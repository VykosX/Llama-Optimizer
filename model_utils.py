#!/usr/bin/env python3
"""
model_utils.py
==============
Self-contained utility library extracted from sweep_engine.py.

Provides GGUF metadata reading, quantization recommendations, model scanning,
GPU/RAM helpers, and the interactive pause function. No dependency on
sweep_engine.py. Used by both batch_runner.py and generate_report.py.

topo_sweep and ctx_sweep are NOT here -- they remain in sweep_engine.py
and are imported optionally by batch_runner.py when that file is present.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path
from typing import Optional

IS_WINDOWS = platform.system() == "Windows"

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    try:
        import subprocess as _sp
        _sp.check_call([sys.executable, "-m", "pip", "install", "psutil", "-q"])
        import psutil
        HAS_PSUTIL = True
    except Exception:
        HAS_PSUTIL = False

try:
    import pynvml as _nvml
    _nvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SKIP_PATTERNS = ["mmproj", "embedding", "embed", "encoder"]
RAM_SAFETY_PCT = float(os.environ.get("RAM_SAFETY_PCT", "0.05"))
RAM_WARN_PCT   = float(os.environ.get("RAM_WARN_PCT",   "0.05"))
MODEL_VRAM_OVERHEAD_MB = 2048   # fixed overhead: KV cache + compute buffers

# GPU info -- populated by detect_gpus()
_GPU_INFO: list[dict] = []

# ---------------------------------------------------------------------------
# GGUF metadata
# ---------------------------------------------------------------------------

def read_gguf_metadata(path: Path) -> dict:
    """Read key-value metadata from a GGUF file without loading tensors.

    Correctly handles large arrays (token vocabularies with 100k+ entries)
    by reading only the first 8 scalar elements and seeking past the rest.
    String arrays (tokenizer vocab) are skipped entirely to avoid reading
    gigabytes of data on every model scan.
    """
    import struct

    GGUF_MAGIC = b"GGUF"
    UINT8, INT8, UINT16, INT16 = 0, 1, 2, 3
    UINT32, INT32, FLOAT32, BOOL = 4, 5, 6, 7
    STRING, ARRAY, UINT64, INT64, FLOAT64 = 8, 9, 10, 11, 12

    # Fixed byte sizes for scalar types (STRING and ARRAY have no fixed size)
    _SCALAR_SIZE = {
        UINT8: 1, INT8: 1, UINT16: 2, INT16: 2,
        UINT32: 4, INT32: 4, FLOAT32: 4, BOOL: 1,
        UINT64: 8, INT64: 8, FLOAT64: 8,
    }

    result = {}
    try:
        with open(path, "rb") as f:
            if f.read(4) != GGUF_MAGIC:
                return {"_parse_error": "not a GGUF file"}
            version = struct.unpack("<I", f.read(4))[0]
            if version not in (2, 3):
                return {"_parse_error": f"unsupported GGUF version {version}"}
            _n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv       = struct.unpack("<Q", f.read(8))[0]

            def read_string() -> str:
                length = struct.unpack("<Q", f.read(8))[0]
                return f.read(length).decode("utf-8", errors="replace")

            def skip_string():
                length = struct.unpack("<Q", f.read(8))[0]
                f.seek(length, 1)

            def read_scalar(t):
                fmt = {UINT8:"<B",INT8:"<b",UINT16:"<H",INT16:"<h",
                       UINT32:"<I",INT32:"<i",FLOAT32:"<f",BOOL:"<B",
                       UINT64:"<Q",INT64:"<q",FLOAT64:"<d"}[t]
                sz = _SCALAR_SIZE[t]
                raw = struct.unpack(fmt, f.read(sz))[0]
                return bool(raw) if t == BOOL else raw

            def skip_value(t):
                """Advance file position past one value of type t."""
                if t == STRING:
                    skip_string()
                elif t == ARRAY:
                    at = struct.unpack("<I", f.read(4))[0]
                    ac = struct.unpack("<Q", f.read(8))[0]
                    skip_array_elements(at, ac)
                elif t in _SCALAR_SIZE:
                    f.seek(_SCALAR_SIZE[t], 1)
                else:
                    raise ValueError(f"unknown type {t}")

            def skip_array_elements(elem_type, count):
                """Skip count elements of elem_type without reading them."""
                if elem_type in _SCALAR_SIZE:
                    # Fixed-size elements: one seek covers all
                    f.seek(_SCALAR_SIZE[elem_type] * count, 1)
                else:
                    # Variable-size (STRING) or nested ARRAY: must iterate
                    # Cap at 65536 to avoid pathological cases
                    for _ in range(min(count, 65536)):
                        skip_value(elem_type)

            def read_value(t):
                """Read and return one value of type t."""
                if t == STRING:
                    return read_string()
                if t in _SCALAR_SIZE:
                    return read_scalar(t)
                if t == ARRAY:
                    at = struct.unpack("<I", f.read(4))[0]
                    ac = struct.unpack("<Q", f.read(8))[0]
                    # For STRING arrays (tokenizer vocab etc.) skip entirely
                    if at == STRING:
                        skip_array_elements(at, ac)
                        return None  # caller drops None values
                    # For scalar arrays read first 8, skip the rest
                    n_read = min(ac, 8)
                    items = [read_scalar(at) for _ in range(n_read)]
                    if ac > n_read:
                        f.seek(_SCALAR_SIZE[at] * (ac - n_read), 1)
                    return items
                raise ValueError(f"unknown GGUF type {t}")

            for _ in range(n_kv):
                key   = read_string()
                vtype = struct.unpack("<I", f.read(4))[0]
                val   = read_value(vtype)
                # Drop None (skipped string arrays) and oversized lists
                if val is not None:
                    if not isinstance(val, list) or len(val) <= 8:
                        result[key] = val

    except Exception as e:
        result["_parse_error"] = str(e)
    return result


def get_model_meta(model_path: Path) -> dict:
    """Return a normalised metadata dict with architecture-independent keys."""
    raw  = read_gguf_metadata(model_path)
    arch = raw.get("general.architecture", "")

    def _get(*keys, default=0):
        for k in keys:
            v = raw.get(k) or raw.get(f"{arch}.{k}")
            if v is not None:
                return int(v) if isinstance(v, float) else v
        return default

    n_layers   = _get("block_count", default=32)
    n_heads_kv = _get("attention.head_count_kv", default=8)
    head_dim   = _get("attention.key_length", default=128)
    n_expert   = _get("expert_count", "n_expert", default=0)
    n_used     = _get("expert_used_count", "n_expert_used", default=0)
    ctx_len    = _get("context_length", default=131072)
    interval   = _get("full_attention_interval", default=0)
    n_attn     = max(1, n_layers // interval) if 0 < interval < n_layers else n_layers

    # Thinking/reasoning model detection via filename + GGUF name field
    _name_check = model_path.stem.lower()
    _general_name = str(raw.get("general.name", "") or "").lower()
    _THINK_RE = re.compile(
        r'\bthink|\breason|\bqwq\b|\br1\b|-r1-|deepseek-r1|0528', re.IGNORECASE)
    is_thinking = bool(_THINK_RE.search(_name_check + " " + _general_name))

    # Parameter count — standard GGUF key added in llama.cpp ~mid-2024
    _pc = raw.get("general.parameter_count")
    parameters_b = round(int(_pc) / 1e9, 2) if _pc and int(_pc) > 0 else None

    return {
        "n_layers":      n_layers,
        "n_heads_kv":    n_heads_kv,
        "head_dim":      head_dim,
        "n_attn_layers": n_attn,
        "n_expert":      n_expert,
        "n_expert_used": n_used,
        "context_length": ctx_len,
        "arch":          arch,
        "parameters_b":  parameters_b,
        "is_moe":        n_expert > 0,
        "is_hybrid":     interval > 0 and n_attn < n_layers,
        "is_thinking":   is_thinking,
        "_parse_error":  raw.get("_parse_error"),
    }


def kv_cache_mb_per_token(meta: dict) -> float:
    """Estimate KV cache size in MB per context token at f16 precision."""
    return (meta["n_attn_layers"] * meta["n_heads_kv"]
            * meta["head_dim"] * 2 * 2) / (1024 * 1024)


# ---------------------------------------------------------------------------
# MTP (Multi-Token Prediction) detection
# ---------------------------------------------------------------------------

# GGUF metadata key that signals built-in MTP heads: {arch}.nextn_predict_layers
# Present in: Qwen3.5/3.6 (arch=qwen35), DeepSeek V3/R1 (arch=deepseek3/deepseek2),
#             Gemma 4 (arch=gemma4)
_MTP_META_KEY_SUFFIX = ".nextn_predict_layers"

# Filename substrings that indicate an MTP-capable GGUF when metadata is unavailable
_MTP_FILENAME_PATTERNS = [
    re.compile(r'[-_\.]mtp[-_\.]', re.IGNORECASE),
    re.compile(r'[-_]mtp[-_\.]', re.IGNORECASE),
    re.compile(r'[-_\.]mtp$', re.IGNORECASE),
    re.compile(r'-MTP-', re.IGNORECASE),
    re.compile(r'\.mtp\.', re.IGNORECASE),
]

# Architectures known to support MTP training (heads may or may not be in GGUF)
_MTP_CAPABLE_ARCHS = {"qwen35", "deepseek3", "deepseek2", "gemma4"}


def detect_mtp(model_path: Path) -> dict:
    """Detect whether a GGUF model has built-in MTP (Multi-Token Prediction) heads.

    Detection cascade (most reliable to least):
      1. GGUF metadata key  {arch}.nextn_predict_layers  -- definitive
      2. Filename pattern matching (-MTP-, .mtp., etc.)  -- probable
      3. Architecture in _MTP_CAPABLE_ARCHS              -- possible (needs confirmation)

    Returns dict with keys:
      has_mtp (bool), mtp_layers (int), source (str), arch (str), confidence (str)
    """
    meta = _cached_meta(model_path)
    arch = meta.get("arch", "")
    stem = model_path.stem

    # Tier 1: definitive -- GGUF metadata contains nextn_predict_layers
    raw = read_gguf_metadata(model_path)
    for k, v in raw.items():
        if k.endswith(_MTP_META_KEY_SUFFIX) and not k.startswith("_"):
            n = int(v) if v else 1
            return {"has_mtp": n > 0, "mtp_layers": n, "source": "metadata",
                    "arch": arch, "confidence": "high"}

    # Tier 2: filename pattern
    for pat in _MTP_FILENAME_PATTERNS:
        if pat.search(stem):
            return {"has_mtp": True, "mtp_layers": 1, "source": "filename",
                    "arch": arch, "confidence": "medium"}

    # Tier 3: architecture hint (family supports MTP but this GGUF may lack heads)
    if arch.lower() in _MTP_CAPABLE_ARCHS:
        return {"has_mtp": False, "mtp_layers": 0, "source": "arch_hint",
                "arch": arch, "confidence": "low"}

    return {"has_mtp": False, "mtp_layers": 0, "source": "none",
            "arch": arch, "confidence": "high"}


_META_CACHE: dict[str, dict] = {}

def _cached_meta(model_path: Path) -> dict:
    key = str(model_path)
    if key not in _META_CACHE:
        _META_CACHE[key] = get_model_meta(model_path)
    return _META_CACHE[key]


import re as _re
import re
_MOE_PATTERNS = [
    _re.compile(r"\d+b[-_]a\d+b",  _re.IGNORECASE),
    _re.compile(r"\d+x\d+b",        _re.IGNORECASE),
    _re.compile(r"moe",               _re.IGNORECASE),
    _re.compile(r"mixture",           _re.IGNORECASE),
]

def is_moe_model(model_path: Path) -> bool:
    """True if the model uses a MoE architecture."""
    meta = _cached_meta(model_path)
    if meta.get("is_moe"):
        return True
    name = model_path.stem.lower()
    return any(p.search(name) for p in _MOE_PATTERNS)


# ---------------------------------------------------------------------------
# Quantization catalogue and recommendations
# ---------------------------------------------------------------------------

_QUANT_CATALOGUE = [
    ("fp16",   16.0, 100, 40,  "FP16 (reference quality, slowest)"),
    ("bf16",   16.0, 100, 40,  "BF16 (reference quality, slowest)"),
    ("f16",    16.0, 100, 40,  "FP16"),
    ("q8_0",    8.5,  90, 85,  "Q8_0 (near-lossless, fast)"),
    ("q6_k",    6.6,  82, 88,  "Q6_K (high quality)"),
    ("q5_k_m",  5.7,  78, 90,  "Q5_K_M (good quality/speed)"),
    ("q5_k_s",  5.5,  76, 91,  "Q5_K_S"),
    ("q5_0",    5.5,  75, 91,  "Q5_0"),
    ("q4_k_m",  4.8,  72, 100, "Q4_K_M (fastest, recommended)"),
    ("q4_k_s",  4.6,  70, 100, "Q4_K_S (fastest, smaller)"),
    ("q4_0",    4.5,  68, 100, "Q4_0"),
    ("iq4_xs",  4.3,  71,  98, "IQ4_XS (imatrix, good quality/size)"),
    ("iq4_nl",  4.5,  72,  99, "IQ4_NL"),
    ("q3_k_m",  3.9,  62,  95, "Q3_K_M"),
    ("q3_k_s",  3.7,  60,  96, "Q3_K_S"),
    ("iq3_xs",  3.3,  58,  96, "IQ3_XS (imatrix)"),
    ("iq3_xxs", 3.1,  55,  97, "IQ3_XXS"),
    ("q2_k",    3.4,  50,  94, "Q2_K (lowest quality)"),
    ("iq2_xs",  2.7,  45,  95, "IQ2_XS"),
    ("iq2_xxs", 2.2,  40,  96, "IQ2_XXS"),
]

_RECOMMEND_ORDER = [
    "q4_k_m", "q4_k_s", "iq4_xs", "q5_k_m", "q5_k_s",
    "q8_0", "q6_k", "q5_0", "q4_0", "fp16", "bf16",
]


def _detect_quant(filename_stem: str) -> Optional[tuple]:
    """Detect quantization from filename. Returns catalogue entry or None."""
    name = filename_stem.lower()
    for entry in sorted(_QUANT_CATALOGUE, key=lambda e: len(e[0]), reverse=True):
        if entry[0] in name:
            return entry
    return None


def _estimate_fp16_size_mb(file_size_mb: float, current_bpw: float) -> float:
    return file_size_mb * (16.0 / current_bpw)


def model_size_mb(model_path: Path) -> float:
    """Return total size in MB. For sharded models (e.g. -00001-of-00006.gguf),
    sums all sibling shards so classification uses the full model size."""
    try:
        _sm = re.search(r'-(\d{5})-of-(\d{5})$', model_path.stem, re.IGNORECASE)
        if _sm:
            # Shard: sum all sibling files matching the same base name pattern
            base = model_path.stem[:_sm.start()]  # e.g. "GLM-5-UD-IQ2_XXS"
            total = 0.0
            for shard in model_path.parent.glob(f"{base}-*-of-{_sm.group(2)}.gguf"):
                try:
                    total += shard.stat().st_size
                except OSError:
                    pass
            return total / (1024 * 1024) if total > 0 else model_path.stat().st_size / (1024 * 1024)
        return model_path.stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0


def recommend_quantizations(
    model_path: Path,
    case: str,
    gpu0_vram_gb: float,
    gpu1_vram_gb: float,
    ram_budget_mb: float,
) -> dict:
    """Recommend alternative quantizations given the model and hardware."""
    stem    = model_path.stem
    file_mb = model_size_mb(model_path)

    current = _detect_quant(stem)
    if current is None:
        return {
            "current_quant": None,
            "note": "Could not detect quantization from filename -- no recommendations.",
            "recommendations": [],
        }

    cur_pattern, cur_bpw, cur_qual, cur_spd, cur_label = current
    fp16_mb     = _estimate_fp16_size_mb(file_mb, cur_bpw)
    gpu0_mb     = gpu0_vram_gb * 1024
    gpu1_mb     = gpu1_vram_gb * 1024
    combined_mb = gpu0_mb + gpu1_mb
    recommendations = []

    for target_pattern in _RECOMMEND_ORDER:
        if target_pattern == cur_pattern:
            continue
        entry = next((e for e in _QUANT_CATALOGUE if e[0] == target_pattern), None)
        if entry is None:
            continue
        t_pat, t_bpw, t_qual, t_spd, t_label = entry
        est_mb = fp16_mb * (t_bpw / 16.0) + MODEL_VRAM_OVERHEAD_MB

        if est_mb <= gpu1_mb:
            fit, fit_label = "A", "fits both GPUs independently (Case A)"
        elif est_mb <= gpu0_mb:
            fit, fit_label = "B", f"fits GPU0 ({gpu0_mb/1024:.0f} GB) only (Case B)"
        elif est_mb <= combined_mb * 0.97:
            fit, fit_label = "C", "requires both GPUs combined (Case C)"
        elif est_mb <= ram_budget_mb:
            fit, fit_label = "D", "requires CPU/RAM offload (Case D)"
        else:
            continue

        size_delta_pct = ((est_mb - MODEL_VRAM_OVERHEAD_MB - file_mb) / file_mb * 100
                          if file_mb > 0 else 0.0)
        qual_delta = t_qual - cur_qual
        spd_delta  = t_spd  - cur_spd
        direction  = "upgrade" if qual_delta > 5 else ("downgrade" if qual_delta < -5 else "sidegrade")

        recommendations.append({
            "quant":          t_pat,
            "label":          t_label,
            "estimated_gb":   round((est_mb - MODEL_VRAM_OVERHEAD_MB) / 1024, 2),
            "case":           fit,
            "fit_label":      fit_label,
            "quality_rank":   t_qual,
            "speed_rank":     t_spd,
            "qual_delta":     qual_delta,
            "spd_delta":      spd_delta,
            "size_delta_pct": round(size_delta_pct, 0),
            "direction":      direction,
        })

    def _sort_key(r):
        order = {"upgrade": 0, "sidegrade": 1, "downgrade": 2}
        return (order[r["direction"]],
                -r["quality_rank"] if r["direction"] != "downgrade" else -r["speed_rank"])
    recommendations.sort(key=_sort_key)

    _meta   = _cached_meta(model_path)
    _is_moe = _meta.get("is_moe") or any(p.search(model_path.stem.lower()) for p in _MOE_PATTERNS)
    moe_note = ""
    if _is_moe:
        n_exp  = _meta.get("n_expert", "?")
        n_used = _meta.get("n_expert_used", "?")
        moe_note = (
            f" NOTE: MoE model ({n_exp} experts, {n_used} active per token). "
            f"All {n_exp} experts must reside in VRAM/RAM -- size estimates are correct "
            f"for memory purposes but speed ranks are less meaningful than for dense models."
        )

    return {
        "current_quant":     cur_pattern,
        "current_label":     cur_label,
        "current_bpw":       cur_bpw,
        "current_case":      case,
        "estimated_fp16_gb": round(fp16_mb / 1024, 2),
        "recommendations":   recommendations,
        "note": (
            "Sizes are estimates based on bits-per-weight ratios. "
            "Actual file sizes vary by architecture." + moe_note
        ),
    }


def classify_model(model_path: Path, gpu0_vram_gb: float, gpu1_vram_gb: float) -> str:
    """Classify model as A/B/C/D based on VRAM fit."""
    size_mb     = model_size_mb(model_path)
    total_mb    = size_mb + MODEL_VRAM_OVERHEAD_MB
    gpu0_mb     = gpu0_vram_gb * 1024
    gpu1_mb     = gpu1_vram_gb * 1024
    combined_mb = gpu0_mb + gpu1_mb
    if total_mb <= min(gpu0_mb, gpu1_mb):
        return "A"
    if total_mb <= gpu0_mb:
        return "B"
    if total_mb <= combined_mb * 0.97:
        return "C"
    return "D"


# ---------------------------------------------------------------------------
# Model scanning
# ---------------------------------------------------------------------------

def find_models(models_dir: Path, filter_str: Optional[str] = None) -> list:
    """Recursively find all GGUF models, skipping non-generative architectures."""
    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        sys.exit(1)

    _NON_GENERATIVE = {
        "bert", "nomic-bert", "jina-bert", "roberta", "distilbert",
        "xlm-roberta", "electra",
    }
    # Shard pattern: modelname-00001-of-00006.gguf
    # Only keep shard 1; llama-server auto-discovers the rest by naming convention.
    _SHARD_RE = re.compile(r'-\d{5}-of-(\d{5})$', re.IGNORECASE)

    out = []
    for f in sorted(models_dir.rglob("*.gguf")):
        if any(s in f.name.lower() for s in SKIP_PATTERNS):
            continue
        if filter_str and filter_str.lower() not in f.name.lower():
            continue
        if not f.exists():
            print(f"  [skip] {f.name} -- file no longer exists")
            continue
        # Skip non-first shards — llama-server loads all shards automatically
        # when given the first one (it finds -00002-, -00003- etc. by itself).
        _sm = _SHARD_RE.search(f.stem)
        if _sm:
            # Extract the shard index from the filename
            _shard_idx = int(re.search(r'-(\d{5})-of-', f.stem, re.IGNORECASE).group(1))
            if _shard_idx != 1:
                continue  # skip shards 2..N
        try:
            _meta = read_gguf_metadata(f)
            _arch = _meta.get("general.architecture", "")
            if _arch.lower() in _NON_GENERATIVE:
                print(f"  [skip] {f.name} -- non-generative arch ({_arch})")
                continue
        except Exception:
            pass
        out.append(f)
    return out


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def detect_gpus() -> list[dict]:
    """Return [{index, name, vram_gb}] sorted by VRAM descending."""
    global _GPU_INFO
    gpus = []
    if HAS_NVML:
        try:
            for i in range(_nvml.nvmlDeviceGetCount()):
                h    = _nvml.nvmlDeviceGetHandleByIndex(i)
                name = _nvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                vram = _nvml.nvmlDeviceGetMemoryInfo(h).total / (1024 ** 3)
                gpus.append({"index": i, "name": name, "vram_gb": vram})
        except Exception:
            gpus = []
    if not gpus:
        g0 = float(os.environ.get("GPU0_VRAM_GB", "0"))
        g1 = float(os.environ.get("GPU1_VRAM_GB", "0"))
        if g0 > 0:
            gpus.append({"index": 0, "name": "GPU0", "vram_gb": g0})
        if g1 > 0:
            gpus.append({"index": 1, "name": "GPU1", "vram_gb": g1})
    gpus.sort(key=lambda g: g["vram_gb"], reverse=True)
    _GPU_INFO = gpus
    return gpus


# ---------------------------------------------------------------------------
# RAM helpers
# ---------------------------------------------------------------------------

def get_ram_info() -> tuple:
    """Returns (total_mb, available_mb, used_pct)."""
    if not HAS_PSUTIL:
        return (256 * 1024, 200 * 1024, 0.22)
    vm = psutil.virtual_memory()
    return (vm.total / (1024*1024), vm.available / (1024*1024), vm.percent / 100.0)


def check_ram_warning():
    """Warn and optionally abort if RAM usage exceeds threshold at startup."""
    total_mb, avail_mb, used_pct = get_ram_info()
    total_gb = total_mb / 1024
    avail_gb = avail_mb / 1024
    used_gb  = (total_mb - avail_mb) / 1024

    if used_pct > RAM_WARN_PCT:
        print(f"\n  {'!'*60}")
        print(f"  RAM WARNING")
        print(f"  Total   : {total_gb:.0f} GB")
        print(f"  Used    : {used_gb:.1f} GB ({used_pct*100:.1f}%)")
        print(f"  Available: {avail_gb:.1f} GB")
        print(f"  High RAM usage may interfere with context ceiling tests.")
        print(f"  {'!'*60}")
        try:
            ans = input("  Continue anyway? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        if ans != "y":
            print("  Aborted.")
            sys.exit(0)
    else:
        print(f"  RAM: {used_gb:.1f}/{total_gb:.0f} GB used "
              f"({used_pct*100:.1f}%)  --  {avail_gb:.1f} GB available")


# ---------------------------------------------------------------------------
# Interactive pause
# ---------------------------------------------------------------------------

def pause_between_models(seconds: int = 5) -> bool:
    """5s countdown; returns False if user pressed n to stop."""
    if not sys.stdin.isatty():
        return True

    _console = sys.stdout
    while hasattr(_console, "_stream"):
        _console = _console._stream

    print(f"\n  -- Press \'n\' within {seconds}s to stop after this model, "
          f"any other key or wait to continue --")

    if IS_WINDOWS:
        import msvcrt
        deadline = time.time() + seconds
        while time.time() < deadline:
            remaining = deadline - time.time()
            _console.write(f"\r  Continuing in {remaining:.0f}s ...  ")
            _console.flush()
            if msvcrt.kbhit():
                ch = msvcrt.getwche()
                _console.write("\n")
                _console.flush()
                if ch.lower() == "n":
                    print("  Stopping after this model -- compiling report...")
                    return False
                return True
            time.sleep(0.1)
    else:
        import select, tty, termios
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        deadline = time.time() + seconds
        try:
            tty.setraw(fd)
            while time.time() < deadline:
                remaining = deadline - time.time()
                _console.write(f"\r  Continuing in {remaining:.0f}s ...  ")
                _console.flush()
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    ch = sys.stdin.read(1)
                    _console.write("\n")
                    _console.flush()
                    if ch.lower() == "n":
                        print("  Stopping after this model -- compiling report...")
                        return False
                    return True
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    _console.write("\r  Continuing...                    \n")
    _console.flush()
    return True
