#!/usr/bin/env python3
"""
sweep_engine.py — GPU Topology Sweep + Context Ceiling Sweep engine.

Provides run_topo_sweep() and run_ctx_sweep() used by batch_runner.py,
plus supporting utilities (model classification, GPU split scenarios,
binary search, probe functions).

This file was previously called run_all_models.py when it also contained
the batch loop. That responsibility moved to batch_runner.py. This file
now focuses exclusively on the sweep logic that characterises each model's
hardware profile before the Optuna optimizer runs.

DO NOT call this file directly. Use batch_runner.py as the entry point.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL CLASSIFICATION (determines which sweeps run)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Case A — fits in BOTH GPUs independently
    • Test gpu0_only and gpu1_only; pick faster winner
    • No split scenarios; no NUMA

  Case B — fits in ONE GPU only (GPU0, the 3090)
    • Test that GPU only
    • No split scenarios; no NUMA

  Case C — requires BOTH GPUs combined to fit in VRAM
    • Skip single-GPU tests (would OOM)
    • Test four split strategies:
        split_prop     proportional by VRAM ratio (e.g. 6.0,4.0)
        split_equal    50/50
        split_g0heavy  80/20 favouring GPU0 (3090 dominant)
        split_kv_aware compensate for KV cache on main-gpu by giving
                       more tensor weight to GPU1 (e.g. 18/22 of 40GB)
    • No NUMA

  Case D — does not fit in combined GPU VRAM (needs CPU offload)
    • Binary-search max layers that fit without OOM (-ngl)
    • Test NUMA policies with that -ngl:
        numa_none      OS default
        numa_dist      --numa distribute (both sockets)
        numa_iso       --numa isolate    (socket 0 only)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT CEILING SWEEP  (--ctx-sweep)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Runs after topology sweep, before optimizer.

  For Case A / B models:
    ctx_gpu_single   max stable ctx on winning single GPU
    ctx_gpu_combined max stable ctx using both GPUs combined
                     (only if combined topology exists)

  For Case C models:
    ctx_gpu_combined max stable ctx on winning split topology

  For Case D models (Case B RAM tests also run here):
    ctx_ram_mixed    max ctx with -ngl=max_fit, KV cache pages
                     to RAM automatically as VRAM fills
                     from the start, VRAM free for weights only)

  All searches use binary search: coarse 4096-token steps,
  then 1024-token fine pass around the bracket.
  Stability = existing speed + large prompts complete without crash.
  The highest stable GPU-only ctx becomes the optimizer's ctx ceiling.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python sweep_engine.py --topo-sweep --ctx-sweep
  python sweep_engine.py --topo-sweep                  # no ctx sweep
  python sweep_engine.py --topo-only                   # topology only
  python sweep_engine.py --ctx-only                    # ctx sweep only
  python sweep_engine.py --topo-sweep --ctx-sweep --skip-ctx-b
  python sweep_engine.py --filter "Qwen" --topo-sweep --ctx-sweep
  python sweep_engine.py --resume --topo-sweep --ctx-sweep
  python sweep_engine.py --report-only
  python sweep_engine.py --dry-run

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENVIRONMENT VARIABLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LLAMA_SERVER         Path to llama-server binary
  LLM_OPT_MODELS_DIR   Models directory (overrides --models-dir)
  LLM_OPT_PORT         llama-server port (default: 8090)
  LLM_OPT_TIMEOUT      Per-model timeout in seconds (default: 2700)
  GPU0_VRAM_GB         Override detected GPU0 VRAM in GB (auto-detected by default)
  GPU1_VRAM_GB         Override detected GPU1 VRAM in GB (auto-detected by default)
  RAM_SAFETY_PCT       Fraction of RAM reserved for OS (default: 0.05)
  RAM_WARN_PCT         Warn if RAM used exceeds this at startup (default: 0.05)
"""

import argparse
import csv
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from statistics import median

# ── optional deps ─────────────────────────────────────────────────────────────
def _pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:
    import requests
except ImportError:
    _pip_install("requests"); import requests

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    _pip_install("psutil"); import psutil; HAS_PSUTIL = True

IS_WINDOWS = platform.system() == "Windows"

# Force CUDA to use PCIe bus ordering, matching NVML device indices.
# Without this, CUDA may number GPUs differently from pynvml, causing
# CUDA_VISIBLE_DEVICES to target the wrong physical card.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# ── log-file tee ──────────────────────────────────────────────────────────────

_LOG_FH = None   # file handle; None until install_log_tee() is called
_NO_JINJA_MODELS: set = set()  # model paths that need --no-jinja
_last_launch_was_crash: bool = False  # True when last launch_server() call returned None due to a hard crash (vs timeout)


class _Tee:
    """Writes to every stream in self._s simultaneously."""
    def __init__(self, *streams):
        self._s = streams
    def write(self, data):
        for s in self._s:
            try:
                s.write(data)
            except Exception:
                pass
    def flush(self):
        for s in self._s:
            try:
                s.flush()
            except Exception:
                pass
    def fileno(self):
        return self._s[0].fileno()
    def isatty(self):
        return False


def install_log_tee(log_path: str, mode: str = "w") -> None:
    """Open log_path and tee sys.stdout + sys.stderr into it.
    Safe to call multiple times — subsequent calls are no-ops."""
    global _LOG_FH
    if _LOG_FH is not None:
        return
    _LOG_FH = open(log_path, mode, encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, _LOG_FH)
    sys.stderr = _Tee(sys.stderr, _LOG_FH)


# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_MODELS_DIR = Path(os.environ.get(
    "LLM_OPT_MODELS_DIR",
    str(Path.home() / ".lmstudio" / "models")
))

DEFAULT_LLAMA_SERVER = Path(os.environ.get(
    "LLAMA_SERVER",
    str(SCRIPT_DIR / ("llama-server.exe" if IS_WINDOWS else "llama-server"))
))

_optimizer_candidates = [
    SCRIPT_DIR / "LLM_Optimiser_lmstudio.py",
    SCRIPT_DIR / "LLM_Optimiser.py",
]
DEFAULT_OPTIMIZER = next(
    (p for p in _optimizer_candidates if p.exists()), _optimizer_candidates[0]
)

TIMEOUT_PER_MODEL  = int(os.environ.get("LLM_OPT_TIMEOUT",       str(60 * 45)))
TRIAL_TIMEOUT_S    = int(os.environ.get("LLM_OPT_TRIAL_TIMEOUT", str(60 * 6)))   # per-trial hard limit
PROBE_TIMEOUT_S    = int(os.environ.get("LLM_OPT_PROBE_TIMEOUT", "30"))           # ctx-sweep probe
STARTUP_TIMEOUT_S  = int(os.environ.get("LLM_OPT_STARTUP_TIMEOUT", "30"))         # server startup wait

# ── hardware config ───────────────────────────────────────────────────────────
PORT           = int(os.environ.get("LLM_OPT_PORT", "8090"))
VERBOSE        = False                                              # set to True via --verbose to show live loading progress
RAM_SAFETY_PCT = float(os.environ.get("RAM_SAFETY_PCT", "0.05"))
RAM_WARN_PCT   = float(os.environ.get("RAM_WARN_PCT",   "0.05"))

# GPU detection — populated at startup by _detect_gpus()
# Each entry: {"index": int, "name": str, "vram_gb": float}
# Sorted by VRAM descending so GPU0 is always the largest card.
_GPU_INFO: list = []

def _detect_gpus() -> list:
    """Query pynvml for all GPUs. Returns list of dicts sorted by VRAM descending.
    Falls back to env-var overrides GPU0_VRAM_GB / GPU1_VRAM_GB if pynvml unavailable.
    """
    gpus = []
    if HAS_NVML:
        try:
            for i in range(pynvml.nvmlDeviceGetCount()):
                h    = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                vram_gb = pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024 ** 3)
                gpus.append({"index": i, "name": name, "vram_gb": vram_gb})
        except Exception:
            gpus = []

    if not gpus:
        # Fallback: env vars or safe defaults
        g0 = float(os.environ.get("GPU0_VRAM_GB", "0"))
        g1 = float(os.environ.get("GPU1_VRAM_GB", "0"))
        if g0 > 0:
            gpus.append({"index": 0, "name": "GPU0", "vram_gb": g0})
        if g1 > 0:
            gpus.append({"index": 1, "name": "GPU1", "vram_gb": g1})

    # Sort by VRAM descending — largest card is always GPU0 in our logic
    gpus.sort(key=lambda g: g["vram_gb"], reverse=True)
    return gpus

# Convenience accessors — set after _detect_gpus() runs in main()
GPU0_VRAM_GB = 0.0   # largest GPU (set at runtime)
GPU1_VRAM_GB = 0.0   # second GPU if present (set at runtime)

SKIP_PATTERNS = ["mmproj", "embedding", "embed", "encoder"]


# ── timing helpers ────────────────────────────────────────────────────────────

class Timer:
    """Lightweight lap timer for structured timing output.

    Usage:
        t = Timer("Phase 0.5")
        t.lap("flag kv_offload")   # prints [0.8s] flag kv_offload
        t.done()                    # prints [12.3s total] Phase 0.5
    """
    _run_start: float = time.time()   # class-level: set once at import

    def __init__(self, label: str = "", *, silent: bool = False):
        self.label   = label
        self.silent  = silent
        self._start  = time.time()
        self._last   = self._start

    @classmethod
    def reset_run(cls) -> None:
        """Call once at the very start of a batch run."""
        cls._run_start = time.time()

    @classmethod
    def run_elapsed(cls) -> float:
        return time.time() - cls._run_start

    def elapsed(self) -> float:
        return time.time() - self._start

    def lap(self, note: str = "") -> float:
        now  = time.time()
        lap  = now - self._last
        self._last = now
        if not self.silent:
            tag = f"  [{lap:.1f}s]" + (f" {note}" if note else "")
            print(tag)
        return lap

    def done(self, note: str = "") -> float:
        total = time.time() - self._start
        if not self.silent:
            label = note or self.label
            run_h, run_m = divmod(self.run_elapsed(), 3600)
            run_m, run_s  = divmod(run_m, 60)
            run_str = (f"{int(run_h)}h " if run_h else "") + f"{int(run_m)}m{int(run_s):02d}s"
            print(f"  [{total:.1f}s total] {label}  (run elapsed: {run_str})")
        return total


def _fmt_s(seconds: float) -> str:
    """Format seconds as e.g. '1m23s' or '45s'."""
    if seconds >= 60:
        return f"{int(seconds)//60}m{int(seconds)%60:02d}s"
    return f"{seconds:.1f}s"

# ── GGUF metadata reader ──────────────────────────────────────────────────────

def read_gguf_metadata(path: Path) -> dict:
    """
    Read key-value metadata from a GGUF file header without loading tensors.
    Returns a dict of string keys to scalar values (int / float / bool / str).
    Array values are truncated to first 8 elements to avoid reading token tables.
    Returns {} with a '_parse_error' key on failure.
    """
    GGUF_MAGIC = b'GGUF'
    UINT8, INT8, UINT16, INT16 = 0, 1, 2, 3
    UINT32, INT32, FLOAT32, BOOL = 4, 5, 6, 7
    STRING, ARRAY, UINT64, INT64, FLOAT64 = 8, 9, 10, 11, 12

    result = {}
    try:
        with open(path, 'rb') as f:
            if f.read(4) != GGUF_MAGIC:
                return {'_parse_error': 'not a GGUF file'}
            version = __import__('struct').unpack('<I', f.read(4))[0]
            if version not in (2, 3):
                return {'_parse_error': f'unsupported GGUF version {version}'}
            import struct
            n_tensors = struct.unpack('<Q', f.read(8))[0]  # noqa: F841
            n_kv      = struct.unpack('<Q', f.read(8))[0]

            def rs():
                return f.read(struct.unpack('<Q', f.read(8))[0]).decode('utf-8', errors='replace')

            _SCALAR_SIZE = {
                UINT8:1,INT8:1,UINT16:2,INT16:2,UINT32:4,INT32:4,
                FLOAT32:4,BOOL:1,UINT64:8,INT64:8,FLOAT64:8,
            }

            def rv(t):
                if t == STRING:
                    return rs()
                if t in _SCALAR_SIZE:
                    fmt = {UINT8:'<B',INT8:'<b',UINT16:'<H',INT16:'<h',
                           UINT32:'<I',INT32:'<i',FLOAT32:'<f',BOOL:'<B',
                           UINT64:'<Q',INT64:'<q',FLOAT64:'<d'}[t]
                    raw = struct.unpack(fmt, f.read(_SCALAR_SIZE[t]))[0]
                    return bool(raw) if t == BOOL else raw
                if t == ARRAY:
                    at = struct.unpack('<I', f.read(4))[0]
                    ac = struct.unpack('<Q', f.read(8))[0]
                    # Skip string arrays (tokenizer vocab) entirely
                    if at == STRING:
                        for _ in range(min(ac, 65536)):
                            length = struct.unpack('<Q', f.read(8))[0]
                            f.seek(length, 1)
                        return None
                    # Fixed-size scalar arrays: read 8, seek past rest
                    if at not in _SCALAR_SIZE:
                        raise ValueError(f'unsupported array elem type {at}')
                    n_read = min(ac, 8)
                    fmt = {UINT8:'<B',INT8:'<b',UINT16:'<H',INT16:'<h',
                           UINT32:'<I',INT32:'<i',FLOAT32:'<f',BOOL:'<B',
                           UINT64:'<Q',INT64:'<q',FLOAT64:'<d'}[at]
                    items = [struct.unpack(fmt, f.read(_SCALAR_SIZE[at]))[0]
                             for _ in range(n_read)]
                    if ac > n_read:
                        f.seek(_SCALAR_SIZE[at] * (ac - n_read), 1)
                    return items
                raise ValueError(f'unknown type {t}')

            for _ in range(n_kv):
                key   = rs()
                vtype = struct.unpack('<I', f.read(4))[0]
                val   = rv(vtype)
                if val is not None and (not isinstance(val, list) or len(val) <= 8):
                    result[key] = val
    except Exception as e:
        result['_parse_error'] = str(e)
    return result


def get_model_meta(model_path: Path) -> dict:
    """
    Return a normalised metadata dict with architecture-independent keys:
        n_layers       — total transformer / SSM block count
        n_heads_kv     — KV attention head count
        head_dim       — key/value head dimension in elements
        n_attn_layers  — number of full-attention layers (< n_layers for hybrids)
        n_expert       — number of MoE experts (0 for dense models)
        n_expert_used  — experts activated per token (0 for dense models)
        context_length — trained max context length
        arch           — architecture string (e.g. "llama", "qwen35", "mixtral")
        is_moe         — True if model has MoE experts
        is_hybrid      — True if model has fewer attention layers than total layers
    Unreadable fields default to sensible fallbacks.
    """
    raw  = read_gguf_metadata(model_path)
    arch = raw.get('general.architecture', '')

    def _get(*keys, default=0):
        for k in keys:
            v = raw.get(k) or raw.get(f'{arch}.{k}')
            if v is not None:
                return int(v) if isinstance(v, float) else v
        return default

    n_layers   = _get('block_count', default=32)
    n_heads_kv = _get('attention.head_count_kv', default=8)
    head_dim   = _get('attention.key_length', default=128)
    n_expert   = _get('expert_count', 'n_expert', default=0)
    n_used     = _get('expert_used_count', 'n_expert_used', default=0)
    ctx_len    = _get('context_length', default=131072)

    # Hybrid models (Mamba/SSM + attention) have full_attention_interval > 0
    # Every N-th layer is a full attention layer; the rest are SSM.
    interval   = _get('full_attention_interval', default=0)
    if interval > 0 and interval < n_layers:
        n_attn = max(1, n_layers // interval)
    else:
        n_attn = n_layers  # pure attention model

    return {
        'n_layers':      n_layers,
        'n_heads_kv':    n_heads_kv,
        'head_dim':      head_dim,
        'n_attn_layers': n_attn,
        'n_expert':      n_expert,
        'n_expert_used': n_used,
        'context_length': ctx_len,
        'arch':          arch,
        'is_moe':        n_expert > 0,
        'is_hybrid':     interval > 0 and n_attn < n_layers,
        # Propagate parse failure so callers can detect unreadable files
        '_parse_error':  raw.get('_parse_error'),
    }


def kv_cache_mb_per_token(meta: dict) -> float:
    """
    Estimate KV cache size in MB per context token at f16 precision.
    Formula: n_attn_layers * n_heads_kv * head_dim * 2(K+V) * 2(bytes) / 1024^2
    For hybrid models only attention layers contribute to the KV cache.
    """
    return (meta['n_attn_layers'] * meta['n_heads_kv']
            * meta['head_dim'] * 2 * 2) / (1024 * 1024)


_META_CACHE: dict[str, dict] = {}   # path → meta, avoids re-reading

def _cached_meta(model_path: Path) -> dict:
    key = str(model_path)
    if key not in _META_CACHE:
        _META_CACHE[key] = get_model_meta(model_path)
    return _META_CACHE[key]


# ── MoE filename detection (fallback when GGUF parse fails) ──────────────────
import re as _re
import re
_MOE_PATTERNS = [
    _re.compile(r'\d+b[-_]a\d+b',  _re.IGNORECASE),   # Qwen / DeepSeek: 35B-A3B
    _re.compile(r'\d+x\d+b',        _re.IGNORECASE),   # Mixtral: 8x7B
    _re.compile(r'moe',              _re.IGNORECASE),   # explicit MoE in name
    _re.compile(r'mixture',          _re.IGNORECASE),
]

def is_moe_model(model_path: Path) -> bool:
    """True if the model uses a MoE architecture (from GGUF metadata or filename)."""
    meta = _cached_meta(model_path)
    if meta.get('is_moe'):
        return True
    name = model_path.stem.lower()
    return any(p.search(name) for p in _MOE_PATTERNS)

# ── quantization catalogue ────────────────────────────────────────────────────
# Each entry: (pattern_in_filename, bits_per_weight, quality_rank, speed_rank, label)
# quality_rank: higher = better quality  (FP16=100, Q2=20)
# speed_rank:   higher = faster on GPU   (Q4=100, FP16=40)
# Patterns are matched case-insensitively against the model filename stem.
_QUANT_CATALOGUE = [
    # name_pattern     bpw    qual  spd   display_label
    ("fp16",           16.0,  100,  40,   "FP16 (reference quality, slowest)"),
    ("bf16",           16.0,  100,  40,   "BF16 (reference quality, slowest)"),
    ("f16",            16.0,  100,  40,   "FP16"),
    ("q8_0",            8.5,   90,  85,   "Q8_0 (near-lossless, fast)"),
    ("q6_k",            6.6,   82,  88,   "Q6_K (high quality)"),
    ("q5_k_m",          5.7,   78,  90,   "Q5_K_M (good quality/speed)"),
    ("q5_k_s",          5.5,   76,  91,   "Q5_K_S"),
    ("q5_0",            5.5,   75,  91,   "Q5_0"),
    ("q4_k_m",          4.8,   72,  100,  "Q4_K_M (fastest, recommended)"),
    ("q4_k_s",          4.6,   70,  100,  "Q4_K_S (fastest, smaller)"),
    ("q4_0",            4.5,   68,  100,  "Q4_0"),
    ("iq4_xs",          4.3,   71,   98,  "IQ4_XS (imatrix, good quality/size)"),
    ("iq4_nl",          4.5,   72,   99,  "IQ4_NL"),
    ("q3_k_m",          3.9,   62,   95,  "Q3_K_M"),
    ("q3_k_s",          3.7,   60,   96,  "Q3_K_S"),
    ("iq3_xs",          3.3,   58,   96,  "IQ3_XS (imatrix)"),
    ("iq3_xxs",         3.1,   55,   97,  "IQ3_XXS"),
    ("q2_k",            3.4,   50,   94,  "Q2_K (lowest quality)"),
    ("iq2_xs",          2.7,   45,   95,  "IQ2_XS"),
    ("iq2_xxs",         2.2,   40,   96,  "IQ2_XXS"),
]

# Ordered recommendations: which quants to suggest as alternatives.
# We always suggest in this priority order for speed: Q4_K_M > Q8_0 > Q5_K_M > Q6_K > FP16
_RECOMMEND_ORDER = [
    "q4_k_m", "q4_k_s", "iq4_xs", "q5_k_m", "q5_k_s",
    "q8_0", "q6_k", "q5_0", "q4_0", "fp16", "bf16",
]


def _detect_quant(filename_stem: str) -> tuple | None:
    """
    Detect the quantization of a model from its filename stem.
    Returns the matching _QUANT_CATALOGUE entry or None.
    """
    name = filename_stem.lower()
    # Try longest match first to avoid q4 matching q4_k_m
    sorted_cat = sorted(_QUANT_CATALOGUE, key=lambda e: len(e[0]), reverse=True)
    for pattern, bpw, qual, spd, label in sorted_cat:
        if pattern in name:
            return (pattern, bpw, qual, spd, label)
    return None


def _estimate_fp16_size_mb(file_size_mb: float, current_bpw: float) -> float:
    """
    Estimate the FP16 baseline size from the current quantized file size.
    FP16 = 16 bpw, so FP16_size = current_size * (16 / current_bpw).
    This is an estimate — embedding layers stay FP16 regardless of quant,
    so actual ratios are slightly higher than pure bpw math suggests.
    """
    return file_size_mb * (16.0 / current_bpw)


def recommend_quantizations(
    model_path: Path,
    case: str,
    gpu0_vram_gb: float,
    gpu1_vram_gb: float,
    ram_budget_mb: float,
) -> dict:
    """
    Given the current model and hardware, recommend alternative quantizations.

    Returns a dict with:
        current_quant       — detected quant name (or None)
        current_bpw         — bits per weight of current quant
        estimated_fp16_gb   — estimated FP16 baseline size
        recommendations     — list of recommendation dicts
        note                — caveat string
    """
    stem = model_path.stem
    file_mb = model_size_mb(model_path)

    current = _detect_quant(stem)
    if current is None:
        return {
            "current_quant": None,
            "note": "Could not detect quantization from filename — no recommendations.",
            "recommendations": [],
        }

    cur_pattern, cur_bpw, cur_qual, cur_spd, cur_label = current
    fp16_mb = _estimate_fp16_size_mb(file_mb, cur_bpw)

    gpu0_mb     = gpu0_vram_gb * 1024
    gpu1_mb     = gpu1_vram_gb * 1024
    combined_mb = gpu0_mb + gpu1_mb

    recommendations = []

    for target_pattern in _RECOMMEND_ORDER:
        if target_pattern == cur_pattern:
            continue  # skip current quant

        # Find catalogue entry
        entry = next((e for e in _QUANT_CATALOGUE if e[0] == target_pattern), None)
        if entry is None:
            continue
        t_pat, t_bpw, t_qual, t_spd, t_label = entry

        # Estimate target size
        est_mb = fp16_mb * (t_bpw / 16.0) + MODEL_VRAM_OVERHEAD_MB

        # Determine where it would run
        if est_mb <= gpu1_mb:
            fit = "A"
            fit_label = f"fits both GPUs independently (Case A)"
        elif est_mb <= gpu0_mb:
            fit = "B"
            _g0n = _GPU_INFO[0]["name"] if _GPU_INFO else "GPU0"
            fit_label = f"fits GPU0 ({_g0n}, {gpu0_mb/1024:.0f} GB) only (Case B)"
        elif est_mb <= combined_mb * 0.97:
            fit = "C"
            fit_label = f"requires both GPUs combined (Case C)"
        elif est_mb <= ram_budget_mb:
            fit = "D"
            fit_label = f"requires CPU/RAM offload (Case D)"
        else:
            continue  # doesn't even fit with RAM — skip

        # Quality vs current
        qual_delta = t_qual - cur_qual
        spd_delta  = t_spd  - cur_spd
        size_delta_pct = ((est_mb - MODEL_VRAM_OVERHEAD_MB - file_mb) / file_mb * 100
                          if file_mb > 0 else 0.0)

        # Classify as upgrade / downgrade / sidegrade
        if qual_delta > 5:
            direction = "upgrade"
        elif qual_delta < -5:
            direction = "downgrade"
        else:
            direction = "sidegrade"

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

    # Sort: upgrades first (by quality desc), then sidegrades, then downgrades (by speed desc)
    def _sort_key(r):
        order = {"upgrade": 0, "sidegrade": 1, "downgrade": 2}
        return (order[r["direction"]], -r["quality_rank"] if r["direction"] != "downgrade"
                else -r["speed_rank"])

    recommendations.sort(key=_sort_key)

    _meta   = _cached_meta(model_path)
    _is_moe = _meta.get("is_moe") or any(p.search(model_path.stem.lower())
                                          for p in _MOE_PATTERNS)
    moe_note = ""
    if _is_moe:
        n_exp  = _meta.get("n_expert", "?")
        n_used = _meta.get("n_expert_used", "?")
        moe_note = (
            f" NOTE: This is a MoE model ({n_exp} experts, {n_used} active per token). "
            f"All {n_exp} experts must reside in VRAM/RAM regardless of how many "
            f"activate at inference — size estimates are correct for memory purposes. "
            f"Speed ranks are less meaningful than for dense models: MoE Q4 is not "
            f"necessarily faster than Q8 due to expert routing overhead. "
            f"Use the optimizer's MoE strategy sweep (cpu_moe / partial_cpu_moe) "
            f"to find the optimal expert placement for your hardware."
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
            "Actual file sizes vary by architecture (embedding layers "
            "remain FP16 regardless of quant). Verify before downloading."
            + moe_note
        ),
    }

# ── model size estimation constants ───────────────────────────────────────────
# Fixed VRAM overhead added to model file size to estimate total VRAM need.
# Accounts for KV cache (ctx=8192 ≈ 0.5-1 GB) + compute buffers + output layer.
# A flat 2 GB is accurate for models from ~3 GB to ~70 GB.
MODEL_VRAM_OVERHEAD_MB = 2048

# ── VRAM/RAM helpers ──────────────────────────────────────────────────────────

def _nvml_device_count() -> int:
    if not HAS_NVML:
        return 0
    try:
        return pynvml.nvmlDeviceGetCount()
    except Exception:
        return 0


def _estimate_ctx_from_vram(
    vram_free_mb: float,
    kv_per_token: float,
    trained_max: int,
    safety: float = 0.85,
    step: int = 4096,
) -> int:
    """
    Estimate the maximum context that will fit in vram_free_mb of free VRAM.
    Applies a safety margin to leave room for compute buffers.
    Returns the estimate rounded down to the nearest step, clamped to
    [step, trained_max].
    """
    if kv_per_token <= 0:
        return trained_max
    predicted = int((vram_free_mb * safety) / kv_per_token)
    predicted = (predicted // step) * step
    return max(step, min(predicted, trained_max))


def _probe_then_search(
    model_path: Path,
    llama_server: Path,
    base_params: dict,
    predicted: int,
    lo: int,
    hi: int,
    label: str = "",
) -> int:
    """
    Smart context search: probe the mathematically predicted maximum first.
    - If it succeeds and predicted >= hi: done in 1 probe.
    - If it succeeds but hi > predicted: search up from predicted to hi.
    - If it fails: binary search downward from predicted to lo.
    Returns the best stable context found.
    """
    predicted = max(lo, min(predicted, hi))
    indent = "      "

    # ── First probe: try the predicted value ─────────────────────────────
    print(f"{indent}[ctx-search] predicted {predicted:,} tokens (from VRAM formula)...")
    wait_cool()
    stop_proc(None)
    time.sleep(1)
    t0 = time.time()
    ok = _probe_stable(model_path, llama_server, {**base_params, "num_ctx": predicted})
    elapsed = time.time() - t0

    _NEEDLE_MIN_CTX = 8192  # below this, skip needle — too small for reliable retrieval

    def _stable_and_coherent(ctx_val: int) -> bool:
        """Load server at ctx_val and run stability + needle quality probe.
        For ctx <= _NEEDLE_MIN_CTX, skip needle and accept stability only.
        """
        proc_inner = launch_server(model_path, llama_server, {**base_params, "num_ctx": ctx_val})
        if proc_inner is None:
            return False
        if not _query_once(_PROBE_PROMPT, timeout=PROBE_TIMEOUT_S):
            stop_proc(proc_inner)
            return False
        if ctx_val <= _NEEDLE_MIN_CTX:
            stop_proc(proc_inner)
            return True  # stability sufficient at small contexts
        needle_ok = _probe_needle(PORT, ctx_val)
        stop_proc(proc_inner)
        if not needle_ok:
            print(f"{indent}  → stable but incoherent at {ctx_val:,} tokens (needle failed)")
        return needle_ok

    if ok:
        print(f"{indent}  → stable  [{elapsed:.1f}s]")
        if predicted >= hi:
            # Predicted == ceiling — verify quality then return
            print(f"{indent}[ctx-quality] verifying coherence at {predicted:,} tokens...")
            wait_cool(); stop_proc(None); time.sleep(1)
            wait_cool(); stop_proc(None); time.sleep(1)
            proc_q = launch_server(model_path, llama_server, {**base_params, "num_ctx": predicted})
            if proc_q is not None:
                needle_ok = (predicted <= _NEEDLE_MIN_CTX) or _probe_needle(PORT, predicted)
                stop_proc(proc_q)
                if needle_ok:
                    print(f"{indent}  → coherent ✓")
                    return predicted
                print(f"{indent}  → incoherent at ceiling, searching down...")
                # Fall through to downward search from predicted
                best = 0
                dn_lo, dn_hi = lo, predicted - 4096
                while dn_lo <= dn_hi:
                    mid = ((dn_lo + dn_hi) // 2 // 4096) * 4096
                    mid = max(mid, dn_lo)
                    print(f"{indent}[ctx-search] trying {mid:,} tokens (quality-down)...")
                    if _stable_and_coherent(mid):
                        print(f"{indent}  → coherent ✓")
                        best = mid; dn_lo = mid + 4096
                    else:
                        dn_hi = mid - 4096
                return best if best > 0 else lo
            return predicted  # couldn't relaunch for quality check, accept as-is

        # Search upward from predicted
        best = predicted
        up_lo, up_hi = predicted + 4096, hi
        while up_lo <= up_hi:
            mid = ((up_lo + up_hi) // 2 // 4096) * 4096
            mid = max(mid, up_lo)
            print(f"{indent}[ctx-search] trying {mid:,} tokens (up)...")
            wait_cool(); stop_proc(None); time.sleep(1)
            t0 = time.time()
            if _probe_stable(model_path, llama_server, {**base_params, "num_ctx": mid}):
                print(f"{indent}  → stable  [{time.time()-t0:.1f}s]")
                best = mid; up_lo = mid + 4096
            else:
                print(f"{indent}  → failed  [{time.time()-t0:.1f}s]")
                up_hi = mid - 4096
        # Verify quality at the highest stable ctx found
        print(f"{indent}[ctx-quality] verifying coherence at {best:,} tokens...")
        while best >= lo:
            wait_cool(); stop_proc(None); time.sleep(1)
            wait_cool(); stop_proc(None); time.sleep(1)
            proc_q = launch_server(model_path, llama_server, {**base_params, "num_ctx": best})
            if proc_q is None:
                best -= 4096; continue
            needle_ok = (best <= _NEEDLE_MIN_CTX) or _probe_needle(PORT, best)
            stop_proc(proc_q)
            if needle_ok:
                print(f"{indent}  → coherent ✓")
                return best
            print(f"{indent}  → incoherent at {best:,}, stepping down...")
            best -= 4096
        return lo

    else:
        print(f"{indent}  → failed  [{elapsed:.1f}s]")
        # Search downward: lo=lo, hi=predicted-4096
        best = 0
        dn_lo, dn_hi = lo, predicted - 4096
        if dn_hi < dn_lo:
            return lo
        while dn_lo <= dn_hi:
            mid = ((dn_lo + dn_hi) // 2 // 4096) * 4096
            mid = max(mid, dn_lo)
            print(f"{indent}[ctx-search] trying {mid:,} tokens (down)...")
            wait_cool(); stop_proc(None); time.sleep(1)
            t0 = time.time()
            if _probe_stable(model_path, llama_server, {**base_params, "num_ctx": mid}):
                print(f"{indent}  → stable  [{time.time()-t0:.1f}s]")
                best = mid; dn_lo = mid + 4096
            else:
                print(f"{indent}  → failed  [{time.time()-t0:.1f}s]")
                dn_hi = mid - 4096
        if best == 0:
            return lo
        # Verify quality at best stable ctx
        print(f"{indent}[ctx-quality] verifying coherence at {best:,} tokens...")
        while best >= lo:
            wait_cool(); stop_proc(None); time.sleep(1)
            wait_cool(); stop_proc(None); time.sleep(1)
            proc_q = launch_server(model_path, llama_server, {**base_params, "num_ctx": best})
            if proc_q is None:
                best -= 4096; continue
            needle_ok = (best <= _NEEDLE_MIN_CTX) or _probe_needle(PORT, best)
            stop_proc(proc_q)
            if needle_ok:
                print(f"{indent}  → coherent ✓")
                return best
            print(f"{indent}  → incoherent at {best:,}, stepping down...")
            best -= 4096
        return lo


def get_vram_mb() -> float:
    """Total VRAM used across all GPUs in MB."""
    if not HAS_NVML:
        return 0.0
    try:
        return sum(
            pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).used
            for i in range(_nvml_device_count())
        ) / (1024 * 1024)
    except Exception:
        return 0.0


def get_vram_total_mb() -> float:
    """Total VRAM across all GPUs in MB."""
    if not HAS_NVML:
        return 0.0
    try:
        return sum(
            pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).total
            for i in range(_nvml_device_count())
        ) / (1024 * 1024)
    except Exception:
        return 0.0


def get_vram_used_per_gpu() -> list:
    """Returns list of (name, used_mb, total_mb) per GPU."""
    if not HAS_NVML:
        return []
    out = []
    try:
        for i in range(_nvml_device_count()):
            h    = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            out.append((name, info.used / (1024*1024), info.total / (1024*1024)))
    except Exception:
        pass
    return out


def get_gpu_temp() -> float:
    if not HAS_NVML:
        return 0.0
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
    except Exception:
        return 0.0


def get_ram_info() -> tuple:
    """Returns (total_mb, available_mb, used_pct)."""
    vm = psutil.virtual_memory()
    return (
        vm.total     / (1024 * 1024),
        vm.available / (1024 * 1024),
        vm.percent / 100.0,
    )


def get_ram_budget_mb() -> float:
    """RAM available for use: total * (1 - RAM_SAFETY_PCT)."""
    total_mb, _, _ = get_ram_info()
    return total_mb * (1.0 - RAM_SAFETY_PCT)


# ── model case classification ─────────────────────────────────────────────────

def model_size_mb(model_path: Path) -> float:
    """Return total size in MB. Sums all shards for sharded models."""
    try:
        _sm = _re.search(r'-(\d{5})-of-(\d{5})$', model_path.stem, _re.IGNORECASE)
        if _sm:
            base = model_path.stem[:_sm.start()]
            total = sum(
                s.stat().st_size
                for s in model_path.parent.glob(f"{base}-*-of-{_sm.group(2)}.gguf")
            )
            return total / (1024 * 1024) if total > 0 else model_path.stat().st_size / (1024 * 1024)
        return model_path.stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0


def classify_model(model_path: Path, gpu0_vram_gb: float, gpu1_vram_gb: float) -> str:
    """
    Return 'A', 'B', 'C', or 'D' based on whether the model fits in VRAM:
      A — both GPUs independently
      B — GPU0 only (not GPU1)
      C — combined VRAM only (not either GPU alone)
      D — not even combined VRAM (needs CPU offload)

    For MoE models all experts must be loaded regardless of how many activate
    per token, so VRAM classification uses total file size as for dense models.

    VRAM overhead scales with model size: larger models need more KV cache
    and compute buffer headroom. Leave 3% headroom on combined VRAM.
    """
    size_mb   = model_size_mb(model_path)
    size_gb   = size_mb / 1024
    # Scale overhead: 2 GB for small models, up to 5 GB for 70B+
    overhead  = min(5120, max(2048, int(size_gb * 0.15 * 1024)))
    estimated = size_mb + overhead
    gpu0_mb     = gpu0_vram_gb * 1024
    gpu1_mb     = gpu1_vram_gb * 1024
    combined_mb = gpu0_mb + gpu1_mb

    if estimated <= gpu1_mb:
        return "A"
    if estimated <= gpu0_mb:
        return "B"
    if estimated <= combined_mb * 0.97:
        return "C"
    return "D"


# ── topology scenario definitions ─────────────────────────────────────────────

def _single_gpu_scenarios(gpu0_vram: float, gpu1_vram: float) -> list:
    """Scenarios for Case A (both fit) and Case B (GPU0 only).
    Uses detected GPU names and indices from _GPU_INFO.
    """
    scenarios = []
    for i, g in enumerate(_GPU_INFO):
        sid   = f"gpu{i}_only"
        label = f"GPU{i} only — {g['name']} ({g['vram_gb']:.0f} GB)"
        scenarios.append((
            sid, label,
            {"cuda_visible_devices": str(g["index"]),
             "tensor_split": None, "main_gpu": None, "numa": None},
        ))
    return scenarios


def _split_scenarios(gpu0_vram: float, gpu1_vram: float) -> list:
    """
    Scenarios for Case C (must use both GPUs).

    tensor_split values are ordered by CUDA device index (physical PCIe order),
    NOT by our sorted _GPU_INFO order. We build a per-CUDA-index split array
    so each card gets the right share regardless of which physical slot it's in.

    main_gpu is set to the largest card's CUDA index so KV cache and scratch
    buffers land on the card with the most headroom.
    """
    # Build a CUDA-index-ordered list of VRAM values.
    # _GPU_INFO is sorted by VRAM desc; each entry has the real CUDA index.
    # We need an array indexed by CUDA device number for tensor_split.
    if not _GPU_INFO:
        return []

    n_cuda = max(g["index"] for g in _GPU_INFO) + 1
    cuda_vram = [0.0] * n_cuda
    for g in _GPU_INFO:
        cuda_vram[g["index"]] = g["vram_gb"]

    total = sum(cuda_vram)
    _main_gpu_idx = _GPU_INFO[0]["index"]  # largest card's CUDA index

    def _ts(shares):
        # shares is indexed by CUDA device; normalise to sum=10 and format
        s = sum(shares)
        return ",".join(str(round(v / s * 10, 2)) for v in shares)

    # Proportional by VRAM
    prop = cuda_vram[:]

    # Largest-card-heavy: give 80% to the largest card, rest proportional
    heavy = [v / total * 2 for v in cuda_vram]
    heavy[_main_gpu_idx] = total / total * 8  # 80% to main

    # KV-aware: main_gpu carries KV overhead, shift 15% of its share to others
    kv = cuda_vram[:]
    kv_shift = kv[_main_gpu_idx] * 0.15
    kv[_main_gpu_idx] -= kv_shift
    # distribute the shift equally to other cards
    others = [i for i in range(n_cuda) if i != _main_gpu_idx]
    if others:
        for i in others:
            kv[i] += kv_shift / len(others)

    _g0name = _GPU_INFO[0]["name"]
    return [
        ("split_prop",
         f"Split proportional by VRAM",
         {"cuda_visible_devices": None,
          "tensor_split": _ts(prop), "main_gpu": _main_gpu_idx, "numa": None}),
        ("split_equal",
         "Split equal (50/50)",
         {"cuda_visible_devices": None,
          "tensor_split": "1,1", "main_gpu": _main_gpu_idx, "numa": None}),
        ("split_g0heavy",
         f"Split {_g0name}-heavy (~80%)",
         {"cuda_visible_devices": None,
          "tensor_split": _ts(heavy), "main_gpu": _main_gpu_idx, "numa": None}),
        ("split_kv_aware",
         f"Split KV-aware (shift weight off {_g0name})",
         {"cuda_visible_devices": None,
          "tensor_split": _ts(kv), "main_gpu": _main_gpu_idx, "numa": None}),
    ]


NUMA_SCENARIOS = [
    ("numa_none",
     "No NUMA policy (OS default)",
     {"numa": None}),
    ("numa_dist",
     "NUMA distribute (both sockets, spread allocation)",
     {"numa": "distribute"}),
    ("numa_iso",
     "NUMA isolate (socket 0 only, lower latency)",
     {"numa": "isolate"}),
]

# ── minimal base config for topology / ctx probe servers ──────────────────────
_PROBE_BASE = {
    "num_gpu_layers": 99,
    "num_ctx":        8192,
    "batch_size":     2048,
    "ubatch_size":    512,
    "num_threads":    8,
    "threads_batch":  0,
    "flash_attn":     "auto",
    "cache_type_k":   "f16",
    "cache_type_v":   "f16",
    "prio":           0,
    "poll":           50,
    # topology fields — overridden per scenario
    "tensor_split":         None,
    "main_gpu":             None,
    "cuda_visible_devices": None,
    "numa":                 None,
    # ctx-sweep flags
    "kv_offload":     True,
    # Limit to 1 parallel slot so KV cache isn't multiplied by 4 (default).
    # n_parallel=4 adds ~10GB KV for large models, causing silent OOM.
    "n_parallel":     1,
}

# ── speed prompts ─────────────────────────────────────────────────────────────
_SPEED_PROMPT = {
    "messages": [
        {"role": "system",
         "content": "You are a senior Python developer. Write clean, efficient code."},
        {"role": "user",
         "content": "Write a Python function that checks if a number is prime. "
                    "Include a docstring and handle edge cases."},
    ],
    "max_tokens": 150,
}

def _get_large_prompt(max_ctx: int = 8192) -> dict:
    """Build a large prompt scaled to fit within max_ctx.
    Targets 90% of ctx with measured constants (37 tok/sentence, 95 tok overhead)
    so tokenizer variance never pushes us over the limit.
    """
    OUTPUT_TOKENS     = 500
    OVERHEAD_TOKENS   = 95   # system prompt + framing text, measured
    TOKENS_PER_SENT   = 37   # measured from actual sentence char lengths
    FILL_RATIO        = 0.90 # 10% real headroom
    input_budget = int(max_ctx * FILL_RATIO) - OUTPUT_TOKENS - OVERHEAD_TOKENS
    n_sentences  = max(5, min(input_budget // TOKENS_PER_SENT, 200))

    topics = ["temperature conversion", "unit testing", "error handling",
              "input validation", "batch processing", "API design",
              "documentation", "type annotations", "edge cases", "performance"]
    sentences = [
        f"Requirement {i+1}: The {topics[i % len(topics)]} module must handle "
        f"scenario {i+1} correctly with proper logging."
        for i in range(n_sentences)
    ]
    return {
        "messages": [
            {"role": "system",
             "content": "You are a senior Python developer. Write clean, well-tested code."},
            {"role": "user",
             "content": "Implement a complete temperature converter module based on "
                        "these requirements:\n\n" + "\n".join(sentences)
                        + "\n\nWrite the full implementation."},
        ],
        "max_tokens": OUTPUT_TOKENS,
    }


def _build_needle_prompt(ctx: int) -> tuple[dict, str]:
    """
    Build a needle-in-haystack prompt scaled to ctx tokens.
    Hides a unique numeric secret near the START of a long filler passage,
    then asks for it at the end — requiring genuine long-range attention.
    Returns (prompt_dict, expected_answer).
    """
    import random as _random
    _random.seed(ctx)  # deterministic per context size

    # Generate a secret that won't appear by chance in filler text
    secret = str(_random.randint(10000, 99999)) + "-XYZQ"
    expected = secret.split("-")[0]  # just the numeric part to check

    # Token budget: fill 80% of ctx, reserve 50 for output + 100 overhead
    OUTPUT_TOKENS  = 50
    OVERHEAD       = 150
    WORDS_PER_TOK  = 0.75   # conservative: ~0.75 words per token for English prose
    fill_tokens    = max(200, int(ctx * 0.70) - OUTPUT_TOKENS - OVERHEAD)  # conservative: tokenizer variance
    fill_words     = int(fill_tokens * WORDS_PER_TOK)

    # Filler sentences — generic prose that stays under radar
    filler_topics = [
        "The weather system moved slowly across the plains bringing rain.",
        "Scientists have studied this phenomenon for several decades now.",
        "The committee reviewed the proposal and requested additional data.",
        "Market conditions remained stable throughout the reporting period.",
        "Engineers designed the system to handle peak loads efficiently.",
        "The research team published their findings in a peer-reviewed journal.",
        "Local authorities responded quickly to the infrastructure concern.",
        "The software update addressed several performance bottlenecks.",
    ]
    # Build filler by repeating sentences until we hit word budget
    filler_sentences = []
    word_count = 0
    i = 0
    while word_count < fill_words:
        s = filler_topics[i % len(filler_topics)]
        filler_sentences.append(s)
        word_count += len(s.split())
        i += 1

    # Place the needle after the first sentence (far from the query at the end)
    needle = f"Important notice: the secret code is {secret}. Remember this code."
    filler_sentences.insert(1, needle)
    passage = " ".join(filler_sentences)

    prompt = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user",   "content":
                f"{passage}\n\n"
                f"Based on the text above, what is the secret code number "
                f"(just the 5 digits before the dash)?"},
        ],
        "max_tokens": OUTPUT_TOKENS,
        "temperature": 0.0,
        "seed": 42,
    }
    return prompt, expected


def _probe_needle(port: int, ctx: int, timeout: int = 120) -> bool:
    """
    Send a needle-in-haystack probe to the already-running server.
    Returns True if the model correctly retrieves the hidden secret.
    The server must already be running — does NOT launch or stop it.
    """
    prompt, expected = _build_needle_prompt(ctx)
    try:
        r = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={**prompt, "model": "test", "stream": False,
                  "chat_template_kwargs": {"enable_thinking": False}},
            timeout=timeout,
        )
        if r.status_code != 200:
            return False
        content = r.json()["choices"][0]["message"]["content"]
        found = expected in content
        if not found:
            print(f"      [needle] FAIL — expected '{expected}' not found in: {content[:120]!r}")
        return found
    except Exception as e:
        print(f"      [needle] error: {e}")
        return False


# ── low-level server helpers ──────────────────────────────────────────────────

def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def kill_competing_processes():
    """Kill Ollama and LM Studio to free VRAM and GPU context before benchmarking."""
    if IS_WINDOWS:
        for proc in ["ollama.exe", "ollama_llama_server.exe", "LM Studio.exe"]:
            subprocess.run(["taskkill", "/F", "/IM", proc], capture_output=True)
    else:
        subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
        subprocess.run(["pkill", "-f", "lm-studio"], capture_output=True)


def kill_server():
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                       capture_output=True)
    else:
        subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    for _ in range(20):
        if not is_port_open(PORT):
            return
        time.sleep(0.5)


def wait_cool(threshold: int = 82, target: int = 72):
    temp = get_gpu_temp()
    if temp > threshold:
        print(f"      Cooling... ({temp:.0f}°C → waiting for {target}°C)")
        while get_gpu_temp() > target:
            time.sleep(3)


def stop_proc(proc):
    """Terminate a server subprocess and kill any lingering llama-server."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    kill_server()


def pause_between_models(seconds: int = 5) -> bool:
    """
    Display a countdown and return True if the run should continue, False if
    the user pressed 'n' (or 'N') within *seconds* seconds.

    On Windows uses msvcrt for non-blocking key reads (no Enter required).
    On POSIX uses select() on stdin with raw terminal mode.
    If stdin is not a real tty (e.g. piped / redirected) the pause is skipped
    and the run continues automatically.

    The countdown line uses \r overwrites and is written directly to the real
    console (bypassing the log-file tee) so the log isn't polluted with \r junk.
    """
    import sys

    # Non-interactive stdin (piped log capture, etc.) — skip silently
    if not sys.stdin.isatty():
        return True

    # _console is the real stdout before any _Tee wrapping — use it for the
    # \r countdown so the log file only gets the clean start/end lines.
    _console = sys.stdout
    while hasattr(_console, "_s"):      # unwrap _Tee layers
        _console = _console._s[0]

    print(f"\n  ── Press 'n' within {seconds}s to stop after this model, "
          f"any other key or wait to continue ──")

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
                    print("  Stopping after this model — compiling report...")
                    return False
                return True
            time.sleep(0.1)
    else:
        import select
        import tty
        import termios
        fd = sys.stdin.fileno()
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
                        print("  Stopping after this model — compiling report...")
                        return False
                    return True
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    _console.write("\r  Continuing...                    \n")
    _console.flush()
    return True


def _build_server_cmd(model_path: Path, llama_server: Path, params: dict) -> list:
    cmd = [
        str(llama_server), "-m", str(model_path),
        "--port", str(PORT), "--host", "127.0.0.1",
        "-ngl",  str(params.get("num_gpu_layers") or 99),
        "-c",    str(params.get("num_ctx", 8192)),
        "-b",    str(params.get("batch_size", 2048)),
        "-ub",   str(params.get("ubatch_size", 512)),
        "-t",    str(params.get("num_threads", 8)),
        "-fa",   str(params.get("flash_attn", "auto")),
        "-ctk",  params.get("cache_type_k", "f16"),
        "-ctv",  params.get("cache_type_v", "f16"),
        "--prio", str(params.get("prio", 0)),
        "--poll", str(params.get("poll", 50)),
        "--no-warmup", "--jinja",
    ]
    if params.get("tensor_split"):
        cmd.extend(["--tensor-split", str(params["tensor_split"])])
    if params.get("main_gpu") is not None:
        cmd.extend(["--main-gpu", str(params["main_gpu"])])
    if params.get("numa") and params["numa"] != "none":
        cmd.extend(["--numa", params["numa"]])
    if not params.get("kv_offload", True):
        cmd.append("-nkvo")
    if not params.get("fit", True):
        cmd.extend(["--fit", "off"])
    if params.get("n_parallel") is not None:
        cmd.extend(["-np", str(params["n_parallel"])])
    return cmd


def _startup_timeout_for(model_path: Path) -> int:
    """Scale server startup timeout with model file size.
    Uses total shard size for sharded models.
    Overridable via LLM_OPT_STARTUP_TIMEOUT env var.
    """
    env_override = os.environ.get("LLM_OPT_STARTUP_TIMEOUT")
    if env_override:
        return int(env_override)
    try:
        from model_utils import model_size_mb as _msz
        size_gb = _msz(model_path) / 1024
    except Exception:
        try:
            size_gb = model_path.stat().st_size / (1024 ** 3)
        except OSError:
            size_gb = 0.0
    t = 30 + size_gb * 4
    if is_moe_model(model_path) or size_gb > 20:
        t *= 1.5
    return min(600, int(t))


# Patterns that indicate llama-server is actively making progress loading the model
_LOADING_PROGRESS_RE = _re.compile(
    r"llm_load|llama_model_load|llama_new_context|ggml_|slot|build info"
    r"|vocab|token|layer|tensor|kv cache|model size|n_ctx|n_batch"
    r"|CUDA|cuBLAS|ROCm|Metal|Vulkan|backend",
    _re.IGNORECASE,
)
# Patterns that are definitive hard failures — no point waiting further
_HARD_FAIL_RE = _re.compile(
    r"failed to load model"
    r"|error loading model"
    r"|CUDA error"
    r"|out of memory"
    r"|cudaMalloc failed"
    r"|GGML_ASSERT"
    r"|Segmentation fault"
    r"|Access violation",
    _re.IGNORECASE,
)
# Patterns that indicate loading is complete and the server is ready
_SERVER_READY_RE = _re.compile(
    r"server is listening"
    r"|all slots are idle"
    r"|model loaded",
    _re.IGNORECASE,
)
# Stall timeout: if no new stderr line for this many seconds, assume hung
_STALL_TIMEOUT_S = 45


def launch_server(model_path: Path, llama_server: Path, params: dict):
    """
    Start llama-server with the given params.
    Returns subprocess.Popen on success, None on failure.

    Uses real-time stderr monitoring instead of a fixed wall-clock timeout:
    - Tracks the timestamp of the last progress line from llama-server
    - Extends the deadline as long as output keeps coming (model still loading)
    - Aborts immediately on hard-failure signatures (CUDA error, failed to load, etc.)
    - Times out only if output stalls for _STALL_TIMEOUT_S seconds
    - Prints a live status line so the user can see what stage loading is at
    """
    global _last_launch_was_crash
    _last_launch_was_crash = False

    cmd = _build_server_cmd(model_path, llama_server, params)
    if str(model_path) in _NO_JINJA_MODELS and "--jinja" in cmd:
        idx = cmd.index("--jinja")
        cmd[idx:idx+1] = ["--no-jinja", "--chat-template", "chatml"]

    env = os.environ.copy()
    bin_dir = str(llama_server.parent)
    env["PATH"] = bin_dir + (";" if IS_WINDOWS else ":") + env.get("PATH", "")
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    cvd = params.get("cuda_visible_devices")
    if cvd is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cvd)
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)

    def _do_launch(cmd_to_run):
        """Spawn the process and return (proc, stderr_lines_list, stdout_lines_list)."""
        try:
            proc = subprocess.Popen(
                cmd_to_run,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=bin_dir, env=env,
            )
        except Exception as e:
            print(f"      Launch failed: {e}")
            return None, [], []

        stdout_lines, stderr_lines = [], []

        def _drain(pipe, buf):
            try:
                for raw in pipe:
                    buf.append(raw.decode("utf-8", errors="replace"))
            except Exception:
                pass

        threading.Thread(target=_drain, args=(proc.stdout, stdout_lines), daemon=True).start()
        threading.Thread(target=_drain, args=(proc.stderr, stderr_lines), daemon=True).start()
        return proc, stderr_lines, stdout_lines

    def _monitor(proc, stderr_lines, stdout_lines, label=""):
        """
        Monitor a running llama-server process until it is ready, fails, or stalls.

        Returns one of:
            "ok"      — server is up and responding
            "crash"   — process exited with a hard error
            "stall"   — no output for _STALL_TIMEOUT_S seconds (hung)
            "jinja"   — jinja template parse error detected
        """
        last_progress_t = time.time()
        last_seen_idx   = 0        # how many stderr lines we've already processed
        last_status     = ""       # last loading-stage message shown to user

        while True:
            # ── Check /health first (fastest path) ──────────────────────────
            try:
                r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return "ok"
            except Exception:
                pass

            # ── Process any new stderr lines ─────────────────────────────────
            new_lines = stderr_lines[last_seen_idx:]
            last_seen_idx += len(new_lines)

            for line in new_lines:
                line_s = line.rstrip()
                if not line_s:
                    continue

                # Any output = still alive and making progress
                last_progress_t = time.time()

                # Definitive ready signal in stderr
                if _SERVER_READY_RE.search(line_s):
                    return "ok"

                # Hard failure
                if _HARD_FAIL_RE.search(line_s):
                    _last_launch_was_crash = True
                    print(f"      Hard failure: {line_s.strip()}")
                    return "crash"

                # Jinja error
                if ("chat template parsing error" in line_s or
                        "Unable to generate parser" in line_s):
                    return "jinja"

                # Progress line — update live status (truncated to 72 chars)
                if _LOADING_PROGRESS_RE.search(line_s) and VERBOSE:
                    status = line_s.strip()[:72]
                    if status != last_status:
                        print(f"      Loading: {status}")
                        last_status = status

            # ── Check for stdout progress lines too ──────────────────────────
            # (llama-server sometimes emits model-load info on stdout)
            for line in stdout_lines[max(0, last_seen_idx - len(new_lines)):]:
                if _SERVER_READY_RE.search(line):
                    return "ok"
                if _HARD_FAIL_RE.search(line):
                    _last_launch_was_crash = True
                    return "crash"

            # ── Check if process has exited ──────────────────────────────────
            if proc.poll() is not None:
                full = "".join(stderr_lines) + "".join(stdout_lines)
                if "chat template parsing error" in full or "Unable to generate parser" in full:
                    return "jinja"
                _last_launch_was_crash = True
                return "crash"

            # ── Stall detection ──────────────────────────────────────────────
            stall_s = time.time() - last_progress_t
            if stall_s > _STALL_TIMEOUT_S:
                print(f"      Stalled — no output for {stall_s:.0f}s")
                return "stall"

            time.sleep(1)

    def _try_launch(cmd_to_run, attempt_label=""):
        """Launch and monitor, handling jinja retry internally."""
        proc, stderr_lines, stdout_lines = _do_launch(cmd_to_run)
        if proc is None:
            return None

        result = _monitor(proc, stderr_lines, stdout_lines, label=attempt_label)

        if result == "ok":
            if VERBOSE:
                print(f"      Model loaded successfully")
            return proc

        if result == "jinja" and "--jinja" in cmd_to_run:
            # Full output for debugging
            print(f"      Jinja template error — retrying without --jinja...")
            stop_proc(proc)
            cmd_nj = [c for c in cmd_to_run if c != "--jinja"] + ["--no-jinja", "--chat-template", "chatml"]
            proc2, stderr2, stdout2 = _do_launch(cmd_nj)
            if proc2 is None:
                return None
            result2 = _monitor(proc2, stderr2, stdout2, label="no-jinja retry")
            if result2 == "ok":
                _NO_JINJA_MODELS.add(str(model_path))
                if VERBOSE:
                    print(f"      Model loaded successfully (no-jinja)")
                return proc2
            stop_proc(proc2)
            full2 = "".join(stderr2)
            if full2.strip():
                print(f"      Retry output: ...{full2[-300:]}")
            return None

        if result in ("crash", "stall"):
            full = "".join(stderr_lines) + "".join(stdout_lines)
            if result == "crash":
                tail = "".join(stderr_lines)[-400:].strip()
                if tail:
                    print(f"      Crashed on startup: ...{tail}")
            stop_proc(proc)
            return None

        stop_proc(proc)
        return None

    return _try_launch(cmd)


def _query_once(prompt: dict, timeout: int = 300) -> bool:
    """
    Send a single inference request.
    Returns True if it completed successfully, False on any error / HTTP 5xx.
    """
    try:
        r = requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json={**prompt, "model": "test", "stream": False,
                  "seed": 42, "temperature": 0.7,
                  "chat_template_kwargs": {"enable_thinking": False}},
            timeout=timeout,
        )
        return r.status_code == 200
    except Exception:
        return False


def _get_actual_server_ctx(fallback: int = 8192) -> int:
    """Query the running server's actual n_ctx.
    Tries /slots, then /props, then an oversized-input probe.
    Returns fallback if all fail.
    """
    try:
        r = requests.get(f"http://127.0.0.1:{PORT}/slots", timeout=5)
        if r.status_code == 200:
            slots = r.json()
            if isinstance(slots, list) and slots:
                v = slots[0].get("n_ctx") or slots[0].get("params", {}).get("n_ctx")
                if v:
                    return int(v)
    except Exception:
        pass
    try:
        r = requests.get(f"http://127.0.0.1:{PORT}/props", timeout=5)
        if r.status_code == 200:
            d = r.json()
            for key in ("n_ctx", "ctx_size"):
                if key in d:
                    return int(d[key])
    except Exception:
        pass
    try:
        oversized_content = "word " * 65000
        probe = {"model": "test", "stream": False, "seed": 42,
                 "max_tokens": 1,
                 "messages": [{"role": "user", "content": oversized_content}],
                 "chat_template_kwargs": {"enable_thinking": False}}
        r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                          json=probe, timeout=15)
        if r.status_code == 400:
            v = r.json().get("error", {}).get("n_ctx")
            if v:
                return int(v)
    except Exception:
        pass
    return fallback


def bench_speed(n_runs: int = 3) -> dict | None:
    """
    Benchmark the running server with speed + large prompts.
    Returns metrics dict or None if all queries failed.
    """
    # Warmup (ignore result)
    _query_once(_SPEED_PROMPT, timeout=120)

    gen_l, prompt_l, ttft_l = [], [], []
    for _ in range(n_runs):
        try:
            r = requests.post(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                json={**_SPEED_PROMPT, "model": "test", "stream": False,
                      "seed": 42, "temperature": 0.7,
                      "chat_template_kwargs": {"enable_thinking": False}},
                timeout=300,
            )
            if r.status_code != 200:
                continue
            t = r.json().get("timings", {})
            _pn = t.get("predicted_n", 0); _pm = t.get("predicted_ms", 0)
            gen_l.append((_pn / _pm * 1000.0) if _pn > 0 and _pm > 0
                         else t.get("predicted_per_second", 0))
            prompt_l.append(t.get("prompt_per_second", 0))
            ttft_l.append(t.get("prompt_ms", 0))
        except Exception:
            continue

    if not gen_l:
        return None

    # Query the server's actual n_ctx before sizing the large prompt — models
    # with a hard-capped context will lie about what we passed via -c.
    _actual_ctx = _get_actual_server_ctx(8192)
    lp = _get_large_prompt(max_ctx=_actual_ctx)
    lg_gen, lg_ttft = [], []
    for _ in range(max(1, n_runs - 1)):
        try:
            r = requests.post(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                json={**lp, "model": "test", "stream": False,
                      "seed": 42, "temperature": 0.7,
                      "chat_template_kwargs": {"enable_thinking": False}},
                timeout=300,
            )
            if r.status_code != 200:
                continue
            t = r.json().get("timings", {})
            _pn = t.get("predicted_n", 0); _pm = t.get("predicted_ms", 0)
            lg_gen.append((_pn / _pm * 1000.0) if _pn > 0 and _pm > 0
                          else t.get("predicted_per_second", 0))
            lg_ttft.append(t.get("prompt_ms", 0))
        except Exception:
            continue

    return {
        "gen_tps":       round(median(gen_l),    2),
        "prompt_tps":    round(median(prompt_l),  2),
        "ttft_ms":       round(median(ttft_l),    1),
        "gen_tps_long":  round(median(lg_gen),    2) if lg_gen  else 0.0,
        "ttft_long_ms":  round(median(lg_ttft),   1) if lg_ttft else 0.0,
        "vram_mb":       round(get_vram_mb(),      0),
        "vram_total_mb": round(get_vram_total_mb(), 0),
    }


def composite_score(m: dict) -> float:
    """Composite score matching the optimizer's compute_score weights."""
    gen   = m.get("gen_tps",      0)
    genL  = m.get("gen_tps_long", 0)
    ttft  = max(m.get("ttft_ms",      1), 1)
    ttftL = max(m.get("ttft_long_ms", 1), 1)
    vram_used  = m.get("vram_mb",       0)
    vram_total = m.get("vram_total_mb", 0) or get_vram_total_mb()
    scale = max(gen, genL, 1.0)
    ts  = min(100.0 / ttft,  1.5) * scale
    tsL = min(500.0 / ttftL, 1.5) * scale
    vram_score = 0.0
    if vram_total > 0 and vram_used > 0:
        vram_score = (1.0 - min(vram_used / vram_total, 1.0)) * scale * 0.10
    if genL <= 0:
        return gen * 0.55 + ts * 0.25 + vram_score
    return gen * 0.35 + genL * 0.25 + ts * 0.15 + tsL * 0.15 + vram_score


# ── single scenario runner ────────────────────────────────────────────────────

def _run_scenario(
    sid: str,
    label: str,
    overlay: dict,
    stype: str,
    model_path: Path,
    llama_server: Path,
    topo_runs: int,
    all_results: list,
    base_overlay: dict | None = None,
) -> dict:
    """
    Launch server with (base_overlay | overlay), benchmark, stop, append result.
    Returns the result dict.
    """
    print(f"\n  [{sid}]  {label}")
    wait_cool()
    stop_proc(None)
    time.sleep(1)

    t_scenario = Timer(f"scenario {sid}", silent=True)
    params = {**_PROBE_BASE, **(base_overlay or {}), **overlay}
    proc   = launch_server(model_path, llama_server, params)

    if proc is None:
        elapsed = t_scenario.elapsed()
        print(f"      FAILED to start  [{_fmt_s(elapsed)}]")
        r = {"scenario": sid, "label": label, "type": stype,
             "status": "failed", "score": 0.0, "params_overlay": overlay}
        all_results.append(r)
        return r

    metrics = bench_speed(topo_runs)
    stop_proc(proc)
    elapsed = t_scenario.elapsed()

    if metrics is None:
        print(f"      FAILED: all queries failed  [{_fmt_s(elapsed)}]")
        r = {"scenario": sid, "label": label, "type": stype,
             "status": "query_failed", "score": 0.0, "params_overlay": overlay}
        all_results.append(r)
        return r

    score = composite_score(metrics)
    # Large-prompt OOM/timeout means the model is unreliable under load on
    # this GPU. Mark degraded and penalise score so it only wins as last resort.
    _large_failed = metrics.get("gen_tps_long", 0) == 0.0
    _status = "degraded" if _large_failed else "ok"
    if _large_failed:
        score *= 0.25
    _degraded_note = "  [DEGRADED: large prompt OOM]" if _large_failed else ""
    print(f"      Gen: {metrics['gen_tps']:.1f} t/s  "
          f"Large: {metrics.get('gen_tps_long', 0):.1f} t/s  "
          f"TTFT: {metrics['ttft_ms']:.0f} ms  "
          f"VRAM: {metrics['vram_mb']:.0f} MB  "
          f"Score: {score:.1f}  [{_fmt_s(elapsed)}]{_degraded_note}")

    r = {"scenario": sid, "label": label, "type": stype,
         "status": _status, "score": round(score, 2),
         "params_overlay": overlay, **metrics}
    all_results.append(r)
    return r


# ── binary-search helpers ─────────────────────────────────────────────────────

_PROBE_PROMPT = {
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 5,
}


def _probe_stable(
    model_path: Path,
    llama_server: Path,
    params: dict,
) -> bool:
    """
    Launch server with params, send a minimal 5-token probe to verify it
    responds without crash.  No large-prompt fill, no benchmark — just a
    health check sufficient to confirm the context size is stable.
    Returns True if the probe succeeds, False otherwise.
    Always stops the server before returning.
    """
    t = Timer(silent=True)
    proc = launch_server(model_path, llama_server, params)
    if proc is None:
        return False

    ok = _query_once(_PROBE_PROMPT, timeout=PROBE_TIMEOUT_S)
    stop_proc(proc)
    _ = t.elapsed()   # time is tracked by caller if needed
    return ok


def _binary_search_ctx(
    model_path: Path,
    llama_server: Path,
    base_params: dict,
    lo: int,
    hi: int,
    step_coarse: int = 4096,
    step_fine:   int = 1024,
) -> int:
    """
    Binary search the maximum stable num_ctx between lo and hi.
    Uses coarse steps first then fine steps around the bracket.
    Returns the highest stable ctx found, or lo if even lo fails.
    """
    # Align to coarse step
    lo = (lo // step_coarse) * step_coarse
    hi = (hi // step_coarse) * step_coarse
    hi = max(lo, hi)

    # ── coarse pass ──────────────────────────────────────────────────────
    best = 0
    lo_c, hi_c = lo, hi
    while lo_c <= hi_c:
        mid = ((lo_c + hi_c) // 2 // step_coarse) * step_coarse
        mid = max(mid, lo_c)
        print(f"      [ctx-search] trying {mid:,} tokens (coarse)...")
        params = {**base_params, "num_ctx": mid}
        wait_cool()
        stop_proc(None)
        time.sleep(1)
        t_probe = time.time()
        if _probe_stable(model_path, llama_server, params):
            print(f"        → stable  [{time.time()-t_probe:.1f}s]")
            best  = mid
            lo_c  = mid + step_coarse
        else:
            print(f"        → failed  [{time.time()-t_probe:.1f}s]")
            hi_c  = mid - step_coarse

    if best == 0:
        print(f"      [ctx-search] even lo={lo:,} failed")
        return lo  # caller will handle the degenerate case

    # ── fine pass: bracket [best, best + step_coarse) ────────────────────
    fine_lo = best
    fine_hi = best + step_coarse - step_fine
    fine_hi = min(fine_hi, hi)

    while fine_lo <= fine_hi:
        mid = ((fine_lo + fine_hi) // 2 // step_fine) * step_fine
        mid = max(mid, fine_lo)
        print(f"      [ctx-search] trying {mid:,} tokens (fine)...")
        params = {**base_params, "num_ctx": mid}
        wait_cool()
        stop_proc(None)
        time.sleep(1)
        t_probe = time.time()
        if _probe_stable(model_path, llama_server, params):
            print(f"        → stable  [{time.time()-t_probe:.1f}s]")
            best    = mid
            fine_lo = mid + step_fine
        else:
            print(f"        → failed  [{time.time()-t_probe:.1f}s]")
            fine_hi = mid - step_fine

    return best


def _read_total_layers(model_path: Path) -> int:
    """
    Read the actual number of transformer/SSM blocks from GGUF metadata.
    Returns 200 if the file can't be parsed — binary search will still find
    the correct value, just with a few extra probe steps at the high end.
    200 is safe for any foreseeable model (GPT-4 style dense = ~120 layers,
    DeepSeek-R1 671B = ~61 MoE layers).
    """
    meta = _cached_meta(model_path)
    # If parsing failed, _parse_error will be set — use safe ceiling
    if meta.get('_parse_error'):
        return 200
    n = meta.get('n_layers', 0)
    if n and n > 0:
        return int(n)
    return 200


def _binary_search_ngl(
    model_path: Path,
    llama_server: Path,
    base_params: dict,
) -> int:
    """
    Binary search the maximum num_gpu_layers that loads without OOM.
    Reads the actual layer count from GGUF metadata to set the search ceiling.
    Returns 0 if even 1 layer fails (pure CPU mode).

    Distinguishes hard crashes (incompatible model/arch — abort immediately)
    from OOM/timeout failures (VRAM issue — keep searching lower).
    A hard crash at any point in the search aborts immediately with 0
    rather than wasting time trying progressively fewer layers.
    """
    total_layers = _read_total_layers(model_path)
    lo, hi, best = 1, total_layers, 0
    print(f"      [ngl-search] model has {total_layers} layers")
    while lo <= hi:
        mid = (lo + hi) // 2
        print(f"      [ngl-search] trying ngl={mid}/{total_layers}...", end="", flush=True)
        params = {**base_params, "num_gpu_layers": mid, "num_ctx": 4096}
        wait_cool()
        stop_proc(None)
        time.sleep(1)
        if _probe_stable(model_path, llama_server, params):
            print(" OK")
            best = mid
            lo   = mid + 1
        else:
            if _last_launch_was_crash:
                print(" hard crash")
                print(f"      [ngl-search] hard crash at ngl={mid} — model incompatible "
                      f"with this llama-server build, aborting search")
                return 0
            print(" OOM/timeout")
            hi = mid - 1
    return best


# ── topology sweep ─────────────────────────────────────────────────────────────

def run_topo_sweep(
    model_path:         Path,
    llama_server:       Path,
    optimizer_script:   Path,
    gpu_filter:         list | None,
    force_numa:         bool,
    topo_runs:          int,
    skip_gpu_indices:   set | None = None,
    results_base:       Path | None = None,
) -> dict:
    """
    Classify the model (A/B/C/D) and run the appropriate topology scenarios.

    Case A  — fits both GPUs independently: test gpu0_only + gpu1_only
    Case B  — fits GPU0 only:              test gpu0_only only
    Case C  — needs combined VRAM:         test four split strategies
    Case D  — needs CPU offload:           binary-search max ngl, test NUMA

    Returns winner_overlay dict for the sidecar.

    Pass results_base to write directly to results_base/<slug>/topo_sweep/
    instead of the legacy LLM_Optimiser/results/<slug>/topo_sweep/ path.
    """
    _rbase = results_base if results_base is not None else optimizer_script
    out_dir = results_dir_for(model_path, _rbase) / "topo_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_topo = Timer("Topology sweep")

    case       = classify_model(model_path, GPU0_VRAM_GB, GPU1_VRAM_GB)
    size_mb    = model_size_mb(model_path)
    total_vram = get_vram_total_mb()

    _tmeta   = _cached_meta(model_path)
    _moe_tag = f"  MoE {_tmeta['n_expert']}x experts ({_tmeta['n_expert_used']} active)"                if _tmeta.get("is_moe") else ""
    _hyb_tag = f"  hybrid ({_tmeta['n_attn_layers']}/{_tmeta['n_layers']} attn layers)"                if _tmeta.get("is_hybrid") else ""
    print(f"\n{'='*70}")
    print(f"  TOPOLOGY SWEEP: {model_path.name}")
    print(f"  Arch  : {_tmeta.get('arch', 'unknown')}{_moe_tag}{_hyb_tag}")
    _overhead_mb = min(5120, max(2048, int(size_mb / 1024 * 0.15 * 1024)))
    print(f"  Size  : {size_mb/1024:.2f} GB  "
          f"(estimated VRAM needed: {(size_mb + _overhead_mb) / 1024:.2f} GB  "
          f"[{_overhead_mb/1024:.1f} GB overhead])")
    _gpu_strs = "  |  ".join(f"GPU{i} = {g['vram_gb']:.0f} GB ({g['name']})" for i, g in enumerate(_GPU_INFO))
    print(f"  {_gpu_strs}  |  Combined = {GPU0_VRAM_GB + GPU1_VRAM_GB:.0f} GB")
    print(f"  Classification: Case {case}  |  Runs per scenario: {topo_runs}")
    print(f"{'='*70}")

    all_results  = []
    max_fit_ngl  = 99   # will be updated for Case D
    winner_overlay = {}
    winner_label   = "default"

    # Kill Ollama/LM Studio before any server launches so they don't hold
    # the CUDA context and cause silent hangs
    kill_competing_processes()
    kill_server()
    time.sleep(2)

    # ── Case A: fits in both GPUs independently ────────────────────────────
    if case == "A":
        print(f"\n  Case A: model fits in both GPUs independently")
        print(f"  Testing single-GPU scenarios — picking the faster one")
        print(f"\n  ── Single-GPU scenarios ──")

        for _gi, (sid, label, overlay) in enumerate(
                _single_gpu_scenarios(GPU0_VRAM_GB, GPU1_VRAM_GB)):
            if gpu_filter and sid not in gpu_filter:
                print(f"  [skip] {label}")
                continue
            if skip_gpu_indices and _gi in skip_gpu_indices:
                print(f"  [skip] {label} (--skip-gpu)")
                continue
            _run_scenario(sid, label, overlay, "gpu",
                          model_path, llama_server, topo_runs, all_results)

    # ── Case B: fits in GPU0 only ──────────────────────────────────────────
    elif case == "B":
        _g0name = _GPU_INFO[0]["name"] if _GPU_INFO else "GPU0"
        print(f"\n  Case B: model fits in GPU0 ({_g0name}, {GPU0_VRAM_GB:.0f} GB) only")
        print(f"  Testing GPU0 only — GPU1 would OOM")
        print(f"\n  ── Single-GPU scenario ──")

        sid, label, overlay = _single_gpu_scenarios(GPU0_VRAM_GB, GPU1_VRAM_GB)[0]
        if not gpu_filter or sid in gpu_filter:
            _run_scenario(sid, label, overlay, "gpu",
                          model_path, llama_server, topo_runs, all_results)
        else:
            print(f"  [skip] {label} (filtered out)")

    # ── Case C: requires both GPUs combined ───────────────────────────────
    elif case == "C":
        print(f"\n  Case C: model may require combined GPU VRAM (size estimate)")
        print(f"  Trying single-GPU first — llama-server --fit may reduce ngl to fit")
        print(f"\n  ── Single-GPU probe ──")

        # Try GPU0 (largest) first — --fit will auto-reduce ngl if needed
        _single = _single_gpu_scenarios(GPU0_VRAM_GB, GPU1_VRAM_GB)
        _gpu0_sid, _gpu0_label, _gpu0_overlay = _single[0]
        if not gpu_filter or _gpu0_sid in gpu_filter:
            # Probe with fit=False and ngl=99 — we only want to reclassify if
            # the model genuinely fits fully on GPU0. If it needs ngl reduction
            # (partial CPU offload) a GPU split will be faster.
            _r = _run_scenario(_gpu0_sid, _gpu0_label,
                               {**_gpu0_overlay, "fit": False}, "gpu",
                               model_path, llama_server, topo_runs, all_results)
            if _r["status"] == "ok":
                # Model fits on GPU0 — now check each remaining GPU individually.
                # Only test a GPU if the model's estimated VRAM fits within it.
                # Works correctly regardless of which GPU has more VRAM.
                _vram_needed_gb = (size_mb + _overhead_mb) / 1024
                _any_other_fits = False
                for _gi, (_sid, _label, _overlay) in enumerate(_single[1:], start=1):
                    _gpu_vram = _GPU_INFO[_gi]["vram_gb"] if _gi < len(_GPU_INFO) else 0.0
                    if _gpu_vram <= 0:
                        continue
                    if _vram_needed_gb <= _gpu_vram:
                        if not gpu_filter or _sid in gpu_filter:
                            _run_scenario(_sid, _label, _overlay, "gpu",
                                          model_path, llama_server, topo_runs, all_results)
                        _any_other_fits = True
                    else:
                        print(f"  Skipping {_label} — "
                              f"model needs ~{_vram_needed_gb:.1f} GB, GPU has {_gpu_vram:.1f} GB")
                case = "A" if _any_other_fits else "B"
                print(f"  Reclassifying as Case {case}")
                # Skip split scenarios — single GPU won
                goto_splits = False
            else:
                goto_splits = True
        else:
            goto_splits = True

        if goto_splits:
            print(f"  Single-GPU failed — testing split strategies")
            print(f"\n  ── Split scenarios ──")
            for sid, label, overlay in _split_scenarios(GPU0_VRAM_GB, GPU1_VRAM_GB):
                if gpu_filter and sid not in gpu_filter:
                    print(f"  [skip] {label}")
                    continue
                _run_scenario(sid, label, overlay, "gpu",
                              model_path, llama_server, topo_runs, all_results)

    # ── Case D: requires CPU offload ──────────────────────────────────────
    elif case == "D":
        print(f"\n  Case D: model exceeds combined GPU VRAM — CPU offload required")

        print(f"\n  ── Binary searching maximum GPU layers (ngl) ──")
        max_fit_ngl = _binary_search_ngl(
            model_path, llama_server, {**_PROBE_BASE},
        )
        if _last_launch_was_crash and max_fit_ngl == 0:
            print(f"  [skip] NUMA scenarios — model crashed during ngl search "
                  f"(incompatible arch, not a VRAM issue)")
        else:
            if max_fit_ngl == 0:
                print(f"  WARNING: no GPU layers fit — running fully on CPU")

            print(f"\n  Max stable ngl = {max_fit_ngl}")
            print(f"\n  ── NUMA scenarios (ngl={max_fit_ngl}) ──")

            ngl_overlay = {"num_gpu_layers": max_fit_ngl}
            for sid, label, numa_overlay in NUMA_SCENARIOS:
                if gpu_filter and sid not in gpu_filter:
                    print(f"  [skip] {label}")
                    continue
                _run_scenario(sid, label, {**ngl_overlay, **numa_overlay}, "numa",
                              model_path, llama_server, topo_runs, all_results)

    # ── pick winner ────────────────────────────────────────────────────────
    ok       = [r for r in all_results if r["status"] == "ok"]
    degraded = [r for r in all_results if r["status"] == "degraded"]
    if not ok and degraded:
        print(f"\n  WARNING: no clean scenarios — best available result is degraded (large-prompt OOM)")
        ok = degraded
    elif degraded:
        print(f"  NOTE: {len(degraded)} scenario(s) degraded (large-prompt OOM) — excluded from winner")

    if not ok:
        print(f"\n  WARNING: all topology scenarios failed — using defaults")
        winner_overlay = {}
        winner_label   = "default (all failed)"
    else:
        best = max(ok, key=lambda r: r["score"])
        winner_label   = best["label"]
        winner_overlay = dict(best["params_overlay"])
        if case == "D":
            winner_overlay["num_gpu_layers"] = max_fit_ngl

        # Show CUDA index alongside label to disambiguate GPU0/GPU1 vs CUDA index
        _cvd = winner_overlay.get("cuda_visible_devices")
        _cvd_str = f"  (CUDA idx {_cvd})" if _cvd is not None else ""
        print(f"\n  {'='*60}")
        print(f"  TOPOLOGY WINNER (Case {case}): {winner_label}{_cvd_str}")
        print(f"  Params: {winner_overlay}")
        print(f"  {'='*60}")
        print(f"\n  ── Ranking ──")
        for i, r in enumerate(sorted(ok, key=lambda x: x["score"], reverse=True), 1):
            star = "★" if r["label"] == winner_label else " "
            print(f"  {star} {i:2}. [{r['scenario']:18}]  "
                  f"score={r['score']:6.1f}  gen={r.get('gen_tps', 0):5.1f} t/s  "
                  f"{r['label']}")

    # Save
    with open(out_dir / "topo_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "model":           model_path.name,
            "model_path":      str(model_path),
            "timestamp":       datetime.now().isoformat(),
            "case":            case,
            "model_size_mb":   round(size_mb, 1),
            "gpu0_vram_gb":    GPU0_VRAM_GB,
            "gpu1_vram_gb":    GPU1_VRAM_GB,
            "max_fit_ngl":     max_fit_ngl,
            "winner":          winner_label,
            "winner_params":   winner_overlay,
            "no_jinja":        str(model_path) in _NO_JINJA_MODELS,
            "scenarios":       all_results,
        }, f, indent=2)

    t_topo.done("Topology sweep complete")
    return winner_overlay, case, max_fit_ngl


# ── context ceiling sweep ─────────────────────────────────────────────────────

def run_ctx_sweep(
    model_path:       Path,
    llama_server:     Path,
    optimizer_script: Path,
    winner_overlay:   dict,
    case:             str,
    max_fit_ngl:      int,
    skip_ram_tests:   bool,
    results_base:     Path | None = None,
    best_kv_type:     str = "f16",
) -> dict:
    """
    Binary-search the maximum stable context length for each relevant
    scenario (GPU-only, combined GPU, RAM-mixed, RAM-explicit).

    Returns a ctx_results dict with fields:
        ctx_gpu_single   — max ctx on winning single-GPU topology (cases A, B)
        ctx_gpu_combined — max ctx on best split topology (cases A, C)
        ctx_ram_mixed    — max ctx with ngl=max_fit, KV spills to RAM (cases A–D)
            recommended_ctx  — highest stable GPU-only ctx (for sidecar)

    Pass results_base to write directly to results_base/<slug>/ctx_sweep/
    instead of the legacy LLM_Optimiser/results/<slug>/ctx_sweep/ path.
    """
    _rbase = results_base if results_base is not None else optimizer_script
    out_dir = results_dir_for(model_path, _rbase) / "ctx_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_ctx = Timer("Context ceiling sweep")

    # Read GGUF metadata first — needed for trained ctx, KV cost, and arch info
    _meta = _cached_meta(model_path)

    # Cap search ceiling at the model's actual trained context length.
    # This prevents wasting probes on context sizes the model was never trained for.
    # Clamp to 262144 as a hard upper limit; round down to 4096 boundary.
    _trained = int(_meta.get("context_length") or 262144)
    TRAINED_MAX = (min(_trained, 262144) // 4096) * 4096 or 4096
    CTX_MIN     = 4096

    # RAM budget ceiling: usable RAM converted to tokens is model-specific.
    ram_total_mb, ram_avail_mb, _ = get_ram_info()
    ram_budget_mb = ram_total_mb * (1.0 - RAM_SAFETY_PCT)
    # Compute KV cache cost per token from GGUF metadata.
    # For hybrid models only attention layers contribute; SSM layers don't.
    kv_per_token = kv_cache_mb_per_token(_meta)
    if kv_per_token <= 0:
        kv_per_token = 0.05   # safe fallback (~1.6 GB per 32k ctx)
    # Adjust KV cost for quantization — quantized KV fits more context per GB
    _KV_RATIOS = {"f16": 1.0, "bf16": 1.0, "q8_0": 0.5, "q5_1": 0.3125,
                  "q5_0": 0.3125, "q4_1": 0.25, "q4_0": 0.25}
    kv_ratio = _KV_RATIOS.get(best_kv_type, 1.0)
    kv_per_token_eff = kv_per_token * kv_ratio
    # Overlay to inject KV cache type into every probe — formula and probe must match
    _kv_overlay = {"cache_type_k": best_kv_type, "cache_type_v": best_kv_type} \
        if best_kv_type not in ("f16", "") else {}
    if kv_ratio < 1.0:
        print(f"  KV quantization ({best_kv_type}): effective KV cost "
              f"{kv_per_token_eff*1024:.1f} MB/1k tokens ({kv_ratio:.2f}x reduction)")
    ram_ctx_ceiling = int(min(TRAINED_MAX, ram_budget_mb / kv_per_token_eff))
    # Round down to nearest 4096
    ram_ctx_ceiling = (ram_ctx_ceiling // 4096) * 4096

    _arch_note = "hybrid" if _meta.get("is_hybrid") else "dense"
    _moe_note  = " MoE" if _meta.get("is_moe") else ""
    print(f"\n{'='*70}")
    print(f"  CONTEXT CEILING SWEEP: {model_path.name}")
    print(f"  Arch: {_meta.get('arch','?')} ({_arch_note}{_moe_note})  |  "
          f"{_meta.get('n_attn_layers','?')} attn layers  |  "
          f"KV: {kv_per_token*1024:.1f} MB per 1k tokens")
    print(f"  Case {case}  |  "
          f"RAM budget: {ram_budget_mb/1024:.0f} GB  |  "
          f"RAM ctx ceiling: {ram_ctx_ceiling:,} tokens")
    print(f"{'='*70}")

    results = {
        "ctx_gpu_single":   None,
        "ctx_gpu_combined": None,
        "ctx_ram_mixed":    None,
        "recommended_ctx":  8192,
    }

    # Estimate model weight VRAM from actual file size — more reliable than
    # reading from topo_results.json which may be from a split run (both GPUs)
    # making it incompatible with single-GPU free VRAM estimates.
    # model_size_mb() sums all shards; multiply by ~1.05 for quant overhead.
    from model_utils import model_size_mb as _msz
    _model_weight_mb = _msz(model_path) * 1.05  # ~5% quant/alignment overhead

    # GPU VRAM budgets: 95% usable (leave 5% for driver/context overhead)
    _GPU0_usable = GPU0_VRAM_GB * 1024 * 0.95
    _GPU1_usable = GPU1_VRAM_GB * 1024 * 0.95
    _gpu_free_mb = max(0.0, (_GPU0_usable + _GPU1_usable) - _model_weight_mb)

    # Detect degenerate winner_overlay (all-None = failed or missing topo).
    # In this case: restrict ctx_gpu_single to GPU0 only (safest single-GPU default).
    _useful_keys = ("cuda_visible_devices", "tensor_split")
    _overlay_is_empty = not any(winner_overlay.get(k) for k in _useful_keys)

    # ── GPU-only context (Cases A and B) ──────────────────────────────────
    if case in ("A", "B"):
        print(f"\n  ── ctx_gpu_single: max context on winning single-GPU topology ──")
        print(f"  Topology: {winner_overlay}")
        # Determine which GPU's VRAM budget to use for the single-GPU estimate.
        _cvd = winner_overlay.get("cuda_visible_devices")
        # CUDA device index for GPU0 (largest) — depends on CUDA_DEVICE_ORDER=PCI_BUS_ID
        _gpu0_cuda_idx = str(int(_GPU_INFO[0]["index"])) if _GPU_INFO else "1"
        if _cvd is not None and str(_cvd) == _gpu0_cuda_idx:
            _single_usable = _GPU0_usable
        elif _cvd is not None:
            _single_usable = _GPU1_usable  # winner is GPU1
        else:
            # No CVD set — degenerate overlay, default to GPU0 (largest/safest)
            _single_usable = _GPU0_usable
        _single_free_mb = max(0.0, _single_usable - _model_weight_mb)
        _predicted_gpu  = _estimate_ctx_from_vram(_single_free_mb, kv_per_token_eff, TRAINED_MAX)
        print(f"  Predicted ctx from VRAM formula: {_predicted_gpu:,} tokens")
        t_sec = time.time()
        # If overlay is degenerate (no CVD or tensor_split), restrict to GPU0 only.
        # This prevents llama-server from greedily splitting across both GPUs.
        _gpu0_cvd = _GPU_INFO[0]["index"] if _GPU_INFO else 1
        _single_overlay = winner_overlay if not _overlay_is_empty else \
            {"cuda_visible_devices": str(_gpu0_cvd), "tensor_split": None,
             "main_gpu": None, "numa": None}
        ctx = _probe_then_search(
            model_path, llama_server,
            {**_PROBE_BASE, **_single_overlay, **_kv_overlay},
            predicted=_predicted_gpu, lo=CTX_MIN, hi=TRAINED_MAX,
        )
        results["ctx_gpu_single"] = ctx
        print(f"  ✓ ctx_gpu_single = {ctx:,} tokens  [{_fmt_s(time.time()-t_sec)}]")

    # ── Combined-GPU context (Cases A and C) ──────────────────────────────
    if case in ("A", "C"):
        # Find the best split topology from the topology sweep results
        topo_path = (results_dir_for(model_path, _rbase)
                     / "topo_sweep" / "topo_results.json")
        combined_overlay = None
        if topo_path.exists():
            try:
                td = json.loads(topo_path.read_text(encoding="utf-8"))
                split_ok = [s for s in td.get("scenarios", [])
                            if s["status"] == "ok"
                            and s.get("scenario", "").startswith("split")]
                if split_ok:
                    best_split = max(split_ok, key=lambda s: s["score"])
                    combined_overlay = best_split["params_overlay"]
            except Exception:
                pass

        # For Case A: only test combined if a split scenario actually succeeded
        # in the topo sweep. If the model fits on one GPU there is no benefit
        # to forcing a split — skip the combined test entirely.

        if combined_overlay:
            print(f"\n  ── ctx_gpu_combined: max context on combined GPU topology ──")
            print(f"  Topology: {combined_overlay}")
            _predicted_comb = _estimate_ctx_from_vram(_gpu_free_mb, kv_per_token_eff, TRAINED_MAX)
            print(f"  Predicted ctx from VRAM formula: {_predicted_comb:,} tokens")
            t_sec = time.time()
            ctx = _probe_then_search(
                model_path, llama_server,
                {**_PROBE_BASE, **combined_overlay, **_kv_overlay},
                predicted=_predicted_comb, lo=CTX_MIN, hi=TRAINED_MAX,
            )
            results["ctx_gpu_combined"] = ctx
            print(f"  ✓ ctx_gpu_combined = {ctx:,} tokens  [{_fmt_s(time.time()-t_sec)}]")

    # ── RAM tests (all cases unless skipped) ──────────────────────────────
    if not skip_ram_tests:

        # B1: mixed — model at max_fit_ngl, KV cache pages to RAM as needed
        print(f"\n  ── ctx_ram_mixed: KV cache pages to RAM automatically ──")
        print(f"  ngl={max_fit_ngl}, kv_offload=True (default)")
        print(f"  Predicted ctx from RAM formula: {ram_ctx_ceiling:,} tokens")
        ram_mixed_params = {
            **_PROBE_BASE,
            **winner_overlay,
            **_kv_overlay,
            "num_gpu_layers": max_fit_ngl,
            "kv_offload": True,
        }
        t_sec = time.time()
        ctx_mixed = _probe_then_search(
            model_path, llama_server, ram_mixed_params,
            predicted=ram_ctx_ceiling, lo=CTX_MIN, hi=ram_ctx_ceiling,
        )
        results["ctx_ram_mixed"] = ctx_mixed
        print(f"  ✓ ctx_ram_mixed = {ctx_mixed:,} tokens  [{_fmt_s(time.time()-t_sec)}]")


    # ── recommended_ctx for optimizer ─────────────────────────────────────
    # Priority: GPU-single > GPU-combined > RAM-mixed > RAM-explicit > 8192
    gpu_ctx = results["ctx_gpu_single"] or results["ctx_gpu_combined"]
    if gpu_ctx:
        results["recommended_ctx"] = gpu_ctx
    elif results["ctx_ram_mixed"]:
        results["recommended_ctx"] = results["ctx_ram_mixed"]

    else:
        results["recommended_ctx"] = 8192

    # ── summary ───────────────────────────────────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  CONTEXT SWEEP RESULTS")
    for k, v in results.items():
        if v is not None and k != "recommended_ctx":
            label_map = {
                "ctx_gpu_single":   "GPU single (best topology)",
                "ctx_gpu_combined": "GPU combined (best split)",
                "ctx_ram_mixed":    "RAM mixed (KV auto-pages)",

            }
            print(f"    {label_map.get(k, k):<35} {v:>8,} tokens")
    print(f"    {'Recommended (for optimizer)':<35} "
          f"{results['recommended_ctx']:>8,} tokens")
    print(f"  {'='*60}")

    # Save
    with open(out_dir / "ctx_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "model":          model_path.name,
            "model_path":     str(model_path),
            "timestamp":      datetime.now().isoformat(),
            "case":           case,
            "trained_max":    TRAINED_MAX,
            "ram_budget_mb":  round(ram_budget_mb, 0),
            "ram_ctx_ceiling": ram_ctx_ceiling,
            **results,
        }, f, indent=2)

    t_ctx.done("Context ceiling sweep complete")
    return results


# ── optimizer invocation ──────────────────────────────────────────────────────

def run_optimizer(
    model_path:       Path,
    llama_server:     Path,
    optimizer:        Path,
    mode_args:        list,
    trials:           int,
    sidecar_data:     dict,
    timeout:          int,
    reduced:          bool = False,
    trial_timeout_s:  int  = TRIAL_TIMEOUT_S,
) -> tuple:
    """
    Invoke LLM_Optimiser for one model with topology + ctx params in sidecar.
    Returns (status, elapsed_s, error_msg).
    """
    env = {**os.environ}
    cvd = sidecar_data.get("cuda_visible_devices")
    if cvd is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cvd)
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)

    # Pass per-trial timeout and reduced flag into the optimizer subprocess
    env["LLM_OPT_TRIAL_TIMEOUT"] = str(trial_timeout_s)
    if reduced:
        env["LLM_OPT_REDUCED"] = "1"
    else:
        env.pop("LLM_OPT_REDUCED", None)

    # Forward the log file path so the optimizer subprocess tees its own output
    if _LOG_FH is not None:
        try:
            env["LLM_OPT_LOG_FILE"] = _LOG_FH.name
        except AttributeError:
            pass
    else:
        env.pop("LLM_OPT_LOG_FILE", None)

    cmd = [
        sys.executable, "-u", str(optimizer),  # -u = unbuffered stdout/stderr
        "--model",        str(model_path),
        "--llama-server", str(llama_server),
        "--trials",       str(trials),
    ] + mode_args

    sidecar = Path(os.environ.get("TEMP", "/tmp")) / f".topo_{model_path.stem}.json"
    sidecar.write_text(json.dumps(sidecar_data, indent=2), encoding="utf-8")
    env["LLM_OPT_TOPO_SIDECAR"] = str(sidecar)

    t0 = time.time()
    status, error_msg = "ok", ""
    try:
        proc = subprocess.run(cmd, timeout=timeout, env=env)
        if proc.returncode != 0:
            status, error_msg = "nonzero_exit", f"Exit code {proc.returncode}"
    except subprocess.TimeoutExpired:
        status, error_msg = "timeout", f"Exceeded {timeout}s"
        print(f"\n  TIMEOUT: {model_path.name}")
    except KeyboardInterrupt:
        sidecar.unlink(missing_ok=True)
        raise
    except Exception as e:
        status, error_msg = "error", str(e)

    sidecar.unlink(missing_ok=True)
    return status, time.time() - t0, error_msg


# ── path / result helpers ─────────────────────────────────────────────────────

def results_dir_for(model_path: Path, results_base_or_script: Path) -> Path:
    """
    Return the per-model results directory.

    Accepts either:
      • a *results_base* directory (e.g. MERGED/results) — the model slug is
        appended directly: results_base/<slug>
      • a legacy *optimizer_script* Path — the old LLM_Optimiser folder logic
        is preserved so existing results remain readable.

    New callers should always pass a results_base directory.
    """
    slug = model_path.stem.lower().replace(" ", "_")
    p = results_base_or_script

    # If it looks like a Python script, use the old derivation for back-compat.
    if p.suffix in (".py", ".exe") or p.name.startswith("LLM_Optimis"):
        work_dir = p.parent
        if work_dir.name != "LLM_Optimiser":
            work_dir = work_dir / "LLM_Optimiser"
        return work_dir / "results" / slug

    # Otherwise treat it as a direct results_base directory.
    return p / slug


def has_existing_results(model_path: Path, optimizer_script: Path) -> bool:
    """True if the optimizer has at least one successful trial for this model."""
    csv_path = results_dir_for(model_path, optimizer_script) / "trials.csv"
    if not csv_path.exists():
        return False
    try:
        with open(csv_path, encoding="utf-8") as f:
            return any(r.get("status") == "ok" for r in csv.DictReader(f))
    except Exception:
        return False


def has_topo_results(model_path: Path, optimizer_script: Path) -> bool:
    """True if a complete topology sweep result exists for this model."""
    p = results_dir_for(model_path, optimizer_script) / "topo_sweep" / "topo_results.json"
    if not p.exists():
        return False
    try:
        td = json.loads(p.read_text(encoding="utf-8"))
        # Must have at least one successful scenario and a winner
        ok = any(s.get("status") == "ok" for s in td.get("scenarios", []))
        return ok and bool(td.get("winner"))
    except Exception:
        return False


def has_ctx_results(model_path: Path, optimizer_script: Path) -> bool:
    """True if a complete context ceiling sweep result exists for this model."""
    p = results_dir_for(model_path, optimizer_script) / "ctx_sweep" / "ctx_results.json"
    if not p.exists():
        return False
    try:
        cd = json.loads(p.read_text(encoding="utf-8"))
        # Must have at least one context value populated
        return any(cd.get(k) for k in [
            "ctx_gpu_single", "ctx_gpu_combined", "ctx_ram_mixed",
        ])
    except Exception:
        return False


def _load_topo_winner(model_path: Path, optimizer_script: Path) -> tuple:
    """
    Load a previously completed topology sweep winner.
    Returns (winner_overlay, case, max_fit_ngl) or ({}, 'A', 99) on failure.
    """
    p = results_dir_for(model_path, optimizer_script) / "topo_sweep" / "topo_results.json"
    try:
        td    = json.loads(p.read_text(encoding="utf-8"))
        over  = td.get("winner_params", {})
        case  = td.get("case", "A")
        ngl   = td.get("max_fit_ngl", 99)
        return over, case, int(ngl)
    except Exception:
        return {}, "A", 99


def _load_ctx_results(model_path: Path, optimizer_script: Path) -> dict:
    """Load a previously completed context ceiling sweep result."""
    p = results_dir_for(model_path, optimizer_script) / "ctx_sweep" / "ctx_results.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_models(models_dir: Path, filter_str: str | None) -> list:
    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        sys.exit(1)
    _NON_GENERATIVE = {"bert", "nomic-bert", "jina-bert", "roberta", "distilbert",
                        "xlm-roberta", "electra"}
    out = []
    for f in sorted(models_dir.rglob("*.gguf")):
        if any(s in f.name.lower() for s in SKIP_PATTERNS):
            continue
        if filter_str and filter_str.lower() not in f.name.lower():
            continue
        if not f.exists():
            print(f"  [skip] {f.name} — file no longer exists (deleted?)")
            continue
        # Skip non-generative architectures (embedding models, encoders)
        try:
            _meta = read_gguf_metadata(f)
            _arch = _meta.get("general.architecture", "")
            if _arch.lower() in _NON_GENERATIVE:
                print(f"  [skip] {f.name} — non-generative arch ({_arch})")
                continue
        except Exception:
            pass
        out.append(f)
    return out


def mode_to_args(mode: str) -> list:
    return {
        "optimize": ["--optimize"],
        "full":     [],
        "baseline": ["--baseline-only"],
        "phase1":   ["--phase1-only"],
    }.get(mode, ["--optimize"])


# ── result reading ────────────────────────────────────────────────────────────

def read_best_result(model_path: Path, optimizer_script: Path) -> dict:
    rdir      = results_dir_for(model_path, optimizer_script)
    topo_dir  = rdir / "topo_sweep"
    ctx_dir   = rdir / "ctx_sweep"

    # Pre-compute quant info from the model file directly — available even
    # before any sweep runs, so it always appears in the report.
    _mpath = Path(model_path)
    _case  = classify_model(_mpath, GPU0_VRAM_GB, GPU1_VRAM_GB)
    _ram_b = get_ram_budget_mb()
    _quant_info = recommend_quantizations(
        _mpath, _case, GPU0_VRAM_GB, GPU1_VRAM_GB, _ram_b)

    result = {
        "model":             model_path.name,
        "model_path":        str(model_path),
        "results_dir":       str(rdir),
        "status":            "no_results",
        "best_gen_tps":      0.0,
        "best_score":        0.0,
        "baseline_gen_tps":  0.0,
        "improvement_pct":   0.0,
        "best_config":       "",
        "topo_case":         "",
        "topo_winner":       "",
        "topo_score":        0.0,
        "ctx_gpu":           None,
        "ctx_ram":           None,
        "recommended_ctx":   None,
        # quantization fields
        "current_quant":     _quant_info.get("current_quant"),
        "current_bpw":       _quant_info.get("current_bpw"),
        "estimated_fp16_gb": _quant_info.get("estimated_fp16_gb"),
        "quant_recs":        _quant_info.get("recommendations", []),
    }

    # Topology results
    topo_path = topo_dir / "topo_results.json"
    if topo_path.exists():
        try:
            td = json.loads(topo_path.read_text(encoding="utf-8"))
            result["topo_case"]   = td.get("case", "")
            result["topo_winner"] = td.get("winner", "")
            ok_s = [s for s in td.get("scenarios", []) if s["status"] == "ok"]
            if ok_s:
                result["topo_score"] = max(ok_s, key=lambda s: s["score"])["score"]
        except Exception:
            pass

    # Context sweep results
    ctx_path = ctx_dir / "ctx_results.json"
    if ctx_path.exists():
        try:
            cd = json.loads(ctx_path.read_text(encoding="utf-8"))
            result["ctx_gpu"] = (cd.get("ctx_gpu_single")
                                 or cd.get("ctx_gpu_combined"))
            result["ctx_ram"] = cd.get("ctx_ram_mixed")
            result["recommended_ctx"] = cd.get("recommended_ctx")
        except Exception:
            pass

    # Baseline
    bl_path = rdir / "baseline.json"
    if bl_path.exists():
        try:
            bl = json.loads(bl_path.read_text(encoding="utf-8"))
            result["baseline_gen_tps"] = bl.get("speed", {}).get("gen_tps", 0.0)
        except Exception:
            pass

    # Optimizer trials
    csv_path = rdir / "trials.csv"
    if not csv_path.exists():
        return result
    try:
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return result

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        result["status"] = "all_failed"
        return result

    has_score = any(r.get("score") for r in ok_rows)
    ok_rows.sort(
        key=lambda r: float(r.get("score") or 0
                            if has_score else r.get("gen_tps") or 0),
        reverse=True,
    )
    best    = ok_rows[0]
    best_tps = float(best.get("gen_tps") or 0)
    bl_tps   = result["baseline_gen_tps"]

    result.update({
        "best_gen_tps":    best_tps,
        "best_score":      float(best.get("score") or 0),
        "improvement_pct": (best_tps - bl_tps) / bl_tps * 100 if bl_tps else 0,
        "best_config": " ".join([
            f"ngl={best.get('num_gpu_layers','?')}",
            f"ctx={best.get('num_ctx','?')}",
            f"b={best.get('batch_size','?')}/{best.get('ubatch_size','?')}",
            f"t={best.get('num_threads','?')}",
            f"fa={best.get('flash_attn','?')}",
            f"ctk={best.get('cache_type_k','?')}",
        ]),
        "total_trials": len(rows),
        "ok_trials":    len(ok_rows),
        "status":       "ok",
    })
    return result


# ── reporting ─────────────────────────────────────────────────────────────────

def print_report(results: list, models_dir: Path):
    ok     = sorted([r for r in results if r["status"] == "ok"],
                    key=lambda r: r["best_gen_tps"], reverse=True)
    failed = [r for r in results if r["status"] != "ok"]

    W = 130
    print(f"\n{'='*W}")
    print(f"  FINAL BATCH REPORT")
    print(f"  Models dir : {models_dir}")
    print(f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Successful : {len(ok)}   No results: {len(failed)}")
    print(f"{'='*W}")
    print(f"\n  {'#':>3}  {'Model':<36} {'Quant':<8} {'Case':<14} "
          f"{'Score':>6} {'Stock':>7} {'Best':>7} {'Gain':>7}  "
          f"{'ctx GPU':>8}  {'ctx RAM':>8}  "
          f"{'Topo winner':<22}  Best config")
    sep = "─"
    print(f"  {sep*3}  {sep*36} {sep*7} {sep*13} "
          f"{sep*6} {sep*7} {sep*7} {sep*7}  "
          f"{sep*8}  {sep*8}  {sep*22}  {sep*22}")

    for i, r in enumerate(ok, 1):
        name  = r["model"][:37] + "…" if len(r["model"]) > 38 else r["model"]
        raw_case = r.get("topo_case", "") or classify_model(
            Path(r["model_path"]), GPU0_VRAM_GB, GPU1_VRAM_GB)
        _CASE_LABELS = {
            "A": "A: both GPUs",
            "B": "B: GPU0 only",
            "C": "C: split GPUs",
            "D": "D: CPU offload",
        }
        case = _CASE_LABELS.get(raw_case, raw_case or "?")
        stock = f"{r['baseline_gen_tps']:.1f}" if r["baseline_gen_tps"] else "n/a"
        best  = f"{r['best_gen_tps']:.1f}"
        gain  = f"{r['improvement_pct']:+.0f}%" if r["baseline_gen_tps"] else "n/a"
        ctx_g = f"{r['ctx_gpu']//1024}k" if r.get("ctx_gpu") else "n/a"
        ctx_r = f"{r['ctx_ram']//1024}k" if r.get("ctx_ram") else "n/a"
        topo  = (r.get("topo_winner") or "not tested")[:24]
        cfg   = r.get("best_config", "")[:24]
        quant    = (r.get("current_quant") or "?")[:7]
        score_s  = f"{r['best_score']:.1f}" if r.get("best_score") else "n/a"
        print(f"  {i:>3}. {name:<36} {quant:<8} {case:<14} "
              f"{score_s:>6} {stock:>7} {best:>7} {gain:>7}  "
              f"{ctx_g:>8}  {ctx_r:>8}  "
              f"{topo:<22}  {cfg}")

    if failed:
        print(f"\n  No-result models ({len(failed)}):")
        for r in failed:
            print(f"    ✗ {r['model']}  ({r['status']})")

    # ── Quantization recommendations (every model with actionable recs) ──────
    models_with_recs = [
        r for r in ok
        if r.get("quant_recs") and r.get("current_quant")
    ]
    if models_with_recs:
        print(f"\n{'='*W}")
        print(f"  QUANTIZATION RECOMMENDATIONS")
        print(f"  Models where a different quant would better suit your hardware")
        print(f"{'='*W}")
        for r in models_with_recs:
            recs = r["quant_recs"]
            upgrades   = [x for x in recs if x["direction"] == "upgrade"][:2]
            sidegrades = [x for x in recs if x["direction"] == "sidegrade"][:1]
            downgrades = [x for x in recs if x["direction"] == "downgrade"][:2]
            # Only print if there's at least one non-sidegrade recommendation
            if not upgrades and not downgrades:
                continue
            _m     = _cached_meta(Path(r["model_path"]))
            _moe_s = (f" [MoE {_m['n_expert']}x]" if _m.get("is_moe") else "")
            fp16_g = r.get("estimated_fp16_gb", "?")
            print(f"\n  {r['model']}{_moe_s}")
            print(f"    Current: {r['current_quant']}  "
                  f"({r.get('best_gen_tps', 0):.0f} t/s)  "
                  f"~{fp16_g} GB FP16 baseline  "
                  f"Case {r.get('topo_case') or r.get('current_case', '?')}")
            for grp, arrow, label in [
                (upgrades,   "↑", "Higher quality"),
                (sidegrades, "↔", "Similar quality"),
                (downgrades, "↓", "Faster/smaller"),
            ]:
                for rec in grp:
                    delta_s = (f"+{rec['size_delta_pct']:.0f}%"
                               if rec["size_delta_pct"] >= 0
                               else f"{rec['size_delta_pct']:.0f}%")
                    print(f"    {arrow} {rec['quant']:<12} "
                          f"~{rec['estimated_gb']:.1f} GB  "
                          f"Case {rec['case']}  "
                          f"size {delta_s:>5}  "
                          f"spd={rec['speed_rank']:3}  qual={rec['quality_rank']:3}  "
                          f"{label}")

    print(f"\n{'='*W}")
    print(f"  TOP 5 FASTEST")
    print(f"{'='*W}")
    for r in ok[:5]:
        print(f"\n  {r['model']}")
        print(f"    Speed : {r['best_gen_tps']:.1f} t/s best  |  "
              f"{r['baseline_gen_tps']:.1f} t/s stock  |  "
              f"{r['improvement_pct']:+.0f}% gain")
        print(f"    Case  : {r.get('topo_case', '?')}  |  "
              f"Topo: {r.get('topo_winner') or 'not tested'}")
        if r.get("ctx_gpu"):
            print(f"    Ctx   : GPU {r['ctx_gpu']:,} tokens  |  "
                  f"RAM {r.get('ctx_ram', 0) or 0:,} tokens  |  "
                  f"Recommended {r.get('recommended_ctx', 0) or 0:,} tokens")
        print(f"    Config: {r['best_config']}")

        # Quant recommendations
        recs = r.get("quant_recs", [])
        if recs:
            cur_q  = r.get("current_quant", "?")
            fp16_g = r.get("estimated_fp16_gb", "?")
            _m     = _cached_meta(Path(r["model_path"]))
            _moe_s = (f"  [MoE: {_m['n_expert']}x experts, "
                      f"{_m['n_expert_used']} active]"
                      if _m.get("is_moe") else "")
            print(f"    Quant : current={cur_q}  "
                  f"(estimated FP16 baseline: {fp16_g} GB){_moe_s}")
            upgrades   = [x for x in recs if x["direction"] == "upgrade"][:3]
            sidegrades = [x for x in recs if x["direction"] == "sidegrade"][:2]
            downgrades = [x for x in recs if x["direction"] == "downgrade"][:2]
            for grp, grp_label in [
                (upgrades,   "  ↑ Higher quality"),
                (sidegrades, "  ↔ Similar quality"),
                (downgrades, "  ↓ Faster/smaller"),
            ]:
                if grp:
                    print(f"    {grp_label}:")
                    for rec in grp:
                        delta_s = (f"+{rec['size_delta_pct']:.0f}%"
                                   if rec["size_delta_pct"] >= 0
                                   else f"{rec['size_delta_pct']:.0f}%")
                        print(f"        {rec['quant']:<12}  "
                              f"~{rec['estimated_gb']:.1f} GB  "
                              f"Case {rec['case']}  "
                              f"size {delta_s:>5}  "
                              f"spd={rec['speed_rank']:3}  qual={rec['quality_rank']:3}  "
                              f"{rec['fit_label']}")
        print(f"    Dir   : {r['results_dir']}")
    print(f"\n{'='*W}")


def save_report(results: list, models_dir: Path, optimizer: Path,
                results_base: Path | None = None) -> Path:
    # Write batch_reports/ next to the results dir (or next to optimizer for legacy callers).
    if results_base is not None:
        report_dir = results_base.parent / "batch_reports"
    else:
        work_dir = optimizer.parent
        if work_dir.name != "LLM_Optimiser":
            work_dir = work_dir / "LLM_Optimiser"
        report_dir = work_dir / "batch_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = report_dir / f"batch_report_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated":   datetime.now().isoformat(),
            "models_dir":  str(models_dir),
            "gpu0_vram_gb": GPU0_VRAM_GB,
            "gpu1_vram_gb": GPU1_VRAM_GB,
            "total":       len(results),
            "successful":  sum(1 for r in results if r["status"] == "ok"),
            "results":     sorted(results,
                                  key=lambda r: r["best_gen_tps"], reverse=True),
        }, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "rank", "model", "current_quant", "current_bpw", "estimated_fp16_gb",
        "topo_case",
        "baseline_gen_tps", "best_gen_tps", "improvement_pct", "best_score",
        "topo_winner", "topo_score",
        "ctx_gpu", "ctx_ram", "recommended_ctx",
        "best_config", "total_trials", "ok_trials", "status", "model_path",
    ]
    csv_path = report_dir / f"batch_report_{ts}.csv"
    ok_r  = sorted([r for r in results if r["status"] == "ok"],
                   key=lambda r: r["best_gen_tps"], reverse=True)
    rest  = [r for r in results if r["status"] != "ok"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(ok_r + rest, 1):
            w.writerow({"rank": i, **r})

    shutil.copy2(csv_path, report_dir / "batch_report_latest.csv")
    print(f"\n  Reports saved:")
    print(f"    JSON  : {json_path}")
    print(f"    CSV   : {csv_path}")
    print(f"    Latest: {report_dir / 'batch_report_latest.csv'}")
    return csv_path


# ── HTML report generation ───────────────────────────────────────────────────

def _maybe_generate_html(
    report_json: Path,
    no_hf:       bool,
    refresh_hf:  bool,
):
    """
    Invoke generate_report.py if it exists alongside this script.
    Called automatically when --html-report is passed.
    """
    candidates = [
        Path(__file__).parent / "generate_report.py",
        Path(__file__).parent.parent / "generate_report.py",
    ]
    gen_script = next((p for p in candidates if p.exists()), None)
    if gen_script is None:
        print("  [html] generate_report.py not found — skipping HTML report")
        print("  [html] Place generate_report.py next to sweep_engine.py")
        return

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("gen_report", gen_script)
        gen  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        gen.generate(
            report_json   = report_json,
            output_path   = None,
            no_hf         = no_hf,
            force_refresh = refresh_hf,
        )
    except Exception as e:
        print(f"  [html] Warning: HTML report generation failed: {e}")


# ── RAM warning check ─────────────────────────────────────────────────────────

def check_ram_warning():
    """
    If RAM usage at startup exceeds RAM_WARN_PCT, print a warning and
    require manual confirmation before proceeding.
    If RAM usage is within the normal range, print a silent info line.
    """
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
        print(f"  Threshold: {RAM_WARN_PCT*100:.0f}%")
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
              f"({used_pct*100:.1f}%)  —  {avail_gb:.1f} GB available")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global GPU0_VRAM_GB, GPU1_VRAM_GB, _GPU_INFO

    parser = argparse.ArgumentParser(
        description="Batch optimizer wrapper with topology + context ceiling sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = parser.add_argument_group("paths")
    g.add_argument("--models-dir",   default=str(DEFAULT_MODELS_DIR))
    g.add_argument("--llama-server", default=str(DEFAULT_LLAMA_SERVER))
    g.add_argument("--optimizer",    default=str(DEFAULT_OPTIMIZER))

    g = parser.add_argument_group("model selection")
    g.add_argument("--filter",  default=None,
                   help="Only process models whose filename contains this string")
    g.add_argument("--resume",  action="store_true",
                   help="Skip models that already have successful optimizer results")
    g.add_argument("--dry-run", action="store_true",
                   help="List models and exit without running anything")

    g = parser.add_argument_group("run mode")
    g.add_argument("--mode", default="optimize",
                   choices=["optimize", "full", "baseline", "phase1"])
    g.add_argument("--trials",      type=int, default=20,
                   help="Phase 1 Optuna trials (default: 20; reduced mode uses 8)")
    g.add_argument("--timeout",     type=int, default=TIMEOUT_PER_MODEL,
                   help=f"Per-model timeout in seconds (default: {TIMEOUT_PER_MODEL})")
    g.add_argument("--trial-timeout", type=int, default=TRIAL_TIMEOUT_S,
                   help=f"Per-trial hard limit in seconds (default: {TRIAL_TIMEOUT_S})")
    g.add_argument("--reduced",     action="store_true",
                   help="Reduced mode: fewer trials, 2 topo runs, skip most Phase 0.5 "
                        "flags, no ctx sweep unless --ctx-sweep also given. "
                        "Gives a quick overview without taking days.")
    g.add_argument("--report-only", action="store_true",
                   help="Compile report from existing results without running")
    g.add_argument("--html-report", action="store_true",
                   help="Generate sortable HTML report after batch completes "
                        "(calls generate_report.py automatically)")
    g.add_argument("--no-hf", action="store_true",
                   help="Skip Hugging Face metadata fetch in HTML report")
    g.add_argument("--refresh-hf", action="store_true",
                   help="Force re-fetch all HF metadata ignoring cache")
    g.add_argument("--no-log", action="store_true",
                   help="Disable automatic log file (default: writes run_log_YYYYMMDD_HHMM.txt "
                        "next to this script)")
    g.add_argument("--interactive", action="store_true",
                   help="Pause for 5s between models — press 'n' to stop cleanly "
                        "after the current model finishes (results are always saved)")

    g = parser.add_argument_group("topology sweep")
    g.add_argument("--topo-sweep",  action="store_true",
                   help="Run topology classification + benchmark before optimizer")
    g.add_argument("--topo-only",   action="store_true",
                   help="Run topology sweep only, skip optimizer")
    g.add_argument("--topo-runs",   type=int, default=3,
                   help="Benchmark runs per topology scenario (default: 3)")
    g.add_argument("--gpu-filter",  nargs="+", default=None,
                   metavar="SCENARIO",
                   help="Only test these scenarios (e.g. gpu0_only split_prop)")
    g.add_argument("--force-numa",  action="store_true",
                   help="Force NUMA tests even for GPU-only models")

    g = parser.add_argument_group("context ceiling sweep")
    g.add_argument("--ctx-sweep",   action="store_true",
                   help="Binary-search max stable context after topology sweep")
    g.add_argument("--ctx-only",    action="store_true",
                   help="Run context sweep only, skip topology and optimizer")
    g.add_argument("--skip-ctx-b",  action="store_true",
                   help="Skip RAM-spill context tests (B1/B2), GPU ceiling only")

    g = parser.add_argument_group("hardware")
    g.add_argument("--gpu0-vram",   type=float, default=None,
                   help="Override detected GPU0 VRAM in GB")
    g.add_argument("--gpu1-vram",   type=float, default=None,
                   help="Override detected GPU1 VRAM in GB")

    args = parser.parse_args()
    # Auto-detect GPUs, then apply any CLI overrides
    _GPU_INFO = _detect_gpus()
    if not _GPU_INFO:
        print("WARNING: No GPUs detected — VRAM classification will be inaccurate")
        _GPU_INFO = [{"index": 0, "name": "GPU0", "vram_gb": args.gpu0_vram or 8.0}]
    else:
        for _gi, _g in enumerate(_GPU_INFO):
            print(f"  Detected GPU{_gi}: [{_g['name']}] {_g['vram_gb']:.0f} GB  (CUDA index {_g['index']})")

    GPU0_VRAM_GB = args.gpu0_vram if args.gpu0_vram is not None else _GPU_INFO[0]["vram_gb"]
    GPU1_VRAM_GB = args.gpu1_vram if args.gpu1_vram is not None else (_GPU_INFO[1]["vram_gb"] if len(_GPU_INFO) > 1 else 0.0)

    # Apply CLI overrides back into _GPU_INFO so labels are consistent
    if args.gpu0_vram is not None:
        _GPU_INFO[0]["vram_gb"] = GPU0_VRAM_GB
    if args.gpu1_vram is not None and len(_GPU_INFO) > 1:
        _GPU_INFO[1]["vram_gb"] = GPU1_VRAM_GB

    models_dir   = Path(args.models_dir)
    llama_server = Path(args.llama_server)
    optimizer    = Path(args.optimizer)

    do_topo = args.topo_sweep or args.topo_only
    do_ctx  = args.ctx_sweep  or args.ctx_only

    # In reduced mode, halve topo runs and trim trials unless explicitly set
    effective_topo_runs = args.topo_runs
    effective_trials    = args.trials
    if args.reduced:
        if args.topo_runs == 3:          # user didn't override, apply reduction
            effective_topo_runs = 2
        if args.trials == 20:            # user didn't override, apply reduction
            effective_trials = 8

    if not args.report_only:
        if not optimizer.exists():
            print(f"Error: Optimizer not found: {optimizer}")
            sys.exit(1)
        if not llama_server.exists():
            print(f"Error: llama-server not found: {llama_server}")
            sys.exit(1)

    models = find_models(models_dir, args.filter)
    if not models:
        print(f"No .gguf models found in: {models_dir}")
        sys.exit(0)

    Timer.reset_run()   # start run-level clock

    # ── log file tee ──────────────────────────────────────────────────────
    if not args.no_log and not args.report_only:
        _ts = datetime.now().strftime("%Y%m%d_%H%M")
        _log_path = SCRIPT_DIR / f"run_log_{_ts}.txt"
        install_log_tee(str(_log_path))
        print(f"  Logging to: {_log_path}")

    print(f"\n{'='*70}")
    print(f"  LLM BATCH OPTIMIZER")
    print(f"  Models dir   : {models_dir}")
    print(f"  llama-server : {llama_server}")
    print(f"  Optimizer    : {optimizer}")
    print(f"  Mode         : {args.mode}  |  Trials: {effective_trials}"
          + ("  [REDUCED]" if args.reduced else ""))
    print(f"  Trial timeout: {args.trial_timeout}s per trial")
    print(f"  Models found : {len(models)}")
    _gstrs = "  |  ".join(f"GPU{i}={g['vram_gb']:.0f}GB ({g['name']})" for i, g in enumerate(_GPU_INFO))
    print(f"  {_gstrs}")
    if do_topo:
        print(f"  Topo sweep   : ON ({effective_topo_runs} runs/scenario)"
              + ("  [reduced from 3]" if args.reduced and effective_topo_runs < args.topo_runs else ""))
        if args.gpu_filter:
            print(f"  GPU filter   : {args.gpu_filter}")
    if do_ctx:
        print(f"  Ctx sweep    : ON"
              + (" (GPU only)" if args.skip_ctx_b else " (GPU + RAM)"))
    if args.reduced and not do_ctx:
        print(f"  Ctx sweep    : OFF (reduced mode — add --ctx-sweep to enable)")
    check_ram_warning()
    print(f"{'='*70}\n")

    if args.report_only:
        all_results = [read_best_result(m, optimizer) for m in models]
        print_report(all_results, models_dir)
        csv_path = save_report(all_results, models_dir, optimizer)
        if args.html_report:
            _maybe_generate_html(
                csv_path.parent / csv_path.name.replace(".csv", ".json"),
                no_hf=args.no_hf, refresh_hf=args.refresh_hf,
            )
        return

    pending = [m for m in models
               if not (args.resume and has_existing_results(m, optimizer))]
    skipped = [m for m in models if m not in pending]
    if skipped:
        print(f"  Skipping {len(skipped)} completed models (--resume)\n")

    if args.dry_run:
        print(f"  DRY RUN — {len(pending)} models would be processed:")
        for i, m in enumerate(pending, 1):
            case = classify_model(m, GPU0_VRAM_GB, GPU1_VRAM_GB)
            print(f"    {i:3}. [Case {case}] {m.name}")
        return

    if not pending:
        print("  Nothing to do. Use --report-only or remove --resume.")
        return

    run_log    = []
    mode_args  = mode_to_args(args.mode)
    _interrupted = False

    for i, model_path in enumerate(pending, 1):
        elapsed_done = sum(r[2] for r in run_log)
        avg = elapsed_done / max(len(run_log), 1)
        eta = f"~{avg * (len(pending) - i + 1) / 60:.0f}m" if run_log else "unknown"

        case = classify_model(model_path, GPU0_VRAM_GB, GPU1_VRAM_GB)
        size = model_size_mb(model_path) / 1024

        run_h, run_m = divmod(Timer.run_elapsed(), 3600)
        run_m, run_s  = divmod(run_m, 60)
        run_str = (f"{int(run_h)}h " if run_h else "") + f"{int(run_m)}m{int(run_s):02d}s"

        print(f"\n{'#'*70}")
        print(f"  MODEL {i}/{len(pending)}: {model_path.name}")
        print(f"  Size: {size:.2f} GB  |  Case: {case}  |  "
              f"ETA remaining: {eta}  |  Run elapsed: {run_str}")
        print(f"{'#'*70}")

        t_model = Timer(f"Model {i}/{len(pending)}: {model_path.name}")

        winner_overlay = {}
        max_fit_ngl    = _read_total_layers(model_path)
        ctx_results    = {}

        # ── topology sweep ─────────────────────────────────────────────────
        if do_topo:
            _topo_done = args.resume and has_topo_results(model_path, optimizer)
            if _topo_done:
                winner_overlay, case, max_fit_ngl = _load_topo_winner(
                    model_path, optimizer)
                print(f"  [resume] Topology sweep already complete — "
                      f"Case {case}, winner: {winner_overlay}")
            else:
                winner_overlay, case, max_fit_ngl = run_topo_sweep(
                    model_path, llama_server, optimizer,
                    gpu_filter  = args.gpu_filter,
                    force_numa  = args.force_numa,
                    topo_runs   = effective_topo_runs,
                )

        if args.topo_only:
            t_model.done()
            run_log.append((model_path.name, "topo_only", t_model.elapsed(), ""))
            continue

        # ── build sidecar (no ctx results yet — optimizer runs first) ──────
        sidecar_data = {**winner_overlay}
        if str(model_path) in _NO_JINJA_MODELS:
            sidecar_data["no_jinja"] = True

        # ── scale trials and timeout for slow models ───────────────────────
        # Read winner gen_tps from topo results if available; otherwise estimate
        # from model size (larger = slower).  For models running < 30 t/s a full
        # 8-trial sweep at 360s/trial already risks the 45-min model timeout.
        # Rule: halve trials when gen_tps < 40 t/s, quarter them when < 15 t/s.
        topo_gen_tps = 0.0
        _topo_path = (results_dir_for(model_path, optimizer)
                      / "topo_sweep" / "topo_results.json")
        if _topo_path.exists():
            try:
                _td = json.loads(_topo_path.read_text(encoding="utf-8"))
                _ok = [s for s in _td.get("scenarios", []) if s.get("status") == "ok"]
                if _ok:
                    topo_gen_tps = max(s.get("gen_tps", 0) for s in _ok)
            except Exception:
                pass

        model_trials   = effective_trials
        model_timeout  = args.timeout
        if topo_gen_tps > 0:
            if topo_gen_tps < 15:
                model_trials  = max(2, effective_trials // 4)
                model_timeout = min(args.timeout, model_trials * args.trial_timeout + 300)
                print(f"  [slow model: {topo_gen_tps:.0f} t/s — reducing to "
                      f"{model_trials} trials, {model_timeout//60:.0f}min timeout]")
            elif topo_gen_tps < 40:
                model_trials  = max(4, effective_trials // 2)
                model_timeout = min(args.timeout, model_trials * args.trial_timeout + 300)
                print(f"  [slow model: {topo_gen_tps:.0f} t/s — reducing to "
                      f"{model_trials} trials, {model_timeout//60:.0f}min timeout]")

        # ── optimizer ──────────────────────────────────────────────────────
        print(f"\n  Starting optimizer "
              f"(mode={args.mode}, trials={model_trials}"
              + (" [reduced]" if args.reduced else "") + ")...")
        if sidecar_data:
            print(f"  Sidecar: {sidecar_data}")

        try:
            status, elapsed_opt, error = run_optimizer(
                model_path, llama_server, optimizer,
                mode_args, model_trials, sidecar_data, model_timeout,
                reduced=args.reduced,
                trial_timeout_s=args.trial_timeout,
            )
        except KeyboardInterrupt:
            print("\n\n  Interrupted — compiling partial report...")
            t_model.done()
            run_log.append((model_path.name, "interrupted", t_model.elapsed(), ""))
            _interrupted = True
            break

        # ── context ceiling sweep (after optimizer — only feeds future runs) ──
        if do_ctx and not args.ctx_only:
            _ctx_done = args.resume and has_ctx_results(model_path, optimizer)
            if _ctx_done:
                ctx_results = _load_ctx_results(model_path, optimizer)
                rec = ctx_results.get("recommended_ctx", "?")
                msg = (f"  [resume] Context sweep already complete — "
                       f"recommended: {rec:,} tokens" if isinstance(rec, int)
                       else "  [resume] Context sweep already complete")
                print(msg)
            else:
                ctx_results = run_ctx_sweep(
                    model_path, llama_server, optimizer,
                    winner_overlay = winner_overlay,
                    case           = case,
                    max_fit_ngl    = max_fit_ngl,
                    skip_ram_tests = args.skip_ctx_b,
                )

        if args.ctx_only:
            t_model.done()
            run_log.append((model_path.name, "ctx_only", t_model.elapsed(), ""))
            continue

        model_elapsed = t_model.elapsed()
        run_log.append((model_path.name, status, model_elapsed, error))
        ok_n = sum(1 for _, s, _, _ in run_log if s == "ok")
        print(f"\n  Model done in {_fmt_s(model_elapsed)} "
              f"(optimizer: {_fmt_s(elapsed_opt)}) | "
              f"Progress {i}/{len(pending)} | "
              f"OK {ok_n} | Failed {len(run_log) - ok_n}")

        # ── interactive pause (only between models, not after the last one) ──
        if args.interactive and i < len(pending):
            if not pause_between_models(seconds=5):
                _interrupted = True
                break

    # ── final report (always runs, even after interruption) ─────────────────
    try:
        all_results = [read_best_result(m, optimizer) for m in models]
    except KeyboardInterrupt:
        _interrupted = True
        all_results = []
        print("  Skipping report — interrupted during result collection.")

    if all_results:
        if _interrupted:
            print("  Compiling report from all completed results...")
        print_report(all_results, models_dir)
        _saved_csv = save_report(all_results, models_dir, optimizer)
        if args.html_report:
            _report_json = (
                _saved_csv.parent
                / _saved_csv.name.replace(".csv", ".json")
            )
            _maybe_generate_html(
                _report_json,
                no_hf=args.no_hf,
                refresh_hf=args.refresh_hf,
            )

    run_total = Timer.run_elapsed()
    run_h, run_m = divmod(run_total, 3600)
    run_m, run_s  = divmod(run_m, 60)
    run_str = (f"{int(run_h)}h " if run_h else "") + f"{int(run_m)}m{int(run_s):02d}s"

    if run_log:
        print(f"\n{'='*70}")
        print(f"  RUN LOG  (total: {run_str})")
        print(f"{'='*70}")
        for name, status, elapsed, error in run_log:
            tag = "OK" if status == "ok" else status.upper()
            err = f" ({error})" if error else ""
            print(f"  {tag:14} {_fmt_s(elapsed):>10}  {name}{err}")
        print(f"{'='*70}")
        print(f"  Total run time: {run_str}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
