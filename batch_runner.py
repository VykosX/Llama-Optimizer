#!/usr/bin/env python3
"""
batch_runner.py
===============
Batch optimizer wrapper.  Runs topology sweep, optional context ceiling sweep,
then drives optimize.py (via optimizer_adapter.py) for every GGUF in a folder.

Usage:
  python batch_runner.py --topo-sweep --preset reduced --html-report --resume --interactive
  python batch_runner.py --preset full --filter qwen --trials compute=80 memory=80
  python batch_runner.py --preset quick --skip-phases moe gpu
  python batch_runner.py --report-only --html-report

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap: add our directory to path and import our modules
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import optimizer_adapter as _adapter
from optimizer_adapter import (
    PRESET_NAMES, PRESETS, RunConfig, ModelResult,
    detect_gpus, kill_competing_processes, startup_timeout_for,
    wait_cool, is_port_open,
)

IS_WINDOWS = platform.system() == "Windows"

# ---------------------------------------------------------------------------
# Default paths -- override via CLI (--models-dir, --llama-server) or env vars
# (LLM_OPT_MODELS_DIR, LLAMA_SERVER). No hardcoded machine-specific paths.
# ---------------------------------------------------------------------------
def _find_llama_server_br() -> Path:
    """Locate llama-server via env var, PATH, or sibling directories."""
    import shutil
    env = os.environ.get("LLAMA_SERVER")
    if env and Path(env).is_file():
        return Path(env)
    found = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if found:
        return Path(found)
    for candidate in [
        _HERE / "llama-server.exe", _HERE / "llama-server",
        _HERE.parent / "llama-server" / "llama-server.exe",
        _HERE.parent / "llama-server" / "llama-server",
        _HERE.parent / "LLama-Server" / "llama-server.exe",
    ]:
        if candidate.is_file():
            return candidate
    return Path("llama-server")  # will fail with a clear error at startup


DEFAULT_MODELS_DIR    = Path(os.environ.get("LLM_OPT_MODELS_DIR", "models"))
DEFAULT_LLAMA_SERVER  = _find_llama_server_br()
DEFAULT_RESULTS_BASE  = _HERE / "results"
DEFAULT_REPORTS_DIR   = _HERE / "batch_reports"
DEFAULT_LOGS_DIR      = _HERE / "logs"
DEFAULT_REPORT_SCRIPT = _HERE / "generate_report.py"


def _find_ik_llama_server_br() -> Path:
    """Locate ik_llama-server via IK_LLAMA_SERVER env var or sibling directories."""
    env = os.environ.get("IK_LLAMA_SERVER", "")
    if env and Path(env).is_file():
        return Path(env)
    for candidate in [
        _HERE / "ik_llama-server.exe", _HERE / "ik_llama-server",
        _HERE.parent / "ik_llama.cpp" / "build" / "bin" / "llama-server.exe",
        _HERE.parent / "ik_llama.cpp" / "build" / "bin" / "llama-server",
        _HERE.parent / "IK_LLama-Server" / "llama-server.exe",
        _HERE.parent / "IK_LLama-Server" / "llama-server",
    ]:
        if Path(candidate).is_file():
            return Path(candidate)
    return Path("")  # empty = not found

PORT              = int(os.environ.get("LLM_OPT_PORT", "8090"))
TIMEOUT_PER_MODEL = int(os.environ.get("LLM_OPT_TIMEOUT", str(60 * 90)))  # 90 min
TRIAL_TIMEOUT_S   = int(os.environ.get("LLM_OPT_TRIAL_TIMEOUT", str(60 * 6)))  # 6 min

SKIP_PATTERNS = ["mmproj", "embedding", "embed", "encoder"]

# ---------------------------------------------------------------------------
# Logging Tee (unchanged from sweep_engine.py)
# ---------------------------------------------------------------------------
_LOG_FH = None

class _Tee:
    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh
    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        try:
            self._fh.write(data)
            self._fh.flush()
        except Exception:
            pass
    def flush(self):
        self._stream.flush()
    def __getattr__(self, name):
        return getattr(self._stream, name)

def install_log_tee(log_path: str, mode: str = "w") -> None:
    global _LOG_FH
    if _LOG_FH is not None:
        return
    try:
        _LOG_FH = open(log_path, mode, encoding="utf-8", buffering=1)
        sys.stdout = _Tee(sys.stdout, _LOG_FH)
        sys.stderr = _Tee(sys.stderr, _LOG_FH)
    except Exception as e:
        print(f"  Warning: could not open log file {log_path}: {e}")

# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------
class Timer:
    _run_start: float = 0.0

    def __init__(self, label: str = "", silent: bool = False):
        self._label = label
        self._start = time.time()
        self._silent = silent
        if not Timer._run_start:
            Timer._run_start = self._start

    def elapsed(self) -> float:
        return time.time() - self._start

    @classmethod
    def run_elapsed(cls) -> float:
        return time.time() - cls._run_start

    def done(self, msg: str = ""):
        if not self._silent:
            e = self.elapsed()
            label = f"[{self._label}]" if self._label else ""
            print(f"  {label} {_fmt_s(e)} total{(' ' + msg) if msg else ''}")

def _fmt_s(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m{s:02d}s"
    return f"{m}m{s:02d}s"

# ---------------------------------------------------------------------------
# model_utils: self-contained GGUF/GPU/RAM helpers (no sweep_engine dependency)
# ---------------------------------------------------------------------------
from model_utils import (
    read_gguf_metadata, _cached_meta, is_moe_model,
    recommend_quantizations, classify_model,
    detect_gpus as _detect_gpus_mu,
    find_models, model_size_mb,
    pause_between_models, check_ram_warning, get_ram_info,
)

# ---------------------------------------------------------------------------
# sweep_engine: GPU topology sweep + context ceiling sweep (required)
# ---------------------------------------------------------------------------
from sweep_engine import run_topo_sweep, run_ctx_sweep

_HAVE_SWEEPS = True
_HAVE_LEGACY = True  # kept for any remaining guard references


# ---------------------------------------------------------------------------
# Results directory logic
# ---------------------------------------------------------------------------
def results_dir_for(model_path: Path, results_base: Path) -> Path:
    slug = model_path.stem.lower().replace(" ", "_")
    return results_base / slug


def _resolve_results_dir(model_path: Path, results_base: Path) -> Path:
    """
    Locate the per-model results directory.
    Tries results_base/<slug> first (the canonical location written by both
    batch_runner and sweep_engine after the LLM_Optimiser path was removed),
    then falls back to _HERE/results/<slug> and CWD/results/<slug> in case
    the report is generated from a different working directory.
    """
    slug = model_path.stem.lower().replace(" ", "_")
    candidates = [
        results_base / slug,
        _HERE / "results" / slug,
        Path.cwd() / "results" / slug,
    ]
    for d in candidates:
        if d.exists():
            return d
    return results_base / slug  # fallback

def has_optimizer_results(model_path: Path, results_base: Path) -> bool:
    d = results_dir_for(model_path, results_base)
    # Check for real phase results first (best outcome)
    for phase in ["memory_audit", "memory", "compute", "compute_audit"]:
        p = d / f"{phase}_results.json"
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if data.get("best_params") is not None:
                    return True
            except Exception:
                pass
    # Fall back to sentinel: optimizer was attempted but all phases failed
    # (e.g. llama-server could never start). Treat as done so --resume skips it.
    if (d / "run_attempted.json").exists():
        return True
    return False

def has_topo_results_new(model_path: Path, results_base: Path) -> bool:
    d = results_dir_for(model_path, results_base)
    return (d / "topo_sweep" / "topo_results.json").exists()

def has_ctx_results_new(model_path: Path, results_base: Path) -> bool:
    d = results_dir_for(model_path, results_base)
    return (d / "ctx_sweep" / "ctx_results.json").exists()


def needs_retry(model_path: Path, results_base: Path) -> bool:
    """
    True if model has a results dir but the previous run failed.
    Covers: topo all-failed, or optimizer attempted but all phases failed.
    """
    d = results_dir_for(model_path, results_base)
    if not d.exists():
        return False
    topo_p = d / "topo_sweep" / "topo_results.json"
    if topo_p.exists():
        try:
            td = json.loads(topo_p.read_text(encoding="utf-8"))
            w = td.get("winner", "")
            if "all failed" in w or w.startswith("default"):
                return True
        except Exception:
            return True
    phase_has_data = any(
        (d / f"{ph}_results.json").exists()
        for ph in ["memory_audit", "memory", "compute_audit", "compute"]
    )
    if (d / "run_attempted.json").exists() and not phase_has_data:
        return True
    return False


def clear_failed_results(model_path: Path, results_base: Path) -> None:
    """Remove stale/failed result files so the model is re-processed fresh."""
    d = results_dir_for(model_path, results_base)
    topo_p = d / "topo_sweep" / "topo_results.json"
    if topo_p.exists():
        try:
            td = json.loads(topo_p.read_text(encoding="utf-8"))
            w = td.get("winner", "")
            if "all failed" in w or w.startswith("default"):
                topo_p.unlink()
                print(f"  [retry] Cleared failed topo result for {model_path.name}")
        except Exception:
            topo_p.unlink(missing_ok=True)
    sentinel = d / "run_attempted.json"
    if sentinel.exists():
        sentinel.unlink()


def load_topo_winner_new(model_path: Path, results_base: Path):
    d = results_dir_for(model_path, results_base)
    p = d / "topo_sweep" / "topo_results.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            # winner_params contains the full overlay dict as saved by run_topo_sweep
            wp = data.get("winner_params") or {}
            overlay = {k: wp.get(k) for k in
                       ["cuda_visible_devices", "tensor_split", "main_gpu", "numa",
                        "num_gpu_layers", "fit"]}
            case = data.get("case", "A")
            ngl = data.get("max_fit_ngl", 99)
            return overlay, case, ngl
        except Exception:
            pass
    return {}, "A", 99

# ---------------------------------------------------------------------------
# Slow-model trial scaling
# ---------------------------------------------------------------------------
def scale_run_config_for_speed(
    run_config: RunConfig,
    topo_gen_tps: float,
    model_timeout: int,
    trial_timeout: int,
) -> tuple[RunConfig, int]:
    """Reduce trial counts for slow models to avoid blowing the per-model timeout."""
    import copy
    rc = copy.deepcopy(run_config)
    mt = model_timeout

    if topo_gen_tps <= 0:
        return rc, mt

    if topo_gen_tps < 15:
        factor = 4
    elif topo_gen_tps < 40:
        factor = 2
    else:
        return rc, mt

    # Populate all preset trials so we scale phases not yet in rc.trials
    for phase, count in PRESETS.get(rc.preset, {}).get("trials", {}).items():
        if phase not in rc.trials:
            rc.trials[phase] = count
    for phase in list(rc.trials.keys()):
        rc.trials[phase] = max(2, rc.trials[phase] // factor)
    # Also tighten timeout
    total_trials = sum(rc.trials.values())
    mt = min(model_timeout, total_trials * trial_timeout + 300)
    print(f"  [slow model: {topo_gen_tps:.0f} t/s -- trial counts halved, timeout {mt//60:.0f}min]")
    return rc, mt

# ---------------------------------------------------------------------------
# Read best result from optimize.py phase results (for reporting)
# ---------------------------------------------------------------------------
def read_best_result_new(model_path: Path, results_base: Path,
                          gpu_info: list[dict]) -> dict:
    d = results_dir_for(model_path, results_base)
    try:
        size_mb = model_size_mb(model_path)
    except OSError:
        size_mb = 0.0

    result = {
        "model":             model_path.name,
        "model_path":        str(model_path),
        "status":            "no_results",
        "best_gen_tps":      0.0,
        "baseline_gen_tps":  0.0,
        "improvement_pct":   0.0,
        "best_score":        0.0,
        "best_config":       "",
        "topo_case":         "",
        "topo_winner":       "",
        "ctx_gpu":           None,
        "ctx_ram":           None,
        "recommended_ctx":   None,
        "results_dir":       str(d),
        # IK contrast fields
        "ik_best_tps":           0.0,
        "ik_gain_vs_llama_pct":  0.0,
        "ik_best_label":         "",
        "ik_available":          False,
    }

    # Try each phase in priority order for the best TPS
    for phase in ["memory_audit", "memory", "compute_audit", "compute"]:
        p = d / f"{phase}_results.json"
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        bm = data.get("best_metrics", {})
        bl = data.get("baseline", {})
        best_tps = bm.get("tps", 0)
        base_tps = bl.get("tps", 0)
        if best_tps > 0:
            result["best_gen_tps"]     = best_tps
            result["baseline_gen_tps"] = base_tps
            result["improvement_pct"]  = ((best_tps - base_tps) / base_tps * 100) if base_tps else 0
            result["best_score"]       = data.get("best_tps", best_tps)
            result["status"]           = "ok"
            bp = data.get("best_params", {})
            result["best_config"] = " ".join(
                f"{k}={v}" for k, v in list(bp.items())[:5]
            )
            break

    # Topo winner
    topo_p = d / "topo_sweep" / "topo_results.json"
    if topo_p.exists():
        try:
            td = json.loads(topo_p.read_text(encoding="utf-8"))
            result["topo_case"]   = td.get("case", "")
            result["topo_winner"] = td.get("winner", "")
        except Exception:
            pass

    # Ctx sweep
    ctx_p = d / "ctx_sweep" / "ctx_results.json"
    if ctx_p.exists():
        try:
            cd = json.loads(ctx_p.read_text(encoding="utf-8"))
            result["ctx_gpu"]         = cd.get("ctx_gpu_single") or cd.get("ctx_gpu_combined")
            result["ctx_ram"]         = cd.get("ctx_ram_mixed")
            result["recommended_ctx"] = cd.get("recommended_ctx")
        except Exception:
            pass

    # IK contrast results
    ik_p = d / "ik_contrast_results.json"
    if ik_p.exists():
        try:
            ik_d = json.loads(ik_p.read_text(encoding="utf-8"))
            result["ik_best_tps"]          = ik_d.get("ik_best_tps", 0.0)
            result["ik_gain_vs_llama_pct"] = ik_d.get("ik_gain_vs_llama_pct", 0.0)
            result["ik_best_label"]        = ik_d.get("ik_best_label", "")
            result["ik_available"]         = True
        except Exception:
            pass

    # Quant recommendations
    if size_mb > 0:
        try:
            gpu0 = gpu_info[0]["vram_gb"] if gpu_info else 24
            gpu1 = gpu_info[1]["vram_gb"] if len(gpu_info) > 1 else 16
            _qr = recommend_quantizations(model_path, result["topo_case"], gpu0, gpu1, 0)
            result["current_quant"]     = _qr.get("current_quant")
            result["current_bpw"]       = _qr.get("current_bpw")
            result["estimated_fp16_gb"] = _qr.get("estimated_fp16_gb")
            result["quant_recs"]        = _qr.get("recommendations", [])
        except Exception:
            pass

    return result

# ---------------------------------------------------------------------------
# HTML report bridge
# ---------------------------------------------------------------------------
def maybe_generate_html(json_path: Path, no_hf: bool = False, refresh_hf: bool = False):
    if not DEFAULT_REPORT_SCRIPT.exists():
        print(f"  [report] generate_report.py not found at {DEFAULT_REPORT_SCRIPT}")
        return
    cmd = [sys.executable, str(DEFAULT_REPORT_SCRIPT), "--report", str(json_path)]
    if no_hf:
        cmd.append("--no-hf")
    if refresh_hf:
        cmd.append("--refresh-hf")
    print(f"\n  Generating HTML report...")
    subprocess.run(cmd, check=False)

# ---------------------------------------------------------------------------
# CSV / JSON batch report
# ---------------------------------------------------------------------------
def save_batch_report(results: list[dict], reports_dir: Path, models_dir: Path = None) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"batch_report_{ts}.json"

    gpu_info = detect_gpus()
    gpu_strs = [f"{g['name']} {g['vram_gb']:.0f}GB" for g in gpu_info]
    gpu0_vram = round(gpu_info[0]["vram_gb"], 2) if gpu_info else 24.0
    gpu1_vram = round(gpu_info[1]["vram_gb"], 2) if len(gpu_info) > 1 else 0.0
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    report = {
        "generated":    datetime.now().isoformat(),
        "gpu_info":     gpu_strs,
        "gpu0_vram_gb": gpu0_vram,
        "gpu1_vram_gb": gpu1_vram,
        "models_dir":   str(models_dir) if models_dir else "",
        "total_models": len(results),
        "total":        len(results),
        "successful":   n_ok,
        "ok":           n_ok,
        "failed":       len(results) - n_ok,
        "results":      results,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = reports_dir / f"batch_report_{ts}.csv"
    fieldnames = ["rank", "model", "status", "best_gen_tps", "baseline_gen_tps",
                  "improvement_pct", "best_score", "topo_case", "topo_winner",
                  "ctx_gpu", "ctx_ram", "recommended_ctx", "best_config",
                  "current_quant", "estimated_fp16_gb",
                  "ik_best_tps", "ik_gain_vs_llama_pct", "ik_best_label"]
    ok = sorted([r for r in results if r.get("status") == "ok"],
                key=lambda r: r.get("best_gen_tps", 0), reverse=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(ok, 1):
            w.writerow({"rank": i, **r})

    print(f"\n  Batch report: {json_path}")
    print(f"  CSV:          {csv_path}")
    return json_path

def print_batch_report(results: list[dict]):
    ok     = sorted([r for r in results if r.get("status") == "ok"],
                    key=lambda r: r.get("best_gen_tps", 0), reverse=True)
    failed = [r for r in results if r.get("status") != "ok"]
    W = 140
    print(f"\n{'='*W}")
    print(f"  FINAL BATCH REPORT  —  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Optimized: {len(ok)}   No results: {len(failed)}")
    print(f"{'='*W}")
    # Header
    hdr = (f"  {'#':>3}  {'Model':<36} {'Quant':<8} {'Case':<6}"
           f" {'Stock':>7} {'Best':>7} {'Gain':>6}"
           f" {'IK t/s':>7} {'IK gain':>8}  {'Topo winner':<22}")
    print(hdr)
    print(f"  {'─'*3}  {'─'*36} {'─'*8} {'─'*6} {'─'*7} {'─'*7} {'─'*6} {'─'*7} {'─'*8}  {'─'*22}")
    for i, r in enumerate(ok, 1):
        name  = (r["model"][:37] + "…") if len(r["model"]) > 38 else r["model"]
        quant = (r.get("current_quant") or "?")[:7]
        case  = r.get("topo_case") or "?"
        stock = f"{r['baseline_gen_tps']:.1f}" if r.get("baseline_gen_tps") else "n/a"
        best  = f"{r['best_gen_tps']:.1f}"
        gain  = f"{r['improvement_pct']:+.0f}%" if r.get("baseline_gen_tps") else "n/a"
        ik_tps  = f"{r['ik_best_tps']:.1f}"  if r.get("ik_best_tps")         else "—"
        ik_gain = f"{r['ik_gain_vs_llama_pct']:+.0f}%" if r.get("ik_available") else "—"
        topo  = (r.get("topo_winner") or "not tested")[:22]
        print(f"  {i:>3}. {name:<36} {quant:<8} {case:<6}"
              f" {stock:>7} {best:>7} {gain:>6}"
              f" {ik_tps:>7} {ik_gain:>8}  {topo:<22}")
    if failed:
        print(f"\n  No-result models ({len(failed)}):")
        for r in failed:
            print(f"    ✗ {r['model']}  ({r.get('status', '?')})")
    print(f"\n{'='*W}")

def _run_ctx_sweep(model_path, llama_server, winner_overlay, case,
                   max_fit_ngl, gpu_info, args, results_base):
    """Run context ceiling sweep. Reads best_kv_type from memory results
    if available so the KV VRAM formula accounts for quantization."""
    import sweep_engine as _se
    _se.GPU0_VRAM_GB = gpu_info[0]["vram_gb"] if gpu_info else 24.0
    _se.GPU1_VRAM_GB = gpu_info[1]["vram_gb"] if len(gpu_info) > 1 else 0.0
    _se._GPU_INFO    = gpu_info
    _se.PORT         = args.port
    _se.VERBOSE      = args.verbose
    _wo, _case, _ngl = winner_overlay, case, max_fit_ngl
    if not _wo and has_topo_results_new(model_path, results_base):
        _wo, _case, _ngl = load_topo_winner_new(model_path, results_base)

    # A degenerate overlay (all keys None) means the stored topo was an all-failed
    # run — functionally equivalent to having no topo data.
    _useful = any(_wo.get(k) for k in ("cuda_visible_devices", "tensor_split", "num_gpu_layers"))
    _topo_is_usable = bool(_wo) and _useful

    # Topo data is mandatory for a meaningful ctx sweep — without it we don't
    # know the case, GPU topology, or max ngl. Run topo sweep now if missing or degenerate.
    if not has_topo_results_new(model_path, results_base) or not _topo_is_usable:
        reason = "no topo results found" if not has_topo_results_new(model_path, results_base) \
                 else "stored topo was all-failed (degenerate overlay)"
        print(f"  [ctx] {reason} — running topo sweep first...")
        try:
            from sweep_engine import run_topo_sweep as _rts
            _wo, _case, _ngl = _rts(
                model_path, llama_server,
                Path(__file__).resolve().parent / "optimizer_adapter.py",
                gpu_filter=getattr(args, "gpu_filter", None),
                force_numa=getattr(args, "force_numa", False),
                topo_runs=getattr(args, "topo_runs", 2),
                skip_gpu_indices=None,
                results_base=results_base,
            )
        except Exception as e:
            print(f"  [ctx] Topo sweep failed ({e}) — ctx sweep will use defaults (less accurate)")
    # Always skip if ctx results already exist — re-running wastes hours
    if has_ctx_results_new(model_path, results_base):
        ctx_p = results_dir_for(model_path, results_base) / "ctx_sweep" / "ctx_results.json"
        try:
            rec = json.loads(ctx_p.read_text(encoding="utf-8")).get("recommended_ctx", "?")
            print(f"  [ctx] Already complete -- recommended_ctx: {rec}")
        except Exception:
            pass
        return
    best_kv_type = "f16"
    mem_p = results_dir_for(model_path, results_base) / "memory_results.json"
    if mem_p.exists():
        try:
            md = json.loads(mem_p.read_text(encoding="utf-8"))
            kv = (md.get("best_params") or {}).get("kv_cache_type") or \
                 md.get("best_kv_type", "")
            if kv and kv not in ("", "none"):
                best_kv_type = kv
        except Exception:
            pass
    if best_kv_type != "f16":
        print(f"  [ctx] Using KV type from memory results: {best_kv_type}")
    try:
        run_ctx_sweep(
            model_path, llama_server, Path(__file__).resolve().parent / "optimizer_adapter.py",
            winner_overlay=_wo,
            case=_case,
            max_fit_ngl=_ngl,
            skip_ram_tests=args.skip_ctx_b,
            results_base=results_base,
            best_kv_type=best_kv_type,
        )
    except Exception as e:
        print(f"  [ctx] error: {e} -- skipping")



def main():
    global _LOG_FH

    parser = argparse.ArgumentParser(
        description="Batch LLM optimizer -- drives optimize.py for every GGUF in a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            [f"  {name:<12} {PRESETS[name]['description']}" for name in PRESET_NAMES]
        )
    )

    g = parser.add_argument_group("paths")
    g.add_argument("--models-dir",    default=str(DEFAULT_MODELS_DIR))
    g.add_argument("--llama-server",  default=str(DEFAULT_LLAMA_SERVER))
    g.add_argument("--ik-llama-server", default=str(_find_ik_llama_server_br()),
                   help="Path to ik_llama-server binary (or set IK_LLAMA_SERVER env var)")
    g.add_argument("--results-base",  default=str(DEFAULT_RESULTS_BASE),
                   help="Root folder for per-model results (default: ./results)")
    g.add_argument("--reports-dir",   default=str(DEFAULT_REPORTS_DIR))
    g.add_argument("--logs-dir",      default=str(DEFAULT_LOGS_DIR),
                   help="Folder for log files (default: ./logs)")

    g = parser.add_argument_group("model selection")
    g.add_argument("--filter",  default=None,
                   help="Only process models whose filename contains this string")
    g.add_argument("--resume",  action="store_true",
                   help="Skip models that already have optimizer results")
    g.add_argument("--retry",   default=None, nargs="?", const="first",
                   metavar="ORDER",
                   help="Retry failed models. ORDER: first=retry before missing (default), last=retry after missing. Combinable with --resume.")
    g.add_argument("--dry-run", action="store_true",
                   help="List models and exit without running")

    g = parser.add_argument_group("preset / phases")
    g.add_argument("--preset", default="standard", choices=PRESET_NAMES,
                   help="Optimization preset (default: standard). Use 'ik' or 'ik_thorough' to include IK contrast.")
    g.add_argument("--phases", nargs="+", default=None,
                   metavar="PHASE",
                   help="Override preset phase list (e.g. --phases compute memory)")
    g.add_argument("--skip-phases", nargs="*", default=None,
                   metavar="PHASE",
                   help="Skip phases. With no args (bare --skip-phases), skips everything "
                        "not in --rerun-phases. With args, skips those specific phases "
                        "(e.g. --skip-phases gpu moe).")
    g.add_argument("--rerun-phases", nargs="+", default=[],
                   metavar="PHASE",
                   help="Force re-run these phases even if results already exist "
                        "(e.g. --rerun-phases fast_moe). Does not clear other phases.")
    g.add_argument("--trials", nargs="+", default=[],
                   metavar="PHASE=N",
                   help="Override trial counts per phase (e.g. --trials compute=40 memory=40)")

    g = parser.add_argument_group("run mode")
    g.add_argument("--timeout",       type=int, default=TIMEOUT_PER_MODEL,
                   help=f"Per-model hard timeout in seconds (default: {TIMEOUT_PER_MODEL})")
    g.add_argument("--trial-timeout", type=int, default=TRIAL_TIMEOUT_S,
                   help=f"Per-trial timeout (default: {TRIAL_TIMEOUT_S})")
    g.add_argument("--port",          type=int, default=PORT)
    g.add_argument("--report-only",   action="store_true",
                   help="Compile report from existing results without running")
    g.add_argument("--html-report",   action="store_true",
                   help="Generate sortable HTML report after batch")
    g.add_argument("--no-hf",         action="store_true")
    g.add_argument("--refresh-hf",    action="store_true")
    g.add_argument("--no-log",        action="store_true",
                   help="Disable automatic log file")
    g.add_argument("--verbose",        action="store_true",
                   help="Show live llama-server loading progress ""(layer counts, CUDA init etc.). Suppressed by default.")
    g.add_argument("--interactive",   action="store_true",
                   help="Pause 5s between models; press n to stop cleanly")

    g = parser.add_argument_group("topology sweep")
    g.add_argument("--topo-sweep",  action="store_true",
                   help="Run GPU topology benchmark before optimizer")
    g.add_argument("--topo-only",   action="store_true")
    g.add_argument("--topo-runs",   type=int, default=2)
    g.add_argument("--gpu-filter",  nargs="+", default=None,
                   help="Only test these scenario IDs (advanced, e.g. gpu0_only)")
    g.add_argument("--skip-gpu",    nargs="+", default=None, metavar="GPU",
                   help="Skip topology scenarios for these GPUs. "
                        "Accepts GPU index (0, 1) or name fragment (5060, Ti, 3090). "
                        "Example: --skip-gpu 1  or  --skip-gpu 5060")
    g.add_argument("--force-numa",  action="store_true")

    g = parser.add_argument_group("context ceiling sweep")
    g.add_argument("--ctx-sweep",   action="store_true")
    g.add_argument("--ctx-only",    action="store_true")
    g.add_argument("--skip-ctx-b",  action="store_true")

    g = parser.add_argument_group("hardware")
    g.add_argument("--gpu0-vram",   type=float, default=None)
    g.add_argument("--gpu1-vram",   type=float, default=None)

    args = parser.parse_args()

    # ---- Initialize run timer ----
    Timer._run_start = time.time()

    # ---- Logging tee ----
    logs_dir = Path(args.logs_dir)
    if not args.no_log and not args.report_only:
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        log_path = logs_dir / f"run_log_{ts}.txt"
        install_log_tee(str(log_path))
        print(f"  Logging to: {log_path}\n")

    # ---- GPU detection ----
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpu_info = detect_gpus()
    if args.gpu0_vram is not None and gpu_info:
        gpu_info[0]["vram_gb"] = args.gpu0_vram
    if args.gpu1_vram is not None and len(gpu_info) > 1:
        gpu_info[1]["vram_gb"] = args.gpu1_vram

    # Resolve --skip-gpu into GPU indices (applied only in Case A)
    # Always defined so the run_topo_sweep call never gets a NameError.
    _gpu_filter        = list(args.gpu_filter) if args.gpu_filter else None
    _skip_gpu_indices: set = set()
    if args.skip_gpu and gpu_info:
        for spec in args.skip_gpu:
            spec_s = str(spec).strip()
            if spec_s.isdigit():
                _skip_gpu_indices.add(int(spec_s))
            else:
                for gi, g in enumerate(gpu_info):
                    if spec_s.lower() in g["name"].lower():
                        _skip_gpu_indices.add(gi)
        if _skip_gpu_indices:
            skipped_names = [gpu_info[i]["name"] for i in _skip_gpu_indices
                             if i < len(gpu_info)]
            print(f"  GPU skip (Case A only): {skipped_names}")
        else:
            print(f"  GPU skip: --skip-gpu {args.skip_gpu!r} matched no detected GPUs")
    args.gpu_filter = _gpu_filter

    gpu_strs = "  |  ".join(
        f"GPU{i}={g['vram_gb']:.0f}GB ({g['name']}) CUDA{g['index']}"
        for i, g in enumerate(gpu_info)
    )

    models_dir    = Path(args.models_dir)
    llama_server  = Path(args.llama_server)
    results_base  = Path(args.results_base)
    reports_dir   = Path(args.reports_dir)
    # IK server: CLI arg > env var > auto-detect
    _ik_arg = getattr(args, "ik_llama_server", "")
    ik_server_path = _ik_arg if (_ik_arg and Path(_ik_arg).is_file()) else \
                     os.environ.get("IK_LLAMA_SERVER", "")

    # ---- Build RunConfig ----
    trial_overrides: dict[str, int] = {}
    for t in args.trials:
        if "=" in t:
            k, v = t.split("=", 1)
            try:
                trial_overrides[k.strip()] = int(v.strip())
            except ValueError:
                pass

    run_config = RunConfig(
        preset=args.preset,
        phases=args.phases or [],
        trials=trial_overrides,
        skip_phases=args.skip_phases,
        rerun_phases=args.rerun_phases,
    )

    do_topo = args.topo_sweep or args.topo_only
    do_ctx  = args.ctx_sweep  or args.ctx_only

    # ---- Find models ----
    if not args.report_only:
        if not models_dir.exists():
            print(f"Error: models dir not found: {models_dir}")
            sys.exit(1)
        if not llama_server.exists():
            print(f"Error: llama-server not found: {llama_server}")
            sys.exit(1)

    models = []
    try:
        models = find_models(models_dir, args.filter)
    except Exception:
        pass
    if not models:
        # Fallback: manual scan (mirrors find_models shard logic)
        _SHARD_RE = re.compile(r'-(\d{5})-of-\d{5}$', re.IGNORECASE)
        for f in sorted(models_dir.rglob("*.gguf")):
            if any(s in f.name.lower() for s in SKIP_PATTERNS):
                continue
            if args.filter and args.filter.lower() not in f.name.lower():
                continue
            if not f.exists():
                continue
            # Skip shards 2..N — llama-server discovers them automatically from shard 1
            _sm = _SHARD_RE.search(f.stem)
            if _sm and int(_sm.group(1)) != 1:
                continue
            models.append(f)

    retry_order    = args.retry  # None | "first" | "last"
    failed_models  = [m for m in models if needs_retry(m, results_base)] if retry_order else []
    missing_models = [m for m in models
                      if not has_optimizer_results(m, results_base)
                      and not needs_retry(m, results_base)]
    completed      = [m for m in models
                      if has_optimizer_results(m, results_base)
                      and not needs_retry(m, results_base)]
    if retry_order == "last":
        pending = (missing_models if (args.resume or retry_order) else models[:]) + failed_models
    elif retry_order == "first":
        pending = failed_models + (missing_models if (args.resume or retry_order) else models[:])
    elif args.resume:
        pending = missing_models
    else:
        pending = models[:]
    if completed and (args.resume or retry_order):
        print(f"  Skipping {len(completed)} completed models")
    if failed_models and retry_order:
        order_str = "first" if retry_order == "first" else "after missing"
        print(f"  Retrying {len(failed_models)} previously-failed models ({order_str})")
        for _m in failed_models:
            clear_failed_results(_m, results_base)

    print(f"\n{'='*70}")
    print(f"  LLM BATCH OPTIMIZER (batch_runner.py)")
    print(f"  Models dir  : {models_dir}")
    print(f"  llama-server: {llama_server}")
    if ik_server_path and Path(ik_server_path).is_file():
        _ik_mode_str = "dual (llama + IK contrast)" if llama_server.is_file() else "IK-only"
        print(f"  ik-server   : {ik_server_path}  [{_ik_mode_str}]")
    else:
        print(f"  ik-server   : not configured (set IK_LLAMA_SERVER or --ik-llama-server)")
    print(f"  Results base: {results_base}")
    print(f"  Logs dir    : {logs_dir}")
    print(f"  Preset      : {args.preset} -- {PRESETS[args.preset]['description']}")
    print(f"  Phases      : {run_config.resolved_phases()}")
    print(f"  {gpu_strs}")
    print(f"  Models found: {len(models)}  |  Pending: {len(pending)}")
    print(f"  Topo sweep  : {'ON' if do_topo else 'OFF'}")
    print(f"  Ctx sweep   : {'ON' if do_ctx else 'OFF'}")
    print(f"{'='*70}\n")

    if args.dry_run:
        for m in pending:
            print(f"  {m.name}")
        sys.exit(0)

    if args.report_only:
        all_results = [read_best_result_new(m, results_base, gpu_info) for m in models]
        print_batch_report(all_results)
        json_path = save_batch_report(all_results, reports_dir, models_dir)
        if args.html_report:
            maybe_generate_html(json_path, args.no_hf, args.refresh_hf)
        return

    check_ram_warning()

    # ---- ETA tracker ----
    elapsed_times: list[float] = []

    def eta_str(remaining: int) -> str:
        if not elapsed_times:
            return "unknown"
        # Use all-time average to avoid ETA spiking after one slow model.
        # If we have enough samples, blend: 70% global avg + 30% recent (last 5)
        global_avg = sum(elapsed_times) / len(elapsed_times)
        if len(elapsed_times) >= 5:
            recent_avg = sum(elapsed_times[-5:]) / 5
            avg = 0.7 * global_avg + 0.3 * recent_avg
        else:
            avg = global_avg
        secs = avg * remaining
        h, m = divmod(int(secs) // 60, 60)
        return f"~{h}h{m:02d}m" if h else f"~{m}m"

    # ---- Main loop ----
    run_log: list[tuple] = []   # (name, status, elapsed, error)
    _interrupted = False

    try:
        for i, model_path in enumerate(pending, 1):
            run_h, run_m = divmod(int(Timer.run_elapsed()), 3600)
            run_m, run_s = divmod(run_m, 60)
            run_str = f"{run_h}h " * bool(run_h) + f"{run_m}m{run_s:02d}s"
            eta = eta_str(len(pending) - i + 1)

            print(f"\n{'#'*70}")
            print(f"  MODEL {i}/{len(pending)}: {model_path.name}")
            try:
                size_gb = model_size_mb(model_path) / 1024
            except OSError:
                size_gb = 0.0
            print(f"  Size: {size_gb:.2f} GB  |  ETA remaining: {eta}  |  Run elapsed: {run_str}")
            print(f"{'#'*70}")

            t_model = Timer(f"Model {i}/{len(pending)}: {model_path.name}")

            winner_overlay: dict = {}
            ctx_results: dict = {}
            case = "A"
            max_fit_ngl = 99
            _topo_ran = False  # set True only when run_topo_sweep succeeds

            # ---- Topology sweep ----
            if do_topo:
                # retry clears stale topo above; only skip if genuinely complete
                _topo_done = (args.resume and not retry_order) and has_topo_results_new(model_path, results_base)
                if _topo_done:
                    winner_overlay, case, max_fit_ngl = load_topo_winner_new(model_path, results_base)
                    print(f"  [resume] Topo already complete -- Case {case}, winner: {winner_overlay}")
                else:
                    # Inject detected GPU info into sweep_engine module globals before calling.
                    import sweep_engine as _se
                    _se.GPU0_VRAM_GB = gpu_info[0]["vram_gb"] if gpu_info else 24.0
                    _se.GPU1_VRAM_GB = gpu_info[1]["vram_gb"] if len(gpu_info) > 1 else 0.0
                    _se._GPU_INFO    = gpu_info
                    _se.PORT         = args.port
                    _se.VERBOSE      = args.verbose
                    _topo_ran = False
                    try:
                        winner_overlay, case, max_fit_ngl = run_topo_sweep(
                            model_path, llama_server, _HERE / "optimizer_adapter.py",
                            gpu_filter=args.gpu_filter,
                            force_numa=args.force_numa,
                            topo_runs=args.topo_runs,
                            skip_gpu_indices=_skip_gpu_indices,
                            results_base=results_base,
                        )
                        _topo_ran = True
                    except Exception as e:
                        print(f"  [topo] error: {e} -- using defaults")

            # Detect unloadable/corrupt model: size=0 OR topo ran successfully
            # but every scenario failed (llama-server could not load the file).
            # Do NOT trigger this when topo threw an exception — that is a code
            # bug, not a bad model file.
            _topo_all_failed = (
                size_gb == 0.0 or
                (do_topo and _topo_ran and not winner_overlay and
                 not has_optimizer_results(model_path, results_base))
            )
            if _topo_all_failed and not args.topo_only:
                print(f"  [!] Model appears unloadable (size={size_gb:.2f} GB, "
                      f"all topo scenarios failed) — skipping optimizer")
                status = "failed"
                error  = "unloadable model"
                model_elapsed = t_model.elapsed()
                elapsed_times.append(model_elapsed)
                run_log.append((model_path.name, status, model_elapsed, error))
                ok_n = sum(1 for _, s, _, _ in run_log if s == "ok")
                print(f"\n  Model done in {_fmt_s(model_elapsed)} | "
                      f"Progress {i}/{len(pending)} | "
                      f"OK {ok_n} | Failed {len(run_log) - ok_n}")
                continue

            if args.topo_only:
                run_log.append((model_path.name, "topo_only", t_model.elapsed(), ""))
                t_model.done()
                continue

            if args.ctx_only:
                # Load topo if not already done this session
                if not do_topo and has_topo_results_new(model_path, results_base):
                    winner_overlay, case, max_fit_ngl = load_topo_winner_new(model_path, results_base)
                if do_ctx:
                    _run_ctx_sweep(model_path, llama_server, winner_overlay, case,
                                   max_fit_ngl, gpu_info, args, results_base)
                run_log.append((model_path.name, "ctx_only", t_model.elapsed(), ""))
                t_model.done()
                continue


            # ---- Scale trials for slow models ----
            topo_gen_tps = 0.0
            topo_p = results_base / (model_path.stem.lower().replace(" ", "_")) / "topo_sweep" / "topo_results.json"
            if topo_p.exists():
                try:
                    td = json.loads(topo_p.read_text(encoding="utf-8"))
                    _ok = [s for s in td.get("scenarios", []) if s.get("status") == "ok"]
                    if _ok:
                        topo_gen_tps = max(s.get("gen_tps", 0) for s in _ok)
                except Exception:
                    pass

            scaled_rc, model_timeout = scale_run_config_for_speed(
                run_config, topo_gen_tps, args.timeout, args.trial_timeout
            )

            # ---- Check no_jinja from topo results ----
            no_jinja = False
            topo_sidecar = results_base / (model_path.stem.lower().replace(" ", "_")) / "topo_sweep" / "topo_results.json"
            if topo_sidecar.exists():
                try:
                    td = json.loads(topo_sidecar.read_text(encoding="utf-8"))
                    no_jinja = td.get("no_jinja", False)
                except Exception:
                    pass

            # ---- Run optimizer via adapter ----
            # Load ctx_recommended from stored results if available;
            # otherwise estimate from VRAM headroom using topo + GGUF metadata.
            ctx_recommended = None
            _ctx_p = results_dir_for(model_path, results_base) / "ctx_sweep" / "ctx_results.json"
            if _ctx_p.exists():
                try:
                    ctx_recommended = json.loads(
                        _ctx_p.read_text(encoding="utf-8")
                    ).get("recommended_ctx")
                except Exception:
                    pass
            if not ctx_recommended:
                # Derive from VRAM: free_vram / kv_cost_per_token
                try:
                    from model_utils import get_model_meta
                    from sweep_engine import kv_cache_mb_per_token
                    _mmeta = get_model_meta(model_path)
                    _kv_pt = kv_cache_mb_per_token(_mmeta)
                    if _kv_pt > 0:
                        # Estimate free VRAM: total - model weight VRAM from topo
                        _gpu_total_mb = (gpu_info[0]["vram_gb"] if gpu_info else 24.0) * 1024
                        _topo_vram = 0.0
                        _tp = results_dir_for(model_path, results_base) / "topo_sweep" / "topo_results.json"
                        if _tp.exists():
                            try:
                                _td = json.loads(_tp.read_text(encoding="utf-8"))
                                _wl = _td.get("winner", "")
                                for _sc in _td.get("scenarios", []):
                                    if _sc.get("label") == _wl and _sc.get("status") == "ok":
                                        _topo_vram = float(_sc.get("vram_mb") or 0)
                                        break
                            except Exception:
                                pass
                        _free_mb = max(0.0, _gpu_total_mb - _topo_vram) * 0.85
                        _trained = int(_mmeta.get("context_length") or 131072)
                        _est = int((_free_mb / _kv_pt) // 4096) * 4096
                        ctx_recommended = max(4096, min(_est, _trained))
                        print(f"  [ctx] Estimated recommended_ctx from VRAM: {ctx_recommended:,}")
                except Exception:
                    pass
            print(f"\n  Starting optimizer (preset={args.preset}, phases={scaled_rc.resolved_phases()})...")
            if winner_overlay:
                print(f"  Topo overlay: {winner_overlay}")

            try:
                model_result = _adapter.run_pipeline(
                    model_path=model_path,
                    llama_server=llama_server,
                    results_dir=results_base / model_path.stem.lower().replace(" ", "_"),
                    run_config=scaled_rc,
                    gpu_info=gpu_info,
                    port=args.port,
                    no_jinja=no_jinja,
                    topo_overlay=winner_overlay,
                    topo_case=case,
                    ctx_recommended=ctx_recommended,
                    resume=args.resume,
                    ik_server_path=ik_server_path,
                )
                status = model_result.status
                error  = model_result.error
            except KeyboardInterrupt:
                print("\n\n  Interrupted -- compiling partial report...")
                t_model.done()
                run_log.append((model_path.name, "interrupted", t_model.elapsed(), ""))
                _interrupted = True
                break


            # ---- Context sweep (last — uses memory/compute results for KV type) ----
            if do_ctx:
                _run_ctx_sweep(model_path, llama_server, winner_overlay, case,
                               max_fit_ngl, gpu_info, args, results_base)

            model_elapsed = t_model.elapsed()
            elapsed_times.append(model_elapsed)
            ok_n = sum(1 for _, s, _, _ in run_log if s == "ok")
            run_log.append((model_path.name, status, model_elapsed, error))
            print(f"\n  Model done in {_fmt_s(model_elapsed)} | "
                  f"Progress {i}/{len(pending)} | "
                  f"OK {ok_n + (1 if status=='ok' else 0)} | "
                  f"Failed {len(run_log) - ok_n - (1 if status=='ok' else 0)}")

            if args.interactive and i < len(pending):
                should_continue = True
                should_continue = pause_between_models(seconds=5)
                if not should_continue:
                    _interrupted = True
                    break

    except KeyboardInterrupt:
        print("\n\n  Ctrl+C -- compiling report from completed results...")
        _interrupted = True

    # ---- Final report (always runs) ----
    try:
        all_results = [read_best_result_new(m, results_base, gpu_info) for m in models]
    except KeyboardInterrupt:
        all_results = []
        print("  Skipping report -- interrupted during result collection.")

    if all_results:
        if _interrupted:
            print("  Compiling report from all completed results...")
        print_batch_report(all_results)
        json_path = save_batch_report(all_results, reports_dir, models_dir)
        if args.html_report:
            maybe_generate_html(json_path, args.no_hf, args.refresh_hf)

    # ---- Run summary ----
    run_total = Timer.run_elapsed()
    h, m = divmod(int(run_total), 3600)
    m, s = divmod(m, 60)
    run_str = (f"{h}h " if h else "") + f"{m}m{s:02d}s"
    print(f"\n  Total run time: {run_str}")

    if run_log:
        ok_n  = sum(1 for _, s, _, _ in run_log if s == "ok")
        fail_n = len(run_log) - ok_n
        print(f"  Processed {len(run_log)} models: {ok_n} OK, {fail_n} failed/skipped")


if __name__ == "__main__":
    main()
