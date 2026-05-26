"""
optimizer_adapter.py
====================
Bridge between batch_runner.py and optimize.py.

Responsibilities:
  - Import optimize.py as a module (no subprocess)
  - Call optimize.reinitialize() to reset globals for each model
  - Translate a RunConfig (preset + phase flags + trial counts) into
    the correct sequence of optimize.phase_*() calls
  - Own all features that are ours, not theirs:
      - Tee / log forwarding
      - Startup timeout scaling (model size + MoE multiplier)
      - Jinja detection and --no-jinja retry
      - GPU detection and CUDA_DEVICE_ORDER alignment
      - Port-open detection for OOM crash
      - kill_competing_processes()
      - VRAM / temp monitoring
  - Expose run_model(model_path, config, run_config) -> ModelResult

optimize.py is never called as a subprocess here.  It is imported once
at module load and its globals are patched via reinitialize() before each
model run.  optimize.py requires only the three minimal patches already
applied (reinitialize, _NO_JINJA, _bootstrap_from_config).
"""

from __future__ import annotations

import json
import math
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Import optimize as a module -- config does NOT run at import time because
# we patched the module-level _config = _load_config() into a function.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
import optimize as _opt

IS_WINDOWS = platform.system() == "Windows"

# ---------------------------------------------------------------------------
# PRESET SYSTEM
# ---------------------------------------------------------------------------
# Each preset is a dict of:
#   phases: list of phase names to run (order matters)
#   trials: dict of phase_name -> trial count
#   description: human-readable label
#
# Phases available from optimize.py:
#   gpu          - GPU layer offload sweep (dense only)
#   moe          - MoE thread sweep
#   experts      - Expert count sweep with quality gate (MoE only, optional)
#   compute      - Compute allocation (threads, speculation, poll, prio)
#   memory       - Memory & throughput (batch, KV, flash-attn)
#   moe_audit    - MoE re-validation after compute+memory
#   compute_audit- Compute re-validation after memory
#   memory_audit - Memory re-validation after compute_audit
#   quality      - Sampling params (temp, top_p, etc.)

PRESETS: dict[str, dict] = {
    "reduced": {
        "description": "Reduced: GPU + compute + memory (30+30 trials). ~25 min/model.",
        "phases": ["gpu", "moe", "compute", "memory"],
        "trials": {"compute": 30, "memory": 30},
    },
    "standard": {
        "description": "Standard: all main phases, no audits or quality (60+60 trials). ~50 min/model.",
        "phases": ["gpu", "moe", "compute", "memory"],
        "trials": {"compute": 60, "memory": 60},
    },
    "thorough": {
        "description": "Thorough: main phases + audits (60+60+60+60 trials). ~2h/model.",
        "phases": ["gpu", "moe", "compute", "memory", "moe_audit", "compute_audit", "memory_audit"],
        "trials": {"compute": 60, "memory": 60, "compute_audit": 60, "memory_audit": 60},
    },
    "full": {
        "description": "Full: all phases including quality/sampling (60+60+60+60+80 trials). ~3h/model.",
        "phases": ["gpu", "moe", "compute", "memory", "moe_audit", "compute_audit", "memory_audit", "quality"],
        "trials": {"compute": 60, "memory": 60, "compute_audit": 60, "memory_audit": 60, "quality": 80},
    },
    "moe_deep": {
        "description": "MoE deep: adds expert count sweep with quality gate. Slow. For MoE models only.",
        "phases": ["gpu", "moe", "experts", "compute", "memory", "moe_audit", "compute_audit", "memory_audit"],
        "trials": {"compute": 60, "memory": 60, "compute_audit": 60, "memory_audit": 60},
    },
    "fast": {
        "description": "Fast: screening + binary GPU/MoE + compute + memory + diagnostics. ~30 min/model.",
        "phases": ["binary_screen", "fast_gpu", "fast_moe", "compute", "memory",
                   "integrity", "reasoning_greedy"],
        "trials": {"compute": 30, "memory": 25},
    },
    "ik": {
        "description": "IK: standard phases + IK_llama.cpp contrast benchmark at the end.",
        "phases": ["gpu", "moe", "compute", "memory", "ik_contrast"],
        "trials": {"compute": 60, "memory": 60},
    },
    "ik_thorough": {
        "description": "IK thorough: full audits + IK contrast. ~2.5h/model.",
        "phases": ["gpu", "moe", "compute", "memory", "moe_audit", "compute_audit", "memory_audit", "ik_contrast"],
        "trials": {"compute": 60, "memory": 60, "compute_audit": 60, "memory_audit": 60},
    },
    "mtp": {
        "description": "MTP: standard phases + MTP draft sweep (MTP-capable models only). ~2–3h/model.",
        "phases": ["gpu", "moe", "compute", "memory", "mtp_spec"],
        "trials": {"compute": 60, "memory": 60},
    },
    "mtp_thorough": {
        "description": "MTP thorough: full audits + MTP sweep. ~4–5h/model.",
        "phases": ["gpu", "moe", "compute", "memory", "moe_audit", "compute_audit", "memory_audit", "mtp_spec"],
        "trials": {"compute": 60, "memory": 60, "compute_audit": 60, "memory_audit": 60},
    },
    "full_plus": {
        "description": "Full+: quality + IK contrast + MTP sweep — everything. ~5–6h/model.",
        "phases": ["gpu", "moe", "compute", "memory", "moe_audit", "compute_audit",
                   "memory_audit", "quality", "ik_contrast", "mtp_spec"],
        "trials": {"compute": 60, "memory": 60, "compute_audit": 60, "memory_audit": 60, "quality": 80},
    },
}

PRESET_NAMES = list(PRESETS.keys())  # auto-derived from PRESETS dict above


@dataclass
class RunConfig:
    """Controls which phases run and how many trials each gets."""
    preset: str = "standard"
    phases: list[str] = field(default_factory=list)      # override preset phases
    trials: dict[str, int] = field(default_factory=dict) # override preset trial counts
    skip_phases: list | None = field(default=None)          # None=not set, []=skip-all-except-rerun, [a,b]=skip those
    rerun_phases: list[str] = field(default_factory=list)  # force-rerun even if results exist
    timeout_per_phase: int = 0  # 0 = no per-phase timeout

    def resolved_phases(self) -> list[str]:
        base = self.phases if self.phases else PRESETS[self.preset]["phases"]
        skip = self.skip_phases
        if skip is None:
            # --skip-phases not passed at all — skip nothing
            return list(base)
        if skip == []:
            # bare --skip-phases with no args — skip everything except rerun_phases
            return [p for p in base if p in self.rerun_phases]
        # --skip-phases a b c — skip those specific phases
        return [p for p in base if p not in skip]

    def trial_count(self, phase: str) -> int:
        if phase in self.trials:
            return self.trials[phase]
        return PRESETS[self.preset]["trials"].get(phase, 60)


@dataclass
class ModelResult:
    """Result from running one model through the adapter."""
    model_path: Path
    status: str               # "ok" | "failed" | "timeout" | "baseline_failed"
    error: str = ""
    elapsed_s: float = 0.0
    best_tps: float = 0.0
    best_score: float = 0.0
    best_config_cmd: str = ""
    phases_run: list[str] = field(default_factory=list)
    phases_results: dict = field(default_factory=dict)   # phase_name -> result dict
    no_jinja: bool = False
    ik_best_tps: float = 0.0
    ik_gain_vs_llama_pct: float = 0.0
    ik_best_label: str = ""
    mtp_best_tps: float = 0.0        # best TPS from MTP sweep
    mtp_gain_pct: float = 0.0        # gain vs no-MTP baseline
    mtp_best_label: str = ""         # winning MTP config description
    mtp_available: bool = False      # whether MTP was detected/forced


# ---------------------------------------------------------------------------
# GPU / hardware helpers (ported from sweep_engine.py)
# ---------------------------------------------------------------------------

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False


def detect_gpus() -> list[dict]:
    """Return list of {index, name, vram_gb} sorted by VRAM descending."""
    gpus = []
    if HAS_NVML:
        try:
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                vram_gb = pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024 ** 3)
                gpus.append({"index": i, "name": name, "vram_gb": vram_gb})
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
    return gpus


def get_gpu_temp() -> float:
    if not HAS_NVML:
        return 0.0
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:
        return 0.0


def wait_cool(threshold: int = 82, target: int = 72):
    while True:
        t = get_gpu_temp()
        if t == 0 or t <= threshold:
            return
        print(f"  [thermal] GPU at {t:.0f}C -- waiting to cool to {target}C...")
        while get_gpu_temp() > target:
            time.sleep(5)
        return


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def kill_competing_processes():
    if IS_WINDOWS:
        for proc in ["ollama.exe", "ollama_llama_server.exe", "LM Studio.exe"]:
            subprocess.run(["taskkill", "/F", "/IM", proc], capture_output=True)
    else:
        subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
        subprocess.run(["pkill", "-f", "lm-studio"], capture_output=True)


def startup_timeout_for(model_path: Path) -> int:
    """Base 30s + 4s/GB, 1.5x for MoE or large models (>20GB), cap 300s."""
    env_override = os.environ.get("LLM_OPT_STARTUP_TIMEOUT")
    if env_override:
        return int(env_override)
    try:
        size_gb = model_path.stat().st_size / (1024 ** 3)
    except OSError:
        size_gb = 0.0
    t = 30 + size_gb * 4
    # MoE detection by filename patterns
    name = model_path.name.lower()
    import re
    is_moe = any([
        re.search(r'\d+b-a\d+b', name),
        re.search(r'\d+x\d+b', name),
        'moe' in name,
    ])
    if is_moe or size_gb > 20:
        t *= 1.5
    return min(300, int(t))


# ---------------------------------------------------------------------------
# Server lifecycle (wraps optimize.py's kill_server / wait_for_server)
# with our jinja retry, port-down detection, and OOM diagnosis.
# ---------------------------------------------------------------------------

_NO_JINJA_MODELS: set[str] = set()   # model paths that confirmed need --no-jinja


def _drain_stdout(proc, buf: list):
    try:
        for line in proc.stdout:
            buf.append(line.decode("utf-8", errors="replace"))
    except Exception:
        pass


def start_and_wait(
    model_path: Path,
    engine_config: dict,
    port: int,
    timeout_s: Optional[int] = None,
) -> tuple[Optional[object], bool]:
    """Start llama-server via optimize.start_server(), wait for health.

    Returns (proc, no_jinja_used).
    Handles: jinja template errors (auto-retry), OOM crash detection,
    port-open monitoring.
    """
    if timeout_s is None:
        timeout_s = startup_timeout_for(model_path)

    # If we already know this model needs --no-jinja, set flag before starting
    model_key = str(model_path)
    if model_key in _NO_JINJA_MODELS:
        _opt._NO_JINJA = True

    kill_competing_processes()
    _opt.kill_server()
    time.sleep(1)

    def _try_start(no_jinja: bool) -> Optional[object]:
        _opt._NO_JINJA = no_jinja
        proc = _opt.start_server(engine_config)

        # Drain stdout in background to prevent pipe buffer deadlock
        stdout_buf: list[str] = []
        if hasattr(proc, 'stdout') and proc.stdout:
            threading.Thread(target=_drain_stdout, args=(proc, stdout_buf), daemon=True).start()
        stderr_buf = getattr(proc, '_stderr_lines', [])

        t0 = time.time()
        port_was_open = False
        in_nojinja_retry = no_jinja

        while time.time() - t0 < timeout_s:
            # Check for crash
            if proc.poll() is not None:
                err = "".join(stderr_buf) + "".join(stdout_buf)
                jinja_fail = ("chat template parsing error" in err or
                              "Unable to generate parser" in err)
                if jinja_fail and not no_jinja:
                    print(f"      Retrying without --jinja (broken chat template)...")
                    _opt.kill_server()
                    return _try_start(no_jinja=True)
                print(f"      Crashed: ...{err[-300:]}")
                return None

            # Check health
            try:
                r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    if in_nojinja_retry:
                        _NO_JINJA_MODELS.add(model_key)
                    return proc
                port_was_open = True
            except Exception:
                # Port went down after being up -> OOM or crash
                if port_was_open and not is_port_open(port):
                    time.sleep(2)  # wait for stdout drain
                    err = "".join(stderr_buf) + "".join(stdout_buf)
                    fit_ok = ("successfully fit params" in err or "no changes needed" in err)
                    jinja_fail = ("chat template parsing error" in err or
                                  "Unable to generate parser" in err)
                    if (jinja_fail or fit_ok) and not no_jinja:
                        reason = "jinja error" if jinja_fail else "post-fit exit (likely jinja)"
                        print(f"      Retrying without --jinja ({reason})...")
                        _opt.kill_server()
                        return _try_start(no_jinja=True)
                    if err:
                        print(f"      Died after partial load: ...{err[-300:]}")
                    else:
                        print(f"      Died after partial load (OOM likely)")
                    return None

            time.sleep(0.1)  # 100ms polling (vs old 1s)

        # Timeout
        time.sleep(2)
        err = "".join(stderr_buf) + "".join(stdout_buf)
        jinja_fail = ("chat template parsing error" in err or "Unable to generate parser" in err)
        if jinja_fail and not no_jinja:
            print(f"      Timeout -- jinja error detected, retrying without --jinja...")
            proc.kill()
            _opt.kill_server()
            return _try_start(no_jinja=True)

        print(f"      Startup timed out ({timeout_s}s)")
        if err.strip():
            print(f"      Last output: ...{err[-200:]}")
        proc.kill()
        return None

    proc = _try_start(no_jinja=(model_key in _NO_JINJA_MODELS))
    no_jinja_used = model_key in _NO_JINJA_MODELS
    return proc, no_jinja_used


# ---------------------------------------------------------------------------
# Phase orchestration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 0.5 binary screening (ported from Checkpoint A LLM_Optimiser_lmstudio.py)
# Runs before Compute phase in the "fast" preset. Tests a small set of binary
# flags that move the needle and that optimize.py's sweep phases don't cover:
#   - kv_offload: disabling KV offload is occasionally faster on PCIe 3.0
#   - ngram_spec: n-gram speculation, +100-200% on repetitive output models
#   - prio: process priority, rarely but meaningfully affects scheduling
# mlock, repack, op_offload etc. are deliberately excluded -- they're already
# swept by optimize.py's Memory phase across 25+ trials.
# ---------------------------------------------------------------------------

_FAST_SCREEN_FLAGS = [
    ("kv_offload",  False, "Disable KV cache offloading"),
    ("ngram_spec",  True,  "Enable n-gram speculation (draftless)"),
]
_FAST_PRIO_VALUES = [2, 3]   # high + realtime only; low/medium rarely win
_SCREEN_RUNS = 2             # median of 2 for stability


def run_binary_screening(results_dir: Path, port: int) -> dict:
    """Fast binary flag screening -- runs 3-4 server starts, ~3 minutes.

    Tests kv_offload, ngram_spec, and top prio values against baseline.
    Returns dict of winning flag overrides to lock into subsequent phases.
    Saves results to screening.json for reporting.
    """
    import requests as _req

    screen_path = results_dir / "screening.json"
    if screen_path.exists():
        try:
            data = json.loads(screen_path.read_text(encoding="utf-8"))
            winners = data.get("winners", {})
            print(f"  [resume] Binary screening already complete -- winners: {winners}")
            return winners
        except Exception:
            pass

    base_url = f"http://127.0.0.1:{port}"

    def _bench(n_runs=_SCREEN_RUNS):
        """Quick TPS measurement using /completion (matches optimize.py measurement)."""
        samples = []
        payload = {
            "prompt": "<|im_start|>user\nWrite a Python function that implements "
                      "binary search. Include docstring and type hints."
                      "<|im_end|>\n<|im_start|>assistant\n",
            "n_predict": 150,
            "temperature": 0.0,
            "cache_prompt": False,
        }
        for _ in range(n_runs):
            try:
                r = _req.post(f"{base_url}/completion",
                              json=payload, timeout=60)
                if r.status_code == 200:
                    t = r.json().get("timings", {})
                    tps = t.get("predicted_per_second", 0)
                    if tps > 0:
                        samples.append(tps)
            except Exception:
                pass
        if not samples:
            return 0.0
        samples.sort()
        return samples[len(samples) // 2]

    # Baseline -- server should already be running from optimize.py phase startup
    print(f"\n  [fast-screen] Measuring baseline TPS...")
    baseline_tps = _bench(3)
    if baseline_tps == 0:
        print(f"  [fast-screen] Baseline failed -- skipping screening")
        return {}

    print(f"  [fast-screen] Baseline: {baseline_tps:.1f} t/s")
    print(f"  [fast-screen] Testing {len(_FAST_SCREEN_FLAGS)} flags + {len(_FAST_PRIO_VALUES)} prio values")

    winners: dict = {}
    screen_results = []

    for flag_name, test_value, description in _FAST_SCREEN_FLAGS:
        # Apply the flag override via optimize.py globals then restart
        wait_cool()
        t0 = time.time()
        print(f"  [fast-screen] {flag_name}={test_value} ({description})...")

        # Translate flag to actual start_server keys and save originals for revert
        saved_keys = {}
        if flag_name == "ngram_spec" and test_value is True:
            ngram_keys = {"spec_type": "ngram-map-k4v", "spec_ngram_n": 16,
                          "spec_ngram_m": 32, "spec_ngram_min_hits": 2,
                          "draft_max": 8, "draft_min": 1, "draft_p_min": 0.7}
            for k, v in ngram_keys.items():
                saved_keys[k] = _opt.NAKED_ENGINE.get(k)
                _opt.NAKED_ENGINE[k] = v
        else:
            saved_keys[flag_name] = _opt.NAKED_ENGINE.get(flag_name)
            _opt.NAKED_ENGINE[flag_name] = test_value

        _opt.kill_server()
        proc = _opt.start_server(_opt.NAKED_ENGINE)
        if not _opt.wait_for_server(proc=proc):
            print(f"    FAILED [{time.time()-t0:.1f}s]")
            # Revert all saved keys
            for k, v in saved_keys.items():
                if v is None:
                    _opt.NAKED_ENGINE.pop(k, None)
                else:
                    _opt.NAKED_ENGINE[k] = v
            screen_results.append({"flag": flag_name, "value": test_value,
                                   "status": "crash", "delta_pct": 0})
            continue

        tps = _bench()
        delta = ((tps - baseline_tps) / baseline_tps * 100) if baseline_tps else 0
        status = "FASTER" if delta > 2 else ("SLOWER" if delta < -2 else "NEUTRAL")
        marker = ">>>" if delta > 2 else ("<<<" if delta < -2 else "   ")
        print(f"    {marker} {tps:.1f} t/s ({delta:+.1f}%) [{status}]  [{time.time()-t0:.1f}s]")

        screen_results.append({"flag": flag_name, "value": test_value,
                               "status": status, "gen_tps": round(tps, 2),
                               "delta_pct": round(delta, 1)})

        if delta > 2:
            winners[flag_name] = test_value
        else:
            # Revert -- flag didn't help
            for k, v in saved_keys.items():
                if v is None:
                    _opt.NAKED_ENGINE.pop(k, None)
                else:
                    _opt.NAKED_ENGINE[k] = v

    # Prio sweep -- keep best if any improve
    best_prio_tps = baseline_tps
    for prio_val in _FAST_PRIO_VALUES:
        wait_cool()
        t0 = time.time()
        print(f"  [fast-screen] prio={prio_val}...")
        orig_prio = _opt.NAKED_ENGINE.get("prio", 0)
        _opt.NAKED_ENGINE["prio"] = prio_val
        _opt.kill_server()
        proc = _opt.start_server(_opt.NAKED_ENGINE)
        if not _opt.wait_for_server(proc=proc):
            print(f"    FAILED [{time.time()-t0:.1f}s]")
            _opt.NAKED_ENGINE["prio"] = orig_prio
            continue

        tps = _bench()
        delta = ((tps - baseline_tps) / baseline_tps * 100) if baseline_tps else 0
        status = "FASTER" if delta > 2 else ("SLOWER" if delta < -2 else "NEUTRAL")
        marker = ">>>" if delta > 2 else "   "
        print(f"    {marker} {tps:.1f} t/s ({delta:+.1f}%) [{status}]  [{time.time()-t0:.1f}s]")

        screen_results.append({"flag": "prio", "value": prio_val,
                               "status": status, "gen_tps": round(tps, 2),
                               "delta_pct": round(delta, 1)})

        if tps > best_prio_tps:
            best_prio_tps = tps
            winners["prio"] = prio_val
        else:
            _opt.NAKED_ENGINE["prio"] = orig_prio

    print(f"\n  [fast-screen] Winners: {winners if winners else 'none'}")

    results_dir.mkdir(parents=True, exist_ok=True)
    screen_path.write_text(
        json.dumps({"baseline_tps": baseline_tps, "results": screen_results,
                    "winners": winners}, indent=2),
        encoding="utf-8"
    )
    return winners


# ---------------------------------------------------------------------------
# Phase 3 integrity -- KV cache quality comparison (ported from Checkpoint A)
# For "fast" preset: tests the single KV type chosen by Memory phase vs f16
# baseline. 10 deterministic prompts, no server restart between prompts.
# Answers: "does q4_0 KV actually degrade output quality on this model?"
# ---------------------------------------------------------------------------

_INTEGRITY_PROMPTS_FAST = [
    {"name": "fibonacci",
     "messages": [{"role": "user", "content":
        "Write a Python function that returns the first 10 Fibonacci numbers as a list. "
        "Output ONLY the function, no explanation."}],
     "max_tokens": 200},
    {"name": "sort_algorithm",
     "messages": [{"role": "user", "content":
        "Write a Python function called bubble_sort that sorts a list of integers. "
        "Output ONLY the function."}],
     "max_tokens": 250},
    {"name": "explain_recursion",
     "messages": [{"role": "user", "content":
        "Explain recursion in exactly 3 sentences. Be precise and technical."}],
     "max_tokens": 150},
    {"name": "sql_query",
     "messages": [{"role": "user", "content":
        "Write a SQL query that finds the top 5 customers by total order amount "
        "from tables customers(id, name) and orders(id, customer_id, amount). "
        "Output ONLY the SQL."}],
     "max_tokens": 150},
]


def run_fast_gpu_sweep(results_dir: Path, port: int) -> int | None:
    """Binary search for best n_gpu_layers. Fast alternative to phase_gpu_offload.

    Tests a small set of strategic layer counts rather than the full middle-out sweep:
      1. All layers on GPU (baseline)
      2. Half layers on GPU
      3. Three-quarter layers on GPU
      4. One-quarter layers on GPU
    Then one refinement pass around the winner using midpoints between adjacent tested values.

    Dense models only -- MoE models skip (MoE phase handles CPU split).
    Returns best n_gpu_layers, or None if model is MoE / already has results.
    Updates optimize.py globals so subsequent phases use the winner.
    """
    import requests as _req

    if _opt.IS_MOE:
        print("\n  [fast-gpu] MoE model -- skipping (MoE phase handles CPU split)")
        return None

    existing = _opt.load_phase_results("gpu")
    if existing and "best_ngl" in existing:
        best = existing["best_ngl"]
        print(f"\n  [fast-gpu] Already complete -- n_gpu_layers={best}")
        _opt.DEFAULT_GPU_LAYERS = best
        _opt.NAKED_ENGINE["n_gpu_layers"] = best
        return best

    max_ngl = _opt.MAX_GPU_LAYERS
    if max_ngl <= 1:
        print(f"\n  [fast-gpu] {max_ngl} layers -- nothing to sweep")
        return max_ngl

    base_url = f"http://127.0.0.1:{port}"

    def _bench_ngl(ngl: int) -> float:
        """Start server at ngl, benchmark, return TPS (0.0 on failure)."""
        config = {**_opt.NAKED_ENGINE, "n_gpu_layers": ngl}
        _opt.kill_server()
        proc = _opt.start_server(config)
        if not _opt.wait_for_server(proc=proc):
            print(f"    ngl={ngl}: server failed to start")
            return 0.0
        # Warmup call — lets CUDA caches settle before the real measurement
        try:
            import requests as _rq
            _rq.post(f"http://127.0.0.1:{port}/completion",
                     json={"prompt": "Hello", "n_predict": 4, "temperature": 0.0},
                     timeout=30)
        except Exception:
            pass
        perf, _ = _opt.measure_perf_adaptive(_best := 0.0)
        _opt.kill_server()
        return perf.get("tps", 0.0)

    # Strategic probe points: all, 3/4, 1/2, 1/4
    probe_fracs = [1.0, 0.75, 0.5, 0.25]
    probe_ngls  = sorted({max(1, int(max_ngl * f)) for f in probe_fracs}, reverse=True)
    # Always include max_ngl explicitly
    if max_ngl not in probe_ngls:
        probe_ngls.insert(0, max_ngl)

    print(f"\n{'='*60}")
    print(f"  Fast GPU Sweep  (dense, max_ngl={max_ngl})")
    print(f"{'='*60}")
    print(f"  Probing {len(probe_ngls)} layer counts: {probe_ngls}")

    scores: dict[int, float] = {}
    for ngl in probe_ngls:
        tps = _bench_ngl(ngl)
        scores[ngl] = tps
        marker = " *" if tps == max(scores.values()) else ""
        print(f"    ngl={ngl:3d}: {tps:.1f} t/s{marker}")

    # One refinement: find midpoints on either side of the current winner
    best_ngl = max(scores, key=scores.get)
    sorted_ngls = sorted(scores.keys())
    idx = sorted_ngls.index(best_ngl)
    refine_candidates = set()
    if idx > 0:
        refine_candidates.add((sorted_ngls[idx - 1] + best_ngl) // 2)
    if idx < len(sorted_ngls) - 1:
        refine_candidates.add((best_ngl + sorted_ngls[idx + 1]) // 2)
    refine_candidates -= set(scores.keys())

    if refine_candidates:
        print(f"  Refining around winner ({best_ngl}): {sorted(refine_candidates)}")
        for ngl in sorted(refine_candidates, reverse=True):
            tps = _bench_ngl(ngl)
            scores[ngl] = tps
            marker = " *" if tps == max(scores.values()) else ""
            print(f"    ngl={ngl:3d}: {tps:.1f} t/s{marker}")

    best_ngl  = max(scores, key=scores.get)
    best_tps  = scores[best_ngl]
    all_tps   = scores.get(max(scores.keys()), 0.0)  # all-GPU score
    gain_pct  = ((best_tps - all_tps) / all_tps * 100) if all_tps > 0 else 0.0

    print(f"\n  Winner: n_gpu_layers={best_ngl}  ({best_tps:.1f} t/s, "
          f"{gain_pct:+.1f}% vs all-GPU)")

    # Update optimize.py globals
    _opt.DEFAULT_GPU_LAYERS = best_ngl
    _opt.NAKED_ENGINE["n_gpu_layers"] = best_ngl
    _opt.save_phase_results("gpu", {
        "phase": "gpu", "best_ngl": best_ngl,
        "best_score": best_tps,
        "all_results": [{"ngl": k, "tps": v} for k, v in sorted(scores.items())],
    })
    return best_ngl


def run_fast_moe_sweep(results_dir: Path, port: int, force: bool = False) -> dict | None:
    """Sparse sweep for best n_cpu_moe. Fast alternative to phase_moe_threads.

    Tests 5 representative thread counts spread across the 0..max_threads range,
    then refines around the winner with up to 3 neighbours.

    MoE models only -- dense models skip.
    Returns {"n_cpu_moe": int, "expert_used_count": int} or None.
    Updates optimize.py globals so subsequent phases use the winner.
    """
    if not _opt.IS_MOE:
        print("\n  [fast-moe] Dense model -- skipping")
        return None

    existing = _opt.load_phase_results("moe_combined")
    if existing and "best_params" in existing and not force:
        bp = existing["best_params"]
        print(f"\n  [fast-moe] Already complete -- n_cpu_moe={bp.get('n_cpu_moe')}")
        return bp

    import os
    max_threads = os.cpu_count() or 16
    moe_max = min(max_threads * 2, 40)

    # 5 representative probes: 0, 25%, 50%, 75%, 100% of range
    probe_vals = sorted({
        0,
        max(1, moe_max // 4),
        max(1, moe_max // 2),
        max(1, moe_max * 3 // 4),
        moe_max,
    })

    def _bench_moe(n: int) -> float:
        config = {**_opt.NAKED_ENGINE, "n_cpu_moe": n}
        _opt.kill_server()
        proc = _opt.start_server(config)
        if not _opt.wait_for_server(proc=proc):
            print(f"    n_cpu_moe={n}: server failed")
            return 0.0
        perf, _ = _opt.measure_perf_adaptive(0.0)
        _opt.kill_server()
        return perf.get("tps", 0.0)

    print(f"\n{'='*60}")
    print(f"  Fast MoE Sweep  (max_threads={max_threads}, moe_max={moe_max})")
    print(f"{'='*60}")
    print(f"  Probing {len(probe_vals)} values: {probe_vals}")

    scores: dict[int, float] = {}
    for n in probe_vals:
        tps = _bench_moe(n)
        scores[n] = tps
        marker = " *" if tps == max(scores.values()) else ""
        print(f"    n_cpu_moe={n:2d}: {tps:.1f} t/s{marker}")

    best_n = max(scores, key=scores.get)

    # Refine: test ±2 neighbours not already tested
    refine = {best_n + d for d in (-2, -1, 1, 2)
              if 0 <= best_n + d <= moe_max and (best_n + d) not in scores}
    if refine:
        print(f"  Refining around winner ({best_n}): {sorted(refine)}")
        for n in sorted(refine):
            tps = _bench_moe(n)
            scores[n] = tps
            marker = " *" if tps == max(scores.values()) else ""
            print(f"    n_cpu_moe={n:2d}: {tps:.1f} t/s{marker}")

    best_n   = max(scores, key=scores.get)
    best_tps = scores[best_n]
    base_tps = scores.get(0, 0.0)
    gain_pct = ((best_tps - base_tps) / base_tps * 100) if base_tps > 0 else 0.0
    print(f"\n  Winner: n_cpu_moe={best_n}  ({best_tps:.1f} t/s, "
          f"{gain_pct:+.1f}% vs no MoE offload)")

    result = {
        "n_cpu_moe": best_n,
        "expert_used_count": _opt.DEFAULT_EXPERTS,
    }

    # Merge with expert count from metadata if available
    if hasattr(_opt, "ARCH") and _opt.ARCH:
        result["expert_used_count"] = _opt.ARCH.get("default_experts", _opt.DEFAULT_EXPERTS)

    _opt.save_phase_results("moe_combined", {
        "phase": "moe_combined",
        "best_params": result,
        "best_tps": best_tps,
        "best_metrics": {"tps": best_tps, "ttft": 0, "prompt_tps": 0, "total_ms": 0},
        "all_results": [{"n_cpu_moe": k, "tps": v} for k, v in sorted(scores.items())],
    })
    return result


def run_integrity_check(best_kv_type: str, results_dir: Path, port: int) -> dict:
    """Lightweight KV integrity check -- f16 vs the Memory phase winner.

    Runs 4 deterministic prompts at temp=0, compares character-level similarity.
    No server restarts between prompts -- just a HTTP benchmark loop.
    Saves results to integrity.json.
    """
    import difflib
    import requests as _req

    integrity_path = results_dir / "integrity.json"
    if integrity_path.exists():
        try:
            print(f"  [integrity] Already complete -- skipping")
            return json.loads(integrity_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    base_url = f"http://127.0.0.1:{port}"

    def _query(messages, max_tokens):
        # Use /completion with explicit ChatML to avoid chat-template issues
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        prompt = "\n".join(parts) + "\n<|im_start|>assistant\n"
        try:
            r = _req.post(f"{base_url}/completion",
                          json={"prompt": prompt, "n_predict": max_tokens,
                                "temperature": 0.0, "cache_prompt": False},
                          timeout=120)
            if r.status_code == 200:
                return r.json().get("content", "")
        except Exception:
            pass
        return ""

    def _similarity(a, b):
        if not a and not b: return 1.0
        if not a or not b: return 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()

    configs_to_test = [("f16", "f16")]  # baseline
    if best_kv_type and best_kv_type != "f16":
        configs_to_test.append((best_kv_type, best_kv_type))

    print(f"\n  [integrity] Testing f16 vs {best_kv_type} KV cache quality ({len(_INTEGRITY_PROMPTS_FAST)} prompts)...")

    ref_outputs = {}
    results = {}

    for kv_k, kv_v in configs_to_test:
        label = f"K={kv_k}/V={kv_v}"
        cfg = {**_opt.NAKED_ENGINE, "cache_type_k": kv_k, "cache_type_v": kv_v}
        if kv_k != "f16" or kv_v != "f16":
            cfg["flash_attn"] = "on"

        _opt.kill_server()
        proc = _opt.start_server(cfg)
        if not _opt.wait_for_server(proc=proc):
            print(f"  [integrity] {label}: server failed")
            results[label] = {"status": "crash", "similarity": 0.0}
            continue

        outputs = {}
        for p in _INTEGRITY_PROMPTS_FAST:
            wait_cool()
            out = _query(p["messages"], p["max_tokens"])
            outputs[p["name"]] = out
            preview = out.replace("\n", " ")[:60]
            print(f"    [{p['name']}] {len(out)} chars | {preview}...")

        if kv_k == "f16" and kv_v == "f16":
            ref_outputs = outputs
            results[label] = {"status": "ok", "similarity": 1.0, "outputs": outputs}
        else:
            sims = [_similarity(ref_outputs.get(n, ""), outputs.get(n, ""))
                    for n in outputs]
            avg_sim = sum(sims) / len(sims) if sims else 0.0
            print(f"  [integrity] {label}: avg similarity to f16 = {avg_sim:.3f} "
                  f"({'OK' if avg_sim >= 0.85 else 'DEGRADED'})")
            results[label] = {"status": "ok", "similarity": round(avg_sim, 4),
                              "outputs": outputs}

    integrity_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


# ---------------------------------------------------------------------------
# Phase 4 reasoning -- greedy-only subset (ported from Checkpoint A)
# For "fast" preset: 19 problems at temp=0 only (greedy). ~2 min.
# Tests math, code, and logic accuracy on the best config from Memory phase.
# Full 6-temperature sweep available in thorough/full presets via optimize.py
# quality phase (which runs sampling optimization after this).
# ---------------------------------------------------------------------------

_REASONING_PROBLEMS_FAST = [
    {"name": "math_arithmetic",  "category": "math",
     "prompt": "A store has 47 apples. 8 customers each buy 3 apples. A delivery of 15 apples arrives. "
               "How many apples does the store have now? Answer with a single number on the last line.",
     "answer": 38},
    {"name": "math_algebra",     "category": "math",
     "prompt": "If x=7 and y=2x^2 - 3x + 1, what is y? Answer with a single number on the last line.",
     "answer": 78},
    {"name": "math_percentage",  "category": "math",
     "prompt": "A jacket costs $80 and is discounted 15%. What is the sale price? "
               "Answer with a single number on the last line.",
     "answer": 68},
    {"name": "math_sequence",    "category": "math",
     "prompt": "What is the 8th number in the Fibonacci sequence (starting 1,1,2,3...)? "
               "Answer with a single number on the last line.",
     "answer": 21},
    {"name": "code_output",      "category": "code",
     "prompt": "What does this Python code print?\n\nx = [1,2,3,4,5]\nprint(sum(x[1:4]))\n\n"
               "Answer with a single number on the last line.",
     "answer": 9},
    {"name": "code_debug",       "category": "code",
     "prompt": "This function always returns None. Why?\n\ndef double(x):\n    y = x * 2\n\n"
               "Answer in one sentence: what is missing?",
     "answer": "return"},
    {"name": "logic_deduction",  "category": "logic",
     "prompt": "All birds have wings. Penguins are birds. Do penguins have wings? "
               "Answer Yes or No on the last line.",
     "answer": "Yes"},
    {"name": "logic_syllogism",  "category": "logic",
     "prompt": "If it rains, the ground gets wet. The ground is wet. Did it definitely rain? "
               "Answer Yes or No on the last line.",
     "answer": "No"},
]


def run_reasoning_greedy(best_config: dict, results_dir: Path, port: int) -> dict:
    """Greedy reasoning check on the best config from Memory phase.

    Runs 8 problems at temperature=0 (greedy decoding).
    Checks exact answer presence in output.
    Saves results to reasoning_greedy.json.
    """
    import requests as _req

    reas_path = results_dir / "reasoning_greedy.json"
    if reas_path.exists():
        try:
            print(f"  [reasoning] Already complete -- skipping")
            return json.loads(reas_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    base_url = f"http://127.0.0.1:{port}"
    cfg = {**_opt.NAKED_ENGINE, **(best_config or {})}

    _opt.kill_server()
    proc = _opt.start_server(cfg)
    if not _opt.wait_for_server(proc=proc):
        print(f"  [reasoning] Server failed to start")
        return {}

    print(f"\n  [reasoning] {len(_REASONING_PROBLEMS_FAST)} problems @ greedy (temp=0)...")
    results = []
    correct_by_cat: dict = {}

    for p in _REASONING_PROBLEMS_FAST:
        wait_cool()
        prompt = f"<|im_start|>user\n{p['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        try:
            r = _req.post(f"{base_url}/completion",
                          json={"prompt": prompt, "n_predict": 300,
                                "temperature": 0.0, "cache_prompt": False},
                          timeout=90)
            content = ""
            if r.status_code == 200:
                content = r.json().get("content", "")
        except Exception:
            content = ""

        answer_str = str(p["answer"]).lower()
        correct = answer_str in content.lower()
        mark = "OK" if correct else "X "
        cat = p["category"]
        correct_by_cat.setdefault(cat, {"correct": 0, "total": 0})
        correct_by_cat[cat]["total"] += 1
        if correct:
            correct_by_cat[cat]["correct"] += 1

        preview = content.replace("\n", " ")[:60]
        print(f"    [{mark}] {p['name']:<22} expected={p['answer']}  | {preview}...")
        results.append({"name": p["name"], "category": cat,
                       "expected": p["answer"], "correct": correct,
                       "content_preview": content[:200]})

    total = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    print(f"\n  [reasoning] Score: {n_correct}/{total} ({n_correct/total*100:.0f}%)")
    for cat, d in correct_by_cat.items():
        print(f"    {cat}: {d['correct']}/{d['total']}")

    output = {"score": n_correct / total if total else 0,
              "correct": n_correct, "total": total,
              "by_category": correct_by_cat, "results": results}
    reas_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def _load_prior_result(results_dir: Path, phase: str) -> Optional[dict]:
    p = results_dir / f"{phase}_results.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def run_pipeline(
    model_path: Path,
    llama_server: Path,
    results_dir: Path,
    run_config: RunConfig,
    gpu_info: list[dict],
    port: int = 8090,
    no_jinja: bool = False,
    topo_overlay: Optional[dict] = None,
    topo_case: str = "",
    ctx_recommended: Optional[int] = None,
    resume: bool = True,
    ik_server_path: Optional[str] = None,
    force_mtp: bool = False,
) -> ModelResult:
    """Run the optimizer pipeline for one model.

    Calls optimize.reinitialize(), then runs each phase in run_config order.
    """
    t_start = time.time()
    phases_run: list[str] = []
    phases_results: dict = {}

    # Detect architecture using get_model_meta which resolves arch-prefixed keys
    # for any architecture (glm-dsa.*, nemotron_h_moe.*, etc.), not just llama.*
    try:
        from model_utils import get_model_meta
        mmeta = get_model_meta(model_path)
        n_expert      = mmeta.get("n_expert", 0)
        n_expert_used = mmeta.get("n_expert_used", 0)
        arch_name     = mmeta.get("arch", "llama")
        is_moe        = bool(n_expert > 0)
        arch_type     = "moe" if is_moe else "dense"
        arch = {
            "type": arch_type,
            "expert_override_key": f"{arch_name}.expert_used_count",
            "default_experts": n_expert_used or 8,
            "max_experts": n_expert or 16,
        }
    except Exception:
        is_moe = False
        arch = {"type": "dense"}

    # GPU layer count
    max_gpu_layers = None
    if topo_overlay and topo_overlay.get("num_gpu_layers"):
        max_gpu_layers = topo_overlay["num_gpu_layers"]

    # Determine CUDA device / split config from topo winner
    cuda_device = None
    if topo_overlay and topo_overlay.get("cuda_visible_devices") is not None:
        cuda_device = str(topo_overlay["cuda_visible_devices"])
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Extract tensor_split and main_gpu from topo overlay for multi-GPU configs
    topo_gpu_flags: dict = {}
    if topo_overlay:
        if topo_overlay.get("tensor_split"):
            topo_gpu_flags["tensor_split"] = topo_overlay["tensor_split"]
        if topo_overlay.get("main_gpu") is not None:
            topo_gpu_flags["main_gpu"] = topo_overlay["main_gpu"]

    # Context size from ctx sweep recommendation
    context_size = ctx_recommended or 4096

    # Initialize optimize globals for this model
    _opt.reinitialize(
        model_path=model_path,
        llama_server_path=llama_server,
        results_dir=results_dir,
        port=port,
        arch=arch,
        max_gpu_layers=max_gpu_layers,
        max_threads=None,   # auto-detect
        no_jinja=(str(model_path) in _NO_JINJA_MODELS) or no_jinja,
        ik_server_path=ik_server_path or os.environ.get("IK_LLAMA_SERVER", ""),
        force_mtp=force_mtp,
    )
    # Set context and multi-GPU flags for all phases
    _opt.NAKED_ENGINE["context"] = context_size
    _opt.NAKED_ENGINE.update(topo_gpu_flags)   # tensor_split, main_gpu (empty for single-GPU)

    resolved = run_config.resolved_phases()

    # Filter out MoE phases for dense models
    if not is_moe:
        resolved = [p for p in resolved if p not in ("moe", "experts", "moe_audit")]

    print(f"\n  Phases to run: {resolved}")

    # ---- Track phase-to-phase state (results chain) ----
    # Dense models use empty dict so phase_compute doesn't inject --n-cpu-moe.
    # MoE models start with sweep-center defaults, overwritten by moe phase.
    best_moe = (
        {"n_cpu_moe": _opt.MOE_SWEEP_CENTER, "expert_used_count": _opt.DEFAULT_EXPERTS}
        if is_moe else {}
    )
    # Ensure dense model flag is propagated into optimize.py globals so
    # phase_compute never injects --n-cpu-moe for non-MoE models.
    _opt.IS_MOE = is_moe
    compute_best: dict = {}
    memory_best: dict = {}

    def _skip(phase: str) -> bool:
        """Return True and print skip message if phase already has results.
        Always returns False for phases in rerun_phases, forcing a re-run."""
        if phase in run_config.rerun_phases:
            print(f"  [rerun] {phase} forced re-run (--rerun-phases)")
            return False
        if resume:
            existing = _load_prior_result(results_dir, phase)
            if existing and "best_params" in existing:
                print(f"  [resume] {phase} already complete -- skipping")
                phases_results[phase] = existing
                return True
        return False

    status = "ok"
    error = ""

    # Write a sentinel immediately so --resume can skip this model even if
    # every phase fails to produce results (e.g. llama-server never starts).
    _sentinel = results_dir / "run_attempted.json"
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        _sentinel.write_text(
            json.dumps({"model": model_path.name,
                        "timestamp": __import__("datetime").datetime.now().isoformat(),
                        "preset": run_config.name if hasattr(run_config, "name") else ""},
                       indent=2), encoding="utf-8")
    except Exception:
        pass

    # Screening winners to pass as base overrides into Compute phase
    _screen_winners: dict = {}
    # Best KV type from Memory phase (for integrity check)
    _best_kv_type: str = "f16"
    # Best compute+memory config for reasoning check
    _best_final_config: dict = {}

    try:
        for phase in resolved:
            wait_cool()

            if phase == "binary_screen":
                # Fast binary screening: kv_offload, ngram_spec, prio
                # Server must be running -- start naked engine first if needed
                _opt.kill_server()
                _proc = _opt.start_server(_opt.NAKED_ENGINE)
                if _opt.wait_for_server(proc=_proc):
                    _screen_winners = run_binary_screening(results_dir, _opt.PORT)
                    # Translate winners into NAKED_ENGINE keys that start_server() understands
                    for k, v in _screen_winners.items():
                        if k == "ngram_spec" and v is True:
                            # Enable n-gram speculation with sensible defaults
                            _opt.NAKED_ENGINE["spec_type"] = "ngram-map-k4v"
                            _opt.NAKED_ENGINE["spec_ngram_n"] = 16
                            _opt.NAKED_ENGINE["spec_ngram_m"] = 32
                            _opt.NAKED_ENGINE["spec_ngram_min_hits"] = 2
                            _opt.NAKED_ENGINE["draft_max"] = 8
                            _opt.NAKED_ENGINE["draft_min"] = 1
                            _opt.NAKED_ENGINE["draft_p_min"] = 0.7
                        elif k == "kv_offload" and v is False:
                            _opt.NAKED_ENGINE["kv_offload"] = False
                        elif k == "prio":
                            _opt.NAKED_ENGINE["prio"] = v
                    phases_run.append("binary_screen")
                _opt.kill_server()

            elif phase == "fast_gpu":
                # Only meaningful for Case D (partial CPU offload) — when the model
                # fits fully in VRAM (Cases A/B/C) ngl=max is always optimal.
                if topo_case and topo_case != "D":
                    print(f"\n  [fast-gpu] Skipping — model fully fits in VRAM (Case {topo_case}), ngl=max is optimal")
                else:
                    _fgpu = run_fast_gpu_sweep(results_dir, _opt.PORT)
                    if _fgpu is not None:
                        phases_run.append("fast_gpu")

            elif phase == "fast_moe":
                # Sparse MoE thread sweep (MoE only, skipped internally for dense)
                _force_moe = "fast_moe" in run_config.rerun_phases
                _fmoe = run_fast_moe_sweep(results_dir, _opt.PORT, force=_force_moe)
                if _fmoe is not None:
                    best_moe = _fmoe
                    phases_run.append("fast_moe")

            elif phase == "integrity":
                # KV integrity: f16 vs best KV type from Memory phase
                _int_result = run_integrity_check(_best_kv_type, results_dir, _opt.PORT)
                phases_run.append("integrity")
                # If the best KV type is degraded, fall back to f16 for
                # subsequent reasoning test and final config
                if _best_kv_type != "f16" and _int_result:
                    _kv_label = f"K={_best_kv_type}/V={_best_kv_type}"
                    _kv_sim = (_int_result.get(_kv_label) or {}).get("similarity", 1.0)
                    if _kv_sim < 0.85:
                        print(f"  [integrity] DEGRADED ({_kv_sim:.3f}) — "
                              f"falling back to f16 KV for reasoning and final config")
                        _best_kv_type = "f16"
                        # Strip degraded KV from final config
                        _best_final_config.pop("kv_cache_type", None)
                        _best_final_config.pop("cache_type_k", None)
                        _best_final_config.pop("cache_type_v", None)
                        _best_final_config.pop("flash_attn", None)
                _opt.kill_server()

            elif phase == "reasoning_greedy":
                # Greedy reasoning check on best final config
                run_reasoning_greedy(_best_final_config, results_dir, _opt.PORT)
                phases_run.append("reasoning_greedy")
                _opt.kill_server()

            elif phase == "gpu":
                if not _skip("gpu"):
                    _opt.phase_gpu_offload()
                    phases_run.append("gpu")
                r = _load_prior_result(results_dir, "gpu")
                if r and "best_ngl" in r:
                    _opt.DEFAULT_GPU_LAYERS = r["best_ngl"]
                    _opt.NAKED_ENGINE["n_gpu_layers"] = r["best_ngl"]
                phases_results["gpu"] = r or {}

            elif phase == "moe":
                if not _skip("moe_combined"):
                    result = _opt.phase_moe(
                        n_trials=run_config.trial_count("moe"),
                        include_experts=False,
                    )
                    if result:
                        best_moe = result
                    phases_run.append("moe")
                r = _load_prior_result(results_dir, "moe_combined")
                if r and "best_params" in r:
                    best_moe = r["best_params"]
                phases_results["moe"] = r or {}

            elif phase == "experts":
                if not _skip("experts"):
                    _opt.phase_experts(locked_moe_threads=best_moe.get("n_cpu_moe", 0))
                    phases_run.append("experts")
                r = _load_prior_result(results_dir, "experts")
                if r and "best_params" in r:
                    best_moe["expert_used_count"] = r["best_params"].get("expert_used_count", _opt.DEFAULT_EXPERTS)
                phases_results["experts"] = r or {}

            elif phase == "compute":
                if not _skip("compute"):
                    compute_best = _opt.phase_compute(
                        n_trials=run_config.trial_count("compute"),
                        phase_name="compute",
                        locked_moe=best_moe,
                    ) or {}
                    phases_run.append("compute")
                else:
                    r = _load_prior_result(results_dir, "compute")
                    compute_best = (r or {}).get("best_params", {})
                phases_results["compute"] = _load_prior_result(results_dir, "compute") or {}

            elif phase == "memory":
                if not _skip("memory"):
                    memory_best = _opt.phase_memory(
                        n_trials=run_config.trial_count("memory"),
                        phase_name="memory",
                        base_compute_config={**compute_best, **best_moe},
                    ) or {}
                    phases_run.append("memory")
                    # Capture best KV type for integrity check
                    _kv = memory_best.get("kv_cache_type") or memory_best.get("cache_type_k")
                    if _kv:
                        _best_kv_type = _kv
                    _best_final_config = {**compute_best, **memory_best, **best_moe}
                else:
                    r = _load_prior_result(results_dir, "memory")
                    memory_best = (r or {}).get("best_params", {})
                    # Also capture KV type on resume
                    _kv = memory_best.get("kv_cache_type") or memory_best.get("cache_type_k")
                    if _kv:
                        _best_kv_type = _kv
                    _best_final_config = {**compute_best, **memory_best, **best_moe}
                phases_results["memory"] = _load_prior_result(results_dir, "memory") or {}

            elif phase == "moe_audit":
                if not _skip("moe_audit"):
                    new_moe = _opt.phase_moe_revalidate(
                        locked_compute=compute_best,
                        locked_moe=best_moe,
                        base_memory_config=memory_best,
                    )
                    if new_moe is not None:
                        best_moe = {**best_moe, "n_cpu_moe": new_moe}
                    phases_run.append("moe_audit")
                phases_results["moe_audit"] = _load_prior_result(results_dir, "moe_audit") or {}

            elif phase == "compute_audit":
                if not _skip("compute_audit"):
                    compute_best = _opt.phase_compute(
                        n_trials=run_config.trial_count("compute_audit"),
                        phase_name="compute_audit",
                        base_memory_config=memory_best,
                        seed_params=compute_best,
                        locked_moe=best_moe,
                    ) or compute_best
                    phases_run.append("compute_audit")
                phases_results["compute_audit"] = _load_prior_result(results_dir, "compute_audit") or {}

            elif phase == "memory_audit":
                if not _skip("memory_audit"):
                    memory_best = _opt.phase_memory(
                        n_trials=run_config.trial_count("memory_audit"),
                        phase_name="memory_audit",
                        base_compute_config={**compute_best, **best_moe},
                        seed_params=memory_best,
                    ) or memory_best
                    phases_run.append("memory_audit")
                phases_results["memory_audit"] = _load_prior_result(results_dir, "memory_audit") or {}

            elif phase == "quality":
                if not _skip("quality"):
                    _opt.phase3(n_trials=run_config.trial_count("quality"))
                    phases_run.append("quality")
                phases_results["quality"] = _load_prior_result(results_dir, "quality") or {}

            elif phase == "ik_contrast":
                # IK_llama.cpp contrast
                if not _opt.IK_MODE:
                    print(f"\n  [ik_contrast] IK_llama.cpp not configured — skipping")
                else:
                    _ik_skip = resume and bool(_load_prior_result(results_dir, "ik_contrast"))
                    if _ik_skip and "ik_contrast" not in run_config.rerun_phases:
                        print(f"  [resume] ik_contrast already complete — skipping")
                        phases_results["ik_contrast"] = _load_prior_result(results_dir, "ik_contrast") or {}
                    else:
                        _opt.phase_ik_contrast(
                            locked_compute=compute_best or None,
                            locked_moe=best_moe or None,
                            locked_memory=memory_best or None,
                        )
                        phases_run.append("ik_contrast")
                    phases_results["ik_contrast"] = _load_prior_result(results_dir, "ik_contrast") or {}

            elif phase == "mtp_spec":
                # MTP draft sweep — only for models with MTP heads or --force-mtp
                _mtp_eligible = _opt.MTP_AVAILABLE or _opt.MTP_FORCE
                if not _mtp_eligible:
                    print(f"\n  [mtp_spec] Model has no MTP heads — skipping")
                    print(f"             (Use --force-mtp to run anyway)")
                else:
                    _mtp_skip = resume and bool(_load_prior_result(results_dir, "mtp_spec"))
                    if _mtp_skip and "mtp_spec" not in run_config.rerun_phases:
                        print(f"  [resume] mtp_spec already complete — skipping")
                        phases_results["mtp_spec"] = _load_prior_result(results_dir, "mtp_spec") or {}
                    else:
                        _opt.phase_mtp(
                            locked_compute=compute_best or None,
                            locked_moe=best_moe or None,
                            locked_memory=memory_best or None,
                        )
                        phases_run.append("mtp_spec")
                    phases_results["mtp_spec"] = _load_prior_result(results_dir, "mtp_spec") or {}

    except KeyboardInterrupt:
        status = "interrupted"
        error = "KeyboardInterrupt"
    except Exception as e:
        status = "error"
        error = str(e)
        import traceback
        traceback.print_exc()
    finally:
        _opt.kill_server()

    elapsed = time.time() - t_start

    # Extract best result from whichever phase ran last and has a score
    best_tps = 0.0
    best_score = 0.0
    best_cmd = ""
    for phase_name in ["memory_audit", "memory", "compute_audit", "compute", "moe"]:
        r = phases_results.get(phase_name, {})
        if r.get("best_tps"):
            bm = r.get("best_metrics", {})
            best_tps = bm.get("tps", r.get("best_tps", 0))
            best_score = r.get("best_tps", 0)
            break

    # Extract IK contrast result
    _ik_r  = phases_results.get("ik_contrast", {})
    _mtp_r = phases_results.get("mtp_spec", {})

    return ModelResult(
        model_path=model_path,
        status=status,
        error=error,
        elapsed_s=elapsed,
        best_tps=best_tps,
        best_score=best_score,
        best_config_cmd=best_cmd,
        phases_run=phases_run,
        phases_results=phases_results,
        no_jinja=(str(model_path) in _NO_JINJA_MODELS),
        ik_best_tps=_ik_r.get("ik_best_tps", 0.0),
        ik_gain_vs_llama_pct=_ik_r.get("ik_gain_vs_llama_pct", 0.0),
        ik_best_label=_ik_r.get("ik_best_label", ""),
        mtp_best_tps=_mtp_r.get("mtp_best_tps", 0.0),
        mtp_gain_pct=_mtp_r.get("mtp_gain_pct", 0.0),
        mtp_best_label=str(_mtp_r.get("best_params", {}).get("spec_draft_n_max", "")),
        mtp_available=bool(_mtp_r.get("mtp_available", False) or _mtp_r.get("mtp_force", False)),
    )
