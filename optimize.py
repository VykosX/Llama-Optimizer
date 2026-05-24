"""
llama-server Parameter Optimizer
================================
Multi-phase coordinate descent using Optuna (GP-Bayesian/TPE).

GPU:            GPU offload sweep — find optimal n_gpu_layers
MoE:            MoE sweep — find optimal n_cpu_moe + expert_used_count
Compute:        Compute allocation — threads, speculation, poll, prio (MoE locked)
Memory:         Memory & throughput — batch, ubatch, KV cache, flash-attn, etc.
Compute Audit:  Re-validate compute — same search on top of Memory best
MoE Audit:      MoE re-validation — re-test best ±2 with compute locked
Memory Audit:   Re-validate memory — same search on top of Compute Audit best
Quality:        Quality / sampling — temp, top-p, etc. scored by eval tasks

Flow: GPU → MoE → Compute → Memory → MoE Audit → Compute Audit → Memory Audit → Quality
Each phase seeds from the previous best. Re-validation catches cross-group interactions.

Usage:
  python optimize.py                    # use defaults
  python optimize.py --model /path/to/model.gguf
  python optimize.py --server /path/to/llama-server --port 8091
  python optimize.py --config config.json
"""

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import optuna
import requests
from scipy.stats import norm

# Auto-install dependencies if missing
for pkg, pip_name in [("sklearn", "scikit-learn"), ("scipy", "scipy")]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"[*] Installing {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "-q"])
        print(f"    Done.")

# ============================================================
# Configuration — defaults (overridable via CLI args or config.json)
# ============================================================

_find_llama_server_helper = None  # defined below


def _find_llama_server():
    """Auto-detect llama-server in PATH, env var, or sibling directories."""
    import shutil
    env = os.environ.get("LLAMA_SERVER")
    if env and Path(env).is_file():
        return env
    found = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if found:
        return found
    _here = Path(__file__).parent
    for candidate in [
        _here / "llama-server.exe",
        _here / "llama-server",
        _here.parent / "llama-server" / "llama-server.exe",
        _here.parent / "llama-server" / "llama-server",
        _here.parent / "LLama-Server" / "llama-server.exe",
    ]:
        if candidate.is_file():
            return str(candidate)
    return "llama-server"  # placeholder; set LLAMA_SERVER env var or use --server


def _find_ik_llama_server():
    """Auto-detect ik_llama-server in env var IK_LLAMA_SERVER or sibling directories."""
    import shutil
    env = os.environ.get("IK_LLAMA_SERVER")
    if env and Path(env).is_file():
        return env
    # Common build locations for ik_llama.cpp
    _here = Path(__file__).parent
    for candidate in [
        _here / "ik_llama-server.exe",
        _here / "ik_llama-server",
        _here.parent / "ik_llama.cpp" / "build" / "bin" / "llama-server.exe",
        _here.parent / "ik_llama.cpp" / "build" / "bin" / "llama-server",
        _here.parent / "IK_LLama-Server" / "llama-server.exe",
        _here.parent / "IK_LLama-Server" / "llama-server",
        _here.parent / "ik-llama-server" / "llama-server.exe",
        _here.parent / "ik-llama-server" / "llama-server",
    ]:
        if candidate.is_file():
            return str(candidate)
    return ""  # empty = not found


_DEFAULTS = {
    # Paths: override via --server / --model CLI args or env vars.
    # No hardcoded user-specific paths -- works on any machine.
    "server": _find_llama_server(),
    "ik_server": _find_ik_llama_server(),   # IK_LLAMA_SERVER env var or auto-detected
    "model": os.environ.get("LLM_OPT_DEFAULT_MODEL", "model.gguf"),
    # chat_template: empty = use --jinja (reads template embedded in GGUF).
    # All modern models include a chat template in their GGUF metadata.
    # Only set this to a .jinja file path if you need to force a specific template.
    "chat_template": os.environ.get("LLM_OPT_CHAT_TEMPLATE", ""),
    "results_dir": str(Path(__file__).parent / "results"),
    "port": 8090,

    # Model architecture — set "type" to "dense" to skip MoE phase entirely
    "architecture": {
        "type": "moe",                                      # "moe" or "dense"
        "expert_override_key": "qwen35moe.expert_used_count",  # GGUF key for expert count override
        "default_experts": 8,                                # trained default active experts
        "max_experts": 16,                                   # max experts to sweep
    },

    # Hardware — auto-detected if not set
    "hardware": {
        "max_threads": None,          # auto: os.cpu_count()
        "moe_sweep_max": None,        # auto: max_threads * 2 (capped at 40)
        "moe_sweep_center": None,     # auto: moe_sweep_max // 2
        "max_gpu_layers": None,       # auto-detected from model metadata, or 99
        "default_gpu_layers": 99,     # default -ngl for naked engine (99 = all GPU)
    },
}


def _detect_model_layers(model_path):
    """Try to read layer count from GGUF metadata.

    Reads the GGUF header to find the block_count key (e.g. 'llama.block_count').
    Returns int or None if detection fails.
    """
    try:
        p = Path(model_path)
        if not p.is_file():
            return None
        with open(p, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None
            version = int.from_bytes(f.read(4), "little")
            _tensor_count = int.from_bytes(f.read(8), "little")
            metadata_kv_count = int.from_bytes(f.read(8), "little")

            def read_string():
                length = int.from_bytes(f.read(8), "little")
                return f.read(length).decode("utf-8", errors="replace")

            def read_value(vtype):
                if vtype == 0:    # UINT8
                    return int.from_bytes(f.read(1), "little")
                elif vtype == 1:  # INT8
                    return int.from_bytes(f.read(1), "little", signed=True)
                elif vtype == 2:  # UINT16
                    return int.from_bytes(f.read(2), "little")
                elif vtype == 3:  # INT16
                    return int.from_bytes(f.read(2), "little", signed=True)
                elif vtype == 4:  # UINT32
                    return int.from_bytes(f.read(4), "little")
                elif vtype == 5:  # INT32
                    return int.from_bytes(f.read(4), "little", signed=True)
                elif vtype == 6:  # FLOAT32
                    import struct
                    return struct.unpack("<f", f.read(4))[0]
                elif vtype == 7:  # BOOL
                    return bool(f.read(1)[0])
                elif vtype == 8:  # STRING
                    return read_string()
                elif vtype == 9:  # ARRAY
                    elem_type = int.from_bytes(f.read(4), "little")
                    count = int.from_bytes(f.read(8), "little")
                    return [read_value(elem_type) for _ in range(count)]
                elif vtype == 10:  # UINT64
                    return int.from_bytes(f.read(8), "little")
                elif vtype == 11:  # INT64
                    return int.from_bytes(f.read(8), "little", signed=True)
                elif vtype == 12:  # FLOAT64
                    import struct
                    return struct.unpack("<d", f.read(8))[0]
                else:
                    return None

            for _ in range(metadata_kv_count):
                key = read_string()
                vtype = int.from_bytes(f.read(4), "little")
                val = read_value(vtype)
                if key.endswith(".block_count"):
                    return int(val)
    except Exception:
        pass
    return None


def _load_config():
    """Load config from CLI args, config.json, or defaults.

    Config file supports nested keys for architecture/hardware:
      {
        "server": "/path/to/llama-server",
        "model": "/path/to/model.gguf",
        "architecture": {"type": "dense"},
        "hardware": {"max_threads": 32}
      }
    """
    parser = argparse.ArgumentParser(description="llama-server Parameter Optimizer", add_help=False)
    parser.add_argument("--server", help="Path to llama-server executable")
    parser.add_argument("--ik-server", help="Path to ik_llama-server executable (IK_LLAMA_SERVER env var)")
    parser.add_argument("--model", help="Path to GGUF model file")
    parser.add_argument("--chat-template", help="Path to chat template file")
    parser.add_argument("--results-dir", help="Path to results directory")
    parser.add_argument("--port", type=int, help="Server port")
    parser.add_argument("--config", help="Path to JSON config file")
    parser.add_argument("--dense", action="store_true", help="Dense model (skip MoE phases)")
    args, _ = parser.parse_known_args()

    import copy
    config = copy.deepcopy(_DEFAULTS)

    # Layer 1: config.json file (deep merge for nested dicts)
    config_path = args.config or os.path.join(_DEFAULTS["results_dir"], "optimizer-config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            file_config = json.load(f)
        for k, v in file_config.items():
            if isinstance(v, dict) and isinstance(config.get(k), dict):
                config[k].update(v)
            else:
                config[k] = v

    # Layer 2: CLI args override everything
    if args.server:
        config["server"] = args.server
    if args.ik_server:
        config["ik_server"] = args.ik_server
    if args.model:
        config["model"] = args.model
    if args.chat_template:
        config["chat_template"] = args.chat_template
    if args.results_dir:
        config["results_dir"] = args.results_dir
    if args.port:
        config["port"] = args.port
    if args.dense:
        config["architecture"]["type"] = "dense"

    # Layer 3: Auto-detect hardware if not explicitly set
    hw = config["hardware"]
    if hw["max_threads"] is None:
        hw["max_threads"] = os.cpu_count() or 16
    if hw["moe_sweep_max"] is None:
        hw["moe_sweep_max"] = min(hw["max_threads"] * 2, 40)
    if hw["moe_sweep_center"] is None:
        hw["moe_sweep_center"] = hw["moe_sweep_max"] // 2
    if hw["max_gpu_layers"] is None:
        hw["max_gpu_layers"] = _detect_model_layers(config.get("model", ""))
    if hw["max_gpu_layers"] is None:
        hw["max_gpu_layers"] = 99  # fallback

    return config


# Global state -- set by reinitialize() before each model run.
# Defaults here are placeholders; they are always overwritten before use.
LLAMA_SERVER = Path("llama-server")
IK_SERVER = Path("")          # empty Path = IK not available
IK_MODE = False               # True when IK_SERVER is valid (pure IK or dual mode)
DUAL_SERVER_MODE = False      # True when BOTH llama-server AND ik_llama-server are available
MODEL = Path("model.gguf")
CHAT_TEMPLATE = Path("")
RESULTS_DIR = Path("results")
LOOKUP_CACHE_FILE = str(RESULTS_DIR / "lookup-cache.bin")
OPTUNA_DB = f"sqlite:///{RESULTS_DIR / 'optuna.db'}"
PORT = 8090
SERVER_URL = f"http://127.0.0.1:{PORT}"
http = requests.Session()

ARCH = {"type": "dense"}
IS_MOE = False
EXPERT_OVERRIDE_KEY = ""
DEFAULT_EXPERTS = 8
MAX_EXPERTS = 16

MAX_THREADS = os.cpu_count() or 16
MOE_SWEEP_MAX = min(MAX_THREADS * 2, 40)
MOE_SWEEP_CENTER = MOE_SWEEP_MAX // 2
MAX_GPU_LAYERS = 99
DEFAULT_GPU_LAYERS = 99

_config = {}
_NO_JINJA = False   # injected by reinitialize(); adds --no-jinja to every server cmd


def reinitialize(model_path, llama_server_path, results_dir, port=8090,
                 arch=None, max_gpu_layers=None, max_threads=None,
                 no_jinja=False, chat_template=None, ik_server_path=None):
    global LLAMA_SERVER, IK_SERVER, IK_MODE, DUAL_SERVER_MODE
    global MODEL, CHAT_TEMPLATE, RESULTS_DIR, LOOKUP_CACHE_FILE
    global OPTUNA_DB, PORT, SERVER_URL, http, _config
    global ARCH, IS_MOE, EXPERT_OVERRIDE_KEY, DEFAULT_EXPERTS, MAX_EXPERTS
    global MAX_THREADS, MOE_SWEEP_MAX, MOE_SWEEP_CENTER
    global MAX_GPU_LAYERS, DEFAULT_GPU_LAYERS, NAKED_ENGINE
    global _quality_baseline, _NO_JINJA

    MODEL = Path(model_path)
    LLAMA_SERVER = Path(llama_server_path)

    # IK server setup: determine mode
    _ik_path = ik_server_path or os.environ.get("IK_LLAMA_SERVER", "")
    IK_SERVER = Path(_ik_path) if (_ik_path and Path(_ik_path).is_file()) else Path("")
    _llama_valid = LLAMA_SERVER.is_file()
    IK_MODE = bool(IK_SERVER and IK_SERVER.is_file())
    DUAL_SERVER_MODE = IK_MODE and _llama_valid
    # If only IK is available, point LLAMA_SERVER at it so all existing code works unmodified
    if not _llama_valid and IK_MODE:
        LLAMA_SERVER = IK_SERVER
    RESULTS_DIR = Path(results_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _lc_path = (RESULTS_DIR / "lookup-cache.bin").resolve()
    try:
        _lc_path.parent.mkdir(parents=True, exist_ok=True)
        _lc_path.touch(exist_ok=True)  # pre-create so llama-server can open it
    except Exception:
        pass
    LOOKUP_CACHE_FILE = str(_lc_path)
    OPTUNA_DB = f"sqlite:///{RESULTS_DIR / 'optuna.db'}"
    PORT = port
    SERVER_URL = f"http://127.0.0.1:{PORT}"
    http = requests.Session()
    CHAT_TEMPLATE = Path(chat_template) if chat_template else Path("")

    if arch is None:
        arch = {"type": "dense"}
    ARCH = arch
    IS_MOE = ARCH.get("type") == "moe"
    EXPERT_OVERRIDE_KEY = ARCH.get("expert_override_key", "")
    DEFAULT_EXPERTS = ARCH.get("default_experts", 8)
    MAX_EXPERTS = ARCH.get("max_experts", 16)

    if max_threads is None:
        max_threads = os.cpu_count() or 16
    MAX_THREADS = max_threads
    MOE_SWEEP_MAX = min(MAX_THREADS * 2, 40)
    MOE_SWEEP_CENTER = MOE_SWEEP_MAX // 2

    detected = _detect_model_layers(str(MODEL))
    MAX_GPU_LAYERS = max_gpu_layers or detected or 99
    DEFAULT_GPU_LAYERS = MAX_GPU_LAYERS
    NAKED_ENGINE = {"context": 4096, "mlock": True, "n_gpu_layers": DEFAULT_GPU_LAYERS}

    _quality_baseline = None
    _NO_JINJA = no_jinja

    _config = {
        "server": str(LLAMA_SERVER), "model": str(MODEL),
        "ik_server": str(IK_SERVER) if IK_MODE else "",
        "ik_mode": IK_MODE, "dual_server_mode": DUAL_SERVER_MODE,
        "results_dir": str(RESULTS_DIR), "port": PORT,
        "architecture": ARCH,
        "hardware": {"max_threads": MAX_THREADS, "max_gpu_layers": MAX_GPU_LAYERS},
    }


def _bootstrap_from_config():
    cfg = _load_config()
    hw = cfg["hardware"]
    reinitialize(
        model_path=cfg["model"],
        llama_server_path=cfg["server"],
        results_dir=cfg["results_dir"],
        port=cfg["port"],
        arch=cfg["architecture"],
        max_gpu_layers=hw.get("max_gpu_layers"),
        max_threads=hw.get("max_threads"),
        chat_template=cfg.get("chat_template", ""),
    )


# Fixed test prompt for TPS measurement
TPS_TEST_PROMPT = "Write a Python function that implements binary search on a sorted list. Include docstring and type hints."

# Quality eval tasks (prompt, expected_answer_contains)
QUALITY_TASKS = [
    ("What is 127 * 43?", "5461"),
    ("Write a Python function to check if a string is a palindrome.", "def"),
    ("Explain what a hash table is in 2 sentences.", "key"),
    ("What are the first 8 prime numbers?", "19"),
    ("If I have 3 red balls and 5 blue balls, what is the probability of picking a red ball?", "3/8"),
]

# Legacy score weight constants -- kept for backward compatibility.
# compute_score() now uses the full VRAM+large-prompt formula directly.
SCORE_WEIGHT_GEN_TPS = 0.75
SCORE_WEIGHT_PROMPT_TPS = 0.15
SCORE_WEIGHT_TTFT = 0.10
SCORE_TTFT_BASELINE = 500   # ms — reference TTFT for normalization
SCORE_PP_BASELINE = 300     # t/s — reference prompt processing speed for normalization

# Large-prompt benchmark constants (ported from Checkpoint A)
LARGE_PROMPT_FILL_RATIO = 0.90
LARGE_PROMPT_OUTPUT_TOKENS = 200
LARGE_PROMPT_OVERHEAD = 95
LARGE_PROMPT_TOKENS_PER_SENT = 37

# VRAM efficiency: total MB cached at server start, used in compute_score()
_vram_total_mb: float = 0.0

# Quality gate: token-level uncertainty measurement
# Instead of average PPL (which gets diluted by high-confidence filler tokens),
# we measure two things:
#   1. Uncertain token count: tokens with logprob < -2.0 (~13% confidence)
#   2. Tail-20% logprob average: average of the worst 20% of logprobs
# Both metrics are sensitive to quality degradation because they focus on where
# the model actually struggles, not the easy tokens it gets right regardless.
QUALITY_GATE_PROMPTS = [
    # GPQA Diamond — graduate-level physics (energy-time uncertainty principle)
    """Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?

(A) 10^-8 eV
(B) 10^-9 eV
(C) 10^-4 eV
(D) 10^-11 eV

Explain your reasoning step by step before giving your final answer.""",
    # GPQA Diamond — graduate-level biology (mitochondrial genetics)
    """Mitochondria are semi-autonomous cellular organelles in charge of energy production. They encode for a part of their own translational machinery and respiratory complexes. Mitochondrial function is governed by over a thousand proteins imported from the cell, contributing to processes like the transport of proteins, ribosome biogenesis and translation regulation, respiratory oxidation, metabolism, and apoptotic signaling cascade. Mutations in the code for mitochondrial protein networks can cause numerous diseases in humans that are inherited through generations. Mutations of which of the mitochondrial proteins listed below are least likely to be genetically transmitted from a father to his children?

(A) Translocase of inner mitochondrial membrane 17B
(B) ATP binding cassette subfamily B member 8
(C) NADH dehydrogenase 2
(D) Tu translation elongation factor, mitochondrial

Explain your reasoning step by step before giving your final answer.""",
]
QUALITY_GATE_SEED = 42
QUALITY_GATE_N_PREDICT = 1024
QUALITY_GATE_UNCERTAIN_THRESHOLD = -0.5  # logprob < this = uncertain token (~60% confidence)
QUALITY_GATE_TAIL_PCT = 0.20             # average the worst 20% of logprobs
# Quality gate thresholds (based on % increase in uncertain tokens or tail degradation)
QUALITY_GATE_CEILING = 0.015    # 1.5% more uncertain tokens = still acceptable
QUALITY_GATE_SOFT_PENALTY = 0.85
QUALITY_GATE_CLIFF = 0.03       # 3% more uncertain tokens = hard disqualify
QUALITY_GATE_CLIFF_PENALTY = 0.1
# Baseline metrics (set during first Expert Count sweep with default experts)
_quality_baseline = None  # dict: {"uncertain_count", "tail_avg", "total_tokens"}


def _init_vram_info():
    """Cache total VRAM for scoring. Called once after warmup_server()."""
    global _vram_total_mb
    try:
        import pynvml
        pynvml.nvmlInit()
        total = 0
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            total += pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024 * 1024)
        _vram_total_mb = total
    except Exception:
        _vram_total_mb = 0.0


def _get_vram_used_mb() -> float:
    """Return MB of VRAM currently in use across all GPUs."""
    try:
        import pynvml
        pynvml.nvmlInit()
        total = 0
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            total += pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 * 1024)
        return total
    except Exception:
        return 0.0


def _get_actual_ctx() -> int:
    """Query the server actual context window size.
    Returns real n_ctx (may differ from requested if model caps it).
    """
    try:
        r = http.get(f"{SERVER_URL}/props", timeout=3)
        if r.status_code == 200:
            n_ctx = r.json().get("n_ctx", 0)
            if n_ctx > 0:
                return n_ctx
    except Exception:
        pass
    try:
        import re
        oversized = "word " * 65000
        r = http.post(f"{SERVER_URL}/completion",
                      json={"prompt": oversized, "n_predict": 1}, timeout=10)
        if r.status_code == 400:
            m = re.search(r"n_ctx[\s=:]+([0-9]+)", r.text)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return 4096


def _build_large_prompt(max_ctx: int) -> str:
    """Build a prompt filling ~90% of max_ctx tokens for large-prompt benchmarking."""
    input_budget = int(max_ctx * LARGE_PROMPT_FILL_RATIO) - LARGE_PROMPT_OUTPUT_TOKENS - LARGE_PROMPT_OVERHEAD
    n_sentences = max(5, min(input_budget // LARGE_PROMPT_TOKENS_PER_SENT, 200))
    sentences = [
        "Implement a thread-safe LRU cache in Python with O(1) get and put operations.",
        "Explain the difference between process and thread scheduling in operating systems.",
        "Write a function that serializes a binary tree to a string and deserializes it.",
        "Describe the CAP theorem and its implications for distributed database design.",
        "Implement Dijkstras shortest path algorithm using a priority queue.",
        "Explain how garbage collection works in generational collectors like CPython.",
        "Write SQL to find employees whose salary is above the department average.",
        "Describe the SOLID principles and give a concrete example of each.",
    ]
    parts = []
    while len(parts) < n_sentences:
        parts.extend(sentences)
    body = " ".join(parts[:n_sentences])
    return (f"<|im_start|>user\n{body}\n\nAnswer each question above concisely."
            f"<|im_end|>\n<|im_start|>assistant\n")


# Naked engine config — bare minimum to boot the server
# context 4096: test prompts only use ~130 tokens, no need to allocate 131K KV cache
# mlock pins model weights in RAM so OS doesn't evict them between restarts
NAKED_ENGINE = {
    "context": 4096,
    "mlock": True,
    "n_gpu_layers": DEFAULT_GPU_LAYERS,
}


# ============================================================
# Helpers
# ============================================================

class GPSampler(optuna.samplers.BaseSampler):
    """Gaussian Process sampler for Optuna using Expected Improvement.

    Fits a GP to all completed trials, then proposes the next trial by maximizing
    Expected Improvement (EI). Handles mixed parameter types by encoding everything
    to [0,1]. Falls back to random sampling for the first n_startup_trials.
    """

    def __init__(self, seed=42, n_startup_trials=10, n_candidates=2000):
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._n_startup = n_startup_trials
        self._n_candidates = n_candidates
        self._random_sampler = optuna.samplers.RandomSampler(seed=seed)

    def infer_relative_search_space(self, study, trial):
        return optuna.search_space.intersection_search_space(study.trials)

    def sample_relative(self, study, trial, search_space):
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and t.value > 0]

        if len(completed) < self._n_startup or not search_space:
            return {}  # fall back to sample_independent (random)

        # Encode completed trials into X matrix and y vector
        param_names = sorted(search_space.keys())
        X, y = [], []
        for t in completed:
            row = []
            for name in param_names:
                if name not in t.params:
                    break
                row.append(self._encode_param(t.params[name], search_space[name]))
            else:
                X.append(row)
                y.append(t.value)

        if len(X) < self._n_startup:
            return {}

        X = np.array(X)
        y = np.array(y)

        # Fit GP
        import warnings
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(X.shape[1])) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=self._seed, normalize_y=True)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X, y)
        except Exception:
            return {}  # GP fit failed, fall back to random

        # Generate random candidates and pick the one with highest EI
        candidates = self._rng.uniform(0, 1, size=(self._n_candidates, len(param_names)))
        mu, sigma = gp.predict(candidates, return_std=True)

        best_y = np.max(y)
        ei = self._expected_improvement(mu, sigma, best_y)

        best_idx = np.argmax(ei)
        best_candidate = candidates[best_idx]

        # Decode back to parameter values
        params = {}
        for i, name in enumerate(param_names):
            params[name] = self._decode_param(best_candidate[i], search_space[name])
        return params

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self._random_sampler.sample_independent(study, trial, param_name, param_distribution)

    @staticmethod
    def _expected_improvement(mu, sigma, best_y, xi=0.01):
        """Compute Expected Improvement. xi is exploration-exploitation tradeoff."""
        with np.errstate(divide='ignore', invalid='ignore'):
            imp = mu - best_y - xi
            Z = np.where(sigma > 1e-8, imp / sigma, 0.0)
            ei = np.where(sigma > 1e-8, imp * norm.cdf(Z) + sigma * norm.pdf(Z), 0.0)
        return ei

    @staticmethod
    def _encode_param(value, distribution):
        """Encode a parameter value to [0, 1] range."""
        if isinstance(distribution, optuna.distributions.CategoricalDistribution):
            choices = distribution.choices
            idx = choices.index(value) if value in choices else 0
            return idx / max(1, len(choices) - 1)
        elif isinstance(distribution, optuna.distributions.IntDistribution):
            low, high = distribution.low, distribution.high
            return (value - low) / max(1, high - low)
        elif isinstance(distribution, optuna.distributions.FloatDistribution):
            low, high = distribution.low, distribution.high
            return (value - low) / max(1e-8, high - low)
        return 0.5

    @staticmethod
    def _decode_param(encoded, distribution):
        """Decode a [0, 1] value back to the parameter's original type/range."""
        if isinstance(distribution, optuna.distributions.CategoricalDistribution):
            choices = distribution.choices
            idx = int(round(encoded * (len(choices) - 1)))
            idx = max(0, min(idx, len(choices) - 1))
            return choices[idx]
        elif isinstance(distribution, optuna.distributions.IntDistribution):
            low, high = distribution.low, distribution.high
            step = distribution.step
            raw = low + encoded * (high - low)
            # Snap to step grid
            return int(round((raw - low) / step) * step + low)
        elif isinstance(distribution, optuna.distributions.FloatDistribution):
            low, high = distribution.low, distribution.high
            return low + encoded * (high - low)
        return encoded


class GPStoppingCallback:
    """Stops optimization when the GP's maximum Expected Improvement drops below a threshold.

    This means the GP is confident that no untested configuration is likely to beat
    the current best — a mathematically principled replacement for patience-based stopping.
    """

    def __init__(self, ei_threshold=0.5, n_candidates=2000, patience_fallback=20, min_trials=15, min_trials_before_stop=30, seed=42, baseline_score=None):
        self._ei_threshold = ei_threshold
        self._n_candidates = n_candidates
        self._patience_fallback = patience_fallback
        self._min_trials = min_trials
        self._min_trials_before_stop = min_trials_before_stop  # must run at least this many before GP can stop
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._trials_without_improvement = 0
        self._best_value = None
        self._baseline_score = baseline_score  # don't stop early if nothing has beaten baseline

    def __call__(self, study, trial):
        # Track patience as fallback
        _bv = _safe_best_value(study)
        if _bv is None or _bv == self._best_value:
            self._trials_without_improvement += 1
        else:
            self._best_value = _bv
            self._trials_without_improvement = 0

        # Hard fallback: stop after patience_fallback trials without improvement
        # But never stop if we haven't beaten baseline — keep searching
        best_so_far = _safe_best_value(study)
        below_baseline = (self._baseline_score is not None and best_so_far is not None
                          and best_so_far < self._baseline_score)
        n_completed = len([t for t in study.trials
                          if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and t.value > 0])
        too_early = n_completed < self._min_trials_before_stop

        if self._trials_without_improvement >= self._patience_fallback:
            if below_baseline or too_early:
                pass  # keep going
            else:
                print(f"\n  [!] GP stopping (fallback): no improvement in {self._patience_fallback} trials.")
                study.stop()
                return

        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and t.value > 0]

        if len(completed) < self._min_trials:
            return  # not enough data to fit GP

        search_space = optuna.search_space.intersection_search_space(study.trials)
        if not search_space:
            return

        param_names = sorted(search_space.keys())
        X, y = [], []
        for t in completed:
            row = []
            for name in param_names:
                if name not in t.params:
                    break
                row.append(GPSampler._encode_param(t.params[name], search_space[name]))
            else:
                X.append(row)
                y.append(t.value)

        if len(X) < self._min_trials:
            return

        X = np.array(X)
        y = np.array(y)

        import warnings
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(X.shape[1])) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=self._seed, normalize_y=True)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X, y)
        except Exception:
            return  # GP fit failed, continue with trials

        candidates = self._rng.uniform(0, 1, size=(self._n_candidates, len(param_names)))
        mu, sigma = gp.predict(candidates, return_std=True)

        best_y = np.max(y)
        ei = GPSampler._expected_improvement(mu, sigma, best_y)
        max_ei = np.max(ei)

        # Scale threshold relative to best score (percentage-based)
        scaled_threshold = self._ei_threshold * best_y / 100

        if max_ei < scaled_threshold:
            if below_baseline or too_early:
                # Don't stop — either below baseline or haven't run enough trials
                pass
            else:
                print(f"\n  [!] GP stopping: max EI={max_ei:.2f} < threshold={scaled_threshold:.2f} "
                      f"(confident no untested config beats {best_y:.1f})")
                study.stop()


class EarlyStoppingCallback:
    """Stops Optuna optimization early if the score hasn't improved after N trials."""
    def __init__(self, patience=15):
        self.patience = patience
        self.best_score = None
        self.trials_without_improvement = 0

    def __call__(self, study, trial):
        _bv = _safe_best_value(study)
        if _bv is None or _bv == self.best_score:
            self.trials_without_improvement += 1
        else:
            self.best_score = _bv
            self.trials_without_improvement = 0
        if self.trials_without_improvement >= self.patience:
            print(f"\n  [!] Early stopping: no improvement in {self.patience} trials.")
            study.stop()


def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def check_duplicate_trial(trial):
    """Check if this exact param combo was already tested. Returns cached score or None."""
    for past in trial.study.trials:
        if past.state == optuna.trial.TrialState.COMPLETE and past.params == trial.params:
            for k, v in past.user_attrs.items():
                trial.set_user_attr(k, v)
            return past.value
    return None


def wait_for_server(proc=None, timeout=300):
    """Wait for llama-server /health to return ok."""
    start = time.time()
    while True:
        if proc is not None and proc.poll() is not None:
            return False
        if time.time() - start > timeout:
            return False

        try:
            r = http.get(f"{SERVER_URL}/health", timeout=0.5)
            if r.status_code == 200 and r.json().get("status") == "ok":
                warmup_server()
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(0.1)  # micro-poll: 100ms instead of 1s


def warmup_server():
    """Send throwaway requests to warm pipelines and prime speculation cache.

    Two requests:
      1. Short prompt + 5 tokens — warms matrix math pipelines
      2. Same test prompt + 30 tokens — primes the ngram speculation cache
         so the first real measurement isn't penalized by an empty cache
    """
    try:
        # Warm pipelines
        http.post(f"{SERVER_URL}/completion", json={
            "prompt": "Write a Python function that implements binary search on a sorted list and returns the index.",
            "n_predict": 5,
            "temperature": 0.0,
            "cache_prompt": False,
        }, timeout=10)
    except:
        pass
    try:
        # Prime speculation cache with same prompt used for measurement
        http.post(f"{SERVER_URL}/completion", json={
            "prompt": f"<|im_start|>user\n{TPS_TEST_PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "n_predict": 30,
            "temperature": 0.4,
            "cache_prompt": False,
        }, timeout=15)
    except:
        pass

    # Cache total VRAM once per server start for VRAM-aware scoring
    _init_vram_info()


def kill_server():
    """Kill any running llama-server process and verify port is free."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    else:
        subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    # Wait for port to actually free up (zombie processes can hold it)
    import socket
    for _ in range(20):  # up to 4 seconds
        time.sleep(0.2)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", PORT)) != 0:
                return  # port is free
    # If still busy after 4s, one more hard kill attempt
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    else:
        subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    time.sleep(2)


def compute_score(perf):
    """Composite score: gen TPS, large-prompt TPS, prompt TPS, TTFT, VRAM efficiency.

    With large-prompt data:  gen*0.35 + long*0.25 + pp*0.15 + ttft*0.15 + vram*0.10
    Without large-prompt:    gen*0.60 + pp*0.25 + ttft*0.15  (quick filter pass)
    """
    gen_tps = perf["tps"]
    prompt_tps = perf["prompt_tps"]
    ttft = perf["ttft"]
    gen_tps_long = perf.get("tps_long", 0.0)
    ttft_long = perf.get("ttft_long", 0.0)
    vram_mb = perf.get("vram_mb", 0.0)

    if gen_tps <= 0:
        return 0.0

    gen_scale = max(gen_tps, gen_tps_long, 1.0)
    pp_factor = (prompt_tps / SCORE_PP_BASELINE) * gen_scale if prompt_tps > 0 else 0.0
    ttft_factor = min(100.0 / ttft, 1.5) * gen_scale if ttft > 0 else gen_scale
    ttft_long_factor = min(500.0 / ttft_long, 1.5) * gen_scale if ttft_long > 0 else gen_scale

    if _vram_total_mb > 0 and vram_mb > 0:
        vram_factor = (1.5 - min(vram_mb / _vram_total_mb, 1.0)) * gen_scale
    else:
        vram_factor = gen_scale

    if gen_tps_long > 0:
        return (gen_tps        * 0.35 +
                gen_tps_long   * 0.25 +
                pp_factor      * 0.15 +
                ttft_factor    * 0.15 +
                vram_factor    * 0.10)
    else:
        return (gen_tps      * 0.60 +
                pp_factor    * 0.25 +
                ttft_factor  * 0.15)


ADAPTIVE_THRESHOLD = 0.70  # score must be >= 70% of best to warrant full measurement


def measure_perf_adaptive(best_score, n_predict=50, spec_params=None):
    """Adaptive measurement with large-prompt and VRAM scoring.

    Pass 1 (all configs): 1 quick small-prompt run.
      - If score < 80% of best: return immediately (filter bad configs fast).
    Pass 2 (competitive only): 2 more runs (median of 3) + large-prompt + VRAM.
    """
    first = _measure_perf_once(n_predict=n_predict, spec_params=spec_params)
    if first is None:
        return {"tps": 0.0, "ttft": 0.0, "prompt_tps": 0.0, "total_ms": 0.0}, False

    quick_score = compute_score(first)

    if best_score > 0 and quick_score < best_score * ADAPTIVE_THRESHOLD:
        return first, False

    # Competitive -- 2 more small-prompt runs, take median
    samples = [first]
    for _ in range(2):
        s = _measure_perf_once(n_predict=n_predict, spec_params=spec_params)
        if s:
            samples.append(s)
    perf = _aggregate_samples(samples)

    # Large-prompt benchmark: skip if quick score is already >30% below best
    quick_score_promoted = compute_score(perf)
    speed_tanked = best_score > 0 and quick_score_promoted < best_score * 0.70
    if not speed_tanked:
        large = _measure_perf_large()
        perf.update(large)

    # VRAM snapshot
    vram_used = _get_vram_used_mb()
    if vram_used > 0:
        perf["vram_mb"] = vram_used

    return perf, True


def is_server_running():
    """Check if llama-server is responding."""
    try:
        r = http.get(f"{SERVER_URL}/health", timeout=3)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except:
        return False


def _measure_perf_once(n_predict=50, spec_params=None):
    """Single measurement run. Returns dict of raw values or None on failure."""
    payload = {
        "prompt": f"<|im_start|>user\n{TPS_TEST_PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "n_predict": n_predict,
        "temperature": 0.4,
        "cache_prompt": False,
    }
    if spec_params:
        payload["speculative"] = spec_params
    try:
        r = http.post(f"{SERVER_URL}/completion", json=payload, timeout=60)
        if r.status_code == 200:
            data = r.json()
            timings = data.get("timings", {})
            # Prefer computing TPS from raw counts/ms rather than trusting
            # predicted_per_second, which llama.cpp can report as ~1e6 for
            # SSM/Mamba models and vision-text components where predicted_ms
            # is near zero (recurrent pass with no per-token timing).
            predicted_n  = timings.get("predicted_n",  0)
            predicted_ms = timings.get("predicted_ms", 0)
            if predicted_n > 0 and predicted_ms > 0:
                tps = (predicted_n / predicted_ms) * 1000.0
            else:
                tps = timings.get("predicted_per_second", 0)
            if tps > 0:
                return {
                    "tps": tps,
                    "ttft": timings.get("prompt_ms", 0),
                    "prompt_tps": timings.get("prompt_per_second", 0),
                    "total_ms": timings.get("prompt_ms", 0) + timings.get("predicted_ms", 0),
                }
    except Exception as e:
        print(f"  [!] Request failed: {e}")
    return None


def _measure_perf_large() -> dict:
    """Large-prompt benchmark for promoted configs. Returns tps_long/ttft_long or {}."""
    try:
        actual_ctx = _get_actual_ctx()
        prompt = _build_large_prompt(actual_ctx)
        r = http.post(f"{SERVER_URL}/completion",
                      json={"prompt": prompt, "n_predict": LARGE_PROMPT_OUTPUT_TOKENS,
                            "temperature": 0.0, "cache_prompt": False},
                      timeout=300)
        if r.status_code == 200:
            t = r.json().get("timings", {})
            predicted_n  = t.get("predicted_n",  0)
            predicted_ms = t.get("predicted_ms", 0)
            if predicted_n > 0 and predicted_ms > 0:
                tps_long = (predicted_n / predicted_ms) * 1000.0
            else:
                tps_long = t.get("predicted_per_second", 0)
            if tps_long > 0:
                return {"tps_long": tps_long, "ttft_long": t.get("prompt_ms", 0)}
    except Exception as e:
        print(f"  [!] Large-prompt failed: {e}")
    return {}


def _aggregate_samples(samples):
    """Given a list of raw measurement dicts, return median-aggregated result."""
    if not samples:
        return {"tps": 0.0, "ttft": 0.0, "prompt_tps": 0.0, "total_ms": 0.0}
    if len(samples) == 1:
        return samples[0]
    # Use median for odd counts, average for even
    result = {}
    for key in ["tps", "ttft", "prompt_tps", "total_ms"]:
        vals = sorted(s[key] for s in samples)
        mid = len(vals) // 2
        if len(vals) % 2 == 1:
            result[key] = vals[mid]
        else:
            result[key] = (vals[mid - 1] + vals[mid]) / 2
    return result


def measure_perf(n_predict=50, spec_params=None, runs=3):
    """Send test prompt and return performance metrics (median of N runs)."""
    samples = []
    for _ in range(runs):
        s = _measure_perf_once(n_predict=n_predict, spec_params=spec_params)
        if s:
            samples.append(s)
    return _aggregate_samples(samples)


def measure_token_uncertainty():
    """Measure token-level uncertainty on quality gate prompts.

    Returns dict with:
      - uncertain_count: number of tokens with logprob < threshold (-2.0)
      - tail_avg: average logprob of the worst 20% of tokens
      - total_tokens: total tokens measured
    Or None on failure.
    """
    all_logprobs = []
    for prompt_text in QUALITY_GATE_PROMPTS:
        payload = {
            "prompt": f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n<think>\n",
            "n_predict": QUALITY_GATE_N_PREDICT,
            "temperature": 0.0,
            "seed": QUALITY_GATE_SEED,
            "cache_prompt": False,
            "n_probs": 1,
        }
        try:
            r = http.post(f"{SERVER_URL}/completion", json=payload, timeout=120)
            if r.status_code != 200:
                print(f"  [!] Quality request returned status {r.status_code}")
            else:
                data = r.json()
                probs = data.get("completion_probabilities", [])
                if not probs:
                    print(f"  [!] No completion_probabilities in response. Keys: {list(data.keys())}")
                for token_info in probs:
                    logprob = token_info.get("logprob")
                    if logprob is not None and logprob < 0:
                        all_logprobs.append(logprob)
        except Exception as e:
            print(f"  [!] Quality measurement failed: {e}")

    if not all_logprobs:
        return None

    # Count uncertain tokens (logprob < -2.0 = less than ~13% confidence)
    uncertain_count = sum(1 for lp in all_logprobs if lp < QUALITY_GATE_UNCERTAIN_THRESHOLD)

    # Tail-20% average: sort logprobs ascending, average the worst 20%
    sorted_lps = sorted(all_logprobs)
    tail_n = max(1, int(len(sorted_lps) * QUALITY_GATE_TAIL_PCT))
    tail_avg = sum(sorted_lps[:tail_n]) / tail_n

    return {
        "uncertain_count": uncertain_count,
        "tail_avg": tail_avg,
        "total_tokens": len(all_logprobs),
    }


def measure_quality_gate(is_baseline=False):
    """Quality gate using token-level uncertainty comparison against baseline.

    Measures two signals:
      1. Uncertain token count increase (tokens with logprob < -2.0)
      2. Tail-20% logprob degradation (worst 20% of tokens)
    Uses the worse of the two signals to determine the quality factor.

    On baseline run (is_baseline=True): measures and stores baseline metrics.
    On subsequent runs: returns quality_factor (0.1-1.0) based on degradation.
    """
    global _quality_baseline

    metrics = measure_token_uncertainty()
    if metrics is None:
        if is_baseline:
            return 1.0
        print(f"  [Q] Quality measurement failed/timed out — applying max penalty")
        return QUALITY_GATE_CLIFF_PENALTY

    if is_baseline or _quality_baseline is None:
        _quality_baseline = metrics
        print(f"  [Q] Baseline: {metrics['uncertain_count']} uncertain tokens "
              f"(of {metrics['total_tokens']}), tail-20% avg: {metrics['tail_avg']:.3f}")
        return 1.0

    # Signal 1: uncertain token count increase
    # When baseline has very few uncertain tokens, use a floor based on total token count
    # to avoid extreme sensitivity (e.g., going from 0→3 out of 1698 shouldn't be a cliff)
    base_uc = _quality_baseline["uncertain_count"]
    uc_floor = max(1, int(_quality_baseline["total_tokens"] * 0.01))  # 1% floor
    base_uc = max(base_uc, uc_floor)
    uc_increase = (metrics["uncertain_count"] - base_uc) / base_uc

    # Signal 2: tail-20% logprob degradation (more negative = worse)
    base_tail = _quality_baseline["tail_avg"]
    if base_tail < 0:
        tail_increase = (base_tail - metrics["tail_avg"]) / abs(base_tail)  # positive = degraded
    else:
        tail_increase = 0.0

    # Use the worse signal
    degradation = max(uc_increase, tail_increase)

    if degradation <= 0:
        quality_factor = 1.0
    elif degradation <= QUALITY_GATE_CEILING:
        # 0% to 15%: gentle slope from 1.0 → 0.85
        penalty_range = 1.0 - QUALITY_GATE_SOFT_PENALTY
        quality_factor = 1.0 - (degradation / QUALITY_GATE_CEILING) * penalty_range
    elif degradation <= QUALITY_GATE_CLIFF:
        # 15% to 30%: steep cliff from 0.85 → 0.1
        t = (degradation - QUALITY_GATE_CEILING) / (QUALITY_GATE_CLIFF - QUALITY_GATE_CEILING)
        quality_factor = QUALITY_GATE_SOFT_PENALTY - t * (QUALITY_GATE_SOFT_PENALTY - QUALITY_GATE_CLIFF_PENALTY)
    else:
        quality_factor = QUALITY_GATE_CLIFF_PENALTY

    print(f"  [Q] Uncertain: {metrics['uncertain_count']} (baseline: {_quality_baseline['uncertain_count']}, "
          f"+{uc_increase:+.0%}) | Tail: {metrics['tail_avg']:.3f} (baseline: {_quality_baseline['tail_avg']:.3f}, "
          f"+{tail_increase:+.0%}) | factor: {quality_factor:.2f}")
    return quality_factor


def measure_quality(sampling_params, tasks=QUALITY_TASKS):
    """Run quality eval tasks and return score (0-100)."""
    correct = 0
    total = len(tasks)

    for prompt, expected in tasks:
        payload = {
            "prompt": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            "n_predict": 512,
            "cache_prompt": False,
            **sampling_params,
        }
        try:
            r = http.post(f"{SERVER_URL}/completion", json=payload, timeout=30)
            if r.status_code == 200:
                content = r.json().get("content", "")
                if expected.lower() in content.lower():
                    correct += 1
        except:
            pass

    return (correct / total) * 100


def start_server(engine_config):
    """Start llama-server with given engine config."""
    env = os.environ.copy()
    if engine_config.get("cuda_graph_opt"):
        env["GGML_CUDA_GRAPH_OPT"] = "1"

    cmd = [
        str(LLAMA_SERVER),
        "-m", str(MODEL),
        "--port", str(PORT),
        "--host", "127.0.0.1",
        "-ngl", str(engine_config.get("n_gpu_layers", 99)),
        "-c", str(engine_config.get("context", 4096)),
        "--parallel", "1",
    ]
    if _NO_JINJA:
        cmd.extend(["--no-jinja", "--chat-template", "chatml"])
    elif CHAT_TEMPLATE and CHAT_TEMPLATE.is_file():
        cmd.extend(["--chat-template-file", str(CHAT_TEMPLATE)])
    else:
        cmd.append("--jinja")
    # Warmup control
    if engine_config.get("warmup") is False:
        cmd.append("--no-warmup")
    # Prompt caching control
    if engine_config.get("cache_prompt") is False:
        cmd.append("--no-cache-prompt")
    # Fit control — disable if we're manually tuning params
    if engine_config.get("fit") is False:
        cmd.append("--fit=off")

    # Only add flags that are explicitly in the config — truly naked otherwise
    if "batch_size" in engine_config:
        cmd.extend(["-b", str(engine_config["batch_size"])])
    if "ubatch_size" in engine_config:
        cmd.extend(["--ubatch-size", str(engine_config["ubatch_size"])])
    if "threads" in engine_config:
        cmd.extend(["-t", str(engine_config["threads"])])
    if "threads_batch" in engine_config:
        cmd.extend(["-tb", str(engine_config["threads_batch"])])
    if "n_cpu_moe" in engine_config:
        cmd.extend(["--n-cpu-moe", str(engine_config["n_cpu_moe"])])
    if "expert_used_count" in engine_config and EXPERT_OVERRIDE_KEY:
        # Only apply override if different from default — override-kv costs ~6 t/s even at default value
        if engine_config["expert_used_count"] != DEFAULT_EXPERTS:
            cmd.extend(["--override-kv", f"{EXPERT_OVERRIDE_KEY}=int:{engine_config['expert_used_count']}"])
    if "poll" in engine_config:
        cmd.extend(["--poll", str(engine_config["poll"])])
    if "poll_batch" in engine_config:
        cmd.extend(["--poll-batch", str(engine_config["poll_batch"])])
    if "prio" in engine_config:
        cmd.extend(["--prio", str(engine_config["prio"])])
    if "prio_batch" in engine_config:
        cmd.extend(["--prio-batch", str(engine_config["prio_batch"])])
    if "cache_type_k" in engine_config:
        cmd.extend(["--cache-type-k", engine_config["cache_type_k"]])
    if "cache_type_v" in engine_config:
        cmd.extend(["--cache-type-v", engine_config["cache_type_v"]])
    if "flash_attn" in engine_config:
        cmd.extend(["--flash-attn", engine_config["flash_attn"]])
    if "n_predict" in engine_config:
        cmd.extend(["--n-predict", str(engine_config["n_predict"])])
    if "temp" in engine_config:
        cmd.extend(["--temp", str(engine_config["temp"])])

    # Speculation params
    if "spec_type" in engine_config:
        cmd.extend(["--spec-type", engine_config["spec_type"]])
    if "spec_ngram_n" in engine_config:
        cmd.extend(["--spec-ngram-size-n", str(engine_config["spec_ngram_n"])])
    if "spec_ngram_m" in engine_config:
        cmd.extend(["--spec-ngram-size-m", str(engine_config["spec_ngram_m"])])
    if "spec_ngram_min_hits" in engine_config:
        cmd.extend(["--spec-ngram-min-hits", str(engine_config["spec_ngram_min_hits"])])
    if "draft_max" in engine_config:
        cmd.extend(["--draft", str(engine_config["draft_max"])])
    if "draft_min" in engine_config:
        cmd.extend(["--draft-min", str(engine_config["draft_min"])])
    if "draft_p_min" in engine_config:
        cmd.extend(["--draft-p-min", str(engine_config["draft_p_min"])])

    # CPU placement
    if "cpu_strict" in engine_config:
        cmd.extend(["--cpu-strict", str(engine_config["cpu_strict"])])
    if "cpu_strict_batch" in engine_config:
        cmd.extend(["--cpu-strict-batch", str(engine_config["cpu_strict_batch"])])

    # Boolean flags — only add if explicitly set
    if engine_config.get("swa_full"):
        cmd.append("--swa-full")
    if engine_config.get("repack") is False:
        cmd.append("--no-repack")
    if engine_config.get("op_offload") is False:
        cmd.append("--no-op-offload")
    if engine_config.get("kv_unified"):
        cmd.append("--kv-unified")
    if engine_config.get("mlock"):
        cmd.append("--mlock")
    if engine_config.get("no_mmap"):
        cmd.append("--no-mmap")
    if engine_config.get("kv_offload") is False:
        cmd.append("--no-kv-offload")
    if engine_config.get("no_host"):
        cmd.append("--no-host")
    if engine_config.get("direct_io"):
        cmd.append("--direct-io")
    if engine_config.get("cont_batching") is False:
        cmd.append("--no-cont-batching")
    if engine_config.get("backend_sampling"):
        cmd.append("--backend-sampling")
    if engine_config.get("context_shift") is False:
        cmd.append("--no-context-shift")

    if "ctx_checkpoints" in engine_config:
        cmd.extend(["--ctx-checkpoints", str(engine_config["ctx_checkpoints"])])
    if "checkpoint_every_n" in engine_config:
        cmd.extend(["--checkpoint-every-n-tokens", str(engine_config["checkpoint_every_n"])])
    if "cache_ram" in engine_config:
        cmd.extend(["--cache-ram", str(engine_config["cache_ram"])])
    if "cache_reuse" in engine_config:
        cmd.extend(["--cache-reuse", str(engine_config["cache_reuse"])])
    if "threads_http" in engine_config:
        cmd.extend(["--threads-http", str(engine_config["threads_http"])])
    if engine_config.get("lookup_cache_dynamic"):
        cmd.extend(["--lookup-cache-dynamic", str(engine_config["lookup_cache_dynamic"])])

    # MoE tensor placement
    if engine_config.get("cpu_moe"):
        cmd.append("--cpu-moe")
    if engine_config.get("override_tensor"):
        cmd.extend(["-ot", engine_config["override_tensor"]])

    # Multi-GPU split (from topology sweep overlay)
    if engine_config.get("tensor_split"):
        cmd.extend(["--tensor-split", str(engine_config["tensor_split"])])
    if engine_config.get("main_gpu") is not None:
        cmd.extend(["--main-gpu", str(engine_config["main_gpu"])])

    # ── IK_llama.cpp exclusive flags ─────────────────────────────────────────
    # Only added when this server binary is ik_llama-server. These flags are
    # silently ignored / invalid on vanilla llama.cpp so we guard them strictly.
    _is_ik = engine_config.get("_ik_server", False)
    if _is_ik:
        # MLA attention: 0=off 1=CPU-only 2=CPU+GPU (best for hybrid) 3=CPU-only v2
        mla = engine_config.get("ik_mla", 0)
        if mla:
            cmd.extend(["-mla", str(mla)])
        # Flash attention is required when MLA is active
        if mla and "flash_attn" not in engine_config:
            cmd.extend(["-fa", "1"])
        # Fused MoE: fuses expert routing kernels on GPU (+20-80% on MoE)
        if engine_config.get("ik_fused_moe"):
            cmd.extend(["-fmoe", "1"])
        # Run-time quant repack: repacks tensor layout for CPU SIMD (+50-80% on CPU)
        if engine_config.get("ik_rtr"):
            cmd.extend(["-rtr", "1"])
        # Attention max batch: MiB for K*Q compute buffer reuse (reduces pressure)
        if engine_config.get("ik_attn_max_batch"):
            cmd.extend(["-amb", str(engine_config["ik_attn_max_batch"])])
        # Smart Expert Reduction: drop N experts per layer to trade quality for speed
        # Format "N,1" — typically 5,1 or 6,1 or 7,1; lower N = better quality
        if engine_config.get("ik_ser"):
            cmd.extend(["-ser", str(engine_config["ik_ser"])])
        # NUMA policy override for CPU (ik build exposes --numa numactl etc.)
        if engine_config.get("ik_numa"):
            cmd.extend(["--numa", str(engine_config["ik_numa"])])

    # Debug: show key flags being passed (helps diagnose perf drops)
    # Skip exe path and model path (args 0-2)
    flags = cmd[3:]  # everything after -m <model>
    print(f"    [debug] cmd: {' '.join(str(f) for f in flags)}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    # Drain stderr in background thread to prevent pipe buffer blocking
    proc._stderr_lines = []
    def _drain_stderr():
        try:
            for line in proc.stderr:
                proc._stderr_lines.append(line.decode("utf-8", errors="replace").rstrip())
        except Exception:
            pass
    t = threading.Thread(target=_drain_stderr, daemon=True)
    t.start()

    return proc


def load_phase_results(phase_name):
    """Load saved results from a completed phase."""
    path = RESULTS_DIR / f"{phase_name}_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def start_ik_server(engine_config):
    """Start ik_llama-server with IK-specific flags merged into engine_config.

    This is a thin wrapper that forces _ik_server=True and routes to the
    IK_SERVER binary instead of LLAMA_SERVER.  All other behaviour (stderr
    drain, env setup, debug print) is identical to start_server().
    """
    ik_cfg = dict(engine_config)
    ik_cfg["_ik_server"] = True
    # Swap the binary: temporarily patch LLAMA_SERVER global for the Popen call
    global LLAMA_SERVER
    _orig = LLAMA_SERVER
    LLAMA_SERVER = IK_SERVER
    try:
        proc = start_server(ik_cfg)
    finally:
        LLAMA_SERVER = _orig
    return proc


def phase_ik_contrast(locked_compute=None, locked_moe=None, locked_memory=None):
    """IK_llama.cpp Contrast Phase: benchmark IK against the best llama.cpp config.

    Runs a structured comparison:
      Step 1 — llama.cpp baseline with best known config (3 runs, median)
      Step 2 — IK with same config (apple-to-apple baseline diff)
      Step 3 — IK with MLA+fused-MoE+RTR enabled  (IK feature pack)
      Step 4 — IK sweep: vary attn_max_batch (128/256/512/1024) to find best
      Step 5 — IK sweep: SER (smart expert reduction) 5,1 / 6,1 / 7,1 if MoE
      Step 6 — IK sweep: MLA mode (1/2/3) if MoE model (DeepSeek-like)

    Saves results to ik_contrast_results.json.
    Returns dict with keys: llama_tps, ik_tps, ik_best_tps, ik_gain_pct, best_ik_config.
    """
    if not IK_MODE:
        print("\n[*] IK_llama.cpp not configured — skipping IK contrast phase.")
        return None

    # Resume guard
    existing = load_phase_results("ik_contrast")
    if existing and "ik_best_tps" in existing:
        print(f"\n[*] IK contrast already complete — "
              f"IK best: {existing['ik_best_tps']:.1f} t/s  "
              f"gain vs llama: {existing.get('ik_gain_pct', 0):.1f}%  (from previous run)")
        return existing

    phase_start = time.time()
    label = "IK_llama.cpp Contrast"

    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)
    print(f"\n  llama-server : {LLAMA_SERVER}")
    print(f"  ik-server    : {IK_SERVER}")
    print(f"  Mode         : {'dual (contrast)' if DUAL_SERVER_MODE else 'IK-only'}")

    # ── Build base config from best known llama.cpp results ─────────────────
    base = dict(NAKED_ENGINE)
    if locked_moe:
        base.update(locked_moe)
    if locked_compute:
        base.update({k: v for k, v in locked_compute.items() if v is not None})
    if locked_memory:
        # Expand kv_cache_type → separate k/v keys if present
        mem = dict(locked_memory)
        if "kv_cache_type" in mem:
            kv = mem.pop("kv_cache_type")
            mem["cache_type_k"] = kv
            mem["cache_type_v"] = kv
        base.update({k: v for k, v in mem.items() if v is not None})
    base = {k: v for k, v in base.items() if v is not None}

    results = []

    def _bench_server(cfg, label_str, use_ik=False, runs=3):
        """Start server (llama or IK), benchmark, stop, return perf dict."""
        print(f"\n  Testing: {label_str} ({'IK' if use_ik else 'llama'})...")
        kill_server()
        proc = start_ik_server(cfg) if use_ik else start_server(cfg)
        if not wait_for_server(proc=proc):
            reason = ""
            try:
                lines = getattr(proc, "_stderr_lines", [])
                for line in reversed(lines):
                    ls = line.strip()
                    if ls and any(kw in ls.lower() for kw in ("error","failed","abort","oom","alloc","cuda","memory","unknown","invalid")):
                        reason = f" → {ls[:100]}"
                        break
            except Exception:
                pass
            print(f"    FAILED{reason}")
            proc.kill()
            return None
        perf = measure_perf(runs=runs)
        large = _measure_perf_large()
        if large:
            perf.update(large)
        vram = _get_vram_used_mb()
        if vram > 0:
            perf["vram_mb"] = vram
        score = compute_score(perf)
        tps = perf["tps"]
        tps_long = perf.get("tps_long", 0)
        server_tag = "IK" if use_ik else "llama"
        print(f"    {server_tag}: {tps:.1f} t/s  (long: {tps_long:.1f} t/s)  "
              f"pp: {perf['prompt_tps']:.0f} t/s  TTFT: {perf['ttft']:.0f}ms  score: {score:.1f}")
        results.append({
            "label": label_str,
            "server": "ik" if use_ik else "llama",
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
            "tps": round(tps, 2),
            "tps_long": round(tps_long, 2),
            "prompt_tps": round(perf["prompt_tps"], 2),
            "ttft": round(perf["ttft"], 1),
            "score": round(score, 2),
        })
        kill_server()
        return perf

    # ── Step 1: llama.cpp baseline ───────────────────────────────────────────
    llama_perf = None
    llama_tps = 0.0
    if DUAL_SERVER_MODE:
        llama_perf = _bench_server(base, "llama.cpp best config", use_ik=False)
        if llama_perf:
            llama_tps = llama_perf["tps"]

    # ── Step 2: IK same config (apple-to-apple) ──────────────────────────────
    ik_same_perf = _bench_server(base, "IK same config (no IK flags)", use_ik=True)
    ik_tps_same = ik_same_perf["tps"] if ik_same_perf else 0.0

    # ── Build IK feature-pack config ─────────────────────────────────────────
    ik_base = dict(base)
    # MLA: use mode 2 (CPU+GPU) for MoE models with DeepSeek-like MLA tensors,
    # mode 0 (off) for dense models where it has no effect / may cause errors.
    # Mode 2 is the safest default — falls back gracefully if no MLA tensors.
    mla_mode = 2 if IS_MOE else 0
    ik_base["ik_mla"] = mla_mode
    if mla_mode:
        ik_base["flash_attn"] = "on"
        ik_base["ik_attn_max_batch"] = 512   # safe default
    if IS_MOE:
        ik_base["ik_fused_moe"] = True
    # RTR: only safe when not using mmap (i.e. we have enough RAM).
    # We enable it — if server crashes, the benchmark marks it failed and we continue.
    ik_base["ik_rtr"] = True

    # ── Step 3: IK feature pack ──────────────────────────────────────────────
    ik_feature_perf = _bench_server(ik_base, "IK + MLA + fused-MoE + RTR", use_ik=True)
    ik_tps_feature = ik_feature_perf["tps"] if ik_feature_perf else 0.0

    # ── Step 4: attn_max_batch sweep (only if MLA active) ───────────────────
    best_ik_tps = ik_tps_feature
    best_ik_config = dict(ik_base)
    best_ik_label = "IK + MLA + fused-MoE + RTR"

    if mla_mode and ik_feature_perf:
        print(f"\n  [IK] Sweeping attn_max_batch...")
        for amb in [128, 256, 512, 1024]:
            cfg = dict(ik_base)
            cfg["ik_attn_max_batch"] = amb
            perf = _bench_server(cfg, f"IK amb={amb}", use_ik=True)
            if perf and perf["tps"] > best_ik_tps:
                best_ik_tps = perf["tps"]
                best_ik_config = dict(cfg)
                best_ik_label = f"IK amb={amb}"

    # ── Step 5: SER sweep (MoE only) ─────────────────────────────────────────
    if IS_MOE and best_ik_tps > 0:
        print(f"\n  [IK] Sweeping Smart Expert Reduction (SER)...")
        for ser_n in [7, 6, 5]:  # higher N = fewer experts = faster but lower quality
            cfg = dict(best_ik_config)
            cfg["ik_ser"] = f"{ser_n},1"
            perf = _bench_server(cfg, f"IK SER={ser_n},1", use_ik=True)
            if perf and perf["tps"] > best_ik_tps:
                best_ik_tps = perf["tps"]
                best_ik_config = dict(cfg)
                best_ik_label = f"IK SER={ser_n},1"
            # SER sacrifices quality — only keep if gain is meaningful (>10%)
            # and quality is not too degraded (guard against runaway speed)
            elif perf:
                gain = (perf["tps"] - ik_tps_feature) / max(ik_tps_feature, 1) * 100
                print(f"    SER={ser_n},1 gain vs feature-pack: {gain:+.1f}% — not faster, skipping")

    # ── Step 6: MLA mode sweep (MoE only) ────────────────────────────────────
    if IS_MOE and mla_mode:
        # Try mode 3 (CPU-only alt) in addition to mode 2
        print(f"\n  [IK] Testing MLA mode 3 (CPU-only v2)...")
        cfg3 = dict(best_ik_config)
        cfg3["ik_mla"] = 3
        perf3 = _bench_server(cfg3, "IK MLA=3 (CPU-only v2)", use_ik=True)
        if perf3 and perf3["tps"] > best_ik_tps:
            best_ik_tps = perf3["tps"]
            best_ik_config = dict(cfg3)
            best_ik_label = "IK MLA=3"

    # ── Summary ───────────────────────────────────────────────────────────────
    ik_gain_vs_llama = (
        (best_ik_tps - llama_tps) / llama_tps * 100
        if llama_tps > 0 else 0.0
    )
    ik_gain_vs_base = (
        (best_ik_tps - ik_tps_same) / ik_tps_same * 100
        if ik_tps_same > 0 else 0.0
    )

    elapsed = time.time() - phase_start
    print(f"\n{'=' * 60}")
    print(f"  {label} — RESULTS")
    print(f"{'=' * 60}")
    if DUAL_SERVER_MODE:
        print(f"  llama.cpp best:  {llama_tps:.1f} t/s")
    print(f"  IK same config:  {ik_tps_same:.1f} t/s  (no IK flags)")
    print(f"  IK feature pack: {ik_tps_feature:.1f} t/s  (MLA+fused-MoE+RTR)")
    print(f"  IK BEST:         {best_ik_tps:.1f} t/s  [{best_ik_label}]")
    if DUAL_SERVER_MODE:
        print(f"  IK gain vs llama: {ik_gain_vs_llama:+.1f}%")
    print(f"  IK gain vs no-IK-flags: {ik_gain_vs_base:+.1f}%")
    print(f"  Duration:        {elapsed / 60:.1f} min")

    output = {
        "phase": "ik_contrast",
        "ik_available": True,
        "dual_mode": DUAL_SERVER_MODE,
        "llama_tps": round(llama_tps, 2),
        "ik_tps_same_config": round(ik_tps_same, 2),
        "ik_tps_feature_pack": round(ik_tps_feature, 2),
        "ik_best_tps": round(best_ik_tps, 2),
        "ik_best_label": best_ik_label,
        "ik_gain_vs_llama_pct": round(ik_gain_vs_llama, 1),
        "ik_gain_vs_base_pct": round(ik_gain_vs_base, 1),
        "best_ik_config": {k: str(v) if isinstance(v, Path) else v
                           for k, v in best_ik_config.items()},
        "duration_seconds": round(elapsed, 1),
        "all_trials": results,
    }
    save_phase_results("ik_contrast", output)
    return output


def start_naked_server():
    """Kill existing server and start a naked one. Returns proc or None on failure."""
    print("\n[*] Starting naked server (no flags)...")
    kill_server()
    proc = start_server(NAKED_ENGINE)
    print("    Waiting for server to load...")
    if wait_for_server(proc=proc):
        print("    Naked server is ready.")
        return proc
    else:
        print("[!] Naked server failed to start. Check GPU/VRAM.")
        return None


def print_trial_result(trial_num, total_trials, tps, perf, params_short, best_score, final_score=None):
    """Print a formatted trial result line. Returns new best_score.

    Args:
        final_score: If provided, use this as the actual score (e.g. after quality gate).
                     Otherwise uses compute_score(perf).
    """
    score = final_score if final_score is not None else compute_score(perf)
    marker = ""
    if score > best_score:
        best_score = score
        marker = " *** NEW BEST ***"

    done = trial_num + 1
    pct = done / total_trials * 100
    bar_len = 20
    filled = int(bar_len * done / total_trials)
    bar = "█" * filled + "░" * (bar_len - filled)

    print(f"  [{bar}] {pct:5.1f}%  Trial {trial_num:3d}/{total_trials}: {tps:6.1f} t/s | "
          f"pp:{perf['prompt_tps']:5.0f} t/s | TTFT:{perf['ttft']:4.0f}ms | "
          f"score:{score:5.1f} | {params_short}{marker}")

    return best_score


def print_param_importance(study):
    """Print a ranked table of parameter importances using fANOVA."""
    # Filter out failed trials (score=0) so fANOVA has clean data
    completed = [t for t in study.trials if t.value is not None and t.value > 0]
    if len(completed) < 10:
        print(f"\n  (Only {len(completed)} successful trials — need 10+ for importance analysis)")
        return {}

    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as e:
        # Fallback to mean decrease impurity if fANOVA fails
        try:
            from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
            importances = optuna.importance.get_param_importances(
                study, evaluator=MeanDecreaseImpurityImportanceEvaluator()
            )
        except Exception:
            print(f"\n  (Could not compute parameter importance: {e})")
            return {}

    if not importances or len(importances) <= 1:
        return importances

    print(f"\n  Parameter Importance:")
    print(f"  {'Param':<28} {'Impact':>7}  {'':}")
    print(f"  {'─' * 28} {'─' * 7}  {'─' * 20}")

    max_bar = 20
    for param, importance in importances.items():
        pct = importance * 100
        bar_len = int(importance / max(importances.values()) * max_bar) if max(importances.values()) > 0 else 0
        bar = "█" * bar_len
        print(f"  {param:<28} {pct:6.1f}%  {bar}")

    return importances


def print_param_histogram(study, param_name, best_value=None):
    """Print a text histogram showing best score per parameter value.

    For single-parameter sweeps — shows how score varies across the parameter range.
    """
    # Collect best score per parameter value
    scores_by_val = {}
    for t in study.trials:
        if t.value is None or t.value <= 0:
            continue
        val = t.params.get(param_name)
        if val is None:
            continue
        if val not in scores_by_val or t.value > scores_by_val[val]:
            scores_by_val[val] = t.value

    if not scores_by_val:
        return

    # Sort by parameter value (left = lowest, right = highest)
    sorted_vals = sorted(scores_by_val.keys())
    max_score = max(scores_by_val.values())
    bar_max = 30  # max bar width

    print(f"\n  Score by {param_name}:")
    print(f"  {'Value':>6}  {'Score':>7}  {'':}")
    print(f"  {'─' * 6}  {'─' * 7}  {'─' * bar_max}")

    for val in sorted_vals:
        score = scores_by_val[val]
        bar_len = int(score / max_score * bar_max) if max_score > 0 else 0
        bar = "█" * bar_len
        marker = " ◄ best" if best_value is not None and val == best_value else ""
        print(f"  {val:>6}  {score:>7.1f}  {bar}{marker}")


def _safe_best_value(study):
    """Get study.best_value without throwing if no valid trials exist."""
    try:
        return study.best_value
    except ValueError:
        return None


def setup_study(study_name, n_trials, seed=42):
    """Create/resume an Optuna study. Returns (study, remaining_trials, completed)."""
    ensure_results_dir()
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=OPTUNA_DB,
        load_if_exists=True,
        sampler=GPSampler(seed=seed),
    )

    completed = len(study.trials)
    remaining = n_trials
    if completed > 0:
        print(f"\n[*] Resuming from trial {completed}/{n_trials} ({completed} completed)")
        remaining = max(0, n_trials - completed)
        if remaining == 0:
            print("    All trials already completed. Use more trials or reset DB.")

    return study, remaining, completed


def save_phase_results(phase_name, results):
    """Save phase results to JSON."""
    ensure_results_dir()
    with open(RESULTS_DIR / f"{phase_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR / f'{phase_name}_results.json'}")


# ============================================================
# MoE Thread Sweep (pure speed, no quality gate)
# Expert Count Sweep (with perplexity quality gate)
# ============================================================

def _server_start_failed(trial_num, params_short, proc):
    """Handle server start failure — extract reason from stderr and report."""
    reason = ""
    try:
        lines = getattr(proc, "_stderr_lines", [])
        for line in reversed(lines):
            line = line.strip()
            if line and ("error" in line.lower() or "failed" in line.lower() or "abort" in line.lower() or "oom" in line.lower() or "alloc" in line.lower() or "CUDA" in line or "memory" in line.lower() or "unknown" in line.lower() or "invalid" in line.lower()):
                reason = f" → {line[:120]}"
                break
        if not reason and lines:
            last_lines = [l.strip() for l in lines if l.strip()]
            if last_lines:
                reason = f" → {last_lines[-1][:120]}"
    except Exception:
        pass
    print(f"  Trial {trial_num}: FAILED | {params_short}{reason}")
    proc.kill()


def phase_gpu_offload():
    """GPU Offload: Find optimal GPU layer offload.

    For MoE models: skips entirely, locks to MAX_GPU_LAYERS (all on GPU).
    MoE models use n_cpu_moe for smart CPU offloading in the MoE phase instead.

    For dense models: sweeps n_gpu_layers using middle-out approach with
    adaptive measurement and per-direction early stopping.

    Updates NAKED_ENGINE and DEFAULT_GPU_LAYERS for all subsequent phases.
    Returns int: best n_gpu_layers value.
    """
    global DEFAULT_GPU_LAYERS, NAKED_ENGINE

    label = "GPU Offload"
    max_ngl = MAX_GPU_LAYERS

    # Check for existing results
    existing = load_phase_results("gpu")
    if existing and "best_ngl" in existing:
        best_ngl = existing["best_ngl"]
        print(f"\n[*] GPU Offload already complete — n_gpu_layers={best_ngl} (from previous run)")
        DEFAULT_GPU_LAYERS = best_ngl
        NAKED_ENGINE = {"context": 4096, "mlock": True, "n_gpu_layers": best_ngl}
        return best_ngl

    # MoE models: always full GPU offload — MoE phase handles smart CPU offloading
    if IS_MOE:
        print(f"\n[*] MoE model — all {max_ngl} layers on GPU (MoE phase handles CPU offloading)")
        DEFAULT_GPU_LAYERS = max_ngl
        NAKED_ENGINE = {"context": 4096, "mlock": True, "n_gpu_layers": max_ngl}
        save_phase_results("gpu", {"phase": "gpu", "best_ngl": max_ngl, "skipped": "moe"})
        return max_ngl

    # Skip if max_gpu_layers is 0 or 1 — nothing to sweep
    if max_ngl <= 1:
        print(f"\n[*] Model has {max_ngl} layers — skipping GPU offload sweep.")
        save_phase_results("gpu", {"phase": "gpu", "best_ngl": max_ngl})
        return max_ngl

    # Middle-out sweep: start at center, expand outward in both directions
    # Each direction stops independently when it drops below 50% of best
    center = max_ngl // 2
    up_range = list(range(center, max_ngl + 1))    # center, center+1, center+2, ...
    down_range = list(range(center - 1, -1, -1))   # center-1, center-2, ..., 0

    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)
    print(f"\n[*] Sweeping n_gpu_layers 0-{max_ngl} (middle-out from {center})")
    print(f"    Each direction stops when score drops below 50% of best\n")

    results = []
    best_score = 0.0
    best_ngl = max_ngl
    up_stopped = False
    down_stopped = False
    up_idx = 0
    down_idx = 0
    trial_num = 0

    def _test_ngl(ngl):
        nonlocal best_score, best_ngl, trial_num
        trial_num += 1
        config = {**NAKED_ENGINE, "n_gpu_layers": ngl}
        kill_server()
        proc = start_server(config)
        if not wait_for_server(proc=proc):
            print(f"  [{trial_num}] ngl={ngl:3d}: FAILED (server didn't start)")
            kill_server()
            return None

        perf, promoted = measure_perf_adaptive(best_score)
        score = compute_score(perf)
        results.append({"ngl": ngl, "perf": perf, "score": score, "promoted": promoted})

        marker = ""
        if score > best_score:
            best_score = score
            best_ngl = ngl
            marker = " *NEW BEST*"

        runs_label = "3 runs" if promoted else "1 run"
        print(f"  [{trial_num}] ngl={ngl:3d}: {perf['tps']:.1f} t/s | "
              f"pp: {perf['prompt_tps']:.0f} t/s | TTFT: {perf['ttft']:.0f}ms | "
              f"Score: {score:.1f} ({runs_label}){marker}")
        kill_server()
        return score

    # Alternate: up, down, up, down... stop each direction independently
    while not (up_stopped and down_stopped):
        # Test upward direction
        if not up_stopped and up_idx < len(up_range):
            score = _test_ngl(up_range[up_idx])
            up_idx += 1
            if score is None:
                pass  # server failed, keep trying
            elif best_score > 0 and score < best_score * 0.50 and up_idx > 2:
                print(f"    ↑ Upward direction stopped (score dropped below 50% of best)")
                up_stopped = True
        else:
            up_stopped = True

        # Test downward direction
        if not down_stopped and down_idx < len(down_range):
            score = _test_ngl(down_range[down_idx])
            down_idx += 1
            if score is None:
                pass
            elif best_score > 0 and score < best_score * 0.50 and down_idx > 2:
                print(f"    ↓ Downward direction stopped (score dropped below 50% of best)")
                down_stopped = True
        else:
            down_stopped = True

    if not results:
        print("[!] All offload levels failed. Using default.")
        return DEFAULT_GPU_LAYERS

    # Re-validate top 3 with fresh 3-run measurements
    top3 = sorted(results, key=lambda r: r["score"], reverse=True)[:3]
    revalidate = [r for r in top3 if not r["promoted"]]  # only those that got 1 run
    if revalidate:
        print(f"\n[*] Re-validating {len(revalidate)} top candidates with 3 fresh runs...")
        for r in revalidate:
            ngl = r["ngl"]
            config = {**NAKED_ENGINE, "n_gpu_layers": ngl}
            kill_server()
            proc = start_server(config)
            if wait_for_server(proc=proc):
                perf = measure_perf(runs=3)
                score = compute_score(perf)
                r["perf"] = perf
                r["score"] = score
                r["promoted"] = True
                print(f"    ngl={ngl:3d}: {perf['tps']:.1f} t/s | Score: {score:.1f} (re-validated)")
                kill_server()

        # Recalculate best after re-validation
        best_r = max(results, key=lambda r: r["score"])
        best_ngl = best_r["ngl"]
        best_score = best_r["score"]

    # Show ranking (top 10)
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    show_n = min(10, len(sorted_results))
    print(f"\n  {'ngl':>5s}  {'t/s':>7s}  {'pp':>7s}  {'TTFT':>7s}  {'Score':>7s}")
    print("  " + "-" * 42)
    for r in sorted_results[:show_n]:
        marker = " <<<" if r["ngl"] == best_ngl else ""
        print(f"  {r['ngl']:5d}  {r['perf']['tps']:7.1f}  {r['perf']['prompt_tps']:7.0f}  "
              f"{r['perf']['ttft']:7.0f}  {r['score']:7.1f}{marker}")

    print(f"\n  Winner: n_gpu_layers={best_ngl} (score {best_score:.1f})")

    # Update globals for all subsequent phases
    DEFAULT_GPU_LAYERS = best_ngl
    NAKED_ENGINE = {
        "context": 4096,
        "mlock": True,
        "n_gpu_layers": DEFAULT_GPU_LAYERS,
    }

    save_phase_results("gpu", {
        "phase": "gpu",
        "best_ngl": best_ngl,
        "best_score": best_score,
        "all_results": [{"ngl": r["ngl"], "tps": r["perf"]["tps"], "score": r["score"]} for r in results],
    })

    return best_ngl


def phase_moe_threads(n_trials=40, base_memory_config=None):
    """MoE Thread Sweep: Sweep n_cpu_moe to find optimal MoE thread count.

    Sequential sweep 0-40 with adaptive measurement:
    - Pass 1: test every value with 1 run (bad configs filtered fast)
    - Automatically promotes competitive configs to 3 runs via adaptive measurement

    Returns int (best n_cpu_moe) or None on failure.
    """
    # Check for existing results
    existing = load_phase_results("moe")
    if existing and "best_params" in existing:
        best_moe = existing["best_params"]["n_cpu_moe"]
        print(f"\n[*] MoE thread sweep already complete — n_cpu_moe={best_moe} (from previous run)")
        return best_moe

    phase_start_time = time.time()
    label = "MoE Thread Sweep"
    # Middle-out sweep: each direction stops independently at 50% of best
    up_range = list(range(MOE_SWEEP_CENTER, MOE_SWEEP_MAX + 1))
    down_range = list(range(MOE_SWEEP_CENTER - 1, -1, -1))

    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)

    base_config = {**NAKED_ENGINE}
    if base_memory_config:
        base_config.update(base_memory_config)

    print("\n[*] Starting baseline server...")
    kill_server()
    proc = start_server(base_config)
    if not wait_for_server(proc=proc):
        print("[!] Baseline server failed to start")
        return None
    baseline = measure_perf(runs=3)
    # Also measure large-prompt for baseline so it uses the same scoring
    # formula as promoted trials (prevents baseline score inflation from
    # the higher pp weight in the short formula)
    _bl_large = _measure_perf_large()
    if _bl_large:
        baseline.update(_bl_large)
    print(f"    Baseline: {baseline['tps']:.1f} t/s | pp: {baseline['prompt_tps']:.0f} t/s | "
          f"TTFT: {baseline['ttft']:.0f}ms | Score: {compute_score(baseline):.1f}")

    total = len(up_range) + len(down_range)
    best_score = compute_score(baseline)
    best_moe = 0

    print(f"\n[*] Sweeping 0-{MOE_SWEEP_MAX} (middle-out from {MOE_SWEEP_CENTER})")
    print(f"    Each direction stops when score drops below 50% of best\n")

    results_by_val = {}  # moe_val -> {moe, score, perf, promoted}
    trial_num = 0

    def _test_moe(moe_val, force_3runs=False):
        """Test a single MoE value. Returns score."""
        nonlocal best_score, trial_num
        trial_num += 1
        config = {**base_config, "n_cpu_moe": moe_val}
        params_short = f"moe={moe_val}"

        label = "restarting server..." if not force_3runs else "re-testing (3runs)..."
        print(f"\n  Trial {trial_num}: {label} | {params_short}")
        kill_server()
        proc = start_server(config)

        if not wait_for_server(proc=proc):
            _server_start_failed(trial_num, params_short, proc)
            results_by_val[moe_val] = {"moe": moe_val, "score": 0.0, "perf": None, "promoted": False}
            return 0.0

        if force_3runs:
            perf = measure_perf(runs=3)
            promoted = True
        else:
            perf, promoted = measure_perf_adaptive(best_score)
        tps = perf["tps"]
        score = compute_score(perf)

        results_by_val[moe_val] = {"moe": moe_val, "score": score, "perf": perf, "promoted": promoted}

        runs_label = "3runs" if promoted else "1run"
        best_score = print_trial_result(trial_num, total, tps, perf, f"{params_short} ({runs_label})", best_score)
        return score

    # Pass 1: middle-out sweep with directional stopping
    up_stopped = False
    down_stopped = False
    up_idx = 0
    down_idx = 0

    while not (up_stopped and down_stopped):
        if not up_stopped and up_idx < len(up_range):
            score = _test_moe(up_range[up_idx])
            up_idx += 1
            if score >= best_score:
                best_moe = up_range[up_idx - 1]
            if best_score > 0 and score < best_score * 0.50 and up_idx > 2:
                print(f"    ↑ Upward direction stopped (score dropped below 50% of best)")
                up_stopped = True
        else:
            up_stopped = True

        if not down_stopped and down_idx < len(down_range):
            score = _test_moe(down_range[down_idx])
            down_idx += 1
            if score >= best_score:
                best_moe = down_range[down_idx - 1]
            if best_score > 0 and score < best_score * 0.50 and down_idx > 2:
                print(f"    ↓ Downward direction stopped (score dropped below 50% of best)")
                down_stopped = True
        else:
            down_stopped = True

    # Pass 2: re-test best ±2 neighbors with fresh 3 runs (always, even if already promoted)
    best_entry = max(results_by_val.values(), key=lambda x: x["score"])
    best_moe = best_entry["moe"]
    retest_range = 2  # ±2 neighbors
    retests = []
    for offset in range(-retest_range, retest_range + 1):
        neighbor = best_moe + offset
        if neighbor in results_by_val:
            retests.append(neighbor)

    print(f"\n  [*] Re-testing best ±{retest_range} neighbors ({len(retests)} values) with fresh 3 runs...")
    for moe_val in retests:
        _test_moe(moe_val, force_3runs=True)

    # Find final best after all retests
    best_entry = max(results_by_val.values(), key=lambda x: x["score"])
    best_moe = best_entry["moe"]
    all_results = [results_by_val[v] for v in sorted(results_by_val.keys())]
    best_perf = best_entry["perf"] or baseline

    print(f"\n{'=' * 60}")
    print(f"  {label} — RESULTS")
    print(f"{'=' * 60}")
    print(f"  Baseline:        {baseline['tps']:.1f} t/s | TTFT: {baseline['ttft']:.0f}ms")
    print(f"  Best MoE threads: {best_moe}")
    print(f"  Best Score:      {best_entry['score']:.1f} (composite)")
    print(f"  Best TPS:        {best_perf['tps']:.1f} t/s")
    print(f"  Best TTFT:       {best_perf['ttft']:.0f}ms")

    # Histogram
    max_score = max(r["score"] for r in all_results) if all_results else 0
    bar_max = 30
    print(f"\n  Score by n_cpu_moe:")
    print(f"  {'Value':>6}  {'Score':>7}  {'Runs':>4}  {'':}")
    print(f"  {'─' * 6}  {'─' * 7}  {'─' * 4}  {'─' * bar_max}")
    for r in all_results:
        score = r["score"]
        bar_len = int(score / max_score * bar_max) if max_score > 0 else 0
        bar = "█" * bar_len
        marker = " ◄ best" if r["moe"] == best_moe else ""
        runs = "3" if r["promoted"] else "1"
        print(f"  {r['moe']:>6}  {score:>7.1f}  {runs:>4}  {bar}{marker}")

    phase_elapsed = time.time() - phase_start_time
    print(f"\n  Duration:        {phase_elapsed / 60:.1f} min")

    results = {
        "phase": "moe",
        "baseline": baseline,
        "best_tps": best_entry["score"],
        "best_metrics": {"tps": best_perf["tps"], "ttft": best_perf["ttft"],
                         "prompt_tps": best_perf["prompt_tps"], "total_ms": best_perf["total_ms"]},
        "best_params": {"n_cpu_moe": best_moe},
        "duration_seconds": round(phase_elapsed, 1),
        "all_trials": [
            {"number": i, "tps": r["score"], "metrics": r["perf"], "params": {"n_cpu_moe": r["moe"]}}
            for i, r in enumerate(all_results)
        ],
    }
    save_phase_results("moe", results)

    return best_moe


def phase_experts(n_trials=20, locked_moe_threads=18, base_memory_config=None):
    """Expert Count Sweep: Sweep expert_used_count with perplexity quality gate.

    Sequential sweep 1-16 with adaptive measurement + quality gate.
    MoE threads are locked from MoE Thread Sweep.

    Returns int (best expert_used_count) or 8 (default) on failure.
    """
    # Check for existing results
    existing = load_phase_results("experts")
    if existing and "best_params" in existing:
        best_exp = existing["best_params"]["expert_used_count"]
        print(f"\n[*] Expert sweep already complete — experts={best_exp} (from previous run)")
        return best_exp

    phase_start_time = time.time()
    label = "Expert Count Sweep"
    # Middle-out: each direction stops independently at 50% of best
    up_range = list(range(DEFAULT_EXPERTS, MAX_EXPERTS + 1))
    down_range = list(range(DEFAULT_EXPERTS - 1, 0, -1))

    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)
    print(f"\n[*] Locked MoE threads: {locked_moe_threads}")

    base_config = {**NAKED_ENGINE, "n_cpu_moe": locked_moe_threads}
    if base_memory_config:
        base_config.update(base_memory_config)

    # Start with default experts (8) to establish baseline
    print(f"\n[*] Starting baseline server (default {DEFAULT_EXPERTS} experts)...")
    kill_server()
    proc = start_server(base_config)
    if not wait_for_server(proc=proc):
        print("[!] Baseline server failed to start")
        return DEFAULT_EXPERTS
    baseline = measure_perf(runs=3)
    # Also measure large-prompt for baseline so it uses the same scoring
    # formula as promoted trials (prevents baseline score inflation from
    # the higher pp weight in the short formula)
    _bl_large = _measure_perf_large()
    if _bl_large:
        baseline.update(_bl_large)
    print(f"    Baseline: {baseline['tps']:.1f} t/s | pp: {baseline['prompt_tps']:.0f} t/s | "
          f"TTFT: {baseline['ttft']:.0f}ms | Score: {compute_score(baseline):.1f}")

    # Establish quality baseline with full experts
    print("[*] Measuring baseline quality (token uncertainty calibration)...")
    baseline_qf = measure_quality_gate(is_baseline=True)
    if _quality_baseline is None:
        print("[!] WARNING: Could not measure baseline quality!")
        print("    The server may not support n_probs / completion_probabilities.")
        print(f"    Falling back to default {DEFAULT_EXPERTS} experts (no quality gate available).")
        return DEFAULT_EXPERTS

    total = len(up_range) + len(down_range)
    best_score = compute_score(baseline)
    best_experts = DEFAULT_EXPERTS
    results_by_val = {}  # expert_count -> result dict
    trial_num = 0

    print(f"\n[*] Sweeping 1-{MAX_EXPERTS} (middle-out from {DEFAULT_EXPERTS})")
    print(f"    Each direction stops when score drops below 50% of best\n")

    def _test_expert(expert_count, force_3runs=False):
        """Test a single expert count. Returns score."""
        nonlocal best_score, trial_num
        trial_num += 1
        config = {**base_config, "expert_used_count": expert_count}
        params_short = f"experts={expert_count}"

        lbl = "restarting server..." if not force_3runs else "re-testing (3runs)..."
        print(f"\n  Trial {trial_num}: {lbl} | {params_short}")
        kill_server()
        proc = start_server(config)

        if not wait_for_server(proc=proc):
            _server_start_failed(trial_num, params_short, proc)
            results_by_val[expert_count] = {"experts": expert_count, "score": 0.0, "speed_score": 0.0,
                                            "perf": None, "quality_factor": 0.0, "promoted": False}
            return 0.0

        if force_3runs:
            perf = measure_perf(runs=3)
            promoted = True
        else:
            perf, promoted = measure_perf_adaptive(best_score)
        tps = perf["tps"]
        speed_score = compute_score(perf)

        # Always measure quality in expert phase — that's the whole point
        quality_factor = measure_quality_gate()
        score = speed_score * quality_factor

        results_by_val[expert_count] = {"experts": expert_count, "score": score, "speed_score": speed_score,
                                        "perf": perf, "quality_factor": quality_factor, "promoted": promoted}

        qf_label = f" q={quality_factor:.2f}" if quality_factor < 1.0 else ""
        runs_label = "3runs" if promoted else "1run"
        best_score = print_trial_result(trial_num, total, tps, perf, f"{params_short} ({runs_label}){qf_label}",
                                        best_score, final_score=score)
        return score

    # Pass 1: middle-out sweep with directional stopping
    up_stopped = False
    down_stopped = False
    up_idx = 0
    down_idx = 0

    while not (up_stopped and down_stopped):
        if not up_stopped and up_idx < len(up_range):
            score = _test_expert(up_range[up_idx])
            up_idx += 1
            if score >= best_score:
                best_experts = up_range[up_idx - 1]
            if best_score > 0 and score < best_score * 0.50 and up_idx > 2:
                print(f"    ↑ Upward direction stopped (score dropped below 50% of best)")
                up_stopped = True
        else:
            up_stopped = True

        if not down_stopped and down_idx < len(down_range):
            score = _test_expert(down_range[down_idx])
            down_idx += 1
            if score >= best_score:
                best_experts = down_range[down_idx - 1]
            if best_score > 0 and score < best_score * 0.50 and down_idx > 2:
                print(f"    ↓ Downward direction stopped (score dropped below 50% of best)")
                down_stopped = True
        else:
            down_stopped = True

    # Pass 2: re-test neighbors of the best with 3 runs if they only got 1
    best_entry = max(results_by_val.values(), key=lambda x: x["score"])
    best_experts = best_entry["experts"]
    retest_range = 2  # ±2 neighbors
    retests = []
    for offset in range(-retest_range, retest_range + 1):
        neighbor = best_experts + offset
        if neighbor in results_by_val:
            retests.append(neighbor)

    print(f"\n  [*] Re-testing best ±{retest_range} neighbors ({len(retests)} values) with fresh 3 runs...")
    for expert_count in retests:
        _test_expert(expert_count, force_3runs=True)

    # Find final best after all retests
    best_entry = max(results_by_val.values(), key=lambda x: x["score"])
    best_experts = best_entry["experts"]
    all_results = [results_by_val[v] for v in sorted(results_by_val.keys())]
    best_perf = best_entry["perf"] or baseline

    print(f"\n{'=' * 60}")
    print(f"  {label} — RESULTS")
    print(f"{'=' * 60}")
    print(f"  Baseline:     {baseline['tps']:.1f} t/s ({DEFAULT_EXPERTS} experts)")
    print(f"  Best experts: {best_experts}")
    print(f"  Best Score:   {best_entry['score']:.1f} (speed × quality)")
    print(f"  Best TPS:     {best_perf['tps']:.1f} t/s")
    print(f"  Quality:      {best_entry['quality_factor']:.2f}")

    # Histogram
    max_score = max(r["score"] for r in all_results) if all_results else 0
    bar_max = 30
    print(f"\n  Score by expert_used_count (quality-adjusted):")
    print(f"  {'Value':>6}  {'Score':>7}  {'QF':>5}  {'Runs':>4}  {'':}")
    print(f"  {'─' * 6}  {'─' * 7}  {'─' * 5}  {'─' * 4}  {'─' * bar_max}")
    for r in all_results:
        score = r["score"]
        bar_len = int(score / max_score * bar_max) if max_score > 0 else 0
        bar = "█" * bar_len
        marker = " ◄ best" if r["experts"] == best_experts else ""
        runs = "3" if r["promoted"] else "1"
        qf = f"{r['quality_factor']:.2f}"
        print(f"  {r['experts']:>6}  {score:>7.1f}  {qf:>5}  {runs:>4}  {bar}{marker}")

    phase_elapsed = time.time() - phase_start_time
    print(f"\n  Duration:     {phase_elapsed / 60:.1f} min")

    results = {
        "phase": "experts",
        "baseline": baseline,
        "baseline_quality": _quality_baseline,
        "best_tps": best_entry["score"],
        "best_metrics": {"tps": best_perf["tps"], "ttft": best_perf["ttft"],
                         "prompt_tps": best_perf["prompt_tps"], "total_ms": best_perf["total_ms"],
                         "quality_factor": best_entry["quality_factor"]},
        "best_params": {"expert_used_count": best_experts},
        "duration_seconds": round(phase_elapsed, 1),
        "all_trials": [
            {"number": i, "tps": r["score"], "metrics": r["perf"], "params": {"expert_used_count": r["experts"]},
             "quality_factor": r["quality_factor"]}
            for i, r in enumerate(all_results)
        ],
    }
    save_phase_results("experts", results)

    return best_experts


def phase_moe(n_trials=60, base_memory_config=None, include_experts=False):
    """MoE: Find optimal MoE config (threads + optionally experts).

    Runs MoE thread sweep (mandatory), then expert count sweep (optional).
      MoE thread sweep — sequential sweep, adaptive measurement
      Expert count sweep — optional, sequential sweep + quality gate

    For dense models (IS_MOE=False), skips entirely and returns empty config.
    n_trials is ignored — both sub-phases sweep their full ranges.

    Returns dict: {"n_cpu_moe": int, "expert_used_count": int} or None on failure.
    """
    if not IS_MOE:
        print("\n[*] Dense model detected — skipping MoE phase.")
        save_phase_results("moe_combined", {"phase": "moe_combined", "best_params": {}})
        return {}

    # Sub-phase 1: MoE threads — full sweep (always)
    best_moe_threads = phase_moe_threads(base_memory_config=base_memory_config)
    if best_moe_threads is None:
        return None

    # Sub-phase 2: Expert count — optional
    best_experts = DEFAULT_EXPERTS
    if include_experts:
        best_experts = phase_experts(locked_moe_threads=best_moe_threads,
                                    base_memory_config=base_memory_config)

    # Save combined result so _get_moe_config() can load it
    combined = {"n_cpu_moe": best_moe_threads, "expert_used_count": best_experts}
    save_phase_results("moe_combined", {
        "phase": "moe_combined",
        "best_params": combined,
    })

    return combined


# ============================================================
# MoE Audit (re-validate MoE with compute params locked)
# ============================================================

def phase_moe_revalidate(locked_compute=None, locked_moe=None, base_memory_config=None):
    """MoE Audit: Re-test best ±2 MoE thread values with compute params locked.

    After Compute Audit finds optimal compute, the MoE sweet spot may shift.
    This does a quick focused sweep (~5 values) instead of the full 0-40.

    Args:
        locked_compute: Compute params from Compute Audit (threads, speculation, etc.)
        locked_moe: Current MoE config from MoE phase {"n_cpu_moe": int, "expert_used_count": int}
        base_memory_config: Memory params from Memory phase (if available)

    Returns int (best n_cpu_moe) or None on failure.
    """
    if not IS_MOE:
        print("\n[*] Dense model detected — skipping MoE re-validation.")
        return None

    existing = load_phase_results("moe_audit")
    if existing and "best_params" in existing:
        best_moe = existing["best_params"]["n_cpu_moe"]
        print(f"\n[*] MoE re-validation already complete — n_cpu_moe={best_moe} (from previous run)")
        return best_moe

    if locked_moe is None:
        locked_moe = {"n_cpu_moe": MOE_SWEEP_CENTER, "expert_used_count": DEFAULT_EXPERTS}

    current_best = locked_moe["n_cpu_moe"]
    expert_count = locked_moe["expert_used_count"]
    retest_range = 2  # ±2 neighbors

    phase_start_time = time.time()
    label = "MoE Audit"

    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)
    print(f"\n[*] Current MoE threads: {current_best} (from MoE phase)")
    if locked_compute:
        print(f"[*] Locked compute from Compute Audit: {len(locked_compute)} params")

    # Build base config with compute + memory params locked
    base_config = {**NAKED_ENGINE}
    if base_memory_config:
        base_config.update(base_memory_config)
    if locked_compute:
        base_config.update(locked_compute)
    base_config["expert_used_count"] = expert_count

    # Measure baseline with current MoE setting
    print(f"\n[*] Starting baseline server (moe={current_best})...")
    kill_server()
    baseline_config = {**base_config, "n_cpu_moe": current_best}
    proc = start_server(baseline_config)
    if not wait_for_server(proc=proc):
        print("[!] Baseline server failed to start")
        return None
    baseline = measure_perf(runs=3)
    baseline_score = compute_score(baseline)
    print(f"    Baseline (moe={current_best}): {baseline['tps']:.1f} t/s | "
          f"pp: {baseline['prompt_tps']:.0f} t/s | TTFT: {baseline['ttft']:.0f}ms | "
          f"Score: {baseline_score:.1f}")

    # Build test values: best ±2, clamped to valid range
    test_values = []
    for offset in range(-retest_range, retest_range + 1):
        val = current_best + offset
        if 1 <= val <= MOE_SWEEP_MAX:
            test_values.append(val)

    print(f"\n[*] Re-testing MoE threads {test_values} with compute params locked...")

    results = {}
    best_score = baseline_score
    best_moe = current_best

    for i, moe_val in enumerate(test_values):
        print(f"\n  Test {i + 1}/{len(test_values)}: moe={moe_val}")
        kill_server()
        config = {**base_config, "n_cpu_moe": moe_val}
        proc = start_server(config)
        if not wait_for_server(proc=proc):
            print(f"    [!] Server failed to start for moe={moe_val}")
            results[moe_val] = {"score": 0.0, "perf": None}
            continue

        perf = measure_perf(runs=3)
        score = compute_score(perf)
        results[moe_val] = {"score": score, "perf": perf}

        marker = " *** NEW BEST ***" if score > best_score else ""
        print(f"    moe={moe_val}: {perf['tps']:.1f} t/s | pp: {perf['prompt_tps']:.0f} t/s | "
              f"TTFT: {perf['ttft']:.0f}ms | Score: {score:.1f}{marker}")

        if score > best_score:
            best_score = score
            best_moe = moe_val

    # Results summary
    phase_elapsed = time.time() - phase_start_time
    changed = best_moe != current_best

    print(f"\n{'=' * 60}")
    print(f"  {label} — RESULTS")
    print(f"{'=' * 60}")
    print(f"  Previous best: moe={current_best} (Score: {baseline_score:.1f})")
    print(f"  New best:      moe={best_moe} (Score: {best_score:.1f})")
    if changed:
        print(f"  ** MoE threads changed from {current_best} to {best_moe} **")
    else:
        print(f"  MoE threads confirmed at {best_moe}")

    # Score table
    print(f"\n  {'Value':>6}  {'Score':>7}  {'TPS':>6}  {'':}")
    print(f"  {'─' * 6}  {'─' * 7}  {'─' * 6}  {'─' * 20}")
    for val in sorted(results.keys()):
        r = results[val]
        perf = r["perf"]
        if perf:
            marker = " ◄ best" if val == best_moe else ""
            tps_str = f"{perf['tps']:.1f}"
            print(f"  {val:>6}  {r['score']:>7.1f}  {tps_str:>6}  {marker}")

    print(f"\n  Duration: {phase_elapsed / 60:.1f} min")

    best_perf = results.get(best_moe, {}).get("perf") or baseline
    save_phase_results("moe_audit", {
        "phase": "moe_audit",
        "baseline": baseline,
        "previous_moe": current_best,
        "best_tps": best_score,
        "best_metrics": {"tps": best_perf["tps"], "ttft": best_perf["ttft"],
                         "prompt_tps": best_perf["prompt_tps"], "total_ms": best_perf["total_ms"]},
        "best_params": {"n_cpu_moe": best_moe},
        "changed": changed,
        "duration_seconds": round(phase_elapsed, 1),
        "all_trials": [
            {"params": {"n_cpu_moe": val}, "score": results[val]["score"],
             "metrics": results[val]["perf"]}
            for val in sorted(results.keys()) if results[val]["perf"]
        ],
    })

    return best_moe


# ============================================================
# Compute / Compute Audit (with MoE locked)
# (threads, speculation, poll, prio)
# ============================================================

def phase_compute(n_trials=60, phase_name="compute", base_memory_config=None, seed_params=None, locked_moe=None):
    """Optimize compute allocation params (with MoE locked from MoE phase).

    Args:
        n_trials: Number of trials to run.
        phase_name: "compute" or "compute_audit" — used for study name and result file.
        base_memory_config: If provided (Compute Audit), use these memory/throughput
                           params as the base. Otherwise starts naked.
        locked_moe: Dict from MoE phase {"n_cpu_moe": int, "expert_used_count": int} (locked, not tuned).
    """
    if not locked_moe:  # None or empty dict -- use defaults (dense model or no MoE phase run)
        locked_moe = {"n_cpu_moe": MOE_SWEEP_CENTER, "expert_used_count": DEFAULT_EXPERTS}
    phase_start_time = time.time()
    labels = {
        "compute": "Compute",
        "compute_audit": "Compute Audit",
    }
    label = labels.get(phase_name, f"{phase_name}: Compute Allocation")

    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)
    moe_n = locked_moe.get('n_cpu_moe', 'none')
    moe_e = locked_moe.get('expert_used_count', 'none')
    print(f"\n[*] Locked MoE: {moe_n} | Experts: {moe_e}")

    if base_memory_config:
        mem_src = "Memory" if phase_name == "compute_audit" else "previous"
        print(f"\n[*] Base memory config from {mem_src}: {len(base_memory_config)} params")

    # Naked baseline (with memory config if revalidating)
    # Check if study is already complete before starting baseline server
    study, remaining, completed = setup_study(phase_name, n_trials)
    if remaining == 0:
        best = study.best_trial
        print(f"\n  Best Score:  {best.value:.1f} | TPS: {best.user_attrs.get('tps', 0):.1f} | "
              f"TTFT: {best.user_attrs.get('ttft', 0):.0f}ms")
        print_param_importance(study)
        return best.params

    base_config = {**NAKED_ENGINE}
    if base_memory_config:
        base_config.update(base_memory_config)
    # Always include locked MoE params in baseline
    if locked_moe:
        base_config.update(locked_moe)

    print("\n[*] Starting baseline server...")
    kill_server()
    proc = start_server(base_config)
    if not wait_for_server(proc=proc):
        print("[!] Baseline server failed to start")
        return None
    baseline = measure_perf(runs=3)
    # Also measure large-prompt for baseline so it uses the same scoring
    # formula as promoted trials (prevents baseline score inflation from
    # the higher pp weight in the short formula)
    _bl_large = _measure_perf_large()
    if _bl_large:
        baseline.update(_bl_large)
    print(f"    Baseline: {baseline['tps']:.1f} t/s | pp: {baseline['prompt_tps']:.0f} t/s | "
          f"TTFT: {baseline['ttft']:.0f}ms | Score: {compute_score(baseline):.1f}")

    # Seed with previous phase's best params so TPE starts from a known good point
    if seed_params and completed == 0:
        print(f"[*] Seeding Trial 0 with previous best config")
        study.enqueue_trial(seed_params)
    elif completed == 0:
        # Seed with proven best compute config from previous runs
        print(f"[*] Seeding Trial 0 with known-good compute config")
        study.enqueue_trial({
            "threads": 4,
            "threads_batch": 12,
            "poll": 0, "poll_batch": 50,
            "prio": 0, "prio_batch": 0,
            "cpu_strict": 1, "cpu_strict_batch": 1,
            "spec_type": "ngram-map-k4v",
            "spec_ngram_n": 23, "spec_ngram_m": 20,
            "spec_ngram_min_hits": 2,
            "draft_max": 9, "draft_min": 8,
            "draft_p_min": 0.9,
            "lookup_cache_dynamic": LOOKUP_CACHE_FILE,
        })
        # Second seed: alternate config with higher threads + speculation
        study.enqueue_trial({
            "threads": 16,
            "threads_batch": 16,
            "poll": 50, "poll_batch": 50,
            "prio": 0, "prio_batch": 3,
            "cpu_strict": 0, "cpu_strict_batch": 1,
            "spec_type": "ngram-cache",
            "spec_ngram_n": 14, "spec_ngram_m": 64,
            "spec_ngram_min_hits": 4,
            "draft_max": 47, "draft_min": 4,
            "draft_p_min": 0.52,
            "lookup_cache_dynamic": None,
        })

    total_trials = completed + remaining
    best_score = compute_score(baseline)
    # When resuming, recalculate best score from existing trials using CURRENT formula
    # (stored values may use an old formula and poison adaptive measurement thresholds)
    if completed > 0:
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs:
                perf = {k: t.user_attrs.get(k, 0) for k in ["tps", "ttft", "prompt_tps", "total_ms"]}
                if perf["tps"] > 0:
                    recalc = compute_score(perf)
                    best_score = max(best_score, recalc)

    def objective(trial):
        nonlocal best_score

        config = {
            **base_config,
            # MoE locked from MoE phase
            **locked_moe,
            # Compute allocation params
            "threads": trial.suggest_int("threads", 4, MAX_THREADS, step=4),
            "threads_batch": trial.suggest_int("threads_batch", 4, MAX_THREADS, step=4),
            "poll": trial.suggest_categorical("poll", [0, 10, 25, 50, 100]),
            "poll_batch": trial.suggest_categorical("poll_batch", [0, 10, 25, 50, 100]),
            "prio": trial.suggest_int("prio", 0, 3),
            "prio_batch": trial.suggest_int("prio_batch", 0, 3),
            "cpu_strict": trial.suggest_categorical("cpu_strict", [0, 1]),
            "cpu_strict_batch": trial.suggest_categorical("cpu_strict_batch", [0, 1]),
            # Speculation params
            "spec_type": trial.suggest_categorical("spec_type", ["ngram-simple", "ngram-cache", "ngram-map-k", "ngram-map-k4v", "ngram-mod"]),
            "spec_ngram_n": trial.suggest_int("spec_ngram_n", 2, 24),
            "spec_ngram_m": trial.suggest_int("spec_ngram_m", 8, 96),
            "spec_ngram_min_hits": trial.suggest_int("spec_ngram_min_hits", 1, 5),
            "draft_max": trial.suggest_int("draft_max", 4, 48),
            "draft_min": trial.suggest_int("draft_min", 0, 8),
            "draft_p_min": trial.suggest_float("draft_p_min", 0.3, 0.99),
            "lookup_cache_dynamic": trial.suggest_categorical("lookup_cache_dynamic", [None, LOOKUP_CACHE_FILE]),
        }
        # Remove None values so start_server doesn't see the key
        config = {k: v for k, v in config.items() if v is not None}

        # Pre-boot pruning
        if config.get("draft_min", 0) >= config.get("draft_max", 4):
            print(f"\n  Trial {trial.number}: pruned (draft_min >= draft_max)")
            return 0.0

        # Check for duplicate config before restarting server
        cached = check_duplicate_trial(trial)
        if cached is not None:
            print(f"\n  Trial {trial.number}: duplicate config — cached score: {cached:.1f}")
            return cached

        params_short = (f"t={config['threads']}/{config['threads_batch']} "
                        f"moe={config['n_cpu_moe']} experts={config['expert_used_count']} "
                        f"poll={config['poll']} prio={config['prio']} "
                        f"spec_n={config['spec_ngram_n']} spec_m={config['spec_ngram_m']} "
                        f"draft={config['draft_max']}")

        print(f"\n  Trial {trial.number}: restarting server... | {params_short}")
        kill_server()
        proc = start_server(config)

        if not wait_for_server(proc=proc):
            # Capture stderr to show why it failed
            reason = ""
            try:
                lines = getattr(proc, "_stderr_lines", [])
                for line in reversed(lines):
                    line = line.strip()
                    if line and ("error" in line.lower() or "failed" in line.lower() or "abort" in line.lower() or "oom" in line.lower() or "alloc" in line.lower() or "CUDA" in line or "memory" in line.lower() or "unknown" in line.lower() or "invalid" in line.lower()):
                        reason = f" → {line[:120]}"
                        break
                if not reason and lines:
                    last_lines = [l.strip() for l in lines if l.strip()]
                    if last_lines:
                        reason = f" → {last_lines[-1][:120]}"
            except Exception:
                pass
            print(f"  Trial {trial.number}: FAILED | {params_short}{reason}")
            proc.kill()
            return 0.0

        perf, promoted = measure_perf_adaptive(best_score)
        tps = perf["tps"]
        score = compute_score(perf)

        trial.set_user_attr("tps", tps)
        trial.set_user_attr("ttft", perf["ttft"])
        trial.set_user_attr("prompt_tps", perf["prompt_tps"])
        trial.set_user_attr("total_ms", perf["total_ms"])

        best_score = print_trial_result(trial.number, total_trials, tps, perf, params_short, best_score)
        return score

    est_minutes = remaining * 20 // 60
    print(f"\n[*] Running {remaining} trials (+ {completed} completed, ~{est_minutes} min)...\n")
    study.optimize(objective, n_trials=remaining, callbacks=[GPStoppingCallback(baseline_score=best_score)], show_progress_bar=False)

    # Results
    best = study.best_trial
    baseline_score = compute_score(baseline)
    beat_baseline = best.value > baseline_score

    print(f"\n{'=' * 60}")
    print(f"  {label} — RESULTS")
    print(f"{'=' * 60}")
    print(f"  Baseline:    {baseline['tps']:.1f} t/s | TTFT: {baseline['ttft']:.0f}ms | Score: {baseline_score:.1f}")
    if beat_baseline:
        print(f"  Best Score:  {best.value:.1f} (composite) — beats baseline by {best.value - baseline_score:.1f}")
    else:
        print(f"  Best Score:  {best.value:.1f} (composite) — BELOW baseline ({baseline_score:.1f})")
        print(f"  [!] No trial beat baseline. Using naked defaults for this phase.")
    print(f"  Best TPS:    {best.user_attrs.get('tps', 0):.1f} t/s")
    print(f"  Best TTFT:   {best.user_attrs.get('ttft', 0):.0f}ms")
    print(f"  Best Prompt: {best.user_attrs.get('prompt_tps', 0):.0f} t/s")
    print(f"  Best params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    importances = print_param_importance(study)

    phase_elapsed = time.time() - phase_start_time
    phase_mins = phase_elapsed / 60
    print(f"\n  Duration:    {phase_mins:.1f} min")

    # If nothing beat baseline, return empty dict so pipeline uses naked defaults
    returned_params = best.params if beat_baseline else {}

    results = {
        "phase": phase_name,
        "baseline": baseline,
        "baseline_score": baseline_score,
        "beat_baseline": beat_baseline,
        "best_tps": best.value,
        "best_metrics": best.user_attrs,
        "best_params": returned_params,
        "base_memory_config": base_memory_config,
        "param_importance": {k: round(v * 100, 1) for k, v in importances.items()},
        "duration_seconds": round(phase_elapsed, 1),
        "duration_minutes": round(phase_mins, 1),
        "all_trials": [
            {"number": t.number, "tps": t.value, "metrics": t.user_attrs, "params": t.params}
            for t in study.trials
        ],
    }
    save_phase_results(phase_name, results)

    return returned_params


# ============================================================
# Memory / Memory Audit
# (batch, ubatch, KV cache, flash-attn, mlock, mmap, etc.)
# ============================================================

def phase_memory(n_trials=60, phase_name="memory", base_compute_config=None, seed_params=None):
    """Optimize memory & throughput params. Each trial restarts the server.

    Args:
        n_trials: Number of trials to run.
        phase_name: "memory" or "memory_audit" — used for study name and result file.
        base_compute_config: Compute allocation params to use as base.
    """
    phase_start_time = time.time()
    is_revalidation = phase_name == "memory_audit"
    label = "Memory Audit" if is_revalidation else "Memory"

    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)

    # Check if study is already complete before starting baseline server
    study, remaining, completed = setup_study(phase_name, n_trials)
    if remaining == 0:
        best = study.best_trial
        print(f"\n  Best Score:  {best.value:.1f} | TPS: {best.user_attrs.get('tps', 0):.1f} | "
              f"TTFT: {best.user_attrs.get('ttft', 0):.0f}ms")
        print_param_importance(study)
        return best.params

    # Build base config from compute results
    base_config = {**NAKED_ENGINE}
    if base_compute_config:
        # Map compute trial params to server config keys
        base_config["threads"] = base_compute_config.get("threads")
        base_config["threads_batch"] = base_compute_config.get("threads_batch")
        base_config["n_cpu_moe"] = base_compute_config.get("n_cpu_moe")
        base_config["expert_used_count"] = base_compute_config.get("expert_used_count")
        base_config["poll"] = base_compute_config.get("poll")
        base_config["poll_batch"] = base_compute_config.get("poll_batch")
        base_config["prio"] = base_compute_config.get("prio")
        base_config["prio_batch"] = base_compute_config.get("prio_batch")
        base_config["cpu_strict"] = base_compute_config.get("cpu_strict")
        base_config["cpu_strict_batch"] = base_compute_config.get("cpu_strict_batch")
        base_config["spec_type"] = base_compute_config.get("spec_type", "ngram-simple")
        base_config["spec_ngram_n"] = base_compute_config.get("spec_ngram_n")
        base_config["spec_ngram_m"] = base_compute_config.get("spec_ngram_m")
        base_config["spec_ngram_min_hits"] = base_compute_config.get("spec_ngram_min_hits")
        base_config["draft_max"] = base_compute_config.get("draft_max")
        base_config["draft_min"] = base_compute_config.get("draft_min")
        base_config["draft_p_min"] = base_compute_config.get("draft_p_min")
        if base_compute_config.get("lookup_cache_dynamic"):
            base_config["lookup_cache_dynamic"] = base_compute_config["lookup_cache_dynamic"]
        # Remove None values
        base_config = {k: v for k, v in base_config.items() if v is not None}
        print(f"\n[*] Base compute config: t={base_compute_config.get('threads')}/{base_compute_config.get('threads_batch')} "
              f"moe={base_compute_config.get('n_cpu_moe')} experts={base_compute_config.get('expert_used_count', 8)} "
              f"spec_n={base_compute_config.get('spec_ngram_n')} spec_m={base_compute_config.get('spec_ngram_m')} "
              f"draft={base_compute_config.get('draft_max')}")
    else:
        print("\n[!] No compute config — running with naked engine.")

    # Baseline
    print("\n[*] Starting baseline server...")
    print(f"    Config keys: {sorted(base_config.keys())}")
    kill_server()
    proc = start_server(base_config)
    if not wait_for_server(proc=proc):
        print("[!] Baseline server failed to start")
        return None
    baseline = measure_perf(runs=3)
    # Also measure large-prompt for baseline so it uses the same scoring
    # formula as promoted trials (prevents baseline score inflation from
    # the higher pp weight in the short formula)
    _bl_large = _measure_perf_large()
    if _bl_large:
        baseline.update(_bl_large)
    print(f"    Baseline: {baseline['tps']:.1f} t/s | pp: {baseline['prompt_tps']:.0f} t/s | "
          f"TTFT: {baseline['ttft']:.0f}ms | Score: {compute_score(baseline):.1f}")

    # Seed with previous phase's best params so TPE starts from a known good point
    if seed_params and completed == 0:
        print(f"[*] Seeding Trial 0 with previous best config")
        study.enqueue_trial(seed_params)
    elif completed == 0:
        # No seed — enqueue a known-good baseline config (f16 KV, small batch, fa on)
        print(f"[*] Seeding Trial 0 with known-good f16 config")
        study.enqueue_trial({
            "batch_size": 512, "ubatch_size": 128, "flash_attn": "on",
            "kv_cache_type": "f16", "swa_full": False, "repack": False,
            "op_offload": False, "mlock": True, "no_mmap": True,
        })

    total_trials = completed + remaining
    best_score = compute_score(baseline)
    # When resuming, recalculate best score from existing trials using CURRENT formula
    if completed > 0:
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs:
                perf = {k: t.user_attrs.get(k, 0) for k in ["tps", "ttft", "prompt_tps", "total_ms"]}
                if perf["tps"] > 0:
                    recalc = compute_score(perf)
                    best_score = max(best_score, recalc)

    ctx = base_config.get("context", 4096)

    def objective(trial):
        nonlocal best_score

        # Batch size capped to context — larger values get silently clamped by fit=True anyway
        batch_opts = [v for v in [512, 1024, 2048, 4096] if v <= ctx]
        batch_size = trial.suggest_categorical("batch_size", batch_opts)
        ubatch_size = trial.suggest_categorical("ubatch_size", [128, 256, 512, 1024])

        # Pre-boot pruning: skip logically impossible configs
        if ubatch_size > batch_size:
            print(f"\n  Trial {trial.number}: pruned (ubatch {ubatch_size} > batch {batch_size})")
            return 0.0

        flash_attn = trial.suggest_categorical("flash_attn", ["on", "off"])
        kv_cache_type = trial.suggest_categorical("kv_cache_type", ["f16", "bf16", "q8_0", "q5_1", "q4_0"])

        # Quantized KV cache requires flash attention — skip impossible combos
        if flash_attn == "off" and kv_cache_type not in ("f16", "bf16"):
            print(f"\n  Trial {trial.number}: pruned (quantized KV {kv_cache_type} requires flash_attn=on)")
            return 0.0

        config = {
            **base_config,
            "context": ctx,
            "batch_size": batch_size,
            "ubatch_size": ubatch_size,
            # Core memory params that actually affect single-user inference
            "flash_attn": flash_attn,
            "kv_cache_type": kv_cache_type,
            "swa_full": trial.suggest_categorical("swa_full", [True, False]),
            "repack": trial.suggest_categorical("repack", [True, False]),
            "op_offload": trial.suggest_categorical("op_offload", [True, False]),
            "mlock": trial.suggest_categorical("mlock", [True, False]),
            "no_mmap": trial.suggest_categorical("no_mmap", [True, False]),
            # GPU layers locked from GPU Offload
            "n_gpu_layers": DEFAULT_GPU_LAYERS,
            "fit": True,
        }
        # Expand matched KV cache type to separate k/v params for server
        if "kv_cache_type" in config:
            kv_type = config.pop("kv_cache_type")
            config["cache_type_k"] = kv_type
            config["cache_type_v"] = kv_type
        config = {k: v for k, v in config.items() if v is not None}

        # Check for duplicate config before restarting server
        cached = check_duplicate_trial(trial)
        if cached is not None:
            print(f"\n  Trial {trial.number}: duplicate config — cached score: {cached:.1f}")
            return cached

        params_short = (f"b={config['batch_size']} ub={config['ubatch_size']} "
                        f"fa={config['flash_attn']} "
                        f"kv={config['cache_type_k']}/{config['cache_type_v']}")

        print(f"\n  Trial {trial.number}: restarting server... | {params_short}")
        kill_server()
        proc = start_server(config)

        if not wait_for_server(proc=proc):
            reason = ""
            try:
                lines = getattr(proc, "_stderr_lines", [])
                for line in reversed(lines):
                    line = line.strip()
                    if line and ("error" in line.lower() or "failed" in line.lower() or "abort" in line.lower() or "oom" in line.lower() or "alloc" in line.lower() or "CUDA" in line or "memory" in line.lower() or "unknown" in line.lower() or "invalid" in line.lower()):
                        reason = f" → {line[:120]}"
                        break
                if not reason and lines:
                    last_lines = [l.strip() for l in lines if l.strip()]
                    if last_lines:
                        reason = f" → {last_lines[-1][:120]}"
            except Exception:
                pass
            print(f"  Trial {trial.number}: FAILED | {params_short}{reason}")
            proc.kill()
            return 0.0

        perf, promoted = measure_perf_adaptive(best_score)
        tps = perf["tps"]
        score = compute_score(perf)

        trial.set_user_attr("tps", tps)
        trial.set_user_attr("ttft", perf["ttft"])
        trial.set_user_attr("prompt_tps", perf["prompt_tps"])
        trial.set_user_attr("total_ms", perf["total_ms"])

        best_score = print_trial_result(trial.number, total_trials, tps, perf, params_short, best_score)
        return score

    est_minutes = remaining * 20 // 60
    print(f"\n[*] Running {remaining} trials (+ {completed} completed, ~{est_minutes} min)...\n")
    study.optimize(objective, n_trials=remaining, callbacks=[GPStoppingCallback(baseline_score=best_score)], show_progress_bar=False)

    # Restore best config for verification
    best = study.best_trial
    print(f"\n[*] Restarting with best config for verification...")
    kill_server()
    verify_config_mapped = {**base_config, **best.params}
    # Map kv_cache_type to the separate k/v params that start_server expects
    if "kv_cache_type" in verify_config_mapped:
        kv_type = verify_config_mapped.pop("kv_cache_type")
        verify_config_mapped["cache_type_k"] = kv_type
        verify_config_mapped["cache_type_v"] = kv_type
    verify_config_mapped = {k: v for k, v in verify_config_mapped.items() if v is not None}
    best_proc = start_server(verify_config_mapped)
    wait_for_server(proc=best_proc)
    verify = measure_perf(runs=3)

    baseline_score = compute_score(baseline)
    beat_baseline = best.value > baseline_score

    print(f"\n{'=' * 60}")
    print(f"  {label} — RESULTS")
    print(f"{'=' * 60}")
    print(f"  Baseline:    {baseline['tps']:.1f} t/s | TTFT: {baseline['ttft']:.0f}ms | Score: {baseline_score:.1f}")
    if beat_baseline:
        print(f"  Best Score:  {best.value:.1f} (composite) — beats baseline by {best.value - baseline_score:.1f}")
    else:
        print(f"  Best Score:  {best.value:.1f} (composite) — BELOW baseline ({baseline_score:.1f})")
        print(f"  [!] No trial beat baseline. Using naked defaults for this phase.")
    print(f"  Best TPS:    {best.user_attrs.get('tps', 0):.1f} t/s")
    print(f"  Best TTFT:   {best.user_attrs.get('ttft', 0):.0f}ms")
    print(f"  Verified:    {verify['tps']:.1f} t/s | TTFT: {verify['ttft']:.0f}ms | Score: {compute_score(verify):.1f}")
    print(f"  Best params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    importances = print_param_importance(study)

    phase_elapsed = time.time() - phase_start_time
    phase_mins = phase_elapsed / 60
    print(f"\n  Duration:    {phase_mins:.1f} min")

    returned_params = best.params if beat_baseline else {}

    results = {
        "phase": phase_name,
        "baseline": baseline,
        "baseline_score": baseline_score,
        "beat_baseline": beat_baseline,
        "best_tps": best.value,
        "best_metrics": best.user_attrs,
        "verified": verify,
        "best_params": returned_params,
        "base_compute_config": base_compute_config,
        "param_importance": {k: round(v * 100, 1) for k, v in importances.items()},
        "duration_seconds": round(phase_elapsed, 1),
        "duration_minutes": round(phase_mins, 1),
        "all_trials": [
            {"number": t.number, "tps": t.value, "metrics": t.user_attrs, "params": t.params}
            for t in study.trials
        ],
    }
    save_phase_results(phase_name, results)

    return returned_params


# ============================================================
# Quality (sampling params)
# ============================================================

def phase3(n_trials=80):
    """Optimize sampling params. Server runs with best compute + memory config."""
    if n_trials <= 0:
        return None
    print("\n" + "=" * 60)
    print("  Quality / Sampling")
    print("=" * 60)

    # Build server config from best available results
    # Prefer revalidation results (1b/2b) over initial (1/2)
    server_config = {**NAKED_ENGINE}

    # Load compute config (prefer 1c over 1b)
    compute_src = load_phase_results("compute_audit") or load_phase_results("compute")
    if compute_src:
        cp = compute_src["best_params"]
        server_config["threads"] = cp.get("threads")
        server_config["threads_batch"] = cp.get("threads_batch")
        server_config["poll"] = cp.get("poll")
        server_config["poll_batch"] = cp.get("poll_batch")
        server_config["prio"] = cp.get("prio")
        server_config["prio_batch"] = cp.get("prio_batch")
        server_config["cpu_strict"] = cp.get("cpu_strict")
        server_config["cpu_strict_batch"] = cp.get("cpu_strict_batch")
        server_config["spec_type"] = cp.get("spec_type", "ngram-simple")
        server_config["spec_ngram_n"] = cp.get("spec_ngram_n")
        server_config["spec_ngram_m"] = cp.get("spec_ngram_m")
        server_config["spec_ngram_min_hits"] = cp.get("spec_ngram_min_hits")
        server_config["draft_max"] = cp.get("draft_max")
        server_config["draft_min"] = cp.get("draft_min")
        server_config["draft_p_min"] = cp.get("draft_p_min")
        if cp.get("lookup_cache_dynamic"):
            server_config["lookup_cache_dynamic"] = cp["lookup_cache_dynamic"]
        # Also load MoE + expert count from MoE phase
        p1a = load_phase_results("moe_combined")
        if p1a:
            moe_cfg = _get_moe_config(p1a)
            server_config.update(moe_cfg)
        src_name = "Compute Audit" if load_phase_results("compute_audit") else "Compute"
        print(f"\n[*] Compute from {src_name}: t={cp.get('threads')}/{cp.get('threads_batch')} "
              f"moe={server_config.get('n_cpu_moe')} experts={server_config.get('expert_used_count', 8)} "
              f"draft={cp.get('draft_max')}")
    else:
        print("\n[!] No compute results — running without compute tuning.")

    # Load memory config (prefer 2b over 2)
    memory_src = load_phase_results("memory_audit") or load_phase_results("memory")
    if memory_src:
        mp = memory_src["best_params"]
        server_config.update(mp)
        src_name = "Memory Audit" if load_phase_results("memory_audit") else "Memory"
        print(f"[*] Memory from {src_name}: {len(mp)} params")
    else:
        print("[!] No memory results — running with naked engine.")

    # Remove None values
    server_config = {k: v for k, v in server_config.items() if v is not None}

    # Start server
    print("\n[*] Starting server with best config...")
    kill_server()
    proc = start_server(server_config)
    if not wait_for_server(proc=proc):
        print("[!] Server failed to start with combined config")
        return None

    # Baseline quality
    print("\n[*] Measuring baseline quality...")
    baseline_score = measure_quality({
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.05,
        "repeat_penalty": 1.0,
    })
    print(f"    Baseline: {baseline_score:.0f}% ({int(baseline_score / 100 * len(QUALITY_TASKS))}/{len(QUALITY_TASKS)} correct)")

    study, remaining, completed = setup_study("quality", n_trials)
    if remaining == 0:
        return study.best_trial.params

    total_trials = completed + remaining
    best_score = baseline_score
    _sbv = _safe_best_value(study)
    if completed > 0 and _sbv is not None:
        best_score = max(best_score, _sbv)

    def objective(trial):
        nonlocal best_score

        # Mirostat overrides temperature/top_p/top_k/min_p — conditional search space
        mirostat = trial.suggest_categorical("mirostat", [0, 1, 2])
        params = {"mirostat": mirostat}

        if mirostat == 0:
            # Standard samplers (only active when mirostat is off)
            params["temperature"] = trial.suggest_float("temperature", 0.0, 1.5)
            params["top_p"] = trial.suggest_float("top_p", 0.5, 1.0)
            params["top_k"] = trial.suggest_int("top_k", 1, 100)
            params["min_p"] = trial.suggest_float("min_p", 0.0, 0.3)
            params["typical_p"] = trial.suggest_float("typical_p", 0.5, 1.0)
            params["top_n_sigma"] = trial.suggest_float("top_n_sigma", -1.0, 3.0)
            params["dynatemp_range"] = trial.suggest_float("dynatemp_range", 0.0, 1.0)
            params["dynatemp_exp"] = trial.suggest_float("dynatemp_exp", 0.5, 2.0)
        else:
            # Mirostat-specific params (only active when mirostat is on)
            params["mirostat_lr"] = trial.suggest_float("mirostat_lr", 0.01, 0.5)
            params["mirostat_ent"] = trial.suggest_float("mirostat_ent", 1.0, 10.0)

        # Penalties and repetition control — always active
        params["repeat_penalty"] = trial.suggest_float("repeat_penalty", 1.0, 1.3)
        params["repeat_last_n"] = trial.suggest_categorical("repeat_last_n", [0, 32, 64, 128, 256])
        params["presence_penalty"] = trial.suggest_float("presence_penalty", 0.0, 0.5)
        params["frequency_penalty"] = trial.suggest_float("frequency_penalty", 0.0, 0.5)

        # XTC and DRY samplers — always active
        params["xtc_probability"] = trial.suggest_float("xtc_probability", 0.0, 0.5)
        params["xtc_threshold"] = trial.suggest_float("xtc_threshold", 0.01, 0.5)
        params["dry_multiplier"] = trial.suggest_float("dry_multiplier", 0.0, 1.0)
        params["dry_base"] = trial.suggest_float("dry_base", 1.0, 3.0)
        params["dry_allowed_length"] = trial.suggest_int("dry_allowed_length", 1, 5)
        params["dry_penalty_last_n"] = trial.suggest_categorical("dry_penalty_last_n", [-1, 0, 64, 128, 256, 512])

        # Adaptive sampling
        params["adaptive_target"] = trial.suggest_float("adaptive_target", 0.0, 1.0)
        params["adaptive_decay"] = trial.suggest_float("adaptive_decay", 0.0, 1.0)

        # Check for duplicate config before running quality eval
        cached = check_duplicate_trial(trial)
        if cached is not None:
            print(f"  Trial {trial.number}: duplicate config — cached score: {cached:.1f}")
            return cached

        score = measure_quality(params)

        marker = ""
        if score > best_score:
            best_score = score
            marker = " *** NEW BEST ***"

        done = trial.number + 1
        pct = done / total_trials * 100
        bar_len = 20
        filled = int(bar_len * done / total_trials)
        bar = "█" * filled + "░" * (bar_len - filled)

        if mirostat == 0:
            detail = (f"temp={params['temperature']:.2f} top_p={params['top_p']:.2f} "
                      f"top_k={params['top_k']:3d} min_p={params['min_p']:.3f}")
        else:
            detail = (f"mirostat={mirostat} lr={params['mirostat_lr']:.3f} "
                      f"ent={params['mirostat_ent']:.1f}")
        print(f"  [{bar}] {pct:5.1f}%  Trial {trial.number:3d}/{total_trials}: {score:5.0f}% | "
              f"{detail}{marker}")

        return score

    print(f"\n[*] Running {remaining} trials (+ {completed} completed)...\n")
    study.optimize(objective, n_trials=remaining, callbacks=[GPStoppingCallback(baseline_score=best_score)], show_progress_bar=False)

    best = study.best_trial
    beat_baseline = best.value > baseline_score

    print(f"\n{'=' * 60}")
    print(f"  Quality — RESULTS")
    print(f"{'=' * 60}")
    print(f"  Baseline:  {baseline_score:.0f}%")
    if beat_baseline:
        print(f"  Best:      {best.value:.0f}% — beats baseline by {best.value - baseline_score:.0f}%")
    else:
        print(f"  Best:      {best.value:.0f}% — BELOW baseline ({baseline_score:.0f}%)")
        print(f"  [!] No trial beat baseline. Using default sampling params.")
    print(f"  Best params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    returned_params = best.params if beat_baseline else {}

    results = {
        "phase": "quality",
        "baseline_score": baseline_score,
        "beat_baseline": beat_baseline,
        "best_score": best.value,
        "best_params": returned_params,
        "eval_tasks": [{"prompt": p, "expected": e} for p, e in QUALITY_TASKS],
        "all_trials": [
            {"number": t.number, "score": t.value, "params": t.params}
            for t in study.trials
        ],
    }
    save_phase_results("quality", results)

    return returned_params


# ============================================================
# Full Pipeline
# ============================================================

def run_full_pipeline(trials_moe=50, trials_p1b=60, trials_p2=60, trials_p1c=60, trials_p2b=60, trials_p3=80, include_ik=True):
    """Run all phases in sequence with result chaining."""
    print("\n" + "=" * 60)
    print("  FULL OPTIMIZATION PIPELINE")
    print(f"  GPU:            GPU offload sweep")
    print(f"  MoE:            MoE thread sweep (experts skipped)")
    print(f"  Compute:        {trials_p1b} trials")
    print(f"  Memory:         {trials_p2} trials")
    print(f"  MoE Audit:      MoE re-validation (best ±2)")
    print(f"  Compute Audit:  {trials_p1c} trials")
    print(f"  Memory Audit:   {trials_p2b} trials")
    print(f"  Quality:        {trials_p3} trials (sampling)")
    if IK_MODE and include_ik:
        print(f"  IK Contrast:    MLA/fused-MoE/RTR/SER sweep")
    print("=" * 60)

    print("\n  Tip: Press Ctrl+C to skip the current phase\n")

    def _run_phase(name, fn):
        """Run a phase, catching Ctrl+C to skip."""
        try:
            return fn()
        except KeyboardInterrupt:
            print(f"\n\n[!] {name} skipped (Ctrl+C)")
            kill_server()
            return None

    # GPU offload
    _run_phase("GPU Offload", phase_gpu_offload)

    # MoE: thread sweep (experts skipped by default)
    best_moe = _run_phase("MoE", lambda: phase_moe(n_trials=trials_moe, include_experts=False))
    if best_moe is None:
        # Check for partial results
        moe_threads_data = load_phase_results("moe")
        p1a = load_phase_results("moe_combined")
        if moe_threads_data and "best_params" in moe_threads_data:
            moe_val = moe_threads_data["best_params"].get("n_cpu_moe", MOE_SWEEP_CENTER)
            best_moe = {"n_cpu_moe": moe_val, "expert_used_count": DEFAULT_EXPERTS}
        elif p1a:
            best_moe = _get_moe_config(p1a)
        else:
            best_moe = _get_moe_config(None)
        print(f"[*] Using MoE config: {best_moe}")

    # Compute (with MoE locked)
    p1b_best = _run_phase("Compute", lambda: phase_compute(n_trials=trials_p1b, phase_name="compute", locked_moe=best_moe))
    if p1b_best is None:
        p1b_data = load_phase_results("compute")
        p1b_best = p1b_data["best_params"] if p1b_data else {}

    # Memory (using Compute results)
    p2_best = _run_phase("Memory", lambda: phase_memory(n_trials=trials_p2, phase_name="memory", base_compute_config={**p1b_best, **best_moe}))
    if p2_best is None:
        p2_data = load_phase_results("memory")
        p2_best = p2_data["best_params"] if p2_data else {}

    # MoE Audit (re-test MoE ±2 with compute + memory locked)
    new_moe = _run_phase("MoE Audit", lambda: phase_moe_revalidate(
        locked_compute=p1b_best, locked_moe=best_moe, base_memory_config=p2_best))
    if new_moe is not None:
        best_moe = {**best_moe, "n_cpu_moe": new_moe}

    # Compute Audit (using Memory + updated MoE), seeded with Compute best
    p1c_best = _run_phase("Compute Audit", lambda: phase_compute(n_trials=trials_p1c, phase_name="compute_audit", base_memory_config=p2_best, seed_params=p1b_best, locked_moe=best_moe))
    if p1c_best is None:
        p1c_data = load_phase_results("compute_audit")
        p1c_best = p1c_data["best_params"] if p1c_data else p1b_best

    # Memory Audit (using Compute Audit + updated MoE), seeded with Memory best
    p2b_result = _run_phase("Memory Audit", lambda: phase_memory(n_trials=trials_p2b, phase_name="memory_audit", base_compute_config={**p1c_best, **best_moe}, seed_params=p2_best))
    p2b_best = p2b_result if p2b_result else p2_best

    # Quality: Sampling (using Compute Audit + Memory Audit)
    _run_phase("Quality", lambda: phase3(n_trials=trials_p3))

    # IK Contrast (optional — runs after llama.cpp pipeline is complete)
    if IK_MODE and include_ik:
        final_compute = p1c_best or p1b_best
        final_memory = p2b_best
        _run_phase("IK Contrast", lambda: phase_ik_contrast(
            locked_compute=final_compute,
            locked_moe=best_moe,
            locked_memory=final_memory,
        ))

    # Final summary
    print("\n" + "=" * 60)
    print("  FULL PIPELINE COMPLETE")
    print("=" * 60)
    for name in ["gpu", "moe_combined", "compute", "memory", "compute_audit", "moe_audit", "memory_audit", "quality", "ik_contrast"]:
        data = load_phase_results(name)
        if data:
            if "best_ngl" in data:
                print(f"  {name:12s}: n_gpu_layers={data['best_ngl']}")
            elif "ik_best_tps" in data:
                print(f"  {name:12s}: IK best {data['ik_best_tps']:.1f} t/s  "
                      f"(gain vs llama: {data.get('ik_gain_vs_llama_pct', 0):+.1f}%)")
            elif "best_tps" in data:
                print(f"  {name:12s}: {data['best_tps']:.1f} t/s")
            elif "best_score" in data:
                print(f"  {name:12s}: {data['best_score']:.0f}% quality")
    print("=" * 60)


# ============================================================
# Interactive Terminal Menu
# ============================================================

def clear_screen():
    os.system("cls" if sys.platform == "win32" else "clear")


def switch_model():
    """Scan models directory for GGUFs and let user pick one."""
    global MODEL, MAX_GPU_LAYERS, DEFAULT_GPU_LAYERS, NAKED_ENGINE, IS_MOE, ARCH
    global EXPERT_OVERRIDE_KEY, DEFAULT_EXPERTS, MAX_EXPERTS

    models_dir = MODEL.parent.parent  # go up from model subdir to models/
    gguf_files = sorted(models_dir.rglob("*.gguf"))
    # Filter out mmproj files (vision projectors, not language models)
    gguf_files = [f for f in gguf_files if "mmproj" not in f.name.lower()
                  and "reranker" not in f.parent.name.lower()
                  and "embedding" not in f.parent.name.lower()]

    if not gguf_files:
        print(f"\n  No GGUF files found in {models_dir}")
        input("\n  Press Enter to continue...")
        return

    print(f"\n  Available models in {models_dir}:\n")
    for i, f in enumerate(gguf_files):
        current = " ← current" if f == MODEL else ""
        size_gb = f.stat().st_size / (1024**3)
        print(f"    [{i+1}] {f.parent.name}/{f.name} ({size_gb:.1f} GB){current}")

    print(f"\n    [0] Enter custom path")
    raw = input("\n  > ").strip()

    if raw == "0":
        path = input("  Path to GGUF: ").strip().strip('"').strip("'")
        if not Path(path).is_file():
            print(f"  File not found: {path}")
            input("\n  Press Enter to continue...")
            return
        new_model = Path(path)
    elif raw.isdigit() and 1 <= int(raw) <= len(gguf_files):
        new_model = gguf_files[int(raw) - 1]
    else:
        return

    # Detect architecture
    print(f"\n  Architecture for {new_model.name}?")
    print("    [1] MoE (Mixture of Experts)")
    print("    [2] Dense")
    arch_choice = input("  > ").strip()
    if arch_choice not in ("1", "2"):
        return

    MODEL = new_model
    _config["model"] = str(MODEL)

    # Update architecture
    if arch_choice == "1":
        key = input("  Expert override key (e.g., qwen35moe.expert_used_count): ").strip()
        default_exp = input("  Default active experts [8]: ").strip()
        max_exp = input("  Max experts [16]: ").strip()
        ARCH = {
            "type": "moe",
            "expert_override_key": key,
            "default_experts": int(default_exp) if default_exp else 8,
            "max_experts": int(max_exp) if max_exp else 16,
        }
        IS_MOE = True
        EXPERT_OVERRIDE_KEY = key
        DEFAULT_EXPERTS = ARCH["default_experts"]
        MAX_EXPERTS = ARCH["max_experts"]
    else:
        ARCH = {"type": "dense"}
        IS_MOE = False
        EXPERT_OVERRIDE_KEY = ""
        DEFAULT_EXPERTS = 8
        MAX_EXPERTS = 16

    _config["architecture"] = ARCH

    # Re-detect layers
    detected = _detect_model_layers(str(MODEL))
    MAX_GPU_LAYERS = detected or 99
    DEFAULT_GPU_LAYERS = MAX_GPU_LAYERS
    NAKED_ENGINE["n_gpu_layers"] = DEFAULT_GPU_LAYERS

    # Reset results dir for new model — each model gets its own folder
    global RESULTS_DIR, LOOKUP_CACHE_FILE, OPTUNA_DB
    model_stem = MODEL.stem.lower().replace(" ", "-")
    RESULTS_DIR = MODEL.parent.parent.parent / "llama-server" / f"optimize-results-{model_stem}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _lc_path = (RESULTS_DIR / "lookup-cache.bin").resolve()
    try:
        _lc_path.parent.mkdir(parents=True, exist_ok=True)
        _lc_path.touch(exist_ok=True)  # pre-create so llama-server can open it
    except Exception:
        pass
    LOOKUP_CACHE_FILE = str(_lc_path)
    OPTUNA_DB = f"sqlite:///{RESULTS_DIR / 'optuna.db'}"
    _config["results_dir"] = str(RESULTS_DIR)

    print(f"\n  Switched to: {MODEL.name}")
    print(f"  Arch: {'MoE' if IS_MOE else 'Dense'} | Layers: {MAX_GPU_LAYERS}")
    print(f"  Results: {RESULTS_DIR}/")
    input("\n  Press Enter to continue...")


def print_header():
    print("=" * 60)
    print("  llama-server Parameter Optimizer")
    print("  GP-Bayesian Coordinate Descent")
    print("=" * 60)
    status = "ONLINE" if is_server_running() else "OFFLINE"
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"  Python: {py_ver}")
    print(f"  Server: {SERVER_URL} [{status}]")
    print(f"  Model:  {MODEL.name}")
    arch_label = f"MoE ({DEFAULT_EXPERTS} experts, {MAX_EXPERTS} max)" if IS_MOE else "Dense"
    print(f"  Arch:   {arch_label}")
    print(f"  GPU:    {DEFAULT_GPU_LAYERS}/{MAX_GPU_LAYERS} layers offloaded")
    print(f"  CPU:    {MAX_THREADS} threads (auto-detected)")
    print(f"  Results: {RESULTS_DIR}/")
    if IK_MODE:
        ik_label = "dual (llama + IK contrast)" if DUAL_SERVER_MODE else "IK-only mode"
        print(f"  IK:     {IK_SERVER.name}  [{ik_label}]")
    else:
        print(f"  IK:     not configured (set IK_LLAMA_SERVER env var to enable)")
    print("=" * 60)


def print_menu():
    print()
    print("  Individual phases:")
    print("  [g]   GPU offload sweep")
    print("  [moe] MoE thread sweep (naked)")
    print("  [ex]  Expert count sweep (optional)")
    print("  [c]   Compute allocation (+ MoE locked)")
    print("  [me]  Memory & throughput (+ Compute best)")
    print("  [mo]  MoE audit (+ Memory best)")
    print("  [ca]  Compute audit (+ MoE Audit best)")
    print("  [ma]  Memory audit (+ Compute Audit best)")
    print("  [s]   Quality / sampling (+ best)")
    if IK_MODE:
        print("  [ik]  IK_llama contrast (MLA/fused-MoE/RTR/SER sweep)")
    print()
    print("  Pipelines:")
    print("  [all] Full pipeline (GPU → MoE → C → ME → MO → CA → MA → Q)")
    print("  [cd]  Coordinate descent (GPU → MoE → C → ME → MO → CA → MA)")
    if IK_MODE:
        print("  [ikall] IK-extended pipeline (adds IK contrast at end)")
    print()
    print("  [v] View past results")
    print("  [m] Switch model")
    print("  [r] Reset (clear DB to start fresh)")
    print("  [q] Quit")
    print()


def _get_moe_config(p1a_results=None):
    """Extract MoE config dict from MoE phase results, with defaults.

    Checks moe_combined results first, then falls back to
    moe thread sweep results if the expert sweep was skipped.
    """
    if not IS_MOE:
        return {}
    # Try combined results first
    if p1a_results and "best_params" in p1a_results:
        bp = p1a_results["best_params"]
        return {
            "n_cpu_moe": bp.get("n_cpu_moe", MOE_SWEEP_CENTER),
            "expert_used_count": bp.get("expert_used_count", DEFAULT_EXPERTS),
        }
    # Fallback: MoE thread sweep completed but expert sweep was skipped
    moe_data = load_phase_results("moe")
    if moe_data and "best_params" in moe_data:
        moe_threads = moe_data["best_params"].get("n_cpu_moe", MOE_SWEEP_CENTER)
        print(f"  [*] Using MoE threads from MoE sweep: {moe_threads} (expert sweep was skipped)")
        return {"n_cpu_moe": moe_threads, "expert_used_count": DEFAULT_EXPERTS}
    return {"n_cpu_moe": MOE_SWEEP_CENTER, "expert_used_count": DEFAULT_EXPERTS}


def ask_trials(phase_label, default):
    """Ask for trial count, return default if user just hits enter."""
    raw = input(f"  {phase_label} trials [{default}]: ").strip()
    if not raw:
        return default
    try:
        n = int(raw)
        return n if n > 0 else default
    except ValueError:
        print(f"  Invalid number, using {default}")
        return default


def reset_db():
    """Delete the Optuna DB and result files so all phases start fresh."""
    db_path = RESULTS_DIR / "optuna.db"
    confirm = input("  Delete all saved trial progress and results? [y/N]: ").strip().lower()
    if confirm == "y":
        if db_path.exists():
            db_path.unlink()
        # Also clean up result JSONs
        for name in ["moe_combined", "moe", "experts", "compute", "memory",
                     "compute_audit", "moe_audit", "memory_audit", "quality", "ik_contrast"]:
            p = RESULTS_DIR / f"{name}_results.json"
            if p.exists():
                p.unlink()
        print("  DB and results deleted. All phases will start fresh.")
    else:
        print("  Cancelled.")
    input("  Press Enter to continue...")


def view_results():
    """Show saved results from previous runs."""
    print()
    for name in ["moe_combined", "moe", "experts", "compute", "memory",
                 "compute_audit", "moe_audit", "memory_audit", "quality", "ik_contrast"]:
        data = load_phase_results(name)
        if data:
            print(f"  --- {name} ---")
            if name == "ik_contrast":
                print(f"  llama.cpp:  {data.get('llama_tps', 0):.1f} t/s")
                print(f"  IK best:    {data.get('ik_best_tps', 0):.1f} t/s  [{data.get('ik_best_label', '')}]")
                print(f"  IK gain:    {data.get('ik_gain_vs_llama_pct', 0):+.1f}% vs llama  "
                      f"/ {data.get('ik_gain_vs_base_pct', 0):+.1f}% vs no-IK-flags")
            elif "best_tps" in data:
                bl = data.get("baseline", {})
                print(f"  Baseline: {bl.get('tps', 0):.1f} t/s | pp: {bl.get('prompt_tps', 0):.0f} t/s | TTFT: {bl.get('ttft', 0):.0f}ms")
                bm = data.get("best_metrics", {})
                print(f"  Best:     {bm.get('tps', data['best_tps']):.1f} t/s | "
                      f"pp: {bm.get('prompt_tps', 0):.0f} t/s | TTFT: {bm.get('ttft', 0):.0f}ms")
                vf = data.get("verified", {})
                if vf:
                    print(f"  Verified: {vf.get('tps', 0):.1f} t/s | pp: {vf.get('prompt_tps', 0):.0f} t/s | TTFT: {vf.get('ttft', 0):.0f}ms")
            elif "best_score" in data:
                print(f"  Baseline: {data.get('baseline_score', '?'):.0f}%")
                print(f"  Best:     {data['best_score']:.0f}%")
            print(f"  Trials:   {len(data.get('all_trials', []))}")
            if name != "ik_contrast":
                print(f"  Params:   {json.dumps(data.get('best_params', {}), indent=2)}")
            print()
        else:
            print(f"  --- {name} --- no results yet")
            print()
    input("  Press Enter to continue...")


def _find_file(pattern_name, extensions, search_hints=None):
    """Try to find a file by searching common locations. Returns path or None."""
    search_hints = search_hints or []
    for hint in search_hints:
        hint = Path(hint).expanduser()
        if hint.is_file():
            return str(hint)
        if hint.is_dir():
            for ext in extensions:
                for f in hint.rglob(f"*{ext}"):
                    return str(f)
    return None


def first_run_setup():
    """Interactive setup wizard for first-time users. Returns config dict."""
    print("=" * 60)
    print("  llama-server Parameter Optimizer — First Run Setup")
    print("=" * 60)
    print()
    print("  This wizard will help you configure the optimizer.")
    print("  Your settings will be saved so you only do this once.")
    print()

    config = {}

    # 1. llama-server path
    print("  [1/5] Path to llama-server executable")
    print("        (e.g., /usr/local/bin/llama-server or C:\\...\\llama-server.exe)")
    while True:
        path = input("        > ").strip().strip('"').strip("'")
        if Path(path).is_file():
            config["server"] = path
            break
        print(f"        File not found: {path}")
        print("        Please enter the full path to the llama-server executable.")

    # 2. Model path
    print()
    print("  [2/5] Path to GGUF model file")
    print("        (e.g., /models/my-model.gguf)")
    while True:
        path = input("        > ").strip().strip('"').strip("'")
        if Path(path).is_file():
            config["model"] = path
            break
        print(f"        File not found: {path}")

    # 3. Chat template
    print()
    print("  [3/5] Path to chat template (.jinja)")
    print("        (press Enter to skip — server will use its default)")
    path = input("        > ").strip().strip('"').strip("'")
    if path and Path(path).is_file():
        config["chat_template"] = path
    else:
        config["chat_template"] = ""

    # 4. Architecture
    print()
    print("  [4/6] Model architecture")
    print("        [1] MoE (Mixture of Experts) — e.g., Qwen 3.5 MoE, Mixtral, DeepSeek")
    print("        [2] Dense — e.g., Llama, Qwen dense, Gemma, Phi")
    while True:
        choice = input("        > ").strip()
        if choice in ("1", "2"):
            break
        print("        Enter 1 or 2.")

    if choice == "1":
        config["architecture"] = {"type": "moe"}
        print()
        print("        GGUF override key for expert count")
        print("        (e.g., qwen35moe.expert_used_count, deepseek2.expert_used_count)")
        print("        Check your model's GGUF metadata if unsure.")
        key = input("        > ").strip()
        config["architecture"]["expert_override_key"] = key

        print()
        print("        Default active experts (how many the model was trained with)")
        default_exp = input("        [8] > ").strip()
        config["architecture"]["default_experts"] = int(default_exp) if default_exp else 8

        print()
        print("        Max experts to sweep")
        max_exp = input("        [16] > ").strip()
        config["architecture"]["max_experts"] = int(max_exp) if max_exp else 16
    else:
        config["architecture"] = {"type": "dense"}

    # 5. GPU offload starting point
    print()
    print("  [5/6] GPU layer offload starting point (-ngl)")
    print("        GPU Offload will automatically sweep all offload levels to find the fastest.")
    print("        This sets the starting default (99 = try full GPU first).")
    print()
    print("        - 99 = start with full GPU (recommended)")
    print("        - 0  = CPU only (no GPU)")
    ngl = input("        [99] > ").strip()
    default_ngl = int(ngl) if ngl else 99
    config["hardware"] = {"default_gpu_layers": default_ngl}

    # 6. Port
    print()
    print("  [6/6] Server port")
    port = input("        [8090] > ").strip()
    config["port"] = int(port) if port else 8090

    # Results dir
    config["results_dir"] = str(Path(config["model"]).parent / "optimize-results")

    # Save config
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    config_path = results_dir / "optimizer-config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print()
    print(f"  Config saved to: {config_path}")
    print("  You can edit this file to change settings later.")
    print()
    input("  Press Enter to start the optimizer...")

    return config


def _needs_setup():
    """Check if the current config points to valid files."""
    return not (Path(_config["server"]).is_file() and Path(_config["model"]).is_file())


def main():
    global _config, LLAMA_SERVER, MODEL, CHAT_TEMPLATE, RESULTS_DIR, LOOKUP_CACHE_FILE
    global OPTUNA_DB, PORT, SERVER_URL
    global ARCH, IS_MOE, EXPERT_OVERRIDE_KEY, DEFAULT_EXPERTS, MAX_EXPERTS
    global HW, MAX_THREADS, MOE_SWEEP_MAX, MOE_SWEEP_CENTER, MAX_GPU_LAYERS, DEFAULT_GPU_LAYERS
    global NAKED_ENGINE

    # First-run setup if config is missing or paths are invalid
    if _needs_setup():
        new_config = first_run_setup()
        # Reload everything with new config
        import copy
        _config = copy.deepcopy(_DEFAULTS)
        for k, v in new_config.items():
            if isinstance(v, dict) and isinstance(_config.get(k), dict):
                _config[k].update(v)
            else:
                _config[k] = v
        # Re-derive all globals
        LLAMA_SERVER = Path(_config["server"])
        MODEL = Path(_config["model"])
        CHAT_TEMPLATE = Path(_config["chat_template"]) if _config.get("chat_template") else Path("")
        RESULTS_DIR = Path(_config["results_dir"])
        _lc_path = (RESULTS_DIR / "lookup-cache.bin").resolve()
        try:
            _lc_path.parent.mkdir(parents=True, exist_ok=True)
            _lc_path.touch(exist_ok=True)  # pre-create so llama-server can open it
        except Exception:
            pass
        LOOKUP_CACHE_FILE = str(_lc_path)
        OPTUNA_DB = f"sqlite:///{RESULTS_DIR / 'optuna.db'}"
        PORT = _config["port"]
        SERVER_URL = f"http://127.0.0.1:{PORT}"
        ARCH = _config["architecture"]
        IS_MOE = ARCH["type"] == "moe"
        EXPERT_OVERRIDE_KEY = ARCH.get("expert_override_key", "")
        DEFAULT_EXPERTS = ARCH.get("default_experts", 8)
        MAX_EXPERTS = ARCH.get("max_experts", 16)
        hw = _config["hardware"]
        if hw["max_threads"] is None:
            hw["max_threads"] = os.cpu_count() or 16
        if hw["moe_sweep_max"] is None:
            hw["moe_sweep_max"] = min(hw["max_threads"] * 2, 40)
        if hw["moe_sweep_center"] is None:
            hw["moe_sweep_center"] = hw["moe_sweep_max"] // 2
        if hw["max_gpu_layers"] is None:
            hw["max_gpu_layers"] = _detect_model_layers(str(MODEL)) or 99
        HW = hw
        MAX_THREADS = hw["max_threads"]
        MOE_SWEEP_MAX = hw["moe_sweep_max"]
        MOE_SWEEP_CENTER = hw["moe_sweep_center"]
        MAX_GPU_LAYERS = hw["max_gpu_layers"]
        DEFAULT_GPU_LAYERS = min(hw.get("default_gpu_layers", 99), MAX_GPU_LAYERS)
        NAKED_ENGINE = {
            "context": 4096,
            "mlock": True,
            "n_gpu_layers": DEFAULT_GPU_LAYERS,
        }

    ensure_results_dir()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    while True:
        clear_screen()
        print_header()
        print_menu()

        choice = input("  > ").strip().lower()

        if choice == "q":
            break
        elif choice == "m":
            switch_model()
        elif choice == "r":
            reset_db()
        elif choice == "v":
            view_results()
        elif choice == "g":
            phase_gpu_offload()
            input("\n  Press Enter to continue...")
        elif choice == "moe":
            phase_moe(include_experts=False)
            input("\n  Press Enter to continue...")
        elif choice == "ex":
            p1a = load_phase_results("moe_combined")
            moe = _get_moe_config(p1a)
            phase_experts(locked_moe_threads=moe["n_cpu_moe"])
            input("\n  Press Enter to continue...")
        elif choice == "c":
            n = ask_trials("Compute", 60)
            p1a = load_phase_results("moe_combined")
            moe = _get_moe_config(p1a)
            phase_compute(n_trials=n, phase_name="compute", locked_moe=moe)
            input("\n  Press Enter to continue...")
        elif choice == "me":
            n = ask_trials("Memory", 60)
            p1a = load_phase_results("moe_combined")
            p1b = load_phase_results("compute")
            moe = _get_moe_config(p1a)
            base = {**p1b["best_params"], **moe} if p1b else None
            phase_memory(n_trials=n, phase_name="memory", base_compute_config=base)
            input("\n  Press Enter to continue...")
        elif choice == "ca":
            n = ask_trials("Compute Audit", 60)
            p1a = load_phase_results("moe_combined")
            p1d = load_phase_results("moe_audit")
            p2 = load_phase_results("memory")
            p1b = load_phase_results("compute")
            moe = _get_moe_config(p1a)
            # Use MoE Audit result if available
            if p1d and "best_params" in p1d:
                moe["n_cpu_moe"] = p1d["best_params"]["n_cpu_moe"]
            base = p2["best_params"] if p2 else None
            seed = p1b["best_params"] if p1b else None
            if base is None:
                print("  [!] No Memory results found. Run Memory first.")
                input("  Press Enter to continue...")
            else:
                phase_compute(n_trials=n, phase_name="compute_audit", base_memory_config=base, seed_params=seed, locked_moe=moe)
                input("\n  Press Enter to continue...")
        elif choice == "mo":
            p1a = load_phase_results("moe_combined")
            p1b = load_phase_results("compute")
            p2 = load_phase_results("memory")
            moe = _get_moe_config(p1a)
            compute = p1b["best_params"] if p1b else None
            mem = p2["best_params"] if p2 else None
            if compute is None:
                print("  [!] No Compute results found. Run Compute first.")
                input("  Press Enter to continue...")
            else:
                phase_moe_revalidate(locked_compute=compute, locked_moe=moe, base_memory_config=mem)
                input("\n  Press Enter to continue...")
        elif choice == "ma":
            n = ask_trials("Memory Audit", 60)
            p1a = load_phase_results("moe_combined")
            p1c = load_phase_results("compute_audit")
            p1d = load_phase_results("moe_audit")
            p2 = load_phase_results("memory")
            moe = _get_moe_config(p1a)
            # Use MoE Audit result if available, otherwise fall back to MoE phase
            if p1d and "best_params" in p1d:
                moe["n_cpu_moe"] = p1d["best_params"]["n_cpu_moe"]
            base = {**p1c["best_params"], **moe} if p1c else None
            seed = p2["best_params"] if p2 else None
            if base is None:
                print("  [!] No Compute Audit results found. Run Compute Audit first.")
                input("  Press Enter to continue...")
            else:
                phase_memory(n_trials=n, phase_name="memory_audit", base_compute_config=base, seed_params=seed)
                input("\n  Press Enter to continue...")
        elif choice == "s":
            n = ask_trials("Quality", 80)
            phase3(n_trials=n)
            input("\n  Press Enter to continue...")
        elif choice == "ik":
            if not IK_MODE:
                print("  [!] IK_llama.cpp not configured. Set IK_LLAMA_SERVER env var.")
                input("  Press Enter to continue...")
            else:
                p1a = load_phase_results("moe_combined")
                p1b = load_phase_results("compute_audit") or load_phase_results("compute")
                p2 = load_phase_results("memory_audit") or load_phase_results("memory")
                moe = _get_moe_config(p1a)
                compute = p1b["best_params"] if p1b else None
                mem = p2["best_params"] if p2 else None
                phase_ik_contrast(locked_compute=compute, locked_moe=moe, locked_memory=mem)
                input("\n  Press Enter to continue...")
        elif choice == "cd":
            n_1b = ask_trials("Compute", 60)
            n_2 = ask_trials("Memory", 60)
            n_1c = ask_trials("Compute Audit", 60)
            n_2b = ask_trials("Memory Audit", 60)
            run_full_pipeline(50, n_1b, n_2, n_1c, n_2b, 0, include_ik=False)  # 0 trials = skip Quality
            input("\n  Press Enter to continue...")
        elif choice in ("all", "12345"):
            n_1b = ask_trials("Compute", 60)
            n_2 = ask_trials("Memory", 60)
            n_1c = ask_trials("Compute Audit", 60)
            n_2b = ask_trials("Memory Audit", 60)
            n_3 = ask_trials("Quality", 80)
            run_full_pipeline(50, n_1b, n_2, n_1c, n_2b, n_3, include_ik=False)
            input("\n  Press Enter to continue...")
        elif choice == "ikall":
            if not IK_MODE:
                print("  [!] IK_llama.cpp not configured. Set IK_LLAMA_SERVER env var.")
                input("  Press Enter to continue...")
            else:
                n_1b = ask_trials("Compute", 60)
                n_2 = ask_trials("Memory", 60)
                n_1c = ask_trials("Compute Audit", 60)
                n_2b = ask_trials("Memory Audit", 60)
                n_3 = ask_trials("Quality", 80)
                run_full_pipeline(50, n_1b, n_2, n_1c, n_2b, n_3, include_ik=True)
                input("\n  Press Enter to continue...")
        else:
            print("  Invalid choice.")
            time.sleep(1)


if __name__ == "__main__":
    _bootstrap_from_config()
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Optimization aborted by user.")
    finally:
        print("\n[*] Cleaning up...")
        kill_server()
        print("    Done.")
