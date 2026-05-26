# Llama Optimizer

**Automatically find the fastest possible settings for running large language models on your machine.**

GP-Bayesian multi-phase optimization for `llama-server` (llama.cpp) and `ik_llama-server` (ik_llama.cpp), with GPU topology sweep, context ceiling detection, MTP (Multi-Token Prediction) draft sweep, IK_llama.cpp contrast benchmarking, and a sortable HTML report enriched with HuggingFace metadata.

**Repository:** https://github.com/VykosX/Llama-Optimizer

---

## Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [What This Does](#what-this-does)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [File Overview](#file-overview)
6. [Running the Optimizer](#running-the-optimizer)
   - [Basic Usage](#basic-usage)
   - [All Command-Line Switches](#all-command-line-switches)
   - [Preset Reference](#preset-reference)
   - [IK_llama.cpp Support](#ik_llamacpp-support)
   - [MTP Support](#mtp-multi-token-prediction-support)
7. [How It All Works — Phase by Phase](#how-it-all-works--phase-by-phase)
   - [Topology Sweep](#phase-0-topology-sweep)
   - [Context Ceiling Sweep](#phase-05-context-ceiling-sweep)
   - [GPU Offload](#phase-1-gpu-offload)
   - [MoE Thread Sweep](#phase-2-moe-thread-sweep)
   - [Expert Count Sweep](#phase-3-expert-count-sweep)
   - [Compute Allocation](#phase-4-compute-allocation)
   - [Memory & Throughput](#phase-5-memory--throughput)
   - [Audit Phases](#phase-6-audit-phases)
   - [Quality / Sampling](#phase-7-quality--sampling)
   - [IK Contrast](#phase-8-ik_llamacpp-contrast)
   - [MTP Draft Sweep](#phase-9-mtp-draft-sweep)
8. [The Scoring System](#the-scoring-system)
9. [The GP-Bayesian Optimizer](#the-gp-bayesian-optimizer)
10. [HTML Report](#html-report)
11. [Understanding Your Results](#understanding-your-results)
12. [Troubleshooting](#troubleshooting)
13. [Example Report Output](#example-report-output)

---

## Quick Start Guide

> **New to this? Start here.** This section explains what the tool does and how to run it in plain language, no deep LLM knowledge required.

### What problem does this solve?

When you run a large language model locally using llama.cpp, there are dozens of settings that affect how fast the model generates text — things like how many layers to put on your GPU, how many CPU threads to use, what batch sizes to use, whether to use flash attention, what kind of KV cache quantization to use, whether to enable MTP draft prediction, and more. Getting these wrong can mean your model runs at 3 tokens per second instead of 15. **This tool automatically tests thousands of combinations and finds the fastest settings for your specific hardware and model.**

### What you need before starting

- **Windows or Linux** with Python 3.10+
- **A working `llama-server.exe`** (or `llama-server` on Linux) from [llama.cpp](https://github.com/ggerganov/llama.cpp/releases)
- **One or more `.gguf` model files** you want to optimize
- **NVIDIA GPU(s)** with CUDA (AMD may work but is untested)
- Optional: `ik_llama-server` from [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) for additional speed gains on hybrid CPU/GPU setups

### Two ways to run — pick whichever suits you

**Option A — Using command-line switches only (no environment variables needed):**

```powershell
# Windows PowerShell
python batch_runner.py `
  --llama-server "C:\path\to\llama-server.exe" `
  --models-dir "C:\path\to\your\models" `
  --preset standard `
  --topo-sweep `
  --html-report `
  --interactive
```

```bash
# Linux / macOS
python batch_runner.py \
  --llama-server /usr/local/bin/llama-server \
  --models-dir /home/user/models \
  --preset standard \
  --topo-sweep \
  --html-report \
  --interactive
```

**Option B — Using environment variables (convenient for repeated runs):**

```powershell
# Windows PowerShell
$env:LLAMA_SERVER = "C:\path\to\llama-server.exe"
$env:LLM_OPT_MODELS_DIR = "C:\path\to\your\models"
python batch_runner.py --preset standard --topo-sweep --html-report --interactive
```

```bash
# Linux / macOS
export LLAMA_SERVER="/usr/local/bin/llama-server"
export LLM_OPT_MODELS_DIR="/home/user/models"
python batch_runner.py --preset standard --topo-sweep --html-report --interactive
```

Both methods work identically. Command-line switches always override environment variables when both are set.

Either run will:
1. Find all `.gguf` files in your models directory
2. Run a GPU topology test to figure out the best GPU configuration for each model
3. Run 60 compute trials + 60 memory trials to find optimal settings
4. Generate an HTML report you can open in any browser

When it finishes, open `batch_reports/report_latest.html` in your browser.

### With IK_llama.cpp (optional, for extra speed on hybrid setups)

```powershell
python batch_runner.py `
  --llama-server "C:\path\to\llama-server.exe" `
  --ik-llama-server "C:\path\to\ik_llama-server.exe" `
  --models-dir "C:\path\to\your\models" `
  --preset ik --topo-sweep --html-report
```

### With MTP draft prediction (for Qwen3.5/3.6, DeepSeek V3/R1, Gemma 4)

MTP is **automatically detected** from GGUF metadata — no manual configuration needed for supported models. Use the `mtp` preset to include the MTP sweep, or `--force-mtp` to test any model regardless of detection:

```powershell
# Auto-detected MTP (Qwen3.5, DeepSeek V3, Gemma 4, etc.)
python batch_runner.py --llama-server "C:\llama-server.exe" --preset mtp --topo-sweep

# Force MTP even when not detected (e.g. custom quants that omit the metadata key)
python batch_runner.py --llama-server "C:\llama-server.exe" --preset mtp --force-mtp --filter "Qwen3.6"
```

### What the results look like

The HTML report shows a sortable table with every model you tested. Key columns:

| Column | Meaning |
|--------|---------|
| **Best t/s** | Fastest tokens per second found by the optimizer |
| **Stock t/s** | Speed with completely default settings (no optimization) |
| **Gain %** | How much faster the optimizer made it |
| **MTP t/s** | Speed with MTP draft prediction enabled (if supported) |
| **MTP gain** | How much MTP added on top of the optimized baseline |
| **IK t/s** | Speed with ik_llama.cpp (if configured) |
| **IK gain** | How much faster ik_llama.cpp is vs vanilla llama.cpp |
| **Case** | How the model fits on your GPU(s) — A=both GPUs, B=one GPU, C=split, D=needs RAM |
| **ctx GPU** | Maximum context length that fits fully in VRAM |
| **ctx RAM** | Maximum context achievable with KV cache spilling to RAM |

### How long does it take?

| Preset | Time per model | What it tests |
|--------|---------------|---------------|
| `fast` | ~25 min | Quick sweep + diagnostics |
| `standard` | ~1–2 hours | Full compute + memory optimization |
| `mtp` | ~2–3 hours | Standard + MTP draft sweep |
| `ik` | ~2–3 hours | Standard + IK_llama.cpp contrast |
| `thorough` | ~3–4 hours | Full optimization + re-validation audits |
| `full` | ~4–5 hours | Everything including sampling params |
| `full_plus` | ~5–6 hours | All phases: audits + quality + IK + MTP |

Press **Ctrl+C** at any time to skip the current phase and move to the next one. Results are saved after every trial so you can always resume with `--resume`.

---

## What This Does

Llama Optimizer is a multi-phase automated benchmarking and optimization system for locally-run large language models. It works by:

**Characterizing your hardware** — Before optimization begins, a topology sweep classifies each model into one of four cases based on how it fits in your VRAM, tests every relevant GPU configuration (single GPU, split across GPUs, NUMA policies), and binary-searches the maximum stable context window. This prevents the optimizer from wasting time on configurations that will OOM.

**Finding optimal inference parameters** — Using Gaussian Process Bayesian optimization (not random search, not grid search), it intelligently explores the parameter space for each model, learning from every trial to propose configurations most likely to improve performance. Parameters explored include thread counts, batch sizes, KV cache quantization, flash attention, speculative decoding, CPU/GPU MoE routing split, and more.

**Sweeping MTP draft prediction** — For models with built-in Multi-Token Prediction heads (Qwen3.5/3.6, DeepSeek V3/R1, Gemma 4), the optimizer runs a dedicated 6-step sweep: draft depth (n_max 1–3), acceptance probability, micro-batch size, and minimum draft threshold. MTP uses auxiliary heads baked into the GGUF itself — no separate draft model needed. Dense models typically see 20–70% gains; MoE models 5–25%.

**Benchmarking IK_llama.cpp** — If you have `ik_llama-server` installed, the optimizer runs a structured contrast benchmark comparing vanilla llama.cpp against IK's exclusive features: MLA attention (critical for DeepSeek-family models), fused MoE routing (20–80% gains on MoE models), run-time quant repacking for CPU execution (50–80% gains for CPU-heavy configs), and smart expert reduction.

**Generating an actionable HTML report** — The final report merges local benchmark results with HuggingFace model metadata (benchmarks, parameters, license, downloads), provides quantization recommendations for alternative quants that would better suit your hardware, and renders everything as a sortable, filterable, self-contained HTML file with no external dependencies.

---

## Requirements

### Software

- Python 3.10 or newer
- `llama-server` from [llama.cpp](https://github.com/ggerganov/llama.cpp/releases) (prebuilt releases available for Windows/Linux/Mac)
- Python packages (auto-installed on first run if missing):
  - `requests`, `optuna`, `numpy`, `scipy`, `scikit-learn`
  - `psutil` (RAM monitoring)
  - `pynvml` (GPU monitoring, optional but recommended)

### Hardware

- **GPU:** NVIDIA GPU with CUDA 11.8+ recommended. 8+ GB VRAM minimum. Works without GPU (CPU-only) but is much slower.
- **RAM:** 16 GB minimum. 64+ GB recommended for large models that need CPU offload.
- **CPU:** Any x86-64. More cores help for CPU-offloaded MoE models. AVX2/AVX-512 support helps significantly for run-time repacking with ik_llama.cpp.

### Models

Any `.gguf` model file compatible with llama.cpp. The optimizer handles:
- Single-file models (`model.gguf`)
- Multi-shard models (`model-00001-of-00004.gguf` — only shard 1 needed, llama-server finds the rest automatically)
- Dense models (Llama, Mistral, Qwen, Phi, Gemma, etc.)
- MoE models (Mixtral, DeepSeek, Qwen MoE, MiniMax, etc.)
- Hybrid/SSM models (Mamba, Jamba)
- MTP-capable models (Qwen3.5/3.6, DeepSeek V3/R1, Gemma 4 — detected automatically)

---

## Installation

```bash
git clone https://github.com/VykosX/Llama-Optimizer
cd Llama-Optimizer
pip install requests optuna numpy scipy scikit-learn psutil pynvml
```

No other installation steps required. All files run directly from the cloned directory. Dependencies are also auto-installed on first run if missing.

---

## File Overview

```
Llama-Optimizer/
├── batch_runner.py         Entry point — run this to optimize your models
├── optimizer_adapter.py    Bridge: translates RunConfig → optimize.py phase calls
├── optimize.py             Core GP-Bayesian optimizer with all phase logic
├── sweep_engine.py         GPU topology sweep + context ceiling sweep
├── model_utils.py          GGUF metadata reader, MTP detection, GPU/RAM helpers
├── generate_report.py      HTML report generator with HuggingFace integration
├── test_benchmarks.py      Standalone script to test HF benchmark fetching
├── results/                Per-model optimization results (auto-created)
├── batch_reports/          Batch summary reports (auto-created)
└── logs/                   Run logs with timestamps (auto-created)
```

| File | Lines | Role |
|------|-------|------|
| `batch_runner.py` | ~870 | Entry point, batch loop, CLI args, report generation |
| `optimizer_adapter.py` | ~320 | Preset system, phase routing, IK/MTP server wiring |
| `optimize.py` | ~3900 | All optimization phases, GP sampler, scoring, MTP/IK phases |
| `sweep_engine.py` | ~3000 | Topo sweep, ctx sweep, GGUF binary reader |
| `model_utils.py` | ~640 | Shared utilities, MTP detection, quant recommendations |
| `generate_report.py` | ~3300 | Self-contained HTML report with JS/CSS, IK/MTP columns |

---

## Running the Optimizer

### Basic Usage

```powershell
# Minimal — specify everything on the command line, no env vars needed
python batch_runner.py `
  --llama-server "C:\path\to\llama-server.exe" `
  --models-dir "C:\path\to\models" `
  --preset standard --topo-sweep --html-report

# Resume interrupted run (skips models that already have results)
python batch_runner.py --preset standard --topo-sweep --resume

# Test a single model by name
python batch_runner.py --filter "Qwen3.6-27B" --preset standard --topo-sweep

# Report only (no benchmarking — just regenerate the HTML)
python batch_runner.py --report-only --html-report

# IK_llama.cpp comparison run
python batch_runner.py `
  --llama-server "C:\path\to\llama-server.exe" `
  --ik-llama-server "C:\path\to\ik_llama-server.exe" `
  --preset ik --topo-sweep --html-report

# MTP sweep — auto-detected for supported models
python batch_runner.py --preset mtp --topo-sweep --html-report

# Force MTP even if not detected in GGUF metadata
python batch_runner.py --preset mtp --force-mtp --filter "PRISM-PRO"

# Everything at once — IK + MTP + quality sampling
python batch_runner.py `
  --llama-server "C:\path\to\llama-server.exe" `
  --ik-llama-server "C:\path\to\ik_llama-server.exe" `
  --preset full_plus --topo-sweep --html-report
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `LLAMA_SERVER` | Path to `llama-server` binary |
| `IK_LLAMA_SERVER` | Path to `ik_llama-server` binary |
| `LLM_OPT_MODELS_DIR` | Default models directory |
| `LLM_OPT_PORT` | Server port (default: 8090) |
| `LLM_OPT_TIMEOUT` | Per-model timeout in seconds (default: 5400) |
| `LLM_OPT_TRIAL_TIMEOUT` | Per-trial timeout in seconds (default: 360) |
| `LLM_OPT_STARTUP_TIMEOUT` | Server startup timeout (auto-scaled by model size if not set) |
| `HF_TOKEN` | HuggingFace read token for benchmark data in reports |
| `GPU0_VRAM_GB` | Override detected GPU0 VRAM in GB (useful when pynvml unavailable) |
| `GPU1_VRAM_GB` | Override detected GPU1 VRAM in GB |

### All Command-Line Switches

#### Path Options

| Switch | Default | Description |
|--------|---------|-------------|
| `--models-dir PATH` | `./models` or `LLM_OPT_MODELS_DIR` | Directory to scan for `.gguf` files (recursive) |
| `--llama-server PATH` | Auto-detected or `LLAMA_SERVER` | Path to `llama-server` binary |
| `--ik-llama-server PATH` | Auto-detected or `IK_LLAMA_SERVER` | Path to `ik_llama-server` binary |
| `--results-base PATH` | `./results` | Root directory for per-model results |
| `--reports-dir PATH` | `./batch_reports` | Directory for batch summary reports |
| `--logs-dir PATH` | `./logs` | Directory for run log files (timestamped per run) |

#### Model Selection

| Switch | Description |
|--------|-------------|
| `--filter TEXT` | Only process models whose filename contains TEXT (case-insensitive) |
| `--resume` | Skip models that already have complete optimization results |
| `--retry [first\|last]` | Retry previously failed models. `first` = retry before new models (default), `last` = retry after |
| `--dry-run` | List models that would be processed, then exit without running anything |

#### Preset and Phase Control

| Switch | Description |
|--------|-------------|
| `--preset NAME` | Optimization preset (see [Preset Reference](#preset-reference)) |
| `--phases PHASE...` | Override preset phase list entirely, e.g. `--phases compute memory mtp_spec` |
| `--skip-phases [PHASE...]` | Skip specific phases. Bare `--skip-phases` (no args) skips everything not in `--rerun-phases` |
| `--rerun-phases PHASE...` | Force re-run named phases even if results exist, e.g. `--rerun-phases mtp_spec ik_contrast` |
| `--trials PHASE=N...` | Override trial counts per phase, e.g. `--trials compute=80 memory=80` |

#### Run Mode

| Switch | Default | Description |
|--------|---------|-------------|
| `--timeout SECONDS` | 5400 (90 min) | Per-model hard timeout. Model is abandoned if exceeded. |
| `--trial-timeout SECONDS` | 360 (6 min) | Per-trial hard timeout. Trial marked failed if exceeded. |
| `--port PORT` | 8090 | Port for llama-server HTTP API |
| `--force-mtp` | — | Force MTP draft sweep even when not detected in GGUF metadata |
| `--report-only` | — | Regenerate HTML report from existing results, no benchmarking |
| `--html-report` | — | Generate HTML report after batch completes |
| `--no-hf` | — | Skip HuggingFace metadata fetch (faster, works offline) |
| `--refresh-hf` | — | Force re-fetch all HF metadata, ignoring the 7-day cache |
| `--no-log` | — | Disable automatic timestamped log file in `./logs/` |
| `--verbose` | — | Show live llama-server loading output (layer counts, CUDA init, etc.) |
| `--interactive` | — | Pause 5 seconds between models; press `n` to stop cleanly after the current model finishes |

#### Topology Sweep

| Switch | Default | Description |
|--------|---------|-------------|
| `--topo-sweep` | — | Run GPU topology benchmark before optimizer |
| `--topo-only` | — | Run topology sweep only, skip optimizer entirely |
| `--topo-runs N` | 2 | Benchmark runs per topology scenario (more = more stable, slower) |
| `--gpu-filter SCENARIO...` | — | Only test specific topology scenarios by ID (e.g. `gpu0_only split_prop`) |
| `--skip-gpu GPU...` | — | Skip GPU by index (0, 1) or name fragment (e.g. `--skip-gpu 5060` or `--skip-gpu 1`) |
| `--force-numa` | — | Force NUMA policy tests even for models that fit in single GPU |

#### Context Ceiling Sweep

| Switch | Description |
|--------|-------------|
| `--ctx-sweep` | Binary-search the maximum stable context length after topology sweep |
| `--ctx-only` | Run context sweep only, skip topology and optimizer |
| `--skip-ctx-b` | Skip RAM-spill context tests (GPU ceiling only, faster) |

#### Hardware Overrides

| Switch | Description |
|--------|-------------|
| `--gpu0-vram FLOAT` | Override detected GPU0 VRAM in GB |
| `--gpu1-vram FLOAT` | Override detected GPU1 VRAM in GB |

### Preset Reference

| Preset | Phases | Trials | Est. Time/Model | Best For |
|--------|--------|--------|-----------------|----------|
| `fast` | binary_screen → fast_gpu → fast_moe → compute → memory → integrity → reasoning | compute=30, memory=25 | ~25 min | Quick first pass, large model collections |
| `standard` | gpu → moe → compute → memory | compute=60, memory=60 | ~1–2h | General purpose, good balance of speed/thoroughness |
| `thorough` | gpu → moe → compute → memory → moe_audit → compute_audit → memory_audit | 60 each | ~3–4h | Final optimization before daily use |
| `full` | All thorough phases + quality/sampling | 60 each + quality=80 | ~4–5h | Complete optimization including sampling params |
| `moe_deep` | All thorough + expert count sweep | 60 each | ~4h | MoE models where expert count matters for quality |
| `ik` | gpu → moe → compute → memory → ik_contrast | compute=60, memory=60 | ~2–3h | When IK_llama.cpp is configured |
| `ik_thorough` | Full thorough phases + ik_contrast | 60 each | ~4–5h | Most complete IK comparison |
| `mtp` | gpu → moe → compute → memory → mtp_spec | compute=60, memory=60 | ~2–3h | MTP-capable models (Qwen3.5/3.6, DeepSeek V3/R1, Gemma 4) |
| `mtp_thorough` | Full thorough phases + mtp_spec | 60 each | ~4–5h | Full optimization + thorough MTP sweep |
| `full_plus` | All thorough + quality + ik_contrast + mtp_spec | 60+80 each | ~5–6h | Complete: every phase, best results |

Override trial counts for any preset: `--trials compute=40 memory=40` halves the trials for a faster run while keeping the same phase structure.

### IK_llama.cpp Support

[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) is a fork of llama.cpp with additional optimizations, particularly valuable for hybrid CPU/GPU setups with MoE models:

| IK Feature | Flag | Benefit |
|-----------|------|---------|
| MLA Attention | `-mla 2` | 50% KV cache reduction for DeepSeek-architecture models with MLA tensors |
| Fused MoE | `-fmoe` | 20–80% faster expert routing on GPU-resident MoE models |
| Run-time Repack | `-rtr` | 50–80% faster CPU execution via AVX-512 optimized tensor layouts |
| Attention Max Batch | `-amb N` | Reduces K×Q compute buffer pressure, improves long-context throughput |
| Smart Expert Reduction | `-ser N,1` | Drop N experts per token to trade quality for speed |

**Setup — three ways, pick one:**

```powershell
# Option 1: environment variable (recommended for daily use)
$env:IK_LLAMA_SERVER = "G:\Tools\ik_llama-server.exe"

# Option 2: CLI flag (for one-off runs)
python batch_runner.py --ik-llama-server "G:\Tools\ik_llama-server.exe" --preset ik

# Option 3: auto-detection — place ik_llama-server.exe next to batch_runner.py
```

**Dual mode vs IK-only mode:**
- **Dual mode** (both `LLAMA_SERVER` and `IK_LLAMA_SERVER` set): All optimization phases use vanilla llama.cpp for fair, consistent results. Then the `ik_contrast` phase runs a head-to-head comparison, starting from the best vanilla config. This gives you the fairest possible apples-to-apples comparison.
- **IK-only mode** (only `IK_LLAMA_SERVER` set, no `LLAMA_SERVER`): All phases run automatically with ik_llama-server. The IK flags (MLA, fused MoE, RTR) are applied globally throughout optimization. Best when you only have the IK build.

The `ik_contrast` phase runs 6 structured steps:
1. Vanilla llama.cpp baseline with best found config (3-run median)
2. IK same config, no IK flags (isolates build-level difference)
3. IK full feature pack: MLA mode 2 + fused MoE + run-time repack + amb=512
4. attn_max_batch sweep: 128 / 256 / 512 / 1024
5. SER sweep (MoE only): smart expert reduction at 7,1 / 6,1 / 5,1
6. MLA mode sweep (MoE only): mode 2 (CPU+GPU) vs mode 3 (CPU-only v2)

### MTP (Multi-Token Prediction) Support

MTP is a form of speculative decoding where the prediction heads are baked directly into the GGUF itself — no separate draft model, no extra VRAM for a second model. A single forward pass produces the main token plus N draft candidates which are verified in parallel. When drafts are accepted, you get multiple tokens per forward pass.

**Supported architectures:**

| Architecture | GGUF metadata key | Representative models |
|-------------|------------------|----------------------|
| `qwen35` | `qwen35.nextn_predict_layers` | Qwen3.5, Qwen3.6, Qwen3.6-27B-PRISM-PRO |
| `deepseek3` | `deepseek3.nextn_predict_layers` | DeepSeek V3, DeepSeek V3-0324 |
| `deepseek2` | `deepseek2.nextn_predict_layers` | DeepSeek R1 |
| `gemma4` | `gemma4.nextn_predict_layers` | Gemma 4 (all sizes) |

**Detection hierarchy (most to least reliable):**

1. **GGUF metadata** (confidence: high) — reads `{arch}.nextn_predict_layers` directly from the binary file header. If this key is present and non-zero, MTP is definitively confirmed. This is the primary signal and the most reliable.
2. **Filename pattern** (confidence: medium) — scans the filename for `-MTP-`, `.mtp.`, `-mtp.gguf` etc. Useful for community uploads that include MTP in the name.
3. **Architecture hint** (confidence: low) — model family is MTP-capable but the key was not found. Reports `source=arch_hint`. Does **not** auto-enable MTP; you must use `--force-mtp` to proceed.

**When to use `--force-mtp`:**
- The GGUF conversion omitted the `nextn_predict_layers` metadata key even though the model has MTP heads (common with some community quantizations)
- You want to test whether any model benefits from the MTP spec flags
- You're testing a model like `Qwen3.6-27B-PRISM-PRO-DQ` whose filename doesn't contain "MTP" but whose architecture (qwen35) is MTP-capable — check with `detect_mtp()` first, then use `--force-mtp` if detection returns `arch_hint`

**Checking a model manually:**
```bash
python -c "
from model_utils import detect_mtp
from pathlib import Path
result = detect_mtp(Path('your_model.gguf'))
print(result)
"
# Example output:
# {'has_mtp': True, 'mtp_layers': 1, 'source': 'metadata', 'arch': 'qwen35', 'confidence': 'high'}
```

**The MTP sweep — 6 steps in detail:**

The sweep runs after all other optimization phases, using the best config found by compute + memory phases as its starting point. It always measures a no-MTP baseline first so the gain is comparable.

| Step | What it tests | Why |
|------|--------------|-----|
| 1 — Baseline | Best config without MTP, 3-run median | Reference point for all gain measurements |
| 2 — Spec-type probe | Tests `--spec-type mtp` vs `--spec-type draft-mtp` | Different llama.cpp build versions use different flag names; we detect which works |
| 3 — Draft depth scan | n_max = 1, 2, 3 | How many draft tokens to predict per step. More = higher potential gain, higher rejection risk |
| 4 — p_min sweep | 0.0 / 0.30 / 0.50 / 0.70 / 0.85 / 0.95 on best n_max | Acceptance probability threshold. Higher = fewer but better drafts. Sweet spot is usually 0.5–0.85 |
| 5 — ubatch re-test | 256 / 512 / 1024 on best n_max + p_min | MTP verification batches N draft tokens in one pass — larger ubatch benefits this operation |
| 6 — n_min test | 0 vs 1 minimum accepted drafts | Whether to force at least one draft token to always be accepted |
| Final | 5-run median validation on overall winner | Stability confirmation before saving results |

**Interpreting MTP gains:**
- **+20–70%:** Strong MTP signal. Highly recommended for daily use. Typical for dense models with well-trained MTP heads.
- **+5–20%:** Moderate gain. Worth enabling. Typical for MoE models where expert routing already reduces per-step compute.
- **0–5%:** Marginal. May not be worth the configuration complexity.
- **Negative:** MTP was counterproductive. Usually means near-full VRAM (MTP auxiliary heads need 2–5% extra), or the model was not trained with strong MTP objectives.

---

## How It All Works — Phase by Phase

The optimizer runs phases sequentially, with each phase seeding the next from its best result. This is coordinate descent: optimize one group of parameters at a time with everything else locked, then rotate to the next group.

```
Topology Sweep ──► Context Ceiling Sweep
                         │
                         ▼
                   GPU Offload (Phase 1, dense models only)
                         │
                         ▼
                   MoE Thread Sweep (Phase 2, MoE models only)
                         │
                         ▼
                   Expert Count (Phase 3, optional, MoE only)
                         │
                         ▼
                   Compute Allocation (Phase 4)  ◄── GP-Bayesian
                         │
                         ▼
                   Memory & Throughput (Phase 5)  ◄── GP-Bayesian
                         │
                         ▼
            MoE Audit ─► Compute Audit ─► Memory Audit (re-validation)
                         │
                         ▼
                   Quality / Sampling (Phase 7)  ◄── GP-Bayesian
                         │
                         ├──► IK_llama Contrast (Phase 8, if IK configured)
                         │
                         └──► MTP Draft Sweep (Phase 9, if MTP detected or --force-mtp)
```

### Phase 0: Topology Sweep

Before any optimization, the topology sweep characterizes how each model relates to your hardware.

**Model classification (Cases A–D):**

Every model is classified into one of four cases based on its estimated VRAM requirement (file size + 2–5 GB overhead scaled by model size for KV cache and compute buffers):

| Case | Condition | Behavior |
|------|-----------|----------|
| A | Fits in *both* GPUs independently | Tests each GPU alone, picks the faster one |
| B | Fits in GPU0 only (largest GPU) | Tests GPU0 only |
| C | Requires *combined* VRAM of both GPUs | Tests four split strategies (see below) |
| D | Exceeds combined VRAM (needs CPU offload) | Binary-searches max GPU layers, tests NUMA policies |

**Case C split strategies:**

When a model needs both GPUs combined, four tensor split ratios are tested. The `tensor_split` values are ordered by physical CUDA device index (PCIe bus order via `CUDA_DEVICE_ORDER=PCI_BUS_ID`), not by our sorted GPU ranking:

- `split_prop` — proportional by VRAM (e.g., 60/40 for 24/16 GB GPUs)
- `split_equal` — 50/50
- `split_g0heavy` — 80/20 favouring the larger card
- `split_kv_aware` — shifts 15% of the main GPU's tensor weight to other cards, compensating for KV cache overhead on the main GPU

Each scenario is benchmarked with 2–3 runs (configurable via `--topo-runs`). The winner is used as the fixed topology for all subsequent optimization phases.

**Case D — GPU layer binary search:**

For models that don't fit in VRAM, a binary search finds the maximum `n_gpu_layers` that loads without OOM. The search reads the actual layer count from GGUF metadata (not assuming a fixed ceiling of 200), making it fast and accurate. A hard-crash detection distinguishes genuine OOM (try fewer layers) from model incompatibility (abort immediately). After finding max layers, NUMA policies are tested:

- `numa_none` — OS default memory allocation
- `numa_distribute` — spread allocation across both NUMA nodes (both CPU sockets)
- `numa_isolate` — restrict to NUMA node 0 (socket 0 only, lower latency)

On dual-socket server CPUs (Xeon, EPYC), NUMA policy can affect CPU-offloaded model throughput by 10–30%.

**Needle-in-a-haystack coherence check:**

The topology sweep doesn't just check that the server starts — it verifies the model produces coherent output at the claimed context size. A unique secret number is hidden near the start of a long passage and the model is asked to retrieve it from the end. This catches configurations that appear to load correctly but produce garbage output (common when context is too large for available memory).

**Results saved to:** `results/<model-slug>/topo_sweep/topo_results.json`

### Phase 0.5: Context Ceiling Sweep

After topology, the optimizer binary-searches for the maximum stable context window for each relevant scenario.

**Why this matters:**
- Using 4096 ctx when your GPU can handle 32768 wastes potential for long-document work
- Using too large a context causes OOM crashes mid-generation
- The correct ceiling depends on the winning topology and KV cache quantization type

**Search strategy — smart probe-first approach:**

Rather than naive binary search, the optimizer first estimates the ceiling from a VRAM formula (free VRAM × 85% safety margin ÷ KV cost per token, adjusted for KV quantization ratio). It probes this predicted value first. If it succeeds and equals the ceiling, done in 1 probe. If it succeeds but there's room to go higher, search upward. If it fails, search downward. This typically finds the ceiling in 3–5 probes instead of 10–15.

**KV quantization awareness:** If the Memory phase found that `q8_0` KV cache works, the context sweep accounts for the 2× compression ratio, correctly predicting a higher context ceiling than at f16.

**Coherence verification:** After the maximum stable context is found, a needle-in-haystack quality probe verifies the model is genuinely coherent at that length, not just stable in terms of not crashing.

**Context types measured:**
- `ctx_gpu_single` — max context with model fully in VRAM (single GPU winner topology)
- `ctx_gpu_combined` — max context using best split topology (Case A/C only)
- `ctx_ram_mixed` — max context with KV cache spilling into system RAM (slower but much larger)

The highest stable GPU-only value becomes the `recommended_ctx` passed to the optimizer for all phases.

**Results saved to:** `results/<model-slug>/ctx_sweep/ctx_results.json`

### Phase 1: GPU Offload

**For dense models:** Sweeps `n_gpu_layers` from 0 to the model's total layer count using a middle-out approach — starts from the midpoint and expands outward in both directions simultaneously. Each direction stops independently when performance drops below 50% of the best observed score, avoiding unnecessary probes at extreme values.

**For MoE models:** Skips entirely. MoE models use a different mechanism — the MoE phase handles expert-level CPU offloading via `--n-cpu-moe` and `--override-tensor exps=CPU`, which is more nuanced than a simple layer cutoff.

**Re-validation:** The top 3 candidates from the sweep are re-validated with fresh 3-run measurements to confirm results aren't noise.

**Results saved to:** `results/<model-slug>/gpu_results.json`

### Phase 2: MoE Thread Sweep

*MoE models only.* The `--n-cpu-moe N` parameter controls how many threads are dedicated to executing MoE expert computations on the CPU. This is separate from the main inference thread count and critically affects performance for models like Mixtral, DeepSeek, Qwen MoE, and MiniMax.

**Why this matters:** MoE models activate only a small fraction of their experts per token (e.g., 8 of 256 for MiniMax-M2.7). The expert weight matrices are large and must be loaded from RAM or VRAM each token. Dedicating too few threads starves the GPU of expert activations; too many threads causes contention with the main inference threads and NUMA cross-socket traffic.

**Sweep strategy:** Middle-out from the center of `[0, max_threads × 2]` (capped at 40), stopping in each direction when performance drops below 50% of the best observed. The ±2 neighborhood of the winner is always re-tested with fresh 3-run measurements to avoid noise selecting a false optimum.

**Results saved to:** `results/<model-slug>/moe_results.json`

### Phase 3: Expert Count Sweep

*Optional, MoE models only — enabled with `--preset moe_deep` or `--phases experts`.* Sweeps `expert_used_count` — the number of experts activated per token. Models are trained with a specific default (e.g., 8 for most MoE models) but some architectures allow this to be changed at inference time.

**Quality gate:** Unlike other phases which optimize purely for speed, this phase applies a token-level uncertainty quality gate. After each expert count is tested for speed, the model is asked two graduate-level science questions from the GPQA Diamond dataset and the distribution of output logprobs is measured. Two signals are computed:

1. **Uncertain token fraction** — tokens with logprob < -0.5 (less than ~60% model confidence), as a percentage of total tokens
2. **Tail-20% logprob average** — average of the worst 20% of logprobs (most uncertain tokens)

Configurations that increase uncertain token fraction by more than 3% vs baseline (or show equivalent tail degradation) are disqualified regardless of speed gain. This prevents the sweep from recommending "use 4 experts instead of 8" which might be 30% faster but produces noticeably degraded output.

### Phase 4: Compute Allocation

The largest and most complex optimization phase. Uses GP-Bayesian search over the compute parameter space with the MoE configuration locked from previous phases.

**Parameters explored:**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `--threads` / `-t` | 4 to max_threads, step 4 | Generation threads |
| `--threads-batch` / `-tb` | 4 to max_threads, step 4 | Prompt processing threads |
| `--poll` | 0, 10, 25, 50, 100 | GPU polling interval (ms). 0=wait-based, 100=spin-poll |
| `--poll-batch` | 0, 10, 25, 50, 100 | GPU polling for batch/prompt processing |
| `--prio` | 0–3 | Process priority for generation thread |
| `--prio-batch` | 0–3 | Process priority for batch/prompt thread |
| `--cpu-strict` | 0, 1 | Strict CPU affinity for generation |
| `--cpu-strict-batch` | 0, 1 | Strict CPU affinity for batch processing |
| `--spec-type` | ngram-simple, ngram-cache, ngram-map-k, ngram-map-k4v, ngram-mod | Speculative decoding algorithm |
| `--spec-ngram-size-n` | 2–24 | N-gram history window size |
| `--spec-ngram-size-m` | 8–96 | N-gram candidate pool size |
| `--spec-ngram-min-hits` | 1–5 | Minimum n-gram hit threshold before speculation fires |
| `--draft` | 4–48 | Maximum speculative draft tokens per step |
| `--draft-min` | 0–8 | Minimum accepted draft tokens |
| `--draft-p-min` | 0.3–0.99 | Minimum acceptance probability for draft tokens |
| `--lookup-cache-dynamic` | None / file path | Dynamic n-gram lookup cache file for cross-session learning |

**Why speculative decoding matters:** N-gram speculation pre-generates multiple candidate tokens using a simple lookup table and lets the main model verify them in parallel. For repetitive text (code, structured output, formal writing), acceptance rates of 60–80% are common, effectively 2–3× throughput. The optimizer finds the best speculation algorithm and parameters for each specific model/hardware combination.

**Seeding:** The first two trials are seeded with known-good configurations based on empirical testing across many models, ensuring the GP has good starting points before exploring.

**Results saved to:** `results/<model-slug>/compute_results.json`

### Phase 5: Memory & Throughput

Optimizes memory and KV cache parameters with compute allocation locked from Phase 4.

**Parameters explored:**

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--batch-size` / `-b` | 512, 1024, 2048, 4096 (capped to context) | Maximum prompt processing batch size |
| `--ubatch-size` / `-ub` | 128, 256, 512, 1024 | Micro-batch size for attention computation |
| `--flash-attn` / `-fa` | on, off | Flash Attention 2 — required for quantized KV cache |
| KV cache type | f16, bf16, q8_0, q5_1, q4_0 | KV cache quantization (applied to both K and V) |
| `--swa-full` | True, False | Sliding window attention — full vs. windowed mode |
| `--repack` | True, False | Tensor repacking for better memory layout |
| `--op-offload` | True, False | Operation offloading to GPU |
| `--mlock` | True, False | Lock model weights in RAM (prevents OS eviction between requests) |
| `--no-mmap` | True, False | Disable memory-mapped loading (loads entirely to RAM upfront) |

**KV cache quantization trade-offs:**

| Type | Memory vs f16 | Quality | Requirements |
|------|--------------|---------|-------------|
| `f16` | 100% (reference) | Reference | None |
| `bf16` | 100% | Reference | None |
| `q8_0` | 50% | Near-lossless | Flash attention required |
| `q5_1` | 31% | Good | Flash attention required |
| `q4_0` | 25% | Acceptable | Flash attention required |

Quantized KV cache frees VRAM which can be used for more context or reduces memory bandwidth per token. The optimizer tests these configurations under its full scoring formula — if quantization degrades output quality enough to show up in the quality gate, it won't be selected even if it's faster.

**Pruning:** Invalid combinations are pruned before server restart. `ubatch_size > batch_size` is logically impossible and skipped. Quantized KV without flash attention is invalid and skipped. This prevents wasting trial slots on configurations that would fail immediately.

**Results saved to:** `results/<model-slug>/memory_results.json`

### Phase 6: Audit Phases

After the main optimization cycle, three re-validation passes catch parameter interactions that coordinate descent can miss.

**Why audits are necessary:** Coordinate descent optimizes parameters in sequence with others locked. But optimal thread count at f16 KV may differ from optimal thread count at q8_0 KV. The audits re-run each phase on top of the results from all other phases, catching these cross-group interactions.

**MoE Audit:** Re-tests the ±2 neighbors of the optimal MoE thread count with compute *and* memory params now locked. The optimal MoE threading often shifts slightly when batch sizes and KV quantization change.

**Compute Audit:** Re-runs the full compute search (Phase 4) on top of the Memory phase's best configuration. The memory layout affects how efficiently different thread counts and speculation parameters perform — for example, a smaller ubatch might prefer different thread counts than a large one.

**Memory Audit:** Re-runs the full memory search (Phase 5) on top of Compute Audit's best configuration. This final re-validation confirms that the memory parameters remain optimal when everything else is at its best.

The three-pass re-validation pattern reliably finds 5–15% additional gains that single-pass coordinate descent misses, at the cost of roughly doubling total run time. Worth it for any model you'll run daily.

### Phase 7: Quality / Sampling

*Only in `full` and `full_plus` presets.* Optimizes sampling parameters — the settings that control how the model selects tokens from its probability distribution. These don't affect raw throughput but significantly affect output quality, creativity, and diversity.

**Parameters explored:**

Standard samplers (when mirostat=0):
- `temperature` (0.0–1.5), `top_p` (0.5–1.0), `top_k` (1–100), `min_p` (0.0–0.3)
- `typical_p` (0.5–1.0), `top_n_sigma` (-1.0 to 3.0)
- `dynatemp_range` (0.0–1.0), `dynatemp_exp` (0.5–2.0)

Mirostat (when mirostat=1 or 2):
- `mirostat_lr` (0.01–0.5) — learning rate for entropy targeting
- `mirostat_ent` (1.0–10.0) — target entropy

Always active:
- `repeat_penalty` (1.0–1.3), `repeat_last_n`, `presence_penalty`, `frequency_penalty`
- `xtc_probability` (0.0–0.5), `xtc_threshold` (0.01–0.5) — extreme token culling
- `dry_multiplier` (0.0–1.0), `dry_base` (1.0–3.0), `dry_allowed_length` (1–5) — DRY repetition penalty
- `adaptive_target` (0.0–1.0), `adaptive_decay` (0.0–1.0)

Quality is measured by running 5 factual and reasoning tasks (arithmetic, prime numbers, probability, code generation, concept definitions) and scoring the model's accuracy. The GP optimizer maximizes correct answers rather than speed in this phase.

### Phase 8: IK_llama.cpp Contrast

*Only when `IK_LLAMA_SERVER` is configured.* Runs a structured head-to-head comparison between vanilla llama.cpp and ik_llama.cpp using the best configuration found in all previous phases. See [IK_llama.cpp Support](#ik_llamacpp-support) for the 6-step methodology.

**Results saved to:** `results/<model-slug>/ik_contrast_results.json`

### Phase 9: MTP Draft Sweep

*Only when MTP is detected in GGUF metadata or `--force-mtp` is specified.* Finds the optimal Multi-Token Prediction settings using the best config from Phases 4–6 as its baseline.

**Key insight:** MTP verification batches N draft tokens in a single forward pass. This means MTP performance is sensitive to `ubatch_size` — the sweep always re-tests ubatch alongside the draft parameters rather than assuming the Memory phase's best ubatch is still optimal when MTP is active.

**spec-type detection:** The sweep probes both `--spec-type mtp` and `--spec-type draft-mtp` at startup to determine which flag name this llama.cpp build uses, then locks to the working name for all subsequent steps.

**Results saved to:** `results/<model-slug>/mtp_spec_results.json`

---

## The Scoring System

Every benchmark measurement produces a composite score that guides the optimizer's next suggestion and determines which configuration is "best".

### Measurement

Each trial generates measurements at two scales:
1. **Short prompt** (Python binary search function, ~50 tokens output) — measures generation TPS and TTFT under typical interactive conditions
2. **Large prompt** (fills 90% of context, 200 tokens output) — measures throughput under long-document conditions; also tests that the configuration is stable at the recommended context size

A VRAM snapshot is taken at peak usage during the large prompt.

### Adaptive measurement — avoiding wasted server restarts

Each trial runs in two passes:

- **Pass 1 (all configs):** A single quick short-prompt measurement. If the score falls below 70% of the current best, the config is discarded immediately. A poor trial takes ~30 seconds total.
- **Pass 2 (competitive configs):** Two more short-prompt runs (median of 3) plus the large-prompt benchmark and VRAM measurement. A promoted trial takes ~3–5 minutes total.

This means clearly bad configurations are filtered in 30 seconds while promising ones get thorough validation.

### Composite score formula

```
With large-prompt data (promoted configs — phases 4, 5, 6 audits):
  score = gen_tps        × 0.35   # generation speed — primary metric
        + long_tps       × 0.25   # large-prompt throughput — long-doc performance
        + pp_factor      × 0.15   # prompt processing speed, normalized to reference
        + ttft_factor    × 0.15   # time-to-first-token, normalized (500ms ref)
        + vram_factor    × 0.10   # VRAM efficiency — lower usage = more headroom

Without large-prompt data (quick filter pass):
  score = gen_tps      × 0.60
        + pp_factor    × 0.25
        + ttft_factor  × 0.15
```

**Why not just maximize tokens/second?**
- A config that generates at 15 t/s but takes 8 seconds for the first token feels extremely slow in interactive use — TTFT matters for perceived responsiveness
- Prompt processing speed matters when re-reading long conversation histories or summarizing documents
- VRAM efficiency matters if you want to run other applications simultaneously or use larger context
- Long-context throughput matters for document summarization, multi-turn conversations, and code review

The 35/25/15/15/10 weights reflect priorities for typical interactive use. The split between `gen_tps` (35%) and `long_tps` (25%) means both short-response and long-response scenarios are optimized.

---

## The GP-Bayesian Optimizer

The optimizer uses Gaussian Process (GP) Expected Improvement rather than random search or grid search. This matters because the compute and memory parameter spaces have 25+ parameters with complex, non-linear interactions — random search would need thousands of trials to explore adequately.

### How it works

1. **Startup phase** (first 10–15 trials): Random sampling to establish initial coverage of the space. Two hand-crafted seeds are always added first based on empirical testing across many models.
2. **GP fitting:** A Matern-5/2 kernel GP is fitted to all completed trials. The Matern-5/2 kernel assumes parameters have moderate smoothness (not perfectly smooth, not rough), which is appropriate for hardware performance surfaces.
3. **Acquisition function:** Expected Improvement (EI) with a small exploration bonus (ξ=0.01). EI balances exploitation (predict high) with exploration (high uncertainty). Configurations with high *predicted* performance OR high *uncertainty* (unexplored regions) are both considered.
4. **Candidate generation:** 2,000 random candidate configurations are encoded to [0,1], scored by the GP, and the best EI is selected. Integer parameters snap to their step grid after decoding.
5. **Encoding:** All parameters are normalized to [0,1] before GP fitting. Categorical parameters are encoded by index. Integer parameters encode position in their range.

### Early stopping — mathematically principled

Rather than running a fixed number of trials, the custom `GPStoppingCallback` evaluates the *maximum Expected Improvement* across 2,000 candidates after each trial. When the GP is confident that no untested configuration can beat the current best by more than 0.5% of its value, optimization stops early.

Two safety conditions prevent premature stopping:
- **Baseline guard:** Never stops early if the best score hasn't beaten the naked-engine baseline. If nothing has improved on defaults, keep searching.
- **Minimum trials:** Never stops before 30 completed trials. Below this, the GP doesn't have enough data to be trusted.

A patience-based fallback (stop after 20 trials with no improvement) catches cases where the GP fit diverges.

This approach saves 20–40% of configured trials on well-explored spaces while ensuring that genuinely difficult spaces are fully explored.

### Duplicate detection

Before starting a server restart for any trial, the optimizer checks if the exact same parameter combination has already been tested in this study (common when the GP converges to a narrow region). Cached scores are returned immediately with zero overhead.

---

## HTML Report

The HTML report is generated by `generate_report.py` and combines local benchmark data with metadata fetched from HuggingFace. It is completely self-contained — one `.html` file, no external CSS/JS dependencies, works offline after generation.

### Local data merged per model

- Benchmark results (best TPS, baseline TPS, gain %, best config flags)
- Topology sweep results (case, winning scenario, all scenario scores for comparison)
- Context ceiling results (GPU ctx, RAM ctx, recommended ctx, trained max ctx)
- GGUF metadata: architecture, layer count, KV head count, head dimension, context length, MoE expert count + active count, MTP layer count, hybrid/SSM detection
- IK contrast results (llama baseline, IK same-config, IK feature pack, IK best TPS, gain %)
- MTP sweep results (baseline, MTP best TPS, gain %, best parameters: n_max/n_min/p_min/ubatch)
- Quantization recommendations with size/quality/speed deltas and fit case for your hardware
- KV integrity check results (similarity score vs f16 baseline, if fast preset was used)
- Reasoning accuracy score (if fast preset was used)

### HuggingFace data merged

For each model, the report fetches (with 7-day cache):
- Model description (first substantive paragraph of README)
- Author, license, downloads, likes, pipeline tag, tags
- Parameter count (from safetensors.total, then cardData, then README, then repo name)
- Benchmark scores from Open LLM Leaderboard V1 and V2
- LM Arena ELO score
- For reference models: Artificial Analysis intelligence index, output TPS, TTFT, API cost

**Benchmark fetching — three-tier cascade:**

**V1 leaderboard** (`open-llm-leaderboard-old/details_{org}__{model}`) — covers ARC, HellaSwag, MMLU (averaged over 57 hendrycksTest subtasks), TruthfulQA (mc2), Winogrande, GSM8K. Era: 2022–mid 2024. Fetches up to 4 result files and merges them with `setdefault` (newest file wins, older files fill gaps).

**V2 leaderboard** (`open-llm-leaderboard/results/{org}/{model}`) — covers ARC, BBH, MATH-Hard, GPQA Diamond, MMLU-Pro. Era: June 2024+. JSON structure uses `data["all"]` key with task names prefixed `leaderboard_` and metric keys with `,none` suffix.

**HF card YAML + README markdown** — parses `model-index:` and `eval_results:` YAML, then falls back to markdown table parsing (three layout types: row-header, column-header, and key:value). Only for non-GGUF source repos.

**Base model chain walking:** If a model has no benchmarks, the fetcher follows `base_model` links up to 4 hops, trying leaderboard lookups and static tech-report data at each ancestor. Benchmarks sourced from an ancestor are marked with an analogue badge in the report (dashed underline on benchmark cells).

**Unified benchmark score:** V2 scores are normalized to V1-equivalent scale using affine calibration (×0.958 + 35.6) derived from models that appear on both leaderboards. This allows V1-era models (Llama 2, Mistral 7B, etc.) and V2-era models (Llama 3, Gemma 2, Qwen 2.5, etc.) to be ranked together on a single score.

**HF leaderboard rank:** The report queries the HF Datasets Server `/rows?where=` endpoint for each model's row in the published ranked table, returning their exact leaderboard position without downloading the entire table.

### 22 reference commercial/open-weights models

Toggle the "Compare reference models" checkbox to show GPT-5.4, GPT-4.1, GPT-4o, GPT-4o mini, Claude Opus/Sonnet/Haiku 4.x, Gemini 3.1 Pro, Gemini 2.5 Pro/Flash, Grok 4.1, DeepSeek V3/R1, Qwen3.5 397B/235B, Qwen2.5 72B, GLM-5/4, Kimi K2, MiniMax M2, Nemotron Super/Nano alongside your local models. Reference-only columns (Arena ELO, AA Intelligence Index, AA output TPS, AA TTFT, API cost $/1M tokens) are hidden when the checkbox is unchecked.

### Report UI features

- **Sortable columns** — click any header to sort ascending; click again for descending; third click resets to insertion order. Sort indicator arrows show current sort state.
- **Text search** — filters by model name, architecture, quantization level, or HF description simultaneously.
- **Case filter** — show only Case A/B/C/D models.
- **Status filter** — show only optimized models or only failed models.
- **Expandable detail rows** — click ▶ on any row for full detail: architecture breakdown, context ceilings, all topology scenarios with scores, IK contrast breakdown, MTP sweep results, quantization recommendations with size/case/quality/speed data, and HF tags.
- **Quantization tooltips** — hover the quant cell to see alternative quantizations and what hardware case they'd fall into.
- **IK columns** — IK t/s and IK gain, color-coded green/yellow/red, with tooltip showing which IK config won (e.g., "IK best config: IK amb=512").
- **MTP columns** — MTP t/s and MTP gain, color-coded, with tooltip showing winning MTP parameters (n_max, p_min, ubatch).
- **Analogue badge** — dashed underline on benchmark cells when scores sourced from a base model ancestor rather than this model directly.

### Generating the report manually

```bash
# From the latest batch report JSON
python generate_report.py

# From a specific batch report
python generate_report.py --report batch_reports/batch_report_20260501_140000.json

# With HuggingFace benchmarks (requires a read token)
python generate_report.py --hf-token hf_yourReadToken

# Offline — skip HF fetch, use cached data only
python generate_report.py --no-hf

# Force re-fetch all HF metadata ignoring the 7-day cache
python generate_report.py --refresh-hf

# Specify output path
python generate_report.py --output my_custom_report.html

# Specify a different reports directory
python generate_report.py --reports-dir /path/to/batch_reports
```

---

## Understanding Your Results

### What "best t/s" actually means

The reported TPS is measured on a specific test prompt (Python binary search function, ~50 tokens output) at temperature 0.4. Real-world TPS varies by:
- **Output length** — longer generations amortize the TTFT cost, so long outputs are slightly faster per token
- **Prompt complexity** — very long prompts spend more time in prompt processing (which is measured separately as `prompt_tps`)
- **Content type** — speculative decoding acceptance rates are much higher for repetitive/structured text (code, lists, formal writing) than for creative prose
- **Temperature and sampling** — higher temperature with diverse sampling adds slight overhead

For interactive conversational use, the reported TPS is a good approximation. For batch document processing, `tps_long` is more representative.

### Baseline vs Best

"Stock t/s" is the speed with a completely unconfigured server — just `-ngl 99 -c 4096` and nothing else. The "Gain %" shows how much the optimizer improved on this. Gains of 10–40% are typical for GPU-resident models; gains of 100–300% are common when speculative decoding fires well on code-generating models.

### Model Cases and their implications

**Case A** (fits both GPUs independently): You have the luxury of full GPU acceleration on either card. The optimizer tests both and picks the faster one — often the larger card wins, but sometimes PCIe lane width or thermal headroom makes the other card competitive. If scores are within 5%, either GPU is effectively equivalent.

**Case B** (largest GPU only): The model fits on your biggest GPU but not the smaller one. Optimization focuses on maximizing single-GPU performance. Consider whether a Q3 or Q2 quant would move it to Case A (fits both GPUs) — this is often worth the quality trade-off if it enables full GPU inference on the faster card.

**Case C** (split required): Both GPUs are needed. Tensor split ratios matter significantly — the optimizer tests four strategies and the winner is used for all subsequent phases. Pay attention to the split scenario scores in the detail panel: if `split_prop` barely beats `split_equal`, the split ratio isn't critical and the configuration is stable. If it wins by 20%+, tensor balance is important for this model.

**Case D** (CPU offload required): The model is larger than your combined VRAM. Performance is fundamentally limited by CPU memory bandwidth. The optimizer will find the optimal GPU/CPU layer split, NUMA policy, and MoE threading, but cannot overcome the physics of RAM bandwidth. For 90+ GB models on consumer hardware (2× 24 GB GPUs), expect 5–15 t/s for Q3-Q4 quants at DDR4-2400 speeds. DDR5 systems perform significantly better.

For Case D models, the quantization downgrade recommendations deserve special attention: moving from a 95 GB Q3 model to a 48 GB Q2 that fits entirely in VRAM can be a genuine 3–5× speed improvement.

### MTP results interpretation

- **+20–70% gain** — Strong MTP signal. The model has well-trained auxiliary heads and your hardware benefits from the draft verification pattern. Recommended for daily use; add the best MTP flags to your llama-server launch command.
- **+5–20% gain** — Moderate but real. Worth enabling. Typical for MoE models where expert routing already reduces per-step compute, leaving less room for draft verification to help.
- **0–5% gain** — Marginal. May not be worth the configuration complexity and potential edge-case instability.
- **Negative gain** — MTP was counterproductive. Most common cause: near-full VRAM (MTP auxiliary heads need 2–5% extra VRAM for the prediction buffer). Try reducing KV cache quantization to free VRAM, then re-run the MTP sweep.

### Context ceiling interpretation

- `ctx_gpu` is the largest context that fits entirely in VRAM at the recommended topology — use this as your `-c` value for best performance and stability.
- `ctx_ram` is the largest context achievable with KV cache spilling to RAM. Much slower (2–5× slower KV reads) but allows much larger contexts for long-document work.
- If `ctx_gpu` is much smaller than the model's trained context (e.g., 8k vs 131k), you're VRAM-limited on KV cache. Reducing KV quantization (from f16 to q8_0 or q4_0) halves or quarters KV memory, directly expanding available context — the ctx sweep accounts for this.

### Quantization recommendations

The report's detail rows and tooltips suggest alternative quantizations for each model based on your specific hardware. Recommendations are direction-labeled:
- ↑ **Upgrade** — higher quality quant that still fits your hardware
- ↓ **Downgrade** — smaller quant that changes the hardware case (e.g., moves from Case C to Case B, enabling full single-GPU operation)
- ↔ **Sidegrade** — similar quality at different size

A downgrade recommendation showing `Q4_K_M → Case A` is often worth taking: moving from a model that requires CPU offload to one that fits entirely in VRAM can be a 3–5× speed improvement, even at lower quantization quality. For very large models (70B+), the speed gain from Case D→C or C→B often outweighs the quality loss from one quantization step down.

---

## Troubleshooting

**Server won't start:**
- Verify `LLAMA_SERVER` or `--llama-server` points to a valid binary with execute permission
- Try `--verbose` to see the full loading output including CUDA initialization
- Model file might be incomplete — check that file size matches expected for the quantization level
- Try removing `--mlock` or `--no-mmap` — some systems have insufficient locked memory limits

**All trials return 0.0:**
- The server is starting but `/completion` requests are failing. Check for port conflicts.
- Try `--verbose` and look for error messages in stderr output
- Context size might be too large for available VRAM — start with a small test using `-c 4096`
- Check that the model file isn't corrupt (try loading it manually with `llama-server` from command line)

**"Speculative decoding context not initialized":**
- This is an informational log message, not an error. Speculative decoding is working correctly.
- If you see this repeatedly with `spec_type=ngram-*`, the lookup cache file may need to be pre-created; the optimizer handles this automatically via `LOOKUP_CACHE_FILE`

**MTP phase skipped / "no MTP heads detected":**
- The GGUF conversion may have omitted the `nextn_predict_layers` metadata key even though the model has MTP heads — common with some community quantizations.
- Use `--force-mtp` to run the MTP sweep regardless of detection.
- Check what was detected: `python -c "from model_utils import detect_mtp; from pathlib import Path; print(detect_mtp(Path('your.gguf')))"`
- For Qwen3.6 models: the architecture key is `qwen35`. If detection returns `arch_hint`, the heads may still be there — use `--force-mtp`.

**MTP slower than baseline:**
- Your VRAM is likely nearly full. MTP auxiliary prediction heads require 2–5% extra VRAM for their activation buffers.
- Try reducing KV cache quantization to q8_0 or q4_0 to free VRAM, then re-run `--rerun-phases mtp_spec`.
- The model may have weak MTP training — not all quantizations of MTP-capable models have equally well-trained heads.
- Also check: is `--flash-attn` enabled? MTP requires flash attention to be efficient.

**IK_llama.cpp: flags not recognized:**
- The `-rtr`, `-fmoe`, `-mla`, `-amb`, `-ser` flags are IK-exclusive. The contrast phase only adds them when `_ik_server=True` in the config dict.
- Verify `--ik-llama-server` points to the IK build, not vanilla llama.cpp. They have the same filename by default.
- Check the `[debug] cmd:` line in the log — it shows every flag being passed.

**Very slow optimization (>15 min per trial):**
- Your model is Case D with significant CPU offload. Each server restart takes 2–5 minutes for a 50+ GB model.
- The optimizer automatically halves trial counts when it detects models running below 15 t/s.
- Use `--preset fast` for initial exploration, then `--preset standard --resume` once you've confirmed the model is worth deep optimization.
- Consider whether a lower-quant variant would move the model to Case C or better.

**HuggingFace benchmark fetch fails:**
- Ensure `HF_TOKEN` is set to a valid **read** token (not fine-grained) from `huggingface.co/settings/tokens`
- The Open LLM Leaderboard datasets require authentication even for read access
- Use `--no-hf` to skip HF fetching and still generate a useful report with local benchmark data only
- Use `--refresh-hf` if you suspect the cache has stale/failed entries

**OOM during context sweep:**
- Expected behavior. The sweep intentionally probes sizes that cause OOM to find the exact boundary.
- If the server crashes repeatedly at small context sizes (≤8192), your model may have a different VRAM layout than the size-based estimate assumes. The sweep will still converge to the correct ceiling.
- If you're running other GPU applications simultaneously, temporarily stop them — the sweep needs predictable VRAM availability.

**Results seem wrong / lower than expected:**
- Check `--verbose` output for the actual command being run per trial
- Look for the `[debug] cmd:` lines in the log — these show every flag passed to llama-server
- The baseline measurement uses the naked engine (no optimization flags). If your baseline TPS is much lower than manually running llama-server, check that `--fit` is not silently reducing `n_gpu_layers`
- Ensure no background processes (Ollama, LM Studio) are holding a GPU context — the optimizer calls `kill_competing_processes()` but this only catches known process names

**Resume not working:**
- Use `--resume` flag explicitly
- Check that `results/<model-slug>/` directory exists and contains at least one `*_results.json` file
- Model slugs are derived from the filename stem (lowercased, spaces replaced with underscores) — verify the directory name matches
- Use `--retry` to explicitly queue previously failed models, `--retry first` to run them before new models

---

## Example Report Output

Below is an illustrative example of what the HTML report's data looks like for a typical multi-model optimization run on a dual-GPU system (2× RTX 3090).

```
===========================================================================================================
  FINAL BATCH REPORT  —  Generated 2026-05-01 14:22
  Optimized: 4   No results: 0
===========================================================================================================
  #   Model                         Quant    Case   Stock   Best   Gain  IK t/s  IK gain  MTP t/s MTP gain  Topo
  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1.  Qwen3.6-27B-PRISM-PRO-DQ     Q4_K_M   B      18.4    31.2   +70%   38.1    +22%     44.7    +43%      gpu0_only
  2.  DeepSeek-R1-8B-Q8_0          Q8_0     A      44.2    67.3   +52%   71.2     +6%     89.1    +32%      gpu1_only
  3.  Llama-3.1-70B-Q5_K_M         Q5_K_M   C      12.1    24.8  +105%   29.4    +19%       —       —        split_prop
  4.  MiniMax-M2.7-UD-Q3_K_XL      Q3_K_XL  D       4.0     8.8  +120%   11.2    +27%       —       —        numa_distribute
===========================================================================================================
```

**Expanded detail for Qwen3.6-27B-PRISM-PRO-DQ (row 1):**

```
  Architecture:   qwen35  |  62 layers  |  8 KV heads  |  128 head dim
  Parameters:     27.4B   |  MTP: 1 prediction layer  (detected: metadata, high confidence)
  Train context:  131,072 tokens
  GGUF quant:     Q4_K_M (4.8 bpw)  |  ~15 GB file

  Context ceilings:
    GPU single:       32,768 tokens  ← use this as -c
    Recommended:      32,768 tokens

  Topology:
    Winner: gpu0_only — GPU0 (RTX 3090, 24 GB)
    Scenarios: gpu0_only=18.4 t/s   gpu1_only=16.2 t/s  (GPU0 is 13% faster)

  Best llama.cpp config:
    -ngl 99 -c 32768 -b 512 --ubatch-size 512
    -t 8 -tb 32
    --spec-type ngram-map-k4v --draft 24 --draft-min 4 --draft-p-min 0.85
    --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
    --mlock --no-mmap
    Score: 31.2 t/s  (baseline was 18.4 t/s)

  MTP draft sweep:
    Baseline (no MTP):              31.2 t/s  (reference)
    Spec-type detected:             --spec-type mtp  (not draft-mtp)
    Draft depth scan:               n_max=1: 38.1   n_max=2: 43.9   n_max=3: 41.2
    p_min sweep (n_max=2):          0.50: 42.1   0.70: 44.7   0.85: 43.2   0.95: 40.8
    ubatch re-test (n_max=2 p=0.70): 256: 41.9   512: 44.7   1024: 43.1
    n_min test:                     n_min=0: 44.7   n_min=1: 43.2
    Final validation (5 runs):      44.7 t/s  ← MTP WINNER
    MTP gain vs no-MTP:             +43.3%

    Best MTP config:
      --spec-type mtp --draft 2 --draft-min 0 --draft-p-min 0.70
      --ubatch-size 512 --flash-attn on

  IK_llama.cpp contrast:
    llama.cpp best:                 31.2 t/s  (reference)
    IK same config (no IK flags):  32.1 t/s  (+2.9%)
    IK + MLA + fused-MoE + RTR:    35.8 t/s  (+14.7%)
    IK amb=512:                     38.1 t/s  (+22.1%)  ← IK WINNER

  HuggingFace benchmarks (Open LLM Leaderboard v2):
    MMLU: 84.5%   GSM8K: 94.0%   MATH: 92.0%   GPQA: 55.0%   BBH: 81.0%
    Local Score: 81.2 (v2)  |  LB Rank: #89 (v2)  |  Downloads: 28,400

  Quant recommendations:
    ↑ Q5_K_M   ~19 GB  Case B  size +27%  spd=90  qual=78  upgrade
    ↔ Q4_K_S   ~14 GB  Case B  size  -7%  spd=100 qual=70  sidegrade
    ↓ Q3_K_M   ~11 GB  Case A  size -27%  spd=95  qual=62  downgrade — would fit BOTH GPUs independently
```

**What this tells you:**

The Qwen3.6-27B PRISM-PRO is a Case B model (fits GPU0 only). Starting from 18.4 t/s stock:

- The optimizer found a **70% gain** to 31.2 t/s through compute+memory tuning — primarily speculative decoding (`ngram-map-k4v`, 24-token drafts) and q8_0 KV cache (halved KV memory, enabling 32k context vs 16k at f16).
- MTP added another **43% on top of that** — n_max=2 at p_min=0.70 means the model drafts 2 tokens per step and accepts them when at least 70% confident. The sweep correctly identified that p_min=0.85 was slightly too conservative (fewer accepted drafts) while p_min=0.50 was too permissive (too many bad drafts requiring re-generation). The 0.70 sweet spot maximizes net throughput.
- IK_llama.cpp added a further **22%** via run-time tensor repacking.
- Combined result: **44.7 t/s vs 18.4 t/s stock — a 2.4× improvement**.

The Q3_K_M downgrade recommendation is worth noting: moving to ~11 GB would make the model Case A (fits both GPUs independently at 24 GB each), meaning it could run on GPU1 if GPU0 is busy with another task. At ~22 tokens/second the Q3 would likely still be 20% faster than the unoptimized Q4 stock speed.

---

## License

MIT License. See `LICENSE` for full text.

---

## Acknowledgements

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — the inference engine this optimizer wraps
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) — extended optimizations for hybrid CPU/GPU inference
- [Optuna](https://optuna.org/) — hyperparameter optimization framework providing the GP sampler infrastructure
- [HuggingFace](https://huggingface.co/) — model cards, leaderboard data, and benchmark results
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) — V1 and V2 benchmark data
- [lmarena.ai](https://lmarena.ai/) — Arena ELO scores for reference models
- [Artificial Analysis](https://artificialanalysis.ai/) — intelligence index and API performance metrics for reference models
