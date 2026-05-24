# Llama Optimizer

**Automatically find the fastest possible settings for running large language models on your machine.**

GP-Bayesian multi-phase optimization for `llama-server` (llama.cpp) and `ik_llama-server` (ik_llama.cpp), with GPU topology sweep, context ceiling detection, HTML reporting, and IK_llama.cpp contrast benchmarking.

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

When you run a large language model locally using llama.cpp, there are dozens of settings that affect how fast the model generates text — things like how many layers to put on your GPU, how many CPU threads to use, what batch sizes to use, whether to use flash attention, what kind of KV cache quantization to use, and more. Getting these wrong can mean your model runs at 3 tokens per second instead of 15. **This tool automatically tests thousands of combinations and finds the fastest settings for your specific hardware and model.**

### What you need before starting

- **Windows or Linux** with Python 3.10+
- **A working `llama-server.exe`** (or `llama-server` on Linux) from [llama.cpp](https://github.com/ggerganov/llama.cpp/releases)
- **One or more `.gguf` model files** you want to optimize
- **NVIDIA GPU(s)** with CUDA (AMD may work but is untested)
- Optional: `ik_llama-server` from [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) for additional speed gains on hybrid CPU/GPU setups

### Quickest possible run

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

This will:
1. Find all `.gguf` files in your models directory
2. Run a GPU topology test to figure out the best GPU configuration for each model
3. Run 60 compute trials + 60 memory trials to find optimal settings
4. Generate an HTML report you can open in any browser

When it finishes, open `batch_reports/report_latest.html` in your browser.

### What the results look like

The HTML report shows a sortable table with every model you tested. Key columns:

| Column | Meaning |
|--------|---------|
| **Best t/s** | Fastest tokens per second found |
| **Stock t/s** | Speed with default settings (no optimization) |
| **Gain %** | How much faster the optimizer made it |
| **Case** | How the model fits on your GPU(s) — A=both GPUs, B=one GPU, C=split, D=needs RAM |
| **ctx GPU** | Maximum context length that fits fully in VRAM |
| **IK t/s** | Speed with ik_llama.cpp (if configured) |
| **IK gain** | How much faster ik_llama.cpp is vs vanilla llama.cpp |

### How long does it take?

| Preset | Time per model | What it tests |
|--------|---------------|---------------|
| `fast` | ~25 min | Quick sweep + diagnostics |
| `standard` | ~1–2 hours | Full compute + memory optimization |
| `thorough` | ~3–4 hours | Full optimization + re-validation passes |
| `full` | ~4–5 hours | Everything including sampling params |
| `ik` | ~2–3 hours | Standard + IK_llama.cpp contrast |

Press **Ctrl+C** at any time to skip the current phase and move to the next one. Results are saved after every trial so you can always resume.

---

## What This Does

Llama Optimizer is a multi-phase automated benchmarking and optimization system for locally-run large language models. It works by:

**Characterizing your hardware** — Before optimization begins, a topology sweep classifies each model into one of four cases based on how it fits in your VRAM, tests every relevant GPU configuration (single GPU, split across GPUs, NUMA policies), and binary-searches the maximum stable context window. This prevents the optimizer from wasting time on configurations that will OOM.

**Finding optimal inference parameters** — Using Gaussian Process Bayesian optimization (not random search, not grid search), it intelligently explores the parameter space for each model, learning from every trial to propose configurations most likely to improve performance. Parameters explored include thread counts, batch sizes, KV cache quantization, flash attention, speculative decoding, CPU/GPU MoE routing split, and more.

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
- **CPU:** Any x86-64. More cores help for CPU-offloaded MoE models. AVX2/AVX-512 support helps significantly.

### Models

Any `.gguf` model file compatible with llama.cpp. The optimizer handles:
- Single-file models (`model.gguf`)
- Multi-shard models (`model-00001-of-00004.gguf` — only shard 1 needed, llama-server finds the rest)
- Dense models (Llama, Mistral, Qwen, Phi, Gemma, etc.)
- MoE models (Mixtral, DeepSeek, Qwen MoE, MiniMax, etc.)
- Hybrid/SSM models (Mamba, Jamba)

---

## Installation

```bash
git clone https://github.com/VykosX/llama-batch-optimizer
cd llama-batch-optimizer

# Install dependencies
pip install requests optuna numpy scipy scikit-learn psutil pynvml

# Optional: for IK_llama.cpp support
# Download ik_llama-server from https://github.com/ikawrakow/ik_llama.cpp
```

No other installation steps required. All files run directly from the cloned directory.

---

## File Overview

```
llm-batch-optimizer/
├── batch_runner.py         Entry point — run this to optimize your models
├── optimizer_adapter.py    Bridge: translates RunConfig → optimize.py phase calls
├── optimize.py             Core GP-Bayesian optimizer with all phase logic
├── sweep_engine.py         GPU topology sweep + context ceiling sweep
├── model_utils.py          GGUF metadata reader, GPU/RAM helpers, quant catalogue
├── generate_report.py      HTML report generator with HuggingFace integration
├── test_benchmarks.py      Standalone script to test HF benchmark fetching
├── results/                Per-model optimization results (auto-created)
├── batch_reports/          Batch summary reports (auto-created)
└── logs/                   Run logs (auto-created)
```

| File | Lines | Role |
|------|-------|------|
| `batch_runner.py` | ~850 | Entry point, batch loop, report generation |
| `optimizer_adapter.py` | ~310 | Preset system, phase routing, IK server wiring |
| `optimize.py` | ~3700 | All optimization phases, GP sampler, scoring |
| `sweep_engine.py` | ~3000 | Topo sweep, ctx sweep, GGUF binary reader |
| `model_utils.py` | ~580 | Shared utilities, quant recommendations |
| `generate_report.py` | ~3200 | Self-contained HTML report with JS/CSS |

---

## Running the Optimizer

### Basic Usage

```powershell
# Minimal — uses LLAMA_SERVER env var, models from current directory
python batch_runner.py

# Specify paths explicitly
python batch_runner.py \
  --llama-server "C:\path\to\llama-server.exe" \
  --models-dir "C:\path\to\models" \
  --preset standard \
  --topo-sweep \
  --html-report

# Resume interrupted run (skips models that already have results)
python batch_runner.py --preset standard --topo-sweep --resume

# Report only (no benchmarking — just regenerate the HTML)
python batch_runner.py --report-only --html-report

# Test a single model
python batch_runner.py --filter "Qwen3.5-9B" --preset standard --topo-sweep

# IK_llama.cpp comparison run
python batch_runner.py \
  --ik-llama-server "C:\path\to\ik_llama-server.exe" \
  --preset ik \
  --topo-sweep \
  --html-report
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
| `LLM_OPT_STARTUP_TIMEOUT` | Server startup timeout in seconds (auto-scaled by model size) |
| `HF_TOKEN` | HuggingFace read token for benchmark data in reports |
| `GPU0_VRAM_GB` | Override detected GPU0 VRAM (useful when pynvml unavailable) |
| `GPU1_VRAM_GB` | Override detected GPU1 VRAM |

### All Command-Line Switches

#### Path Options

| Switch | Default | Description |
|--------|---------|-------------|
| `--models-dir PATH` | `./models` or `LLM_OPT_MODELS_DIR` | Directory to scan for `.gguf` files (recursive) |
| `--llama-server PATH` | Auto-detected or `LLAMA_SERVER` | Path to `llama-server` binary |
| `--ik-llama-server PATH` | Auto-detected or `IK_LLAMA_SERVER` | Path to `ik_llama-server` binary |
| `--results-base PATH` | `./results` | Root directory for per-model results |
| `--reports-dir PATH` | `./batch_reports` | Directory for batch summary reports |
| `--logs-dir PATH` | `./logs` | Directory for run log files |

#### Model Selection

| Switch | Description |
|--------|-------------|
| `--filter TEXT` | Only process models whose filename contains TEXT (case-insensitive) |
| `--resume` | Skip models that already have complete optimization results |
| `--retry [first\|last]` | Retry previously failed models. `first` = retry before new models, `last` = after |
| `--dry-run` | List models that would be processed, then exit without running |

#### Preset and Phase Control

| Switch | Description |
|--------|-------------|
| `--preset NAME` | Optimization preset (see [Preset Reference](#preset-reference)) |
| `--phases PHASE...` | Override preset phase list entirely, e.g. `--phases compute memory` |
| `--skip-phases [PHASE...]` | Skip specific phases. Bare `--skip-phases` skips everything not in `--rerun-phases` |
| `--rerun-phases PHASE...` | Force re-run named phases even if results exist, e.g. `--rerun-phases ik_contrast` |
| `--trials PHASE=N...` | Override trial counts per phase, e.g. `--trials compute=80 memory=80` |

#### Run Mode

| Switch | Default | Description |
|--------|---------|-------------|
| `--timeout SECONDS` | 5400 (90 min) | Per-model hard timeout. Model is abandoned if exceeded |
| `--trial-timeout SECONDS` | 360 (6 min) | Per-trial hard timeout. Trial is marked failed if exceeded |
| `--port PORT` | 8090 | Port for llama-server HTTP API |
| `--report-only` | — | Regenerate HTML report from existing results, no benchmarking |
| `--html-report` | — | Generate HTML report after batch completes |
| `--no-hf` | — | Skip HuggingFace metadata fetch (faster, offline-safe) |
| `--refresh-hf` | — | Force re-fetch all HF metadata, ignoring 7-day cache |
| `--no-log` | — | Disable automatic log file in `./logs/` |
| `--verbose` | — | Show live llama-server loading output (layer counts, CUDA init) |
| `--interactive` | — | Pause 5 seconds between models, press `n` to stop cleanly after current model |

#### Topology Sweep

| Switch | Default | Description |
|--------|---------|-------------|
| `--topo-sweep` | — | Run GPU topology benchmark before optimizer |
| `--topo-only` | — | Run topology sweep only, skip optimizer |
| `--topo-runs N` | 2 | Benchmark runs per topology scenario (more = more stable, slower) |
| `--gpu-filter SCENARIO...` | — | Only test specific topology scenarios by ID (e.g. `gpu0_only split_prop`) |
| `--skip-gpu GPU...` | — | Skip topology scenarios for specific GPUs by index (0, 1) or name fragment |
| `--force-numa` | — | Force NUMA policy tests even for models that fit in single GPU |

#### Context Ceiling Sweep

| Switch | Description |
|--------|-------------|
| `--ctx-sweep` | Binary-search the maximum stable context length after topology sweep |
| `--ctx-only` | Run context sweep only, skip topology and optimizer |
| `--skip-ctx-b` | Skip RAM-spill context tests (faster, GPU ceiling only) |

#### Hardware Overrides

| Switch | Description |
|--------|-------------|
| `--gpu0-vram FLOAT` | Override detected GPU0 VRAM in GB |
| `--gpu1-vram FLOAT` | Override detected GPU1 VRAM in GB |

### Preset Reference

| Preset | Phases | Trials | Est. Time/Model | Best For |
|--------|--------|--------|-----------------|----------|
| `fast` | binary_screen → fast_gpu → fast_moe → compute → memory → integrity → reasoning | compute=30, memory=25 | ~25 min | Quick first pass, large model collections |
| `standard` | gpu → moe → compute → memory | compute=60, memory=60 | ~1–2h | General purpose, good balance |
| `thorough` | gpu → moe → compute → memory → moe_audit → compute_audit → memory_audit | 60 each | ~3–4h | Final optimization before daily use |
| `full` | All thorough phases + quality/sampling | 60 each + quality=80 | ~4–5h | Complete optimization including sampling params |
| `moe_deep` | All thorough + expert count sweep | 60 each | ~4h | MoE models where expert count matters for quality |
| `ik` | gpu → moe → compute → memory → ik_contrast | compute=60, memory=60 | ~2–3h | When IK_llama.cpp is configured |
| `ik_thorough` | Full thorough phases + ik_contrast | 60 each | ~4–5h | Most complete IK comparison |

Override trial counts for any preset: `--trials compute=40 memory=40` halves the standard preset's trials for a faster run.

### IK_llama.cpp Support

[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) is a fork of llama.cpp with additional optimizations, particularly valuable for hybrid CPU/GPU setups with MoE models:

| IK Feature | Flag | Benefit |
|-----------|------|---------|
| MLA Attention | `-mla 2` | 50% KV cache reduction for DeepSeek-architecture models |
| Fused MoE | `-fmoe` | 20–80% faster expert routing on GPU |
| Run-time Repack | `-rtr` | 50–80% faster CPU execution via SIMD-optimized tensor layouts |
| Attention Max Batch | `-amb N` | Reduces K×Q compute buffer pressure, helps long context |
| Smart Expert Reduction | `-ser N,1` | Drop N experts per token to trade ~quality for speed |

**Setup:**

```powershell
# Option 1: environment variable (recommended)
$env:IK_LLAMA_SERVER = "G:\Tools\ik_llama-server.exe"

# Option 2: CLI flag
python batch_runner.py --ik-llama-server "G:\Tools\ik_llama-server.exe" --preset ik

# Option 3: place ik_llama-server.exe next to batch_runner.py — auto-detected
```

**Behavior:**
- If **only** `IK_LLAMA_SERVER` is set (no `LLAMA_SERVER`): all phases run with IK automatically, no separate contrast phase
- If **both** are set: all optimization phases use vanilla llama.cpp for fair results, then `ik_contrast` runs a structured comparison
- The contrast phase runs 6 steps: llama.cpp baseline → IK same config → IK with full feature pack → attn_max_batch sweep → SER sweep (MoE only) → MLA mode sweep (MoE only)

---

## How It All Works — Phase by Phase

The optimizer runs phases sequentially, with each phase seeding the next from its best result. This is coordinate descent: optimize one group of parameters at a time with everything else locked, then rotate to the next group.

```
Topology Sweep ──► Context Ceiling Sweep
                         │
                         ▼
                   GPU Offload (Phase 1)
                         │
                         ▼
                   MoE Thread Sweep (Phase 2, MoE only)
                         │
                         ▼
                   Expert Count (Phase 3, optional)
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
                         ▼
                   IK_llama Contrast (Phase 8, if configured)
```

### Phase 0: Topology Sweep

Before any optimization, the topology sweep characterizes how each model relates to your hardware.

**Model classification (Cases A–D):**

Every model is classified into one of four cases based on its estimated VRAM requirement (file size + 2–5 GB overhead for KV cache and compute buffers):

| Case | Condition | Behavior |
|------|-----------|----------|
| A | Fits in *both* GPUs independently | Tests each GPU alone, picks the faster one |
| B | Fits in GPU0 only (largest GPU) | Tests GPU0 only |
| C | Requires *combined* VRAM of both GPUs | Tests four split strategies |
| D | Exceeds combined VRAM (needs CPU offload) | Binary-searches max GPU layers, tests NUMA policies |

**Case C split strategies:**

When a model needs both GPUs combined, four tensor split ratios are tested:
- `split_prop` — proportional by VRAM (e.g., 60/40 for 24/16 GB GPUs)
- `split_equal` — 50/50
- `split_g0heavy` — 80/20 favouring the larger card
- `split_kv_aware` — shifts weight off the main GPU which carries KV cache overhead

Each scenario is benchmarked with 2–3 runs (configurable via `--topo-runs`). The winner is used as the fixed topology for all subsequent optimization phases.

**Case D GPU layer search:**

For models that don't fit in VRAM, a binary search finds the maximum `n_gpu_layers` that loads without OOM. The search reads the actual layer count from GGUF metadata rather than assuming 200, making it fast and accurate. After finding max layers, NUMA policies are tested:

- `numa_none` — OS default memory allocation
- `numa_distribute` — spread allocation across both NUMA nodes (both CPU sockets)
- `numa_isolate` — restrict to NUMA node 0 (socket 0 only, lower latency)

On dual-socket server CPUs (Xeon, EPYC), NUMA policy can affect CPU-offloaded model throughput by 10–30%.

**Needle-in-a-haystack coherence check:**

The topology sweep doesn't just check that the server starts — it verifies that the model actually produces coherent output at its claimed context length. A hidden secret number is placed early in a long passage and the model is asked to retrieve it. This prevents configurations that appear to load correctly but produce garbage output from being selected as the topology winner.

**Results saved to:** `results/<model-slug>/topo_sweep/topo_results.json`

### Phase 0.5: Context Ceiling Sweep

After topology, the optimizer binary-searches for the maximum stable context window. This matters because:
- Using 4096 ctx when your GPU can handle 32768 wastes potential
- Using too large a context causes OOM crashes mid-generation
- The correct ceiling depends on the winning topology (single GPU vs split vs hybrid)

**Search strategy:** The optimizer first estimates the maximum context from a VRAM formula (free VRAM ÷ KV cost per token, accounting for quantization), probes that predicted value first, then binary-searches up or down from there. This typically finds the ceiling in 3–5 probes instead of 15+.

**Context types measured:**
- `ctx_gpu_single` — max context with model fully in VRAM (fastest)
- `ctx_gpu_combined` — max context using both GPUs (Case A/C only)
- `ctx_ram_mixed` — max context with KV cache spilling to RAM (slower but much larger)

**Recommended ctx** — the highest stable GPU-only value — is passed to the optimizer as its context size for all subsequent phases.

**Results saved to:** `results/<model-slug>/ctx_sweep/ctx_results.json`

### Phase 1: GPU Offload

**For dense models:** Sweeps `n_gpu_layers` from 0 to the model's total layer count using a middle-out approach (starting from the midpoint and expanding outward in both directions). Each direction stops independently when performance drops below 50% of the best observed score, avoiding unnecessary probes at extreme values.

**For MoE models:** Skips entirely. MoE models use a different mechanism — the MoE phase handles expert-level CPU offloading with `--n-cpu-moe` and `--override-tensor exps=CPU`, which is more nuanced than a simple layer cutoff.

**Re-validation:** The top 3 candidates from the sweep are re-validated with fresh 3-run measurements to confirm results.

**Results saved to:** `results/<model-slug>/gpu_results.json`

### Phase 2: MoE Thread Sweep

*MoE models only.* The `--n-cpu-moe N` parameter controls how many threads are dedicated to executing MoE expert computations on the CPU. This is separate from the main thread count and critically affects performance for models like Mixtral, DeepSeek, Qwen MoE, and MiniMax.

**Why this matters:** MoE models activate only a small fraction of their experts per token (e.g., 8 of 256 for MiniMax). The expert weight matrices are large and must be loaded from RAM or VRAM each token. Dedicating too few threads starves the GPU of expert activations; too many threads causes contention with the main inference threads.

**Sweep strategy:** Middle-out from `max_threads` (typically half of total CPU threads), stopping in each direction when performance drops below 50% of best. The ±2 neighborhood of the winner is always re-tested with fresh 3-run measurements to avoid measurement noise selecting a false optimum.

**Results saved to:** `results/<model-slug>/moe_results.json`

### Phase 3: Expert Count Sweep

*Optional, MoE models only, enabled with `--preset moe_deep` or `--phases experts`.* Sweeps `expert_used_count` — the number of experts activated per token. Models are trained with a default (e.g., 8 for most MoE models) but some architectures allow this to be changed at runtime.

**Quality gate:** Unlike other phases which optimize purely for speed, this phase applies a token-level uncertainty quality gate. After each expert count is tested for speed, the model is asked two graduate-level science questions (from the GPQA Diamond dataset) and the distribution of logprobs is measured. Configurations that increase the proportion of uncertain tokens (logprob < -0.5) by more than 3% relative to baseline are disqualified, regardless of speed.

This prevents the sweep from recommending "use 4 experts instead of 8" which might be 30% faster but produces noticeably degraded output.

### Phase 4: Compute Allocation

The largest and most complex optimization phase. Uses GP-Bayesian search over the compute parameter space with the MoE configuration locked from previous phases.

**Parameters explored:**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `--threads` / `-t` | 4 to max_threads, step 4 | Generation threads |
| `--threads-batch` / `-tb` | 4 to max_threads, step 4 | Prompt processing threads |
| `--poll` | 0, 10, 25, 50, 100 | GPU polling interval (ms). 0=wait, 100=spin |
| `--poll-batch` | 0, 10, 25, 50, 100 | GPU polling for batch processing |
| `--prio` | 0–3 | Process priority for generation |
| `--prio-batch` | 0–3 | Process priority for batch processing |
| `--cpu-strict` | 0, 1 | Strict CPU affinity for generation |
| `--cpu-strict-batch` | 0, 1 | Strict CPU affinity for batch |
| `--spec-type` | ngram-simple, ngram-cache, ngram-map-k, ngram-map-k4v, ngram-mod | Speculative decoding algorithm |
| `--spec-ngram-size-n` | 2–24 | N-gram history window size |
| `--spec-ngram-size-m` | 8–96 | N-gram candidate pool size |
| `--spec-ngram-min-hits` | 1–5 | Minimum n-gram hit threshold |
| `--draft` | 4–48 | Maximum speculative draft tokens |
| `--draft-min` | 0–8 | Minimum accepted draft tokens |
| `--draft-p-min` | 0.3–0.99 | Minimum acceptance probability |
| `--lookup-cache-dynamic` | None / file path | Dynamic n-gram lookup cache file |

**Why speculative decoding matters:** N-gram speculation pre-generates multiple candidate tokens using a simple lookup table and lets the main model verify them in parallel. For repetitive text (code, structured output, formal writing), acceptance rates of 60–80% are common, effectively 2–3× throughput. The optimizer finds the best speculation parameters for each model/hardware combination.

**Seeding:** The first two trials are seeded with known-good configurations based on empirical testing across many models, ensuring the GP has good starting points before exploring.

**Results saved to:** `results/<model-slug>/compute_results.json`

### Phase 5: Memory & Throughput

Optimizes memory and KV cache parameters with compute allocation locked from Phase 4.

**Parameters explored:**

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--batch-size` / `-b` | 512, 1024, 2048, 4096 | Maximum prompt processing batch size (capped to context) |
| `--ubatch-size` / `-ub` | 128, 256, 512, 1024 | Micro-batch size for attention computation |
| `--flash-attn` / `-fa` | on, off | Flash Attention 2 (required for quantized KV) |
| `--cache-type-k/v` | f16, bf16, q8_0, q5_1, q4_0 | KV cache quantization |
| `--swa-full` | True, False | Sliding window attention mode |
| `--repack` | True, False | Tensor repacking for better memory layout |
| `--op-offload` | True, False | Operation offloading to GPU |
| `--mlock` | True, False | Lock model weights in RAM (prevents OS eviction) |
| `--no-mmap` | True, False | Disable memory-mapped loading (loads entirely to RAM) |

**KV cache quantization trade-offs:**

| Type | Memory | Quality | Notes |
|------|--------|---------|-------|
| `f16` | 100% | Reference | Safe for any model |
| `bf16` | 100% | Reference | BFloat16, similar to f16 |
| `q8_0` | 50% | Near-lossless | Requires flash attention |
| `q5_1` | 31% | Good | Requires flash attention |
| `q4_0` | 25% | Acceptable | Requires flash attention |

Quantized KV cache frees VRAM for more context or reduces memory bandwidth requirements. The optimizer tests these configurations and applies its quality-aware scoring to detect if quantization is degrading output.

**Results saved to:** `results/<model-slug>/memory_results.json`

### Phase 6: Audit Phases

After the main optimization cycle, three re-validation passes catch parameter interactions that coordinate descent can miss.

**MoE Audit:** Re-tests the best ±2 neighbors of the optimal MoE thread count with compute *and* memory params now locked. The optimal MoE threading often shifts slightly when other parameters change.

**Compute Audit:** Re-runs the full compute search (Phase 4) but now on top of the Memory phase's best configuration. The memory layout affects how efficiently different compute configurations perform.

**Memory Audit:** Re-runs the full memory search (Phase 5) on top of Compute Audit's best configuration. This is the final re-validation that confirms the chosen parameters remain optimal when everything is combined.

This three-pass re-validation pattern reliably finds 5–15% additional gains that a single-pass approach misses, at the cost of roughly doubling total run time.

### Phase 7: Quality / Sampling

*Only in `full` preset.* Optimizes sampling parameters — the settings that control how the model selects tokens from its probability distribution. These don't affect raw throughput but significantly affect output quality and diversity.

**Parameters explored:**
- Temperature, top-p, top-k, min-p (standard samplers)
- Mirostat mode 1/2 with learning rate and entropy targets
- Typical-p, top-n-sigma, dynamic temperature range/exponent
- Repeat penalty, frequency penalty, presence penalty
- XTC (extreme token culling) probability and threshold
- DRY (Don't Repeat Yourself) multiplier, base, window
- Adaptive sampling target and decay

Quality is measured by running 5 factual and reasoning tasks (arithmetic, prime numbers, probability, code generation, definitions) and scoring the model's accuracy. The GP optimizer maximizes accuracy rather than speed in this phase.

### Phase 8: IK_llama.cpp Contrast

*Only when `IK_LLAMA_SERVER` is configured.* Runs a structured head-to-head comparison between vanilla llama.cpp and ik_llama.cpp on the same model with the same best configuration found in previous phases.

**Six-step process:**

1. **Vanilla llama.cpp baseline** — Run 3-measurement median with the best config from phases 4–6 using the vanilla server. This is the "optimized llama.cpp" number.
2. **IK same config, no IK flags** — Run ik_llama-server with identical parameters but no IK-specific flags. This isolates the base performance difference between the two builds.
3. **IK feature pack** — Add `-mla 2` (MLA attention), `-fmoe 1` (fused MoE), `-rtr 1` (run-time repack), `-amb 512` (attention max batch). This is the expected best-case IK configuration.
4. **Attn-max-batch sweep** — Test `-amb` values of 128, 256, 512, 1024. The optimal value depends on available VRAM and batch size.
5. **SER sweep** (MoE only) — Test smart expert reduction with N=7, 6, 5 (dropping 1–3 experts per token). Higher N = faster but lower quality.
6. **MLA mode sweep** (MoE only) — Test MLA modes 2 (CPU+GPU) and 3 (CPU-only v2). Primarily beneficial for DeepSeek-architecture models with dedicated MLA tensors.

The best result from all six steps is recorded as `ik_best_tps` with its configuration labeled (e.g., "IK amb=512 SER=7,1").

**Results saved to:** `results/<model-slug>/ik_contrast_results.json`

---

## The Scoring System

Every benchmark measurement produces a composite score that guides the optimizer's next suggestion.

### Measurement

Each trial generates three measurements by default:
1. A **short prompt** (50 tokens output) for generation speed and TTFT
2. A **large prompt** (filling 90% of context, 200 tokens output) for long-context throughput
3. A **VRAM snapshot** at peak usage

### Adaptive measurement

To avoid spending 10 minutes on clearly bad configurations:
- **Pass 1:** Single quick measurement for every candidate
- **If score < 70% of best:** Return immediately (fast filter)
- **If competitive:** Two more measurements for median stability, plus large-prompt and VRAM measurement

This means good configurations get careful 3-measurement validation while bad configurations are discarded in ~30 seconds.

### Composite score formula

```
With large-prompt data (promoted configs):
  score = gen_tps × 0.35
        + long_tps × 0.25
        + pp_factor × 0.15    # prompt processing, normalized
        + ttft_factor × 0.15  # time-to-first-token, normalized
        + vram_factor × 0.10  # VRAM efficiency

Without large-prompt data (quick filter):
  score = gen_tps × 0.60
        + pp_factor × 0.25
        + ttft_factor × 0.15
```

**Why not just maximize tokens/second?**
- A config that generates at 15 t/s but takes 8 seconds for the first token feels slow in practice
- Prompt processing speed matters when you're re-reading long documents
- VRAM efficiency matters if you want to run other applications simultaneously
- Long-context throughput matters for document summarization and multi-turn conversations

The 35/25/15/15/10 weights reflect these priorities for typical interactive use.

---

## The GP-Bayesian Optimizer

The optimizer uses Gaussian Process Expected Improvement rather than random search or grid search. This matters because the parameter space for compute and memory phases has ~25 parameters with complex interactions — random search would need thousands of trials to explore it adequately.

### How it works

1. **Startup phase** (first 10–15 trials): Random sampling to establish initial coverage
2. **GP fitting:** A Matern-5/2 kernel GP is fit to all completed trials, learning the shape of the performance surface
3. **Acquisition:** The GP predicts performance *and* uncertainty across the parameter space. Configurations with high predicted performance *or* high uncertainty (unexplored regions) are prioritized via Expected Improvement
4. **Encoding:** All parameters are encoded to [0,1] before GP fitting. Integer parameters snap to their step grid after decoding.

### Early stopping

Rather than running a fixed number of trials, a custom `GPStoppingCallback` evaluates the maximum Expected Improvement remaining after each trial. When the GP is confident no untested configuration can beat the current best by more than 0.5% of its value, optimization stops early. This typically saves 20–40% of configured trials on well-explored spaces.

The callback also enforces two safety conditions:
- Never stop early if the best score hasn't beaten the naked-engine baseline (keep searching if we haven't improved anything)
- Never stop before 30 trials minimum (ensures the GP has enough data)

### Duplicate detection

Before starting a server restart for any trial, the optimizer checks if the exact same parameter combination has been tested before (common when the GP converges). Cached scores are returned immediately, avoiding expensive server restarts.

---

## HTML Report

The HTML report is generated by `generate_report.py` and combines local benchmark data with metadata fetched from HuggingFace.

### Local data merged

- Per-model benchmark results (best TPS, baseline TPS, gain %, best config)
- Topology sweep results (case, winning scenario, all scenario scores)
- Context ceiling results (GPU ctx, RAM ctx, recommended ctx)
- GGUF metadata (architecture, layer count, KV head count, context length, MoE expert count)
- IK contrast results (IK best TPS, gain vs llama.cpp, best IK config)
- Quantization recommendations (which alternative quants would fit your hardware better)

### HuggingFace data merged

For each model, the report fetches:
- Model description, author, license, downloads, likes
- Parameter count (from safetensors metadata, model card, README, or repo name)
- Benchmark scores from the Open LLM Leaderboard (both v1 and v2)
- LM Arena ELO score
- Artificial Analysis intelligence index, output speed, and TTFT (for reference API models)

**Benchmark versions:**
- **V1 leaderboard** (2022–mid 2024): ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K
- **V2 leaderboard** (June 2024+): ARC, BBH, MATH-Hard, GPQA, MMLU-Pro

Benchmarks are merged into a unified "Local Score" that normalizes V2 scores to V1-equivalent scale (×0.958 + 35.6 calibration derived from overlapping models) so models across both leaderboard eras can be compared.

**HF cache:** Metadata is cached in `batch_reports/hf_cache.json` and refreshed only when both conditions are true: the cache entry is older than 7 days AND the model file was modified within the last 30 days. Use `--refresh-hf` to force a full re-fetch.

### Reference models

The report includes 22 reference commercial and open-weights models (GPT-4o, Claude 3.5, Gemini 2.5, DeepSeek R1, Qwen3, GLM-5, Kimi K2, etc.) for comparison. Toggle them on/off with the "Compare reference models" checkbox in the report UI. Reference models show Arena ELO, Artificial Analysis intelligence index, output speed, TTFT, and API cost columns that are hidden for local models.

### Report features

- **Sortable columns:** Click any header to sort; click again to reverse; click again to reset
- **Text search:** Filter by model name, architecture, quantization level, or HF description
- **Case filter:** Show only Case A/B/C/D models
- **Status filter:** Show only optimized models or only failed models
- **Expandable rows:** Click ▶ on any row to see full detail: architecture breakdown, context ceilings, all topology scenarios, IK contrast results, quantization recommendations, and HF tags
- **Quantization tooltips:** Hover over the quant column to see recommended alternative quantizations and what case they'd fall into on your hardware
- **IK columns:** IK t/s and IK gain columns color-coded green (IK faster) / yellow (neutral) / red (IK slower), with tooltip showing which IK config won

### Generating the report manually

```bash
# From latest batch report
python generate_report.py

# From specific report
python generate_report.py --report batch_reports/batch_report_20260412_194529.json

# With HuggingFace benchmarks (requires token)
python generate_report.py --hf-token hf_yourReadToken

# Offline (no HF fetch)
python generate_report.py --no-hf
```

---

## Understanding Your Results

### What "best t/s" actually means

The reported TPS is measured on a specific test prompt (Python binary search function, ~50 token output) at temperature 0.4. Real-world TPS varies by:
- **Output length:** Longer generations amortize the TTFT cost
- **Prompt complexity:** Very long prompts spend more time in prompt processing
- **Temperature and sampling:** Higher temperature + top-k adds overhead
- **Content type:** Speculative decoding works much better on repetitive/structured text

For interactive conversational use, the reported TPS is a reasonable approximation. For batch document processing, the `tps_long` value (large-prompt throughput) is more representative.

### Baseline vs Best

"Stock t/s" is the speed with a completely unconfigured server — just `-ngl 99 -c 4096` and nothing else. The "Gain %" shows how much the optimizer improved over this. Gains of 10–40% are typical for GPU-resident models; gains of 100–300% are common when speculative decoding is enabled for code-generating models.

### Model Cases and their implications

**Case A** (fits both GPUs independently): You have the luxury of full GPU acceleration on either card. The optimizer will test both and pick the faster one — often the larger card wins, but sometimes the card with faster PCIe lanes or better cooling wins.

**Case B** (largest GPU only): The model fits on your biggest GPU but not the smaller one. Optimization focuses on maximizing single-GPU performance.

**Case C** (split required): Both GPUs are needed. Tensor split ratios matter significantly — the optimizer tests four strategies and the winner is used for all subsequent phases. Pay attention to the split scenario scores in the detail panel; if `split_prop` barely beats `split_equal`, the split ratio isn't critical. If it wins by 20%+, tensor balance matters for this model.

**Case D** (CPU offload required): The model is larger than your combined VRAM. Performance is fundamentally limited by CPU memory bandwidth. The optimizer will find the optimal GPU/CPU split, NUMA policy, and MoE threading, but cannot overcome the physics of RAM bandwidth. For 95+ GB models on consumer hardware (2× 24 GB GPUs), expect 5–15 t/s for Q3-Q4 quants.

### Context ceiling interpretation

- `ctx_gpu` in the report is the largest context that fits entirely in VRAM at the recommended topology. Use this as your `-c` value for best performance.
- `ctx_ram` is the largest context achievable with KV cache spilling to RAM. Much slower but allows long-document work.
- If `ctx_gpu` is much smaller than the model's trained context (e.g., 32k vs 196k), you're VRAM-limited. Reducing KV cache quantization (from f16 to q8_0 or q4_0) directly expands available context.

### Quantization recommendations

The report's detail rows and tooltips suggest alternative quantizations for each model based on your specific hardware. Recommendations are direction-labeled:
- ↑ **Upgrade** — higher quality quant that still fits your hardware
- ↓ **Downgrade** — smaller quant that changes the hardware case (e.g., moves from Case C to Case B, enabling full single-GPU operation)
- ↔ **Sidegrade** — similar quality at different size

A downgrade recommendation showing `Q4_K_M → Case A` is often worth taking: moving from a model that requires CPU offload to one that fits entirely in VRAM can be a 3–5× speed improvement.

---

## Troubleshooting

**Server won't start:**
- Check that `LLAMA_SERVER` points to a valid binary with execute permission
- Try `--verbose` to see the full loading output
- Model file might be incomplete — verify file size matches expected

**All trials return 0.0:**
- The server is starting but `/completion` requests are failing
- Try `--verbose` and look for error messages
- Context size might be too large for available VRAM — start with `-c 4096`

**"Speculative decoding context not initialized":**
- This is a log message, not an error. Speculative decoding is working.
- If you see this with `spec_type=ngram-*`, the lookup cache file may need to be pre-created

**Very slow optimization (>15 min per trial):**
- Your model is Case D with significant CPU offload. Each server restart takes 2–5 minutes for a 50+ GB model.
- The optimizer automatically halves trial counts when it detects models running below 15 t/s
- Use `--preset fast` for initial exploration, then `--preset standard --resume` once you know the model is worth optimizing

**HuggingFace benchmark fetch fails:**
- Ensure `HF_TOKEN` is set to a valid **read** token (not fine-grained) from `huggingface.co/settings/tokens`
- The leaderboard datasets require authentication
- Use `--no-hf` to skip and still generate a report with local data only

**OOM during context sweep:**
- Expected. The sweep intentionally probes sizes that cause OOM to find the boundary.
- If the server crashes repeatedly at small context sizes (≤8192), your model may have a different VRAM layout than estimated. The sweep will still find the correct ceiling.

**Results seem wrong / lower than expected:**
- Check `--verbose` output for the actual command being run
- Look for the `[debug] cmd:` lines in the log — these show the exact flags passed
- The baseline measurement uses the naked engine (no optimization flags). If your baseline TPS is much lower than expected, check that `llama_params_fit` is not silently reducing `n_gpu_layers`

**Resume not working:**
- Use `--resume` flag explicitly
- Check that `results/<model-slug>/` directory exists and contains at least one `*_results.json` file
- Use `--retry` to explicitly re-run failed models

---

## Example Report Output

Below is an illustrative example of what the HTML report's data looks like for a typical multi-model optimization run on a dual-GPU system.

```
===================================================================================================================================
  FINAL BATCH REPORT  —  Generated 2026-04-15 14:22
  Optimized: 8   No results: 0
===================================================================================================================================
  #   Model                                Quant    Case   Stock   Best   Gain   IK t/s  IK gain   Topo winner
  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1.  Qwen2.5-72B-Instruct-Q4_K_M         Q4_K_M   B      18.4    31.2   +70%   38.1    +22%      gpu0_only
  2.  Llama-3.1-70B-Instruct-Q5_K_M       Q5_K_M   C      12.1    24.8   +105%  29.4    +19%      split_prop
  3.  DeepSeek-R1-8B-Q8_0                 Q8_0     A      44.2    67.3   +52%   71.2    +6%       gpu1_only
  4.  Qwen3-30B-A3B-Q4_K_M               Q4_K_M   B      15.3    22.1   +44%   28.7    +30%      gpu0_only
  5.  Mistral-7B-Instruct-v0.3-Q4_K_M    Q4_K_M   A      58.3    82.4   +41%   84.1    +2%       gpu0_only
  6.  phi-3-medium-128k-instruct-Q5_K_M  Q5_K_M   A      36.1    51.2   +42%   52.8    +3%       gpu0_only
  7.  MiniMax-M2.7-UD-Q3_K_XL            Q3_K_XL  D       4.0     8.8  +120%   11.2    +27%      numa_distribute
  8.  gemma-3-27b-it-Q4_K_M              Q4_K_M   B      22.4    38.9   +74%   40.1    +3%       gpu0_only
===================================================================================================================================
```

**Expanded detail for Qwen2.5-72B-Instruct-Q4_K_M (row 1):**

```
  Model:            Qwen2.5-72B-Instruct
  Architecture:     qwen2  |  80 layers  |  8 KV heads  |  128 head dim
  Parameters:       72.7B
  Train context:    128k tokens
  GGUF quant:       Q4_K_M (4.8 bpw)  |  ~42 GB file

  Context ceilings:
    GPU single:       16,384 tokens  (fits in GPU0, 24 GB)
    Recommended:      16,384 tokens  ← use this as -c

  Topology:
    Winner: gpu0_only — GPU0 (RTX 3090, 24 GB)
    Scenario scores: gpu0_only=18.4  gpu1_only=16.2  (GPU0 is 13% faster)

  Best llama.cpp config:
    -ngl 99 -c 16384 -b 512 --ubatch-size 512
    -t 8 -tb 32
    --spec-type ngram-map-k4v --draft 24 --draft-min 4 --draft-p-min 0.85
    --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
    --mlock --no-mmap

  IK_llama.cpp contrast:
    llama.cpp best:      31.2 t/s
    IK same config:      32.1 t/s   (+2.9%)
    IK feature pack:     35.8 t/s   (+14.7%, fused-MoE + RTR)
    IK best (amb=512):   38.1 t/s   (+22.1%)  ← IK winner

  HuggingFace benchmarks (Open LLM Leaderboard v2):
    MMLU: 85.3%   GSM8K: 88.0%   MATH: 83.1%   GPQA: 49.0%
    BBH: 72.0%    HumanEval: 72.0%
    Local Score: 78.4 (v2)   |   LB Rank: #142 (v2)

  Quant recommendations:
    ↑ Q5_K_M   ~54 GB  Case C  size +29%  spd=90  qual=78  upgrade
    ↓ Q3_K_M   ~31 GB  Case A  size -26%  spd=95  qual=62  downgrade — would fit BOTH GPUs
    ↔ Q4_K_S   ~40 GB  Case B  size -5%   spd=100 qual=70  sidegrade
```

**What this tells you:**

The Qwen2.5-72B is Case B — it fits on the larger RTX 3090 but not the smaller GPU. The optimizer found a 70% speed improvement primarily from speculative decoding (`ngram-map-k4v` with 24-token drafts) and quantized KV cache (halved KV memory, allowing 16k context vs 8k at f16). IK_llama.cpp adds another 22% on top via run-time tensor repacking. The Q3_K_M downgrade recommendation is interesting: moving to ~31 GB would make the model Case A (fits both GPUs independently), allowing you to run it on the faster GPU1 if GPU0 is busy — though at meaningfully lower quality.

---

## Acknowledgements

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — the inference engine this optimizer wraps
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) — extended optimizations for hybrid CPU/GPU inference
- [Optuna](https://optuna.org/) — hyperparameter optimization framework providing the GP sampler infrastructure
- [HuggingFace](https://huggingface.co/) — model cards, leaderboard data, and benchmark results
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) — V1 and V2 benchmark data
- [lmarena.ai](https://lmarena.ai/) — Arena ELO scores for reference models
- [Artificial Analysis](https://artificialanalysis.ai/) — intelligence index and API performance metrics for reference models
