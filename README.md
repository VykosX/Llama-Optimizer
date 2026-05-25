# Llama Optimizer

> **Automatically find the fastest possible settings for running large language models on your machine.**

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
8. [The Scoring System](#the-scoring-system)
9. [The GP-Bayesian Optimizer](#the-gp-bayesian-optimizer)
10. [HTML Report](#html-report)
11. [Understanding Your Results](#understanding-your-results)
12. [Troubleshooting](#troubleshooting)
13. [Example Report Output](#example-report-output)

---

## Quick Start Guide

> **New to this? Start here.** Plain language, no deep LLM knowledge required.

### What problem does this solve?

When you run a large language model locally using llama.cpp, dozens of settings affect how fast the model generates text — GPU layer counts, thread allocations, batch sizes, KV cache quantization, speculative decoding, MTP draft depth, and more. Getting these wrong can mean your model runs at 3 tokens/second instead of 15. **This tool automatically tests thousands of combinations and finds the fastest settings for your specific hardware and model.**

### What you need

- **Windows or Linux** with Python 3.10+
- **`llama-server.exe`** (or `llama-server`) from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases)
- **One or more `.gguf` model files**
- **An NVIDIA GPU** (optional but strongly recommended)
- **Optional:** `ik_llama-server` from [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) for additional performance on CPU-heavy configs

### Two ways to run — pick whichever suits you

**Option A — Using command-line switches (no environment variables needed):**

```powershell
# Windows
python batch_runner.py ^
  --llama-server "C:\path\to\llama-server.exe" ^
  --models-dir "C:\path\to\models" ^
  --preset standard ^
  --topo-sweep ^
  --html-report
```

```bash
# Linux / macOS
python batch_runner.py \
  --llama-server /usr/local/bin/llama-server \
  --models-dir /home/user/models \
  --preset standard \
  --topo-sweep \
  --html-report
```

**Option B — Using environment variables (convenient for repeated runs):**

```powershell
# Windows PowerShell
$env:LLAMA_SERVER = "C:\path\to\llama-server.exe"
$env:LLM_OPT_MODELS_DIR = "C:\path\to\models"
python batch_runner.py --preset standard --topo-sweep --html-report
```

```bash
# Linux / macOS
export LLAMA_SERVER="/usr/local/bin/llama-server"
export LLM_OPT_MODELS_DIR="/home/user/models"
python batch_runner.py --preset standard --topo-sweep --html-report
```

Both methods work identically. Command-line switches always override environment variables.

### With IK_llama.cpp (optional, for extra speed)

```powershell
python batch_runner.py ^
  --llama-server "C:\path\to\llama-server.exe" ^
  --ik-llama-server "C:\path\to\ik_llama-server.exe" ^
  --models-dir "C:\path\to\models" ^
  --preset ik ^
  --topo-sweep ^
  --html-report
```

### With MTP (for Qwen3.5/3.6, DeepSeek V3/R1, Gemma 4 models)

MTP is detected automatically from GGUF metadata. Use the `mtp` preset to include the MTP sweep, or `--force-mtp` to test any model:

```powershell
python batch_runner.py ^
  --llama-server "C:\path\to\llama-server.exe" ^
  --models-dir "C:\path\to\models" ^
  --preset mtp ^
  --topo-sweep ^
  --html-report

# Force MTP even if not detected in metadata
python batch_runner.py --preset mtp --force-mtp --filter "Qwen3"
```

### What you get when it's done

Open `batch_reports/report_latest.html` in any browser. You'll see:

| Column | What it means |
|--------|--------------|
| **Best t/s** | Fastest tokens per second found |
| **Stock t/s** | Speed with zero optimization |
| **Gain %** | How much faster the optimizer made it |
| **MTP t/s** | Speed with MTP draft enabled (if applicable) |
| **MTP gain** | How much MTP helped |
| **IK t/s** | Speed with ik_llama.cpp (if configured) |
| **IK gain** | How much ik_llama.cpp helped vs vanilla llama.cpp |
| **Case** | How the model fits your GPU(s): A=both, B=one, C=split, D=needs RAM |
| **ctx GPU** | Maximum context that fits fully in VRAM |

### How long does it take?

| Preset | Time/model | What it does |
|--------|-----------|--------------|
| `fast` | ~25 min | Quick compute + memory sweep |
| `standard` | ~1–2h | Full compute + memory optimization |
| `mtp` | ~2–3h | Standard + MTP draft sweep |
| `ik` | ~2–3h | Standard + IK contrast |
| `thorough` | ~3–4h | Full + re-validation audits |
| `full_plus` | ~5–6h | Everything: audits + quality + IK + MTP |

Press **Ctrl+C** at any time to skip the current phase. Results are saved after every trial — you can always resume.

---

## What This Does

Llama Optimizer is a multi-phase automated benchmarking and optimization system for locally-run large language models. It:

- **Characterizes your hardware** — Topology sweep classifies each model (Case A–D) based on VRAM fit, tests GPU configurations (single GPU, split, NUMA), and binary-searches the maximum stable context window.
- **Finds optimal inference parameters** — Gaussian Process Bayesian optimization intelligently explores 25+ parameters per phase (threads, batch sizes, KV quantization, speculation, MTP depth), learning from every trial to converge on the fastest configuration.
- **Benchmarks MTP** — Detects MTP heads from GGUF metadata and runs a structured 6-step sweep: draft depth scan → acceptance probability sweep → ubatch re-test → n_min test → final validation.
- **Benchmarks IK_llama.cpp** — Structured contrast between vanilla llama.cpp and IK's exclusive features (MLA attention, fused MoE, run-time repack, SER).
- **Generates an actionable HTML report** — Merges local benchmarks with HuggingFace metadata (benchmarks, parameters, license), provides quantization recommendations, fully sortable and filterable.

---

## Requirements

### Software
- Python 3.10+
- `llama-server` from [llama.cpp](https://github.com/ggerganov/llama.cpp/releases)
- Python packages (auto-installed if missing): `requests`, `optuna`, `numpy`, `scipy`, `scikit-learn`, `psutil`, `pynvml`

### Hardware
- **GPU:** NVIDIA with CUDA 11.8+. 8+ GB VRAM minimum. CPU-only works but is much slower.
- **RAM:** 16 GB minimum, 64+ GB for large models needing CPU offload.
- **CPU:** Any x86-64. AVX2/AVX-512 helps significantly for CPU-offloaded models.

### Models
Any `.gguf` compatible with llama.cpp: single-file, multi-shard, dense, MoE, hybrid/SSM, MTP-capable.

---

## Installation

```bash
git clone https://github.com/VykosX/Llama-Optimizer
cd Llama-Optimizer
pip install requests optuna numpy scipy scikit-learn psutil pynvml
```

No other steps required. All files run directly from the cloned directory.

---

## File Overview

```
Llama-Optimizer/
├── batch_runner.py         Entry point — run this
├── optimizer_adapter.py    Bridge: RunConfig → optimize.py phase calls
├── optimize.py             Core GP-Bayesian optimizer, all phase logic
├── sweep_engine.py         GPU topology sweep + context ceiling sweep
├── model_utils.py          GGUF metadata, MTP detection, GPU/RAM helpers
├── generate_report.py      Self-contained HTML report generator
├── test_benchmarks.py      Standalone HF benchmark fetch tester
├── results/                Per-model results (auto-created)
├── batch_reports/          Batch summary reports (auto-created)
└── logs/                   Run logs (auto-created)
```

---

## Running the Optimizer

### Basic Usage

```bash
# Standard run — specify everything on the command line
python batch_runner.py \
  --llama-server /path/to/llama-server \
  --models-dir /path/to/models \
  --preset standard \
  --topo-sweep \
  --html-report

# Resume an interrupted run
python batch_runner.py --preset standard --topo-sweep --resume

# Test a single model
python batch_runner.py --filter "Qwen3" --preset standard --topo-sweep

# Report only (no benchmarking)
python batch_runner.py --report-only --html-report

# MTP sweep with force flag (for models without confirmed MTP metadata)
python batch_runner.py --preset mtp --force-mtp --filter "Qwen3.6-27B"

# Full everything — IK + MTP + quality sampling
python batch_runner.py \
  --llama-server /path/to/llama-server \
  --ik-llama-server /path/to/ik_llama-server \
  --preset full_plus \
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
| `LLM_OPT_STARTUP_TIMEOUT` | Server startup timeout (auto-scaled by model size) |
| `HF_TOKEN` | HuggingFace read token for benchmark data in reports |
| `GPU0_VRAM_GB` | Override detected GPU0 VRAM in GB |
| `GPU1_VRAM_GB` | Override detected GPU1 VRAM in GB |

### All Command-Line Switches

#### Path Options

| Switch | Default | Description |
|--------|---------|-------------|
| `--llama-server PATH` | Auto-detected or `LLAMA_SERVER` | Path to `llama-server` binary |
| `--ik-llama-server PATH` | Auto-detected or `IK_LLAMA_SERVER` | Path to `ik_llama-server` binary |
| `--models-dir PATH` | `./models` or `LLM_OPT_MODELS_DIR` | Directory to scan for `.gguf` files (recursive) |
| `--results-base PATH` | `./results` | Root directory for per-model results |
| `--reports-dir PATH` | `./batch_reports` | Directory for batch summary reports |
| `--logs-dir PATH` | `./logs` | Directory for run log files |

#### Model Selection

| Switch | Description |
|--------|-------------|
| `--filter TEXT` | Only process models whose filename contains TEXT (case-insensitive) |
| `--resume` | Skip models that already have complete results |
| `--retry [first\|last]` | Retry failed models. `first` = before new models, `last` = after |
| `--dry-run` | List models that would run, then exit |

#### Preset and Phase Control

| Switch | Description |
|--------|-------------|
| `--preset NAME` | Optimization preset (see table below) |
| `--phases PHASE...` | Override preset phase list, e.g. `--phases compute memory mtp_spec` |
| `--skip-phases [PHASE...]` | Skip specific phases. Bare `--skip-phases` skips all except `--rerun-phases` |
| `--rerun-phases PHASE...` | Force re-run named phases even if results exist |
| `--trials PHASE=N...` | Override trial counts, e.g. `--trials compute=40 memory=40` |

#### Run Mode

| Switch | Default | Description |
|--------|---------|-------------|
| `--timeout SECONDS` | 5400 | Per-model hard timeout |
| `--trial-timeout SECONDS` | 360 | Per-trial hard timeout |
| `--port PORT` | 8090 | llama-server HTTP API port |
| `--force-mtp` | — | Run MTP sweep even when MTP heads not detected in GGUF |
| `--report-only` | — | Regenerate HTML report from existing results only |
| `--html-report` | — | Generate HTML report after batch completes |
| `--no-hf` | — | Skip HuggingFace metadata fetch |
| `--refresh-hf` | — | Force re-fetch all HF metadata |
| `--no-log` | — | Disable automatic log file |
| `--verbose` | — | Show live llama-server loading output |
| `--interactive` | — | Pause 5s between models; press `n` to stop after current |

#### Topology Sweep

| Switch | Default | Description |
|--------|---------|-------------|
| `--topo-sweep` | — | Run GPU topology benchmark before optimizer |
| `--topo-only` | — | Topology sweep only, skip optimizer |
| `--topo-runs N` | 2 | Benchmark runs per topology scenario |
| `--gpu-filter SCENARIO...` | — | Only test specific scenario IDs |
| `--skip-gpu GPU...` | — | Skip GPU by index (0, 1) or name fragment |
| `--force-numa` | — | Force NUMA tests even for single-GPU models |

#### Context Ceiling Sweep

| Switch | Description |
|--------|-------------|
| `--ctx-sweep` | Binary-search maximum stable context after topology |
| `--ctx-only` | Context sweep only |
| `--skip-ctx-b` | Skip RAM-spill context tests |

#### Hardware Overrides

| Switch | Description |
|--------|-------------|
| `--gpu0-vram FLOAT` | Override detected GPU0 VRAM in GB |
| `--gpu1-vram FLOAT` | Override detected GPU1 VRAM in GB |

### Preset Reference

| Preset | Phases | Est. Time/Model | Best For |
|--------|--------|-----------------|----------|
| `fast` | binary_screen → fast_gpu → fast_moe → compute(30) → memory(25) → integrity → reasoning | ~25 min | Quick first pass, large collections |
| `standard` | gpu → moe → compute(60) → memory(60) | ~1–2h | General purpose |
| `thorough` | standard + moe_audit + compute_audit + memory_audit | ~3–4h | Final optimization |
| `full` | thorough + quality(80) | ~4–5h | Including sampling params |
| `moe_deep` | thorough + expert count sweep | ~4h | MoE models needing expert tuning |
| `ik` | standard + ik_contrast | ~2–3h | IK_llama.cpp comparison |
| `ik_thorough` | thorough + ik_contrast | ~4–5h | Full IK comparison |
| `mtp` | standard + mtp_spec | ~2–3h | MTP-capable models |
| `mtp_thorough` | thorough + mtp_spec | ~4–5h | Full MTP optimization |
| `full_plus` | thorough + quality + ik_contrast + mtp_spec | ~5–6h | Complete: everything |

Override trial counts for any preset: `--trials compute=30 memory=30` for a faster pass.

### IK_llama.cpp Support

[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) is a fork with additional optimizations valuable for hybrid CPU/GPU setups:

| Feature | Flag | Benefit |
|---------|------|---------|
| MLA Attention | `-mla 2` | 50% KV cache reduction for DeepSeek-architecture models |
| Fused MoE | `-fmoe` | 20–80% faster expert routing on GPU-resident MoE models |
| Run-time Repack | `-rtr` | 50–80% faster CPU execution via SIMD tensor layout optimization |
| Attention Max Batch | `-amb N` | Reduces K×Q compute buffer pressure at long context |
| Smart Expert Reduction | `-ser N,1` | Drop N experts/token, trading some quality for speed |

**Modes:**
- **IK-only** (only `IK_LLAMA_SERVER` set): all phases run with IK automatically
- **Dual mode** (both servers set): llama.cpp runs all optimization phases, then `ik_contrast` compares the two head-to-head on the best found config

### MTP (Multi-Token Prediction) Support

MTP is a form of speculative decoding where the prediction heads are baked directly into the GGUF — no separate draft model needed. A single forward pass produces the main token plus N draft tokens, all verified in parallel.

**Supported architectures:**

| Architecture | GGUF key | Models |
|-------------|---------|--------|
| `qwen35` | `qwen35.nextn_predict_layers` | Qwen3.5, Qwen3.6 |
| `deepseek3` / `deepseek2` | `deepseek3.nextn_predict_layers` | DeepSeek V3, R1 |
| `gemma4` | `gemma4.nextn_predict_layers` | Gemma 4 |

**Detection hierarchy:**
1. **GGUF metadata** (highest confidence) — reads `nextn_predict_layers` key directly from the file header. This is definitive.
2. **Filename pattern** — matches `-MTP-`, `.mtp.`, `-mtp.gguf` etc. Medium confidence.
3. **Architecture hint** — model family is MTP-capable but key not found. Reports as `arch_hint`, does not auto-enable MTP; requires `--force-mtp`.

**The `--force-mtp` flag** bypasses detection and runs the MTP sweep regardless. Use this for:
- Models from MTP-capable families where the GGUF conversion omitted the metadata key
- Testing whether any model benefits from the spec-type flags
- The `Ex0bit/Qwen3.6-27B-PRISM-PRO-DQ` model (Qwen3.6 architecture supports MTP even though the filename doesn't include "MTP")

```bash
python batch_runner.py --preset mtp --force-mtp --filter "PRISM-PRO"
```

**What the MTP sweep tests:**

1. **Baseline** — best config without MTP (3-run median)
2. **Spec-type probe** — detects whether this llama.cpp build uses `--spec-type mtp` or `draft-mtp`
3. **Draft depth scan** — n_max = 1, 2, 3 (tokens drafted per step)
4. **p_min sweep** — acceptance probability 0.0 → 0.30 → 0.50 → 0.70 → 0.85 → 0.95 on best n_max
5. **ubatch re-test** — 256, 512, 1024 on best n_max + p_min (MTP verification benefits from larger ubatch)
6. **n_min test** — 0 vs 1 minimum accepted draft tokens
7. **Final validation** — 5-run median on overall winner for stability

---

## How It All Works — Phase by Phase

The optimizer runs phases sequentially. Each phase seeds the next from its best result — coordinate descent: optimize one parameter group at a time with everything else locked, then rotate.

```
Topology Sweep ──► Context Ceiling Sweep
                         │
                         ▼
                   GPU Offload (dense models)
                         │
                         ▼
                   MoE Thread Sweep (MoE models)
                         │
                         ▼
                   Expert Count (optional, MoE)
                         │
                         ▼
                   Compute Allocation  ◄── GP-Bayesian (25 params)
                         │
                         ▼
                   Memory & Throughput  ◄── GP-Bayesian (9 params)
                         │
                         ▼
                 MoE Audit → Compute Audit → Memory Audit
                         │
                         ▼
                   Quality / Sampling  ◄── GP-Bayesian (sampling params)
                         │
                         ├──► IK_llama Contrast (if configured)
                         │
                         └──► MTP Draft Sweep (if MTP detected or --force-mtp)
```

### Phase 0: Topology Sweep

Classifies each model into Cases A–D based on VRAM fit, then tests the relevant GPU scenarios:

| Case | Condition | Scenarios Tested |
|------|-----------|-----------------|
| A | Fits both GPUs independently | gpu0_only, gpu1_only — picks faster |
| B | Fits GPU0 only | gpu0_only |
| C | Requires combined VRAM | split_prop, split_equal, split_g0heavy, split_kv_aware |
| D | Exceeds combined VRAM | Binary-search max ngl, then numa_none / numa_distribute / numa_isolate |

A **needle-in-a-haystack coherence test** verifies each scenario actually produces correct output, not just that the server starts. A hidden secret number is planted in a long passage; the model must retrieve it.

### Phase 0.5: Context Ceiling Sweep

Binary-searches maximum stable context for each topology. Uses a VRAM formula to predict the ceiling first, probing that value before binary-searching up or down — typically finds the answer in 3–5 probes. Measures GPU-only ctx, combined-GPU ctx, and RAM-spill ctx separately.

### Phase 1: GPU Offload

Dense models only. Middle-out sweep of `n_gpu_layers` from 0 to total layers. Each direction (up/down from midpoint) stops independently when performance drops below 50% of best. MoE models skip this phase — their CPU/GPU split is handled by n_cpu_moe and override-tensor in later phases.

### Phase 2: MoE Thread Sweep

MoE models only. Sweeps `--n-cpu-moe` (dedicated MoE routing threads) using the same middle-out approach. Always re-validates the ±2 neighborhood of the winner with fresh 3-run measurements.

### Phase 3: Expert Count Sweep

Optional (preset `moe_deep` only). Sweeps `expert_used_count` with a **quality gate**: each candidate is measured for speed AND tested on two GPQA Diamond graduate-level science questions. Configurations that increase uncertain token counts by more than 3% are disqualified regardless of speed gain.

### Phase 4: Compute Allocation

GP-Bayesian search over 25 compute parameters: threads, polling intervals, process priority, CPU affinity, and the full speculative decoding parameter space (spec type, n-gram window, draft size, acceptance probability). Two known-good configurations are seeded as the first trials to give the GP solid starting points.

### Phase 5: Memory & Throughput

GP-Bayesian search over 9 memory parameters: batch sizes, flash attention, KV cache quantization (f16/bf16/q8_0/q5_1/q4_0), SWA mode, repack, operation offload, mlock, mmap. Quantized KV requires flash attention — invalid combinations are pruned before server restart.

### Phase 6: Audit Phases

Three re-validation passes catch parameter interactions that single-pass coordinate descent misses. Run in order: MoE Audit (re-tests ±2 MoE neighbors with compute+memory locked) → Compute Audit (full compute search on top of Memory results) → Memory Audit (full memory search on top of Compute Audit results). Typically finds 5–15% additional gains.

### Phase 7: Quality / Sampling

`full` and `full_plus` presets only. Optimizes sampling parameters (temperature, top-p/k, mirostat, repeat penalty, XTC, DRY, etc.) by measuring accuracy on 5 factual/reasoning tasks. GP maximizes correct answers rather than speed.

### Phase 8: IK_llama.cpp Contrast

Six-step structured comparison: vanilla llama baseline → IK same config (no IK flags) → IK feature pack (MLA + fused-MoE + RTR) → attn_max_batch sweep (128/256/512/1024) → SER sweep for MoE (7,1 / 6,1 / 5,1) → MLA mode sweep (mode 2 vs 3).

### Phase 9: MTP Draft Sweep

Six-step MTP optimization (see [MTP Support](#mtp-multi-token-prediction-support) above for full details). Runs after all llama.cpp optimization is complete so the base config is already optimal. Reports gain vs the no-MTP baseline using the same best config.

---

## The Scoring System

### Composite score formula

```
With large-prompt data (promoted configs):
  score = gen_tps    × 0.35   # short-prompt generation speed
        + long_tps   × 0.25   # large-prompt throughput (90% context fill)
        + pp_factor  × 0.15   # prompt processing speed, normalized
        + ttft_factor × 0.15  # time-to-first-token, normalized
        + vram_factor × 0.10  # VRAM efficiency

Without large-prompt (quick filter pass):
  score = gen_tps   × 0.60
        + pp_factor × 0.25
        + ttft_factor × 0.15
```

### Adaptive measurement

- **Pass 1 (all configs):** Single quick measurement
- **Score < 70% of best:** Return immediately — bad config filtered in ~30s
- **Score ≥ 70%:** Two more measurements for median stability + large-prompt + VRAM snapshot

This means promising configurations get careful 3-measurement validation while poor ones are discarded fast.

---

## The GP-Bayesian Optimizer

Uses a Gaussian Process with Matern-5/2 kernel and Expected Improvement acquisition. The GP learns the shape of the performance surface from completed trials, balancing exploitation (configurations predicted to be good) with exploration (high-uncertainty regions).

**Early stopping:** A custom `GPStoppingCallback` monitors maximum Expected Improvement after each trial. When the GP is confident no untested configuration can beat the current best by more than 0.5%, optimization stops early. Saves 20–40% of configured trials on well-explored spaces. Never stops early if nothing has beaten the naked baseline, and never before 30 trials minimum.

---

## HTML Report

Generated by `generate_report.py`. Self-contained HTML — works offline after generation, no external dependencies.

### Local data

Benchmark results, topology sweep (case, winner, all scenario scores), context ceilings, GGUF metadata (architecture, layers, KV heads, context length, MoE experts, MTP layers), IK contrast results, MTP sweep results, quantization recommendations.

### HuggingFace data

Model description, author, license, downloads, parameter count, Open LLM Leaderboard benchmarks (V1: ARC/HellaSwag/MMLU/TruthfulQA/Winogrande/GSM8K; V2: ARC/BBH/MATH/GPQA/MMLU-Pro), LM Arena ELO. Cached in `batch_reports/hf_cache.json`, refreshed after 7 days if model recently downloaded.

### 22 reference models

GPT-5.4, GPT-4.1, GPT-4o, Claude Opus/Sonnet/Haiku 4.x, Gemini 3.1/2.5 Pro/Flash, Grok 4.1, DeepSeek V3/R1, Qwen3.5 397B/235B/72B, GLM-5, Kimi K2, MiniMax M2, Nemotron Super/Nano. Toggle with the "Compare reference models" checkbox. Shows Arena ELO, Artificial Analysis intelligence index/speed/TTFT, and API cost columns.

### Generating the report manually

```bash
python generate_report.py                                      # latest report
python generate_report.py --report path/to/batch_report.json  # specific report
python generate_report.py --hf-token hf_yourReadToken         # with HF benchmarks
python generate_report.py --no-hf                             # offline
python generate_report.py --refresh-hf                        # force re-fetch
```

---

## Understanding Your Results

### Model Cases

| Case | Meaning | Performance Expectation |
|------|---------|------------------------|
| A | Fits both GPUs independently | Excellent — full GPU speed on either card |
| B | Fits GPU0 only | Good — full GPU speed on the larger card |
| C | Requires combined VRAM | Good — split-GPU with some PCIe overhead |
| D | Requires CPU offload | Limited by RAM bandwidth (~5–15 t/s for 90+ GB models on DDR4) |

### MTP results interpretation

- **+20–70% gain:** Strong MTP, recommended for daily use. Dense models in this range.
- **+5–20% gain:** Modest but real. Worth enabling. Typical for MoE models.
- **0–5% gain:** Marginal. May not be worth the config complexity.
- **Negative gain:** MTP was counterproductive. Usually means near-full VRAM (MTP heads need 2–5% extra) or the model was not trained with strong MTP objectives.

### Context ceiling interpretation

`ctx_gpu` is the maximum context fitting entirely in VRAM at the winning topology — use this as your `-c` value for best performance. If much smaller than the model's trained context, reducing KV quantization (f16 → q8_0 or q4_0) directly expands available context.

### Quantization recommendations

The report suggests alternative quants based on your hardware:
- **↑ Upgrade** — higher quality quant that still fits your hardware
- **↓ Downgrade** — smaller quant that moves to a better Case (e.g., D→C or C→B)
- **↔ Sidegrade** — similar quality at different size

A "↓ Q4_K_M → Case A" recommendation means dropping to Q4 would make the model fit in both GPUs independently — often a 3–5× speed improvement worth taking.

---

## Troubleshooting

**MTP phase skipped / "no MTP heads"**
- Check that your GGUF was converted with MTP heads included. Not all quantizations of MTP-capable models include the heads.
- Use `--force-mtp` to run the sweep anyway.
- For Qwen3.6 models, the architecture key should be `qwen35`. Verify with: `python -c "from model_utils import detect_mtp; from pathlib import Path; print(detect_mtp(Path('your_model.gguf')))"`

**MTP slower than baseline**
- Your VRAM may be nearly full. MTP heads require 2–5% extra VRAM for the auxiliary prediction buffers.
- Try reducing KV cache quantization first to free VRAM, then re-run MTP sweep.
- The model may have weak MTP training signal — not all MTP GGUFs have equally well-trained heads.

**IK_llama.cpp: server crashes with "unknown flag"**
- The `-rtr`, `-fmoe`, `-mla`, `-amb`, `-ser` flags are IK-specific. The IK contrast phase only adds these when `_ik_server=True`.
- Verify you're pointing `--ik-llama-server` at the ik_llama build, not vanilla llama.cpp.

**Server won't start**
- Verify path to binary with `--verbose`
- Model file may be incomplete — check file size
- Try `--force-mtp` without MTP first to confirm baseline works

**Very slow optimization (>15 min/trial)**
- Case D model with significant CPU offload. Each restart takes 2–5 min for 50+ GB models.
- Optimizer auto-halves trial counts for models below 15 t/s.
- Use `--preset fast` for initial exploration, then `--preset standard --resume`.

**HuggingFace benchmarks missing**
- Set `HF_TOKEN` to a **read** token from huggingface.co/settings/tokens (not fine-grained)
- Use `--no-hf` to skip and still generate a useful local-data-only report

---

## Example Report Output

```
========================================================================================
  FINAL BATCH REPORT  —  Generated 2026-05-01 14:22
  Optimized: 3   No results: 0
========================================================================================
  #   Model                         Quant    Case   Stock   Best   Gain  IK t/s  IK gain  MTP t/s  MTP gain  Topo winner
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1.  Qwen3.6-27B-PRISM-PRO-DQ     Q4_K_M   B      18.4    31.2   +70%   38.1    +22%     44.7      +43%     gpu0_only
  2.  DeepSeek-R1-8B-Q8_0          Q8_0     A      44.2    67.3   +52%   71.2     +6%     89.1      +32%     gpu1_only
  3.  MiniMax-M2.7-Q3_K_XL         Q3_K_XL  D       4.0     8.8  +120%   11.2    +27%      —          —      numa_distribute
========================================================================================
```

**Expanded detail for Qwen3.6-27B-PRISM-PRO-DQ:**

```
  Architecture:   qwen35  |  62 layers  |  8 KV heads  |  128 head dim
  Parameters:     27.4B   |  MTP: 1 prediction layer (detected from metadata)
  Train context:  131,072 tokens

  Context:        ctx_gpu = 32,768 tokens  (recommended)

  Best llama.cpp config:
    -ngl 99 -c 32768 -b 512 --ubatch-size 512
    -t 8 -tb 32
    --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
    --mlock --no-mmap

  MTP sweep results:
    Baseline (no MTP):     31.2 t/s
    MTP best:              44.7 t/s  (+43.3%)
      --spec-type mtp --spec-draft-n-max 2 --spec-draft-p-min 0.70
      --spec-draft-n-min 0 --ubatch-size 512

  IK contrast results:
    llama.cpp best:        31.2 t/s
    IK + MLA + fused-MoE:  38.1 t/s  (+22.1%)
    IK best (amb=512):     38.1 t/s

  HF Benchmarks:  MMLU: 84.5%  GSM8K: 94.0%  MATH: 92.0%  GPQA: 55.0%
```

**Reading this:** The Qwen3.6 model (Case B — fits GPU0 only) got a 70% gain from the optimizer's compute+memory tuning, another 22% from IK_llama.cpp, and a further 43% from MTP draft speculation — ultimately running at 44.7 t/s vs the 18.4 t/s stock speed, a combined **2.4× improvement** from baseline.

The MTP gain of 43% is strong for a 27B model — consistent with Qwen3.x's well-trained MTP heads. n_max=2 with p_min=0.70 means 2 draft tokens are predicted per step and accepted when the model is at least 70% confident, balancing acceptance rate against the cost of rejected drafts.

---

## License

MIT License. See `LICENSE` for full text.

---

## Acknowledgements

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — the inference engine this optimizer wraps
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) — extended optimizations for hybrid CPU/GPU inference
- [Optuna](https://optuna.org/) — hyperparameter optimization framework
- [HuggingFace](https://huggingface.co/) — model metadata, leaderboard data, benchmark results
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) — V1 and V2 benchmark data
- [lmarena.ai](https://lmarena.ai/) — Arena ELO scores
- [Artificial Analysis](https://artificialanalysis.ai/) — intelligence index and API performance metrics
