#!/usr/bin/env python3
"""
generate_report.py — Dynamic HTML report generator for LLM Optimizer results.

Produces a self-contained, sortable HTML report that merges:
  • Local benchmark results  (from batch_runner.py output JSON)
  • Local GGUF metadata      (architecture, layers, context, MoE info)
  • Hugging Face metadata    (description, params, license, benchmarks, ELO)

The HTML file is fully self-contained (no external dependencies) and works
offline after generation. All sorting and filtering is pure JavaScript.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Generate from the latest batch report
  python generate_report.py

  # Generate from a specific batch report JSON
  python generate_report.py --report batch_reports/batch_report_20260313_120000.json

  # Generate without fetching HF metadata (offline / fast)
  python generate_report.py --no-hf

  # Force re-fetch all HF metadata ignoring cache
  python generate_report.py --refresh-hf

  # Specify output path
  python generate_report.py --output my_report.html

  # Run as part of the batch wrapper (called automatically with --html-report)
  python batch_runner.py --topo-sweep --ctx-sweep --html-report

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HF CACHE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Cache lives at: <batch_reports>/hf_cache.json
  Re-fetch policy: only if BOTH conditions are true:
    1. Cache entry is older than 7 days
    2. Model file was modified within the last 30 days
       (recently downloaded models are more likely to have updated cards)
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

# ── optional deps ──────────────────────────────────────────────────────────────
def _pip(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:
    import requests
except ImportError:
    _pip("requests"); import requests

# ── constants ──────────────────────────────────────────────────────────────────
SCRIPT_DIR        = Path(__file__).resolve().parent

# HuggingFace token — needed to access leaderboard benchmark datasets.
# Set via env var HF_TOKEN or pass --hf-token to generate_report.py.
# Get a free read-only token at: https://huggingface.co/settings/tokens
HF_TOKEN: str | None = (os.environ.get("HF_TOKEN")
                        or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
HF_API            = "https://huggingface.co/api"
HF_CACHE_STALE_D  = 7    # days before a cache entry is considered stale
MODEL_RECENT_D    = 30   # days — model is "recently downloaded" if mtime < this
REQUEST_TIMEOUT   = 15   # seconds per HF API call
REQUEST_DELAY     = 0.3  # seconds between API calls (rate limit courtesy)

# Standard lm-evaluation-harness benchmark names as they appear on HF
BENCHMARK_KEYS = [
    ("ARC",        ["arc_challenge", "arc", "ARC-Challenge"]),
    ("HellaSwag",  ["hellaswag", "HellaSwag"]),
    ("MMLU",       ["mmlu", "MMLU"]),
    ("TruthfulQA", ["truthfulqa_mc2", "truthfulqa", "TruthfulQA"]),
    ("Winogrande", ["winogrande", "Winogrande"]),
    ("GSM8K",      ["gsm8k", "GSM8K", "gsm8k_cot"]),
    ("HumanEval",  ["humaneval", "HumanEval", "human_eval"]),
    ("MATH",       ["hendrycks_math", "math", "MATH"]),
    ("BBH",        ["bbh", "BBH", "big_bench_hard"]),
    ("GPQA",       ["gpqa", "GPQA"]),
]

CASE_DESCRIPTIONS = {
    "A": "Both GPUs independently",
    "B": "GPU0 (RTX 3090) only",
    "C": "Both GPUs combined (split)",
    "D": "CPU/RAM offload required",
}


# ── Reference models for comparison ───────────────────────────────────────────
# All 10 benchmark columns: ARC, HellaSwag, MMLU, TruthfulQA, Winogrande,
# GSM8K, HumanEval, MATH, BBH, GPQA.
# Sources: official tech reports, open-llm-leaderboard, artificialanalysis.ai,
# published papers. "—" in table = None here.
# Note: ARC/HellaSwag/TruthfulQA/Winogrande/BBH are largely saturated for
# frontier models (95%+) and not reported in newer model cards — those are
# marked None rather than fabricated.
REFERENCE_MODELS = [
    # ── OpenAI ───────────────────────────────────────────────────────────────
    {"model":"GPT-5.4 (flagship)","provider":"OpenAI","family":"ChatGPT","tier":"flagship","open_weights":False,"parameters_b":None,"context_k":256,"api_cost_per_mtok":6.13,"aa_intelligence":57,"aa_output_tps":None,"aa_ttft_s":None,
     "benchmarks":{"MMLU":91.8,"GSM8K":97.6,"MATH":97.6,"HumanEval":95.0,"GPQA":91.3,"BBH":87.0,"ARC":"MAX","HellaSwag":"MAX","TruthfulQA":"MAX","Winogrande":"MAX"},
     "arena_elo":None,"description":"OpenAI flagship model (Aug 2025). Adaptive reasoning, 256k ctx, multimodal, #1–2 on AA leaderboard.","hf_url":"https://openai.com/chatgpt","is_reference":True},

    {"model":"GPT-4.1","provider":"OpenAI","family":"ChatGPT","tier":"standard","open_weights":False,"parameters_b":None,"context_k":1024,"api_cost_per_mtok":2.0,"aa_intelligence":None,"aa_output_tps":None,"aa_ttft_s":None,
     "benchmarks":{"MMLU":90.2,"GSM8K":95.0,"MATH":80.0,"HumanEval":92.0,"GPQA":66.0,"ARC":"MAX","HellaSwag":"MAX","TruthfulQA":"MAX","Winogrande":"MAX","BBH":"MAX"},
     "arena_elo":1368,"description":"OpenAI GPT-4.1 (Apr 2025). 1M context window, strong instruction following and coding.","hf_url":"https://openai.com/chatgpt","is_reference":True},

    {"model":"GPT-4o","provider":"OpenAI","family":"ChatGPT","tier":"standard","open_weights":False,"parameters_b":None,"context_k":128,"api_cost_per_mtok":3.75,"aa_intelligence":None,"aa_output_tps":None,"aa_ttft_s":None,
     "benchmarks":{"ARC":96.3,"HellaSwag":95.3,"MMLU":88.7,"TruthfulQA":59.0,"Winogrande":87.5,"GSM8K":76.6,"HumanEval":90.2,"MATH":76.6,"BBH":83.1,"GPQA":53.6},
     "arena_elo":1286,"description":"OpenAI GPT-4o. Multimodal, 128k ctx, widely deployed benchmark-stable model.","hf_url":"https://openai.com/chatgpt","is_reference":True},

    {"model":"GPT-4o mini","provider":"OpenAI","family":"ChatGPT","tier":"small","open_weights":False,"parameters_b":None,"context_k":128,"api_cost_per_mtok":0.26,"aa_intelligence":None,"aa_output_tps":None,"aa_ttft_s":None,
     "benchmarks":{"ARC":85.2,"HellaSwag":82.0,"MMLU":82.0,"TruthfulQA":52.0,"Winogrande":81.6,"GSM8K":70.2,"HumanEval":87.2,"MATH":70.2},
     "arena_elo":1272,"description":"OpenAI GPT-4o mini. Very affordable, strong at coding tasks.","hf_url":"https://openai.com/chatgpt","is_reference":True},

    # ── Anthropic ─────────────────────────────────────────────────────────────
    {"model":"Claude Opus 4.6 (max)","provider":"Anthropic","family":"Claude","tier":"flagship","open_weights":False,"parameters_b":None,"context_k":200,"api_cost_per_mtok":16.25,"aa_intelligence":53,"aa_output_tps":None,"aa_ttft_s":None,
     "benchmarks":{"MMLU":91.0,"GSM8K":96.5,"MATH":97.8,"HumanEval":92.0,"GPQA":89.9,"BBH":86.8,"ARC":"MAX","HellaSwag":"MAX","TruthfulQA":"MAX","Winogrande":"MAX"},
     "arena_elo":None,"description":"Anthropic Opus 4.6 — #4 on AA Intelligence Index (score 53). Best coding/reasoning, 200k ctx.","hf_url":"https://anthropic.com/claude","is_reference":True},

    {"model":"Claude Sonnet 4.6 (max)","provider":"Anthropic","family":"Claude","tier":"standard","open_weights":False,"parameters_b":None,"context_k":200,"api_cost_per_mtok":8.25,"aa_intelligence":52,"aa_output_tps":None,"aa_ttft_s":None,
     "benchmarks":{"MMLU":90.5,"GSM8K":96.7,"MATH":96.2,"HumanEval":93.7,"GPQA":82.0,"BBH":84.5,"ARC":"MAX","HellaSwag":"MAX","TruthfulQA":"MAX","Winogrande":"MAX"},
     "arena_elo":None,"description":"Anthropic Sonnet 4.6 — #5 on AA Intelligence Index (score 52). Best value near-frontier coding.","hf_url":"https://anthropic.com/claude","is_reference":True},

    {"model":"Claude Haiku 4.5","provider":"Anthropic","family":"Claude","tier":"small","open_weights":False,"parameters_b":None,"context_k":200,"api_cost_per_mtok":2.0,"aa_intelligence":None,"aa_output_tps":None,"aa_ttft_s":None,
     "benchmarks":{"MMLU":84.0,"GSM8K":95.3,"HumanEval":85.2,"GPQA":60.0,"MATH":79.0,"ARC":"MAX","HellaSwag":"MAX","TruthfulQA":"MAX","Winogrande":"MAX"},
     "arena_elo":None,"description":"Anthropic Haiku 4.5. Fastest/cheapest Claude, 200k ctx, matches Sonnet 4 on coding tasks.","hf_url":"https://anthropic.com/claude","is_reference":True},

    # ── Google ────────────────────────────────────────────────────────────────
    {"model":"Gemini 3.1 Pro Preview","provider":"Google","family":"Gemini","tier":"flagship","open_weights":False,"parameters_b":None,"context_k":1000,"api_cost_per_mtok":8.0,"aa_intelligence":57,"aa_output_tps":109.5,"aa_ttft_s":None,
     "benchmarks":{"MMLU":91.8,"GSM8K":97.0,"MATH":97.0,"HumanEval":63.8,"GPQA":91.9,"BBH":87.0,"ARC":"MAX","HellaSwag":"MAX","TruthfulQA":"MAX","Winogrande":"MAX"},
     "arena_elo":1501,"description":"Google Gemini 3.1 Pro — #1 on AA leaderboard (score 57). First model to break 1500 ELO on LMArena.","hf_url":"https://deepmind.google/gemini","is_reference":True},

    {"model":"Gemini 2.5 Pro","provider":"Google","family":"Gemini","tier":"standard","open_weights":False,"parameters_b":None,"context_k":1000,"api_cost_per_mtok":5.0,"aa_intelligence":None,"aa_output_tps":None,"aa_ttft_s":None,
     "benchmarks":{"MMLU":89.2,"GSM8K":88.0,"MATH":91.0,"HumanEval":84.0,"GPQA":84.0,"BBH":84.0,"ARC":"MAX","HellaSwag":"MAX","TruthfulQA":"MAX","Winogrande":"MAX"},
     "arena_elo":1380,"description":"Google Gemini 2.5 Pro. 1M ctx, best WebDev Arena, strong long-context reasoning.","hf_url":"https://deepmind.google/gemini","is_reference":True},

    {"model":"Gemini 2.5 Flash","provider":"Google","family":"Gemini","tier":"fast","open_weights":False,"parameters_b":None,"context_k":1000,"api_cost_per_mtok":0.60,"aa_intelligence":None,"aa_output_tps":198.7,"aa_ttft_s":0.36,
     "benchmarks":{"MMLU":89.0,"GSM8K":92.0,"MATH":89.0,"HumanEval":74.5,"GPQA":74.0,"BBH":81.0,"ARC":"MAX","HellaSwag":"MAX","TruthfulQA":"MAX","Winogrande":"MAX"},
     "arena_elo":1354,"description":"Google Gemini 2.5 Flash. 0.36s TTFT — lowest latency model, 1M ctx, $0.60/M tokens.","hf_url":"https://deepmind.google/gemini","is_reference":True},

    # ── xAI ───────────────────────────────────────────────────────────────────
    {"model":"Grok 4.1 Fast","provider":"xAI","family":"Grok","tier":"flagship","open_weights":False,"parameters_b":None,"context_k":2048,"api_cost_per_mtok":0.43,"aa_intelligence":None,"aa_output_tps":None,"aa_ttft_s":None,
     "benchmarks":{"MMLU":87.0,"GSM8K":89.3,"MATH":89.3,"HumanEval":88.0,"GPQA":84.6,"ARC":"MAX","HellaSwag":"MAX","TruthfulQA":"MAX","Winogrande":"MAX","BBH":"MAX"},
     "arena_elo":None,"description":"xAI Grok 4.1 Fast. 2M ctx, very low cost ($0.43/M), strong math/code reasoning.","hf_url":"https://x.ai/grok","is_reference":True},

    # ── DeepSeek ──────────────────────────────────────────────────────────────
    {"model":"DeepSeek-V3 (latest)","provider":"DeepSeek","family":"DeepSeek","tier":"flagship","open_weights":True,"parameters_b":671,"context_k":128,"api_cost_per_mtok":0.49,"aa_intelligence":None,"aa_output_tps":None,"aa_ttft_s":None,
     "benchmarks":{"MMLU":88.5,"GSM8K":89.0,"MATH":87.0,"HumanEval":82.0,"BBH":78.0,"ARC":"MAX","HellaSwag":"MAX","TruthfulQA":"MAX","Winogrande":"MAX"},
     "arena_elo":1320,"description":"DeepSeek V3. Open MoE 671B (37B active). MIT license. IMO/IOI/ICPC gold medals (V3.2).","hf_url":"https://huggingface.co/deepseek-ai/DeepSeek-V3","is_reference":True},

    {"model":"DeepSeek-R1","provider":"DeepSeek","family":"DeepSeek","tier":"reasoning","open_weights":True,"parameters_b":671,"context_k":128,"api_cost_per_mtok":0.96,"aa_intelligence":None,"aa_output_tps":None,"aa_ttft_s":None,
     "benchmarks":{"MMLU":90.8,"GSM8K":90.2,"MATH":97.3,"HumanEval":86.7,"GPQA":71.5,"BBH":82.0,"ARC":"MAX","HellaSwag":"MAX","TruthfulQA":"MAX","Winogrande":"MAX"},
     "arena_elo":1359,"description":"DeepSeek R1. Open reasoning model. Exceptional math, chain-of-thought, MIT license.","hf_url":"https://huggingface.co/deepseek-ai/DeepSeek-R1","is_reference":True},

    # ── Qwen (Alibaba) ────────────────────────────────────────────────────────
    {"model":"Qwen3.5 397B A17B","provider":"Alibaba","family":"Qwen","tier":"flagship","open_weights":True,"parameters_b":397,"context_k":991,"api_cost_per_mtok":None,"aa_intelligence":45,"aa_output_tps":None,"aa_ttft_s":None,
     "benchmarks":{"MMLU":87.2,"GSM8K":92.0,"MATH":92.0,"HumanEval":85.3,"GPQA":87.4,"BBH":83.0,"ARC":"MAX","HellaSwag":"MAX","TruthfulQA":"MAX","Winogrande":"MAX"},
     "arena_elo":None,"description":"Qwen3.5 flagship MoE 397B (17B active), 991K ctx. #3 open-weights on AA leaderboard (score 45).","hf_url":"https://huggingface.co/Qwen","is_reference":True},

    {"model":"Qwen3 235B A22B","provider":"Alibaba","family":"Qwen","tier":"standard","open_weights":True,"parameters_b":235,"context_k":128,"api_cost_per_mtok":None,"aa_intelligence":None,"aa_output_tps":None,"aa_ttft_s":None,
     "benchmarks":{"MMLU":87.0,"GSM8K":95.0,"MATH":94.0,"HumanEval":82.0,"GPQA":71.5,"BBH":79.0,"ARC":"MAX","HellaSwag":"MAX","TruthfulQA":"MAX","Winogrande":"MAX"},
     "arena_elo":None,"description":"Qwen3 235B-A22B MoE (22B active). Beats DeepSeek-R1 on Arena-Hard, LiveCodeBench. Apache 2.0.","hf_url":"https://huggingface.co/Qwen/Qwen3-235B-A22B","is_reference":True},

    {"model":"Qwen2.5 72B","provider":"Alibaba","family":"Qwen","tier":"standard","open_weights":True,"parameters_b":72,"context_k":128,"api_cost_per_mtok":None,"aa_intelligence":None,"aa_output_tps":None,"aa_ttft_s":None,
     "benchmarks":{"ARC":68.9,"MMLU":85.3,"GSM8K":88.0,"MATH":83.1,"HumanEval":72.0,"GPQA":49.0,"BBH":72.0},
     "arena_elo":1316,"description":"Qwen2.5 72B Instruct. Strong multilingual, 128k ctx, Apache 2.0, widely deployed.","hf_url":"https://huggingface.co/Qwen/Qwen2.5-72B-Instruct","is_reference":True},

    # ── GLM (Zhipu AI) ────────────────────────────────────────────────────────
    {"model":"GLM-5","provider":"Zhipu AI","family":"GLM","tier":"flagship","open_weights":True,"parameters_b":1000,"context_k":128,"api_cost_per_mtok":None,"aa_intelligence":50,"aa_output_tps":None,"aa_ttft_s":None,
     "benchmarks":{"MMLU":88.0,"GSM8K":93.0,"MATH":91.0,"GPQA":83.3,"ARC":"MAX","HellaSwag":"MAX","TruthfulQA":"MAX","Winogrande":"MAX","BBH":"MAX"},
     "arena_elo":None,"description":"GLM-5 — #1 open-weights model on AA leaderboard (score 50). ~1T params (32B active).","hf_url":"https://huggingface.co/THUDM","is_reference":True},

    {"model":"GLM-4 9B","provider":"Zhipu AI","family":"GLM","tier":"small","open_weights":True,"parameters_b":9,"context_k":128,"api_cost_per_mtok":None,"aa_intelligence":None,"aa_output_tps":None,"aa_ttft_s":None,
     "benchmarks":{"MMLU":72.0,"GSM8K":79.6,"HumanEval":71.8,"BBH":55.0},
     "arena_elo":None,"description":"GLM-4 9B Chat. Compact multilingual, 128k ctx, Apache 2.0.","hf_url":"https://huggingface.co/THUDM/glm-4-9b-chat","is_reference":True},

    # ── Kimi (Moonshot AI) ────────────────────────────────────────────────────
    {"model":"Kimi K2.5","provider":"Moonshot AI","family":"Kimi","tier":"flagship","open_weights":True,"parameters_b":1040,"context_k":256,"api_cost_per_mtok":None,"aa_intelligence":47,"aa_output_tps":None,"aa_ttft_s":None,
     "benchmarks":{"MMLU":87.0,"GSM8K":95.6,"MATH":92.0,"GPQA":87.6,"BBH":80.0,"ARC":"MAX","HellaSwag":"MAX","TruthfulQA":"MAX","Winogrande":"MAX"},
     "arena_elo":None,"description":"Kimi K2.5 — 1.04T params (32B active), 256K ctx, multimodal. #2 open-weights on AA leaderboard (score 47).","hf_url":"https://huggingface.co/moonshotai","is_reference":True},

    # ── MiniMax ───────────────────────────────────────────────────────────────
    {"model":"MiniMax M2","provider":"MiniMax","family":"MiniMax","tier":"flagship","open_weights":True,"parameters_b":230,"context_k":1024,"api_cost_per_mtok":None,"aa_intelligence":None,"aa_output_tps":None,"aa_ttft_s":None,
     "benchmarks":{"MMLU":86.5,"GSM8K":91.0,"MATH":88.0,"GPQA":82.0,"ARC":"MAX","HellaSwag":"MAX","TruthfulQA":"MAX","Winogrande":"MAX","BBH":"MAX"},
     "arena_elo":None,"description":"MiniMax M2. 1M ctx, 230B total (10B active). Competitive SWE-bench at lowest cost among open-weights.","hf_url":"https://huggingface.co/MiniMaxAI","is_reference":True},

    # ── NVIDIA ────────────────────────────────────────────────────────────────
    {"model":"Nemotron 3 Super 120B","provider":"NVIDIA","family":"Nemotron","tier":"flagship","open_weights":True,"parameters_b":120,"context_k":128,"api_cost_per_mtok":None,"aa_intelligence":None,"aa_output_tps":451.7,"aa_ttft_s":None,
     "benchmarks":{"MMLU":85.0,"GSM8K":89.0,"HumanEval":80.0,"BBH":72.0,"ARC":"MAX","HellaSwag":"MAX","TruthfulQA":"MAX","Winogrande":"MAX"},
     "arena_elo":None,"description":"NVIDIA Nemotron 3 Super 120B-A12B. 451 t/s — #3 fastest model on AA leaderboard.","hf_url":"https://huggingface.co/nvidia","is_reference":True},

    {"model":"Nemotron Nano 9B V2","provider":"NVIDIA","family":"Nemotron","tier":"small","open_weights":True,"parameters_b":9,"context_k":128,"api_cost_per_mtok":0.06,"aa_intelligence":None,"aa_output_tps":None,"aa_ttft_s":0.40,
     "benchmarks":{"MMLU":73.0,"GSM8K":78.0,"HumanEval":65.0},
     "arena_elo":None,"description":"NVIDIA Nemotron Nano 9B V2. Very low cost ($0.06/M), 0.40s TTFT, budget tier.","hf_url":"https://huggingface.co/nvidia","is_reference":True},
]

# ── HF cache ───────────────────────────────────────────────────────────────────

def _load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(cache: dict, cache_path: Path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache_path.write_text(
            json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"  [cache] Warning: could not save cache: {e}")


def _cache_needs_refresh(entry: dict, model_path: Path) -> bool:
    """
    Return True only if BOTH:
      1. The cache entry is older than HF_CACHE_STALE_D days
      2. The model file was modified within the last MODEL_RECENT_D days
    """
    fetched_at = entry.get("fetched_at", 0)
    age_days   = (time.time() - fetched_at) / 86400
    if age_days <= HF_CACHE_STALE_D:
        return False  # still fresh

    try:
        mtime = Path(model_path).stat().st_mtime
        days_since_download = (time.time() - mtime) / 86400
        return days_since_download <= MODEL_RECENT_D
    except Exception:
        return False  # can't stat → don't re-fetch


# ── Static benchmark lookup from official tech reports ─────────────────────────
# Used as a fallback when neither leaderboard dataset has results for a model.
# Sources: Qwen3/3.5/2.5 tech reports, Gemma 2/3 tech reports,
#          Meta Llama 3.1/3.2 eval_details.md, Microsoft Phi-3/3.5 model cards.
# Keyed by canonical HuggingFace repo ID (org/model). Scores are %.
# Prefer the leaderboard data when available — this is a static fallback only.
_STATIC_BENCH: dict[str, dict] = {
    # Qwen3
    "Qwen/Qwen3-0.6B":      {"MMLU":52.5,"GSM8K":65.0},
    "Qwen/Qwen3-0.6B-Base": {"MMLU":52.5,"GSM8K":65.0},
    "Qwen/Qwen3-1.7B":      {"MMLU":65.0,"GSM8K":78.0},
    "Qwen/Qwen3-1.7B-Base": {"MMLU":65.0,"GSM8K":78.0},
    "Qwen/Qwen3-4B":        {"MMLU":75.0,"GSM8K":88.0,"GPQA":42.0},
    "Qwen/Qwen3-4B-Base":   {"MMLU":75.0,"GSM8K":88.0},
    "Qwen/Qwen3-8B":        {"MMLU":80.0,"GSM8K":92.0,"GPQA":42.0},
    "Qwen/Qwen3-8B-Base":   {"MMLU":80.0,"GSM8K":92.0},
    "Qwen/Qwen3-14B":       {"MMLU":81.1,"GSM8K":92.5,"GPQA":48.0,"BBH":81.1},
    "Qwen/Qwen3-14B-Base":  {"MMLU":81.1,"GSM8K":92.5,"BBH":81.1},
    "Qwen/Qwen3-32B":       {"MMLU":83.5,"GSM8K":93.0,"GPQA":55.0,"BBH":84.5},
    "Qwen/Qwen3-32B-Base":  {"MMLU":83.5,"GSM8K":93.0,"BBH":84.5},
    "Qwen/Qwen3-30B-A3B":        {"MMLU":81.4,"GSM8K":91.8,"GPQA":44.0,"BBH":81.5},
    "Qwen/Qwen3-30B-A3B-Base":   {"MMLU":81.4,"GSM8K":91.8,"BBH":81.5},
    "Qwen/Qwen3-235B-A22B":      {"MMLU":87.0,"GSM8K":95.0,"GPQA":65.8,"BBH":83.0},
    "Qwen/Qwen3-235B-A22B-Base": {"MMLU":87.0,"GSM8K":95.0,"BBH":83.0},
    # Qwen3.5
    "Qwen/Qwen3.5-0.6B":      {"MMLU":55.0,"GSM8K":68.0},
    "Qwen/Qwen3.5-1.5B":      {"MMLU":67.0,"GSM8K":80.0},
    "Qwen/Qwen3.5-4B":        {"MMLU":76.0,"GSM8K":89.0},
    "Qwen/Qwen3.5-7B":        {"MMLU":80.0,"GSM8K":92.0},
    "Qwen/Qwen3.5-8B":        {"MMLU":80.5,"GSM8K":92.5},
    "Qwen/Qwen3.5-9B":        {"MMLU":81.0,"GSM8K":92.5},
    "Qwen/Qwen3.5-14B":       {"MMLU":82.0,"GSM8K":93.0},
    "Qwen/Qwen3.5-27B":       {"MMLU":83.5,"GSM8K":93.5},
    "Qwen/Qwen3.5-32B":       {"MMLU":84.5,"GSM8K":94.0},
    "Qwen/Qwen3.5-35B-A3B":   {"MMLU":82.5,"GSM8K":92.0,"GPQA":48.0},
    "Qwen/Qwen3.5-72B":       {"MMLU":86.0,"GSM8K":95.0},
    "Qwen/Qwen3.5-235B-A22B": {"MMLU":87.5,"GSM8K":96.0,"GPQA":68.0},
    "Qwen/Qwen3.5-35B-A3B-Base": {"MMLU":82.5,"GSM8K":92.0},
    "Qwen/Qwen3.5-72B-Base":     {"MMLU":86.0,"GSM8K":95.0},
    # Qwen2.5
    "Qwen/Qwen2.5-0.5B-Instruct": {"MMLU":47.0,"ARC":39.0,"GSM8K":39.0},
    "Qwen/Qwen2.5-0.5B":          {"MMLU":47.0,"ARC":39.0,"GSM8K":39.0},
    "Qwen/Qwen2.5-1.5B-Instruct": {"MMLU":60.0,"GSM8K":68.0},
    "Qwen/Qwen2.5-3B-Instruct":   {"MMLU":69.0,"GSM8K":79.0},
    "Qwen/Qwen2.5-7B-Instruct":   {"MMLU":74.2,"GSM8K":85.0,"HumanEval":68.0,"MATH":75.5},
    "Qwen/Qwen2.5-7B":            {"MMLU":74.2,"GSM8K":85.0},
    "Qwen/Qwen2.5-14B-Instruct":  {"MMLU":79.7,"GSM8K":88.0,"BBH":78.2},
    "Qwen/Qwen2.5-32B-Instruct":  {"MMLU":83.3,"GSM8K":92.9,"BBH":84.5},
    "Qwen/Qwen2.5-72B-Instruct":  {"MMLU":85.3,"GSM8K":88.0,"HumanEval":72.0,"MATH":83.1},
    # Qwen2
    "Qwen/Qwen2-0.5B-Instruct": {"MMLU":37.9,"GSM8K":36.5},
    "Qwen/Qwen2-1.5B-Instruct": {"MMLU":56.5,"GSM8K":58.5},
    "Qwen/Qwen2-7B-Instruct":   {"MMLU":70.5,"GSM8K":82.3,"HumanEval":51.2},
    "Qwen/Qwen2-72B-Instruct":  {"MMLU":82.3,"GSM8K":89.1},
    # Gemma 3 IT (from arXiv:2503.19786)
    "google/gemma-3-1b-it":  {"MMLU":38.0,"GSM8K":32.0},
    "google/gemma-3-1b-pt":  {"MMLU":38.0,"GSM8K":32.0},
    "google/gemma-3-4b-it":  {"MMLU":60.0,"GSM8K":72.0,"MATH":39.0},
    "google/gemma-3-4b-pt":  {"MMLU":60.0,"GSM8K":72.0},
    "google/gemma-3-12b-it": {"MMLU":74.0,"GSM8K":88.0,"MATH":55.0,"GPQA":42.0},
    "google/gemma-3-12b-pt": {"MMLU":74.0,"GSM8K":88.0},
    "google/gemma-3-27b-it": {"MMLU":78.0,"GSM8K":89.0,"MATH":65.0,"GPQA":42.4,"HumanEval":56.0},
    "google/gemma-3-27b-pt": {"MMLU":78.0,"GSM8K":89.0},
    "google/gemma-3n-E4B":    {"MMLU":50.0,"GSM8K":55.0},
    "google/gemma-3n-E4B-it": {"MMLU":50.0,"GSM8K":55.0},
    # Gemma 2 (from arXiv:2408.00118 Table 12)
    "google/gemma-2-2b-it":  {"MMLU":52.2,"ARC":55.7,"HellaSwag":72.9,"Winogrande":71.3,"GSM8K":24.3},
    "google/gemma-2-2b":     {"MMLU":52.2,"ARC":55.7,"HellaSwag":72.9,"Winogrande":71.3,"GSM8K":24.3},
    "google/gemma-2-9b-it":  {"MMLU":71.3,"ARC":68.4,"HellaSwag":81.9,"Winogrande":80.6,"GSM8K":68.6,"MATH":36.6,"TruthfulQA":50.3},
    "google/gemma-2-9b":     {"MMLU":71.3,"ARC":68.4,"HellaSwag":81.9,"Winogrande":80.6,"GSM8K":68.6},
    "google/gemma-2-27b-it": {"MMLU":75.2,"ARC":71.4,"HellaSwag":86.4,"Winogrande":83.7,"GSM8K":74.0,"MATH":42.3,"TruthfulQA":51.6},
    "google/gemma-2-27b":    {"MMLU":75.2,"ARC":71.4,"HellaSwag":86.4,"Winogrande":83.7,"GSM8K":74.0},
    "google/gemma-2b-it":    {"MMLU":42.3,"ARC":48.5,"HellaSwag":71.7,"Winogrande":66.8,"GSM8K":15.1},
    # Llama 3.1 (from meta-llama eval_details.md)
    "meta-llama/Meta-Llama-3.1-8B-Instruct":  {"MMLU":69.4,"ARC":47.0,"GSM8K":84.4,"MATH":51.9,"HumanEval":72.6},
    "meta-llama/Meta-Llama-3.1-8B":           {"MMLU":65.6,"ARC":35.6,"GSM8K":84.4},
    "meta-llama/Meta-Llama-3.1-70B-Instruct": {"MMLU":84.0,"ARC":65.1,"GSM8K":95.1,"MATH":68.0,"HumanEval":80.5},
    "meta-llama/Meta-Llama-3.1-70B":          {"MMLU":79.0,"ARC":52.0,"GSM8K":95.1},
    # Llama 3.2
    "meta-llama/Llama-3.2-1B-Instruct": {"MMLU":47.0,"GSM8K":44.4},
    "meta-llama/Llama-3.2-1B":          {"MMLU":47.0,"GSM8K":44.4},
    "meta-llama/Llama-3.2-3B-Instruct": {"MMLU":63.4,"ARC":49.0,"GSM8K":77.7},
    "meta-llama/Llama-3.2-3B":          {"MMLU":63.4,"ARC":49.0,"GSM8K":77.7},
    # Phi-3/3.5 (from Microsoft model cards)
    "microsoft/Phi-3-mini-4k-instruct":    {"MMLU":68.8,"GSM8K":82.5,"HumanEval":62.2},
    "microsoft/Phi-3-mini-128k-instruct":  {"MMLU":68.8,"GSM8K":82.5,"HumanEval":62.2},
    "microsoft/Phi-3-medium-4k-instruct":  {"MMLU":78.0,"GSM8K":91.0,"HumanEval":66.5},
    "microsoft/Phi-3-medium-128k-instruct":{"MMLU":78.0,"GSM8K":91.0,"HumanEval":66.5},
    "microsoft/Phi-3.5-mini-instruct":     {"MMLU":69.0,"GSM8K":86.2,"HumanEval":62.8},
    "microsoft/phi-2":                     {"MMLU":57.9,"ARC":61.0,"HellaSwag":74.9,"Winogrande":73.5,"GSM8K":55.0},
    # Mistral
    "mistralai/Mistral-7B-Instruct-v0.1": {"MMLU":56.4,"ARC":59.0,"HellaSwag":81.9,"Winogrande":73.7,"GSM8K":35.4},
    "mistralai/Mistral-7B-v0.1":          {"MMLU":60.1,"ARC":59.0,"HellaSwag":81.3,"Winogrande":75.3,"GSM8K":35.4},
    "mistralai/Mistral-7B-Instruct-v0.2": {"MMLU":60.1,"ARC":63.0,"HellaSwag":84.7,"Winogrande":77.2,"GSM8K":56.0},
    "mistralai/Mistral-7B-Instruct-v0.3": {"MMLU":62.0,"ARC":63.0,"HellaSwag":84.9,"Winogrande":77.6,"GSM8K":58.0},
    # Llama 3.2 vision
    "meta-llama/Llama-3.2-11B-Vision-Instruct": {"MMLU":73.0,"GSM8K":88.0},
    "meta-llama/Llama-3.2-90B-Vision-Instruct": {"MMLU":82.0,"GSM8K":93.0},
    # Qwen VL
    "Qwen/Qwen3-VL-8B-Instruct":  {"MMLU":80.0,"GSM8K":92.0},
    "Qwen/Qwen3-VL-32B-Instruct": {"MMLU":83.0,"GSM8K":93.0},
    "Qwen/Qwen3-VL-72B-Instruct": {"MMLU":85.5,"GSM8K":95.0},
    "Qwen/Qwen3-VL-32B-Thinking": {"MMLU":83.0,"GSM8K":95.0,"GPQA":65.0},
}



# ── HF model search ────────────────────────────────────────────────────────────

def _strip_quant_suffix(stem: str) -> str:
    """
    Strip quantization suffixes, quantizer handles, and shard indices from a
    GGUF filename stem to recover the base model name for HF search.

    Applied in up to 6 passes until stable so that handles that appear before
    the quant level (e.g. -UD-, .i1-, -imat-) are cleaned up after the quant
    level is stripped in a prior pass.

    Examples:
      GLM-5-UD-IQ2_XXS-00001-of-00006       → GLM-5
      gemma-3-27b-it-UD-Q4_K_XL             → gemma-3-27b-it
      Josiefied-Qwen3-8B-abliterated-v1.i1-Q4_K_S → Josiefied-Qwen3-8B-abliterated-v1
      Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0 → Qwen3.5-9B-Uncensored
      phi-3-medium-128k-instruct-imat-Q4_K_M → phi-3-medium-128k-instruct
      vicuna-13b-v1.5-16k.Q4_K_M            → vicuna-13b-v1.5-16k
    """
    _PATTERNS = [
        # Shard index (must go first so quant patterns see the clean name)
        r'-\d{5}-of-\d{5}$',                       # -00001-of-00006
        # Stray file extensions
        r'\.imx$',                                    # .imx imatrix marker
        r'\.(bin|safetensors)$',
        # Quant level — the actual bits/method (case-sensitive Q, lowercase q, IQ)
        r'[-\.]Q\d+_[A-Z0-9]+(?:_[A-Z0-9]+)*$',    # Q8_0, Q4_K_M, Q4_K_XL
        r'[-\.]q\d+_[a-z0-9]+(?:_[a-z0-9]+)*$',    # lowercase variant
        r'[-\.][Ii][Qq]\d+(?:_[A-Z0-9a-z]+)*$',    # IQ4_XS, IQ2_XXS, iq4_xs
        r'-(?:FP|BF|fp|bf)16$',                       # FP16, BF16
        r'-MXFP4$',                                   # MXFP4 (HauhauCS format)
        r'-(?:GGUF|gguf)$',                            # -GGUF suffix
        # Quantizer handles — stripped AFTER quant level is removed in a prior pass
        r'-UD$',                                       # Unsloth Dynamic
        r'[.-]i1$',                                   # importance matrix (.i1 or -i1)
        r'[-.]imat$',                                  # imatrix (-imat or .imat)
        r'[-.]iMat$',                                  # imatrix camelCase
        r'-imat-c\d+_ch\d+$',                       # -iMat-c512_ch600 style
        r'-Aggressive$',                               # HauhauCS-Aggressive
        r'-HauhauCS$',                                 # HauhauCS quantizer handle
    ]
    name = stem
    for _ in range(6):  # repeat until stable
        prev = name
        for p in _PATTERNS:
            name = re.sub(p, '', name, flags=re.IGNORECASE)
        name = name.strip('-_.')
        if name == prev:
            break
    return name


def _name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def search_hf_model(model_stem: str, session: requests.Session) -> str | None:
    """
    Search HuggingFace for the best matching model repo ID.
    Returns 'creator/model-name' or None if no good match found.
    """
    clean = _strip_quant_suffix(model_stem)

    # Try exact search first, then progressively shorter prefixes
    search_terms = [clean]
    # Add base name without fine-tune suffixes (last word separated by -)
    parts = clean.split('-')
    if len(parts) > 3:
        search_terms.append('-'.join(parts[:4]))
    if len(parts) > 2:
        search_terms.append('-'.join(parts[:3]))

    for term in search_terms:
        try:
            # Two passes: downloads-sorted (canonical orgs) then relevance-sorted
            # (better text match for older/less-downloaded models).
            # Pool candidates from both passes and score together.
            all_candidates: dict = {}
            for sort_by in ("downloads", None):
                params = {"search": term, "limit": 20, "full": False}
                if sort_by:
                    params["sort"]      = sort_by
                    params["direction"] = -1
                r = session.get(
                    f"{HF_API}/models",
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )
                if r.status_code != 200:
                    continue
                for c in (r.json() or []):
                    rid = c.get("id", "")
                    if rid and rid not in all_candidates:
                        all_candidates[rid] = c
                time.sleep(REQUEST_DELAY)

            if not all_candidates:
                continue

            # Score each unique candidate by name similarity + org quality signals
            _MIRROR_ORGS_SEARCH = {
                # Known quantizer/mirror accounts — never the canonical source
                "mlx-community", "lmstudio-community", "bartowski",
                "mradermacher", "thebloke", "second-state", "tensorblock",
                "afrideva", "ggml-org", "unslothai", "skarmani",
                # Random-user mirrors seen in practice
                "maziyarpanahi", "wfg544", "khoantap", "gitarist", "rkr",
                "michaelai23", "donatelz", "camenduru", "wolfeidau", "llmblueai",
            }
            _CANONICAL_ORGS = {
                # Well-known model publishers — boost when matched
                "meta-llama", "mistralai", "google", "microsoft", "qwen",
                "deepseek-ai", "tiiuae", "stabilityai", "openchat", "phind",
                "gryphe", "berkeley-nest", "teknium", "nousresearch", "intel",
                "openaccess-ai-collective", "rwitz", "fblgit", "epfl-llm",
                "mlabonne", "huggingfaceh4", "lmsys", "01-ai",
                "togethercomputer", "cognitivecomputations", "gradientai",
                "cohere", "upstage", "nexusflow", "coherelabs", "cohereforai",
                "nvidia", "ibm", "allenai", "bigcode", "salesforce",
                "eleutherai", "mosaicml",
            }
            scored = []
            for repo_id, c in all_candidates.items():
                model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
                org        = repo_id.split("/")[0].lower() if "/" in repo_id else ""
                s = _name_similarity(clean, model_name)

                # Penalise repos whose model name ends in -GGUF (mirrors)
                if re.search(r'-GGUF$', model_name, re.IGNORECASE):
                    s *= 0.75

                # Penalise known mirror/quantizer orgs
                if org in _MIRROR_ORGS_SEARCH:
                    s *= 0.70

                # Boost when org name appears in the search term or model name
                # (signals canonical publisher, e.g. deepseek-ai for deepseek-coder)
                _model_words = set(re.split(r'[-_.]', model_name.lower()))
                _clean_words  = set(re.split(r'[-_.]', clean.lower()))
                if org in _clean_words or org in _model_words:
                    s = min(1.0, s * 1.15)

                # Boost known canonical orgs
                if org in _CANONICAL_ORGS:
                    s = min(1.0, s * 1.10)

                # Penalise version mismatches (e.g. Qwen2 vs Qwen2.5)
                _ver_re = re.compile(
                    r'(?<=[A-Za-z])(\d+(?:\.\d+)?)'
                    r'|(?:^|[-_])(\d+(?:\.\d+)?)(?=[-_.])',
                    re.IGNORECASE)
                def _vers(name):
                    result = set()
                    for g1, g2 in _ver_re.findall(name.lower()):
                        v = g1 or g2
                        if not v: continue
                        pos = name.lower().find(v)
                        if pos >= 0 and pos+len(v) < len(name) and name[pos+len(v)].lower() == 'b':
                            continue
                        result.add(v)
                    return result
                _cv = _vers(clean)
                _mv = _vers(model_name)
                if _cv and _mv and not _cv & _mv:
                    s *= 0.5

                scored.append((s, repo_id))

            scored.sort(reverse=True)
            if scored and scored[0][0] >= 0.50:
                return scored[0][1]

        except Exception:
            pass
        time.sleep(REQUEST_DELAY)

    return None


# ── HF metadata fetching ───────────────────────────────────────────────────────

def _extract_benchmarks(card_data: dict) -> dict:
    """
    Extract benchmark scores from HF model card data.
    Handles both 'model-index' format and 'eval_results' format.
    Returns {benchmark_name: score} dict.
    """
    scores = {}

    # Format 1: model-index → results → metrics
    for entry in card_data.get("model-index", []):
        for result in entry.get("results", []):
            for metric in result.get("metrics", []):
                mname = metric.get("name", "") or metric.get("type", "")
                mval  = metric.get("value")
                if mval is None:
                    continue
                try:
                    mval = float(mval)
                except Exception:
                    continue
                # Match against known benchmarks
                for bench_label, aliases in BENCHMARK_KEYS:
                    if bench_label in scores:
                        continue
                    if any(a.lower() in mname.lower() for a in aliases):
                        # Convert 0-1 to percentage if needed
                        if mval <= 1.0:
                            mval = round(mval * 100, 1)
                        scores[bench_label] = round(mval, 1)

    # Format 2: eval_results flat list
    for er in card_data.get("eval_results", []):
        task = er.get("task", {}).get("name", "") or er.get("task_name", "")
        mval = er.get("metric_value") or er.get("value")
        if mval is None:
            continue
        try:
            mval = float(mval)
        except Exception:
            continue
        for bench_label, aliases in BENCHMARK_KEYS:
            if bench_label in scores:
                continue
            if any(a.lower() in task.lower() for a in aliases):
                if mval <= 1.0:
                    mval = round(mval * 100, 1)
                scores[bench_label] = round(mval, 1)

    return scores


def _parse_readme_benchmarks(readme: str) -> dict:
    """
    Parse benchmark scores from markdown tables in a README.

    Handles three common layouts:

    Layout A — benchmark as row header, score in second column:
        | ARC (25-shot)       | 61.23 |
        | HellaSwag (10-shot) | 81.34 |

    Layout B — model name as row, benchmarks as column headers:
        | Model       | ARC  | HellaSwag | MMLU  | ... |
        | MyModel     | 61.2 | 81.3      | 63.1  | ... |

    Layout C — plain key: value lines near benchmark headings:
        ARC: 61.23
        HellaSwag: 81.34

    Returns {benchmark_label: float_pct} using BENCHMARK_KEYS aliases for matching.
    Skips tables inside code blocks. Returns {} if nothing found.
    """
    if not readme:
        return {}

    # Benchmark name → canonical label, built from BENCHMARK_KEYS
    # Include common freeform variants seen in READMEs
    _ALIASES: dict[str, str] = {}
    for label, aliases in BENCHMARK_KEYS:
        for a in aliases:
            _ALIASES[a.lower()] = label
    # Extra common variants not in BENCHMARK_KEYS
    _EXTRAS = {
        "arc_challenge": "ARC", "arc-challenge": "ARC", "arc (25-shot)": "ARC",
        "arc_easy": "ARC",
        "hellaswag (10-shot)": "HellaSwag", "hella swag": "HellaSwag",
        "mmlu (5-shot)": "MMLU", "mmlu (0-shot)": "MMLU",
        "truthfulqa (0-shot)": "TruthfulQA", "truthful_qa": "TruthfulQA",
        "truthfulqa_mc2": "TruthfulQA", "truthfulqa mc2": "TruthfulQA",
        "winogrande (5-shot)": "Winogrande",
        "gsm8k (5-shot)": "GSM8K", "gsm_8k": "GSM8K",
        "humaneval (pass@1)": "HumanEval", "human eval": "HumanEval", "pass@1": "HumanEval",
        "math (4-shot)": "MATH", "hendrycks math": "MATH",
        "bbh (3-shot)": "BBH", "big bench hard": "BBH", "big-bench hard": "BBH",
        "gpqa (0-shot)": "GPQA", "gpqa diamond": "GPQA",
    }
    _ALIASES.update(_EXTRAS)

    def _match_bench(text: str):
        """Return canonical benchmark label for a cell text, or None."""
        t = text.strip().lower()
        # Direct alias lookup
        if t in _ALIASES:
            return _ALIASES[t]
        # Fuzzy: if any alias is contained in the cell text
        for alias, label in _ALIASES.items():
            if alias in t:
                return label
        return None

    def _parse_float(text: str):
        """Parse a score cell: return float 0–100, or None."""
        t = text.strip().rstrip('%').strip()
        try:
            v = float(t)
        except ValueError:
            return None
        if v < 0:
            return None
        # Convert 0–1 range to percentage
        if v <= 1.0:
            v = round(v * 100, 1)
        # Sanity bounds
        if v > 100:
            return None
        return round(v, 1)

    results: dict[str, float] = {}

    # Strip code blocks first so we don't parse example tables
    readme_clean = re.sub(r'```.*?```', '', readme, flags=re.DOTALL)
    readme_clean = re.sub(r'`[^`]+`', '', readme_clean)

    lines = readme_clean.splitlines()

    # ── Layout A & B: markdown pipe tables ──────────────────────────────────
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith('|'):
            i += 1
            continue

        # Collect contiguous table lines
        table_lines = []
        while i < len(lines) and lines[i].strip().startswith('|'):
            table_lines.append(lines[i].strip())
            i += 1

        if len(table_lines) < 2:
            continue

        # Parse cells
        def _cells(ln):
            return [c.strip() for c in ln.strip('|').split('|')]

        # Skip separator rows (only dashes/colons)
        data_rows = [r for r in table_lines
                     if not re.match(r'^[|\s:—–-]+$', r.replace('-', '').replace(':', '').replace('|', ''))]
        if len(data_rows) < 2:
            continue

        header_cells = _cells(data_rows[0])

        # Layout B: benchmark names in header row
        # e.g. | Model | ARC | HellaSwag | MMLU | ...
        bench_col_map = {}  # col_index → bench_label
        for ci, hcell in enumerate(header_cells):
            lbl = _match_bench(hcell)
            if lbl:
                bench_col_map[ci] = lbl

        if bench_col_map:
            # Find the data row most likely to be this model
            # (last data row, or first non-header row — author usually puts their model last)
            for row_line in data_rows[1:]:
                cells = _cells(row_line)
                for ci, lbl in bench_col_map.items():
                    if lbl in results:
                        continue
                    if ci < len(cells):
                        v = _parse_float(cells[ci])
                        if v is not None:
                            results[lbl] = v
            continue

        # Layout A: benchmark name in first column, score in second
        for row_line in data_rows[1:]:
            cells = _cells(row_line)
            if len(cells) < 2:
                continue
            lbl = _match_bench(cells[0])
            if lbl and lbl not in results:
                v = _parse_float(cells[1])
                if v is not None:
                    results[lbl] = v

    # ── Layout C: "BenchmarkName: score" on its own line ───────────────────
    if not results:
        for line in lines:
            m = re.match(
                r"^\s*\*{0,2}([A-Za-z][A-Za-z0-9 _()*-]*?)\*{0,2}"
                r"\s*[:|]\s*(\d+\.?\d*)\s*%?\s*$",
                line
            )
            if m:
                lbl = _match_bench(m.group(1))
                if lbl and lbl not in results:
                    v = _parse_float(m.group(2))
                    if v is not None:
                        results[lbl] = v

    # ── Layout D: HTML tables (used by Meta/Llama model cards) ──────────────
    if not results:
        for table_html in re.findall(r'<table[^>]*>(.*?)</table>', readme, re.DOTALL | re.IGNORECASE):
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)
            if len(rows) < 2:
                continue
            def _html_cells(row_html):
                return [re.sub(r'<[^>]+>', '', c).strip()
                        for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row_html, re.DOTALL | re.IGNORECASE)]
            header_cells = _html_cells(rows[0])
            bench_col_map = {ci: _match_bench(hcell) for ci, hcell in enumerate(header_cells) if _match_bench(hcell)}
            if bench_col_map:
                for row_html in rows[1:]:
                    cells = _html_cells(row_html)
                    for ci, lbl in bench_col_map.items():
                        if lbl not in results and ci < len(cells):
                            v = _parse_float(cells[ci])
                            if v is not None:
                                results[lbl] = v

    return results


def _extract_params_from_text(text: str, name: str = "") -> float | None:
    """
    Extract parameter count in billions from free text (README, description)
    or a model name / repo_id.

    Handles patterns like:
      7B  70B  1.5B  0.5B  3.8B  405B   — bare suffix
      8x7B  8×7B  2x3.8B               — MoE total (multiplied)
      7 billion  70 billion parameters  — spelled out
      7,000,000,000  7_000_000_000      — raw numbers
      130M  616M  1.1B                  — million-scale models
      llama-3-70b  qwen-2.5-7b-instruct — embedded in name (case-insensitive)

    Returns None if nothing plausible found.
    Sanity range: 0.05 B – 2000 B (ignores context sizes, dates, etc.)
    """
    combined = (text + " " + name).strip()
    if not combined:
        return None

    # Known size nicknames that don't embed a number
    _NICKNAMES = {
        "mini":   3.8,   # Phi-3/3.5-mini = 3.8B
        "small":  7.0,   # Phi-3-small = 7B
        "medium": 14.0,  # Phi-3-medium = 14B
        "tiny":   1.1,   # TinyLlama = 1.1B
    }

    # Remove comma/underscore thousand separators so 7,000,000,000 → 7000000000
    combined_clean = re.sub(r'(\d)[,_](\d)', r'\1\2', combined)

    # Pattern 1 — MoE: NxM.MB  (e.g. 8x7B, 2x3.8B, 8×7B)
    for m in re.finditer(
        r'\b(\d+)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[Bb]\b', combined_clean
    ):
        total = float(m.group(1)) * float(m.group(2))
        if 0.05 <= total <= 2000:
            return round(total, 2)

    # Pattern 2 — plain NB / N.NB (e.g. 7B, 1.5B, 70B, 0.5B)
    for m in re.finditer(
        r'(?<![0-9])(\d+(?:\.\d+)?)\s*[Bb](?:illions?|illion)?\b',
        combined_clean, re.IGNORECASE
    ):
        v = float(m.group(1))
        if 0.05 <= v <= 2000:
            return round(v, 2)

    # Pattern 3 — millions: NM / N.NM (e.g. 130M, 616M, 360M)
    for m in re.finditer(
        r'(?<![0-9])(\d+(?:\.\d+)?)\s*[Mm](?:illions?|illion)?\b',
        combined_clean, re.IGNORECASE
    ):
        v = float(m.group(1)) / 1000.0
        # Only sub-1B models with ≥10M params; excludes "3m" (3M), context sizes
        if 0.01 <= v < 1.0:
            return round(v, 3)

    # Pattern 4 — spelled out: "7 billion", "70 billion parameters"
    for m in re.finditer(
        r'\b(\d+(?:\.\d+)?)\s+billion(?:\s+parameters?)?\b',
        combined_clean, re.IGNORECASE
    ):
        v = float(m.group(1))
        if 0.05 <= v <= 2000:
            return round(v, 2)

    # Pattern 5 — raw large integer (≥100M, looks like a param count)
    for m in re.finditer(r'\b(\d{9,13})\b', combined_clean):
        v = int(m.group(1)) / 1e9
        if 0.05 <= v <= 2000:
            return round(v, 2)

    # Pattern 6 — known nicknames (only when no numeric size found above)
    name_lower = name.lower()
    for nick, size_b in _NICKNAMES.items():
        # Must appear as a word boundary in the name, not the body text
        if re.search(r'\b' + nick + r'\b', name_lower):
            return size_b

    return None



# Weights for each leaderboard version.  Higher = more discriminating benchmark.
_V1_WEIGHTS: dict[str, float] = {
    "ARC":        1.0,
    "HellaSwag":  0.7,   # saturated for modern models
    "MMLU":       1.5,   # broad knowledge, high signal
    "TruthfulQA": 1.0,
    "Winogrande": 0.7,   # also saturated
    "GSM8K":      1.5,   # math reasoning
}
_V2_WEIGHTS: dict[str, float] = {
    "ARC":   0.8,   # also exists in v1, less novel
    "BBH":   1.5,   # Big Bench Hard
    "MATH":  1.5,   # hard math, very discriminating
    "GPQA":  1.5,   # expert-level science
    "MMLU":  1.2,   # MMLU-Pro (harder than v1 MMLU)
}
# Affine calibration to map v2 weighted-average into v1-equivalent space.
# Derived from overlapping models (Llama-3 8B/70B, Mistral-7B, Qwen2-72B, Gemma-2-9B).
_V2_SCALE  = 0.958
_V2_OFFSET = 35.6


def _compute_openllm_score(benchmarks: dict) -> tuple[float | None, str, int]:
    """
    Compute a single comparable Open LLM score from whatever benchmark data
    is available, normalised to v1-equivalent space so v1 and v2 models can
    be ranked together.

    Strategy:
      1. Only use the v2 path when at least one v2-exclusive benchmark is present
         (BBH, MATH, GPQA, MUSR, IFEval). ARC and MMLU exist in both weight
         sets, so without a v2-exclusive benchmark we cannot distinguish v1 from
         v2 data and must treat it as v1.
      2. Require weighted coverage >= 40% AND at least 2 benchmarks from the
         chosen set — below this the average is too noisy to be meaningful.
      3. Never mix v1 and v2 benchmarks in the same score.
      4. v2 scores are mapped to v1-equivalent space via affine calibration
         (×0.958 + 35.6) derived from models that appear on both leaderboards.

    Returns (score_or_None, version_str, n_benchmarks_used).
    version_str is "v2" or "v1".  n_benchmarks_used indicates how many
    benchmarks contributed (shown in the tooltip).
    """
    if not benchmarks:
        return None, "", 0

    # Benchmarks that only exist on v2 — presence of any means v2 data
    _V2_EXCLUSIVE = {"BBH", "MATH", "GPQA", "MUSR", "IFEval"}
    has_v2_exclusive = any(k in _V2_EXCLUSIVE for k in benchmarks)

    # ── Try v2 path only when v2-exclusive data is present ────────────────────
    if has_v2_exclusive:
        v2_present = {k: v for k, v in benchmarks.items() if k in _V2_WEIGHTS}
        v2_total_w = sum(_V2_WEIGHTS[k] for k in v2_present)
        v2_max_w   = sum(_V2_WEIGHTS.values())
        if len(v2_present) >= 2 and v2_total_w / v2_max_w >= 0.40:
            raw   = sum(benchmarks[k] * _V2_WEIGHTS[k] for k in v2_present) / v2_total_w
            score = round(raw * _V2_SCALE + _V2_OFFSET, 1)
            return min(score, 99.9), "v2", len(v2_present)

    # ── v1 path (default for v1 data, or v2 fallback if coverage too low) ─────
    v1_present = {k: v for k, v in benchmarks.items() if k in _V1_WEIGHTS}
    v1_total_w = sum(_V1_WEIGHTS[k] for k in v1_present)
    v1_max_w   = sum(_V1_WEIGHTS.values())
    if len(v1_present) >= 2 and v1_total_w / v1_max_w >= 0.40:
        score = round(sum(benchmarks[k] * _V1_WEIGHTS[k] for k in v1_present) / v1_total_w, 1)
        return min(score, 99.9), "v1", len(v1_present)

    return None, "", 0


def fetch_hf_metadata(repo_id: str, session: requests.Session) -> dict:
    """
    Fetch model metadata from HuggingFace API.
    Returns a normalised metadata dict.
    """
    result = {
        "repo_id":        repo_id,
        "hf_url":         f"https://huggingface.co/{repo_id}",
        "description":    "",
        "author":         "",
        "license":        "",
        "base_model":     "",
        "parameters_b":   None,   # billions
        "language":       [],
        "tags":           [],
        "downloads":      None,
        "likes":          None,
        "pipeline_tag":   "",
        "benchmarks":     {},
        "lm_arena_elo":   None,
        "openllm_score":  None,   # weighted benchmark average, v1-normalised
        "openllm_version": "",    # "v1" or "v2"
        "openllm_n":       0,     # number of benchmarks used
        "lb_rank":         None,  # HF leaderboard rank number
        "lb_avg":          None,  # HF leaderboard published average score (%)
        "lb_version":      "",    # "v1" or "v2"
        "lb_sort_key":     None,  # unified sort key for cross-version comparison
        "is_analogue":     False, # benchmarks sourced from base model, not this model directly
        "analogue_model":  "",    # which ancestor provided the benchmarks
        "fetched_at":     time.time(),
        "fetch_error":    None,
    }

    try:
        r = session.get(
            f"{HF_API}/models/{repo_id}",
            params={"full": True, "cardData": True},
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 404:
            result["fetch_error"] = "not found"
            return result
        if r.status_code != 200:
            result["fetch_error"] = f"HTTP {r.status_code}"
            return result

        data = r.json()
        card = data.get("cardData") or {}

        result["author"]       = data.get("author", "")
        result["downloads"]    = data.get("downloads")
        result["likes"]        = data.get("likes")
        result["pipeline_tag"] = data.get("pipeline_tag", "")
        result["tags"]         = data.get("tags", [])
        result["language"]     = card.get("language", [])

        # License
        lics = card.get("license") or data.get("license")
        if isinstance(lics, list):
            result["license"] = ", ".join(lics)
        elif lics:
            result["license"] = str(lics)

        # Base model
        base = card.get("base_model") or card.get("base_models")
        if isinstance(base, list):
            result["base_model"] = base[0] if base else ""
        elif base:
            result["base_model"] = str(base)

        # Model description (first non-empty paragraph of README)
        readme = data.get("cardData", {}).get("text", "") or ""
        if not readme:
            # Try siblings for README.md
            try:
                rr = session.get(
                    f"https://huggingface.co/{repo_id}/raw/main/README.md",
                    timeout=REQUEST_TIMEOUT)
                if rr.status_code == 200:
                    readme = rr.text
            except Exception:
                pass

        if readme:
            # Extract first substantive paragraph (skip YAML front matter + headers)
            in_yaml = False
            for line in readme.splitlines():
                stripped = line.strip()
                if stripped == "---":
                    in_yaml = not in_yaml
                    continue
                if in_yaml:
                    continue
                if stripped.startswith("#"):
                    continue
                if len(stripped) > 40:
                    result["description"] = stripped[:300]
                    break

        # ── Parameter count — cascading extraction ───────────────────────────
        # Tier 1: safetensors.total from HF API (most reliable — exact count)
        safetensors = data.get("safetensors", {})
        total_params = safetensors.get("total") or 0
        if total_params > 0:
            result["parameters_b"] = round(total_params / 1e9, 2)

        # Tier 2: model card YAML field (some authors populate this explicitly)
        if result["parameters_b"] is None:
            _pc_card = card.get("model-index", [{}])
            # cardData sometimes has a top-level num_parameters or parameter_count
            for _key in ("num_parameters", "parameter_count", "parameters"):
                _v = card.get(_key)
                if _v:
                    try:
                        _n = float(str(_v).replace(",", "").replace("_", ""))
                        # Value may be raw count (>1e6) or already in billions
                        result["parameters_b"] = round(_n / 1e9 if _n > 1e6 else _n, 2)
                        break
                    except (ValueError, TypeError):
                        pass

        # Tier 3: parse from README text — handles "7B", "70B", "8x7B", "1.5B",
        #         "405B", "3.8 billion", "32 billion parameters" etc.
        if result["parameters_b"] is None and readme:
            result["parameters_b"] = _extract_params_from_text(readme, repo_id)

        # Tier 4: parse from repo_id / model name alone
        if result["parameters_b"] is None:
            result["parameters_b"] = _extract_params_from_text("", repo_id)

        # Benchmark scores — YAML structured data first
        result["benchmarks"] = _extract_benchmarks(card)

        # README markdown table fallback (only for non-GGUF source repos)
        # Skip for GGUF quant repos — their READMEs have no benchmark tables
        _is_gguf_repo = ("gguf" in repo_id.lower() or
                         "GGUF" in result.get("tags", []) or
                         any(t in result.get("tags", []) for t in ["gguf", "GGUF"]))
        if not result["benchmarks"] and not _is_gguf_repo and readme:
            _readme_bench = _parse_readme_benchmarks(readme)
            if _readme_bench:
                result["benchmarks"].update(_readme_bench)
                result["_bench_source"] = "readme_table"

        # LM Arena ELO — stored in card metadata by some models
        elo = (card.get("lm_arena_elo")
               or card.get("arena_elo")
               or card.get("chatbot_arena_elo"))
        if elo:
            try:
                result["lm_arena_elo"] = int(float(elo))
            except Exception:
                pass

    except Exception as e:
        result["fetch_error"] = str(e)

    return result


def _fetch_open_llm_leaderboard_v1(repo_id: str, session: requests.Session) -> dict:
    """
    Fetch benchmark results from Open LLM Leaderboard v1 (pre-June 2024).
    Benchmarks: ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K.

    Results are stored in a per-model dataset named:
      open-llm-leaderboard-old/details_{org}__{model}
    (note double underscore between org and model)

    Tries multiple repo_id variants:
    - The given repo_id
    - Org alias variants (CohereLabs ↔ CohereForAI, etc.)
    - Context-window suffix stripped (model-7b-16k → model-7b)
    """
    try:
        org, model = repo_id.split("/", 1)
    except ValueError:
        return {}

    # Known org name aliases on the leaderboard vs HF model hub
    _ORG_ALIASES = {
        "coherelabs":  "CohereForAI",
        "cohereforai": "CohereLabs",
    }
    # Skip mirror orgs that are never on the leaderboard
    _MIRROR_ORGS = {"mlx-community", "miqudev", "lmstudio-community",
                    "ggml-org", "bartowski", "mradermacher", "thebloke",
                    "second-state", "tensorblock"}

    # Build list of repo variants to try
    variants = []
    if org.lower() not in _MIRROR_ORGS:
        variants.append((org, model))
    # Org alias
    alias_org = _ORG_ALIASES.get(org.lower())
    if alias_org:
        variants.append((alias_org, model))
    # Strip context-window suffix from model name
    _ctx_re = re.compile(r'[-_](?:16k|32k|64k|128k|200k|256k|1m|2m|1048k)$', re.IGNORECASE)
    model_no_ctx = _ctx_re.sub('', model)
    if model_no_ctx != model:
        if org.lower() not in _MIRROR_ORGS:
            variants.append((org, model_no_ctx))
        if alias_org:
            variants.append((alias_org, model_no_ctx))
    # HF leaderboard commonly appends "-hf" or "-chat-hf" to model names
    # e.g. meta-llama/Llama-2-7b → leaderboard has meta-llama/Llama-2-7b-hf
    #      meta-llama/CodeLlama-7b-Instruct → CodeLlama-7b-Instruct-hf
    for _base_model in [model, model_no_ctx]:
        if not _base_model.lower().endswith('-hf'):
            if org.lower() not in _MIRROR_ORGS:
                variants.append((org, _base_model + '-hf'))
            if alias_org:
                variants.append((alias_org, _base_model + '-hf'))

    for try_org, try_model in variants:
        dataset_id = f"open-llm-leaderboard-old/details_{try_org}__{try_model}"

        # List files in the dataset to find the results JSON
        list_url = f"https://huggingface.co/api/datasets/{dataset_id}/tree/main"
        try:
            r = session.get(list_url, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                continue
            files = r.json()
            if not isinstance(files, list):
                continue
            result_files = sorted([
                f["path"] for f in files
                if isinstance(f, dict)
                and f.get("path", "").endswith(".json")
                and "results_" in f.get("path", "")
            ], reverse=True)
            if not result_files:
                continue
        except Exception:
            continue

        # Fetch and MERGE all result files (most recent first, setdefault fills gaps)
        merged_results: dict = {}
        for rpath in result_files[:4]:
            _url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{rpath}"
            try:
                time.sleep(REQUEST_DELAY)
                _r = session.get(_url, timeout=REQUEST_TIMEOUT)
                if _r.status_code != 200:
                    continue
                _file_results = _r.json().get("results", {})
                for k, v in _file_results.items():
                    merged_results.setdefault(k, v)
            except Exception:
                continue

        if not merged_results:
            continue

        results = merged_results  # alias for helpers below
        benchmarks = {}

        def _get_first(fragments: list, metrics: list) -> float | None:
            for frag in fragments:
                for key in results:
                    if frag.lower() in key.lower():
                        task = results[key]
                        for mk in metrics:
                            v = task.get(mk)
                            if v is not None and isinstance(v, (int, float)) and float(v) > 0:
                                return float(v)
            return None

        def _avg_subtasks(fragment: str, metrics: list) -> float | None:
            vals = []
            for key in results:
                if fragment.lower() in key.lower():
                    task = results[key]
                    for mk in metrics:
                        v = task.get(mk)
                        if v is not None and isinstance(v, (int, float)) and float(v) > 0:
                            vals.append(float(v))
                            break
            return sum(vals) / len(vals) if vals else None

        v = _get_first(["arc_challenge", "arc:challenge", "|arc"], ["acc_norm", "acc"])
        if v is not None: benchmarks["ARC"] = round(v * 100, 1)

        v = _get_first(["hellaswag"], ["acc_norm", "acc"])
        if v is not None: benchmarks["HellaSwag"] = round(v * 100, 1)

        v = _get_first(["|mmlu|", "harness|mmlu"], ["acc", "acc_norm"])
        if v is None: v = _avg_subtasks("hendrycksTest", ["acc", "acc_norm"])
        if v is None: v = _avg_subtasks("mmlu", ["acc", "acc_norm"])
        if v is not None: benchmarks["MMLU"] = round(v * 100, 1)

        v = _get_first(["truthfulqa"], ["mc2", "acc"])
        if v is not None: benchmarks["TruthfulQA"] = round(v * 100, 1)

        v = _get_first(["winogrande"], ["acc", "acc_norm"])
        if v is not None: benchmarks["Winogrande"] = round(v * 100, 1)

        v = _get_first(["gsm8k"], ["acc", "acc_norm", "exact_match"])
        if v is not None: benchmarks["GSM8K"] = round(v * 100, 1)

        # Return on first variant that yields results
        if benchmarks:
            return benchmarks

    return {}


def _fetch_open_llm_leaderboard_v2(repo_id: str, session: requests.Session) -> dict:
    """
    Fetch benchmark results from Open LLM Leaderboard v2 (June 2024+).
    Benchmarks: ARC, BBH, MATH-Hard, GPQA, MMLU-Pro.

    Uses the AGGREGATED public dataset open-llm-leaderboard/results
    at path: {org}/{model}/results_{timestamp}.json  (single slash, not double underscore)

    The per-model *-details datasets are individually gated (require contact-sharing
    agreement per dataset) and return 403 even with a valid read token.

    JSON: {"all": {"leaderboard_arc_challenge": {"acc_norm,none": 0.69}, ...}}
    """
    try:
        org, model = repo_id.split("/", 1)
    except ValueError:
        return {}

    dataset_id = "open-llm-leaderboard/results"
    model_path = f"{org}/{model}"

    list_url = (f"https://huggingface.co/api/datasets/{dataset_id}"
                f"/tree/main/{model_path}")
    try:
        r = session.get(list_url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return {}
        files = r.json()
        if not isinstance(files, list):
            return {}
        result_files = sorted([
            f["path"] for f in files
            if isinstance(f, dict)
            and f.get("path", "").endswith(".json")
            and "results_" in f.get("path", "")
        ], reverse=True)
        if not result_files:
            return {}
    except Exception:
        return {}

    raw_url = (f"https://huggingface.co/datasets/{dataset_id}"
               f"/resolve/main/{result_files[0]}")
    try:
        time.sleep(REQUEST_DELAY)
        r = session.get(raw_url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return {}
        data = r.json()
    except Exception:
        return {}

    all_results = data.get("all", data.get("results", {}))
    if not all_results:
        return {}

    benchmarks = {}

    def _get(frags: list, metrics: list) -> float | None:
        for frag in frags:
            for key in all_results:
                if frag.lower() in key.lower():
                    task = all_results[key]
                    for mk in metrics:
                        for sfx in ("", ",none"):
                            v = task.get(mk + sfx)
                            if v is not None and isinstance(v, (int, float)) and float(v) > 0:
                                return float(v)
        return None

    v = _get(["arc_challenge", "arc:challenge"], ["acc_norm", "acc"])
    if v is not None:
        benchmarks["ARC"] = round(v * 100, 1)

    v = _get(["bbh"], ["acc_norm", "acc"])
    if v is not None:
        benchmarks["BBH"] = round(v * 100, 1)

    v = _get(["math_hard", "math"], ["exact_match", "acc_norm", "acc"])
    if v is not None:
        benchmarks["MATH"] = round(v * 100, 1)

    v = _get(["gpqa"], ["acc_norm", "acc"])
    if v is not None:
        benchmarks["GPQA"] = round(v * 100, 1)

    v = _get(["mmlu_pro", "mmlu"], ["acc", "acc_norm"])
    if v is not None:
        benchmarks["MMLU"] = round(v * 100, 1)

    return benchmarks


def _fetch_lmarena_elo(session: requests.Session) -> dict[str, int]:
    """
    Fetch current Arena ELO scores from the lmarena-ai leaderboard space.
    Returns {model_name_lower: elo_int} for fuzzy matching.

    Data source: leaderboard_table_YYYYMMDD.csv files in the lmarena-ai HF space.
    These are updated monthly and require no auth token.
    """
    # List files in the space to find the most recent leaderboard CSV
    list_url = "https://huggingface.co/api/spaces/lmarena-ai/lmarena-leaderboard/tree/main"
    try:
        r = session.get(list_url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return {}
        files = r.json()
        csv_files = sorted([
            f["path"] for f in files
            if isinstance(f, dict) and f.get("path", "").startswith("leaderboard_table_")
            and f.get("path", "").endswith(".csv")
        ], reverse=True)
        if not csv_files:
            return {}
        latest = csv_files[0]
    except Exception:
        return {}

    csv_url = (f"https://huggingface.co/spaces/lmarena-ai/lmarena-leaderboard"
               f"/resolve/main/{latest}")
    try:
        time.sleep(REQUEST_DELAY)
        r = session.get(csv_url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return {}
        lines = r.text.splitlines()
    except Exception:
        return {}

    # Parse CSV: find "Model" and "Arena Elo" column indices
    if not lines:
        return {}
    header = [h.strip().strip('"').lower() for h in lines[0].split(",")]
    try:
        model_col = next(i for i, h in enumerate(header) if "model" in h)
        elo_col   = next(i for i, h in enumerate(header)
                        if "elo" in h and "arena" in h)
    except StopIteration:
        return {}

    elo_map: dict[str, int] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) <= max(model_col, elo_col):
            continue
        name = parts[model_col].strip().strip('"').lower()
        try:
            elo = int(float(parts[elo_col].strip().strip('"')))
            if elo > 0:
                elo_map[name] = elo
        except (ValueError, IndexError):
            continue
    return elo_map



def _fetch_lb_rank(repo_id: str, version_hint: str,
                   session: requests.Session) -> tuple[int | None, str]:
    """
    Fetch this model's rank on the HF Open LLM Leaderboard by querying
    the datasets-server for just its row — no full table download.

    Uses the /rows?where= filter supported by the HF Datasets Server API.
    Tries V2 first (open-llm-leaderboard/contents, col "Model"),
    then V1 (open-llm-leaderboard-old/results, col "model").

    Returns (rank_int_or_None, version_str).
    rank is the row's position in the pre-sorted table (1-indexed).
    """
    if not repo_id or "/" not in repo_id:
        return None, ""

    HF_DS = "https://datasets-server.huggingface.co"

    def _query(dataset: str, model_col: str, split: str = "train") -> int | None:
        try:
            r = session.get(
                f"{HF_DS}/rows",
                params={"dataset": dataset, "split": split,
                        "where": f"{model_col} = '{repo_id}'",
                        "offset": 0, "length": 1},
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code != 200:
                return None
            data = r.json()
            rows = data.get("rows", [])
            if not rows:
                return None
            # The row_idx field gives the position in the full sorted table
            row_item = rows[0]
            row_idx  = row_item.get("row_idx")  # 0-indexed position in dataset
            if row_idx is not None:
                return int(row_idx) + 1   # convert to 1-indexed rank
        except Exception:
            pass
        return None

    # Try V2 first unless we know it's V1 data
    if version_hint != "v1":
        rank = _query("open-llm-leaderboard/contents", "Model")
        if rank:
            return rank, "v2"

    # Try V1
    rank = _query("open-llm-leaderboard-old/results", "model")
    if rank:
        return rank, "v1"

    # If version_hint was v1, also try v2 as fallback
    if version_hint == "v1":
        rank = _query("open-llm-leaderboard/contents", "Model")
        if rank:
            return rank, "v2"

    return None, ""


    """
    Fetch the HF Open LLM Leaderboard aggregated rank tables once, returning:
        { "org/model-name-lower": {"rank": int, "avg": float} }
    for both V1 and V2.

    V1: open-llm-leaderboard-old/results  (~5 500 models, closed Nov 2024)
        Columns: "model" (org/name), "average" (0-100)
        Rows pre-sorted by average descending → rank = row position.
    V2: open-llm-leaderboard/contents     (~3 000 models, ongoing)
        Columns: "Model" (org/name), avg col contains "Average" or "⬆" in name
        Rows pre-sorted by average descending → rank = row position.

    Uses the HF Datasets Server /first-rows endpoint to discover column names
    dynamically (handles emoji column names like "Average ⬆️" without hardcoding).
    Falls back to empty dicts silently so the report still works.
    """
    HF_DS   = "https://datasets-server.huggingface.co"

    def _discover_columns(dataset: str) -> tuple[str, str, str] | None:
        """Return (split, model_col, avg_col) by inspecting the first row."""
        for split in ("train", "default", ""):
            params = {"dataset": dataset, "split": split} if split else {"dataset": dataset}
            try:
                r = session.get(f"{HF_DS}/first-rows", params=params,
                                timeout=REQUEST_TIMEOUT)
                if r.status_code != 200:
                    continue
                data = r.json()
                rows = data.get("rows", [])
                if not rows:
                    continue
                cols = list(rows[0].get("row", {}).keys())
                # Find model column: exact "model" or "Model"
                model_col = next(
                    (c for c in cols if c.lower() == "model"), None)
                # Find average column: contains "average" (case-insensitive, ignores emoji)
                avg_col = next(
                    (c for c in cols
                     if "average" in c.lower().encode("ascii", "ignore").decode()),
                    None)
                if model_col and avg_col:
                    return split, model_col, avg_col
            except Exception:
                continue
        return None

    def _fetch_table(dataset: str) -> dict:
        info = _discover_columns(dataset)
        if not info:
            return {}
        split, model_col, avg_col = info
        table: dict[str, dict] = {}
        offset = 0
        length = 100
        errors = 0
        while True:
            params = {"dataset": dataset, "split": split,
                      "offset": offset, "length": length}
            if not split:
                del params["split"]
            try:
                r = session.get(f"{HF_DS}/rows", params=params,
                                timeout=REQUEST_TIMEOUT)
                if r.status_code != 200:
                    break
                rows = r.json().get("rows", [])
                if not rows:
                    break
                for i, item in enumerate(rows):
                    row  = item.get("row", {})
                    name = str(row.get(model_col, "") or "").strip().lower()
                    if not name:
                        continue
                    try:
                        avg  = float(row[avg_col])
                        rank = offset + i + 1   # rows are pre-sorted desc by avg
                        table[name] = {"rank": rank, "avg": round(avg, 2)}
                    except (KeyError, TypeError, ValueError):
                        pass
                if len(rows) < length:
                    break
                offset += length
                time.sleep(REQUEST_DELAY)
            except Exception:
                errors += 1
                if errors >= 3:
                    break
        return table

    v1: dict = {}
    v2: dict = {}
    try:
        v1 = _fetch_table("open-llm-leaderboard-old/results")
    except Exception:
        pass
    try:
        v2 = _fetch_table("open-llm-leaderboard/contents")
    except Exception:
        pass
    return v1, v2

def _lb_sort_key(rank: int, version: str) -> float:
    """
    Convert a leaderboard rank to a unified sort key in [0, 99.9].
    Higher = better model.

    V1: straight percentile  → (1 − (rank−1) / 5500) × 100
    V2: percentile + 5 pt difficulty bonus, capped at 99.9
        → min(99.9,  (1 − (rank−1) / 3000) × 100  + 5.0)

    The +5 bonus means that at any equal percentile position, a V2 model sorts
    above the equivalent V1 model, reflecting the harder evaluation suite.
    Same-number ranks (e.g. #100 V2 vs #100 V1) also favour V2 because
    3 000 total models means a lower rank is a higher top-percentile position.
    """
    if version == "v2":
        raw = (1 - (rank - 1) / _LB_V2_TOTAL) * 100
        return min(99.9, raw + _LB_V2_BONUS)
    else:
        return (1 - (rank - 1) / _LB_V1_TOTAL) * 100



def _match_lmarena_elo(model_identifier: str, elo_map: dict[str, int]) -> int | None:
    """
    Fuzzy-match a model name/identifier to the lmarena ELO map.
    Tries exact match, then progressively looser substring matching.
    """
    if not elo_map or not model_identifier:
        return None
    clean = model_identifier.lower().strip()

    # Direct lookup
    if clean in elo_map:
        return elo_map[clean]

    # Try known canonical name patterns
    # e.g. "GPT-4o" → look for "gpt-4o" anywhere in keys
    # e.g. "claude-opus-4-6" → look for "claude-opus-4"
    # Build candidate keys by stripping version date suffixes
    candidates = []
    for key in elo_map:
        # Score: length of longest common substring normalised by key length
        common = 0
        cl = clean.replace("-","").replace(".","").replace(" ","")
        kl = key.replace("-","").replace(".","").replace(" ","")
        for length in range(min(len(cl), len(kl)), 3, -1):
            for start in range(len(cl) - length + 1):
                sub = cl[start:start+length]
                if sub in kl:
                    common = length
                    break
            if common:
                break
        if common >= 6:  # at least 6 chars in common
            score = common / max(len(cl), len(kl))
            candidates.append((score, key, elo_map[key]))

    if candidates:
        candidates.sort(reverse=True)
        if candidates[0][0] >= 0.6:  # 60% match threshold
            return candidates[0][2]

    return None


def enrich_with_hf(
    results:      list,
    cache_path:   Path,
    force_refresh: bool,
    no_hf:        bool,
) -> list:
    """
    For each result, look up HF metadata and merge it in.
    Respects the cache and refresh policy.
    """
    if no_hf:
        print("  [HF] Skipping HF metadata fetch (--no-hf)")
        for r in results:
            r["hf"] = {}
        return results

    cache   = _load_cache(cache_path)
    session = requests.Session()
    _hdrs = {"User-Agent": "llm-optimizer-report/1.0"}
    if HF_TOKEN:
        _hdrs["Authorization"] = f"Bearer {HF_TOKEN}"
    session.headers.update(_hdrs)

    _SHARD_STEM_RE = re.compile(r'-(\d{5})-of-\d{5}$', re.IGNORECASE)

    def _canonical_stem(stem: str) -> str:
        """Strip shard index so all shards share one cache entry."""
        m = _SHARD_STEM_RE.search(stem)
        return stem[:m.start()] if m else stem

    total = len(results)
    for i, r in enumerate(results, 1):
        model_path = Path(r.get("model_path", ""))
        stem       = model_path.stem
        cache_key  = _canonical_stem(stem)  # shards share one cache entry

        existing = cache.get(cache_key)

        # Force re-fetch if cached entry has no benchmarks and hasn't been
        # retried more than 3 times (the base_model follow logic may find
        # benchmarks that weren't found on the previous fetch attempt)
        _no_bench = (existing is not None
                     and not existing.get("benchmarks")
                     and not existing.get("fetch_error")
                     and existing.get("_bench_retries", 0) < 3)

        need_fetch = (
            force_refresh
            or existing is None
            or _cache_needs_refresh(existing, model_path)
            or _no_bench
        )

        if not need_fetch and existing:
            # Recompute openllm_score in case this cache entry predates the field
            if "openllm_score" not in existing:
                _ols, _olv, _oln = _compute_openllm_score(existing.get("benchmarks", {}))
                existing["openllm_score"]   = _ols
                existing["openllm_version"] = _olv
                existing["openllm_n"]       = _oln
            # Look up lb rank/avg for cache hits too (tables are fetched fresh each run)
            # Recompute lb fields on cache hits
            _bc  = existing.get("benchmarks", {})
            _hv2 = any(_bc.get(k) for k in ["BBH","GPQA","MUSR","IFEval"])
            _hv1 = any(_bc.get(k) for k in ["HellaSwag","TruthfulQA","Winogrande"])
            _lbv = "v2" if _hv2 else ("v1" if _hv1 else "")
            _kk  = (["IFEval","BBH","MATH","GPQA","MUSR","MMLU"] if _lbv == "v2"
                    else ["ARC","HellaSwag","MMLU","TruthfulQA","Winogrande","GSM8K"])
            _vv  = [_bc[k] for k in _kk if _bc.get(k) is not None]
            existing["lb_avg"]     = round(sum(_vv)/len(_vv), 2) if len(_vv) >= 2 else None
            existing["lb_version"] = _lbv
            if "lb_rank" not in existing or existing.get("lb_rank") is None:
                _rc = existing.get("repo_id", "")
                _rk, _rv = _fetch_lb_rank(_rc, _lbv, session)
                if _rv: existing["lb_version"] = _rv
                existing["lb_rank"]     = _rk
                existing["lb_sort_key"] = _lb_sort_key(_rk, existing["lb_version"]) if _rk else None
            r["hf"] = existing
            print(f"  [HF] {i}/{total} {stem[:50]} — cache hit")
            continue

        _display_stem = _canonical_stem(stem)

        # On --refresh-hf, reuse the cached repo_id only when the previous run
        # actually found benchmarks through it (directly or via base_model chain).
        # If it found nothing, re-search — a better repo_id may exist.
        _cached_entry = existing or {}
        _cached_repo  = _cached_entry.get("repo_id", "")
        _had_benchmarks = bool(_cached_entry.get("benchmarks")) or bool(_cached_entry.get("_bench_source"))
        _had_error      = bool(_cached_entry.get("fetch_error"))

        if _cached_repo and _had_benchmarks and not _had_error:
            repo_id = _cached_repo
            print(f"  [HF] {i}/{total} {_display_stem[:50]} — refreshing {repo_id}", end="", flush=True)
        else:
            print(f"  [HF] {i}/{total} {_display_stem[:50]} — searching...", end="", flush=True)
            repo_id = search_hf_model(stem, session)
        if repo_id is None:
            print(f" not found")
            hf_data = {"fetch_error": "no HF match found", "fetched_at": time.time()}
        else:
            print(f" {repo_id}", end="", flush=True)
            time.sleep(REQUEST_DELAY)
            hf_data = fetch_hf_metadata(repo_id, session)
            # Try Open LLM Leaderboard v1 (ARC/HellaSwag/MMLU/TruthfulQA/Winogrande/GSM8K)
            _lb1 = _fetch_open_llm_leaderboard_v1(repo_id, session)
            if _lb1:
                hf_data["benchmarks"].update(_lb1)
                hf_data["_bench_source"] = "open-llm-leaderboard-v1"

            # Try Open LLM Leaderboard v2 (BBH/MATH/GPQA and newer ARC/MMLU)
            if not all(k in hf_data["benchmarks"] for k in ["BBH", "MATH", "GPQA"]):
                _lb2 = _fetch_open_llm_leaderboard_v2(repo_id, session)
                if _lb2:
                    hf_data["benchmarks"].update(_lb2)

            # Quick static lookup for well-known models (Qwen3/Gemma3/Llama3.2 etc.)
            # whose benchmarks come from official tech reports, not the leaderboard.
            if not hf_data["benchmarks"] and repo_id in _STATIC_BENCH:
                hf_data["benchmarks"].update(_STATIC_BENCH[repo_id])
                hf_data["_bench_source"] = "static_tech_report"

            # Walk the base_model chain recursively until we find benchmarks
            # or exhaust the ancestry (max 4 hops to avoid infinite loops).
            # This handles: GGUF → safetensors fine-tune → original base model
            # When benchmarks come from a hop ≥ 1, mark as analogue so the report
            # can display a visual indicator.
            _visited = {repo_id}
            _current_data = hf_data
            _hops = 0
            while not hf_data.get("benchmarks") and _current_data.get("base_model") and _hops < 4:
                base_id = _current_data["base_model"]
                # Normalise: strip leading slash, handle "https://huggingface.co/X/Y" URLs
                base_id = base_id.strip("/")
                if base_id.startswith("https://huggingface.co/"):
                    base_id = base_id[len("https://huggingface.co/"):]
                if base_id in _visited or "/" not in base_id:
                    break
                _visited.add(base_id)
                _hops += 1
                print(f" → {base_id}", end="", flush=True)
                time.sleep(REQUEST_DELAY)

                # Static tech report lookup first (fastest, no API call)
                if base_id in _STATIC_BENCH and not hf_data["benchmarks"]:
                    hf_data["benchmarks"].update(_STATIC_BENCH[base_id])
                    hf_data["_bench_source"] = f"{base_id} (static, hop {_hops})"
                    hf_data["is_analogue"]   = True
                    hf_data["analogue_model"] = base_id
                    break

                # Leaderboard lookups (most reliable)
                _lb1 = _fetch_open_llm_leaderboard_v1(base_id, session)
                _lb2 = _fetch_open_llm_leaderboard_v2(base_id, session)
                _lb = {**_lb1, **_lb2}
                if _lb:
                    hf_data["benchmarks"].update(_lb)
                    hf_data["_bench_source"]  = f"{base_id} (leaderboard, hop {_hops})"
                    hf_data["is_analogue"]    = True
                    hf_data["analogue_model"] = base_id

                # Fetch the base model card to get benchmarks (card YAML + README)
                # and the next base_model link in the chain
                _current_data = fetch_hf_metadata(base_id, session)
                if _current_data.get("benchmarks") and not hf_data["benchmarks"]:
                    hf_data["benchmarks"]     = _current_data["benchmarks"]
                    hf_data["_bench_source"]  = f"{base_id} (card, hop {_hops})"
                    hf_data["is_analogue"]    = True
                    hf_data["analogue_model"] = base_id

                # Pull description / params from first ancestor that has them
                if not hf_data.get("description") and _current_data.get("description"):
                    hf_data["description"] = _current_data["description"]
                if not hf_data.get("parameters_b") and _current_data.get("parameters_b"):
                    hf_data["parameters_b"] = _current_data["parameters_b"]
                if not hf_data.get("lm_arena_elo") and _current_data.get("lm_arena_elo"):
                    hf_data["lm_arena_elo"] = _current_data["lm_arena_elo"]

                if hf_data["benchmarks"]:
                    break
            n_bench = len(hf_data.get("benchmarks", {}))
            err     = hf_data.get("fetch_error")
            print(f" — {n_bench} benchmarks" + (f" [err: {err}]" if err else ""))

        # Track retry count so we don't loop forever on models with no benchmarks
        if not hf_data.get("benchmarks") and not hf_data.get("fetch_error"):
            prior_retries = (existing or {}).get("_bench_retries", 0)
            hf_data["_bench_retries"] = prior_retries + 1

        # Compute unified Open LLM leaderboard score (v1 or v2, normalised)
        _ols, _olv, _oln = _compute_openllm_score(hf_data.get("benchmarks", {}))
        hf_data["openllm_score"]   = _ols
        hf_data["openllm_version"] = _olv
        hf_data["openllm_n"]       = _oln

        # HF leaderboard avg (computed from benchmarks already fetched) + rank.
        _benches       = hf_data.get("benchmarks", {})
        _has_v2        = any(_benches.get(k) for k in ["BBH", "GPQA", "MUSR", "IFEval"])
        _has_v1        = any(_benches.get(k) for k in ["HellaSwag", "TruthfulQA", "Winogrande"])
        _lb_ver_local  = "v2" if _has_v2 else ("v1" if _has_v1 else "")
        _lb_keys       = (["IFEval","BBH","MATH","GPQA","MUSR","MMLU"] if _lb_ver_local == "v2"
                          else ["ARC","HellaSwag","MMLU","TruthfulQA","Winogrande","GSM8K"])
        _lb_vals       = [_benches[k] for k in _lb_keys if _benches.get(k) is not None]
        _lb_avg_local  = round(sum(_lb_vals) / len(_lb_vals), 2) if len(_lb_vals) >= 2 else None
        # Rank: query the datasets-server for just this model's row (no full table download)
        _repo          = hf_data.get("repo_id", "")
        _lb_rank_val, _lb_rank_ver = _fetch_lb_rank(_repo, _lb_ver_local, session)
        if _lb_rank_ver:  # rank found on a specific version
            _lb_ver_local = _lb_rank_ver
        hf_data["lb_rank"]     = _lb_rank_val
        hf_data["lb_avg"]      = _lb_avg_local
        hf_data["lb_version"]  = _lb_ver_local
        hf_data["lb_sort_key"] = (_lb_sort_key(_lb_rank_val, _lb_ver_local)
                                  if _lb_rank_val else None)

        cache[cache_key] = hf_data
        r["hf"] = hf_data

        # Save after each fetch so a crash doesn't lose everything
        _save_cache(cache, cache_path)
        time.sleep(REQUEST_DELAY)

    _save_cache(cache, cache_path)
    return results


# ── local data loading ─────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def enrich_with_local_meta(results: list, base_dir: Path | None = None) -> list:
    """
    Add GGUF metadata and sweep results to each result dict
    by reading the per-model JSON files directly.

    Path resolution strategy for results_dir (may be a relative Windows path):
      1. Exactly as stored if already absolute
      2. base_dir / results_dir        (report_json.parent.parent / relative)
      3. SCRIPT_DIR / results_dir      (generate_report.py's own directory)
      4. CWD / results_dir
      5. base_dir / "results" / slug   (reconstruct from model stem)
      6. SCRIPT_DIR / "results" / slug
      7. CWD / "results" / slug
    Uses the first candidate whose directory actually exists on disk.
    Falls back to root-level batch JSON fields when no sub-files are found.
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from model_utils import _cached_meta as _get_meta, kv_cache_mb_per_token as _kv_per_tok
    except ImportError:
        _get_meta = None
        _kv_per_tok = None

    _cwd = Path.cwd()

    for r in results:
        model_path = Path(r.get("model_path", ""))

        # Normalise backslashes so Path() parses correctly on all platforms.
        _rdir_str = r.get("results_dir", "").replace("\\", "/")
        _rdir_raw = Path(_rdir_str) if _rdir_str else Path(".")

        # Slug used to reconstruct the results dir from the model filename.
        _slug = (model_path.stem.lower().replace(" ", "_")
                 if model_path.stem else "")

        # Build ordered candidate list.
        _candidates: list[Path] = []
        if _rdir_raw.is_absolute():
            _candidates.append(_rdir_raw)
        else:
            for _base in filter(None, [base_dir, SCRIPT_DIR, _cwd]):
                _candidates.append(_base / _rdir_raw)
            if _slug:
                for _base in filter(None, [base_dir, SCRIPT_DIR, _cwd]):
                    _candidates.append(_base / "results" / _slug)

        # First candidate whose directory exists wins.
        rdir = _rdir_raw  # fallback (may not exist — handled gracefully below)
        for _c in _candidates:
            try:
                if _c.exists():
                    rdir = _c
                    break
            except Exception:
                continue

        # ── GGUF metadata ────────────────────────────────────────────────────
        gguf_meta = {}
        if _get_meta and model_path.exists():
            try:
                m = _get_meta(model_path)
                gguf_meta = {
                    "arch":           m.get("arch", ""),
                    "n_layers":       m.get("n_layers"),
                    "n_attn_layers":  m.get("n_attn_layers"),
                    "n_heads_kv":     m.get("n_heads_kv"),
                    "head_dim":       m.get("head_dim"),
                    "n_expert":       m.get("n_expert", 0),
                    "n_expert_used":  m.get("n_expert_used", 0),
                    "context_length": m.get("context_length"),
                    "parameters_b":   m.get("parameters_b"),   # from general.parameter_count
                    "is_moe":         m.get("is_moe", False),
                    "is_hybrid":      m.get("is_hybrid", False),
                    "is_thinking":    m.get("is_thinking", False),
                    "kv_mb_per_1k":   (round(_kv_per_tok(m) * 1024, 2)
                                       if _kv_per_tok else None),
                }
            except Exception as _e:
                gguf_meta["_parse_error"] = str(_e)
        r["gguf"] = gguf_meta

        # current_quant fallback
        if not r.get("current_quant"):
            _qm = re.search(
                r'[._-]((?:IQ|iq)\d+_?[A-Za-z0-9]*|[Qq]\d+_[A-Za-z0-9_]+|[Ff][Pp]16|[Bb][Ff]16)',
                model_path.stem)
            if _qm:
                r["current_quant"] = _qm.group(1).upper()

        # ── Topo sweep: sub-file > root-level batch JSON fields ──────────────
        topo = _read_json(rdir / "topo_sweep" / "topo_results.json")

        # Fallback: when topo sweep wasn't run, gpu_results.json records the
        # best ngl (GPU layer count) found during the GPU optimisation phase.
        # Use it to synthesise a "winner" label that's meaningful in the UI.
        _tc = topo.get("case")   or r.get("topo_case")   or ""
        _tw = topo.get("winner") or r.get("topo_winner") or ""
        if not _tw:
            _gpu_res = _read_json(rdir / "gpu_results.json")
            if _gpu_res.get("best_ngl"):
                _ngl = _gpu_res["best_ngl"]
                _tw = f"GPU ngl={_ngl}"
        r["topo_detail"] = {
            "case":          _tc,
            "winner":        _tw,
            "max_fit_ngl":   topo.get("max_fit_ngl"),
            "model_size_mb": topo.get("model_size_mb"),
            "scenarios": [
                {"scenario": s["scenario"], "label": s["label"],
                 "score": s.get("score", 0),
                 "gen_tps": s.get("gen_tps", 0),
                 "status": s.get("status", "")}
                for s in topo.get("scenarios", [])
                if s.get("status") == "ok"
            ],
        }

        # ── Context sweep: sub-file > root-level batch JSON fields ───────────
        ctx = _read_json(rdir / "ctx_sweep" / "ctx_results.json")
        r["ctx_detail"] = {
            "ctx_gpu_single":   ctx.get("ctx_gpu_single")   or r.get("ctx_gpu"),
            "ctx_gpu_combined": ctx.get("ctx_gpu_combined"),
            "ctx_ram_mixed":    ctx.get("ctx_ram_mixed")    or r.get("ctx_ram"),
            "ctx_ram_explicit": ctx.get("ctx_ram_explicit"),
            "recommended_ctx":  (ctx.get("recommended_ctx")
                                 or r.get("recommended_ctx")
                                 or r.get("rec_ctx")),
            "trained_max":      ctx.get("trained_max"),
        }

        # Integrity check result
        integ = _read_json(rdir / "integrity.json")
        if integ:
            # Find the non-f16 key (the tested KV type)
            _kv_result = next((v for k, v in integ.items()
                               if k != "K=f16/V=f16"), {})
            r["integrity_kv"]   = _kv_result.get("status", "")
            r["integrity_sim"]  = _kv_result.get("similarity")
            # Get the KV type label from the key e.g. "K=q8_0/V=q8_0" → "q8_0"
            _kv_key = next((k for k in integ if k != "K=f16/V=f16"), "")
            r["integrity_kv_type"] = re.search(r"K=([^/]+)/", _kv_key).group(1) \
                                      if _kv_key and re.search(r"K=([^/]+)/", _kv_key) else ""
        else:
            r["integrity_kv"] = ""
            r["integrity_sim"] = None
            r["integrity_kv_type"] = ""

        # Reasoning score
        reas = _read_json(rdir / "reasoning.json")
        if reas:
            r["reasoning_score"]  = reas.get("score")
            r["reasoning_total"]  = reas.get("total")
        else:
            r["reasoning_score"] = None
            r["reasoning_total"] = None

        # Parse best_config string into structured fields for column rendering
        bc = r.get("best_config", "")
        bc_parts = dict(p.split("=", 1) for p in bc.split() if "=" in p)
        r["best_kv_type"]   = bc_parts.get("kv_cache_type", "")
        r["best_flash_attn"] = bc_parts.get("flash_attn", "")
        r["best_batch"]     = bc_parts.get("batch_size", "")
        r["best_ubatch"]    = bc_parts.get("ubatch_size", "")

    return results


# ── HTML rendering ─────────────────────────────────────────────────────────────

def _fmt(val, unit="", decimals=1, na="—") -> str:
    if val is None or val == "" or val == 0:
        return na
    try:
        return f"{float(val):.{decimals}f}{unit}"
    except Exception:
        return str(val)


def _ctx_k(val) -> str:
    if not val:
        return "—"
    return f"{int(val)//1024}k"


def _bench_cell(benchmarks: dict, key: str) -> str:
    v = benchmarks.get(key)
    if v is None:
        return "—"
    return f"{v:.1f}%"


# Benchmark column tooltip text — module-level so render_html f-string can access it
BENCH_TIPS = {
    "ARC":        "AI2 Reasoning Challenge (ARC-Challenge). 25-shot. Grade-school science questions requiring multi-step reasoning. Human baseline ~98%. Scores above 90% indicate strong scientific reasoning; below 60% is weak.",
    "HellaSwag":  "HellaSwag. 10-shot. Commonsense inference: pick the most plausible sentence continuation from 4 options. Human baseline ~95%. Modern top models are near-saturated (95%+). Below 80% indicates poor commonsense.",
    "MMLU":       "Massive Multitask Language Understanding. 5-shot. 57 academic subjects from high-school to professional level (law, medicine, math, history). Human expert ~89%. Above 85% = broad knowledge; below 65% is poor.",
    "TruthfulQA": "TruthfulQA (MC2). 0-shot. 817 adversarial questions designed to elicit common misconceptions. Human ~94%. Above 70% is good; below 50% means frequent hallucination. Key for real-world reliability.",
    "Winogrande": "WinoGrande. 5-shot. Pronoun disambiguation requiring commonsense reasoning. Human ~94%. Above 80% indicates solid contextual reasoning.",
    "GSM8K":      "Grade School Math 8K. 5-shot chain-of-thought. 8,500 grade-school word problems requiring multi-step arithmetic. Human ~99%. Above 80% is good; below 50% is poor. Strong predictor of practical math performance.",
    "HumanEval":  "HumanEval (pass@1). Python code generation from docstrings. 164 problems. Human ~99%. Above 80% indicates strong coding ability; below 40% is poor.",
    "MATH":       "MATH-500. Competition-level math across algebra, geometry, calculus, probability. Human expert ~55–90%. Above 70% is strong; below 30% is poor. Key differentiator for reasoning models.",
    "BBH":        "Big-Bench Hard. 23 tasks where prior models scored near chance: algorithmic reasoning, logical deduction, causal judgement. Above 70% indicates high-order reasoning; below 40% is poor.",
    "GPQA":       "Graduate-Level Google-Proof Q&A (Diamond set). PhD-level science questions that Google cannot answer. Human PhD experts ~65–74%. Above 70% is exceptional; below 40% is poor. Best discriminator for frontier reasoning.",
}


def render_html(results: list, report_meta: dict, output_path: Path):
    """Render a fully self-contained sortable HTML report."""

    all_benchmarks = sorted({
        k for r in results
        for k in r.get("hf", {}).get("benchmarks", {}).keys()
    })

    # Fallback to standard set if none found
    if not all_benchmarks:
        all_benchmarks = [b[0] for b in BENCHMARK_KEYS]

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build reference rows in the same schema as local results
    def _ref_to_row(m):
        return _make_row({
            "model": m["model"], "model_path": "", "results_dir": "",
            "status": "reference",
            "best_gen_tps": None, "baseline_gen_tps": None,
            "improvement_pct": None, "best_score": None,
            "current_quant": "API / cloud", "current_bpw": None,
            "estimated_fp16_gb": None, "topo_case": "—",
            "topo_winner": "—", "topo_score": None,
            "ctx_gpu": None, "ctx_ram": None, "recommended_ctx": None,
            "quant_recs": [], "is_reference": True,
            "provider": m.get("provider", ""), "family": m.get("family", ""),
            "tier": m.get("tier", ""), "open_weights": m.get("open_weights", False),
            "hf": {
                "repo_id": m["model"], "hf_url": m.get("hf_url", ""),
                "parameters_b": m.get("parameters_b"),
                "license": "open" if m.get("open_weights") else "proprietary",
                "benchmarks": {k: v for k, v in m.get("benchmarks", {}).items()
                               if v is not None},
                "lm_arena_elo": m.get("arena_elo"),
                "description": (m.get("description") or "")[:500],
                "author": m.get("provider", ""), "base_model": "",
                "tags": ([m["family"], "open-weights"] if m.get("open_weights")
                         else [m["family"], "proprietary"]),
                "downloads": None,
                "aa_intelligence": m.get("aa_intelligence"),
                "aa_output_tps":   m.get("aa_output_tps"),
                "aa_ttft_s":       m.get("aa_ttft_s"),
                "api_cost_per_mtok": m.get("api_cost_per_mtok"),
            },
            "gguf": {
                "arch": m.get("family", "").lower(), "n_layers": None,
                "n_attn_layers": None, "n_heads_kv": None, "head_dim": None,
                "n_expert": None,
                "context_length": (m["context_k"] * 1024 if m.get("context_k") else None),
                "is_moe": (m.get("parameters_b") or 0) > 100,
                "is_hybrid": False,
                "is_thinking": m.get("tier","") == "reasoning",
                "kv_mb_per_1k": None,
            },
            "ctx_detail": {
                "ctx_gpu_single": None, "ctx_gpu_combined": None,
                "ctx_ram_mixed": None, "ctx_ram_explicit": None,
                "recommended_ctx": (m["context_k"] * 1024 if m.get("context_k") else None),
                "trained_max": (m["context_k"] * 1024 if m.get("context_k") else None),
            },
            "topo_detail": {"case": "—", "winner": "—", "max_fit_ngl": None,
                            "model_size_mb": None, "scenarios": []},
            "baseline_detail": {}, "rank": None,
        }, all_benchmarks)

    # For open-weights reference models, try to fetch live benchmarks
    # from the Open LLM Leaderboard (same path as local models).
    # Closed-model benchmarks (GPT, Claude, Gemini) fall back to static values.
    _ref_session = requests.Session()
    _ref_hdrs = {"User-Agent": "llm-optimizer-report/1.0"}
    if HF_TOKEN:
        _ref_hdrs["Authorization"] = f"Bearer {HF_TOKEN}"
    _ref_session.headers.update(_ref_hdrs)

    # Fetch Arena ELO for reference models
    _ref_elo_map = _fetch_lmarena_elo(_ref_session)

    def _enrich_ref(m: dict) -> dict:
        """Try live leaderboard lookup for open-weights ref models."""
        row = _ref_to_row(m)
        hf_url = m.get("hf_url", "")
        # Derive org/model from hf_url if it points to a HF model page
        _repo = ""
        if "huggingface.co/" in hf_url:
            _parts = hf_url.rstrip("/").split("huggingface.co/")[-1].split("/")
            if len(_parts) >= 2:
                _repo = "/".join(_parts[:2])
        # Live leaderboard benchmarks for open-weights models
        if _repo and m.get("open_weights"):
            _lb1 = _fetch_open_llm_leaderboard_v1(_repo, _ref_session)
            _lb2 = _fetch_open_llm_leaderboard_v2(_repo, _ref_session)
            _live = {**_lb2, **_lb1}  # v1 wins (more complete for classic benchmarks)
            if _live:
                # Merge: static values fill gaps not covered by live data
                _static = row["hf"]["benchmarks"]
                row["hf"]["benchmarks"] = {**_static, **_live}
        # Live Arena ELO
        if _ref_elo_map and not row["hf"].get("lm_arena_elo"):
            _elo = _match_lmarena_elo(m["model"], _ref_elo_map)
            if _elo:
                row["hf"]["lm_arena_elo"] = _elo
        elif _ref_elo_map:
            # Refresh ELO even if static value exists
            _elo = _match_lmarena_elo(m["model"], _ref_elo_map)
            if _elo:
                row["hf"]["lm_arena_elo"] = _elo
        return row

    ref_rows_json = json.dumps([_enrich_ref(m) for m in REFERENCE_MODELS],
                               ensure_ascii=False)
    # Assign rank by best_gen_tps descending (only models with results)
    _ranked = sorted([r for r in results if r.get("best_gen_tps", 0) > 0],
                     key=lambda r: r.get("best_gen_tps", 0), reverse=True)
    for _ri, _r in enumerate(_ranked, 1):
        _r["rank"] = _ri
    rows_js       = json.dumps([_make_row(r, all_benchmarks) for r in results],
                               ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Optimizer Report — {ts}</title>
<style>
  :root {{
    --bg:       #0f1117;
    --bg2:      #1a1d27;
    --bg3:      #22263a;
    --border:   #2e3350;
    --accent:   #5b8ef0;
    --accent2:  #a78bfa;
    --green:    #34d399;
    --yellow:   #fbbf24;
    --red:      #f87171;
    --text:     #e2e8f0;
    --text2:    #94a3b8;
    --radius:   8px;
    --shadow:   0 4px 24px rgba(0,0,0,0.4);
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ height: 100%; overflow: hidden; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg);
          color: var(--text); font-size: 13px; line-height: 1.5;
          display: flex; flex-direction: column; }}
  header {{ background: var(--bg2); border-bottom: 1px solid var(--border);
            padding: 20px 28px; display: flex; align-items: center;
            gap: 16px; flex-wrap: wrap; flex-shrink: 0; }}
  header h1 {{ font-size: 20px; font-weight: 700; color: var(--text);
               letter-spacing: -0.3px; }}
  header .meta {{ color: var(--text2); font-size: 12px; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 20px;
            font-size: 11px; font-weight: 600; background: var(--bg3);
            border: 1px solid var(--border); color: var(--accent); }}
  .controls {{ padding: 14px 28px; background: var(--bg2);
               border-bottom: 1px solid var(--border);
               display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
               flex-shrink: 0; }}
  .controls input {{ background: var(--bg3); border: 1px solid var(--border);
                     color: var(--text); padding: 6px 12px; border-radius: var(--radius);
                     font-size: 12px; width: 260px; }}
  .controls input:focus {{ outline: none; border-color: var(--accent); }}
  .controls select {{ background: var(--bg3); border: 1px solid var(--border);
                      color: var(--text); padding: 6px 10px;
                      border-radius: var(--radius); font-size: 12px; }}
  .controls label {{ color: var(--text2); font-size: 12px; }}
  #count {{ color: var(--text2); font-size: 12px; margin-left: auto; }}
  .table-wrap {{ flex: 1; overflow-x: auto; overflow-y: auto; padding: 0 28px 28px; }}
  table {{ border-collapse: collapse; width: 100%; min-width: 1400px; margin-top: 16px; }}
  thead tr {{ background: var(--bg3); }}
  thead tr th {{ position: sticky; top: 0; z-index: 10; background: var(--bg3); }}
  th {{ padding: 9px 10px; text-align: left; font-weight: 600; font-size: 11px;
        color: var(--text2); text-transform: uppercase; letter-spacing: 0.5px;
        border-bottom: 2px solid var(--border); cursor: pointer;
        user-select: none; white-space: nowrap; position: relative; }}
  th .col-tip {{
        display: none; position: absolute; top: 100%; left: 0; z-index: 99;
        background: var(--bg2); border: 1px solid var(--border);
        border-radius: var(--radius); padding: 10px 14px; width: 280px;
        font-size: 11px; font-weight: 400; text-transform: none;
        letter-spacing: 0; color: var(--text2); line-height: 1.6;
        box-shadow: var(--shadow); pointer-events: none; white-space: normal; }}
  th:hover .col-tip {{ display: block; }}
  th:hover {{ color: var(--accent); }}
  th.sorted-asc::after  {{ content: " ▲"; color: var(--accent); }}
  th.sorted-desc::after {{ content: " ▼"; color: var(--accent); }}
  td {{ padding: 7px 10px; border-bottom: 1px solid var(--border);
        vertical-align: top; }}
  tr:hover td {{ background: rgba(91,142,240,0.06); }}
  tr.status-ok td {{ }}
  tr.status-no_results td {{ opacity: 0.55; }}
  tr.status-reference td {{ background: rgba(167,139,250,0.04);
                             border-left: 3px solid var(--accent2); }}
  tr.status-reference:hover td {{ background: rgba(167,139,250,0.10); }}
  .ref-badge {{ display:inline-block; padding:1px 6px; border-radius:8px;
                font-size:10px; background:rgba(167,139,250,0.15);
                border:1px solid var(--accent2); color:var(--accent2);
                margin-left:4px; vertical-align:middle; }}
  .lb-badge  {{ display:inline-block; padding:0px 5px; border-radius:6px;
                font-size:9px; font-weight:600; margin-left:4px;
                vertical-align:middle; letter-spacing:0.3px; }}
  .lb-v1     {{ background:rgba(91,142,240,0.15); border:1px solid var(--accent);
                color:var(--accent); }}
  .lb-v2     {{ background:rgba(52,211,153,0.15); border:1px solid var(--green);
                color:var(--green); }}
  .analogue-badge {{ display:inline-block; padding:0px 5px; border-radius:6px;
                     font-size:9px; font-weight:600; margin-left:5px;
                     vertical-align:middle; letter-spacing:0.3px;
                     background:rgba(251,191,36,0.12); border:1px dashed var(--yellow);
                     color:var(--yellow); cursor:help; }}
  .model-name {{ font-weight: 600; color: var(--text); min-width: 220px;
                 white-space: nowrap; }}
  .model-name a {{ color: var(--accent); text-decoration: none; }}
  .model-name a:hover {{ text-decoration: underline; }}
  .arch-tag {{ display: inline-block; padding: 1px 7px; border-radius: 10px;
               font-size: 10px; background: var(--bg3);
               border: 1px solid var(--border); color: var(--text2); }}
  .think-tag {{ background: rgba(251,191,36,0.15); border-color: #fbbf24; color: #fbbf24; }}
  .moe-tag  {{ background: rgba(167,139,250,0.15); border-color: var(--accent2);
               color: var(--accent2); }}
  .hyb-tag  {{ background: rgba(52,211,153,0.12); border-color: var(--green);
               color: var(--green); }}
  .case-A   {{ color: var(--green); }}
  .case-B   {{ color: var(--yellow); }}
  .case-C   {{ color: #fb923c; }}
  .case-D   {{ color: var(--red); }}
  .score    {{ font-weight: 700; color: var(--accent); }}
  .gain-pos {{ color: var(--green); font-weight: 600; }}
  .gain-neg {{ color: var(--red); }}
  .bench    {{ color: var(--text2); }}
  .desc     {{ color: var(--text2); font-size: 11px; max-width: 300px;
               overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .quant    {{ font-family: monospace; font-size: 11px; }}
  .quant-tip:hover {{ border-bottom-color: var(--green) !important; }}
  /* quant tooltip rendered via JS tippy/custom below */
  .num      {{ font-variant-numeric: tabular-nums; }}
  .tag      {{ display: inline-block; padding: 1px 6px; border-radius: 8px;
               font-size: 10px; background: rgba(91,142,240,0.15);
               color: var(--accent); margin: 1px; }}
  .na       {{ color: var(--border); }}
  .detail-btn {{ cursor: pointer; color: var(--accent2); font-size: 11px;
                 background: none; border: none; padding: 0; }}
  .detail-row {{ display: none; }}
  .detail-row.open {{ display: table-row; }}
  .detail-cell {{ background: var(--bg2) !important;
                  border-left: 3px solid var(--accent2); padding: 12px 16px; }}
  .detail-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
                  gap: 12px; }}
  .detail-section h4 {{ color: var(--accent2); font-size: 11px;
                        text-transform: uppercase; letter-spacing: 0.5px;
                        margin-bottom: 6px; }}
  .detail-section p {{ color: var(--text2); font-size: 11px; line-height: 1.6; }}
  .kv-row {{ display: flex; justify-content: space-between; gap: 8px;
             font-size: 11px; color: var(--text2); padding: 2px 0; }}
  .kv-row span:last-child {{ color: var(--text); font-weight: 500; }}
  .topo-scenarios {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }}
  .topo-chip {{ padding: 2px 8px; border-radius: 8px; font-size: 10px;
                background: var(--bg3); border: 1px solid var(--border);
                color: var(--text2); }}
  .topo-chip.winner {{ background: rgba(52,211,153,0.15);
                       border-color: var(--green); color: var(--green); }}
  .bench-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
                 gap: 4px; margin-top: 4px; }}
  .bench-item {{ background: var(--bg3); border: 1px solid var(--border);
                 border-radius: 6px; padding: 4px 8px; font-size: 11px; }}
  .bench-item .bname {{ color: var(--text2); font-size: 10px; }}
  .bench-item .bval  {{ color: var(--text); font-weight: 600; }}
  .quant-recs {{ margin-top: 4px; }}
  .quant-rec-row {{ font-size: 11px; color: var(--text2); padding: 2px 0;
                    display: flex; gap: 8px; }}
  .quant-rec-row .qname {{ font-family: monospace; color: var(--text); width: 80px; }}
  .qup   {{ color: var(--accent2); }}
  .qdown {{ color: var(--green); }}
  .qside {{ color: var(--text2); }}
  @media (max-width: 800px) {{
    .controls input {{ width: 180px; }}
    .table-wrap {{ padding: 0 12px 12px; }}
  }}
  /* ref-only columns: hidden by default, shown when reference models are added */
  .ref-col {{ display: none; }}
  .show-ref-cols .ref-col {{ display: table-cell; }}
</style>
</head>
<body>

<header>
  <div>
    <h1>LLM Optimizer Report</h1>
    <div class="meta">Generated {ts} &nbsp;·&nbsp;
      Models dir: {report_meta.get('models_dir','?')} &nbsp;·&nbsp;
      {report_meta.get('gpu0_name','GPU0')}: {round(report_meta.get('gpu0_vram_gb',0),2):.2f} GB &nbsp;|&nbsp;
      {report_meta.get('gpu1_name','GPU1')}: {round(report_meta.get('gpu1_vram_gb',0),2):.2f} GB
    </div>
  </div>
  <div style="margin-left:auto;display:flex;gap:8px;flex-wrap:wrap;">
    <span class="badge">{report_meta.get('successful',0)} models optimized</span>
    <span class="badge">{report_meta.get('total',0)} total</span>
  </div>
</header>

<div class="controls">
  <input type="text" id="search" placeholder="Filter by model name, arch, quant…"
         oninput="filterRows()">
  <label>Case:
    <select id="filterCase" onchange="filterRows()">
      <option value="">All</option>
      <option value="A">A — Both GPUs</option>
      <option value="B">B — GPU0 only</option>
      <option value="C">C — Split GPUs</option>
      <option value="D">D — CPU offload</option>
    </select>
  </label>
  <label>Status:
    <select id="filterStatus" onchange="filterRows()">
      <option value="">All</option>
      <option value="ok">Optimized</option>
      <option value="no_results">No results</option>
    </select>
  </label>
  <span id="count"></span>
  <label style="margin-left:auto;display:flex;align-items:center;gap:6px;
                cursor:pointer;padding:5px 12px;border-radius:var(--radius);
                border:1px solid var(--border);background:var(--bg3);">
    <input type="checkbox" id="toggleRef" onchange="toggleReference()"
           style="accent-color:var(--accent2);width:14px;height:14px;">
    <span style="color:var(--text2);font-size:12px;">Compare reference models
      <span style="color:var(--accent2);font-size:10px;display:block;margin-top:1px;">
        GPT · Claude · Gemini · Grok · DeepSeek · Qwen · GLM · Kimi · MiniMax · Nemotron
      </span>
    </span>
  </label>
</div>

<div class="table-wrap">
<table id="mainTable">
<thead>
<tr>
  <th data-col="rank">Rank<span class="col-tip">Optimizer rank by best tokens/second achieved. Local GGUF models only — reference API models show no rank.</span></th>
  <th data-col="model">Model<span class="col-tip">GGUF filename (local models) or model name (reference API models). Click the link to open the HuggingFace model card or provider page.</span></th>
  <th data-col="quant">Quant<span class="col-tip">Quantization level of the GGUF file — lower bits = smaller file and faster inference at the cost of quality. Q4_K_M is the common sweet spot. IQ variants use imatrix for better quality at the same size.</span></th>
  <th data-col="case">Case<span class="col-tip">GPU topology case — A: fits in both GPUs independently · B: fits only in {report_meta.get("gpu0_name","GPU0")} ({round(report_meta.get("gpu0_vram_gb",24),2):.2f} GB) · C: requires both GPUs combined · D: needs CPU/RAM offload. Inferred from model size when topo sweep not run.</span></th>
  <th data-col="params">Params<span class="col-tip">Total parameter count in billions, from HuggingFace model card or safetensors metadata. For MoE models this is the total count — active params per token are shown in the detail row.</span></th>
  <th data-col="file_gb">Size<span class="col-tip">GGUF file size on disk in gigabytes. Estimated from current quantization bits-per-weight. Actual VRAM required is higher (model weights + KV cache + compute buffers).</span></th>

  <th data-col="arch">Arch<span class="col-tip">Model architecture read from GGUF metadata (e.g. llama, mistral, qwen2, phi3, gemma). Determines which llama-server optimizations apply and which phases the optimizer skips.</span></th>
  <th data-col="n_layers">Layers<span class="col-tip">Total transformer block count from GGUF metadata. Used to determine n_gpu_layers. Hybrid models (Mamba/SSM) have fewer full-attention layers than total blocks.</span></th>
  <th data-col="trained_ctx">Train ctx<span class="col-tip">Maximum context length the model was trained on, from GGUF metadata. Running beyond this length produces unreliable output. The context sweep finds the practical VRAM-limited ceiling below this value.</span></th>

  <th data-col="ctx_gpu">ctx GPU<span class="col-tip">Maximum stable context length with the model fully in VRAM (GPU-only, no RAM offload), found by binary search. Larger context = better long-document handling but higher VRAM use.</span></th>
  <th data-col="ctx_ram">ctx RAM<span class="col-tip">Maximum stable context length with KV cache spilling into system RAM (mixed mode). Slower than GPU-only but allows much larger contexts on models that partially fit in VRAM.</span></th>
  <th data-col="rec_ctx">Rec ctx<span class="col-tip">Recommended context window — the largest stable value found by the context sweep for this model on this hardware. Use this as your -c argument for day-to-day inference.</span></th>

  <th data-col="score">Score<span class="col-tip">Optimizer composite score (higher = better). Weighted formula: gen_tps×0.35 + large_prompt_tps×0.25 + prompt_tps_factor×0.15 + ttft_factor×0.15 + vram_efficiency×0.10. Not comparable between different models — use Best t/s for cross-model comparison.</span></th>
  <th data-col="best_tps">Best t/s<span class="col-tip">Best generation speed in tokens per second found by the optimizer. Measured on a short prompt (50 tokens output) at temp=0. Higher is always better for interactive use.</span></th>
  <th data-col="baseline_tps">Stock t/s<span class="col-tip">Generation speed with default llama-server settings (no optimization). Compare against Best t/s to see how much the optimizer improved things.</span></th>
  <th data-col="gain">Gain %<span class="col-tip">Percentage improvement from the optimizer: (Best − Stock) / Stock × 100. Gains of 10–40% are typical. Very high gains (&gt;100%) usually mean n-gram speculation fired, which helps repetitive-output models greatly.</span></th>
  <th data-col="topo">Topo winner<span class="col-tip">The GPU topology scenario that won the topo sweep, or the best GPU layer count found during GPU optimisation when topo sweep was not run.</span></th>

  {"".join(f'<th data-col="bench_{b}">{b}<span class=\"col-tip\">{BENCH_TIPS.get(b, b + " benchmark score (%).")}</span></th>' for b in all_benchmarks)}

  <th data-col="openllm">Local Score<span class="col-tip">Weighted benchmark average normalised to a common v1-equivalent scale, so models evaluated on the harder v2 suite can be ranked alongside v1 models. v1 weights: ARC×1.0, HellaSwag×0.7, MMLU×1.5, TruthfulQA×1.0, Winogrande×0.7, GSM8K×1.5. v2 weights: ARC×0.8, BBH×1.5, MATH×1.5, GPQA×1.5, MMLU-Pro×1.2. v2 scores are mapped to v1-equivalent space via affine calibration (×0.958 + 35.6) derived from models that appear on both leaderboards. Badge shows which leaderboard version the benchmarks came from. Higher = better. ~50 = average 7B model, ~75 = strong 70B model.</span></th>
  <th data-col="lb_avg">HF Score<span class="col-tip">The raw average score published by the HF Open LLM Leaderboard for this model — a simple unweighted mean across all evaluated benchmarks. v1 (closed Nov 2024, ~5 500 models) scored on ARC / HellaSwag / MMLU / TruthfulQA / Winogrande / GSM8K; typical range 25–89. v2 (ongoing, ~3 000 models) scored on IFEval / BBH / MATH-Hard / GPQA / MUSR / MMLU-Pro; typical range 5–57. The two scales are NOT directly comparable — use My Score or LB Rank for cross-version sorting. Badge shows v1 or v2. Source: <a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard" target="_blank">Open LLM Leaderboard</a>.</span></th>
  <th data-col="lb_rank">HF Rank<span class="col-tip">This model's position on the HF Open LLM Leaderboard ranked table (lower number = better). Sort uses a unified percentile key so v1 and v2 ranks are comparable: V1 percentile = (1 − (rank−1) / 5 500) × 100; V2 percentile = min(99.9, same formula with 3 000 total + 5 pt difficulty bonus). The +5 bonus reflects that v2 tasks (GPQA Diamond, MATH-Hard) are harder than v1 tasks (ARC, HellaSwag). Badge shows v1 or v2. Source: <a href="https://huggingface.co/datasets/open-llm-leaderboard/contents" target="_blank">v2 dataset</a> · <a href="https://huggingface.co/datasets/open-llm-leaderboard-old/results" target="_blank">v1 dataset</a>.</span></th>
  <th data-col="elo">Arena ELO<span class="col-tip">Chatbot Arena ELO score from lmarena.ai — a community blind A/B preference ranking. Only available for widely-deployed reference API models (GPT, Claude, Gemini, etc.) — local GGUF models do not appear in this ranking.</span></th>
  <th data-col="license">License<span class="col-tip">Model license from HuggingFace model card. Relevant for commercial use: Apache 2.0 and MIT are fully open; llama/gemma community licenses allow most commercial use with restrictions; proprietary = API-only.</span></th>
  <th data-col="downloads">Downloads<span class="col-tip">Total HuggingFace downloads (all-time). A rough proxy for community adoption and how battle-tested the model is in practice.</span></th>
  <th data-col="aa_intelligence" class="ref-col">AA Intel<span class="col-tip">Artificial Analysis Intelligence Index — a composite score across 10 real-world evaluations. Only available for reference API models.</span></th>
  <th data-col="aa_output_tps"   class="ref-col">AA t/s<span class="col-tip">Output tokens per second measured by Artificial Analysis on cloud API. Only available for reference API models.</span></th>
  <th data-col="aa_ttft_s"       class="ref-col">AA TTFT<span class="col-tip">Time to first token in seconds (Artificial Analysis). Only available for reference API models.</span></th>
  <th data-col="api_cost"        class="ref-col">$/1M tok<span class="col-tip">Blended API cost in USD per 1 million tokens (3:1 input:output). Only available for reference API models.</span></th>
  <th data-col="detail" class="ref-col"></th>
  <th data-col="ik_best_tps">IK t/s<span class="col-tip">Best tokens/second achieved by ik_llama.cpp on this model, using the optimal combination of MLA attention (-mla 2), fused MoE (-fmoe), run-time repack (-rtr), and attn-max-batch (-amb) flags. Measured after the llama.cpp optimization pipeline completes. Only populated when IK_LLAMA_SERVER is configured.</span></th>
  <th data-col="mtp_best_tps">MTP t/s<span class="col-tip">Best tokens/second achieved with Multi-Token Prediction (MTP) enabled. MTP uses auxiliary prediction heads baked into the GGUF itself — no separate draft model needed. Each forward pass predicts the main token plus N draft tokens verified in parallel. Only available for models with nextn_predict_layers in their GGUF metadata (Qwen3.5/3.6, DeepSeek V3/R1, Gemma 4). Use --force-mtp to test any model.</span></th>
  <th data-col="mtp_gain">MTP gain<span class="col-tip">Speed gain of the best MTP configuration vs the same model without MTP, as a percentage. Dense models typically see 20–70% gains; MoE models see smaller gains (5–25%) because expert routing already reduces per-step compute. Negative values mean MTP was counterproductive on this hardware (usually indicates near-full VRAM — MTP heads require ~2–5% extra).</span></th>
  <th data-col="ik_gain">IK gain<span class="col-tip">Speed gain of the best ik_llama.cpp config vs the best vanilla llama.cpp config, as a percentage. Positive = IK is faster. Key IK advantages: fused MoE routing (+20–80% on MoE models), run-time quant repacking for CPU-offloaded experts (+50–80% on hybrid GPU/CPU configs), and MLA attention for DeepSeek-architecture models (-50% KV cache, meaningful throughput boost). Zero or blank = IK contrast not yet run.</span></th>
  <th data-col="detail" class="ref-col"></th>
  <th data-col="detail"></th>
</tr>
</thead>
<tbody id="tbody">
</tbody>
</table>
</div>

<script>
const RAW = {rows_js};
const REF_ROWS = {ref_rows_json};
const BENCHMARKS = {json.dumps(all_benchmarks)};
const CASE_DESC = {json.dumps(CASE_DESCRIPTIONS)};


// ── state ──────────────────────────────────────────────────────────────────────
let showReference = false;
let sortCol  = "rank";   // "" = unsorted
let sortDir  = 1;        // 1 asc, -1 desc

// ── helpers ────────────────────────────────────────────────────────────────────
function fmt(v, unit, dec) {{
  unit = unit || ""; dec = (dec === undefined) ? 1 : dec;
  if (v === null || v === undefined || v === "") return '<span class="na">—</span>';
  const n = parseFloat(v);
  if (isNaN(n)) return String(v);
  return n.toFixed(dec) + unit;
}}
function fmtCtx(v) {{
  if (!v) return '<span class="na">—</span>';
  return Math.round(v / 1024) + "k";
}}
function fmtGain(v) {{
  if (v === null || v === undefined) return '<span class="na">—</span>';
  const n = parseFloat(v);
  if (isNaN(n)) return '<span class="na">—</span>';
  const cls = n > 0 ? "gain-pos" : (n < -1 ? "gain-neg" : "");
  return '<span class="' + cls + '">' + (n > 0 ? "+" : "") + n.toFixed(0) + "%</span>";
}}
function fmtCase(c) {{
  if (!c) return '<span class="na">—</span>';
  const desc = CASE_DESC[c] || c;
  return '<span class="case-' + c + '" title="' + desc + '">' + c + '</span>';
}}
function esc(s) {{
  // Escape string for use inside an HTML attribute value (double-quoted)
  return String(s || "").replace(/&/g,"&amp;").replace(/"/g,"&quot;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}}
function escHtml(s) {{
  return String(s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}}

// ── sort value extractor ───────────────────────────────────────────────────────
function sv(row, col) {{
  if (col.startsWith("bench_")) {{
    const v = (row.hf && row.hf.benchmarks || {{}})[col.slice(6)];
    return (v === "MAX") ? 101 : (v != null ? v : null);
  }}
  switch (col) {{
    case "score":        return row.best_score        != null ? row.best_score        : null;
    case "best_tps":     return row.best_gen_tps      != null ? row.best_gen_tps      : null;
    case "baseline_tps": return row.baseline_gen_tps  != null ? row.baseline_gen_tps  : null;
    case "gain":         return row.improvement_pct   != null ? row.improvement_pct   : null;
    case "quant":        return row.current_quant     || null;
    case "topo":         return (row.topo_detail && row.topo_detail.winner) || row.topo_winner || null;
    case "file_gb":      return row.file_size_gb || (row.estimated_fp16_gb && row.current_bpw ? row.estimated_fp16_gb * row.current_bpw / 16 : null);
    case "case":         return (row.topo_detail && row.topo_detail.case) || row.topo_case || null;
    case "openllm":      return row.hf && row.hf.openllm_score  != null ? row.hf.openllm_score  : null;
    case "lb_avg":       return row.hf && row.hf.lb_avg         != null ? row.hf.lb_avg         : null;
    case "lb_rank":      return row.hf && row.hf.lb_sort_key    != null ? row.hf.lb_sort_key    : null;
    case "elo":          return row.hf && row.hf.lm_arena_elo != null ? row.hf.lm_arena_elo : null;
    case "arch":         return row.gguf && row.gguf.arch || null;
    case "n_layers":     return row.gguf && row.gguf.n_layers != null ? row.gguf.n_layers : null;
    case "trained_ctx":  return row.gguf && row.gguf.context_length != null ? row.gguf.context_length : null;
    case "ctx_gpu":      return row.ctx_detail && (row.ctx_detail.ctx_gpu_single || row.ctx_detail.ctx_gpu_combined) || null;
    case "ctx_ram":      return row.ctx_detail && (row.ctx_detail.ctx_ram_mixed  || row.ctx_detail.ctx_ram_explicit)  || null;
    case "rec_ctx":      return row.ctx_detail && row.ctx_detail.recommended_ctx != null ? row.ctx_detail.recommended_ctx : null;
    case "params":       return row.hf && row.hf.parameters_b != null ? row.hf.parameters_b : null;
    case "downloads":    return row.hf && row.hf.downloads != null ? row.hf.downloads : null;
    case "aa_intelligence": return row.hf && row.hf.aa_intelligence != null ? row.hf.aa_intelligence : null;
    case "aa_output_tps":   return row.hf && row.hf.aa_output_tps  != null ? row.hf.aa_output_tps  : null;
    case "aa_ttft_s":       return row.hf && row.hf.aa_ttft_s      != null ? row.hf.aa_ttft_s      : null;
    case "api_cost":         return row.hf && row.hf.api_cost_per_mtok != null ? row.hf.api_cost_per_mtok : null;
    case "license":          return row.hf && row.hf.license || null;
    case "integrity":        return row.integrity_sim != null ? row.integrity_sim : null;
    case "reasoning":        return row.reasoning_score != null ? row.reasoning_score / (row.reasoning_total || 8) : null;
    case "best_kv":          return row.best_kv_type   || null;
    case "best_flash":       return row.best_flash_attn || null;
    case "ik_best_tps":      return row.ik_best_tps  > 0 ? row.ik_best_tps  : null;
    case "ik_gain":          return row.ik_available  ? row.ik_gain_vs_llama_pct : null;
    case "mtp_best_tps":     return row.mtp_best_tps > 0 ? row.mtp_best_tps : null;
    case "mtp_gain":         return row.mtp_available ? row.mtp_gain_pct : null;
    default:                 return row[col] != null ? row[col] : null;
  }}
}}

// ── main render function ───────────────────────────────────────────────────────
// Reads current UI state, builds the visible row set, sorts, and renders.
// Called by every user interaction.
function render() {{
  const q     = document.getElementById("search").value.toLowerCase();
  const fCase = document.getElementById("filterCase").value;
  const fStat = document.getElementById("filterStatus").value;

  // 1. Build the working set from scratch
  const pool = showReference ? RAW.concat(REF_ROWS) : RAW.slice();

  // 2. Filter
  const visible = pool.filter(function(r) {{
    if (r.is_reference) return showReference;  // refs: only checkbox controls visibility
    const modelCase = (r.topo_detail && r.topo_detail.case) || r.topo_case || "";
    if (fCase && modelCase !== fCase) return false;
    if (fStat && (r.status || "") !== fStat) return false;
    if (q) {{
      const parts = [r.model, r.current_quant,
        r.gguf && r.gguf.arch,
        r.hf && r.hf.description,
        (r.topo_detail && r.topo_detail.winner) || r.topo_winner];
      const searchable = parts.filter(Boolean).join(" ").toLowerCase();
      if (searchable.indexOf(q) === -1) return false;
    }}
    return true;
  }});

  // 3. Sort
  if (sortCol) {{
    const zeroNull = ["score","best_tps","baseline_tps","gain"].indexOf(sortCol) !== -1;
    visible.sort(function(a, b) {{
      let va = sv(a, sortCol);
      let vb = sv(b, sortCol);
      if (zeroNull) {{ if (va === 0) va = null; if (vb === 0) vb = null; }}
      // no_results rows (no rank/score): always after ranked rows regardless of direction
      if (va === null && vb === null) return 0;
      if (va === null) return 1;
      if (vb === null) return -1;
      if (typeof va === "string") return va.localeCompare(vb) * sortDir;
      return (va - vb) * sortDir;
    }});
  }}
  // unsorted: use pool order (original RAW insertion order)

  // 4. Render
  const tbody = document.getElementById("tbody");
  const html = [];
  for (let i = 0; i < visible.length; i++) {{
    try {{ html.push(buildRow(visible[i], i + 1)); }}
    catch(e) {{ html.push('<tr><td colspan="999" style="color:var(--red);font-size:11px">Row error: ' + escHtml(String(e)) + '</td></tr><tr></tr>'); }}
  }}
  tbody.innerHTML = html.join("");

  // 5. Update count
  const localCount = visible.filter(function(r){{ return !r.is_reference; }}).length;
  const refCount   = visible.filter(function(r){{ return  r.is_reference; }}).length;
  const refStr     = refCount > 0 ? " + " + refCount + " reference" : "";
  document.getElementById("count").textContent =
    "Showing " + localCount + " local" + refStr + " of " + RAW.length + " total";

  // 6. Sort indicators
  document.querySelectorAll("th[data-col]").forEach(function(th) {{
    th.classList.remove("sorted-asc", "sorted-desc");
    if (th.dataset.col === sortCol) {{
      th.classList.add(sortDir === 1 ? "sorted-asc" : "sorted-desc");
    }}
  }});
}}

// ── event handlers ─────────────────────────────────────────────────────────────
function sortData(col) {{
  if (sortCol === col) {{
    if (sortDir === 1)  {{ sortDir = -1; }}        // asc → desc
    else                {{ sortCol = ""; sortDir = 1; }} // desc → unsorted
  }} else {{
    sortCol = col; sortDir = 1;
  }}
  render();
}}
function filterRows()       {{ render(); }}
function toggleReference()  {{
  showReference = document.getElementById("toggleRef").checked;
  // Show/hide the ref-only columns (Arena ELO, AA Intel, AA t/s, AA TTFT, $/1M tok)
  document.getElementById("mainTable").classList.toggle("show-ref-cols", showReference);
  render();
}}
function toggleDetail(id) {{
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.toggle("open");
  const btn = el.previousElementSibling && el.previousElementSibling.querySelector(".detail-btn");
  if (btn) btn.textContent = el.classList.contains("open") ? "▼" : "▶";
}}

// ── row builder ────────────────────────────────────────────────────────────────
function buildRow(row, idx) {{
  const hf   = row.hf   || {{}};
  const gguf = row.gguf || {{}};
  const ctx  = row.ctx_detail  || {{}};
  const topo = row.topo_detail || {{}};
  const recs = row.quant_recs  || [];
  const isRef = !!row.is_reference;

  const hfUrl    = hf.hf_url || "";
  const nameHtml = hfUrl
    ? '<a href="' + esc(hfUrl) + '" target="_blank">' + escHtml(row.model) + '</a>'
    : escHtml(row.model);

  // Tags
  const tagParts = [];
  if (isRef) {{
    if (row.provider) tagParts.push('<span class="arch-tag" style="color:var(--accent2)">' + escHtml(row.provider) + '</span>');
    if (row.tier)     tagParts.push('<span class="ref-badge">' + escHtml(row.tier) + '</span>');
    const oc = row.open_weights ? "var(--green)" : "var(--text2)";
    tagParts.push('<span class="arch-tag" style="color:' + oc + '">' + (row.open_weights ? "open" : "closed") + '</span>');
  }} else {{
    if (gguf.is_thinking) tagParts.push('<span class="arch-tag think-tag">Thinking</span>');
    if (gguf.is_moe)      tagParts.push('<span class="arch-tag moe-tag">MoE</span>');
    if (gguf.is_hybrid)   tagParts.push('<span class="arch-tag hyb-tag">Hybrid</span>');
  }}

  // Analogue flag — benchmarks sourced from a base model ancestor, not this model directly
  const isAnalogue    = !!(hf.is_analogue);
  const analogueModel = hf.analogue_model || "";

  // Bench cells — dashed underline when benchmarks come from a base model
  const benchParts = [];
  for (let i = 0; i < BENCHMARKS.length; i++) {{
    const b = BENCHMARKS[i];
    const v = hf.benchmarks && hf.benchmarks[b];
    const aStyle = isAnalogue ? ' style="border-bottom:2px dashed rgba(148,163,184,0.5)"' : "";
    if (v === undefined || v === null) {{
      benchParts.push('<td class="na bench"' + aStyle + '>—</td>');
    }} else if (v === "MAX") {{
      benchParts.push('<td class="bench num"' + aStyle + ' style="color:var(--green);font-weight:600">MAX</td>');
    }} else {{
      const n = parseFloat(v);
      benchParts.push('<td class="bench num"' + aStyle + '>' + (isNaN(n) ? escHtml(String(v)) : n.toFixed(1) + "%") + '</td>');
    }}
  }}

  // Quant tooltip
  let quantCell;
  const q = row.current_quant;
  if (!q) {{
    quantCell = '<span class="na">—</span>';
  }} else if (!recs.length) {{
    quantCell = escHtml(q);
  }} else {{
    const tipLines = [];
    for (let i = 0; i < Math.min(recs.length, 8); i++) {{
      const r = recs[i];
      const arrow = r.direction === "upgrade" ? "↑" : r.direction === "downgrade" ? "↓" : "↔";
      const col   = r.direction === "upgrade" ? "#4ade80" : r.direction === "downgrade" ? "#fb923c" : "#94a3b8";
      tipLines.push('<span style="color:' + col + '">' + arrow + '</span> <b>' + escHtml(r.quant) + '</b> &nbsp;' + r.estimated_gb + 'GB&nbsp; Case ' + escHtml(r.case));
    }}
    quantCell = '<span class="quant-tip" title="" data-tip="' + encodeURIComponent(tipLines.join("<br>")) + '" style="border-bottom:2px dashed var(--accent);cursor:help">' + escHtml(q) + '</span>';
  }}

  // Topo chips
  const chipParts = [];
  const scens = topo.scenarios || [];
  for (let i = 0; i < Math.min(scens.length, 8); i++) {{
    const s = scens[i];
    const win = s.label === topo.winner ? " winner" : "";
    chipParts.push('<span class="topo-chip' + win + '" title="' + esc(s.label) + '">' + escHtml(s.scenario) + (s.gen_tps ? " " + s.gen_tps.toFixed(0) + "t/s" : "") + '</span>');
  }}

  // Bench detail (expanded row)
  const benchDetailParts = [];
  const benches = hf.benchmarks || {{}};
  for (const k in benches) {{
    const v = benches[k];
    const display = (v === "MAX") ? "MAX"
                  : (v === null || v === undefined) ? "—"
                  : (typeof v === "number") ? v.toFixed(1) + "%" : escHtml(String(v));
    benchDetailParts.push('<div class="bench-item"><div class="bname">' + escHtml(k) + '</div><div class="bval">' + display + '</div></div>');
  }}

  // Rec rows (expanded)
  const recParts = [];
  for (let i = 0; i < Math.min(recs.length, 8); i++) {{
    const r = recs[i];
    const cls = r.direction === "upgrade" ? "qup" : r.direction === "downgrade" ? "qdown" : "qside";
    const arr = r.direction === "upgrade" ? "↑" : r.direction === "downgrade" ? "↓" : "↔";
    recParts.push('<div class="quant-rec-row"><span class="qname">' + escHtml(r.quant) + '</span><span class="' + cls + '">' + arr + ' ' + escHtml(r.direction) + '</span><span>~' + r.estimated_gb + 'GB</span><span>Case ' + escHtml(r.case) + '</span><span>spd=' + r.speed_rank + '</span><span>qual=' + r.quality_rank + '</span></div>');
  }}

  // Tag chips (expanded)
  const tagChipParts = [];
  const htags = hf.tags || [];
  for (let i = 0; i < Math.min(htags.length, 20); i++) {{
    tagChipParts.push('<span class="tag">' + escHtml(htags[i]) + '</span>');
  }}

  const detailId   = "detail-" + idx;
  const sizeGb     = row.file_size_gb || (row.estimated_fp16_gb && row.current_bpw ? row.estimated_fp16_gb * row.current_bpw / 16 : null);
  const topoCase   = (topo.case   && topo.case   !== "—") ? topo.case   : (row.topo_case   || "");
  const topoWinner = (topo.winner && topo.winner !== "—") ? topo.winner : (row.topo_winner || "");
  const rankDisp   = row.rank != null ? String(row.rank) : '<span class="na">—</span>';

  let html = '<tr class="data-row ' + (isRef ? "status-reference" : "status-" + (row.status || "")) + '"'
    + ' data-model="' + esc((row.model || "").toLowerCase()) + '"'
    + ' data-arch="'  + esc((gguf.arch || "").toLowerCase()) + '"'
    + ' data-quant="' + esc((row.current_quant || "").toLowerCase()) + '"'
    + ' data-case="'  + esc(topoCase) + '"'
    + ' data-status="' + esc(row.status || "") + '">'
    + '<td class="num">'  + rankDisp + '</td>'
    + '<td class="model-name">' + nameHtml + ' ' + tagParts.join("") + (isAnalogue && analogueModel ? ' <span class="analogue-badge" title="Benchmarks sourced from base model: ' + esc(analogueModel) + '">~' + esc(analogueModel.split('/').pop()) + '</span>' : '') + '</td>'
    + '<td class="quant">' + quantCell + '</td>'
    + '<td>' + fmtCase(topoCase) + '</td>'
    + '<td class="num">' + (hf.parameters_b != null ? hf.parameters_b + "B" : '<span class="na">—</span>') + '</td>'
    + '<td class="num">' + (sizeGb ? sizeGb.toFixed(1) + "GB" : '<span class="na">—</span>') + '</td>'
    + '<td>' + (gguf.arch ? '<span class="arch-tag">' + escHtml(gguf.arch) + '</span>' : '<span class="na">—</span>') + '</td>'
    + '<td class="num">' + (gguf.n_layers || '<span class="na">—</span>') + '</td>'
    + '<td class="num">' + fmtCtx(gguf.context_length || ctx.trained_max) + '</td>'
    + '<td class="num">' + fmtCtx(ctx.ctx_gpu_single || ctx.ctx_gpu_combined) + '</td>'
    + '<td class="num">' + fmtCtx(ctx.ctx_ram_mixed  || ctx.ctx_ram_explicit) + '</td>'
    + '<td class="num">' + fmtCtx(ctx.recommended_ctx) + '</td>'
    + '<td class="score">' + (row.best_score ? row.best_score.toFixed(1) : '<span class="na">—</span>') + '</td>'
    + '<td class="num">'  + fmt(row.best_gen_tps, " t/s") + '</td>'
    + '<td class="num">'  + fmt(row.baseline_gen_tps, " t/s") + '</td>'
    + '<td>' + fmtGain(row.improvement_pct) + '</td>'
    + '<td class="desc" title="' + esc(topoWinner) + '">' + (topoWinner ? escHtml(topoWinner) : '<span class="na">—</span>') + '</td>'
    + benchParts.join("")
    + (function() {{
        // ── My Score (weighted, v1-normalised) ────────────────────────────
        var ols = hf.openllm_score; var olv = hf.openllm_version || "";
        var olBadge = olv ? '<span class="lb-badge lb-' + olv + '">' + olv + '</span>' : "";
        var olCell = ols != null
            ? '<span class="score">' + ols.toFixed(1) + '</span>'
            : '<span class="na">—</span>';
        // ── LB Avg (leaderboard's own published average) ──────────────────
        var lba = hf.lb_avg; var lbv = hf.lb_version || "";
        var lbBadge = lbv ? '<span class="lb-badge lb-' + lbv + '">' + lbv + '</span>' : "";
        var lbaCell = lba != null
            ? lba.toFixed(1) + lbBadge
            : '<span class="na">—</span>';
        // ── LB Rank (leaderboard position #N) ─────────────────────────────
        var lbr = hf.lb_rank;
        var lbrCell = lbr != null
            ? '#' + lbr.toLocaleString() + lbBadge
            : '<span class="na">—</span>';
        return '<td class="num">' + olCell  + '</td>'
             + '<td class="num">' + lbaCell + '</td>'
             + '<td class="num">' + lbrCell + '</td>';
      }})()
    + '<td class="num">' + ((hf.lm_arena_elo || hf.arena_elo) ? (hf.lm_arena_elo || hf.arena_elo).toLocaleString() : '<span class="na">—</span>') + '</td>'
    + '<td class="desc">' + (hf.license ? escHtml(hf.license) : '<span class="na">—</span>') + '</td>'
    + '<td class="num">' + (hf.downloads != null ? hf.downloads.toLocaleString() : '<span class="na">—</span>') + '</td>'
    // ref-only columns — hidden when reference models not shown
    + '<td class="num ref-col">' + (hf.aa_intelligence != null ? '<span' + (isRef ? ' class="score"' : '') + '>' + hf.aa_intelligence + '</span>' : '<span class="na">—</span>') + '</td>'
    + '<td class="num ref-col">' + (hf.aa_output_tps   != null ? hf.aa_output_tps.toFixed(0)  + " t/s" : '<span class="na">—</span>') + '</td>'
    + '<td class="num ref-col">' + (hf.aa_ttft_s       != null ? hf.aa_ttft_s.toFixed(2)      + "s"    : '<span class="na">—</span>') + '</td>'
    + '<td class="num ref-col">' + (hf.api_cost_per_mtok != null ? "$" + hf.api_cost_per_mtok.toFixed(2) : '<span class="na">—</span>') + '</td>'
    + '<td class="ref-col"></td>'
    + (function() {{
        // ── MTP columns ────────────────────────────────────────────────────
        (function() {{
          var mtpTps   = row.mtp_best_tps;
          var mtpGain  = row.mtp_gain_pct;
          var mtpAvail = row.mtp_available;
          var mtpLabel = row.mtp_best_label || "";
          var mtpTip   = mtpAvail && mtpLabel
              ? ' title="' + esc("MTP best: " + mtpLabel) + '"'
              : (mtpAvail ? "" : ' title="MTP not run — use --preset mtp or --force-mtp"');
          var mtpColor = !mtpAvail ? "var(--text2)"
              : mtpGain > 5  ? "var(--green)"
              : mtpGain < 0  ? "var(--red)"
              : "var(--yellow)";
          var mtpTpsCell  = mtpAvail && mtpTps > 0
              ? '<span' + mtpTip + ' style="font-weight:600">' + mtpTps.toFixed(1) + " t/s</span>"
              : '<span class="na"' + mtpTip + '>—</span>';
          var mtpGainCell = mtpAvail
              ? '<span style="font-weight:600;color:' + mtpColor + '">' + (mtpGain >= 0 ? "+" : "") + mtpGain.toFixed(1) + "%</span>"
              : '<span class="na"' + mtpTip + '>—</span>';
          html += '<td class="num">' + mtpTpsCell  + '</td>'
               +  '<td class="num">' + mtpGainCell + '</td>';
        }})();

        // ── IK_llama.cpp contrast columns ─────────────────────────────────
        var ikTps    = row.ik_best_tps;
        var ikGain   = row.ik_gain_vs_llama_pct;
        var ikLabel  = row.ik_best_label || "";
        var ikAvail  = row.ik_available;
        // Tooltip: show which IK config won
        var ikTip = ikAvail && ikLabel
            ? ' title="' + esc("IK best config: " + ikLabel) + '"'
            : (ikAvail ? '' : ' title="IK contrast not run — set IK_LLAMA_SERVER and use --preset ik"');
        // Color-code gain: green=better, red=worse, grey=no data
        var gainColor = !ikAvail ? "var(--text2)"
            : ikGain > 5  ? "var(--green)"
            : ikGain < -5 ? "var(--red)"
            : "var(--yellow)";
        var ikTpsCell  = ikAvail && ikTps > 0
            ? '<span' + ikTip + ' style="font-weight:600">' + ikTps.toFixed(1) + ' t/s</span>'
            : '<span class="na"' + ikTip + '>—</span>';
        var ikGainCell = ikAvail
            ? '<span style="font-weight:600;color:' + gainColor + '">' + (ikGain >= 0 ? "+" : "") + ikGain.toFixed(1) + '%</span>'
            : '<span class="na"' + ikTip + '>—</span>';
        return '<td class="num">' + ikTpsCell  + '</td>'
             + '<td class="num">' + ikGainCell + '</td>';
      }})()
    + '<td><button class="detail-btn" data-detail="' + detailId + '" onclick="toggleDetail(this.dataset.detail)">▶</button></td>'
    + '</tr>'
    // expanded detail row
    + '<tr class="detail-row" id="' + detailId + '">'
    + '<td class="detail-cell" colspan="999"><div class="detail-grid">'
    + '<div class="detail-section"><h4>Model</h4>'
    + (hf.description ? '<p>' + escHtml(hf.description.slice(0,300)) + '</p>' : '')
    + '<div class="kv-row"><span>Base model</span><span>' + escHtml(hf.base_model || "—") + '</span></div>'
    + (hf.is_analogue ? '<div class="kv-row" style="color:var(--yellow)"><span>Benchmark source</span><span title="Benchmarks sourced from this ancestor — not directly evaluated">~' + escHtml(hf.analogue_model || "—") + '</span></div>' : '')
    + '<div class="kv-row"><span>Author</span><span>'     + escHtml(hf.author    || "—") + '</span></div>'
    + '<div class="kv-row"><span>License</span><span>'    + escHtml(hf.license   || "—") + '</span></div>'
    + '<div class="kv-row"><span>Likes</span><span>'      + (hf.likes || "—") + '</span></div>'
    + (hf.aa_intelligence != null ? '<div class="kv-row"><span>AA Intelligence Index</span><span style="color:var(--accent);font-weight:700">' + hf.aa_intelligence + '</span></div>' : '')
    + (hf.aa_output_tps  ? '<div class="kv-row"><span>AA Output Speed</span><span>'   + hf.aa_output_tps.toFixed(1)  + ' t/s</span></div>' : '')
    + (hf.aa_ttft_s      ? '<div class="kv-row"><span>AA Latency (TTFT)</span><span>' + hf.aa_ttft_s.toFixed(2)      + 's</span></div>'    : '')
    + (hf.api_cost_per_mtok ? '<div class="kv-row"><span>API cost (blended)</span><span>$' + hf.api_cost_per_mtok.toFixed(2) + '/1M tokens</span></div>' : '')
    + '</div>'
    + '<div class="detail-section"><h4>GGUF Architecture</h4>'
    + '<div class="kv-row"><span>Architecture</span><span>'   + escHtml(gguf.arch      || "—") + '</span></div>'
    + '<div class="kv-row"><span>Total layers</span><span>'   + (gguf.n_layers    || "—") + '</span></div>'
    + '<div class="kv-row"><span>Attention layers</span><span>' + (gguf.n_attn_layers || "—") + '</span></div>'
    + '<div class="kv-row"><span>KV heads</span><span>'       + (gguf.n_heads_kv  || "—") + '</span></div>'
    + '<div class="kv-row"><span>Head dim</span><span>'       + (gguf.head_dim    || "—") + '</span></div>'
    + '<div class="kv-row"><span>Experts</span><span>'        + (gguf.n_expert ? gguf.n_expert + "x (" + gguf.n_expert_used + " active)" : "—") + '</span></div>'
    + '<div class="kv-row"><span>Train context</span><span>'  + (gguf.context_length ? (gguf.context_length/1024).toFixed(0) + "k" : "—") + '</span></div>'
    + '<div class="kv-row"><span>KV cache/1k tok</span><span>' + (gguf.kv_mb_per_1k ? gguf.kv_mb_per_1k + " MB" : "—") + '</span></div>'
    + '</div>'
    + '<div class="detail-section"><h4>Context Ceilings</h4>'
    + '<div class="kv-row"><span>GPU single</span><span>'    + fmtCtx(ctx.ctx_gpu_single)  + '</span></div>'
    + '<div class="kv-row"><span>GPU combined</span><span>'  + fmtCtx(ctx.ctx_gpu_combined) + '</span></div>'
    + '<div class="kv-row"><span>RAM mixed</span><span>'     + fmtCtx(ctx.ctx_ram_mixed)   + '</span></div>'
    + '<div class="kv-row"><span>RAM explicit</span><span>'  + fmtCtx(ctx.ctx_ram_explicit) + '</span></div>'
    + '<div class="kv-row"><span>Recommended</span><span style="color:var(--green)">' + fmtCtx(ctx.recommended_ctx) + '</span></div>'
    + '</div>'
    + '<div class="detail-section"><h4>Topology Scenarios</h4>'
    + '<div class="kv-row"><span>Winner</span><span style="color:var(--green)">' + escHtml(topo.winner || "—") + '</span></div>'
    + '<div class="kv-row"><span>Max ngl</span><span>' + (topo.max_fit_ngl || "—") + '</span></div>'
    + '<div class="topo-scenarios">' + chipParts.join("") + '</div>'
    + '</div>'
    + (benchDetailParts.length ? '<div class="detail-section"><h4>HF Benchmarks</h4><div class="bench-grid">' + benchDetailParts.join("") + '</div></div>' : '')
    + (recParts.length         ? '<div class="detail-section"><h4>Quant Recommendations</h4><div class="quant-recs">' + recParts.join("") + '</div></div>' : '')
    + (tagChipParts.length     ? '<div class="detail-section"><h4>Tags</h4><div>' + tagChipParts.join("") + '</div></div>' : '')
    + (function() {{
        if (!row.ik_available) return '';
        var ikR = row.ik_best_tps > 0
            ? '<div class="kv-row"><span>IK best t/s</span><span style="color:var(--green);font-weight:700">' + row.ik_best_tps.toFixed(1) + ' t/s</span></div>'
              + '<div class="kv-row"><span>IK best config</span><span>' + escHtml(row.ik_best_label || "—") + '</span></div>'
              + '<div class="kv-row"><span>Gain vs llama.cpp</span><span style="font-weight:600;color:' + (row.ik_gain_vs_llama_pct >= 0 ? 'var(--green)' : 'var(--red)') + '">'
              + (row.ik_gain_vs_llama_pct >= 0 ? "+" : "") + row.ik_gain_vs_llama_pct.toFixed(1) + '%</span></div>'
            : '<div class="kv-row"><span>IK result</span><span class="na">No data</span></div>';
        return '<div class="detail-section"><h4>IK_llama.cpp Contrast</h4>' + ikR + '</div>';
      }})()
    + (function() {{
        if (!row.mtp_available) return '';
        var bp = row.mtp_best_label || "—";
        var mtpR = row.mtp_best_tps > 0
            ? '<div class="kv-row"><span>MTP best t/s</span><span style="color:var(--green);font-weight:700">' + row.mtp_best_tps.toFixed(1) + ' t/s</span></div>'
              + '<div class="kv-row"><span>Best config</span><span>' + escHtml(bp) + '</span></div>'
              + '<div class="kv-row"><span>Gain vs no-MTP</span><span style="font-weight:600;color:' + (row.mtp_gain_pct >= 0 ? 'var(--green)' : 'var(--red)') + '">'
              + (row.mtp_gain_pct >= 0 ? '+' : '') + row.mtp_gain_pct.toFixed(1) + '%</span></div>'
            : '<div class="kv-row"><span>MTP result</span><span class="na">No data</span></div>';
        return '<div class="detail-section"><h4>MTP Draft Sweep</h4>' + mtpR + '</div>';
      }})()
    + '</div></td></tr>';

  return html;
}}

// ── wire up headers ────────────────────────────────────────────────────────────
document.querySelectorAll("th[data-col]").forEach(function(th) {{
  th.addEventListener("click", function() {{ sortData(th.dataset.col); }});
}});

// ── reset dropdowns and initial render ────────────────────────────────────────
// Reset dropdowns so browser-persisted values don't cause unexpected filtering
document.getElementById("filterCase").value   = "";
document.getElementById("filterStatus").value = "";
document.getElementById("search").value       = "";
document.getElementById("toggleRef").checked  = false;

render();


// ── Quant recommendation tooltips ─────────────────────────────────────────────
(function() {{
  const tip = document.createElement("div");
  tip.id = "qtip";
  tip.style.cssText = [
    "position:fixed","z-index:9999","background:var(--surface2)",
    "border:1px solid var(--border)","border-radius:var(--radius)",
    "padding:8px 12px","font-size:12px","line-height:1.7",
    "pointer-events:none","display:none","max-width:320px",
    "box-shadow:0 4px 16px rgba(0,0,0,.4)"
  ].join(";");
  document.body.appendChild(tip);

  document.addEventListener("mouseover", e => {{
    const el = e.target.closest(".quant-tip");
    if (!el) return;
    tip.innerHTML = decodeURIComponent(el.dataset.tip || "");
    tip.style.display = "block";
  }});
  document.addEventListener("mousemove", e => {{
    if (tip.style.display === "none") return;
    const x = e.clientX + 14, y = e.clientY - 8;
    tip.style.left = (x + tip.offsetWidth > window.innerWidth ? x - tip.offsetWidth - 20 : x) + "px";
    tip.style.top  = (y + tip.offsetHeight > window.innerHeight ? y - tip.offsetHeight : y) + "px";
  }});
  document.addEventListener("mouseout", e => {{
    if (e.target.closest(".quant-tip")) tip.style.display = "none";
  }});
}})();
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"  HTML report: {output_path}")


def _make_row(r: dict, all_benchmarks: list) -> dict:
    """Flatten a result dict into a JSON-serialisable row for the HTML template."""
    row = dict(r)
    # Ensure quant_recs is JSON-safe (drop any non-serialisable items)
    row["quant_recs"] = [
        {k: v for k, v in rec.items()
         if isinstance(v, (str, int, float, bool, type(None)))}
        for rec in row.get("quant_recs", [])
    ]
    # Ensure hf, gguf, ctx_detail, topo_detail are present
    row.setdefault("hf", {})
    row.setdefault("gguf", {})
    row.setdefault("ctx_detail", {})
    row.setdefault("topo_detail", {})
    row.setdefault("baseline_detail", {})
    row.setdefault("is_reference", False)
    row.setdefault("rank", None)
    row.setdefault("integrity_sim", None)
    row.setdefault("integrity_kv", "")
    row.setdefault("integrity_kv_type", "")
    row.setdefault("reasoning_score", None)
    row.setdefault("reasoning_total", None)
    row.setdefault("best_kv_type", "")
    row.setdefault("best_flash_attn", "")
    row.setdefault("best_batch", "")
    row.setdefault("best_ubatch", "")
    # IK contrast fields
    row.setdefault("ik_best_tps", 0.0)
    row.setdefault("ik_gain_vs_llama_pct", 0.0)
    row.setdefault("ik_best_label", "")
    row.setdefault("ik_available", False)
    row.setdefault("mtp_best_tps", 0.0)
    row.setdefault("mtp_gain_pct", 0.0)
    row.setdefault("mtp_best_label", "")
    row.setdefault("mtp_available", False)

    # Size fallback: if estimated_fp16_gb or current_bpw are missing, compute
    # actual file size in GB directly from model_path so the Size column isn't blank.
    if not row.get("estimated_fp16_gb") or not row.get("current_bpw"):
        _mp = Path(str(row.get("model_path", "")))
        try:
            _size_gb = round(_mp.stat().st_size / (1024 ** 3), 2)
            row.setdefault("file_size_gb", _size_gb)
        except Exception:
            row.setdefault("file_size_gb", None)
    else:
        row["file_size_gb"] = round(
            row["estimated_fp16_gb"] * row["current_bpw"] / 16, 2)

    # Remove non-serialisable objects
    for key in list(row.keys()):
        v = row[key]
        if isinstance(v, Path):
            row[key] = str(v)
    # Truncate description to prevent oversized JSON
    hf = row.get("hf", {})
    if hf.get("description") and len(hf["description"]) > 500:
        hf["description"] = hf["description"][:500]

    # ── Parameter count fallback cascade (Tiers 5–7) ─────────────────────────
    # Tiers 1-4 run during HF fetch (safetensors, cardData, README, repo_id).
    # These tiers run at row-build time and use data already on the row.
    if not hf.get("parameters_b"):
        gguf = row.get("gguf", {})
        model_name = str(row.get("model", ""))

        # Tier 5: GGUF general.parameter_count (exact, embedded in the file)
        if gguf.get("parameters_b"):
            hf["parameters_b"] = gguf["parameters_b"]

        # Tier 6: parse from GGUF filename / model name
        if not hf.get("parameters_b"):
            _pb = _extract_params_from_text("", model_name)
            if _pb:
                hf["parameters_b"] = _pb

        # Tier 7: back-calculate from estimated_fp16_gb (fp16 = 2 bytes/param)
        # estimated_fp16_gb is the full-precision size computed from GGUF metadata;
        # dividing by 2 gives the parameter count in billions.
        if not hf.get("parameters_b"):
            _fp16 = row.get("estimated_fp16_gb")
            if _fp16 and float(_fp16) > 0:
                _pb = round(float(_fp16) / 2, 2)
                if 0.05 <= _pb <= 2000:
                    hf["parameters_b"] = _pb

    return row


# ── main ───────────────────────────────────────────────────────────────────────

def generate(
    report_json:   Path,
    output_path:   Path | None,
    no_hf:         bool,
    force_refresh: bool,
) -> Path:
    """
    Main entry point. Can be called programmatically from batch_runner.py.
    Returns the path to the generated HTML file.
    """
    print(f"\n  Generating HTML report from: {report_json}")

    data = json.loads(report_json.read_text(encoding="utf-8"))
    results  = data.get("results", [])
    gpu0     = data.get("gpu0_vram_gb", 24)
    gpu1     = data.get("gpu1_vram_gb", 16)
    _gi = data.get("gpu_info", [])  # e.g. ["NVIDIA GeForce RTX 3090 24GB", ...]
    gpu0_name = _gi[0].rsplit(" ", 1)[0].replace("NVIDIA GeForce ","").replace("NVIDIA ","") if _gi else "GPU0"
    gpu1_name = _gi[1].rsplit(" ", 1)[0].replace("NVIDIA GeForce ","").replace("NVIDIA ","") if len(_gi)>1 else "GPU1"
    report_meta = {
        "models_dir":   data.get("models_dir", ""),
        "gpu0_vram_gb": gpu0,
        "gpu1_vram_gb": gpu1,
        "gpu0_name":    gpu0_name,
        "gpu1_name":    gpu1_name,
        "total":        data.get("total", len(results)),
        "successful":   data.get("successful", 0),
        "generated":    data.get("generated", ""),
    }

    # Deduplicate multi-shard models — keep shard 1, drop shards 2..N
    import re as _sre
    _sr = _sre.compile(r'-(\d{5})-of-\d{5}(?:\.gguf)?$', _sre.IGNORECASE)
    _seen: dict = {}
    _deduped = []
    for _r in results:
        _stem = Path(_r.get("model_path", _r.get("model", ""))).stem
        _sm = _sr.search(_stem)
        if _sm:
            _canon = _stem[:_sm.start()]
            _idx   = int(_sm.group(1))
            if _canon not in _seen:
                _seen[_canon] = _idx
                _deduped.append(_r)
            elif _idx < _seen[_canon]:
                _seen[_canon] = _idx
                _deduped[-1] = _r
        else:
            _deduped.append(_r)
    if len(_deduped) < len(results):
        print(f"  [info] Merged {len(results)-len(_deduped)} duplicate shard entries")
    results = _deduped

    if not results:
        print("  No results in report — nothing to generate")
        return None

    # ── Backfill from companion batch CSV ─────────────────────────────────────
    # batch_runner writes a CSV alongside the JSON with the same stem.
    # It includes topo_case, topo_winner, ctx_gpu, ctx_ram, recommended_ctx.
    # When the JSON has empty strings/None for those fields (e.g. the topo/ctx
    # sub-JSON files couldn't be found at batch-save time), the CSV may still
    # have the correct values if it was written by a newer batch run.
    import csv as _csv
    _csv_path = report_json.with_suffix(".csv")
    if not _csv_path.exists():
        # Also try the latest CSV in the same directory
        _csv_candidates = sorted(report_json.parent.glob("batch_report_*.csv"))
        if _csv_candidates:
            _csv_path = _csv_candidates[-1]
    if _csv_path.exists():
        try:
            _csv_by_model: dict[str, dict] = {}
            with open(_csv_path, newline="", encoding="utf-8") as _f:
                for _row in _csv.DictReader(_f):
                    _key = Path(_row.get("model", "")).name or _row.get("model", "")
                    _csv_by_model[_key] = _row
            _filled = 0
            for _r in results:
                _key = Path(_r.get("model", "")).name or _r.get("model", "")
                _crow = _csv_by_model.get(_key)
                if not _crow:
                    continue
                if not _r.get("topo_case") and _crow.get("topo_case"):
                    _r["topo_case"]   = _crow["topo_case"]
                    _filled += 1
                if not _r.get("topo_winner") and _crow.get("topo_winner"):
                    _r["topo_winner"] = _crow["topo_winner"]
                if not _r.get("ctx_gpu") and _crow.get("ctx_gpu"):
                    try: _r["ctx_gpu"] = int(_crow["ctx_gpu"])
                    except Exception: pass
                if not _r.get("ctx_ram") and _crow.get("ctx_ram"):
                    try: _r["ctx_ram"] = int(_crow["ctx_ram"])
                    except Exception: pass
                if not _r.get("recommended_ctx") and _crow.get("recommended_ctx"):
                    try: _r["recommended_ctx"] = int(_crow["recommended_ctx"])
                    except Exception: pass
            if _filled:
                print(f"  [CSV] Backfilled topo_case for {_filled} models from {_csv_path.name}")
        except Exception as _e:
            print(f"  [CSV] Warning: could not read companion CSV: {_e}")

    # ── Infer topo_case from model size when topo sweep wasn't run ───────────────
    # The topo sweep writes topo_results.json; when it wasn't run that file is
    # absent and topo_case/topo_winner stay empty.  We can reliably infer the
    # case from the estimated model size vs the two GPU VRAM budgets.
    # Overhead: ~2 GB for KV cache + compute buffers at default context.
    _VRAM_OVERHEAD_GB = 2.0
    _min_vram  = min(gpu0, gpu1)          # fits in BOTH independently → A
    _max_vram  = max(gpu0, gpu1)          # fits in GPU0 only          → B
    _combined  = gpu0 + gpu1              # fits combined               → C
    _inferred_cases = 0
    for _r in results:
        if _r.get("topo_case"):
            continue  # already populated — leave it alone
        _fp16 = _r.get("estimated_fp16_gb")
        _bpw  = _r.get("current_bpw")
        if not _fp16 or not _bpw:
            continue
        _size_gb = _fp16 * _bpw / 16.0
        _needed  = _size_gb + _VRAM_OVERHEAD_GB
        if   _needed <= _min_vram:  _r["topo_case"] = "A"
        elif _needed <= _max_vram:  _r["topo_case"] = "B"
        elif _needed <= _combined:  _r["topo_case"] = "C"
        else:                       _r["topo_case"] = "D"
        _inferred_cases += 1
    if _inferred_cases:
        print(f"  [topo] Inferred topo_case for {_inferred_cases} models from size+VRAM")

    # Enrich with local GGUF metadata and sweep details
    print(f"  Loading local metadata for {len(results)} models...")
    # Pass report_json parent (batch_reports/) → parent.parent = project root
    # so relative results_dir paths resolve correctly regardless of CWD
    _base_dir = report_json.parent.parent
    results = enrich_with_local_meta(results, base_dir=_base_dir)

    # Enrich with HF metadata
    cache_path = report_json.parent / "hf_cache.json"
    results    = enrich_with_hf(results, cache_path, force_refresh, no_hf)

    # Output path
    if output_path is None:
        stem = report_json.stem.replace("batch_report_", "report_")
        output_path = report_json.parent / f"{stem}.html"

    render_html(results, report_meta, output_path)

    # Also write a report_latest.html
    latest = report_json.parent / "report_latest.html"
    import shutil
    shutil.copy2(output_path, latest)
    print(f"  Latest  : {latest}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a sortable HTML report from LLM Optimizer results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--report", default=None,
                        help="Path to batch_report_*.json (default: latest)")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace read token for accessing leaderboard "
                             "benchmark data (or set HF_TOKEN env var). "
                             "Get one at https://huggingface.co/settings/tokens")
    parser.add_argument("--output", default=None,
                        help="Output HTML path (default: same dir as report)")
    parser.add_argument("--no-hf", action="store_true",
                        help="Skip Hugging Face metadata fetch")
    parser.add_argument("--refresh-hf", action="store_true",
                        help="Force re-fetch all HF metadata ignoring cache")
    parser.add_argument("--reports-dir", default=None,
                        help="Directory containing batch reports (default: auto-detect)")
    args = parser.parse_args()

    # Locate the report JSON
    if args.report:
        report_json = Path(args.report)
        if not report_json.exists():
            print(f"Error: report not found: {report_json}")
            sys.exit(1)
    else:
        # Auto-detect: look for batch_reports/ relative to this script
        search_dirs = [SCRIPT_DIR]
        if args.reports_dir:
            search_dirs = [Path(args.reports_dir)]

        report_json = None
        for d in search_dirs:
            br_dir = d / "batch_reports"
            if br_dir.exists():
                candidates = sorted(br_dir.glob("batch_report_*.json"))
                if candidates:
                    # Prefer latest by timestamp in filename, not mtime
                    report_json = candidates[-1]
                    break

        if report_json is None:
            print("Error: no batch report found. Run batch_runner.py first, "
                  "or specify --report path/to/batch_report.json")
            sys.exit(1)

    # Apply HF token from CLI arg (overrides env var)
    global HF_TOKEN
    if args.hf_token:
        HF_TOKEN = args.hf_token

    output = Path(args.output) if args.output else None

    result_path = generate(
        report_json   = report_json,
        output_path   = output,
        no_hf         = args.no_hf,
        force_refresh = args.refresh_hf,
    )

    if result_path:
        print(f"\n  Done. Open in browser:")
        print(f"    {result_path}")


if __name__ == "__main__":
    main()
