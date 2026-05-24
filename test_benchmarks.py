#!/usr/bin/env python3
"""
test_benchmarks.py — Tests the leaderboard benchmark fetching against 40 real models.

Usage:
    python test_benchmarks.py --token hf_yourToken

Or set env var:
    $env:HF_TOKEN = "hf_yourToken"
    python test_benchmarks.py
"""
import argparse, json, os, re, sys, time
from pathlib import Path

try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

# ─── 40 models sampled from Victor's 166 ────────────────────────────────────
# Mix of: base models found directly, GGUF repos that need base_model follow,
# v1 leaderboard era (2023–2024), v2 era (2024–2025), and expected-empty ones.
TEST_MODELS = [
    # ── Direct base model repos (no GGUF wrapper) ────────────────────────
    ("Meta-Llama-3-8B-Instruct-Q4_K_M",  "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("Meta-Llama-3-70B-Instruct-IQ1_M",  "meta-llama/Meta-Llama-3-70B-Instruct"),
    ("mistral-7b-instruct-v0.1.Q4_K_M",  "mistralai/Mistral-7B-Instruct-v0.1"),
    ("nous-hermes-llama2-13b.Q4_0",       "NousResearch/Nous-Hermes-Llama2-13b"),
    ("gemma-2-27b-it-IQ2_M",             "google/gemma-2-27b-it"),
    ("aya-23-35B-IQ4_NL",                "CohereLabs/aya-23-35B"),
    ("Codestral-22B-v0.1-hf.IQ4_XS",    "mistralai/Codestral-22B-v0.1"),
    ("Starling-LM-7B-beta-Q4_K_M",       "Nexusflow/Starling-LM-7B-beta"),
    ("gemma-3-27b-it-UD-Q4_K_XL",        "google/gemma-3-27b-it"),
    ("Llama-3.2-3B-Instruct-UD-Q4_K_XL", "meta-llama/Llama-3.2-3B-Instruct"),
    # ── GGUF repos → need base_model follow ──────────────────────────────
    ("vicuna-13b-v1.5-16k.Q4_K_M",       "lmsys/vicuna-13b-v1.5-16k"),
    ("zephyr-7b-beta.Q4_K_M",            "HuggingFaceH4/zephyr-7b-beta"),
    ("mythomax-l2-13b.Q4_K_M",           "Gryphe/MythoMax-L2-13b"),
    ("neuralhermes-2.5-mistral-7b.Q4_K_M","mlabonne/NeuralHermes-2.5-Mistral-7B"),
    ("starling-lm-7b-alpha.Q5_K_M",      "berkeley-nest/Starling-LM-7B-alpha"),
    ("dolphin-2_6-phi-2.Q2_K",           "cognitivecomputations/dolphin-2_6-phi-2"),
    ("openhermes-2.5-mistral-7b-16k",    "teknium/OpenHermes-2.5-Mistral-7B"),
    ("phind-codellama-34b-v2.Q3_K_S",    "Phind/Phind-CodeLlama-34B-v2"),
    ("yarn-mistral-7b-128k.Q4_K_M",      "NousResearch/Yarn-Mistral-7b-128k"),
    ("Llama-3-8B-Instruct-Gradient-1048k","gradientai/Llama-3-8B-Instruct-Gradient-1048k"),
    # ── v2 leaderboard era (June 2024+) ──────────────────────────────────
    ("Qwen2-7B-Instruct.Q5_K_M",         "Qwen/Qwen2-7B-Instruct"),
    ("gemma-2-27b-it.Q3_K_S",            "google/gemma-2-27b-it"),
    ("Gemma-2-9B-It-SPPO-Iter3-Q4_K_M",  "UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3"),
    ("phi-2.Q8_0",                        "microsoft/phi-2"),
    ("Meta-Llama-3.1-8B-Instruct-abliterated","meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("qwen2.5-coder-7b-instruct.Q4_K_M", "Qwen/Qwen2.5-Coder-7B-Instruct"),
    ("DeepSeek-R1-0528-Qwen3-8B-Q8_0",   "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"),
    ("Llama-3.2-3B-Instruct-uncensored",  "meta-llama/Llama-3.2-3B-Instruct"),
    # ── Expected empty (custom fine-tunes, no leaderboard submission) ────
    ("miqu-evil-dpo.i1-IQ1_S",           "maywell/miqu-evil-dpo"),
    ("tinystories-gpt-0.1-3m.Q4_K_M",    "segestic/Tinystories-gpt-0.1-3m"),
    ("medllama3-v20.Q4_K_M",             "ProbeMedicalYonseiMAILab/medllama3-v20"),
    ("BigMaid-20B-v1.0.IQ4_XS",          "TheDrummer/BigMaid-20B"),
    ("Qwen3.5-35B-A3B-UD-Q4_K_XL",       "Qwen/Qwen3.5-35B-A3B"),
    ("GLM-5-UD-IQ2_XXS",                 "THUDM/GLM-5"),
    ("Devstral-Small-2-24B-Instruct",     "mistralai/Devstral-Small-2-24B-Instruct-2512"),
    ("gemma-3n-E4B-it-Q8_0",             "google/gemma-3n-E4B-it"),
    ("Nemotron-3-Nano-30B-A3B",          "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"),
    ("falcon-mamba-7B-Q4_K_M",           "tiiuae/falcon-mamba-7b"),
    ("CodeQwen1.5-7B-Chat-Q4_K_M",       "Qwen/CodeQwen1.5-7B-Chat"),
    ("BioMistral-7B.Q4_K_M",             "BioMistral/BioMistral-7B"),
]

REQUEST_TIMEOUT = 15
REQUEST_DELAY   = 0.3


def fetch_v1(repo_id: str, session) -> dict:
    try:
        org, model = repo_id.split("/", 1)
    except ValueError:
        return {}
    details_name = f"details_{org}__{model}"
    dataset_id   = f"open-llm-leaderboard-old/{details_name}"
    list_url     = f"https://huggingface.co/api/datasets/{dataset_id}/tree/main"
    try:
        r = session.get(list_url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return {}
        files = r.json()
        result_files = [f["path"] for f in files if isinstance(f, dict)
                        and f.get("path","").endswith(".json")
                        and "results_" in f.get("path","")]
        if not result_files:
            return {}
        result_files.sort(reverse=True)
        result_path = result_files[0]
    except Exception as e:
        return {}

    raw_url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{result_path}"
    try:
        time.sleep(REQUEST_DELAY)
        r = session.get(raw_url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return {}
        data = r.json()
    except Exception:
        return {}

    results = data.get("results", {})
    if not results:
        return {}

    benchmarks = {}

    def _get_first(frags, metrics):
        for frag in frags:
            for key in results:
                if frag.lower() in key.lower():
                    task = results[key]
                    for mk in metrics:
                        v = task.get(mk)
                        if v is not None and isinstance(v, (int,float)) and float(v) > 0:
                            return float(v)
        return None

    def _avg_subtasks(frag, metrics):
        vals = []
        for key in results:
            if frag.lower() in key.lower():
                task = results[key]
                for mk in metrics:
                    v = task.get(mk)
                    if v is not None and isinstance(v,(int,float)) and float(v) > 0:
                        vals.append(float(v)); break
        return sum(vals)/len(vals) if vals else None

    v = _get_first(["arc_challenge","arc:challenge","|arc"], ["acc_norm","acc"])
    if v: benchmarks["ARC"] = round(v*100,1)
    v = _get_first(["hellaswag"], ["acc_norm","acc"])
    if v: benchmarks["HellaSwag"] = round(v*100,1)
    v = _get_first(["|mmlu|","harness|mmlu"], ["acc","acc_norm"])
    if v is None: v = _avg_subtasks("hendrycksTest", ["acc","acc_norm"])
    if v is None: v = _avg_subtasks("mmlu", ["acc","acc_norm"])
    if v: benchmarks["MMLU"] = round(v*100,1)
    v = _get_first(["truthfulqa"], ["mc2","acc"])
    if v: benchmarks["TruthfulQA"] = round(v*100,1)
    v = _get_first(["winogrande"], ["acc","acc_norm"])
    if v: benchmarks["Winogrande"] = round(v*100,1)
    v = _get_first(["gsm8k"], ["acc","acc_norm","exact_match"])
    if v: benchmarks["GSM8K"] = round(v*100,1)
    return benchmarks


def fetch_v2(repo_id: str, session) -> dict:
    try:
        org, model = repo_id.split("/", 1)
    except ValueError:
        return {}
    details_name = f"details_{org}__{model}"
    dataset_id   = f"open-llm-leaderboard/{details_name}"
    list_url     = f"https://huggingface.co/api/datasets/{dataset_id}/tree/main"
    try:
        r = session.get(list_url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return {}
        files = r.json()
        result_files = [f["path"] for f in files if isinstance(f, dict)
                        and f.get("path","").endswith(".json")
                        and "results_" in f.get("path","")]
        if not result_files:
            return {}
        result_files.sort(reverse=True)
        result_path = result_files[0]
    except Exception:
        return {}

    raw_url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{result_path}"
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

    def _get(frags, metrics):
        for frag in frags:
            for key in all_results:
                if frag.lower() in key.lower():
                    task = all_results[key]
                    for mk in metrics:
                        for sfx in ("",",none"):
                            v = task.get(mk+sfx)
                            if v is not None and isinstance(v,(int,float)) and float(v)>0:
                                return float(v)
        return None

    v = _get(["arc_challenge","arc:challenge"], ["acc_norm","acc"])
    if v: benchmarks["ARC"] = round(v*100,1)
    v = _get(["bbh"], ["acc_norm","acc"])
    if v: benchmarks["BBH"] = round(v*100,1)
    v = _get(["math_hard","math"], ["exact_match","acc_norm","acc"])
    if v: benchmarks["MATH"] = round(v*100,1)
    v = _get(["gpqa"], ["acc_norm","acc"])
    if v: benchmarks["GPQA"] = round(v*100,1)
    v = _get(["mmlu_pro","mmlu"], ["acc","acc_norm"])
    if v: benchmarks["MMLU"] = round(v*100,1)
    return benchmarks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN",""))
    args = parser.parse_args()

    if not args.token:
        print("ERROR: No HF token. Pass --token hf_... or set HF_TOKEN env var.")
        sys.exit(1)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "llm-optimizer-benchmark-test/1.0",
        "Authorization": f"Bearer {args.token}"
    })

    # Verify token
    r = session.get("https://huggingface.co/api/whoami-v2", timeout=10)
    if r.status_code != 200:
        print(f"Token check FAILED: {r.status_code} {r.text[:100]}")
        sys.exit(1)
    print(f"Token OK — logged in as: {r.json().get('name','?')}\n")

    cols = ["ARC","HellaSwag","MMLU","TruthfulQA","Winogrande","GSM8K","BBH","MATH","GPQA"]
    header = f"{'Model':<52} {'Src':<4} " + " ".join(f"{c:>9}" for c in cols)
    print(header)
    print("─" * len(header))

    found_any = 0
    for filename, base_id in TEST_MODELS:
        b1 = fetch_v1(base_id, session)
        b2 = fetch_v2(base_id, session)
        bench = {**b1, **b2}
        src = ("v1+2" if b1 and b2 else "v1" if b1 else "v2" if b2 else "—")
        if bench:
            found_any += 1
        row = f"{filename[:51]:<52} {src:<4} "
        row += " ".join(f"{bench.get(c,'—'):>9}" if bench.get(c) else f"{'—':>9}" for c in cols)
        print(row)
        time.sleep(0.1)

    print(f"\n{'─'*len(header)}")
    print(f"Found benchmarks for {found_any}/{len(TEST_MODELS)} models")

if __name__ == "__main__":
    main()
