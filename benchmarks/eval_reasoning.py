#!/usr/bin/env python3
"""
eval_reasoning.py - Official lm-evaluation-harness Reasoning Benchmark Runner

Evaluates local models against authoritative reasoning and mathematical benchmarks
(e.g., GSM8K, MATH, ARC-Challenge) using EleutherAI's lm-evaluation-harness.
Automatically handles virtualenv execution, endpoint routing, and structured summary reporting.

Usage:
  python3 benchmarks/eval_reasoning.py think
  python3 benchmarks/eval_reasoning.py think --tasks gsm8k --limit 50
  python3 benchmarks/eval_reasoning.py main --tasks gsm8k --num-concurrent 4
  python3 benchmarks/eval_reasoning.py think --tasks minerva_math --limit 100
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = REPO_ROOT / "benchmarks" / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python3"
LM_EVAL_BIN = VENV_DIR / "bin" / "lm-eval"
if not LM_EVAL_BIN.exists():
    LM_EVAL_BIN = VENV_DIR / "bin" / "lm_eval"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results" / "reasoning"

def ensure_venv():
    """Re-executes script inside benchmarks/.venv if running under system python."""
    if not VENV_PYTHON.exists() or not LM_EVAL_BIN.exists():
        print(f"[!] Error: lm-eval virtual environment not found at {VENV_DIR}")
        print("    Please run: python3 -m venv benchmarks/.venv && benchmarks/.venv/bin/pip install 'lm-eval[api]'")
        sys.exit(1)

    if sys.prefix != str(VENV_DIR):
        cmd = [str(VENV_PYTHON)] + sys.argv
        os.execv(str(VENV_PYTHON), cmd)

def parse_args():
    parser = argparse.ArgumentParser(description="Official lm-evaluation-harness Reasoning Benchmark")
    parser.add_argument("model", nargs="?", default="think", help="Model name in llama-swap (default: think)")
    parser.add_argument("--tasks", default="gsm8k", help="Comma-separated tasks (default: gsm8k, e.g. minerva_math, arc_challenge)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems per task (for testing/quick runs)")
    parser.add_argument("--num-concurrent", type=int, default=1, help="Number of concurrent requests (default: 1; use 2 or 4 for main)")
    parser.add_argument("--max-gen-toks", type=int, default=2048, help="Maximum generation tokens per problem (default: 2048)")
    parser.add_argument("--base-url", default=os.environ.get("LLAMA_SWAP_URL", "http://localhost:8090") + "/v1/chat/completions",
                        help="OpenAI-compatible chat completions endpoint")
    return parser.parse_args()

def print_summary(results_file: Path, model_name: str, tasks: str, elapsed: float):
    if not results_file.exists():
        print(f"[!] Warning: Result file not found at {results_file}")
        return

    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})
    if not results:
        print("[!] No results found in json output.")
        return

    print("\n" + "=" * 80)
    print(f"📊 LM-EVALUATION-HARNESS RESULTS — {model_name.upper()}")
    print("=" * 80)
    print(f"  • Tasks Evaluated:       {tasks}")
    print(f"  • Total Time Elapsed:    {elapsed:.1f}s")
    print("-" * 80)

    for task_name, task_metrics in results.items():
        sample_len = task_metrics.get("sample_len", "N/A")
        print(f"\n  Task: {task_name.upper()} (N={sample_len})")
        for k, v in sorted(task_metrics.items()):
            if k in ("name", "alias", "sample_len"):
                continue
            if isinstance(v, float):
                if "stderr" in k:
                    print(f"    - {k:<32}: ± {v:.4f}")
                else:
                    print(f"    - {k:<32}: {v * 100:.2f}% ({v:.4f})")
            else:
                print(f"    - {k:<32}: {v}")

    print("=" * 80 + "\n")

def isolate_model(target_model: str, base_url: str):
    """Ensures single dynamic heavy model residency (main or think) during benchmark."""
    from urllib.parse import urlparse
    p = urlparse(base_url)
    swap_root = f"{p.scheme}://{p.netloc}"
    try:
        req = urllib.request.Request(f"{swap_root}/running")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            running = [item.get("model") for item in data.get("running", []) if item.get("model")]
            conflicts = [m for m in running if m != target_model and m not in ("embed", "rerank")]
            if conflicts:
                print(f"[*] Memory Isolation: Conflicting model(s) {conflicts} active in VRAM.")
                for c in conflicts:
                    print(f"[*] Evicting '{c}' to guarantee dedicated memory bandwidth for '{target_model}'...")
                    payload = json.dumps({"model": c}).encode("utf-8")
                    unload_req = urllib.request.Request(
                        f"{swap_root}/api/models/unload",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST"
                    )
                    with urllib.request.urlopen(unload_req, timeout=15):
                        pass
                time.sleep(1.0)
                print(f"[✓] VRAM cleared. Only '{target_model}' (and zero-eviction embeddings) will reside.")
            else:
                print(f"[✓] Memory Isolation: VRAM clean (currently active: {running}).")
    except Exception as e:
        print(f"[!] Warning: Memory isolation check failed: {e}")

def main():
    ensure_venv()
    args = parse_args()
    isolate_model(args.model, args.base_url)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_tasks = args.tasks.replace(",", "_")
    output_stem = f"{args.model}_{clean_tasks}_{timestamp}"
    output_path = RESULTS_DIR / f"{output_stem}.json"

    model_args = f"model={args.model},base_url={args.base_url},max_gen_toks={args.max_gen_toks},max_length=16384"
    if args.num_concurrent > 1:
        model_args += f",num_concurrent={args.num_concurrent}"

    cmd = [
        str(LM_EVAL_BIN),
        "run",
        "--model", "local-chat-completions",
        "--model_args", model_args,
        "--tasks", args.tasks,
        "--apply_chat_template",
        "--gen_kwargs", 'until=["<|im_end|>","<|endoftext|>"]',
        "--log_samples",
        "--output_path", str(output_path),
    ]

    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])

    print("=" * 80)
    print(f"🧠 Official Reasoning Evaluation: {args.model.upper()}")
    print(f"   Tasks:        {args.tasks}")
    print(f"   Limit:        {args.limit if args.limit else 'Full Dataset'}")
    print(f"   Concurrency:  {args.num_concurrent}")
    print(f"   Endpoint:     {args.base_url}")
    print(f"   Output File:  {output_path}")
    print("=" * 80)

    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] Evaluation runner failed with exit code: {e.returncode}")
        sys.exit(e.returncode)
    elapsed = time.time() - t0

    actual_file = output_path
    if not actual_file.exists():
        matches = sorted(RESULTS_DIR.glob(f"{args.model}_{clean_tasks}*.json"), key=os.path.getmtime)
        if matches:
            actual_file = matches[-1]

    print_summary(actual_file, args.model, args.tasks, elapsed)

if __name__ == "__main__":
    main()
