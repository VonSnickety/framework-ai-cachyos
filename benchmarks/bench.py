#!/usr/bin/env python3
"""
bench.py - Unified Performance & Feature Benchmark Suite

Measures serial generation speed (tok/s), prompt ingestion (PP tok/s),
Time To First Token (TTFT), memory bandwidth (GB/s), speculative decoding (MTP),
prompt prefix cache efficiency, and reasoning effort controls.

Usage:
  python3 benchmarks/bench.py main
  python3 benchmarks/bench.py main --spec
  python3 benchmarks/bench.py main --cache
  python3 benchmarks/bench.py main --no-think
  python3 benchmarks/bench.py main --context-sweep 512,2048,8192,32768
"""

import argparse
import base64
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

API_URL = os.environ.get("LLAMA_SWAP_URL", "http://localhost:8090") + "/v1/chat/completions"
API_KEY = os.environ.get("LLAMA_SWAP_API_KEY", "")

# Hardware constants
ROOFLINE_GB_S = 256.0

# Weight footprint of the GGUF actually served by each model ID, in decimal GB
# to match ROOFLINE_GB_S. Re-measure with `stat -c %s` after any requant — a stale
# entry here silently corrupts every bandwidth figure downstream.
#   main  36.4 GiB  Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf
#   think 23.6 GiB  Qwen3.8-27B-UD-Q6_K_XL.gguf   (Dynamic v3.0)
#   admin  4.4 GiB  qwen2.5-coder-7b-instruct.Q4_K_M.gguf
#   embed  0.6 GiB  bge-m3-q8_0.gguf
MODEL_WEIGHTS = {
    "main": {"weights_gb": 39.1, "active_gb": 3.0, "type": "moe"},
    "think": {"weights_gb": 25.3, "active_gb": None, "type": "dense"},
    "flash": {"weights_gb": 93.7, "active_gb": 4.5, "type": "moe"},
    "tiny": {"weights_gb": 4.4, "active_gb": None, "type": "dense"},
    "vision": {"weights_gb": 36.8, "active_gb": None, "type": "dense"},
    "revil": {"weights_gb": 27.2, "active_gb": None, "type": "dense"},
    "revil2": {"weights_gb": 27.2, "active_gb": None, "type": "dense"},
    "embed": {"weights_gb": 0.6, "active_gb": None, "type": "dense"},
}

def strip_think(text: str) -> str:
    """Removes thinking blocks from output if present."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def calc_bandwidth(gen_tok_s: float, weights_gb: float, active_gb: Optional[float] = None,
                   accept_ratio: float = 0.0) -> Tuple[float, str]:
    """Computes effective memory bandwidth vs the hardware roofline.

    DRAM traffic is driven by *forward passes*, not emitted tokens. With MTP
    speculative decoding every accepted draft token rides along on a pass that
    already happened, so `tok/s * weights` overstates real traffic by the MTP
    speedup factor — which is how this metric used to report figures above the
    256 GB/s physical bus. Discount by the measured draft acceptance ratio.
    """
    per_token_gb = active_gb if active_gb else weights_gb
    basis = "active params" if active_gb else "full weights"

    passes_s = gen_tok_s * (1.0 - max(0.0, min(accept_ratio, 0.99)))
    eff_bw = passes_s * per_token_gb
    pct = (eff_bw / ROOFLINE_GB_S) * 100.0

    note = f"{basis}"
    if accept_ratio > 0:
        note += f", MTP-adjusted {accept_ratio * 100:.0f}% accepted"
    label = f"{eff_bw:.1f} GB/s ({pct:.1f}% of {ROOFLINE_GB_S:.0f} GB/s bus, {note})"
    if eff_bw > ROOFLINE_GB_S:
        # Physically impossible — means a stale MODEL_WEIGHTS entry or a bad
        # acceptance reading. Surface it instead of publishing the number.
        label += "  [!] EXCEEDS ROOFLINE - check MODEL_WEIGHTS"
    return eff_bw, label

def accept_ratio_from(res: dict) -> float:
    """Fraction of *emitted tokens* that came from accepted MTP drafts (0.0 if no spec decoding).

    Note this is NOT the same as the conventional "draft acceptance rate"
    (accepted / drafted, see draft_accept_rate). Coverage is the correct basis
    for discounting bandwidth; acceptance is the correct basis for judging the
    p-min confidence gate. Reporting one under the other's name is how `main`
    ended up documented at "96.87% acceptance" alongside ~54% coverage — both
    true, different denominators.
    """
    t = res.get("timings") or {}
    predicted = t.get("predicted_n") or 0
    accepted = t.get("draft_n_accepted") or 0
    if predicted <= 0 or accepted <= 0:
        return 0.0
    return min(accepted / predicted, 0.99)

def draft_accept_rate(res: dict) -> float:
    """Conventional MTP acceptance: accepted / drafted (0.0 if no drafts were issued)."""
    t = res.get("timings") or {}
    drafted = t.get("draft_n") or 0
    accepted = t.get("draft_n_accepted") or 0
    if drafted <= 0:
        return 0.0
    return accepted / drafted

def post_chat(payload: dict, timeout: int = 180) -> dict:
    """Executes a chat completion request against llama-swap."""
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}" if API_KEY else ""
        }
    )
    start_wall = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    data["_wall_time_s"] = time.perf_counter() - start_wall
    return data

def warmup(model: str) -> None:
    """Forces llama-swap to load the model before any scored request.

    Without this, the first scored prompt's wall time includes the multi-second
    model load, which used to drag the reported average well below the real
    generation rate (e.g. main: 11.6 tok/s cold vs ~54 tok/s warm).
    """
    print("▶ Warmup (loading model, not scored) ...", end="", flush=True)
    t0 = time.perf_counter()
    try:
        post_chat({"model": model, "messages": [{"role": "user", "content": "ok"}], "max_tokens": 8}, timeout=900)
        print(f" ready in {time.perf_counter() - t0:.1f}s")
    except Exception as e:
        print(f" ⚠ warmup failed: {e}")

def gen_rate(res: dict, comp_tokens: int, wall_time: float) -> float:
    """Pure decode rate from llama-server timings, falling back to end-to-end."""
    t = res.get("timings") or {}
    pps = t.get("predicted_per_second")
    if pps:
        return float(pps)
    return comp_tokens / max(wall_time, 0.001)

def run_standard_bench(model: str, max_tokens: int = 256, temp: float = 0.0, no_think: bool = False):
    """Runs primary serial generation, TTFT, and prefill tests."""
    print(f"\n🚀 Running Performance Benchmark: {model}")
    print("=" * 60)

    # Resolve model profile
    profile = MODEL_WEIGHTS.get(model, {"weights_gb": 35.0, "active_gb": None, "type": "dense"})
    
    prompts = [
        ("Short Python Refactor", "Write a Python function to compute the longest common subsequence of two strings. Return only code."),
        ("Architectural Analysis", "Explain the trade-offs between zero-copy shared memory and distributed message queues in high-throughput streaming systems."),
        ("Concurrency Synthesis", "Write a thread-safe LRU cache in Go with sync.RWMutex, eviction callbacks, and O(1) get/put operations.")
    ]

    warmup(model)

    results = []
    for title, prompt in prompts:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temp,
            "stream": False
        }
        if no_think:
            payload["chat_template_kwargs"] = {"reasoning_effort": "low"}
            payload["messages"][0]["content"] = "<|think_off|>" + prompt

        print(f"▶ Testing: {title} ...", end="", flush=True)
        try:
            res = post_chat(payload)
            usage = res.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            comp_tokens = usage.get("completion_tokens", 0)
            wall_time = res.get("_wall_time_s", 0.0)
            # Decode rate (what "tok/s" should mean); e2e includes prefill + TTFT.
            tok_s = gen_rate(res, comp_tokens, wall_time)
            e2e_tok_s = comp_tokens / max(wall_time, 0.001)

            acc = accept_ratio_from(res)
            bw_val, bw_str = calc_bandwidth(tok_s, profile["weights_gb"], profile["active_gb"], acc)

            results.append({
                "accept_ratio": acc,
                "title": title,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": comp_tokens,
                "wall_time_s": wall_time,
                "tok_s": tok_s,
                "e2e_tok_s": e2e_tok_s,
                "bw_str": bw_str
            })
            print(f" {tok_s:.2f} tok/s decode | {e2e_tok_s:.2f} tok/s e2e ({wall_time:.2f}s)")
        except Exception as e:
            print(f" ❌ Failed: {e}")

    if results:
        avg_tok_s = sum(r["tok_s"] for r in results) / len(results)
        avg_acc = sum(r.get("accept_ratio", 0.0) for r in results) / len(results)
        print("\n" + "=" * 60)
        print(f"📊 SUMMARY RESULTS: {model}")
        avg_e2e = sum(r["e2e_tok_s"] for r in results) / len(results)
        print(f"   Average Generation Speed : {avg_tok_s:.2f} tok/s (decode only)")
        print(f"   Peak / Min Generation    : {max(r['tok_s'] for r in results):.2f} / {min(r['tok_s'] for r in results):.2f} tok/s")
        print(f"   Average End-to-End Rate  : {avg_e2e:.2f} tok/s (incl. prefill + TTFT)")
        bw_val, bw_str = calc_bandwidth(avg_tok_s, profile["weights_gb"], profile["active_gb"], avg_acc)
        print(f"   Memory Bandwidth Saturation: {bw_str}")
        print("=" * 60)

def run_spec_bench(model: str):
    """Measures Speculative Decoding (MTP) draft acceptance and speedup."""
    print(f"\n🔮 Running Speculative Decoding (MTP) Benchmark: {model}")
    print("=" * 60)
    
    test_cases = [
        "Write a quicksort implementation in Rust with generic type constraints.",
        "Synthesize a SQL query with window functions to find 7-day rolling active user retention.",
        "Implement a binary search tree in C++ with recursive tree traversal."
    ]

    warmup(model)

    for idx, prompt in enumerate(test_cases, 1):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.0
        }
        res = post_chat(payload)
        wall_s = res.get("_wall_time_s", 0.001)
        usage = res.get("usage", {})
        comp_tokens = usage.get("completion_tokens", 0)
        tok_s = gen_rate(res, comp_tokens, wall_s)
        cov = accept_ratio_from(res)
        acc = draft_accept_rate(res)
        print(f"[{idx}/3] MTP: {tok_s:.2f} tok/s decode | {acc * 100:.1f}% of drafts accepted "
              f"| {cov * 100:.1f}% of output drafted ({comp_tokens} tokens in {wall_s:.2f}s)")

def run_context_sweep(model: str, sizes: List[int]):
    """Measures prompt ingestion (prefill) rate and TTFT across context lengths."""
    print(f"\n📥 Running Prompt Processing (PP) Context Sweep: {model}")
    print("=" * 60)

    warmup(model)

    # "token " tokenizes to ~1 token per repetition on Qwen; actual count is
    # reported alongside the target so the reader never has to trust the estimate.
    filler = "token "
    print(f"\n{'Target':>8}  {'Actual':>8}  {'PP tok/s':>10}  {'TTFT':>10}")
    print("-" * 44)
    for target in sizes:
        prompt = filler * max(1, target - 12)  # -12 leaves room for the chat template wrapper
        try:
            res = post_chat({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "temperature": 0.0
            }, timeout=600)
        except Exception as e:
            print(f"{target:>8}  {'—':>8}  {'failed':>10}  {str(e)[:20]}")
            continue

        t = res.get("timings") or {}
        actual = t.get("prompt_n") or res.get("usage", {}).get("prompt_tokens", 0)
        pp_s = t.get("prompt_per_second") or 0.0
        prompt_ms = t.get("prompt_ms") or (res.get("_wall_time_s", 0.0) * 1000)
        print(f"{target:>8}  {actual:>8}  {pp_s:>10.1f}  {prompt_ms:>8.0f} ms")
    print("=" * 60)

def run_cache_bench(model: str):
    """Benchmarks Cold vs Warm prefix cache hit latency."""
    print(f"\n⚡ Running Prompt Prefix Cache Benchmark: {model}")
    print("=" * 60)

    warmup(model)  # otherwise "cold" measures the model load, not a cache miss

    system_preamble = "You are an expert systems engineer. " * 150  # ~600 tokens
    prompt_cold = "What is process scheduling?"
    prompt_warm = "Explain memory paging."

    # Cold run
    print("▶ 1. Cold Prompt Prefill ...", end="", flush=True)
    t0 = time.perf_counter()
    post_chat({"model": model, "messages": [{"role": "system", "content": system_preamble}, {"role": "user", "content": prompt_cold}], "max_tokens": 10})
    cold_time = (time.perf_counter() - t0) * 1000
    print(f" {cold_time:.1f} ms")

    # Warm run (shared system prompt)
    print("▶ 2. Warm Cache Hit Prefill ...", end="", flush=True)
    t0 = time.perf_counter()
    post_chat({"model": model, "messages": [{"role": "system", "content": system_preamble}, {"role": "user", "content": prompt_warm}], "max_tokens": 10})
    warm_time = (time.perf_counter() - t0) * 1000
    print(f" {warm_time:.1f} ms")

    speedup = cold_time / max(warm_time, 0.1)
    print(f"\n✅ Cache Hit Speedup: {speedup:.2f}x faster TTFT on repeated prefix")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified LLM Performance & Feature Benchmark Suite.")
    parser.add_argument("model", default="main", nargs="?", help="Model ID to benchmark (default: main)")
    parser.add_argument("--spec", action="store_true", help="Run speculative decoding / MTP benchmark")
    parser.add_argument("--cache", action="store_true", help="Run prompt prefix cache hit/miss benchmark")
    parser.add_argument("--no-think", action="store_true", help="Disable thinking tokens for zero-shot mode")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max completion tokens per prompt")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--context-sweep", type=str, default=None,
                        help="Prompt-processing sweep, e.g. 512,2048,8192")
    args = parser.parse_args()

    if args.context_sweep:
        run_context_sweep(args.model, [int(x) for x in args.context_sweep.split(",") if x.strip()])
    elif args.spec:
        run_spec_bench(args.model)
    elif args.cache:
        run_cache_bench(args.model)
    else:
        run_standard_bench(args.model, args.max_tokens, args.temp, args.no_think)
