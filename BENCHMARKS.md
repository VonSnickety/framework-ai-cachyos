# Benchmarks: Local LLMs on AMD Strix Halo

Measured on a Framework Desktop: Ryzen AI Max+ 395, Radeon 8060S (40 CU RDNA 3.5, `gfx1151`), 128 GB unified LPDDR5X (~256 GB/s), CachyOS, ROCm 10.0 containers, routed through [llama-swap](https://github.com/mostlygeek/llama-swap). Setup overview: [README.md](README.md).

Measured between 2026-08-24 and 2026-10-07. Each section gives its own date. Container images change over time, so older sections may not match today's builds.

## Contents

1. [How to read these numbers](#1-how-to-read-these-numbers)
2. [Lineup summary](#2-lineup-summary)
3. [Engine showdown: 27B dense and Flash-Next](#3-engine-showdown-27b-dense-and-flash-next-2026-09-28)
4. [Per-model deep dives](#4-per-model-deep-dives)
5. [Speculative decoding strategy](#5-speculative-decoding-strategy)
6. [MoE compression: REAP pruning vs quantization](#6-moe-compression-reap-pruning-vs-quantization-2026-09-14)
7. [8-agent swarm stress test](#7-8-agent-swarm-stress-test)
8. [Quality evals (lm-eval)](#8-quality-evals-lm-eval)
9. [Tuning levers tested](#9-tuning-levers-tested)
10. [Batch geometry re-check (negative result)](#10-batch-geometry-re-check-negative-result-2026-08-24)
11. [Slot count vs per-slot context](#11-slot-count-vs-per-slot-context)
12. [Methodology corrections](#12-methodology-corrections-2026-08-24)
13. [ROCm environment variables](#13-rocm-environment-variables-2026-10-10)

---

## 1. How to read these numbers

- **Decode** is the pure generation rate from llama-server's `timings.predicted_per_second`. It excludes model load and prefill. Unless stated otherwise it is the average over three warm prompts, after one cold-start measurement.
- **Avg / Peak**: speculative decoding makes speed depend on the task. Copy-heavy prompts such as file edits and refactors run much faster than fresh writing, so both numbers are given.
- **TTFT** is time to first token. **PP** is prompt processing (prefill) speed.
- **Draft acceptance** (accepted ÷ drafted) shows how well the draft gate is tuned. **Coverage** (accepted ÷ emitted) shows how much of the output came for free. These are different numbers and are labelled separately.
- **Effective bandwidth** is discounted by draft coverage and capped at the 256 GB/s physical bus. See [§12](#12-methodology-corrections-2026-08-24) for why.
- Single runs per cell differ from run to run by a few percent. Differences that small are noise unless stated otherwise.

---

## 2. Lineup summary

Rows marked *retired* are kept for comparison and are no longer served.

| Model | Type | Quant | Slots × ctx (as tested) | Cold load | Decode avg / peak | Warm TTFT | PP (~2K) | Weights |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3.6-35B-A3B | MoE, 3B active | UD-Q8_K_XL | 2 × 262K | ~8.2 s | 56.8 / 195.6 tok/s | 185 ms | 1,386 tok/s | 36.4 GiB |
| Qwen3.6-35B-A3B REAP-48 | pruned MoE, 133 experts | iq3_s / q6_K | 8 × 128K | ~4.5 s | 53.2 tok/s (147.0 across 8) | 120 ms | 510 tok/s | 8.78 GiB |
| Swift-Qwen3.8-27B | dense | Q6_K | 1 × 32K | 6.53 s | 26.6 / 37.3 tok/s | 367 ms | 561 tok/s | 21.3 GiB |
| OrcaSAQ-2 27B | dense, abliterated | mixed-precision imatrix | 1 × 262K | 3.47 s | 35.3 / 49.3 tok/s | 286 ms | n/a | 15.7 GB |
| Qwen3.8-27B Uncensored | dense, abliterated | Q6_K | 1 × 262K | 6.53 s | 28.0 / 43.6 tok/s | 361 ms | n/a | 22.4 GB |
| Qwen3.8-Flash-Next | hybrid linear attention | UD-Q4_K_XL | 1 × 32K | 13.69 s | 43.5 / 57.5 tok/s | 224 ms | 1,467 tok/s (1,663 @ 32K) | 103.7 GiB |
| EmbeddingGemma 2 | embeddings | Q8_0 | 8 × 8K | resident | n/a | 7.7 ms | 15,067 tok/s (368 chunks/s) | 0.3 GiB |
| BGE-Reranker-v2-M3 | reranker | Q8_0 | 4 slots | resident | n/a | 13 ms | 250 pairs/s | 0.6 GiB |
| LFM2.5-8B-A1B | MoE, ~1B active | Q6_K | 2 × 32K | resident | ~120 tok/s | n/a | 2,295 tok/s (398-token prompt) | 6.5 GiB |
| *retired:* Qwen3.8-27B (base) | dense | UD-Q6_K_XL | 1 × 32K | 7.49 s | 23.9 / 31.6 tok/s | 308 ms | 561 tok/s | 23.6 GiB |
| *retired:* Qwen3.8-27B on Halogen | dense | FP8/NVFP4 | 1 × 262K | 18.02 s | 31.4 / 39.3 tok/s | 280 ms | 601 tok/s | 23.5 GiB |
| *retired:* Flash-Next on Halogen | hybrid linear attention | 5.53 bpw | 1 × 262K | 41.54 s | 42.8 / 47.5 tok/s | 759 ms | 970 tok/s | 115.5 GiB |
| *retired:* Swift-1.5-Flash-Next (llama.cpp) | hybrid linear attention | Q4_K_S | 1 × 32K | 71.18 s | 24.9 / 26.3 tok/s | 395 ms | 558 tok/s | 91.0 GiB |

---

## 3. Engine showdown: 27B dense and Flash-Next (2026-09-28)

Three engines on the same hardware: **[Gufo](https://github.com/gufo-org/gufo)**, **Halogen** ([27B](https://github.com/peonist-ai/halogen-server), [Flash-Next](https://github.com/peonist-ai/halogen-flash-server)) and stock **llama.cpp** (ROCm 10 llama-server). Every run started cold: all other models were evicted and baseline GTT memory was checked first.

### Cold and warm profile

| Tier | Model | Engine / quant | Cold load | Cold TTFT | Warm TTFT | Decode avg / peak | Draft acceptance | Effective bandwidth |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 27B | Swift-Qwen3.8-27B | Gufo DFlash2 (Q6_K) | **6.53 s** | 262 ms | 368 ms | 26.6 / 37.3 | 68.1% | 193.9 GB/s (76%) |
| | Qwen3.8-27B (base) | Gufo DFlash2 (UD-Q6_K_XL) | 7.49 s | 220 ms | 308 ms | 23.9 / 31.6 | 68.1% | 193.1 GB/s (75%) |
| | Qwen3.8-27B on Halogen | Halogen (FP8/NVFP4) | 18.02 s | 713 ms | 280 ms | **31.4 / 39.3** | 74.0% | 191.8 GB/s (75%) |
| Flash-Next | Qwen3.8-Flash-Next | Gufo MTP (UD-Q4_K_XL) | **13.69 s** | **344 ms** | **224 ms** | **43.5 / 57.5** | 59.4% | 93.7 GB/s (37%) |
| | Flash-Next on Halogen | Halogen (5.53 bpw) | 41.54 s | 1,205 ms | 759 ms | 42.8 / 47.5 | 36.7% | 148.9 GB/s (58%) |
| | Swift-1.5-Flash-Next | llama.cpp (Q4_K_S) | 71.18 s | 970 ms | 395 ms | 24.9 / 26.3 | 6.2% (n-gram) | 116.8 GB/s (46%) |

### Prefill as context grows

| Model | 512 | 2,048 | 8,192 | 16,384 | 32,000 |
|---|:---:|:---:|:---:|:---:|:---:|
| Qwen3.8-27B base (Gufo, dense attention) | 398 tok/s · 1.39 s | 561 · 3.72 s | 628 · 13.1 s | 613 · 26.8 s | n/a |
| Qwen3.8-27B on Halogen | 606 · 0.91 s | 601 · 3.48 s | 620 · 9.97 s | 548 · 15.0 s | 542 · 28.9 s |
| Qwen3.8-Flash-Next (Gufo, Gated DeltaNet linear attention) | 1,055 · 0.52 s | 1,467 · 1.42 s | 1,587 · 5.19 s | 1,628 · 10.1 s | **1,663 · 19.3 s** |
| Flash-Next on Halogen | 483 · 1.14 s | 970 · 1.63 s | 1,300 · 4.76 s | 1,380 · 5.97 s | 1,481 · 10.6 s |
| Swift-1.5-Flash-Next (llama.cpp) | 417 · 1.32 s | 558 · 3.67 s | 577 · 12.4 s | 550 · 16.8 s | 458 · 36.3 s |

Each cell shows the prefill rate and the TTFT. The dense 27B levels off around 600 tok/s. Flash-Next's linear attention keeps speeding up as the prompt grows, reaching about 3× the dense rate at 32K.

### Real task: thread-safe token-bucket rate limiter

Prompt: write a complete Python token-bucket rate limiter (burst capacity, fractional refill, monotonic clock) plus 3 multi-threaded contention tests using 10 threads. The model's code is written to a sandbox and run with `python3`.

| Model | Wall clock | Decode | Total tokens | Thinking share | Code tokens | Tests |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3.8-Flash-Next (Gufo MTP) | **42.6 s** | **52.7 tok/s** | 1,520 | 19.5% | 1,223 | pass |
| Swift-Qwen3.8-27B (Gufo) | 65.3 s | 39.8 tok/s | 2,599 | 52.7% | 1,229 | pass |
| Flash-Next on Halogen | 83.7 s | 46.8 tok/s | 2,066 | 12.1% | 1,817 | pass |
| Qwen3.8-27B base (Gufo) | 84.8 s | 35.1 tok/s | 2,771 | 28.9% | 1,971 | pass |
| Swift-1.5-Flash-Next (llama.cpp) | 106.6 s | 24.2 tok/s | 1,366 | 18.4% | 1,115 | pass |

### Decisions

- **27B tier: Swift-Qwen3.8-27B on Gufo.** Qwen3.8-27B on Halogen decodes faster on simple prompts (31.4 vs 26.6 tok/s), but Swift-Qwen3.8-27B finished the real task 19.5 s sooner than the base 27B because it spends fewer tokens thinking in circles. It also loads in 6.5 s against Halogen's 18 s, so on-demand swaps barely show.
- **Flash-Next tier: Qwen3.8-Flash-Next on Gufo.** It was fastest on the real task, had the highest peak decode and the lowest warm TTFT, loads ~104 GiB in 13.7 s, and has the fastest long-context prefill.

---

## 4. Per-model deep dives

### Qwen3.6-35B-A3B MoE (UD-Q8_K_XL, 36.4 GiB), 2026-08-24

| Metric | Value |
|---|---|
| Decode, average of 3 prompts | **56.8 tok/s** |
| Spread | 195.6 tok/s on refactors, 49.9 on reasoning |
| End-to-end rate (including prefill) | 55.0 tok/s |
| Effective bandwidth | 75.2 GB/s (29% of the bus; MoE, so only active parameters are read) |

**Prefill**

| Prompt tokens | Rate | Time |
|:---:|:---:|:---:|
| 552 | 1,007 tok/s | 548 ms |
| 2,046 | 1,386 tok/s | 1,476 ms |
| 7,172 | 1,326 tok/s | 5,410 ms |

**Prefix cache (`--cache-ram`, `-sps 0.30`)**

| Turn | Prompt | Prefill latency | Speedup |
|---|:---:|:---:|:---:|
| 1 (cold) | ~1,200 tok | 1,198 ms | 1.0× |
| 2 (same slot) | ~1,250 tok | 113 ms | **10.6×** |
| 3 | ~1,300 tok | 109 ms | 11.0× |

**Speculative decoding (MTP n=4, p-min 0.75, plus n-gram), tuned 2026-08-28**
- 87.1–99.8% of drafts accepted. 81.3–93.3% of emitted tokens were drafted.
- Refactor: **195.6 tok/s** (99.8% accepted, 3.37 s). File edit: 97.4 tok/s (87.1% accepted).

### Swift-Qwen3.8-27B (UkisAI Q6_K, 21.3 GiB), 2026-09-28

An on-policy-distilled fine-tune of Qwen3.8-27B, trained to cut redundant deliberation. It stays compatible with the DFlash2 block drafter.

| Metric | Value |
|---|---|
| Decode, average of 3 prompts | **26.9 tok/s** (+11.6% over the base Q6) |
| Spread | 38.2 tok/s on refactors, 20.9 on concurrency |
| End-to-end rate | 25.8 tok/s |
| Effective bandwidth | 196.5 GB/s (77% of the bus) |

**Against the base 27B on a practical task (IPv4 validation)**

| Model | Wall clock | Tokens | Reasoning characters |
|---|:---:|:---:|:---:|
| base Qwen3.8-27B | 23.8 s | 752 | 2,151 |
| Swift | **15.8 s** (1.5× faster) | **501** (−33%) | 1,426 |

**Earlier llama.cpp baseline for the base 27B (2026-08-28, retired path):** 15.4 tok/s average, 23.3 on file edits, 12.5 on long reasoning. Prefill was 313–381 tok/s. Two concurrent agents gave 29.3 tok/s combined (1.58×). Moving to Gufo with DFlash2 raised average decode to 24.1 tok/s on the same weights.

### OrcaSAQ-2 27B (mixed-precision imatrix, 15.7 GB), 2026-10-03

The same GGUF served four ways, with Qwen3.8-27B Uncensored (Q6_K) re-run in the same session for reference. One run per prompt, so gaps of a few tok/s are noise.

| Engine | Speculation | Avg decode | Peak / min | Warm TTFT | Cold load | GTT | Acceptance |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Gufo | DFlash2 | **35.3 tok/s** | 49.3 / 26.9 | 286 ms | 3.47 s | 23.6 GB | 69.5% |
| llama.cpp (build 11366) | `draft-dflash`, same drafter | 29.8 tok/s | 36.8 / 25.2 | 330 ms | 3.73 s | 29.9 GB | 64.7% |
| llama.cpp | embedded MTP head, n=4 | 22.4 tok/s | 30.5 / 18.2 | 384 ms | 4.78 s | 29.6 GB | 53.0% |
| llama.cpp | none | 13.4 tok/s | 13.5 / 13.4 | 301 ms | 3.60 s | 27.2 GB | n/a |
| *Qwen3.8-27B Uncensored (Gufo, Q6_K)* | DFlash2 | 28.0 tok/s | 43.6 / 19.9 | 361 ms | 6.53 s | 23.5 GB | 67.3% |

- **The drafter type matters most on llama.cpp.** DFlash2 beats the model's own MTP head by ~33%, and both beat no speculation.
- **With the same drafter, the engine still matters.** Gufo is ~19% faster than llama.cpp and uses ~6 GB less GTT.
- The smaller quant decodes ~26% faster than the Q6_K at similar acceptance, which fits fewer bytes being read per token. I haven't measured accuracy for either quant. The quant's published fidelity figures (+0.80% perplexity, 94.4% top-1 vs BF16) come from its author.

### Qwen3.8-Flash-Next (UD-Q4_K_XL, 103.7 GiB), 2026-09-28

A Gated DeltaNet + sparse attention hybrid, so most layers keep a fixed-size state instead of a growing KV cache.

| Metric | Value |
|---|---|
| Decode, average of 3 prompts | **43.8 tok/s** |
| Spread | 58.2 tok/s on refactors, 35.8 on concurrency |
| End-to-end rate | 42.0 tok/s |
| Effective bandwidth | 94.3 GB/s (37% of the bus; active parameters, MTP-discounted) |
| Cold load | 13.9 s for 104 GiB |

**Earlier llama.cpp path (IQ4_XS, 87.2 GiB, 2026-08-28, retired):** 20.8 tok/s with almost no spread between prompt types. Prefill was 324–437 tok/s, and `-b 4096 -ub 1024` gave +17.7% at 8K. Total memory use was 92 GiB. llama.cpp had no MTP support for this architecture at the time, which made it a clean test of n-gram drafting alone (see [§5](#5-speculative-decoding-strategy)).

### EmbeddingGemma 2 (Q8_0, 0.3 GiB), 2026-10-07

768-dim, mean pooling, can be truncated to 512/256/128 (Matryoshka). 8 slots × 8,192 tokens.

| Batch | Tokens | Wall time | Chunks/s | Tokens/s |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 41 | 7.7 ms | 130.5 | 5,350 |
| 16 | 656 | 46.0 ms | 347.6 | 14,250 |
| 64 | 2,624 | 174.2 ms | **367.5** | **15,067** |
| 128 | 5,248 | 352.2 ms | 363.4 | 14,901 |
| 256 | 10,496 | 706.9 ms | 362.1 | 14,847 |

The model it replaced, BGE-M3 (1024-dim), peaked at 240.1 chunks/s and took 12.5 ms for a single chunk. EmbeddingGemma 2 has **1.53× the peak rate** and **1.63× lower latency**, with 25% less vector storage.

### BGE-Reranker-v2-M3 (Q8_0, 0.6 GiB)

| Pairs | Latency | Pairs/s |
|:---:|:---:|:---:|
| 1 | 13.0 ms | 76.9 |
| 10 | 58.2 ms | 171.7 |
| 25 | 118.3 ms | 211.3 |
| 50 | 204.7 ms | 244.3 |
| 100 | 400.4 ms | **249.8** |

### LFM2.5-8B-A1B (Q6_K, 6.5 GiB), 2026-10-07

Pulls durable facts out of agent conversations, and writes session titles. All three sizes ran the coding agent's real extraction prompt (one fact per line, or `NO_FACTS`) over 8 fixed messages, at temperature 0 with a 2,048-token budget.

| Message | LFM2.5-230M | LFM2.5-2.6B | **LFM2.5-8B-A1B** |
|---|---|---|---|
| Small talk / one-off plan / weather | ✅ `NO_FACTS` ×3 | ✅ ×3 | ✅ ×3 |
| Stack fact (port + TTL) | ❌ refusal | ✅ | ✅ |
| Explicit instruction ("always use fish shell syntax") | ❌ refusal | ❌ `NO_FACTS` | ✅ |
| Fact buried in chatter | ❌ refusal | ⚠️ split into subject-less lines | ⚠️ value kept, subject dropped |
| Corrected value | ❌ refusal | ✅ latest value | ✅ latest value |
| German input | ❌ refusal | ⚠️ bare username | ✅ full sentence, language kept |
| **Score** | **3/8** | **6/8** | **8/8** |
| Latency per message | 0.02–0.1 s | 2–13 s | 1–4 s |

All three always open a `<think>` block. Forcing it shut with `--reasoning-budget 0` made the output worse (reasoning and stray tool-call tokens leaked into the answer), so thinking stays on and is split out with `reasoning_content`.

---

## 5. Speculative decoding strategy

`--spec-type` takes a comma-separated list, and drafters can be combined. Qwen3.6-35B-A3B runs `draft-mtp,ngram-mod`. This was chosen by A/B testing on 2026-08-24 (llama.cpp b10603) at production settings.

### One benchmark would have given the wrong answer

Three prompts with different amounts of repetition:
- **fresh-gen**: write a class from scratch. There is nothing in the context to copy.
- **file-edit**: return a ~60-line file with one method changed. The output is mostly a copy of the input.
- **refactor-rename**: rename two identifiers everywhere in the same file. It is repetitive, but every copied n-gram mismatches at the renamed token.

**Qwen3.6-35B-A3B, decode tok/s at one slot**

| Strategy | fresh-gen | file-edit | refactor-rename |
|---|---:|---:|---:|
| `draft-mtp` | 58.35 | 71.53 | **70.05** |
| `ngram-mod` alone | 43.35 | 100.80 | 43.64 |
| **`draft-mtp,ngram-mod`** | **58.57** | **105.57** | 68.84 |
| `ngram-map-k4v` alone | 43.42 | 53.65 | 45.29 |

**n-gram alone is a trap.** It is 41% faster on file edits, but **26% slower on fresh writing**, because it accepts 0% of its drafts when there is nothing to match. It is **38% slower on renames**, where acceptance falls to 19.5%. Testing only the copy-heavy case would have justified a change that slowed down two of the three real workloads. MTP covers fresh text, n-gram covers copying, and together they raise file-edit coverage from 69.6% to 82.6%.

### At production settings (Qwen3.6-35B-A3B 4 × 262K, 27B dense 2 × 262K at the time)

| | Metric | `draft-mtp` | `draft-mtp,ngram-mod` | Change |
|---|---|---:|---:|---:|
| **main** | GTT | 55.8 GiB | 55.8 GiB | none |
| | fresh-gen | 57.97 | 57.80 | −0.3% |
| | file-edit | 69.85 | **104.95** | **+50%** |
| | refactor-rename | 69.71 | 67.68 | −2.9% |
| | 4 streams combined | 83.12 | **116.85** | **+41%** |
| **27B dense** | GTT | 49.4 GiB | 49.4 GiB | none |
| | fresh-gen | 21.25 | 21.28 | +0.1% |
| | file-edit | 28.27 | **67.57** | **+139%** |
| | refactor-rename | 27.42 | **43.61** | **+59%** |
| | 2 streams combined | 45.51 | **123.26** | **+171%** |

The n-gram pools cost no extra memory at full context, and the gain grows under concurrency. The dense model gains most because it is bandwidth-bound: each skipped forward pass is worth more at 21 tok/s than at 57. Repeat runs agreed within 0.7%.

### Flash-Next on llama.cpp: n-gram only (2026-08-28, retired path)

| Strategy | fresh-gen | file-edit | refactor-rename | reasoning | Turn time |
|---|---:|---:|---:|---:|:---:|
| none | 21.37 | 21.01 | 21.05 | 21.10 | 27.6 s |
| `ngram-simple` | **22.69** | **26.09** | 73.89 | 22.50 | 9.0 s |
| `ngram-mod` | 21.99 | 25.40 | 71.02 | 22.00 | 9.3 s |
| `ngram-simple,ngram-mod` | 22.32 | 22.06 | **74.15** | **22.82** | **8.9 s** |
| `ngram-map-k` | 21.20 | 22.27 | 26.48 | 21.30 | 22.3 s |
| `ngram-map-k4v` | 21.29 | 22.00 | 26.69 | 21.30 | 22.1 s |
| `ngram-cache` | 21.52 | 19.70 | 17.63 | 19.94 | 32.7 s (+8 GB RAM) |

Unlike Qwen3.6-35B-A3B, the slower model (~21 tok/s) has enough time per token to hide the CPU cost of checking n-gram drafts. It lost nothing on fresh writing and gained ~3.5× on refactors.

---

## 6. MoE compression: REAP pruning vs quantization (2026-09-14)

**Question:** running 8 agents at once needs a lot of KV cache. Can a 35B MoE shrink enough to leave room for it without losing coding ability?

Compared on Qwen3.6-35B-A3B: [Crucible](https://github.com/peonist-ai/crucible) REAP expert pruning against three quantization levels. Each tier ran 15 HumanEval problems (0–14) with a 4,096-token limit in an isolated sandbox.

| Tier | Checkpoint | Experts | Size | BPW | HumanEval (15) | Decode | Prefill | Wall time |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline | UD-Q8_K_XL | 256 | 36.4 GiB | 8.81 | **13/15** | 61.6 tok/s | 530 tok/s | 6m 47s |
| Q4 | Q4_K_M | 256 | 20.2 GiB | 4.89 | **13/15** | **80.1 tok/s** | 540 tok/s | **5m 08s** |
| Crucible 16G | 16G-Q3K | 256 | 15.7 GiB | 3.81 | 12/15 | 74.8 tok/s | 525 tok/s | 5m 43s |
| Crucible 12G | 12G6-Q2K | 256 | 12.6 GiB | 3.05 | 11/15 | 72.3 tok/s | **78 tok/s** | 7m 32s |
| **REAP-48** | REAP-48-v2 | **133** | **8.78 GiB** | 2.12 | **13/15** | 53.2 tok/s | 510 tok/s | 8m 23s |

<details>
<summary>Problem-by-problem results</summary>

| Problem | Q8 | Q4_K_M | 16G-Q3K | 12G-Q2K | REAP-48 |
|---|:---:|:---:|:---:|:---:|:---:|
| HumanEval/0 | FAIL | PASS | FAIL | PASS | FAIL (syntax) |
| HumanEval/1 | PASS | PASS | PASS | PASS | PASS |
| HumanEval/2 | PASS | PASS | FAIL | PASS | PASS |
| HumanEval/3 | PASS | PASS | PASS | PASS | PASS |
| HumanEval/4 | PASS | PASS | PASS | PASS | PASS |
| HumanEval/5 | FAIL | PASS | FAIL | FAIL | PASS |
| HumanEval/6 | PASS | PASS | PASS | FAIL | PASS |
| HumanEval/7 | PASS | PASS | PASS | PASS | PASS |
| HumanEval/8 | PASS | FAIL | PASS | PASS | FAIL |
| HumanEval/9 | PASS | PASS | PASS | PASS | PASS |
| HumanEval/10 | PASS | FAIL | PASS | FAIL | PASS (3,897 tokens) |
| HumanEval/11 | PASS | PASS | PASS | PASS | PASS |
| HumanEval/12 | PASS | PASS | PASS | FAIL | PASS |
| HumanEval/13 | PASS | PASS | PASS | PASS | PASS |
| HumanEval/14 | PASS | PASS | PASS | PASS | PASS |

</details>

**Findings**
- **Pruning beat aggressive quantization.** Removing 123 of 256 experts took out 15.48B parameters (35B down to 19.5B total), while active compute stayed at ~3B. With `iq3_s` routed experts and `q6_K` shared experts and attention, it matched the baseline's 13/15 at 8.78 GiB. The uniform 2-bit quant scored 11/15 at 12.6 GiB.
- **Q2_K stalls prefill on ROCm.** It ran at 78 tok/s against ~530 for every other tier. REAP at `iq3_s` avoids this.
- **Q4_K_M is the fastest decoder** at 80.1 tok/s (+30% over Q8) with no score loss.
- 15 problems is a small sample. One problem is about 7 percentage points, so 13/15 versus 12/15 is suggestive, not conclusive.

---

## 7. 8-agent swarm stress test

Qwen3.6-35B-A3B REAP-48 with 8 slots and 1,048,576 tokens of total context. Eight agents ran at the same time, each reading a different ~90K-token codebase with a unique needle string hidden at a different depth (11% to 98%).

| Agent | Needle depth | Result | Prompt tokens | Output tokens | Latency |
|---|:---:|:---:|:---:|:---:|:---:|
| 1 | 11% | pass | 90,288 | 322 | 919.6 s |
| 2 | 24% | pass | 90,286 | 322 | 920.8 s |
| 3 | 40% | pass | 90,287 | 325 | 910.9 s |
| 4 | 53% | pass | 90,286 | 331 | 916.9 s |
| 5 | 67% | pass | 90,285 | 303 | 903.2 s |
| 6 | 80% | pass | 90,287 | 308 | 916.8 s |
| 7 | 91% | pass | 90,285 | 310 | 911.9 s |
| 8 | 98% | pass | 90,286 | 313 | 921.3 s |

| Aggregate | Value |
|---|---|
| Tokens in flight | **722,290** prompt + 2,534 generated |
| Needles found | **8/8** |
| Combined throughput | 786.8 tok/s over 921 s |
| Short-prompt generation, 8 streams | 147.0 tok/s combined (18.4 per stream) |
| Memory | **24.6 GB** of 125.1 GB GTT; 100.5 GB host RAM still free |
| Junction temperature / board power | 53 °C / 29–38 W |

**Why 1M tokens of context needs only ~10.7 GiB:** Qwen3.6 has 40 layers, but only 10 use full attention (`full_attention_interval = 4`). The other 30 are linear recurrent layers with a fixed-size state. At `q8_0` KV that works out to ~10.24 KB per token across the whole model.

---

## 8. Quality evals (lm-eval)

[EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) through `local-chat-completions` against llama-swap, with the chat template applied and unconstrained `<think>` traces. Competing models were evicted before each run.

| Model | Type | Task | N | Strict match | Flexible extract | Stderr |
|---|---|---|:---:|:---:|:---:|:---:|
| Qwen3.6-35B-A3B | MoE | GSM8K 5-shot CoT | 200 | **97.0%** (194/200) | 92.5% | ±0.012 |
| Qwen3.8-27B (base) | dense | GSM8K 5-shot CoT | 25 | 88.0% (22/25) | **96.0%** (24/25) | ±0.040 |

---

## 9. Tuning levers tested

| Area | Lever | Tested | Result | Kept |
|---|---|---|---|---|
| Speculation | Draft sampling backend | GPU vs CPU | CPU ~3% faster on dense. GPU +15% on MoE file edits (68.7 → 79.2 tok/s) by removing host–GPU syncs | CPU for dense, GPU for MoE |
| Speculation | `ngram-mod` window on MoE | 24/48/64 vs 16/32/48 vs 24/48/96 | Short window −13.7% on edits (truncated drafts). Long window overshot (61% acceptance) | Default 24/48/64 |
| Speculation | Temperature | 0.6 vs 0.2 vs 0.0 on Qwen3.6-35B-A3B | Coverage 55–74% at 0.6, 88–96% at 0.0. Refactors reach 135–183 tok/s | Greedy is fastest for code |
| Speculation | Draft depth on Qwen3.6-35B-A3B | n=4 p=0.75 vs n=3 p=0.8 vs n=5 p=0.75 | n=4: 195.6 tok/s refactor, 97.4 edit. n=5 peaked at 200.4 but truncated multi-line edits | n=4, p=0.75 |
| Speculation | Draft depth on 27B dense (llama.cpp) | n=6 p=0.6 vs n=4 p=0.5 vs n=6 p=0.4 | File edits 15.8 → 23.3 tok/s (+47.8%), turn time 36.6 → 25.3 s | n=6, p=0.6 |
| Speculation | Tree branching | `--spec-draft-p-split` 0.0 vs 0.25 | 15.07 vs 15.12 tok/s, no difference | Linear (default) |
| Speculation | DFlash2 GGUF on llama.cpp (Aug 2026) | Q8_0 drafter | Failed to load: tensor count mismatch with the parser at the time. Later builds load it (see OrcaSAQ-2 27B) | Superseded |
| Batching | `-b 4096 -ub 1024` on MoE | 512 → 4096 | Prefill 1,133 → 1,386 tok/s (+22%). `-ub 4096` slower at long context | `-b 4096 -ub 1024` |
| Batching | `-b 8192 -ub 8192` on EmbeddingGemma 2 | batch 1 → 256 | Embedding throughput levels off around batch 64 | `-b 8192 -ub 8192` |
| Context | Flash attention, 1K → 64K on 27B | 4 depths | Decode fell only 23% by 32K (17.3 → 13.3 tok/s) | `--flash-attn on` |
| Scheduling | `--prio 2` | 0 vs 2 | Steadier timings, background tasks don't preempt | `--prio 2` |
| Threading | `-t 8 -tb 16` vs `-t 16 -tb 16` | | Same speed (15.09 vs 15.06), half the CPU power, 8 cores free | `-t 8 -tb 16` |
| KV cache | f16 vs q8_0 on 27B | | f16 14.9% faster on fresh text. q8_0 halves memory, which makes 262K+ context fit | q8_0 |
| Prefix cache | `--cache-ram` 16 GB vs 32 GB | | Turn-2 prefill 1,198 → 113 ms. 16 GB was enough | 16 GB tested; 8 GB in use now |
| Loading | `--no-mmap` vs mmap | | Same TTFT on Gen4 NVMe; mmap boots containers faster | mmap |

---

## 10. Batch geometry re-check (negative result, 2026-08-24)

`-b 4096 -ub 1024` was tuned under older settings, so it was re-tested on b10603 with Qwen3.6-35B-A3B at 4 × 262K and `draft-mtp,ngram-mod`. Prefill rate by actual prompt length:

| Geometry | 552 | 2,046 | 8,180 | 32,756 | Decode | GTT |
|---|---:|---:|---:|---:|---:|---:|
| **`-b 4096 -ub 1024`** (kept) | 997 | 1,376 | 1,303 | 929 | 55.01 | **55.8 GiB** |
| `-b 4096 -ub 2048` | 992 | 1,386 | **1,326** | 927 | 57.06 | 57.0 GiB |
| `-b 8192 -ub 2048` | 984 | **1,394** | 1,310 | 923 | 57.01 | 57.0 GiB |
| `-b 8192 -ub 4096` | 996 | **1,407** | 1,209 | 878 | 56.45 | 59.4 GiB |

- The first three rows are within 1–2%. With one run per cell, that is noise. The honest conclusion is "no change justified", not "provably optimal".
- **`-ub 4096` is worse:** −7% at 8K and −5% at 32K, and it costs +3.6 GiB.
- **The prefill drop from 2K to 32K is not batch starvation.** Every geometry falls by the same proportion, so it comes from attention cost growing with sequence length. Don't try to fix it with batch flags.

---

## 11. Slot count vs per-slot context

`--ctx-size` is the total that llama.cpp splits across slots. If you change `-np` without scaling `--ctx-size`, each request silently gets a different context size. Check `n_ctx_slot` in the startup log, not the flag you passed.

This caught Qwen3.6-35B-A3B once. It went from 2 slots to 8 without touching `--ctx-size 524288`, which cut per-slot context from 262K to 65K while the docs still said 262K.

| Setting | `-np` | `--ctx-size` | Per slot | Peak combined decode | GTT |
|---|:---:|:---:|:---:|:---:|:---:|
| 8 slots | 8 | 524,288 | 65,536 | 73.75 tok/s | ~50 GB |
| 4 slots | 4 | 1,048,576 | 262,144 | 70.23 tok/s | ~55 GB |
| **now** | **2** | **524,288** | **262,144** | n/a | n/a |

Going from 8 to 4 slots cost 4.8% peak combined throughput and ~5 GB, in return for 4× the context per slot. Single-stream decode didn't change. Qwen3.6-35B-A3B now runs 2 slots; parallel agents go to the REAP-48 model.

The same mistake hit EmbeddingGemma 2: `--ctx-size 8192` with 8 slots left 1,024 tokens per slot, and any longer chunk failed with `400 input is larger than the max context size`.

---

## 12. Methodology corrections (2026-08-24)

Three corrections were applied on 2026-08-24, and every figure above was measured after them. Earlier numbers I had published were not comparable, and two of them were physically impossible.

1. **Warmup.** Tokens per second used to be `completion_tokens / wall_time`, and the first prompt's wall time included llama-swap loading the model. On Qwen3.6-35B-A3B, prompt 1 scored 11.6 tok/s against ~52–54 for prompts 2 and 3, which dragged the average to 39. An unscored warmup request now goes first. In concurrency sweeps the load had been charged to the 1-stream baseline, which inflated every scaling ratio (published 1.43× at 8 streams; actual 1.34×).
2. **Decode rate vs end-to-end rate.** Tokens divided by total request time includes prefill and TTFT, so it is not a generation rate. The headline now comes from `timings.predicted_per_second`, and the end-to-end rate is reported separately.
3. **Effective bandwidth.** The old formula, `tok/s × weight size`, assumes one full pass over the weights per token. With speculative decoding, several tokens come out of one forward pass. Results exceeded the 256 GB/s physical bus (one model was published at 411 GB/s). Bandwidth is now discounted by measured draft coverage and uses decimal GB to match the bus figure. Anything still over the roofline gets flagged instead of published.

**Acceptance vs coverage:** Qwen3.6-35B-A3B measures 96.4% draft acceptance but 54% coverage. Quoting the first as if it were the second made MTP look near-perfect.

**Reproducing:** after any requant, update the model weight sizes used for the bandwidth maths (`stat -c %s` on the GGUF, in decimal GB). A stale entry silently corrupts every bandwidth figure.

---

## 13. ROCm environment variables (2026-10-10)

Two variables are often recommended for Strix Halo: `HSA_ENABLE_SDMA=0` and `ROCBLAS_USE_HIPBLASLT=1`. I tested all four on/off combinations with `llama-bench` on the two production llama.cpp builds, using the same flags as the servers (`-fa on -ctk q8_0 -ctv q8_0 -b 4096 -ub 1024 -t 8`, no speculative decoding). Each combination ran twice, the second pass in reverse order, 3 repetitions per cell.

**Qwen3.6-35B-A3B** (UD-Q8_K_XL, llama.cpp build 11583), tok/s

| Variables | pp512 | pp2048 | pp8192 | tg128 |
|---|---:|---:|---:|---:|
| none | 1,898 | 2,302 | 2,163 | 56.5 |
| `HSA_ENABLE_SDMA=0` | 1,893 (−0.3%) | 2,318 (+0.7%) | 2,161 (−0.1%) | 56.2 |
| `ROCBLAS_USE_HIPBLASLT=1` | 1,909 (+0.5%) | **2,406 (+4.5%)** | **2,242 (+3.6%)** | 56.0 |
| both | 1,904 (+0.3%) | **2,393 (+3.9%)** | **2,240 (+3.6%)** | 55.8 |

The first run of the session (the "none" row, pass 1) came in at 38.4 tok/s decode and 1,953 tok/s at 8K, a cold-start outlier. The table uses its pass-2 run. Every other cell's two passes agreed within ~1%.

**Qwen3.6-35B-A3B REAP-48** (iq3_s/q6_K, llama.cpp build 11458), tok/s

| Variables | pp512 | pp2048 | pp8192 | tg128 |
|---|---:|---:|---:|---:|
| none | 1,221 | 1,524 | 1,502 | 58.6 |
| `HSA_ENABLE_SDMA=0` | 1,214 (−0.6%) | 1,524 (0.0%) | 1,500 (−0.2%) | 56.8 (−3.1%) |
| `ROCBLAS_USE_HIPBLASLT=1` | 1,246 (+2.1%) | **1,569 (+3.0%)** | **1,558 (+3.7%)** | 58.5 |
| both | 1,207 (−1.1%) | **1,575 (+3.4%)** | **1,565 (+4.2%)** | 57.0 (−2.7%) |

**Model load** (container start + 36 GiB model load + 16-token prompt, Qwen3.6-35B-A3B, 3 runs each, weights in page cache)

| Variables | Mean |
|---|---:|
| none | 3.31 s |
| `HSA_ENABLE_SDMA=0` | 3.75 s |
| `ROCBLAS_USE_HIPBLASLT=1` | 3.24 s |
| both | 3.73 s |

**Findings**
- **`ROCBLAS_USE_HIPBLASLT=1` helps prefill a little.** It gave +3–5% at 2K–8K tokens on both models, in both passes. It made no difference at 512 tokens or to decode.
- **`HSA_ENABLE_SDMA=0` brings no speed benefit.** Prefill and decode were unchanged. Model load was ~0.4 s slower, because it turns off the DMA engine for host-to-GPU copies. On REAP-48, decode was 3% lower in the second pass only, which is too inconsistent to call.
- `HSA_ENABLE_SDMA=0` was originally a fix for SDMA hangs. A short benchmark can't show whether the current kernel and ROCm still need it.
- A small LFM2.5-8B-A1B server stayed loaded but idle during most runs, the same in every combination.

---

### Tools used

[llama.cpp](https://github.com/ggerganov/llama.cpp) · [Gufo](https://github.com/gufo-org/gufo) · [Halogen](https://github.com/peonist-ai/halogen-server) · [Halogen Flash](https://github.com/peonist-ai/halogen-flash-server) · [llama-swap](https://github.com/mostlygeek/llama-swap) · [AMD Strix Halo Toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) · [Crucible](https://github.com/peonist-ai/crucible) · [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) · [EvalPlus](https://github.com/evalplus/evalplus)
