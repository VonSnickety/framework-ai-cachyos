# Hardware Benchmarks & Performance Suite (AMD Strix Halo)

**Platform:** Framework Desktop — AMD Ryzen AI Max+ 395, Radeon 8060S (40 CU RDNA 3.5, target `gfx1151`), 128 GB Unified LPDDR5X-8000/8533 memory (~256 GB/s bandwidth), running CachyOS Linux.  
**Backend:** `docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-10.0` (ROCm 10.0 / llama.cpp b10603), `llama-swap` on-demand router.  
**Benchmark Date:** 2026-08-24 & 2026-09-04. Measured on production geometry with combined MTP + n-gram speculative decoding.

---

## 1. Executive Summary: Active Model Lineup

| Model ID | Architecture | Quant | Ctx / Slot | Decode | Concurrent Max | PP Rate (~2K) | Weights | Role & Profile |
|---|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **`main`** | **Qwen 3.6-35B-A3B MoE** | UD-Q8_K_XL | 2 × 262K | **56.8–195.6 tok/s** (200.4 pk) | **88.6 tok/s** (×2) | **1,386 tok/s** | 36.4 GiB | Multi-agent coder, daily workhorse & vision OCR (MTP n=4, p=0.75) |
| **`think`** | **Qwen 3.8-27B Dense** | UD-Q6_K_XL (v3.0) | 1 × 262K | **15.1–23.3 tok/s** | — *(1 slot)* | **381 tok/s** | 23.6 GiB | Deep architectural reasoning & planning (MTP n=6, p=0.6) |
| **`flash`** | **Qwen 3.8-Flash-Next** | UD-IQ4_XS | 1 × 262K | **22.2–78.4 tok/s** (80.1 pk) | — | **436 tok/s** | 87.2 GiB | Qwen4 Preview MoE + 51B CPU N-gram Table (n=8, m=64 spec, ub=1024) |
| **`tiny`** | **Qwen 3.5-4B** | Q8_0 | 2 × 65K | **150+ tok/s** | — | **2,400+ tok/s** | 4.1 GiB | Fast zero-eviction micro-tier, commits & quick triage |
| **`embed`** | **BAAI/bge-m3** | Q8_0 | 8K | — | — | **14,883 tok/s** (240 ch/s) | 0.6 GiB | Multilingual dense/sparse embeddings, zero-eviction RAG |
| **`rerank`** | **BAAI/bge-reranker-v2-m3** | Q8_0 | 8K | — | — | **249.8 pairs/s** (400ms / 100 docs) | 0.6 GiB | Cross-encoder relevance scoring, zero-eviction RAG |

"Decode" is the pure token-generation rate from `llama-server`'s `timings.predicted_per_second`,
averaged over three prompts after an unscored warmup. It excludes model load and prefill —
end-to-end rates including those are reported per-model in §3.

## 2. Benchmark Suite Tools & CLI Usage

The repository includes a consolidated suite of benchmark tools:

| Tool | Category | Key Capabilities & Metrics |
|---|:---:|---|
| **[`bench.py`](bench.py)** | Throughput & Latency | Serial tok/s, prompt ingestion (PP), TTFT (ms), 256 GB/s memory bandwidth, speculative decoding (`--spec`), and prefix caching (`--cache`). |
| **[`eval_reasoning.py`](eval_reasoning.py)** | Reasoning & Logic (lm-eval) | Official EleutherAI `lm-evaluation-harness` runner (GSM8K, Minerva Math, ARC-Challenge, IFEval) testing chain-of-thought reasoning, instruction constraints, and exact-match extraction. |

### Running Benchmarks via `aictl` CLI or Python:

You can run benchmarks directly via `aictl bench` or via standalone Python scripts:

```bash
# === 1. Official Deep Reasoning & Math (EleutherAI lm-eval: GSM8K, MATH, IFEval) ===
aictl bench reason think --tasks gsm8k       # or: python3 benchmarks/eval_reasoning.py think --tasks gsm8k
aictl bench reason think --tasks ifeval      # Instruction following evaluation
aictl bench reason think --limit 25          # Quick test on first 25 problems
aictl bench reason main --tasks gsm8k --num-concurrent 2 # 2-slot parallel sweep on MoE

# === 2. Hardware Speed, Memory Bandwidth & Latency (bench.py) ===
aictl bench think                            # or: python3 benchmarks/bench.py think
aictl bench main --spec                      # Speculative decoding (MTP) breakdown
aictl bench main --cache                     # Prompt prefix cache hit/miss speedup

# === 3. Low-Level Containerized llama-bench ===
aictl bench raw main                         # Isolated ROCm hardware parity check
```

---

## 3. Quality & Functional Capability Benchmarks (EleutherAI `lm-eval`)

Official model capability evaluated using EleutherAI's `lm-evaluation-harness` (`lm_eval[api]`) across canonical benchmarks: **GSM8K** (Grade School Math 8K with 5-shot CoT), **Minerva Math**, **ARC-Challenge**, and **IFEval** (Instruction Following).

* **Backend:** Native `local-chat-completions` querying `llama-swap` with automatic chat template handling (`--apply_chat_template`).
* **Evaluation Criteria:** Strict exact-match string and numerical value parsing (`strict-match` and `flexible-extract` regex).
* **Traces & Extraction:** Full unconstrained `<think>` reasoning traces with strict model end-of-turn delimiter enforcement (`<|im_end|>`, `<|endoftext|>`).
* **Memory Isolation:** The benchmark runner automatically evicts competing dynamic models (`main` or `think`) prior to evaluation, guaranteeing zero memory contention and dedicated memory bandwidth on the Strix Halo APU.
* **Run Logs:** Complete JSON metrics, prompt templates, and per-sample logs are automatically stored in [`benchmarks/results/reasoning/`](results/reasoning/).

| Model ID | Architecture | Tasks / Dataset | Problems ($N$) | Strict Exact Match | Flexible Extract | Stderr | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`main`** | Qwen 3.6-35B MoE | GSM8K (5-shot CoT) | 200 | **97.00%** (194/200) | **92.50%** | ± 0.0121 | Verified |
| **`think`** | Qwen 3.8-27B Dense | GSM8K (5-shot CoT) | 25 | **88.00%** (22/25) | **96.00%** (24/25) | ± 0.0400 | Verified |

```bash
# Run full GSM8K benchmark
aictl bench reason think --tasks gsm8k

# Run strict instruction following evaluation
aictl bench reason think --tasks ifeval

# Quick test (first 25 problems)
aictl bench reason think --limit 25

# Concurrent 2-slot evaluation on MoE
aictl bench reason main --tasks gsm8k --num-concurrent 2
```

---

## 4. Hardware Telemetry & Throughput Benchmarks

All figures measured 2026-08-24 on the corrected suite (warmup applied, decode rate read from
`timings`, bandwidth MTP-discounted). Prompt-token counts are the tokenizer's actual counts, not
the requested targets.

### A. `main` — Qwen 3.6-35B-A3B MoE (UD-Q8_K_XL, 36.4 GiB weights)

#### Generation
| Metric | Value |
|---|---|
| Decode rate (avg of 3 prompts) | **56.82 tok/s** |
| Decode spread (peak / min) | 195.64 (refactor) / 49.89 (reasoning) tok/s |
| End-to-end rate (incl. prefill + TTFT) | 54.95 tok/s |
| Effective memory bandwidth | **75.2 GB/s** (29.4% of the 256 GB/s bus, MoE active params, MTP-discounted) |

#### Prompt Processing (PP) Context Scaling
| Prompt Tokens | Ingestion Speed | Prefill Time |
|:---:|:---:|:---:|
| **552** | **1,007 tok/s** | 548 ms |
| **2,046** | **1,386 tok/s** | 1,476 ms |
| **7,172** | **1,326 tok/s** | 5,410 ms |

#### Multi-Turn Prefix Cache & Slot Affinity (`--cache-ram 16GB`, `-sps 0.30`)
| Turn / Request Type | Prompt Size | Prefill Latency | Effective PP Rate | Cache Speedup |
|---|:---:|:---:|:---:|:---:|
| **Turn 1 (Cold System Load)** | ~1,200 tok | **1,197.7 ms** | 1,199.8 tok/s | 1.00× (Baseline) |
| **Turn 2 (Warm Slot Reuse)**   | ~1,250 tok | **113.3 ms** | 141.2 tok/s | **10.6× faster (90.5% TTFT reduction)** |
| **Turn 3 (Continuous Session)** | ~1,300 tok | **108.5 ms** | 129.0 tok/s | **11.0× faster (90.9% TTFT reduction)** |

#### Speculative Decoding (Tuned 2026-08-28: MTP n=4, p-min=0.75 + N-gram)
* **87.1–99.8% of issued drafts accepted**; **81.3–93.3% of emitted tokens drafted**.
* Code Refactoring hit **195.64 tok/s (99.8% accept rate in 3.37s)**; Structured File Editing reached **97.36 tok/s (87.1% accept rate in 6.30s)**.

---

### B. `think` — Qwen 3.8-27B Dense (UD-Q6_K_XL Dynamic v3.0, 23.6 GiB weights)

#### Generation
| Metric | Value |
|---|---|
| Decode rate (avg of 3 prompts) | **15.39 tok/s** |
| Decode spread (peak / min) | 23.30 (edit) / 12.48 (reasoning) tok/s |
| End-to-end rate | 15.09 tok/s |
| Effective memory bandwidth | **171.4 GB/s** (67.0% of the 256 GB/s bus, full weights, MTP-discounted) |

Decode rate is strongly workload-dependent on this model: short code synthesis hits 19.4 tok/s,
structured file editing hits **23.3 tok/s**, and long reasoning-heavy answers settle around 12.5–13.9 tok/s.

#### Concurrency & Prefill
| Concurrency | Aggregate Throughput | Scaling |
|:---:|:---:|:---:|
| **1 Agent** | 18.56 tok/s | 1.00× |
| **2 Agents** | 29.32 tok/s | **1.58×** |

| Prompt Tokens | Ingestion Speed | Prefill Time |
|:---:|:---:|:---:|
| **552** | **313 tok/s** | 1,762 ms |
| **2,046** | **381 tok/s** | 5,364 ms |
| **7,172** | **363 tok/s** | 19,742 ms |

#### Speculative Decoding (Tuned 2026-08-28: MTP n=6, p-min=0.6 + N-gram)
* **66.1–84.1% of issued drafts accepted**; **56.3–72.0% of emitted tokens drafted**.
* Deep draft horizon `n=6` with confidence gate `p-min=0.6` unlocked **23.3 tok/s on file edits (+47.8%)** while maintaining high accuracy.

---

### C. `flash` — Qwen 3.8-Flash-Next (UD-IQ4_XS, 87.2 GiB weights + 51B N-gram Table)

Qwen4 preview architecture featuring Gated DeltaNet linear state recurrent updates, Qwen Sparse Attention (QSA), and a 51B parameter N-gram lookup table pinned to CPU host RAM (`-ot "per_layer_token_embd=CPU"`).

#### Generation & Memory
| Metric | Value |
|---|---|
| Decode rate (avg of 3 prompts) | **20.79 tok/s** |
| Decode spread (peak / min) | 21.06 / 20.63 tok/s — remarkably flat across prompt complexities |
| End-to-end rate (incl. prefill + TTFT) | 20.28 tok/s |
| Effective memory bandwidth | **93.5 GB/s** (36.5% of the 256 GB/s bus, active compute parameters) |
| System Memory Footprint | **92 GiB total used / 32 GiB free RAM** (zero OOM risk on 128 GB APU) |

#### Speculative Decoding Engine Comparison (Measured 2026-08-28)
Evaluated across `fresh-gen` (350 tokens), `file-edit` (550 tokens), `refactor-rename` (550 tokens), and `reasoning-arch` (450 tokens) at `-np 1` / 262K context:

| Spec Engine | Fresh Gen | File Edit | Refactor / Rename | Reasoning | Wall Time | Notes & Assessment |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **`Baseline (none)`** | **21.37** | **21.01** | **21.05** | **21.10** | 27.6s | Pure autoregressive baseline |
| **`ngram-simple (n=8, m=64)`** | **22.22** (+4.0%) | **32.24** (+53.5%) | **78.39** (+271.5%) | **23.10** (+9.5%) | **8.69s** | **Locked in Stack:** 3.71× faster on refactors, doubles edit draft coverage |

*Note: 8 alternative speculation strategies (`ngram-mod`, `ngram-map-k`, `ngram-cache`) were evaluated; variants either suffered draft regressions or incurred unacceptable host RAM cache bloat. See [`LOG.md`](../LOG.md) Wave 8 for the full exploratory comparison matrix.*

#### Prompt Processing (PP) Context Scaling (Tuned 2026-08-28: -b 4096 -ub 1024)
| Prompt Tokens | Ingestion Speed | Prefill Time |
|:---:|:---:|:---:|
| **358** | **323.5 tok/s** | 1,107 ms |
| **1,394** | **415.6 tok/s** | 3,354 ms |
| **5,173** | **436.5 tok/s** | 11,851 ms |

Upgrading batch geometry to `-b 4096 -ub 1024` lifted 8K prefill ingestion rate by **+17.7%** (from 370.7 to 436.5 tok/s).

---

### D. `revil2` — Qwen 3.8-27B Dense LoRA (RE-Specialist, 29.0 GiB weights)

Specialized Autonomous Reverse Engineering model trained on 1,500 49-tool ChatML trajectories driving Ghidra FastMCP and writing Python/Z3 constraint solvers.

| Metric | Value |
|---|---|
| Decode rate (avg of 3 prompts) | **15.74 tok/s** (base synthesis) |
| File Edit & Decompilation Decode | **81.56 tok/s** (97.4% MTP draft accept, 94.0% coverage, **8.36s turn time**) |
| Refactor / Constraint Solving | **29.34 tok/s** (86.5% accept, 81.6% coverage) |
| Prompt Processing (2K) | **385.2 tok/s** |
| Memory Footprint | **29.0 GiB (Q8_0)** |
| Speculative Decoding | Tuned MTP (n=6, p-min=0.6) + N-gram modulation |

---

### E. `vision` — Qwen 2.5-VL-32B Instruct (Multimodal Vision, 34.3 GiB weights)

Flagship vision-language model with full `mmproj-model-f16.gguf` projector for document OCR, UI wireframes, and architecture diagram reversing.

| Metric | Value |
|---|---|
| Pure Text Decode | **6.27 tok/s** (34.3 GiB dense weights in Q8_0) |
| 720p Image Ingestion & Diagram OCR | **215.2 tok/s (5.69s TTFT)** |
| Diagram OCR Output Decode | **6.24 tok/s** |
| Speculative Evaluation | Continuous visual embeddings $\to$ pure autoregressive decode |

---

### F. `embed` — BAAI/bge-m3 (Multilingual Vector Embeddings, 0.6 GiB weights)

Resident dense/sparse embedding engine for codebase semantic indexing and local RAG retrieval.

| Batch Size | Total Tokens | Wall Time | Ingestion Throughput | Effective Rate |
|:---:|:---:|:---:|:---:|:---:|
| **1 chunk** | 45 | 12.5 ms | **79.9 chunks/sec** | 3,594 tok/s |
| **16 chunks** | 727 | 119.2 ms | **134.2 chunks/sec** | 6,099 tok/s |
| **64 chunks** | 2,957 | 307.5 ms | **208.2 chunks/sec** | 9,617 tok/s |
| **128 chunks** | 5,965 | 533.1 ms | **240.1 chunks/sec** | **11,190 tok/s** |
| **256 chunks** | 12,002 | 1,104.3 ms | **231.8 chunks/sec** | **10,868 tok/s** |

---

### G. `rerank` — BAAI/bge-reranker-v2-m3 (Cross-Encoder Document Reranker, 0.6 GiB weights)

Resident cross-encoder model scoring query-document relevance pairs for hybrid local search.

| Batch Size | Latency | Throughput | Role & Scaling Profile |
|:---:|:---:|:---:|---|
| **1 pair** | 13.0 ms | **76.9 pairs/sec** | Interactive single-result scoring |
| **10 pairs** | 58.2 ms | **171.7 pairs/sec** | Standard web search candidate re-ranking |
| **25 pairs** | 118.3 ms | **211.3 pairs/sec** | Hybrid vector/keyword retrieval scoring |
| **50 pairs** | 204.7 ms | **244.3 pairs/sec** | Deep codebase snippet reranking |
| **100 pairs** | 400.4 ms | **249.8 pairs/sec** | Bus-saturating batch scoring (250 docs in 400ms) |

---

## 4. Levers, Knobs & Architectural Assessment Matrix

| Category | Lever / Knob | Tested Setting | Empirical Result | Architectural Decision & Rationale |
|---|---|---|---|:---:|
| **Speculative Decoding** | Draft Sampling Offload | `--spec-draft-backend-sampling` (GPU) vs CPU | Host CPU sampling is ~3% faster on dense models (`think`, 15.09 vs 14.96 tok/s), but GPU queue sampling boosted `main` (MoE) file editing from 68.7 to **79.2 tok/s (+15.3%)** by eliminating host-GPU sync stalls at high token rates. | **Locked: CPU for Dense, GPU for MoE** |
| **Speculative Decoding** | `ngram-mod` Window Geometry on MoE | `match=24, min=48, max=64` vs `match=16, min=32, max=48` vs `match=24, min=48, max=96` | Short window (`max=48`) caused −13.7% regression on file editing due to draft span truncation (59.3 vs 68.7 tok/s); deep window (`max=96`) overshot variable boundaries (61.1% acceptance). Stack default is optimal. | **Locked at stack default (24/48/64)** |
| **Speculative Decoding** | Sampling Temperature & Entropy | `temp 0.6` vs `0.2` vs `0.0` on `main` | Lowering temperature collapses token entropy, lifting draft coverage from 55–74% (`temp 0.6`) to **88–96% (`temp 0.0`)**, unlocking **135–183 tok/s** on code refactors. | **Deterministic/Greedy optimal for code** |
| **Speculative Decoding** | `--spec-draft-n-max` & `p-min` on `main` (MoE) | `n=4, p=0.75` vs `n=3, p=0.8` vs `n=5, p=0.75` | Expanding horizon to `n=4, p=0.75` boosted code refactoring to **195.6 tok/s (99.8% draft accept, 3.37s)** and file editing to **97.4 tok/s (87.1% accept)**. `n=5` peaked at 200.4 tok/s on refactoring but suffered truncations on multi-line edits. | **Locked at `n=4, p=0.75`** |
| **Speculative Decoding** | `--spec-draft-n-max` & `p-min` on `think` (Dense) | `n=6, p=0.6` vs `n=4, p=0.5` vs `n=6, p=0.4` | Raised structured file editing from **15.8 $\to$ 23.3 tok/s (+47.8%)** with **84.1% draft acceptance** and **72.0% draft coverage**, slashing turn latency from 36.6s to **25.3s**. | **Locked at `n=6, p=0.6`** |
| **Speculative Decoding** | `--spec-draft-n-max` & `p-min` on `revil2` (RE) | `n=6, p=0.6` vs `n=4, p=0.5` | Smashed reverse engineering decompilation / code modification decode to **81.6 tok/s (97.4% draft acceptance, 94% coverage)**, cutting turn time from 36s to **8.36s**. | **Locked at `n=6, p=0.6`** |
| **Speculative Decoding** | N-gram window tuning on `flash` | `n=8, m=64` vs `n=12, m=48` vs `n=12, m=64` | Short-trigger `n=8, m=64` boosted file editing to **32.2 tok/s (+53.5%)** and refactoring to **78.4 tok/s (3.71× faster)** with 86.9% draft coverage (8.69s turn time). | **Locked at `n=8, m=64`** |
| **Speculative Decoding** | N-gram engine suite on `flash` | `simple`, `mod`, `map-k`, `map-k4v`, `cache`, `combo` | `simple` (73.9 tok/s) and `mod` (71.0 tok/s) deliver 3.5× speedup at 0 GB extra RAM. `cache` regressed decode (−16%) and cost +8 GB RAM. | **Locked at `ngram-simple`** |
| **Speculative Decoding** | DFlash2 GGUF Compatibility | `incoai/Qwen3.8-27B-DFlash2-Q8_0.gguf` | Failed on boot (`wrong number of tensors; expected 81, got 58`) due to DFlash2 schema mismatch with upstream llama.cpp DFlash v1 parser. Native MTP remains king. | **Rejected DFlash2; Keep native MTP** |
| **Speculative Decoding** | Multimodal Vision Speculation | Pure Autoregressive vs `ngram-mod` on `vision` | Visual tokens are continuous ViT embeddings, yielding 0.0% text draft hits. Kept pure autoregressive to avoid unnecessary string matching overhead. | **Pure Autoregressive on Vision** |
| **Speculative Decoding** | Tree Speculation Branching | `--spec-draft-p-split 0.0` vs `0.25` | Negligible variance (15.07 vs 15.12 tok/s) because linear MTP confidence at p-min=0.6 is already >84% accurate. | **Keep linear (default)** |
| **Micro-Batching** | `-b 4096 -ub 1024` on `flash` | `2048/512` vs `4096/1024` | Lifted 8K prompt prefill ingestion from **370.7 $\to$ 436.5 tok/s (+17.7%)**, dropping 8K TTFT from 12.57s to 11.85s. | **Locked at `-b 4096 -ub 1024`** |
| **Micro-Batching** | `-b 4096 -ub 1024` on `main` (MoE) | `512` vs `1024` vs `2048` vs `4096` | Boosted MoE prefill **1,133 $\to$ 1,386 tok/s (+22.3%)**. `-ub 4096` caused regressions at long context. | **Locked at `-b 4096 -ub 1024`** |
| **Micro-Batching** | `-b 8192 -ub 8192` on `embed` (RAG) | `1` vs `16` vs `64` vs `128` vs `256` batch size | Saturated Strix memory bus at **240.1 chunks/sec (11,190 tok/s in 533ms)**, enabling full codebase vector indexing in seconds. | **Locked at `-b 8192 -ub 8192`** |
| **Context Scaling** | Flash Attention 1K $\to$ 64K depth | 1K vs 16K vs 32K vs 64K tokens on `think` | Rock-solid attention scaling: decode decayed only 22.9% out at 32K context depth (17.3 $\to$ 13.3 tok/s), holding 127K+ tokens in KV cache without memory pressure. | **Locked at `--flash-attn on`** |
| **Process Scheduling** | `--prio 2` on Linux/CachyOS | `--prio 0` (CFS) vs `--prio 2` (High) | Flat, ultra-stable generation timings across all workloads; eliminates OS background task preemption. | **Active & Locked at `--prio 2`** |
| **Threading** | Asymmetric Zen 5 CCX Allocation | `-t 8 -tb 16` vs `-t 16 -tb 16` | `-t 8` achieved identical generation throughput (15.09 vs 15.06 tok/s) while saving 50% CPU power and freeing 8 cores for host multitasking. | **Locked at `-t 8 -tb 16`** |
| **GEMM Dispatch** | ROCm hipBLASLt vs Tensile | `ROCBLAS_USE_HIPBLASLT=1` | Directs GEMM operations to optimized hipBLASLt kernels on RDNA 3.5 (`gfx1151`). | **Locked at `ROCBLAS_USE_HIPBLASLT=1`** |
| **KV Cache Precision** | `f16` vs `q8_0` on `think` | `f16` vs `q8_0` KV cache | `f16` is 14.9% faster on fresh synthesis (17.3 vs 15.1 tok/s). `q8_0` uses 50% less RAM, maintaining 262K-524K stability and hitting 23.3 tok/s on edits. | **Locked at `q8_0/q8_0`** |
| **Prefix Caching** | `--cache-ram 16384` vs `32768` + `-sps 0.30` | 16GB vs 32GB host RAM cache | Reusing shared system prompt dropped turn-2 prefill latency from **1,197.7ms $\to$ 113.3ms (10.6× faster / 90.5% TTFT reduction)**. 16GB cache is the lean sweet spot. | **Locked at 16GB + `-sps 0.30`** |
| **Memory Mapping** | `--no-mmap` vs `mmap` | Direct physical RAM vs Virtual demand-paging | Identical TTFT (0.62s warmup) on Gen4 NVMe. `mmap` preferred for faster container boot. | **Locked at `mmap`** |
| **Daemon Hot-Reload** | `--watch-config` in `llama-swap` | Enabled in systemd service | Instant zero-downtime hot-reloading on YAML file save without dropping active connections. | **Active & Live in systemd** |
| **Tool Calling** | `tool_call_format` | `xml` vs `json` | Qwen native XML tags reduce parser recovery errors by ~15%. | **Locked at `xml`** |

---

## 5. Measurement Methodology (and what it used to get wrong)

Every figure above is measured after three corrections applied on 2026-08-24. The pre-2026-08-24
numbers in git history are not comparable, and two of them were physically impossible.

**1. Warmup.** `tok_s` was `completion_tokens / wall_time`, and the first scored prompt's wall
time included llama-swap's model load. Measured cold-vs-warm on `main`: **11.63 tok/s** for prompt
1 against 51.95 and 53.87 for prompts 2 and 3 — dragging a ~54 tok/s model to a 39.15 tok/s
average. Both `bench.py` and `load-bench.py` now issue an unscored warmup request first. In
`load-bench.py` this mattered doubly: the sweep runs the ×1 level first, so the whole model load
was charged to the baseline and inflated every scaling ratio measured against it (published
1.43× at ×8; actual **1.34×**).

**2. Decode rate vs end-to-end rate.** Tokens ÷ total request time is not a generation rate — it
is diluted by prefill and TTFT. The headline number now comes from `llama-server`'s
`timings.predicted_per_second`, with the end-to-end rate reported separately.

**3. Effective bandwidth.** The old formula was `tok/s × total_weight_GB`, which assumes one full
sweep of the weights per emitted token. With MTP speculative decoding several tokens come out of a
single forward pass, so the product overstates real DRAM traffic by roughly the MTP speedup. The
result exceeded the 256 GB/s physical bus — `think` was published at 411 GB/s and `admin` at
335 GB/s, and an older tool using the *correct* weight still reported 546 GB/s. Stale
`MODEL_WEIGHTS` constants (Q8 export sizes for models that shipped as Q6 and Q4_K_M) made it
worse but were not the root cause. `calc_bandwidth` now discounts by measured draft coverage,
uses decimal GB to match the bus figure, and prints an explicit
`[!] EXCEEDS ROOFLINE` marker rather than publishing an impossible number.

**On acceptance rates.** Two different denominators are both legitimate and were previously
conflated: *drafts accepted* (accepted ÷ drafted — how well the p-min gate is tuned) and
*coverage* (accepted ÷ emitted — how much of the output was free). `main` measures 96.4% and 54%
respectively; quoting the first as if it were the second is what made MTP look near-perfect.
Bandwidth is discounted by coverage; the gate is judged by acceptance.

**Re-deriving these numbers.** `MODEL_WEIGHTS` in `bench.py` must be updated after any requant —
`stat -c %s` on the GGUF, converted to decimal GB. A stale entry silently corrupts every bandwidth
figure downstream, which is exactly how this happened.

**On raw run data:** results are published as tables in this file, and raw runs stay local by
design — `benchmarks/runs/` is gitignored and its existing contents come from the 13 retired
scripts (different schema, not comparable to the current tooling). If you are reproducing these
numbers, re-run the commands in §2 on your own hardware rather than diffing against stored runs.

---

## 6. Slot Count vs Per-Slot Context (`-np`)

`--ctx-size` is a **total** that llama.cpp divides across slots. Changing `-np` without scaling
`--ctx-size` silently changes how much context each request actually gets — verify with the
`n_ctx_slot` value logged at startup, not the flag you passed.

This bit `main`: it was scaled from 2 slots to 8 without touching `--ctx-size 524288`, quietly
cutting per-slot context from 262K to **65K** while every doc still claimed 262K. Corrected on
2026-08-24 to `-np 4 / --ctx-size 1048576`.

| Setting | `-np` | `--ctx-size` | `n_ctx_slot` | Peak aggregate | GTT |
|---|:---:|:---:|:---:|:---:|:---:|
| old | 8 | 524288 | 65,536 | 73.75 tok/s (×8) | ~50 GB |
| **current** | **4** | **1048576** | **262,144** | **70.23 tok/s (×4)** | **~55 GB** |

The trade is 4.8% peak aggregate throughput and ~5 GB for 4× per-slot context and 4 (not 8)
concurrent agents. Serial decode is unaffected (56.6 → 56.3 tok/s, inside run-to-run noise), as is
prefill. It matters more than usual here because `--context-shift` does not work on this model:
there is no graceful eviction at the ceiling, so the ceiling needs to be high enough.

### Vision (`--mmproj`)

`main` and `think` both load a vision projector (~0.85 GiB each) and genuinely read screenshots —
verified by feeding `think` a rendered terminal screenshot, which it read back correctly
("ERROR CODE 8231, faulted device nvme1n1p3"). Removing the projector to try to recover
`--cache-reuse` was measured and does **not** work: identical prefill, cache hits and decode rate
with and without it, the levers stay disabled either way, and you lose image input. Keep it.

---

## 7. Speculative Decoding Strategy (why `draft-mtp,ngram-mod`)

`--spec-type` takes a **comma-separated list**, and drafters compose: `main` and `think` both run
`draft-mtp,ngram-mod`. Chosen by A/B on 2026-08-24 (b10603) at production geometry.

### The workloads matter more than the strategy

Three prompts with deliberately different repetition profiles, because a single benchmark would
have given the wrong answer:

* **fresh-gen** — write a class from scratch. Nothing in context to copy.
* **file-edit** — given a ~60-line file, return it with one method changed. Output is mostly a
  verbatim copy of the input.
* **refactor-rename** — same file, rename two identifiers everywhere. Repetitive *but* every
  n-gram drawn from context mismatches at the renamed token.

### `main` (Qwen 3.6-35B MoE), decode tok/s at `-np 1`

| Strategy | fresh-gen | file-edit | refactor-rename |
|---|---:|---:|---:|
| `draft-mtp` (was production) | 58.35 | 71.53 | **70.05** |
| `ngram-mod` alone | 43.35 | 100.80 | 43.64 |
| **`draft-mtp,ngram-mod`** | **58.57** | **105.57** | 68.84 |
| `ngram-map-k4v` alone | 43.42 | 53.65 | 45.29 |

**n-gram alone is a trap.** It is 41% faster on file-edit but **26% slower on fresh-gen** (0.0%
draft acceptance — it drafts from a history with nothing to match) and **38% slower on
refactor-rename** (acceptance collapses to 19.5%). Benchmarking only the copy-heavy case would have
justified a change that made two of three real workloads much slower. `ngram-map-k4v` is worse than
`ngram-mod` everywhere and was discarded.

MTP covers the non-repetitive case; n-gram covers the copy-heavy case; together they raise draft
coverage on file-edit from 69.6% to 82.6% rather than competing.

### Production-geometry validation

`main` at `-np 4` / 262K per slot, `think` at `-np 2` / 262K per slot:

| | metric | `draft-mtp` | `draft-mtp,ngram-mod` | Δ |
|---|---|---:|---:|---:|
| **main** | GTT footprint | 55.8 GiB | 55.8 GiB | none |
| | fresh-gen | 57.97 | 57.80 | −0.3% |
| | file-edit | 69.85 | **104.95** | **+50%** |
| | refactor-rename | 69.71 | 67.68 | −2.9% |
| | aggregate ×4 | 83.12 | **116.85** | **+41%** |
| **think** | GTT footprint | 49.4 GiB | 49.4 GiB | none |
| | fresh-gen | 21.25 | 21.28 | +0.1% |
| | file-edit | 28.27 | **67.57** | **+139%** |
| | refactor-rename | 27.42 | **43.61** | **+59%** |
| | aggregate ×2 | 45.51 | **123.26** | **+171%** |

Two things worth noting: the n-gram pools cost **no additional memory** at full context, and the
benefit **grows** under concurrency rather than degrading. `think` gains most because it is
bandwidth-bound — each forward pass skipped is worth more at 21 tok/s than at 57.

### Flash-Next Validation (Qwen 3.8-Flash-Next, 2026-08-28)

`qwen38-flash-next` (DeltaNet + QSA) lacks upstream MTP support in llama.cpp, making it an ideal testbed for runtime speculative engines alone (`ngram-mod`, `ngram-simple`, `ngram-map-k`):

| Strategy | fresh-gen | file-edit | refactor-rename | reasoning | Turn Wall Time | Notes |
|---|---:|---:|---:|---:|:---:|---|
| `Baseline (none)` | 21.37 | 21.01 | 21.05 | 21.10 | 27.6s | Pure autoregressive |
| `ngram-simple` | **22.69** | **26.09** | **73.89** | 22.50 | **9.0s** | Fastest raw speed, 98.5% acceptance on edits |
| `ngram-mod` | 21.99 | 25.40 | **71.02** | 22.00 | **9.3s** | **Locked in Stack:** +237% on refactors, 0 extra RAM |
| `ngram-simple,ngram-mod` | 22.32 | 22.06 | **74.15** | **22.82** | **8.9s** | Peak overall throughput (74.2 tok/s peak) |
| `ngram-map-k` | 21.20 | 22.27 | 26.48 | 21.30 | 22.3s | Discarded: low draft coverage (24% max) |
| `ngram-map-k4v` | 21.29 | 22.00 | 26.69 | 21.30 | 22.1s | Discarded: similar to map-k; low span length |
| `ngram-cache` | 21.52 | 19.70 | 17.63 | 19.94 | 32.7s | Avoid: Lookup overhead regresses decode & uses +8 GB RAM |

**Key architectural finding:** Unlike `main` (58 tok/s MoE), where standalone n-gram without MTP regressed on fresh generation (−26%), `flash` (~21 tok/s base) has enough per-token compute time to completely hide CPU-side n-gram verification overhead. It suffered **zero regression on fresh generation** (+2.9% to +6.2% due to boilerplate keyword hits) while delivering a **3.5× speedup** on structured code edits. `ngram-mod` is locked into `flash` in `llama-swap.yaml`.

### Reproducibility

Repeat runs of both `main` arms agreed within 0.7% (file-edit 71.59/71.46 for MTP,
105.94/105.20 for the combination), so the effect is not run-to-run noise.

`ngram-mod` window geometry was empirically verified across sweeps (2026-08-28): default geometry (`--spec-ngram-mod-n-match 24`, `-n-min 48`, `-n-max 64`) proved optimal against shorter (`16/32/48`: −13.7% truncation penalty on edits) and deeper (`24/48/96`: variable boundary overshoot and rollback discards) configurations.

---

## 8. Batch Geometry Re-validation (negative result, 2026-08-24)

The `-b 4096 -ub 1024` setting was tuned on 2026-08-21 under conditions that no longer apply:
`main` ran `-np 8` @ 65K/slot, on llama.cpp b10438, without n-gram speculation. It was re-tested on
**b10603 at the current `-np 4` @ 262K/slot with `draft-mtp,ngram-mod`** to check it had not gone
stale. Prompt-processing rate by actual prompt length:

| Geometry | 552 | 2,046 | 8,180 | 32,756 | decode | GTT |
|---|---:|---:|---:|---:|---:|---:|
| **`-b 4096 -ub 1024`** (current) | 997 | 1,376 | 1,303 | 929 | 55.01 | **55.8 GiB** |
| `-b 4096 -ub 2048` | 992 | 1,386 | **1,326** | 927 | 57.06 | 57.0 GiB |
| `-b 8192 -ub 2048` | 984 | **1,394** | 1,310 | 923 | 57.01 | 57.0 GiB |
| `-b 8192 -ub 4096` | 996 | **1,407** | 1,209 | 878 | 56.45 | 59.4 GiB |

**Conclusion: keep `-b 4096 -ub 1024`.** The spread across the first three rows is 1–2% on prefill,
which a single run per cell cannot distinguish from noise — the honest claim is "no change
justified", not "provably optimal". The decode column looks like +3.7% for the `ub 2048` rows, but
`main` decode measured anywhere from 54.9 to 58.9 tok/s across prompts the same day, so that is
also noise rather than a batch effect.

Two things *are* outside the noise:

* **`-ub 4096` is worse.** Best at 2K (1,407) but **−7% at 8K** (1,209 vs 1,303) and −5% at 32K,
  for **+3.6 GiB**. Micro-batches stop paying once the chunk exceeds what the pipeline can keep fed.
* **`-ub 2048` costs +1.2 GiB** for at most ~2% at mid-context.

**A hypothesis this killed:** prefill peaks near 2K and falls by 32K (1,376 → 929), which looks like
micro-batch starvation. It is not — *every* geometry falls off by the same proportion, including
`-ub 4096`. That curve is context scaling (attention cost growing with sequence length), so do not
go chasing it with batch flags.

If re-running this, use 2–3 repeats per cell; at these effect sizes one run per cell is not enough
to resolve the differences, which is itself the reason no change was made.

---

## 9. Upstream Evaluation Frameworks & Repositories

All quality and capability evaluations in this repository rely strictly on authoritative, open-source community standards:

* **EvalPlus:** [https://github.com/evalplus/evalplus](https://github.com/evalplus/evalplus)
  * *Paper:* ["Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Code Generation with EvalPlus"](https://arxiv.org/abs/2305.01210) (NeurIPS 2023).
  * *Leaderboard:* [https://evalplus.github.io/leaderboard.html](https://evalplus.github.io/leaderboard.html)
  * *Provides:* 80× rigorous unit-test generation and execution sandboxing for HumanEval (`HumanEval+`) and MBPP (`MBPP+`).
* **EleutherAI `lm-evaluation-harness`:** [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
  * *Documentation:* [https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs)
  * *Powers:* [Hugging Face Open LLM Leaderboard (v1 & v2)](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard).
  * *Provides:* Canonical dataset ingestion, verified prompt formatting, and exact-match answer parsing for GSM8K, MATH, ARC-Challenge, and MMLU.
* **`llama-swap`:** [https://github.com/mostlygeek/llama-swap](https://github.com/mostlygeek/llama-swap)
  * *Provides:* Dynamic multi-model routing, process lifecycle management, on-demand multi-residency, and prefix cache preservation.
* **`llama.cpp`:** [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
  * *Provides:* ROCm `gfx1151` inference backend, GGUF quants, MTP speculative decoding, and native KV caching.
* **AMD Strix Halo Toolboxes:** [https://github.com/kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)
  * *Provides:* Optimized ROCm 10.0 runtime container images for AMD Ryzen AI Max+ 395 APUs.
