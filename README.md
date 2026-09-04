# Framework Desktop AI Stack (AMD Strix Halo + CachyOS)

Personal local AI setup running on a Framework Desktop (AMD Ryzen AI Max+ 395 / Strix Halo `gfx1151`, 128 GB unified LPDDR5X memory) with CachyOS Linux.

This repository tracks configuration, routing rules, container environments, and benchmark results for local models on unified memory APUs.

---

## Active Model Matrix

Models are routed on-demand via [`llama-swap`](https://github.com/mostlygeek/llama-swap) on port `8090`:

### Generative Models (with Vision)

| Model ID | Architecture & Quant | Context | Slots | Decode Speed | Primary Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`main`** | **Qwen 3.6-35B-A3B MoE** *(UD-Q8_K_XL)* | 262K | 2 | **56.8 tok/s** | Daily workhorse, coding & vision OCR |
| **`think`** | **Qwen 3.8-27B Dense** *(UD-Q6_K_XL)* | 262K | 1 | **15.1 tok/s** | Deep reasoning, architecture & planning |
| **`flash`** | **Qwen 3.8-Flash-Next** *(UD-IQ4_XS)* | 262K | 1 | **22.2 tok/s** | Qwen4 architecture preview (Gated DeltaNet) |

### Retrieval & Voice Services

| Model / Service | Quant / Engine | Context / Endpoint | Compute | Role |
| :--- | :--- | :---: | :---: | :--- |
| **`embed` (bge-m3)** | Q8_0 | 8K | GPU (ROCm) | Multilingual vector embeddings for semantic search |
| **`rerank` (bge-reranker-v2-m3)** | Q8_0 | 8K | GPU (ROCm) | Cross-encoder reranker for document retrieval |
| **Speaches TTS (Kokoro)** | 82M ONNX | `http://localhost:8088` | CPU (int8, 8 threads) | Text-to-speech (`af_heart`) |
| **Speaches STT (Faster-Whisper)** | Large-v3-turbo CT2 | `http://localhost:8088` | CPU (int8, 8 threads) | Audio transcription & voice dictation |

---

## Memory & Lifecycle Management

With a shared 128 GB unified memory pool:
* **Coexistence:** `think` (1 slot, ~38 GB) and `main` (2 slots, ~48.8 GB) can run concurrently in memory (~87 GB total) without swapping.
* **Auto-Unload:** Models automatically unload after idle timeout (300s default; 1 hour for `main`) to free RAM for host applications.

---

## Stack Components

* **Router:** `llama-swap` (`:8090`) for on-demand model loading, multi-residency, and OpenAI-compatible endpoints.
* **Web UI:** Open WebUI (`:3001`) for chat, document RAG, and web search.
* **Search:** SearXNG (`:8085`) with FlareSolverr.
* **Voice:** Speaches (`:8088`, Kokoro TTS + Faster-Whisper STT) running isolated on CPU.
* **Coding Agents:** Oh My Pi (`omp`) and `pi`.
* **MCP Servers:** Serena (LSP), Semgrep (SAST), SonarQube, Burp Suite, Ghidra, Puppeteer, SearxNcrawl, and Mnemopi.
* **Remote Access:** Tailscale Aperture.

---

## Hardware, Kernel & Engine Settings

### 1. Bootloader Parameters
Set in `/etc/default/grub` or systemd-boot loader entry:
```text
amdgpu.gttsize=131072
ttm.pages_limit=33554432
amd_iommu=off
```

### 2. ROCm Container Environment
Set on the ROCm runner (`docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-10.0` in `llama-swap.yaml`):
```bash
# Prevent SDMA queue timeout on APU unified memory
HSA_ENABLE_SDMA=0

# Enable tuned hipBLASLt GEMM kernel dispatch on gfx1151
ROCBLAS_USE_HIPBLASLT=1
```

### 3. Engine Levers (`llama-swap.yaml`)
* **Asymmetric Threading (`-t 8 -tb 16`):** Limits serial generation to 8 physical cores on CCX0 to prevent cache bounce; uses all 16 cores for prompt prefill.
* **Batch Sizing (`-b 4096 -ub 1024`):** Micro-batching of 1024 for prompt ingestion (`embed` and `rerank` use `-b 8192 -ub 8192` to ingest full 8K windows).
* **Speculative Decoding:** MTP draft heads (`--spec-type draft-mtp`) on `main` ($n=4$) and `think` ($n=6$); N-gram (`--spec-type ngram-simple`, $n=8, m=64$) on `flash`.
* **Host RAM Prefix Cache (`--cache-ram`):** Allocates host RAM (8–16 GB) to retain multi-turn conversation KV prefix trees for instant TTFT.

---

## Verified Quality Benchmarks ([EleutherAI `lm-evaluation-harness`](benchmarks/eval_reasoning.py))

Official evaluation across canonical reasoning benchmarks using unconstrained chain-of-thought (`<think>`) traces:

| Model ID | Task / Dataset | Problems | Metric | Strict Match | Flexible Extract |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`main`** | GSM8K (5-shot CoT) | 200 | Exact Match | **97.0%** (194/200) | **92.5%** |
| **`think`** | GSM8K (5-shot CoT) | 25 | Exact Match | **88.0%** (22/25) | **96.0%** (24/25) |

---

## CLI Management (`aictl`)

The [`aictl`](aictl) script manages stack lifecycle, inspection, and benchmarks:

```bash
# === General Help ===
aictl --help                            # Full command line usage and options

# === Service & Telemetry ===
aictl status                            # Service status and loaded models
aictl top                               # Real-time GPU & system telemetry
aictl test main "Hello"                 # Quick inference sanity check
aictl unload                            # Free memory by unmounting models

# === Benchmarks ===
aictl bench think                       # Serial tok/s decode, TTFT & bandwidth
aictl bench reason think                # EleutherAI lm-eval (GSM8K)
aictl bench reason think --tasks ifeval # Instruction following evaluation
```

---

## Upstream References & Ecosystem

* **[AMD Strix Halo Toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes):** Pre-built ROCm container images tuned for AMD Ryzen AI Max+ 395 (`gfx1151`).
* **[`llama.cpp`](https://github.com/ggerganov/llama.cpp):** Core C/C++ inference engine with ROCm, GGUF, and MTP speculative decoding support.
* **[`llama-swap`](https://github.com/mostlygeek/llama-swap):** Dynamic multi-model proxy and on-demand lifecycle manager for `llama-server`.
* **[EleutherAI `lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness):** Standard open-source framework for evaluating LLMs on reasoning benchmarks (powers the [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)).
* **[Speaches](https://github.com/speaches-ai/speaches):** Local OpenAI-compatible audio API for Kokoro TTS and Faster-Whisper STT.
* **[SearXNG](https://github.com/searxng/searxng):** Self-hosted metasearch engine for local web retrieval.
* **[Open WebUI](https://github.com/open-webui/open-webui):** Web interface for chat, document RAG, and agent orchestration.
* **[Tailscale Aperture](https://github.com/tailscale/aperture-cli):** Secure identity-aware AI agent gateway and proxy across tailnets.
