# Qwen3.6-35B-A3B on 4× Intel Arc Pro B70 — reproducible benchmark bundle

Two turn-key vLLM-XPU serving configs for the Qwen3.6-35B-A3B MoE, plus a one-command harness that
reproduces the published throughput, latency, and capability numbers.

- **`bf16-tp4`** — bf16 across 4 cards, custom DMA copy-engine all-reduce (2.5× single-stream prefill).
- **`int8-tp2`** — experts_int8 across 2 cards. Matches bf16-tp4 on speed *and* capability, on half the cards.

Both run MTP (k=2) speculative decode + `FULL_DECODE_ONLY` cudagraphs. Everything (custom kernels, the
all-reduce `.so`, MTP/GDN patches, int8 kernels, and the full eval harness) is **baked into the image** —
no runtime mounts except the model.

## Requirements

- A box with Intel Arc Pro B70 GPUs (4 for `bf16-tp4`, 2 for `int8-tp2`) + the Level-Zero/compute stack.
- Docker with `--device /dev/dri` GPU access.
- Python 3 (stdlib only) on the host to run `reproduce.py`.
- The **Qwen3.6-35B-A3B** model directory on disk (HF-format).

Exact software/driver versions baked into the image are in **[VERSIONS.md](VERSIONS.md)**.

## 1. Get the image

```bash
docker pull ghcr.io/ragingnoper/qwen36-b70-ship:latest   # ~11 GB download, ~48 GB on disk
docker images ghcr.io/ragingnoper/qwen36-b70-ship        # confirm it's there
```

## 2. Use it — serve the model

```bash
python3 serve.py --config int8-tp2 --model /path/to/Qwen3.6-35B-A3B
```

Starts the model and **leaves it running**, then prints its connection details. It serves the standard
**OpenAI-compatible API**, so point any chat UI or app at it — Open WebUI, LibreChat, the `openai`
Python library, `curl`, etc. (the server and the UI are fully independent; use whatever you like). See
**GETTING_STARTED.md → "Part 6b — Just use the model"** for the one-line Open WebUI setup. Stop with
`docker rm -f qwen36-serve`.

## 2b. (Optional) Verify the published numbers

```bash
python3 reproduce.py --config int8-tp2 --model /path/to/Qwen3.6-35B-A3B            # full, ~1-2 h
python3 reproduce.py --config bf16-tp4 --model /path/to/Qwen3.6-35B-A3B --suite quick   # ~15 min
```

`reproduce.py` boots the server, runs the benchmarks **inside** the container (offline — datasets are
baked in), prints a results table, and tears the container down.

## 3. Expected results (temp 0, seed 42)

| | bf16-tp4 (4 cards) | int8-tp2 (2 cards) |
|---|---|---|
| Prefill TTFT @1024 tok | 382 ms | 350 ms |
| Raw decode, single stream | ~128 t/s | ~145 t/s |
| Peak throughput @ c32 (ShareGPT) | 487 t/s | 518 t/s |
| MMLU (5-shot) | 83.0% | 82.9% |
| IFEval prompt-strict | 83.4% | 83.6% |
| HumanEval pass@1 | 93.9% | 94.5% |
| GSM8K | 98% | 97% |

Numbers reproduce within run-to-run noise. The capability axes are statistically identical between the
two configs — int8 costs no measurable quality.

> **int8-tp2 note:** its prefill depends on the PCIe bandwidth between the two GPUs you use. `reproduce.py`
> defaults to `--devices 2,3` (a high-P2P pair on the reference box). On a 2-GPU box use `--devices 0,1`;
> on other 4-GPU boxes, if prefill is slow, try another adjacent pair. `bf16-tp4` uses all 4, so it's not affected.

## What's inside

`reproduce.py` (host) → boots `qwen36-b70-ship` → `bench_inner.py` (in-container) drives `vllm bench serve`
+ `lm-eval` (MMLU/IFEval) + `humaneval_eval.py` + `gsm8k_eval2.py`. Server flags are in `reproduce.py`
(`CONFIGS` + `CFG`); the two configs differ only in TP size, quantization, gpu-util, and the DMA all-reduce.

---

## Publishing the image (maintainer note)

The image lives on GitHub Container Registry, so recipients just `docker pull` — no tarball. To publish
a new build, push it to GHCR (needs a token with `write:packages`):

```bash
docker tag qwen36-b70-ship:latest ghcr.io/ragingnoper/qwen36-b70-ship:latest
echo "$GHCR_TOKEN" | docker login ghcr.io -u ragingnoper --password-stdin
docker push ghcr.io/ragingnoper/qwen36-b70-ship:latest
# then, one time, make the package public in the GitHub package settings
```

The model is **not** in the image (licensing + size) — recipients obtain Qwen3.6-35B-A3B separately.
