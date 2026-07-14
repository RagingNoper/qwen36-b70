# Qwen3.6-35B-A3B on 4× Intel Arc Pro B70 (vLLM-XPU): full benchmarks + one-command reproducible builds

Follow-up to my earlier posts on getting this MoE running on Battlemage. It's been about six weeks of yak-shaving; I finally have two configs I'm happy with, benchmarked them properly (throughput, latency, *and* capability), and packaged everything so you can `docker load` an image and serve the model (or re-run every benchmark) with one Python script. Origin story, hurdles, numbers, and repro below.

**Hardware:** 4× Intel Arc Pro B70 (32 GB each, Battlemage/Xe2), Threadripper Pro on a WRX80 board. Model is Qwen3.6-35B-A3B (35B total, ~3B active MoE). Serving is vLLM-XPU with a pile of custom kernels.

## The road here

**Last month's build was just getting the thing to run *correctly* at all.** Battlemage compute on Linux is bleeding-edge, and "35B MoE on 4× Arc Pro B70 via vLLM" had no beaten path — the stock XPU stack got me almost nothing, so most of this is custom. The big early ones:

- **torch.compile emitted NaNs** on the model's gated-delta-net attention. Fixed by writing an unconditional GDN custom op + a dedicated decode kernel.
- **cudagraphs on XPU** — the thing that makes decode fast — took a lot of coaxing to capture and replay correctly.
- **The tensor-parallel all-reduce was broken under graph capture:** oneCCL (Intel's collective library) mis-replays inside a captured graph. So I wrote a **custom all-reduce from scratch** — Level-Zero IPC peer pointers + a device-resident barrier + a SYCL reduce. This one kept coming back to haunt me (see the prefill bit below).

Then the hardware fought back:

- **PCIe bifurcation:** the board's BIOS had quietly split the x16 slots into x4×4, so every GPU was crawling on a single x4 lane. A BIOS fix to Gen4 x16 unlocked a big chunk of bandwidth.
- **The 3-of-4-GPU mystery:** after some reboots, only 3 of the 4 cards would enumerate. I chased it through slots and *even a board swap* before it turned out to be a single **bad PCIe power cable.**

And the toolchain:

- An **oneAPI compiler regression** broke the ESIMD path my all-reduce barrier used → rewrote the barrier in plain SYCL.
- Fun gotcha that cost me a day: the Intel driver **reserves host RAM equal to total VRAM** (~120 GB across 4 cards), invisible to normal tools — a concurrent kernel build kept OOM-killing the running server until I understood that.

That got me to a stable **~100 tok/s decode** baseline. This month was about the last wall: single-stream **prefill** (the DMA all-reduce story further down).

## The two configs

- **bf16-tp4** — bf16 across all 4 cards.
- **int8-tp2** — `experts_int8` across 2 cards (leaves 2 free for a second instance).

Both use MTP (k=2) speculative decode + `FULL_DECODE_ONLY` cudagraphs.

## Performance (temp 0, seed 42)

| | bf16-tp4 (4 cards) | int8-tp2 (2 cards) |
|---|---|---|
| Prefill TTFT @1024 tok | 382 ms | **350 ms** |
| Prefill TTFT @4096 tok | 1398 ms | **882 ms** |
| Single-stream decode | ~128 t/s | **~145 t/s** |
| Throughput @ c32 (ShareGPT) | 487 t/s | **518 t/s** |
| p99 TTFT @ c32 (static) | 11.5 s | **7.5 s** |

**int8-tp2 matches or beats the 4-card bf16 config on basically everything — on half the GPUs.** bf16-tp4 only pulls ahead at mid concurrency (c8–c16).

Single-stream decode is ~128 t/s (bf16) / ~145 t/s (int8), TPOT ~7–8 ms — and it *stays* fast because MTP (speculative decode) is carrying most of it; MTP acceptance on real content runs 78–86%. Combined output throughput as you pile on concurrent requests:

| combined tok/s | c1 | c8 | c16 | c32 |
|---|---|---|---|---|
| bf16-tp4 (ShareGPT) | 128 | 207 | 358 | 487 |
| int8-tp2 (ShareGPT) | 132 | 195 | 330 | 518 |
| bf16-tp4 (static 1024/256) | 107 | 165 | 273 | 344 |
| int8-tp2 (static 1024/256) | 114 | 154 | 256 | 387 |

Both scale cleanly to c32. int8-tp2 wins the ends (c1 latency and c32 peak) and has a far better tail — **7.5 s vs 11.5 s p99 TTFT at c32** — while bf16-tp4 takes the mid-range. And remember int8-tp2 is doing all this on **2 cards**, so the other two are free for a second replica (≈2× aggregate).

### Prefill throughput (tok/s), for the prompt-processing crowd

Prefill rate rises with prompt length (fixed per-request overhead amortizes), then flattens:

| prompt tokens | bf16-tp4 | int8-tp2 |
|---|---|---|
| 1024 | 2,680 t/s | 2,930 t/s |
| 2048 | 2,910 t/s | 3,880 t/s |
| 4096 | 2,930 t/s | 4,650 t/s |

**Peak (batched, output=1): bf16-tp4 ~2,970 t/s, int8-tp2 ~4,670 t/s.** Notable finding: it **saturates by ~concurrency 16** (c32 is identical), and batched ≈ single-stream — a single big prompt already maxes the GPUs, so there's no hidden batched headroom for prefill on a model this size. int8-tp2 works out to **~2,335 t/s per card**; for reference the popular single-B70 llama.cpp+Vulkan writeups land ~1,824 t/s per card (on Q4 — we're on int8, i.e. reading *more* bytes/token), so per-card we're comfortably ahead.

## Capability

All measured in **thinking mode** (this is a reasoning model — non-thinking scores undersell it, and it's the mode you actually deploy), `<think>` stripped before scoring. **int8 and bf16 land within ~1pp on every benchmark — quantization is free** — so the numbers below are **averaged across the two configs**:

| benchmark | score |
|---|---|
| MMLU-Redux 2.0 | 93.4% |
| IFEval | 93.7% |
| HumanEval pass@1 | 97.3% |
| GSM8K | 97% |

Low-90s knowledge, ~97 code/math, and instruction-following on par with the same-shape predecessor (Qwen3.5-35B-A3B) and the wider Qwen3.5 family. (The IFEval figure averages its four sub-metrics — prompt- and instruction-level × strict and loose — which span 91–96%.)

**Benchmarking a reasoning model — lessons that cost me real points:**
- **Use a relabeled knowledge set.** Standard MMLU is saturated and has mislabeled gold answers — it undersold this model by ~5pts (read 88%). MMLU-Redux 2.0 (corrected labels) is the honest number: **93.4%**.
- **Thinking ON, strip `<think>` before scoring.** A bad strip regex tanked IFEval to 10% until I noticed Qwen closes `</think>` with *no opening tag* in the completion.
- **Give reasoning room.** lm-eval's default 1280-token generation cap truncates the trace and craters the strict per-prompt score (all-or-nothing per prompt). Use ≥32k; the model's context is 262k so a big cap is nearly free.
- **Sample, don't greedy-decode.** With the model's recommended sampling (temp 0.6 / top_p 0.95) instead of greedy, IFEval recovered ~2 points. Greedy sends ~2% of hard *lexical*-constraint prompts (letter-frequency, no-comma, all-caps) into infinite self-verification loops — each a guaranteed fail. (A mild `presence_penalty` / `no_repeat_ngram` fixes it in production.)

## The interesting engineering bit

Single-stream prefill was the wall — it was ~84% all-reduce. Battlemage has no fast collective and vLLM's default path was reading peers over PCIe at ~7% of link bandwidth. I wrote a **custom all-reduce that gathers peer data with the GPU copy engine** (full PCIe bandwidth) instead of a compute kernel → **2.5× single-stream prefill** on bf16-tp4. A couple of gotchas that ate days:

- oneCCL migrates `malloc_shared` device-only, so the host read of the peer-pointer array segfaulted → cache the pointers host-side.
- The copy-engine gather is incompatible with **piecewise** cudagraph capture (async host-side copies aren't recorded) → `FULL_DECODE_ONLY` (decode still fully captured/fast, prefill runs eager). Turns out piecewise + the barrier-based custom AR corrupts short prompts *regardless* of the DMA path, so FDO is the right call anyway.

Interesting counter-result: DMA all-reduce is a **TP4-specific** win. On int8-tp2 it's break-even — 2 ranks means each GPU reads only 1 peer instead of 3, so the all-reduce was never the bottleneck there. int8-tp2 already prefills faster than bf16-tp4-*with*-DMA.

## Run it yourself

Self-contained image (~11 GB download, ~48 GB on disk once loaded) with every patch + the eval harness baked in — nothing to mount but the model. You need: B70 GPUs, Docker with `/dev/dri`, and the Qwen3.6-35B-A3B weights.

```bash
docker pull ghcr.io/ragingnoper/qwen36-b70-ship:latest

# just serve it (leaves it running):
python3 serve.py --config int8-tp2 --model /path/to/Qwen3.6-35B-A3B
```

`serve.py` brings the model up and hands you a standard **OpenAI-compatible endpoint**, so you point whatever you like at it — Open WebUI, LibreChat, the `openai` python lib, curl. The engine and the UI are fully independent; use any client.

Want to verify the numbers above? `python3 reproduce.py --config int8-tp2 --model ...` runs the whole perf + MMLU-Redux/IFEval/HumanEval/GSM8K suite inside the container (offline, in thinking mode) and prints the table (`--suite quick` for a ~15-min sanity run; the full capability suite is ~3-4 h). A layman-friendly step-by-step (drivers → docker → serve → connect a UI) is included.

Everything (`serve.py`, `reproduce.py`, and a layman-friendly setup guide): **https://github.com/RagingNoper/qwen36-b70** — the image is just `docker pull ghcr.io/ragingnoper/qwen36-b70-ship`.

Happy to answer questions on the kernels, the cudagraph stuff, or Battlemage serving in general.
