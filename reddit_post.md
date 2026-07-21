# Qwen3.6-35B-A3B on 4× Intel Arc Pro B70 (vLLM-XPU): four tuned configs, full benchmarks, one-command reproducible builds

Follow-up to my earlier posts on getting this MoE running on Battlemage. It started as "can I fully tax four B70s with one big model," and after a lot of testing it turned into **four** serving configs I'm happy with — a single-stream latency champion, a 2-card option, a high-concurrency config, and a full-precision one — all shipping in a single Docker image where you pick the config at launch. Benchmarked properly (throughput, latency, *and* capability) and packaged so you can `docker pull` and serve (or re-run every benchmark) with one Python script. Origin story, the configs, numbers, the interesting engineering, and repro below.

**Hardware:** 4× Intel Arc Pro B70 (32 GB each, Battlemage/Xe2), Threadripper Pro on a WRX80 board. Model is Qwen3.6-35B-A3B (35B total, ~3B active MoE). Serving is vLLM-XPU with a pile of custom kernels.

## How this became a four-config release

The original goal was simple: **saturate all four cards** with one model and get it as fast as possible in bf16. But a clean capability harness flipped the design. First, **int8 cost nothing in quality during capability testing** — within ~1 point of bf16 on every benchmark, no measurable capability difference (details below). Second — and this is what flipped it — **int8 isn't just as-good-as bf16, it's faster silicon**: at matched settings (spec-decode off on both) int8 decodes **~1.4× faster than bf16 on the same four cards (142 vs 101 tok/s)**, from reading half the weight bytes. So the "premium" full-precision config had **no accuracy edge and no decode edge**, and bf16 stopped being the default.

From there it was **which int8 config for which job**, and reaching 206 tok/s single-stream took the whole custom stack pulling in the same direction: a **from-scratch batch-1 int8 MoE GEMV kernel** (the stock grouped GEMM is occupancy-starved at ~1 row per expert, so I wrote a direct expert-indexed streaming kernel that ~3.5×'d it), **MTP speculative decode drafting three tokens deep**, that GEMV kernel **widened to also serve the speculative *verify* batch** — which otherwise dropped back to the slow grouped GEMM on every step — the **16-byte-vectorized custom all-reduce** (reduce-scatter/all-gather), and **`FULL_DECODE_ONLY` cudagraph capture** wrapping all of it so none of that orchestration hits per-token launch overhead. The result is a **4-card int8 config at 206 tok/s single-stream** — the fastest of everything here, and faster than 2 cards: at 4-way the ¼-of-the-model-per-card weight-read win outruns the extra all-reduce once that reduce is cheap. That's `int8-tp4-latency`. Only have two cards? `int8-tp2` gives **174 tok/s** on half the hardware. And running int8 across four cards *without* speculation instead spends that budget on a **1.37M-token KV cache** and batch headroom for a lot of concurrent users — `int8-tp4-concurrency`. `bf16-tp4` stays in the box because the data's done and validated, not because it wins anything.

So: four configs, one image, choose at launch.

## The four configs

- **`int8-tp4-latency`** — `experts_int8` across **4 cards**, MTP + the widened MoE-GEMV + vectorized all-reduce → **the single-stream champion, 206 tok/s decode** (177 combined). If you have four cards and want the fastest possible single response, this is it. ~816k-token KV.
- **`int8-tp2`** — `experts_int8` across **2 cards** (64 GB), MTP → **174 tok/s single-stream** and the **fastest prefill of any config (6,268 t/s)** on just two B70s. The pick if you have a 64 GB box or want the other two cards free. ~267k-token KV.
- **`int8-tp4-concurrency`** — `experts_int8` across 4 cards, no speculation, a throughput-tuned vectorized all-reduce → the **biggest KV cache (~1.37M tokens)** and **~965 tok/s at 64 concurrent requests**. The "host a bunch of users" config.
- **`bf16-tp4`** — full bf16 across 4 cards, MTP. In the box for completeness (full-precision weights if you specifically want them). Once the int8 MoE kernel was autotuned, it **wins on no axis** — int8 matches or beats it on capability, decode, prefill, and concurrency alike. Kept because it's done and validated, not because it's better. ~380k-token KV.

All configs use `FULL_DECODE_ONLY` cudagraphs; the three MTP configs (`int8-tp4-latency`, `int8-tp2`, `bf16-tp4`) run speculative decode, `int8-tp4-concurrency` does not.

## The road here

**The early months were just getting this to run *correctly* at all.** Battlemage compute on Linux is still immature, and "35B MoE on 4× Arc Pro B70 via vLLM" had no beaten path — the stock XPU stack got me almost nothing, so most of this is custom:

- **torch.compile emitted NaNs** on the model's gated-delta-net attention → wrote an unconditional GDN custom op + a dedicated decode kernel.
- **cudagraphs on XPU** — the thing that makes decode fast — took a lot of coaxing to capture and replay correctly.
- **The tensor-parallel all-reduce was broken under graph capture:** oneCCL mis-replays inside a captured graph, so I wrote a **custom all-reduce from scratch** — Level-Zero IPC peer pointers + a device-resident barrier + a SYCL reduce. This one kept coming back to haunt me.
- An **oneAPI compiler regression** broke the ESIMD path the barrier used → rewrote it in plain SYCL. And a fun one that cost a day: the Intel driver **reserves host RAM equal to total VRAM** (~120 GB across 4 cards), invisible to normal tools — a concurrent kernel build kept OOM-killing the running server until I figured out what it was. I recently chased that one to the root and **fixed it with a one-function kernel patch** (~100 GB of host RAM reclaimed, capability-neutral) — see the RAM section below.

That got me to a stable ~100 tok/s decode baseline — which turned out to be the *start* of the decode work, not the end. Roughly doubling it to 206 took a from-scratch batch-1 MoE-GEMV kernel, speculative decode with a widened verify path, the vectorized all-reduce, and a lot of profiling to find where each token's time actually went. Prefill and concurrency were their own separate pushes on top.

## Performance (seed 42)

The configs are tuned for different jobs, so read this as **"which config for which job,"** not one leaderboard. Single-request numbers are on **two shapes**: **static** (fixed 1024-in / 256-out) and **ShareGPT** (real chat prompts + real EOS variable output; the prompts are short, median ~31 tok).

**1. A single request (latency).** Single-stream, sent sequentially (no queue effect). *decode* = steady-state, prefill excluded; *TTFT* = first-token latency; *combined* = end-to-end, prefill included. Two shapes:

| config | ShareGPT — decode / TTFT / combined | static — decode / TTFT / combined | KV cache |
|---|---|---|---|
| **`int8-tp4-latency`** | 196 / 134 ms / 179 | **206 / 206 ms / 177** | 816k tok |
| `int8-tp2` (2 cards) | 178 / 122 ms / 165 | 174 / 173 ms / 156 | 267k tok |
| `int8-tp4-concurrency` | 147 / 122 ms / 142 | 142 / 195 ms / 128 | **1.37M tok** |
| `bf16-tp4` | 186 / 109 ms / 175 | 175 / 205 ms / 154 | 380k tok |

*(decode/combined in tok/s. ShareGPT prompts are short — median ~31 tok — so its TTFT is short-prompt latency and combined ≈ decode.)*

**`int8-tp4-latency` is the single-stream pick** — fastest here (206 decode / 177 combined static). int8 is faster silicon: spec-decode off on both, **int8 decodes ~1.4× faster than bf16 on the same four cards (142 vs 101 tok/s)** from reading half the weight bytes; MTP + the widened batch-1 MoE kernel gets you to 206. `int8-tp2` gives up ~15% of decode to run on **two** cards (174 vs 206).

**2. Prefill processing** (prompt-len ÷ TTFT, tok/s):

| config | @1024 | @2048 | @4096 |
|---|---|---|---|
| **`int8-tp2`** | **6,268** | **7,002** | **7,368** |
| `int8-tp4-latency` | 5,147 | 5,415 | 5,454 |
| `int8-tp4-concurrency` | 5,148 | 5,385 | 5,447 |
| `bf16-tp4` | 5,139 | 5,647 | 5,828 |

**`int8-tp2` has the fastest prefill of any config** — a 2-card box out-prefilling 4-card bf16, by pairing a freshly-autotuned int8 MoE kernel with the cheap 2-card all-reduce (engineering section below). The 4-card int8 configs match bf16. **So bf16-tp4 leads on no axis** — capability, decode, prefill, concurrency all favor int8. (For reference, single-B70 llama.cpp+Vulkan writeups land ~1,824 tok/s per card on Q4 prefill.)

**3. Many concurrent requests (throughput) — `int8-tp4-concurrency`.** Output tok/s (prefill included) as simultaneous requests scale — mean and peak, static 1024/256:

| concurrency | mean | peak |
|---|---|---|
| 8 | 320 | 560 |
| 16 | 564 | 912 |
| 32 | 718 | 1,088 |
| 64 | **965** | **1,600** |

`int8-tp4-concurrency` is the throughput config — **965 tok/s at 64 concurrent** — and its **1.37M-token KV cache** (~4–5× the others) is what lets it hold that many simultaneous conversations. The latency configs aren't built for this; they spend their compute on single-stream speculation, not batch.

## Capability

This section is a **control, not a leaderboard flex** — it shows the months of custom-kernel / MTP / quantization / all-reduce / autotuning work didn't quietly degrade the model. All four configs share the same weights (`experts_int8`, or bf16) and land within ~1 point of each other, so one representative is shown — **`int8-tp4-concurrency`**. Measured in **thinking mode** (the deploy mode), `<think>` stripped before scoring, recommended sampling, large generation budget:

| benchmark | `int8-tp4-concurrency` |
|---|---|
| MMLU-Redux 2.0 | 93.4% |
| IFEval | 92.7% |
| HumanEval pass@1 | 97.0% |
| GSM8K | 98% |

int8 tracks bf16 within ~1 point on every benchmark, the vec-reduce/RS-AG all-reduce is numerically faithful, and the MoE-GEMV widening, deeper speculation, *and* the autotuned MoE config were each separately GSM8K-verified lossless (96–98%, temp-0). The scores are exactly where a healthy Qwen3.6-35B-A3B should be — none of the kernel / MTP / quant / tuning work cost measurable quality. (IFEval averages its four sub-metrics — prompt/instruction × strict/loose — and has real ~2-3 point run-to-run variance in thinking mode.)

**Benchmarking a reasoning model — lessons that cost me real points:**
- **Use a relabeled knowledge set.** Standard MMLU is saturated with mislabeled gold answers — it undersold this model by ~5pts (read 88%). MMLU-Redux 2.0 (corrected labels) is the honest number: **93.3–93.5%**.
- **Thinking ON, strip `<think>` before scoring.** A bad strip regex tanked IFEval to 10% until I noticed Qwen closes `</think>` with *no opening tag*.
- **Give reasoning room + sample, don't greedy-decode.** lm-eval's default 1280-token cap truncates the trace and craters the strict per-prompt score; use ≥32k. And greedy sends ~2% of hard *lexical*-constraint prompts (letter-frequency, no-comma, all-caps) into infinite self-verification loops — each a guaranteed fail — so use the model's recommended sampling (temp 0.6 / top_p 0.95). Getting these two wrong is a ~2-3 point swing, and I re-learned it the hard way benchmarking the third config.

## The interesting engineering bits

**Prefill (the latency configs).** Single-stream prefill was ~84% all-reduce; Battlemage has no fast collective and vLLM's default read peers over PCIe at ~7% of link bandwidth. I wrote a custom all-reduce that gathers peer data with the **GPU copy engine** (full PCIe bandwidth) → **2.5× single-stream prefill** on `bf16-tp4`. It's a **TP4-specific** win: on `int8-tp2` it's break-even (2 ranks means each GPU reads only 1 peer, so the all-reduce was never the bottleneck). Gotcha that ate days: the copy-engine gather is incompatible with **piecewise** cudagraph capture, so everything runs `FULL_DECODE_ONLY` (decode fully captured/fast, prefill eager) — which is the right call anyway since piecewise + the barrier corrupts short prompts.

**The single-stream champion (`int8-tp4-latency`).** Two stacked tricks get 4-card int8 to 206 tok/s. First, speculative decode (MTP) — but its *verify* pass runs the target on a small batch of candidate tokens (k+1), which fell just outside my batch-1 int8 MoE-GEMV kernel's window and dropped back to the occupancy-starved grouped GEMM on every speculative step; widening the kernel to cover the verify batch recovered the fast path there. Second, drafting one token deeper. Both are lossless (exact rejection sampling, GSM8K-verified) and free — the net is the fastest single response of any config, on four cards you'd otherwise use for concurrency.

**Concurrency (`int8-tp4-concurrency`).** Getting it to high throughput was a separate all-reduce project. The decode all-reduce at high batch was the bottleneck, and profiling showed the reduce kernel was reading peer memory **two bytes per lane** — a leftover from the original scalar kernel. Rewriting it to 16-byte vectorized loads was a **~3.5× per-byte** speedup on the reduce alone (+36% end-to-end at c64), and layering a **reduce-scatter/all-gather** collective on top (halving the wire bytes per rank) added a few more percent. (I also tried a push-based collective and moving the logits all-gather onto the custom path — both turned out to be dead ends after a lot of measurement; happy to go into why in the comments.)

**Autotuning the int8 MoE kernel (a late, free multiplier).** Profiling the prefill gap turned up something dumb-but-large: the int8 MoE GEMM was running the stock Triton kernel with a **default block config** — no autotuned config existed for this expert shape on Battlemage — and the default is **pathologically bad at large batch** (10–11× off optimal at ≥1024 tokens, i.e. the entire prefill / high-concurrency regime; it's fine at batch-1, which is why single-stream decode never showed it). Sweeping block sizes and dropping in a per-shape config — **a JSON file, no kernel change, bit-identical output** — closed it: **int8 prefill jumped to bf16 parity or better** (`int8-tp2` to 6,268 t/s, fastest of any config) and **concurrency rose ~19% at c64** (811→965). Single-stream *decode* is untouched — that path uses the custom batch-1 GEMV, not the Triton kernel — so this is a pure prefill + throughput win.

**The two-card PCIe saga (why `int8-tp2` cares which cards you give it).** Worth a warning if you run the 2-card config. The four B70s do **not** all talk to each other equally — they hang off different quadrants of the CPU's IO die through per-card onboard PCIe switches, and peer-to-peer bandwidth between a given *pair* depends on which slots/quadrant they're in. On my box, one specific pair (the two cards sharing an IO-die quadrant) reads peer memory at **~20 GB/s**, while any cross-quadrant pair manages only **~5.5 GB/s** — a 3.7× difference purely from topology. For `int8-tp2` that means **which two cards you pick materially affects prefill speed**; `serve.py` defaults to the fast pair on my box (`--devices 2,3`), with a `--devices` override for yours. (`tp4` uses all four, so it's unaffected.)

**Reclaiming ~100 GB of host RAM (a one-function kernel fix).** The weirdest problem here: while serving, the box eats **host RAM equal to the total VRAM working set** — ~72 GB on 2 cards, **~121 GB on 4** — invisible to `top`/RSS/page-cache accounting, released only when the model stops. It's what pushed me to 128 GB of system RAM just to run a 35B model whose weights live entirely in VRAM. I finally instrumented the Intel `xe` kernel driver's dma-buf path and found it: on tensor-parallel serving, each GPU exports its buffers so peers can read them, and `xe_gem_prime_export()` → `ttm_bo_setup_export()` → `ttm_tt_populate()` **allocates a full-size system-memory copy of every exported buffer** — even though the buffer stays in VRAM and the peers read it over **PCIe P2P** (I counted: ~99% of the reads are P2P, 8 out of 562 touch system pages). So the entire cross-GPU working set gets duplicated in host RAM for nothing. A one-function patch adds an `xe.force_p2p_vram=1` param that skips that populate: **bf16-tp4 went 121 GB → 22 GB host RAM, int8-tp2 72 GB → 14 GB, with zero capability change** (re-ran the full MMLU-Redux/IFEval/GSM8K suite under the patched driver — all on reference; a broken P2P path would've tanked those by tens of points, not held steady). It's a **host-side kernel-module change** (can't ship in the container — containers share the host kernel), it's optional, and it's headed upstream — the real fix is Intel making that populate lazy/P2P-aware. Patch + build/install script + writeup are in [`ram-fix/`](https://github.com/RagingNoper/qwen36-b70/tree/main/ram-fix). If you run multi-GPU `xe`, this is a lot of RAM back.

## Run it yourself

One self-contained image (~11 GB download, ~48 GB on disk) with every patch, both all-reduce kernels, the int8 kernels, and the eval harness baked in — nothing to mount but the model. You need: B70 GPUs (2 for `int8-tp2`, 4 for the `tp4` configs), Docker with `/dev/dri`, and the Qwen3.6-35B-A3B weights.

```bash
docker pull ghcr.io/ragingnoper/qwen36-b70-ship:latest

# pick a config and serve it (leaves it running):
python3 serve.py --config int8-tp4-latency     --model /path/to/Qwen3.6-35B-A3B   # fastest single-stream (4 cards)
python3 serve.py --config int8-tp2             --model /path/to/Qwen3.6-35B-A3B   # fast single-stream / 2 cards
python3 serve.py --config int8-tp4-concurrency --model /path/to/Qwen3.6-35B-A3B   # many users / huge KV
python3 serve.py --config bf16-tp4             --model /path/to/Qwen3.6-35B-A3B   # full precision
```

`serve.py` brings the model up and hands you a standard **OpenAI-compatible endpoint**, so point whatever you like at it — Open WebUI, LibreChat, the `openai` python lib, curl.

Want to verify the numbers? `python3 reproduce.py --config <cfg> --model ...` runs the whole perf + MMLU-Redux/IFEval/HumanEval/GSM8K suite inside the container (offline, thinking mode) and prints the table (`--suite quick` for a ~15-min sanity run; the full capability suite is ~3-4 h). A layman-friendly step-by-step (drivers → docker → serve → connect a UI) is included.

**Optional — get your host RAM back.** If you run a multi-GPU config and want to stop the driver from mirroring the whole VRAM working set into system RAM (the ~100 GB thing above), `ram-fix/` has the kernel patch + a `build-and-install.sh`. It's a **host-side, root, one-time** step (a kernel-module change — not part of the `docker pull`), fully reversible, default-off. Skip it entirely if you don't care about the RAM.

Everything — `serve.py`, `reproduce.py`, and the setup guide — with full instructions in the **[README](https://github.com/RagingNoper/qwen36-b70)**: **https://github.com/RagingNoper/qwen36-b70** (image: `docker pull ghcr.io/ragingnoper/qwen36-b70-ship`).

Happy to answer questions on the kernels, the cudagraph/all-reduce stuff, or Battlemage serving in general.
