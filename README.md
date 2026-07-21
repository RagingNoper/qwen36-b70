# Qwen3.6-35B-A3B on 4× Intel Arc Pro B70

Run the Qwen3.6-35B-A3B model on Intel Arc Pro B70 GPUs with one Docker image and one command.
Everything is baked into the image — you only supply the model files. Four ready-made setups; you
pick one at launch. New here? Follow **[GETTING_STARTED.md](GETTING_STARTED.md)** for a click-by-click
walkthrough (install Docker → get the model → serve → connect a chat app).

## Which one should I pick?

| Your situation | Use this | What you get |
|---|---|---|
| I have **4 cards** and mostly **one person** chatting at a time | **`int8-tp4-latency`** | The fastest replies (~206 tokens/sec). |
| I only have **2 cards** (or want to keep two free) | **`int8-tp2`** | Almost as fast (~174 tok/s), and the quickest to start answering long prompts. |
| I have **4 cards** and want to serve **lots of people at once** | **`int8-tp4-concurrency`** | ~965 tok/s across 64 simultaneous chats, and room for ~150 conversations at once. |
| I specifically want **full-precision** weights | **`bf16-tp4`** | Full bf16. (The int8 setups match it on quality and beat it on speed, so most people don't need this.) |

All four give **identical answer quality** — the int8 versions are quantized in a way that costs no
measurable quality on this model. Pick based on your hardware and how many people will use it.

## What you need

- A machine with Intel Arc Pro B70 GPUs — **2** for `int8-tp2`, **4** for the other three.
- **Docker** installed, with GPU access (`--device /dev/dri`).
- The **Qwen3.6-35B-A3B** model files on disk (download separately — not in the image).
- Nothing else: the image carries its own GPU runtime; the host just needs a Battlemage GPU driver.

(Exact software versions in the image: **[VERSIONS.md](VERSIONS.md)**.)

## 1. Get the image

```bash
docker pull ghcr.io/ragingnoper/qwen36-b70-ship:latest   # ~11 GB download
```

## 2. Start the model

Pick one line (point `--model` at your Qwen3.6-35B-A3B folder):

```bash
python3 serve.py --config int8-tp4-latency     --model /path/to/Qwen3.6-35B-A3B   # fastest replies (4 cards)
python3 serve.py --config int8-tp2             --model /path/to/Qwen3.6-35B-A3B   # 2 cards
python3 serve.py --config int8-tp4-concurrency --model /path/to/Qwen3.6-35B-A3B   # many people at once
python3 serve.py --config bf16-tp4             --model /path/to/Qwen3.6-35B-A3B   # full precision
```

The first start takes ~8–12 min (it compiles GPU kernels), then it prints a URL and **keeps running**.
Stop it any time with `docker rm -f qwen36-serve`.

## 3. Connect a chat app

The server speaks the standard **OpenAI API**, so any OpenAI-compatible chat app works. Point it at the
URL that `serve.py` printed (e.g. `http://<your-machine-ip>:8107/v1`), model name `qwen3.6-35b-a3b`,
API key left blank. **[GETTING_STARTED.md](GETTING_STARTED.md)** has a copy-paste Open WebUI setup.

## What to expect (reference box, seed 42)

| | int8-tp4-latency | int8-tp2 | int8-tp4-concurrency | bf16-tp4 |
|---|---|---|---|---|
| Single reply speed (decode) | **206 tok/s** | 174 tok/s | 142 tok/s | 175 tok/s |
| Time to first token (1024-tok prompt) | 206 ms | 173 ms | 195 ms | 205 ms |
| Prompt-reading speed (1024 tok) | 5,147 tok/s | **6,268 tok/s** | 5,148 tok/s | 5,139 tok/s |
| Throughput at 64 users | — | — | **965 tok/s** | — |
| Simultaneous conversations (KV cache) | 816k tok | 267k tok | **1.37M tok** | 380k tok |

**Quality is the same across all four** (thinking mode, recommended sampling): MMLU-Redux 2.0 **93.4%**,
IFEval **92.7%**, HumanEval **97.0%**, GSM8K **98%** — right where a healthy Qwen3.6-35B-A3B should be.
int8 costs no measurable quality vs bf16.

**Want to double-check the numbers yourself?**
```bash
python3 reproduce.py --config int8-tp4-latency --model /path/to/Qwen3.6-35B-A3B --suite quick   # ~15 min
python3 reproduce.py --config int8-tp4-concurrency --model /path/to/Qwen3.6-35B-A3B             # full, ~3–4 h
```
It runs the benchmarks inside the container (datasets are baked in), prints a table, and cleans up.

> **`int8-tp2` — which two cards matter.** On a 4-card box the pairs aren't equal: two cards on the same
> part of the motherboard talk ~3.7× faster than a cross-board pair, and that affects how fast `int8-tp2`
> reads prompts. `serve.py` defaults to the fast pair (`--devices 2,3`) on the reference box; on a 2-GPU
> box use `--devices 0,1`; on other boxes, if prompt-reading is slow, try a different adjacent pair. The
> 4-card configs use all four and aren't affected.

## Optional: get ~100 GB of host RAM back (multi-GPU only)

When serving across multiple GPUs, the Intel `xe` driver quietly mirrors the whole GPU working set into
system RAM (~72 GB on 2 cards, ~121 GB on 4). [`ram-fix/`](ram-fix/) has a small, reversible host kernel
patch that reclaims it (down to ~14–22 GB), with zero quality change. It's a **host-side, root, one-time**
step — not part of `docker pull`. Skip it unless you're tight on RAM. See [`ram-fix/README.md`](ram-fix/README.md).

---

The model is **not** in the image (licensing + size) — get Qwen3.6-35B-A3B separately.
