# Running Qwen3.6-35B-A3B on Intel Arc B70 — step-by-step guide

This walks you all the way from a fresh Linux box with Intel Arc Pro B70 GPUs to a running model that
prints benchmark numbers. It assumes you can copy-paste into a terminal and read the output, but **not**
that you know anything about Intel GPU drivers, Docker, or vLLM. Take it one part at a time; each part
ends with a check so you know it worked before moving on.

**Choose your configuration.** You run everything in this guide with **`--config <tag>`** — you pick one of
the four tags below and use that same tag in every `serve.py` and `reproduce.py` command:

| `--config` tag | GPUs | best for |
|---|---|---|
| `int8-tp4-latency` | 4 | The fastest replies, when it's mostly one person chatting at a time |
| `int8-tp2` | 2 | Nearly as fast on just **two** cards (or to keep the other two free) |
| `int8-tp4-concurrency` | 4 | Serving **many people at once** (biggest KV cache + throughput) |
| `bf16-tp4` | 4 | Full-precision weights — most people don't need it (int8 matches its quality) |

So a real command looks like `python3 serve.py --config int8-tp4-latency --model /path/to/your/model`.
**Heads-up:** the examples throughout this guide use `int8-tp2` as a stand-in — wherever you see it, put in
whichever tag you picked from the table above.

Rough time: 30–60 min of setup (mostly the GPU driver), then ~15 min for a quick test run.

---

## Part 0 — What you need

- A Linux PC/server with **2 or 4 Intel Arc Pro B70** GPUs installed and powered (each needs its PCIe
  power cables). Validated on **Ubuntu 26.04**, but any recent Linux with Intel GPU support works.
- About **60 GB of free disk** for the container image, plus room for the model (~70 GB).
- An internet connection for the driver install.
- The **Qwen3.6-35B-A3B** model files (HuggingFace format) — see Part 4.
- The scripts from the repo: `serve.py`, `reproduce.py`, and this folder. (The container image is
  pulled from GitHub in Part 5 — you don't download it separately.)

---

## Part 1 — Confirm Linux sees your GPUs

Open a terminal and run:

```bash
lspci | grep -i battlemage
```

You should see **one line per card**, e.g.:

```
03:00.0 VGA compatible controller: Intel Corporation Battlemage G31 [Intel Graphics]
23:00.0 VGA compatible controller: Intel Corporation Battlemage G31 [Intel Graphics]
...
```

✅ **Check:** the number of `Battlemage` lines equals the number of cards you installed (2 or 4).
If a card is missing, it's usually a power cable or a reseat — fix that before continuing. (On some
boards a missing card after a warm reboot needs a full power-off, not just a restart.)

---

## Part 2 — Install the Intel GPU drivers (the important part)

Your GPUs need Intel's **compute runtime** (the software that lets programs actually use the GPU for
math, not just display). Battlemage needs a **recent** version — this bundle was validated with the
Intel client GPU packages at version **26.05.x** (compute runtime) and **Level-Zero loader 1.28**.

Follow Intel's official instructions for your distro — they keep the commands current:
**https://dgpu-docs.intel.com/driver/client/overview.html**

For Ubuntu it comes down to adding Intel's GPU package repository and installing these packages:

```bash
# after adding Intel's repo per the link above:
sudo apt update
sudo apt install -y intel-opencl-icd libze-intel-gpu1 libze1 intel-level-zero-gpu
```

Then **reboot** so the kernel driver loads cleanly:

```bash
sudo reboot
```

After it comes back, confirm the GPUs show up as compute devices:

```bash
ls -l /dev/dri/
```

✅ **Check:** you should see one `renderD1xx` entry per card, e.g. `renderD128`, `renderD129`, … These
`renderD*` nodes are how programs talk to the GPUs. (You'll also see `card0`, `card1`, … — those are
fine to ignore.) If there are no `renderD*` entries, the compute driver didn't install — recheck the
Intel doc and that you rebooted.

> **Note:** the GPU driver is the one step that varies by machine and by how new your OS is. If you get
> stuck, the Intel link above and their forums are the right place — everything after this is easy.
>
> **Why exact versions here don't need to match ours:** the container ships with its *own* Intel GPU
> userspace runtime (the part that runs the model), so on the host you really only need a **kernel driver
> that supports Battlemage** plus the `/dev/dri` render nodes. The package versions above are just a
> known-good set; a newer driver is fine. (Details in `VERSIONS.md`.)

---

## Part 3 — Install Docker and grant GPU access

Docker runs the model in a sealed "container" so you don't have to install vLLM or its kernels yourself.

Install Docker:

```bash
sudo apt install -y docker.io
```

Let your user run Docker and reach the GPUs without typing `sudo` every time:

```bash
sudo usermod -aG docker,render,video $USER
```

**Log out and back in** (or reboot) for that to take effect, then check:

```bash
docker run --rm hello-world
```

✅ **Check:** you see "Hello from Docker!". If instead you get a permission error, you didn't log out/in
after the `usermod` command — do that.

---

## Part 4 — Get the model

You need the **Qwen3.6-35B-A3B** model in HuggingFace format (a folder of `.safetensors` files plus
`config.json`, `tokenizer.json`, etc.). Download it from wherever you obtained access, and put it in a
folder, e.g.:

```
/home/you/models/Qwen3.6-35B-A3B/
```

✅ **Check:** `ls /home/you/models/Qwen3.6-35B-A3B` shows a `config.json` and several `.safetensors`
files. Remember this path — you'll pass it in Part 6.

---

## Part 5 — Get the container image

The image is hosted on GitHub, so you just pull it (it's ~11 GB to download, ~48 GB once unpacked, so
give it a few minutes):

```bash
docker pull ghcr.io/ragingnoper/qwen36-b70-ship:latest
```

✅ **Check:**

```bash
docker images ghcr.io/ragingnoper/qwen36-b70-ship
```

shows a line for the image.

---

## Part 6 — Run it

From the folder that contains `reproduce.py`, start a **quick** test first (~15 min). Point `--model`
at your model folder from Part 4:

```bash
python3 reproduce.py --config int8-tp2 --model /home/you/models/Qwen3.6-35B-A3B --suite quick
```

(Have 4 cards? Use `--config int8-tp4-latency` for the fastest single-user setup, or `--config int8-tp4-concurrency` to serve many users at once.)

What happens: it starts the model server, waits for it to say `READY` (the first start compiles some
kernels and can take 5–10 minutes — that's normal), runs the benchmarks, prints a results table, and
shuts the server down.

When you're happy it works, run the **full** suite (drop `--suite quick`; takes ~3–4 hours — the
capability evals run in thinking mode with a large token budget, which is slow but correct — and produces
all the published numbers including MMLU / HumanEval / GSM8K):

```bash
python3 reproduce.py --config int8-tp2 --model /home/you/models/Qwen3.6-35B-A3B
```

### Which two GPUs? (only matters for `int8-tp2`)

`int8-tp2` uses **2** of your GPUs, and they talk to each other over PCIe during every prompt. **How
fast that link is depends on which two cards you pick** — on a 4-card board, some pairs sit on the same
part of the PCIe tree (fast) and some cross a slower hop. On the reference machine, cards **2,3** are the
fast pair (prefill ~350 ms) and cards 0,1 are ~40% slower for the same work.

- The script defaults to `--devices 2,3` (the fast pair on the reference box). If you have **4 cards**,
  start there.
- If you only have **2 cards**, add `--devices 0,1`.
- If prefill looks slow, try a different pair (`--devices 1,2`, `0,2`, etc.) — pick whichever gives the
  best "TTFT@1024" number. Adjacent cards on the same PCIe slot group are usually best.

(The 4-card configs use all four cards, so there's no pair to choose — this only applies to `int8-tp2`.)

---

## Part 6b — Just use the model (serve it + connect a chat UI)

The benchmark script above starts the model, tests it, and shuts it down. To actually **use** the model
day-to-day, use `serve.py` instead — it starts the model and **leaves it running**:

```bash
python3 serve.py --config int8-tp2 --model /home/you/models/Qwen3.6-35B-A3B
```

(Same `--config` tags as the table up top — swap `int8-tp2` for `int8-tp4-latency`, `int8-tp4-concurrency`,
or `bf16-tp4` depending on your hardware and how many people will use it.)

When it's ready it prints a box with the connection details, the important ones being:

```
API base URL : http://<this-machine-ip>:8107/v1
Model name   : qwen3.6-35b-a3b
API key      : none
```

**How this works (worth understanding):** the model runs as a *server* that speaks the "OpenAI API" — the
same format ChatGPT's API uses. The server and your chat window are **separate programs**. The server just
loads the model and answers requests; you bring whatever chat interface you like and point it at the URL
above. Nothing is locked together — you can even run several different apps against the same running model.

### Option A — a chat UI (recommended: Open WebUI)

Open WebUI is a self-hosted, ChatGPT-style web interface. Start it with one command:

```bash
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8107/v1 \
  -e OPENAI_API_KEY=none \
  --name openwebui ghcr.io/open-webui/open-webui:main
```

Then open **http://localhost:3000** in your browser, make an account (it's local-only), and the
`qwen3.6-35b-a3b` model appears in the model picker. That's it — chat away. (Other UIs like LibreChat or
SillyTavern work the same way: point them at `http://<ip>:8107/v1`, model `qwen3.6-35b-a3b`, no API key.)

### Option B — from code (Python)

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8107/v1", api_key="none")
r = client.chat.completions.create(model="qwen3.6-35b-a3b",
        messages=[{"role": "user", "content": "Explain PCIe in one sentence."}])
print(r.choices[0].message.content)
```

### Option C — quick test with curl

```bash
curl http://localhost:8107/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Hello!"}]}'
```

Stop the server when you're done: `docker rm -f qwen36-serve`.

---

## Part 7 — Reading the results

At the end you'll get a table like:

```
-- prefill --
  TTFT@1024               350.0 ms      <- time to first token for a 1024-token prompt
-- static --
  c1 out_tput             114.4 t/s     <- tokens/sec, one request at a time
  c1 TPOT                  6.9  ms       <- time per output token (lower = faster typing)
-- capability (thinking mode) --
  GSM8K                    97.0 %        <- accuracy on grade-school math
  MMLU-Redux 2.0           93.4 %        <- broad knowledge (corrected-label MMLU)
  IFEval (avg of 4)        93.7 %        <- instruction-following
  HumanEval pass@1         97.3 %        <- coding
```

These should match the published numbers within a few percent of run-to-run variation. If they do,
you've reproduced the result. 🎉

---

## Part 8 — If something goes wrong

- **"container died during boot"** — run `docker logs qwen36-repro` and look near the bottom.
  - *"visible devices (N)"* where N is less than expected → not all cards enumerated (see Part 1; try a
    full power-off/on).
  - *out of memory / OOM* → another program is using the GPUs, or you tried a 4-card config (`int8-tp4-latency`,
    `int8-tp4-concurrency`, `bf16-tp4`) with fewer than 4 cards. Close other GPU programs; make sure the card
    count matches the config.
- **Boot takes a very long time the first run** — normal; it compiles GPU kernels. The second run is faster.
- **"permission denied" talking to Docker or the GPU** — you skipped the log-out/in after Part 3's
  `usermod`. Log out and back in.
- **Wrong number of GPUs** — `int8-tp2` needs 2; the three `-tp4` configs need 4. You can point at specific cards with
  `--devices 0,1` (first two) if your machine numbers them differently.

Still stuck? The server log (`docker logs qwen36-repro`) almost always names the problem on one of its
last lines.
