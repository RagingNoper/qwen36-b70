# What's inside the image (exact versions)

Everything below is **baked into** `ghcr.io/ragingnoper/qwen36-b70-ship` at these exact versions — you get
all of it from `docker pull`, with nothing to install and no version drift.

Important: the image is built **FROM a prebuilt vLLM-XPU base image** (the author's custom build). The
`Dockerfile` in this repo only adds the kernel patches + the eval harness on top, so the versions below
come from that base image, **not** from the Dockerfile. The image is the source of truth; this file just
documents what's in it.

| Component | Version |
|---|---|
| Container OS | Ubuntu 24.04.3 LTS |
| Python | 3.12.3 |
| PyTorch (XPU) | 2.12.0+xpu |
| vLLM | 0.1.dev1+gdec860fb1 (main-branch build) |
| Triton (XPU) | 3.7.1 |
| Intel oneAPI DPC++/C++ compiler | 2025.3.2 |
| Intel oneCCL | 2021.17.2 |
| Intel compute runtime (Level-Zero GPU, in-container) | 26.09.37435.12 |
| Level-Zero loader (in-container) | 1.28.0 |
| lm-eval-harness (+ transformers, evaluate, datasets) | 0.4.12 |
| Custom bits | DMA copy-engine all-reduce, MTP drafter + GDN cudagraph fix, experts_int8 kernels (see `patches/`) |

## GPU driver — host vs. container (read this)

Intel GPU compute is split across two layers, and only one of them is your responsibility:

- **Kernel driver — on your HOST (you install this).** The `i915`/`xe` kernel module + Battlemage
  firmware, exposed as `/dev/dri/renderD*`. See `GETTING_STARTED.md` Part 2. Reference machine: Ubuntu
  26.04, kernel 7.1, Intel client-GPU packages **26.05.x**.
- **Userspace runtime — in the CONTAINER (already baked, you do nothing).** `libze-intel-gpu` **26.09** +
  Level-Zero loader **1.28.0**. This is what actually executes the model on the GPU.

Because the userspace lives in the container, **your host userspace version does not need to match** — the
container's 26.09 runtime is in fact *newer* than the reference host's 26.05 and runs correctly. The only
hard host requirement is a **Battlemage-capable kernel driver** plus `/dev/dri` access to the container.

## Reproducing the base image

The Dockerfile is intentionally a thin layer; it is **not** a from-scratch recipe, because the base image
(`t212-vllm-graph-head2-mtp`) is a large custom vLLM-XPU build (patched kernels, a custom all-reduce, a
paged-decode patch, triton-xpu, etc.). The intended distribution is the prebuilt GHCR image above, not a
rebuild-from-source — the pinned versions in the table are how you verify you have the right one.
