# What's inside the image (exact versions)

Everything below is **baked into** `ghcr.io/ragingnoper/qwen36-b70-ship` at these exact versions — you get
all of it from `docker pull`, with nothing to install and no version drift.

Important: the image is built **FROM a prebuilt vLLM-XPU base image** (the author's custom build). The
`Dockerfile` in this repo only adds the eval harness on top, so the versions below come from that base
image, **not** from the Dockerfile. The image is the source of truth; this file just documents what's in it.

| Component | Version |
|---|---|
| Container OS | Ubuntu 24.04.3 LTS |
| Python | 3.12.3 |
| PyTorch (XPU) | 2.13.0+xpu |
| vLLM | main-branch build (0.1.dev) |
| Triton (XPU) | 3.7.2 |
| Intel oneAPI DPC++/C++ compiler | 2026.1.0 |
| Intel oneCCL | 2022 (pip `oneccl`, torch-native `xccl` backend) |
| Intel compute runtime (Level-Zero GPU, in-container) | 26.09.37435.12 |
| Level-Zero loader (in-container) | 1.28.0 |
| lm-eval-harness (+ transformers, evaluate, datasets) | 0.4.12 |
| Custom bits | custom all-reduce (capture+small route: vec-reduce + reduce-scatter/all-gather), MTP drafter + GDN / mamba V2-align cudagraph fix, `experts_int8` kernels + batch-1 MoE GEMV + autotuned MoE block configs, turboquant (int8) KV-cache dtype, prefix-cache (mamba-align) support (see `patches/`) |

> **Note on the stack move.** This is the **oneAPI-2026** build. The previous release (oneAPI 2025.3 /
> torch 2.12 / oneCCL 2021) is preserved at git tag `v1.0-oneapi2025` and image tag
> `ghcr.io/ragingnoper/qwen36-b70-ship:oneapi2025`. The new runtime is leaner (more of each card's VRAM
> goes to KV cache) and prefills ~2x faster; see the writeup for the full before/after.

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
(`t212-vllm-oneapi2026:prod2`) is a large custom vLLM-XPU build (patched kernels, a custom all-reduce, a
paged-decode patch, triton-xpu, oneCCL 2022, etc.) on oneAPI 2026.1. The intended distribution is the
prebuilt GHCR image above, not a rebuild-from-source — the pinned versions in the table are how you verify
you have the right one.
