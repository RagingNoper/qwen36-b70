# Self-contained reproduction image for Qwen3.6-35B-A3B on 4x Intel Arc Pro B70 (vLLM-XPU).
# Bakes in every runtime patch + the eval harness so a reproducer needs only: this image + the model.
#
# NOTE: this is a THIN layer on a prebuilt vLLM-XPU base image; the heavy stack below comes from the
# base, not from this Dockerfile (so it is not a from-scratch recipe). Exact versions in the image
# (see VERSIONS.md):
#   Ubuntu 24.04.3  |  Python 3.12.3  |  torch 2.13.0+xpu  |  vLLM main-branch build  |  triton-xpu 3.7.2
#   oneAPI DPC++ 2026.1.0  |  oneCCL 2022 (pip / xccl backend)  |  compute runtime 26.09.37435.12  |  Level-Zero 1.28.0
# The container carries its OWN Intel GPU userspace runtime; the HOST only needs a Battlemage kernel
# driver + /dev/dri (host userspace version need not match — see VERSIONS.md).
#
# The base image already has ALL of the custom bits baked in and validated (mount-free):
#   - custom all-reduce, capture+small route (custom_ar_v4.so at /work/ext/) — vec-reduce + reduce-scatter/all-gather
#   - MTP spec-decode + GDN / mamba V2-align cudagraph fixes (FULL_DECODE_ONLY capture)
#   - experts_int8 kernels + batch-1 MoE GEMV + autotuned per-shape MoE block configs
#   - turboquant (int8) KV-cache dtype + prefix-cache (mamba-align) support
# The repo's patches/ folder documents these mods for transparency; they are ALREADY in the base image,
# so this Dockerfile does not re-copy them — that keeps the published image byte-identical to the
# validated build.
FROM t212-vllm-oneapi2026:prod2

# --- eval harness (runs in-container; Py3.12 so HumanEval code_eval works) ---
RUN pip install --no-cache-dir "lm-eval[api]==0.4.12" transformers evaluate langdetect immutabledict nltk datasets \
    && python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"

COPY scripts/            /work/repro/scripts/
COPY bench_inner.py      /work/repro/bench_inner.py
COPY datasets/           /work/repro/datasets/
COPY hfcache/            /root/.cache/huggingface/

ENV HF_ALLOW_CODE_EVAL=1
LABEL repro="qwen36-b70-ship" stack="oneapi-2026" built="2026-07-24"
