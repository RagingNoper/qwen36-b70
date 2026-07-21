# Self-contained reproduction image for Qwen3.6-35B-A3B on 4x Intel Arc Pro B70 (vLLM-XPU).
# Bakes in every runtime patch + the eval harness so a reproducer needs only: this image + the model.
#
# NOTE: this is a THIN layer on a prebuilt vLLM-XPU base image; the heavy stack below comes from the
# base, not from this Dockerfile (so it is not a from-scratch recipe). Exact versions in the image
# (see VERSIONS.md):
#   Ubuntu 24.04.3  |  Python 3.12.3  |  torch 2.12.0+xpu  |  vLLM 0.1.dev1+gdec860fb1  |  triton-xpu 3.7.1
#   oneAPI DPC++ 2025.3.2  |  oneCCL 2021.17.2  |  Intel compute runtime 26.09.37435.12  |  Level-Zero 1.28.0
# The container carries its OWN Intel GPU userspace runtime; the HOST only needs a Battlemage kernel
# driver + /dev/dri (host userspace version need not match — see VERSIONS.md).
FROM t212-vllm-graph-head2-mtp

# --- custom all-reduce + TP communicator ---
#   custom_ar.so     = DMA copy-engine reduce (default; used by int8-tp2, where 2-rank AR is already cheap)
#   custom_ar.so.v4  = vec-reduce + reduce-scatter/all-gather (int8-tp4-latency, int8-tp4-concurrency, bf16-tp4)
#   the communicator picks between them via the VLLM_XPU_CAR_SO env (default = the DMA .so)
COPY patches/custom_ar.so               /work/ext/custom_ar.so
COPY patches/custom_ar.so.v4            /work/ext/custom_ar.so.v4
COPY patches/xpu_communicator.car.py    /opt/vllm-main/vllm/distributed/device_communicators/xpu_communicator.py
# --- MTP + GDN cudagraph fix ---
COPY patches/_xpu_ops.py                /opt/vllm-main/vllm/_xpu_ops.py
COPY patches/v2_speculator.py           /opt/vllm-main/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
COPY patches/gdn_attn_nbfix.py          /opt/vllm-main/vllm/v1/attention/backends/gdn_attn.py
# --- int8 (experts_int8) kernels ---
COPY patches/triton_moe_experts.py      /opt/vllm-main/vllm/model_executor/layers/fused_moe/experts/triton_moe.py
COPY patches/online_int8.py             /opt/vllm-main/vllm/model_executor/layers/quantization/online/int8.py
COPY patches/triton_moe_int8_native.py  /opt/vllm-main/vllm/model_executor/layers/quantization/online/_int8_gemv.py
COPY patches/experts_int8.py            /opt/vllm-main/vllm/model_executor/layers/quantization/experts_int8.py
COPY patches/_int8_linear.py            /opt/vllm-main/vllm/model_executor/layers/quantization/online/_int8_linear.py
# --- autotuned int8 MoE block configs (per-shape; closes the prefill + concurrency gap, bit-identical output) ---
COPY patches/moe-configs/               /opt/vllm-main/vllm/model_executor/layers/fused_moe/configs/

# --- eval harness (runs in-container; Py3.12 so HumanEval code_eval works) ---
RUN pip install --no-cache-dir "lm-eval[api]==0.4.12" transformers evaluate langdetect immutabledict nltk datasets \
    && python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"

COPY scripts/            /work/repro/scripts/
COPY bench_inner.py      /work/repro/bench_inner.py
COPY datasets/           /work/repro/datasets/
COPY hfcache/            /root/.cache/huggingface/

ENV HF_ALLOW_CODE_EVAL=1
LABEL repro="qwen36-b70-ship" built="2026-07-18"
