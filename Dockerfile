# Self-contained reproduction image for Qwen3.6-35B-A3B on 4x Intel Arc Pro B70 (vLLM-XPU).
# Bakes in every runtime patch + the eval harness so a reproducer needs only: this image + the model.
FROM t212-vllm-graph-head2-mtp

# --- custom all-reduce (DMA copy-engine) + TP communicator ---
COPY patches/custom_ar.so               /work/ext/custom_ar.so
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

# --- eval harness (runs in-container; Py3.12 so HumanEval code_eval works) ---
RUN pip install --no-cache-dir "lm-eval[api]==0.4.12" transformers evaluate langdetect immutabledict nltk datasets \
    && python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"

COPY scripts/            /work/repro/scripts/
COPY bench_inner.py      /work/repro/bench_inner.py
COPY datasets/           /work/repro/datasets/
COPY hfcache/            /root/.cache/huggingface/

ENV HF_ALLOW_CODE_EVAL=1
LABEL repro="qwen36-b70-ship" built="2026-07-13"
