# patches/ — the custom bits, for transparency

These are the custom kernels / integration files that make Qwen3.6-35B-A3B fast on Arc B70. **They are
already baked into the base image** (`t212-vllm-oneapi2026:prod2`) — the repo `Dockerfile` does **not**
re-copy them, so the published image stays byte-identical to the validated build. This folder exists so
you can see exactly what's custom.

| file | what it is | where it lives in the image |
|---|---|---|
| `custom_ar_v4.so` | custom all-reduce — 16-byte-vectorized reduce + reduce-scatter/all-gather; the `capture+small` route handles the graph-captured decode + small prefill embed AR | `/work/ext/custom_ar_v4.so` |
| `xpu_communicator.car.py` | TP communicator that routes collectives to the custom AR (`CAR_ROUTE`) vs oneCCL | `vllm/distributed/device_communicators/xpu_communicator.py` |
| `experts_int8.py`, `triton_moe_experts.py`, `online_int8.py`, `triton_moe_int8_native.py`, `_int8_linear.py` | `experts_int8` W8A8 MoE + batch-1 MoE GEMV kernels | `vllm/model_executor/layers/quantization/…` and `…/fused_moe/experts/triton_moe.py` |
| `moe-configs/*.json` | autotuned per-shape int8 MoE block configs (bit-identical output; closes the prefill / concurrency gap) | `vllm/model_executor/layers/fused_moe/configs/` |
| `gdn_attn.py` | GDN linear-attention backend with the cudagraph decode fix | `vllm/v1/attention/backends/gdn_attn.py` |
| `v2_speculator.py` | MTP speculative-decode drafter (V2 runner, graph-capture correct) | `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` |
| `_xpu_ops.py` | XPU op shims (MTP + GDN) | `vllm/_xpu_ops.py` |

Also baked into the base (integration plumbing, not shown here): the V2 model runner + mamba-align prefix
caching (enables `--enable-prefix-caching` on this hybrid model), the FULL_DECODE_ONLY breakable-cudagraph
path, the turboquant (int8) KV-cache dtype, and the sampler/warmup shims. See `../VERSIONS.md`.
