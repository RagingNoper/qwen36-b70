#!/usr/bin/env python3
"""
Just serve the model — boot one of the four tuned configs and leave it running so you can point
any chat UI / app at it.

  python3 serve.py --config int8-tp4-latency     --model /path/to/Qwen3.6-35B-A3B   # fastest single response (4 cards)
  python3 serve.py --config int8-tp2             --model /path/to/Qwen3.6-35B-A3B   # fast single response / 2 cards
  python3 serve.py --config int8-tp4-concurrency --model /path/to/Qwen3.6-35B-A3B   # many users at once / huge KV cache
  python3 serve.py --config bf16-tp4             --model /path/to/Qwen3.6-35B-A3B   # full-precision weights

Which one?
  int8-tp4-latency     : 4x B70. int8 + MTP spec-decode -> the FASTEST single response (~206 tok/s). Pick this
                         if you have 4 cards and mostly one user at a time. ~816k-token KV cache.
  int8-tp2             : 2x B70 (64 GB). int8 + MTP -> ~174 tok/s single response AND the fastest prefill of any
                         config. Pick this on a 2-card box, or to leave the other two cards free. ~267k-token KV.
  int8-tp4-concurrency : 4x B70. int8, NO spec-decode, throughput-tuned all-reduce -> ~965 tok/s at 64 concurrent
                         requests and the biggest KV cache (~1.37M tokens). Pick this to host many users.
  bf16-tp4             : 4x B70, full bf16 weights + MTP. In the box for completeness; int8 matches or beats it
                         everywhere. ~380k-token KV cache.

Capability (quality) is identical across all four — int8 is quant-free vs bf16 here. See GETTING_STARTED.md.
It prints the connection details once ready, then leaves the server running in the background.
Stop it with:  docker rm -f qwen36-serve      Stdlib only.
"""
import argparse, subprocess, sys, time, urllib.request, urllib.error

IMAGE = "ghcr.io/ragingnoper/qwen36-b70-ship:latest"; NAME = "qwen36-serve"
SPLIT = ('"vllm::unified_attention","vllm::unified_attention_with_output","vllm::gdn_attention_core",'
         '"vllm::mamba_mixer2","vllm::mamba_mixer","vllm::linear_attention"')
PASS = ('"pass_config":{"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false,'
        '"enable_sp":false,"fuse_gemm_comms":false,"fuse_allreduce_rms":false,'
        '"enable_qk_norm_rope_fusion":false,"fuse_rope_kvcache_cat_mla":false,"fuse_act_padding":false,'
        '"fuse_mla_dual_rms_norm":false,"fuse_rope_kvcache":false}')
def cfg_json(ladder):
    return ('{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","splitting_ops":[' + SPLIT + '],' + PASS +
            ',"cudagraph_capture_sizes":[' + ladder + ']}')
def mtp_arg(k):  # MTP speculative decode drafting k tokens deep
    return ["--speculative-config", '{"method":"mtp","num_speculative_tokens":%d}' % k]
COMMON_ENV = ["VLLM_USE_V2_MODEL_RUNNER=1", "VLLM_SPEC_EAGER=1", "VLLM_XPU_ENABLE_XPU_GRAPH=1",
              "VLLM_XPU_CUSTOM_AR=1", "CCL_ENABLE_SYCL_KERNELS=1", "TORCHINDUCTOR_CACHE_DIR=/tmp/ind"] + \
             [f"DISABLE_ESIMD_{x}=1" for x in ("FUSED_INPUT", "QKV", "ATTN_GEMV", "DENSE", "MOE")]
# v4 all-reduce (16-byte-vectorized reduce + reduce-scatter/all-gather). CAR_DMA_MIN=huge keeps decode off the
# copy engine; CAR_RSAG_MIN turns on reduce-scatter/all-gather above 64 KiB.
V4_AR = ["VLLM_XPU_CAR_DMA=1", "VLLM_XPU_CAR_SO=/work/ext/custom_ar.so.v4",
         "VLLM_XPU_CAR_RSAG_MIN=65536", "VLLM_XPU_CAR_DMA_MIN=9999999999"]

# per-config: which GPUs, shm, extra env (AR + int8 kernel flags), vllm serve args, MTP depth (spec_k / None),
# capture ladder, and max-num-batched-tokens. The autotuned int8 MoE configs are baked into the image.
CONFIGS = {
    # FASTEST SINGLE RESPONSE: int8 experts, 4 cards, v4 all-reduce, MTP (k=3) + widened batch-1 MoE GEMV
    # (GEMV_MAX_T=4 so the k+1 speculative-verify batch stays on the fast kernel). ~206 tok/s single-stream.
    "int8-tp4-latency": dict(devices="0,1,2,3", shm="64g",
                     env=["VLLM_INT8_GEMV_MAX_T=4", "VLLM_INT8_LMHEAD=1"] + V4_AR,
                     ladder="1,2,3,4,5,8,16,32", mbt="4096", spec_k=3,
                     serve=["--tensor-parallel-size", "4", "--dtype", "bfloat16",
                            "--quantization", "experts_int8", "--gpu-memory-utilization", "0.85"]),
    # 2-CARD: int8 experts, 2 cards, DMA all-reduce (cheap at 2 ranks), MTP (k=3) + widened GEMV.
    # ~174 tok/s single-stream and the fastest prefill of any config. 64 GB.
    "int8-tp2": dict(devices="2,3", shm="32g",
                     env=["VLLM_INT8_GEMV_MAX_T=4", "VLLM_INT8_LMHEAD=1", "VLLM_XPU_CAR_DMA=1"],
                     ladder="1,2,3,4,5,8,16,32", mbt="4096", spec_k=3,
                     serve=["--tensor-parallel-size", "2", "--dtype", "bfloat16",
                            "--quantization", "experts_int8", "--gpu-memory-utilization", "0.88"]),
    # MANY USERS: int8 experts, 4 cards, v4 all-reduce, NO spec-decode, wide ladder + fat batches.
    # ~965 tok/s at 64 concurrent and the biggest KV cache (~1.37M tokens).
    "int8-tp4-concurrency": dict(devices="0,1,2,3", shm="64g",
                     env=["VLLM_INT8_GEMV_MAX_T=1", "VLLM_INT8_LMHEAD=1"] + V4_AR,
                     ladder="1,2,4,8,16,32,48,64,100", mbt="16384", spec_k=None,
                     serve=["--tensor-parallel-size", "4", "--dtype", "bfloat16",
                            "--quantization", "experts_int8", "--gpu-memory-utilization", "0.88"]),
    # FULL PRECISION: bf16 weights, 4 cards, v4 all-reduce, MTP (k=3). Kept for completeness.
    "bf16-tp4": dict(devices="0,1,2,3", shm="64g",
                     env=list(V4_AR),
                     ladder="1,2,3,4,8,16,32", mbt="4096", spec_k=3,
                     serve=["--tensor-parallel-size", "4", "--dtype", "bfloat16",
                            "--gpu-memory-utilization", "0.80"]),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", default="8107")
    ap.add_argument("--max-model-len", default="40960")
    ap.add_argument("--devices", default=None, help="override which GPUs, e.g. 0,1")
    a = ap.parse_args()
    c = CONFIGS[a.config]; dev = a.devices or c["devices"]; port = a.port

    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    env = [f"ZE_AFFINITY_MASK={dev}"] + COMMON_ENV + c["env"]
    serve = (["vllm", "serve", "/model", "--max-model-len", a.max_model_len,
              "--max-num-batched-tokens", c["mbt"]]
             + (mtp_arg(c["spec_k"]) if c["spec_k"] else [])
             + ["--compilation-config", cfg_json(c["ladder"]), "--host", "0.0.0.0", "--port", str(port),
                "--served-model-name", "qwen3.6-35b-a3b"]
             + c["serve"])
    cmd = (["docker", "run", "-d", "--name", NAME, "--restart", "unless-stopped", "--device", "/dev/dri",
            "--ipc", "host", "--shm-size", c["shm"], "-p", f"{port}:{port}"]
           + sum([["-e", e] for e in env], [])
           + ["-v", f"{a.model}:/model", "--entrypoint", "", IMAGE] + serve)
    print(f"[serve] starting {a.config} on GPUs {dev} ... (first start compiles kernels, ~8-12 min)", flush=True)
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print("docker run failed:\n" + r.stderr.decode(errors="replace").strip()); sys.exit(1)

    for i in range(240):
        try:
            if urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5).status == 200:
                break
        except (urllib.error.URLError, OSError): pass
        if subprocess.run(["docker", "ps", "-q", "-f", f"name={NAME}"], capture_output=True,
                          text=True).stdout.strip() == "":
            print("[serve] container died during boot:")
            subprocess.run(["docker", "logs", "--tail", "40", NAME]); sys.exit(1)
        time.sleep(15)
    else:
        print("[serve] boot timed out"); sys.exit(1)

    print(f"""
==================================================================
  MODEL IS SERVING ({a.config}) and will keep running (even across reboots).

  API base URL : http://<this-machine-ip>:{port}/v1
  (on this machine: http://localhost:{port}/v1)
  Model name   : qwen3.6-35b-a3b
  API key      : none (leave blank, or use any dummy value)

  It speaks the OpenAI API, so point ANY OpenAI-compatible chat UI
  or app at the URL above (Open WebUI, LibreChat, the openai python
  library, etc.). See GETTING_STARTED.md -> "Using the model".

  Stop it with:  docker rm -f {NAME}
==================================================================
""")

if __name__ == "__main__":
    main()
