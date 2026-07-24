#!/usr/bin/env python3
"""
Just serve the model — boot one of the four tuned configs and leave it running so you can point
any chat UI / app at it.

  python3 serve.py --config int8-tp4-latency     --model /path/to/Qwen3.6-35B-A3B   # fastest single response (4 cards)
  python3 serve.py --config int8-tp2             --model /path/to/Qwen3.6-35B-A3B   # fast single response / 2 cards
  python3 serve.py --config int8-tp4-concurrency --model /path/to/Qwen3.6-35B-A3B   # many users at once / huge KV cache
  python3 serve.py --config bf16-tp4             --model /path/to/Qwen3.6-35B-A3B   # full-precision weights

Which one?
  int8-tp4-latency     : 4x B70. int8 + MTP spec-decode + custom all-reduce -> the FASTEST single response
                         (~201 tok/s) at 256K context. Pick this with 4 cards and mostly one user. ~1.47M-token KV.
  int8-tp2             : 2x B70 (64 GB). int8 + MTP -> ~196 tok/s single response at 256K context. Pick this on a
                         2-card box, or to leave the other two cards free. ~614k-token KV.
  int8-tp4-concurrency : 4x B70. int8, NO spec-decode, turboquant int8 KV, throughput-tuned oneCCL all-reduce ->
                         ~990 tok/s at 64 concurrent and the biggest KV cache (~4.14M tokens). Host many users.
  bf16-tp4             : 4x B70, full bf16 weights + MTP. In the box for completeness; int8 matches or beats it
                         everywhere except raw prefill. ~612k-token KV.

Capability (quality) is identical across all four — int8 is quant-free vs bf16 here. See GETTING_STARTED.md.
All four enable prefix caching, so long multi-turn sessions only prefill new tokens (prior turns are cached).
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

# Common env for every config (new stack: oneAPI 2026.1 / torch 2.13 / oneCCL 2022, V2 runner, graph mode).
# CCL_WORKER_COUNT=1 is required — a second worker uses scheduler algos that can't record into the graph capture.
COMMON_ENV = ["VLLM_USE_V2_MODEL_RUNNER=1", "VLLM_SPEC_EAGER=1", "VLLM_XPU_ENABLE_XPU_GRAPH=1",
              "VLLM_WORKER_MULTIPROC_METHOD=spawn", "CCL_ENABLE_SYCL_KERNELS=1",
              "CCL_ZE_IPC_EXCHANGE=sockets", "CCL_WORKER_COUNT=1", "TORCHINDUCTOR_CACHE_DIR=/tmp/ind"] + \
             [f"DISABLE_ESIMD_{x}=1" for x in ("FUSED_INPUT", "QKV", "ATTN_GEMV", "DENSE", "MOE")]
# Custom all-reduce, capture+small route: the 16-byte-vectorized reduce (reduce-scatter/all-gather) handles the
# graph-captured decode + small prefill embed AR; larger prefill ARs fall through to oneCCL. Used by the three
# MTP configs. The concurrency config does NOT set this — at high batch, oneCCL's collective wins (see writeup).
CAR_ENV = ["VLLM_XPU_CUSTOM_AR=1", "CAR_ROUTE=capture+small", "VLLM_XPU_CAR_DMA=1",
           "VLLM_XPU_CAR_SO=/work/ext/custom_ar_v4.so", "VLLM_XPU_CAR_RSAG_MIN=65536",
           "VLLM_XPU_CAR_DMA_MIN=9999999999", "VLLM_XPU_CAR_MAX=16777216"]

# per-config: which GPUs, shm, extra env (AR + int8 kernel flags), vllm serve args, MTP depth (spec_k / None),
# capture ladder, max-num-batched-tokens, max-model-len, and optional kv-cache dtype. Prefix caching is on for
# all four. The autotuned int8 MoE configs and all kernel patches are baked into the image (see VERSIONS.md).
CONFIGS = {
    # FASTEST SINGLE RESPONSE: int8 experts, 4 cards, custom AR (capture+small), MTP (k=3) + widened batch-1 MoE
    # GEMV (GEMV_MAX_T=4 keeps the k+1 speculative-verify batch on the fast kernel). ~201 tok/s at 256K context.
    "int8-tp4-latency": dict(devices="0,1,2,3", shm="64g",
                     env=["VLLM_INT8_GEMV_MAX_T=4", "VLLM_INT8_LMHEAD=1"] + CAR_ENV,
                     ladder="4,8,16,32", mbt="4096", maxlen="262144", spec_k=3, kvd=None,
                     serve=["--tensor-parallel-size", "4", "--dtype", "bfloat16",
                            "--quantization", "experts_int8", "--gpu-memory-utilization", "0.85"]),
    # 2-CARD: int8 experts, 2 cards, custom AR, MTP (k=3) + widened GEMV. ~196 tok/s at 256K context. 64 GB.
    "int8-tp2": dict(devices="2,3", shm="32g",
                     env=["VLLM_INT8_GEMV_MAX_T=4", "VLLM_INT8_LMHEAD=1"] + CAR_ENV,
                     ladder="4,8,16,32", mbt="4096", maxlen="262144", spec_k=3, kvd=None,
                     serve=["--tensor-parallel-size", "2", "--dtype", "bfloat16",
                            "--quantization", "experts_int8", "--gpu-memory-utilization", "0.88"]),
    # MANY USERS: int8 experts, 4 cards, pure oneCCL all-reduce (wins at batch), NO spec-decode, turboquant int8
    # KV (2.25x capacity), wide ladder + fat batches. ~990 tok/s at 64 concurrent, ~4.14M-token KV cache.
    "int8-tp4-concurrency": dict(devices="0,1,2,3", shm="64g",
                     env=["VLLM_INT8_GEMV_MAX_T=4", "VLLM_INT8_LMHEAD=1"],
                     ladder="1,2,4,8,16,32,48,64", mbt="8192", maxlen="65536", spec_k=None, kvd="turboquant_k8v4",
                     serve=["--tensor-parallel-size", "4", "--dtype", "bfloat16",
                            "--quantization", "experts_int8", "--gpu-memory-utilization", "0.88"]),
    # FULL PRECISION: bf16 weights, 4 cards, custom AR, MTP (k=3). Kept for completeness; int8 matches or beats it
    # everywhere except raw prefill (bf16's mature vendor GEMM leads there). ~197 tok/s at 256K context.
    "bf16-tp4": dict(devices="0,1,2,3", shm="64g",
                     env=list(CAR_ENV),
                     ladder="4,8,16,32", mbt="4096", maxlen="262144", spec_k=3, kvd=None,
                     serve=["--tensor-parallel-size", "4", "--dtype", "bfloat16",
                            "--gpu-memory-utilization", "0.78"]),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", default="8107")
    ap.add_argument("--max-model-len", default=None, help="override the config's default context length")
    ap.add_argument("--devices", default=None, help="override which GPUs, e.g. 0,1")
    a = ap.parse_args()
    c = CONFIGS[a.config]; dev = a.devices or c["devices"]; port = a.port
    maxlen = a.max_model_len or c["maxlen"]

    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    env = [f"ZE_AFFINITY_MASK={dev}"] + COMMON_ENV + c["env"]
    serve = (["vllm", "serve", "/model", "--max-model-len", maxlen,
              "--max-num-batched-tokens", c["mbt"], "--enable-prefix-caching"]
             + (["--kv-cache-dtype", c["kvd"]] if c["kvd"] else [])
             + (mtp_arg(c["spec_k"]) if c["spec_k"] else [])
             + ["--compilation-config", cfg_json(c["ladder"]), "--host", "0.0.0.0", "--port", str(port),
                "--served-model-name", "qwen3.6-35b-a3b"]
             + c["serve"])
    cmd = (["docker", "run", "-d", "--name", NAME, "--restart", "unless-stopped", "--device", "/dev/dri",
            "-v", "/dev/dri/by-path:/dev/dri/by-path",  # oneCCL 2022 enumerates GPUs via by-path
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
