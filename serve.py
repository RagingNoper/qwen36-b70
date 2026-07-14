#!/usr/bin/env python3
"""
Just serve the model — boot it and leave it running so you can point any chat UI / app at it.

  python3 serve.py --config int8-tp2 --model /path/to/Qwen3.6-35B-A3B
  python3 serve.py --config bf16-tp4 --model /path/to/Qwen3.6-35B-A3B --port 8107

It prints the connection details once the server is ready, then leaves it running in the background.
Stop it later with:  docker rm -f qwen36-serve
Stdlib only. (Same server config as reproduce.py, minus the benchmarks and the teardown.)
"""
import argparse, subprocess, sys, time, urllib.request, urllib.error

IMAGE = "ghcr.io/ragingnoper/qwen36-b70-ship:latest"; NAME = "qwen36-serve"
CFG = ('{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","splitting_ops":["vllm::unified_attention",'
       '"vllm::unified_attention_with_output","vllm::gdn_attention_core","vllm::mamba_mixer2",'
       '"vllm::mamba_mixer","vllm::linear_attention"],"pass_config":{"fuse_norm_quant":false,'
       '"fuse_act_quant":false,"fuse_attn_quant":false,"enable_sp":false,"fuse_gemm_comms":false,'
       '"fuse_allreduce_rms":false,"enable_qk_norm_rope_fusion":false,"fuse_rope_kvcache_cat_mla":false,'
       '"fuse_act_padding":false,"fuse_mla_dual_rms_norm":false,"fuse_rope_kvcache":false},'
       '"cudagraph_capture_sizes":[1,2,4,8,16,32]}')
COMMON_ENV = ["VLLM_USE_V2_MODEL_RUNNER=1", "VLLM_SPEC_EAGER=1", "VLLM_XPU_ENABLE_XPU_GRAPH=1",
              "VLLM_XPU_CUSTOM_AR=1", "CCL_ENABLE_SYCL_KERNELS=1", "TORCHINDUCTOR_CACHE_DIR=/tmp/ind"] + \
             [f"DISABLE_ESIMD_{x}=1" for x in ("FUSED_INPUT", "QKV", "ATTN_GEMV", "DENSE", "MOE")]
CONFIGS = {
    "bf16-tp4": dict(devices="0,1,2,3", shm="64g", env=["VLLM_XPU_CAR_DMA=1"],
                     serve=["--tensor-parallel-size", "4", "--dtype", "bfloat16",
                            "--gpu-memory-utilization", "0.80"]),
    "int8-tp2": dict(devices="2,3", shm="32g", env=["VLLM_INT8_GEMV_MAX_T=1", "VLLM_INT8_LMHEAD=1"],
                     serve=["--tensor-parallel-size", "2", "--dtype", "bfloat16",
                            "--quantization", "experts_int8", "--gpu-memory-utilization", "0.88"]),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", default="8107")
    ap.add_argument("--devices", default=None, help="override which GPUs, e.g. 0,1")
    a = ap.parse_args()
    c = CONFIGS[a.config]; dev = a.devices or c["devices"]; port = a.port

    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    env = [f"ZE_AFFINITY_MASK={dev}"] + COMMON_ENV + c["env"]
    cmd = (["docker", "run", "-d", "--name", NAME, "--restart", "unless-stopped", "--device", "/dev/dri",
            "--ipc", "host", "--shm-size", c["shm"], "-p", f"{port}:{port}"]
           + sum([["-e", e] for e in env], [])
           + ["-v", f"{a.model}:/model", "--entrypoint", "", IMAGE,
              "vllm", "serve", "/model", "--max-model-len", "40960", "--max-num-batched-tokens", "4096",
              "--speculative-config", '{"method":"mtp","num_speculative_tokens":2}',
              "--compilation-config", CFG, "--host", "0.0.0.0", "--port", str(port),
              "--served-model-name", "qwen3.6-35b-a3b"]
           + c["serve"])
    print(f"[serve] starting {a.config} on GPUs {dev} ... (first start compiles kernels, ~5-10 min)", flush=True)
    if subprocess.run(cmd, capture_output=True).returncode != 0:
        print("docker run failed"); sys.exit(1)

    for i in range(200):
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
  MODEL IS SERVING and will keep running (even across reboots).

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
