#!/usr/bin/env python3
"""
Reproduce the Qwen3.6-35B-A3B / Arc B70 benchmark + capability numbers.

  python3 reproduce.py --config int8-tp4-concurrency --model /path/to/Qwen3.6-35B-A3B
  python3 reproduce.py --config int8-tp2 --model /path/to/Qwen3.6-35B-A3B --suite quick

Configs: int8-tp4-latency | int8-tp2 | int8-tp4-concurrency | bf16-tp4

Needs only: Docker w/ Intel GPU access, the loaded `qwen36-b70-ship` image, and the model on disk.
Stdlib only. Boots the server, waits, runs the in-container benchmarks, prints results, tears down.
"""
import argparse, subprocess, sys, time, urllib.request, urllib.error

IMAGE = "ghcr.io/ragingnoper/qwen36-b70-ship:latest"; NAME = "qwen36-repro"; PORT = 8107
SPLIT = ('"vllm::unified_attention","vllm::unified_attention_with_output","vllm::gdn_attention_core",'
         '"vllm::mamba_mixer2","vllm::mamba_mixer","vllm::linear_attention"')
PASS = ('"pass_config":{"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false,'
        '"enable_sp":false,"fuse_gemm_comms":false,"fuse_allreduce_rms":false,'
        '"enable_qk_norm_rope_fusion":false,"fuse_rope_kvcache_cat_mla":false,"fuse_act_padding":false,'
        '"fuse_mla_dual_rms_norm":false,"fuse_rope_kvcache":false}')
def cfg_json(ladder):
    return ('{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","splitting_ops":[' + SPLIT + '],' + PASS +
            ',"cudagraph_capture_sizes":[' + ladder + ']}')
def mtp_arg(k):
    return ["--speculative-config", '{"method":"mtp","num_speculative_tokens":%d}' % k]
# new stack (oneAPI 2026.1 / torch 2.13 / oneCCL 2022). CCL_WORKER_COUNT=1 is required (graph capture).
COMMON_ENV = ["VLLM_USE_V2_MODEL_RUNNER=1", "VLLM_SPEC_EAGER=1", "VLLM_XPU_ENABLE_XPU_GRAPH=1",
              "VLLM_WORKER_MULTIPROC_METHOD=spawn", "CCL_ENABLE_SYCL_KERNELS=1", "CCL_ZE_IPC_EXCHANGE=sockets",
              "CCL_WORKER_COUNT=1", "TORCHINDUCTOR_CACHE_DIR=/tmp/ind",
              "HF_HUB_OFFLINE=1", "HF_DATASETS_OFFLINE=1"] + \
             [f"DISABLE_ESIMD_{x}=1" for x in ("FUSED_INPUT", "QKV", "ATTN_GEMV", "DENSE", "MOE")]
# custom all-reduce, capture+small route (the three MTP configs; the concurrency config uses pure oneCCL).
CAR_ENV = ["VLLM_XPU_CUSTOM_AR=1", "CAR_ROUTE=capture+small", "VLLM_XPU_CAR_DMA=1",
           "VLLM_XPU_CAR_SO=/work/ext/custom_ar_v4.so", "VLLM_XPU_CAR_RSAG_MIN=65536",
           "VLLM_XPU_CAR_DMA_MIN=9999999999", "VLLM_XPU_CAR_MAX=16777216"]
CONFIGS = {
    "int8-tp4-latency": dict(devices="0,1,2,3", shm="64g",
                     env=["VLLM_INT8_GEMV_MAX_T=4", "VLLM_INT8_LMHEAD=1"] + CAR_ENV,
                     ladder="4,8,16,32", mbt="4096", maxlen="262144", spec_k=3, kvd=None,
                     serve=["--tensor-parallel-size", "4", "--dtype", "bfloat16",
                            "--quantization", "experts_int8", "--gpu-memory-utilization", "0.85"]),
    "int8-tp2": dict(devices="2,3", shm="32g",
                     env=["VLLM_INT8_GEMV_MAX_T=4", "VLLM_INT8_LMHEAD=1"] + CAR_ENV,
                     ladder="4,8,16,32", mbt="4096", maxlen="262144", spec_k=3, kvd=None,
                     serve=["--tensor-parallel-size", "2", "--dtype", "bfloat16",
                            "--quantization", "experts_int8", "--gpu-memory-utilization", "0.88"]),
    "int8-tp4-concurrency": dict(devices="0,1,2,3", shm="64g",
                     env=["VLLM_INT8_GEMV_MAX_T=4", "VLLM_INT8_LMHEAD=1"],
                     ladder="1,2,4,8,16,32,48,64", mbt="8192", maxlen="65536", spec_k=None, kvd="turboquant_k8v4",
                     serve=["--tensor-parallel-size", "4", "--dtype", "bfloat16",
                            "--quantization", "experts_int8", "--gpu-memory-utilization", "0.88"]),
    "bf16-tp4": dict(devices="0,1,2,3", shm="64g", env=list(CAR_ENV),
                     ladder="4,8,16,32", mbt="4096", maxlen="262144", spec_k=3, kvd=None,
                     serve=["--tensor-parallel-size", "4", "--dtype", "bfloat16",
                            "--gpu-memory-utilization", "0.78"]),
}

def sh(args, **k): return subprocess.run(args, **k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--model", required=True, help="host path to the Qwen3.6-35B-A3B directory")
    ap.add_argument("--suite", default="full", choices=["full", "quick"])
    ap.add_argument("--devices", default=None, help="override ZE_AFFINITY_MASK, e.g. 0,1")
    a = ap.parse_args()
    c = CONFIGS[a.config]; dev = a.devices or c["devices"]

    sh(["docker", "rm", "-f", NAME], capture_output=True)
    env = [f"ZE_AFFINITY_MASK={dev}"] + COMMON_ENV + c["env"]
    serve = (["vllm", "serve", "/model", "--max-model-len", c["maxlen"], "--max-num-batched-tokens", c["mbt"],
              "--enable-prefix-caching"]
             + (["--kv-cache-dtype", c["kvd"]] if c["kvd"] else [])
             + (mtp_arg(c["spec_k"]) if c["spec_k"] else [])
             + ["--compilation-config", cfg_json(c["ladder"]), "--port", str(PORT),
                "--served-model-name", "qwen3.6-35b-a3b"]
             + c["serve"])
    cmd = (["docker", "run", "-d", "--name", NAME, "--device", "/dev/dri",
            "-v", "/dev/dri/by-path:/dev/dri/by-path",  # oneCCL 2022 enumerates GPUs via by-path
            "--ipc", "host", "--shm-size", c["shm"], "-p", f"{PORT}:{PORT}"]
           + sum([["-e", e] for e in env], [])
           + ["-v", f"{a.model}:/model", "--entrypoint", "", IMAGE] + serve)
    if a.config == "int8-tp2" and a.devices is None:
        print("[reproduce] NOTE: int8-tp2 uses 2 GPUs; prefill speed depends on the PCIe bandwidth\n"
              "           between them, and on a 4-card box the pairs are NOT equal. Default is GPUs 2,3\n"
              "           (a high-P2P pair on the reference box). 2-GPU box? add `--devices 0,1`.", flush=True)
    print(f"[reproduce] booting {a.config} on GPUs {dev} ...", flush=True)
    if sh(cmd, capture_output=True).returncode != 0:
        print("docker run failed"); sys.exit(1)

    for i in range(240):
        try:
            if urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=5).status == 200:
                print("[reproduce] server READY", flush=True); break
        except (urllib.error.URLError, OSError): pass
        if sh(["docker", "ps", "-q", "-f", f"name={NAME}"], capture_output=True, text=True).stdout.strip() == "":
            print("[reproduce] container died during boot:")
            sh(["docker", "logs", "--tail", "40", NAME]); sys.exit(1)
        time.sleep(15)
    else:
        print("[reproduce] boot timed out"); sys.exit(1)

    sh(["docker", "exec", NAME, "python3", "/work/repro/bench_inner.py",
        "--config", a.config, "--suite", a.suite])
    sh(["docker", "rm", "-f", NAME], capture_output=True)
    print("[reproduce] done.")

if __name__ == "__main__":
    main()
