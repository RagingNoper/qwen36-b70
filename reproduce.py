#!/usr/bin/env python3
"""
Reproduce the Qwen3.6-35B-A3B / Arc B70 benchmark + capability numbers.

  python3 reproduce.py --config int8-tp2 --model /path/to/Qwen3.6-35B-A3B
  python3 reproduce.py --config bf16-tp4 --model /path/to/Qwen3.6-35B-A3B --suite quick

Needs only: Docker w/ Intel GPU access, the loaded `qwen36-b70-ship` image, and the model on disk.
Stdlib only. Boots the server, waits, runs the in-container benchmarks, prints results, tears down.
"""
import argparse, subprocess, sys, time, urllib.request, urllib.error

IMAGE = "ghcr.io/ragingnoper/qwen36-b70-ship:latest"; NAME = "qwen36-repro"; PORT = 8107
CFG = ('{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","splitting_ops":["vllm::unified_attention",'
       '"vllm::unified_attention_with_output","vllm::gdn_attention_core","vllm::mamba_mixer2",'
       '"vllm::mamba_mixer","vllm::linear_attention"],"pass_config":{"fuse_norm_quant":false,'
       '"fuse_act_quant":false,"fuse_attn_quant":false,"enable_sp":false,"fuse_gemm_comms":false,'
       '"fuse_allreduce_rms":false,"enable_qk_norm_rope_fusion":false,"fuse_rope_kvcache_cat_mla":false,'
       '"fuse_act_padding":false,"fuse_mla_dual_rms_norm":false,"fuse_rope_kvcache":false},'
       '"cudagraph_capture_sizes":[1,2,4,8,16,32]}')
COMMON_ENV = ["VLLM_USE_V2_MODEL_RUNNER=1", "VLLM_SPEC_EAGER=1", "VLLM_XPU_ENABLE_XPU_GRAPH=1",
              "VLLM_XPU_CUSTOM_AR=1", "CCL_ENABLE_SYCL_KERNELS=1", "TORCHINDUCTOR_CACHE_DIR=/tmp/ind",
              "HF_HUB_OFFLINE=1", "HF_DATASETS_OFFLINE=1"] + \
             [f"DISABLE_ESIMD_{x}=1" for x in ("FUSED_INPUT", "QKV", "ATTN_GEMV", "DENSE", "MOE")]
CONFIGS = {
    "bf16-tp4": dict(devices="0,1,2,3", shm="64g", env=["VLLM_XPU_CAR_DMA=1"],
                     serve=["--tensor-parallel-size", "4", "--dtype", "bfloat16",
                            "--gpu-memory-utilization", "0.80"]),
    "int8-tp2": dict(devices="2,3", shm="32g", env=["VLLM_INT8_GEMV_MAX_T=1", "VLLM_INT8_LMHEAD=1"],
                     serve=["--tensor-parallel-size", "2", "--dtype", "bfloat16",
                            "--quantization", "experts_int8", "--gpu-memory-utilization", "0.88"]),
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
    cmd = (["docker", "run", "-d", "--name", NAME, "--device", "/dev/dri", "--ipc", "host",
            "--shm-size", c["shm"], "-p", f"{PORT}:{PORT}"]
           + sum([["-e", e] for e in env], [])
           + ["-v", f"{a.model}:/model", "--entrypoint", "", IMAGE,
              "vllm", "serve", "/model", "--max-model-len", "40960", "--max-num-batched-tokens", "4096",
              "--speculative-config", '{"method":"mtp","num_speculative_tokens":2}',
              "--compilation-config", CFG, "--port", str(PORT), "--served-model-name", "qwen3.6-35b-a3b"]
           + c["serve"])
    if a.config == "int8-tp2" and a.devices is None:
        print("[reproduce] NOTE: int8-tp2 uses 2 GPUs; prefill speed depends on the PCIe bandwidth\n"
              "           between them. Default is GPUs 2,3 (a high-P2P pair on the reference box).\n"
              "           2-GPU box? add `--devices 0,1`. Prefill slow? try another same-root pair.", flush=True)
    print(f"[reproduce] booting {a.config} on GPUs {dev} ...", flush=True)
    if sh(cmd, capture_output=True).returncode != 0:
        print("docker run failed"); sys.exit(1)

    for i in range(200):
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
