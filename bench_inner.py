#!/usr/bin/env python3
"""Runs INSIDE the ship container against the local vLLM server. Reproduces the published
perf + capability numbers. Usage: bench_inner.py --config int8-tp2 --suite full|quick"""
import argparse, subprocess, re, sys, os
PORT = 8107; MODEL = "qwen3.6-35b-a3b"; D = "/work/repro"
SG = f"{D}/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

def sh(cmd, timeout=3600):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr

def g(txt, pat):
    m = re.search(pat + r".*?([0-9.]+)", txt)
    return float(m.group(1)) if m else float("nan")

def bench(dataset, il, ol, c, np, extra=""):
    o = sh(f"vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --host localhost "
           f"--port {PORT} --model /model --served-model-name {MODEL} --dataset-name {dataset} {extra} "
           f"--num-prompts {np} --max-concurrency {c} --seed 42 "
           f"--percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99")
    return dict(tput=g(o, "Output token throughput \\(tok/s\\):"), ttft=g(o, "Median TTFT \\(ms\\):"),
                p99=g(o, "P99 TTFT \\(ms\\):"), tpot=g(o, "Median TPOT \\(ms\\):"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="?"); ap.add_argument("--suite", default="full")
    a = ap.parse_args(); full = a.suite == "full"
    R = []
    print(f"\n===== REPRODUCTION: {a.config} / {a.suite} =====", flush=True)

    # ---- warmup: compile every prefill shape + warm decode/autotune so measured numbers are steady-state ----
    print("[warmup] compiling prefill shapes + warming kernels (first run only is slow) ...", flush=True)
    for L in ([16, 256, 1024, 2048, 4096] if full else [1024, 4096]):
        sh(f"vllm bench serve --backend openai --endpoint /v1/completions --host localhost --port {PORT} "
           f"--model /model --served-model-name {MODEL} --dataset-name random --random-input-len {L} "
           f"--random-output-len 8 --ignore-eos --num-prompts 4 --max-concurrency 1 --seed 1")

    # ---- prefill TTFT vs length ----
    print("[1/5] prefill TTFT sweep ...", flush=True)
    lens = [16, 128, 256, 1024, 2048, 4096] if full else [1024, 4096]
    for L in lens:
        o = sh(f"vllm bench serve --backend openai --endpoint /v1/completions --host localhost --port {PORT} "
               f"--model /model --served-model-name {MODEL} --dataset-name random --random-input-len {L} "
               f"--random-output-len 1 --ignore-eos --num-prompts 12 --max-concurrency 1 --seed 42")
        R.append(("prefill", f"TTFT@{L}", round(g(o, "Median TTFT \\(ms\\):"), 1), "ms"))

    # ---- static (fixed 1024/256) throughput/latency ----
    print("[2/5] static workload (1024-in/256-out) ...", flush=True)
    cs = [1, 8, 16, 32] if full else [1]
    for c in cs:
        r = bench("random", 1024, 256, c, 12 if c == 1 else c*8,
                  "--random-input-len 1024 --random-output-len 256 --ignore-eos")
        R += [("static", f"c{c} out_tput", round(r["tput"], 1), "t/s"),
              ("static", f"c{c} TTFT", round(r["ttft"], 0), "ms"),
              ("static", f"c{c} TPOT", round(r["tpot"], 2), "ms")]

    # ---- sharegpt (variable) ----
    if full:
        print("[3/5] ShareGPT workload ...", flush=True)
        for c in [1, 8, 16, 32]:
            r = bench("sharegpt", 0, 0, c, 12 if c == 1 else c*8, f"--dataset-path {SG}")
            R += [("sharegpt", f"c{c} out_tput", round(r["tput"], 1), "t/s"),
                  ("sharegpt", f"c{c} p99_TTFT", round(r["p99"], 0), "ms")]

    # ---- GSM8K ----
    print("[4/5] GSM8K ...", flush=True)
    o = sh(f"python3 {D}/scripts/gsm8k_eval2.py http://localhost:{PORT}/v1/chat/completions "
           f"repro-gsm {100 if full else 40} 16 0.0 {MODEL}")
    R.append(("capability", "GSM8K", g(o, "GSM8K = "), "%"))

    # ---- MMLU-Redux / IFEval / HumanEval (full only) — all in THINKING mode ----
    # This is a reasoning model; capability is measured with thinking ON and the <think> block stripped
    # before scoring. See the report for why standard MMLU / greedy / small budgets undersell it.
    if full:
        print("[5/5] MMLU-Redux + IFEval + HumanEval (thinking mode) ...", flush=True)
        TA = (f"model={MODEL},base_url=http://localhost:{PORT}/v1/chat/completions,"
              f"num_concurrent=16,max_retries=4,tokenized_requests=False,timeout=1800")
        # MMLU-Redux 2.0 (corrected-label MMLU), generative CoT. Subsampled for runtime; the published
        # 93.4% is the full 5330-question set, so expect a few points of sampling noise here.
        o = sh(f"python3 {D}/scripts/run_lmeval_think.py --model local-chat-completions "
               f"--model_args '{TA}' --tasks mmlu_redux_generative --apply_chat_template "
               f"--gen_kwargs max_gen_toks=32000 --limit 8", timeout=14400)
        m = re.search(r"mmlu_redux \(generative\).*?exact_match.*?([0-9.]+)", o)
        R.append(("capability", "MMLU-Redux", round(float(m.group(1))*100, 1) if m else float("nan"), "%"))
        # IFEval: thinking + the model's recommended sampling (temp 0.6/top_p 0.95, not greedy) + large
        # budget so reasoning traces are never truncated (both matter — see report).
        o = sh(f"python3 {D}/scripts/run_lmeval_think.py --model local-chat-completions "
               f"--model_args '{TA}' --tasks ifeval --apply_chat_template "
               f"--gen_kwargs do_sample=true,temperature=0.6,top_p=0.95,max_gen_toks=32000", timeout=14400)
        # IFEval reported as the average of its four sub-metrics (prompt/instruction x strict/loose).
        ifm = [g(o, m + r"\s*\|[^|]*\|") for m in
               ("prompt_level_strict_acc", "prompt_level_loose_acc",
                "inst_level_strict_acc", "inst_level_loose_acc")]
        R.append(("capability", "IFEval (avg of 4)", round(sum(ifm) / len(ifm) * 100, 1), "%"))
        # HumanEval pass@1, thinking mode.
        o = sh(f"python3 {D}/scripts/humaneval_think.py http://localhost:{PORT}/v1/chat/completions "
               f"repro 164 12 {MODEL}", timeout=7200)
        R.append(("capability", "HumanEval pass@1", g(o, "pass@1 = "), "%"))

    # ---- print ----
    print(f"\n\n========== RESULTS: {a.config} ==========")
    grp = None
    for cat, name, val, unit in R:
        if cat != grp: print(f"\n-- {cat} --"); grp = cat
        print(f"  {name:<22} {val:>8} {unit}")
    print("\n(compare against the published report; numbers should match within run-to-run noise)")

if __name__ == "__main__":
    main()
