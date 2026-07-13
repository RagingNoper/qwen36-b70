#!/usr/bin/env python3
# Honest single-stream decode wall-clock: same fixed prompt, time a 1-token request and a 512-token
# request; decode t/s = (512-1 output tokens) / (t512 - t1). Cancels prefill+TTFT, counts REAL tokens
# from usage (so MTP's speculative bursts are measured by wall clock, not flattered by median-TPOT).
import sys, json, time, urllib.request, statistics
URL, MODEL = sys.argv[1], sys.argv[2]
NTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 512
PROMPT = ("You are a helpful assistant. Write a long, detailed technical essay about the history of "
          "computing hardware, from mechanical calculators to modern GPUs and accelerators. Begin:")
def run(ntok):
    body = json.dumps({"model": MODEL, "prompt": PROMPT, "max_tokens": ntok,
                       "temperature": 0, "ignore_eos": True}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        obj = json.load(r)
    return time.time() - t0, obj["usage"]["completion_tokens"]
run(8)  # warm
res = []
for _ in range(3):
    t1, c1 = run(1)
    tN, cN = run(NTOK)
    res.append((cN - c1) / (tN - t1))
print(f"decode wall-clock t/s = {statistics.median(res):.1f}  "
      f"(runs {[round(x,1) for x in res]}, prefill+1tok≈{t1*1000:.0f}ms, out_tokens={cN})")
