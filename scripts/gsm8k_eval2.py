"""GSM8K v2 — bigger token budget, strict answer-format instruction, robust extraction.
args: URL LABEL N CONC TEMP"""
import sys, json, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL   = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8100/v1/chat/completions"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "run"
N     = int(sys.argv[3]) if len(sys.argv) > 3 else 200
CONC  = int(sys.argv[4]) if len(sys.argv) > 4 else 32
TEMP  = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6
MODEL = sys.argv[6] if len(sys.argv) > 6 else "qwen3.6-35b-a3b"
MAXTOK = 4096

rows = [json.loads(l) for l in open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)),"gsm8k_test.jsonl"))][:N]

INSTR = ("\n\nSolve it step by step. Then, on the final line, output your answer in EXACTLY "
         "this format:\n#### <number>\nwhere <number> is the final numeric answer only — no units, "
         "no commas, no extra words.")

def gold_of(a): return a.split("####")[-1].strip().replace(",", "")

def extract(content, reasoning):
    for t in (content, reasoning):
        if not t: continue
        m = re.findall(r"####\s*\$?(-?[\d,]+\.?\d*)", t)
        if m: return m[-1].replace(",", "").rstrip(".")
        m = re.findall(r"\\boxed\{\s*\$?(-?[\d,]+\.?\d*)", t)
        if m: return m[-1].replace(",", "").rstrip(".")
        m = re.findall(r"(?:answer|total|result)\s*(?:is|=|:)\s*\$?(-?[\d,]+\.?\d*)", t, re.I)
        if m: return m[-1].replace(",", "").rstrip(".")
    # last-resort: last number in content (then reasoning)
    for t in (content, reasoning):
        if not t: continue
        m = re.findall(r"-?\$?\d[\d,]*\.?\d*", t.replace(",", ""))
        if m: return m[-1].replace("$", "").rstrip(".")
    return None

def ask(i, q):
    body = {"model": MODEL, "messages": [{"role": "user", "content": q + INSTR}],
            "max_tokens": MAXTOK, "temperature": TEMP, "seed": 0, "stream": False}
    if TEMP > 0:
        body["top_p"] = 0.95; body["top_k"] = 20
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        m = r["choices"][0]["message"]
        pred = extract(m.get("content") or "", m.get("reasoning_content") or m.get("reasoning") or "")
        return i, pred, r.get("usage", {}).get("completion_tokens", 0), None
    except Exception as e:
        return i, None, 0, str(e)[:60]

def num_eq(a, b):
    try: return abs(float(a) - float(b)) < 1e-4
    except: return str(a) == str(b)

def main():
    golds = [gold_of(r["answer"]) for r in rows]
    correct = done = errs = toks = trunc = 0
    res = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        futs = [ex.submit(ask, i, r["question"]) for i, r in enumerate(rows)]
        for f in as_completed(futs):
            i, pred, ct, err = f.result()
            done += 1; toks += ct
            if err: errs += 1
            if ct >= MAXTOK: trunc += 1
            ok = pred is not None and num_eq(pred, golds[i]); res[i] = ok
            if ok: correct += 1
            if done % 40 == 0:
                print(f"  [{LABEL}] {done}/{len(rows)} acc={correct/done*100:.1f}%", flush=True)
    dt = time.time() - t0; p = correct/len(rows); se = (p*(1-p)/len(rows))**0.5*100
    print(f">>> [{LABEL}] temp={TEMP} GSM8K = {p*100:.1f}% ({correct}/{len(rows)})  ±{se:.1f}pp  "
          f"trunc={trunc}  errs={errs}  {dt:.0f}s", flush=True)
    json.dump({"label": LABEL, "temp": TEMP, "acc": p*100, "correct": correct, "n": len(rows),
               "se_pp": se, "trunc": trunc, "errs": errs, "per_q": res, "secs": dt},
              open(f"/tmp/gsm8k_{LABEL}.json", "w"))

if __name__ == "__main__":
    main()
