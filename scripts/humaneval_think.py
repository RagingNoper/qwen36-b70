#!/usr/bin/env python3
# Self-contained HumanEval pass@1 (subprocess exec + timeout; no code_eval/multiprocessing -> Py3.14-safe).
# Chat endpoint, enable_thinking=False. Usage: humaneval_eval.py URL TAG N CONC MODEL
import sys, json, re, subprocess, urllib.request, concurrent.futures, os
from datasets import load_dataset
URL, TAG, N, CONC, MODEL = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
import time
for _i in range(12):
    try: items = list(load_dataset("openai/openai_humaneval", split="test")); break
    except Exception as e:
        if _i==11: raise
        time.sleep(3)
if N > 0: items = items[:N]
def gen(prompt):
    body = json.dumps({"model":MODEL,"max_tokens":12000,"temperature":0,
        "messages":[{"role":"user","content":
          "Complete this Python function. Return ONLY the full function definition (with signature) in a single ```python code block.\n\n```python\n"+prompt+"\n```"}]}).encode()
    req=urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=900) as r:
        return json.load(r)["choices"][0]["message"]["content"]
def extract(text):
    text = __import__("re").sub(r"(?s)^.*?</think>","",text)
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m[0] if m else text
def check(item):
    try:
        code = extract(gen(item["prompt"]))
        pre = "\n".join(l for l in item["prompt"].splitlines() if l.startswith(("import ","from ")))
        prog = pre+"\n"+code+"\n"+item["test"]+f"\ncheck({item['entry_point']})\n"
        r = subprocess.run(["python3","-c",prog], capture_output=True, timeout=20)
        return r.returncode == 0
    except Exception:
        return False
with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as ex:
    res = list(ex.map(check, items))
p = sum(res)
print(f">>> [{TAG}] HumanEval pass@1 = {p/len(items)*100:.1f}% ({p}/{len(items)})")
