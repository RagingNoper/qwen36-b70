#!/usr/bin/env python3
# lm-eval wrapper for REASONING mode: thinking stays ON (we do NOT disable it), but the <think>...</think>
# scratchpad is stripped from each generation before the task scores it -> measures the ANSWER's
# instruction-following, not the scratchpad. Plus the evaluate.load retry (flaky HF hub).
import re, time
import evaluate
_eload = evaluate.load
def _retry_load(*a, **k):
    for i in range(12):
        try: return _eload(*a, **k)
        except Exception:
            if i == 11: raise
            time.sleep(3)
evaluate.load = _retry_load

import lm_eval.models.openai_completions as oc
_pg = oc.LocalChatCompletion.parse_generations
def _pg_strip(outputs, **k):
    res = _pg(outputs, **k)
    return [re.sub(r"(?s)^.*?</think>\s*", "", x).strip() for x in res]
oc.LocalChatCompletion.parse_generations = staticmethod(_pg_strip)

from lm_eval.__main__ import cli_evaluate
cli_evaluate()
