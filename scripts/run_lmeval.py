#!/usr/bin/env python3
# lm-eval wrapper: (1) inject chat_template_kwargs={"enable_thinking":False} so the reasoning model
# answers cleanly; (2) retry evaluate.load (HuggingFace hub intermittently resets, and humaneval's
# code_eval metric loads at import with no retry of its own).
import time
import evaluate
_eload = evaluate.load
def _retry_load(*a, **k):
    for i in range(12):
        try:
            return _eload(*a, **k)
        except Exception as e:
            if i == 11:
                raise
            time.sleep(3)
evaluate.load = _retry_load

import lm_eval.models.openai_completions as oc
_orig = oc.LocalChatCompletion._create_payload
def _patched(self, *a, **k):
    p = _orig(self, *a, **k)
    if isinstance(p, dict):
        p["chat_template_kwargs"] = {"enable_thinking": False}
    return p
oc.LocalChatCompletion._create_payload = _patched

from lm_eval.__main__ import cli_evaluate
cli_evaluate()
