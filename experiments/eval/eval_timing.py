"""Warm answer time per model on the 20 answer scenarios.

Each model is loaded with a short warm-up call first, so the times
measure answering only, not loading the model from disk.

    python experiments/eval/eval_timing.py
"""

import time

import common  # noqa: F401  (sets the project root and working directory)
from common import save
from langchain_ollama import ChatOllama

import eval_responses as er
from services import llm_service
from services.llm_service import stream_answer
from services.prompt_service import build_healthcare_prompt

MODELS = ["llama3.2", "llama3", "qwen2.5:7b-instruct"]


def main():
    out = {}
    for model in MODELS:
        llm_service.llm = ChatOllama(model=model, temperature=0, num_ctx=8192)
        "".join(stream_answer("Say OK.", "You are a test."))  # warm-up: load the model
        secs, passed = [], 0
        for sc in er.SCENARIOS:
            sid, name, key, q, ref, every, check = sc[:7]
            hist = sc[7] if len(sc) > 7 else None
            ctx = er.report(key) if key else er.LABEL
            ver = er.verified_for(ctx, q, every) if key else None
            system, prompt = build_healthcare_prompt(ctx, q, history=hist, verified=ver,
                                                     reference_context=ref)
            t = time.monotonic()
            answer = "".join(stream_answer(prompt, system)).strip()
            secs.append(time.monotonic() - t)
            passed += bool(check(answer))
        s = sorted(secs)
        mid = len(s) // 2
        out[model] = {"mean_seconds": round(sum(secs) / len(secs), 2),
                      "median_seconds": round((s[mid - 1] + s[mid]) / 2 if len(s) % 2 == 0 else s[mid], 2),
                      "max_seconds": round(max(secs), 2), "passed_this_run": passed}
        print(model, out[model], flush=True)
    save("responses_timing_warm", out)


if __name__ == "__main__":
    main()
