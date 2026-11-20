"""Compare the three answering models the project used, on the same tasks.

  1. Range comparison with no help from code: given a value and its
     printed range, is it within, below or above? All 40 labelled
     values from the generated reports. This is the task the project
     moved out of the model.
  2. Rule compliance: the response scenarios, with the
     verified checks supplied as the application supplies them.
  3. Source selection: the 40 routing questions.

    python experiments/eval/eval_models.py
"""

import json
import os
import re
import subprocess
import sys
import time

from common import DATASET, EVAL, save

from langchain_ollama import ChatOllama
import eval_responses
from services import llm_service

MODELS = ["llama3", "llama3.2", "qwen2.5:7b-instruct"]
TRUTH = json.loads((DATASET / "ground_truth.json").read_text())


def compare_ranges(model):
    llm = ChatOllama(model=model, temperature=0, num_ctx=2048)
    right = 0
    rows = 0
    secs = 0.0
    misses = []
    for report in TRUTH.values():
        sex = "female" if report["sex"] == "F" else "male"
        for r in report["rows"]:
            prompt = (f"A laboratory result for a {sex} patient.\n"
                      f"Test: {r['test']}\nMeasured value: {r['value']} {r['unit']}\n"
                      f"Reference range: {r['range']}\n"
                      "Is the measured value within, below or above the reference "
                      "range? Reply with exactly one word: within, below or above.")
            t = time.monotonic()
            answer = llm.invoke(prompt).content.lower()
            secs += time.monotonic() - t
            word = next(iter(re.findall(r"within|below|above", answer)), None)
            rows += 1
            if word == r["status"]:
                right += 1
            else:
                misses.append(f"{r['test']} {r['value']} vs {r['range']}: said {word}, truth {r['status']}")
    return {"correct": right, "total": rows, "mean_seconds": round(secs / rows, 2),
            "misses": misses}


def run_responses(model):
    llm_service.llm = ChatOllama(model=model, temperature=0, num_ctx=8192)
    results = eval_responses.main(save_as="responses_" + re.sub(r"[:.]", "_", model))
    return {"passed": sum(r["passed"] for r in results), "total": len(results),
            "mean_seconds": round(sum(r["seconds"] for r in results) / len(results), 2)}


def run_routing(model):
    env = dict(os.environ, MCP_ROUTER_MODEL=model)
    out = subprocess.run([sys.executable, str(EVAL / "eval_routing.py")], env=env,
                         capture_output=True, text=True).stdout
    data = json.loads((EVAL / "results" / ("routing_" + re.sub(r"[:.]", "_", model) + ".json")).read_text())
    res = data["results"]
    return {"correct": sum(r["model_ok"] for r in res), "total": len(res),
            "fallbacks": sum(r["model"] == "fallback" for r in res),
            "natural_correct": sum(r["model_ok"] for r in res if r["phrasing"] == "natural"),
            "mean_seconds": round(sum(r["seconds"] for r in res) / len(res), 2),
            "log_tail": out.strip().splitlines()[-4:]}


summary = {}
for m in MODELS:
    print(f"\n######## {m}", flush=True)
    summary[m] = {"ranges": compare_ranges(m)}
    print("ranges:", {k: v for k, v in summary[m]["ranges"].items() if k != "misses"}, flush=True)
    summary[m]["responses"] = run_responses(m)
    print("responses:", summary[m]["responses"], flush=True)
    summary[m]["routing"] = run_routing(m)
    print("routing:", {k: v for k, v in summary[m]["routing"].items() if k != "log_tail"}, flush=True)

save("models", summary)
