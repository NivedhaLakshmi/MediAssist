"""Invented-number check on the saved answer-scenario outputs.

For each model's recorded answer-scenario answers, rebuild the exact prompt the model
was given (report or label, code-verified checks, reference text,
question) and flag any number in the answer that does not appear in
that prompt. Nothing is re-run; this reads results/responses_*.json.

    python experiments/eval/eval_numbers.py
"""

import json
import re

from common import ROOT, save
import eval_responses as er
from services.prompt_service import build_healthcare_prompt

FILES = {"Llama 3.2 3B": "responses_llama3_2", "Llama 3 8B": "responses_llama3",
         "Qwen 2.5 7B": "responses_qwen2_5_7b-instruct"}
NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*")
LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s")


def numbers(text):
    text = LIST_MARKER.sub(" ", text)
    out = set()
    for m in NUM.findall(text):
        s = m.replace(",", "").rstrip(".")
        try:
            out.add(float(s))
        except ValueError:
            pass
    return out


def prompts():
    built = {}
    for scenario in er.SCENARIOS:
        sid, _, key, question, reference, everything, _ = scenario[:7]
        history = scenario[7] if len(scenario) > 7 else None
        context = er.report(key) if key else er.LABEL
        verified = er.verified_for(context, question, everything) if key else None
        system, prompt = build_healthcare_prompt(context, question, history=history,
                                                 verified=verified, reference_context=reference)
        built[sid] = system + "\n" + prompt
    return built


def main():
    given = prompts()
    out = {}
    for model, f in FILES.items():
        rows = json.load(open(ROOT / "experiments/eval/results" / f"{f}.json"))["results"]
        flagged = []
        for r in rows:
            extra = sorted(numbers(r["answer"]) - numbers(given[r["id"]]))
            if extra:
                flagged.append({"id": r["id"], "invented": extra, "answer": r["answer"]})
        out[model] = {"answers": len(rows), "with_invented_numbers": len(flagged), "flagged": flagged}
        print(f"\n{model}: {len(flagged)}/{len(rows)} answers contain a number not in the prompt")
        for x in flagged:
            print(f"  {x['id']}: {x['invented']}  | {x['answer'][:200]}")
    save("numbers", out)


if __name__ == "__main__":
    main()
