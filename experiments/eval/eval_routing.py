"""MCP source selection: the production model router against the keyword rules.

28 questions with a known correct source. 16 are phrased with the
words the keyword rules look for; 12 are phrased the way patients and
nurses actually ask, without those words. The model router is the one
the server uses, called directly.

    python experiments/eval/eval_routing.py
"""

import subprocess
import sys

from common import ROOT, save

CASES = [
    # (question, correct source, phrasing)
    ("what are the side effects of metformin", "dailymed", "keyword"),
    ("what is the usual dosage for amoxicillin", "dailymed", "keyword"),
    ("can i take ibuprofen with warfarin", "dailymed", "keyword"),
    ("what are the warnings for atorvastatin", "dailymed", "keyword"),
    ("are there any clinical trials for type 2 diabetes", "clinical_trials", "keyword"),
    ("what new treatments are being studied for psoriasis", "clinical_trials", "keyword"),
    ("is there an ongoing study for alzheimer's disease", "clinical_trials", "keyword"),
    ("any experimental therapy for sickle cell", "clinical_trials", "keyword"),
    ("what is hypertension", "medlineplus", "keyword"),
    ("what causes anaemia", "medlineplus", "keyword"),
    ("how is asthma usually managed", "medlineplus", "keyword"),
    ("what does a thyroid test measure", "medlineplus", "keyword"),
    ("what is a complete blood count", "medlineplus", "keyword"),
    ("what does high bilirubin mean", "medlineplus", "keyword"),
    ("what is vitamin d deficiency", "medlineplus", "keyword"),
    ("what is a normal sodium level", "medlineplus", "keyword"),
    ("how much paracetamol for a 5 year old", "dailymed", "natural"),
    ("is amoxicillin okay during pregnancy", "dailymed", "natural"),
    ("atorvastatin and grapefruit", "dailymed", "natural"),
    ("what happens if i take too much metformin", "dailymed", "natural"),
    ("is dolo 650 safe for a child", "dailymed", "natural"),
    ("what is in the pipeline for als", "clinical_trials", "natural"),
    ("is anything being developed for huntington's disease", "clinical_trials", "natural"),
    ("are researchers testing anything for parkinson's", "clinical_trials", "natural"),
    ("what do studies show about vitamin d supplementation", "pubmed", "natural"),
    ("is there evidence linking sleep and immune function", "pubmed", "natural"),
    ("has intermittent fasting been shown to help", "pubmed", "natural"),
    ("what is the current scientific consensus on red meat", "pubmed", "natural"),
]

# Run inside the server's directory, whose own utils package would
# otherwise shadow the application's.
probe = '''
import sys, json, time, os
sys.path.insert(0, ".")
from utils import model_router
from tools.search import _keyword_order
cases = json.loads(sys.argv[1])
out = []
for q, _, _ in cases:
    t = time.monotonic()
    pick = model_router.choose_source(q)
    out.append({"rules": _keyword_order(q)[0].replace("search_", ""),
                "model": (pick or "fallback").replace("search_", ""),
                "seconds": round(time.monotonic() - t, 2)})
print("OUT=" + json.dumps(out))
'''
import json  # noqa: E402
res = subprocess.run([sys.executable, "-c", probe, json.dumps(CASES)],
                     cwd=ROOT / "medical_mcp", capture_output=True, text=True)
rows = json.loads(res.stdout.split("OUT=")[-1])

results = []
for (q, truth, kind), r in zip(CASES, rows):
    results.append({"question": q, "correct": truth, "phrasing": kind, **r,
                    "model_ok": r["model"] == truth, "rules_ok": r["rules"] == truth})
    print(f"{'OK ' if r['model']==truth else 'BAD'} model={r['model']:<16}"
          f"{'OK ' if r['rules']==truth else 'BAD'} rules={r['rules']:<16}{kind:<8}{q}")

for kind in ("keyword", "natural", None):
    sub = [x for x in results if kind is None or x["phrasing"] == kind]
    label = kind or "all"
    print(f"{label:<8} model {sum(x['model_ok'] for x in sub)}/{len(sub)}   "
          f"rules {sum(x['rules_ok'] for x in sub)}/{len(sub)}")
print(f"mean routing time {sum(x['seconds'] for x in results)/len(results):.2f}s")
import os  # noqa: E402
_model = os.getenv("MCP_ROUTER_MODEL")
save("routing" if not _model else "routing_" + _model.replace(":", "_").replace(".", "_"),
     {"model": _model or "default", "results": results})
