"""RAG and report retrieval: scenarios against a throwaway vector store.

Profile A holds three reports for one patient (blood count, lipids,
thyroid). Profile B holds two others. The real patient store is never
touched. Each scenario checks one retrieval behaviour, and the
analyte questions also check that the verdict computed from what was
retrieved is the right one.

    python experiments/eval/eval_retrieval.py
"""

import json
import tempfile

from common import DATASET, norm, parse_finding, save

from langchain_community.vectorstores import Chroma
from services import patient_memory
from services.embedding_service import embedding
from services.lab_verifier import verify_lab_ranges

TRUTH = json.loads((DATASET / "ground_truth.json").read_text())
tmp = tempfile.mkdtemp()
patient_memory.patient_vectordb = Chroma(
    persist_directory=tmp, embedding_function=embedding)

PROFILES = {"A": ["R1_cbc_male", "R3_lipid_one_sided", "R5_thyroid_electrolytes"],
            "B": ["R2_cbc_female_sexed", "R4_lft_kft_to_ranges"]}

for profile, keys in PROFILES.items():
    for key in keys:
        text = (DATASET / key / "report.txt").read_text()
        patient_memory.add_patient_report(text, f"{key}.pdf", "Lab Report",
                                          profile_id=profile)

scenarios = []


def record(sid, name, passed, detail):
    scenarios.append({"id": sid, "scenario": name, "passed": bool(passed),
                      "detail": detail})
    print(f"{sid:<5}{'PASS' if passed else 'FAIL'}  {name}  ({detail})", flush=True)


def truth_row(key, test):
    return next(r for r in TRUTH[key]["rows"] if r["test"] == test)


# --- Analyte questions: right report retrieved, right verdict computed ---
ANALYTE = [
    ("RAG-1", "A", "what is my TSH level", "R5_thyroid_electrolytes", "TSH"),
    ("RAG-2", "A", "is my total cholesterol high", "R3_lipid_one_sided", "Total Cholesterol"),
    ("RAG-3", "A", "how is my white blood cell count", "R1_cbc_male", "Total WBC Count"),
    ("RAG-4", "A", "is my sodium okay", "R5_thyroid_electrolytes", "Sodium"),
    ("RAG-5", "A", "is my hba1c normal", "R3_lipid_one_sided", "HbA1c"),
    ("RAG-6", "B", "is my haemoglobin low", "R2_cbc_female_sexed", "Haemoglobin"),
    ("RAG-7", "B", "what is my albumin result", "R4_lft_kft_to_ranges", "Albumin"),
    ("RAG-8", "B", "is my platelet count normal", "R4_lft_kft_to_ranges", "Platelet Count"),
]
for sid, profile, q, expect_report, test in ANALYTE:
    docs, context, _, best = patient_memory.search_patient_reports(q, profile_id=profile)
    # The search returns chunks sorted by report for reading order, so
    # rank is taken from the raw similarity ordering instead.
    ranked = patient_memory.patient_vectordb.similarity_search_with_score(
        q, k=6, filter={"profile": profile})
    top = ranked[0][0].metadata.get("report_name", "") if ranked else "(none)"
    sexes = {d.metadata.get("patient_sex") for d in docs if d.metadata.get("patient_sex")}
    sex = sexes.pop() if len(sexes) == 1 else None
    found = [parse_finding(f) for f in verify_lab_ranges(context, patient_sex=sex)
             if norm(f["name"]) == norm(test)]
    row = truth_row(expect_report, test)
    verdict_ok = bool(found) and found[0][2] == row["status"]
    retrieved_ok = top == f"{expect_report}.pdf"
    record(sid, f"'{q}' ranks the right report first and gives the right verdict",
           retrieved_ok and verdict_ok,
           f"best distance {best:.2f}, top {top}, verdict "
           f"{found[0][2] if found else 'none'} (expected {row['status']})")

# --- Profile isolation ---
docs, context, _, _ = patient_memory.search_patient_reports(
    "what is my albumin result", profile_id="A")
leak = [d for d in docs if "R4" in d.metadata.get("report_name", "")]
record("RAG-9", "Profile A cannot retrieve profile B's albumin report",
       not leak and "Albumin" not in context, f"{len(leak)} foreign chunks")

docs, context, _, _ = patient_memory.search_patient_reports(
    "is my tsh normal", profile_id="B")
record("RAG-10", "Profile B cannot retrieve profile A's thyroid report",
       "TSH" not in context, f"TSH present: {'TSH' in context}")

# --- Whole-report summary bypass ---
docs, context, _, best = patient_memory.search_patient_reports(
    "give me a summary of my report", profile_id="B", whole_report=True)
names = {d.metadata.get("report_name") for d in docs}
record("RAG-11", "A summary request returns a whole report despite low similarity",
       bool(docs), f"best distance {best:.2f}, {len(docs)} chunks from {sorted(names)}")

# --- General question is not treated as answered by the report ---
docs, _, _, best = patient_memory.search_patient_reports("what is malaria", profile_id="A")
record("RAG-12", "A general question is not a strong match, so external sources run",
       not patient_memory.is_strong_match(best),
       f"best distance {best:.2f}, strong-match gate 1.25")

# --- Duplicate detection ---
text = (DATASET / "R1_cbc_male" / "report.txt").read_text()
again = patient_memory.add_patient_report(text, "R1_again.pdf", "Lab Report", profile_id="A")
record("RAG-13", "Uploading the same report twice stores it once",
       again is None, "second upload skipped" if again is None else "stored twice")

# --- Sex recorded on every chunk ---
docs = patient_memory.get_all_patient_documents("B")
r2 = [d for d in patient_memory.patient_vectordb.get(
    where={"profile": "B"}, include=["metadatas"])["metadatas"]
    if "R2" in d.get("report_name", "")]
record("RAG-14", "Patient sex from the header is stored on every chunk",
       r2 and all(m.get("patient_sex") == "F" for m in r2),
       f"{sum(m.get('patient_sex') == 'F' for m in r2)}/{len(r2)} chunks tagged F")

# --- Deleting one profile leaves the other intact ---
before_b = patient_memory.count_patient_reports("B")
patient_memory.delete_all_patient_reports("A")
record("RAG-15", "Deleting profile A's reports leaves profile B's untouched",
       patient_memory.count_patient_reports("A")[0] == 0
       and patient_memory.count_patient_reports("B") == before_b,
       f"A now {patient_memory.count_patient_reports('A')}, B {patient_memory.count_patient_reports('B')}")

# --- General knowledge base threshold (known limitation) ---
import rag  # noqa: E402
kb = []
for q in ["what is diabetes", "what causes high blood sugar", "symptoms of prediabetes",
          "how is prediabetes treated", "what is hba1c"]:
    scores = [s for _, s in rag.vectordb.similarity_search_with_score(q, k=3)]
    kb.append((q, round(min(scores), 2) if scores else None))
on_topic = kb[:4]   # the reference collection is about diabetes and prediabetes
passing = sum(1 for _, s in on_topic if s is not None and s < rag.LOCAL_SCORE_THRESHOLD)
record("RAG-16", "Knowledge base returns material for every on-topic question",
       passing == len(on_topic),
       f"{passing}/{len(on_topic)} under its 1.0 threshold; best distances {[s for _, s in on_topic]}")

passed = sum(s["passed"] for s in scenarios)
print(f"\n{passed}/{len(scenarios)} retrieval scenarios passed")
save("retrieval", {"scenarios": scenarios, "kb_distances": kb})
