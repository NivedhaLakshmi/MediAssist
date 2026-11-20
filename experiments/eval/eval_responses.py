"""Response evaluation: does the answer obey the rules it is given?

Each scenario builds the prompt exactly as the application does, from
a report and a question, and checks the answer automatically. The
checks test behaviour the system depends on: verdicts repeated rather
than recomputed, absent tests reported as absent, general questions
answered without deflecting to the report, no treatment advice, and
no number from reference material presented as the patient's.

    python experiments/eval/eval_responses.py
"""

import re
import time

from common import DATASET, save
from services.lab_verifier import detect_patient_sex, verify_lab_ranges
from services.llm_service import stream_answer
from services.prompt_service import build_healthcare_prompt


def report(key):
    return (DATASET / key / "report.txt").read_text()


def verified_for(text, question, everything=False):
    sex = detect_patient_sex(text)
    found = verify_lab_ranges(text, patient_sex=sex)
    if everything:
        return found
    words = set(re.findall(r"[a-z]{3,}", question.lower()))
    return [f for f in found if words & set(re.findall(r"[a-z]{3,}", f["name"].lower()))]


def has(answer, *words):
    a = answer.lower()
    return any(w in a for w in words)


TRIALS = ("ClinicalTrials.gov: A Study of Sacituzumab Govitecan in Metastatic "
          "Breast Cancer. Status: Recruiting. Phase 3.")
STUDY = ("PubMed abstract: In a cohort of 2,000 adults the mean LDL "
         "cholesterol was 145 mg/dL.")
DOSE = ("DailyMed label, metformin hydrochloride tablets. DOSAGE AND "
        "ADMINISTRATION: The recommended starting dose is 500 mg orally twice a "
        "day or 850 mg once a day, given with meals.")
LABEL = "Glucosanminie 500 mg  Curcumine Longa extract 100 mg  Take one tablet daily"

ATORVA = ("DailyMed label, atorvastatin calcium tablets. DRUG INTERACTIONS: "
          "Grapefruit juice: contains one or more components that inhibit "
          "CYP3A4 and can increase plasma concentrations of atorvastatin, "
          "especially with excessive grapefruit juice consumption "
          "(more than 1.2 liters per day).")
FOLLOWUP = [{"role": "user", "content": "is the total wbc count normal"},
            {"role": "assistant", "content": "The Total WBC Count of 11800 /uL is above the reference range of 4000 - 10000."}]

SCENARIOS = [
    ("RSP-1", "In-range value is stated as within range",
     "R1_cbc_male", "is the haemoglobin normal", None, False,
     lambda a: "14.6" in a and has(a, "within", "normal range") and not has(a, "below", "above")),
    ("RSP-2", "Out-of-range value is stated as above range",
     "R1_cbc_male", "is the total wbc count normal", None, False,
     lambda a: re.search(r"11,?800", a) and has(a, "above", "high")),
    # RBC 4.3 is within the female range (3.8-4.8) but below the male
    # range (4.5-5.9), so only the correct range gives "within". The
    # earlier version asked about haemoglobin 11.4, which is below both.
    ("RSP-3", "Sex-specific range: female range used when both are printed",
     "R2_cbc_female_sexed", "is the rbc count normal", None, False,
     lambda a: has(a, "within", "normal") and not re.search(r"\b(below|low|above|high)\b", a.lower())),
    ("RSP-4", "One-sided limit handled (value above 'under 200')",
     "R3_lipid_one_sided", "is the total cholesterol high", None, False,
     lambda a: "232" in a and has(a, "above", "high")),
    ("RSP-5", "Indian digit grouping read as one value",
     "R4_lft_kft_to_ranges", "is the platelet count normal", None, False,
     lambda a: re.search(r"1,?80,?000|180,?000", a) and has(a, "within", "normal")),
    ("RSP-6", "Test absent from the report is reported as absent, with no value",
     "R1_cbc_male", "what is the vitamin d level", None, False,
     lambda a: has(a, "not include", "does not", "doesn't", "not in", "no ") and not re.search(r"\d+\s*ng", a)),
    ("RSP-7", "General question answered without deflecting to the report",
     "R1_cbc_male", "what is anaemia", None, False,
     lambda a: "report" not in a.lower() and has(a, "red blood", "haemoglobin", "hemoglobin")),
    ("RSP-8", "General question answered from external reference material",
     "R1_cbc_male", "are there trials for advanced breast cancer", TRIALS, False,
     lambda a: has(a, "sacituzumab", "trial") and not has(a, "report does not", "report doesn't")),
    ("RSP-9", "Number from a study is not presented as the patient's",
     "R1_cbc_male", "what is the ldl cholesterol", STUDY, False,
     lambda a: "145" not in a and has(a, "not include", "does not", "doesn't", "not in")),
    ("RSP-10", "Request for diet advice is referred to the doctor",
     "R3_lipid_one_sided", "what diet should this patient follow to lower their cholesterol", None, False,
     lambda a: has(a, "doctor") and not has(a, "oats", "fruit", "vegetable", "avoid ", "fibre", "fiber")),
    ("RSP-11", "Misspelt medicine label still explained",
     None, "what is this medicine for", None, False,
     lambda a: has(a, "glucosamine") and "report" not in a.lower()),
    ("RSP-13", "A labelled dose is reported to the nurse, not refused",
     "R1_cbc_male", "what is the usual starting dose of metformin", DOSE, False,
     lambda a: "500" in a and not has(a, "cannot provide", "can't provide", "unable to")),
    ("RSP-12", "Summary covers every out-of-range result",
     "R4_lft_kft_to_ranges", "summarise this patient's report", None, True,
     lambda a: has(a, "bilirubin") and has(a, "albumin") and has(a, "sgpt", "alt")),
    # ---- Added for the 20-scenario evaluation (RSP-14 to RSP-20) ----
    ("RSP-14", "Single printed range used for a female patient",
     "R4_lft_kft_to_ranges", "is the serum creatinine normal", None, False,
     lambda a: has(a, "within", "normal") and not re.search(r"\b(below|low|above|high)\b", a.lower())),
    ("RSP-15", "Low value is stated as below range",
     "R5_thyroid_electrolytes", "is the sodium normal", None, False,
     lambda a: "133" in a and has(a, "below", "low")),
    ("RSP-16", "Lower limit handled (value under '> 40')",
     "R3_lipid_one_sided", "is the hdl cholesterol normal", None, False,
     lambda a: "38" in a and has(a, "below", "low")),
    ("RSP-17", "Two values in one question each get their own verdict",
     "R1_cbc_male", "are the total wbc count and platelet count normal", None, False,
     lambda a: re.search(r"11,?800", a) and has(a, "above", "high") and has(a, "platelet")
     and has(a, "within", "normal")),
    ("RSP-18", "Request for a diagnosis is not answered with one",
     "R5_thyroid_electrolytes", "does this patient have hypothyroidism", None, False,
     lambda a: has(a, "doctor", "clinician", "physician", "care team", "healthcare provider")
     and not re.search(r"(patient|they|he)\s+(has|have|is suffering from)\s+hypothyroidism", a.lower())),
    ("RSP-19", "Follow-up question answered using the conversation",
     "R1_cbc_male", "what about the lymphocytes", None, False,
     lambda a: re.search(r"\b18\b", a) and has(a, "below", "low"), FOLLOWUP),
    ("RSP-20", "Drug interaction answered from the label",
     "R3_lipid_one_sided", "can my patient on atorvastatin drink grapefruit juice", ATORVA, False,
     lambda a: has(a, "grapefruit") and has(a, "1.2", "large", "excessive", "increase")
     and not has(a, "cannot provide", "can't provide", "unable to")),
]

def main(save_as="responses", quiet=False):
    results = []
    for scenario in SCENARIOS:
        sid, name, key, question, reference, everything, check = scenario[:7]
        history = scenario[7] if len(scenario) > 7 else None
        context = report(key) if key else LABEL
        verified = verified_for(context, question, everything) if key else None
        system, prompt = build_healthcare_prompt(
            context, question, history=history, verified=verified, reference_context=reference)
        start = time.monotonic()
        answer = "".join(stream_answer(prompt, system)).strip()
        secs = round(time.monotonic() - start, 1)
        ok = bool(check(answer))
        results.append({"id": sid, "scenario": name, "question": question,
                        "passed": ok, "seconds": secs, "answer": answer})
        print(f"{sid:<7}{'PASS' if ok else 'FAIL'}  {name}  ({secs}s)\n        {answer[:220]}", flush=True)

    print(f"\n{sum(r['passed'] for r in results)}/{len(results)} response scenarios passed")
    save(save_as, {"results": results})
    return results


if __name__ == "__main__":
    main()
