"""Document and image processing: every report, every rendering, scored.

For each of the five reports and four renderings, the full reading
pipeline is run and each labelled value is scored as read correctly
(right value and right in/out-of-range verdict), read wrongly, or
missed. Images and scans are also read by character recognition alone,
which is how the system read them before the vision model was added.
A non-document image (a chest X-ray) checks that nothing is invented.

    python experiments/eval/eval_documents.py
"""

import json
import time

from common import DATASET, Upload, save, score
from services import image_service, pdf_service
from services.lab_verifier import (
    detect_patient_sex, merge_findings, verify_extracted_rows,
    verify_lab_ranges,
)

ROUTES = [
    ("digital PDF", "digital.pdf", "application/pdf"),
    ("scanned PDF", "scanned.pdf", "application/pdf"),
    ("clean image", "clean.png", "image/png"),
    ("phone photo", "photo.jpg", "image/jpeg"),
]


def findings_from_text(text, sex=None):
    sex = sex or detect_patient_sex(text)
    return verify_lab_ranges(text, patient_sex=sex)


_fallback_used = []
_real_tesseract = image_service._tesseract_text


def _tracking_tesseract(upload):
    _fallback_used.append(True)
    return _real_tesseract(upload)


def run_pipeline(path, mime):
    data = path.read_bytes()
    _fallback_used.clear()
    image_service._tesseract_text = _tracking_tesseract
    up = Upload(data, path.name, mime)
    start = time.monotonic()
    if mime == "application/pdf":
        text, how = pdf_service.read_pdf(up)
        findings = findings_from_text(text)
    else:
        text, rows = image_service.read_image(up)
        sex = detect_patient_sex(text)
        findings = merge_findings(
            verify_lab_ranges(text, patient_sex=sex),
            verify_extracted_rows(rows, sex),
        )
        if _fallback_used and text:
            how = "OCR fallback"
        elif text:
            how = "MedGemma"
        else:
            how = "nothing read"
    image_service._tesseract_text = _real_tesseract
    return findings, how, round(time.monotonic() - start, 1)


def run_ocr_only(path, mime):
    data = path.read_bytes()
    start = time.monotonic()
    if mime == "application/pdf":
        text = "\n".join(
            pdf_service._tesseract_page(png)
            for png in pdf_service.render_pages(Upload(data, path.name, mime))
        )
    else:
        text = image_service._tesseract_text(Upload(data, path.name, mime))
    return findings_from_text(text), round(time.monotonic() - start, 1)


def main():
    truth = json.loads((DATASET / "ground_truth.json").read_text())
    results = []
    for key, report in truth.items():
        for route, filename, mime in ROUTES:
            path = DATASET / key / filename
            findings, how, secs = run_pipeline(path, mime)
            s = score(findings, report["rows"])
            entry = {"report": key, "route": route, "read_by": how,
                     "seconds": secs, "pipeline": s}
            if route != "digital PDF":
                ocr_findings, ocr_secs = run_ocr_only(path, mime)
                entry["ocr_only"] = score(ocr_findings, report["rows"])
                entry["ocr_seconds"] = ocr_secs
            results.append(entry)
            o = entry.get("ocr_only")
            print(f"{key:<26}{route:<13} pipeline {s['correct']:>2}/{len(report['rows'])} "
                  f"wrong {s['wrong']} spurious {s['spurious']} {secs:>5}s via {how}"
                  + (f" | OCR only {o['correct']}/{len(report['rows'])} wrong {o['wrong']}" if o else ""),
                  flush=True)

    # A radiograph is not a document. Nothing should be read from it.
    xray = DATASET / "xray_chest.jpg"
    if xray.exists():
        findings, how, secs = run_pipeline(xray, "image/jpeg")
        results.append({"report": "chest X-ray", "route": "non-document image",
                        "findings": len(findings), "seconds": secs})
        print(f"chest X-ray (non-document): {len(findings)} findings produced, {secs}s")

    save("documents", {"results": results})


if __name__ == "__main__":
    main()
