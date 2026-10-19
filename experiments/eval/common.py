"""Shared helpers for the evaluation scripts."""

import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = Path(__file__).resolve().parent
DATASET = EVAL / "dataset"
RESULTS = EVAL / "results"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The application opens its stores by relative path ("chroma_db",
# "patient_db"), so the scripts must run from the project root or they
# silently open empty stores.
import os  # noqa: E402
os.chdir(ROOT)


class Upload(io.BytesIO):
    """Stands in for a Streamlit upload: bytes plus name, type, size."""

    def __init__(self, data, name, mime):
        super().__init__(data)
        self.name, self.type, self.size = name, mime, len(data)


def norm(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def parse_finding(finding):
    text = finding["text"]
    value = re.search(r"measured value ([\d.]+)", text)
    if " within" in text:
        status = "within"
    elif " below" in text:
        status = "below"
    elif " above" in text:
        status = "above"
    else:
        status = None
    return norm(finding["name"]), float(value.group(1)) if value else None, status


def score(findings, truth_rows):
    """Count truth rows read correctly, read wrongly, or missed."""
    parsed = [parse_finding(f) for f in findings]
    used = set()
    out = {"correct": 0, "wrong": 0, "missing": 0, "name_misread": 0, "details": []}
    for row in truth_rows:
        key = norm(row["test"])
        expected = float(row["value"].replace(",", ""))
        match = next((i for i, p in enumerate(parsed) if p[0] == key), None)
        if match is None:
            loose = [i for i, p in enumerate(parsed)
                     if i not in used and (key in p[0] or p[0] in key)]
            match = loose[0] if len(loose) == 1 else None
        if match is None:
            # Right value and verdict under a garbled name ("HBAlC" for
            # HbA1c) is neither correct nor wrong: the number is safe,
            # but a question naming the test will not find it.
            variant = next((i for i, p in enumerate(parsed) if i not in used
                            and p[1] is not None and abs(p[1] - expected) < 1e-6
                            and p[2] == row["status"]), None)
            if variant is not None:
                used.add(variant)
                out["name_misread"] += 1
                out["details"].append((row["test"], "name_misread",
                                       f"read as '{findings[variant]['name']}'"))
                continue
            out["missing"] += 1
            out["details"].append((row["test"], "missing", None))
            continue
        used.add(match)
        _, value, status = parsed[match]
        ok = value is not None and abs(value - expected) < 1e-6 and status == row["status"]
        out["correct" if ok else "wrong"] += 1
        out["details"].append((row["test"], "correct" if ok else "wrong",
                               f"read {value} {status}, expected {expected} {row['status']}"))
    out["spurious"] = len(parsed) - len(used)
    return out


def save(name, payload):
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload["run"] = datetime.now().isoformat(timespec="seconds")
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))
