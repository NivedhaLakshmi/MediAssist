"""Probe MedGemma on radiological images, outside the application.

Nothing here is imported by MediAssist and nothing here imports
MediAssist. The purpose is to answer one question that the product
code deliberately does not ask: given a radiograph rather than a
printed document, what does the vision model actually produce?

Four prompts are run against each image:

  baseline        the application's own extraction prompt, verbatim,
                  so the probe records what happens today
  radiology       free-text findings, which is the use Google
                  evaluated the model on
  structured      findings as JSON, to see whether the model will
                  hold a schema on this input
  patient         a plain-language explanation, the register
                  MediAssist answers in

Each prompt is timed and its stop reason recorded. A reply that ends
on the token limit is the signature of the repetition loop seen
elsewhere in this project, so it is reported rather than hidden.

Usage:

    python experiments/probe_medgemma_xray.py chest.png
    python experiments/probe_medgemma_xray.py *.png --repeat 2
    python experiments/probe_medgemma_xray.py chest.png --only radiology

Results are written to experiments/results/ as Markdown.
"""

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


MODEL = os.getenv("MEDGEMMA_MODEL", "medgemma:4b")
TIMEOUT_SECONDS = float(os.getenv("MEDGEMMA_TIMEOUT", "180"))
KEEP_ALIVE = os.getenv("MEDGEMMA_KEEP_ALIVE", "10m")

# Generous compared with the application's 1200. A findings report is
# longer than a lab table, and the probe wants to see the model run
# out of things to say rather than run out of budget.
MAX_TOKENS = 1024

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------

# Copied verbatim from services/report_extractor.py rather than
# imported, so this file records the prompt as it was at the time of
# the experiment and cannot be changed by later edits to the product.
BASELINE_PROMPT = """You are reading a photograph of a medical document.

Return ONLY a JSON object, with no explanation and no code fence:

{
  "document_type": "lab_report" | "medicine_label" | "other",
  "text": "every line of text you can read, in the order it appears",
  "values": [["test name", "measured value", "unit", "reference range"]]
}

Each entry in "values" is an array of exactly four strings, copied
from the page exactly as printed: the test name, the measured value,
the unit, and the whole reference range including any "M-" / "F-"
labels and any "<" or ">".

Rules:
- Copy every character exactly. Never round, reformat or interpret.
- Only include a row if you can read its measured value.
- Use "" for anything that is not printed.
- Never say whether a value is normal, high or low. Report only what
  is printed.
- If the image is not a medical document, use "other" and an empty
  "values" list.
"""

RADIOLOGY_PROMPT = """Describe the findings in this radiograph.

State the modality and the view. Then describe what you observe,
region by region. Finish with a one-line impression.

If the image is not a radiograph, say so and stop.
"""

STRUCTURED_PROMPT = """Return ONLY a JSON object, with no explanation
and no code fence:

{
  "modality": "",
  "view": "",
  "findings": [{"region": "", "observation": "", "confidence": "high" | "medium" | "low"}],
  "impression": "",
  "image_quality": ""
}

Report only what is visible in the image. Use an empty list if you
can see nothing to report. Do not invent a finding to fill the
schema.
"""

PATIENT_PROMPT = """A patient has uploaded this image and asked what
it shows.

Explain in plain English, in four sentences or fewer, what kind of
image this is and what can be seen in it. Do not give a diagnosis and
do not give advice. If you cannot tell what the image shows, say so.
"""

PROMPTS = {
    "baseline": BASELINE_PROMPT,
    "radiology": RADIOLOGY_PROMPT,
    "structured": STRUCTURED_PROMPT,
    "patient": PATIENT_PROMPT,
}


# ----------------------------------------------------------------------
# Image handling
# ----------------------------------------------------------------------

def upright(image_bytes):
    """Apply the file's recorded orientation, if it has one.

    Reimplemented rather than imported, for the same reason the prompt
    is copied. A phone records how it was held in the metadata instead
    of rotating the pixels, and a model shown a sideways image reads
    nothing.
    """

    try:
        from PIL import Image, ImageOps

        image = Image.open(io.BytesIO(image_bytes))
        corrected = ImageOps.exif_transpose(image)

        if corrected is image:
            return image_bytes

        buffer = io.BytesIO()
        corrected.save(buffer, format=image.format or "PNG")

        return buffer.getvalue()

    except Exception as error:  # noqa: BLE001
        print(f"  ! could not check orientation: {error}")
        return image_bytes


def describe_image(image_bytes):
    """Size and mode, so the write-up can state what was fed in."""

    try:
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes))

        return f"{image.width}x{image.height} {image.mode} {image.format}"

    except Exception:  # noqa: BLE001
        return f"{len(image_bytes)} bytes, unreadable by PIL"


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

def preflight():
    """Check the runtime and the model before spending time on images."""

    try:
        import ollama

    except ImportError:
        sys.exit(
            "The ollama package is not installed in this environment.\n"
            "  pip install ollama"
        )

    try:
        installed = ollama.Client(timeout=10).list()

    except Exception as error:  # noqa: BLE001
        sys.exit(
            f"Could not reach Ollama: {error}\n"
            "Start it with 'ollama serve'."
        )

    names = []

    for entry in installed.get("models", []):
        name = entry.get("model") or entry.get("name") or ""
        if name:
            names.append(name)

    if not any(name.startswith(MODEL.split(":")[0]) for name in names):
        sys.exit(
            f"{MODEL} is not installed. Found: {', '.join(names) or 'none'}\n"
            f"  ollama pull {MODEL}"
        )

    print(f"Ollama reachable, {MODEL} present.")


def ask(image_bytes, prompt):
    """One call. Returns a dict of reply text and measurements."""

    import ollama

    started = time.monotonic()

    try:
        reply = ollama.Client(timeout=TIMEOUT_SECONDS).chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_bytes],
                }
            ],
            options={
                "temperature": 0,
                "num_predict": MAX_TOKENS,
            },
            keep_alive=KEEP_ALIVE,
        )

    except Exception as error:  # noqa: BLE001
        return {
            "error": str(error),
            "seconds": round(time.monotonic() - started, 1),
        }

    return {
        "text": (reply.get("message") or {}).get("content", "") or "",
        "done_reason": reply.get("done_reason", ""),
        "output_tokens": reply.get("eval_count"),
        "seconds": round(time.monotonic() - started, 1),
    }


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def looks_like_json(text):
    """Did the model hold the schema it was given?"""

    stripped = text.strip()

    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.partition("\n")[2]

    try:
        json.loads(stripped[stripped.find("{"):stripped.rfind("}") + 1])
        return True

    except Exception:  # noqa: BLE001
        return False


def run(paths, chosen, repeats):
    preflight()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report = RESULTS_DIR / f"xray-probe-{stamp}.md"

    lines = [
        "# MedGemma radiology probe",
        "",
        f"Model: `{MODEL}`  ",
        f"Run: {datetime.now().isoformat(timespec='seconds')}  ",
        f"Token cap: {MAX_TOKENS}, temperature 0, "
        f"{repeats} run(s) per prompt",
        "",
        "Replies ending on `length` hit the token cap, which in this "
        "project has consistently meant the model was repeating itself "
        "rather than still reading.",
        "",
    ]

    summary = []

    for path in paths:
        path = Path(path)

        if not path.is_file():
            print(f"! {path} is not a file, skipped")
            continue

        image_bytes = upright(path.read_bytes())
        description = describe_image(image_bytes)

        print(f"\n=== {path.name}  ({description})")

        lines += [f"## {path.name}", "", f"`{description}`", ""]

        for key in chosen:
            for attempt in range(1, repeats + 1):
                label = key if repeats == 1 else f"{key} (run {attempt})"

                print(f"  {label} ... ", end="", flush=True)

                result = ask(image_bytes, PROMPTS[key])

                if "error" in result:
                    print(f"failed after {result['seconds']}s")

                    lines += [
                        f"### {label}", "",
                        f"**Failed** after {result['seconds']}s: "
                        f"`{result['error']}`", "",
                    ]

                    summary.append((path.name, label, "failed", ""))
                    continue

                text = result["text"]
                reason = result["done_reason"]
                valid = looks_like_json(text) if key in (
                    "baseline", "structured"
                ) else None

                print(
                    f"{result['seconds']}s, "
                    f"{result['output_tokens']} tokens, "
                    f"stop={reason or 'unknown'}"
                    + ("" if valid is None else
                       f", json={'ok' if valid else 'broken'}")
                )

                lines += [
                    f"### {label}",
                    "",
                    f"{result['seconds']}s · "
                    f"{result['output_tokens']} output tokens · "
                    f"stop reason `{reason or 'unknown'}`"
                    + ("" if valid is None else
                       f" · JSON {'parsed' if valid else 'did not parse'}"),
                    "",
                    "```",
                    text.strip() or "(empty reply)",
                    "```",
                    "",
                ]

                summary.append((
                    path.name,
                    label,
                    f"{result['seconds']}s / {result['output_tokens']}t",
                    reason or "",
                ))

    lines += ["## Summary", "", "| Image | Prompt | Cost | Stop reason |",
              "|---|---|---|---|"]

    for row in summary:
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    report.write_text("\n".join(lines))

    print(f"\nWritten to {report}")


def main():
    parser = argparse.ArgumentParser(
        description="Probe MedGemma on radiological images.",
    )
    parser.add_argument("images", nargs="+", help="image files to read")
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(PROMPTS),
        help="run only this prompt (repeatable)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="runs per prompt, to check consistency at temperature 0",
    )

    args = parser.parse_args()

    run(args.images, args.only or sorted(PROMPTS), args.repeat)


if __name__ == "__main__":
    main()
