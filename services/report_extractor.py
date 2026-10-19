"""Read a photographed medical document with a vision model.

Tesseract transcribes characters. It has no idea it is looking at a
lab table, so a photo taken at an angle, under glare, or with the
value and its reference range in separate columns comes back as a
jumble that the row parser cannot use. Ingredient lists on curved
bottles fare worse.

MedGemma is a vision-language model whose image encoder was
pre-trained on medical images, so it can be asked for the document's
*structure* rather than its pixels: which analyte, which value, which
reference range.

Two rules keep this safe:

  * The model never decides whether a value is normal. It reports
    name, value and range; :mod:`services.lab_verifier` still does
    the arithmetic in Python.
  * Anything it returns that cannot be validated is dropped, and any
    failure at all falls back to Tesseract. The worst case is the
    behaviour that existed before this module.
"""

import functools
import io
import json
import os
import re

from services.lab_verifier import parse_reference


MODEL = os.getenv("MEDGEMMA_MODEL", "medgemma:4b")

# Reading an image takes a few seconds. It happens once, when a
# document is added - never while answering a question.
TIMEOUT_SECONDS = float(os.getenv("MEDGEMMA_TIMEOUT", "120"))

# Roughly twice what a full lab page needs (a dense slip reads in
# about 630 tokens). The cap is deliberately not generous: its other
# job is to end a repetition loop quickly, and at 2048 a cluttered
# photograph span 85 seconds before anyone noticed it was stuck.
MAX_TOKENS = 1200

# A name longer than this is a sentence the model wrote, not an
# analyte it read.
MAX_NAME_LENGTH = 60

# How long Ollama keeps the model in memory after a read. A
# cold load costs about five seconds on top of every upload.
KEEP_ALIVE = os.getenv("MEDGEMMA_KEEP_ALIVE", "10m")

# A cluttered photograph - a blister pack held in the hand - can send
# the model into a repetition loop: it reads "Boswellic Acid" once and
# then writes it again until it hits the token limit, 90 seconds
# later, and the whole reply has to be discarded.
#
# Widening Ollama's repetition window ends the loop, but it must not
# be on by default: a lab table legitimately repeats itself - the same
# JSON shape, the same units, "to" in every range - and penalising
# that took a clean slip from 12/12 to 5/12 and a photographed one to
# 0/12. So it is applied only to a retry, after a read has already run
# to the limit and been thrown away. It cannot damage a read that
# worked, because a read that worked never reaches it.
REPEAT_PENALTY = 1.3
REPEAT_LAST_N = 256


# Rows are arrays of strings copied straight off the page, for two
# reasons.
#
# Speed: generation is the whole cost - the image is encoded to 611
# tokens whatever its size - and repeating six JSON key names per row
# cost 865 tokens and 29 seconds against 641 tokens and 22 seconds
# here.
#
# A rule telling the model not to guess was tried, to curb the
# invented ingredients a crumpled blister pack produced. It took a
# clean lab slip from 12/12 to 6/12 with six wrong values - a
# caution about reading turns into hesitancy about reporting. The
# instructions stay descriptive; an unreadable photograph is handled
# by discarding the read, not by asking the model to be careful.
#
# Accuracy: asked for "reference_low" and "reference_high" as
# separate numbers in a positional array, the model put the whole of
# "4000 to 10000" in the low slot and left high empty, turning every
# range into a one-sided limit. Transcribing the range as printed and
# parsing it in Python plays to what each side is good at.
#
# The unit is asked for but never stored. Dropping it from the schema
# was 100 fewer tokens and 16% faster, and cost 12/12 -> 0/12 on a
# clean scan: without that slot the model rambled past its token
# limit. This four-field shape is what it answers stably, and
# stability is worth more than four seconds.
PROMPT = """You are reading a photograph of a medical document.

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


def upright(image_bytes):
    """Apply the photo's own orientation tag before anything reads it.

    A phone holds the sensor still and records which way up the
    picture was taken in an EXIF tag; every viewer rotates it on the
    way to the screen. Reading the file's raw pixels skips that, so a
    label photographed the right way up arrives on its side - and a
    model given sideways text loops until it runs out of tokens.

    Two photographs of a bottle failed this way, and read in fifteen
    seconds once turned upright.

    Returns the original bytes if there is nothing to correct.
    """

    try:
        from PIL import Image, ImageOps

        image = Image.open(io.BytesIO(image_bytes))

        if (image.getexif() or {}).get(274, 1) in (1, None):
            return image_bytes

        buffer = io.BytesIO()

        ImageOps.exif_transpose(image).convert("RGB").save(
            buffer, "JPEG", quality=92
        )

        print("[VISION] Turned the photograph upright.")

        return buffer.getvalue()

    except Exception as error:  # noqa: BLE001
        print("[VISION] Could not check the orientation:", error)
        return image_bytes


@functools.lru_cache(maxsize=1)
def is_available():
    """Whether the vision model is installed and Ollama is running."""

    try:
        import ollama

        installed = {
            model.get("model") or model.get("name")
            for model in ollama.list().get("models", [])
        }

    except Exception as error:  # noqa: BLE001
        print("[VISION] Ollama unavailable:", error)
        return False

    if MODEL not in installed:
        print(
            f"[VISION] {MODEL} is not installed "
            f"- falling back to Tesseract. "
            f"Install it with 'ollama pull {MODEL}'."
        )
        return False

    return True


def _parse_payload(raw):
    """Pull the JSON object out of whatever the model wrapped it in."""

    if not raw:
        return {}

    text = raw.strip()

    # ```json ... ``` fences.
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)

    if fence:
        text = fence.group(1).strip()

    for candidate in _json_candidates(text):

        payload = _loads(candidate)

        if isinstance(payload, dict):
            return payload

    return {}


# JSON allows a backslash only before one of these.
_BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu]|u[0-9a-fA-F]{4})')


def _loads(text):
    """json.loads, retried with the model's stray backslashes fixed.

    It transcribes what is printed, so a label reading "50 mg" then
    "Chondroitin" came back as "50 mg\\Chondroitin". That is not a
    valid escape, and the whole read - ingredients and all - was
    thrown away for one character.
    """

    try:
        return json.loads(text)

    except ValueError:
        pass

    try:
        return json.loads(_BAD_ESCAPE.sub(r"\\\\", text))

    except ValueError:
        return None


def _json_candidates(text):
    """The whole reply, then just the object inside it."""

    yield text

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        yield match.group()


def _as_number(value):
    """Return a float, or None when the model wrote something else."""

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        return None

    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))

    return float(match.group()) if match else None


def _clean_rows(rows):
    """Keep only rows that are complete and internally consistent.

    A row whose reference range names both sexes becomes one entry
    per sex, matching what the text parser produces.
    """

    if not isinstance(rows, list):
        return []

    cleaned = []

    for row in rows:

        if isinstance(row, (list, tuple)):

            if len(row) < 3:
                continue

            name, raw_value = row[0], row[1]

            # The range is the last item, so a model that omits
            # the unit is still read correctly.
            reference = row[-1]

        elif isinstance(row, dict):

            # An older shape, still accepted so a model that answers
            # with named fields is not thrown away.
            name = row.get("name")
            raw_value = row.get("value")
            reference = row.get("reference") or ""

        else:
            continue

        name = str(name or "").strip()

        if not name or len(name) > MAX_NAME_LENGTH:
            continue

        value = _as_number(raw_value)

        if value is None:
            continue

        for sex, low, high in parse_reference(str(reference or "")):

            cleaned.append(
                {
                    "name": name,
                    "value": value,
                    "low": low,
                    "high": high,
                    "sex": sex if sex in ("M", "F") else "",
                }
            )

    return cleaned


def _ask(image_bytes, penalise_repetition=False):
    """One call to the model, or None if it could not be made."""

    options = {
        "temperature": 0,
        "num_predict": MAX_TOKENS,
    }

    if penalise_repetition:
        options["repeat_penalty"] = REPEAT_PENALTY
        options["repeat_last_n"] = REPEAT_LAST_N

    try:
        import ollama

        return ollama.Client(timeout=TIMEOUT_SECONDS).chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT,
                    "images": [image_bytes],
                }
            ],
            options=options,
            keep_alive=KEEP_ALIVE,
        )

    except Exception as error:  # noqa: BLE001
        print("[VISION] Could not read the image:", error)
        return None


def _without_prompt_echo(text):
    """Drop lines the model copied out of the instructions.

    Asked for "every line of text you can read", it sometimes writes
    that sentence into the answer as though it were printed on the
    document. Left in, it reaches the patient's context as a line of
    the label.
    """

    if not text:
        return text

    instructions = PROMPT.lower()

    kept = [
        line for line in text.splitlines()
        if not line.strip()
        or line.strip().lower() not in instructions
    ]

    removed = len(text.splitlines()) - len(kept)

    if removed:
        print(f"[VISION] Dropped {removed} line(s) echoed from the "
              "instructions.")

    return "\n".join(kept).strip()


def read_document(image_bytes):
    """Return ``(text, rows)`` read from an image.

    ``text`` is everything the model could read, for the answer to
    quote. ``rows`` are validated measurements awaiting comparison -
    this module never classifies them.

    Returns ``("", [])`` on any failure, so the caller can fall back.
    """

    if not image_bytes or not is_available():
        return "", []

    image_bytes = upright(image_bytes)

    response = _ask(image_bytes)

    if response is None:
        return "", []

    # Running to the token limit means the reply was cut mid-JSON,
    # and on this model it almost always means a repetition loop on
    # a photograph it cannot read.
    #
    # A penalised retry does break the loop and returns fluent text -
    # which is the problem. On a crumpled blister pack it produced a
    # different invented ingredient list each time, once naming
    # hydroxychloroquine, a prescription drug the pack does not
    # contain. Fluent and wrong is the worst thing this application
    # can hand a patient, so the read is abandoned instead.
    if response.get("done_reason") == "length":

        print(
            "[VISION] The model ran to its token limit - the reply "
            "is incomplete and has been discarded."
        )

        return "", []

    payload = _parse_payload(
        (response.get("message") or {}).get("content", "")
    )

    if not payload:
        print("[VISION] The model did not return usable JSON.")
        return "", []

    text = _without_prompt_echo(
        str(payload.get("text") or "").strip()
    )

    rows = _clean_rows(payload.get("values"))

    print(
        f"[VISION] {payload.get('document_type', 'unknown')}: "
        f"{len(text)} characters, {len(rows)} measurement(s)."
    )

    return text, rows


def rows_as_text(rows):
    """Render measurements back into lines a text parser can read.

    The model puts the numbers in ``values`` and leaves ``text`` as a
    list of the test names, which is right for the image path - the
    rows go straight to the verifier. The PDF path stores text and
    parses it again at question time, so on that path the rows have
    to be written back out or twelve correct measurements are simply
    dropped.

    The forms below are the ones the text parser already handles.
    """

    lines = []

    for row in rows or []:

        low, high, sex = row["low"], row["high"], row["sex"]

        if low is not None and high is not None:
            reference = f"{low:g} to {high:g}"
            if sex:
                reference = f"{sex}-({reference})"

        elif high is not None:
            reference = f"< {high:g}"

        elif low is not None:
            reference = f"> {low:g}"

        else:
            continue

        lines.append(f"{row['name']} {row['value']:g} {reference}")

    return "\n".join(lines)

