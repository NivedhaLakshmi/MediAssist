"""Deterministic lab-value range checks.

Small language models are unreliable at deciding whether a value
sits inside a reference range. This module parses ``value`` /
``low - high`` pairs straight out of the retrieved report text and
classifies each one in Python, so the classification never depends
on the model's arithmetic.
"""

import re


# A number, optionally grouped with commas in either the Western
# (1,500,000) or Indian (1,50,000) style.
#
# Without comma support the range pattern matched "000 to 5" inside
# "1,50,000 to 5,00,000" and read a platelet count as a 0 - 5 range.
# Wrapped in a non-capturing group: it is an alternation, and
# embedding it bare in a larger pattern split that pattern in two.
_NUMBER_PATTERN = (
    r"(?:\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
)

# "12 - 15", "12.0 to 15.5", "12–15", "0.6 -1.2", "1,50,000 to 5,00,000"
_RANGE = re.compile(
    rf"(?P<low>{_NUMBER_PATTERN})\s*(?:-|–|—|to)\s*"
    rf"(?P<high>{_NUMBER_PATTERN})"
)

_NUMBER = re.compile(_NUMBER_PATTERN)

# A grouped number followed by a loose digit run, as produced when
# PDF extraction splits "2,89,000" into "2,89,00 0". The true value
# cannot be recovered without guessing, so the row is skipped.
_SPLIT_NUMBER = re.compile(r"\d,\d{2,3}[\d,]*\s+\d")


def _to_float(text):
    return float(text.replace(",", ""))

# Rows that are never a single reference-range result even though
# they carry numbers and range-looking tokens.
_SKIP_LINE = re.compile(
    r"report\s*name|patient\s*name|patient\s*id|\bage\b|\bsex\b|\bdob\b|"
    r"date\s*of\s*birth|sample|collected|received|reported\s*on|"
    r"registration|accession|ref(?:erred)?\s*by|lab\s*no|\.pdf\b",
    re.IGNORECASE,
)

# Category words used by lipid panels. With a low-high range present
# they signal a band table, where the applicable band is ambiguous.
_BAND_WORDS = re.compile(
    r"borderline|desirable|near\s*optimal|\boptimal\b|high\s*risk",
    re.IGNORECASE,
)

# A one-sided limit: "< 200", "<200mg/dl", "less than 200",
# "> 240mg/dl". Lipid panels state these instead of a range, and
# skipping them left the model to do the comparison itself - it
# called a total cholesterol of 207 "within the desirable range of
# less than 200 mg/dl".
_UPPER_LIMIT = re.compile(
    rf"(?:<|≤|less\s+than|under|up\s*to|below)\s*({_NUMBER_PATTERN})",
    re.IGNORECASE,
)

_LOWER_LIMIT = re.compile(
    rf"(?:>|≥|greater\s+than|more\s+than|above|at\s+least)\s*"
    rf"({_NUMBER_PATTERN})",
    re.IGNORECASE,
)


def _single_limit(window):
    """Return ("upper"|"lower", limit, position) for a lone limit.

    Only a single limit qualifies. A row listing several bands -
    "Desirable < 200  Borderline 200 - 239  High > 240" - gives no
    way to know which one the value should be judged against, so it
    is left alone.
    """

    uppers = list(_UPPER_LIMIT.finditer(window))
    lowers = list(_LOWER_LIMIT.finditer(window))

    if len(uppers) + len(lowers) != 1:
        return None, None, None

    match = (uppers or lowers)[0]

    return (
        "upper" if uppers else "lower",
        _to_float(match.group(1)),
        match.start(),
    )

# An extracted analyte name may not be one of these bare words, and
# may not contain the listed fragments.
_BAD_NAME = re.compile(
    r"reference|normal\s*range|\brange\b|result|units?|method|"
    r"report\s*name|patient|interval",
    re.IGNORECASE,
)

_NON_ANALYTE = {
    "below", "above", "normal", "abnormal", "high", "low",
    "borderline", "desirable", "optimal", "value", "values",
    "result", "results", "range", "ranges", "test", "tests",
    "total", "serum", "blood", "report", "reference", "interval",
    "and", "the", "with", "not",
}

# Interpretation words that must not sit between a value and its
# range - if they do, the range almost certainly belongs to a
# different row (bad OCR alignment).
_INTERLOPER = re.compile(
    r"\b(below|above|normal|abnormal|high|low|borderline|desirable|"
    r"optimal|negative|positive|\bnil\b|trace|reactive)\b",
    re.IGNORECASE,
)


def _is_continuation(line):
    """Whether ``line`` carries only a range, continuing the row above.

    "Reference Range : 15 - 45" continues the previous line. But
    "1 Hour 05 mm 0 to 20" is a result in its own right - it has its
    own value - and joining it to the line above produced a finding
    named "ESR 1/2 Hour 02 mm 1 Hour".
    """

    match = _RANGE.search(line)

    if not match:
        return False

    return not _NUMBER.search(line[: match.start()])


def _leading_label(line):
    """The alphabetic label a row starts with, such as "ESR"."""

    match = re.match(r"([A-Za-z][A-Za-z.&/'-]*(?:\s+[A-Za-z][A-Za-z.&/'-]*)?)\s+\d", line)

    return match.group(1).strip() if match else ""


# "Age/Sex : 31 /Male", "Sex: F", "Gender : Female".
_PATIENT_SEX = re.compile(
    r"\b(?:sex|gender)\b[^A-Za-z]{0,15}(female|male|f|m)\b",
    re.IGNORECASE,
)

# A reference range labelled with the sex it applies to:
# "M-(13.5 to 18.0)", "F-(11.5 to 16.4)", "Male : 13 - 17".
_SEXED_RANGE = re.compile(
    rf"\b(?P<sex>male|female|m|f)\b\s*[-–—:.]?\s*\(?\s*"
    rf"(?P<low>{_NUMBER_PATTERN})\s*(?:-|–|—|to)\s*"
    rf"(?P<high>{_NUMBER_PATTERN})",
    re.IGNORECASE,
)

# A line carrying nothing but the other sex's range, as happens when
# "M-(13.5 to 18.0)" and "F-(11.5 to 16.4)" are printed on separate
# lines of the same result row.
_SEX_ONLY_LINE = re.compile(
    rf"^\s*(?:male|female|m|f)\b\s*[-–—:.]?\s*\(?\s*"
    rf"{_NUMBER_PATTERN}\s*(?:-|–|—|to)\s*{_NUMBER_PATTERN}"
    rf"\s*\)?\s*$",
    re.IGNORECASE,
)


def detect_patient_sex(text):
    """Return "M", "F", or "" when the report does not say."""

    match = _PATIENT_SEX.search(text or "")

    if not match:
        return ""

    return match.group(1)[0].upper()


def _sexed_ranges(window):
    """Map "M"/"F" to the (low, high, position) range labelled for it."""

    found = {}

    for match in _SEXED_RANGE.finditer(window):

        sex = match.group("sex")[0].upper()

        low = _to_float(match.group("low"))
        high = _to_float(match.group("high"))

        if low >= high or sex in found:
            continue

        found[sex] = (low, high, match.start())

    return found


def _classify(value, low, high):
    if value < low:
        return "below the reference range"
    if value > high:
        return "above the reference range"
    return "within the reference range"


def _extract_name(prefix):
    """Analyte name = the cleaned text before the first number."""

    text = prefix

    # "(Modified Berthelot)", "(F)" and similar balanced notes.
    text = re.sub(r"\([^()]*\)", " ", text)

    # A method note whose opening "(" was cut off: "... picrate) Uric Acid".
    if ")" in text:
        text = text.split(")")[-1]

    # A qualifier whose closing ")" is missing: "Sodium ( Na".
    text = re.sub(r"\([^()]*$", " ", text)

    # Strip a leading ALL-CAPS section heading such as "KIDNEY
    # FUNCTION TESTS" or "ELECTROLYTES", but keep short ALL-CAPS
    # analyte abbreviations like "WBC Count" or "LDL Cholesterol".
    # Heading words are long; analyte abbreviations are not.
    tokens = text.split()

    caps_run = 0

    while caps_run < len(tokens) and re.fullmatch(
        r"[A-Z0-9][A-Z0-9'&./\-]*",
        tokens[caps_run],
    ):
        caps_run += 1

    if (
        caps_run
        and caps_run < len(tokens)
        and any(len(token) >= 6 for token in tokens[:caps_run])
    ):
        text = " ".join(tokens[caps_run:])

    text = text.strip(" :\t-|.")

    match = re.search(
        r"([A-Za-z][A-Za-z0-9' .\-/]{1,38}[A-Za-z0-9])\s*$",
        text,
    )

    if not match:
        return ""

    name = re.sub(r"\s+", " ", match.group(1)).strip()

    if len(name) < 3 or _BAD_NAME.search(name):
        return ""

    if name.lower() in _NON_ANALYTE:
        return ""

    return name


def verify_lab_ranges(context, max_findings=25, patient_sex=None):
    """Return authoritative "<name>: <value> ... <status>" lines.

    Only rows where a value and a numeric ``low - high`` range can be
    read with confidence are included. Anything ambiguous is skipped
    so the model is never handed a wrong classification.

    ``patient_sex`` ("M"/"F") overrides what the report states, for
    callers that know it from elsewhere. Many haematology rows print
    one range for men and another for women; taking the first one
    told a healthy woman her haemoglobin was below range.
    """

    if not context:
        return []

    lines = [line.strip() for line in context.splitlines()]

    if patient_sex:
        patient_sex = patient_sex[0].upper()
    else:
        patient_sex = detect_patient_sex(context)

    findings = []
    seen = set()

    for index, line in enumerate(lines):

        if not line:
            continue

        # Allow the value on one line and the range on the next, but
        # only when this line actually carries a value. A line with no
        # digits is a section heading and must not absorb the next row.
        window = line

        if (
            not _RANGE.search(window)
            and _NUMBER.search(line)
            and index + 1 < len(lines)
            and _is_continuation(lines[index + 1])
        ):
            window = f"{line} {lines[index + 1]}"

        # "Haemoglobin 15.8 gms% M-(13.5 to 18.0)" is followed by
        # "F-(11.5 to 16.4)" on its own line. Both belong to this
        # row, and without the second one there is no way to know
        # the first is sex-specific at all.
        elif (
            index + 1 < len(lines)
            and _SEX_ONLY_LINE.match(lines[index + 1])
        ):
            window = f"{line} {lines[index + 1]}"

        range_match = _RANGE.search(window)

        limit_kind = limit_value = limit_start = None

        if not range_match:

            limit_kind, limit_value, limit_start = _single_limit(
                window
            )

            if not limit_kind:
                continue

        if _SKIP_LINE.search(window):
            continue

        # Band words rule out a range row, where the applicable band
        # cannot be known. On a lone limit they are just the label
        # the lab printed beside it ("Desirable < 200").
        if range_match and _BAND_WORDS.search(window):
            continue

        if _SPLIT_NUMBER.search(window):
            continue

        if limit_kind:
            low = high = None
            range_start = limit_start
        else:
            low = _to_float(range_match.group("low"))
            high = _to_float(range_match.group("high"))
            range_start = range_match.start()

        # Rows that print one range for men and another for women.
        # Only engaged when the row actually offers a choice, so an
        # analyte whose name ends in "F" is not mistaken for one.
        sexed = {} if limit_kind else _sexed_ranges(window)

        if sexed:

            if not patient_sex:

                # Guessing is what produced the wrong verdict. With
                # two ranges and no stated sex there is no basis to
                # choose, so the row gets no verdict at all.
                if len(sexed) > 1:
                    print(
                        "[VERIFY] Skipped a sex-specific row: the "
                        "report does not state the patient's sex."
                    )
                    continue

            elif patient_sex not in sexed:

                # The row is labelled for the other sex only.
                continue

            else:
                low, high, _ = sexed[patient_sex]

            # The value sits before every range on the row. Taking it
            # from before the *chosen* range read 6.5 - the top of
            # the male range - as the patient's RBC count.
            range_start = min(
                [range_start] + [start for _, _, start in sexed.values()]
            )

        if not limit_kind and low >= high:
            continue

        prefix = window[:range_start]

        value_matches = list(_NUMBER.finditer(prefix))

        if not value_matches:
            continue

        value_match = value_matches[-1]
        value = _to_float(value_match.group())

        # Words like "Below" / "Normal" between the value and the
        # range mean the range belongs to another row.
        between = window[value_match.end():range_start]

        # On a limit row the label the lab printed - "Desirable",
        # "High Cholesterol:" - sits between the value and the
        # limit, so the interloper check would reject every one of
        # them. Requiring exactly one limit and no range in the row
        # does that job instead.
        if not limit_kind and _INTERLOPER.search(between):
            continue

        name = _extract_name(prefix[: value_match.start()])

        # A sub-row such as "1 Hour 05 mm 0 to 20" names only part of
        # the measurement: the analyte is on the line above ("ESR
        # 1/2 Hour 02 mm"). Taken alone it produced a finding called
        # "Hour". Carry the label down so it reads "ESR 1 Hour".
        if re.match(r"\s*\d", line) and index > 0:

            label = _leading_label(lines[index - 1])

            if label and not _BAD_NAME.search(label):

                own = re.sub(
                    r"\s+", " ", prefix[: value_match.start()]
                ).strip(" :\t-|.")

                name = f"{label} {own}".strip() if own else label

        if not name:
            continue

        key = name.lower()

        if key in seen:
            continue

        seen.add(key)

        if limit_kind == "upper":
            text = (
                f"{name}: measured value {value:g} against the "
                f"recommended limit of under {limit_value:g} is "
                f"{'within' if value < limit_value else 'above'} "
                "that limit."
            )

        elif limit_kind == "lower":
            text = (
                f"{name}: measured value {value:g} against the "
                f"recommended minimum of {limit_value:g} is "
                f"{'below' if value < limit_value else 'at or above'} "
                "that minimum."
            )

        else:
            text = (
                f"{name}: measured value {value:g} against reference "
                f"range {low:g} to {high:g} is "
                f"{_classify(value, low, high)}."
            )

        findings.append({"name": name, "text": text})

        if len(findings) >= max_findings:
            break

    return findings


# ------------------------------------------------------------------
# Values read by the vision model
# ------------------------------------------------------------------

def verify_extracted_rows(rows, patient_sex=None):
    """Classify rows from :mod:`services.report_extractor`.

    The vision model reports what is printed - name, value, range -
    and nothing else. The comparison happens here, in Python, so a
    verdict never depends on a language model's arithmetic. The
    output is the same shape the text parser produces, so callers
    cannot tell the two apart.
    """

    if not rows:
        return []

    patient_sex = (patient_sex or "")[:1].upper()

    # Rows split by sex arrive as separate entries with the same
    # name, so they are grouped before one is chosen.
    grouped = {}

    for row in rows:
        grouped.setdefault(row["name"], []).append(row)

    findings = []

    for name, candidates in grouped.items():

        # A sex label is only meaningful when the row actually
        # offers a choice. The vision model sometimes tags a single
        # row - it labelled a cholesterol row "M" - and honouring
        # that would drop the row entirely for a female patient.
        offered = {row["sex"] for row in candidates if row["sex"]}

        if len(offered) > 1:

            if not patient_sex:
                # A choice with no stated sex is no basis to choose,
                # exactly as in the text parser.
                continue

            matching = [
                row for row in candidates
                if row["sex"] == patient_sex
            ]

            if not matching:
                continue

            chosen = matching[0]

        else:
            chosen = candidates[0]

        value = chosen["value"]
        low = chosen["low"]
        high = chosen["high"]

        if low is not None and high is not None:
            text = (
                f"{name}: measured value {value:g} against reference "
                f"range {low:g} to {high:g} is "
                f"{_classify(value, low, high)}."
            )

        elif high is not None:
            text = (
                f"{name}: measured value {value:g} against the "
                f"recommended limit of under {high:g} is "
                f"{'within' if value < high else 'above'} that limit."
            )

        else:
            text = (
                f"{name}: measured value {value:g} against the "
                f"recommended minimum of {low:g} is "
                f"{'below' if value < low else 'at or above'} "
                "that minimum."
            )

        findings.append({"name": name, "text": text})

    return findings


def merge_findings(text_findings, vision_findings):
    """Combine verdicts from the text parser and the vision model.

    Where both read the same analyte and agree, one copy is kept.
    Where they disagree, neither is kept: two readers of the same row
    reaching different numbers is exactly the ambiguity this module
    refuses to guess at, and a wrong verdict is worse than none.
    """

    merged = {}

    for finding in text_findings:
        merged[finding["name"].lower()] = finding

    for finding in vision_findings:

        key = finding["name"].lower()

        existing = merged.get(key)

        if existing is None:
            merged[key] = finding

        elif existing["text"] != finding["text"]:
            print(
                "[VERIFY] Readers disagree on "
                f"'{finding['name']}' - no verdict given."
            )
            merged[key] = None

    return [finding for finding in merged.values() if finding]


def parse_reference(reference):
    """Turn a printed reference range into (sex, low, high) tuples.

    The vision model transcribes the range exactly as the lab
    printed it - "M- (13.5 to 18.0)", "4000 to 10000", "<200mg/dl" -
    and the parsing happens here, with the same patterns the text
    parser uses. Asking the model for low and high as separate
    numbers instead put "4000 to 10000" in the low field and left
    high empty, turning every range into a one-sided limit.

    ``low`` or ``high`` is None for a one-sided limit. An empty sex
    means the range is not sex-specific.
    """

    if not reference:
        return []

    sexed = _sexed_ranges(reference)

    if sexed:
        return [
            (sex, low, high)
            for sex, (low, high, _) in sexed.items()
        ]

    match = _RANGE.search(reference)

    if match:

        low = _to_float(match.group("low"))
        high = _to_float(match.group("high"))

        return [("", low, high)] if low < high else []

    kind, limit, _ = _single_limit(reference)

    if kind == "upper":
        return [("", None, limit)]

    if kind == "lower":
        return [("", limit, None)]

    return []


def has_damaged_numbers(text):
    """Whether extraction split a grouped number, as in "2,89,00 0".

    A PDF's text layer can break a number across column boundaries.
    The row is skipped rather than guessed, so the value is lost
    silently - which is the signal to read the page as an image
    instead.
    """

    return bool(text) and bool(_SPLIT_NUMBER.search(text))

