"""Generate laboratory reports with known correct values.

No public dataset of laboratory reports with labelled values could be
downloaded without an account, and real patient reports cannot be
redistributed. Each report here is therefore written from a ground
truth table and then rendered four ways, so every value the system
reads can be scored against what is actually printed:

  digital.pdf   a PDF with a text layer, as a lab system exports it
  scanned.pdf   the same page as an image only, as a scanner produces
  clean.png     a straight, sharp image of the page
  photo.jpg     a phone-style capture: tilted, blurred, unevenly lit,
                JPEG-compressed and stored sideways with an EXIF
                orientation tag, as phones store portrait photos

The reports vary the things that break parsers in practice: ranges
written with hyphens or "to", sex-specific ranges, one-sided limits,
and Indian digit grouping.

    python experiments/eval/make_dataset.py
"""

import io
import json
from pathlib import Path

import fitz
from PIL import Image, ImageEnhance, ImageFilter

OUT = Path(__file__).resolve().parent / "dataset"

# (test, value as printed, unit, range as printed, expected status)
REPORTS = {
    "R1_cbc_male": {
        "panel": "COMPLETE BLOOD COUNT",
        "patient": ("Mr. Daniel Tan", "45/M"),
        "rows": [
            ("Haemoglobin", "14.6", "g/dL", "13.0 - 17.0", "within"),
            ("Total WBC Count", "11800", "/uL", "4000 - 10000", "above"),
            ("Neutrophils", "72", "%", "40 - 75", "within"),
            ("Lymphocytes", "18", "%", "20 - 40", "below"),
            ("Eosinophils", "3", "%", "1 - 6", "within"),
            ("Monocytes", "5", "%", "2 - 10", "within"),
            ("Platelet Count", "245000", "/uL", "150000 - 410000", "within"),
            ("RBC Count", "4.9", "mill/uL", "4.5 - 5.9", "within"),
            ("PCV", "44.1", "%", "40 - 50", "within"),
            ("MCV", "88", "fL", "83 - 101", "within"),
            ("MCH", "29.8", "pg", "27 - 32", "within"),
            ("MCHC", "33.9", "g/dL", "31.5 - 34.5", "within"),
        ],
    },
    "R2_cbc_female_sexed": {
        "panel": "HAEMATOLOGY",
        "patient": ("Ms. Priya Nair", "29/F"),
        "rows": [
            ("Haemoglobin", "11.4", "g/dL", "M-13.0-17.0 F-12.0-15.0", "below"),
            ("PCV", "35.2", "%", "M-40-50 F-36-46", "below"),
            ("RBC Count", "4.3", "mill/uL", "M-4.5-5.9 F-3.8-4.8", "within"),
            ("Ferritin", "12", "ng/mL", "M-30-400 F-13-150", "below"),
            ("ESR", "18", "mm/hr", "M-0-15 F-0-20", "within"),
            ("Serum Creatinine", "0.7", "mg/dL", "M-0.7-1.3 F-0.5-1.1", "within"),
        ],
    },
    "R3_lipid_one_sided": {
        "panel": "LIPID PROFILE AND GLUCOSE",
        "patient": ("Mr. Ahmad Rahman", "52/M"),
        "rows": [
            ("Total Cholesterol", "232", "mg/dL", "< 200", "above"),
            ("Triglycerides", "140", "mg/dL", "< 150", "within"),
            ("HDL Cholesterol", "38", "mg/dL", "> 40", "below"),
            ("LDL Cholesterol", "162", "mg/dL", "< 100", "above"),
            ("Fasting Glucose", "108", "mg/dL", "70 - 100", "above"),
            ("HbA1c", "5.9", "%", "4.0 - 5.6", "above"),
        ],
    },
    "R4_lft_kft_to_ranges": {
        "panel": "LIVER AND KIDNEY FUNCTION",
        "patient": ("Mdm. Lim Siew Hong", "61/F"),
        "rows": [
            ("Total Bilirubin", "1.6", "mg/dL", "0.3 to 1.2", "above"),
            ("SGOT", "32", "U/L", "8 to 40", "within"),
            ("SGPT", "58", "U/L", "5 to 35", "above"),
            ("Alkaline Phosphatase", "96", "U/L", "44 to 147", "within"),
            ("Albumin", "3.2", "g/dL", "3.5 to 5.0", "below"),
            ("Blood Urea Nitrogen", "14", "mg/dL", "8 to 25", "within"),
            ("Serum Creatinine", "1.0", "mg/dL", "0.6 to 1.2", "within"),
            ("Platelet Count", "1,80,000", "/uL", "1,50,000 to 4,50,000", "within"),
        ],
    },
    "R5_thyroid_electrolytes": {
        "panel": "THYROID AND ELECTROLYTES",
        "patient": ("Mr. Wei Jie Ong", "38/M"),
        "rows": [
            ("TSH", "6.8", "uIU/mL", "0.4 - 4.0", "above"),
            ("Free T4", "1.1", "ng/dL", "0.8 - 1.8", "within"),
            ("Sodium", "133", "mmol/L", "135 - 145", "below"),
            ("Potassium", "4.2", "mmol/L", "3.5 - 5.1", "within"),
            ("Chloride", "101", "mmol/L", "98 - 107", "within"),
            ("Calcium", "9.4", "mg/dL", "8.6 - 10.3", "within"),
            ("Vitamin D", "18", "ng/mL", "30 - 100", "below"),
            ("Vitamin B12", "410", "pg/mL", "200 - 900", "within"),
        ],
    },
}


def report_lines(report):
    name, age_sex = report["patient"]
    lines = [
        "CITYCARE DIAGNOSTIC LABORATORY",
        "Accredited Clinical Laboratory",
        "",
        f"Patient Name : {name:<24} Age/Sex : {age_sex}",
        "Sample Date  : 12-03-2026               Ref. By : Dr. K. Menon",
        "",
        report["panel"],
        f"{'Test':<24}{'Result':<12}{'Unit':<10}Reference Range",
        "-" * 70,
    ]
    for test, value, unit, ref, _ in report["rows"]:
        lines.append(f"{test:<24}{value:<12}{unit:<10}{ref}")
    lines += ["-" * 70, "", "*** End of Report ***"]
    return lines


def digital_pdf(lines):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 60
    for line in lines:
        page.insert_text((40, y), line, fontname="cour", fontsize=10)
        y += 16
    return doc.tobytes()


def scanned_pdf(digital):
    """Rasterise the page and rebuild a PDF with no text layer."""
    pix = fitz.open(stream=digital, filetype="pdf")[0].get_pixmap(dpi=150)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    image = image.filter(ImageFilter.GaussianBlur(0.4))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=buf.getvalue())
    return doc.tobytes()


def clean_png(digital):
    pix = fitz.open(stream=digital, filetype="pdf")[0].get_pixmap(dpi=150)
    return pix.tobytes("png")


def phone_photo(png):
    """Tilt, blur, uneven light, JPEG, stored sideways with EXIF 6."""
    page = Image.open(io.BytesIO(png)).convert("RGB")
    page = page.rotate(-2.5, expand=True, fillcolor=(235, 232, 225))
    page = page.filter(ImageFilter.GaussianBlur(0.8))

    # Lighter at the top, darker at the bottom, as under a desk lamp.
    shade = Image.linear_gradient("L").resize(page.size)
    shade = ImageEnhance.Brightness(shade.convert("RGB")).enhance(0.35)
    page = Image.blend(page, Image.new("RGB", page.size, (0, 0, 0)), 0)
    page = Image.composite(
        ImageEnhance.Brightness(page).enhance(0.8), page, shade.convert("L")
    )

    # A phone held upright stores the sensor image sideways and records
    # orientation 6, leaving the viewer to rotate it for display.
    stored = page.rotate(90, expand=True)
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    stored.save(buf, format="JPEG", quality=60, exif=exif)
    return buf.getvalue()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    truth = {}
    for key, report in REPORTS.items():
        folder = OUT / key
        folder.mkdir(exist_ok=True)
        lines = report_lines(report)
        digital = digital_pdf(lines)
        png = clean_png(digital)
        (folder / "digital.pdf").write_bytes(digital)
        (folder / "scanned.pdf").write_bytes(scanned_pdf(digital))
        (folder / "clean.png").write_bytes(png)
        (folder / "photo.jpg").write_bytes(phone_photo(png))
        (folder / "report.txt").write_text("\n".join(lines))
        truth[key] = {
            "sex": report["patient"][1][-1],
            "rows": [
                {"test": t, "value": v, "unit": u, "range": r, "status": s}
                for t, v, u, r, s in report["rows"]
            ],
        }
    (OUT / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    total = sum(len(r["rows"]) for r in truth.values())
    print(f"{len(truth)} reports, {total} labelled values, 4 renderings each")


if __name__ == "__main__":
    main()
