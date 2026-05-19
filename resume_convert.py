"""
Resume PDF Text Extractor — Multi-Column Layout Fix
=====================================================
Uses pdfplumber crop-based column detection to handle
sidebar + main content resume layouts correctly.

Install:
    pip install pdfplumber pytesseract pdf2image Pillow

Windows extras:
    Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
    Poppler:   https://github.com/oschwartz10612/poppler-windows/releases
               Add the /bin folder to your system PATH.
"""

import os
import sys
import re

import pdfplumber
import pytesseract
from pdf2image import convert_from_path


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Uncomment if Tesseract is not on PATH:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Uncomment if Poppler is not on PATH (Windows):
# POPPLER_PATH = r"C:\poppler\Library\bin"
POPPLER_PATH    = None

OCR_DPI         = 300
OCR_LANG        = "eng"
MIN_TEXT_LENGTH = 50    # chars — if less, fall back to OCR
SAVE_OUTPUT     = True


# ─────────────────────────────────────────────────────────────
# COLUMN SPLIT DETECTION
# ─────────────────────────────────────────────────────────────

def find_column_splits(words, page_width, search_min=60, search_max=0.65, min_gap=15):
    """
    Find vertical column boundaries by looking for the largest gap in
    word left-edges (x0) within the likely sidebar zone (60px to 65% of width).
    Returns a list of x-coordinates where columns split.
    """
    max_x = page_width * search_max
    # Collect unique rounded x0 values in the search zone
    x0s = sorted(set(round(w["x0"]) for w in words if search_min < w["x0"] < max_x))

    if not x0s:
        return []

    # Find largest gap between consecutive x-positions
    best_gap, best_split = 0, None
    for i in range(len(x0s) - 1):
        gap = x0s[i + 1] - x0s[i]
        if gap > best_gap and gap >= min_gap:
            best_gap = gap
            best_split = x0s[i] + gap / 2   # midpoint of the gap

    return [best_split] if best_split else []


# ─────────────────────────────────────────────────────────────
# PDFPLUMBER COLUMN-AWARE EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_page(page):
    """
    Detect columns on a page by analysing word positions,
    then crop and extract each column independently.
    """
    words = page.extract_words(x_tolerance=3, y_tolerance=3)
    if not words:
        return ""

    splits = find_column_splits(words, page.width)

    if not splits:
        # Single-column — standard extraction
        return page.extract_text() or ""

    # Build column bounding boxes
    boundaries = [0] + splits + [page.width]
    column_texts = []

    for i in range(len(boundaries) - 1):
        x0, x1 = boundaries[i], boundaries[i + 1]
        cropped = page.crop((x0, 0, x1, page.height))
        text = cropped.extract_text()
        if text and text.strip():
            column_texts.append(text.strip())

    return "\n\n".join(column_texts)


def extract_with_pdfplumber(pdf_path):
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = extract_page(page)
            if text.strip():
                pages_text.append(
                    f"{'='*54}\n  PAGE {i}\n{'='*54}\n{text.strip()}"
                )
    return "\n\n".join(pages_text)


# ─────────────────────────────────────────────────────────────
# OCR FALLBACK (scanned PDFs)
# ─────────────────────────────────────────────────────────────

def extract_with_ocr(pdf_path):
    kwargs = {"dpi": OCR_DPI}
    if POPPLER_PATH:
        kwargs["poppler_path"] = POPPLER_PATH

    print("  [OCR] Converting PDF pages to images ...")
    pages = convert_from_path(pdf_path, **kwargs)
    results = []

    for i, img in enumerate(pages, start=1):
        print(f"  [OCR] Tesseract on page {i}/{len(pages)} ...")
        text = pytesseract.image_to_string(img.convert("L"), lang=OCR_LANG)
        if text.strip():
            results.append(f"{'='*54}\n  PAGE {i}\n{'='*54}\n{text.strip()}")

    return "\n\n".join(results)


# ─────────────────────────────────────────────────────────────
# TEXT CLEANUP
# ─────────────────────────────────────────────────────────────

RESUME_HEADERS = [
    # Exact-match headers (case-insensitive)
    "summary", "professional summary", "profile", "objective",
    "career objective", "about me", "about",
    "experience", "work experience", "professional experience",
    "employment history", "work history",
    "education", "academic background", "academic qualifications",
    "skills", "technical skills", "core competencies", "key skills",
    "projects", "personal projects", "academic projects",
    "certifications", "certificates", "licenses & certifications",
    "achievements", "awards", "honors", "awards & achievements",
    "interests", "hobbies", "hobbies & interests",
    "languages", "publications", "references",
    "volunteer", "volunteer experience",
    "training", "professional development",
    "contact", "contact information", "personal information",
]

# Build a regex that matches any header on its own line
_header_pattern = re.compile(
    r"^(" + "|".join(re.escape(h) for h in sorted(RESUME_HEADERS, key=len, reverse=True)) + r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _fuzzy_header_match(line):
    """Check if a line matches a known header after removing stray spaces (PDF artifact)."""
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return None
    # Remove single spaces within words (PDF extraction artifact like "Skil ls" -> "Skills")
    collapsed = re.sub(r"(?<=\w)\s(?=\w)", "", stripped)
    for header in RESUME_HEADERS:
        if collapsed.lower() == header.lower():
            return header.upper()
    return None


def label_resume_headers(text):
    """Replace lines that match known resume headers with [HEADER] format."""
    # First pass: exact regex match
    text = _header_pattern.sub(lambda m: f"[{m.group(1).strip().upper()}]", text)
    # Second pass: fuzzy match for PDF-broken words (e.g. "Skil ls")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("[") and line.endswith("]"):
            continue  # already labelled
        match = _fuzzy_header_match(line)
        if match:
            lines[i] = f"[{match}]"
    return "\n".join(lines)


def clean_text(text):
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)
    text = label_resume_headers(text)
    return text.strip()


# ─────────────────────────────────────────────────────────────
# MAIN HYBRID LOGIC
# ─────────────────────────────────────────────────────────────

def extract_resume_text(pdf_path):
    pdf_path = os.path.normpath(pdf_path)
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"\n  File   : {os.path.basename(pdf_path)}")
    print("-" * 54)

    print("  [1/2] Column-aware pdfplumber extraction ...")
    plumber_text  = extract_with_pdfplumber(pdf_path)
    plumber_chars = len(plumber_text.strip())
    print(f"        {plumber_chars:,} characters extracted")

    if plumber_chars >= MIN_TEXT_LENGTH:
        print("  [2/2] Result looks good — skipping OCR")
        final_text = plumber_text
    else:
        print(f"  [2/2] Too short (<{MIN_TEXT_LENGTH}) — running OCR fallback ...")
        ocr_text  = extract_with_ocr(pdf_path)
        final_text = ocr_text if not plumber_text.strip() else (
            "=== pdfplumber ===\n\n" + plumber_text +
            "\n\n=== OCR ===\n\n" + ocr_text
        )

    final_text = clean_text(final_text)
    print(f"  Done  : {len(final_text):,} characters total\n")
    return final_text


# ─────────────────────────────────────────────────────────────
# SAVE & PREVIEW HELPERS
# ─────────────────────────────────────────────────────────────

def save_text(text, pdf_path):
    out = os.path.splitext(pdf_path)[0] + "_extracted.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved : {out}")
    return out


def display_preview(text, chars=2500):
    print("=" * 54)
    print("  EXTRACTED TEXT PREVIEW")
    print("=" * 54)
    print(text[:chars])
    if len(text) > chars:
        print(f"\n  ... [{len(text) - chars:,} more chars — open the .txt file for full content]")
    print("=" * 54)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 54)
    print("   RESUME PDF → TEXT  (multi-column aware)")
    print("=" * 54)

    if len(sys.argv) > 1:
        # e.g.  python resume_extractor.py "C:\path\to\resume.pdf"
        pdf_path = " ".join(sys.argv[1:])
    else:
        print(r"\nPaste the full path to the PDF and press Enter.")
        print(r'  e.g.  C:\Users\bhanu\Desktop\chintan.pdf' + "\n")
        pdf_path = input("PDF path: ").strip()

    pdf_path = pdf_path.strip('"').strip("'")
    if not pdf_path:
        print("  No path entered. Exiting.")
        sys.exit(1)

    try:
        text = extract_resume_text(pdf_path)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    if SAVE_OUTPUT:
        save_path = os.path.splitext(pdf_path)[0] + "_extracted.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  Saved : {save_path}")

    display_preview(text)
    return text          # usable when imported as a module


if __name__ == "__main__":
    main()