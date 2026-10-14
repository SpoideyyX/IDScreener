"""
ocr_validator.py

Phase 3 module — OCR text extraction + rule-based field validation.

Approach
--------
1. Run EasyOCR over the document to get all text boxes + confidences.
2. Use regex patterns to pull out candidate values for Name, DOB, ID Number,
   and Expiry Date from the raw OCR text (documents don't come with labeled
   fields, so we pattern-match on *shape*: dates look like dates, ID numbers
   look like alphanumeric codes of a certain length, etc.)
3. Run logical/rule-based checks on what we found:
     - DOB must be a real, past calendar date, and imply a plausible age
     - Expiry must be a real calendar date
     - ID number must match the expected format (configurable regex)
     - Expiry should typically be after DOB (sanity check, catches garbled OCR)
4. Combine per-field validity + OCR confidence into a single 0-1 score.

Note for your viva: OCR-based validation like this catches "logic" fraud
(mismatched dates, malformed ID numbers) that a forger might not think to
fix even when the pixels look convincing — it's a complementary signal to
ELA, not a replacement for it.
"""

import re
from datetime import date

import easyocr
import numpy as np

import config

# --------------------------------------------------------------------------- #
# Regex patterns
# --------------------------------------------------------------------------- #

# Matches dates like 01/01/1999, 1-1-99, 2030.12.31, etc.
DATE_PATTERN = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")

# Matches common ID-number shapes: letters+digits, 6-12 chars, at least one digit.
ID_NUMBER_PATTERN = re.compile(r"\b(?=[A-Z0-9]{6,12}\b)(?=[A-Z0-9]*\d)[A-Z0-9]{6,12}\b")

# A crude "looks like a person's name" pattern: 2-4 words, each either
# Title Case ("John Smith") or ALL CAPS ("JOHN SMITH") — most printed ID
# cards use all-caps for the name field, so both forms must be matched.
# A lone single letter (middle initial, e.g. "JOHN A SMITH") is allowed too.
NAME_PATTERN = re.compile(
    r"\b("
    r"[A-Z][a-z]+(?:\s+[A-Z](?:[a-z]+)?){1,3}"
    r"|[A-Z]{2,}(?:\s+[A-Z]{1,})?(?:\s+[A-Z]{2,}){0,2}"
    r")\b"
)

# Field-label words that can get swept up by NAME_PATTERN (e.g. "DOB EXP")
# and should be discarded if that's all a "name" match turns out to be.
_NAME_STOPWORDS = {
    "DOB", "EXP", "EXPIRY", "EXPIRES", "ID", "DATE", "BIRTH",
    "ISSUED", "ISSUE", "NATIONALITY", "SEX", "GENDER", "ADDRESS", "SIGNATURE",
}


def _looks_like_real_name(candidate: str) -> bool:
    """Reject a NAME_PATTERN match that's actually just field-label text."""
    words = {w.upper() for w in candidate.split()}
    return not words.issubset(_NAME_STOPWORDS)

_reader_cache = None


def _get_reader() -> "easyocr.Reader":
    global _reader_cache
    if _reader_cache is None:
        _reader_cache = easyocr.Reader(["en"], gpu=False)
    return _reader_cache


# --------------------------------------------------------------------------- #
# OCR extraction
# --------------------------------------------------------------------------- #

def run_ocr(image_path: str) -> list:
    """Returns a list of (text, confidence) tuples from EasyOCR, filtered by
    config.MIN_OCR_CONFIDENCE."""
    reader = _get_reader()
    raw_results = reader.readtext(image_path)  # [(bbox, text, confidence), ...]

    filtered = [
        (text.strip(), float(conf))
        for (_bbox, text, conf) in raw_results
        if conf >= config.MIN_OCR_CONFIDENCE and text.strip()
    ]
    return filtered


def _parse_date(match: "re.Match") -> "date | None":
    """Try to build a real date.date from a regex match, trying both
    DD/MM/YYYY and MM/DD/YYYY interpretations since ID formats vary by
    country and we can't assume which one we're looking at."""
    d1, d2, y = match.groups()
    d1, d2 = int(d1), int(d2)
    y = int(y)
    if y < 100:
        y += 2000 if y < 50 else 1900

    for day, month in ((d1, d2), (d2, d1)):
        try:
            return date(y, month, day)
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------- #
# Field extraction
# --------------------------------------------------------------------------- #

def extract_fields(ocr_results: list) -> dict:
    """Scan OCR text lines for date-like, ID-like, and name-like strings.
    This is intentionally simple pattern matching rather than a labeled-field
    parser — appropriate scope for a course project, and a good place to
    mention 'template-specific field-position matching' as future work."""
    all_text = [text for text, _conf in ocr_results]
    joined = " ".join(all_text)

    dates_found = []
    for m in DATE_PATTERN.finditer(joined):
        parsed = _parse_date(m)
        if parsed:
            dates_found.append(parsed)

    # Heuristic: earliest plausible past date = DOB, latest date = expiry.
    today = date(config.CURRENT_YEAR, 1, 1)
    past_dates = sorted(d for d in dates_found if d < today)
    future_or_recent = sorted(dates_found, reverse=True)

    dob = past_dates[0] if past_dates else None
    expiry = future_or_recent[0] if future_or_recent else None
    if dob is not None and expiry == dob and len(dates_found) > 1:
        expiry = future_or_recent[1] if len(future_or_recent) > 1 else None

    id_number = None
    for text in all_text:
        m = ID_NUMBER_PATTERN.search(text.replace(" ", ""))
        if m:
            id_number = m.group(0)
            break

    # Search line-by-line (not the joined string) so a name can't accidentally
    # bridge across two unrelated OCR detections (e.g. end of one field +
    # start of the next).
    name = None
    for line in all_text:
        for name_match in NAME_PATTERN.finditer(line):
            candidate = name_match.group(1)
            if _looks_like_real_name(candidate) and len(candidate.split()) >= 2:
                name = candidate
                break
        if name:
            break

    return {
        "name": name,
        "dob": dob,
        "id_number": id_number,
        "expiry": expiry,
        "raw_text": all_text,
    }


# --------------------------------------------------------------------------- #
# Validation rules
# --------------------------------------------------------------------------- #

def validate_fields(fields: dict) -> list:
    """Run rule-based checks on extracted fields. Returns a list of human-
    readable error strings (empty list = all checks passed)."""
    errors = []
    today = date(config.CURRENT_YEAR, 1, 1)

    if fields["name"] is None:
        errors.append("Could not locate a name field on the document.")

    if fields["id_number"] is None:
        errors.append("Could not locate a valid-format ID number.")

    dob = fields["dob"]
    if dob is None:
        errors.append("Could not locate a valid date of birth.")
    else:
        age_years = (today - dob).days / 365.25
        if age_years < 0:
            errors.append("Date of birth is in the future.")
        elif age_years > 120:
            errors.append("Date of birth implies an implausible age (>120 years).")

    expiry = fields["expiry"]
    if expiry is None:
        errors.append("Could not locate a valid expiry date.")
    elif dob is not None and expiry <= dob:
        errors.append("Expiry date is not after date of birth — likely tampered or misread field.")

    return errors


# --------------------------------------------------------------------------- #
# Scoring + Phase-1 interface contract
# --------------------------------------------------------------------------- #

def _confidence_score(ocr_results: list) -> float:
    """Average OCR confidence across all detected text, as a proxy for how
    'readable'/clean the document capture was."""
    if not ocr_results:
        return 0.0
    return float(np.mean([conf for _text, conf in ocr_results]))


def extract_and_validate(image_path: str) -> dict:
    """Phase-1 interface contract.

    Returns
    -------
    dict with:
      fields : dict — extracted name / dob / id_number / expiry / raw_text
      score  : float in [0, 1] — combined OCR confidence + field-validity score
      errors : list[str] — human-readable validation failures
    """
    ocr_results = run_ocr(image_path)
    fields = extract_fields(ocr_results)
    errors = validate_fields(fields)

    ocr_confidence = _confidence_score(ocr_results)

    # Field validity: what fraction of the 4 expected fields were both found
    # AND passed their individual checks. Missing/invalid fields are exactly
    # what `errors` flags, so we derive this count from how many error-free
    # fields we ended up with rather than duplicating the logic.
    expected_fields = 4  # name, dob, id_number, expiry
    found_fields = sum(1 for k in ("name", "dob", "id_number", "expiry") if fields[k] is not None)
    field_validity = found_fields / expected_fields
    # Penalize logical errors (e.g. expiry <= dob) beyond simple "missing field" errors.
    logic_error_penalty = 0.15 * max(0, len(errors) - (expected_fields - found_fields))
    field_validity = max(0.0, field_validity - logic_error_penalty)

    # Final OCR/validation score blends how confidently we could read the
    # document with how internally consistent what we read turned out to be.
    score = 0.4 * ocr_confidence + 0.6 * field_validity

    return {
        "fields": fields,
        "score": float(np.clip(score, 0.0, 1.0)),
        "errors": errors,
    }


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run OCR extraction + validation on a document image.")
    parser.add_argument("image_path", type=str)
    args = parser.parse_args()

    result = extract_and_validate(args.image_path)
    print("Extracted fields:")
    for k, v in result["fields"].items():
        if k != "raw_text":
            print(f"  {k:12s}: {v}")
    print(f"\nOCR/validation score: {result['score']:.3f}")
    if result["errors"]:
        print("Errors:")
        for e in result["errors"]:
            print(f"  - {e}")
    else:
        print("No validation errors found.")
    sys.exit(0)
