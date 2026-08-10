"""
PaddleOCR fallback engine.

Tesseract handles clean scans well but fails on degraded phone photos.
PaddleOCR's detection+recognition pipeline is far more robust on those.
Loaded lazily as a singleton because initialization takes several seconds.

All inference runs locally in-process - no network calls with image data.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("deed-ocr-service.paddle")

_lock = threading.Lock()
_engine = None
_engine_failed = False

MAX_WIDTH = 1600


def _get_engine():
    """Lazy singleton. Returns None if PaddleOCR is unavailable."""
    global _engine, _engine_failed
    if _engine is not None or _engine_failed:
        return _engine
    with _lock:
        if _engine is not None or _engine_failed:
            return _engine
        try:
            from paddleocr import PaddleOCR
            _engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            logger.info("PaddleOCR engine initialized")
        except Exception:  # noqa: BLE001
            logger.exception("PaddleOCR unavailable - falling back to Tesseract only")
            _engine_failed = True
    return _engine


def get_text_lines(img: np.ndarray) -> list[str]:
    """Run PaddleOCR on a BGR image, return recognized text lines (top-down)."""
    engine = _get_engine()
    if engine is None:
        return []

    h, w = img.shape[:2]
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / float(w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    try:
        result = engine.ocr(img, cls=True)
    except Exception:  # noqa: BLE001
        logger.exception("PaddleOCR inference failed")
        return []

    lines = []
    for page in result or []:
        for entry in page or []:
            # entry: [box, (text, confidence)]
            try:
                text, conf = entry[1]
            except (IndexError, TypeError):
                continue
            if conf >= 0.5 and text and text.strip():
                lines.append(text.strip())
    return lines


_MRZ_CHARS_RE = re.compile(r"[^A-Z0-9<]")


def mrz_candidate_lines(lines: list[str]) -> list[str]:
    """Clean PaddleOCR lines into MRZ-shaped candidates."""
    out = []
    for raw in lines:
        cleaned = _MRZ_CHARS_RE.sub("", raw.upper().replace(" ", ""))
        if len(cleaned) >= 20:
            out.append(cleaned)
    return out


_MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
_DATE_NUM_RE = re.compile(r"\b(\d{1,2})[\s/.\-](\d{1,2})[\s/.\-](\d{4})\b")
_DATE_MON_RE = re.compile(r"\b(\d{1,2})[\s/.\-]?([A-Z]{3})[\s/.\-]?(\d{4})\b", re.IGNORECASE)
_DATE_ISO_RE = re.compile(r"\b(\d{4})[\s/.\-](\d{1,2})[\s/.\-](\d{1,2})\b")

_ISSUE_LABEL_RE = re.compile(
    r"date\s*of\s*issue|issue\s*date|issued\s*on|date\s*de\s*d[ée]livrance|fecha\s*de\s*expedici[oó]n",
    re.IGNORECASE,
)
_EXPIRY_LABEL_RE = re.compile(r"expir", re.IGNORECASE)

_POB_LABEL_RE = re.compile(
    r"place\s*of\s*birth|birth\s*place|lieu\s*de\s*naissance|lugar\s*de\s*nacimiento",
    re.IGNORECASE,
)


def _parse_date_from(text: str) -> Optional[str]:
    """Extract the first plausible date in a line, returned as ISO YYYY-MM-DD."""
    m = _DATE_MON_RE.search(text)
    if m:
        d, mon, y = int(m.group(1)), _MONTHS.get(m.group(2).upper()), int(m.group(3))
        if mon and 1 <= d <= 31 and 1980 <= y <= 2050:
            return "%04d-%02d-%02d" % (y, mon, d)
    m = _DATE_NUM_RE.search(text)
    if m:
        d, mon, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mon <= 12 and 1 <= d <= 31 and 1980 <= y <= 2050:
            return "%04d-%02d-%02d" % (y, mon, d)
    m = _DATE_ISO_RE.search(text)
    if m:
        y, mon, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mon <= 12 and 1 <= d <= 31 and 1980 <= y <= 2050:
            return "%04d-%02d-%02d" % (y, mon, d)
    return None


def find_issue_date(lines: list[str]) -> Optional[str]:
    """Find the printed Date of Issue. Skips lines that mention expiry."""
    from datetime import date as _date
    for i, line in enumerate(lines):
        if _ISSUE_LABEL_RE.search(line) and not _EXPIRY_LABEL_RE.search(line):
            for candidate in (line, lines[i + 1] if i + 1 < len(lines) else "",
                              lines[i + 2] if i + 2 < len(lines) else ""):
                parsed = _parse_date_from(candidate)
                # Issue dates are in the past; reject future dates (those are
                # likely the expiry sitting on an adjacent line)
                if parsed and parsed <= _date.today().isoformat():
                    return parsed
    return None


_POB_VALUE_RE = re.compile(r"[A-Za-z][A-Za-z\s,.'\-]{2,39}")


def find_place_of_birth(lines: list[str]) -> Optional[str]:
    """Find the printed Place of Birth (Latin-script values only)."""
    for i, line in enumerate(lines):
        m = _POB_LABEL_RE.search(line)
        if not m:
            continue
        # value on the same line after the label, else the next line
        candidates = [line[m.end():]]
        if i + 1 < len(lines):
            candidates.append(lines[i + 1])
        for cand in candidates:
            cand = cand.strip(" :;/.-")
            # skip if this is another label or mostly digits
            if _POB_LABEL_RE.search(cand) or _ISSUE_LABEL_RE.search(cand):
                continue
            if sum(ch.isdigit() for ch in cand) > 2:
                continue
            vm = _POB_VALUE_RE.search(cand)
            if vm:
                value = vm.group(0).strip()
                if len(value) >= 3 and value.upper() not in ("SEX", "DATE"):
                    return value.title()
    return None


_SEX_RE = re.compile(r"\b(?:sex|sexe|sexo)\b[^A-Za-z0-9]{0,8}([MF])\b", re.IGNORECASE)
# Standalone M/F line right after a Sex label line (labels and values often
# land in separate detection boxes)
_SEX_LABEL_RE = re.compile(r"\b(?:sex|sexe|sexo)\b", re.IGNORECASE)
_MF_ONLY_RE = re.compile(r"^\s*([MF])\s*$", re.IGNORECASE)


def find_sex(lines: list[str]) -> Optional[str]:
    """Look for the printed Sex field in recognized lines."""
    for i, line in enumerate(lines):
        m = _SEX_RE.search(line)
        if m:
            return m.group(1).upper()
        if _SEX_LABEL_RE.search(line):
            # check the same line minus label, then the next two lines
            for j in (i + 1, i + 2):
                if j < len(lines):
                    m2 = _MF_ONLY_RE.match(lines[j])
                    if m2:
                        return m2.group(1).upper()
    return None


# --------------------------------------------------------------------------
# Label-independent field extraction
#
# Diagnosis (2026-08-10): on real passports the VALUES are large and read
# cleanly, but the LABELS are small stylised multilingual print that OCR
# mangles - "Place of Birth" -> "Place of Birh", "Place" -> "Pace",
# "Date of Issue" -> "Dat". Anchoring on labels therefore loses data that is
# plainly visible. These helpers anchor on the MRZ and on value shape instead.
# --------------------------------------------------------------------------

# Country names used only for the no-readable-label fallback. Kept local so
# this module never imports from app.ocr (which would be circular).
_COUNTRY_NAMES = [
    "United Arab Emirates","Jordan","Sri Lanka","Syria","United Kingdom","India","Pakistan",
    "Egypt","Saudi Arabia","United States","Germany","Kuwait","Palestine","Philippines",
    "Nigeria","Canada","France","Australia","South Africa","Lebanon","Iraq","Yemen","Morocco",
    "Tunisia","Algeria","Bahrain","Oman","Qatar","Turkey","Russia","China","Japan","South Korea",
    "Bangladesh","Nepal","Afghanistan","Iran","Italy","Spain","Netherlands","Belgium",
    "Switzerland","Sweden","Norway","Poland","Ukraine","Greece","Portugal","Ireland",
    "New Zealand","Singapore","Malaysia","Indonesia","Thailand","Vietnam","Ethiopia","Kenya",
    "Ghana","Sudan","Somalia","Libya","Cyprus","Maldives","Uganda","Tanzania","Myanmar",
    "Hong Kong","Colombia","Israel","Cameroon","Eritrea","Azerbaijan","Kazakhstan","Uzbekistan",
    "Czech Republic","Austria","Denmark","Finland","Hungary","Romania","Bulgaria","Croatia",
    "Serbia","Brazil","Argentina","Mexico","Georgia","Armenia","Malta","Luxembourg","Estonia",
    "Latvia","Lithuania",
]

_MONTHS3 = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
            "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

# OCR routinely swaps these glyphs inside month abbreviations
_GLYPH_FIX = {"0":"O","1":"I","5":"S","8":"B","2":"Z","6":"G","4":"A"}


def _repair_month(tok: str):
    """Map a possibly-garbled 3-char month token to a month number."""
    t = "".join(_GLYPH_FIX.get(c, c) for c in tok.upper())
    if t in _MONTHS3:
        return _MONTHS3[t]
    # allow a single-character difference (e.g. 'NOU' -> 'NOV')
    for name, num in _MONTHS3.items():
        if len(t) == 3 and sum(1 for a, b in zip(t, name) if a != b) <= 1:
            return num
    return None


_DATE_PATTERNS = [
    # 06OCT2022 / 060CT2022 / 19AUG 2025 / 18 AUG2035
    re.compile(r"(\d{1,2})\s*([A-Z0-9]{3})\s*(\d{4})", re.I),
    # 10/08/2026, 10-08-2026, 10.08.2026
    re.compile(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})"),
    # 2026-08-10
    re.compile(r"(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})"),
]


def find_all_dates(lines):
    """Return every plausible date on the page as ISO strings, deduped."""
    from datetime import date as _date
    found = []
    text = " ".join(lines)
    for i, pat in enumerate(_DATE_PATTERNS):
        for m in pat.finditer(text):
            try:
                if i == 0:
                    d = int(m.group(1)); mon = _repair_month(m.group(2)); y = int(m.group(3))
                elif i == 1:
                    d = int(m.group(1)); mon = int(m.group(2)); y = int(m.group(3))
                else:
                    y = int(m.group(1)); mon = int(m.group(2)); d = int(m.group(3))
            except (TypeError, ValueError):
                continue
            if not mon or not (1 <= mon <= 12) or not (1 <= d <= 31):
                continue
            if not (1900 <= y <= 2060):
                continue
            try:
                iso = _date(y, mon, d).isoformat()
            except ValueError:
                continue
            if iso not in found:
                found.append(iso)
    return found


def classify_issue_date(all_dates, dob_iso, expiry_iso):
    """Pick the issue date by elimination against the MRZ-verified dates.

    The MRZ gives DOB and expiry with checksums, so whatever else appears on
    the page and sits sensibly between them is the issue date. Prefers a
    candidate exactly 5 or 10 years before expiry, the usual passport terms.
    """
    from datetime import date as _date
    today = _date.today().isoformat()
    cands = [d for d in all_dates if d != dob_iso and d != expiry_iso]
    cands = [d for d in cands if d <= today]
    if dob_iso:
        cands = [d for d in cands if d > dob_iso]
    if expiry_iso:
        cands = [d for d in cands if d < expiry_iso]
    if not cands:
        return None, ""
    if expiry_iso:
        for term in (10, 5, 7):
            for d in cands:
                try:
                    if int(expiry_iso[:4]) - int(d[:4]) == term and d[5:7] == expiry_iso[5:7]:
                        return d, "matched %d-year passport term against MRZ expiry" % term
                except ValueError:
                    continue
    # otherwise the most recent qualifying date
    cands.sort(reverse=True)
    return cands[0], "only past date between MRZ date-of-birth and expiry"


_LABEL_HINTS = ("place of birth", "lieu de naissance", "lugar de nacimiento",
                "birth place", "place of birh", "pace of birth")


def _label_index(lines):
    """Fuzzy-locate the place-of-birth label, tolerating OCR damage."""
    for i, line in enumerate(lines):
        low = re.sub(r"[^a-z ]", "", line.lower())
        for hint in _LABEL_HINTS:
            if hint in low:
                return i
        # tolerate dropped characters: 'place of birh', 'pace', 'plce'
        compact = low.replace(" ", "")
        if ("birh" in compact or "birth" in compact or "naissance" in compact) and len(compact) <= 30:
            return i
        if compact.startswith("pace") or compact.startswith("plce"):
            return i
    return -1


_NOT_A_PLACE = {"male", "female", "sex", "date", "type", "code", "passport",
                "authority", "signature", "holder", "copy", "national", "nationality",
                "surname", "given", "names", "expiry", "issue"}


def _plausible_place(v):
    s = str(v or "").strip(" :;/.-,")
    if len(s) < 4 or len(s) > 40:
        return None
    if not re.match(r"^[A-Za-z][A-Za-z\s'.,\-]+$", s):
        return None
    low = s.lower()
    if any(w in low for w in _NOT_A_PLACE):
        return None
    if not re.search(r"[aeiou]", low):
        return None
    if _repair_month(s[:3]) and len(s) <= 4:
        return None
    return s.strip()


def find_place_of_birth_v2(lines, exclude=()):
    """Value-first place-of-birth extraction.

    Strategy 1: fuzzy-match the label, then take the best plausible value from
                the following few lines (skipping short OCR noise like 'MAS').
    Strategy 2: no usable label - accept a line that exactly names a known
                country, provided it is not the document's own issuing country.
    Returns (value, basis) or (None, "").
    """
    excl = set(str(e).strip().lower() for e in exclude if e)

    idx = _label_index(lines)
    if idx >= 0:
        for j in range(idx, min(idx + 4, len(lines))):
            candidate = lines[j]
            if j == idx:
                # value may sit after the label on the same line
                parts = re.split(r"(?i)birh|birth|naissance|nacimiento|pace", candidate)
                candidate = parts[-1] if len(parts) > 1 else ""
            v = _plausible_place(candidate)
            if v and v.lower() not in excl:
                return v, "found next to the place-of-birth label"

    known = {name.upper(): name for name in _COUNTRY_NAMES}
    for line in lines:
        s = str(line).strip(" :;/.-,").upper()
        if s in known and s.lower() not in excl:
            return known[s], "matched a country name on the page (no readable label)"
    return None, ""
