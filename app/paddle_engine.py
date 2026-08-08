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
