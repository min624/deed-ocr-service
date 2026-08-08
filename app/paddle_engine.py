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
