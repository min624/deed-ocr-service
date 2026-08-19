"""
MRZ (Machine Readable Zone) extraction and parsing for passports / ID cards.

Pipeline:
    bytes -> PIL/OpenCV image -> preprocess -> locate MRZ region
          -> Tesseract OCR -> candidate line cleanup
          -> mrz library (TD1 / TD2 / TD3 checkers) -> structured fields

Everything below runs locally in-process. No network calls are made while
handling image bytes, so passport/ID data never leaves the container.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps, UnidentifiedImageError

from mrz.checker.td1 import TD1CodeChecker
from mrz.checker.td2 import TD2CodeChecker
from mrz.checker.td3 import TD3CodeChecker

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class OcrError(Exception):
    """Base class for all handled OCR/MRZ failures. Carries an HTTP status."""

    status_code = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidImageError(OcrError):
    status_code = 400


class ImageTooSmallError(OcrError):
    status_code = 422


class MrzNotFoundError(OcrError):
    status_code = 422


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

MIN_DIMENSION_PX = 200  # reject images smaller than this on either axis
MAX_UPSCALE_WIDTH = 1600  # cap upscaling so huge/tiny images normalize similarly

MRZ_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
TESSERACT_CONFIG = (
    "--psm 6 --oem 3 "
    f"-c tessedit_char_whitelist={MRZ_CHARSET}"
)

# (line_count, line_length, mrz_type, checker_class)
_MRZ_SHAPES = [
    (2, 44, "TD3", TD3CodeChecker),  # passports
    (3, 30, "TD1", TD1CodeChecker),  # ID cards
    (2, 36, "TD2", TD2CodeChecker),  # ID cards / visas
]

_LINE_TOKEN_RE = re.compile(r"[A-Z0-9<]{20,}")


@dataclass
class ParsedMrz:
    mrz_type: str
    fields: dict
    raw_mrz: str


# --------------------------------------------------------------------------
# Image loading / preprocessing
# --------------------------------------------------------------------------


def _pdf_to_image(pdf_bytes: bytes) -> Image.Image:
    """Render the first page of a PDF to a PIL image at ~200 DPI."""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise InvalidImageError("PDF received but PDF support unavailable.") from exc
    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
        page = pdf[0]
        bitmap = page.render(scale=200 / 72)
        return bitmap.to_pil()
    except Exception as exc:  # noqa: BLE001
        raise InvalidImageError(f"Could not render PDF: {exc}") from exc


def load_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into a BGR OpenCV image, validating it's a real image.

    Accepts standard image formats and PDFs (first page rendered).
    Honors EXIF orientation so phone photos come in upright.
    """
    if not image_bytes:
        raise InvalidImageError("Empty image payload.")
    if image_bytes[:5] == b"%PDF-":
        pil_img = _pdf_to_image(image_bytes)
    else:
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.load()
            pil_img = ImageOps.exif_transpose(pil_img)
        except UnidentifiedImageError as exc:
            raise InvalidImageError("File is not a readable image.") from exc
        except Exception as exc:  # noqa: BLE001
            raise InvalidImageError(f"Could not decode image: {exc}") from exc

    width, height = pil_img.size
    if width < MIN_DIMENSION_PX or height < MIN_DIMENSION_PX:
        raise ImageTooSmallError(
            f"Image too small ({width}x{height}px). "
            f"Minimum {MIN_DIMENSION_PX}x{MIN_DIMENSION_PX}px required."
        )

    pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _upscale_if_needed(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if w < MAX_UPSCALE_WIDTH:
        scale = MAX_UPSCALE_WIDTH / float(w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return img


def _preprocess_variants(img: np.ndarray):
    """Yield a few preprocessed versions of the image to try OCR against.

    Real-world passport photos vary a lot (lighting, skew, background), so
    instead of a single "correct" pipeline we try a small set of cheap
    variants and keep whichever one yields a valid MRZ.
    """
    img = _upscale_if_needed(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Plain grayscale
    yield gray

    # 2. Otsu threshold (good for clean scans / high-contrast crops)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield otsu

    # 3. Adaptive threshold (good for uneven lighting / phone photos)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    yield adaptive

    # 4. Bottom third only, upscaled further, Otsu-thresholded.
    # The MRZ is always at the bottom of a passport/ID; isolating it removes
    # noise from photos/backgrounds elsewhere in the frame.
    h = gray.shape[0]
    bottom = gray[int(h * 0.55):, :]
    bottom = cv2.resize(bottom, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    _, bottom_otsu = cv2.threshold(bottom, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield bottom_otsu

    # 5. CLAHE contrast enhancement (glare / low-contrast phone photos)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    yield enhanced
    _, enh_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield enh_otsu

    # 6. Rotations - passports are often photographed sideways
    for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
        yield cv2.rotate(gray, rot)


# --------------------------------------------------------------------------
# OCR + line cleanup
# --------------------------------------------------------------------------


def _ocr_candidate_lines(img: np.ndarray) -> list[str]:
    text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
    lines = []
    for raw_line in text.splitlines():
        cleaned = raw_line.strip().upper().replace(" ", "")
        if not cleaned:
            continue
        match = _LINE_TOKEN_RE.search(cleaned)
        if match:
            lines.append(match.group(0))
    return lines


def _normalize_length(line: str, target_len: int) -> Optional[str]:
    """Pad/trim a slightly-mis-OCR'd line to the expected MRZ line length.

    Tesseract commonly drops a trailing filler '<' or two. We only correct
    lines within 2 characters of the target to avoid fabricating data.
    """
    diff = target_len - len(line)
    if diff == 0:
        return line
    if 0 < diff <= 2:
        return line + ("<" * diff)
    if -2 <= diff < 0:
        return line[:target_len]
    return None


def _try_parse_shape(lines: list[str]) -> Optional[ParsedMrz]:
    for count, length, mrz_type, checker_cls in _MRZ_SHAPES:
        if len(lines) < count:
            continue
        # try every contiguous window of `count` lines, in case OCR picked
        # up stray tokens before/after the actual MRZ block
        for start in range(0, len(lines) - count + 1):
            window = lines[start:start + count]
            normalized = [_normalize_length(l, length) for l in window]
            if any(n is None for n in normalized):
                continue
            mrz_text = "\n".join(normalized)
            try:
                checker = checker_cls(mrz_text)
            except Exception:  # noqa: BLE001
                continue
            if not (checker.document_number_hash
                    and checker.birth_date_hash
                    and checker.expiry_date_hash):
                continue
            fields = checker.fields()
            return ParsedMrz(mrz_type=mrz_type, fields=fields, raw_mrz=mrz_text)
    return None


# TD3 line 2 layout: doc#(0-8) chk(9) nat(10-12) dob(13-18) chk(19) sex(20)
# expiry(21-26) chk(27). Used for relaxed extraction when checksums fail.
_TD3_LINE2_RE = re.compile(
    r"^(?P<doc>[A-Z0-9<]{9})(?P<doc_chk>[0-9<])(?P<nat>[A-Z0-9<]{3})"
    r"(?P<dob>\d{6})(?P<dob_chk>[0-9<])(?P<sex>[MF<X])"
    r"(?P<exp>\d{6})(?P<exp_chk>[0-9<])"
)


def _relaxed_line2_extract(lines: list[str]) -> Optional[dict]:
    """Positional TD3 line-2 parse for when strict checksum validation fails.

    Only trusts fields with strong internal structure: sex must be M/F flanked
    by two valid 6-digit dates. Field checksums are verified individually so a
    garbled read can still yield a usable DOB or gender.
    """
    def check_digit_ok(value: str, chk: str) -> bool:
        if not chk.isdigit():
            return False
        weights = [7, 3, 1]
        total = 0
        for i, ch in enumerate(value):
            if ch.isdigit():
                v = int(ch)
            elif ch == "<":
                v = 0
            else:
                v = ord(ch) - 55
            total += v * weights[i % 3]
        return total % 10 == int(chk)

    for line in lines:
        m = _TD3_LINE2_RE.match(line)
        if not m:
            continue
        out = {}
        if m.group("sex") in ("M", "F"):
            # Trust sex only when at least one adjacent date passes its checksum
            dob_ok = check_digit_ok(m.group("dob"), m.group("dob_chk"))
            exp_ok = check_digit_ok(m.group("exp"), m.group("exp_chk"))
            if dob_ok or exp_ok:
                out["sex"] = m.group("sex")
                if dob_ok:
                    out["birth_date"] = m.group("dob")
                if exp_ok:
                    out["expiry_date"] = m.group("exp")
                if check_digit_ok(m.group("doc"), m.group("doc_chk")):
                    out["document_number"] = m.group("doc").replace("<", "")
                nat = m.group("nat").replace("<", "")
                if nat.isalpha() and len(nat) == 3:
                    out["nationality"] = nat
        if out:
            return out
    return None


def extract_mrz(img: np.ndarray) -> ParsedMrz:
    all_lines: list[list[str]] = []
    for variant in _preprocess_variants(img):
        lines = _ocr_candidate_lines(variant)
        if lines:
            all_lines.append(lines)
        parsed = _try_parse_shape(lines)
        if parsed is not None:
            return parsed

    # Strict parse failed everywhere - try relaxed positional extraction
    # across every variant's lines, best (most fields) first.
    best = None
    for lines in all_lines:
        relaxed = _relaxed_line2_extract(lines)
        if relaxed and (best is None or len(relaxed) > len(best)):
            best = relaxed
    if best:
        return ParsedMrz(mrz_type="TD3-RELAXED", fields=best, raw_mrz="")

    raise MrzNotFoundError(
        "No valid MRZ block could be found in the image. "
        "Make sure the machine-readable zone (bottom lines of the "
        "passport/ID) is fully visible, in focus, and well lit."
    )


# --------------------------------------------------------------------------
# Field normalization
# --------------------------------------------------------------------------


def _yymmdd_to_iso(value: str, *, is_expiry: bool) -> Optional[str]:
    if not value or len(value) != 6 or not value.isdigit():
        return None
    yy, mm, dd = int(value[0:2]), int(value[2:4]), int(value[4:6])
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None

    current_yy = date.today().year % 100
    current_century = (date.today().year // 100) * 100

    if is_expiry:
        # Expiry dates are generally within ~15 years of "now"; if the
        # 2-digit year looks like it's far in the past, it's next century.
        century = current_century if yy >= current_yy - 1 else current_century + 100
    else:
        # Birth dates: assume no one in the dataset was born in the future.
        century = current_century if yy <= current_yy else current_century - 100

    try:
        return date(century + yy, mm, dd).isoformat()
    except ValueError:
        return None


def _clean_name(value: str) -> str:
    return re.sub(r"<+", " ", value or "").strip()


def build_response_fields(parsed: ParsedMrz) -> dict:
    f = parsed.fields

    def get(name):
        if isinstance(f, dict):
            return f.get(name, "")
        return getattr(f, name, "")

    surname = _clean_name(get("surname"))
    given_names = _clean_name(get("name"))
    sex = (get("sex") or "").strip().upper()
    if sex not in ("M", "F"):
        sex = "X"

    return {
        "surname": surname,
        "given_names": given_names,
        "passport_number": (get("document_number") or "").replace("<", "").strip(),
        "nationality": (get("nationality") or "").strip(),
        "date_of_birth": _yymmdd_to_iso(get("birth_date"), is_expiry=False),
        "gender": sex,
        "expiry_date": _yymmdd_to_iso(get("expiry_date"), is_expiry=True),
        "issuing_country": (get("country") or "").strip(),
    }


# --------------------------------------------------------------------------
# Visual zone (VIZ) fallback - the printed "Sex: M/F" field
# --------------------------------------------------------------------------

# "Sex" label variants across passports (English/French/Spanish), followed by
# a short gap of separators, then a single M or F. The letter is Latin even on
# Arabic-script passports.
_VIZ_SEX_RE = re.compile(
    r"\b(?:sex|sexe|sexo)\b[^A-Za-z0-9]{0,8}([MF])\b", re.IGNORECASE
)


VIZ_MAX_WIDTH = 1400  # cap resolution so the sparse-text pass stays fast


def extract_viz_sex(img: np.ndarray) -> Optional[str]:
    """Full-page OCR looking for the printed Sex field. Returns 'M'/'F' or None.

    Kept deliberately cheap: downscaled image, two variants, no rotations
    (EXIF orientation is already applied at load), --psm 6 which is much
    faster than sparse-text mode on large pages.
    """
    h, w = img.shape[:2]
    if w > VIZ_MAX_WIDTH:
        scale = VIZ_MAX_WIDTH / float(w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

    votes = []
    for variant in (gray, clahe):
        try:
            text = pytesseract.image_to_string(variant, config="--psm 6 --oem 3")
        except Exception:  # noqa: BLE001
            continue
        m = _VIZ_SEX_RE.search(text)
        if m:
            votes.append(m.group(1).upper())
    if not votes:
        return None
    if votes.count("M") > votes.count("F"):
        return "M"
    if votes.count("F") > votes.count("M"):
        return "F"
    return None


# --------------------------------------------------------------------------
# Public entrypoint
# --------------------------------------------------------------------------


def _paddle_fallback(img: np.ndarray, cached_lines: Optional[list] = None) -> Optional[ParsedMrz]:
    """Try PaddleOCR: strict MRZ parse on its lines, then relaxed, then VIZ sex."""
    from app import paddle_engine

    lines = cached_lines if cached_lines is not None else paddle_engine.get_text_lines(img)
    if not lines:
        return None

    mrz_lines = paddle_engine.mrz_candidate_lines(lines)
    parsed = _try_parse_shape(mrz_lines)
    if parsed is not None:
        parsed.mrz_type = parsed.mrz_type + "-PADDLE"
        return parsed

    relaxed = _relaxed_line2_extract(mrz_lines)
    if relaxed:
        return ParsedMrz(mrz_type="TD3-RELAXED-PADDLE", fields=relaxed, raw_mrz="")

    sex = paddle_engine.find_sex(lines)
    if sex:
        return ParsedMrz(mrz_type="VIZ-PADDLE", fields={"sex": sex}, raw_mrz="")
    return None


def parse_document(image_bytes: bytes, extract_viz_fields: bool = False, debug_text: bool = False) -> dict:
    """Full pipeline: raw image bytes -> structured MRZ response dict.

    Tier 1: Tesseract MRZ (strict, then relaxed line-2).
    Tier 2: PaddleOCR (strict MRZ, relaxed, printed Sex field).
    Tier 3: Tesseract visual-zone Sex field.

    extract_viz_fields additionally runs PaddleOCR over the printed page to
    recover Date of Issue and Place of Birth (not present in any MRZ).
    Raises an OcrError subclass on any handled failure.
    """
    img = load_image(image_bytes)

    from app import paddle_engine
    _paddle_lines_cache = [None]

    def _get_paddle_lines():
        if _paddle_lines_cache[0] is None:
            _paddle_lines_cache[0] = paddle_engine.get_text_lines(img)
        return _paddle_lines_cache[0]

    parsed = None
    try:
        parsed = extract_mrz(img)
    except MrzNotFoundError:
        parsed = _paddle_fallback(img, cached_lines=_get_paddle_lines())
        if parsed is None:
            sex = extract_viz_sex(img)
            if sex:
                parsed = ParsedMrz(mrz_type="VIZ-ONLY", fields={"sex": sex}, raw_mrz="")
        if parsed is None and extract_viz_fields:
            lines = _get_paddle_lines()
            if lines and (paddle_engine.find_issue_date(lines) or paddle_engine.find_place_of_birth_v2(lines)[0]):
                parsed = ParsedMrz(mrz_type="VIZ-ONLY", fields={}, raw_mrz="")
        if parsed is None:
            raise

    fields = build_response_fields(parsed)
    if fields.get("gender") == "X":
        better = _paddle_fallback(img, cached_lines=_get_paddle_lines())
        if better is not None:
            better_fields = build_response_fields(better)
            if better_fields.get("gender") in ("M", "F"):
                fields["gender"] = better_fields["gender"]
                fields["gender_source"] = better.mrz_type
                for key, val in better_fields.items():
                    if val and not fields.get(key):
                        fields[key] = val
        if fields.get("gender") == "X":
            sex = extract_viz_sex(img)
            if sex:
                fields["gender"] = sex
                fields["gender_source"] = "viz"

    raw_lines = []
    if extract_viz_fields or debug_text:
        lines = _get_paddle_lines()
        raw_lines = lines
        if lines:
            issue = paddle_engine.find_issue_date(lines)
            basis = "read next to the date-of-issue label" if issue else ""
            if not issue:
                all_dates = paddle_engine.find_all_dates(lines)
                issue, basis = paddle_engine.classify_issue_date(
                    all_dates, fields.get("date_of_birth"), fields.get("expiry_date")
                )
            if issue:
                fields["issue_date"] = issue
                fields["issue_date_basis"] = basis

            # Strategy 2 of find_place_of_birth_v2 compares against full
            # country names, so the exclude list must contain both the 3-letter
            # codes from the MRZ and the resolved full names.
            _ISO3_RESOLVE = {"ARE":"United Arab Emirates","JOR":"Jordan","LKA":"Sri Lanka","SYR":"Syria","GBR":"United Kingdom","IND":"India","PAK":"Pakistan","EGY":"Egypt","SAU":"Saudi Arabia","USA":"United States","DEU":"Germany","KWT":"Kuwait","PSE":"Palestine","PHL":"Philippines","NGA":"Nigeria","CAN":"Canada","FRA":"France","AUS":"Australia","ZAF":"South Africa","LBN":"Lebanon","IRQ":"Iraq","YEM":"Yemen","MAR":"Morocco","TUN":"Tunisia","DZA":"Algeria","BHR":"Bahrain","OMN":"Oman","QAT":"Qatar","TUR":"Turkey","RUS":"Russia","CHN":"China","JPN":"Japan","KOR":"South Korea","BGD":"Bangladesh","NPL":"Nepal","AFG":"Afghanistan","IRN":"Iran","ITA":"Italy","ESP":"Spain","NLD":"Netherlands","BEL":"Belgium","CHE":"Switzerland","SWE":"Sweden","NOR":"Norway","POL":"Poland","UKR":"Ukraine","GRC":"Greece","PRT":"Portugal","IRL":"Ireland","NZL":"New Zealand","SGP":"Singapore","MYS":"Malaysia","IDN":"Indonesia","THA":"Thailand","VNM":"Vietnam","ETH":"Ethiopia","KEN":"Kenya","GHA":"Ghana","SDN":"Sudan","SOM":"Somalia","LBY":"Libya","CYP":"Cyprus","MDV":"Maldives","UGA":"Uganda","TZA":"Tanzania","MMR":"Myanmar","HKG":"Hong Kong","COL":"Colombia","ISR":"Israel","CMR":"Cameroon","AZE":"Azerbaijan","KAZ":"Kazakhstan","UZB":"Uzbekistan","CZE":"Czech Republic","AUT":"Austria","DNK":"Denmark","FIN":"Finland","HUN":"Hungary","ROU":"Romania","BGR":"Bulgaria","BRA":"Brazil","ARG":"Argentina","MEX":"Mexico","GEO":"Georgia","ARM":"Armenia","MLT":"Malta","EST":"Estonia","LVA":"Latvia","LTU":"Lithuania","MUS":"Mauritius","SVK":"Slovakia","SVN":"Slovenia","HRV":"Croatia","SRB":"Serbia","ZWE":"Zimbabwe"}
            nat_code = fields.get("nationality", "")
            iss_code = fields.get("issuing_country", "")
            exclude = [nat_code, iss_code,
                       _ISO3_RESOLVE.get(nat_code.upper(), ""),
                       _ISO3_RESOLVE.get(iss_code.upper(), "")]
            pob, pob_basis = paddle_engine.find_place_of_birth_v2(lines, exclude=exclude)
            if pob:
                fields["place_of_birth"] = pob
                fields["place_of_birth_basis"] = pob_basis

    out = {
        "success": True,
        "mrz_type": parsed.mrz_type,
        "fields": fields,
        "raw_mrz": parsed.raw_mrz,
    }
    if debug_text:
        out["raw_text_lines"] = raw_lines
        out["raw_text_count"] = len(raw_lines)
    return out
