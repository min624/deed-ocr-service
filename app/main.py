"""
Passport / ID MRZ OCR microservice.

POST /parse  - accepts an image via multipart upload, base64 JSON, or an
               image URL to fetch, extracts the MRZ, and returns structured
               identity fields.
GET  /health - liveness probe.

All image bytes are processed entirely in-process (Tesseract + OpenCV +
the `mrz` library run locally). No image data or extracted PII is ever
sent to a third-party service.
"""

from __future__ import annotations

import base64
import binascii
import logging
from typing import Optional

import requests
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.ocr import OcrError, parse_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deed-ocr-service")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
URL_FETCH_TIMEOUT_SECONDS = 10

app = FastAPI(
    title="Deed OCR Service",
    description="Local passport/ID MRZ OCR microservice (Tesseract + mrz).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ParseBody(BaseModel):
    """JSON body for /parse when not using multipart upload.

    Provide exactly one of `image_base64` or `image_url`.
    """

    image_base64: Optional[str] = Field(
        default=None, description="Base64-encoded image bytes (with or without data: URI prefix)."
    )
    image_url: Optional[str] = Field(
        default=None, description="HTTP(S) URL to fetch the image from."
    )


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": message},
    )


def _decode_base64_image(data: str) -> bytes:
    if "," in data and data.strip().lower().startswith("data:"):
        # strip a data: URI prefix, e.g. "data:image/jpeg;base64,...."
        data = data.split(",", 1)[1]
    try:
        return base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise OcrError(f"Invalid base64 image data: {exc}") from exc


def _fetch_image_url(url: str) -> bytes:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise OcrError("image_url must be an http:// or https:// URL.")
    try:
        resp = requests.get(url, timeout=URL_FETCH_TIMEOUT_SECONDS, stream=True)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise OcrError(f"Could not fetch image_url: {exc}") from exc

    content = resp.content
    if len(content) > MAX_UPLOAD_BYTES:
        raise OcrError(
            f"Fetched image exceeds max allowed size "
            f"({MAX_UPLOAD_BYTES // (1024 * 1024)}MB)."
        )
    return content


@app.get("/health")
async def health():
    return {"status": "ok", "service": "deed-ocr-service"}


@app.post("/parse")
async def parse(request: Request, file: Optional[UploadFile] = File(default=None)):
    """Extract MRZ data from a passport/ID image.

    Accepts ONE of:
      - multipart/form-data upload with field name "file"
      - JSON body: {"image_base64": "..."}
      - JSON body: {"image_url": "https://..."}
    """
    try:
        image_bytes: Optional[bytes] = None

        if file is not None:
            image_bytes = await file.read()
            if len(image_bytes) > MAX_UPLOAD_BYTES:
                return _error_response(
                    413,
                    f"Uploaded file exceeds max allowed size "
                    f"({MAX_UPLOAD_BYTES // (1024 * 1024)}MB).",
                )
        else:
            # No multipart file provided - try a JSON body instead.
            try:
                payload = await request.json()
            except Exception:
                payload = None

            if not payload:
                return _error_response(
                    400,
                    "No image provided. Send multipart form field 'file', "
                    "or a JSON body with 'image_base64' or 'image_url'.",
                )

            body = ParseBody(**payload)
            if body.image_base64:
                image_bytes = _decode_base64_image(body.image_base64)
            elif body.image_url:
                image_bytes = _fetch_image_url(body.image_url)
            else:
                return _error_response(
                    400,
                    "JSON body must include either 'image_base64' or 'image_url'.",
                )

        result = parse_document(image_bytes)
        return result

    except OcrError as exc:
        logger.info("OCR failure: %s", exc.message)
        return _error_response(exc.status_code, exc.message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error while parsing document")
        return _error_response(500, f"Internal error: {exc}")
