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

import asyncio
import base64
import binascii
import ipaddress
import logging
import os
import socket
from typing import Optional
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app.ocr import OcrError, parse_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deed-ocr-service")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
URL_FETCH_TIMEOUT_SECONDS = 10
MAX_CONCURRENT_OCR = 2

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

_ocr_semaphore = asyncio.Semaphore(MAX_CONCURRENT_OCR)


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
    extract_viz_fields: bool = Field(
        default=False,
        description="Also OCR the printed page for Date of Issue and Place of Birth (slower).",
    )
    debug_text: bool = Field(
        default=False,
        description="Return the raw recognised text lines, for diagnosing why a field was not found.",
    )


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": message},
    )


def _decode_base64_image(data: str) -> bytes:
    if "," in data and data.strip().lower().startswith("data:"):
        data = data.split(",", 1)[1]
    padding = data.count("=") if data.endswith("=") else 0
    estimated_bytes = (len(data) * 3) // 4 - padding
    if estimated_bytes > MAX_UPLOAD_BYTES:
        raise OcrError(
            f"Base64 image exceeds max allowed size "
            f"({MAX_UPLOAD_BYTES // (1024 * 1024)}MB)."
        )
    try:
        image_bytes = base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise OcrError(f"Invalid base64 image data: {exc}") from exc
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise OcrError(
            f"Base64 image exceeds max allowed size "
            f"({MAX_UPLOAD_BYTES // (1024 * 1024)}MB)."
        )
    return image_bytes


def _validate_url(url: str) -> None:
    """Reject URLs that resolve to private/reserved IPs (SSRF prevention)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise OcrError("image_url must be an http:// or https:// URL.")
    hostname = parsed.hostname
    if not hostname:
        raise OcrError("image_url has no hostname.")
    try:
        addr_infos = socket.getaddrinfo(
            hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
        )
    except socket.gaierror as exc:
        raise OcrError(f"Could not resolve hostname: {hostname}") from exc
    for _family, _type, _proto, _canonname, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise OcrError(
                "image_url must not point to a private or reserved network address."
            )


def _fetch_image_url(url: str) -> bytes:
    _validate_url(url)
    try:
        resp = requests.get(
            url,
            timeout=URL_FETCH_TIMEOUT_SECONDS,
            stream=True,
            allow_redirects=False,
        )
        if resp.is_redirect:
            raise OcrError("image_url returned a redirect; redirects are not allowed.")
        resp.raise_for_status()
    except OcrError:
        raise
    except requests.RequestException as exc:
        raise OcrError(f"Could not fetch image_url: {exc}") from exc

    chunks: list[bytes] = []
    fetched = 0
    for chunk in resp.iter_content(chunk_size=64 * 1024):
        fetched += len(chunk)
        if fetched > MAX_UPLOAD_BYTES:
            resp.close()
            raise OcrError(
                f"Fetched image exceeds max allowed size "
                f"({MAX_UPLOAD_BYTES // (1024 * 1024)}MB)."
            )
        chunks.append(chunk)
    return b"".join(chunks)


BUILD_COMMIT = (
    os.environ.get("RAILWAY_GIT_COMMIT_SHA")
    or os.environ.get("GIT_COMMIT_SHA")
    or "unknown"
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "deed-ocr-service",
        "commit": BUILD_COMMIT[:12],
    }


@app.post("/parse")
async def parse(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    extract_viz_fields: Optional[bool] = Form(default=None),
    debug_text: Optional[bool] = Form(default=None),
):
    """Extract MRZ data from a passport/ID image.

    Accepts ONE of:
      - multipart/form-data upload with field name "file"
        (optionally with extract_viz_fields / debug_text form fields)
      - JSON body: {"image_base64": "..."}
      - JSON body: {"image_url": "https://..."}
    """
    try:
        image_bytes: Optional[bytes] = None
        viz = extract_viz_fields or False
        dbg = debug_text or False

        if file is not None:
            chunks: list[bytes] = []
            received = 0
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                received += len(chunk)
                if received > MAX_UPLOAD_BYTES:
                    return _error_response(
                        413,
                        f"Uploaded file exceeds max allowed size "
                        f"({MAX_UPLOAD_BYTES // (1024 * 1024)}MB).",
                    )
                chunks.append(chunk)
            image_bytes = b"".join(chunks)
        else:
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

            if not isinstance(payload, dict):
                return _error_response(400, "JSON body must be an object, not an array or scalar.")

            try:
                body = ParseBody(**payload)
            except (ValidationError, TypeError) as exc:
                return _error_response(422, f"Invalid request body: {exc}")

            viz = body.extract_viz_fields
            dbg = body.debug_text
            if body.image_base64:
                image_bytes = _decode_base64_image(body.image_base64)
            elif body.image_url:
                image_bytes = _fetch_image_url(body.image_url)
            else:
                return _error_response(
                    400,
                    "JSON body must include either 'image_base64' or 'image_url'.",
                )

        async with _ocr_semaphore:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, parse_document, image_bytes, viz, dbg
            )
        return result

    except OcrError as exc:
        logger.info("OCR failure: %s", exc.message)
        return _error_response(exc.status_code, exc.message)
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error while parsing document")
        return _error_response(500, "Internal server error.")
