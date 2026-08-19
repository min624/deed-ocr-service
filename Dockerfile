# Deed OCR Service - passport/ID MRZ extraction
# Tesseract OCR + OpenCV + mrz library, all running locally in-container.

FROM python:3.11-slim

# Tesseract OCR engine (English is enough for MRZ, which is A-Z/0-9/< only).
# libgl1/libglib2.0-0 are the minimal runtime libs opencv-python-headless needs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download PaddleOCR models at build time so the first request is fast.
# || true kept: Railway's build network can be flaky, and a failed download
# is recoverable at runtime (PaddleOCR retries on first inference).
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en', show_log=False)" || true

COPY app ./app

RUN useradd --system --no-create-home ocr \
    && chown -R ocr:ocr /srv
USER ocr

# Railway injects PORT at runtime; default it for local `docker run`.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Shell form so $PORT expands; Railway sets PORT, otherwise falls back to 8000.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
