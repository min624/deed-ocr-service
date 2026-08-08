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
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Railway injects PORT at runtime; default it for local `docker run`.
ENV PORT=8000
EXPOSE 8000

# Shell form so $PORT expands; Railway sets PORT, otherwise falls back to 8000.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
