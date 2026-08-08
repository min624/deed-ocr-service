# Deed OCR Service

Lightweight FastAPI microservice that OCRs a passport/ID photo and returns the
parsed MRZ (Machine Readable Zone) as structured JSON. Runs entirely offline
inside the container — Tesseract OCR + OpenCV + the `mrz` Python library do
all the work locally; no image or extracted PII is ever sent to a third party.

## Endpoints

### `GET /health`
Liveness check. Returns `{"status": "ok", "service": "deed-ocr-service"}`.

### `POST /parse`
Accepts **one** of the following:

- **Multipart upload** — form field named `file` containing the image.
- **Base64 JSON** — `{"image_base64": "<base64 string, data: URI prefix OK>"}`
- **Image URL** — `{"image_url": "https://..."}` (fetched server-side, still
  processed locally — nothing is sent onward except the outbound GET to
  retrieve the image itself).

Example (multipart):
```bash
curl -X POST http://localhost:8000/parse -F "file=@passport.jpg"
```

Example (base64):
```bash
curl -X POST http://localhost:8000/parse \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "'"$(base64 -w0 passport.jpg)"'"}'
```

Success response:
```json
{
  "success": true,
  "mrz_type": "TD3",
  "fields": {
    "surname": "SMITH",
    "given_names": "JOHN",
    "passport_number": "AB1234567",
    "nationality": "GBR",
    "date_of_birth": "1990-01-15",
    "gender": "M",
    "expiry_date": "2030-05-20",
    "issuing_country": "GBR"
  },
  "raw_mrz": "P<GBRSMITH<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<\nAB12345671GBR9001158M3005209<<<<<<<<<<<<<<04"
}
```

Error response (400/422/500):
```json
{
  "success": false,
  "error": "No valid MRZ block could be found in the image. ..."
}
```

Supports passport (TD3), and ID-card formats TD1/TD2.

## Local run

```bash
docker build -t deed-ocr-service .
docker run -p 8000:8000 deed-ocr-service
curl http://localhost:8000/health
```

## Deploying to Railway

1. Push this directory to a git repo (or a subfolder of one).
2. In Railway: **New Project → Deploy from GitHub repo**, pick the repo/folder
   containing this `Dockerfile`. Railway auto-detects the Dockerfile build.
3. Railway sets the `PORT` env var automatically — the container's `CMD`
   already reads `$PORT`, no config needed.
4. No other environment variables are required. There are no external API
   keys because everything runs locally in the container.
5. Once deployed, Railway gives you a public URL — use
   `https://<your-app>.up.railway.app/parse` as the n8n HTTP Request node
   target. CORS is already open (`*`) so browser-based callers work too; if
   you want to lock it down to just your n8n instance, restrict
   `allow_origins` in `app/main.py`.

## Notes / limitations

- Image minimum size is 200x200px; smaller images are rejected with a 422.
- Max request/body size is capped at 15MB (both uploads and fetched URLs).
- `image_url` fetches are local-server-side HTTP GETs with a 10s timeout —
  if you're processing untrusted input, consider restricting which hosts are
  allowed to be fetched (SSRF hardening) before exposing this publicly.
- OCR accuracy depends heavily on image quality: the MRZ block (bottom of
  the document) should be flat, in focus, well-lit, and not obstructed.
- No PII is logged; only high-level failure reasons are logged on error.
