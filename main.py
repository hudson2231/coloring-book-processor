# main.py
import os
import io
import time
import base64
import hashlib
from datetime import timedelta
from typing import List, Tuple

import requests
from flask import Flask, request, jsonify

from google.cloud import storage
from openai import OpenAI

# ----------- Config -----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OUTPUT_BUCKET = "memory-books-output"  # <— hardcoded bucket name
DEFAULT_PROMPT = (
    "Convert this photo into clean black-and-white line art suitable for a "
    "coloring book. Keep facial features recognizable, simplify shading to outlines, "
    "remove background clutter, and produce crisp, printable lines on white."
)
TIMEOUT = (8, 30)  # connect, read
# ------------------------------

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
if not OUTPUT_BUCKET:
    raise RuntimeError("OUTPUT_BUCKET is not set")

client = OpenAI(api_key=OPENAI_API_KEY)
gcs = storage.Client()
bucket = gcs.bucket(OUTPUT_BUCKET)

app = Flask(__name__)

def _download(url: str) -> bytes:
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content

def _as_png_rgb(raw: bytes) -> bytes:
    """
    Ensure the image is PNG and in RGB mode (OpenAI edits require RGB/RGBA PNG).
    Uses Pillow; keep import here to avoid import cost if /health is called.
    """
    from PIL import Image  # Pillow
    with Image.open(io.BytesIO(raw)) as im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()

def _call_openai_edit(png_bytes: bytes, prompt: str) -> bytes:
    """
    Calls gpt-image-1 edit with a PNG (RGB/RGBA). The SDK wants a file handle,
    so write to a temp file then reopen.
    """
    tmp_path = f"/tmp/in_{time.time_ns()}.png"
    with open(tmp_path, "wb") as f:
        f.write(png_bytes)

    with open(tmp_path, "rb") as f:
        resp = client.images.edits(
            model="gpt-image-1",
            image=f,
            prompt=prompt or DEFAULT_PROMPT,
            size="1024x1024",
        )

    # result is base64 PNG
    b64 = resp.data[0].b64_json
    return base64.b64decode(b64)

def _gcs_save_and_sign(data: bytes, key: str) -> Tuple[str, str]:
    """
    Upload to GCS and return (gs_uri, signed_url).
    Signed URL valid for 7 days.
    """
    blob = bucket.blob(key)
    blob.upload_from_string(data, content_type="image/png")
    url = blob.generate_signed_url(
        version="v4",
        method="GET",
        expiration=timedelta(days=7),
        response_disposition=f'inline; filename="{os.path.basename(key)}"',
        content_type="image/png",
    )
    return f"gs://{OUTPUT_BUCKET}/{key}", url

def _safe_key(order_id: str, idx: int, src_url: str) -> str:
    # stable, readable key with short hash of source
    h = hashlib.sha1(src_url.encode("utf-8")).hexdigest()[:8]
    return f"orders/{order_id or 'noid'}/page_{idx+1}_{h}.png"

@app.get("/health")
def health():
    return jsonify({"ok": True}), 200

@app.post("/process")
def process():
    """
    Request JSON:
    {
      "order_id": "123",
      "image_urls": ["https://...png", "..."],
      "prompt": "optional override"
    }
    """
    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    image_urls: List[str] = payload.get("image_urls") or []
    order_id: str = str(payload.get("order_id") or "")
    prompt: str = payload.get("prompt") or DEFAULT_PROMPT

    if not image_urls:
        return jsonify({"error": "image_urls is required and must be a non-empty array"}), 400

    outputs = []
    for i, url in enumerate(image_urls):
        try:
            raw = _download(url)
            png_in = _as_png_rgb(raw)
            png_out = _call_openai_edit(png_in, prompt)
            key = _safe_key(order_id, i, url)
            gs_uri, signed = _gcs_save_and_sign(png_out, key)
            outputs.append(
                {
                    "index": i,
                    "source_url": url,
                    "gcs_uri": gs_uri,
                    "signed_url": signed,
                    "status": "ok",
                }
            )
        except requests.HTTPError as e:
            outputs.append(
                {"index": i, "source_url": url, "status": "download_error", "detail": str(e)}
            )
        except Exception as e:
            outputs.append(
                {"index": i, "source_url": url, "status": "processing_error", "detail": str(e)}
            )

    return jsonify(
        {
            "order_id": order_id,
            "count": len(outputs),
            "prompt_used": prompt,
            "results": outputs,
        }
    ), 200

if __name__ == "__main__":
    # For local testing only; Cloud Run uses gunicorn via Dockerfile CMD
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))

