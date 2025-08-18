import os
import io
import time
import base64
import requests
from flask import Flask, request, jsonify
from google.cloud import storage
from openai import OpenAI

# --- Config ---
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
bucket_name = os.environ.get("GCS_BUCKET", "coloring-book-results")

DEFAULT_PROMPT = (
    "Convert this photo into clean black-and-white line art suitable for a coloring book. "
    "Keep facial features recognizable, simplify shading to outlines, remove background clutter, "
    "and produce crisp, printable lines on white."
)

app = Flask(__name__)
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)


# --- Helpers ---
def _sanitize_text(s: str) -> str:
    """Strip invisible Unicode separators and ensure clean utf-8."""
    if not isinstance(s, str):
        s = str(s)
    s = s.replace(u"\u2028", " ").replace(u"\u2029", " ")
    return s.encode("utf-8", "ignore").decode("utf-8")


def _download_image(url: str) -> bytes:
    """Download image bytes from a URL."""
    headers = {"User-Agent": "ColoringBookProcessor/1.0"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.content


def _call_openai_edit(png_bytes: bytes, prompt: str) -> bytes:
    """Send PNG to OpenAI image edit API and return edited PNG bytes."""
    tmp_path = f"/tmp/in_{time.time_ns()}.png"
    with open(tmp_path, "wb") as f:
        f.write(png_bytes)

    clean_prompt = _sanitize_text(prompt or DEFAULT_PROMPT)

    with open(tmp_path, "rb") as f:
        resp = client.images.edit(
            model="gpt-image-1",
            image=f,
            prompt=clean_prompt,
            size="1024x1024",
        )

    b64 = resp.data[0].b64_json
    return base64.b64decode(b64)


def _upload_to_gcs(order_id: str, idx: int, img_bytes: bytes) -> str:
    """Upload PNG to GCS and return signed URL."""
    blob_name = f"{order_id}/{int(time.time())}_{idx}.png"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(img_bytes, content_type="image/png")
    return blob.generate_signed_url(expiration=3600)


# --- Routes ---
@app.route("/process", methods=["POST"])
def process():
    payload = request.get_json(force=True)
    order_id = payload.get("order_id", f"order_{int(time.time())}")
    image_urls = payload.get("image_urls", [])
    prompt = payload.get("prompt", DEFAULT_PROMPT)

    results = []
    for idx, url in enumerate(image_urls):
        try:
            raw = _download_image(url)
            edited = _call_openai_edit(raw, prompt)
            signed = _upload_to_gcs(order_id, idx, edited)
            results.append({
                "status": "ok",
                "index": idx,
                "source_url": url,
                "signed_url": signed,
            })
        except Exception as e:
            results.append({
                "status": "processing_error",
                "index": idx,
                "source_url": url,
                "detail": str(e),
            })

    return jsonify({
        "count": len(results),
        "order_id": order_id,
        "prompt_used": _sanitize_text(prompt),
        "results": results,
    })


@app.route("/")
def health():
    return "OK", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
