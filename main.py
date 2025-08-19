import os
import io
import time
import base64
import requests
from flask import Flask, request, Response
import json
from google.cloud import storage
from openai import OpenAI
import re
from PIL import Image
from datetime import timedelta

# --- Config ---
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=api_key)
bucket_name = os.environ.get("OUTPUT_BUCKET", "memory-books-output")
DEFAULT_PROMPT = (
    "Convert this photo into a professional adult coloring book page with clean continuous "
    "black outlines only on a pure white background"
)

# Limits
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20MB hard cap
REQUEST_TIMEOUT = 30  # seconds for HTTP GETs

app = Flask(__name__)

# Initialize storage client (best-effort)
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    print(f"[init] Connected to GCS bucket: {bucket_name}")
except Exception as e:
    print(f"[init] GCS not available: {e}")
    storage_client = None
    bucket = None

# --- Helpers ---
def safe_str(obj):
    """Convert any object to a safe ASCII string, removing problematic Unicode."""
    try:
        s = str(obj)
        # strip separators/zero-width nasties
        s = (
            s.replace("\u2028", " ")
             .replace("\u2029", " ")
             .replace("\u00a0", " ")
             .replace("\u200b", "")
             .replace("\u200c", "")
             .replace("\u200d", "")
        )
        # enforce ASCII
        return "".join(ch if ord(ch) < 128 else "?" for ch in s)
    except Exception:
        return "Error converting to string"

def make_safe_dict(obj):
    """Recursively cast arbitraries into ASCII-safe structures for JSON."""
    if isinstance(obj, dict):
        return {safe_str(k): make_safe_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_safe_dict(item) for item in obj]
    elif isinstance(obj, str):
        return safe_str(obj)
    else:
        return safe_str(obj)

def safe_json_response(data, status_code=200):
    safe_data = make_safe_dict(data)
    json_str = json.dumps(safe_data, ensure_ascii=True, separators=(",", ":"))
    return Response(json_str, status=status_code, mimetype="application/json")

def sanitize_text(s: str) -> str:
    return safe_str(s).strip()

def extract_direct_image_url(url: str) -> str:
    """Pull direct image URL from UploadKit HTML pages if needed."""
    if url.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
        return url

    if "uploadkit" in url or "download.html" in url:
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "ColoringBookProcessor/1.0"})
            resp.raise_for_status()
            html = resp.text
            patterns = [
                r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
                r'<img[^>]+src=["\']([^"\']+)["\']',
                r'href=["\']([^"\']+\.(?:jpg|jpeg|png|gif|webp)[^"\']*)["\']',
            ]
            for p in patterns:
                m = re.search(p, html, re.IGNORECASE)
                if m:
                    img_url = m.group(1)
                    if img_url.startswith("http"):
                        return img_url
                    if img_url.startswith("//"):
                        return "https:" + img_url
                    if img_url.startswith("/"):
                        base = "/".join(url.split("/")[:3])
                        return base + img_url
        except Exception as e:
            print(f"[extract] failed to mine HTML: {safe_str(e)}")
    return url

def download_image(url: str) -> bytes:
    direct = extract_direct_image_url(url)
    print(f"[fetch] {direct[:120]}")

    headers = {"User-Agent": "ColoringBookProcessor/1.0"}
    with requests.get(direct, headers=headers, timeout=REQUEST_TIMEOUT, stream=True) as r:
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "text/html" in ct.lower():
            raise ValueError("Got HTML instead of image")
        # enforce size cap
        total = 0
        chunks = []
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    raise ValueError("Image exceeds max size (20MB)")
                chunks.append(chunk)
        data = b"".join(chunks)
    return data

def call_openai_edit(image_bytes: bytes, prompt: str) -> bytes:
    """Send image to OpenAI image edit API and return edited PNG bytes."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        clean_prompt = sanitize_text(prompt) if (prompt and len(prompt) > 10) else sanitize_text(DEFAULT_PROMPT)
        if not clean_prompt or len(clean_prompt) < 10:
            clean_prompt = "Convert to line art coloring book page"

        print(f"[openai] prompt: {clean_prompt[:80]}")

        # Use URL response to avoid base64 headaches
        resp = client.images.edit(
            model="gpt-image-1",
            image=buf,
            prompt=clean_prompt,
            size="1024x1024",
            response_format="url",
        )
        img_url = resp.data[0].url
        img_resp = requests.get(img_url, timeout=REQUEST_TIMEOUT)
        img_resp.raise_for_status()
        return img_resp.content

    except Exception as e:
        print(f"[openai] primary failed: {safe_str(e)}; retrying with minimal prompt")
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            resp = client.images.edit(
                model="gpt-image-1",
                image=buf,
                prompt="line art coloring page",
                size="1024x1024",
                response_format="url",
            )
            img_url = resp.data[0].url
            img_resp = requests.get(img_url, timeout=REQUEST_TIMEOUT)
            img_resp.raise_for_status()
            return img_resp.content
        except Exception as e2:
            raise Exception(f"Image processing failed: {safe_str(e2)}")

def upload_to_gcs(order_id: str, idx: int, img_bytes: bytes) -> str:
    """Upload PNG to GCS and return signed URL."""
    if not bucket:
        # fallback: return data URL preview
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    blob_name = f"{order_id}/{int(time.time())}_{idx}.png"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(img_bytes, content_type="image/png")

    return blob.generate_signed_url(expiration=timedelta(days=7))  # v4 signed URL

# --- Routes ---
@app.route("/process", methods=["POST"])
def process():
    try:
        payload = request.get_json(force=True) or {}
        order_id = sanitize_text(payload.get("order_id", f"order_{int(time.time())}"))
        image_urls = payload.get("image_urls") or payload.get("urls") or []

        if isinstance(image_urls, str):
            image_urls = [u.strip() for u in image_urls.split(",") if u.strip()]

        raw_prompt = payload.get("prompt", DEFAULT_PROMPT)
        prompt = sanitize_text(raw_prompt) if raw_prompt else DEFAULT_PROMPT

        print(f"[process] {order_id} - {len(image_urls)} image(s)")

        results = []
        for idx, url in enumerate(image_urls):
            try:
                raw = download_image(url)
                print(f"[process] downloaded {len(raw)} bytes")
                edited = call_openai_edit(raw, prompt)
                print(f"[process] openai done -> {len(edited)} bytes")
                signed = upload_to_gcs(order_id, idx, edited)
                results.append({
                    "status": "ok",
                    "index": idx,
                    "source_url": url,
                    "result_url": signed if bucket else None,
                    "result_base64": None if bucket else signed,  # data URL in no-GCS mode
                    "storage_type": "gcs" if bucket else "data-url"
                })
            except Exception as e:
                err = sanitize_text(str(e))
                print(f"[process] error image {idx}: {err}")
                results.append({"status": "error", "index": idx, "source_url": url, "error": err})

        return safe_json_response({
            "success": True,
            "count": len(results),
            "order_id": order_id,
            "prompt_used": (prompt.encode("ascii", "ignore").decode("ascii"))[:100],
            "results": results
        })

    except Exception as e:
        err = safe_str(e)
        print(f"[process] request failed: {err}")
        return safe_json_response({"success": False, "error": err}, 500)

@app.route("/test", methods=["GET", "POST"])
def test():
    test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"
    try:
        raw = download_image(test_url)
        edited = call_openai_edit(raw, "Convert to coloring book page")
        b64 = base64.b64encode(edited).decode("utf-8")
        return safe_json_response({
            "success": True,
            "message": "Test successful!",
            "original_url": test_url,
            "result_base64_preview": b64[:100] + "...",
            "result_size": len(edited)
        })
    except Exception as e:
        return safe_json_response({"success": False, "error": safe_str(e)}, 500)

@app.route("/health", methods=["GET"])
def health():
    return safe_json_response({
        "status": "healthy",
        "service": "coloring-book-processor",
        "gcs_available": bucket is not None,
        "openai_configured": api_key is not None,
        "bucket_name": bucket_name
    })

@app.route("/", methods=["GET"])
def index():
    return safe_json_response({
        "service": "Coloring Book Processor",
        "endpoints": {
            "/process": "POST - Process images to line art",
            "/test": "GET/POST - Test with sample image",
            "/health": "GET - Health check",
            "/": "GET - This help"
        },
        "example_request": {
            "url": "/process",
            "method": "POST",
            "body": {
                "order_id": "order_123",
                "urls": ["https://example.com/image1.jpg"],
                "prompt": "Convert to coloring book page"
            }
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
