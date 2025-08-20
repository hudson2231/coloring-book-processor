# main.py
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
from datetime import timedelta  # kept for compatibility with existing imports

# --- Force-disable proxies that break OpenAI client on Cloud Run ---
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    if os.environ.get(_k):
        print(f"[net] ignoring proxy env {_k}")
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

app = Flask(__name__)

# ---------- Utilities (Unicode hardening) ----------
def safe_str(obj):
    """Convert any object to a safe ASCII string, removing problematic Unicode."""
    try:
        s = str(obj)
        s = (
            s.replace("\u2028", " ")  # line sep
             .replace("\u2029", " ")  # paragraph sep
             .replace("\u00a0", " ")  # NBSP
             .replace("\u200b", "")   # zero-width space
             .replace("\u200c", "")   # ZWNJ
             .replace("\u200d", "")   # ZWJ
        )
        return "".join(ch if ord(ch) < 128 else "?" for ch in s)
    except Exception:
        return "Error converting to string"

def make_safe_dict(obj):
    if isinstance(obj, dict):
        return {safe_str(k): make_safe_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_safe_dict(item) for item in obj]
    elif isinstance(obj, str):
        return safe_str(obj)
    else:
        return safe_str(obj)

def safe_json_response(data, status_code=200):
    txt = json.dumps(make_safe_dict(data), ensure_ascii=True, separators=(",", ":"))
    return Response(txt, status=status_code, mimetype="application/json")

def sanitize_text(s: str) -> str:
    return safe_str(s).strip()

def sanitize_key(s: str) -> str:
    """Extra strict: remove invisible chars AND spaces from API keys."""
    s = sanitize_text(s)
    return s.replace(" ", "")

# ---------- OpenAI client (lazy init; sanitizes API key) ----------
client = None
def get_openai():
    global client
    if client is None:
        raw_key = os.environ.get("OPENAI_API_KEY", "")
        key = sanitize_key(raw_key)
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        os.environ["OPENAI_API_KEY"] = key  # overwrite raw env to avoid fallbacks using bad value
        client = OpenAI(api_key=key)
        print(f"[openai] client initialized")
    return client

# ---------- Config ----------
bucket_name = os.environ.get("OUTPUT_BUCKET", "memory-books-output")
DEFAULT_PROMPT = (
    "Convert this photo into a professional adult coloring book page with clean continuous "
    "black outlines only on a pure white background"
)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
REQUEST_TIMEOUT = 30

# ---------- GCS client ----------
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    print(f"[init] Connected to GCS bucket: {bucket_name}")
except Exception as e:
    print(f"[init] GCS not available: {safe_str(e)}")
    storage_client = None
    bucket = None

# ---------- Image helpers ----------
def extract_direct_image_url(url: str) -> str:
    url = sanitize_text(url)
    lower = url.lower()
    if lower.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
        return url
    if "uploadkit" in lower or "download.html" in lower:
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
                    if img_url.startswith("http"): return img_url
                    if img_url.startswith("//"):   return "https:" + img_url
                    if img_url.startswith("/"):    return "/".join(url.split("/")[:3]) + img_url
        except Exception as e:
            print(f"[extract] failed to mine HTML: {safe_str(e)}")
    return url

def download_image(url: str) -> bytes:
    direct = extract_direct_image_url(url)
    print(f"[fetch] {direct[:120]}")
    headers = {"User-Agent": "ColoringBookProcessor/1.0"}
    with requests.get(direct, headers=headers, timeout=REQUEST_TIMEOUT, stream=True) as r:
        r.raise_for_status()
        ct = (r.headers.get("content-type", "") or "").lower()
        if "text/html" in ct:
            raise ValueError("Got HTML instead of image")
        total, chunks = 0, []
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    raise ValueError("Image exceeds max size (20MB)")
                chunks.append(chunk)
    return b"".join(chunks)

def _decode_image_response(resp) -> bytes:
    """Handle both b64_json and url response shapes."""
    d = resp.data[0]
    b64 = getattr(d, "b64_json", None)
    if b64:
        return base64.b64decode(b64)
    url = getattr(d, "url", None)
    if url:
        ir = requests.get(url, timeout=REQUEST_TIMEOUT)
        ir.raise_for_status()
        return ir.content
    raise Exception("No image data in response")

def call_openai_edit(image_bytes: bytes, prompt: str) -> bytes:
    """Send image to OpenAI Images API and return edited PNG bytes."""
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
        clean_prompt = clean_prompt.encode("ascii", "ignore").decode("ascii")  # belt & suspenders

        print(f"[openai] prompt: {clean_prompt[:80]}")

        # Preferred modern call: edits (plural). Fallback to edit if SDK alias exists.
        try:
            resp = get_openai().images.edits(
                model="gpt-image-1",
                image=buf,
                prompt=clean_prompt,
                size="1024x1024",
            )
        except AttributeError:
            resp = get_openai().images.edit(
                model="gpt-image-1",
                image=buf,
                prompt=clean_prompt,
                size="1024x1024",
            )

        return _decode_image_response(resp)

    except Exception as e:
        print(f"[openai] primary failed: {safe_str(e)}; retrying with minimal prompt")
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            try:
                resp = get_openai().images.edits(
                    model="gpt-image-1",
                    image=buf,
                    prompt="line art coloring page",
                    size="1024x1024",
                )
            except AttributeError:
                resp = get_openai().images.edit(
                    model="gpt-image-1",
                    image=buf,
                    prompt="line art coloring page",
                    size="1024x1024",
                )

            return _decode_image_response(resp)
        except Exception as e2:
            raise Exception(f"Image processing failed: {safe_str(e2)}")

# ---------- Upload helper (SIGNED URLs; no public ACLs) ----------
def upload_to_gcs(order_id: str, idx: int, img_bytes: bytes) -> str:
    if not bucket:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    blob_name = f"{order_id}/{int(time.time())}_{idx}.png"
    blob = bucket.blob(blob_name)

    # Good CDN/browser caching for immutable artifacts
    blob.cache_control = "public, max-age=31536000, immutable"

    # Upload bytes
    blob.upload_from_string(img_bytes, content_type="image/png")
    # Persist cache headers (best effort)
    try:
        blob.patch()
    except Exception as e:
        print(f"[gcs] patch failed: {safe_str(e)}")

    # Return a time-limited signed URL (v4). No public ACLs.
    try:
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(days=7),
            method="GET",
        )
        return url
    except Exception as e:
        # Fallback: direct path (may not be accessible if bucket/object isn't public)
        print(f"[gcs] signed URL failed: {safe_str(e)}; falling back to public path")
        return f"https://storage.googleapis.com/{bucket_name}/{blob_name}"

# ---------- Routes ----------
@app.route("/process", methods=["POST"])
def process():
    try:
        payload = request.get_json(force=True) or {}
        order_id = sanitize_text(payload.get("order_id", f"order_{int(time.time())}"))

        image_urls = payload.get("image_urls") or payload.get("urls") or []
        if isinstance(image_urls, str):
            image_urls = [u.strip() for u in image_urls.split(",") if u.strip()]
        image_urls = [sanitize_text(u) for u in image_urls]

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
                final_url = upload_to_gcs(order_id, idx, edited)
                results.append({
                    "status": "ok",
                    "index": idx,
                    "source_url": url,
                    "result_url": final_url if bucket else None,
                    "result_base64": None if bucket else final_url,
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

@app.route("/test", methods=["GET", 'POST'])
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
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
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
