import os
import io
import time
import base64
import re
import json
import http
import urllib.parse as urlparse

import requests
from flask import Flask, request, Response
from google.cloud import storage
from openai import OpenAI
from PIL import Image

# ---------------- Version tag (for /health and logs) ----------------
VERSION = "cbp-v1.1-fetch-uploadkit"

# --- Force-disable proxies that break OpenAI client on Cloud Run ---
for _k in (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
    "OPENAI_PROXY", "OPENAI_HTTP_PROXY", "OPENAI_HTTPS_PROXY"
):
    if os.environ.get(_k):
        print(f"[net] ignoring proxy env {_k}")
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

app = Flask(__name__)

# ---------- Utilities (Unicode hardening) ----------
def safe_str(obj):
    try:
        s = str(obj)
        s = (
            s.replace("\u2028", " ")
             .replace("\u2029", " ")
             .replace("\u00a0", " ")
             .replace("\u200b", "")
             .replace("\u200c", "")
             .replace("\u200d", "")
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
    s = sanitize_text(s)
    return s.replace(" ", "")

# ---------- OpenAI client (lazy init; sanitizes API key) ----------
_client = None
def get_openai():
    global _client
    if _client is None:
        raw_key = os.environ.get("OPENAI_API_KEY", "")
        key = sanitize_key(raw_key)
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")

        # belt & suspenders: ensure no proxies have crept back in
        for _k in (
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy",
            "OPENAI_PROXY", "OPENAI_HTTP_PROXY", "OPENAI_HTTPS_PROXY"
        ):
            os.environ.pop(_k, None)
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"

        _client = OpenAI(api_key=key)
        print("[openai] client initialized")
    return _client

# ---------- Config ----------
bucket_name = os.environ.get("OUTPUT_BUCKET", "memory-books-output")
DEFAULT_PROMPT = (
    "Convert this photo into a professional adult coloring book page with clean continuous "
    "black outlines only on a pure white background"
)
MAX_IMAGE_BYTES = 20 * 1024 * 1024

# timeouts
REQUEST_CONNECT_TIMEOUT = 10
REQUEST_READ_TIMEOUT = 120
REQUEST_TIMEOUT = (REQUEST_CONNECT_TIMEOUT, REQUEST_READ_TIMEOUT)

# ---------- GCS client ----------
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    print(f"[init] Connected to GCS bucket: {bucket_name} | VERSION={VERSION}")
except Exception as e:
    print(f"[init] GCS not available: {safe_str(e)}")
    storage_client = None
    bucket = None

# ---------- HTTP headers / fetch helpers ----------
def _origin(url: str) -> str:
    try:
        u = urlparse.urlsplit(url)
        return f"{u.scheme}://{u.netloc}"
    except Exception:
        return ""

def _browser_headers(base_url: str, referer: str | None = None, is_image: bool = True) -> dict:
    # Build browser-like headers to satisfy CDNs
    hdrs = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": (
            "image/avif,image/webp,image/apng,image/*,*/*;q=0.8" if is_image
            else "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        ),
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    # Some hosts require a referer — use origin by default if provided
    if referer:
        hdrs["Referer"] = referer
    else:
        ori = _origin(base_url)
        if ori:
            hdrs["Referer"] = ori
    return hdrs

def _is_likely_image_url(url: str) -> bool:
    lower = url.lower()
    return lower.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"))

def _absolute_url(base_url: str, candidate: str) -> str:
    try:
        return urlparse.urljoin(base_url, candidate)
    except Exception:
        return candidate

def _fetch_once(url: str, headers: dict, stream: bool = True):
    return requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, stream=stream, allow_redirects=True)

def fetch_html(url: str) -> str:
    headers = _browser_headers(url, referer=None, is_image=False)
    r = _fetch_once(url, headers=headers, stream=False)
    r.raise_for_status()
    ct = (r.headers.get("content-type") or "").lower()
    if "text/html" not in ct and "application/xhtml+xml" not in ct:
        raise ValueError(f"Expected HTML, got content-type {ct or 'unknown'}")
    return r.text

def mine_image_from_html(html: str, base_url: str) -> str | None:
    # 1) og:image
    m = re.search(
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        html, re.IGNORECASE
    )
    if m:
        return _absolute_url(base_url, m.group(1))

    # 2) link rel=image_src
    m = re.search(
        r'<link[^>]+rel=["\']image_src["\'][^>]+href=["\']([^"\']+)["\']',
        html, re.IGNORECASE
    )
    if m:
        return _absolute_url(base_url, m.group(1))

    # 3) img srcset (pick the largest width if available)
    srcset_matches = re.findall(
        r'<img[^>]+srcset=["\']([^"\']+)["\']',
        html, re.IGNORECASE
    )
    for ss in srcset_matches:
        # srcset like: "https://... 320w, https://... 640w, ..."
        parts = [p.strip() for p in ss.split(",")]
        best = None
        best_w = -1
        for p in parts:
            # "URL WIDTHw"
            m2 = re.match(r'(.+?)\s+(\d+)w$', p)
            if m2:
                urlp = m2.group(1).strip()
                w = int(m2.group(2))
                if w > best_w:
                    best_w = w
                    best = urlp
        if best:
            return _absolute_url(base_url, best)

    # 4) fallback: any <img src="...">
    m = re.search(
        r'<img[^>]+src=["\']([^"\']+\.(?:jpg|jpeg|png|gif|webp|bmp|tiff)(?:\?[^"\']*)?)["\']',
        html, re.IGNORECASE
    )
    if m:
        return _absolute_url(base_url, m.group(1))

    # 5) last resort: try to discover CDN links inside JSON/script
    m = re.search(
        r'(https?://[^\s\'"]+\.(?:jpg|jpeg|png|gif|webp|bmp|tiff)(?:\?[^\'"\s<>]*)?)',
        html, re.IGNORECASE
    )
    if m:
        return m.group(1)

    return None

# ---------- Image download (robust: supports UploadKit/Shopify/CDNs) ----------
def download_image(url: str) -> bytes:
    url = sanitize_text(url)
    print(f"[fetch] {url[:200]}")
    base_origin = _origin(url)

    # Special-case: UploadKit/Shopify commonly gives direct CDN file URLs with query params.
    # Treat as direct image if it *looks* like a file OR host hints (getuploadkit.com, uploadcare, cloudfront).
    parsed = urlparse.urlsplit(url)
    host_lc = (parsed.netloc or "").lower()
    looks_uploadkitish = any(k in host_lc for k in [
        "getuploadkit.com", "uploadkit", "uploadcare", "cloudfront", "amazonaws.com"
    ])
    directish = _is_likely_image_url(url) or looks_uploadkitish

    # 1) Try direct download first
    headers_img = _browser_headers(url, referer=None, is_image=True)
    try:
        r = _fetch_once(url, headers=headers_img, stream=True)
        # If forbidden, retry with explicit referer (host origin)
        if r.status_code in (401, 403):
            r.close()
            headers_img_ref = _browser_headers(url, referer=base_origin, is_image=True)
            r = _fetch_once(url, headers=headers_img_ref, stream=True)

        r.raise_for_status()
        ct = (r.headers.get("content-type") or "").lower()

        # If we expected image and got image: stream it
        if "image/" in ct or directish:
            total = 0
            chunks = []
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    total += len(chunk)
                    if total > MAX_IMAGE_BYTES:
                        r.close()
                        raise ValueError("Image exceeds max size (20MB)")
                    chunks.append(chunk)
            r.close()
            return b"".join(chunks)

        # If we received HTML (wrapper page), fall through to HTML mining.
        r.close()

    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        # retry with referer if we didn't already above
        if status in (401, 403):
            headers_img_ref = _browser_headers(url, referer=base_origin, is_image=True)
            r2 = _fetch_once(url, headers=headers_img_ref, stream=True)
            if r2.status_code == 200 and "image/" in (r2.headers.get("content-type","").lower()):
                total = 0
                chunks = []
                for chunk in r2.iter_content(chunk_size=8192):
                    if chunk:
                        total += len(chunk)
                        if total > MAX_IMAGE_BYTES:
                            r2.close()
                            raise ValueError("Image exceeds max size (20MB)")
                        chunks.append(chunk)
                r2.close()
                return b"".join(chunks)
            r2.close()
        # else: proceed to HTML mining
    except Exception as e:
        print(f"[fetch] direct attempt failed: {safe_str(e)}; will try HTML scrape if possible")

    # 2) HTML mining path (for pages that wrap the real file)
    try:
        html = fetch_html(url)
        candidate = mine_image_from_html(html, url)
        if not candidate:
            raise ValueError("Could not find image on page")

        # download mined image url; try without, then with referer
        headers_img2 = _browser_headers(candidate, referer=None, is_image=True)
        r = _fetch_once(candidate, headers=headers_img2, stream=True)
        if r.status_code in (401, 403):
            r.close()
            headers_img2 = _browser_headers(candidate, referer=_origin(url), is_image=True)
            r = _fetch_once(candidate, headers=headers_img2, stream=True)
        r.raise_for_status()
        ct = (r.headers.get("content-type") or "").lower()
        if "image/" not in ct:
            r.close()
            raise ValueError(f"Resolved URL not an image (content-type: {ct or 'unknown'})")

        total = 0
        chunks = []
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    r.close()
                    raise ValueError("Image exceeds max size (20MB)")
                chunks.append(chunk)
        r.close()
        return b"".join(chunks)

    except Exception as e:
        raise ValueError(f"Failed to resolve image: {safe_str(e)}")

# ---------- OpenAI image edit ----------
def _decode_image_response(resp) -> bytes:
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
        clean_prompt = clean_prompt.encode("ascii", "ignore").decode("ascii")

        print(f"[openai] prompt: {clean_prompt[:80]}")

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

# ---------- Upload to GCS (public URL if bucket policy allows; else fallback to canonical) ----------
def upload_to_gcs(order_id: str, idx: int, img_bytes: bytes) -> str:
    if not bucket:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    blob_name = f"{order_id}/{int(time.time())}_{idx}.png"
    blob = bucket.blob(blob_name)
    blob.cache_control = "public, max-age=31536000, immutable"
    blob.upload_from_string(img_bytes, content_type="image/png")
    try:
        blob.patch()
    except Exception as e:
        print(f"[gcs] patch failed: {safe_str(e)}")
    try:
        blob.make_public()
    except Exception as e:
        print(f"[gcs] make_public failed (likely uniform access/PAP): {safe_str(e)}")

    url = blob.public_url
    if isinstance(url, bytes):
        url = url.decode("utf-8", "ignore")
    if not url or url.startswith("gs://"):
        url = f"https://storage.googleapis.com/{bucket.name}/{blob_name}"
    return url

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
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "bucket_name": bucket_name,
        "version": VERSION
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
