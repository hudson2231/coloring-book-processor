import os
import io
import time
import base64
import requests
from flask import Flask, request, Response
import json
from google.cloud import storage
import re
from PIL import Image
from typing import Optional

# --- hard kill any proxy env that could affect OpenAI/http ---
for _k in (
    "HTTP_PROXY","HTTPS_PROXY","ALL_PROXY",
    "http_proxy","https_proxy","all_proxy",
    "OPENAI_PROXY","OPENAI_HTTP_PROXY","OPENAI_HTTPS_PROXY"
):
    if os.environ.get(_k):
        print(f"[net] ignoring proxy env {_k}")
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
VERSION = "cbp-v1.3-py39-fix"
app = Flask(__name__)

# ---------- Utilities ----------
def safe_str(obj):
    try:
        s = str(obj)
        s = (s.replace("\u2028"," ").replace("\u2029"," ").replace("\u00a0"," ")
               .replace("\u200b","").replace("\u200c","").replace("\u200d",""))
        return "".join(ch if ord(ch) < 128 else "?" for ch in s)
    except Exception:
        return "Error converting to string"

def make_safe_dict(obj):
    if isinstance(obj, dict):
        return {safe_str(k): make_safe_dict(v) for k,v in obj.items()}
    elif isinstance(obj, list):
        return [make_safe_dict(x) for x in obj]
    elif isinstance(obj, str):
        return safe_str(obj)
    else:
        return safe_str(obj)

def safe_json_response(data, status_code=200):
    txt = json.dumps(make_safe_dict(data), ensure_ascii=True, separators=(",",":"))
    return Response(txt, status=status_code, mimetype="application/json")

def sanitize_text(s: str) -> str:
    return safe_str(s).strip()

def sanitize_key(s: str) -> str:
    return sanitize_text(s).replace(" ","")

# ---------- OpenAI REST API (bypass SDK completely) ----------
OPENAI_API_BASE = "https://api.openai.com/v1"

def get_api_key() -> str:
    key = sanitize_key(os.environ.get("OPENAI_API_KEY",""))
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return key

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
    print(f"[init] Connected to GCS bucket: {bucket_name} | VERSION={VERSION}")
except Exception as e:
    print(f"[init] GCS not available: {safe_str(e)}")
    storage_client = None
    bucket = None

# ---------- Image fetch helpers ----------
UA_BROWSER = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
BASE_HEADERS = {"User-Agent": UA_BROWSER, "Accept": "*/*"}

def mine_html_for_image(html: str, page_url: str) -> Optional[str]:
    patterns = [
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<img[^>]+srcset=["\']([^"\']+)["\']',
        r'<img[^>]+src=["\']([^"\']+)["\']',
        r'href=["\']([^"\']+\.(?:jpg|jpeg|png|gif|webp)(?:\?[^"\']*)?)["\']',
    ]
    for p in patterns:
        m = re.search(p, html, re.IGNORECASE)
        if m:
            img_url = m.group(1).strip()
            if img_url.startswith("//"):   return "https:" + img_url
            if img_url.startswith("/"):    return "/".join(page_url.split("/")[:3]) + img_url
            return img_url
    return None

def extract_direct_image_url(url: str) -> str:
    url = sanitize_text(url)
    lower = url.lower()
    # UploadKit direct CDN or any URL that already looks like an image
    if any(lower.endswith(ext) for ext in (".jpg",".jpeg",".png",".gif",".webp")) or "files.getuploadkit.com" in lower:
        return url
    # If it might be a landing page, try to mine HTML
    try:
        resp = requests.get(url, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200 and "text/html" in resp.headers.get("content-type","").lower():
            mined = mine_html_for_image(resp.text, url)
            if mined: return mined
    except Exception as e:
        print(f"[extract] failed to mine HTML: {safe_str(e)}")
    return url

def download_image(url: str) -> bytes:
    direct = extract_direct_image_url(url)
    print(f"[fetch] {direct[:200]}")
    headers = dict(BASE_HEADERS)
    # Some CDNs require a referer (UploadKit/Shopify flows)
    headers["Referer"] = "https://shopify.com/"
    
    # First attempt
    with requests.get(direct, headers=headers, timeout=REQUEST_TIMEOUT, stream=True, allow_redirects=True) as r:
        if r.status_code in (401,403):
            # Retry with even more browsery headers
            headers.update({
                "Accept-Language": "en-US,en;q=0.9",
                "Sec-Fetch-Mode": "no-cors",
                "Sec-Fetch-Site": "cross-site",
            })
            r.close()
            r = requests.get(direct, headers=headers, timeout=REQUEST_TIMEOUT, stream=True, allow_redirects=True)
        r.raise_for_status()
        ct = (r.headers.get("content-type","") or "").lower()
        if "text/html" in ct:
            # Final attempt to mine inside HTML
            html = r.text
            mined = mine_html_for_image(html, direct)
            if not mined:
                raise ValueError("Got HTML instead of image")
            r.close()
            r = requests.get(mined, headers=headers, timeout=REQUEST_TIMEOUT, stream=True, allow_redirects=True)
            r.raise_for_status()
        total, chunks = 0, []
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    raise ValueError("Image exceeds max size (20MB)")
                chunks.append(chunk)
    return b"".join(chunks)

def decode_image_response(response_json) -> bytes:
    data = response_json.get("data", [])
    if not data:
        raise Exception("No data in OpenAI response")
    
    item = data[0]
    
    # Try b64_json first
    if "b64_json" in item and item["b64_json"]:
        return base64.b64decode(item["b64_json"])
    
    # Try URL
    if "url" in item and item["url"]:
        img_resp = requests.get(item["url"], headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        img_resp.raise_for_status()
        return img_resp.content
    
    raise Exception("No image data found in OpenAI response")

def call_openai_edit_rest(image_png_bytes: bytes, prompt: str) -> bytes:
    """Call OpenAI images/edit API directly via REST, no SDK."""
    api_key = get_api_key()
    
    # Create a session with no proxy environment variables
    session = requests.Session()
    session.trust_env = False  # Ignore all proxy env vars
    
    files = {
        "image": ("image.png", image_png_bytes, "image/png")
    }
    data = {
        "model": "gpt-image-1",  # Keep your reverse-engineered model name
        "prompt": prompt,
        "size": "1024x1024",
        "response_format": "url"  # Get URL to avoid base64 issues
    }
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    print(f"[openai-rest] calling edit API with prompt: {prompt[:50]}...")
    
    response = session.post(
        f"{OPENAI_API_BASE}/images/edits",
        headers=headers,
        data=data,
        files=files,
        timeout=(10, 300)  # 10s connect, 300s read
    )
    
    if response.status_code != 200:
        try:
            error_info = response.json()
        except:
            error_info = {"error": {"message": response.text}}
        raise Exception(f"OpenAI API error {response.status_code}: {safe_str(error_info)}")
    
    return decode_image_response(response.json())

def call_openai_edit(image_bytes: bytes, prompt: str) -> bytes:
    try:
        # Normalize image to PNG (RGB on white background if RGBA)
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255,255,255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        
        clean_prompt = sanitize_text(prompt) if (prompt and len(prompt) > 10) else sanitize_text(DEFAULT_PROMPT)
        if not clean_prompt or len(clean_prompt) < 10:
            clean_prompt = "Convert to line art coloring book page"
        clean_prompt = clean_prompt.encode("ascii","ignore").decode("ascii")
        
        print(f"[openai] processing with prompt: {clean_prompt[:80]}")
        
        return call_openai_edit_rest(png_bytes, clean_prompt)
        
    except Exception as e:
        print(f"[openai] primary failed: {safe_str(e)}; retrying with minimal prompt")
        try:
            # Try again with minimal prompt
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            
            return call_openai_edit_rest(buf.getvalue(), "line art coloring page")
            
        except Exception as e2:
            raise Exception(f"Image processing failed: {safe_str(e2)}")

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
                results.append({"status":"error","index":idx,"source_url":url,"error":err})
        
        return safe_json_response({
            "success": True,
            "count": len(results),
            "order_id": order_id,
            "prompt_used": (prompt.encode("ascii","ignore").decode("ascii"))[:100],
            "results": results
        })
    except Exception as e:
        err = safe_str(e)
        print(f"[process] request failed: {err}")
        return safe_json_response({"success": False, "error": err}, 500)

@app.route("/test", methods=["GET","POST"])
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
