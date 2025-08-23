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
VERSION = "cbp-v2.0-dalle2-test"
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

# ---------- Config ----------
bucket_name = os.environ.get("OUTPUT_BUCKET", "memory-books-output")

# DALL-E 2 optimized coloring book prompt
DALLE2_COLORING_PROMPT = (
    "Convert this photograph into a premium adult coloring book illustration. "
    "Create bold, consistent black outlines with 2-3 pixel thickness on pure white background. "
    "Preserve ALL facial features accurately - maintain proper proportions, expressions, and recognizable details. "
    "Include rich background elements like architectural details, furniture, other people, and environmental context. "
    "Maintain accurate representation of all objects - do not change food items, drinks, or other recognizable elements. "
    "Remove all colors, shadows, and photographic textures while preserving structural details. "
    "Ensure all outlines are continuous, closed, and perfect for coloring with markers or colored pencils. "
    "Style should match professional adult coloring books with engaging detail levels."
)

# Model and sizing configuration  
MODEL_NAME = "dall-e-2"  # Official OpenAI model
COLORING_BOOK_SIZE = "1024x1024"
MAX_IMAGES_PER_ORDER = 24
MAX_IMAGE_BYTES = 20 * 1024 * 1024
REQUEST_TIMEOUT = 45
OPENAI_TIMEOUT = 600

# ---------- OpenAI REST API ----------
OPENAI_API_BASE = "https://api.openai.com/v1"

def get_api_key() -> str:
    key = sanitize_key(os.environ.get("OPENAI_API_KEY",""))
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return key

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

def call_dalle2_edit(image_png_bytes: bytes, prompt: str) -> bytes:
    """Call DALL-E 2 images/edit API directly via REST with full-image mask."""
    api_key = get_api_key()
    
    # Create a session with no proxy environment variables
    session = requests.Session()
    session.trust_env = False  # Ignore all proxy env vars
    
    # Create full white mask (edit entire image)
    img = Image.open(io.BytesIO(image_png_bytes))
    mask = Image.new("RGBA", img.size, (255, 255, 255, 255))  # Full white mask
    mask_buffer = io.BytesIO()
    mask.save(mask_buffer, format="PNG")
    mask_bytes = mask_buffer.getvalue()
    
    files = {
        "image": ("image.png", image_png_bytes, "image/png"),
        "mask": ("mask.png", mask_bytes, "image/png")
    }
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "size": COLORING_BOOK_SIZE,
        "n": 1
    }
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    print(f"[dalle2] calling edit API with {MODEL_NAME} + full mask for {COLORING_BOOK_SIZE}...")
    
    response = session.post(
        f"{OPENAI_API_BASE}/images/edits",
        headers=headers,
        data=data,
        files=files,
        timeout=(15, OPENAI_TIMEOUT)
    )
    
    if response.status_code != 200:
        try:
            error_info = response.json()
        except:
            error_info = {"error": {"message": response.text}}
        raise Exception(f"DALL-E 2 API error {response.status_code}: {safe_str(error_info)}")
    
    return decode_image_response(response.json())

def call_openai_edit(image_bytes: bytes, prompt: str) -> bytes:
    try:
        # DALL-E 2 requires RGBA format for editing
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGBA (with alpha channel) for DALL-E 2
        if img.mode != "RGBA":
            # Create RGBA version
            if img.mode == "RGB":
                # Add full opacity alpha channel
                img = img.convert("RGBA")
            elif img.mode in ("L", "LA"):
                # Grayscale to RGBA
                img = img.convert("RGBA")
            else:
                # Any other mode to RGBA
                img = img.convert("RGBA")
        
        buf = io.BytesIO()
        img.save(buf, format="PNG")  # PNG preserves RGBA
        png_bytes = buf.getvalue()
        
        # ALWAYS use the DALL-E 2 optimized prompt
        final_prompt = DALLE2_COLORING_PROMPT
        
        print(f"[dalle2] processing RGBA image with optimized DALL-E 2 prompt...")
        
        return call_dalle2_edit(png_bytes, final_prompt)
        
    except Exception as e:
        print(f"[dalle2] primary failed: {safe_str(e)}; retrying with simplified prompt")
        try:
            # Try again with simplified fallback
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            
            return call_dalle2_edit(buf.getvalue(), "Transform into clean black line art coloring book page preserving facial features and background details")
            
        except Exception as e2:
            raise Exception(f"DALL-E 2 processing failed: {safe_str(e2)}")

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
        
        # Validate image count
        if len(image_urls) > MAX_IMAGES_PER_ORDER:
            return safe_json_response({
                "success": False, 
                "error": f"Too many images. Maximum {MAX_IMAGES_PER_ORDER} per order."
            }, 400)
        
        if not image_urls:
            return safe_json_response({
                "success": False, 
                "error": "No image URLs provided"
            }, 400)
        
        print(f"[process] {order_id} - processing {len(image_urls)} image(s) with DALL-E 2 model")
        
        results = []
        total_success = 0
        
        for idx, url in enumerate(image_urls):
            try:
                print(f"[process] Processing image {idx + 1}/{len(image_urls)}")
                
                raw = download_image(url)
                print(f"[process] Downloaded {len(raw)} bytes for image {idx + 1}")
                
                edited = call_openai_edit(raw, "")  # Empty prompt since we override it
                print(f"[process] DALL-E 2 processing complete for image {idx + 1} -> {len(edited)} bytes")
                
                final_url = upload_to_gcs(order_id, idx, edited)
                print(f"[process] Uploaded image {idx + 1} to: {final_url[:100]}...")
                
                results.append({
                    "status": "success",
                    "index": idx,
                    "source_url": url,
                    "result_url": final_url if bucket else None,
                    "result_base64": None if bucket else final_url,
                    "storage_type": "gcs" if bucket else "data-url",
                    "size": COLORING_BOOK_SIZE
                })
                total_success += 1
                
            except Exception as e:
                err = sanitize_text(str(e))
                print(f"[process] ERROR processing image {idx + 1}: {err}")
                results.append({
                    "status": "error",
                    "index": idx,
                    "source_url": url,
                    "error": err
                })
        
        return safe_json_response({
            "success": True,
            "total_images": len(image_urls),
            "successful_images": total_success,
            "failed_images": len(image_urls) - total_success,
            "order_id": order_id,
            "prompt_used": "DALLE2_COLORING_PROMPT (facial accuracy focused)",
            "model_used": MODEL_NAME,
            "image_size": COLORING_BOOK_SIZE,
            "results": results
        })
        
    except Exception as e:
        err = safe_str(e)
        print(f"[process] Request failed: {err}")
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
            "message": "DALL-E 2 test successful!",
            "original_url": test_url,
            "result_base64_preview": b64[:100] + "...",
            "result_size": len(edited),
            "model_used": MODEL_NAME
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
        "version": VERSION,
        "model": MODEL_NAME
    })

@app.route("/", methods=["GET"])
def index():
    return safe_json_response({
        "service": "Coloring Book Processor",
        "version": VERSION,
        "model": MODEL_NAME,
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
                "prompt": "ignored - using DALL-E 2 optimized prompt"
            }
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
