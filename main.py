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
from typing import Optional, Dict, List, Any

VERSION = "cbp-v3.0-bulletproof"

# --- Nuke any proxy env that could interfere ---
for _k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy",
           "OPENAI_PROXY","OPENAI_HTTP_PROXY","OPENAI_HTTPS_PROXY"):
    if os.environ.get(_k):
        print(f"[net] ignoring proxy env {_k}")
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

app = Flask(__name__)

# ---------- PROMPT STRATEGIES FOR A/B TESTING ----------
PROMPTS = {
    "MINIMAL": "Convert to clean line art coloring book page",
    
    "DETAILED": (
        "Transform this photograph into a professional adult coloring book illustration. "
        "Create bold, consistent black outlines on pure white background. "
        "Preserve all facial features accurately with proper proportions. "
        "Include detailed backgrounds and maintain all recognizable objects exactly as they appear. "
        "Remove colors and shadows while keeping structural detail. "
        "Ensure continuous closed outlines perfect for coloring."
    ),
    
    "FACE_FOCUSED": (
        "Convert to line art coloring book preserving exact facial features and expressions. "
        "Maintain accurate proportions and recognizable details. "
        "Bold black outlines, white background, detailed environment."
    ),
    
    "TECHNICAL": (
        "Generate black line art illustration: 2-3px line weight, closed paths, "
        "facial accuracy priority, preserve background context, "
        "remove photographic textures, coloring book style"
    )
}

# Get prompt strategy from environment (default to DETAILED)
PROMPT_STRATEGY = os.environ.get("PROMPT_STRATEGY", "DETAILED")
SELECTED_PROMPT = PROMPTS.get(PROMPT_STRATEGY, PROMPTS["DETAILED"])

# ---------- Enhanced Config ----------
bucket_name = os.environ.get("OUTPUT_BUCKET", "memory-books-output")
MAX_IMAGE_BYTES = 25 * 1024 * 1024  # Increased to 25MB
MAX_IMAGES_PER_ORDER = 24
REQUEST_TIMEOUT = 45  # Increased timeout
OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "600"))  # 10 minutes
OPENAI_RETRY_ATTEMPTS = 3
IMAGE_QUALITY = 95  # High quality PNG compression

# ---------- Utilities ----------
def safe_str(obj) -> str:
    try:
        s = str(obj)
        s = (s.replace("\u2028"," ").replace("\u2029"," ").replace("\u00a0"," ")
               .replace("\u200b","").replace("\u200c","").replace("\u200d",""))
        return "".join(ch if ord(ch) < 128 else "?" for ch in s)
    except Exception:
        return "Error converting to string"

def make_safe_dict(obj) -> Any:
    if isinstance(obj, dict):
        return {safe_str(k): make_safe_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_safe_dict(item) for item in obj]
    elif isinstance(obj, str):
        return safe_str(obj)
    else:
        return safe_str(obj)

def safe_json_response(data: Dict, status_code: int = 200) -> Response:
    txt = json.dumps(make_safe_dict(data), ensure_ascii=True, separators=(",", ":"))
    return Response(txt, status=status_code, mimetype="application/json")

def sanitize_text(s: str) -> str:
    return safe_str(s).strip()

def sanitize_key(s: str) -> str:
    s = sanitize_text(s)
    return s.replace(" ", "").replace("\n", "").replace("\t", "")

def get_api_key() -> str:
    raw = os.environ.get("OPENAI_API_KEY", "")
    key = sanitize_key(raw)
    if not key or len(key) < 20:  # Basic validation
        raise RuntimeError("OPENAI_API_KEY not set or invalid")
    return key

def validate_image_url(url: str) -> bool:
    """Validate image URL format and accessibility."""
    if not url or len(url) < 10:
        return False
    if not (url.startswith("http://") or url.startswith("https://")):
        return False
    return True

# ---------- GCS client with error handling ----------
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    # Test bucket accessibility
    bucket.exists()
    print(f"[init] Connected to GCS bucket: {bucket_name} | VERSION={VERSION}")
except Exception as e:
    print(f"[init] GCS not available: {safe_str(e)}")
    storage_client = None
    bucket = None

# ---------- Enhanced Image Processing ----------
def extract_direct_image_url(url: str) -> str:
    url = sanitize_text(url)
    lower = url.lower()
    
    # Direct image URLs
    if any(lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")):
        return url
        
    # UploadKit or HTML pages
    if any(keyword in lower for keyword in ("uploadkit", "download.html", "files.getuploadkit.com")):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive"
            }
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
            resp.raise_for_status()
            html = resp.text
            
            # More comprehensive patterns
            patterns = [
                r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
                r'<img[^>]+src=["\']([^"\']+\.(jpg|jpeg|png|gif|webp|bmp)[^"\']*)["\'][^>]*>',
                r'href=["\']([^"\']+\.(jpg|jpeg|png|gif|webp|bmp)(?:\?[^"\']*)?)["\']',
                r'url\(["\']?([^"\']+\.(jpg|jpeg|png|gif|webp|bmp))["\']?\)',
                r'"(https?://[^"]+\.(jpg|jpeg|png|gif|webp|bmp)(?:\?[^"]*)?)"'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, html, re.IGNORECASE)
                for match in matches:
                    img_url = match[0] if isinstance(match, tuple) else match
                    img_url = img_url.strip()
                    
                    if img_url.startswith("http"):
                        return img_url
                    elif img_url.startswith("//"):
                        return "https:" + img_url
                    elif img_url.startswith("/"):
                        base_url = "/".join(url.split("/")[:3])
                        return base_url + img_url
                        
        except Exception as e:
            print(f"[extract] HTML mining failed: {safe_str(e)}")
    
    return url

def download_image(url: str) -> bytes:
    if not validate_image_url(url):
        raise ValueError(f"Invalid image URL: {url[:100]}")
        
    direct = extract_direct_image_url(url)
    print(f"[fetch] {direct[:150]}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://uploadkit.com/",
        "Cache-Control": "no-cache"
    }
    
    with requests.get(direct, headers=headers, timeout=REQUEST_TIMEOUT, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        
        # Verify content type
        content_type = r.headers.get("content-type", "").lower()
        if "text/html" in content_type:
            raise ValueError(f"Server returned HTML instead of image from {direct[:50]}")
        
        # Verify it's actually an image
        if not any(img_type in content_type for img_type in ["image/", "application/octet-stream"]):
            print(f"[warn] Unexpected content-type: {content_type}")
        
        total, chunks = 0, []
        for chunk in r.iter_content(chunk_size=16384):  # Larger chunks for efficiency
            if chunk:
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    raise ValueError(f"Image exceeds maximum size ({MAX_IMAGE_BYTES/1024/1024:.1f}MB)")
                chunks.append(chunk)
        
        if total < 1024:  # Image too small (likely error page)
            raise ValueError("Downloaded data too small to be a valid image")
            
        print(f"[fetch] Downloaded {total:,} bytes")
        return b"".join(chunks)

def _decode_image_json(response_json: Dict) -> bytes:
    data = response_json.get("data", [])
    if not data:
        raise Exception("No data in OpenAI response")
        
    item = data[0]
    
    # Try b64_json first (more reliable)
    if "b64_json" in item and item["b64_json"]:
        try:
            return base64.b64decode(item["b64_json"])
        except Exception as e:
            print(f"[openai] b64 decode failed: {safe_str(e)}")
    
    # Fallback to URL
    if "url" in item and item["url"]:
        try:
            session = requests.Session()
            session.trust_env = False
            with session.get(item["url"], timeout=(15, OPENAI_TIMEOUT)) as r:
                r.raise_for_status()
                content = r.content
                if len(content) < 1024:
                    raise Exception("Downloaded image too small")
                return content
        except Exception as e:
            print(f"[openai] URL download failed: {safe_str(e)}")
    
    raise Exception("No valid image data found in OpenAI response")

# ---------- OpenAI with retry logic ----------
OPENAI_IMAGES_EDITS = "https://api.openai.com/v1/images/edits"

def _rest_image_edit(image_png: bytes, prompt: str, attempt: int = 1) -> bytes:
    key = get_api_key()
    
    session = requests.Session()
    session.trust_env = False

    files = {"image": ("image.png", image_png, "image/png")}
    data = {
        "model": "gpt-image-1",
        "prompt": prompt,
        "size": "1024x1024",
    }
    headers = {"Authorization": f"Bearer {key}"}
    timeout = (15, OPENAI_TIMEOUT)

    print(f"[openai-http] attempt {attempt}/{OPENAI_RETRY_ATTEMPTS} calling images/edits")
    
    try:
        resp = session.post(OPENAI_IMAGES_EDITS, headers=headers, data=data, files=files, timeout=timeout)
        
        if resp.status_code >= 400:
            try:
                err = resp.json()
                error_msg = err.get("error", {}).get("message", resp.text)
            except:
                error_msg = resp.text
            
            # Don't retry on client errors (400-499)
            if 400 <= resp.status_code < 500:
                raise Exception(f"OpenAI client error {resp.status_code}: {safe_str(error_msg)}")
            else:
                raise Exception(f"OpenAI server error {resp.status_code}: {safe_str(error_msg)}")

        return _decode_image_json(resp.json())
        
    except requests.exceptions.Timeout as e:
        if attempt < OPENAI_RETRY_ATTEMPTS:
            print(f"[openai] timeout on attempt {attempt}, retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff
            return _rest_image_edit(image_png, prompt, attempt + 1)
        raise Exception(f"OpenAI timeout after {attempt} attempts")
    
    except requests.exceptions.RequestException as e:
        if attempt < OPENAI_RETRY_ATTEMPTS:
            print(f"[openai] request error on attempt {attempt}: {safe_str(e)}, retrying...")
            time.sleep(2 ** attempt)
            return _rest_image_edit(image_png, prompt, attempt + 1)
        raise Exception(f"OpenAI request failed after {attempt} attempts: {safe_str(e)}")

def call_openai_edit(image_bytes: bytes, prompt: str) -> bytes:
    # Enhanced image preprocessing
    try:
        img = Image.open(io.BytesIO(image_bytes))
        original_size = img.size
        print(f"[openai] processing {img.mode} image {original_size[0]}x{original_size[1]}")
        
        # Convert to RGB with white background
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode in ("P", "L", "LA"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        
        # Optimize image size for processing (maintain aspect ratio)
        max_dimension = 1024
        if max(img.size) > max_dimension:
            ratio = max_dimension / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            print(f"[openai] resized to {new_size[0]}x{new_size[1]} for processing")
        
        # Save with high quality
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        png_bytes = buf.getvalue()
        
        # Use selected prompt strategy
        clean_prompt = sanitize_text(prompt) if (prompt and len(prompt) > 10) else SELECTED_PROMPT
        if not clean_prompt or len(clean_prompt) < 5:
            clean_prompt = PROMPTS["MINIMAL"]
        
        # Ensure ASCII encoding
        clean_prompt = clean_prompt.encode("ascii", "ignore").decode("ascii")
        print(f"[openai] using {PROMPT_STRATEGY} strategy: {clean_prompt[:100]}")

        return _rest_image_edit(png_bytes, clean_prompt)
        
    except Exception as e:
        print(f"[openai] primary processing failed: {safe_str(e)}")
        
        # Fallback with minimal prompt
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != "RGB":
                if img.mode == "RGBA":
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                else:
                    img = img.convert("RGB")
            
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            
            print("[openai] retrying with minimal prompt")
            return _rest_image_edit(buf.getvalue(), PROMPTS["MINIMAL"])
            
        except Exception as e2:
            raise Exception(f"Image processing failed completely: {safe_str(e2)}")

# ---------- Enhanced Upload with validation ----------
def upload_to_gcs(order_id: str, idx: int, img_bytes: bytes) -> str:
    if not img_bytes or len(img_bytes) < 1024:
        raise ValueError("Invalid image data for upload")
        
    if not bucket:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    # Generate unique blob name with timestamp
    timestamp = int(time.time() * 1000)  # Millisecond precision
    blob_name = f"{sanitize_text(order_id)}/{timestamp}_{idx:03d}.png"
    blob = bucket.blob(blob_name)

    try:
        # Upload with metadata
        blob.cache_control = "public, max-age=31536000, immutable"
        blob.metadata = {
            "order_id": sanitize_text(order_id),
            "image_index": str(idx),
            "processed_at": str(timestamp),
            "version": VERSION
        }
        
        blob.upload_from_string(img_bytes, content_type="image/png")
        
        # Apply settings
        try:
            blob.patch()
        except Exception as e:
            print(f"[gcs] metadata patch failed: {safe_str(e)}")
            
        # Make public (ignore errors due to bucket policies)
        try:
            blob.make_public()
        except Exception as e:
            print(f"[gcs] make_public failed (expected with uniform access): {safe_str(e)}")

        # Generate URL
        url = blob.public_url
        if isinstance(url, bytes):
            url = url.decode("utf-8", "ignore")
        if not url or url.startswith("gs://"):
            url = f"https://storage.googleapis.com/{bucket.name}/{blob_name}"
        
        print(f"[gcs] uploaded to {url}")
        return url
        
    except Exception as e:
        print(f"[gcs] upload failed: {safe_str(e)}")
        # Fallback to base64 if GCS fails
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"

# ---------- Enhanced Routes ----------
@app.route("/process", methods=["POST"])
def process():
    start_time = time.time()
    
    try:
        payload = request.get_json(force=True) or {}
        order_id = sanitize_text(payload.get("order_id", f"order_{int(time.time())}"))

        # Validate and process image URLs
        image_urls = payload.get("image_urls") or payload.get("urls") or []
        if isinstance(image_urls, str):
            image_urls = [u.strip() for u in image_urls.split(",") if u.strip()]
        
        # Filter and validate URLs
        valid_urls = []
        for url in image_urls:
            clean_url = sanitize_text(url)
            if validate_image_url(clean_url):
                valid_urls.append(clean_url)
            else:
                print(f"[process] skipping invalid URL: {clean_url[:100]}")
        
        if not valid_urls:
            return safe_json_response({
                "success": False,
                "error": "No valid image URLs provided"
            }, 400)
            
        if len(valid_urls) > MAX_IMAGES_PER_ORDER:
            return safe_json_response({
                "success": False,
                "error": f"Too many images. Maximum {MAX_IMAGES_PER_ORDER} per order, got {len(valid_urls)}"
            }, 400)

        # Process prompt
        raw_prompt = payload.get("prompt", "")
        prompt = sanitize_text(raw_prompt) if raw_prompt else ""

        print(f"[process] {order_id} - processing {len(valid_urls)} image(s) with {PROMPT_STRATEGY} strategy")

        results = []
        total_success = 0
        total_errors = 0

        for idx, url in enumerate(valid_urls):
            image_start = time.time()
            try:
                print(f"[process] Processing image {idx + 1}/{len(valid_urls)}: {url[:100]}")
                
                # Download
                raw_bytes = download_image(url)
                download_time = time.time() - image_start
                
                # Process with OpenAI
                processing_start = time.time()
                edited_bytes = call_openai_edit(raw_bytes, prompt)
                processing_time = time.time() - processing_start
                
                # Upload
                upload_start = time.time()
                result_url = upload_to_gcs(order_id, idx, edited_bytes)
                upload_time = time.time() - upload_start
                
                image_total_time = time.time() - image_start
                
                print(f"[process] Image {idx + 1} completed in {image_total_time:.1f}s "
                      f"(download: {download_time:.1f}s, process: {processing_time:.1f}s, upload: {upload_time:.1f}s)")
                
                results.append({
                    "status": "success",
                    "index": idx,
                    "source_url": url,
                    "result_url": result_url,
                    "storage_type": "gcs" if bucket else "base64",
                    "processing_time_seconds": round(image_total_time, 1),
                    "file_size_bytes": len(edited_bytes)
                })
                total_success += 1
                
            except Exception as e:
                error_time = time.time() - image_start
                error_msg = safe_str(e)
                print(f"[process] ERROR processing image {idx + 1} after {error_time:.1f}s: {error_msg}")
                
                results.append({
                    "status": "error",
                    "index": idx,
                    "source_url": url,
                    "error": error_msg,
                    "processing_time_seconds": round(error_time, 1)
                })
                total_errors += 1

        total_time = time.time() - start_time
        
        return safe_json_response({
            "success": True,
            "order_id": order_id,
            "prompt_strategy": PROMPT_STRATEGY,
            "total_images": len(valid_urls),
            "successful_images": total_success,
            "failed_images": total_errors,
            "total_processing_time_seconds": round(total_time, 1),
            "results": results
        })
        
    except Exception as e:
        error_time = time.time() - start_time
        error_msg = safe_str(e)
        print(f"[process] Request failed after {error_time:.1f}s: {error_msg}")
        return safe_json_response({
            "success": False,
            "error": error_msg,
            "processing_time_seconds": round(error_time, 1)
        }, 500)

@app.route("/test", methods=["GET", "POST"])
def test():
    test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"
    try:
        raw = download_image(test_url)
        edited = call_openai_edit(raw, "Convert to coloring book page")
        b64_preview = base64.b64encode(edited).decode("utf-8")[:200]
        
        return safe_json_response({
            "success": True,
            "message": "Test successful!",
            "original_url": test_url,
            "result_base64_preview": b64_preview + "...",
            "result_size_bytes": len(edited),
            "prompt_strategy": PROMPT_STRATEGY,
            "version": VERSION
        })
    except Exception as e:
        return safe_json_response({
            "success": False,
            "error": safe_str(e),
            "version": VERSION
        }, 500)

@app.route("/health", methods=["GET"])
def health():
    return safe_json_response({
        "status": "healthy",
        "service": "coloring-book-processor",
        "version": VERSION,
        "gcs_available": bucket is not None,
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "bucket_name": bucket_name,
        "prompt_strategy": PROMPT_STRATEGY,
        "max_images_per_order": MAX_IMAGES_PER_ORDER,
        "openai_timeout_seconds": OPENAI_TIMEOUT,
        "available_strategies": list(PROMPTS.keys())
    })

@app.route("/", methods=["GET"])
def index():
    return safe_json_response({
        "service": "Coloring Book Processor",
        "version": VERSION,
        "prompt_strategy": PROMPT_STRATEGY,
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
                "urls": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"],
                "prompt": "optional custom prompt"
            }
        },
        "available_prompt_strategies": PROMPTS
    })

if __name__ == "__main__":
    print(f"[startup] Coloring Book Processor {VERSION}")
    print(f"[startup] Using prompt strategy: {PROMPT_STRATEGY}")
    print(f"[startup] Selected prompt: {SELECTED_PROMPT[:100]}...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
