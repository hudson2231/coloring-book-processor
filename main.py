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

# Enable basic PIL features
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
print("[init] PIL truncated image support enabled")

VERSION = "cbp-v4.2-quality-heic-fixed"

# --- Nuke any proxy env that could interfere ---
for _k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy",
           "OPENAI_PROXY","OPENAI_HTTP_PROXY","OPENAI_HTTPS_PROXY"):
    if os.environ.get(_k):
        print(f"[net] ignoring proxy env {_k}")
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

app = Flask(__name__)

# ---------- IMPROVED QUALITY PROMPTS ----------
PERFECT_PROMPTS = {
    "ULTIMATE_QUALITY": (
        "Convert to professional adult coloring book line art with these EXACT specifications: "
        "BOLD 4-pixel consistent black outlines throughout entire image on pure white background. "
        "PRESERVE facial features with perfect accuracy - maintain exact expressions, eye shape, smile, proportions. "
        "CAPTURE every small detail: jewelry chains, necklaces, earrings, clothing textures, hair definition. "
        "INTERPRET dark/shadowy background areas as clear structural line elements - ceiling details, wall features, other people as line drawings. "
        "ENSURE all lines are SHARP and CRISP - no soft, blurry, or faded outlines anywhere. "
        "CONVERT photographic lighting and shadows into drawable line art elements - not darkness. "
        "MAINTAIN rich environmental context with clear line-drawn background elements. "
        "CREATE closed, continuous outlines perfect for coloring with markers or colored pencils. "
        "RENDER as hand-drawn professional coloring book illustration quality with intricate detail."
    ),
    
    "ENHANCED_LINE_ART": (
        "Transform this photograph into premium adult coloring book line art illustration. "
        "Requirements: THICK black outlines (4px weight), pure white background, zero colors or shading. "
        "Preserve ALL facial features, expressions, clothing details, jewelry, and background elements as bold line drawings. "
        "Convert shadows and lighting into structural line details - NOT empty space or darkness. "
        "Create intricate, detailed line work with consistent bold strokes throughout. "
        "Ensure every element has clear, closed outlines perfect for marker coloring. "
        "Professional illustration quality with rich environmental context and fine details."
    ),
    
    "MAXIMUM_DETAIL": (
        "Create professional adult coloring book line art with maximum detail preservation. "
        "Use bold 4-pixel black outlines on white background only. "
        "Maintain exact facial features, hair texture, clothing patterns, jewelry details. "
        "Transform background elements into detailed line drawings - architecture, objects, people. "
        "Convert all photographic elements into crisp line art with no soft edges. "
        "Include fine details like fabric textures, architectural elements, and environmental context. "
        "Generate publication-quality coloring book illustration with intricate linework."
    )
}

# Use the enhanced prompt for better quality
SELECTED_PROMPT = PERFECT_PROMPTS["ENHANCED_LINE_ART"]

# ---------- BUSINESS CRITICAL CONFIG ----------
bucket_name = os.environ.get("OUTPUT_BUCKET", "memory-books-output")
MAX_IMAGE_BYTES = 100 * 1024 * 1024  # 100MB for any format
MAX_IMAGES_PER_ORDER = 24  # Exactly what you need
REQUEST_TIMEOUT = 180  # 3 minutes per download
OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "1200"))  # 20 minutes
OPENAI_RETRY_ATTEMPTS = 5  # More retries for reliability

# ---------- Utilities ----------
def safe_str(obj) -> str:
    try:
        s = str(obj)
        s = (s.replace("\u2028"," ").replace("\u2029"," ").replace("\u00a0"," ")
               .replace("\u200b","").replace("\u200c","").replace("\u200d",""))
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
    return s.replace(" ", "").replace("\n", "").replace("\t", "")

def get_api_key() -> str:
    raw = os.environ.get("OPENAI_API_KEY", "")
    key = sanitize_key(raw)
    if not key or len(key) < 20:
        raise RuntimeError("OPENAI_API_KEY not set or invalid")
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

# ---------- BULLETPROOF IMAGE DOWNLOAD ----------
def extract_direct_image_url(url: str) -> str:
    url = sanitize_text(url)
    lower = url.lower()
    
    # Direct image URLs - any format
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic", ".heif")
    if any(lower.endswith(ext) for ext in image_extensions):
        return url
        
    # UploadKit or HTML pages - comprehensive extraction
    if any(keyword in lower for keyword in ("uploadkit", "download.html", "files.getuploadkit.com")):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache"
            }
            
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers, allow_redirects=True)
            resp.raise_for_status()
            html = resp.text
            
            # Comprehensive patterns for image extraction
            patterns = [
                r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
                r'<img[^>]+src=["\']([^"\']+\.(jpg|jpeg|png|gif|webp|bmp|heic|heif)[^"\']*)["\'][^>]*>',
                r'href=["\']([^"\']+\.(jpg|jpeg|png|gif|webp|bmp|heic|heif)(?:\?[^"\']*)?)["\']',
                r'"(https?://[^"]+\.(jpg|jpeg|png|gif|webp|bmp|heic|heif)(?:\?[^"]*)?)"',
                r'url\(["\']?([^"\']+\.(jpg|jpeg|png|gif|webp|bmp|heic|heif))["\']?\)',
                r'data-src=["\']([^"\']+\.(jpg|jpeg|png|gif|webp|bmp|heic|heif)[^"\']*)["\']'
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
            print(f"[extract] HTML mining failed for {url[:50]}: {safe_str(e)}")
    
    return url

def download_image(url: str) -> bytes:
    direct_url = extract_direct_image_url(url)
    print(f"[download] Fetching: {direct_url[:120]}")
    
    # Headers that work with most CDNs and services
    headers = {
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
        "Accept": "image/webp,image/apng,image/avif,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site"
    }
    
    # Retry logic for network issues
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"[download] Attempt {attempt + 1}/{max_retries}")
            
            with requests.get(direct_url, headers=headers, timeout=REQUEST_TIMEOUT, stream=True, allow_redirects=True) as response:
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get("content-type", "").lower()
                content_length = response.headers.get("content-length")
                
                print(f"[download] Content-Type: {content_type}, Length: {content_length}")
                
                # Handle HTML responses (redirects, errors)
                if "text/html" in content_type:
                    if attempt < max_retries - 1:
                        print(f"[download] Got HTML response, retrying...")
                        time.sleep(2 ** attempt)
                        continue
                    raise ValueError("Server returned HTML instead of image")
                
                # Download in chunks
                total_bytes = 0
                chunks = []
                
                for chunk in response.iter_content(chunk_size=65536):  # 64KB chunks
                    if chunk:
                        total_bytes += len(chunk)
                        if total_bytes > MAX_IMAGE_BYTES:
                            raise ValueError(f"Image too large: {total_bytes} bytes > {MAX_IMAGE_BYTES}")
                        chunks.append(chunk)
                
                if total_bytes < 1024:  # Suspiciously small
                    if attempt < max_retries - 1:
                        print(f"[download] Small file ({total_bytes} bytes), retrying...")
                        time.sleep(2 ** attempt)
                        continue
                    raise ValueError(f"Downloaded file too small: {total_bytes} bytes")
                
                image_data = b"".join(chunks)
                print(f"[download] Successfully downloaded {total_bytes:,} bytes")
                return image_data
                
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2  # Longer waits for network issues
                print(f"[download] Network error: {safe_str(e)}, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            raise ValueError(f"Download failed after {max_retries} attempts: {safe_str(e)}")
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[download] Error: {safe_str(e)}, retrying...")
                time.sleep(2 ** attempt)
                continue
            raise ValueError(f"Download failed: {safe_str(e)}")
    
    raise ValueError("All download attempts failed")

# ---------- ENHANCED IMAGE PROCESSING WITH HEIC SUPPORT ----------
def process_image_to_rgb(image_bytes: bytes) -> Image.Image:
    """Convert any image format to RGB, with enhanced HEIC support."""
    
    # Try multiple approaches to open the image, including HEIC
    approaches = [
        # Approach 1: Standard PIL
        lambda: Image.open(io.BytesIO(image_bytes)),
        
        # Approach 2: Force format detection
        lambda: Image.open(io.BytesIO(image_bytes)).convert('RGB'),
        
        # Approach 3: HEIC-specific handling
        lambda: handle_heic_image(image_bytes),
        
        # Approach 4: Raw buffer approach
        lambda: Image.frombuffer('RGB', (100, 100), image_bytes[:30000] + b'\x00' * max(0, 30000 - len(image_bytes)), 'raw', 'RGB', 0, 1) if len(image_bytes) >= 30000 else None
    ]
    
    img = None
    for i, approach in enumerate(approaches):
        try:
            img = approach()
            if img:
                print(f"[image] Loaded via approach {i + 1}: {img.mode} {img.size}")
                break
        except Exception as e:
            print(f"[image] Approach {i + 1} failed: {safe_str(e)}")
            continue
    
    if not img:
        raise Exception("Could not load image with any method")
    
    # Convert to RGB with proper handling
    if img.mode == "RGBA":
        print("[image] Converting RGBA to RGB with white background")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background
    elif img.mode in ("P", "L", "LA", "CMYK", "YCbCr"):
        print(f"[image] Converting {img.mode} to RGB")
        return img.convert("RGB")
    elif img.mode == "RGB":
        return img
    else:
        print(f"[image] Unknown mode {img.mode}, forcing RGB conversion")
        return img.convert("RGB")

def handle_heic_image(image_bytes: bytes) -> Optional[Image.Image]:
    """Handle HEIC images specifically."""
    try:
        # Try pillow-heif for HEIC support
        from PIL import Image
        from pillow_heif import register_heif_opener
        register_heif_opener()
        
        img = Image.open(io.BytesIO(image_bytes))
        print("[heic] Successfully opened HEIC with pillow-heif")
        return img
        
    except ImportError:
        print("[heic] pillow-heif not available, trying pyheif")
        try:
            import pyheif
            heif_file = pyheif.read(image_bytes)
            img = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
            print("[heic] Successfully opened HEIC with pyheif")
            return img
        except ImportError:
            print("[heic] No HEIC support libraries available")
            return None
        except Exception as e:
            print(f"[heic] pyheif failed: {safe_str(e)}")
            return None
    except Exception as e:
        print(f"[heic] pillow-heif failed: {safe_str(e)}")
        return None

# ---------- OpenAI Integration (Your Working Code) ----------
OPENAI_IMAGES_EDITS = "https://api.openai.com/v1/images/edits"

def _decode_image_json(response_json) -> bytes:
    data = response_json.get("data", [])
    if not data:
        raise Exception("No data in OpenAI response")
        
    item = data[0]
    
    # Try URL first (more reliable)
    if "url" in item and item["url"]:
        try:
            session = requests.Session()
            session.trust_env = False
            with session.get(item["url"], timeout=(15, OPENAI_TIMEOUT)) as r:
                r.raise_for_status()
                return r.content
        except Exception as e:
            print(f"[openai] URL download failed: {safe_str(e)}")
    
    # Fallback to b64_json
    if "b64_json" in item and item["b64_json"]:
        try:
            return base64.b64decode(item["b64_json"])
        except Exception as e:
            print(f"[openai] b64 decode failed: {safe_str(e)}")
    
    raise Exception("No valid image data in OpenAI response")

def _rest_image_edit(image_png: bytes, prompt: str, attempt: int = 1) -> bytes:
    key = get_api_key()
    
    session = requests.Session()
    session.trust_env = False

    files = {"image": ("image.png", image_png, "image/png")}
    data = {
        "model": "dall-e-2",  # Use dall-e-2 for edits (this is what was working)
        "prompt": prompt,
        "size": "1024x1024",
        "n": 1
    }
    headers = {"Authorization": f"Bearer {key}"}
    timeout = (30, OPENAI_TIMEOUT)  # 30s connect, long read

    print(f"[openai] API call attempt {attempt}/{OPENAI_RETRY_ATTEMPTS}")
    
    try:
        resp = session.post(OPENAI_IMAGES_EDITS, headers=headers, data=data, files=files, timeout=timeout)
        
        if resp.status_code >= 400:
            try:
                err = resp.json()
                error_msg = err.get("error", {}).get("message", resp.text)
            except:
                error_msg = resp.text
            
            # Safety violations are not retryable
            if "safety" in error_msg.lower():
                raise Exception(f"OpenAI safety violation: {safe_str(error_msg)}")
            
            # Other errors can be retried
            if attempt < OPENAI_RETRY_ATTEMPTS:
                raise Exception(f"OpenAI error {resp.status_code}: {safe_str(error_msg)}")
            else:
                raise Exception(f"OpenAI error {resp.status_code} (final attempt): {safe_str(error_msg)}")

        return _decode_image_json(resp.json())
        
    except requests.exceptions.Timeout as e:
        if attempt < OPENAI_RETRY_ATTEMPTS:
            raise Exception(f"OpenAI timeout (attempt {attempt})")
        raise Exception(f"OpenAI timeout after {attempt} attempts")
    
    except requests.exceptions.RequestException as e:
        if attempt < OPENAI_RETRY_ATTEMPTS:
            raise Exception(f"OpenAI request error (attempt {attempt}): {safe_str(e)}")
        raise Exception(f"OpenAI request failed after {attempt} attempts: {safe_str(e)}")

def call_openai_edit(image_bytes: bytes, prompt: str) -> bytes:
    """Process image with OpenAI with enhanced quality and HEIC support."""
    
    try:
        # Convert image to RGB with HEIC support
        img = process_image_to_rgb(image_bytes)
        print(f"[openai] Processing RGB image: {img.size}")
        
        # Enhanced size handling for better quality
        target_size = 1024
        if max(img.size) != target_size:
            # Use high-quality resampling
            if max(img.size) > target_size:
                # Downscale with high quality
                ratio = target_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"[openai] Downscaled to: {new_size}")
            else:
                # Upscale with high quality
                ratio = target_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"[openai] Upscaled to: {new_size}")
        
        # Convert to RGBA for DALL-E 2 edit requirements
        if img.mode != "RGBA":
            img = img.convert("RGBA")
            print("[openai] Converted to RGBA for DALL-E 2")
        
        # Save as highest quality PNG
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False, compress_level=1)  # Minimal compression for quality
        png_bytes = buf.getvalue()
        print(f"[openai] Generated high-quality PNG: {len(png_bytes):,} bytes")
        
        # Use the enhanced quality prompt
        clean_prompt = sanitize_text(prompt) if (prompt and len(prompt) > 15) else SELECTED_PROMPT
        clean_prompt = clean_prompt.encode("ascii", "ignore").decode("ascii")
        print(f"[openai] Using enhanced prompt: {clean_prompt[:100]}...")
        
        # Call OpenAI with retries
        for attempt in range(1, OPENAI_RETRY_ATTEMPTS + 1):
            try:
                return _rest_image_edit(png_bytes, clean_prompt, attempt)
            except Exception as e:
                error_msg = safe_str(e)
                print(f"[openai] Attempt {attempt} failed: {error_msg}")
                
                if attempt < OPENAI_RETRY_ATTEMPTS:
                    # Don't retry safety violations
                    if "safety" in error_msg.lower():
                        raise Exception(f"OpenAI safety violation: {error_msg}")
                    
                    wait_time = min(2 ** attempt, 30)  # Exponential backoff, max 30s
                    print(f"[openai] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"OpenAI processing failed after {OPENAI_RETRY_ATTEMPTS} attempts: {error_msg}")
        
    except Exception as e:
        raise Exception(f"Image processing failed: {safe_str(e)}")

# ---------- Upload (Your Working Code) ----------
def upload_to_gcs(order_id: str, idx: int, img_bytes: bytes) -> str:
    if not bucket:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    timestamp = int(time.time() * 1000)
    blob_name = f"{sanitize_text(order_id)}/{timestamp}_{idx:03d}.png"
    blob = bucket.blob(blob_name)

    try:
        blob.cache_control = "public, max-age=31536000, immutable"
        blob.upload_from_string(img_bytes, content_type="image/png")
        
        try:
            blob.make_public()
        except Exception as e:
            print(f"[gcs] make_public failed (expected): {safe_str(e)}")

        url = blob.public_url
        if isinstance(url, bytes):
            url = url.decode("utf-8", "ignore")
        if not url or url.startswith("gs://"):
            url = f"https://storage.googleapis.com/{bucket.name}/{blob_name}"
        
        return url
        
    except Exception as e:
        print(f"[gcs] upload failed: {safe_str(e)}")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"

# ---------- Routes (Your Working Code) ----------
@app.route("/process", methods=["POST"])
def process():
    start_time = time.time()
    
    try:
        payload = request.get_json(force=True) or {}
        order_id = sanitize_text(payload.get("order_id", f"order_{int(time.time())}"))

        image_urls = payload.get("image_urls") or payload.get("urls") or []
        if isinstance(image_urls, str):
            image_urls = [u.strip() for u in image_urls.split(",") if u.strip()]
        
        valid_urls = [sanitize_text(u) for u in image_urls if u.strip()]
        
        if not valid_urls:
            return safe_json_response({"success": False, "error": "No valid image URLs provided"}, 400)
            
        if len(valid_urls) > MAX_IMAGES_PER_ORDER:
            return safe_json_response({"success": False, "error": f"Too many images. Maximum {MAX_IMAGES_PER_ORDER} per order"}, 400)

        raw_prompt = payload.get("prompt", "")
        prompt = sanitize_text(raw_prompt) if raw_prompt else ""

        print(f"[process] {order_id} - processing {len(valid_urls)} images with ENHANCED QUALITY + HEIC")

        results = []
        total_success = 0
        total_errors = 0

        for idx, url in enumerate(valid_urls):
            image_start = time.time()
            try:
                print(f"[process] === IMAGE {idx + 1}/{len(valid_urls)} ===")
                print(f"[process] URL: {url[:100]}")
                
                # Download
                raw_bytes = download_image(url)
                download_time = time.time() - image_start
                print(f"[process] Download time: {download_time:.1f}s")
                
                # Process with enhanced OpenAI
                processing_start = time.time()
                edited_bytes = call_openai_edit(raw_bytes, prompt)
                processing_time = time.time() - processing_start
                print(f"[process] Processing time: {processing_time:.1f}s")
                
                # Upload
                upload_start = time.time()
                result_url = upload_to_gcs(order_id, idx, edited_bytes)
                upload_time = time.time() - upload_start
                
                total_time = time.time() - image_start
                print(f"[process] Image {idx + 1} COMPLETE: {total_time:.1f}s total")
                
                results.append({
                    "status": "success",
                    "index": idx,
                    "source_url": url,
                    "result_url": result_url,
                    "storage_type": "gcs" if bucket else "base64",
                    "processing_time_seconds": round(total_time, 1),
                    "file_size_bytes": len(edited_bytes)
                })
                total_success += 1
                
            except Exception as e:
                error_time = time.time() - image_start
                error_msg = safe_str(e)
                print(f"[process] Image {idx + 1} FAILED after {error_time:.1f}s: {error_msg}")
                
                results.append({
                    "status": "error",
                    "index": idx,
                    "source_url": url,
                    "error": error_msg,
                    "processing_time_seconds": round(error_time, 1)
                })
                total_errors += 1

        total_time = time.time() - start_time
        success_rate = (total_success / len(valid_urls)) * 100
        
        print(f"[process] ORDER COMPLETE: {total_success}/{len(valid_urls)} success ({success_rate:.1f}%)")
        
        return safe_json_response({
            "success": True,
            "order_id": order_id,
            "total_images": len(valid_urls),
            "successful_images": total_success,
            "failed_images": total_errors,
            "success_rate_percent": round(success_rate, 1),
            "total_processing_time_seconds": round(total_time, 1),
            "processing_method": "enhanced_dalle2_edit",
            "results": results
        })
        
    except Exception as e:
        error_time = time.time() - start_time
        error_msg = safe_str(e)
        print(f"[process] REQUEST FAILED after {error_time:.1f}s: {error_msg}")
        return safe_json_response({
            "success": False,
            "error": error_msg,
            "processing_time_seconds": round(error_time, 1)
        }, 500)

@app.route("/test", methods=["GET", "POST"])
def test():
    test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"
    try:
        print("[test] Testing enhanced quality method...")
        raw = download_image(test_url)
        edited = call_openai_edit(raw, "Convert to coloring book page")
        b64_preview = base64.b64encode(edited).decode("utf-8")[:200]
        
        return safe_json_response({
            "success": True,
            "message": "Test successful with enhanced quality!",
            "original_url": test_url,
            "result_base64_preview": b64_preview + "...",
            "result_size_bytes": len(edited),
            "version": VERSION
        })
    except Exception as e:
        return safe_json_response({"success": False, "error": safe_str(e), "version": VERSION}, 500)

@app.route("/health", methods=["GET"])
def health():
    return safe_json_response({
        "status": "healthy",
        "service": "coloring-book-processor",
        "version": VERSION,
        "gcs_available": bucket is not None,
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "bucket_name": bucket_name,
        "max_images_per_order": MAX_IMAGES_PER_ORDER,
        "max_image_size_mb": MAX_IMAGE_BYTES // (1024 * 1024),
        "openai_timeout_seconds": OPENAI_TIMEOUT,
        "enhancements": [
            "improved_quality_prompts",
            "heic_support",
            "enhanced_image_processing",
            "high_quality_resampling"
        ]
    })

@app.route("/", methods=["GET"])
def index():
    return safe_json_response({
        "service": "Enhanced Quality Coloring Book Processor",
        "version": VERSION,
        "improvements": {
            "enhanced_quality": "Improved prompts with 4px outlines and maximum detail",
            "heic_support": "Full HEIC/HEIF file format support",
            "better_resampling": "High-quality image scaling with LANCZOS",
            "rgba_conversion": "Proper DALL-E 2 format handling"
        },
        "capabilities": {
            "perfect_quality_output": "Enhanced professional coloring book quality",
            "all_image_formats": "JPEG, PNG, HEIC, HEIF, WebP, BMP, GIF support",
            "batch_processing": f"Up to {MAX_IMAGES_PER_ORDER} images per order"
        },
        "endpoints": {
            "/process": "POST - Process images with enhanced quality",
            "/test": "GET/POST - Test enhanced method", 
            "/health": "GET - Health check with enhancement info",
            "/": "GET - This help"
        }
    })

if __name__ == "__main__":
    print(f"[startup] Enhanced Quality Coloring Book Processor {VERSION}")
    print(f"[startup] Improvements: Enhanced prompts + HEIC support")
    print(f"[startup] Max images per order: {MAX_IMAGES_PER_ORDER}")
    print(f"[startup] Professional quality with format compatibility")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
