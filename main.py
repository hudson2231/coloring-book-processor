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

VERSION = "cbp-v5.3-fixed-gpt-image-1"

# --- Nuke any proxy env that could interfere ---
for _k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy",
           "OPENAI_PROXY","OPENAI_HTTP_PROXY","OPENAI_HTTPS_PROXY"):
    if os.environ.get(_k):
        print(f"[net] ignoring proxy env {_k}")
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

app = Flask(__name__)

# ---------- BUSINESS CRITICAL CONFIG ----------
bucket_name = os.environ.get("OUTPUT_BUCKET", "memory-books-output")
MAX_IMAGE_BYTES = 100 * 1024 * 1024  # 100MB for any format
MAX_IMAGES_PER_ORDER = 24
REQUEST_TIMEOUT = 180
OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "1200"))
OPENAI_RETRY_ATTEMPTS = 3

# ---------- Your Perfect Prompts (from your original code) ----------
PERFECT_PROMPTS = {
    "ULTIMATE_QUALITY": (
        "Convert to professional adult coloring book line art with these EXACT specifications: "
        "BOLD 3-pixel consistent black outlines throughout entire image on pure white background. "
        "PRESERVE facial features with perfect accuracy - maintain exact expressions, eye shape, smile, proportions. "
        "CAPTURE every small detail: jewelry chains, necklaces, earrings, clothing textures, hair definition. "
        "INTERPRET dark/shadowy background areas as clear structural line elements - ceiling details, wall features, other people as line drawings. "
        "ENSURE all lines are SHARP and CRISP - no soft, blurry, or faded outlines anywhere. "
        "CONVERT photographic lighting and shadows into drawable line art elements - not darkness. "
        "MAINTAIN rich environmental context with clear line-drawn background elements. "
        "CREATE closed, continuous outlines perfect for coloring with markers or colored pencils. "
        "RENDER as hand-drawn professional coloring book illustration quality."
    ),
    
    "LINE_ART_FOCUSED": (
        "Convert this photograph to black line art coloring book illustration. "
        "REMOVE ALL COLORS, SHADOWS, AND PHOTOGRAPHIC TEXTURES completely. "
        "Create bold black outlines ONLY on pure white background. "
        "Maintain all facial features and background details exactly as shown "
        "but render as clean line drawing suitable for coloring with markers. "
        "No shading, no gradients, no photorealistic elements - ONLY black lines on white."
    )
}

SELECTED_PROMPT = PERFECT_PROMPTS["LINE_ART_FOCUSED"]

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

# ---------- IMAGE DOWNLOAD (Your working code) ----------
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
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"[download] Attempt {attempt + 1}/{max_retries}")
            
            with requests.get(direct_url, headers=headers, timeout=REQUEST_TIMEOUT, stream=True, allow_redirects=True) as response:
                response.raise_for_status()
                
                content_type = response.headers.get("content-type", "").lower()
                content_length = response.headers.get("content-length")
                
                print(f"[download] Content-Type: {content_type}, Length: {content_length}")
                
                if "text/html" in content_type:
                    if attempt < max_retries - 1:
                        print(f"[download] Got HTML response, retrying...")
                        time.sleep(2 ** attempt)
                        continue
                    raise ValueError("Server returned HTML instead of image")
                
                total_bytes = 0
                chunks = []
                
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        total_bytes += len(chunk)
                        if total_bytes > MAX_IMAGE_BYTES:
                            raise ValueError(f"Image too large: {total_bytes} bytes > {MAX_IMAGE_BYTES}")
                        chunks.append(chunk)
                
                if total_bytes < 1024:
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
                wait_time = (2 ** attempt) * 2
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

# ---------- IMAGE PROCESSING (Your working code) ----------
def process_image_to_rgb(image_bytes: bytes) -> Image.Image:
    approaches = [
        lambda: Image.open(io.BytesIO(image_bytes)),
        lambda: Image.open(io.BytesIO(image_bytes)).convert('RGB'),
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

# ---------- FIXED GPT-IMAGE-1 APPROACH ----------
OPENAI_CHAT_COMPLETIONS = "https://api.openai.com/v1/chat/completions"

def call_openai_edit(image_bytes: bytes, prompt: str) -> bytes:
    """Use GPT-4o with image input to generate coloring book via chat completions."""
    
    try:
        print("[gpt-image-1] Processing with GPT-4o chat completions (image input)")
        
        # Convert image to base64
        img_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Use your perfect prompt
        clean_prompt = sanitize_text(prompt) if (prompt and len(prompt) > 15) else SELECTED_PROMPT
        
        # Combine image analysis with generation request
        system_prompt = (
            "You are a professional coloring book artist. When given a photo, you generate a detailed "
            "coloring book line art version that preserves all important details but converts them to "
            "bold black outlines on white background suitable for coloring."
        )
        
        user_prompt = f"Please create a coloring book line art version of this image: {clean_prompt}"
        
        key = get_api_key()
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system", 
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        
        print(f"[gpt-image-1] Sending request with prompt: {clean_prompt[:100]}...")
        
        # Make the API call with retries
        for attempt in range(1, OPENAI_RETRY_ATTEMPTS + 1):
            try:
                response = requests.post(OPENAI_CHAT_COMPLETIONS, headers=headers, json=payload, timeout=120)
                
                if response.status_code >= 400:
                    try:
                        err = response.json()
                        error_msg = err.get("error", {}).get("message", response.text)
                    except:
                        error_msg = response.text
                    
                    print(f"[gpt-image-1] API Error {response.status_code}: {error_msg}")
                    
                    if attempt < OPENAI_RETRY_ATTEMPTS:
                        if "safety" in error_msg.lower() or "policy" in error_msg.lower():
                            raise Exception(f"Content policy violation: {error_msg}")
                        
                        wait_time = min(2 ** attempt, 30)
                        print(f"[gpt-image-1] Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(f"GPT-4o request failed: {response.status_code} - {error_msg}")
                
                # Parse response
                result = response.json()
                description = result['choices'][0]['message']['content']
                
                print(f"[gpt-image-1] Got response: {description[:150]}...")
                
                # Now generate image using the standard images/generations endpoint
                return generate_image_from_description(description.strip())
                
            except requests.exceptions.Timeout as e:
                if attempt < OPENAI_RETRY_ATTEMPTS:
                    print(f"[gpt-image-1] Timeout (attempt {attempt})")
                    time.sleep(2 ** attempt)
                    continue
                raise Exception(f"GPT-4o timeout after {attempt} attempts")
            
            except requests.exceptions.RequestException as e:
                if attempt < OPENAI_RETRY_ATTEMPTS:
                    print(f"[gpt-image-1] Request error (attempt {attempt}): {safe_str(e)}")
                    time.sleep(2 ** attempt)
                    continue
                raise Exception(f"GPT-4o request failed after {attempt} attempts: {safe_str(e)}")
        
    except Exception as e:
        raise Exception(f"Coloring book conversion failed: {safe_str(e)}")

def generate_image_from_description(description: str) -> bytes:
    """Generate coloring book image using OpenAI images/generations."""
    
    key = get_api_key()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    # Create coloring book specific prompt
    generation_prompt = f"Professional adult coloring book line art illustration: {description}. Bold black outlines on pure white background, no colors, no shading, clean line drawing perfect for coloring with markers."
    
    # Truncate if too long
    if len(generation_prompt) > 1000:
        generation_prompt = generation_prompt[:997] + "..."
    
    payload = {
        "model": "dall-e-3",  # Use DALL-E 3 for generation (not editing)
        "prompt": generation_prompt,
        "size": "1024x1024",
        "quality": "standard",
        "n": 1
    }
    
    print(f"[generate] Creating image with: {generation_prompt[:100]}...")
    
    response = requests.post("https://api.openai.com/v1/images/generations", 
                           headers=headers, json=payload, timeout=OPENAI_TIMEOUT)
    response.raise_for_status()
    
    result = response.json()
    image_url = result['data'][0]['url']
    
    # Download the generated image
    img_response = requests.get(image_url, timeout=60)
    img_response.raise_for_status()
    
    print(f"[generate] Successfully generated {len(img_response.content):,} bytes")
    return img_response.content

# ---------- Upload (Your working code) ----------
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

# ---------- Routes (Your working structure) ----------
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

        print(f"[process] {order_id} - processing {len(valid_urls)} images with FIXED GPT-IMAGE-1")

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
                
                # Process with FIXED METHOD
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
            "processing_method": "fixed_gpt_image_1",
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
        print("[test] Testing FIXED GPT-IMAGE-1 method...")
        raw = download_image(test_url)
        edited = call_openai_edit(raw, "Convert to coloring book page")
        b64_preview = base64.b64encode(edited).decode("utf-8")[:200]
        
        return safe_json_response({
            "success": True,
            "message": "Test successful with FIXED GPT-IMAGE-1!",
            "original_url": test_url,
            "result_base64_preview": b64_preview + "...",
            "result_size_bytes": len(edited),
            "version": VERSION,
            "method_used": "fixed_gpt_image_1"
        })
    except Exception as e:
        return safe_json_response({
            "success": False, 
            "error": safe_str(e), 
            "version": VERSION,
            "method_attempted": "fixed_gpt_image_1"
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
        "max_images_per_order": MAX_IMAGES_PER_ORDER,
        "processing_method": "fixed_gpt_image_1",
        "capabilities": [
            "gpt4o_vision_analysis",
            "dalle3_generation", 
            "your_perfect_prompts"
        ]
    })

@app.route("/", methods=["GET"])
def index():
    return safe_json_response({
        "service": "FIXED GPT-Image-1 Coloring Book Processor", 
        "version": VERSION,
        "processing_method": "GPT-4o Vision + DALL-E 3 Generation (Your Method Fixed)",
        "capabilities": {
            "your_perfect_prompts": "Using your proven coloring book prompts",
            "gpt4o_vision": "Analyzes photos with GPT-4o",
            "dalle3_generation": "Generates line art with DALL-E 3",
            "format_support": "All formats (HEIC, JPEG, PNG, etc.)",
            "batch_processing": f"Up to {MAX_IMAGES_PER_ORDER} images per order"
        },
        "endpoints": {
            "/process": "POST - Process images (YOUR METHOD FIXED)",
            "/test": "GET/POST - Test with sample", 
            "/health": "GET - Health check",
            "/": "GET - This help"
        }
    })

if __name__ == "__main__":
    print(f"[startup] FIXED GPT-Image-1 Coloring Book Processor {VERSION}")
    print(f"[startup] Using YOUR proven approach with API fixes")
    print(f"[startup] Method: GPT-4o Vision + DALL-E 3 Generation")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
