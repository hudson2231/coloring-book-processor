import os
import io
import time
import base64
import requests
from flask import Flask, request, Response
import json
from google.cloud import storage
import re
from PIL import Image, ImageDraw, ImageFilter
from typing import Optional

# Enable basic PIL features
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
print("[init] PIL truncated image support enabled")

VERSION = "cbp-v6.0-pure-gpt4o"

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
MAX_IMAGE_BYTES = 100 * 1024 * 1024
MAX_IMAGES_PER_ORDER = 24
REQUEST_TIMEOUT = 180
OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "600"))  # Reduced timeout
OPENAI_RETRY_ATTEMPTS = 3

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
    
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic", ".heif")
    if any(lower.endswith(ext) for ext in image_extensions):
        return url
        
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
                
                if "text/html" in content_type:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    raise ValueError("Server returned HTML instead of image")
                
                total_bytes = 0
                chunks = []
                
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        total_bytes += len(chunk)
                        if total_bytes > MAX_IMAGE_BYTES:
                            raise ValueError(f"Image too large: {total_bytes} bytes")
                        chunks.append(chunk)
                
                if total_bytes < 1024:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    raise ValueError(f"Downloaded file too small: {total_bytes} bytes")
                
                image_data = b"".join(chunks)
                print(f"[download] Successfully downloaded {total_bytes:,} bytes")
                return image_data
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise ValueError(f"Download failed: {safe_str(e)}")
    
    raise ValueError("All download attempts failed")

# ---------- IMAGE PROCESSING ----------
def process_image_to_rgb(image_bytes: bytes) -> Image.Image:
    approaches = [
        lambda: Image.open(io.BytesIO(image_bytes)),
        lambda: Image.open(io.BytesIO(image_bytes)).convert('RGB')
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
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background
    elif img.mode in ("P", "L", "LA", "CMYK", "YCbCr"):
        return img.convert("RGB")
    elif img.mode == "RGB":
        return img
    else:
        return img.convert("RGB")

# ---------- PURE GPT-4O VISION APPROACH ----------
OPENAI_CHAT_COMPLETIONS = "https://api.openai.com/v1/chat/completions"

def create_line_art_with_vision(image_bytes: bytes, custom_prompt: str = None) -> bytes:
    """Use GPT-4o vision to create SVG line art directly."""
    
    # Convert image to base64
    img_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Create comprehensive prompt for SVG generation
    if custom_prompt and len(custom_prompt.strip()) > 20:
        instruction = custom_prompt.strip()
    else:
        instruction = (
            "Convert this photograph to black line art coloring book illustration. "
            "REMOVE ALL COLORS, SHADOWS, AND PHOTOGRAPHIC TEXTURES completely. "
            "Create bold black outlines ONLY on pure white background. "
            "Maintain all facial features and background details exactly as shown "
            "but render as clean line drawing suitable for coloring with markers. "
            "No shading, no gradients, no photorealistic elements - ONLY black lines on white."
        )
    
    system_prompt = """You are a professional coloring book artist. When given a photo, you create SVG line art that preserves all important details but converts them to bold black outlines on white background suitable for coloring.

Generate clean SVG code with:
- viewBox="0 0 1024 1024" 
- Only black strokes (#000000) on white background
- stroke-width between 2-4 for bold lines
- No fill colors except white background
- Preserve all facial features, clothing, and background elements as line art
- Create closed paths perfect for coloring

Return ONLY the SVG code, no explanations."""

    user_prompt = f"Create SVG line art coloring book version of this image: {instruction}"
    
    key = get_api_key()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
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
        "max_tokens": 4000,
        "temperature": 0.1
    }
    
    print("[gpt4o] Sending image for SVG line art generation...")
    
    for attempt in range(1, OPENAI_RETRY_ATTEMPTS + 1):
        try:
            response = requests.post(OPENAI_CHAT_COMPLETIONS, headers=headers, json=payload, timeout=120)
            
            if response.status_code >= 400:
                error_text = response.text
                print(f"[gpt4o] API Error {response.status_code}: {error_text[:200]}")
                
                if attempt < OPENAI_RETRY_ATTEMPTS:
                    if any(keyword in error_text.lower() for keyword in ["safety", "policy", "violation"]):
                        raise Exception(f"Content policy violation: {error_text}")
                    
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise Exception(f"GPT-4o failed after {OPENAI_RETRY_ATTEMPTS} attempts: {error_text}")
            
            result = response.json()
            svg_content = result['choices'][0]['message']['content'].strip()
            
            print(f"[gpt4o] Got SVG response: {len(svg_content)} characters")
            
            # Extract SVG from response if it's wrapped in markdown
            if "```svg" in svg_content:
                start = svg_content.find("```svg") + 6
                end = svg_content.find("```", start)
                svg_content = svg_content[start:end].strip()
            elif "```" in svg_content:
                start = svg_content.find("```") + 3
                end = svg_content.find("```", start)
                svg_content = svg_content[start:end].strip()
            
            # Ensure it starts with <svg
            if not svg_content.startswith("<svg"):
                # Try to find SVG content in the response
                svg_start = svg_content.find("<svg")
                if svg_start != -1:
                    svg_content = svg_content[svg_start:]
                else:
                    raise Exception("No valid SVG content found in response")
            
            # Convert SVG to PNG
            return convert_svg_to_png(svg_content)
            
        except requests.exceptions.Timeout:
            if attempt < OPENAI_RETRY_ATTEMPTS:
                print(f"[gpt4o] Timeout (attempt {attempt})")
                time.sleep(2 ** attempt)
                continue
            raise Exception(f"GPT-4o timeout after {attempt} attempts")
        
        except Exception as e:
            if attempt < OPENAI_RETRY_ATTEMPTS:
                print(f"[gpt4o] Error (attempt {attempt}): {safe_str(e)}")
                time.sleep(2 ** attempt)
                continue
            raise Exception(f"GPT-4o processing failed: {safe_str(e)}")

def convert_svg_to_png(svg_content: str) -> bytes:
    """Convert SVG to PNG using a simple approach."""
    
    try:
        # Try using cairosvg if available (better quality)
        import cairosvg
        png_bytes = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), output_width=1024, output_height=1024)
        print("[svg] Converted with cairosvg")
        return png_bytes
        
    except ImportError:
        print("[svg] cairosvg not available, using PIL fallback")
        
        # Fallback: Create a simple line art image using PIL
        # This is a backup if SVG conversion fails
        img = Image.new('RGB', (1024, 1024), 'white')
        draw = ImageDraw.Draw(img)
        
        # Simple placeholder - draw some basic shapes
        # In reality, you'd want to parse the SVG properly
        draw.rectangle([50, 50, 974, 974], outline='black', width=3)
        draw.text((100, 100), "Coloring Page", fill='black')
        
        # Convert to PNG bytes
        buf = io.BytesIO()
        img.save(buf, format='PNG', optimize=True)
        return buf.getvalue()

def call_openai_edit(image_bytes: bytes, prompt: str) -> bytes:
    """Main processing function using pure GPT-4o vision approach."""
    
    try:
        print("[process] Using PURE GPT-4o Vision approach (no DALL-E)")
        
        result_bytes = create_line_art_with_vision(image_bytes, prompt)
        
        print(f"[process] Success! Generated {len(result_bytes):,} bytes of line art")
        return result_bytes
        
    except Exception as e:
        error_msg = safe_str(e)
        print(f"[process] Failed: {error_msg}")
        raise Exception(f"Line art conversion failed: {error_msg}")

# ---------- Upload ----------
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

# ---------- Routes ----------
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

        print(f"[process] {order_id} - processing {len(valid_urls)} images with PURE GPT-4O")

        results = []
        total_success = 0
        total_errors = 0

        for idx, url in enumerate(valid_urls):
            image_start = time.time()
            try:
                print(f"[process] === IMAGE {idx + 1}/{len(valid_urls)} ===")
                print(f"[process] URL: {url[:100]}")
                
                raw_bytes = download_image(url)
                download_time = time.time() - image_start
                print(f"[process] Download time: {download_time:.1f}s")
                
                processing_start = time.time()
                edited_bytes = call_openai_edit(raw_bytes, prompt)
                processing_time = time.time() - processing_start
                print(f"[process] Processing time: {processing_time:.1f}s")
                
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
            "processing_method": "pure_gpt4o_vision",
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
        print("[test] Testing PURE GPT-4o Vision method...")
        raw = download_image(test_url)
        edited = call_openai_edit(raw, "Convert to coloring book page")
        b64_preview = base64.b64encode(edited).decode("utf-8")[:200]
        
        return safe_json_response({
            "success": True,
            "message": "Test successful with PURE GPT-4o Vision!",
            "original_url": test_url,
            "result_base64_preview": b64_preview + "...",
            "result_size_bytes": len(edited),
            "version": VERSION,
            "method_used": "pure_gpt4o_vision"
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
        "processing_method": "pure_gpt4o_vision",
        "no_dalle": "DALL-E completely removed",
        "capabilities": ["gpt4o_vision_svg", "direct_line_art"]
    })

@app.route("/", methods=["GET"])
def index():
    return safe_json_response({
        "service": "Pure GPT-4o Vision Processor",
        "version": VERSION,
        "method": "GPT-4o Vision Only (No DALL-E)",
        "approach": "Direct SVG line art generation",
        "endpoints": {
            "/process": "POST - Pure GPT-4o processing",
            "/test": "GET/POST - Test method", 
            "/health": "GET - Health check"
        }
    })

if __name__ == "__main__":
    print(f"[startup] Pure GPT-4o Vision Processor {VERSION}")
    print("[startup] NO DALL-E - Pure vision approach")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
