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

VERSION = "cbp-v1.1-heic-only-fix"

# --- Nuke any proxy env that could interfere ---
for _k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy",
           "OPENAI_PROXY","OPENAI_HTTP_PROXY","OPENAI_HTTPS_PROXY"):
    if os.environ.get(_k):
        print(f"[net] ignoring proxy env {_k}")
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

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

def get_api_key() -> str:
    raw = os.environ.get("OPENAI_API_KEY", "")
    key = sanitize_key(raw)
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return key

# ---------- Config ----------
bucket_name = os.environ.get("OUTPUT_BUCKET", "memory-books-output")
DEFAULT_PROMPT = (
    "Transform this photograph into a professional adult coloring book illustration with these MANDATORY requirements: "
    "PRESERVE EVERY VISIBLE ELEMENT - people, faces, hair, clothing, jewelry, furniture, walls, ceiling, floors, windows, doors, signs, decorations, food, drinks, plants, vehicles, and ALL background objects. "
    "CONVERT photographic shadows, lighting, and dark areas into clear structural LINE ART ELEMENTS - NOT empty white space. "
    "CREATE bold consistent black OUTLINES ONLY (3-4 pixel width) throughout the ENTIRE image on pure white background. "
    "NO SHADING, NO CROSSHATCHING, NO DIAGONAL LINES, NO FILL PATTERNS - only clean black outlines on white. "
    "MAINTAIN exact facial expressions, eye shapes, smiles, and proportions with precise detail. "
    "RENDER background architecture as detailed line drawings - include ceiling details, wall textures, window frames, architectural elements. "
    "TRANSFORM all people in background into clear line art figures - do not eliminate them. "
    "CONVERT all objects, furniture, and environmental elements into detailed colorable line art sections with OUTLINES ONLY. "
    "ENSURE every area has clear, closed black outlines with NO INTERIOR SHADING - perfect for coloring with markers. "
    "GENERATE publication-quality adult coloring book page with maximum environmental context and rich detail throughout entire scene using OUTLINES ONLY."
)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
REQUEST_TIMEOUT = 30
OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "300"))  # seconds

# ---------- GCS client ----------
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    print(f"[init] Connected to GCS bucket: {bucket_name} | VERSION={VERSION}")
except Exception as e:
    print(f"[init] GCS not available: {safe_str(e)}")
    storage_client = None
    bucket = None

# ---------- Image helpers ----------
def extract_direct_image_url(url: str) -> str:
    url = sanitize_text(url)
    lower = url.lower()
    # ONLY CHANGE: Added HEIC support
    if lower.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif")):
        return url
    if "uploadkit" in lower or "download.html" in lower:
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "ColoringBookProcessor/1.0"})
            resp.raise_for_status()
            html = resp.text
            patterns = [
                r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
                r'<img[^>]+src=["\']([^"\']+)["\']',
                r'href=["\']([^"\']+\.(?:jpg|jpeg|png|gif|webp|heic|heif)[^"\']*)["\']',
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

def _decode_image_json(j) -> bytes:
    data = j.get("data") or []
    if not data:
        raise Exception(f"No data in response")
    item = data[0]
    if "b64_json" in item and item["b64_json"]:
        return base64.b64decode(item["b64_json"])
    if "url" in item and item["url"]:
        # Fetch the URL with env proxies disabled
        s = requests.Session()
        s.trust_env = False
        with s.get(item["url"], timeout=(10, OPENAI_TIMEOUT)) as r:
            r.raise_for_status()
            return r.content
    raise Exception("No image data found in response")

# ---------- OpenAI (REST, not SDK) ----------
OPENAI_IMAGES_EDITS = "https://api.openai.com/v1/images/edits"

def _rest_image_edit(image_png: bytes, prompt: str) -> bytes:
    key = get_api_key()

    # Disable reading proxy env
    s = requests.Session()
    s.trust_env = False  # ignore *_PROXY envs entirely

    files = {
        "image": ("image.png", image_png, "image/png"),
    }
    data = {
        "model": "gpt-image-1",
        "prompt": prompt,
        "size": "1024x1024",
    }
    headers = {
        "Authorization": f"Bearer {key}",
    }
    # 2-tuple timeout: (connect, read)
    tmo = (10, OPENAI_TIMEOUT)

    print("[openai-http] calling images/edits")
    resp = s.post(OPENAI_IMAGES_EDITS, headers=headers, data=data, files=files, timeout=tmo)
    if resp.status_code >= 400:
        try:
            err = resp.json()
        except Exception:
            err = {"error": {"message": resp.text}}
        raise Exception(f"OpenAI HTTP {resp.status_code}: {safe_str(err)}")

    return _decode_image_json(resp.json())

def call_openai_edit(image_bytes: bytes, prompt: str) -> bytes:
    """Convert image to line art using OpenAI, with robust HEIC support"""
    img = None
    
    # Enhanced HEIC detection - check multiple signatures
    is_heic = (
        image_bytes[4:12] == b'ftypheic' or 
        image_bytes[4:12] == b'ftypheif' or
        image_bytes[4:12] == b'ftypmif1' or  # HEIF variant
        b'heic' in image_bytes[:32].lower() or
        b'heif' in image_bytes[:32].lower()
    )
    
    print(f"[image] Processing image, size: {len(image_bytes)} bytes, HEIC detected: {is_heic}")
    
    # Try to open the image with HEIC support first
    if is_heic:
        print("[heic] HEIC/HEIF file detected, processing...")
        try:
            # Import and register HEIF support
            import pillow_heif
            pillow_heif.register_heif_opener()
            
            # Open the HEIC file
            img = Image.open(io.BytesIO(image_bytes))
            print(f"[heic] Successfully opened HEIC file: {img.size[0]}x{img.size[1]} pixels, mode: {img.mode}")
            
        except ImportError as e:
            print(f"[heic] pillow-heif not available: {safe_str(e)}")
            raise ValueError(
                "HEIC files require special processing libraries that are not available. "
                "Please convert your HEIC image to JPG format and try again."
            )
        except Exception as e:
            print(f"[heic] Failed to process HEIC file: {safe_str(e)}")
            raise ValueError(f"Unable to process HEIC file: {safe_str(e)}")
    
    # If not HEIC or HEIC processing failed, try standard PIL
    if not img:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            print(f"[image] Opened standard format: {img.format if hasattr(img, 'format') else 'Unknown'} ({img.size[0]}x{img.size[1]})")
        except Exception as e:
            print(f"[image] Failed to open image with PIL: {safe_str(e)}")
            raise ValueError(f"Unable to process image file: {safe_str(e)}")
    
    # Ensure we have a valid image
    if not img:
        raise ValueError("Failed to load image - unsupported format or corrupted file")
    
    # Convert to RGB for OpenAI processing (your exact original logic)
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
        return _rest_image_edit(buf.getvalue(), clean_prompt)
    except Exception as e:
        print(f"[openai] primary failed: {safe_str(e)}; retrying with minimal prompt")
        try:
            return _rest_image_edit(buf.getvalue(), "line art coloring page")
        except Exception as e2:
            raise Exception(f"Image processing failed: {safe_str(e2)}")

# ---------- Upload helper (public URL) ----------
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
        blob.make_public()  # if PAP/UBLA prevents, URL still works for signed URLs (not used here)
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
