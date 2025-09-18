import os
import io
import time
import base64
import requests
from flask import Flask, request, Response
import json
from google.cloud import storage
from google.cloud import tasks_v2
import re
from PIL import Image, ImageDraw, ImageFont
import uuid
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

VERSION = "cbp-v1.27-country-fix"

# --- Nuke any proxy env that could interfere ---
for _k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy",
          "OPENAI_PROXY","OPENAI_HTTP_PROXY","OPENAI_HTTPS_PROXY"):
   if os.environ.get(_k):
       print(f"[net] ignoring proxy env {_k}")
       os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

app = Flask(__name__)

# Batch configuration
BATCH_SIZE = 8  # Process 8 images per batch to stay under timeout

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
# BALANCED PROMPT - PERFECT FACES WITH PROPER BACKGROUNDS
DEFAULT_PROMPT = (
   "Transform this photograph into a professional adult coloring book illustration with these MANDATORY requirements: "
   "CRITICAL COUNTING RULE: Count every person in the photo first. If photo shows 1 person, output must show 1 person. If photo shows 2 people, output must show exactly 2 people. Never add extra people or figures. "
   "FACE RECOGNITION RULE: Each face must remain recognizable. Preserve unique features through line placement - do not use generic cartoon faces. "
   "PRESERVE EVERY VISIBLE ELEMENT - maintain exact positions of people, faces, hair, clothing, furniture, and background objects as they appear in the photo. "
   "ENVIRONMENT RULE: Outdoor scenes with grass, fences, and sky must remain outdoor. Indoor scenes must remain indoor. Simple backgrounds must stay simple - empty grass is empty grass, not a crowd. "
   "CREATE bold consistent black OUTLINES ONLY (3-4 pixel width) throughout the ENTIRE image on pure white background. "
   "NO SHADING, NO CROSSHATCHING, NO DIAGONAL LINES, NO FILL PATTERNS - only clean black outlines on white. "
   "MAINTAIN exact facial expressions, eye shapes, smiles, and proportions with precise detail for ALL visible faces. "
   "BACKGROUND PRECISION: Include background elements that exist in the photo using simple line work. Grass = horizon line and texture hints. Sky = cloud outlines or boundary. Walls = edge lines. Fences = vertical lines. Do not leave backgrounds empty, but do not add elements not in the original photo. "
   "FORBIDDEN ADDITIONS: Do NOT add people in background, do NOT add furniture not present, do NOT add architectural elements not in photo. "
   "TECHNICAL EXECUTION: This is photographic line tracing, not artistic interpretation. Extract edges exactly as they appear. "
   "FINAL CHECK: Output must have same number of people, same setting (indoor/outdoor), and same objects as input photograph."
)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
REQUEST_TIMEOUT = 30
OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "300"))

# Lulu configuration
LULU_CLIENT_ID = os.environ.get('LULU_CLIENT_ID')
LULU_CLIENT_SECRET = os.environ.get('LULU_CLIENT_SECRET')
LULU_USE_SANDBOX = os.environ.get('LULU_USE_SANDBOX', 'true').lower() == 'true'

# ---------- GCS client ----------
try:
   storage_client = storage.Client()
   bucket = storage_client.bucket(bucket_name)
   print(f"[init] Connected to GCS bucket: {bucket_name} | VERSION={VERSION}")
except Exception as e:
   print(f"[init] GCS not available: {safe_str(e)}")
   storage_client = None
   bucket = None

# ---------- Helper for placeholder pages ----------
def create_placeholder_page(text="Image could not be processed"):
   """Create a simple coloring page with text when image fails"""
   img = Image.new('RGB', (1024, 1024), 'white')
   draw = ImageDraw.Draw(img)
   
   # Draw a decorative border
   draw.rectangle([50, 50, 974, 974], outline='black', width=3)
   draw.rectangle([70, 70, 954, 954], outline='black', width=2)
   
   # Add text in center
   text_bbox = draw.textbbox((0, 0), text)
   text_width = text_bbox[2] - text_bbox[0]
   text_height = text_bbox[3] - text_bbox[1]
   x = (1024 - text_width) // 2
   y = (1024 - text_height) // 2
   draw.text((x, y), text, fill='black')
   
   # Convert to bytes
   buffer = io.BytesIO()
   img.save(buffer, format='PNG')
   buffer.seek(0)
   return buffer.getvalue()

# ---------- Image helpers ----------
def extract_direct_image_url(url: str) -> str:
   url = sanitize_text(url)
   lower = url.lower()
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

def download_image_with_retry(url: str, max_retries: int = 3) -> bytes:
   """Download image with retry logic for network failures"""
   for attempt in range(max_retries):
       try:
           direct = extract_direct_image_url(url)
           print(f"[fetch] Attempt {attempt+1}: {direct[:120]}")
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
       except (requests.Timeout, requests.ConnectionError) as e:
           if attempt < max_retries - 1:
               wait_time = (attempt + 1) * 2  # Exponential backoff
               print(f"[fetch] Network error, retrying in {wait_time}s: {safe_str(e)}")
               time.sleep(wait_time)
           else:
               raise
       except Exception as e:
           raise  # Non-network errors, don't retry

def download_image(url: str) -> bytes:
   """Wrapper for backward compatibility"""
   return download_image_with_retry(url)

def _decode_image_json(j) -> bytes:
   data = j.get("data") or []
   if not data:
       raise Exception(f"No data in response")
   item = data[0]
   if "b64_json" in item and item["b64_json"]:
       return base64.b64decode(item["b64_json"])
   if "url" in item and item["url"]:
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
   s = requests.Session()
   s.trust_env = False
   
   files = {"image": ("image.png", image_png, "image/png")}
   data = {"model": "gpt-image-1", "prompt": prompt, "size": "1024x1024"}
   headers = {"Authorization": f"Bearer {key}"}
   
   print("[openai-http] calling images/edits")
   resp = s.post(OPENAI_IMAGES_EDITS, headers=headers, data=data, files=files, timeout=(10, OPENAI_TIMEOUT))
   
   if resp.status_code >= 400:
       try:
           err = resp.json()
       except Exception:
           err = {"error": {"message": resp.text}}
       raise Exception(f"OpenAI HTTP {resp.status_code}: {safe_str(err)}")
   
   return _decode_image_json(resp.json())

def call_openai_edit_with_retry(image_bytes: bytes, prompt: str, max_retries: int = 2) -> bytes:
   """Call OpenAI with retry logic and fallback for safety violations"""
   img = None
   
   is_heic = (
       image_bytes[4:12] == b'ftypheic' or 
       image_bytes[4:12] == b'ftypheif' or
       image_bytes[4:12] == b'ftypmif1' or
       b'heic' in image_bytes[:32].lower() or
       b'heif' in image_bytes[:32].lower()
   )
   
   print(f"[image] Processing image, size: {len(image_bytes)} bytes, HEIC detected: {is_heic}")
   
   if is_heic:
       print("[heic] HEIC/HEIF file detected, processing with pillow-heif...")
       try:
           from pillow_heif import register_heif_opener
           register_heif_opener()
           img = Image.open(io.BytesIO(image_bytes))
           print(f"[heic] Successfully opened HEIC file: {img.size[0]}x{img.size[1]} pixels")
       except ImportError:
           raise ValueError("HEIC support not available")
       except Exception as e:
           print(f"[heic] Failed to process HEIC file: {safe_str(e)}")
           raise ValueError(f"HEIC processing failed: {safe_str(e)}")
   
   if not img:
       try:
           img = Image.open(io.BytesIO(image_bytes))
           print(f"[image] Opened standard format: {img.format if hasattr(img, 'format') else 'Unknown'} ({img.size[0]}x{img.size[1]})")
       except Exception as e:
           print(f"[image] Failed to open image with PIL: {safe_str(e)}")
           raise ValueError(f"Unable to process image file: {safe_str(e)}")
   
   if not img:
       raise ValueError("Failed to load image - unsupported format or corrupted file")
   
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

   for attempt in range(max_retries):
       try:
           return _rest_image_edit(buf.getvalue(), clean_prompt)
       except Exception as e:
           error_str = str(e)
           
           # Check for safety violation
           if "safety_violations" in error_str or "moderation" in error_str:
               print(f"[openai] Safety violation detected, using placeholder")
               return create_placeholder_page("Image filtered - draw your own design!")
           
           # Check for rate limit
           elif "rate_limit" in error_str.lower():
               wait_time = min(30, (attempt + 1) * 5)
               print(f"[openai] Rate limit hit, waiting {wait_time}s")
               time.sleep(wait_time)
               continue
           
           # Try with minimal prompt on first failure
           elif attempt == 0:
               print(f"[openai] primary failed: {safe_str(e)}; retrying with minimal prompt")
               try:
                   return _rest_image_edit(buf.getvalue(), "line art coloring page")
               except Exception as e2:
                   if "safety_violations" in str(e2):
                       return create_placeholder_page("Image filtered - draw your own design!")
                   if attempt < max_retries - 1:
                       continue
                   raise Exception(f"Image processing failed: {safe_str(e2)}")
           else:
               raise Exception(f"Image processing failed after {attempt+1} attempts: {safe_str(e)}")

def call_openai_edit(image_bytes: bytes, prompt: str) -> bytes:
   """Wrapper for backward compatibility"""
   return call_openai_edit_with_retry(image_bytes, prompt)

# ---------- Upload helper (public URL) ----------
def upload_to_gcs(order_id: str, idx: int, img_bytes: bytes) -> str:
   if not bucket:
       b64 = base64.b64encode(img_bytes).decode("utf-8")
       return f"data:image/png;base64,{b64}"

   # Strip the # from Shopify order numbers
   order_id = order_id.replace("#", "")
   
   blob_name = f"{order_id}/{int(time.time())}_{idx}.png"
   blob = bucket.blob(blob_name)
   blob.cache_control = "public, max-age=31536000, immutable"
   blob.upload_from_string(img_bytes, content_type="image/png")
   
   # Always construct URL manually
   url = f"https://storage.googleapis.com/{bucket.name}/{blob_name}"
   return url

# ---------- Lulu Integration ----------
def get_lulu_token():
    """Get OAuth token from Lulu"""
    if not LULU_CLIENT_ID or not LULU_CLIENT_SECRET:
        raise Exception("Lulu credentials not configured")
    
    # Strip whitespace from credentials
    client_id = LULU_CLIENT_ID.strip()
    client_secret = LULU_CLIENT_SECRET.strip()
    
    response = requests.post(
        'https://api.lulu.com/auth/realms/glasstree/protocol/openid-connect/token',
        data={
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret
        },
        headers={
            'Content-Type': 'application/x-www-form-urlencoded'
        }
    )
    
    print(f"[lulu-auth] Token request status: {response.status_code}")
    if response.status_code != 200:
        print(f"[lulu-auth] Token request failed: {response.text}")
        raise Exception(f"Lulu auth failed: {response.text}")
    
    token_data = response.json()
    print(f"[lulu-auth] Token obtained successfully")
    return token_data['access_token']

@app.route("/send-to-lulu", methods=["POST"])
def send_to_lulu():
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Get data from Zapier
            data = request.get_json(force=True, silent=True) or {}
            
            # Parse the image URLs - handle both newline and comma separated
            image_urls_str = data.get('image_urls', '')
            if isinstance(image_urls_str, list):
                image_urls = [url.replace("#", "") for url in image_urls_str if url.startswith('http')]
            else:
                # Check for newlines first, then commas
                if '\n' in image_urls_str:
                    image_urls = [u.strip().replace("#", "") for u in image_urls_str.split('\n') 
                                 if u.strip() and u.strip().startswith('http')]
                else:
                    image_urls = [u.strip().replace("#", "") for u in image_urls_str.split(',') 
                                 if u.strip() and u.strip().startswith('http')]
            
            order_id = data.get('order_id', f"order_{int(time.time())}").replace("#", "")
            
            # Download and prepare images
            images = []
            failed_count = 0
            for url in image_urls:
                if url and not url.startswith("ERROR") and not url.startswith("SKIP"):
                    try:
                        img_bytes = download_image(url)
                        img = Image.open(io.BytesIO(img_bytes))
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        # Resize to 6x9 at 300 DPI
                        img = img.resize((1800, 2700), Image.Resampling.LANCZOS)
                        
                        # Save to bytes
                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format='PNG', dpi=(300, 300))
                        images.append(img_buffer.getvalue())
                    except Exception as e:
                        print(f"[lulu] Failed to process image {url}: {safe_str(e)}")
                        failed_count += 1
            
            if not images:
                return safe_json_response({"success": False, "error": "No valid images to process"}, 400)
            
            # Add blank pages to reach minimum page count (24 pages = 12 images)
            original_count = len(images)
            while len(images) < 12:
                # Create a blank white page
                blank = Image.new('RGB', (1800, 2700), 'white')
                blank_buffer = io.BytesIO()
                blank.save(blank_buffer, format='PNG', dpi=(300, 300))
                images.append(blank_buffer.getvalue())
                print(f"[lulu] Added blank page {len(images)}/12")
            
            # Create PDF with ReportLab for proper formatting
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=(6*72, 9*72))  # 6x9 inches at 72 points/inch
            
            for img_bytes in images:
                img_reader = ImageReader(io.BytesIO(img_bytes))
                c.drawImage(img_reader, 0, 0, width=6*72, height=9*72, preserveAspectRatio=True)
                c.showPage()
            
            c.save()
            pdf_bytes = pdf_buffer.getvalue()
            
            # Upload interior PDF to GCS and get URL
            pdf_blob_name = f"lulu_pdfs/{order_id}_{int(time.time())}.pdf"
            pdf_blob = bucket.blob(pdf_blob_name)
            pdf_blob.upload_from_string(pdf_bytes, content_type="application/pdf")
            pdf_url = f"https://storage.googleapis.com/{bucket.name}/{pdf_blob_name}"
            print(f"[lulu] Interior PDF uploaded: {pdf_url}")
            
            # USE STATIC COVER URL INSTEAD OF GENERATING
            cover_url = "https://storage.googleapis.com/memory-books-output/covers/standard_cover.pdf"
            print(f"[lulu] Using static cover: {cover_url}")
            
            # Page count should now match what we tell Lulu
            page_count = len(images) * 2  # Double-sided pages
            pod_package_id = '0600X0900BWSTDSS060UW444MXX'  # 6x9, B&W, Saddle Stitch, 60# White
            
            # Get Lulu token
            token = get_lulu_token()
            
            # Parse address - SIMPLIFIED SINCE WE NOW GET 2-LETTER CODES
            shipping_address = data.get('shipping_address', '')
            customer_name = data.get('customer_name', 'Customer')
            customer_email = data.get('customer_email', 'no-reply@example.com')
            
            # Log raw data for debugging
            print(f"[lulu] Raw shipping address: {shipping_address}")
            print(f"[lulu] Raw customer name: {customer_name}")
            print(f"[lulu] Raw customer email: {customer_email}")
            
            # Parse address (expecting: "Street, City, State_Code, ZIP, Country_Code")
            addr_parts = [p.strip() for p in shipping_address.split(',')]
            print(f"[lulu] Parsed address parts: {addr_parts}")
            
            # Ensure we have enough parts
            while len(addr_parts) < 5:
                addr_parts.append('')
            
            # Get country code - it's already a 2-letter code from Shopify
            country_code = addr_parts[4].strip().upper() if addr_parts[4] else 'US'
            
            # Validate it's 2 characters
            if len(country_code) != 2:
                print(f"[lulu] WARNING: Invalid country code '{country_code}', defaulting to US")
                country_code = 'US'
            
            # State code is already correct from Shopify
            state_code = addr_parts[2].strip().upper() if addr_parts[2] else 'CA'
            
            print(f"[lulu] Final country code: {country_code}")
            print(f"[lulu] Final state code: {state_code}")
            
            # Correct order structure with URLs
            order_data = {
                'external_id': order_id,
                'contact_email': customer_email or 'no-reply@example.com',  # Required field
                'line_items': [{
                    'title': f'Memory Book - {order_id}',
                    'quantity': 1,
                    'printable_normalization': {
                        'interior': {
                            'source_url': pdf_url  # URL instead of file
                        },
                        'cover': {
                            'source_url': cover_url  # URL instead of file
                        },
                        'pod_package_id': pod_package_id
                    }
                }],
                'shipping_address': {
                    'name': customer_name or 'Customer',
                    'street1': addr_parts[0] if addr_parts[0] else '123 Main St',
                    'city': addr_parts[1] if addr_parts[1] else 'City',
                    'state_code': state_code,
                    'postcode': addr_parts[3] if addr_parts[3] else '12345',
                    'country_code': country_code,
                    'email': customer_email or 'no-reply@example.com',
                    'phone_number': data.get('phone', '+1234567890')  # Required with default
                },
                'shipping_level': 'MAIL',
                'production_delay': 120  # Optional: 2 hour delay to allow cancellations
            }
            
            # Determine endpoint based on sandbox setting
            if LULU_USE_SANDBOX:
                api_url = 'https://api.sandbox.lulu.com/print-jobs/'
                print(f"[lulu] Using SANDBOX endpoint")
            else:
                api_url = 'https://api.lulu.com/print-jobs/'
                print(f"[lulu] Using PRODUCTION endpoint")
            
            # Add detailed logging
            print(f"[lulu] Sending request to Lulu API: {api_url}")
            print(f"[lulu] Order data: {json.dumps(order_data, indent=2)}")
            
            # Send as JSON, not multipart
            response = requests.post(
                api_url,
                headers={
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json'  # JSON header
                },
                json=order_data  # Send as JSON, not files
            )
            
            # Log response details
            print(f"[lulu] Response status: {response.status_code}")
            print(f"[lulu] Response text: {response.text[:1000]}")
            
            if response.status_code == 401:  # Auth failed
                print("[lulu] Auth failed, refreshing token")
                token = get_lulu_token()
                continue  # Retry with new token
                
            if response.status_code in [200, 201]:
                lulu_order = response.json()
                return safe_json_response({
                    'success': True,
                    'lulu_order_id': lulu_order.get('id', 'unknown'),
                    'status': 'sent_to_lulu',
                    'pdf_url': pdf_url,  # Include for debugging
                    'cover_url': cover_url,
                    'failed_images': failed_count
                })
            elif response.status_code >= 500:  # Server error, retry
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"[lulu] Server error, retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                    
            return safe_json_response({
                'success': False,
                'error': f"Lulu API error: {response.text}"
            }, 500)
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 3
                print(f"[lulu] Error on attempt {attempt+1}, retrying in {wait_time}s: {safe_str(e)}")
                time.sleep(wait_time)
            else:
                print(f"[lulu] Failed after {max_retries} attempts: {safe_str(e)}")
                import traceback
                print(f"[lulu] Traceback: {traceback.format_exc()}")
                return safe_json_response({
                    'success': False,
                    'error': safe_str(e)
                }, 500)

# ---------- Routes ----------
@app.route("/process", methods=["POST"])
def process():
   try:
       payload = request.get_json(force=True, silent=True)
       
       if not payload:
           payload = {}
           payload['order_id'] = request.form.get('order_id', f"order_{int(time.time())}")
           image_urls = request.form.get('image_urls', '')
           if image_urls:
               try:
                   image_urls = json.loads(image_urls)
               except:
                   image_urls = [u.strip() for u in image_urls.split(',') if u.strip()]
           else:
               image_urls = request.form.getlist('image_url')
           payload['image_urls'] = image_urls
           payload['prompt'] = request.form.get('prompt', DEFAULT_PROMPT)
           
           # EXTRACT CUSTOMER DATA FROM FORM
           payload['customer_name'] = request.form.get('customer_name', 'N/A')
           payload['customer_email'] = request.form.get('customer_email', 'N/A')
           payload['shipping_address'] = request.form.get('shipping_address', 'N/A')
           
           print(f"[process] Received customer_name: {payload['customer_name']}")
           print(f"[process] Received customer_email: {payload['customer_email']}")
           print(f"[process] Received shipping_address: {payload['shipping_address']}")
       
       image_urls = payload.get('image_urls', [])
       order_id = payload.get('order_id')
       print(f"[process] order_id={order_id}, total urls={len(image_urls)}")
       
       # Store original URLs for the spreadsheet
       payload['original_urls'] = image_urls.copy()
       
       # Split into batches
       total_batches = (len(image_urls) + BATCH_SIZE - 1) // BATCH_SIZE
       
       # Queue to Cloud Tasks
       try:
           client = tasks_v2.CloudTasksClient()
           parent = client.queue_path('coloring-book-processor', 'us-central1', 'image-processing')
           
           # Create tasks for each batch with deduplication
           tasks_created = 0
           for i in range(0, len(image_urls), BATCH_SIZE):
               batch = image_urls[i:i+BATCH_SIZE]
               batch_num = (i // BATCH_SIZE) + 1
               batch_payload = {
                   **payload,
                   'image_urls': batch,
                   'batch_number': batch_num,
                   'total_batches': total_batches,
                   'batch_start_index': i,
                   'total_images': len(image_urls)
               }
               
               task = {
                   'http_request': {
                       'http_method': tasks_v2.HttpMethod.POST,
                       'url': 'https://coloring-book-processor-585071603431.us-central1.run.app/process-worker',
                       'headers': {'Content-Type': 'application/json'},
                       'body': json.dumps(batch_payload).encode()
                   }
               }
               
               # Use deterministic task name to prevent duplicates
               urls_hash = str(hash(''.join(batch)))[-6:]
               task_id = f"{order_id}-b{batch_num}-{urls_hash}".replace('#', '').replace('/', '-')[:100]
               
               try:
                   task_with_name = {
                       **task,
                       'name': f"{parent}/tasks/{task_id}"
                   }
                   client.create_task(parent=parent, task=task_with_name)
                   tasks_created += 1
                   print(f"[process] Queued batch {batch_num}/{total_batches} with task ID: {task_id}")
               except Exception as task_error:
                   if "already exists" in str(task_error).lower():
                       print(f"[process] Task {task_id} already exists, skipping duplicate")
                   else:
                       client.create_task(parent=parent, task=task)
                       tasks_created += 1
                       print(f"[process] Queued batch {batch_num}/{total_batches} (unnamed)")
           
           if tasks_created == 0:
               return safe_json_response({
                   "success": True,
                   "status": "already_processing",
                   "order_id": order_id,
                   "message": "Order is already being processed"
               })
           
           print(f"[process] {tasks_created} new batches queued to Cloud Tasks")
           
       except Exception as e:
           print(f"[process] Cloud Tasks not available, processing synchronously: {safe_str(e)}")
           return process_worker_internal(payload)
       
       return safe_json_response({
           "success": True,
           "status": "queued",
           "order_id": order_id,
           "total_images": len(image_urls),
           "batches": total_batches,
           "message": f"Processing {len(image_urls)} images in {total_batches} batch(es)"
       })
       
   except Exception as e:
       err = safe_str(e)
       print(f"[process] request failed: {err}")
       return safe_json_response({"success": False, "error": err}, 500)

@app.route("/process-worker", methods=["POST"])
def process_worker():
   payload = request.get_json(force=True)
   return process_worker_internal(payload)

def process_worker_internal(payload):
   from google.oauth2 import service_account
   from googleapiclient.discovery import build
   
   try:
       order_id = payload.get("order_id", f"order_{int(time.time())}")
       image_urls = payload.get("image_urls", [])
       original_urls = payload.get("original_urls", [])
       prompt = payload.get("prompt", DEFAULT_PROMPT)
       batch_number = payload.get("batch_number", 1)
       total_batches = payload.get("total_batches", 1)
       batch_start_index = payload.get("batch_start_index", 0)
       total_images = payload.get("total_images", len(image_urls))
       retry_count = payload.get("retry_count", 0)
       
       print(f"[worker] Processing batch {batch_number}/{total_batches} for order {order_id}: {len(image_urls)} images")
       print(f"[worker] Customer data - name: {payload.get('customer_name')}, email: {payload.get('customer_email')}")
       
       # NEW SPREADSHEET ID
       spreadsheet_id = '1fVKN0Lf6FHnfFvNZW80nNuxn_Vx3HHHR0cbwcfj5gwc'
       
       # Early check: if this is batch 1 and order already exists as complete, skip
       if batch_number == 1:
           try:
               creds = service_account.Credentials.from_service_account_file('key.json')
               service = build('sheets', 'v4', credentials=creds)
               
               sheet_data = service.spreadsheets().values().get(
                   spreadsheetId=spreadsheet_id,
                   range='A:F'
               ).execute()
               
               values = sheet_data.get('values', [])
               for row in values:
                   if row and len(row) > 0 and row[0] == order_id:
                       if len(row) > 5 and row[5] == 'complete':
                           print(f"[worker] Order {order_id} already complete, skipping batch {batch_number}")
                           return safe_json_response({"success": True, "message": "Order already complete"})
           except Exception as e:
               print(f"[worker] Could not check existing status: {safe_str(e)}")
       
       results = []
       error_count = 0
       
       for idx, url in enumerate(image_urls):
           try:
               actual_idx = batch_start_index + idx
               print(f"[worker] Processing image {idx+1}/{len(image_urls)} (overall #{actual_idx+1})")
               raw = download_image(url)
               edited = call_openai_edit(raw, prompt)
               final_url = upload_to_gcs(order_id, actual_idx, edited)
               results.append(final_url)
           except Exception as e:
               error_str = str(e)
               print(f"[worker] Error processing image {idx}: {safe_str(e)}")
               
               # Check if it's a safety violation - use placeholder
               if "safety" in error_str.lower() or "filtered" in error_str.lower():
                   placeholder = create_placeholder_page(f"Image {actual_idx+1} - Create your own!")
                   final_url = upload_to_gcs(order_id, actual_idx, placeholder)
                   results.append(final_url)
                   print(f"[worker] Used placeholder for filtered image {idx}")
               else:
                   results.append(f"SKIPPED: Image {actual_idx+1} failed")
                   error_count += 1
       
       # Update Google Sheets - either update existing row or create new
       try:
           creds = service_account.Credentials.from_service_account_file('key.json')
           service = build('sheets', 'v4', credentials=creds)
           
           # Read existing data to find the row
           sheet_data = service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
               range='A:Q'  # Extended to include column Q for original URLs
           ).execute()
           
           values = sheet_data.get('values', [])
           row_index = None
           existing_urls = ""
           existing_status = ""
           existing_error_count = 0
           
           # Find existing row for this order
           for i, row in enumerate(values):
               if row and len(row) > 0 and row[0] == order_id:
                   row_index = i + 1
                   existing_urls = row[4] if len(row) > 4 else ""
                   existing_status = row[5] if len(row) > 5 else ""
                   existing_error_count = int(row[14]) if len(row) > 14 and row[14].isdigit() else 0
                   break
           
           # Combine error counts
           total_error_count = existing_error_count + error_count
           
           # Remove duplicates when combining URLs - USING NEWLINES
           successful_urls = [r for r in results if not r.startswith("ERROR") and not r.startswith("SKIP") and r.startswith("http")]
           
           if existing_urls:
               # Parse existing URLs - handle both comma and newline separators
               if '\n' in existing_urls:
                   existing_list = [u.strip() for u in existing_urls.split('\n') if u.strip() and u.strip().startswith('http')]
               else:
                   existing_list = [u.strip() for u in existing_urls.split(',') if u.strip() and u.strip().startswith('http')]
               
               seen = set(existing_list)
               for new_url in successful_urls:
                   if new_url not in seen and new_url.startswith('http'):
                       existing_list.append(new_url)
                       seen.add(new_url)
               # USE NEWLINES TO PREVENT GOOGLE SHEETS CORRUPTION
               all_urls = '\n'.join(existing_list)
           else:
               # USE NEWLINES FOR NEW ENTRIES
               all_urls = '\n'.join([u for u in successful_urls if u.startswith('http')])
           
           # Create comma-separated version for Zapier
           zapier_urls = all_urls.replace('\n', ',')
           
           # Format original URLs with newlines for column Q
           if batch_number == 1 and original_urls:
               # Only set original URLs on first batch
               original_urls_str = '\n'.join(original_urls)
           else:
               # Keep existing original URLs for subsequent batches
               original_urls_str = ''
           
           # Count total successful images
           url_list = [u.strip() for u in all_urls.split('\n') if u.strip()]
           total_successful = len(url_list)
           
           # Update status properly for each batch
           if batch_number == total_batches:
               if total_successful >= total_images * 0.75:  # At least 75% success
                   status = 'complete'
               else:
                   status = 'partial_complete'
               notes = f'All {total_batches} batch(es) processed. {total_successful}/{total_images} images successful, {total_error_count} errors.'
           else:
               expected_status = f'processing_batch_{batch_number}/{total_batches}'
               if existing_status and 'processing_batch_' in existing_status:
                   try:
                       existing_batch = int(existing_status.split('_')[-1].split('/')[0])
                       if existing_batch >= batch_number:
                           status = existing_status
                           notes = f'Batch {batch_number}/{total_batches} updated (retry)'
                       else:
                           status = expected_status
                           notes = f'Batch {batch_number}/{total_batches} complete. Processing continues...'
                   except:
                       status = expected_status
                       notes = f'Batch {batch_number}/{total_batches} complete. Processing continues...'
               else:
                   status = expected_status
                   notes = f'Batch {batch_number}/{total_batches} complete. Processing continues...'
           
           # Prepare row data with POD columns and original URLs
           row_data = [
               order_id,                          # A - Order ID
               payload.get('customer_name', 'N/A'),  # B - Customer Name
               payload.get('customer_email', 'N/A'), # C - Customer Email
               payload.get('shipping_address', 'N/A'), # D - Shipping Address
               all_urls,                          # E - Image URLs (processed)
               status,                            # F - Status
               time.strftime('%Y-%m-%d %H:%M:%S'), # G - Timestamp
               total_successful,                  # H - Number of Images
               payload.get('shopify_order_number', order_id), # I - Shopify Order Number
               notes,                             # J - Notes
               'FALSE',                           # K - Send to POD
               '',                                # L - Lulu Order ID
               '',                                # M - POD Date
               zapier_urls,                       # N - Zapier URLs (comma-separated)
               str(total_error_count),            # O - Error Count
               str(retry_count),                  # P - Retry Count
               original_urls_str if batch_number == 1 else ''  # Q - Original URLs (only on first batch)
           ]
           
           if row_index:
               print(f"[worker] Updating row {row_index} for order {order_id} - Status: {status}")
               # Don't overwrite column Q if it already has data
               if batch_number != 1:
                   # Read existing column Q value
                   existing_row = values[row_index - 1] if row_index <= len(values) else []
                   if len(existing_row) > 16:  # Column Q is index 16
                       row_data[16] = existing_row[16]  # Keep existing original URLs
               
               service.spreadsheets().values().update(
                   spreadsheetId=spreadsheet_id,
                   range=f'A{row_index}:Q{row_index}',
                   valueInputOption='RAW',
                   body={'values': [row_data]}
               ).execute()
           else:
               print(f"[worker] Creating new row for order {order_id} - Status: {status}")
               service.spreadsheets().values().append(
                   spreadsheetId=spreadsheet_id,
                   range='A:Q',
                   valueInputOption='RAW',
                   body={'values': [row_data]}
               ).execute()
           
           print(f"[worker] Batch {batch_number}/{total_batches} for order {order_id} complete. {len(successful_urls)} images processed, {error_count} errors.")
           
       except Exception as e:
           print(f"[worker] Failed to write to sheets: {safe_str(e)}")
           # Store backup locally if Sheets fails
           try:
               backup_data = {
                   "order_id": order_id,
                   "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                   "results": results,
                   "error": str(e)
               }
               with open(f"/tmp/backup_{order_id}_{batch_number}.json", "w") as f:
                   json.dump(backup_data, f)
               print(f"[worker] Backup saved to /tmp/backup_{order_id}_{batch_number}.json")
           except:
               pass
       
       return safe_json_response({"success": True, "batch": batch_number, "results": results, "errors": error_count})
       
   except Exception as e:
       print(f"[worker] Job failed: {safe_str(e)}")
       return safe_json_response({"success": False, "error": safe_str(e)}, 500)

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
       "lulu_configured": bool(LULU_CLIENT_ID and LULU_CLIENT_SECRET),
       "bucket_name": bucket_name,
       "version": VERSION,
       "lulu_mode": "SANDBOX" if LULU_USE_SANDBOX else "PRODUCTION"
   })

@app.route("/", methods=["GET"])
def index():
   return safe_json_response({
       "service": "Coloring Book Processor",
       "endpoints": {
           "/process": "POST - Process images to line art",
           "/process-worker": "POST - Worker endpoint for Cloud Tasks",
           "/send-to-lulu": "POST - Send to Lulu for printing",
           "/test": "GET/POST - Test with sample image",
           "/health": "GET - Health check",
           "/": "GET - This help"
       }
   })

if __name__ == "__main__":
   app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
