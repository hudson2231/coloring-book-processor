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
from PIL import Image
import uuid
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

VERSION = "cbp-v1.13-shopify-hash-fix"

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

def call_openai_edit(image_bytes: bytes, prompt: str) -> bytes:
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

   # FIXED: Strip the # from Shopify order numbers
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
    try:
        # Get data from Zapier
        data = request.get_json(force=True, silent=True) or {}
        
        # Parse the image URLs - FIX THE HASH IN URLS
        image_urls_str = data.get('image_urls', '')
        if isinstance(image_urls_str, list):
            image_urls = [url.replace("#", "") for url in image_urls_str]
        else:
            image_urls = [u.strip().replace("#", "") for u in image_urls_str.split(',') if u.strip()]
        
        order_id = data.get('order_id', f"order_{int(time.time())}").replace("#", "")
        
        # Download and prepare images
        images = []
        for url in image_urls:
            if url and not url.startswith("ERROR"):
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
        
        # Create and upload a simple cover PDF  
        # FIX: For saddle stitch, cover needs to be double width
        cover_buffer = io.BytesIO()
        cover_canvas = canvas.Canvas(cover_buffer, pagesize=(12.188*72, 9.188*72))  # Full spread for saddle stitch
        cover_canvas.setFillColorRGB(1, 1, 1)  # White background
        cover_canvas.rect(0, 0, 12.188*72, 9.188*72, fill=1)
        cover_canvas.showPage()
        cover_canvas.save()
        
        cover_blob_name = f"lulu_covers/{order_id}_{int(time.time())}_cover.pdf"
        cover_blob = bucket.blob(cover_blob_name)
        cover_blob.upload_from_string(cover_buffer.getvalue(), content_type="application/pdf")
        cover_url = f"https://storage.googleapis.com/{bucket.name}/{cover_blob_name}"
        print(f"[lulu] Cover PDF uploaded: {cover_url}")
        
        # Page count should now match what we tell Lulu
        page_count = len(images) * 2  # Double-sided pages
        pod_package_id = '0600X0900BWSTDSS060UW444MXX'  # 6x9, B&W, Saddle Stitch, 60# White
        
        # Get Lulu token
        token = get_lulu_token()
        
        # Parse address
        shipping_address = data.get('shipping_address', '')
        customer_name = data.get('customer_name', 'Customer')
        customer_email = data.get('customer_email', 'no-reply@example.com')
        
        # Simple address parsing (expecting: "Street, City, State, ZIP, Country")
        addr_parts = [p.strip() for p in shipping_address.split(',')]
        
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
                'street1': addr_parts[0] if len(addr_parts) > 0 else '123 Main St',
                'city': addr_parts[1] if len(addr_parts) > 1 else 'City',
                'state_code': addr_parts[2] if len(addr_parts) > 2 else 'CA',
                'postcode': addr_parts[3] if len(addr_parts) > 3 else '12345',
                'country_code': addr_parts[4] if len(addr_parts) > 4 else 'US',
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
        
        if response.status_code in [200, 201]:
            lulu_order = response.json()
            return safe_json_response({
                'success': True,
                'lulu_order_id': lulu_order.get('id', 'unknown'),
                'status': 'sent_to_lulu',
                'pdf_url': pdf_url,  # Include for debugging
                'cover_url': cover_url
            })
        else:
            return safe_json_response({
                'success': False,
                'error': f"Lulu API error: {response.text}"
            }, 500)
            
    except Exception as e:
        print(f"[lulu] Error: {safe_str(e)}")
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
       
       image_urls = payload.get('image_urls', [])
       order_id = payload.get('order_id')
       print(f"[process] order_id={order_id}, total urls={len(image_urls)}")
       
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
       prompt = payload.get("prompt", DEFAULT_PROMPT)
       batch_number = payload.get("batch_number", 1)
       total_batches = payload.get("total_batches", 1)
       batch_start_index = payload.get("batch_start_index", 0)
       total_images = payload.get("total_images", len(image_urls))
       
       print(f"[worker] Processing batch {batch_number}/{total_batches} for order {order_id}: {len(image_urls)} images")
       
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
       for idx, url in enumerate(image_urls):
           try:
               actual_idx = batch_start_index + idx
               print(f"[worker] Processing image {idx+1}/{len(image_urls)} (overall #{actual_idx+1})")
               raw = download_image(url)
               edited = call_openai_edit(raw, prompt)
               final_url = upload_to_gcs(order_id, actual_idx, edited)
               results.append(final_url)
           except Exception as e:
               print(f"[worker] Error processing image {idx}: {safe_str(e)}")
               results.append(f"ERROR: {safe_str(e)}")
       
       # Update Google Sheets - either update existing row or create new
       try:
           creds = service_account.Credentials.from_service_account_file('key.json')
           service = build('sheets', 'v4', credentials=creds)
           
           # Read existing data to find the row
           sheet_data = service.spreadsheets().values().get(
               spreadsheetId=spreadsheet_id,
               range='A:M'
           ).execute()
           
           values = sheet_data.get('values', [])
           row_index = None
           existing_urls = ""
           existing_status = ""
           
           # Find existing row for this order
           for i, row in enumerate(values):
               if row and len(row) > 0 and row[0] == order_id:
                   row_index = i + 1
                   existing_urls = row[4] if len(row) > 4 else ""
                   existing_status = row[5] if len(row) > 5 else ""
                   break
           
           # Remove duplicates when combining URLs
           successful_urls = [r for r in results if not r.startswith("ERROR")]
           
           if existing_urls:
               existing_list = [u.strip() for u in existing_urls.split(',') if u.strip()]
               seen = set(existing_list)
               for new_url in successful_urls:
                   if new_url not in seen:
                       existing_list.append(new_url)
                       seen.add(new_url)
               all_urls = ','.join(existing_list)
           else:
               all_urls = ','.join(successful_urls)
           
           # Count total successful images
           url_list = [u.strip() for u in all_urls.split(',') if u.strip()]
           total_successful = len(url_list)
           
           # Update status properly for each batch
           if batch_number == total_batches:
               status = 'complete'
               notes = f'All {total_batches} batch(es) processed. {total_successful}/{total_images} images successful.'
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
           
           # Prepare row data with POD columns
           row_data = [
               order_id,                          # A
               payload.get('customer_name', 'N/A'),  # B
               payload.get('customer_email', 'N/A'), # C
               payload.get('shipping_address', 'N/A'), # D
               all_urls,                          # E
               status,                            # F
               time.strftime('%Y-%m-%d %H:%M:%S'), # G
               total_successful,                  # H
               payload.get('shopify_order_number', order_id), # I
               notes,                             # J
               'FALSE',                           # K - Send to POD
               '',                                # L - Lulu Order ID
               ''                                 # M - POD Date
           ]
           
           if row_index:
               print(f"[worker] Updating row {row_index} for order {order_id} - Status: {status}")
               service.spreadsheets().values().update(
                   spreadsheetId=spreadsheet_id,
                   range=f'A{row_index}:M{row_index}',
                   valueInputOption='RAW',
                   body={'values': [row_data]}
               ).execute()
           else:
               print(f"[worker] Creating new row for order {order_id} - Status: {status}")
               service.spreadsheets().values().append(
                   spreadsheetId=spreadsheet_id,
                   range='A:M',
                   valueInputOption='RAW',
                   body={'values': [row_data]}
               ).execute()
           
           print(f"[worker] Batch {batch_number}/{total_batches} for order {order_id} complete. {len(successful_urls)} new images added.")
           
       except Exception as e:
           print(f"[worker] Failed to write to sheets: {safe_str(e)}")
       
       return safe_json_response({"success": True, "batch": batch_number, "results": results})
       
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
