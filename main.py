import os
import io
import time
import base64
import requests
from flask import Flask, request, jsonify
from google.cloud import storage
from openai import OpenAI
import re
from PIL import Image
import sys
import codecs

# Force UTF-8 encoding for the entire application
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# --- Config ---
# NEVER put API keys in code! Use environment variables
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=api_key)
# Fixed to match your environment variable name
bucket_name = os.environ.get("OUTPUT_BUCKET", "memory-books-output")

# Keep your optimized prompt but ensure it's ASCII-safe
DEFAULT_PROMPT = "Convert this photo into a professional adult coloring book page with clean continuous black outlines only on pure white background"

app = Flask(__name__)

# Initialize storage client only if we have credentials
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    print(f"Connected to GCS bucket: {bucket_name}")
except Exception as e:
    print(f"GCS not available: {e}")
    storage_client = None
    bucket = None

# --- Helpers ---
def safe_str(obj):
    """Convert any object to a safe ASCII string, removing all problematic Unicode."""
    try:
        s = str(obj)
        # Remove specific problematic Unicode characters
        s = s.replace('\u2028', ' ').replace('\u2029', ' ').replace('\u00a0', ' ')
        s = s.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')  # Zero-width chars
        # Keep only ASCII characters
        return ''.join(char if ord(char) < 128 else '?' for char in s)
    except:
        return "Error converting to string"

def sanitize_text(s: str) -> str:
    """Strip invisible Unicode separators and ensure clean ASCII."""
    return safe_str(s).strip()

def extract_direct_image_url(url: str) -> str:
    """Extract the actual image URL from UploadKit HTML pages."""
    # If it's already a direct image URL, return it
    if url.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
        return url
    
    # Handle UploadKit HTML download pages
    if 'uploadkit' in url or 'download.html' in url:
        try:
            # Download the HTML page
            resp = requests.get(url, timeout=10)
            html_content = resp.text
            
            # Look for direct image URL in the HTML
            patterns = [
                r'<meta property="og:image" content="([^"]+)"',
                r'<img[^>]+src="([^"]+)"',
                r'href="([^"]+\.(jpg|jpeg|png|gif|webp)[^"]*)"'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, html_content, re.IGNORECASE)
                if match:
                    img_url = match.group(1)
                    # Make sure it's a full URL
                    if img_url.startswith('http'):
                        return img_url
                    elif img_url.startswith('//'):
                        return 'https:' + img_url
                    elif img_url.startswith('/'):
                        base = '/'.join(url.split('/')[:3])
                        return base + img_url
        except Exception as e:
            print(f"Failed to extract image from HTML: {e}")
    
    return url

def download_image(url: str) -> bytes:
    """Download image bytes from a URL."""
    direct_url = extract_direct_image_url(url)
    print(f"Downloading from: {direct_url[:100]}...")
    
    headers = {"User-Agent": "ColoringBookProcessor/1.0"}
    resp = requests.get(direct_url, headers=headers, timeout=20)
    resp.raise_for_status()
    
    content_type = resp.headers.get('content-type', '')
    if 'text/html' in content_type:
        raise ValueError(f"Got HTML instead of image from {direct_url[:50]}...")
    
    return resp.content

def call_openai_edit(image_bytes: bytes, prompt: str) -> bytes:
    """Send image to OpenAI image edit API and return edited PNG bytes."""
    try:
        # Clean the image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            if img.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            else:
                img = img.convert('RGB')
        
        # Save as clean PNG
        clean_img = io.BytesIO()
        img.save(clean_img, format='PNG')
        clean_img.seek(0)
        
        # Use simple ASCII-only prompt
        if prompt and len(prompt) > 10:
            # Ensure prompt is ASCII-safe
            clean_prompt = sanitize_text(prompt)
        else:
            clean_prompt = sanitize_text(DEFAULT_PROMPT)
        
        if not clean_prompt or len(clean_prompt) < 10:
            clean_prompt = "Convert to line art coloring book page"
        
        print(f"Using prompt: {clean_prompt[:50]}...")
        
        # Call OpenAI API with URL response format (avoids base64 Unicode issues)
        resp = client.images.edit(
            model="gpt-image-1",  # Keep your reverse-engineered model
            image=clean_img,
            prompt=clean_prompt,
            size="1024x1024",
            response_format="url"  # Get URL instead of base64 to avoid Unicode issues
        )
        
        # Download the result from URL
        img_url = resp.data[0].url
        print(f"Downloading result from OpenAI URL...")
        img_resp = requests.get(img_url, timeout=30)
        img_resp.raise_for_status()
        
        return img_resp.content
        
    except Exception as e:
        # Nuclear option for Unicode handling
        error_msg = safe_str(e)
        print(f"Error in call_openai_edit: {error_msg}")
        
        # Try with simplest possible prompt as fallback
        try:
            print("Retrying with minimal prompt...")
            
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            minimal_img = io.BytesIO()
            img.save(minimal_img, format='PNG')
            minimal_img.seek(0)
            
            resp = client.images.edit(
                model="gpt-image-1",  # Keep your reverse-engineered model
                image=minimal_img,
                prompt="line art coloring page",
                size="1024x1024",
                response_format="url"
            )
            
            img_url = resp.data[0].url
            img_resp = requests.get(img_url, timeout=30)
            return img_resp.content
            
        except Exception as e2:
            error_msg2 = safe_str(e2)
            print(f"Fallback also failed: {error_msg2}")
            raise Exception(f"Image processing failed: {error_msg2}")

def upload_to_gcs(order_id: str, idx: int, img_bytes: bytes) -> str:
    """Upload PNG to GCS and return signed URL."""
    if not bucket:
        # If no GCS, return base64 data URL
        b64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/png;base64,{b64}"
    
    blob_name = f"{order_id}/{int(time.time())}_{idx}.png"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(img_bytes, content_type="image/png")
    
    # Generate a signed URL that expires in 7 days
    return blob.generate_signed_url(expiration=604800)

# --- Routes ---
@app.route("/process", methods=["POST"])
def process():
    """Main processing endpoint."""
    try:
        payload = request.get_json(force=True)
        order_id = sanitize_text(payload.get("order_id", f"order_{int(time.time())}"))
        
        # Handle both 'image_urls' and 'urls' keys
        image_urls = payload.get("image_urls") or payload.get("urls", [])
        
        # If it's a single string, convert to list
        if isinstance(image_urls, str):
            image_urls = [url.strip() for url in image_urls.split(',') if url.strip()]
        
        # Get and sanitize prompt
        raw_prompt = payload.get("prompt", DEFAULT_PROMPT)
        prompt = sanitize_text(raw_prompt) if raw_prompt else DEFAULT_PROMPT
        
        print(f"Processing order {order_id} with {len(image_urls)} images")

        results = []
        for idx, url in enumerate(image_urls):
            try:
                print(f"Processing image {idx + 1}/{len(image_urls)}")
                
                # Download the image
                raw = download_image(url)
                print(f"Downloaded {len(raw)} bytes")
                
                # Process with OpenAI
                edited = call_openai_edit(raw, prompt)
                print(f"OpenAI processing complete")
                
                # Upload to storage
                if bucket:
                    signed = upload_to_gcs(order_id, idx, edited)
                    storage_type = "gcs"
                else:
                    # Return base64 if no GCS
                    signed = base64.b64encode(edited).decode('utf-8')
                    storage_type = "base64"
                
                results.append({
                    "status": "ok",
                    "index": idx,
                    "source_url": url,
                    "result_url": signed if bucket else None,
                    "result_base64": signed if not bucket else None,
                    "storage_type": storage_type
                })
                
            except Exception as e:
                # Sanitize error message
                error_msg = sanitize_text(str(e))
                print(f"Error processing image {idx}: {error_msg}")
                results.append({
                    "status": "error",
                    "index": idx,
                    "source_url": url,
                    "error": error_msg
                })

        # Force ASCII encoding on response
        safe_prompt = prompt.encode('ascii', 'ignore').decode('ascii')
        
        return jsonify({
            "success": True,
            "count": len(results),
            "order_id": order_id,
            "prompt_used": safe_prompt[:100],
            "results": results
        })
        
    except Exception as e:
        # Sanitize error message
        error_msg = sanitize_text(str(e))
        print(f"Request failed: {error_msg}")
        return jsonify({
            "success": False,
            "error": error_msg
        }), 500

@app.route("/test", methods=["GET", "POST"])
def test():
    """Test endpoint with a sample image."""
    test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"
    
    try:
        raw = download_image(test_url)
        edited = call_openai_edit(raw, "Convert to coloring book page")
        
        # Return as base64 for testing
        b64 = base64.b64encode(edited).decode('utf-8')
        
        return jsonify({
            "success": True,
            "message": "Test successful!",
            "original_url": test_url,
            "result_base64_preview": b64[:100] + "...",
            "result_size": len(edited)
        })
        
    except Exception as e:
        # Nuclear Unicode handling
        error_msg = safe_str(e)
        return jsonify({
            "success": False,
            "error": error_msg
        }), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "coloring-book-processor",
        "gcs_available": bucket is not None,
        "openai_configured": api_key is not None,
        "bucket_name": bucket_name  # Added to verify correct bucket
    })

@app.route("/", methods=["GET"])
def index():
    """Root endpoint with usage instructions."""
    return jsonify({
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
