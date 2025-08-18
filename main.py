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
import tempfile

# --- Config ---
# NEVER put API keys in code! Use environment variables
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=api_key)
bucket_name = os.environ.get("GCS_BUCKET", "coloring-book-results")

DEFAULT_PROMPT = (
    "Convert this photo into clean black-and-white line art suitable for a coloring book. "
    "Keep facial features recognizable, simplify shading to outlines, remove background clutter, "
    "and produce crisp, printable lines on white."
)

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
def sanitize_text(s: str) -> str:
    """Strip invisible Unicode separators and ensure clean utf-8."""
    if not isinstance(s, str):
        s = str(s)
    # Remove problematic Unicode characters
    s = s.replace(u"\u2028", " ").replace(u"\u2029", " ")
    # Remove any other non-ASCII characters that might cause issues
    s = s.encode("ascii", "ignore").decode("ascii")
    return s

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
            # UploadKit usually has the image URL in meta tags or as a direct link
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
                        # Extract base URL from original
                        base = '/'.join(url.split('/')[:3])
                        return base + img_url
            
            # If we can't find an image URL, try to extract from URL parameters
            if 'fi=' in url:
                # The 'fi' parameter might contain the encoded filename
                import urllib.parse
                parsed = urllib.parse.urlparse(url)
                params = urllib.parse.parse_qs(parsed.query)
                if 'fi' in params:
                    # Decode the base64 filename
                    try:
                        filename = base64.b64decode(params['fi'][0]).decode('utf-8')
                        # Construct a direct URL (this is a guess at the pattern)
                        base_path = url.split('/files/')[0] + '/files/'
                        return base_path + filename
                    except:
                        pass
                        
        except Exception as e:
            print(f"Failed to extract image from HTML: {e}")
    
    # Return original URL as fallback
    return url

def download_image(url: str) -> bytes:
    """Download image bytes from a URL."""
    # First try to get the direct image URL
    direct_url = extract_direct_image_url(url)
    print(f"Downloading from: {direct_url[:100]}...")
    
    headers = {"User-Agent": "ColoringBookProcessor/1.0"}
    resp = requests.get(direct_url, headers=headers, timeout=20)
    resp.raise_for_status()
    
    # Check if we got HTML instead of an image
    content_type = resp.headers.get('content-type', '')
    if 'text/html' in content_type:
        raise ValueError(f"Got HTML instead of image from {direct_url[:50]}...")
    
    return resp.content

def call_openai_edit(image_bytes: bytes, prompt: str) -> bytes:
    """Send image to OpenAI image edit API and return edited PNG bytes."""
    try:
        # Clean the prompt of any problematic characters
        clean_prompt = sanitize_text(prompt or DEFAULT_PROMPT)
        
        # Clean the image by re-saving it without metadata
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert RGBA to RGB if necessary (OpenAI doesn't like transparency)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create a white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Call OpenAI API with clean image
        resp = client.images.edit(
            model="gpt-image-1",
            image=img_byte_arr,
            prompt=clean_prompt,
            size="1024x1024"
        )
        
        # Get the base64 result
        b64 = resp.data[0].b64_json
        return base64.b64decode(b64)
        
    except Exception as e:
        print(f"OpenAI API error: {e}")
        raise

def upload_to_gcs(order_id: str, idx: int, img_bytes: bytes) -> str:
    """Upload PNG to GCS and return signed URL."""
    if not bucket:
        # If no GCS, return base64 data URL instead
        b64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/png;base64,{b64[:100]}..." # Truncated for response
    
    blob_name = f"{order_id}/{int(time.time())}_{idx}.png"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(img_bytes, content_type="image/png")
    
    # Generate a signed URL that expires in 7 days
    return blob.generate_signed_url(expiration=604800)  # 7 days

# --- Routes ---
@app.route("/process", methods=["POST"])
def process():
    """Main processing endpoint."""
    try:
        payload = request.get_json(force=True)
        order_id = payload.get("order_id", f"order_{int(time.time())}")
        
        # Handle both 'image_urls' and 'urls' keys
        image_urls = payload.get("image_urls") or payload.get("urls", [])
        
        # If it's a single string, convert to list
        if isinstance(image_urls, str):
            image_urls = [url.strip() for url in image_urls.split(',') if url.strip()]
        
        prompt = payload.get("prompt", DEFAULT_PROMPT)
        
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
                print(f"Error processing image {idx}: {e}")
                results.append({
                    "status": "error",
                    "index": idx,
                    "source_url": url,
                    "error": str(e)
                })

        return jsonify({
            "success": True,
            "count": len(results),
            "order_id": order_id,
            "prompt_used": sanitize_text(prompt),
            "results": results
        })
        
    except Exception as e:
        print(f"Request failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/test", methods=["GET", "POST"])
def test():
    """Test endpoint with a sample image."""
    # Use a public domain test image
    test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"
    
    try:
        raw = download_image(test_url)
        edited = call_openai_edit(raw, "Convert to simple line art coloring page")
        
        # Return as base64 for easy testing
        b64 = base64.b64encode(edited).decode('utf-8')
        
        return jsonify({
            "success": True,
            "message": "Test successful!",
            "original_url": test_url,
            "result_base64_preview": b64[:100] + "...",
            "result_size": len(edited)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "coloring-book-processor",
        "gcs_available": bucket is not None,
        "openai_configured": api_key is not None
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
