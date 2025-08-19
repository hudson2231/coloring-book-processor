import os
import io
import base64
import requests
from flask import Flask, request, jsonify
from google.cloud import storage
from PIL import Image
from openai import OpenAI

# --- Force-disable proxies that break OpenAI client on Cloud Run ---
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    if os.environ.get(_k):
        print(f"[net] ignoring proxy env {_k}")
        os.environ.pop(_k, None)

# also block any system proxy fallback
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

# --- Setup ---
app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

bucket_name = os.environ.get("OUTPUT_BUCKET", "memory-books-output")
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)


def download_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def upload_to_gcs(image: Image.Image, path: str) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    blob = bucket.blob(path)
    blob.upload_from_file(buf, content_type="image/png")
    return f"gs://{bucket_name}/{path}"


def generate_coloring_page(image: Image.Image, prompt: str) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    # call OpenAI images API
    resp = client.images.edit(
        model="gpt-image-1",
        image=buf,
        prompt=prompt,
        size="1024x1024",
        n=1
    )
    img_b64 = resp.data[0].b64_json
    out_bytes = base64.b64decode(img_b64)
    return Image.open(io.BytesIO(out_bytes)).convert("RGB")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "coloring-book-processor",
        "gcs_available": str(bucket.exists()),
        "openai_configured": str(bool(os.environ.get("OPENAI_API_KEY"))),
        "bucket_name": bucket_name
    })


@app.route("/test", methods=["GET"])
def test_openai():
    try:
        resp = client.models.list()
        return jsonify({"success": True, "models_count": len(resp.data)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/process", methods=["POST"])
def process():
    data = request.get_json()
    order_id = data.get("order_id", "unknown")
    prompt = data.get("prompt", "Convert this photo into a coloring page")
    image_urls = data.get("image_urls", [])

    results = []
    for idx, url in enumerate(image_urls):
        try:
            img = download_image(url)
            out_img = generate_coloring_page(img, prompt)
            out_path = f"{order_id}_{idx}.png"
            gcs_path = upload_to_gcs(out_img, out_path)
            results.append({
                "status": "success",
                "index": str(idx),
                "source_url": url,
                "gcs_path": gcs_path
            })
        except Exception as e:
            results.append({
                "status": "error",
                "index": str(idx),
                "source_url": url,
                "error": f"Image processing failed: {e}"
            })

    return jsonify({
        "success": "True",
        "count": str(len(results)),
        "order_id": order_id,
        "prompt_used": prompt,
        "results": results
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
