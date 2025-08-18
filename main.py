from flask import Flask, jsonify, request

app = Flask(__name__)

@app.get("/health")
def health_check():
    return jsonify({"status": "healthy"}), 200

# simple stub to receive your Zapier POST later
@app.post("/process")
def process():
    data = request.get_json(silent=True) or {}
    # TODO: run your image pipeline here
    return jsonify({"ok": True, "received_keys": list(data.keys())}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
