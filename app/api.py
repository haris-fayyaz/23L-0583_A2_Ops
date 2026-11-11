import hmac
import logging
import os
import time

from flask import Flask, jsonify, request

from app import inference

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("visionguard-api")

ALLOWED = {"png", "jpg", "jpeg"}
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
API_KEY = os.getenv("API_KEY", "")

# Fail fast: an API with no key must never start
if not API_KEY:
    raise RuntimeError("API_KEY is not set")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


def error(code, msg):
    return jsonify(error=msg), code


@app.before_request
def check_key():
    if request.path == "/health":
        return None
    sent = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(sent.encode(), API_KEY.encode()):
        log.warning("rejected request: wrong or missing API key")
        return error(401, "Missing or wrong API key.")
    return None


@app.post("/predict")
def predict():
    f = request.files.get("file")
    if f is None or not f.filename:
        return error(400, "Send an image in the form field 'file'.")
    ext = os.path.splitext(f.filename)[1].lower().lstrip(".")
    if ext not in ALLOWED:
        return error(415, "Allowed types: " + ", ".join(sorted(ALLOWED)) + ".")
    raw = f.read()
    if not raw:
        return error(400, "The file has no content.")
    start = time.perf_counter()
    try:
        res = inference.predict(raw)
    except inference.ImageDecodeError:
        return error(422, "The file looks like an image but cannot be decoded.")
    ms = round((time.perf_counter() - start) * 1000)
    log.info("predict label=%s confidence=%.4f model=%s ms=%d",
             res["label"], res["confidence"], res["model"], ms)
    return jsonify(**res)


@app.get("/health")
def health():
    return jsonify(status="ok", model=inference.MODEL_NAME,
                   threshold=inference.THRESHOLD)


@app.errorhandler(413)
def too_large(e):
    return error(413, f"Maximum size is {MAX_MB} MB.")


@app.errorhandler(404)
def not_found(e):
    return error(404, "Not found.")


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", "5000")), debug=False)
