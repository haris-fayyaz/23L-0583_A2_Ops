import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from app import db

load_dotenv()

# With API_URL set, predictions come from the hosted API. Otherwise the local model is used.
if os.getenv("API_URL"):
    from app import client as backend
    RemoteError = backend.RemoteError
else:
    from app import inference as backend

    class RemoteError(Exception):
        """Never raised in local mode."""

ALLOWED = {"png", "jpg", "jpeg"}
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

db.init_db()


def error_page(code, title, msg):
    if request.path == "/predict":
        return jsonify(error=msg), code
    return render_template("error.html", code=code, title=title, msg=msg), code


@app.get("/")
def upload():
    return render_template("index.html")


@app.post("/predict")
def predict():
    f = request.files.get("file")
    if f is None or not f.filename:
        return error_page(400, "No file", "Choose an image first.")
    ext = os.path.splitext(f.filename)[1].lower().lstrip(".")
    if ext not in ALLOWED:
        return error_page(415, "Wrong file type",
                          "Allowed types: " + ", ".join(sorted(ALLOWED)) + ".")
    name = secure_filename(f.filename) or "image"
    raw = f.read()
    if not raw:
        return error_page(400, "Empty file", "The file has no content.")
    try:
        res = backend.predict(raw)
    except backend.ImageDecodeError:
        return error_page(422, "Cannot read image",
                          "The file looks like an image but cannot be decoded.")
    except RemoteError as e:
        return error_page(e.code, "Prediction service unavailable", e.message)
    db.log_request(name, res["label"], res["confidence"],
                   res["threshold"], res["model"])
    return jsonify(name=name, **res)


@app.get("/history")
def history():
    return render_template("history.html", rows=db.recent())


@app.get("/health")
def health():
    if os.getenv("API_URL"):
        return jsonify(status="ok", mode="remote")
    return jsonify(status="ok", model=backend.MODEL_NAME,
                   threshold=backend.THRESHOLD)


@app.errorhandler(413)
def too_large(e):
    return error_page(413, "File too large", f"Maximum size is {MAX_MB} MB.")


@app.errorhandler(404)
def not_found(e):
    return error_page(404, "Page not found", "That page does not exist.")


@app.errorhandler(500)
def server_error(e):
    return error_page(500, "Something went wrong",
                      "Please try again with a different image.")


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", "5000")), debug=False)
