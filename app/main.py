import os

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from app import db, inference

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
        res = inference.predict(raw)
    except inference.ImageDecodeError:
        return error_page(422, "Cannot read image",
                          "The file looks like an image but cannot be decoded.")
    db.log_request(name, res["label"], res["confidence"],
                   res["threshold"], res["model"])
    return jsonify(name=name, **res)


@app.get("/history")
def history():
    return render_template("history.html", rows=db.recent())


@app.get("/health")
def health():
    return jsonify(status="ok", model=inference.MODEL_NAME,
                   threshold=inference.THRESHOLD)


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
