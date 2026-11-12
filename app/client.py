import json
import os
import urllib.error
import urllib.request
import uuid

from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "").rstrip("/")
API_KEY = os.getenv("API_KEY", "")
TIMEOUT = int(os.getenv("API_TIMEOUT", "150"))  # a cold start of the hosted API takes about 75 s

if not API_URL or not API_KEY:
    raise RuntimeError("API_URL and API_KEY must be set to use the hosted API")


class ImageDecodeError(ValueError):
    """The hosted API could not decode the image."""


class RemoteError(Exception):
    """The hosted API is unreachable or refused the request."""

    def __init__(self, message, code=502):
        super().__init__(message)
        self.message = message
        self.code = code


def _multipart(raw):
    boundary = uuid.uuid4().hex
    head = (f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="upload.jpg"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n").encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return boundary, head + raw + tail


def _error_text(err):
    try:
        return json.load(err).get("error", "")
    except (ValueError, AttributeError):
        return ""


def predict(raw):
    boundary, body = _multipart(raw)
    req = urllib.request.Request(
        API_URL + "/predict", data=body, method="POST",
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary,
                 "X-API-Key": API_KEY})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            res = json.load(resp)
    except urllib.error.HTTPError as e:
        text = _error_text(e)
        if e.code == 422:
            raise ImageDecodeError(text) from e
        if e.code in (401, 403):
            raise RemoteError("The prediction service rejected the API key.") from e
        raise RemoteError(text or "The prediction service returned an error.") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RemoteError("Could not reach the prediction service. Try again.", 504) from e
    except ValueError as e:
        raise RemoteError("The prediction service sent an unreadable answer.") from e
    if not all(k in res for k in ("label", "confidence", "threshold", "model")):
        raise RemoteError("The prediction service sent an unexpected answer.")
    return res
