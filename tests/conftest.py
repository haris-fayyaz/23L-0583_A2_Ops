import os
import sys
import tempfile
import types

import pytest

# Settings must be set before app.main is imported
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tempfile.mkdtemp(), "test.db")


# Stub model: CI has no model file and no TensorFlow
class ImageDecodeError(ValueError):
    pass


def predict(raw):
    if not raw.startswith(b"\xff\xd8"):
        raise ImageDecodeError("not a jpeg")
    return {"label": "AI-generated", "confidence": 0.92,
            "threshold": 0.25, "model": "stub"}


stub = types.ModuleType("app.inference")
stub.ImageDecodeError = ImageDecodeError
stub.predict = predict
stub.MODEL_NAME = "stub"
stub.THRESHOLD = 0.25
sys.modules["app.inference"] = stub


@pytest.fixture()
def client():
    from app.main import app
    return app.test_client()
