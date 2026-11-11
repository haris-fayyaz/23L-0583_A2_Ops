import os
import sys

import keras
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/mobilenet_s2.keras")
MODEL_NAME = os.getenv("MODEL_NAME", "mobilenet_s2")
THRESHOLD = float(os.getenv("THRESHOLD", "0.25"))
IMG_SIZE = (224, 224)  # keep in sync with scripts/pipeline.py


class ImageDecodeError(ValueError):
    """Bytes look like an image but cannot be decoded."""


# Loaded once at start-up, not per request
_model = keras.models.load_model(MODEL_PATH, compile=False)
# Warm-up call so the first user request is not slow
_model.predict(np.zeros((1, *IMG_SIZE, 3), dtype="float32"), verbose=0)


def preprocess(raw):
    try:
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    except tf.errors.InvalidArgumentError as e:
        raise ImageDecodeError("Cannot decode image") from e
    img = tf.image.resize(img, IMG_SIZE, antialias=True)
    return tf.expand_dims(img, 0)


def predict(raw):
    x = preprocess(raw)
    p = float(_model.predict(x, verbose=0)[0][0])
    label = "AI-generated" if p >= THRESHOLD else "Real"
    return {"label": label, "confidence": p,
            "threshold": THRESHOLD, "model": MODEL_NAME}


if __name__ == "__main__":
    for path in sys.argv[1:]:
        with open(path, "rb") as f:
            print(path)
            print("  ", predict(f.read()))