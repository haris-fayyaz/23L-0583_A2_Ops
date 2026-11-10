import json
import os
import sys

import mlflow
import mlflow.keras
import tensorflow as tf
from dotenv import load_dotenv

from pipeline import load_image

NAME = "23L-0583_A2_Ops"

load_dotenv()
for v in ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]:
    if not os.getenv(v):
        raise SystemExit(f"Missing environment variable: {v}")
if len(sys.argv) != 2:
    raise SystemExit("usage: python scripts/predict_by_alias.py <image path>")

thr = json.load(open("reports/thresholds.json"))["mobilenet_s2"]
model = mlflow.keras.load_model(f"models:/{NAME}@champion")
img, _ = load_image(sys.argv[1], 0)
p = float(model.predict(tf.expand_dims(img, 0), verbose=0)[0][0])
label = "AI-generated" if p >= thr else "Real"
print(f"image: {sys.argv[1]}")
print(f"P(AI) = {p:.3f}, threshold = {thr}")
print(f"verdict: {label}")