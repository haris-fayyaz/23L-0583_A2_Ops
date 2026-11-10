import os
import subprocess
import sys

import keras
import mlflow
import mlflow.keras
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient

NAME = "23L-0583_A2_Ops"
V1_MODEL_ID = "m-538c587d54184ef18e8bc4cb4a689bac"  # logged model of run s2_manual

load_dotenv()
for v in ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]:
    if not os.getenv(v):
        raise SystemExit(f"Missing environment variable: {v}")

if len(sys.argv) != 2 or sys.argv[1] not in ("v1", "v2"):
    raise SystemExit("usage: python scripts/register_models.py v1|v2")
client = MlflowClient()

if sys.argv[1] == "v1":
    mv = mlflow.register_model(f"models:/{V1_MODEL_ID}", NAME)
    print("registered version", mv.version, "from s2_manual")
else:
    with open("data/raw/archive.dvc") as f:
        dvc_md5 = next(l.split(":")[1].strip() for l in f if "md5:" in l)
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    row = pd.read_csv("reports/test_metrics.csv")
    row = row[(row.model == "mobilenet_s2") & (row.rule == "val")].iloc[0]
    thr = float(row.threshold)

    mlflow.set_experiment("visionguard")
    with mlflow.start_run(run_name="official_mobilenet_s2"):
        mlflow.log_params({"model": "mobilenet_v2", "stage": 2, "source": "models/mobilenet_s2.keras",
                           "threshold": thr, "threshold_rule": "best F1 on validation",
                           "seed": 42, "img_size": 224})
        mlflow.set_tags({"git_commit": commit, "dvc_data_md5": dvc_md5, "seed": "42",
                         "role": "official model", "test_set": "evaluated once in Task 2"})
        mlflow.log_metrics({
            "test_accuracy": float(row.accuracy), "test_precision": float(row.precision),
            "test_recall": float(row.recall), "test_f1": float(row.f1), "test_auc": float(row.auc),
            "test_tn": int(row.TN), "test_fp": int(row.FP), "test_fn": int(row.FN), "test_tp": int(row.TP),
        })
        for f in ["reports/confusion_matrices.png", "reports/roc_curves.png",
                  "reports/test_metrics.csv", "reports/thresholds.json"]:
            mlflow.log_artifact(f)
        mlflow.log_artifacts("splits", artifact_path="splits")
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        mlflow.log_text(freeze, "requirements_freeze.txt")
        model = keras.models.load_model("models/mobilenet_s2.keras")
        mlflow.keras.log_model(model, name="model", registered_model_name=NAME)
    versions = client.search_model_versions(f"name='{NAME}'")
    newest = max(int(v.version) for v in versions)
    client.set_registered_model_alias(NAME, "champion", str(newest))
    print("alias champion -> version", newest)