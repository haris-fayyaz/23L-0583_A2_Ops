import os
import subprocess
import sys

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from models import unfreeze_last_block
from pipeline import SEED, make_ds, seed_everything

load_dotenv()
for name in ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]:
    if not os.getenv(name):
        raise SystemExit(f"Missing environment variable: {name}")

with open("data/raw/archive.dvc") as f:
    DVC_MD5 = next(line.split(":")[1].strip() for line in f if "md5:" in line)
LR = 1e-5
EPOCHS = 8
commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


class MlflowLogger(keras.callbacks.Callback):
    """Send every per-epoch value in Keras `logs` to MLflow.

    Keep this callback AFTER ReduceLROnPlateau in the callbacks list:
    ReduceLROnPlateau adds `learning_rate` to `logs`, so it is logged too.
    """

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            mlflow.log_metric(k, float(v), step=epoch + 1)


def log_val_and_artifacts(y, p, thr=0.5):
    """Log validation metrics, confusion matrix, ROC curve, splits and package list."""
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    mlflow.log_metrics({
        "final_val_accuracy": accuracy_score(y, pred),
        "final_val_precision": precision_score(y, pred, zero_division=0),
        "final_val_recall": recall_score(y, pred),
        "final_val_f1": f1_score(y, pred),
        "final_val_auc": roc_auc_score(y, p),
    })
    mlflow.log_param("eval_threshold", thr)

    cm = [[tn, fp], [fn, tp]]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Real", "AI"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Real", "AI"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Validation confusion matrix (threshold {thr})")
    fig.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix_val.png")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC {roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Validation ROC curve"); ax.legend(loc="lower right")
    fig.tight_layout()
    mlflow.log_figure(fig, "roc_curve_val.png")
    plt.close(fig)

    mlflow.log_artifacts("splits", artifact_path="splits")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    mlflow.log_text(freeze, "requirements_freeze.txt")


seed_everything()
train_ds = make_ds("train", training=True)
val_ds = make_ds("val")
model = keras.models.load_model("models/mobilenet_s1.keras")
unfreeze_last_block(model, LR)

mlflow.set_experiment("visionguard")
with mlflow.start_run(run_name="s2_manual"):
    mlflow.log_params(
        {"model": "mobilenet_v2", "stage": 2, "init": "mobilenet_s1", "lr": LR,
         "dropout": model.get_layer("dropout").rate, "epochs_max": EPOCHS, "batch": 32,
         "img_size": 224, "seed": SEED, "optimizer": "adam", "loss": "binary_crossentropy",
         "augment": True, "unfrozen_from": "block_16_expand"}
    )
    mlflow.set_tags(
        {"git_commit": commit, "dvc_data_md5": DVC_MD5, "seed": str(SEED), "logging": "manual"}
    )
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(
                "models/mlflow_manual.keras", monitor="val_loss", save_best_only=True
            ),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6),
            MlflowLogger(),
        ],
    )
    vl = hist.history["val_loss"]
    mlflow.log_metric("best_epoch", vl.index(min(vl)) + 1)
    mlflow.log_metric("best_val_loss", min(vl))

    # Score the saved best checkpoint on VALIDATION only (test set is not touched).
    best_model = keras.models.load_model("models/mlflow_manual.keras")
    val_df = pd.read_csv("splits/val.csv")
    p = best_model.predict(val_ds, verbose=0).ravel()
    assert len(p) == len(val_df), "row count mismatch"
    log_val_and_artifacts(val_df["label"].values, p)
    mlflow.keras.log_model(best_model, name="model")
    print("run done, best val_loss", round(min(vl), 4))