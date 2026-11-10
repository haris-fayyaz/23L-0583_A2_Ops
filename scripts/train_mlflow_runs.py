import argparse
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
from keras import layers
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

import pipeline
from models import compile_model, unfreeze_last_block
from pipeline import SEED, make_ds, seed_everything

CONFIGS = {
    "s2_lr1e-4": dict(lr=1e-4, unfreeze="block_16", dropout=0.3, augment=True,
                      change="learning rate 1e-5 -> 1e-4"),
    "s2_noaug": dict(lr=1e-5, unfreeze="block_16", dropout=0.3, augment=False,
                     change="augmentation off"),
    "s2_unfreeze33": dict(lr=1e-5, unfreeze=33, dropout=0.3, augment=True,
                          change="last 33 base layers unfrozen"),
    "s2_head_only": dict(lr=1e-5, unfreeze=0, dropout=0.3, augment=True,
                         change="base fully frozen, head only"),
    "s2_dropout0.5": dict(lr=1e-5, unfreeze="block_16", dropout=0.5, augment=True,
                          change="dropout 0.3 -> 0.5"),
}

p = argparse.ArgumentParser()
p.add_argument("name", choices=list(CONFIGS))
args = p.parse_args()
cfg = CONFIGS[args.name]

load_dotenv()
for v in ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]:
    if not os.getenv(v):
        raise SystemExit(f"Missing environment variable: {v}")

with open("data/raw/archive.dvc") as f:
    DVC_MD5 = next(line.split(":")[1].strip() for line in f if "md5:" in line)
EPOCHS = 8
commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()

if not cfg["augment"]:
    pipeline.augment = lambda x, training=False: x


class MlflowLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for k, v in (logs or {}).items():
            mlflow.log_metric(k, float(v), step=epoch + 1)


def log_val_and_artifacts(y, p, thr=0.5):
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
model.get_layer("dropout").rate = cfg["dropout"]

if cfg["unfreeze"] == "block_16":
    unfreeze_last_block(model, cfg["lr"])
else:
    base = model.get_layer("mobilenetv2_base")
    n = cfg["unfreeze"]
    if n == 0:
        base.trainable = False
    else:
        base.trainable = True
        start = len(base.layers) - n
        for i, layer in enumerate(base.layers):
            layer.trainable = i >= start and not isinstance(layer, layers.BatchNormalization)
    compile_model(model, cfg["lr"])
n_train = sum(w.numpy().size for w in model.trainable_weights)
print(f"{args.name}: trainable params {n_train:,}")

mlflow.set_experiment("visionguard")
with mlflow.start_run(run_name=args.name):
    mlflow.log_params({
        "model": "mobilenet_v2", "stage": 2, "init": "mobilenet_s1", "lr": cfg["lr"],
        "dropout": cfg["dropout"], "epochs_max": EPOCHS, "batch": 32, "img_size": 224,
        "seed": SEED, "optimizer": "adam", "loss": "binary_crossentropy",
        "augment": cfg["augment"], "unfrozen": str(cfg["unfreeze"]),
        "trainable_params": n_train,
    })
    mlflow.set_tags({"git_commit": commit, "dvc_data_md5": DVC_MD5, "seed": str(SEED),
                     "logging": "manual", "change": cfg["change"]})
    ckpt = f"models/{args.name}.keras"
    hist = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6),
            MlflowLogger(),
        ],
    )
    vl = hist.history["val_loss"]
    mlflow.log_metric("best_epoch", vl.index(min(vl)) + 1)
    mlflow.log_metric("best_val_loss", min(vl))
    best = keras.models.load_model(ckpt)
    val_df = pd.read_csv("splits/val.csv")
    pv = best.predict(val_ds, verbose=0).ravel()
    assert len(pv) == len(val_df), "row count mismatch"
    log_val_and_artifacts(val_df["label"].values, pv)
    mlflow.keras.log_model(best, name="model")
    print(args.name, "done, best val_loss", round(min(vl), 4))