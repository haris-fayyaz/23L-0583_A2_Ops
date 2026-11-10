import os
import subprocess

import keras
import mlflow
from dotenv import load_dotenv

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

mlflow.set_experiment("visionguard")
mlflow.tensorflow.autolog(log_models=False)

seed_everything()
train_ds = make_ds("train", training=True)
val_ds = make_ds("val")
model = keras.models.load_model("models/mobilenet_s1.keras")
unfreeze_last_block(model, LR)

with mlflow.start_run(run_name="s2_autolog"):
    mlflow.set_tags(
        {"git_commit": commit, "dvc_data_md5": DVC_MD5, "seed": str(SEED), "logging": "auto"}
    )
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(
                "models/mlflow_auto.keras", monitor="val_loss", save_best_only=True
            ),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6),
        ],
    )
    vl = hist.history["val_loss"]
    print("run done, best val_loss", round(min(vl), 4), "best epoch", vl.index(min(vl)) + 1)