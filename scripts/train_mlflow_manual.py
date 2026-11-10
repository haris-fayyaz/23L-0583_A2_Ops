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

DVC_MD5 = "b2cec67839f742df4653f66d4bcbbafb.dir"
LR = 1e-5
EPOCHS = 8
commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


class MlflowLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            mlflow.log_metric(k, float(v), step=epoch + 1)
        lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        mlflow.log_metric("lr", lr, step=epoch + 1)


seed_everything()
train_ds = make_ds("train", training=True)
val_ds = make_ds("val")
model = keras.models.load_model("models/mobilenet_s1.keras")
unfreeze_last_block(model, LR)

mlflow.set_experiment("visionguard")
with mlflow.start_run(run_name="s2_manual_v2"):
    mlflow.log_params(
        {"model": "mobilenet_v2", "stage": 2, "init": "mobilenet_s1", "lr": LR,
         "dropout": 0.3, "epochs_max": EPOCHS, "batch": 32, "img_size": 224, "seed": SEED}
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
    print("run done, best val_loss", round(min(vl), 4))