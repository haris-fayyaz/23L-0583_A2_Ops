import argparse
import os
import time

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from models import build_baseline, compile_model
from pipeline import SEED, make_ds, seed_everything

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--name", default="baseline")
args = p.parse_args()

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

seed_everything()
train_ds = make_ds("train", training=True)
val_ds = make_ds("val")

model = build_baseline(lr=args.lr)
if not model.compiled:
    compile_model(model, args.lr)

ckpt_path = f"models/{args.name}.keras"
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_loss", save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6
    ),
]

t0 = time.time()
hist = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    callbacks=callbacks,
    verbose=2,
)
mins = (time.time() - t0) / 60

h = pd.DataFrame(hist.history)
h.insert(0, "epoch", range(1, len(h) + 1))
h.to_csv(f"reports/{args.name}_history.csv", index=False)
best = int(h["val_loss"].idxmin()) + 1

acc = "accuracy" if "accuracy" in h else "binary_accuracy"
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, key, title in [(ax[0], "loss", "Loss"), (ax[1], acc, "Accuracy")]:
    a.plot(h["epoch"], h[key], marker="o", label="train")
    a.plot(h["epoch"], h["val_" + key], marker="o", label="val")
    a.axvline(best, color="red", linestyle="--", label=f"best epoch {best}")
    a.set_title(title)
    a.set_xlabel("Epoch")
    a.legend()
fig.suptitle(f"{args.name} (seed {SEED})")
fig.tight_layout()
fig.savefig(f"reports/{args.name}_curves.png", dpi=150)

print(f"best epoch: {best} of {len(h)}")
print(f"best val_loss: {h['val_loss'].min():.4f}")
print(f"val {acc} at best: {h.loc[best - 1, 'val_' + acc]:.4f}")
print(f"time: {mins:.1f} min")
print(f"saved: {ckpt_path}")