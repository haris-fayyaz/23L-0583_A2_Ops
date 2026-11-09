import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from pipeline import ROOT, load_image

THR = 0.25
model = keras.models.load_model("models/mobilenet_s2.keras")
pre = model.get_layer("mobilenet_preprocess")
base = model.get_layer("mobilenetv2_base")
gap = model.get_layer("gap")
drop = model.get_layer("dropout")
head = model.get_layer("prob_ai")

df = pd.read_csv("reports/preds_test.csv")
df["called_ai"] = df["mobilenet_s2"] >= THR
picks = []
groups = [
    ("real->real", (df.label == 0) & ~df.called_ai, "mobilenet_s2", True),
    ("fake->fake", (df.label == 1) & df.called_ai, "mobilenet_s2", False),
    ("real->fake", (df.label == 0) & df.called_ai, "mobilenet_s2", False),
    ("fake->real", (df.label == 1) & ~df.called_ai, "mobilenet_s2", True),
]
for kind, mask, col, asc in groups:
    sub = df[mask].sort_values(col, ascending=asc).head(2)
    for _, r in sub.iterrows():
        picks.append((kind, r["path"], r["class"], r["mobilenet_s2"]))


def heatmap(img):
    x = tf.expand_dims(img, 0)
    with tf.GradientTape() as tape:
        conv = base(pre(x), training=False)
        p = head(drop(gap(conv), training=False))[:, 0]
    g = tape.gradient(p, conv)
    w = tf.reduce_mean(g, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * w[:, None, None, :], axis=-1))[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (224, 224))[..., 0]
    return cam.numpy(), float(p[0])


fig, axes = plt.subplots(4, 4, figsize=(14, 14))
rows = []
for i, (kind, path, cls, csv_p) in enumerate(picks):
    img, _ = load_image((ROOT / path).as_posix(), 0)
    cam, p = heatmap(img)
    print(f"{kind:11s} csv={csv_p:.3f} now={p:.3f} {path.split('/')[-1][:40]}")
    r, c = i // 2, (i % 2) * 2
    shown = np.clip(img.numpy() / 255.0, 0, 1)
    axes[r, c].imshow(shown)
    axes[r, c + 1].imshow(shown)
    axes[r, c + 1].imshow(cam, cmap="jet", alpha=0.45)
    axes[r, c].set_title(f"{kind} ({cls})", fontsize=9)
    axes[r, c + 1].set_title(f"AI prob {p:.2f}", fontsize=9)
    rows.append({"kind": kind, "path": path, "class": cls, "prob": round(p, 3)})
for a in axes.ravel():
    a.axis("off")
plt.tight_layout()
plt.savefig("reports/gradcam_grid.png", dpi=110)
pd.DataFrame(rows).to_csv("reports/gradcam_picks.csv", index=False)
print("saved reports/gradcam_grid.png and reports/gradcam_picks.csv")