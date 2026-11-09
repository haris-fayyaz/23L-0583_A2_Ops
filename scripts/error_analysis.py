import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = "data/raw/archive"
MODEL = "mobilenet_s2"
thr = json.load(open("reports/thresholds.json"))[MODEL]
df = pd.read_csv("reports/preds_test.csv")


def info(p):
    full = os.path.join(ROOT, p)
    with Image.open(full) as im:
        return im.width, im.height, im.mode, im.format, os.path.getsize(full) / 1024


meta = pd.DataFrame(df["path"].apply(info).tolist(),
                    columns=["w", "h", "mode", "fmt", "kb"], index=df.index)
df = pd.concat([df, meta], axis=1)
df["pred"] = (df[MODEL] >= thr).astype(int)
df["wrong"] = df["pred"] != df["label"]
df["conf_wrong"] = np.where(df["pred"] == 1, df[MODEL], 1 - df[MODEL])

wrong = df[df["wrong"]]
fp = wrong[wrong["label"] == 0].nlargest(5, "conf_wrong")
fn = wrong[wrong["label"] == 1].nlargest(5, "conf_wrong")
sel = pd.concat([fp, fn]).reset_index(drop=True)
sel.index = sel.index + 1

show = sel.assign(file=sel["path"].str.split("/").str[-1].str[:38])
print(show[["file", "label", MODEL, "w", "h", "kb"]].round(2).to_string())
print()
print("threshold:", thr, "| wrong:", int(df["wrong"].sum()), "of", len(df))
print(df.groupby(["class", "wrong"]).agg(n=("path", "size"), w=("w", "median"),
                                          h=("h", "median"),
                                          kb=("kb", "median")).round(0))

sel[["path", "class", "label", MODEL, "w", "h", "mode", "fmt", "kb"]].round(3) \
    .to_csv("reports/misclassified.csv", index=False)

fig, axes = plt.subplots(2, 5, figsize=(15, 7))
for ax, (i, r) in zip(axes.ravel(), sel.iterrows()):
    with Image.open(os.path.join(ROOT, r["path"])) as im:
        ax.imshow(im.convert("RGB"))
    kind = "AI" if r["label"] == 1 else "Real"
    ax.set_title(f"#{i} true {kind}, P(AI)={r[MODEL]:.2f}", fontsize=10)
    ax.axis("off")
fig.suptitle("Top row: real called AI. Bottom row: AI called real.")
fig.tight_layout()
fig.savefig("reports/misclassified_grid.png", dpi=150)
plt.close(fig)
print("saved grid")