import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

val = pd.read_csv("reports/preds_val.csv")
test = pd.read_csv("reports/preds_test.csv")
NAMES = ["baseline", "mobilenet_s2"]
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def pick_threshold(y, p):
    scores = [f1_score(y, (p >= t).astype(int)) for t in GRID]
    best = max(scores)
    cands = [t for t, s in zip(GRID, scores) if s >= best - 1e-12]
    return float(min(cands, key=lambda t: abs(t - 0.5)))


y = test["label"].values
rows, thr, cm_at = [], {}, {}
for n in NAMES:
    t = pick_threshold(val["label"].values, val[n].values)
    thr[n] = t
    p = test[n].values
    auc = roc_auc_score(y, p)
    for tag, th in [("0.5", 0.5), ("val", t)]:
        pred = (p >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({"model": n, "rule": tag, "threshold": th,
                     "accuracy": accuracy_score(y, pred),
                     "precision": precision_score(y, pred, zero_division=0),
                     "recall": recall_score(y, pred),
                     "f1": f1_score(y, pred), "auc": auc,
                     "TN": tn, "FP": fp, "FN": fn, "TP": tp})
        if tag == "val":
            cm_at[n] = (tn, fp, fn, tp, th)

table = pd.DataFrame(rows)
table.round(4).to_csv("reports/test_metrics.csv", index=False)
print(table.round(4).to_string(index=False))
print("thresholds chosen on val:", thr)
with open("reports/thresholds.json", "w") as f:
    json.dump(thr, f)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, n in zip(axes, NAMES):
    tn, fp, fn, tp, th = cm_at[n]
    m = np.array([[tn, fp], [fn, tp]])
    ax.imshow(m, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Real", "AI"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Real", "AI"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{n} (threshold {th})")
fig.suptitle("Test set confusion matrices")
fig.tight_layout()
fig.savefig("reports/confusion_matrices.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 5))
for n in NAMES:
    fpr, tpr, _ = roc_curve(y, test[n].values)
    ax.plot(fpr, tpr, label=f"{n} (AUC {roc_auc_score(y, test[n].values):.3f})")
    tn, fp, fn, tp, th = cm_at[n]
    ax.scatter([fp / (fp + tn)], [tp / (tp + fn)], s=50, zorder=3)
ax.plot([0, 1], [0, 1], "k--", label="chance")
ax.set_xlabel("False positive rate (real called AI)")
ax.set_ylabel("True positive rate (AI called AI)")
ax.set_title("Test ROC curves (dots = chosen threshold)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("reports/roc_curves.png", dpi=150)
plt.close(fig)
print("saved figures")