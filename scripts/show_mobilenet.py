import os

import numpy as np
import pandas as pd

from models import build_mobilenet, unfreeze_last_block


def trainable_rows(model, stage):
    base = model.get_layer("mobilenetv2_base")
    rows = []
    for part, layer_list in [("base", base.layers), ("head", [
        l for l in model.layers if l is not base
    ])]:
        for l in layer_list:
            if l.trainable_weights:
                n = sum(int(np.prod(w.shape)) for w in l.trainable_weights)
                rows.append((stage, part, l.name, type(l).__name__, n))
    return rows


model = build_mobilenet()
total = model.count_params()
rows = trainable_rows(model, "stage1_frozen")

unfreeze_last_block(model, 1e-5)
rows += trainable_rows(model, "stage2_last_block")

df = pd.DataFrame(rows, columns=["stage", "part", "layer", "type", "trainable_params"])
os.makedirs("reports", exist_ok=True)
df.to_csv("reports/mobilenet_trainable_layers.csv", index=False)

print(f"total params: {total:,}")
for stage, g in df.groupby("stage", sort=False):
    n = g["trainable_params"].sum()
    print(f"\n{stage}: {len(g)} trainable layers, {n:,} trainable params")
    print(g[["part", "layer", "type", "trainable_params"]].to_string(index=False))