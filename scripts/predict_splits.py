import numpy as np
import pandas as pd
import keras
from pipeline import make_ds, seed_everything

seed_everything()

MODELS = {
    "baseline": "models/baseline.keras",
    "mobilenet_s2": "models/mobilenet_s2.keras",
}
loaded = {n: keras.models.load_model(p, compile=False) for n, p in MODELS.items()}

for split in ["val", "test"]:
    df = pd.read_csv(f"splits/{split}.csv")
    ds = make_ds(split)
    y = np.concatenate([lab.numpy().ravel() for _, lab in ds])
    assert len(y) == len(df), "row count mismatch"
    assert (y == df["label"].values).all(), "order mismatch"
    out = df[["path", "class", "label"]].copy()
    for name, model in loaded.items():
        out[name] = model.predict(ds, verbose=0).ravel()
        if split == "val":
            acc = ((out[name] > 0.5) == out["label"]).mean()
            print(name, "val acc at 0.5:", round(acc, 4))
    out.to_csv(f"reports/preds_{split}.csv", index=False)
    print(split, len(out), "rows saved")