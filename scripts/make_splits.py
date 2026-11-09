import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 42
OUT = Path("splits")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("reports/audit_images.csv")
df = df[df["readable"]].drop_duplicates("sha256", keep="first").copy()
df["label"] = (df["class"] == "AiArtData").astype(int)
df = df[["path", "class", "label", "sha256"]]

train, rest = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=SEED
)
val, test = train_test_split(
    rest, test_size=0.50, stratify=rest["label"], random_state=SEED
)

# no image may appear in two splits
assert not set(train.sha256) & set(val.sha256)
assert not set(train.sha256) & set(test.sha256)
assert not set(val.sha256) & set(test.sha256)

for name, part in [("train", train), ("val", val), ("test", test)]:
    part.to_csv(OUT / f"{name}.csv", index=False)
    print(name, len(part), part["class"].value_counts().to_dict())
print("total", len(df))