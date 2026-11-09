"""Audit the image dataset: one row per file, plus corrupt and duplicate checks."""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 80)


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def inspect_image(path):
    """Return image facts, or the error if the file cannot be decoded."""
    try:
        with Image.open(path) as im:
            im.verify()  # checks file structure
        with Image.open(path) as im:  # verify() invalidates the object, reopen
            im.load()  # fully decodes pixels, catches truncated files
            return {"width": im.width, "height": im.height,
                    "mode": im.mode, "format": im.format, "error": ""}
    except Exception as exc:
        return {"width": None, "height": None, "mode": None,
                "format": None, "error": f"{type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/archive"))
    ap.add_argument("--out", type=Path, default=Path("reports/audit_images.csv"))
    args = ap.parse_args()

    files = sorted(p for p in args.root.rglob("*") if p.is_file())
    rows = []
    for p in files:
        rel = p.relative_to(args.root)
        rows.append({
            "path": rel.as_posix(),
            "class": rel.parts[0],  # the label is the top-level folder
            "file_size": p.stat().st_size,
            "sha256": sha256_of(p),
            **inspect_image(p),
        })

    df = pd.DataFrame(rows)
    df["readable"] = df["error"] == ""
    df["is_duplicate"] = df.duplicated("sha256", keep="first")  # extra copies only
    df["file_kb"] = (df["file_size"] / 1024).round(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Total files: {len(df)}  (table saved to {args.out})\n")

    print("== Counts per class ==")
    counts = df.groupby("class").agg(
        files=("path", "count"),
        readable=("readable", "sum"),
        corrupt=("readable", lambda s: int((~s).sum())),
        extra_duplicates=("is_duplicate", "sum"),
    )
    print(counts, "\n")

    print("== Min / median / max (readable images only) ==")
    ok = df[df["readable"]]
    stats = ok.groupby("class")[["width", "height", "file_kb"]].agg(["min", "median", "max"])
    print(stats, "\n")

    print("== Format per class ==")
    print(pd.crosstab(df["class"], df["format"].fillna("unreadable")), "\n")
    print("== Colour mode per class ==")
    print(pd.crosstab(df["class"], df["mode"].fillna("unreadable")), "\n")

    print("== Corrupt or unreadable files ==")
    bad = df[~df["readable"]][["path", "error"]]
    print(bad.to_string(index=False) if len(bad) else "none", "\n")

    print("== Exact duplicates (same SHA-256) ==")
    dups = df[df.duplicated("sha256", keep=False)]
    groups = list(dups.groupby("sha256"))
    cross = sum(g["class"].nunique() > 1 for _, g in groups)
    print(f"duplicate groups: {len(groups)}, redundant files: {int(df['is_duplicate'].sum())}, "
          f"groups spanning both classes: {cross}")
    for _, g in groups[:15]:
        print(" -", " | ".join(g["path"]))


if __name__ == "__main__":
    main()