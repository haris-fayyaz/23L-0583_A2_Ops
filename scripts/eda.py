"""Visual EDA: sample grids and histograms per class, from the audit table."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path("data/raw/archive")
OUT = Path("reports/figures")
SEED = 42
N_ROWS, N_COLS = 2, 6


def load_thumb(path, size=256):
    with Image.open(path) as im:
        im.draft("RGB", (size, size))  # fast downscaled decode for JPEG
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((size, size))
        return im.copy()


def sample_grid(df, cls):
    pick = df[(df["class"] == cls) & df["readable"]].sample(
        N_ROWS * N_COLS, random_state=SEED)
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(2.4 * N_COLS, 2.6 * N_ROWS))
    for ax, (_, r) in zip(axes.ravel(), pick.iterrows()):
        ax.imshow(load_thumb(ROOT / r["path"]))
        ax.set_title(f'{r["width"]}x{r["height"]}', fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Random samples: {cls}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f"samples_{cls}.png", dpi=150)
    plt.close(fig)


def histograms(df):
    ok = df[df["readable"]]
    for col, label in [("width", "Width (px)"), ("height", "Height (px)"),
                       ("file_kb", "File size (KB)")]:
        bins = np.logspace(np.log10(ok[col].min()), np.log10(ok[col].max()), 30)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for cls, g in ok.groupby("class"):
            ax.hist(g[col], bins=bins, alpha=0.55,
                    label=f"{cls} (median {g[col].median():.0f})")
        ax.set_xscale("log")
        ticks = [t for t in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
                 if bins[0] <= t <= bins[-1]]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:,}" for t in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(label + ", log scale")
        ax.set_ylabel("Number of images")
        ax.set_title(f"{label} per class")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"hist_{col}.png", dpi=150)
        plt.close(fig)


def main():
    df = pd.read_csv("reports/audit_images.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    for cls in sorted(df["class"].unique()):
        sample_grid(df, cls)
    histograms(df)
    print("Saved:", *sorted(p.as_posix() for p in OUT.glob("*.png")), sep="\n- ")


if __name__ == "__main__":
    main()