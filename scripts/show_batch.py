from pathlib import Path

import matplotlib.pyplot as plt

from pipeline import make_ds, seed_everything

seed_everything()
x, y = next(iter(make_ds("train", training=True)))

fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
for ax, img, lab in zip(axes.ravel(), x[:8], y[:8]):
    ax.imshow(img.numpy().clip(0, 255).astype("uint8"))
    ax.set_title("AI" if lab == 1 else "Real")
    ax.axis("off")
fig.suptitle("One augmented training batch (first 8 of 32), 224x224")
fig.tight_layout()

Path("reports").mkdir(exist_ok=True)
fig.savefig("reports/aug_batch.png", dpi=150)
print("saved reports/aug_batch.png")