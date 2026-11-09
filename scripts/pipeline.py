from pathlib import Path

import keras
import pandas as pd
import tensorflow as tf

SEED = 42
IMG_SIZE = (224, 224)
BATCH = 32
ROOT = Path("data/raw/archive")
SPLITS = Path("splits")
AUTOTUNE = tf.data.AUTOTUNE

augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomContrast(0.1),
])


def seed_everything(seed=SEED):
    keras.utils.set_random_seed(seed)


def load_image(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE, antialias=True)
    return img, tf.cast(label, tf.float32)


def make_ds(split, training=False):
    df = pd.read_csv(SPLITS / f"{split}.csv")
    paths = [(ROOT / p).as_posix() for p in df["path"]]
    ds = tf.data.Dataset.from_tensor_slices((paths, df["label"].values))
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH)
    if training:
        ds = ds.map(lambda x, y: (augment(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


if __name__ == "__main__":
    import time

    seed_everything()
    for split in ["train", "val", "test"]:
        t = time.time()
        ds = make_ds(split, training=(split == "train"))
        n, first, lo, hi = 0, None, 255.0, 0.0
        for x, y in ds:
            n += x.shape[0]
            first = first or x.shape
            lo = min(lo, float(tf.reduce_min(x)))
            hi = max(hi, float(tf.reduce_max(x)))
        print(split, n, tuple(first), x.dtype.name, round(lo, 1), round(hi, 1),
              f"{time.time() - t:.0f}s")