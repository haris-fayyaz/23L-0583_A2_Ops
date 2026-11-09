import keras
from keras import layers

from models import compile_model
from pipeline import SEED, make_ds, seed_everything

seed_everything()
train_ds = make_ds("train", training=True)
val_ds = make_ds("val")


def show(model, tag):
    r = model.evaluate(val_ds, verbose=0, return_dict=True)
    print(f"{tag}: val_loss={r['loss']:.4f} acc={r['accuracy']:.4f} auc={r['auc']:.4f}")


for name in ["mobilenet_s1", "mobilenet_s2"]:
    m = keras.models.load_model(f"models/{name}.keras")
    compile_model(m, 1e-5)
    show(m, f"saved {name}")


def one_epoch(tag, unfrozen, dropout, lr):
    keras.utils.set_random_seed(SEED)
    m = keras.models.load_model("models/mobilenet_s1.keras")
    m.get_layer("dropout").rate = dropout
    base = m.get_layer("mobilenetv2_base")
    base.trainable = True
    start = len(base.layers) - unfrozen
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= start and not isinstance(
            layer, layers.BatchNormalization
        )
    compile_model(m, lr)
    n = sum(w.numpy().size for w in m.trainable_weights)
    print(f"{tag}: trainable={n:,}")
    m.fit(train_ds, epochs=1, validation_data=val_ds, verbose=2)
    show(m, f"{tag} after 1 epoch")


one_epoch("tuner best", 55, 0.2, 4.126800832382481e-05)
one_epoch("stage2-like", 11, 0.3, 1e-5)