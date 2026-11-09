import keras
import keras_tuner as kt
import pandas as pd
from keras import layers

from models import compile_model
from pipeline import SEED, make_ds, seed_everything

EPOCHS = 4
MAX_TRIALS = 8

seed_everything()
train_ds = make_ds("train", training=True)
val_ds = make_ds("val")


def build_model(hp):
    lr = hp.Float("lr", 1e-5, 1e-3, sampling="log")
    dropout = hp.Float("dropout", 0.0, 0.5, step=0.1)
    unfrozen = hp.Choice("unfrozen", [0, 11, 33, 55])

    keras.utils.set_random_seed(SEED)
    model = keras.models.load_model("models/mobilenet_s1.keras")
    model.get_layer("dropout").rate = dropout
    assert model.get_layer("dropout").rate == dropout

    base = model.get_layer("mobilenetv2_base")
    if unfrozen == 0:
        base.trainable = False
    else:
        base.trainable = True
        start = len(base.layers) - unfrozen
        for i, layer in enumerate(base.layers):
            layer.trainable = i >= start and not isinstance(
                layer, layers.BatchNormalization
            )
    compile_model(model, lr)
    n = sum(w.numpy().size for w in model.trainable_weights)
    print(f"unfrozen={unfrozen} dropout={dropout} lr={lr:.6f} trainable={n:,}")
    return model


tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=MAX_TRIALS,
    seed=SEED,
    directory="tuner_runs",
    project_name="visionguard",
    overwrite=True,
)
tuner.search(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    verbose=2,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ],
)

rows = []
for t in tuner.oracle.trials.values():
    v = t.hyperparameters.values
    loss = [o.value[0] for o in t.metrics.get_history("val_loss")]
    auc = [o.value[0] for o in t.metrics.get_history("val_auc")]
    acc = [o.value[0] for o in t.metrics.get_history("val_accuracy")]
    b = loss.index(min(loss))
    rows.append(
        {
            "trial": t.trial_id,
            "unfrozen": v["unfrozen"],
            "dropout": round(v["dropout"], 1),
            "lr": round(v["lr"], 6),
            "best_epoch": b + 1,
            "val_loss": round(loss[b], 4),
            "val_auc": round(auc[b], 4),
            "val_acc": round(acc[b], 4),
        }
    )
df = pd.DataFrame(rows).sort_values("val_loss")
df.to_csv("reports/tuner_vg_results.csv", index=False)
print()
print(df.to_string(index=False))
print("Best settings:", tuner.get_best_hyperparameters(1)[0].values)