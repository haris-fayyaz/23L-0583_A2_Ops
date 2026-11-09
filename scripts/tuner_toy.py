import keras
import keras_tuner as kt
import pandas as pd

SEED = 42
keras.utils.set_random_seed(SEED)

(x_all, y_all), _ = keras.datasets.fashion_mnist.load_data()
x_all = x_all.astype("float32") / 255.0
x_tr, y_tr = x_all[:10000], y_all[:10000]
x_val, y_val = x_all[10000:12000], y_all[10000:12000]


def build_model(hp):
    units = hp.Int("units", 32, 128, step=32)
    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")
    model = keras.Sequential(
        [
            keras.layers.Input((28, 28)),
            keras.layers.Flatten(),
            keras.layers.Dense(units, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=6,
    seed=SEED,
    directory="tuner_runs",
    project_name="toy_fashion",
    overwrite=True,
)
tuner.search(
    x_tr, y_tr, epochs=3, batch_size=64,
    validation_data=(x_val, y_val), verbose=2,
)

rows = []
for t in tuner.oracle.trials.values():
    rows.append(
        {
            "trial": t.trial_id,
            "units": t.hyperparameters.values["units"],
            "lr": round(t.hyperparameters.values["lr"], 5),
            "val_accuracy": round(t.score, 4),
        }
    )
df = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
df.to_csv("reports/tuner_toy_results.csv", index=False)
print()
print(df.to_string(index=False))
print("Best settings:", tuner.get_best_hyperparameters(1)[0].values)