import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras

SEED = 42
mlflow.set_tracking_uri("sqlite:///mlflow_toy.db")
mlflow.set_experiment("toy_fashion_mnist")

(x_all, y_all), _ = keras.datasets.fashion_mnist.load_data()
x_all = x_all.astype("float32") / 255.0
x_tr, y_tr = x_all[:10000], y_all[:10000]
x_val, y_val = x_all[10000:12000], y_all[10000:12000]


def run(lr, units=64, epochs=5):
    keras.utils.set_random_seed(SEED)
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
    with mlflow.start_run(run_name=f"lr_{lr}"):
        mlflow.log_params({"lr": lr, "units": units, "epochs": epochs, "seed": SEED})
        mlflow.set_tag("purpose", "toy")
        h = model.fit(
            x_tr, y_tr, epochs=epochs, batch_size=64,
            validation_data=(x_val, y_val), verbose=2,
        )
        for e in range(epochs):
            for k, v in h.history.items():
                mlflow.log_metric(k, float(v[e]), step=e + 1)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3))
        ax[0].plot(h.history["loss"], label="train")
        ax[0].plot(h.history["val_loss"], label="val")
        ax[0].set_title("Loss")
        ax[0].legend()
        ax[1].plot(h.history["accuracy"], label="train")
        ax[1].plot(h.history["val_accuracy"], label="val")
        ax[1].set_title("Accuracy")
        ax[1].legend()
        fig.suptitle(f"lr = {lr}")
        mlflow.log_figure(fig, "curves.png")
        plt.close(fig)
        mlflow.keras.log_model(model, "model")


run(1e-3)
run(1e-4)
print("done")