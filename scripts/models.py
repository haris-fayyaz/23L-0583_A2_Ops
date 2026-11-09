import keras
from keras import layers

from pipeline import IMG_SIZE


def conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(momentum=0.9, name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    return layers.MaxPooling2D(name=f"{name}_pool")(x)


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_baseline(lr=1e-3, dropout=0.3):
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="image")
    x = layers.Rescaling(1 / 255.0, name="rescale")(inputs)
    for i, f in enumerate([32, 64, 128], start=1):
        x = conv_block(x, f, f"block{i}")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob_ai")(x)
    model = keras.Model(inputs, outputs, name="baseline_cnn")
    return compile_model(model, lr)