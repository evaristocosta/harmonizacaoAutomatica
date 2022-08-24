from keras.models import Sequential
from keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
)

""" p = {
    "activation": "elu",
    "dropout": 0.30,
    "dense_1": 128,
    "dense_2": 128,
    "lr": 0.1,
    "optimizer": Adam,
    "batch_size": 32,
    "epochs": 300,
} """


def model(params):
    model = Sequential()

    model.add(
        Conv2D(
            32,
            kernel_size=3,
            activation=params["activation"],
            input_shape=(96, 96, 4),
        )
    )
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(params["dropout"]))

    model.add(Conv2D(64, kernel_size=3, activation=params["activation"]))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(params["dropout"]))

    model.add(Conv2D(128, kernel_size=3, activation=params["activation"]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(params["dropout"]))

    model.add(Flatten())
    model.add(Dense(params["dense_1"], activation=params["activation"]))
    model.add(BatchNormalization())
    model.add(Dropout(params["dropout"]))

    model.add(Dense(params["dense_2"], activation=params["activation"]))
    model.add(BatchNormalization())
    model.add(Dropout(params["dropout"]))

    model.add(Dense(24, activation="softmax"))

    return model, model.get_weights()
