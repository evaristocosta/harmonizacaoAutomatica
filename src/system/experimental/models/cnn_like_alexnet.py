from keras.models import Sequential
from keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
)


def model(params):
    model = Sequential(
        [
            Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=params["input_shape"],
            ),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(128, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(params["output_shape"], activation="softmax"),
        ]
    )
    return model
