# https://github.com/eweill/keras-deepcv/blob/master/models/classification/alexnet.py

"""
AlexNet Keras Implementation with optimization parameters

BibTeX Citation:

@inproceedings{krizhevsky2012imagenet,
    title={Imagenet classification with deep convolutional neural networks},
    author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
    booktitle={Advances in neural information processing systems},
    pages={1097--1105},
    year={2012}
}
"""

# Import necessary components to build LeNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2


def model(l2_reg=0.0):
    # Initialize model
    alexnet = Sequential()

    # Layer 1
    alexnet.add(
        Conv2D(
            96,
            (11, 11),
            input_shape=(96, 96, 4),
            padding="same",
            kernel_regularizer=l2(l2_reg),
        )
    )
    alexnet.add(BatchNormalization())
    alexnet.add(Activation("relu"))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    alexnet.add(Dropout(0.25))

    # Layer 2
    alexnet.add(Conv2D(64, (5, 5), padding="same"))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation("relu"))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    alexnet.add(Dropout(0.25))

    # Layer 3, 4 e 5
    for i in range(3):
        multiplicador = 2 ** (int(i / 2) + 1)
        alexnet.add(ZeroPadding2D((1, 1)))
        alexnet.add(Conv2D(64 * multiplicador, (3, 3), padding="same"))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation("relu"))
        # se 1 é impar, então:
        if i % 2 == 1:
            alexnet.add(MaxPooling2D(pool_size=(2, 2)))

        alexnet.add(Dropout(0.25))

    # -------
    # Layer 6 (excluido)
    alexnet.add(Flatten())

    # Layer 7
    alexnet.add(Dense(32))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation("relu"))
    alexnet.add(Dropout(0.25))

    # Layer 8
    alexnet.add(Dense(24))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation("softmax"))

    return alexnet, alexnet.get_weights()
