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
from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from keras.regularizers import l2


def model(params, l2_reg=0.0):

    # Initialize model
    alexnet = Sequential()

    # Layer 1
    alexnet.add(
        Conv2D(
            96,
            (11, 11),
            input_shape=eval(params["input_shape"]),
            padding="same",
            kernel_regularizer=l2(l2_reg),
        )
    )
    alexnet.add(BatchNormalization())
    alexnet.add(Activation(params["activation"]))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    alexnet.add(Conv2D(params["neurons"], (5, 5), padding="same"))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation(params["activation"]))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(params["neurons"] * 2, (3, 3), padding="same"))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation(params["activation"]))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 4
    if params["layer_4"] is True:
        alexnet.add(ZeroPadding2D((1, 1)))
        alexnet.add(Conv2D(params["neurons"] * 4, (3, 3), padding="same"))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation(params["activation"]))

    # Layer 5
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(params["neurons"] * 4, (3, 3), padding="same"))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation(params["activation"]))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(params["dense_1"]))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation(params["activation"]))
    alexnet.add(Dropout(params["dropout"]))

    # Layer 7
    alexnet.add(Dense(params["dense_2"]))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation(params["activation"]))
    alexnet.add(Dropout(params["dropout"]))

    # Layer 8
    alexnet.add(Dense(params["output_shape"]))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation("softmax"))

    return alexnet, alexnet.get_weights()
