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
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

def model(hp, l2_reg=0.0):

    ativacao = hp.Choice("activation", ["relu", "elu"])
    neuronios_cnn = hp.Choice("neurons", [64, 128, 256])
    dropout = hp.Choice("dropout", [0.0, 0.25, 0.5])

    # Initialize model
    alexnet = Sequential()

    # Layer 1
    alexnet.add(
        Conv2D(
            96,
            (11, 11),
            input_shape=(96, 96, 3),
            padding="same",
            kernel_regularizer=l2(l2_reg),
        )
    )
    alexnet.add(BatchNormalization())
    alexnet.add(Activation(ativacao))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    alexnet.add(Conv2D(neuronios_cnn, (5, 5), padding="same"))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation(ativacao))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3, 4 e 5
    for i in range(hp.Int("cnn_layers", 0, 4)):
        multiplicador = 2 ** (int(i/2) + 1)
        alexnet.add(ZeroPadding2D((1, 1)))
        alexnet.add(Conv2D(neuronios_cnn * multiplicador, (3, 3), padding="same"))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation(ativacao))
        # se 1 é impar, então:
        if i % 2 == 1:
            alexnet.add(MaxPooling2D(pool_size=(2, 2)))
    
    # -------
    # Layer 6
    alexnet.add(Flatten())
    if hp.Boolean("dense_layer"):
        alexnet.add(Dense(hp.Choice("dense_1", [1024, 2048, 3072])))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation(ativacao))
        alexnet.add(Dropout(dropout))

    # Layer 7
    alexnet.add(Dense(hp.Choice("dense_2", [512, 1024, 2048])))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation(ativacao))
    alexnet.add(Dropout(dropout))

    # Layer 8
    alexnet.add(Dense(24))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation("softmax"))

    learning_rate = hp.Choice("learning_rate", [1e-1, 1e-2, 1e-3])
    optimizer_name = hp.Choice("optimizer", ["sgd", "rmsprop", "adam"])

    if optimizer_name == "sgd":
        optimizer = SGD(
            lr=learning_rate,
            momentum=0.9,
        )
    elif optimizer_name == "adam":
        optimizer = Adam(lr=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = RMSprop(lr=learning_rate, epsilon=0.9)

    alexnet.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

    return alexnet, alexnet.get_weights()
