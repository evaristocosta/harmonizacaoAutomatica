import numpy as np
from tensorflow.keras.optimizers import SGD, Adam
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import (
    Input,
    Dense,
)


def mlp_2_hidden(params):
    model = Sequential()
    model.add(Input(shape=(params["input_shape"],)))
    model.add(Dense(params["neurons"], activation=params["activation"]))
    model.add(Dense(params["neurons"], activation=params["activation"]))
    model.add(Dense(params["output_shape"], activation="softmax"))

    return model
