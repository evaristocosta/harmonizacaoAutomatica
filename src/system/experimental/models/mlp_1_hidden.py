import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import (
    Input,
    Dense,
)


def model(params):
    model = Sequential()
    model.add(Input(shape=(params["input_shape"],)))
    model.add(Dense(params["neurons"], activation=params["activation"]))
    model.add(Dense(params["output_shape"], activation="softmax"))

    return model
