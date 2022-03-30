import argparse
import os
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import talos

from sklearn.metrics import accuracy_score


from load_data import carrega, separa
from models import *


parser = argparse.ArgumentParser(description="Model fit and train")
parser.add_argument(
    "--model",
    help="Which model to train",
    default="mlp_1_hidden",
    choices=["mlp_1_hidden", "mlp_2_hidden", "rbf", "esn", "elm", "cnn_like_alexnet"],
)
parser.add_argument(
    "--optimizer",
    help="What optimizer to use",
    default="sgd",
    choices=["sgd", "adam"],
)
args = parser.parse_args()

MODEL = args.model
OPTIMIZER = args.optimizer


def fit():
    X, Y = carrega(data="encoded")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = separa(X, Y, ratio_train=0.7)
    input_shape = X_train.shape[1]
    output_shape = Y_train.shape[1]

    params = {
        "input_shape": input_shape,
        "output_shape": output_shape,
        "neurons": 64,  # 128, 256
        "activation": "tanh",
        "batch_size": 1,
        "epochs": 300,
        "optimizer": SGD,
        "learning_rate": 0.001 * 100.0,
        "model": "mlp_1_hidden",
    }

    if MODEL != "elm" and MODEL != "esn":
        if MODEL == "mlp_1_hidden":
            params["model"] = "mlp_1_hidden"
        elif MODEL == "mlp_2_hidden":
            params["model"] = "mlp_2_hidden"
        elif MODEL == "rbf":
            params["model"] = "rbf"
        elif MODEL == "cnn_like_alexnet":
            params["model"] = "cnn_like_alexnet"

        if OPTIMIZER == "sgd":
            params["optimizer"] = SGD
        elif OPTIMIZER == "adam":
            params["optimizer"] = Adam
        _, modelo = model_fit(X_train, Y_train, X_val, Y_val, params)

    else:
        if MODEL == "elm":
            modelo = elm.ELM(
                params["input_shape"], params["output_shape"], params["neurons"]
            )
            modelo.train(X_train, Y_train)

        elif MODEL == "esn":
            modelo = esn.ESN(
                n_inputs=params["input_shape"],
                n_outputs=params["output_shape"],
            )
            modelo.fit(X_train, Y_train)

    # igual pra todos
    predicao = modelo.predict(X_test)

    # transforma em array
    predicao = np.squeeze(np.asarray(predicao))
    predicao = np.argmax(predicao, axis=1)
    Y_test = np.argmax(Y_test, axis=1)

    acerto = accuracy_score(Y_test, predicao)

    print("Acerto:", acerto)


def model_fit(X_train, Y_train, X_val, Y_val, params):
    if params["model"] == "mlp_1_hidden":
        modelo = mlp_1_hidden.model(params)
    elif params["model"] == "mlp_2_hidden":
        modelo = mlp_2_hidden.model(params)
    elif params["model"] == "rbf":
        modelo = rbf.model(params, X_train)
    elif params["model"] == "cnn_like_alexnet":
        modelo = cnn_like_alexnet.model(params)

    modelo.summary()

    modelo.compile(
        loss="categorical_crossentropy",
        optimizer=params["optimizer"](
            lr=talos.utils.lr_normalizer(params["learning_rate"], params["optimizer"])
        ),
        metrics=["accuracy"],
    )

    checkpoint = ModelCheckpoint(
        "modelo.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
    )

    historico = modelo.fit(
        X_train,
        Y_train,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        validation_data=(X_val, Y_val),
        callbacks=[checkpoint],
    )

    # LOAD BEST MODEL to evaluate the performance of the model
    modelo.load_weights("modelo.h5")
    os.remove("modelo.h5")

    return historico, modelo


if __name__ == "__main__":
    fit()
