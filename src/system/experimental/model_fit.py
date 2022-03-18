import argparse
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score

from load_data import carrega
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
    X_train, Y_train, X_val, Y_val, X_test, Y_test = carrega()
    input_shape = X_train.shape[1]
    output_shape = Y_train.shape[1]

    params = {
        "input_shape": input_shape,
        "output_shape": output_shape,
        "neurons": 128,
        "activation": "relu",
        "learning_rate": 0.01,
        "batch_size": 128,
        "epochs": 10,
    }

    if MODEL != "elm" and MODEL != "esn":
        if MODEL == "mlp_1_hidden":
            modelo = mlp_1_hidden.model(params)
        elif MODEL == "mlp_2_hidden":
            modelo = mlp_2_hidden.model(params)
        elif MODEL == "rbf":
            modelo = rbf.model(params, X_train)
        elif MODEL == "cnn_like_alexnet":
            modelo = cnn_like_alexnet.model(params)

        if OPTIMIZER == "sgd":
            optimizer = SGD(lr=params["learning_rate"])
        elif OPTIMIZER == "adam":
            optimizer = Adam(lr=params["learning_rate"])

        modelo.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            metrics=["categorical_accuracy"],
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

        metricas = modelo.evaluate(X_test, Y_test, verbose=0)
        print(
            f"Resultados:\n {modelo.metrics_names[0]} de {metricas[0]};\n {modelo.metrics_names[1]} de {metricas[1]*100}%"
        )
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
        
        historico = ""

    # igual pra todos
    predicao = modelo.predict(X_test)

    # transforma em array
    predicao = np.squeeze(np.asarray(predicao))
    predicao = np.argmax(predicao, axis=1)
    Y_test = np.argmax(Y_test, axis=1)

    acerto = accuracy_score(Y_test, predicao)

    print("Acerto:", acerto)

    return predicao, Y_test, historico


if __name__ == "__main__":
    fit()
