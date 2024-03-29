import sys
sys.path.insert(1, "/home/lucas/harmonizacaoAutomatica/src/")

import gc
import json
import time
import os
import argparse
import pickle
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, CSVLogger
import talos
from npy_append_array import NpyAppendArray

from load_data import carrega, separa, carrega_arquivo
from models import *
from analysis.performance_measures import (
    print_basic_performance,
    calc_accuracy,
    calc_log_loss,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

parser = argparse.ArgumentParser(description="Cross validate models")
parser.add_argument(
    "--model",
    help="Which model to train",
    default="mlp_1_hidden",
    choices=[
        "mlp_1_hidden",
        "mlp_2_hidden",
        "rbf",
        "esn",
        "elm",
        "cnn_like_alexnet",
        "ensemble",
        "vgg16",
        "resnet101",
        "inceptionv3",
        "densenet201",
    ],
)
parser.add_argument(
    "--repetitions",
    help="How many cross validation repetions",
    default=30,
    type=int,
)
parser.add_argument(
    "--neurons",
    help="How many neurons in the hidden layer",
    default=64,
    type=int,
)
args = parser.parse_args()

MODEL = args.model
REPETITIONS = args.repetitions
NEURONS = args.neurons


# https://stackoverflow.com/a/67138072
class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()


def init_weight(same_old_model, first_weights):
    # https://stackoverflow.com/a/45785545
    # we can uncomment the line below to reshufle the weights themselves so they are not exactly the same between folds
    weights = [np.random.permutation(x.flat).reshape(x.shape) for x in first_weights]

    same_old_model.set_weights(weights)


def cross_val():
    agora = str(int(time.time()))
    teste = MODEL
    caminho = "src/system/results/" + teste + "_" + agora + "/"
    if not os.path.isdir(caminho):
        os.makedirs(caminho)
        os.makedirs(caminho + "pesos/")
        os.makedirs(caminho + "modelos/")
        os.makedirs(caminho + "logs/")
        os.makedirs(caminho + "output/")

    log = open(caminho + "log_resultados.csv", "w")
    log.write("rodada,loss,accuracy")
    log.close()

    # X, Y = carrega(data="encoded")
    # X_train, Y_train, X_val, Y_val, X_test, Y_test = separa(X, Y, ratio_train=0.7)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = carrega_arquivo()

    input_shape = X_train.shape[1:]
    output_shape = Y_train.shape[1]

    params = {
        "input_shape": input_shape,
        "output_shape": output_shape,
        "neurons": NEURONS,  # 64, 128, 256
        "activation": "sigmoid",
        "batch_size": 512,
        "epochs": 2,
        "optimizer": Adam,
        "learning_rate": 0.001 * 100.0,
        "model": MODEL,
        "ensemble_models": ["mlp_1_hidden", "mlp_2_hidden"],
        "ensemble_voting": "wta",
        "application": MODEL,
    }

    # controle
    loss_por_fold = []
    acc_por_fold = []
    predicao_por_fold = []
    real_por_fold = []

    repetitions = REPETITIONS

    if params["model"] not in ["elm", "esn", "ensemble"]:
        if params["model"] == "mlp_1_hidden":
            modelo, pesos = mlp_1_hidden.model(params)
        elif params["model"] == "mlp_2_hidden":
            modelo, pesos = mlp_2_hidden.model(params)
        elif params["model"] == "rbf":
            modelo, pesos = rbf.model(params, X_train)
        elif params["model"] == "cnn_like_alexnet":
            modelo, pesos = cnn_like_alexnet.model(params)
        elif params["model"] in ["vgg16", "resnet101", "inceptionv3", "densenet201"]:
            X_train, X_val, X_test = keras_application.preprocess(
                X_train, X_val, X_test, params
            )
            modelo, pesos = keras_application.model(params)

        modelo.compile(
            loss="categorical_crossentropy",
            optimizer=params["optimizer"](
                lr=talos.utils.lr_normalizer(
                    params["learning_rate"], params["optimizer"]
                )
            ),
            metrics=["accuracy"],
            # run_eagerly=True,  # https://stackoverflow.com/a/67138072
        )

    for rodada in range(repetitions):
        if params["model"] not in ["elm", "esn", "ensemble"]:
            init_weight(modelo, pesos)

            checkpoint = ModelCheckpoint(
                caminho + "pesos/" + str(rodada + 1) + ".h5",
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="min",
            )
            train_log = CSVLogger(
                caminho + "logs/" + str(rodada + 1) + ".csv", append=False
            )
            lista_callbacks = [
                checkpoint,
                train_log,
                ClearMemory(),
            ]  # https://stackoverflow.com/a/67138072

            modelo.fit(
                X_train,
                Y_train,
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                validation_data=(X_val, Y_val),
                callbacks=lista_callbacks,
            )

            # Carrega melhor modelo
            modelo.load_weights(caminho + "pesos/" + str(rodada + 1) + ".h5")
            modelo.save(caminho + "modelos/" + str(rodada + 1) + ".h5")

        else:
            if params["model"] == "elm":
                modelo = elm.ELM(
                    params["input_shape"], params["output_shape"], params["neurons"]
                )
                modelo.train(X_train, Y_train)

            elif params["model"] == "esn":
                modelo = esn.ESN(
                    n_inputs=params["input_shape"],
                    n_outputs=params["output_shape"],
                )
                modelo.fit(X_train, Y_train)

            elif params["model"] == "ensemble":
                modelo = ensemble.Ensemble(
                    params["ensemble_voting"], params["ensemble_models"]
                )

            with open(caminho + "pesos/" + str(rodada + 1) + ".pkl", "wb") as f:
                pickle.dump(modelo, f)

        predicao = modelo.predict(X_test)

        loss = calc_log_loss(Y_test, predicao)
        acc = calc_accuracy(Y_test, predicao)

        # print basic performance
        print_basic_performance(Y_test, predicao)

        log = open(caminho + "log_resultados.csv", "a")
        log.write(f"\n{rodada+1},{loss},{acc}")
        log.close()

        loss_por_fold.append(loss)
        acc_por_fold.append(acc)

        predicao_por_fold.append(np.array(predicao))
        real_por_fold.append(np.array(Y_test))

        del predicao

        # https://forums.fast.ai/t/how-could-i-release-gpu-memory-of-keras/2023/7
        gc.collect()
        tf.keras.backend.clear_session()

    params["optimizer"] = str(params["optimizer"])
    with open(caminho + "params.json", "w") as f:
        json.dump(params, f)

    np.save(caminho + "output/predicao_por_fold.npy", predicao_por_fold)
    np.save(caminho + "output/real_por_fold.npy", real_por_fold)

    melhor_fold_loss = loss_por_fold.index(min(loss_por_fold))
    print("Melhor fold:", melhor_fold_loss + 1)

    sumario = open("src/system/results/summary.csv", "a")
    sumario.write(
        f"{agora},{teste},{melhor_fold_loss + 1},{loss_por_fold[melhor_fold_loss]},{acc_por_fold[melhor_fold_loss]}\n"
    )
    sumario.close()


if __name__ == "__main__":
    cross_val()
