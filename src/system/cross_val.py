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
from keras.callbacks import ModelCheckpoint, CSVLogger
import talos

from load_data import carrega, separa
from models import *
from performance_measures import print_basic_performance, calc_accuracy, calc_log_loss


parser = argparse.ArgumentParser(description="Cross validate models")
parser.add_argument(
    "--model",
    help="Which model to train",
    default="mlp_1_hidden",
    choices=["mlp_1_hidden", "mlp_2_hidden", "rbf", "esn", "elm", "cnn_like_alexnet"],
)
parser.add_argument(
    "--repetitions",
    help="How many cross validation repetions",
    default=30,
    type=int,
)
args = parser.parse_args()

MODEL = args.model
REPETITIONS = args.repetitions


def cross_val():
    agora = str(int(time.time()))
    teste = MODEL
    caminho = "src/system/results/" + teste + "_" + agora + "/"
    if not os.path.isdir(caminho):
        os.mkdir(caminho)
        os.mkdir(caminho + "pesos/")
        os.mkdir(caminho + "modelos/")
        os.mkdir(caminho + "logs/")
        os.mkdir(caminho + "output/")

    log = open(caminho + "log_resultados.csv", "w")
    log.write("rodada,loss,accuracy")

    X, Y = carrega(data="encoded")
    input_shape = X.shape[1]
    output_shape = Y.shape[1]

    params = {
        "input_shape": input_shape,
        "output_shape": output_shape,
        "neurons": 64,  # 128, 256
        "activation": "tanh",
        "batch_size": 128,
        "epochs": 10,
        "optimizer": SGD,
        "learning_rate": 0.001 * 100.0,
        "model": MODEL,
    }

    # controle
    loss_por_fold = []
    acc_por_fold = []
    predicao_por_fold = []
    real_por_fold = []

    X_train, Y_train, X_val, Y_val, X_test, Y_test = separa(X, Y, ratio_train=0.7)
    repetitions = REPETITIONS

    for rodada in range(repetitions):
        if params["model"] != "elm" and params["model"] != "esn":
            if params["model"] == "mlp_1_hidden":
                modelo = mlp_1_hidden.model(params)
            elif params["model"] == "mlp_2_hidden":
                modelo = mlp_2_hidden.model(params)
            elif params["model"] == "rbf":
                modelo = rbf.model(params, X_train)
            elif params["model"] == "cnn_like_alexnet":
                modelo = cnn_like_alexnet.model(params)

            modelo.compile(
                loss="categorical_crossentropy",
                optimizer=params["optimizer"](
                    lr=talos.utils.lr_normalizer(
                        params["learning_rate"], params["optimizer"]
                    )
                ),
                metrics=["accuracy"],
            )

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
            lista_callbacks = [checkpoint, train_log]

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

            with open(caminho + "pesos/" + str(rodada + 1) + ".pkl", "wb") as f:
                pickle.dump(modelo, f)

        predicao = modelo.predict(X_test)

        loss = calc_log_loss(Y_test, predicao)
        acc = calc_accuracy(Y_test, predicao)

        # print basic performance
        print_basic_performance(Y_test, predicao)

        log.write(f"\n{rodada+1},{loss},{acc}")

        loss_por_fold.append(loss)
        acc_por_fold.append(acc)

        predicao_por_fold.append(predicao)
        real_por_fold.append(Y_test)

        del modelo

        # https://forums.fast.ai/t/how-could-i-release-gpu-memory-of-keras/2023/7
        tf.keras.backend.clear_session()
        gc.collect()

    log.close()

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
