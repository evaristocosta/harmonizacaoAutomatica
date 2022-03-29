import numpy as np
import gc
import time
import os
import keras
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
import talos

from load_data import carrega, separa
from models import *


def cross_val():
    agora = str(int(time.time()))
    teste = "mlp_1_hidden"
    caminho = "src/results/" + teste + "_" + agora + "/"
    if not os.path.isdir(caminho):
        os.mkdir(caminho)
        os.mkdir(caminho + "pesos/")
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
        "model": "mlp_1_hidden",
    }
    open(caminho + "params.txt", "w").write(str(params))

    # controle
    loss_por_fold = []
    acc_por_fold = []
    predicao_por_fold = []
    real_por_fold = []

    X_train, Y_train, X_val, Y_val, X_test, Y_test = separa(X, Y, ratio_train=0.7)
    repetitions = 30

    for rodada in range(repetitions):
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
        metricas = modelo.evaluate(
            X_test, Y_test, verbose=1, batch_size=params["batch_size"]
        )
        print("\n==========\nResultados para fold " + str(rodada + 1) + ":\n")
        log.write(f"\n{rodada+1},")

        for metrica in modelo.metrics_names:
            print(metrica + ": " + str(metricas[modelo.metrics_names.index(metrica)]))
            log.write(str(metricas[modelo.metrics_names.index(metrica)]) + ",")

        loss_por_fold.append(metricas[0])
        acc_por_fold.append(metricas[1])

        predicao = modelo.predict(X_test)
        # transforma em array
        predicao = np.squeeze(np.asarray(predicao))
        predicao = np.argmax(predicao, axis=1)
        Y_categorico = np.argmax(Y_test, axis=1)

        predicao_por_fold.append(predicao)
        real_por_fold.append(Y_categorico)

        # https://forums.fast.ai/t/how-could-i-release-gpu-memory-of-keras/2023/7
        tf.keras.backend.clear_session()
        gc.collect()


if __name__ == "__main__":
    cross_val()
