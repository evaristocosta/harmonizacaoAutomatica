import numpy as np
import gc
import keras
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam 
from keras.callbacks import ModelCheckpoint, CSVLogger


from load_data import carrega
from models import *


# controle
loss_por_fold = []
acc_por_fold = []
predicao_por_fold = []
real_por_fold = []

r = open("log_resultados.csv", "w")
r.write("rodada,loss,accuracy,f1score")

X_train, Y_train, X_val, Y_val, X_test, Y_test = carrega()
input_shape = X_train.shape[1]
output_shape = Y_train.shape[1]

params = {
    "input_shape": input_shape,
    "output_shape": output_shape,
    "neurons": 128,
    "activation": "relu",
    "batch_size": 128,
    "epochs": 10,
    "optimizer": Adam,
    "learning_rate": 0.01,
    "model": "mlp_1_hidden",
}

num_folds = 5

for rodada in range(num_folds):
    if params["model"] == "mlp_1_hidden":
        modelo = mlp_1_hidden.model(params)
    elif params["model"] == "mlp_2_hidden":
        modelo = mlp_2_hidden.model(params)
    elif params["model"] == "rbf":
        modelo = rbf.model(params, X_train)
    elif params["model"] == "cnn_like_alexnet":
        modelo = cnn_like_alexnet.model(params)

    checkpoint = ModelCheckpoint(
        str(rodada + 1) + ".h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
    )
    log = CSVLogger( str(rodada + 1) + ".csv", append=False)
    lista_callbacks = [checkpoint, log]
    
    modelo.fit(
        X_train,
        Y_train,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        validation_data=(X_val, Y_val),
        callbacks=lista_callbacks,
    )

    # Carrega melhor modelo
    modelo.load_weights( str(rodada + 1) + ".h5")
    metricas = modelo.evaluate(X_test, Y_test, verbose=1, batch_size=params["batch_size"])
    print("\n==========\nResultados para fold " + str(rodada + 1) + ":\n")
    r.write(f"\n{rodada+1},")

    for metrica in modelo.metrics_names:
        print(metrica + ": " + str(metricas[modelo.metrics_names.index(metrica)]))
        r.write(str(metricas[modelo.metrics_names.index(metrica)]) + ",")
    loss_por_fold.append(metricas[0])
    acc_por_fold.append(metricas[1])

    # salva o modelo
    modelo_json = modelo.to_json()
    caminho_modelo =  str(rodada + 1) + ".json"
    open(caminho_modelo, "w").write(modelo_json)

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