import pickle
import json
import re
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from keras.models import load_model

from load_data import carrega, separa
from models.rbf import RBFLayer

experiment = "mlp_1_hidden"
date = "1648674681"
best_run = "2"
path = "src/system/results/" + experiment + "_" + date + "/"

X, Y = carrega(data="encoded")
X_train, Y_train, X_val, Y_val, X_test, Y_test = separa(X, Y, ratio_train=0.7)


if experiment != "elm" and experiment != "esn":
    if experiment != "rbf":
        modelo = load_model(path + "modelos/" + best_run + ".h5")
    else:
        modelo = load_model(path + "modelos/" + best_run + ".h5", custom_objects={'RBFLayer': RBFLayer})

    with open(path + "params.json") as json_file:
        params = json.load(json_file)

    regex = r"\.([^.]*)'>"
    opt = re.search(regex, str(params["optimizer"])).group(1)

else:
    file = open(path + "pesos/" + best_run + ".pkl", "rb")
    modelo = pickle.load(file)
    file.close()



predicao = modelo.predict(X_test)
predicao = np.squeeze(np.asarray(predicao))
predicao_categorico = np.argmax(predicao, axis=1)
Y_categorico = np.argmax(Y_test, axis=1)

loss = log_loss(Y_categorico, predicao)
acc = accuracy_score(Y_categorico, predicao_categorico)
print("loss: ", loss)
print("acc: ", acc)