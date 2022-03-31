import numpy as np
from predict import return_model
from load_data import carrega, separa

# selecionar modelos
selecoes = ["mlp_1_hidden", "elm", "esn"]

modelos = []
# carregar modelos
for selecao in selecoes:
    modelos.append(return_model(selecao))

# fazer predicoes com modelos
X, Y = carrega(data="encoded")
_, _, _, _, X, Y = separa(X, Y, ratio_train=0.7)

predicoes = []

for modelo in modelos:
    predicoes.append(np.squeeze(np.asarray(modelo.predict(X))))

# redefine pra iteracao
predicoes = np.transpose(np.array(predicoes), (1, 0, 2))

# escolha das respostas (soft voting ou wta)
predicao_ensemble = []

# soft voting:
""" for predicao in predicoes:
    soma = np.sum(predicao, axis=0) / len(predicao)
    predicao_ensemble.append(soma) """


# wta:
for predicao in predicoes:
    maiores = [np.max(maior) for maior in predicao]
    ganhador = np.argmax(maiores)
    predicao_ensemble.append(predicao[ganhador])

# predicao final
# resultados
