import os
import numpy as np
from imblearn.under_sampling import (
    RandomUnderSampler,
    NearMiss,
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    AllKNN,
    CondensedNearestNeighbour,
    OneSidedSelection,
    NeighbourhoodCleaningRule,
)
from collections import Counter

functions = [
    RandomUnderSampler(),
    NearMiss(),
    #EditedNearestNeighbours(),
    #RepeatedEditedNearestNeighbours(),
    #AllKNN(),
    CondensedNearestNeighbour(),
    #OneSidedSelection(),
    #NeighbourhoodCleaningRule(),
]


data = "separated"

X = np.load("data/" + data + "/vetor_entrada.npy", mmap_mode="r")
Y = np.load("data/" + data + "/vetor_saida.npy", mmap_mode="r")

original_input_shape = X.shape[1:]

X_reshaped = np.reshape(X, (X.shape[0], -1))

for ros in functions:
    print(ros)
    X_resampled, Y_resampled = ros.fit_resample(X_reshaped, Y)

    """ X_resampled_reshaped = np.reshape(
        X_resampled, ((X_resampled.shape[0],) + original_input_shape)
    )

    if not os.path.isdir("data/balanced"):
        os.mkdir("data/balanced")

    if os.path.isfile("data/balanced/vetor_entrada.npy"):
        os.remove("data/balanced/vetor_entrada.npy")
    if os.path.isfile("data/balanced/vetor_saida.npy"):
        os.remove("data/balanced/vetor_saida.npy")

    np.save("data/balanced/vetor_entrada.npy", X_resampled_reshaped)
    np.save("data/balanced/vetor_saida.npy", Y_resampled) """

    Y_cat = np.argmax(Y_resampled, axis=1)
    print(sorted(Counter(Y_cat).items()))


""" 
Resultados:

Origina = [(0, 4349), (1, 95), (2, 61), (3, 34), (4, 599), (5, 1192), (6, 94), (7, 43), (8, 464), (9, 348), (10, 1736), (11, 181), (12, 23), (13, 59), (14, 2443), (15, 168), (16, 101), (17, 14), (18, 358), (19, 1185), (20, 250), (21, 25), (22, 97), (23, 85)]
RandomUnderSampler = [(0, 14), (1, 14), (2, 14), (3, 14), (4, 14), (5, 14), (6, 14), (7, 14), (8, 14), (9, 14), (10, 14), (11, 14), (12, 14), (13, 14), (14, 14), (15, 14), (16, 14), (17, 14), (18, 14), (19, 14), (20, 14), (21, 14), (22, 14), (23, 14)]
NearMiss = [(0, 14), (1, 14), (2, 14), (3, 14), (4, 14), (5, 14), (6, 14), (7, 14), (8, 14), (9, 14), (10, 14), (11, 14), (12, 14), (13, 14), (14, 14), (15, 14), (16, 14), (17, 14), (18, 14), (19, 14), (20, 14), (21, 14), (22, 14), (23, 14)]
EditedNearestNeighbours = [(0, 951), (3, 4), (4, 18), (5, 27), (6, 8), (8, 9), (9, 2), (10, 89), (11, 4), (14, 325), (17, 14), (18, 8), (19, 37), (20, 1)]
RepeatedEditedNearestNeighbours = [(0, 951), (3, 4), (4, 18), (5, 27), (6, 8), (8, 9), (9, 2), (10, 89), (11, 4), (14, 325), (17, 14), (18, 8), (19, 37), (20, 1)]
AllKNN = [(0, 2749), (1, 19), (2, 15), (3, 8), (4, 142), (5, 320), (6, 20), (7, 12), (8, 116), (9, 62), (10, 696), (11, 49), (12, 1), (13, 9), (14, 1168), (15, 47), (16, 13), (17, 14), (18, 82), (19, 429), (20, 56), (21, 4), (22, 16), (23, 10)]
CondensedNearestNeighbour = [(0, 74), (1, 16), (2, 14), (3, 9), (4, 42), (5, 48), (6, 13), (7, 13), (8, 54), (9, 35), (10, 66), (11, 30), (12, 11), (13, 17), (14, 73), (15, 20), (16, 14), (17, 14), (18, 38), (19, 48), (20, 21), (21, 9), (22, 12), (23, 23)]
OneSidedSelection = [(0, 2519), (1, 65), (2, 42), (3, 26), (4, 395), (5, 772), (6, 82), (7, 35), (8, 361), (9, 281), (10, 1253), (11, 140), (12, 17), (13, 25), (14, 1866), (15, 109), (16, 64), (17, 14), (18, 257), (19, 862), (20, 124), (21, 22), (22, 64), (23, 44)]
NeighbourhoodCleaningRule = [(0, 3392), (1, 15), (2, 10), (3, 4), (4, 151), (5, 379), (6, 12), (7, 2), (8, 87), (9, 32), (10, 564), (11, 17), (12, 1), (13, 4), (14, 1004), (15, 8), (16, 3), (17, 14), (18, 30), (19, 185), (20, 13), (22, 5), (23, 4)]

import numpy as np
import matplotlib.pyplot as plt
from statistics import stdev

Origina.sort(key=lambda x: x[0], reverse=False)
RandomUnderSampler.sort(key=lambda x: x[0], reverse=False)
NearMiss.sort(key=lambda x: x[0], reverse=False)
# AllKNN.sort(key=lambda x: x[0], reverse=False)
CondensedNearestNeighbour.sort(key=lambda x: x[0], reverse=False)
# OneSidedSelection.sort(key=lambda x: x[0], reverse=False)


fig, axs = plt.subplots(2, 3, figsize=(20, 10))

def plot_samples(sampler, axs):
	classe = []
	qtde = []
	for item in sampler:
		classe.append(item[0])
		qtde.append(item[1])
	total = sum(qtde)
	print("total:", total)
	print("range:", max(qtde) - min(qtde))
	print("stdev:", stdev(qtde))
	qtde = [x / total for x in qtde]
	x_pos = np.arange(len(classe)) 
	axs.bar(x_pos, qtde, align='center')


plot_samples(Origina, axs[0, 0])
plot_samples(RandomUnderSampler, axs[0, 1])
plot_samples(NearMiss, axs[0, 2])
plot_samples(CondensedNearestNeighbour, axs[1, 1])
plt.show()


>>> plot_samples(Origina, axs[0, 0])
total: 14004
range: 4335
stdev: 1011.9418268208727
>>> plot_samples(RandomUnderSampler, axs[0, 1])
total: 336
range: 0
stdev: 0.0
>>> plot_samples(NearMiss, axs[0, 2])
total: 336
range: 0
stdev: 0.0
>>> plot_samples(CondensedNearestNeighbour, axs[1, 1])
total: 714
range: 65
stdev: 20.89778436352399


"""