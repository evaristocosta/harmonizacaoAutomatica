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
    InstanceHardnessThreshold,
)
from collections import Counter

data = "separated"

X = np.load("data/" + data + "/vetor_entrada.npy", mmap_mode="r")
Y = np.load("data/" + data + "/vetor_saida.npy", mmap_mode="r")

original_input_shape = X.shape[1:]

X_reshaped = np.reshape(X, (X.shape[0], -1))

ros = CondensedNearestNeighbour(random_state=0)
X_resampled, Y_resampled = ros.fit_resample(X_reshaped, Y)

X_resampled_reshaped = np.reshape(
    X_resampled, ((X_resampled.shape[0],) + original_input_shape)
)

Y_cat = np.argmax(Y_resampled, axis=1)
print(sorted(Counter(Y_cat).items()))
