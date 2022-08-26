import sys

sys.path.insert(1, "/home/lucas/repos/harmonizacao/src/")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from analysis.friedman_test import load_losses

plt.style.use("seaborn-paper")

matplotlib.rc("xtick", labelsize=14)
matplotlib.rc("ytick", labelsize=14)

losses = load_losses()
tests = [
    "MLP1",
    "MLP2",
    "RBF",
    "ELM",
    "ESN",
    "ENS5",
    "ENS3",
    "ENS2",
]

plt.boxplot(losses, patch_artist=True, showfliers=True)
plt.ylabel("Loss", fontsize=14)
plt.xticks(np.arange(1, len(tests) + 1), tests)

plt.show()
