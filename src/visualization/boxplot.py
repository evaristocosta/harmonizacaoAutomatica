import sys
sys.path.insert(1, '/home/lucas/repos/harmonizacao/src/')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from analysis.friedman_test import load_losses

plt.style.use("seaborn-paper")

matplotlib.rc("xtick", labelsize=14)
matplotlib.rc("ytick", labelsize=14)

losses = load_losses()
tests = [
    "mlp_1_hidden",
    "mlp_2_hidden",
    "rbf",
    "elm",
    "esn",
    "ensemble_all",
    "ensemble_mlps_rbf",
    "ensemble_mlps",
]

plt.boxplot(losses, patch_artist=True, showfliers=True)
# plt.title('MSE por rodada')
plt.ylabel("Loss", fontsize=14)
# plt.xlabel('Rodada')
plt.xticks(np.arange(1, len(tests) +1 ), tests)

plt.show()
# plt.savefig('resultados_q2/boxplot_mlp_neuronios%s_base%s.svg' % (neuronios, base))
# plt.close()
