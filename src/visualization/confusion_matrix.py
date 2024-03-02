import sys

sys.path.insert(1, "/home/lucas/harmonizacaoAutomatica/src/")

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={"figure.figsize": (8, 7)})
sns.set_theme(context="talk", style="white")
np.set_printoptions(threshold=np.inf)


def main():
    parser = argparse.ArgumentParser(description="Plot date based confusion matrix")
    parser.add_argument(
        "-d",
        "--date",
        help="Training date to be considered",
        type=int,
    )
    args = parser.parse_args()

    DATE = args.date

    dicio_acordes = [
        "C maj",
        "C min",
        "C# maj",
        "C# min",
        "D maj",
        "D min",
        "D# maj",
        "D# min",
        "E maj",
        "E min",
        "F maj",
        "F min",
        "F# maj",
        "F# min",
        "G maj",
        "G min",
        "G# maj",
        "G# min",
        "A maj",
        "A min",
        "A# maj",
        "A# min",
        "B maj",
        "B min",
    ]
    class_names = np.array(dicio_acordes)

    plot_confusion_matrix(classes=class_names, date=DATE, normalize=True)
    plt.grid(linestyle="--", linewidth=0.5)
    plt.show()


def plot_confusion_matrix(
    classes,
    date,
    normalize=True,
    anotacoes=False,
    # pylint: disable=no-member
    cmap=plt.cm.bone_r,
):
    """
    melhores:
    - bone_r
    - ocean_r *
    """

    matplotlib.rc("xtick", labelsize=12)
    matplotlib.rc("ytick", labelsize=12)
    np.set_printoptions(precision=2)

    df = pd.read_csv("src/system/results/summary.csv")
    df = df[df["date"] == date]

    # pega informações da melhor execução do experimento
    experiment = str(df["experiment"].values[0])
    best_run = df["best_run"].values[0] - 1

    print("Experiment:", experiment)
    path = "src/system/results/" + experiment + "_" + str(date) + "/"

    y_true = np.load(path + "output/real_por_fold.npy")
    y_pred = np.load(path + "output/predicao_por_fold.npy")

    y_true_cat = np.argmax(y_true[best_run], axis=1)
    y_pred_cat = np.argmax(y_pred[best_run], axis=1)

    cm = confusion_matrix(y_true_cat, y_pred_cat)

    if normalize:
        cm = cm / cm.astype(float).sum(axis=1)

    # inicia construcao da imagem
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
    )
    ax.set_ylabel("True label", fontsize=17)
    ax.set_xlabel("Predicted label", fontsize=17)

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(),
        rotation=90,
        ha="right",
        va="center",
        rotation_mode="anchor",
    )

    # Loop over data dimensions and create text annotations.
    if anotacoes:
        fmt = ".1f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt) + "%",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

    fig.tight_layout()
    return ax


if __name__ == "__main__":
    main()
