import glob
import csv
import numpy as np
from collections import Counter
import argparse
import matplotlib
import matplotlib.pyplot as plt

plt.style.use("seaborn-paper")

matplotlib.rc("xtick", labelsize=14)
matplotlib.rc("ytick", labelsize=14)

parser = argparse.ArgumentParser(description="Visualization of total notes and chords")
parser.add_argument(
    "--data",
    type=str,
    default="standardized",
    help="Data to visualize",
    choices=["standardized", "filtered", "separated"],
)
parser.add_argument(
    "--type",
    type=str,
    default="percentage",
    help="Visualize as totals or percentages",
    choices=["percentage", "totals"],
)
parser.add_argument(
    "--plot",
    type=str,
    default="all",
    help="Visualize only notes, chords or both",
    choices=["all", "notes", "chords"],
)
args = parser.parse_args()

NIVEL = args.data
TIPO = args.type
PLOT = args.plot

DICIO_NOTAS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def plot_graph(
    total_notas_organizado,
    total_acordes_maiores_organizado,
    total_acordes_menores_organizado,
):
    x = np.arange(len(DICIO_NOTAS))  # the label locations

    if PLOT == "notes":
        ncol = 1
        fig, ax = plt.subplots()
        ax.bar(x, total_notas_organizado, 1, label="Notes")
    elif PLOT == "chords":
        ncol = 2
        width = 0.8  # the width of the bars

        fig, ax = plt.subplots()
        ax.bar(
            x - width / 4, total_acordes_maiores_organizado, width / 2, label="Major c."
        )
        ax.bar(
            x + width / 4, total_acordes_menores_organizado, width / 2, label="Minor c."
        )
    elif PLOT == "all":
        ncol = 3
        width = 0.6  # the width of the bars

        fig, ax = plt.subplots()
        ax.bar(x - width / 3, total_notas_organizado, width / 3, label="Notes")
        ax.bar(x, total_acordes_maiores_organizado, width / 3, label="Major c.")
        ax.bar(
            x + width / 3, total_acordes_menores_organizado, width / 3, label="Minor c."
        )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    yl = "Amount (%)" if TIPO == "percentage" else "Amount"
    ax.set_ylabel(yl, fontsize=14)
    # ax.set_title('Quantidade de notas e acordes')
    ax.set_xticks(x)
    ax.set_xticklabels(DICIO_NOTAS)
    ax.legend(loc="upper right", fontsize=14, ncol=ncol)

    fig.tight_layout()

    plt.show()


def graph_total_notes_chords():
    np.set_printoptions(threshold=np.inf)

    caminho = "data/" + NIVEL + "/*.csv"
    arquivos = glob.glob(caminho)

    sequencia_notas = []
    sequencia_acordes = []

    # extrai os acordes e notas
    for caminho_csv in arquivos:
        csv_aberto = open(caminho_csv, "r", encoding="utf-8")
        next(csv_aberto)
        leitor = csv.reader(csv_aberto)

        compasso_anterior = None

        for linha in leitor:
            compasso = int(linha[0])
            acorde = linha[1]
            nota = linha[2]

            if compasso_anterior is None or compasso_anterior == compasso:
                if nota != "rest":
                    sequencia_notas.append(nota)

                if compasso_anterior is None:
                    sequencia_acordes.append(acorde)

            else:
                sequencia_acordes.append(acorde)

            compasso_anterior = compasso

    # separa acordes maiores e menores
    sequencia_acordes_maiores = []
    sequencia_acordes_menores = []

    for acorde in sequencia_acordes:
        if acorde.split(":")[1] == "maj":
            sequencia_acordes_maiores.append(acorde)
        else:
            sequencia_acordes_menores.append(acorde)

    # conta o total de notas e acordes
    total_notas = Counter(sequencia_notas)
    total_acordes_maiores = Counter(sequencia_acordes_maiores)
    total_acordes_menores = Counter(sequencia_acordes_menores)

    # separa e organiza
    total_notas_organizado = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    total_acordes_maiores_organizado = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    total_acordes_menores_organizado = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for item in list(total_notas.items()):
        total_notas_organizado[DICIO_NOTAS.index(item[0])] = item[1]

    for item in list(total_acordes_maiores.items()):
        total_acordes_maiores_organizado[
            DICIO_NOTAS.index(item[0].split(":")[0])
        ] = item[1]

    for item in list(total_acordes_menores.items()):
        total_acordes_menores_organizado[
            DICIO_NOTAS.index(item[0].split(":")[0])
        ] = item[1]

    # calcula a percentage
    if TIPO == "percentage":
        total_notas_organizado = [
            (total / sum(total_notas_organizado)) * 100
            for total in total_notas_organizado
        ]
        total_acordes_maiores_organizado = [
            (total / sum(total_acordes_maiores_organizado)) * 100
            for total in total_acordes_maiores_organizado
        ]
        total_acordes_menores_organizado = [
            (total / sum(total_acordes_menores_organizado)) * 100
            for total in total_acordes_menores_organizado
        ]

    print("\nTotal de notas:", total_notas_organizado)
    print("Total de acordes maiores:", total_acordes_maiores_organizado)
    print("Total de acordes menores:", total_acordes_menores_organizado)

    plot_graph(
        total_notas_organizado,
        total_acordes_maiores_organizado,
        total_acordes_menores_organizado,
    )


if __name__ == "__main__":
    graph_total_notes_chords()
