import os
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description="Data separation")
parser.add_argument(
    "-m",
    help="Método de separação",
    choices=["amostral", "porcentagem"],
    default="porcentagem",
)
parser.add_argument(
    "-p",
    help="Porcentagem escolhida caso -m seja porcentagem",
    type=float,
    default=0.30,
)

args = parser.parse_args()
METHOD = args.m
RATIO = args.p


def tamanho_amostral():
    print("Calculando tamanho amostral ideal...")

    vetor_entrada = np.load("data/encoded/vetor_entrada.npy", mmap_mode="r")
    vetor_saida = np.load("data/encoded/vetor_saida.npy", mmap_mode="r")

    print("Tamanho inicial (shape entrada): ", vetor_entrada.shape)

    # calcula tamanho amostral
    # http://www.cienciasecognicao.org/portal/wp-content/uploads/2011/09/Tamanho-da-Amostra-1-1.pdf
    # TRIOLA, Mário F. Introdução à Estatística. 7a. Ed. Rio de Janeiro: LTC, 1999. pg. 163
    # https://www.statisticshowto.com/probability-and-statistics/find-sample-size/
    N = vetor_entrada.shape[0]
    confianca = 0.99
    alpha = 1 - confianca
    # https://www.statology.org/z-critical-value-python/
    Z = norm.ppf(1 - alpha / 2)
    pq = 0.25
    e = 0.01

    n = round((N * (Z ** 2) * pq) / ((N - 1) * e ** 2 + (Z ** 2) * pq))
    porcentagem = n / N

    X, _, Y, _ = train_test_split(
        vetor_entrada,
        vetor_saida,
        train_size=porcentagem,
        shuffle=True,
        random_state=42,
        stratify=vetor_saida,
    )

    del vetor_entrada
    del vetor_saida

    if not os.path.isdir("data/separated"):
        os.mkdir("data/separated")

    np.save("data/separated/vetor_entrada.npy", X)
    np.save("data/separated/vetor_saida.npy", Y)

    print("Tamanho original: ", N)
    print("Número de amostras: ", n)
    print("Porcentagem: ", porcentagem)
    print("Tamanho final (shape entrada): ", X.shape)


def porcentagem_fixa():
    vetor_entrada = np.load("data/encoded/vetor_entrada.npy", mmap_mode="r")
    vetor_saida = np.load("data/encoded/vetor_saida.npy", mmap_mode="r")
    print("Tamanho inicial (shape entrada): ", vetor_entrada.shape)

    X, _, Y, _ = train_test_split(
        vetor_entrada,
        vetor_saida,
        train_size=RATIO,
        shuffle=True,
        random_state=42,
        stratify=vetor_saida,
    )

    del vetor_entrada
    del vetor_saida

    if not os.path.isdir("data/separated"):
        os.mkdir("data/separated")

    np.save("data/separated/vetor_entrada.npy", X)
    np.save("data/separated/vetor_saida.npy", Y)

    print("Porcentagem: ", RATIO)
    print("Tamanho final (shape entrada): ", X.shape)


def separate():
    if METHOD == "amostral":
        tamanho_amostral()
    elif METHOD == "porcentagem":
        porcentagem_fixa()


if __name__ == "__main__":
    separate()
