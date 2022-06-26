import numpy as np
import pandas as pd
import csv
import os
from scipy.stats import norm
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

    caminho_csv = "data/filtered/filtrado.csv"
    number_lines = sum(1 for row in (open(caminho_csv)))

    print("Tamanho inicial: ", number_lines)

    # calcula tamanho amostral
    # http://www.cienciasecognicao.org/portal/wp-content/uploads/2011/09/Tamanho-da-Amostra-1-1.pdf
    # TRIOLA, Mário F. Introdução à Estatística. 7a. Ed. Rio de Janeiro: LTC, 1999. pg. 163
    N = number_lines
    confianca = 0.99
    alpha = 1 - confianca
    # https://www.statology.org/z-critical-value-python/
    Z = norm.ppf(1 - alpha / 2)
    pq = 0.25
    e = 0.01

    n = round((N * (Z**2) * pq) / ((N - 1) * e**2 + (Z**2) * pq))
    porcentagem = n / N

    total_rows = int(number_lines * porcentagem)
    df = pd.read_csv(caminho_csv, header=None, nrows=total_rows)
    saida = "data/separated/separated.csv"
    df.to_csv(
        saida,
        index=False,
        mode="w",
        header=not os.path.exists(saida),
    )

    print("Tamanho original: ", N)
    print("Número de amostras: ", n)
    print("Porcentagem: ", porcentagem)
    print("Tamanho final: ", total_rows)


def porcentagem_fixa():
    caminho_csv = "data/filtered/filtrado.csv"

    number_lines = sum(1 for row in (open(caminho_csv)))
    print("Tamanho inicial (num linhas): ", number_lines)
    print("Porcentagem: ", RATIO)

    total_rows = int(number_lines * RATIO)
    print("Tamanho final: ", total_rows)

    df = pd.read_csv(caminho_csv, header=None, nrows=total_rows)

    saida = "data/separated/"
    if not os.path.isdir(saida):
        os.mkdir(saida)

    df.to_csv(
        saida + "separated.csv",
        index=False,
        mode="w",
        header=not os.path.exists(saida),
    )


def separate():
    if METHOD == "amostral":
        tamanho_amostral()
    elif METHOD == "porcentagem":
        porcentagem_fixa()


if __name__ == "__main__":
    separate()
