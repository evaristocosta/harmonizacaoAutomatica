import glob
import os
import numpy as np
import argparse
from encoders.one_hot import processamento_one_hot
from encoders.imagem_compasso import processamento_ic

parser = argparse.ArgumentParser(description="Data encoder")
parser.add_argument(
    "-e",
    help="Qual codificação usar (oh: one hot, ic: imagem de compasso)",
    default="oh",
    choices=["oh", "ic"],
)
parser.add_argument(
    "-d",
    help="Qual tipo de dados usar (st: standard, fl: filtered, bl: balanced)",
    default="st",
    choices=["st", "fl", "bl"],
)
args = parser.parse_args()

DATASET = args.d
OPTION = args.e

def encode():
    # remove anteriores
    if os.path.isfile('data/encoded/vetor_entrada.npy'):
        os.remove('data/encoded/vetor_entrada.npy')
    if os.path.isfile('data/encoded/vetor_saida.npy'):
        os.remove('data/encoded/vetor_saida.npy')

    
    if DATASET == "st":
        caminho = "data/standardized/*.csv"
    elif DATASET == "fl":
        caminho = "data/filtered/*.csv"
    elif DATASET == "bl":
        caminho = "data/balanced/*.csv"

    arquivos_csv = glob.glob(caminho)
    print(len(arquivos_csv), "arquivos encontrados.\nIniciando codificação...")

    # encode
    if OPTION == "oh":
        processamento_one_hot(arquivos_csv)
    elif OPTION == "ic":
        processamento_ic(arquivos_csv)

    s = np.load("data/encoded/vetor_saida.npy", mmap_mode="r")
    e = np.load("data/encoded/vetor_entrada.npy", mmap_mode="r")
    print("----\nDiferença entre quantidade de dados:", e.shape[0] - s.shape[0])
    print("Total de dados:", e.shape[0])


if __name__ == "__main__":
    encode()
