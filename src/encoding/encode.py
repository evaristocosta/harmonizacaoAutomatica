import glob
import os
import numpy as np
import argparse
from encoders.one_hot import processamento_one_hot
from encoders.imagem_compasso import processamento_ic

parser = argparse.ArgumentParser(description="Data encoder")
parser.add_argument(
    "-e",
    "--encoder",
    help="Qual codificação usar (oh: one hot, ic: imagem de compasso)",
    default="oh",
    choices=["oh", "ic"],
)
parser.add_argument(
    "-d",
    "--dataset",
    help="Qual tipo de dados usar (st: standard, fl: filtered, bl: balanced, sp: separated)",
    default="sp",
    choices=["st", "fl", "bl", "sp"],
)
parser.add_argument(
    "--single",
    action="store_true",
    help="Salvar tudo em um único arquivo (defalt: True)",
)
parser.add_argument("--no-single", dest="single", action="store_false")
parser.set_defaults(single=True)

parser.add_argument(
    "-c",
    "--color",
    help="Usar formato RGBA ou RGB",
    default="rgba",
    choices=["rgba", "rgb"],
)
args = parser.parse_args()

DATASET = args.dataset
OPTION = args.encoder
COLOR = args.color
SINGLE = args.single


def encode():
    if not os.path.isdir("data/encoded/"):
        os.mkdir("data/encoded/")
    # remove anteriores
    if SINGLE:
        if os.path.isfile("data/encoded/vetor_entrada.npy"):
            os.remove("data/encoded/vetor_entrada.npy")
        if os.path.isfile("data/encoded/vetor_saida.npy"):
            os.remove("data/encoded/vetor_saida.npy")

    if DATASET == "st":
        caminho = "data/standardized/*.csv"
    elif DATASET == "fl":
        caminho = "data/filtered/*.csv"
    elif DATASET == "bl":
        caminho = "data/balanced/*.csv"
    elif DATASET == "sp":
        caminho = "data/separated/separated.csv"

    arquivos_csv = glob.glob(caminho)
    print(len(arquivos_csv), "arquivo(s) encontrado(s).\nIniciando codificação...")

    # encode
    if SINGLE:
        if OPTION == "oh":
            processamento_one_hot(arquivos_csv)
        elif OPTION == "ic":
            processamento_ic(arquivos_csv, False if COLOR == "rgba" else True)

        s = np.load("data/encoded/vetor_saida.npy", mmap_mode="r")
        e = np.load("data/encoded/vetor_entrada.npy", mmap_mode="r")
        print("----\nDiferença entre quantidade de dados:", e.shape[0] - s.shape[0])
        print("Total de dados:", e.shape[0])
    else:
        for i, arquivo in enumerate(arquivos_csv):
            print("-- Arquivo:", i)
            nome = os.path.basename(arquivo).split(".")[0]
            if OPTION == "oh":
                processamento_one_hot([arquivo], nome=nome)
            elif OPTION == "ic":
                processamento_ic(
                    [arquivo], False if COLOR == "rgba" else True, nome=nome
                )


if __name__ == "__main__":
    encode()
