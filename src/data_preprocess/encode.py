import glob
import numpy as np
from npy_append_array import NpyAppendArray
from encoders.one_hot import processamento_one_hot


def construtor():
    entrada = "1"
    if entrada == "1":
        caminho = "dados/novo_treino/*.csv"
    elif entrada == "2":
        caminho = "dados/novo_validacao/*.csv"
    else:
        print("Erro de entrada")
        return None

    print(
        "-- Sempre será somente o primeiro acorde do compasso --\n"
        "1. Todas as notas e pausas (binário)\n"
        "2. Todas as notas e pausas (2D)\n"
        "3. Todas as notas e pausas (2D somadas)\n"
        "4. Todas as notas e pausas (considerando tempos)\n"
        "5. Todas as notas e pausas (2D somadas, sem pausas)\n"
    )
    processamento = input("Selecione o modo de processamento: ")

    arquivos_csv = glob.glob(caminho)
    print(len(arquivos_csv), "arquivos encontrados.\nIniciando processamento...")

    # encode
    processamento_one_hot(processamento, arquivos_csv)

    s = np.load("vetor_saida.npy", mmap_mode="r")
    e = np.load("vetor_entrada.npy", mmap_mode="r")
    print("----\nDiferença entre quantidade de dados:", e.shape[0] - s.shape[0])
    print("Total de dados:", e.shape[0])


if __name__ == "__main__":
    construtor()
