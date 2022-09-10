import argparse
import numpy as np
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="Separa dados de treinamento e salva")
parser.add_argument(
    "-d",
    "--data",
    help="Dados a serem separados",
    default="encoded",
    choices=["encoded", "separated", "balanced"],
)
parser.add_argument(
    "-r", "--ratio", help="Porcentagem de treinamento", type=float, default=0.6
)
args = parser.parse_args()

DATA = args.data
RATIO = args.ratio


def salva():
    data = DATA

    X = np.load("data/" + data + "/vetor_entrada.npy", mmap_mode="r")
    Y = np.load("data/" + data + "/vetor_saida.npy", mmap_mode="r")

    X_train, Y_train, X_val, Y_val, X_test, Y_test = separa(X, Y, ratio_train=RATIO)

    np.save("data/separated/input_train.npy", X_train)
    np.save("data/separated/output_train.npy", Y_train)
    np.save("data/separated/input_val.npy", X_val)
    np.save("data/separated/output_val.npy", Y_val)
    np.save("data/separated/input_test.npy", X_test)
    np.save("data/separated/output_test.npy", Y_test)
    print("Arquivos salvos com sucesso!")


def separa(X, Y, ratio_train=0.6):
    # divisao
    # https://datascience.stackexchange.com/a/55322
    """Exemplo:
    ratio_train = 0.60
    ratio_val = 0.20
    ratio_test = 0.20
    """
    ratio_val = (1 - ratio_train) / 2
    ratio_test = ratio_val

    ratio_remaining = 1 - ratio_test
    ratio_val_adjusted = ratio_val / ratio_remaining

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=ratio_test,
        shuffle=True,
        random_state=42,
        stratify=Y,
    )

    del X
    del Y

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train,
        Y_train,
        test_size=ratio_val_adjusted,
        shuffle=True,
        random_state=42,
        stratify=Y_train,
    )

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


if __name__ == "__main__":
    salva()
