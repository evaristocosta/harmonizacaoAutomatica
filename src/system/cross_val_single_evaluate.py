import sys
sys.path.insert(1, "/home/lucas/harmonizacaoAutomatica/src/")

import time
import os
import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description="Cross validate models")
parser.add_argument(
    "--model",
    help="Which model to train",
    default="mlp_1_hidden",
    choices=[
        "mlp_1_hidden",
        "mlp_2_hidden",
        "rbf",
        "esn",
        "elm",
        "cnn_like_alexnet",
        "alexnet",
        "vgg16",
        "resnet101",
        "inceptionv3",
        "densenet201",
    ],
)
parser.add_argument("--time", help="Timestamp", default=str(int(time.time())))
args = parser.parse_args()

MODEL = args.model
TIME = args.time


def cross_val_single_evaluate():
    agora = TIME
    teste = MODEL
    caminho = "src/system/results/" + teste + "_" + agora + "/log_resultados.csv"

    df = pd.read_csv(caminho)
    df = df.sort_values(by=["loss"]).iloc[0]

    melhor_fold_loss = int(df["rodada"])
    print("Melhor fold:", melhor_fold_loss)

    sumario = open("src/system/results/summary.csv", "a")
    sumario.write(f"{agora},{teste},{melhor_fold_loss},{df['loss']},{df['accuracy']}\n")
    sumario.close()


if __name__ == "__main__":
    cross_val_single_evaluate()
