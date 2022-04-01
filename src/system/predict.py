import argparse
import pickle
import json
import re
import pandas as pd
import numpy as np
from keras.models import load_model

from load_data import carrega, separa
from models.rbf import RBFLayer
from performance_measures import print_basic_performance


def predict():
    parser = argparse.ArgumentParser(description="Make predictions using a trained model.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="mlp_1_hidden",
        help="Experiment to be considered",
    )
    parser.add_argument("--separate", type=bool, default=True, help="Separate data?")
    parser.add_argument("--print_chords", type=bool, default=False, help="Print chords?")

    args = parser.parse_args()
    EXPERIMENT = args.experiment
    SEPARATE = args.separate
    PRINT_CHORDS = args.print_chords

    # carrega dados
    X, Y = carrega(data="encoded")
    if SEPARATE:
        _, _, _, _, X, Y = separa(X, Y, ratio_train=0.7)

    modelo = return_model(EXPERIMENT)

    predicao = modelo.predict(X)
    print_basic_performance(Y, predicao)

    if PRINT_CHORDS:
        acordes_df = print_acordes(Y, predicao)
        print(acordes_df.to_string())


def return_model(experiment):

    # abre sumário e seleciona dados do experimento, ordenados por erro
    df = pd.read_csv("src/system/results/summary.csv")
    df = df[df["experiment"] == experiment].sort_values(by=["loss"])

    # pega informações da melhor execução do experimento
    date = str(df["date"].values[0])
    best_run = str(df["best_run"].values[0])

    path = "src/system/results/" + experiment + "_" + date + "/"

    """ 
    Se precisar dos parametros:
    with open(path + "params.json") as json_file:
        params = json.load(json_file)
    """

    if experiment != "elm" and experiment != "esn":
        if experiment != "rbf":
            modelo = load_model(path + "modelos/" + best_run + ".h5")
        else:
            modelo = load_model(
                path + "modelos/" + best_run + ".h5",
                custom_objects={"RBFLayer": RBFLayer},
            )

        """ 
        Caso precise do otimizador:
        regex = r"\.([^.]*)'>"
        opt = re.search(regex, str(params["optimizer"])).group(1)
        """

    else:
        file = open(path + "pesos/" + best_run + ".pkl", "rb")
        modelo = pickle.load(file)
        file.close()

    return modelo


def print_acordes(Y, predicao):
    dicio_acordes = [
        "C:maj",
        "C:min",
        "C#:maj",
        "C#:min",
        "D:maj",
        "D:min",
        "D#:maj",
        "D#:min",
        "E:maj",
        "E:min",
        "F:maj",
        "F:min",
        "F#:maj",
        "F#:min",
        "G:maj",
        "G:min",
        "G#:maj",
        "G#:min",
        "A:maj",
        "A:min",
        "A#:maj",
        "A#:min",
        "B:maj",
        "B:min",
    ]

    acordes_real = []
    acordes_predicao = []

    predicao = np.squeeze(np.asarray(predicao))
    predicao_categorico = np.argmax(predicao, axis=1)
    Y_categorico = np.argmax(Y, axis=1)

    for acorde_real, acorde_pred in zip(Y_categorico, predicao_categorico):
        acordes_real.append(dicio_acordes[acorde_real])
        acordes_predicao.append(dicio_acordes[acorde_pred])

    acordes = {"real": acordes_real, "pred": acordes_predicao}
    acordes_df = pd.DataFrame(acordes)
    return acordes_df


if __name__ == "__main__":
    predict()
