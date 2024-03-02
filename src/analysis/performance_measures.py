import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
)


# calculate from experiment results
def calc_best_from_date(date):
    # abre sumário e seleciona dados do experimento, ordenados por erro
    df = pd.read_csv("src/system/results/summary.csv")
    df = df[df["date"] == date]

    # pega informações da melhor execução do experimento
    best_run = int(df["best_run"].values[0]) - 1
    experiment = str(df["experiment"].values[0])
    print("Experiment:", experiment)

    path = "src/system/results/" + experiment + "_" + str(date) + "/output/"

    predicao = np.load(path + "predicao_por_fold.npy")
    real = np.load(path + "real_por_fold.npy")

    y_true = real[best_run]
    y_pred = predicao[best_run]

    print_all_performance(y_true, y_pred)


def calc_average_from_date(date):
    # abre sumário e seleciona dados do experimento, ordenados por erro
    df = pd.read_csv("src/system/results/summary.csv")
    df = df[df["date"] == date]

    # pega informações da melhor execução do experimento
    experiment = str(df["experiment"].values[0])
    print("Experiment:", experiment)

    path = "src/system/results/" + experiment + "_" + str(date) + "/output/"

    predicao = np.load(path + "predicao_por_fold.npy")
    real = np.load(path + "real_por_fold.npy")

    resultados = pd.DataFrame(columns=["loss", "acc", "f1", "mcc", "kappa"])

    for y_true, y_pred in zip(real, predicao):
        loss, acc, _, f1, mcc, kappa = print_all_performance(y_true, y_pred)
        resultados = pd.concat(
            [
                resultados,
                pd.DataFrame(
                    {
                        "loss": [loss],
                        "acc": [acc],
                        "f1": [f1],
                        "mcc": [mcc],
                        "kappa": [kappa],
                    }
                ),
            ],
            ignore_index=True,
        )

    print("\n\nMédia:\n", resultados.mean())
    return resultados.mean(), experiment, date


# loss
def calc_log_loss(y_true, y_pred):
    y_true_cat = np.argmax(y_true, axis=1)
    labels = np.arange(0, y_pred.shape[1])
    loss = log_loss(y_true_cat, y_pred, labels=labels)

    return loss


# acc
def calc_accuracy(y_true, y_pred):
    y_true_cat = np.argmax(y_true, axis=1)
    y_pred_cat = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_true_cat, y_pred_cat)

    return acc


# bacc
def calc_bacc(y_true, y_pred):
    y_true_cat = np.argmax(y_true, axis=1)
    y_pred_cat = np.argmax(y_pred, axis=1)
    bacc = balanced_accuracy_score(y_true_cat, y_pred_cat)

    return bacc


# f-measure
def calc_f1(y_true, y_pred):
    y_true_cat = np.argmax(y_true, axis=1)
    y_pred_cat = np.argmax(y_pred, axis=1)
    f1 = f1_score(y_true_cat, y_pred_cat, average="macro")

    return f1


# mcc
def calc_mcc(y_true, y_pred):
    y_true_cat = np.argmax(y_true, axis=1)
    y_pred_cat = np.argmax(y_pred, axis=1)
    mcc = matthews_corrcoef(y_true_cat, y_pred_cat)

    return mcc


# kappa
def calc_kappa(y_true, y_pred):
    y_true_cat = np.argmax(y_true, axis=1)
    y_pred_cat = np.argmax(y_pred, axis=1)
    kappa = cohen_kappa_score(y_true_cat, y_pred_cat)

    return kappa


# print only loss and accuracy
def print_basic_performance(y_true, y_pred):
    loss = calc_log_loss(y_true, y_pred)
    acc = calc_accuracy(y_true, y_pred)

    print("Loss: {:.4f}".format(loss))
    print("Accuracy: {:.4f}".format(acc))


# print all performance measures
def print_all_performance(y_true, y_pred):
    loss = calc_log_loss(y_true, y_pred)
    acc = calc_accuracy(y_true, y_pred)
    bacc = calc_bacc(y_true, y_pred)
    f1 = calc_f1(y_true, y_pred)
    mcc = calc_mcc(y_true, y_pred)
    kappa = calc_kappa(y_true, y_pred)

    print("Loss: {:.4f}".format(loss))
    print("Accuracy: {:.4f}".format(acc))
    print("Balanced Accuracy: {:.4f}".format(bacc))
    print("F1: {:.4f}".format(f1))
    print("Matthews Correlation Coefficient: {:.4f}".format(mcc))
    print("Cohen's Kappa: {:.4f}".format(kappa))

    return loss, acc, bacc, f1, mcc, kappa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance measures helper")
    parser.add_argument("date", type=int, help="Experiment date of execution")
    parser.add_argument(
        "-c",
        "--choice",
        type=str,
        help="Choice of performance measure calculation",
        choices=["best", "average"],
        default="best",
    )

    args = parser.parse_args()
    date = args.date
    choice = args.choice

    if choice == "best":
        calc_best_from_date(date)
    elif choice == "average":
        calc_average_from_date(date)
