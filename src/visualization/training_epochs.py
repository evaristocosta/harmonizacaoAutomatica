import pandas as pd
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot date based accuracy and loss training results"
    )
    parser.add_argument(
        "-d",
        "--date",
        help="Training date to be considered",
        type=int,
    )
    args = parser.parse_args()

    date = args.date

    df = pd.read_csv("src/system/results/summary.csv")
    df = df[df["date"] == date]

    # pega informações da melhor execução do experimento
    experiment = df["experiment"].values[0]
    best_run = df["best_run"].values[0] - 1

    print("Experiment:", experiment)
    path = f"src/system/results/{experiment}_{date}/logs/{best_run}.csv"

    resultados = pd.read_csv(path)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Experimento: {experiment}")
    # acerto
    axs[0].plot(resultados["epoch"], resultados["accuracy"], label="Treino")
    axs[0].plot(resultados["epoch"], resultados["val_accuracy"], label="Validação")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Acerto")
    axs[0].legend()
    # loss
    axs[1].plot(resultados["epoch"], resultados["loss"], label="Treino")
    axs[1].plot(resultados["epoch"], resultados["val_loss"], label="Validação")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    plt.show()


if __name__ == "__main__":
    main()