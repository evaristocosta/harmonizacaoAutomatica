import argparse
import os
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Create CSV with optimization data after running keras tuner"
    )
    parser.add_argument(
        "folder",
        help="Folder to analyse",
        type=str,
    )
    args = parser.parse_args()

    folder = args.folder

    path = "src/system/results/"

    subfolders = [f.path for f in os.scandir(path + folder) if f.is_dir()]
    # remove the element that have the substring "logs"
    subfolders = [x for x in subfolders if not "logs" in x]
    arquivo = "trial.json"
    csv_folder = f"{path}optimization_{folder}.csv"

    # create csv file with headers only
    colunas = [
        "id",
        "activation",
        "neurons",
        "dropout",
        "cnn_layers",
        "dense_layer",
        "dense_2",
        "learning_rate",
        "optimizer",
        "best_step",
        "loss",
        "accuracy",
        "val_loss",
        "val_accuracy",
    ]
    df = pd.DataFrame(columns=colunas)
    df.to_csv(csv_folder, index=False)

    for subfolder in subfolders:
        with open(subfolder + "/" + arquivo) as f:
            data = json.load(f)

            id = data["trial_id"]

            activation = data["hyperparameters"]["values"]["activation"]
            neurons = data["hyperparameters"]["values"]["neurons"]
            dropout = data["hyperparameters"]["values"]["dropout"]
            cnn_layers = data["hyperparameters"]["values"]["cnn_layers"]
            dense_layer = data["hyperparameters"]["values"]["dense_layer"]
            dense_2 = data["hyperparameters"]["values"]["dense_2"]
            learning_rate = data["hyperparameters"]["values"]["learning_rate"]
            optimizer = data["hyperparameters"]["values"]["optimizer"]

            best_step = data["best_step"]

            loss = data["metrics"]["metrics"]["loss"]["observations"][0]["value"][0]
            accuracy = data["metrics"]["metrics"]["accuracy"]["observations"][0][
                "value"
            ][0]
            val_loss = data["metrics"]["metrics"]["val_loss"]["observations"][0][
                "value"
            ][0]
            val_accuracy = data["metrics"]["metrics"]["val_accuracy"]["observations"][
                0
            ]["value"][0]

            # create a datafranme with the data
            df = pd.DataFrame(
                data=[
                    [
                        id,
                        activation,
                        neurons,
                        dropout,
                        cnn_layers,
                        dense_layer,
                        dense_2,
                        learning_rate,
                        optimizer,
                        best_step,
                        loss,
                        accuracy,
                        val_loss,
                        val_accuracy,
                    ]
                ]
            )
            # append df as csv
            df.to_csv(csv_folder, mode="a", header=False, index=False)


if __name__ == "__main__":
    main()
