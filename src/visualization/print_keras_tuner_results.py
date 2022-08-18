import argparse
import pandas as pd
import matplotlib.pyplot as plt
import paxplot
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description="Print tuner results")
    parser.add_argument(
        "file",
        help="File to analyse",
        type=str,
    )
    args = parser.parse_args()

    file = args.file
    path = "src/system/results/"

    # Import data
    path_to_data = path + file
    df = pd.read_csv(path_to_data)

    print(df.sort_values(by="val_loss", ascending=True))
    print("Correlação\n", df.corrwith(df.val_loss, method="kendall"))

    df = df.drop(columns=["id", "accuracy", "val_accuracy"])
    cols = df.columns

    df_paxplot = (
        df.copy()
        .replace("elu", 0)
        .replace("relu", 1)
        .replace("rmsprop", 0)
        .replace("sgd", 1)
        .replace("adam", 2)
        .replace(True, 1)
        .replace(False, 0)
    )

    # Create figure
    paxfig = paxplot.pax_parallel(n_axes=len(cols))
    paxfig.plot(df_paxplot.to_numpy())

    # Add labels
    paxfig.set_labels(cols)

    # Add colorbar
    color_col = len(cols) - 1
    paxfig.add_colorbar(
        ax_idx=color_col, cmap="viridis", colorbar_kwargs={"label": cols[color_col]}
    )

    plt.show()

    corr = df.corr(method="kendall")
    sns.heatmap(
        corr,
        xticklabels=corr.columns.values,
        yticklabels=corr.columns.values,
        annot=True,
    )
    plt.show()


if __name__ == "__main__":
    main()
