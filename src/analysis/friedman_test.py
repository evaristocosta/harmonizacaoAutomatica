from scipy import stats
import pandas as pd

# seleciona melhores manualmente, array de dates
bests = [
    1649169086,
    1649239086,
    1649299088,
    1649428853,
    1649429262,
    1649556804,
    1649556868,
    1649556910,
]


def load_losses():
    # abrir log_resultados de cada um (array de loss)
    df = pd.read_csv("src/system/results/summary.csv")
    losses = []

    for best in bests:
        line = df[df["date"] == best]
        experiment = str(line["experiment"].values[0])
        print("Experiment:", experiment)

        results_path = (
            "src/system/results/" + experiment + "_" + str(best) + "/log_resultados.csv"
        )

        results = pd.read_csv(results_path)
        loss = results["loss"].tolist()
        losses.append(loss)

    return losses


def calc_friedman():
    # carregar losses
    losses = load_losses()

    stat, pvalue = stats.friedmanchisquare(*losses)

    print("Estatística:", stat)
    print("P-valor:", pvalue)


if __name__ == "__main__":
    calc_friedman()
