import argparse
import numpy as np
from predict import return_model
from load_data import carrega, separa
from performance_measures import print_all_performance


def ensemble():
    parser = argparse.ArgumentParser(description="Ensemble learning")
    parser.add_argument("--models", nargs="+", help="Models to be used", required=True)
    parser.add_argument(
        "--vote",
        type=str,
        help="Voting method",
        default="majority",
        choices=["majority", "wta"],
    )

    args = parser.parse_args()
    MODELS = args.models
    VOTING = args.vote

    X, Y = carrega(data="encoded")
    _, _, _, _, X, Y = separa(X, Y, ratio_train=0.7)

    predicao_ensemble = predict(VOTING, MODELS, X)

    # print all performance
    print_all_performance(Y, predicao_ensemble)


def predict(votacao, selecoes, X):
    modelos = []
    # carregar modelos
    for selecao in selecoes:
        modelos.append(return_model(selecao))

    predicoes = []

    for modelo in modelos:
        predicoes.append(np.squeeze(np.asarray(modelo.predict(X))))

    # redefine pra iteracao
    predicoes = np.transpose(np.array(predicoes), (1, 0, 2))

    # escolha das respostas (soft voting ou wta)
    predicao_ensemble = []

    if votacao == "majority":
        # soft voting:
        for predicao in predicoes:
            soma = np.sum(predicao, axis=0) / len(predicao)
            predicao_ensemble.append(soma)
    elif votacao == "wta":
        # wta:
        for predicao in predicoes:
            maiores = [np.max(maior) for maior in predicao]
            ganhador = np.argmax(maiores)
            predicao_ensemble.append(predicao[ganhador])

    # resultados
    predicao_ensemble = np.squeeze(np.asarray(predicao_ensemble))
    return predicao_ensemble


if __name__ == "__main__":
    ensemble()
