import glob
import ntpath
import csv
import os
import argparse
from pandas import DataFrame

parser = argparse.ArgumentParser(description="Filter for standalized data")
parser.add_argument(
    "--empty", action="store_true", help="Remover compassos vazios (defalt: True)"
)
parser.add_argument("--no-empty", dest="empty", action="store_false")
parser.set_defaults(empty=True)

parser.add_argument(
    "--harmonic",
    action="store_true",
    help="Remover acordes fora do campo harmônico (defalt: False)",
)
parser.add_argument("--no-harmonic", dest="harmonic", action="store_false")
parser.set_defaults(harmonic=False)

parser.add_argument(
    "--single",
    action="store_true",
    help="Salvar tudo em um único arquivo (defalt: True)",
)
parser.add_argument("--no-single", dest="single", action="store_false")
parser.set_defaults(single=True)

args = parser.parse_args()

FILTRO_COMPASSO_VAZIO = args.empty
FILTRO_CAMPO_HARMONICO = args.harmonic
ARQUIVO_UNICO = args.single
SEMIBREVE = 96.0


def filter():
    caminho = "data/standardized/*.csv"
    novo_caminho = "data/filtered/"

    campo_harmonico = [
        "C:maj",
        "D:min",
        "E:min",
        "F:maj",
        "G:maj",
        "A:min",
        "B:min",
    ]

    # Encontra todos arquivos no caminho especificado
    arquivos_csv = glob.glob(caminho)
    print("Filtrando " + str(len(arquivos_csv)) + " arquivos...")

    if not os.path.isdir(novo_caminho):
        os.mkdir(novo_caminho)

    # Remoção de anteriores
    arquivos_antigos = glob.glob(novo_caminho + "*")
    for arquivo in arquivos_antigos:
        os.remove(arquivo)

    # iteracao no vetor de arquivos mantendo o indice
    for i, caminho_csv in enumerate(arquivos_csv):
        # pega somente nome do arquivo
        # o indice eh usado indiretamente aqui
        nome_arquivo = ntpath.basename(caminho_csv)
        print("Filtrando: " + nome_arquivo, "(" + str(i + 1) + ")")

        # abre arquivo csv como leitura
        csv_aberto = open(caminho_csv, "r", encoding="utf-8")
        next(csv_aberto)  # pula a primeira linha (info de colunas)
        # capacita leitor de csv
        leitor = csv.reader(csv_aberto)

        # vetores de armazenamento
        compasso_lista = []
        acorde_lista = []
        nota_lista = []
        nota_oitava_lista = []
        duracao_lista = []

        # leitura de todas linhas do csv
        for linha in leitor:
            # padronização reduziu o banco de dados a 5 colunas de infomacao:
            #    0       1      2      3        4
            # measure, chord, note, octave, duration
            compasso = linha[0]
            acorde = linha[1]
            nota = linha[2]
            nota_oitava = linha[3]
            duracao = linha[4]

            # opcao 1: remover compassos vazios - teste: Close every door from Joseph and the Amazing Techcolor Dream Coat.csv
            if nota == "rest" and eval(duracao) == SEMIBREVE and FILTRO_COMPASSO_VAZIO:
                continue

            # opcao 2: remover acordes fora do campo harmônico
            if acorde not in campo_harmonico and FILTRO_CAMPO_HARMONICO:
                continue

            compasso_lista.append(compasso)
            acorde_lista.append(acorde)
            nota_lista.append(nota)
            nota_oitava_lista.append(nota_oitava)
            duracao_lista.append(duracao)

        # produz novo csv
        dados_coletados = {
            "measure": compasso_lista,
            "chord": acorde_lista,
            "note": nota_lista,
            "octave": nota_oitava_lista,
            "duration": duracao_lista,
        }
        # elaboracao da tabela
        df = DataFrame(dados_coletados)

        if ARQUIVO_UNICO:
            df.to_csv(
                novo_caminho + "filtrado.csv",
                encoding="utf-8",
                index=False,
                mode="a",
                header=(i == 0),
            )
        else:
            df.to_csv(novo_caminho + nome_arquivo, encoding="utf-8", index=False)


if __name__ == "__main__":
    filter()
