import glob
import ntpath
import csv
import os
import argparse
from pandas import DataFrame

parser = argparse.ArgumentParser(
    description="Standardization of raw data considering chord types"
)
parser.add_argument(
    "--chord_type",
    type=str,
    default="triad",
    help="Chord type to be considered",
    choices=["triad", "seventh"],
)
parser.add_argument(
    "--db_percentage", type=float, default=1.0, help="Percentage of data to be used"
)
parser.add_argument(
    "--semibreve", type=int, default=96, help="Number of semibreve duration/ticks"
)
args = parser.parse_args()

# Tipo de acorde
ACORDE = args.chord_type
# Quantidade de dados
PORCENTAGEM_BD = args.db_percentage
# Duração da semibreve (x)
SEMIBREVE = args.semibreve


def conversor_tipo_acorde_triade(chord):
    """Simplificacao dos tipos de acorde: maiores e menores"""
    return {
        "maj": "maj",
        "major": "maj",
        "major-sixth": "maj",
        "major-seventh": "maj",
        "maj7": "maj",
        "major-ninth": "maj",
        "maj69": "maj",
        "maj9": "maj",
        "major-minor": "maj",
        "minor": "min",
        "min": "min",
        "minor-sixth": "min",
        "minor-seventh": "min",
        "min7": "min",
        "minor-ninth": "min",
        "minor-11th": "min",
        "minor-13th": "min",
        "minor-major": "min",
        "minMaj7": "min",
        "6": "maj",
        "7": "maj",
        "9": "maj",
        "dominant": "maj",
        "dominant-seventh": "maj",
        "dominant-ninth": "maj",
        "dominant-11th": "maj",
        "dominant-13th": "maj",
        "augmented": "maj",
        "aug": "maj",
        "augmented-seventh": "maj",
        "augmented-ninth": "maj",
        "dim": "min",
        "diminished": "min",
        "diminished-seventh": "min",
        "half-diminished": "min",
        "m7b5": "min",
        "dim7": "min",
        " dim7": "min",
        "suspended-second": "maj",
        "suspended-fourth": "maj",
        "sus47": "maj",
        "power": "maj",
    }.get(chord, "nan")


def conversor_tipo_acorde_setima(chord):
    """Simplificacao dos tipos de acorde: maiores e menores"""
    return {
        "maj": "maj",
        "major": "maj",
        "major-sixth": "maj",
        "major-seventh": "maj",
        "maj7": "7",
        "major-ninth": "maj",
        "maj69": "maj",
        "maj9": "maj",
        "major-minor": "maj",
        "minor": "min",
        "min": "min",
        "minor-sixth": "min",
        "minor-seventh": "min",
        "min7": "7",
        "minor-ninth": "min",
        "minor-11th": "min",
        "minor-13th": "min",
        "minor-major": "min",
        "minMaj7": "min",
        "6": "maj",
        "7": "7",
        "9": "maj",
        "dominant": "maj",
        "dominant-seventh": "7",
        "dominant-ninth": "maj",
        "dominant-11th": "maj",
        "dominant-13th": "maj",
        "augmented": "maj",
        "aug": "maj",
        "augmented-seventh": "maj",
        "augmented-ninth": "maj",
        "dim": "min",
        "diminished": "min",
        "diminished-seventh": "min",
        "half-diminished": "min",
        "m7b5": "min",
        "dim7": "min",
        " dim7": "min",
        "suspended-second": "maj",
        "suspended-fourth": "maj",
        "sus47": "maj",
        "power": "maj",
    }.get(chord, "nan")


def traduz_indice(raiz):
    """Converte nota para valor inteiro"""
    return {
        "B#": 0,
        "C0": 0,
        "C2": 0,
        "Db": 1,
        "C#": 1,
        "D0": 2,
        "D-2": 2,
        "Eb": 3,
        "D#": 3,
        "E0": 4,
        "Fb": 4,
        "F0": 5,
        "E#": 5,
        "F2": 5,
        "Gb": 6,
        "F#": 6,
        "G0": 7,
        "Ab": 8,
        "G#": 8,
        "A0": 9,
        "A2": 9,
        "Bb": 10,
        "A#": 10,
        "B-2": 10,
        "Cb": 11,
        "B0": 11,
        "rest": 12,
    }.get(raiz, "nan")


def transpoe(raiz, tom):
    """transpoe para C."""
    # variaveis de retorno
    raiz_dicionario = [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
        "rest",
    ]
    # pega intervalo para transposicao
    calcular_num = calculador_transposicao(tom)

    # encontra indice da raiz
    raiz_indice = traduz_indice(raiz)  # resultado: 0 ~ 12
    # no caso de ser pausa, retorna direto
    if raiz_indice == 12:
        return raiz_dicionario[raiz_indice]

    # realiza transposicao
    transposicao_indice = raiz_indice + calcular_num
    # mod 12 pois desconsidera pausa (rest)
    transposicao_indice %= 12

    return raiz_dicionario[transposicao_indice]


def calculador_transposicao(tom):
    """Calcula o intervalo de transposicao"""
    return {
        "-5": -1,
        "-4": 4,
        "-3": -3,
        "-2": 2,
        "-1": -5,
        "0": 0,
        "1": 5,
        "2": -2,
        "3": 3,
        "4": -4,
        "5": 1,
        "6": 6,
        "-6": 6,
        "7": -1,
    }.get(tom, "nan")


def standardization():
    # de acordo com padrão da base de dados
    normalizador_bd = 16 / SEMIBREVE

    caminho = "data/raw/*.csv"
    novo_caminho = "data/standardized/"

    # Encontra todos arquivos no caminho especificado
    arquivos_csv = glob.glob(caminho)
    limite = int(len(arquivos_csv) * PORCENTAGEM_BD)
    print("Padronizando " + str(limite) + " arquivos...")

    # Remoção de anteriores
    arquivos_antigos = glob.glob(novo_caminho + "*")
    for arquivo in arquivos_antigos:
        os.remove(arquivo)

    # iteracao no vetor de arquivos mantendo o indice
    for i, caminho_csv in enumerate(arquivos_csv):
        if i >= limite:
            break

        # pega somente nome do arquivo
        # o indice eh usado indiretamente aqui
        nome_arquivo = ntpath.basename(caminho_csv)
        print("Padronizando: " + nome_arquivo, "(" + str(i + 1) + ")")

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
            # formato do cabecalho dos csvs:
            #   0      1         2          3         4            5           6           7           8
            # time, measure, key_fifths, key_mode, chord_root, chord_type, note_root, note_octave, note_duration
            tempo = linha[0]
            duracao = linha[8]
            compasso = linha[1]
            quintas = linha[2]
            nota_raiz_acorde = linha[4]
            tipo_acorde = linha[5]
            nota_raiz = linha[6]
            nota_oitava = linha[7]

            # normaliza a duracao de acordo com o tempo da pauta
            duracao_real = ((1 / eval(tempo)) * float(duracao)) / normalizador_bd

            # simplifica o tipo do acorde (maiores, menores e com 7)
            if ACORDE == "triad":
                resultado_tipo_acorde = conversor_tipo_acorde_triade(tipo_acorde)
            elif ACORDE == "seventh":
                resultado_tipo_acorde = conversor_tipo_acorde_setima(tipo_acorde)

            # ignora notas e/ou acordes desconhecidos
            # compassos marcados com 'X1' (erro no próprio xml)
            # formato das notas encontradas no banco de dados
            dicionario_completo = [
                "C0",
                "Db",
                "D0",
                "Eb",
                "E0",
                "F0",
                "Gb",
                "G0",
                "Ab",
                "A0",
                "Bb",
                "B0",
                "B#",
                "C#",
                "D#",
                "Fb",
                "E#",
                "F#",
                "G#",
                "A#",
                "Cb",
                "rest",
            ]

            if (
                nota_raiz not in dicionario_completo
                or nota_raiz_acorde not in dicionario_completo
                or resultado_tipo_acorde == "nan"
                or compasso == "X1"
            ):
                continue

            # transpoe para C
            resultado_acorde = transpoe(nota_raiz_acorde, quintas)
            resultado_nota = transpoe(nota_raiz, quintas)

            compasso_lista.append(compasso)
            duracao_lista.append(duracao_real)
            acorde_lista.append(resultado_acorde + ":" + resultado_tipo_acorde)
            nota_lista.append(resultado_nota)
            nota_oitava_lista.append(nota_oitava)

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

        # verifica existencia da pasta
        if not os.path.isdir(novo_caminho):
            os.mkdir(novo_caminho)
        df.to_csv(novo_caminho + nome_arquivo, encoding="utf-8", index=False)


if __name__ == "__main__":
    standardization()
