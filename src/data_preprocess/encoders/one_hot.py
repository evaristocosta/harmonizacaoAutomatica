import csv
import numpy as np
from npy_append_array import NpyAppendArray



# dicionario de notas e acordes: correspondem com todas possibilidades do banco de dados processado
dicio_notas = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "rest"]
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


def codificacao_one_hot(length, one_index):
    # Retorna vetor codificado
    # vetor de zeros
    vetor = [0] * length
    # posicao do 1 no vetor
    vetor[one_index] = 1

    return vetor



def processamento_one_hot(arquivos_csv):
    # pega tamanho dos dicionarios
    dicio_notas_tamanho = len(dicio_notas)
    dicio_acordes_tamanho = len(dicio_acordes)

    # declaracao das listas usadas durante o processamento
    # a mesma lista de treino é usada para processar o conjunto de validacao
    matriz_dados_entrada = []
    matriz_dados_saida = []

    # arquivos de salvamento
    vetor_entrada = NpyAppendArray("vetor_entrada.npy")
    vetor_saida = NpyAppendArray("vetor_saida.npy")

    # construcao das matrizes a partir dos arquivos csv
    # logica como do processamento
    for i, caminho_csv in enumerate(arquivos_csv):
        print(caminho_csv, "--", i + 1)

        csv_aberto = open(caminho_csv, "r", encoding="utf-8")
        next(csv_aberto)
        leitor = csv.reader(csv_aberto)

        # lista da sequencia de notas sendo processada
        sequencia_notas = []
        # para o caso de imagem
        controle_resto = 0
        # lista de cada musica (bloco) processada
        lista_musicas = []
        # variavel de controle de compassos
        compasso_anterior = None
        duracao_soma = 0
        # indice de controle
        indice = None

        for linha in leitor:
            # processamento reduziu o banco de dados a 5 colunas de infomacao:
            #    0       1      2      3        4
            # measure, chord, note, octave, duration
            compasso = int(linha[0])
            acorde = linha[1]
            nota = linha[2]
            oitava = linha[3]
            duracao = float(linha[4])

            duracao_soma += duracao

            # pega posicao da nota no dicionario
            nota_indice = dicio_notas.index(nota)
            acorde_indice = dicio_acordes.index(acorde)

            # tecnica de codificacao "one-hot" (https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)
            cod_vetor_nota = codificacao_one_hot(dicio_notas_tamanho, nota_indice)
            cod_vetor_acorde = codificacao_one_hot(dicio_acordes_tamanho, acorde_indice)

            # ------------------------------------------------
            # ---------------- PROCESSAMENTO -----------------
            # ------------------------------------------------

            """ 
            MODO DE PROCESSAMENTO 5:
            - Salva todas as notas de cada compasso
            - Ignora todos silencios
            - Soma cada vetor de notas em um vetor só
            """

            if compasso_anterior is None or compasso_anterior == compasso:
                if nota != "rest":
                    sequencia_notas.append(cod_vetor_nota)

                    if compasso_anterior is None:
                        matriz_dados_saida.append(cod_vetor_acorde)

            else:
                # soma as notas nas posicoes
                sequenciaNotasSomadas = np.sum(sequencia_notas, axis=0)
                lista_musicas.append(sequenciaNotasSomadas)
                matriz_dados_entrada.append(sequenciaNotasSomadas)
                matriz_dados_saida.append(cod_vetor_acorde)

                if nota != "rest":
                    sequencia_notas = [cod_vetor_nota]
                else:
                    sequencia_notas = []

            # ------------------------------------------------
            # -------------FIM DO PROCESSAMENTO --------------
            # ------------------------------------------------

            # atualiza variavel de compasso e repete
            compasso_anterior = compasso

        # ultima sequencia de notas do ultimo compasso da musica atual
        sequencia_notas = np.sum(sequencia_notas, axis=0)

        matriz_dados_entrada.append(sequencia_notas)
        lista_musicas.append(sequencia_notas)

        vetor_saida.append(np.array(matriz_dados_saida))
        vetor_entrada.append(np.array(matriz_dados_entrada))

        matriz_dados_entrada = []
        matriz_dados_saida = []

        csv_aberto.close()