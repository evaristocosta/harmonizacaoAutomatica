import csv
import numpy as np
from npy_append_array import NpyAppendArray
from .dictionary import dicio_notas, dicio_acordes
from .one_hot_function import codificacao_one_hot


def processamento_one_hot(arquivos_csv, nome="vetor"):
    # pega tamanho dos dicionarios
    dicio_notas_tamanho = len(dicio_notas) - 1  # -1 para o rest
    dicio_acordes_tamanho = len(dicio_acordes)

    # declaracao das matrizes
    matriz_dados_entrada = []
    matriz_dados_saida = []

    # arquivos de salvamento
    vetor_entrada = NpyAppendArray(f"data/encoded/{nome}_entrada.npy")
    vetor_saida = NpyAppendArray(f"data/encoded/{nome}_saida.npy")

    # construcao das matrizes a partir dos arquivos csv
    for i, caminho_csv in enumerate(arquivos_csv):
        print(caminho_csv, "--", i + 1)

        csv_aberto = open(caminho_csv, "r", encoding="utf-8")
        next(csv_aberto)
        leitor = csv.reader(csv_aberto)

        # lista da sequencia de notas sendo processada
        sequencia_notas = []
        # variavel de controle de compassos
        compasso_anterior = None

        for linha in leitor:
            # padronização reduziu o banco de dados a 5 colunas de infomacao:
            #    0       1      2      3        4
            # measure, chord, note, octave, duration
            compasso = int(linha[0])
            acorde = linha[1]
            nota = linha[2]

            # pausas não importam
            if nota == "rest":
                continue

            # pega posicao da nota no dicionario
            nota_indice = dicio_notas.index(nota)
            acorde_indice = dicio_acordes.index(acorde)

            # tecnica de codificacao "one-hot" (https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)
            cod_vetor_nota = codificacao_one_hot(dicio_notas_tamanho, nota_indice)
            cod_vetor_acorde = codificacao_one_hot(dicio_acordes_tamanho, acorde_indice)

            if compasso_anterior is None or compasso_anterior == compasso:
                sequencia_notas.append(cod_vetor_nota)

                if compasso_anterior is None:
                    matriz_dados_saida.append(cod_vetor_acorde)

            else:
                # soma as notas nas posicoes
                sequencia_notas_somadas = np.sum(sequencia_notas, axis=0)
                # normaliza o vetor
                sequencia_notas_somadas = sequencia_notas_somadas / max(
                    sequencia_notas_somadas
                )

                matriz_dados_entrada.append(sequencia_notas_somadas)
                matriz_dados_saida.append(cod_vetor_acorde)

                sequencia_notas = [cod_vetor_nota]

            # atualiza variavel de compasso e repete
            compasso_anterior = compasso

        # ultima sequencia de notas do ultimo compasso da musica atual
        if len(sequencia_notas) == 0:
            continue

        sequencia_notas = np.sum(sequencia_notas, axis=0)

        # normaliza o vetor
        sequencia_notas = sequencia_notas / max(sequencia_notas)

        matriz_dados_entrada.append(sequencia_notas)

        if matriz_dados_entrada and matriz_dados_saida:
            vetor_saida.append(np.array(matriz_dados_saida))
            vetor_entrada.append(np.array(matriz_dados_entrada))

        matriz_dados_entrada = []
        matriz_dados_saida = []

        csv_aberto.close()
