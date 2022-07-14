import csv
import numpy as np
from npy_append_array import NpyAppendArray
from math import modf
from .dictionary import dicio_notas, dicio_acordes
from .one_hot_function import codificacao_one_hot
from .rgba2rgb import rgba2rgb

PRECISAO = 0.000001


def codificacao_nota(nota, oitava, resto, indice):
    # pega tamanho do dicionario (menos pausa)
    dicio_notas_tamanho = len(dicio_notas) - 1

    # pega posicao da nota no dicionario
    nota_indice = dicio_notas.index(nota)

    # One-hot
    # vetor de zeros
    vetor = [0] * dicio_notas_tamanho
    # posicao do 1 no vetor
    # se for pausa, não considera
    if nota != "rest":
        vetor[nota_indice] = 1

    # razão considerando 10 oitavas (int(255/10)) ou 1/10
    razao = 1 / 10

    alfa = 1
    cod_nota = []
    cod_nota_primeira = []
    cod_nota_final = []

    for item in vetor:
        if item == 0:
            cod_nota.append([0, 0, 0, alfa])
            cod_nota_primeira.append([0, 0, 0, alfa])
            cod_nota_final.append([0, 0, 0, alfa])
        else:
            cod_nota.append([0, 0, razao * float(oitava), alfa])
            cod_nota_primeira.append([0, 0, razao * float(oitava), alfa])
            cod_nota_final.append([1, 0, 0, alfa * resto if resto != 0 else alfa])

    # se for apogiatura, marca como verde
    if indice is not None:
        cod_nota_primeira[indice] = [0, razao * float(oitava), 0, alfa]

    # alargador
    cod_nota = [item for item in cod_nota for _ in range(8)]
    cod_nota_primeira = [item for item in cod_nota_primeira for _ in range(8)]
    cod_nota_final = [item for item in cod_nota_final for _ in range(8)]
    return cod_nota_primeira, cod_nota, cod_nota_final


def sequencia_notas_nao_valida(sequencia_notas):
    if len(sequencia_notas) == 0:
        return False

    # verifica se é todo de pausas
    # https://stackoverflow.com/questions/37777529/comparing-multiple-numpy-arrays
    verificador = np.stack(sequencia_notas, axis=0)
    tudo_igual = np.max(np.abs(np.diff(verificador, axis=0))) <= 0
    # verifica se tamanho está correto
    altura = len(sequencia_notas) != 96
    largura = len(sequencia_notas[0]) != 96
    if tudo_igual or altura or largura:
        return True
    else:
        return False


def cria_sequencia_notas(primeira_nota, nota, ultima_nota, quantidade):
    sequencia = []
    # salva as notas dessa sequencia
    sequencia.append(np.array(primeira_nota))

    for _ in range(quantidade - 2):
        sequencia.append(np.array(nota))

    sequencia.append(np.array(ultima_nota))

    sequencia = np.array(sequencia)

    return sequencia


def processamento_ic(arquivos_csv, rgb=False):
    dicio_acordes_tamanho = len(dicio_acordes)

    # declaracao das matrizes
    matriz_dados_entrada = []
    matriz_dados_saida = []

    # arquivos de salvamento
    vetor_entrada = NpyAppendArray("data/encoded/vetor_entrada.npy")
    vetor_saida = NpyAppendArray("data/encoded/vetor_saida.npy")

    # construcao das matrizes a partir dos arquivos csv
    for i, caminho_csv in enumerate(arquivos_csv):
        print(caminho_csv, "--", i + 1)

        csv_aberto = open(caminho_csv, "r", encoding="utf-8")
        next(csv_aberto)
        leitor = csv.reader(csv_aberto)

        # lista da sequencia de notas sendo processada
        sequencia_notas = []
        controle_resto = 0

        # variavel de controle de compassos
        compasso_anterior = None
        indice = None

        for j, linha in enumerate(leitor):
            # processamento reduziu o banco de dados a 5 colunas de infomacao:
            #    0       1      2      3        4
            # measure, chord, note, octave, duration
            compasso = int(linha[0])
            acorde = linha[1]
            nota = linha[2]
            oitava = linha[3]
            duracao = float(linha[4])

            resto, inteiro = modf(duracao)
            inteiro = int(inteiro)
            controle_resto += resto

            if 1 - controle_resto < PRECISAO:
                inteiro += 1
                # pode ser que controle_resto = 0.9999999999999...
                controle_resto = 0 if controle_resto < PRECISAO else controle_resto - 1

            # tem casos onde controle_resto fica extremamente pequeno e pode ser desconsiderado
            if controle_resto < PRECISAO:
                controle_resto = 0

            # appoggiatura
            if duracao == 0:
                indice = dicio_notas.index(nota)
                continue

            # codificacao da nota em forma de linha de pixels
            (
                cod_vetor_primeira_nota,
                cod_vetor_nota,
                cod_vetor_ultima_nota,
            ) = codificacao_nota(nota, oitava, controle_resto, indice)

            # salva os compassos
            if compasso_anterior is None or compasso_anterior == compasso:
                sequencia_notas.extend(
                    cria_sequencia_notas(
                        cod_vetor_primeira_nota,
                        cod_vetor_nota,
                        cod_vetor_ultima_nota,
                        inteiro,
                    )
                )

            else:
                if sequencia_notas_nao_valida(sequencia_notas):
                    sequencia_notas = []
                else:
                    sequencia_notas = np.array(sequencia_notas)
                    if rgb and len(sequencia_notas.shape) == 3:
                        sequencia_notas = rgba2rgb(sequencia_notas)

                if len(sequencia_notas) > 0:
                    matriz_dados_entrada.append(sequencia_notas)
                    matriz_dados_saida.append(cod_vetor_acorde)

                sequencia_notas = []
                # salva as notas dessa sequencia
                sequencia_notas.extend(
                    cria_sequencia_notas(
                        cod_vetor_primeira_nota,
                        cod_vetor_nota,
                        cod_vetor_ultima_nota,
                        inteiro,
                    )
                )

            # gera acorde depois de verificar o compasso
            acorde_indice = dicio_acordes.index(acorde)
            cod_vetor_acorde = codificacao_one_hot(dicio_acordes_tamanho, acorde_indice)

            # atualiza variavel de compasso e repete
            compasso_anterior = compasso
            indice = None

        if sequencia_notas_nao_valida(sequencia_notas):
            sequencia_notas = []
        else:
            sequencia_notas = np.array(sequencia_notas)
            if rgb and len(sequencia_notas.shape) == 3:
                sequencia_notas = rgba2rgb(sequencia_notas)

        if len(sequencia_notas) > 0:
            matriz_dados_entrada.append(sequencia_notas)
            matriz_dados_saida.append(cod_vetor_acorde)

        if matriz_dados_entrada and matriz_dados_saida:
            vetor_saida.append(np.array(matriz_dados_saida))
            vetor_entrada.append(np.array(matriz_dados_entrada))

        matriz_dados_entrada = []
        matriz_dados_saida = []

        csv_aberto.close()
