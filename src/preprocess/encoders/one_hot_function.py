def codificacao_one_hot(length, one_index):
    # Retorna vetor codificado
    # vetor de zeros
    vetor = [0] * length
    # posicao do 1 no vetor
    vetor[one_index] = 1

    return vetor
