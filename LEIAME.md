# Harmonização Musical Automática

> Um framework para automatizar o processo de harmonização de uma música.

## Introdução

O problema de harmonização musical automática envolve o uso de ferramentas que analisam melodias musicais e predizem ou classificam para determinados trechos, acordes que corretamente os harmonizem.

Diversos estudos foram realizados com esse objetivo envolvendo métodos baseados em regras, estatísticos, algoritmos genéticos e redes neurais artificiais.

Este projeto tem por objetivo automatizar o processo de harmonização de uma música usando diversos métodos de redes neurais e também de modelagem de entradas e saídas.

A base de dados básica é a [*CSV Leadsheet Database*](http://marg.snu.ac.kr/chord_generation/), porém pode-se utilizar qualquer dado desde que adaptado e processado de maneira identica à existente na citada.

Este trabalho é fruto de estudos desenvolvidos durante a graduação e mestrado de [**Lucas Costa**](http://lattes.cnpq.br/8890492090241097) (quem vos escreve).

## Modo de usar

Deve-se criar um diretório na raiz do projeto chamado `data` e dentro dele, outro diretório `raw` contendo todas as músicas em formato CSV de acordo com estrutura do [**CSV Leadsheet Database**](http://marg.snu.ac.kr/chord_generation/).

Os passos seguintes, em ordem, são:
- Preprocessamento:
    - Padronização;
    - Filtro;
    - Balanço (em construção).
- Codificação:
    - Duas possibilidades. Primeira:
        - Separação antes da codificação;
        - Codificação;
    - Segunda:
        - Codificação;
        - Separação após a codificação;
        - Separação em conjuntos de treino, validação e teste.
- Treinamento:
    - `model_runner`: teste individual de modelos;
    - `optimize`: encontrar melhores hiperparâmetros;
    - `cross_val`: performar validação cruzada;
    - Em casos onde há pouca memória disponível (<8GB), pode-se usar o script `cross_val_single_execution.sh`;
    - `performance_measures` e `predict` podem ser usados individualmente para avaliar modelos;
- Visualização: geração de gráficos para visualização de resultados e dados (em desenvolvimento).

## Trabalhos relacionados

> COSTA, Lucas F. P.; OGOSHI, André Y. C.; MARTINS, Marcella S. R.; SIQUEIRA, Hugo V. Developing a Measure Image and Applying to Deep Learning. Music Encoding Conference (MEC), Halifax, Canada, 2022. - Accepted to be published in May 22, 2022.

> Lucas F. P. Costa, Tathiana M. Barchi, Erikson F. de Morais, Andrés E. Coca, Marcella S. R. Martins and Hugo V. Siqueira. Neural Networks To Automatic Musical Harmonization: A Performance Comparison. NÃO PUBLICADO - EM CONSTRUÇÃO.

> COSTA, Lucas. Harmonização Musical Automática Baseada em Redes Neurais Artificiais. 2019. 54 f. Trabalho de Conclusão de Curso – Curso de Engenharia de Computação, Universidade Tecnológica Federal do Paraná. Toledo, 2019.

> COSTA, L. F. P.; JERONYMO, D. C. . Transcrição E Harmonização De Melodias Musicais Usando Redes Neurais. In: XXIV Seminário de Iniciação Científica e Tecnológica, 2019, Pato Branco - PR. Anais Do XXIV Seminário de Iniciação Científica e Tecnológica, 2019.