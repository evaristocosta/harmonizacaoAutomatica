# Automatic Musical Harmonization

> A framework to automate the process of harmonizing a song.

## Introduction

The problem of automatic musical harmonization involves the use of tools that analyze musical melodies and predict or classify for certain parts, chords that correctly harmonize them.

Several studies were carried out with this objective involving methods based on rules, statistics, genetic algorithms and artificial neural networks.

This project aims to automate the process of harmonizing a song using different methods of neural networks and also modeling inputs and outputs.

The basic database is the [*CSV Leadsheet Database*](http://marg.snu.ac.kr/chord_generation/), however, any data can be used as long as it is adapted and processed in the same way as in the aforementioned.

This work is the result of studies developed during [**Lucas Costa's**](http://lattes.cnpq.br/8890492090241097) graduation and master's degree (who writes).

## How to use

DYou must create a directory in the project root called `data` and inside it, another `raw` directory containing all the songs in CSV format according to the [**CSV Leadsheet Database**](http://marg.snu.ac.kr/chord_generation/) structure.

The following steps, in order, are:
- Preprocessing:
    - Standardization;
    - Filter;
    - Balance (under construction).
- Encoding:
    - Two possibilities. First:
        - Separation before encoding;
        - Encode;
    - Second:
        - Encode;
        - Separation after encoding;
        - Separation into training, validation and test sets.
- Training:
    - `model_runner`: individual model testing;
    - `optimize`: find best hyperparameters;
    - `cross_val`: perform cross validation;
    - In cases where there is little memory available (<8GB), you can use the script `cross_val_single_execution.sh`;
    - `performance_measures` and `predict` can be used individually to evaluate models;
- Visualization: generation of graphs for visualization of results and data (under development).

## Related works

> COSTA, Lucas F. P.; OGOSHI, Andr?? Y. C.; MARTINS, Marcella S. R.; SIQUEIRA, Hugo V. Developing a Measure Image and Applying to Deep Learning. Music Encoding Conference (MEC), Halifax, Canada, 2022. - Accepted to be published in May 22, 2022.

> Lucas F. P. Costa, Tathiana M. Barchi, Erikson F. de Morais, Andr??s E. Coca, Marcella S. R. Martins and Hugo V. Siqueira. Neural Networks To Automatic Musical Harmonization: A Performance Comparison. N??O PUBLICADO - EM CONSTRU????O.

> COSTA, Lucas. Harmoniza????o Musical Autom??tica Baseada em Redes Neurais Artificiais. 2019. 54 f. Trabalho de Conclus??o de Curso ??? Curso de Engenharia de Computa????o, Universidade Tecnol??gica Federal do Paran??. Toledo, 2019.

> COSTA, L. F. P.; JERONYMO, D. C. . Transcri????o E Harmoniza????o De Melodias Musicais Usando Redes Neurais. In: XXIV Semin??rio de Inicia????o Cient??fica e Tecnol??gica, 2019, Pato Branco - PR. Anais Do XXIV Semin??rio de Inicia????o Cient??fica e Tecnol??gica, 2019.