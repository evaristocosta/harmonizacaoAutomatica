#!/bin/bash

MODEL=$1
NEURONS=$2
TIMESTAMP=$(date +%s)
ROLLS=30

for i in $(seq $ROLLS); do
    nohup python src/system/cross_val_single.py --time $TIMESTAMP --roll $i --model $MODEL --neurons $NEURONS &> ${MODEL}_${NEURONS}.out
done

python src/system/cross_val_single_evaluate.py --time $TIMESTAMP --model $MODEL
