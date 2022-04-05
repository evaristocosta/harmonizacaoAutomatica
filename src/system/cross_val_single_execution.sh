#!/bin/bash

TIMESTAMP=$(date +%s)
ROLLS=30

for i in $(seq $ROLLS); do
    python src/system/cross_val_single.py --time $TIMESTAMP --roll $i
done

python src/system/cross_val_single_evaluate.py --time $TIMESTAMP
