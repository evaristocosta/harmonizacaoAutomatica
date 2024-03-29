#!/bin/bash

MODEL=$1
BATCH=$2
EPOCH=$3
OPTIMIZER=$4
LR=$5
MOMENTUM=$6
NESTEROV=$7
TIMESTAMP=$(date +%s)
ROLLS=$8
ROLLSTART=$9

for i in $(seq $ROLLS); do
    nohup python src/system/cross_val_single.py --time $TIMESTAMP --roll $((i+ROLLSTART)) --model $MODEL --batch $BATCH --epoch $EPOCH --optimizer $OPTIMIZER --lr $LR --momentum $MOMENTUM --nesterov $NESTEROV &> ${MODEL}_${BATCH}.out
done

python src/system/cross_val_single_evaluate.py --time $TIMESTAMP --model $MODEL


# src/system/cross_val_apps_single_execution.sh alexnet 128 90 sgd 0.1 0.9 0 &
# src/system/cross_val_apps_single_execution.sh vgg16 256 100 sgd 0.01 0.9 0 6 4 &
# src/system/cross_val_apps_single_execution.sh resnet101 256 400 sgd 0.0001 0.9 0 1 5 &
# src/system/cross_val_apps_single_execution.sh inceptionv3 32 100 rmsprop 0.045 1.0 0 10 0 &
# src/system/cross_val_apps_single_execution.sh densenet201 256 90 sgd 0.1 0.9 1 10 0 &

# src/system/cross_val_apps_single_execution.sh alexnet_optimization 64 20 sgd 1 0.9 0 10 0 &