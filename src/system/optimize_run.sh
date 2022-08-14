#!/bin/bash

for i in $(seq 50); do
    nohup python src/system/optimize_keras_tuner.py &> alexnet_opt_${i}.out
done

sudo poweroff
