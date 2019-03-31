#!/bin/bash
mkdir -p logs

layer_types=( RNN-tanh RNN-ReLU GRU LSTM NAC NALU )
operations=( add sub mul div squared root )
seeds=( 0 1 2 3 4 5 6 7 8 9 )

for operation in "${operations[@]}"
do
    for layer_type in "${layer_types[@]}"
    do
        for seed in "${seeds[@]}"
        do
            python3 -u experiments/simple_function_recurrent.py --max-iterations 1 \
                --operation $operation --layer-type $layer_type --seed $seed \
                |& tee logs/simple_function_recurrent_$operation_$layer_type_$seed.log
        done
    done
done
