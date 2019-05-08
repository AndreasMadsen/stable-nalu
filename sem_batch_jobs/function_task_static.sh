#!/bin/bash

#layer_types=( Tanh Sigmoid ReLU6 Softsign SELU ELU ReLU linear NAC NALU )
baseline_layer_types=( ReLU6 ReLU linear )
#operations=( add sub mul div squared root )
operations=( add sub mul )
seeds=( 0 1 2 3 4 5 6 7 8 9 )

for seed in "${seeds[@]}"
do
    for operation in "${operations[@]}"
    do
        for layer_type in "${baseline_layer_types[@]}"
        do
            sem -j 8 python3 experiments/simple_function_static.py \
                --operation ${operation} --layer-type ${layer_type} --seed ${seed} --verbose --remove-existing-data \
                --name-prefix nalu_simple_function_static_experiment_baseline --verbose --remove-existing-data
        done

        sem -j 8 python3 experiments/simple_function_static.py \
            --operation ${operation} --layer-type NAC --seed ${seed} --max-iterations 1000000 \
            --name-prefix nalu_simple_function_static_experiment_baseline --verbose --remove-existing-data

        sem -j 8 python3 experiments/simple_function_static.py \
            --operation ${operation} --layer-type NAC --nac-mul normal --seed ${seed} --max-iterations 1000000 \
            --name-prefix nalu_simple_function_static_experiment_baseline --verbose --remove-existing-data

        sem -j 8 python3 experiments/simple_function_static.py \
            --operation ${operation} --layer-type NALU --seed ${seed} --max-iterations 1000000 \
            --name-prefix nalu_simple_function_static_experiment_baseline --verbose --remove-existing-data
    done
done

sem --wait
