#!/bin/bash

for seed in {0..99}
do
    sem -j 16 python3 experiments/simple_function_static.py --simple \
        --operation mul --layer-type NAC --nac-mul normal --seed ${seed} --max-iterations 200000 \
        --name-prefix nac_mul_simple --verbose --remove-existing-data

    sem -j 16 python3 experiments/simple_function_static.py --simple \
        --operation mul --layer-type NALU --seed ${seed} --max-iterations 200000 \
        --name-prefix nac_mul_simple --verbose --remove-existing-data

    sem -j 16 python3 experiments/simple_function_static.py --simple \
        --operation mul --layer-type ReRegualizedLinearNAC --nac-mul mnac --seed ${seed} --max-iterations 200000 \
        --name-prefix nac_mul_simple --verbose --remove-existing-data
done
