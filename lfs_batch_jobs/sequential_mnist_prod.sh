#!/bin/bash
experiment_name='sequential_mnist_prod'

for seed in {0..4}
do
    bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation cumprod --layer-type NAC --nac-mul normal --remove-zero \
        --interpolation-length 3 --extrapolation-lengths '[1,2,3,5,6,7,8,9]' \
        --seed ${seed} --max-epochs 10000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation cumprod --layer-type NALU --remove-zero \
        --interpolation-length 3 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9]' \
        --seed ${seed} --max-epochs 10000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation cumprod --layer-type ReRegualizedLinearNAC --nac-mul mnac --remove-zero \
        --interpolation-length 3 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9]' \
        --seed ${seed} --max-epochs 10000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation cumprod --layer-type LSTM --remove-zero \
        --interpolation-length 3 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9]' \
        --seed ${seed} --max-epochs 10000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data
done
