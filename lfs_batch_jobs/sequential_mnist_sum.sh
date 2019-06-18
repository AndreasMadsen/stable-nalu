#!/bin/bash
experiment_name='sequential_mnist_sum'

for seed in {0..4}
do
    bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation cumsum --layer-type NAC \
        --interpolation-length 10 --extrapolation-lengths '[1,10,100,1000]' \
        --seed ${seed} --max-epochs 10000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation cumsum --layer-type NALU \
        --interpolation-length 10 --extrapolation-lengths '[1,10,100,1000]' \
        --seed ${seed} --max-epochs 10000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation cumsum --layer-type ReRegualizedLinearNAC \
        --interpolation-length 10 --extrapolation-lengths '[1,10,100,1000]' \
        --seed ${seed} --max-epochs 10000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation cumsum --layer-type LSTM \
        --interpolation-length 10 --extrapolation-lengths '[1,10,100,1000]' \
        --seed ${seed} --max-epochs 10000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data
done
