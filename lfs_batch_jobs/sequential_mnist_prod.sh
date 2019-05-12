#!/bin/bash
experiment_name='sequential_mnist_prod'

for seed in {0..10}
do
    bsub -q gpuvoltasxm2i -n 2 -gpu "num=1:mode=exclusive_process" -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name} -e /work3/$USER/logs/${experiment_name} -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation prod --layer-type NAC --nac-mul normal \
        --interpolation-length 3 --extrapolation-short-length 6 --extrapolation-long-length 9 \
        --seed ${seed} --max-epochs 10000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q gpuvoltasxm2i -n 2 -gpu "num=1:mode=exclusive_process" -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name} -e /work3/$USER/logs/${experiment_name} -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation prod --layer-type NALU \
        --interpolation-length 3 --extrapolation-short-length 6 --extrapolation-long-length 9 \
        --seed ${seed} --max-epochs 10000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q gpuvoltasxm2i -n 2 -gpu "num=1:mode=exclusive_process" -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name} -e /work3/$USER/logs/${experiment_name} -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation prod --layer-type ReRegualizedLinearNAC --nac-mul mnac \
        --interpolation-length 3 --extrapolation-short-length 6 --extrapolation-long-length 9 \
        --seed ${seed} --max-epochs 10000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data
done
