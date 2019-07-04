#!/bin/bash
experiment_name='sequential_mnist_prod_debug'

for seed in {0..1}
do
    # bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
    #     experiments/sequential_mnist.py \
    #     --operation cumprod --layer-type ReRegualizedLinearNAC --nac-mul mnac --mnist-digits 123456789 \
    #     --interpolation-length 3 --extrapolation-lengths '[1,2,3,4,5]' \
    #     --seed ${seed} --max-epochs 10000 --verbose \
    #     --name-prefix ${experiment_name} --remove-existing-data

    bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation cumprod --layer-type ReRegualizedLinearNAC --nac-mul mnac --mnist-digits 123456789 \
        --softmax-transform \
        --interpolation-length 3 --extrapolation-lengths '[1,2,3,4,5]' \
        --seed ${seed} --max-epochs 10000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation cumprod --layer-type ReRegualizedLinearNAC --nac-mul mnac --mnist-digits 23 \
        --softmax-transform \
        --interpolation-length 3 --extrapolation-lengths '[1,2,3,4,5]' \
        --seed ${seed} --max-epochs 10000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation cumprod --layer-type ReRegualizedLinearNAC --nac-mul mnac --mnist-digits 123456789 \
        --solved-accumulator --softmax-transform \
        --interpolation-length 3 --extrapolation-lengths '[1,2,3,4,5]' \
        --seed ${seed} --max-epochs 10000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation cumprod --layer-type ReRegualizedLinearNAC --nac-mul mnac --mnist-digits 23 \
        --solved-accumulator --softmax-transform \
        --interpolation-length 3 --extrapolation-lengths '[1,2,3,4,5]' \
        --seed ${seed} --max-epochs 10000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    # bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
    #     experiments/sequential_mnist.py \
    #     --operation cumprod --layer-type ReRegualizedLinearNAC --nac-mul mnac --mnist-digits 123456789 \
    #     --solved-accumulator \
    #     --interpolation-length 3 --extrapolation-lengths '[1,2,3,4,5]' \
    #     --seed ${seed} --max-epochs 10000 --verbose \
    #     --name-prefix ${experiment_name} --remove-existing-data
done

python3 experiments/sequential_mnist.py \
--operation cumprod --layer-type ReRegualizedLinearNAC --nac-mul mnac \
--mnist-digits 123456789 --mnist-outputs 2 --regualizer-z 1 --regualizer 100 \
--interpolation-length 2 --extrapolation-lengths '[1,2,3]' \
--seed ${seed} --max-epochs 1000 --verbose \
--name-prefix ${experiment_name} --remove-existing-data

python3 experiments/sequential_mnist.py \
--operation cumprod --layer-type ReRegualizedLinearNAC --nac-mul mnac \
--mnist-digits 123456789 --mnist-outputs 1 --regualizer-z 1 --regualizer 100 \
--interpolation-length 2 --extrapolation-lengths '[1,2,3]' \
--seed ${seed} --max-epochs 1000 --verbose \
--name-prefix ${experiment_name} --remove-existing-data

python3 experiments/sequential_mnist.py \
--operation cumprod --layer-type ReRegualizedLinearNAC --nac-mul mnac \
--mnist-digits 123456789 --mnist-outputs 1 --regualizer-z 1 \
--model-simplification pass-through \
--interpolation-length 1 --extrapolation-lengths '[1]' \
--seed ${seed} --max-epochs 1000 --verbose \
--name-prefix ${experiment_name} --remove-existing-data

python3 experiments/sequential_mnist.py \
--operation cumprod --layer-type ReRegualizedLinearNAC --nac-mul mnac \
--mnist-digits 123456789 --mnist-outputs 1 \
--model-simplification solved-accumulator \
--interpolation-length 2 --extrapolation-lengths '[1,2,3]' \
--seed ${seed} --max-epochs 1000 --verbose \
--name-prefix ${experiment_name} --remove-existing-data
