#!/bin/bash
export LSB_JOB_REPORT_MAIL=N
experiment_name='sequential_mnist_sum_long'

for seed in {0..9}
do
    # bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 6:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
    #     experiments/sequential_mnist.py \
    #     --operation cumsum --layer-type NAC \
    #     --regualizer-z 0 \
    #     --interpolation-length 10 --extrapolation-lengths '[1,10,100,200,300,400,500,600,700,800,900,1000]' \
    #     --seed ${seed} --max-epochs 1000 --verbose \
    #     --name-prefix ${experiment_name} --remove-existing-data

    bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 6:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation cumsum --layer-type NAC \
        --regualizer-z 1 \
        --interpolation-length 10 --extrapolation-lengths '[1,10,100,200,300,400,500,600,700,800,900,1000]' \
        --seed ${seed} --max-epochs 1000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    # bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 6:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
    #     experiments/sequential_mnist.py \
    #     --operation cumsum --layer-type NALU \
    #     --interpolation-length 10 --extrapolation-lengths '[1,10,100,200,300,400,500,600,700,800,900,1000]' \
    #     --seed ${seed} --max-epochs 1000 --verbose \
    #     --name-prefix ${experiment_name} --remove-existing-data

    # bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 6:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
    #     experiments/sequential_mnist.py \
    #     --operation cumsum --layer-type ReRegualizedLinearNAC \
    #     --interpolation-length 10 --extrapolation-lengths '[1,10,100,200,300,400,500,600,700,800,900,1000]' \
    #     --regualizer-z 0 \
    #     --seed ${seed} --max-epochs 1000 --verbose \
    #     --name-prefix ${experiment_name} --remove-existing-data

    bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 6:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation cumsum --layer-type ReRegualizedLinearNAC \
        --interpolation-length 10 --extrapolation-lengths '[1,10,100,200,300,400,500,600,700,800,900,1000]' \
        --regualizer-z 1 \
        --seed ${seed} --max-epochs 1000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    # bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 6:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
    #     experiments/sequential_mnist.py \
    #     --operation cumsum --layer-type LSTM \
    #     --interpolation-length 10 --extrapolation-lengths '[1,10,100,200,300,400,500,600,700,800,900,1000]' \
    #     --seed ${seed} --max-epochs 1000 --verbose \
    #     --name-prefix ${experiment_name} --remove-existing-data
done
