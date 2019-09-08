#!/bin/bash
export LSB_JOB_REPORT_MAIL=N
experiment_name='sequential_mnist_prod_reference'

for seed in {0..9}
do
    bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 8:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation cumprod --layer-type ReRegualizedLinearNAC --nac-mul mnac --mnist-digits 123456789 \
        --model-simplification solved-accumulator \
        --interpolation-length 2 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]' \
        --seed ${seed} --max-epochs 1000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data
done
