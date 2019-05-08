#!/bin/bash
experiment_name = 'simple_mul'

for seed in {0..99}
do
    bsub -q compute -n 8 -W 0:20 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name} -e /work3/$USER/logs/${experiment_name} -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/simple_function_static.py \
        --operation mul --layer-type NAC --nac-mul normal \
        --simple \
        --seed ${seed} --max-iterations 200000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q compute -n 8 -W 0:20 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name} -e /work3/$USER/logs/${experiment_name} -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/simple_function_static.py \
        --operation mul --layer-type NALU \
        --simple \
        --seed ${seed} --max-iterations 200000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q compute -n 8 -W 0:20 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name} -e /work3/$USER/logs/${experiment_name} -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/simple_function_static.py \
        --operation mul --layer-type ReRegualizedLinearNAC --nac-mul mnac \
        --simple \
        --seed ${seed} --max-iterations 200000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data
done
