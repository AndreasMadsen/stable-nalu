#!/bin/bash
experiment_name='function_task_static_ablation'

for seed in {0..9}
do
    bsub -q compute -n 8 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/simple_function_static.py \
        --operation mul --layer-type NAC --nac-mul normal \
        --seed ${seed} --max-iterations 5000000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q compute -n 8 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/simple_function_static.py \
        --operation mul --layer-type PosNAC --nac-mul normal --first-layer NAC \
        --seed ${seed} --max-iterations 5000000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q compute -n 8 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/simple_function_static.py \
        --operation mul --layer-type ReRegualizedLinearNAC --nac-mul mnac \
        --seed ${seed} --max-iterations 5000000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q compute -n 8 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/simple_function_static.py \
        --operation mul --layer-type SillyReRegualizedLinearNAC --nac-mul mnac --first-layer ReRegualizedLinearNAC \
        --seed ${seed} --max-iterations 5000000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q compute -n 8 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/simple_function_static.py \
        --operation mul --layer-type ReRegualizedLinearNAC --nac-mul mnac \
        --regualizer 0 \
        --seed ${seed} --max-iterations 5000000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q compute -n 8 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/simple_function_static.py \
        --operation mul --layer-type ReRegualizedLinearNAC --nac-mul mnac \
        --regualizer-oob 0 \
        --seed ${seed} --max-iterations 5000000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data
done
