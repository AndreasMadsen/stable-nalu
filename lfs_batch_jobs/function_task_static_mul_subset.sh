#!/bin/bash
experiment_name='function_task_static_mul_subset'
operation=mul
subset_ratios=( 0.05 0.10 0.25 0.50 )
for seed in {0..9}
do
    for subset_ratio in "${subset_ratios[@]}"
    do
        bsub -q compute -n 8 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py --subset-ratio ${subset_ratio} \
            --operation ${operation} --layer-type NAC --nac-mul normal \
            --seed ${seed} --max-iterations 5000000 --verbose \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q compute -n 8 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py --subset-ratio ${subset_ratio} \
            --operation ${operation} --layer-type NALU \
            --seed ${seed} --max-iterations 5000000 --verbose \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q compute -n 8 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py --subset-ratio ${subset_ratio} \
            --operation ${operation} --layer-type ReRegualizedLinearNAC --nac-mul mnac \
            --seed ${seed} --max-iterations 5000000 --verbose \
            --name-prefix ${experiment_name} --remove-existing-data
        done
done
