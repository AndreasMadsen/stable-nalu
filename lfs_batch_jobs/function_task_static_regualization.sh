#!/bin/bash
experiment_name='function_task_static_regualization'
regualizations=( 0 0.01 0.1 1 10 )
#operations=( add sub mul div squared root )
verbose_flag=''
for seed in {0..9}
do
    for regualization in "${regualizations[@]}"
    do
        bsub -q compute -n 8 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py \
            --operation add --layer-type ReRegualizedLinearNAC \
            --regualizer ${regualization} --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q compute -n 8 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py \
            --operation sub --layer-type ReRegualizedLinearNAC \
            --regualizer ${regualization} --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q compute -n 8 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py \
            --operation mul --layer-type ReRegualizedLinearNAC --nac-mul mnac \
            --regualizer ${regualization} --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
            --name-prefix ${experiment_name} --remove-existing-data
    done
done
