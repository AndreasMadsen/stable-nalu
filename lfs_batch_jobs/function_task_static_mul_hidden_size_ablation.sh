#!/bin/bash
export LSB_JOB_REPORT_MAIL=N
experiment_name='function_task_static_mul_hidden_size_ablation'
operation=mul
hidden_sizes=( 2 3 4 5 6 )
verbose_flag=''
for seed in {0..24}
do
    for hidden_size in "${hidden_sizes[@]}"
    do
        bsub -q compute -n 2 -W 14:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py --hidden-size ${hidden_size} \
            --operation ${operation} --layer-type SillyReRegualizedLinearNAC --nac-mul mnac --first-layer ReRegualizedLinearNAC \
            --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
            --name-prefix ${experiment_name} --remove-existing-data
    done
done
