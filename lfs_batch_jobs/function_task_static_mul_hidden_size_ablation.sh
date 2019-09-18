#!/bin/bash
export LSB_JOB_REPORT_MAIL=N
experiment_name='function_task_static_mul_hidden_size_ablation'
operation=mul
hidden_sizes=( 2 3 4 5 6 7 8 9 10 )
verbose_flag=''
for seed in {0..49}
do
    for hidden_size in "${hidden_sizes[@]}"
    do

        bsub -q compute -n 1 -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=8GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py --hidden-size ${hidden_size} \
            --operation ${operation} --layer-type ReRegualizedLinearNAC --nac-mul mnac \
            --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q compute -n 1 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py --hidden-size ${hidden_size} \
            --operation ${operation} --layer-type ReRegualizedLinearNAC --nac-mul mnac \
            --regualizer 0 \
            --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q compute -n 1 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py --hidden-size ${hidden_size} \
            --operation ${operation} --layer-type ReRegualizedLinearNAC --nac-mul mnac \
            --oob-mode regualized --regualizer-oob 0 \
            --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q compute -n 1 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py --hidden-size ${hidden_size} \
            --operation ${operation} --layer-type ReRegualizedLinearNAC --nac-mul mnac \
            --oob-mode regualized --regualizer-oob 0 --regualizer 0 \
            --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
            --name-prefix ${experiment_name} --remove-existing-data
    done
done
