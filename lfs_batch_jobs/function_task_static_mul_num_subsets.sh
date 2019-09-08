#!/bin/bash
export LSB_JOB_REPORT_MAIL=N
experiment_name='function_task_static_mul_num_subsets'
operation=mul
num_subsets=( 2 3 4 5 2 3 4 5 )
hidden_size=( 2 3 4 5 4 6 8 10 )
verbose_flag=''
for seed in {0..24}
do
    for i in "${!num_subsets[@]}"
    do
        bsub -q compute -n 1 -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py \
            --hidden-size ${hidden_size[$i]} --num-subsets ${num_subsets[$i]} \
            --operation ${operation} --layer-type NAC --nac-mul normal \
            --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q compute -n 1 -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py \
            --hidden-size ${hidden_size[$i]} --num-subsets ${num_subsets[$i]} \
            --operation ${operation} --layer-type PosNAC --nac-mul normal --first-layer NAC \
            --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q compute -n 1 -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py \
            --hidden-size ${hidden_size[$i]} --num-subsets ${num_subsets[$i]} \
            --operation ${operation} --layer-type NALU \
            --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q compute -n 1 -W 24:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py \
            --hidden-size ${hidden_size[$i]} --num-subsets ${num_subsets[$i]} \
            --operation ${operation} --layer-type ReRegualizedLinearNAC --nac-mul mnac \
            --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q compute -n 1 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py --num-subsets ${num_subsets[$i]} \
            --operation ${operation} --layer-type ReRegualizedLinearPosNAC --nac-mul normal --first-layer ReRegualizedLinearNAC \
            --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
            --name-prefix ${experiment_name} --remove-existing-data
    done
done
