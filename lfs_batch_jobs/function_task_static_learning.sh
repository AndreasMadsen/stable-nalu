#!/bin/bash
export LSB_JOB_REPORT_MAIL=N
experiment_name='function_task_static_learning'
operations=( add sub mul div )
#operations=( add sub mul div squared root )
learning_rates=( 1e-5 1e-4 1e-3 1e-2 1e-1 )
verbose_flag=''
for seed in {0..24}
do
    for operation in "${operations[@]}"
    do
        for learning_rate in "${learning_rates[@]}"
        do
            # Adam
            bsub -q compute -n 1 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
                experiments/simple_function_static.py \
                --operation ${operation} --layer-type NAC \
                --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
                --learning-rate ${learning_rate} --optimizer adam \
                --name-prefix ${experiment_name} --remove-existing-data

            bsub -q compute -n 1 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
                experiments/simple_function_static.py \
                --operation ${operation} --layer-type NAC --nac-mul normal \
                --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
                --learning-rate ${learning_rate} --optimizer adam \
                --name-prefix ${experiment_name} --remove-existing-data

            bsub -q compute -n 1 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
                experiments/simple_function_static.py \
                --operation ${operation} --layer-type NALU \
                --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
                --learning-rate ${learning_rate} --optimizer adam \
                --name-prefix ${experiment_name} --remove-existing-data

            # SGD
            bsub -q compute -n 1 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
                experiments/simple_function_static.py \
                --operation ${operation} --layer-type NAC \
                --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
                --learning-rate ${learning_rate} --optimizer sgd \
                --name-prefix ${experiment_name} --remove-existing-data

            bsub -q compute -n 1 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
                experiments/simple_function_static.py \
                --operation ${operation} --layer-type NAC --nac-mul normal \
                --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
                --learning-rate ${learning_rate} --optimizer sgd \
                --name-prefix ${experiment_name} --remove-existing-data

            bsub -q compute -n 1 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
                experiments/simple_function_static.py \
                --operation ${operation} --layer-type NALU \
                --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
                --learning-rate ${learning_rate} --optimizer sgd \
                --name-prefix ${experiment_name} --remove-existing-data

            # SGD(momentum=0.9)
            bsub -q compute -n 1 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
                experiments/simple_function_static.py \
                --operation ${operation} --layer-type NAC \
                --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
                --learning-rate ${learning_rate} --optimizer sgd --momentum 0.9 \
                --name-prefix ${experiment_name} --remove-existing-data

            bsub -q compute -n 1 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
                experiments/simple_function_static.py \
                --operation ${operation} --layer-type NAC --nac-mul normal \
                --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
                --learning-rate ${learning_rate} --optimizer sgd --momentum 0.9 \
                --name-prefix ${experiment_name} --remove-existing-data

            bsub -q compute -n 1 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
                experiments/simple_function_static.py \
                --operation ${operation} --layer-type NALU \
                --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
                --learning-rate ${learning_rate} --optimizer sgd --momentum 0.9 \
                --name-prefix ${experiment_name} --remove-existing-data
        done
    done
done
