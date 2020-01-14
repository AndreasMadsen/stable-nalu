#!/bin/bash
export LSB_JOB_REPORT_MAIL=N
experiment_name='function_task_static_nalu'
operations=( add mul )
verbose_flag=''
for seed in {0..99}
do
    for operation in "${operations[@]}"
    do
        # bsub -q compute -n 1 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
        #     experiments/simple_function_static.py \
        #     --operation ${operation} --layer-type NALU --first-layer NAC \
        #     --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
        #     --name-prefix ${experiment_name} --remove-existing-data

        # bsub -q compute -n 1 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
        #     experiments/simple_function_static.py \
        #     --operation ${operation} --layer-type NALU --first-layer NAC --nalu-two-nac \
        #     --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
        #     --name-prefix ${experiment_name} --remove-existing-data

        bsub -q compute -n 1 -W 12:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py \
            --operation ${operation} --layer-type ReRegualizedLinearNALU --first-layer ReRegualizedLinearNAC --nalu-two-nac --nalu-mul mnac \
            --seed ${seed} --max-iterations 5000000 ${verbose_flag} \
            --name-prefix ${experiment_name} --remove-existing-data
    done
done
