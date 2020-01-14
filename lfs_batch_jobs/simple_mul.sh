#!/bin/bash
export LSB_JOB_REPORT_MAIL=N
experiment_name='simple_mul'
verbose_flag=''

for seed in {0..99}
do
    # bsub -q compute -n 2 -W 0:20 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
    #     experiments/simple_function_static.py \
    #     --operation mul --layer-type NAC --nac-mul normal \
    #     --simple \
    #     --seed ${seed} --max-iterations 200000 ${verbose_flag} \
    #     --name-prefix ${experiment_name} --remove-existing-data

    # bsub -q compute -n 2 -W 0:20 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
    #     experiments/simple_function_static.py \
    #     --operation mul --layer-type PosNAC --nac-mul normal --first-layer NAC \
    #     --simple \
    #     --seed ${seed} --max-iterations 200000 ${verbose_flag} \
    #     --name-prefix ${experiment_name} --remove-existing-data

    # bsub -q compute -n 2 -W 0:20 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
    #     experiments/simple_function_static.py \
    #     --operation mul --layer-type NALU \
    #     --simple \
    #     --seed ${seed} --max-iterations 200000 ${verbose_flag} \
    #     --name-prefix ${experiment_name} --remove-existing-data

    # bsub -q compute -n 2 -W 0:20 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
    #     experiments/simple_function_static.py \
    #     --operation mul --layer-type ReRegualizedLinearNAC --nac-mul mnac \
    #     --simple \
    #     --seed ${seed} --max-iterations 200000 ${verbose_flag} \
    #     --name-prefix ${experiment_name} --remove-existing-data

    bsub -q compute -n 2 -W 0:20 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
        experiments/simple_function_static.py \
        --operation mul --layer-type ReRegualizedLinearNALU --nalu-two-nac --nalu-mul mnac \
        --simple \
        --seed ${seed} --max-iterations 200000 ${verbose_flag} \
        --name-prefix ${experiment_name} --remove-existing-data
done
