#!/bin/bash
export LSB_JOB_REPORT_MAIL=N
job_name='function_task_static_csv_metrics'
# experiment_names=( 'function_task_static' 'function_task_static_learning' 'function_task_static_mul_hidden_size' 'function_task_static_mul_hidden_size_ablation' 'function_task_static_mul_input_size' 'function_task_static_mul_overlap' 'function_task_static_mul_subset' 'function_task_static_mul_range' 'function_task_static_regualization' 'simple_mul' )
experiment_names=( 'function_task_static' 'function_task_static_mul_hidden_size' 'function_task_static_mul_input_size' 'function_task_static_mul_overlap' 'function_task_static_mul_subset' 'function_task_static_mul_range' 'simple_mul' )
for experiment_name in "${experiment_names[@]}"
do
    bsub -q hpc -n 16 -W 3:00 -J ${job_name} -o /work3/$USER/logs/${job_name}/ -e /work3/$USER/logs/${job_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
        export/simple_function_static.py \
        --tensorboard-dir /work3/$USER/tensorboard/${experiment_name}/ \
        --csv-out ./results/${experiment_name}.csv
done

# bsub -q hpc -n 16 -W 3:00 -J ${job_name} -o /work3/$USER/logs/${job_name}/ -e /work3/$USER/logs/${job_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
#     export/simple_function_static.py \
#     --tensorboard-dir /work3/$USER/tensorboard/function_task_static_nalu/ \
#     --csv-out ./results/function_task_static_nalu.csv \
#     --export-gate
