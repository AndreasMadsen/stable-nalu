
job_name='function_task_static_csv_metrics'
experiment_names=( 'function_task_static_mul_input_size' 'function_task_static_mul_overlap' 'function_task_static_mul_range' 'function_task_static_mul_subset' 'function_task_static_regualization' 'function_task_static_ablation' 'function_task_static' 'simple_mul' )

for experiment_name in "${experiment_names[@]}"
do
    bsub -q compute -n 16 -W 3:00 -J ${job_name} -o /work3/$USER/logs/${job_name}/ -e /work3/$USER/logs/${job_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        export/simple_function_static.py \
        --tensorboard-dir /work3/$USER/tensorboard/${experiment_name}/ \
        --csv-out ./results/${experiment_name}.csv
done
