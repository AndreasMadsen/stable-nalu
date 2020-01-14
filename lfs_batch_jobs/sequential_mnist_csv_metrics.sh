#!/bin/bash
export LSB_JOB_REPORT_MAIL=N
job_name='sequential_mnist_csv_metrics'
#experiment_names=( 'sequential_mnist_prod_long' 'sequential_mnist_prod_reference' 'sequential_svhn_prod_long' 'sequential_svhn_prod_reference' )
experiment_names=( 'sequential_mnist_sum_long' 'sequential_mnist_sum_reference' )

for experiment_name in "${experiment_names[@]}"
do
    bsub -q hpc -n 16 -W 3:00 -J ${job_name} -o /work3/$USER/logs/${job_name}/ -e /work3/$USER/logs/${job_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
        export/sequential_mnist.py \
        --tensorboard-dir /work3/$USER/tensorboard/${experiment_name}/ \
        --csv-out ./results/${experiment_name}.csv
done
