#!/bin/bash
export LSB_JOB_REPORT_MAIL=N
experiment_name='sequential_svhn_prod_long'
resnet_models=( resnet18 resnet34 resnet50 )

for seed in {0..4}
do
    for resnet_model in "${resnet_models[@]}"
    do
        bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 10:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/sequential_svhn.py \
            --operation cumprod --layer-type NAC --nac-mul normal \
            --svhn-digits 123456789 --svhn-outputs 1 --resnet ${resnet_model} \
            --interpolation-length 2 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]' \
            --seed ${seed} --max-epochs 1000 --verbose \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 10:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/sequential_svhn.py \
            --operation cumprod --layer-type PosNAC --nac-mul normal \
            --svhn-digits 123456789 --svhn-outputs 1 --resnet ${resnet_model} \
            --interpolation-length 2 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]' \
            --seed ${seed} --max-epochs 1000 --verbose \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 10:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/sequential_svhn.py \
            --operation cumprod --layer-type NALU \
            --svhn-digits 123456789 --svhn-outputs 1 --resnet ${resnet_model} \
            --interpolation-length 2 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]' \
            --seed ${seed} --max-epochs 1000 --verbose \
            --name-prefix ${experiment_name} --remove-existing-data

        # bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 6:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
        #     experiments/sequential_svhn.py \
        #     --operation cumprod --layer-type ReRegualizedLinearNAC --nac-mul mnac \
        #     --svhn-digits 123456789 --svhn-outputs 1 --resnet ${resnet_model} \
        #     --regualizer-z 0 \
        #     --interpolation-length 2 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]' \
        #     --seed ${seed} --max-epochs 250 --verbose \
        #     --name-prefix ${experiment_name} --remove-existing-data

        bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 10:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/sequential_svhn.py \
            --operation cumprod --layer-type ReRegualizedLinearNAC --nac-mul mnac \
            --svhn-digits 123456789 --svhn-outputs 1 --resnet ${resnet_model} \
            --regualizer-z 1 \
            --interpolation-length 2 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]' \
            --seed ${seed} --max-epochs 1000 --verbose \
            --name-prefix ${experiment_name} --remove-existing-data

        # bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 6:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
        #     experiments/sequential_svhn.py \
        #     --operation cumprod --layer-type ReRegualizedLinearPosNAC --nac-mul normal \
        #     --svhn-digits 123456789 --svhn-outputs 1 --resnet ${resnet_model} \
        #     --regualizer-z 0 \
        #     --interpolation-length 2 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]' \
        #     --seed ${seed} --max-epochs 250 --verbose \
        #     --name-prefix ${experiment_name} --remove-existing-data

        bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 10:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/sequential_svhn.py \
            --operation cumprod --layer-type ReRegualizedLinearPosNAC --nac-mul normal \
            --svhn-digits 123456789 --svhn-outputs 1 --resnet ${resnet_model} \
            --regualizer-z 1 \
            --interpolation-length 2 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]' \
            --seed ${seed} --max-epochs 1000 --verbose \
            --name-prefix ${experiment_name} --remove-existing-data

        # bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 6:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
        #     experiments/sequential_svhn.py \
        #     --operation cumprod --layer-type LSTM \
        #     --svhn-digits 123456789 --svhn-outputs 1 --resnet ${resnet_model} \
        #     --interpolation-length 2 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]' \
        #     --seed ${seed} --max-epochs 250 --verbose \
        #     --name-prefix ${experiment_name} --remove-existing-data
    done
done
