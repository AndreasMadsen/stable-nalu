#!/bin/bash
experiment_name='function_task_static_mul_range'
operation=mul
interpolation_ranges=( '[-2,2]' '[0,1]' '[0.1,0.2]' '[1,2]' '[10,20]' )
extrapolation_ranges=( '[-6,6]' '[0,5]' '[0,2]' '[1,6]' '[1,40]' )
for seed in {0..9}
do
    for i in ${!interpolation_ranges[@]}
    do
        bsub -q compute -n 8 -W 14:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py --interpolation-range ${interpolation_ranges[$i]} --extrapolation-range ${extrapolation_ranges[$i]} \
            --operation ${operation} --layer-type NAC --nac-mul normal \
            --seed ${seed} --max-iterations 5000000 --verbose \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q compute -n 8 -W 14:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py --interpolation-range ${interpolation_ranges[$i]} --extrapolation-range ${extrapolation_ranges[$i]} \
            --operation ${operation} --layer-type NALU \
            --seed ${seed} --max-iterations 5000000 --verbose \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q compute -n 8 -W 14:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py --interpolation-range ${interpolation_ranges[$i]} --extrapolation-range ${extrapolation_ranges[$i]} \
            --operation ${operation} --layer-type ReRegualizedLinearNAC --nac-mul mnac \
            --seed ${seed} --max-iterations 5000000 --verbose \
            --name-prefix ${experiment_name} --remove-existing-data
    done
done
