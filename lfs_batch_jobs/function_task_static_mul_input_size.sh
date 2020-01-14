#!/bin/bash
export LSB_JOB_REPORT_MAIL=N
experiment_name='function_task_static_mul_input_size'
operation=mul
input_sizes=( 4 10 25 50 75 100 125 150 175 200 225 250 275 300 )
wall_times=( 10 10 10 15 15  15  30  30  45  45  60  60  75  75 )
max_miters=(  5  5  5  5  5   5  10  10  10  10  15  15  15  15 )

verbose_flag=''
for seed in {0..49}
do
    for i in ${!input_sizes[@]}
    do
        # bsub -q compute -n 1 -W ${wall_times[$i]}:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
        #     experiments/simple_function_static.py --input-size ${input_sizes[$i]} \
        #     --operation ${operation} --layer-type NAC --nac-mul normal \
        #     --seed ${seed} --max-iterations ${max_miters[$i]}000000 ${verbose_flag} \
        #     --name-prefix ${experiment_name} --remove-existing-data

        # bsub -q compute -n 1 -W ${wall_times[$i]}:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
        #     experiments/simple_function_static.py --input-size ${input_sizes[$i]} \
        #     --operation ${operation} --layer-type PosNAC --nac-mul normal --first-layer NAC \
        #     --seed ${seed} --max-iterations ${max_miters[$i]}000000 ${verbose_flag} \
        #     --name-prefix ${experiment_name} --remove-existing-data

        # bsub -q compute -n 1 -W ${wall_times[$i]}:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
        #     experiments/simple_function_static.py --input-size ${input_sizes[$i]} \
        #     --operation ${operation} --layer-type NALU \
        #     --seed ${seed} --max-iterations ${max_miters[$i]}000000 ${verbose_flag} \
        #     --name-prefix ${experiment_name} --remove-existing-data

        # bsub -q compute -n 1 -W ${wall_times[$i]}:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
        #     experiments/simple_function_static.py --input-size ${input_sizes[$i]} \
        #     --operation ${operation} --layer-type ReRegualizedLinearNAC --nac-mul mnac \
        #     --seed ${seed} --max-iterations ${max_miters[$i]}000000 ${verbose_flag} \
        #     --name-prefix ${experiment_name} --remove-existing-data

        # bsub -q compute -n 1 -W ${wall_times[$i]}:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
        #     experiments/simple_function_static.py --input-size ${input_sizes[$i]} \
        #     --operation ${operation} --layer-type ReRegualizedLinearPosNAC --nac-mul normal --first-layer ReRegualizedLinearNAC \
        #     --seed ${seed} --max-iterations ${max_miters[$i]}000000 ${verbose_flag} \
        #     --name-prefix ${experiment_name} --remove-existing-data

        bsub -q compute -n 1 -W ${wall_times[$i]}:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/simple_function_static.py --input-size ${input_sizes[$i]} \
            --operation ${operation} --layer-type ReRegualizedLinearNALU --nalu-two-nac --nalu-mul mnac \
            --seed ${seed} --max-iterations ${max_miters[$i]}000000 ${verbose_flag} \
            --name-prefix ${experiment_name} --remove-existing-data
    done
done
