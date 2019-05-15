#!/bin/sh

export CUDA_VERSION='10.0'
export CUDNN_VERSION='7.4.2.24'
export TENSORBOARD_DIR=/work3/$USER/tensorboard
export SAVE_DIR=/work3/$USER/saves

module load python3
module load gcc/4.9.2
module load cuda/$CUDA_VERSION
module load cudnn/v$CUDNN_VERSION-prod-cuda-$CUDA_VERSION

export PYTHONPATH=./
source ~/stdpy3/bin/activate

python3 -u "$@"