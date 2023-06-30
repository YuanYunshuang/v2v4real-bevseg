#!/bin/bash

module load GCC/10.2.0  CUDA/11.1.1  OpenMPI/4.0.5 cuDNN/8.0.4.30-CUDA-11.1.1 \
Python/3.8.6 PyTorch/1.10.0 Boost/1.74.0
module load Miniconda3

conda activate testenv

PATHONPATH=. python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_fax.yaml