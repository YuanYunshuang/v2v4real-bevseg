#!/bin/bash --login
#SBATCH --job-name=web-login/sys/myjobs/default/v2vreal_bev
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output test_gou-job_%j.out
#SBATCH --error test_gpu-job_%j.err

#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=1:00
nvidia-smi

# run program
echo "I am running on $HOSTNAME"
module load cuDNN/8.0.4.30-CUDA-11.1.1 \
GCC/10.2.0  CUDA/11.1.1  OpenMPI/4.0.5 \
PyTorch/1.10.0 \
Python/3.8.6 \
Boost/1.74.0 \
Miniconda3

conda activate testenv

PATHONPATH=. python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_fax.yaml