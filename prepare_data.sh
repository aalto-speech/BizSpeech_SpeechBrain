#!/bin/bash
#SBATCH --job-name=200_biz
#SBATCH --constraint hsw
#SBATCH -N 1                 # on one node
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=1GB
#SBATCH --exclusive

module load cuda
python tokenizer_train.py hparams/tokenizer.yaml
#python train.py hparams/train.yaml
# srun --job-name=bizspeech_train -p dgx-spa -c 16 --time=10:00:00 --gres=gpu:v100:1 --mem-per-cpu=16GB
