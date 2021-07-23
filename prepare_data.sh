#!/bin/bash
#SBATCH --job-name=200_biz
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=6GB
#SBATCH -N 1                 # on one node
#SBATCH --exclusive

module load cuda
python tokenizer_train.py hparams/tokenizer.yaml
#python train.py hparams/train.yaml
# srun --job-name=bizspeech_train -p dgx-spa -c 16 --time=10:00:00 --gres=gpu:v100:1 --mem-per-cpu=16GB
