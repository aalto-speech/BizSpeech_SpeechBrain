#!/bin/bash
#SBATCH --job-name=spgi
#SBATCH -p gpu-nvlink,dgx-spa,dgx-common
#SBATCH --account dgx-spa
#SBATCH -c 2
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=12GB

module load cuda
python tokenizer_train.py hparams/tokenizer.yaml
python train.py hparams/train.yaml
