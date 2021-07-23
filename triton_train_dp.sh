#!/bin/bash
#SBATCH --job-name=wds_biz
#SBATCH -p dgx-spa
#SBATCH --account dgx-spa
#SBATCH -c 4
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=16GB

module load cuda
#python tokenizer_train.py hparams/tokenizer.yaml
python train.py hparams/train_dp.yaml --data_parallel_backend
