#!/bin/bash
#SBATCH --job-name=wds_biz
#SBATCH -c 16
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4GB

SBATCH --partition=gpu-nvlink
SBATCH --account=dgx-spa
module load cuda
python tokenizer_train.py hparams/tokenizer.yaml
python train_transformer.py hparams/transformer.yaml
# srun --job-name=bizspeech_train -p dgx-spa -c 16 --time=10:00:00 --gres=gpu:v100:1 --mem-per-cpu=16GB
