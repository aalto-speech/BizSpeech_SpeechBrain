#!/bin/bash
#SBATCH --job-name=bizspeech_train
#SBATCH -p dgx-spa
#SBATCH -c 16
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-cpu=6GB

module load cuda
python tokenizer_train.py tokenizer.yaml
python train.py train.yaml
# srun --job-name=bizspeech_train -p dgx-spa -c 16 --time=10:00:00 --gres=gpu:v100:1 --mem-per-cpu=16GB
