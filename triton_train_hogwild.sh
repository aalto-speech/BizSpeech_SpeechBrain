#!/bin/bash
#SBATCH --job-name=test_hogwild
#SBATCH -p dgx-spa
#SBATCH --account dgx-spa
#SBATCH -c 8
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=16GB

module load cuda
python tokenizer_train.py hparams/tokenizer.yaml
python train_hogwild.py hparams/train_hogwild.yaml
