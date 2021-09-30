#!/bin/bash
#SBATCH --job-name=biz_ddp
#SBATCH -p dgx-spa
#SBATCH --account dgx-spa
#SBATCH -c 8
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=16GB

module load cuda
python tokenizer_train.py hparams/tokenizer.yaml
python -m torch.distributed.launch --nproc_per_node=4 train.py hparams/train_ddp.yaml --distributed_launch --distributed_backend='nccl'