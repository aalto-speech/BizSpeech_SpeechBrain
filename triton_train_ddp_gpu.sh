#!/bin/bash
#SBATCH --job-name=biz_ddp
#SBATCH -p gpu-nvlink
#SBATCH -c 4
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=80GB

## Set monitoring interval ##
MONITORING_INTERVAL=120

## Set gpu stats file ##
MONITORING_FILE=slurm-${SLURM_JOB_ID}.out


## Remove monitoring file ##
#[ ! -z ${MONITORING_FILE} ] && echo 'Removing old monitoring file '${MONITORING_FILE} && rm -f ${MONITORING_FILE}

## Start monitoring ##
MONITORING_PID=-1
gpustat --no-color --no-header -c -i ${MONITORING_INTERVAL} >> ${MONITORING_FILE} &
MONITORING_PID=$!
echo 'Monitoring PID:         '$MONITORING_PID
echo 'Monitoring output file: '$MONITORING_FILE

## Define monitoring stopping function and a trap for exit ##
stop_monitoring() {
  echo 'Stopping GPU monitoring'
  if [[ "${MONITORING_PID}" -gt 0 ]]; then
    echo 'Killing monitoring with PID '$MONITORING_PID
    kill -9 $MONITORING_PID
  fi
}
trap stop_monitoring EXIT


module load cuda
python tokenizer_train.py hparams/tokenizer.yaml
python -m torch.distributed.launch --nproc_per_node=4 train.py hparams/train_ddp.yaml --distributed_launch --distributed_backend='nccl'