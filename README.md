# **<center>Speechbrain with Bizspeech data for distributed training.</center>**

## **<center>Repo for the [Masters Thesis](https://github.com/anandcu3/verbose-eureka)</center>**

This is the repository for distributed training using SpeechBrain with BizSpeech dataset.

### **Project Files Description**
This Project includes the following files:

#### **Executable Files**

- **train.py** - Starting point for single GPU and Synchronous (DDP) training. Uses the functions from `data_prepare` folder to load the datasets and configuration files in `hparams` folder.

- **train_hogwild.py** - Starting point for Asynchronous (Hogwild!) training. Uses the functions from `data_prepare` folder to load the datasets and configuration files in `hparams` folder. Doesnt use SpeechBrain `Brain` class but uses the modified version of the methods from the class to enable shared memory training. 

- **tokenizer_train.py** - Script to train the tokenizer using SentencePiece Tokenizer.

#### **Other Files**

- **conda_env.yaml**

- **Experiments.xlsx**

#### **Triton SBatch Scripts**

- **triton_train_ddp_gpu.sh** - Sbatch script to request multiple GPUs and CPUs to train using DDP. This script monitors the GPU usage and dumps the logs to `slurm-${SLURM_JOB_ID}.out` when running on non-DGX machines. The logging doesnt work on DGX machines due to a dependency issue. Uses the tool `gpustat` for monitoring GPU and is part of the `conda_env.yaml`.

- **triton_train_ddp.sh** - Sbatch script to request multiple GPUs and CPUs to train using DDP. This is for DGX machine without any monitoring.

- **triton_train_gpu.sh** - Sbatch script to request single GPU to train. This script monitors the GPU usage and dumps the logs to `slurm-${SLURM_JOB_ID}.out` when running on non-DGX machines. The logging doesnt work on DGX machines due to a dependency issue. Uses the tool `gpustat` for monitoring GPU and is part of the `conda_env.yaml`.

- **triton_train_hogwild.sh** - Sbatch script to request single GPU to train using Hogwild. Requests more CPUs than general single GPU script.

- **triton_train.sh** - Sbatch script to request single GPU to train. This is for DGX machine without any monitoring.

### **`data_prepare` Folder Description**

- **bizspeech_prepare.py** - 

- **librispeech.py** - 

- **spgispeech.py** - 

### **`hparams` Folder Description**

- **dataset\*.yaml** - 

- **tokenizer.yaml** - 

- **train_ddp.yaml** - 

- **train.yaml** - 

- **train_hogwild.yaml** - 