# **<center>Repo for the [Masters Thesis](https://github.com/anandcu3/verbose-eureka/blob/main/Large%20Scale%20Speech%20Recognition%20with%20Deep%20Learning.pdf)**

## **<center>Speechbrain with Bizspeech data for distributed training.</center>**

This is the repository for distributed training using SpeechBrain with BizSpeech dataset.

### **Project Files Description**
This Project includes the following files:

#### **Executable Files**

- **train.py** - Starting point for single GPU and Synchronous (DDP) training. Uses the functions from `data_prepare` folder to load the datasets and configuration files in `hparams` folder.

- **train_hogwild.py** - Starting point for Asynchronous (Hogwild!) training. Uses the functions from `data_prepare` folder to load the datasets and configuration files in `hparams` folder. Doesnt use SpeechBrain `Brain` class but uses the modified version of the methods from the class to enable shared memory training. 

- **tokenizer_train.py** - Script to train the tokenizer using SentencePiece Tokenizer.

#### **Triton SBatch Scripts**

- **triton_train_ddp_gpu.sh** - Sbatch script to request multiple GPUs and CPUs to train using DDP. This script monitors the GPU usage and dumps the logs to `slurm-${SLURM_JOB_ID}.out` when running on non-DGX machines. The logging doesnt work on DGX machines due to a dependency issue. Uses the tool `gpustat` for monitoring GPU and is part of the `conda_env.yaml`.

- **triton_train_ddp.sh** - Sbatch script to request multiple GPUs and CPUs to train using DDP. This is for DGX machine without any monitoring.

- **triton_train_gpu.sh** - Sbatch script to request single GPU to train. This script monitors the GPU usage and dumps the logs to `slurm-${SLURM_JOB_ID}.out` when running on non-DGX machines. The logging doesnt work on DGX machines due to a dependency issue. Uses the tool `gpustat` for monitoring GPU and is part of the `conda_env.yaml`.

- **triton_train_hogwild.sh** - Sbatch script to request single GPU to train using Hogwild. Requests more CPUs than general single GPU script.

- **triton_train.sh** - Sbatch script to request single GPU to train. This is for DGX machine without any monitoring.

#### **Other Files**

- **conda_env.yaml** - Conda environment setup with all the packages used. The same environment was used on Triton and on the local workstation.

- **Experiments.xlsx** - Excel file has all the results tabulated to keep track of all the training runs.

### **`data_prepare` Folder Description**

- **bizspeech_prepare.py** - DataSet class to prepare and write out Bizspeech data in TAR format, supports multiprocessing. 

- **librispeech.py** - dataio pipeline for librispeech test splits

- **spgispeech.py** - dataio pipeline for spgispeech val split

### **`hparams` Folder Description**

- **dataset\*.yaml** - Configuration parameters related to the dataset, different files are used for different splits.

- **tokenizer.yaml** - Configuration parameters related to the tokenizer

- **train_ddp.yaml** - Configuration parameters related to ddp experiments. The major change is only related to the Batch Size

- **train.yaml** - Configuration parameters related to single GPU experiments. This can also be used for test set evaluation

- **train_hogwild.yaml** - Configuration parameters related to hogwild experiments. The major change is only related to the Batch Size