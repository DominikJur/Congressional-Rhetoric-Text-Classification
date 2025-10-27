#!/bin/bash -l
#SBATCH -N 1               # Number of nodes. ALWAYS set to 1
#SBATCH -n 1               # Number of tasks. ALWAYS set to 1
#SBATCH -c 32             # Number of CPU cores. Can go as high as 128. Each additional CPU core adds around 1.9GB of RAM so to get more memory, add more CPU cores.
#SBATCH -t 1:0:0           # Number of hours to run (H:M:S). Change as needed.
#SBATCH -A cis220051-gpu   # The TDM account to charge for this. Don't change.
#SBATCH -p gpu             # Partition to use -> gpu | gpu-debug if less than 15 minutes
#SBATCH --gpus-per-node=1  # Must be just one GPU.

# These three lines use the TDM python
module use /anvil/projects/tdm/opt/core
module load tdm
module load python/seminar r/seminar

cd ../

# This is where you specify the program you want to run!
python transcription_runner.py
