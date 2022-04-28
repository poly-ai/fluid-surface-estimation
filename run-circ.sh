#!/bin/bash

#SBATCH --job-name=circ
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
# SBATCH --account=eecs545s0
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=12000m

# SBATCH --mem-per-cpu=1000m

source "env/bin/activate"
python3 "src/main_RL_Circ.py"
