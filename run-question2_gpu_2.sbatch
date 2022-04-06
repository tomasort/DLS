#!/bin/bash
#SBATCH --job-name=gpu_2_training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=%x.out
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:rtx8000:2

module purge
source /scratch/tor213/.env/bin/activate
cd /scratch/tor213/DLS

python training_script_p2.py
