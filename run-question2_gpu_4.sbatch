#!/bin/bash
#SBATCH --job-name=gpu_4_training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x.out
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:rtx8000:4

module purge
source /scratch/tor213/.env/bin/activate
cd /scratch/tor213/DLS

python training_script_p2.py
