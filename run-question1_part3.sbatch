#!/bin/bash
#SBATCH --job-name=problem1_dropout
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=%x.out
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:v100:1

module purge
source /scratch/tor213/.env/bin/activate
cd /scratch/tor213/DLS

python training_script_p1_dropout.py 
