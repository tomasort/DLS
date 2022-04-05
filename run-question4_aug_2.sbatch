#!/bin/bash
#SBATCH --job-name=training_m_2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=%x.out
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

module purge
source /scratch/tor213/.env/bin/activate
cd /scratch/tor213/DLS

python question4/Hoffer/main.py --results-dir results/ --model-config "{'depth': 44}"  --duplicates 2 --cutout -b 64 --epochs 100