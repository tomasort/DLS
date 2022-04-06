#!/bin/bash
#SBATCH --job-name=gpu-test_four
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=%x.out
#SBATCH --mem=8GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=tor213@nyu.edu
#SBATCH --gres=gpu:4

module purge
source /scratch/tor213/.env/bin/activate
cd /scratch/tor213/DLS

python test_number_of_gpus