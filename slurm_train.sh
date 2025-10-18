#!/bin/bash
#SBATCH --job-name=rl_adder_training
#SBATCH --account=course_eel6938
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_output_%j.out
#SBATCH --error=slurm_error_%j.err

# Load necessary modules (adjust based on your cluster)
module load python/3.10
module load cuda/12.1

# Change to the working directory
export BASE_DIR=CHANGEME
cd $BASE_DIR

# Activate virtual environment
source .venv/bin/activate

# Run the training script
python reinforcement-learning/train.py