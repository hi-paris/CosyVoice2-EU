#!/bin/bash
#SBATCH --job-name=emonet_de
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tim.horstmann@ip-paris.fr

# ---- env setup----
# source /home/infres/horstmann-24/.bashrc
# conda activate cosyvoice

# Hugging Face cache (you asked to use this path)
export HF_HOME=/tsi/hi-paris/tts/Luka/models

# Prevent BLAS from oversubscribing threads when multiprocessing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Use the allocated CPUs for Python workers if you like
export PYTHONWARNINGS="ignore"

# Run your script via srun so Slurm tracks resources correctly
srun python download_emonet_german.py