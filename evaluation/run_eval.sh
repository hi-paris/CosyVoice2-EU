#!/bin/bash
#SBATCH --job-name=11-de
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tim.horstmann@ip-paris.fr


CONDA_ENV_NAME="cosyvoice" # set to cosyvoice-modern for HF-based models, cosyvoice for others


# Activate conda environment
source /home/infres/horstmann-24/.bashrc
conda activate $CONDA_ENV_NAME

# Navigate to working directory
cd /home/infres/horstmann-24/TTS2/evaluation

# Set up path (optional)
if [ -f ./path.sh ]; then
	source ./path.sh
else
	echo "(run_eval.sh) Notice: path.sh not found, continuing without it." >&2
fi



# export HF caches
export HF_HOME="/tsi/hi-paris/tts/Luka/models"
export TRANSFORMERS_CACHE="/tsi/hi-paris/tts/Luka/models"



# Print system info
echo "========================================"
echo "System Information:"
echo "GPU Info:"
nvidia-smi
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h)"
echo "Disk space: $(df -h .)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo "========================================"

# Example: override language/hours from CLI for parallel SBATCH runs
# Accept env overrides LANG and HOURS if provided, else fallback to defaults from config

# default values for lang: fr, hours: 50,100, mix: false
# hours to test: 50, 100, 250, 500, 1000, 1500
LANG_ARG=${LANGUAGE_OVERRIDE:-de}
HOURS_ARG=${HOURS_OVERRIDE:-50,100}
MIX_ARG=${USE_MIXED_MODEL:-true}
ADD_LANG_HINT=${ADD_LANG_HINT:-false}

echo "[sanity] LANG_ARG=${LANG_ARG}  HOURS_ARG=${HOURS_ARG}  MIX_ARG=${MIX_ARG}  ADD_LANG_HINT=${ADD_LANG_HINT}"

ARGS=(--config eval_config_elevenlabs.yaml)
if [ -n "$LANG_ARG" ]; then
	ARGS+=(--language "$LANG_ARG")
fi
if [ -n "$HOURS_ARG" ]; then
	ARGS+=(--hours "$HOURS_ARG")
fi
# Only enable mixed models if explicitly requested (true/1/yes)
case "${MIX_ARG,,}" in
	true|1|yes)
		ARGS+=(--use-mixed-model)
		;;
	*)
		;;
esac

# Add language hint to prompt
if [ "$ADD_LANG_HINT" = true ]; then
	ARGS+=(--add-language-hint)
fi

python run_baseline_evaluation.py "${ARGS[@]}"


echo "========================================"
echo "Job completed successfully"
echo "End time: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "========================================"
