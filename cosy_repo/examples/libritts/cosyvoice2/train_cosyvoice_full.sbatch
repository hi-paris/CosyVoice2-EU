#!/bin/bash
#SBATCH --job-name=l-3k-mi
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=H100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tim.horstmann@ip-paris.fr

# naming convention for job name I try to follow: <model_abbreviation>-<training_hours>-<backbone>
# Usage: sbatch train_cosyvoice_full.sbatch \
#   [STAGE] [STOP_STAGE] [TRAIN_HOURS] [NUM_WORKERS] [MODELS] [TRAIN_ENGINE] [CUDA_DEVICES] [GPU_PARTITION] [BACKBONE] [MAX_FRAMES] [GRAD_CHECKPOINT] [LORA_ENABLE]
# Example: sbatch train_cosyvoice_full.sbatch 5 6 3000 2 "llm" torch_ddp 0,1 A40 blanken 5000 true false

# Parse command line arguments
STAGE=${1:-5}
STOP_STAGE=${2:-6}
TRAIN_HOURS=${3:-3000}  # Default to 100h training data - e.g. mix: 100, 200, 500, 1000, 2000, 3000, de/fr: 50, 100, 250, 500, 1000, 1500
NUM_WORKERS=${4:-2}
MODELS=${5:-"llm"}
TRAIN_ENGINE=${6:-"torch_ddp"}
CUDA_DEVICES=${7:-"0"}  # Default to use 2 GPUs as specified in SBATCH
GPU_PARTITION=${8:-$SLURM_JOB_PARTITION}  # Auto-detect from SLURM or use provided value
BACKBONE=${9:-"hf:mistralai/Mistral-7B-v0.3"} # "blanken" | "hf:<repo>" | "local:/abs/or/relative/path"
                        #  "hf:Qwen/Qwen2.5-0.5B", "hf:Qwen/Qwen3-0.6B", "hf:utter-project/EuroLLM-1.7B-Instruct"
                        # "hf:mistralai/Mistral-7B-v0.3"
MAX_FRAMES=${10:-8000} # Maximum frames in a batch; around 10k in LoRa-setting, around 3000 to 5000 in normal setting
GRAD_CHECKPOINT=${11:-true}
LORA_ENABLE=${12:-true} # currently only supported for HF-based models, i.e. not for blanken

LANGUAGE=${13:-"mix"} # Language code (e.g., "FR", "DE", or "mix" for both)

# Resume controls (HF backbones only)
CONTINUE=${14:-true}           # true|false
WANDB_RUN_ID=${15:-"w4cgp3zp"}         # required when CONTINUE=true
RESUME_CHECKPOINT=${16:-"/tsi/hi-paris/tts/Luka/exp-3000h-hf-mistralai-Mistral-7B-v0.3-lora_true_mix/cosyvoice2/llm/torch_ddp/epoch_1_whole.pt"}    # optional explicit checkpoint path

CONDA_ENV_NAME="cosyvoice-modern" # set to cosyvoice-modern for HF-based models, cosyvoice for others


# Export feature flags so run.sh can pick them up
export COSY_GRAD_CHECKPOINT="$GRAD_CHECKPOINT"
export COSY_LORA_ENABLE="$LORA_ENABLE"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1


# STAGES OVERVIEW:
# -1: Data Download
#  0: Data Preparation
#  1: Extract Speaker Embedding
#  2: Extract Speech Token
#  3: Prepare Parquet Data
#  5: Train Model
#  6: Average Model
#  7: Export Models
#  8: Upload Final Model to Hugging Face

# Print job info
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Stage: $STAGE"
echo "Stop stage: $STOP_STAGE"
echo "Number of workers: $NUM_WORKERS"
echo "Models to train: $MODELS"
echo "Training engine: $TRAIN_ENGINE"
echo "CUDA devices: $CUDA_DEVICES"
echo "Training hours: $TRAIN_HOURS"
echo "GPU partition: $GPU_PARTITION"
echo "Gradient checkpointing: $GRAD_CHECKPOINT"
echo "LoRA enabled: $LORA_ENABLE"
echo "Backbone: $BACKBONE"
echo "Language: $LANGUAGE"
echo "Max frames in batch: $MAX_FRAMES"
echo "Continue (resume): $CONTINUE"
if [ "$CONTINUE" = true ]; then
    echo "W&B run id: $WANDB_RUN_ID"
    echo "Resume checkpoint: ${RESUME_CHECKPOINT:-auto}"
fi
echo "========================================"

# Activate conda environment
source /home/infres/horstmann-24/.bashrc
conda activate $CONDA_ENV_NAME

# Navigate to working directory
cd /home/infres/horstmann-24/TTS2/cosy_repo/examples/libritts/cosyvoice2

# Set up path
source ./path.sh

# Set up W&B (install if needed)
echo "Setting up Weights & Biases..."

# export WANDB_DIR
export WANDB_DIR="/tsi/hi-paris/tts/Luka/wandb"

# export HF caches
export HF_HOME="/tsi/hi-paris/tts/Luka/models"
export TRANSFORMERS_CACHE="/tsi/hi-paris/tts/Luka/models"

python3 -c "import wandb; print(f'W&B version: {wandb.__version__}')" 2>/dev/null || {
    echo "W&B not found. Installing..."
    pip install wandb --quiet
}

# Check W&B login (continue even if not logged in)
python3 -c "
import wandb
try:
    api = wandb.Api()
    user = api.viewer
    print(f'✓ Logged in to W&B as: {user.username if hasattr(user, \"username\") else \"unknown\"}')
except Exception as e:
    print('⚠ Not logged in to W&B. Run \"wandb login\" for full functionality.')
    print('Training will continue without W&B logging.')
" 2>/dev/null || echo "⚠ W&B setup incomplete but continuing..."

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

# Set CUDA_VISIBLE_DEVICES for the current environment
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

# Set W&B project name for training
export WANDB_PROJECT="CosyVoice2-EU"


# Choose a free MASTER_PORT (prefer stable per job; fall back to random)
if [ -z "${MASTER_PORT:-}" ]; then
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    # derive from job id, keep in safe ephemeral range
    export MASTER_PORT=$(( (SLURM_JOB_ID % 20000) + 20000 ))
  else
    # random free port probe
    export MASTER_PORT=$(python - <<'PY'
import socket, random
for _ in range(50):
    port = random.randint(20000, 45000)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            print(port)
            break
        except OSError:
            pass
PY
)
  fi
fi
echo "Using MASTER_PORT=$MASTER_PORT"

# (Optional) also pin master addr to IPv4 loopback to dodge IPv6 surprises
export MASTER_ADDR=127.0.0.1


# Set up error handling
set -e
trap 'echo "Error occurred at line $LINENO. Exit code: $?" >&2' ERR

# Monitor system resources in background
(
    while true; do
        echo "$(date): GPU usage:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
        echo "$(date): Memory usage: $(free -h | grep Mem | awk '{print $3"/"$2}')"
        sleep 300  # Log every 5 minutes
    done
) &
MONITOR_PID=$!

# Run the pipeline
echo "Starting CosyVoice2 pipeline at $(date)"
echo "Running stages $STAGE to $STOP_STAGE"
echo "Training models: $MODELS"
echo "Using training engine: $TRAIN_ENGINE"
echo "Export directory: $EXPORT_DIR"
echo "CUDA devices: $CUDA_DEVICES"
echo "Training hours: $TRAIN_HOURS"
echo "GPU partition: $GPU_PARTITION"
echo "Max frames in batch: $MAX_FRAMES"

./run.sh \
    --stage "$STAGE" \
    --stop_stage "$STOP_STAGE" \
    --models "$MODELS" \
    --num_workers "$NUM_WORKERS" \
    --train_engine "$TRAIN_ENGINE" \
    --train_hours "$TRAIN_HOURS" \
    --gpu_partition "$GPU_PARTITION" \
    --backbone "$BACKBONE" \
    --max_frames_in_batch "$MAX_FRAMES" \
    --grad_checkpoint "$GRAD_CHECKPOINT" \
    --lora_enable "$LORA_ENABLE" \
    --language "$LANGUAGE" \
    --continue "$CONTINUE" \
    --wandb_run_id "$WANDB_RUN_ID" \
    $( [ -n "$RESUME_CHECKPOINT" ] && echo --resume_from_checkpoint "$RESUME_CHECKPOINT" )

echo "Pipeline completed successfully at $(date)"

# Kill the monitoring process
kill $MONITOR_PID 2>/dev/null || true

# Upload SLURM output files to W&B if they exist
echo "Attempting to upload SLURM output files to W&B..."
python3 -c "
import os
import sys
import glob

# Add current directory to path for any local imports
sys.path.insert(0, os.getcwd())

try:
    import wandb
    
    # Look for SLURM output files in current directory
    slurm_out = 'slurm-${SLURM_JOB_ID}.out'
    slurm_err = 'slurm-${SLURM_JOB_ID}.err'
    
    # Try to find the most recent wandb run
    wandb_dir = 'wandb'
    if os.path.exists(wandb_dir):
        runs = [d for d in os.listdir(wandb_dir) if d.startswith('run-')]
        if runs:
            # Get the most recent run
            latest_run = max(runs, key=lambda x: os.path.getctime(os.path.join(wandb_dir, x)))
            run_id = latest_run.split('-')[1] if len(latest_run.split('-')) > 1 else None
            
            if run_id:
                print(f'Found W&B run ID: {run_id}')
                # Initialize wandb with the existing run
                run = wandb.init(project='CosyVoice2-EU', id=run_id, resume='must')
                
                files_uploaded = 0
                # Upload SLURM files if they exist
                if os.path.exists(slurm_out):
                    wandb.save(slurm_out)
                    print(f'✓ Uploaded {slurm_out}')
                    files_uploaded += 1
                if os.path.exists(slurm_err):
                    wandb.save(slurm_err)
                    print(f'✓ Uploaded {slurm_err}')
                    files_uploaded += 1
                
                # Log final job info
                wandb.log({
                    'slurm/job_completed': 1,
                    'slurm/total_runtime_seconds': ${SECONDS},
                    'slurm/job_id': '${SLURM_JOB_ID}',
                    'slurm/node': '${SLURM_NODELIST}',
                    'slurm/files_uploaded': files_uploaded
                })
                
                wandb.finish()
                print(f'✓ W&B logging completed ({files_uploaded} files uploaded)')
            else:
                print('⚠ Could not extract run ID from wandb directory')
        else:
            print('⚠ No wandb runs found in wandb directory')
    else:
        print('⚠ No wandb directory found - training may not have used W&B')

except ImportError:
    print('⚠ W&B not available - install with: pip install wandb')
except Exception as e:
    print(f'⚠ W&B upload failed: {e}')
    print('This is non-critical - training completed successfully')
"

# Optional: Compress and backup results
# tar -czf results_${SLURM_JOB_ID}.tar.gz /tsi/hi-paris/tts/Luka/exp/cosyvoice2/

echo "========================================"
echo "Job completed successfully"
echo "End time: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "========================================"
