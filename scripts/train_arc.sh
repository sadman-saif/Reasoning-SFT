#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=a100_normal_q
#SBATCH --account=tml_2026
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

CONFIG=${1:?Usage: sbatch train_a100.sh CONFIG.yaml}

SCR=/scratch/$USER
PROJECT=$SCR/Project-Reasoning-SFT-LLM
ENV=$SCR/envs/reasoning_sft
WORKDIR=$PROJECT/LLaMA-Factory

module reset
module load Miniforge3/25.11.0-1
module load CUDA/12.6.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

# ARC jupyter-base previously overrode Python.
export PATH="$ENV/bin:$PATH"
hash -r

# Persistent caches on scratch.
export HF_HOME=$SCR/hf_cache
export TORCH_EXTENSIONS_DIR=$SCR/torch_extensions
export TRITON_CACHE_DIR=$SCR/triton_cache
export PIP_CACHE_DIR=$SCR/pip_cache
export TMPDIR=$SCR/tmp
export XDG_CACHE_HOME=$SCR/.cache

export CUDA_HOME=$(dirname $(dirname $(readlink -f $(which nvcc))))
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

mkdir -p \
    "$HF_HOME" \
    "$TORCH_EXTENSIONS_DIR" \
    "$TRITON_CACHE_DIR" \
    "$PIP_CACHE_DIR" \
    "$TMPDIR" \
    "$XDG_CACHE_HOME" \
    "$WORKDIR/logs"

cd "$WORKDIR"

echo "=================================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Hostname:     $(hostname)"
echo "Config:       $CONFIG"
echo "Python:       $(which python)"
python --version
echo "CUDA_HOME:    $CUDA_HOME"
nvcc --version
nvidia-smi
echo "=================================================="

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
PY

# Read output_dir from the YAML.
OUTPUT_REL=$(awk '$1=="output_dir:" {print $2}' "$CONFIG")
OUTPUT="$WORKDIR/$OUTPUT_REL"

echo "Output directory: $OUTPUT"

# Do not accidentally restart a completed experiment.
if [ -f "$OUTPUT/TRAINING_COMPLETE" ]; then
    echo "Training is already marked COMPLETE."
    exit 0
fi

# Find newest checkpoint, if one exists.
LATEST=$(
    find "$OUTPUT" \
        -maxdepth 1 \
        -type d \
        -name 'checkpoint-*' \
        2>/dev/null \
        | sort -V \
        | tail -n 1 \
        || true
)

if [ -n "$LATEST" ]; then
    echo "=================================================="
    echo "RESUMING FROM:"
    echo "$LATEST"
    echo "=================================================="

    FORCE_TORCHRUN=1 \
    MPLBACKEND=Agg \
    llamafactory-cli train "$CONFIG" \
        resume_from_checkpoint="$LATEST"
else
    echo "=================================================="
    echo "Starting new training run"
    echo "=================================================="

    FORCE_TORCHRUN=1 \
    MPLBACKEND=Agg \
    llamafactory-cli train "$CONFIG"
fi

# This is reached only if training exits successfully.
touch "$OUTPUT/TRAINING_COMPLETE"

echo "=================================================="
echo "TRAINING COMPLETED SUCCESSFULLY"
echo "$(date)"
echo "=================================================="
