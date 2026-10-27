#!/bin/bash



#SBATCH --job-name=qwen_adv15k_2ep

#SBATCH --account=tml_2026

#SBATCH --partition=a100_normal_q



#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8



#SBATCH --gres=gpu:a100:1

#SBATCH --mem=128G



#SBATCH --time=23:50:00



#SBATCH --output=advanced15k_2ep_%j.out

#SBATCH --error=advanced15k_2ep_%j.err





echo "=========================================="

echo "Advanced-15K Full SFT"

echo "2 epochs / cutoff 10,000"

echo "Started: $(date)"

echo "Hostname: $(hostname)"

echo "SLURM Job ID: $SLURM_JOB_ID"

echo "=========================================="



module reset

module load Miniforge3/25.11.0-1

module load CUDA/12.6.0



source activate /scratch/sadmansaif/envs/reasoning_sft



export PATH="$CONDA_PREFIX/bin:$PATH"

hash -r



export PIP_CACHE_DIR=/scratch/sadmansaif/pip_cache

export TMPDIR=/scratch/sadmansaif/tmp

export XDG_CACHE_HOME=/scratch/sadmansaif/.cache



mkdir -p "$PIP_CACHE_DIR"

mkdir -p "$TMPDIR"

mkdir -p "$XDG_CACHE_HOME"



export FORCE_TORCHRUN=1



cd /scratch/sadmansaif/Project-Reasoning-SFT-LLM/LLaMA-Factory



echo

echo "Python:"

which python

python --version



echo

echo "GPU:"

nvidia-smi



echo

echo "Training configuration:"

grep -E \

"dataset:|cutoff_len:|num_train_epochs:|learning_rate:|gradient_accumulation_steps:|output_dir:|finetuning_type:" \

configs/advanced15k_full_sft.yaml



echo

echo "Starting training..."

echo



llamafactory-cli train \

    configs/advanced15k_full_sft.yaml



STATUS=$?



echo

echo "=========================================="

echo "Training command finished"

echo "Time: $(date)"

echo "Exit status: $STATUS"

echo "=========================================="



exit $STATUS
