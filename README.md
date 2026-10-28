# Reasoning SFT with Random vs. Diversity-Aware Data Selection

This repository contains the data-selection, supervised fine-tuning (SFT), and evaluation workflow used to study whether selected 15,000 reasoning examples can provide better training data than a randomly selected 15,000-example baseline for `Qwen/Qwen2.5-3B-Instruct`.

The project uses the AceReason-1.1 100K reasoning dataset and compares two training subsets:

1. Random-15K — created with random sampling.
2. Advanced-15K — diversity-aware semantic clustering combined with a cluster solution-length criterion.



Repository Structure

Reasoning-SFT/
│
├── configs/
│   ├── random15k_full_sft.yaml
│   └── advanced15k_full_sft.yaml
│
├── scripts/
│   ├── create_random_15k.py
│   ├── create_advanced_15k.py
│   ├── train_arc.sh
│   ├── train_advanced_arc.sh
│   └── eval.py
│
├── metadata/
│   └── advanced_selection_metadata.csv
│
├── results/
│   └── evaluation summaries / CSV files
│
├── .gitignore
└── README.md
```

File descriptions

| File | Purpose |
|---|---|
| `configs/random15k_full_sft.yaml` | Training configuration used for the Random-15K ARC full-SFT experiment |
| `configs/advanced15k_full_sft.yaml` | Training configuration used for the time-constrained Advanced-15K full-SFT experiment |
| `scripts/create_random_15k.py` | Generates the reproducible Random-15K dataset |
| `scripts/create_advanced_15k.py` | Generates the diversity-aware Advanced-15K dataset |
| `scripts/train_arc.sh` | Generic SLURM launcher for LLaMA-Factory training |
| `scripts/train_advanced_arc.sh` | SLURM launcher used for the Advanced experiment |
| `scripts/eval.py` | vLLM benchmark evaluator |
| `metadata/advanced_selection_metadata.csv` | Metadata from Advanced subset selection |
| `results/` | Compact benchmark summaries and processed results |
| `.gitignore` | Prevents large checkpoints, weights, caches, environments, and datasets from entering Git |

Dataset Construction

Random-15K

The Random baseline is generated using:

seed = 42
sample size = 15,000

The implementation is provided in:
scripts/create_random_15k.py


Advanced-15K Selection Strategy

The Advanced subset was designed to improve both semantic coverage and reasoning-solution complexity coverage.

The pipeline is:

AceReason 100K
      │
      ▼
Remove empty instruction/output entries
      │
      ▼
Normalize prompts
      │
      ▼
Remove duplicate normalized prompts
      │
      ▼
84,094 usable unique examples
      │
      ▼
Encode instructions with all-MiniLM-L6-v2
      │
      ▼
384-dimensional normalized semantic embeddings
      │
      ▼
MiniBatchKMeans clustering
300 clusters
      │
      ▼
Rank solution length within each cluster
      │
      ▼
Prefer examples near the 70th percentile
      │
      ▼
Approximately 50 examples per cluster
      │
      ▼
Global score-based fill
      │
      ▼
15,000 Advanced examples


The implementation is available in:
scripts/create_advanced_15k.py


Detailed selection metadata is stored in:
metadata/advanced_selection_metadata.csv


Environment Setup

Clone this repository

```bash
git clone https://github.com/sadman-saif/Reasoning-SFT.git
cd Reasoning-SFT
```

## 4.2 Create a Python environment

A Python 3.10/3.11 environment is recommended.

Example:

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip


For dataset selection:


pip install numpy pandas scikit-learn sentence-transformers huggingface_hub


LLaMA-Factory Setup

Training was performed using LLaMA-Factory.

Clone LLaMA-Factory:



The training system used:
GPU: NVIDIA A100
CUDA module: 12.6.0
Miniforge module: 25.11.0-1
Training framework: LLaMA-Factory
Precision: BF16
FlashAttention2: enabled
DeepSpeed ZeRO-3 CPU offload: enabled
Gradient checkpointing: enabled


Obtain the Source Dataset

Download:


redsgnaoh/acereason11_100k
```

Required file:


data/acereason11_100k.json

Place the source JSON at the path expected by the selection scripts.

The generated datasets should ultimately be available to LLaMA-Factory as:

LLaMA-Factory/data/acereason_random_15k.json
LLaMA-Factory/data/acereason_advanced_15k.json

Recreate the Training Subsets

From the project root, run:

python scripts/create_random_15k.py

and:

python scripts/create_advanced_15k.py


Expected outputs:

acereason_random_15k.json
acereason_advanced_15k.json


The Advanced pipeline will additionally produce metadata describing the clustering and selection process.

Register the Datasets with LLaMA-Factory

Place the generated JSON files under:

LLaMA-Factory/data/

Each training example contains an instruction/problem and an output/solution.

Confirm the dataset names used by the configurations with:
"dataset:" configs/*.yaml


Then register the corresponding JSON files in LLaMA-Factory's:

data/dataset_info.json


Random-15K Full SFT

The Random configuration is:


configs/random15k_full_sft.yaml


Important training settings include:

Base model: Qwen/Qwen2.5-3B-Instruct
Fine-tuning type: full SFT
Training examples: 15,000
Maximum sequence length: 16,384
Per-device batch size: 1
Gradient accumulation: 8
Effective batch size: 8
Learning rate: 5e-6
Scheduler: cosine
Planned epochs: 3
Precision: BF16
Seed: 42
DeepSpeed: ZeRO-3 with CPU offload
FlashAttention2: enabled
Gradient checkpointing: enabled
```

On an ARC/SLURM system:


sbatch scripts/train_arc.sh configs/random15k_full_sft.yaml

---

# 10. Advanced-15K Full SFT

The Advanced configuration is:
configs/advanced15k_full_sft.yaml

The ARC Advanced experiment was designed under a stricter remaining compute/time budget.

Its configuration uses approximately:

Training examples: 15,000
Fine-tuning type: full SFT
Maximum sequence length: 10,000
Epochs: 2
Per-device batch size: 1
Gradient accumulation: 8
Learning rate: 5e-6
Scheduler: cosine
Precision: BF16
Seed: 42
DeepSpeed ZeRO-3 CPU offload
FlashAttention2
Gradient checkpointing


Launch using:
sbatch scripts/train_advanced_arc.sh

Evaluation

Evaluation is implemented in:
scripts/eval.py


The evaluator uses vLLM.

The main ARC evaluation environment included:

Python: 3.11
torch: 2.4.0+cu121
vllm: 0.6.1
transformers: 4.46.0
datasets: 2.21.0
numpy: 1.26.4
sympy: 1.12


The official-style generation settings used were:

temperature = 0.6
top_p = 0.95
n_sampling = 8
max_tokens = 32768
seed = 0
k = 1
prompt format = qwen-instruct


The local evaluator was modified to use:

swap_space=60

and:

gpu_memory_utilization = 0.96

From the evaluation the logs of the trained with random15K dataset model's output. Those files have the answers that the model produces during evaluation.
