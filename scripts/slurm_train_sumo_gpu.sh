#!/bin/bash
#SBATCH --account=def-gkaddoum
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --job-name=tgnn-sumo
#SBATCH --output=runs/logs/tgnn-sumo-%j.out

set -euo pipefail

cd /lustre06/project/6009314/marwani/Event_based_Temporal_Graph_Neural_Network_for_Radio_Resource_Management/github_checkout
source .venv/bin/activate

mkdir -p runs/logs runs/checkpoints runs/plots
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TRACE="${TRACE:-runs/sumo_benchmark_fcd.xml}"
EPOCHS="${EPOCHS:-100}"
TRAIN_STEPS="${TRAIN_STEPS:-9000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
MEMORY_DIM="${MEMORY_DIM:-16}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
MIN_RATE_BPS="${MIN_RATE_BPS:-3000}"
OPTIMIZER="${OPTIMIZER:-adam}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
MAX_INTERFERENCE_NEIGHBORS="${MAX_INTERFERENCE_NEIGHBORS:-4}"
JOB_ID="${SLURM_JOB_ID:-local}"

python -c "import torch; print('cuda_available', torch.cuda.is_available()); print('cuda_device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
nvidia-smi || true

python scripts/train_sumo_unsupervised.py "$TRACE" \
  --device cuda \
  --epochs "$EPOCHS" \
  --train-steps "$TRAIN_STEPS" \
  --eval-steps "$EVAL_STEPS" \
  --memory-dim "$MEMORY_DIM" \
  --hidden-dim "$HIDDEN_DIM" \
  --min-rate-bps "$MIN_RATE_BPS" \
  --optimizer "$OPTIMIZER" \
  --grad-clip-norm "$GRAD_CLIP_NORM" \
  --max-interference-neighbors "$MAX_INTERFERENCE_NEIGHBORS" \
  --metrics-csv "runs/logs/sumo_gpu_metrics_${JOB_ID}.csv" \
  --checkpoint "runs/checkpoints/sumo_tgnn_${JOB_ID}.pt"

python scripts/plot_training_metrics.py \
  "runs/logs/sumo_gpu_metrics_${JOB_ID}.csv" \
  --output-dir "runs/plots/sumo_gpu_${JOB_ID}"
