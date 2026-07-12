#!/usr/bin/env bash
#SBATCH --job-name=hdpde-main-rtx5090
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=1-120%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --signal=B:TERM@120

set -euo pipefail
module load python3.12/3.12
module load cuda/13.0
source .venv/bin/activate
mkdir -p logs
slurm/run_array_task.sh "${MANIFEST:-runs/main.csv}" rtx5090
