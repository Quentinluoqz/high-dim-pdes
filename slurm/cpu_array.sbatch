#!/usr/bin/env bash
#SBATCH --job-name=hdpde-rank-cpu
#SBATCH --partition=CPU-6530
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --array=1-45%8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --signal=B:TERM@120

set -euo pipefail
module load python3.12/3.12
source .venv/bin/activate
mkdir -p logs
slurm/run_array_task.sh "${MANIFEST:-runs/rank.csv}" cpu
