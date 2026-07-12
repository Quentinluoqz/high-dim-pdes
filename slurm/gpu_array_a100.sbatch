#!/usr/bin/env bash
#SBATCH --job-name=hdpde-prec-a100
#SBATCH --partition=P107-A100
#SBATCH --qos=qos_p107-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=3,6,9,12,15,18,21,24,27,30,33,36%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --signal=B:TERM@120

set -euo pipefail
module load python3.12/3.12
module load cuda/13.0
source .venv/bin/activate
mkdir -p logs
slurm/run_array_task.sh "${MANIFEST:-runs/prec.csv}" a100
