# USTC 107 cloud handoff

This file is the operator checklist for the user-controlled cloud phase. Run commands in
order and return the named evidence files. Do not paste or return `SLURM_JWT`.

## 1. Upload and verify

Upload `dist/hdpde-cloud.tar.gz` and its `.sha256` sidecar to your shared home directory.

```bash
cd /public/home/$USER
sha256sum -c hdpde-cloud.tar.gz.sha256
tar -xzf hdpde-cloud.tar.gz
cd hdpde
cat SOURCE_COMMIT
```

## 2. Environment

```bash
module purge
module load python3.12/3.12 cuda/13.0
python3 -m venv /public/home/$USER/venv/hdpde
source /public/home/$USER/venv/hdpde/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" \
  -i https://mirrors.ustc.edu.cn/pypi/web/simple \
  --trusted-host mirrors.ustc.edu.cn
```

If the platform module already provides a CUDA-enabled Torch satisfying the lock, record
its version before changing it. Do not install from the login node while jobs are running.

## 3. Preflight

```bash
bash scripts/cloud_preflight.sh preflight/initial
module list 2> preflight/initial/module_list.txt
```

Review `preflight/initial/cluster_profile.json` against the raw outputs. Correct only values
confirmed by `sinfo`, `scontrol`, and `sacctmgr`; preserve the raw snapshot.

## 4. Local/CPU gates on the login node

Only lightweight tests and manifest generation are permitted here.

```bash
python -m pytest -m "not slow and not gpu"
python scripts/make_manifest.py --config configs/experiments/main.yaml --dry-run
python scripts/make_manifest.py --config configs/experiments/rank.yaml --dry-run
python scripts/make_manifest.py --config configs/experiments/prec.yaml --dry-run
python scripts/make_manifest.py --config configs/experiments/scale.yaml --dry-run
```

Expected counts are 120, 45, 36, and 6.

## 5. Generate reviewed manifests and arrays

```bash
mkdir -p runs slurm/generated logs
python scripts/make_manifest.py --config configs/experiments/main.yaml --out runs/main.csv \
  --preflight preflight/initial/cluster_profile.json
python scripts/make_manifest.py --config configs/experiments/rank.yaml --out runs/rank.csv \
  --preflight preflight/initial/cluster_profile.json
python scripts/make_manifest.py --config configs/experiments/prec.yaml --out runs/prec.csv \
  --preflight preflight/initial/cluster_profile.json
python scripts/make_manifest.py --config configs/experiments/scale.yaml --out runs/scale.csv \
  --preflight preflight/initial/cluster_profile.json

python scripts/make_slurm.py runs/main.csv --preflight preflight/initial/cluster_profile.json
python scripts/make_slurm.py runs/rank.csv --preflight preflight/initial/cluster_profile.json
python scripts/make_slurm.py runs/prec.csv --preflight preflight/initial/cluster_profile.json
python scripts/make_slurm.py runs/scale.csv --preflight preflight/initial/cluster_profile.json
```

Inspect every generated `#SBATCH` header before submission. Submit a one-row pilot manifest
first; do not submit the complete arrays until E1, E3 resume, and TT smoke evidence pass.

## 6. Monitor and account

```bash
squeue -u "$USER"
sacct -u "$USER" -X --format=JobID,JobName,Partition,Elapsed,State,ExitCode
python scripts/collect_sacct.py runs/main.csv --output ledger-main.csv
```

For optional read-only REST monitoring:

```bash
export SLURM_JWT='token value only'
python scripts/slurm_api.py --base-url http://107.ustc.edu.cn:6820 \
  --endpoint slurm/v0.0.41/jobs --output preflight/initial/rest-jobs.json
unset SLURM_JWT
```

## 7. Return bundle contents

Return these paths through SCOW/SFTP:

- `SOURCE_COMMIT`, `SHA256SUMS.json`
- `preflight/`
- `runs/*.csv`, `ledger-*.csv`
- `results/*/config.json`, `metrics.json`, and `train.log`
- Slurm `.out/.err` logs
- Grafana/SCOW screenshots
- only the selected best checkpoint needed by the demo

Do not return all intermediate checkpoints, the virtual environment, package caches, `.env`,
SSH keys, 2FA material, or JWTs.
