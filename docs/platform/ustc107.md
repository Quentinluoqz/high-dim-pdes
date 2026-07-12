# USTC 107 execution notes

The values below are defaults transcribed from the 2026-06-28 training material. They
are not permanent facts. Run `scripts/cloud_preflight.sh` before generating or submitting
jobs and retain its output with the campaign evidence.

## User-controlled handoff

1. Upload the clean cloud bundle to `/public/home/$USER/hdpde` using SCOW or SFTP.
2. Extract it, load `python3.12/3.12` and the required CUDA module, then create a venv.
3. Run the preflight on the login node. It performs queries only; it does not train.
4. Submit CPU/GPU pilots with `sbatch`. Never train on the login node.
5. Return metrics, frozen configs, ledgers, Slurm logs, environment snapshots, and
   screenshots. Do not return JWTs or unrelated credentials.

Default competition partitions are `P107-RTX5090` with QoS `qos_p107-rtx5090`
and `P107-A100` with QoS `qos_p107-a100`. Their live association and TRES limits
must be checked with `scontrol` and `sacctmgr`.

JWTs produced by `scontrol token` are password-equivalent, short-lived secrets. Export
`SLURM_JWT` in the active shell only. `scripts/slurm_api.py` is intentionally read-only.
