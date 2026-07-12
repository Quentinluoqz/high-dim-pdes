#!/usr/bin/env bash
# Capture evidence before generating any real Slurm job scripts.
set -euo pipefail

OUTPUT_DIR="${1:-preflight/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUTPUT_DIR"

capture() {
  local output="$1"
  shift
  if command -v "$1" >/dev/null 2>&1; then
    "$@" >"$OUTPUT_DIR/$output" 2>&1 || true
  else
    printf '%s is unavailable\n' "$1" >"$OUTPUT_DIR/$output"
  fi
}

capture sinfo.txt sinfo -Nel
capture partitions.txt scontrol show partition -o
capture qos.txt sacctmgr -nP show qos format=Name,MaxJobs,MaxTRESPU,GrpTRES
capture modules.txt module list
capture python.txt python3 --version
capture torch_cuda.txt python3 -c 'import json, torch; print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "devices": torch.cuda.device_count()}))'
capture filesystem.txt df -h .

# This machine-readable profile starts from documented fallbacks. Review and
# edit it against the raw evidence above; make_slurm requires it for real jobs.
python3 - "$OUTPUT_DIR" <<'PY'
import datetime
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
profile = {
    "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "evidence_dir": str(out),
    "clusters": {
        "rtx5090": {"partition": "P107-RTX5090", "qos": "qos_p107-rtx5090", "throttle": 4, "cpus_per_task": 4, "gpus": 1, "time_limit": "12:00:00"},
        "a100": {"partition": "P107-A100", "qos": "qos_p107-a100", "throttle": 2, "cpus_per_task": 4, "gpus": 1, "time_limit": "12:00:00"},
        "cpu": {"partition": "CPU-6530", "qos": "", "throttle": 8, "cpus_per_task": 16, "gpus": 0, "time_limit": "08:00:00"},
    },
}
(out / "cluster_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
PY

printf 'Preflight evidence: %s\nReview %s/cluster_profile.json before use.\n' "$OUTPUT_DIR" "$OUTPUT_DIR"
