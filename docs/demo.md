# Five-minute demonstration

1. Show the clean source commit and frozen E3 d=100 pilot config.
2. Submit the reviewed pilot with `sbatch`; show the returned JobID.
3. Use `squeue` and the read-only REST query to show queued/running state.
4. Show the Grafana GPU-utilization panel and the retained preflight evidence.
5. Open a completed run's `metrics.json`, config hash, and `sacct` ledger row.
6. Load the selected best checkpoint inside an allocated GPU job and time inference.
7. Contrast the measured representation with the order of magnitude of a direct
   tensor grid in 100 dimensions.

If the live queue delays the new pilot, use the already completed evidence bundle and
state that the submitted job is pending. Never fabricate a running state.
