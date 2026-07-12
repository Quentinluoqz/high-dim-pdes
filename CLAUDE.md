# hdpde project context

This project compares Deep BSDE and tensor-train approaches on five high-dimensional
parabolic PDE benchmarks. Read `AGENTS.md` and `hdpde_project_plan.md` before changes.

The stable atomic interface is:

```text
python scripts/run_one.py --method M --equation E --dim D --seed S --precision P
```

Every run must freeze configuration and provenance. Missing references produce a null
relative error. Numerical failures are terminal metrics, not exceptions to hide. Local
development is CPU-only; GPU claims require returned USTC Slurm evidence.
