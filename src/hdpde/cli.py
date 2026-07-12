"""Command-line interface for one atomic experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hdpde.runner import run_atomic
from hdpde.utils.config import load_run_config

EQUATION_FILES = {
    "E1": "e1_bsb.yaml",
    "e1": "e1_bsb.yaml",
    "e1_bsb": "e1_bsb.yaml",
    "E2": "e2_lq_control.yaml",
    "e2": "e2_lq_control.yaml",
    "e2_lq_control": "e2_lq_control.yaml",
    "E3": "e3_hjb.yaml",
    "e3": "e3_hjb.yaml",
    "e3_hjb": "e3_hjb.yaml",
    "E4": "e4_allen_cahn.yaml",
    "e4": "e4_allen_cahn.yaml",
    "e4_allen_cahn": "e4_allen_cahn.yaml",
    "E5": "e5_heat.yaml",
    "e5": "e5_heat.yaml",
    "e5_heat": "e5_heat.yaml",
}
METHOD_FILES = {
    "deep_bsde_orig": "deep_bsde_orig.yaml",
    "deep_bsde_shared": "deep_bsde_shared.yaml",
    "tt_cross": "tt_cross.yaml",
    "tt_solver": "tt_solver.yaml",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--method", choices=sorted(METHOD_FILES), required=True)
    result.add_argument("--equation", choices=sorted(EQUATION_FILES), required=True)
    result.add_argument("--dim", type=int, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--precision", choices=["fp32", "tf32", "fp64"], required=True)
    result.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    result.add_argument("--config", type=Path, action="append", default=[])
    result.add_argument("--config-root", type=Path, default=Path("configs"))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    paths: list[Path] = [
        args.config_root / "defaults.yaml",
        args.config_root / "equations" / EQUATION_FILES[args.equation],
        args.config_root / "methods" / METHOD_FILES[args.method],
        *args.config,
    ]
    overrides = [
        f"equation.name={args.equation}",
        f"equation.dim={args.dim}",
        f"method.name={args.method}",
        f"seed={args.seed}",
        f"precision={args.precision}",
        f"device={args.device}",
        f"resume={'true' if args.resume else 'false'}",
        *args.override,
    ]
    config = load_run_config(paths, overrides)
    metrics = run_atomic(config)
    if int(os.getenv("RANK", "0")) == 0:
        print(json.dumps(metrics.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
