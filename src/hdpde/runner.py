"""Atomic experiment assembly and execution."""

from __future__ import annotations

import os
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from hdpde.equations import make_equation
from hdpde.solvers.deep_bsde import DeepBSDESolver
from hdpde.types import RunMetrics, RunStatus
from hdpde.utils.config import RunConfig
from hdpde.utils.io import atomic_write_json, make_run_id, prepare_run_dir, source_commit

EQUATION_ALIASES = {
    "e1": "E1",
    "e1_bsb": "E1",
    "black_scholes_barenblatt": "E1",
    "e2": "E2",
    "e2_lq_control": "E2",
    "lq_control": "E2",
    "e3": "E3",
    "e3_hjb": "E3",
    "hjblq": "E3",
    "e4": "E4",
    "e4_allen_cahn": "E4",
    "allen_cahn": "E4",
    "e5": "E5",
    "e5_heat": "E5",
    "heat_equation": "E5",
}


def finalize_provenance(config: RunConfig, repo_root: str | Path = ".") -> RunConfig:
    """Bind an otherwise resolved config to an immutable source witness."""

    commit = config.source_commit
    if commit in {"", "unknown", "TBD"}:
        commit = source_commit(repo_root)
    return replace(config, source_commit=commit)


def _assemble_equation(config: RunConfig) -> tuple[Any, RunConfig]:
    alias = EQUATION_ALIASES.get(config.equation.name.lower())
    if alias is None:
        raise ValueError(f"unknown equation alias: {config.equation.name}")
    params = dict(config.equation.params)
    reference_samples = params.pop("reference_samples", None)
    reference_batch = params.pop("reference_batch_size", None)
    equation = make_equation(alias, config.equation.dim, **params)
    if reference_samples is not None or reference_batch is not None:
        method_params = dict(config.method.params)
        options = dict(method_params.get("reference_options", {}))
        if reference_samples is not None:
            options["sample_count"] = int(reference_samples)
        if reference_batch is not None:
            options["batch_size"] = int(reference_batch)
        options.setdefault("seed", config.seed)
        method_params["reference_options"] = options
        config = replace(config, method=replace(config.method, params=method_params))
    return equation, config


def _run_tt(config: RunConfig, equation: Any) -> RunMetrics:
    from hdpde.solvers.tt import TTConfig, run_tier1

    params = dict(config.method.params)
    aliases = {
        "target_error": "target_eps",
        "validation_size": "validation_samples",
        "test_size": "grid_test_samples",
        "continuous_test_size": "continuous_test_samples",
        "max_evaluations": "max_samples",
        "rank_growth_min": "rank_increment_min",
        "rank_growth_max": "rank_increment_max",
    }
    for alias, canonical in aliases.items():
        if alias in params:
            params[canonical] = params.pop(alias)
    result = run_tier1(
        TTConfig(
            dim=config.equation.dim,
            equation_id=EQUATION_ALIASES[config.equation.name.lower()],
            seed=config.seed,
            **params,
        ),
        equation,
    )
    reference_result = equation.reference_solution()
    reference = reference_result.to_dict()
    reference_value = reference_result.value
    abs_error = None
    rel_error = None
    if result.estimate is not None and reference_value is not None:
        abs_error = abs(result.estimate - reference_value)
        if reference_value != 0:
            rel_error = abs_error / abs(reference_value)
    return RunMetrics(
        run_id=make_run_id(config),
        method=config.method.name,
        equation=config.equation.name,
        dim=config.equation.dim,
        seed=config.seed,
        precision=config.precision,
        status=RunStatus(result.status),
        config_hash=config.config_hash,
        source_commit=config.source_commit,
        estimate=result.estimate,
        abs_error=abs_error,
        rel_error=rel_error,
        reference=reference,
        wall_clock_sec=result.total_seconds,
        train_sec=result.build_seconds,
        n_params=result.n_params_tt,
        tt_ranks=result.ranks,
        extra={
            "target_error": result.target_eps,
            "grid_rel_l2": result.grid_rel_l2,
            "continuous_rel_l2": result.continuous_rel_l2,
            "max_rank": result.max_rank,
            "full_grid_elements": result.full_grid_elements,
            "compression_ratio": result.compression_ratio,
            "function_evaluations": result.function_evaluations,
            "sweeps": result.sweeps,
            "stop_reason": result.stop_reason,
        },
    )


def run_atomic(config: RunConfig, repo_root: str | Path = ".") -> RunMetrics:
    """Run exactly one method/equation point and always retain failures as data."""

    config = finalize_provenance(config, repo_root)
    equation, config = _assemble_equation(config)
    rank = int(os.getenv("RANK", "0"))
    run_dir = prepare_run_dir(config)
    if rank == 0:
        atomic_write_json(run_dir / "config.json", config.to_dict())
    try:
        if config.method.name in {"deep_bsde_orig", "deep_bsde_shared", "m1a", "m1b"}:
            metrics = DeepBSDESolver(equation).run(config)
        elif config.method.name == "tt_cross":
            metrics = _run_tt(config, equation)
        elif config.method.name == "tt_solver":
            raise RuntimeError(
                "Tier-2 is disabled until external/PDE-backward-solver "
                "passes license and build review"
            )
        else:
            raise ValueError(f"unknown method: {config.method.name}")
    except Exception as exc:
        metrics = RunMetrics(
            run_id=make_run_id(config),
            method=config.method.name,
            equation=config.equation.name,
            dim=config.equation.dim,
            seed=config.seed,
            precision=config.precision,
            status=RunStatus.FAILED,
            config_hash=config.config_hash,
            source_commit=config.source_commit,
            message=f"{type(exc).__name__}: {exc}",
            artifacts={"traceback": str(run_dir / "traceback.txt")},
        )
        if rank == 0:
            (run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    if rank == 0:
        atomic_write_json(run_dir / "metrics.json", metrics.to_dict())
        (run_dir / "train.log").write_text(
            f"status={metrics.status.value}\nmessage={metrics.message or ''}\n",
            encoding="utf-8",
        )
    return metrics
