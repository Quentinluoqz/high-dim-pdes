"""Tier-1 TT-cross approximation of an equation's exact solution.

The optional :mod:`teneva` dependency is imported only when the solver runs, so
the rest of the package (and GPU jobs) remain usable in minimal environments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from math import prod
from time import perf_counter
from typing import Any, Protocol

import numpy as np


class TenevaUnavailableError(RuntimeError):
    """Raised when a TT run is requested without the optional dependency."""


class ExactEquation(Protocol):
    """Small interface needed by Tier 1 (implemented by E1, E2 and E5)."""

    def exact_solution(self, t: float, x: Any) -> Any: ...


_DEFAULT_BOXES: dict[str, tuple[float, float]] = {
    "E1": (0.5, 1.5),
    "E2": (-2.0, 2.0),
    "E5": (-2.0, 2.0),
}


@dataclass(frozen=True, slots=True)
class TTConfig:
    """Configuration for the reproducible Tier-1 baseline."""

    dim: int
    target_eps: float
    equation_id: str | None = None
    lower: float | tuple[float, ...] | None = None
    upper: float | tuple[float, ...] | None = None
    grid_size: int = 32
    initial_rank: int = 1
    max_sweeps: int = 20
    max_samples: int = 200_000
    validation_samples: int = 2_000
    grid_test_samples: int = 10_000
    continuous_test_samples: int = 2_000
    rank_increment_min: int = 1
    rank_increment_max: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError("dim must be positive")
        if not 0.0 < self.target_eps < 1.0:
            raise ValueError("target_eps must be in (0, 1)")
        for field in (
            "grid_size",
            "initial_rank",
            "max_sweeps",
            "max_samples",
            "validation_samples",
            "grid_test_samples",
            "continuous_test_samples",
            "rank_increment_min",
            "rank_increment_max",
        ):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive")
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least two")
        if self.rank_increment_min > self.rank_increment_max:
            raise ValueError("rank_increment_min cannot exceed rank_increment_max")


@dataclass(slots=True)
class TTResult:
    """Serializable scientific and resource metrics for one TT-cross run."""

    status: str
    estimate: float | None
    target_eps: float
    grid_rel_l2: float
    continuous_rel_l2: float
    ranks: list[int]
    max_rank: int
    n_params_tt: int
    full_grid_elements: int
    compression_ratio: float
    function_evaluations: int
    sweeps: int
    stop_reason: str
    build_seconds: float
    total_seconds: float
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_teneva() -> Any:
    try:
        return import_module("teneva")
    except ImportError as exc:
        raise TenevaUnavailableError(
            "TT Tier 1 requires teneva==0.14.11; install the project TT dependencies "
            "or run this experiment in the CPU environment."
        ) from exc


def _equation_id(equation: Any, configured: str | None) -> str:
    if configured:
        return configured.upper()
    for attr in ("equation_id", "id", "name"):
        value = getattr(equation, attr, None)
        if isinstance(value, str) and value.upper() in _DEFAULT_BOXES:
            return value.upper()
    names = {
        "BlackScholesBarenblatt": "E1",
        "LQControl": "E2",
        "HeatEquation": "E5",
    }
    try:
        return names[type(equation).__name__]
    except KeyError as exc:
        raise ValueError(
            "Tier 1 needs equation_id or an E1/E2/E5 equation with exact_solution"
        ) from exc


def _bounds(config: TTConfig, equation_id: str) -> tuple[np.ndarray, np.ndarray]:
    default_lower, default_upper = _DEFAULT_BOXES[equation_id]

    def expand(value: float | tuple[float, ...] | None, default: float) -> np.ndarray:
        if value is None:
            return np.full(config.dim, default, dtype=np.float64)
        result = np.asarray(value, dtype=np.float64)
        if result.ndim == 0:
            return np.full(config.dim, float(result), dtype=np.float64)
        if result.shape != (config.dim,):
            raise ValueError(f"bound must be scalar or have shape ({config.dim},)")
        return result

    lower = expand(config.lower, default_lower)
    upper = expand(config.upper, default_upper)
    if np.any(lower >= upper):
        raise ValueError("each lower bound must be strictly smaller than upper")
    return lower, upper


def _exact_values(equation: ExactEquation, x: np.ndarray) -> np.ndarray:
    """Evaluate the torch-facing equation interface and return flat float64 data."""
    try:
        torch = import_module("torch")
    except ImportError as exc:  # pragma: no cover - project always installs torch
        raise RuntimeError("Equation evaluation requires torch") from exc
    x_tensor = torch.as_tensor(x, dtype=torch.float64, device="cpu")
    with torch.no_grad():
        values = equation.exact_solution(0.0, x_tensor)
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.shape[0] != x.shape[0]:
        raise ValueError("exact_solution must return one scalar per input point")
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("exact_solution returned NaN or infinity")
    return result


def _relative_l2(predicted: np.ndarray, expected: np.ndarray) -> float:
    denominator = float(np.linalg.norm(expected))
    numerator = float(np.linalg.norm(predicted - expected))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def run_tt_cross(equation: ExactEquation, config: TTConfig) -> TTResult:
    """Approximate ``equation.exact_solution(0, x)`` on a Chebyshev grid.

    Validation data controls early stopping inside TT-cross. Accuracy is then
    reported on both independent unseen grid indices and continuous points.
    """
    teneva = _load_teneva()
    equation_id = _equation_id(equation, config.equation_id)
    lower, upper = _bounds(config, equation_id)
    shape = [config.grid_size] * config.dim
    rng = np.random.default_rng(config.seed)

    def indices_to_points(indices: np.ndarray) -> np.ndarray:
        return np.asarray(
            teneva.ind_to_poi(indices, lower, upper, shape, kind="cheb"),
            dtype=np.float64,
        )

    def target(indices: np.ndarray) -> np.ndarray:
        return _exact_values(equation, indices_to_points(indices))

    validation_indices = rng.integers(
        0, config.grid_size, size=(config.validation_samples, config.dim)
    )
    validation_values = target(validation_indices)
    initial = teneva.rand(shape, r=config.initial_rank, seed=config.seed)
    info: dict[str, Any] = {}
    started = perf_counter()
    tensor = teneva.cross(
        target,
        initial,
        m=config.max_samples,
        nswp=config.max_sweeps,
        dr_min=config.rank_increment_min,
        dr_max=config.rank_increment_max,
        info=info,
        cache={},
        I_vld=validation_indices,
        y_vld=validation_values,
        e_vld=config.target_eps,
        log=False,
    )
    build_seconds = perf_counter() - started

    # Round only below the requested tolerance so this cannot manufacture a pass.
    tensor = teneva.truncate(tensor, e=config.target_eps * 0.1)
    grid_indices = rng.integers(0, config.grid_size, size=(config.grid_test_samples, config.dim))
    grid_expected = target(grid_indices)
    grid_predicted = np.asarray(teneva.get_many(tensor, grid_indices))
    grid_error = _relative_l2(grid_predicted, grid_expected)

    continuous_points = rng.uniform(lower, upper, size=(config.continuous_test_samples, config.dim))
    coefficients = teneva.func_int(tensor)
    continuous_predicted = np.asarray(
        teneva.func_get(continuous_points, coefficients, lower, upper)
    )
    continuous_expected = _exact_values(equation, continuous_points)
    continuous_error = _relative_l2(continuous_predicted, continuous_expected)

    ranks = [int(rank) for rank in teneva.ranks(tensor)]
    params = int(teneva.size(tensor))
    full_size = int(prod(shape))
    total_seconds = perf_counter() - started
    x0 = getattr(equation, "x0", None)
    estimate = None
    if x0 is not None:
        if hasattr(x0, "detach"):
            x0 = x0.detach().cpu().numpy()
        point = np.asarray(x0, dtype=np.float64).reshape(1, config.dim)
        estimate = float(teneva.func_get(point, coefficients, lower, upper))

    return TTResult(
        status="completed" if grid_error <= config.target_eps else "not_reached",
        estimate=estimate,
        target_eps=config.target_eps,
        grid_rel_l2=grid_error,
        continuous_rel_l2=continuous_error,
        ranks=ranks,
        max_rank=max(ranks),
        n_params_tt=params,
        full_grid_elements=full_size,
        compression_ratio=float(full_size / params),
        function_evaluations=int(info.get("m", 0)),
        sweeps=int(info.get("nswp", 0)),
        stop_reason=str(info.get("stop", "unknown")),
        build_seconds=build_seconds,
        total_seconds=total_seconds,
        config=asdict(config),
    )


def run_tier1(config: TTConfig, equation: ExactEquation) -> TTResult:
    """Config-first compatibility entry point used by the unified runner."""
    return run_tt_cross(equation, config)
