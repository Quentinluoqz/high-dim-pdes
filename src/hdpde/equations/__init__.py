"""Benchmark equation public API."""

from hdpde.utils.registry import register_equation

from .base import Equation
from .problems import (
    EQUATION_REGISTRY,
    HJBLQ,
    AllenCahn,
    BlackScholesBarenblatt,
    HeatEquation,
    LQControl,
    make_equation,
)

# Importing the benchmark package makes all stable IDs available to the
# project-wide configuration registry used by the atomic runner.
for _equation_id, _equation_type in EQUATION_REGISTRY.items():
    register_equation(_equation_id)(_equation_type)

# Concise aliases are convenient in experiment manifests and notebooks.
E1 = BlackScholesBarenblatt
E2 = LQControl
E3 = HJBLQ
E4 = AllenCahn
E5 = HeatEquation

__all__ = [
    "EQUATION_REGISTRY",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "AllenCahn",
    "BlackScholesBarenblatt",
    "Equation",
    "HJBLQ",
    "HeatEquation",
    "LQControl",
    "make_equation",
]
