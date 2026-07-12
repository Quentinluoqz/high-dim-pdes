"""Tensor-train baselines and optional external solver integration."""

from .tier1_cross import TenevaUnavailableError, TTConfig, TTResult, run_tier1, run_tt_cross
from .tier2_wrapper import Tier2Diagnostics, diagnose_tier2, require_tier2

__all__ = [
    "TTConfig",
    "TTResult",
    "TenevaUnavailableError",
    "Tier2Diagnostics",
    "diagnose_tier2",
    "require_tier2",
    "run_tier1",
    "run_tt_cross",
]
