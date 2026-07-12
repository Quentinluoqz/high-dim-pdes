"""Shared solver protocol."""

from __future__ import annotations

from typing import Protocol

from hdpde.types import RunMetrics
from hdpde.utils.config import RunConfig


class Solver(Protocol):
    """Contract implemented by all runnable methods."""

    def run(self, config: RunConfig) -> RunMetrics:
        """Run one fully resolved experiment."""
