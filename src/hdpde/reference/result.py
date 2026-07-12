"""Typed results returned by equation reference solvers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    """A scalar reference value together with its provenance.

    ``value`` is ``None`` when no defensible reference is available.  Exact
    references have ``ci95=None``; Monte Carlo references contain a two-sided
    95% confidence interval.
    """

    value: float | None
    ci95: tuple[float, float] | None
    kind: str
    source: str
    sample_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""

        return asdict(self)
