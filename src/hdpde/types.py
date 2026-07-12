"""Stable, JSON-serializable public result types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class RunStatus(StrEnum):
    """Terminal states retained in the scientific ledger."""

    COMPLETED = "completed"
    DIVERGED = "diverged"
    FAILED = "failed"
    TIMEOUT = "timeout"
    NOT_REACHED = "not_reached"


@dataclass(slots=True)
class RunMetrics:
    """Versioned metrics emitted by every solver implementation."""

    run_id: str
    method: str
    equation: str
    dim: int
    seed: int
    precision: str
    status: RunStatus
    config_hash: str
    source_commit: str
    estimate: float | None = None
    abs_error: float | None = None
    rel_error: float | None = None
    reference: dict[str, Any] | None = None
    wall_clock_sec: float | None = None
    train_sec: float | None = None
    throughput: float | None = None
    peak_gpu_mem_mb: float | None = None
    n_params: int | None = None
    tt_ranks: list[int] | None = None
    iterations: int | None = None
    hardware: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    message: str | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary, including string enum values."""

        data = asdict(self)
        data["status"] = self.status.value
        return data
