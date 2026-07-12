"""Configuration and precision handling for Deep BSDE solvers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

Precision = Literal["fp32", "tf32", "fp64"]
Variant = Literal["m1a", "m1b"]


@dataclass(frozen=True, slots=True)
class DeepBSDEConfig:
    """All numerical settings needed for a reproducible Deep BSDE run."""

    dimension: int
    steps: int = 20
    batch_size: int = 64
    world_size: int = 1
    global_batch_size: int | None = None
    iterations: int = 1_000
    learning_rate: float = 1.0e-3
    scheduler_gamma: float = 1.0
    seed: int = 0
    precision: Precision = "fp32"
    device: str = "cpu"
    variant: Variant = "m1a"
    hidden_width_add: int = 10
    gradient_clip: float | None = None
    checkpoint_interval: int = 0
    max_seconds: float | None = None
    external_config_hash: str | None = None

    def __post_init__(self) -> None:
        positive_ints = {
            "dimension": self.dimension,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "iterations": self.iterations,
            "world_size": self.world_size,
        }
        for name, value in positive_ints.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_width_add < 0:
            raise ValueError("hidden_width_add must be non-negative")
        if self.global_batch_size is not None:
            if self.global_batch_size < 1:
                raise ValueError("global_batch_size must be positive when set")
            if self.global_batch_size != self.batch_size * self.world_size:
                raise ValueError("global_batch_size must equal batch_size * world_size")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.scheduler_gamma <= 1:
            raise ValueError("scheduler_gamma must be in (0, 1]")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive when set")
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be non-negative")
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive when set")
        if self.precision not in {"fp32", "tf32", "fp64"}:
            raise ValueError(f"unsupported precision: {self.precision}")
        if self.variant not in {"m1a", "m1b"}:
            raise ValueError(f"unsupported variant: {self.variant}")

    @property
    def dtype(self) -> torch.dtype:
        return torch.float64 if self.precision == "fp64" else torch.float32

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()


def configure_precision(precision: Precision) -> torch.dtype:
    """Configure CUDA TF32 switches and return the parameter dtype.

    TF32 is deliberately represented by float32 tensors plus enabled CUDA
    tensor-core math. It never enables AMP or half precision.
    """

    if precision not in {"fp32", "tf32", "fp64"}:
        raise ValueError(f"unsupported precision: {precision}")
    use_tf32 = precision == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    # This also makes the distinction explicit on recent PyTorch versions.
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    return torch.float64 if precision == "fp64" else torch.float32
