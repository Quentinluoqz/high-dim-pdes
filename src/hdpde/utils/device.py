"""Device and numerical precision selection."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PrecisionContext:
    device: torch.device
    dtype: torch.dtype
    tf32: bool


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve auto to CUDA, MPS, then CPU; explicit requests fail closed."""

    require_cuda = os.getenv("HDPDE_REQUIRE_CUDA") == "1"
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if require_cuda:
            raise RuntimeError("HDPDE_REQUIRE_CUDA=1 but CUDA is unavailable")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def precision_context(precision: str, requested_device: str = "auto") -> PrecisionContext:
    """Configure TF32 explicitly and return device/dtype choices."""

    device = resolve_device(requested_device)
    if precision not in {"fp32", "tf32", "fp64"}:
        raise ValueError(f"unsupported precision: {precision}")
    if precision == "tf32" and device.type != "cuda":
        raise RuntimeError("TF32 requires CUDA hardware")
    tf32 = precision == "tf32"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
    return PrecisionContext(
        device=device, dtype=torch.float64 if precision == "fp64" else torch.float32, tf32=tf32
    )
