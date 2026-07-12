"""Memory-bounded Monte Carlo utilities for reference values."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from .result import ReferenceResult


def cole_hopf_reference(
    terminal: Callable[[torch.Tensor], torch.Tensor],
    *,
    x: torch.Tensor,
    remaining_time: float,
    diffusion_scale: float,
    lambda_: float,
    sample_count: int,
    batch_size: int,
    seed: int,
    source: str,
) -> ReferenceResult:
    """Estimate a Cole--Hopf reference without retaining all samples.

    For ``u_t + 0.5 sigma^2 Delta u - lambda |grad u|^2 = 0``, the
    transform is ``u = -sigma^2/(2 lambda) log E[exp(-2 lambda g/sigma^2)]``.
    The implementation accumulates first and second moments in float64 and
    maps the confidence interval for the transformed mean back to ``u``.
    """

    if sample_count <= 1:
        raise ValueError("sample_count must be greater than one")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if remaining_time < 0:
        raise ValueError("remaining_time must be non-negative")
    if diffusion_scale <= 0:
        raise ValueError("diffusion_scale must be positive")
    if lambda_ <= 0:
        raise ValueError("lambda_ must be positive")
    if x.ndim != 1:
        raise ValueError("Monte Carlo reference expects a single state vector")

    # A CPU generator makes seeded references stable and avoids materialising
    # the formal 10^7-sample production calculation on an accelerator.
    x_cpu = x.detach().to(device="cpu", dtype=torch.float64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    alpha = 2.0 * lambda_ / (diffusion_scale * diffusion_scale)
    noise_scale = diffusion_scale * math.sqrt(remaining_time)

    total = torch.zeros((), dtype=torch.float64)
    total_sq = torch.zeros((), dtype=torch.float64)
    completed = 0
    while completed < sample_count:
        n = min(batch_size, sample_count - completed)
        noise = torch.randn((n, x_cpu.numel()), dtype=torch.float64, generator=generator)
        states = x_cpu.unsqueeze(0) + noise_scale * noise
        transformed = torch.exp(-alpha * terminal(states).reshape(-1).double())
        total += transformed.sum()
        total_sq += transformed.square().sum()
        completed += n

    mean = float(total / sample_count)
    # Unbiased variance, formed from aggregate moments. Clamp only protects
    # against tiny negative roundoff for nearly deterministic terminal data.
    variance = float(
        torch.clamp(
            (total_sq - total.square() / sample_count) / (sample_count - 1),
            min=0.0,
        )
    )
    # Moment subtraction can leave an O(eps) positive remainder for an
    # exactly constant random variable.  Treat only that roundoff scale as
    # zero; genuine low-variance estimates remain untouched.
    if variance <= 32.0 * torch.finfo(torch.float64).eps * mean * mean:
        variance = 0.0
    standard_error = math.sqrt(variance / sample_count)
    radius = 1.959963984540054 * standard_error
    lower_mean = max(mean - radius, torch.finfo(torch.float64).tiny)
    upper_mean = max(mean + radius, torch.finfo(torch.float64).tiny)
    coefficient = -(diffusion_scale * diffusion_scale) / (2.0 * lambda_)

    value = coefficient * math.log(mean)
    # The logarithmic map is decreasing because coefficient is negative.
    ci95 = (
        coefficient * math.log(upper_mean),
        coefficient * math.log(lower_mean),
    )
    return ReferenceResult(
        value=value,
        ci95=ci95,
        kind="monte_carlo_cole_hopf",
        source=source,
        sample_count=sample_count,
    )
