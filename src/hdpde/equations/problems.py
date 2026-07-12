"""Benchmark equations E1--E5."""

from __future__ import annotations

import math

import torch

from hdpde.reference import ReferenceResult, cole_hopf_reference

from .base import Equation, ensure_output_shape, isotropic_diffusion


def _exact_reference(value: torch.Tensor, *, source: str) -> ReferenceResult:
    if value.numel() != 1:
        raise ValueError("reference_solution requires a single state")
    return ReferenceResult(
        value=float(value.detach().cpu().reshape(())),
        ci95=None,
        kind="exact",
        source=source,
        sample_count=None,
    )


class BlackScholesBarenblatt(Equation):
    """E1: Black--Scholes--Barenblatt benchmark with quadratic terminal data."""

    equation_id = "E1"
    name = "black_scholes_barenblatt"

    def __init__(self, dim: int, *, T: float = 1.0, r: float = 0.05, sigma: float = 0.4) -> None:
        super().__init__(dim, T=T)
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.r = float(r)
        self.sigma = float(sigma)

    @property
    def x0(self) -> torch.Tensor:
        return torch.ones(self.dim)

    def drift(self, t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        return torch.zeros_like(x)

    def diffusion(self, t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        return torch.diag_embed(self.sigma * x)

    def nonlinearity(
        self,
        t: float | torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_x(x)
        if z.shape != x.shape:
            raise ValueError("z must have the same shape as x")
        summed_z = z.sum(dim=-1, keepdim=True)
        value = -self.r * (y - ensure_output_shape(summed_z / self.sigma, y))
        return value

    def terminal(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        return x.square().sum(dim=-1, keepdim=True)

    def exact_solution(self, t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        time = torch.as_tensor(t, dtype=x.dtype, device=x.device)
        factor = torch.exp((self.r + self.sigma**2) * (self.T - time))
        return factor * x.square().sum(dim=-1, keepdim=True)

    def reference_solution(
        self, t: float = 0.0, x: torch.Tensor | None = None, **kwargs: object
    ) -> ReferenceResult:
        state = self._default_reference_x(x)
        return _exact_reference(
            self.exact_solution(t, state),
            source="analytic Black--Scholes--Barenblatt quadratic solution",
        )


class LQControl(Equation):
    """E2: isotropic linear-quadratic Hamilton--Jacobi equation."""

    equation_id = "E2"
    name = "lq_control"

    def __init__(
        self, dim: int, *, T: float = 1.0, q: float = 1.0, q_terminal: float = 0.5
    ) -> None:
        super().__init__(dim, T=T)
        if q <= 0 or q_terminal < 0:
            raise ValueError("q must be positive and q_terminal non-negative")
        self.q = float(q)
        self.q_terminal = float(q_terminal)

    @property
    def x0(self) -> torch.Tensor:
        return torch.zeros(self.dim)

    def drift(self, t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        return torch.zeros_like(x)

    def diffusion(self, t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        return isotropic_diffusion(x, math.sqrt(2.0))

    def nonlinearity(
        self,
        t: float | torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_x(x)
        if z.shape != x.shape:
            raise ValueError("z must have the same shape as x")
        value = -0.25 * z.square().sum(dim=-1, keepdim=True) + 0.5 * self.q * x.square().sum(
            dim=-1, keepdim=True
        )
        return ensure_output_shape(value, y)

    def terminal(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        return 0.5 * self.q_terminal * x.square().sum(dim=-1, keepdim=True)

    def _riccati(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tau = self.T - t
        root_q = math.sqrt(self.q)
        argument = root_q * tau
        denominator = torch.cosh(argument) + (self.q_terminal / root_q) * torch.sinh(argument)
        numerator = root_q * torch.sinh(argument) + self.q_terminal * torch.cosh(argument)
        return numerator / denominator, torch.log(denominator)

    def exact_solution(self, t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        time = torch.as_tensor(t, dtype=x.dtype, device=x.device)
        p, integral_p = self._riccati(time)
        return 0.5 * p * x.square().sum(dim=-1, keepdim=True) + self.dim * integral_p

    def reference_solution(
        self, t: float = 0.0, x: torch.Tensor | None = None, **kwargs: object
    ) -> ReferenceResult:
        state = self._default_reference_x(x)
        return _exact_reference(
            self.exact_solution(t, state),
            source="analytic scalar Riccati solution",
        )


class HJBLQ(Equation):
    """E3: Cole--Hopf-solvable HJB benchmark."""

    equation_id = "E3"
    name = "hjblq"

    def __init__(self, dim: int, *, T: float = 1.0, lambda_: float = 1.0) -> None:
        super().__init__(dim, T=T)
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive")
        self.lambda_ = float(lambda_)
        self.diffusion_scale = math.sqrt(2.0)

    @property
    def x0(self) -> torch.Tensor:
        return torch.zeros(self.dim)

    def drift(self, t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        return torch.zeros_like(x)

    def diffusion(self, t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        return isotropic_diffusion(x, self.diffusion_scale)

    def nonlinearity(
        self,
        t: float | torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_x(x)
        if z.shape != x.shape:
            raise ValueError("z must have the same shape as x")
        value = -0.5 * self.lambda_ * z.square().sum(dim=-1, keepdim=True)
        return ensure_output_shape(value, y)

    def terminal(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        return torch.log((1.0 + x.square().sum(dim=-1, keepdim=True)) / 2.0)

    def reference_solution(
        self,
        t: float = 0.0,
        x: torch.Tensor | None = None,
        **kwargs: object,
    ) -> ReferenceResult:
        if not 0.0 <= t <= self.T:
            raise ValueError(f"t must lie in [0, {self.T}]")
        state = self._default_reference_x(x)
        sample_count = int(kwargs.pop("sample_count", 10_000_000))
        batch_size = int(kwargs.pop("batch_size", 100_000))
        seed = int(kwargs.pop("seed", 0))
        if kwargs:
            raise TypeError(f"unknown reference options: {sorted(kwargs)}")
        return cole_hopf_reference(
            self.terminal,
            x=state,
            remaining_time=self.T - t,
            diffusion_scale=self.diffusion_scale,
            lambda_=self.lambda_,
            sample_count=sample_count,
            batch_size=batch_size,
            seed=seed,
            source="Cole--Hopf transform with batched Gaussian Monte Carlo",
        )


class AllenCahn(Equation):
    """E4: Allen--Cahn benchmark with a literature reference at d=100 only."""

    equation_id = "E4"
    name = "allen_cahn"
    CANONICAL_REFERENCE = 0.052802

    def __init__(self, dim: int, *, T: float = 0.3) -> None:
        super().__init__(dim, T=T)

    @property
    def x0(self) -> torch.Tensor:
        return torch.zeros(self.dim)

    def drift(self, t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        return torch.zeros_like(x)

    def diffusion(self, t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        return isotropic_diffusion(x, math.sqrt(2.0))

    def nonlinearity(
        self,
        t: float | torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_x(x)
        return y - y.pow(3)

    def terminal(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        return 0.5 / (1.0 + 0.2 * x.square().sum(dim=-1, keepdim=True))

    def reference_solution(
        self, t: float = 0.0, x: torch.Tensor | None = None, **kwargs: object
    ) -> ReferenceResult:
        state = self._default_reference_x(x)
        is_canonical = (
            self.dim == 100 and abs(t) <= 1e-15 and bool(torch.count_nonzero(state).item() == 0)
        )
        if not is_canonical:
            return ReferenceResult(
                value=None,
                ci95=None,
                kind="unavailable",
                source="literature reference is only valid for d=100, t=0, x=0",
                sample_count=None,
            )
        return ReferenceResult(
            value=self.CANONICAL_REFERENCE,
            ci95=None,
            kind="literature_approximation",
            source="Han, Jentzen & E (2018), 100D Allen--Cahn benchmark",
            sample_count=None,
        )


class HeatEquation(Equation):
    """E5: heat equation with a Gaussian terminal condition."""

    equation_id = "E5"
    name = "heat_equation"

    def __init__(self, dim: int, *, T: float = 1.0) -> None:
        super().__init__(dim, T=T)

    @property
    def x0(self) -> torch.Tensor:
        return torch.zeros(self.dim)

    def drift(self, t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        return torch.zeros_like(x)

    def diffusion(self, t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        return isotropic_diffusion(x, math.sqrt(2.0))

    def nonlinearity(
        self,
        t: float | torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_x(x)
        return torch.zeros_like(y)

    def terminal(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        return torch.exp(-x.square().sum(dim=-1, keepdim=True) / (2.0 * self.dim))

    def exact_solution(self, t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        time = torch.as_tensor(t, dtype=x.dtype, device=x.device)
        remaining = self.T - time
        denominator = self.dim + 2.0 * remaining
        prefactor = (self.dim / denominator).pow(0.5 * self.dim)
        return prefactor * torch.exp(-x.square().sum(dim=-1, keepdim=True) / (2.0 * denominator))

    def reference_solution(
        self, t: float = 0.0, x: torch.Tensor | None = None, **kwargs: object
    ) -> ReferenceResult:
        state = self._default_reference_x(x)
        return _exact_reference(
            self.exact_solution(t, state), source="closed-form Gaussian heat kernel"
        )


EQUATION_REGISTRY: dict[str, type[Equation]] = {
    "E1": BlackScholesBarenblatt,
    "E2": LQControl,
    "E3": HJBLQ,
    "E4": AllenCahn,
    "E5": HeatEquation,
}


def make_equation(equation_id: str, dim: int, **kwargs: object) -> Equation:
    """Construct a benchmark equation by its stable E1--E5 identifier."""

    key = equation_id.upper()
    try:
        equation_type = EQUATION_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown equation {equation_id!r}; choose from {sorted(EQUATION_REGISTRY)}"
        ) from exc
    return equation_type(dim=dim, **kwargs)
