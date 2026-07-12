"""Common interface for semilinear parabolic equations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from hdpde.reference import ReferenceResult


class Equation(ABC):
    """An equation in the Deep-BSDE convention.

    The represented PDE is

    ``u_t + mu.grad(u) + 0.5 Tr(sigma sigma^T Hess(u)) + f(t,x,u,z) = 0``

    where ``z = sigma^T grad(u)``.  State tensors use the final axis for the
    spatial dimension.  Diffusions have two final matrix axes.
    """

    equation_id: str
    name: str

    def __init__(self, dim: int, *, T: float) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if T <= 0:
            raise ValueError("T must be positive")
        self.dim = dim
        self.T = float(T)

    @property
    @abstractmethod
    def x0(self) -> torch.Tensor:
        """Default initial state as a one-dimensional CPU tensor."""

    def _validate_x(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim < 1 or x.shape[-1] != self.dim:
            raise ValueError(f"expected x.shape[-1] == {self.dim}, got {x.shape}")
        if not (x.dtype.is_floating_point or x.dtype.is_complex):
            raise TypeError("x must have a floating-point dtype")

    def _default_reference_x(self, x: torch.Tensor | None) -> torch.Tensor:
        if x is None:
            return self.x0.to(dtype=torch.float64)
        self._validate_x(x)
        if x.ndim != 1:
            raise ValueError("reference_solution returns one scalar; x must be one-dimensional")
        return x

    @abstractmethod
    def drift(self, t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return ``mu(t,x)`` with the same shape, dtype and device as ``x``."""

    @abstractmethod
    def diffusion(self, t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return ``sigma(t,x)`` with shape ``x.shape[:-1] + (d,d)``."""

    @abstractmethod
    def nonlinearity(
        self,
        t: float | torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``f(t,x,y,z)`` with the same shape as ``y``."""

    @abstractmethod
    def terminal(self, x: torch.Tensor) -> torch.Tensor:
        """Return terminal data with shape ``x.shape[:-1] + (1,)``."""

    @abstractmethod
    def reference_solution(
        self, t: float = 0.0, x: torch.Tensor | None = None, **kwargs: object
    ) -> ReferenceResult:
        """Return a scalar reference value and provenance."""


def isotropic_diffusion(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Construct a batched ``scale * I`` preserving dtype and device."""

    dim = x.shape[-1]
    identity = torch.eye(dim, dtype=x.dtype, device=x.device)
    return identity.expand(x.shape[:-1] + (dim, dim)) * scale


def ensure_output_shape(value: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Make a scalar-per-sample tensor conform to the solver's y shape."""

    if value.shape == y.shape:
        return value
    if value.shape == y.shape + (1,):
        return value.squeeze(-1)
    if value.shape + (1,) == y.shape:
        return value.unsqueeze(-1)
    return value.expand_as(y)
