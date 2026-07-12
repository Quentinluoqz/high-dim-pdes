"""Neural time-discretizations used by the Deep BSDE solver."""

from __future__ import annotations

from typing import Protocol, cast, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class EquationProtocol(Protocol):
    """Structural interface consumed by :class:`DeepBSDEModel`."""

    T: float
    x0: torch.Tensor

    def drift(self, t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor: ...

    def diffusion(self, t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor: ...

    def nonlinearity(
        self,
        t: float | torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor: ...

    def terminal(self, x: torch.Tensor) -> torch.Tensor: ...

    def reference_solution(
        self,
        t: float = 0.0,
        x: torch.Tensor | None = None,
        **kwargs: object,
    ) -> object: ...


def _stepwise_mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    """Original M1a block: two Linear--BatchNorm--ReLU stages."""

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _shared_mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    """Smooth shared network used by M1b."""

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class StepwiseZNetwork(nn.Module):
    """M1a: an independent ``x -> z`` network at each internal time step."""

    def __init__(self, dimension: int, steps: int, hidden_dim: int) -> None:
        super().__init__()
        self.networks = nn.ModuleList(
            [_stepwise_mlp(dimension, dimension, hidden_dim) for _ in range(max(steps - 1, 0))]
        )

    def forward(self, step: int, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        if step <= 0:
            raise ValueError("the trainable z0 must be used at step zero")
        return cast(torch.Tensor, self.networks[step - 1](x))


class SharedZNetwork(nn.Module):
    """M1b: one shared ``(t, x) -> z`` network for all internal steps."""

    def __init__(self, dimension: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = _shared_mlp(dimension + 1, dimension, hidden_dim)

    def forward(self, step: int, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del step
        t_column = t.expand(x.shape[0], 1)
        return cast(torch.Tensor, self.network(torch.cat((t_column, x), dim=1)))


class DeepBSDEModel(nn.Module):
    """Euler-Maruyama Deep BSDE discretization with trainable ``u0``/``z0``."""

    def __init__(
        self,
        dimension: int,
        steps: int,
        variant: str,
        hidden_width_add: int = 10,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.steps = steps
        self.variant = variant
        self.u0 = nn.Parameter(torch.zeros(()))
        self.z0 = nn.Parameter(torch.zeros(dimension))
        hidden = dimension + hidden_width_add
        if variant == "m1a":
            self.z_network: nn.Module = StepwiseZNetwork(dimension, steps, hidden)
        elif variant == "m1b":
            self.z_network = SharedZNetwork(dimension, hidden)
        else:
            raise ValueError(f"unsupported Deep BSDE variant: {variant}")

    def _initial_state(
        self, equation: EquationProtocol, batch_size: int, *, device: torch.device
    ) -> torch.Tensor:
        x0 = torch.as_tensor(equation.x0, device=device, dtype=self.u0.dtype)
        if x0.ndim == 0:
            x0 = x0.repeat(self.dimension)
        x0 = x0.reshape(-1)
        if x0.numel() != self.dimension:
            raise ValueError(f"equation x0 has {x0.numel()} entries, expected {self.dimension}")
        return x0.unsqueeze(0).expand(batch_size, -1).clone()

    @staticmethod
    def _as_batch_vector(value: torch.Tensor, batch_size: int) -> torch.Tensor:
        if value.ndim == 0:
            return value.expand(batch_size)
        if value.shape == (batch_size, 1):
            return value[:, 0]
        if value.shape == (batch_size,):
            return value
        raise ValueError(f"expected scalar batch output, got shape {tuple(value.shape)}")

    @staticmethod
    def _apply_diffusion(sigma: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
        batch, dimension = dw.shape
        if sigma.ndim == 0:
            return sigma * dw
        if sigma.shape == (dimension,):
            return sigma.unsqueeze(0) * dw
        if sigma.shape == (batch, dimension):
            return sigma * dw
        if sigma.shape == (dimension, dimension):
            return torch.einsum("ij,bj->bi", sigma, dw)
        if sigma.shape == (batch, dimension, dimension):
            return torch.einsum("bij,bj->bi", sigma, dw)
        raise ValueError(f"unsupported diffusion shape: {tuple(sigma.shape)}")

    @staticmethod
    def _as_state_drift(mu: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if mu.ndim == 0:
            return mu.expand_as(x)
        if mu.shape == (x.shape[1],):
            return mu.unsqueeze(0).expand_as(x)
        if mu.shape == x.shape:
            return mu
        raise ValueError(f"unsupported drift shape: {tuple(mu.shape)}")

    def simulate(
        self,
        equation: EquationProtocol,
        normal_increments: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return propagated terminal ``Y`` and the terminal condition target."""

        if normal_increments.ndim != 3:
            raise ValueError("normal_increments must have shape (batch, steps, dimension)")
        batch_size, steps, dimension = normal_increments.shape
        if (steps, dimension) != (self.steps, self.dimension):
            raise ValueError(
                f"increments have {(steps, dimension)}, expected {(self.steps, self.dimension)}"
            )
        device = normal_increments.device
        dtype = self.u0.dtype
        horizon = torch.as_tensor(equation.T, device=device, dtype=dtype)
        dt = horizon / self.steps
        sqrt_dt = torch.sqrt(dt)
        x = self._initial_state(equation, batch_size, device=device)
        y = self.u0.expand(batch_size)

        for step in range(self.steps):
            t = dt * step
            z = self.z0.unsqueeze(0).expand(batch_size, -1)
            if step > 0:
                z = self.z_network(step, t.reshape(1, 1), x)
            dw = normal_increments[:, step, :] * sqrt_dt
            driver = torch.as_tensor(equation.nonlinearity(t, x, y, z), device=device, dtype=dtype)
            driver = self._as_batch_vector(driver, batch_size)
            y = y - driver * dt + torch.sum(z * dw, dim=1)
            mu = torch.as_tensor(equation.drift(t, x), device=device, dtype=dtype)
            sigma = torch.as_tensor(equation.diffusion(t, x), device=device, dtype=dtype)
            x = x + self._as_state_drift(mu, x) * dt + self._apply_diffusion(sigma, dw)

        target = torch.as_tensor(equation.terminal(x), device=device, dtype=dtype)
        return y, self._as_batch_vector(target, batch_size)

    def loss(self, equation: EquationProtocol, normal_increments: torch.Tensor) -> torch.Tensor:
        terminal_y, target = self.simulate(equation, normal_increments)
        return torch.mean(torch.square(terminal_y - target))

    def forward(self, equation: EquationProtocol, normal_increments: torch.Tensor) -> torch.Tensor:
        """DDP-compatible forward entry returning the terminal mismatch loss."""

        return self.loss(equation, normal_increments)
