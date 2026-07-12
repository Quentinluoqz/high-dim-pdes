"""Training, status handling, and resumable checkpoints for Deep BSDE."""

from __future__ import annotations

import math
import os
import platform
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from hdpde.types import RunMetrics
from hdpde.types import RunStatus as PublicRunStatus
from hdpde.utils.config import RunConfig
from hdpde.utils.device import resolve_device
from hdpde.utils.io import make_run_id, prepare_run_dir

from .config import DeepBSDEConfig, configure_precision
from .model import DeepBSDEModel, EquationProtocol

RunStatus = Literal["completed", "diverged", "failed", "timeout", "not_reached"]


@dataclass(slots=True)
class TrainingResult:
    status: RunStatus
    estimate: float | None
    final_loss: float | None
    iterations_completed: int
    training_seconds: float
    total_seconds: float
    throughput: float | None
    parameter_count: int
    precision: str
    config_hash: str
    device: str
    hardware: str
    error: str | None = None
    history: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "estimate": self.estimate,
            "final_loss": self.final_loss,
            "iterations_completed": self.iterations_completed,
            "training_seconds": self.training_seconds,
            "total_seconds": self.total_seconds,
            "throughput": self.throughput,
            "parameter_count": self.parameter_count,
            "precision": self.precision,
            "config_hash": self.config_hash,
            "device": self.device,
            "hardware": self.hardware,
            "error": self.error,
            "history": self.history,
        }


def _hardware_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Metal Performance Shaders"
    return platform.processor() or platform.machine() or "CPU"


class DeepBSDETrainer:
    """Own the optimizer state and reproducible lifecycle of one solver run."""

    CHECKPOINT_VERSION = 1

    def __init__(self, equation: EquationProtocol, config: DeepBSDEConfig) -> None:
        self.equation = equation
        self.config = config
        dtype = configure_precision(config.precision)
        environment_world_size = int(os.getenv("WORLD_SIZE", "1"))
        if environment_world_size != config.world_size:
            raise RuntimeError(
                "configured world_size does not match torchrun environment: "
                f"configured={config.world_size}, actual={environment_world_size}"
            )
        self.world_size = config.world_size
        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        if self.world_size > 1 and not dist.is_initialized():
            backend = "nccl" if config.device.startswith("cuda") else "gloo"
            dist.init_process_group(backend=backend)
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and self.world_size > 1:
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")

        rank_seed = config.seed + self.rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rank_seed)

        self.model = DeepBSDEModel(
            config.dimension,
            config.steps,
            config.variant,
            config.hidden_width_add,
        ).to(device=self.device, dtype=dtype)
        self.train_model: torch.nn.Module = self.model
        if self.world_size > 1:
            device_ids = [self.local_rank] if self.device.type == "cuda" else None
            self.train_model = DistributedDataParallel(self.model, device_ids=device_ids)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=config.scheduler_gamma
        )
        self.iteration = 0
        self.history: list[float] = []
        self.training_seconds = 0.0

    def _random_batch(self) -> torch.Tensor:
        return torch.randn(
            self.config.batch_size,
            self.config.steps,
            self.config.dimension,
            device=self.device,
            dtype=self.config.dtype,
        )

    def _all_ranks_true(self, value: bool) -> bool:
        if self.world_size == 1:
            return value
        flag = torch.tensor(1 if value else 0, dtype=torch.int32, device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def _local_rng_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _checkpoint_state(self, rank_rng_states: list[dict[str, Any]]) -> dict[str, Any]:
        state: dict[str, Any] = {
            "version": self.CHECKPOINT_VERSION,
            "config_hash": self.config.config_hash,
            "config": self.config.canonical_dict(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "iteration": self.iteration,
            "history": self.history,
            "training_seconds": self.training_seconds,
            "rank_rng_states": rank_rng_states,
            # Version-1 single-rank keys remain for backward compatibility.
            "python_rng_state": rank_rng_states[0]["python"],
            "numpy_rng_state": rank_rng_states[0]["numpy"],
            "torch_rng_state": rank_rng_states[0]["torch"],
        }
        if "cuda" in rank_rng_states[0]:
            state["cuda_rng_state_all"] = rank_rng_states[0]["cuda"]
        return state

    def save_checkpoint(self, path: str | os.PathLike[str]) -> None:
        local_rng = self._local_rng_state()
        rank_states: list[dict[str, Any]] | None = None
        if self.world_size > 1:
            if self.rank == 0:
                rank_states = [{} for _ in range(self.world_size)]
            dist.gather_object(local_rng, rank_states, dst=0)
            if self.rank != 0:
                return
        else:
            rank_states = [local_rng]
        assert rank_states is not None
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        torch.save(self._checkpoint_state(rank_states), temporary)
        os.replace(temporary, destination)

    def load_checkpoint(self, path: str | os.PathLike[str]) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("version") != self.CHECKPOINT_VERSION:
            raise ValueError("unsupported Deep BSDE checkpoint version")
        stored_hash = checkpoint.get("config_hash")
        if stored_hash != self.config.config_hash:
            raise ValueError(
                "checkpoint configuration mismatch: "
                f"stored={stored_hash}, current={self.config.config_hash}"
            )
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.iteration = int(checkpoint["iteration"])
        self.history = [float(value) for value in checkpoint["history"]]
        self.training_seconds = float(checkpoint["training_seconds"])
        rank_states = checkpoint.get("rank_rng_states")
        if rank_states:
            if len(rank_states) != self.world_size:
                raise ValueError("checkpoint world size does not match the current run")
            rng_state = rank_states[self.rank]
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"].cpu())
            if torch.cuda.is_available() and "cuda" in rng_state:
                torch.cuda.set_rng_state_all(rng_state["cuda"])
        else:
            random.setstate(checkpoint["python_rng_state"])
            np.random.set_state(checkpoint["numpy_rng_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

    def _result(
        self,
        status: RunStatus,
        started_at: float,
        *,
        error: str | None = None,
    ) -> TrainingResult:
        total_seconds = time.monotonic() - started_at
        samples = self.iteration * self.config.batch_size * self.world_size
        throughput = samples / self.training_seconds if self.training_seconds > 0 else None
        estimate_value = float(self.model.u0.detach().cpu())
        estimate = estimate_value if math.isfinite(estimate_value) else None
        final_loss = self.history[-1] if self.history else None
        return TrainingResult(
            status=status,
            estimate=estimate,
            final_loss=final_loss,
            iterations_completed=self.iteration,
            training_seconds=self.training_seconds,
            total_seconds=total_seconds,
            throughput=throughput,
            parameter_count=sum(parameter.numel() for parameter in self.model.parameters()),
            precision=self.config.precision,
            config_hash=self.config.config_hash,
            device=str(self.device),
            hardware=_hardware_name(self.device),
            error=error,
            history=list(self.history),
        )

    def train(
        self,
        *,
        checkpoint_path: str | os.PathLike[str] | None = None,
        stop_after: int | None = None,
    ) -> TrainingResult:
        """Train up to the configured iteration count or a testing interruption.

        ``stop_after`` is an absolute iteration number. It lets callers create an
        intermediate checkpoint without changing the hashed configuration.
        """

        started_at = time.monotonic()
        target = self.config.iterations
        if stop_after is not None:
            if stop_after < self.iteration:
                raise ValueError("stop_after precedes the restored iteration")
            target = min(target, stop_after)
        try:
            while self.iteration < target:
                within_time = not (
                    self.config.max_seconds is not None
                    and self.training_seconds >= self.config.max_seconds
                )
                if not self._all_ranks_true(within_time):
                    if checkpoint_path is not None:
                        self.save_checkpoint(checkpoint_path)
                    return self._result("timeout", started_at, error="time limit reached")

                step_started = time.monotonic()
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.train_model(self.equation, self._random_batch())
                if not self._all_ranks_true(bool(torch.isfinite(loss))):
                    return self._result("diverged", started_at, error="non-finite loss")
                loss.backward()  # type: ignore[no-untyped-call]
                gradients_finite = not any(
                    parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                    for parameter in self.model.parameters()
                )
                if not self._all_ranks_true(gradients_finite):
                    return self._result("diverged", started_at, error="non-finite gradient")
                if self.config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                self.optimizer.step()
                self.scheduler.step()
                parameters_finite = not any(
                    not bool(torch.isfinite(parameter).all())
                    for parameter in self.model.parameters()
                )
                if not self._all_ranks_true(parameters_finite):
                    return self._result("diverged", started_at, error="non-finite parameter")
                self.iteration += 1
                self.history.append(float(loss.detach().cpu()))
                self.training_seconds += time.monotonic() - step_started

                if (
                    checkpoint_path is not None
                    and self.config.checkpoint_interval > 0
                    and self.iteration % self.config.checkpoint_interval == 0
                ):
                    self.save_checkpoint(checkpoint_path)

            if checkpoint_path is not None:
                self.save_checkpoint(checkpoint_path)
            return self._result("completed", started_at)
        except (KeyboardInterrupt, SystemExit):
            if checkpoint_path is not None:
                self.save_checkpoint(checkpoint_path)
            raise
        except Exception as exc:  # Preserve every scientific failure as run data.
            return self._result("failed", started_at, error=f"{type(exc).__name__}: {exc}")


class DeepBSDESolver:
    """Small public facade matching ``Solver.run(config)`` semantics."""

    def __init__(self, equation: EquationProtocol) -> None:
        self.equation = equation

    @staticmethod
    def _from_run_config(config: RunConfig) -> tuple[DeepBSDEConfig, bool, dict[str, Any]]:
        params = dict(config.method.params)
        method_name = config.method.name.lower().replace("-", "_")
        default_variant = "m1b" if "m1b" in method_name or "shared" in method_name else "m1a"
        variant = str(params.pop("variant", default_variant)).lower()
        # Friendly aliases are accepted at this boundary; the internal config
        # remains strict and has one canonical spelling per field.
        aliases = {
            "n_steps": "steps",
            "num_steps": "steps",
            "num_iterations": "iterations",
            "lr": "learning_rate",
            "grad_clip": "gradient_clip",
        }
        for alias, canonical in aliases.items():
            if alias in params:
                if canonical in params:
                    raise ValueError(f"both {alias!r} and {canonical!r} were supplied")
                params[canonical] = params.pop(alias)
        compute_reference = bool(params.pop("compute_reference", True))
        reference_options = params.pop("reference_options", {})
        if not isinstance(reference_options, dict):
            raise TypeError("method.params.reference_options must be a mapping")
        world_size = int(params.pop("world_size", 1))
        global_batch_size_value = params.pop("global_batch_size", None)
        global_batch_size = (
            int(global_batch_size_value) if global_batch_size_value is not None else None
        )
        if global_batch_size is not None:
            if global_batch_size % world_size != 0:
                raise ValueError("global_batch_size must be divisible by world_size")
            params["batch_size"] = global_batch_size // world_size
        allowed = {
            "steps",
            "batch_size",
            "iterations",
            "learning_rate",
            "scheduler_gamma",
            "hidden_width_add",
            "gradient_clip",
            "max_seconds",
        }
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"unknown Deep BSDE method parameters: {sorted(unknown)}")
        device = str(resolve_device(config.device))
        internal = DeepBSDEConfig(
            dimension=config.equation.dim,
            seed=config.seed,
            precision=config.precision,  # type: ignore[arg-type]
            device=device,
            variant=variant,  # type: ignore[arg-type]
            world_size=world_size,
            global_batch_size=global_batch_size,
            checkpoint_interval=config.output.checkpoint_every,
            external_config_hash=config.config_hash,
            **params,
        )
        return internal, compute_reference, dict(reference_options)

    def _reference(self, options: dict[str, Any]) -> tuple[dict[str, Any] | None, float | None]:
        result = self.equation.reference_solution(**options)
        value = getattr(result, "value", None)
        if hasattr(result, "to_dict"):
            payload = result.to_dict()
        elif isinstance(result, dict):
            payload = dict(result)
        else:
            payload = {
                "value": value,
                "ci95": getattr(result, "ci95", None),
                "kind": getattr(result, "kind", "unknown"),
                "source": getattr(result, "source", "unknown"),
                "sample_count": getattr(result, "sample_count", None),
            }
        return payload, float(value) if value is not None else None

    def _run_public(self, config: RunConfig) -> RunMetrics:
        internal, compute_reference, reference_options = self._from_run_config(config)
        run_directory = prepare_run_dir(config)
        checkpoint = run_directory / "ckpt" / "last.pt"
        trainer = DeepBSDETrainer(self.equation, internal)
        if config.resume:
            trainer.load_checkpoint(checkpoint)
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)
        result = trainer.train(checkpoint_path=checkpoint)

        reference: dict[str, Any] | None = None
        reference_value: float | None = None
        reference_error: str | None = None
        if compute_reference and result.status == "completed":
            try:
                reference, reference_value = self._reference(reference_options)
            except Exception as exc:
                # The estimate is still valid when an optional external/MC
                # reference fails; retain the reason rather than fabricating an error.
                reference_error = f"reference failed: {type(exc).__name__}: {exc}"
        abs_error: float | None = None
        rel_error: float | None = None
        if result.estimate is not None and reference_value is not None:
            abs_error = abs(result.estimate - reference_value)
            if reference_value != 0:
                rel_error = abs_error / abs(reference_value)

        peak_gpu_mem_mb: float | None = None
        if trainer.device.type == "cuda":
            peak_gpu_mem_mb = torch.cuda.max_memory_allocated(trainer.device) / (1024**2)
        message_parts = [item for item in (result.error, reference_error) if item]
        return RunMetrics(
            run_id=make_run_id(config),
            method=config.method.name,
            equation=config.equation.name,
            dim=config.equation.dim,
            seed=config.seed,
            precision=config.precision,
            status=PublicRunStatus(result.status),
            config_hash=config.config_hash,
            source_commit=config.source_commit,
            estimate=result.estimate,
            abs_error=abs_error,
            rel_error=rel_error,
            reference=reference,
            wall_clock_sec=result.total_seconds,
            train_sec=result.training_seconds,
            throughput=result.throughput,
            peak_gpu_mem_mb=peak_gpu_mem_mb,
            n_params=result.parameter_count,
            iterations=result.iterations_completed,
            hardware={
                "device": result.device,
                "name": result.hardware,
                "world_size": internal.world_size,
                "rank": trainer.rank,
                "global_batch_size": internal.global_batch_size or internal.batch_size,
            },
            artifacts={"checkpoint": str(checkpoint)},
            message="; ".join(message_parts) if message_parts else None,
        )

    @overload
    def run(
        self,
        config: DeepBSDEConfig,
        *,
        checkpoint_path: str | os.PathLike[str] | None = None,
        resume: bool = False,
    ) -> TrainingResult: ...

    @overload
    def run(
        self,
        config: RunConfig,
        *,
        checkpoint_path: None = None,
        resume: bool = False,
    ) -> RunMetrics: ...

    def run(
        self,
        config: DeepBSDEConfig | RunConfig,
        *,
        checkpoint_path: str | os.PathLike[str] | None = None,
        resume: bool = False,
    ) -> TrainingResult | RunMetrics:
        if isinstance(config, RunConfig):
            if checkpoint_path is not None or resume:
                raise ValueError("public RunConfig controls resume and output paths itself")
            return self._run_public(config)
        trainer = DeepBSDETrainer(self.equation, config)
        if resume:
            if checkpoint_path is None:
                raise ValueError("resume requires checkpoint_path")
            trainer.load_checkpoint(checkpoint_path)
        return trainer.train(checkpoint_path=checkpoint_path)
