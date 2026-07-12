"""Deep BSDE solvers (M1a stepwise and M1b shared-network variants)."""

from .config import DeepBSDEConfig, Precision, Variant, configure_precision
from .model import DeepBSDEModel, EquationProtocol, SharedZNetwork, StepwiseZNetwork
from .trainer import DeepBSDESolver, DeepBSDETrainer, RunStatus, TrainingResult

__all__ = [
    "DeepBSDEConfig",
    "DeepBSDEModel",
    "DeepBSDESolver",
    "DeepBSDETrainer",
    "EquationProtocol",
    "Precision",
    "RunStatus",
    "SharedZNetwork",
    "StepwiseZNetwork",
    "TrainingResult",
    "Variant",
    "configure_precision",
]
