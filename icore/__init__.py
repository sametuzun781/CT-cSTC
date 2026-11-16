"""ICORE: one primal-dual IPM with automatic structure reduction."""

from .api import Solver
from .model import ChainQP

__all__ = ["Solver", "ChainQP"]
__version__ = "0.20.0"
