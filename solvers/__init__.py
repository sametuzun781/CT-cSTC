"""Five selected solve paths for the trajectory optimization examples."""
__all__ = ["ChainSolver"]


def __getattr__(name):
    if name == "ChainSolver":
        from .engine import ChainSolver
        return ChainSolver
    raise AttributeError(name)
