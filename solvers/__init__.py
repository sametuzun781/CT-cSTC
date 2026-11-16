"""Five selected solve paths for the trajectory optimization examples."""
__all__ = ["ChainSolver"]


def __getattr__(name):
    if name == "ChainSolver":
        from .engine import ChainSolver
        return ChainSolver
    raise AttributeError(name)


try:
    import cvxpy as cp
except ImportError as _import_error:
    class _MissingCVXPY:
        def __getattr__(self, name):
            raise ImportError(
                "The cvxpy_qoco backend requires CVXPY and QOCO. "
                "Install requirements-comparators.txt or select icore_cpp/icore_c."
            )
    cp = _MissingCVXPY()
