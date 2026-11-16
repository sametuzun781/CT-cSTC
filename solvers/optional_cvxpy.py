"""Load CVXPY only when the optional CVXPY modeling path is used."""
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
