"""Direct calls to the official QOCO C API through a small ctypes bridge."""
from __future__ import annotations

import ctypes as ct
from pathlib import Path

import numpy as np
from scipy import sparse

QOCO_VERSION = "0.3.2"

def library_path():
    from ..._comparators import build_comparators
    return build_comparators()["qoco"]

_ICORE = ct.POINTER(ct.c_double)
_IP = ct.POINTER(ct.c_int)


def _icore(a):
    return a.ctypes.data_as(_ICORE)


def _ip(a):
    return a.ctypes.data_as(_IP)


def _csc(a):
    a = sparse.csc_matrix(a, dtype=np.float64, copy=True)
    a.sum_duplicates()
    a.sort_indices()
    if max(*a.shape, a.nnz) > np.iinfo(np.int32).max:
        raise ValueError("QOCO uses 32-bit sparse indices in this benchmark.")
    a.indices = np.ascontiguousarray(a.indices, dtype=np.int32)
    a.indptr = np.ascontiguousarray(a.indptr, dtype=np.int32)
    return a


class NativeQoco:
    """Reusable C workspace; each solve starts from QOCO's default initialization.

    The public Solver wrapper builds its library during preparation. Output
    arrays are independently owned. native_seconds covers qoco_solve, including
    final unscaling, but not output copies; Python wall time is measured separately.
    """

    def __init__(self, canonical, *, tolerance=1e-9, max_iterations=120, library=None):
        path = Path(library) if library else library_path()
        if not path.is_file():
            raise FileNotFoundError("Build the QOCO library before creating its C workspace.")
        self.lib = lib = ct.CDLL(str(path))
        self.library = str(path)
        self.tolerance, self.max_iterations = tolerance, max_iterations
        self.handle = None
        lib.benchmark_qoco_create.argtypes = ([ct.c_int] * 4 + [_IP, _IP, _ICORE, _ICORE]
            + [ct.c_int, _IP, _IP, _ICORE, _ICORE] * 2
            + [ct.c_int, ct.c_int, _IP, ct.c_double, ct.c_int, _IP])
        lib.benchmark_qoco_create.restype = ct.c_void_p
        lib.benchmark_qoco_solve.argtypes = [ct.c_void_p] + [_ICORE] * 5 + [_IP]
        lib.benchmark_qoco_solve.restype = None
        lib.benchmark_qoco_destroy.argtypes = [ct.c_void_p]
        lib.benchmark_qoco_destroy.restype = None
        lib.benchmark_qoco_update.argtypes = [ct.c_void_p] + [_ICORE] * 6
        lib.benchmark_qoco_update.restype = None
        self.P, self.A, self.G = (_csc(a) for a in
            (sparse.triu(canonical.P, format="csc"), canonical.A, canonical.G))
        self.c, self.b, self.h = (np.array(a, dtype=np.float64, order="C", copy=True)
                                 for a in (canonical.c, canonical.b, canonical.h))
        self.cones = np.array(canonical.cone_sizes, dtype=np.int32)
        self.n, self.p, self.m = len(self.c), len(self.b), len(self.h)
        if self.P.shape != (self.n, self.n) or self.A.shape != (self.p, self.n) or self.G.shape != (self.m, self.n):
            raise ValueError("Canonical matrix and vector shapes disagree.")
        if not all(np.isfinite(a).all() for a in
                   (self.P.data, self.A.data, self.G.data, self.c, self.b, self.h)):
            raise ValueError("QOCO input data must be finite.")
        if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
            raise ValueError("Tolerance and maximum iterations must be positive.")
        error = ct.c_int()
        args = [self.n, self.m, self.p]
        for a, vector in ((self.P, self.c), (self.A, self.b), (self.G, self.h)):
            args.extend((a.nnz, _ip(a.indptr), _ip(a.indices), _icore(a.data), _icore(vector)))
        args.extend((int(canonical.l), len(self.cones), _ip(self.cones),
                     float(tolerance), int(max_iterations), ct.byref(error)))
        self.handle = lib.benchmark_qoco_create(*args)
        if not self.handle:
            raise RuntimeError(f"QOCO C setup failed with error {error.value}.")
        self.version = QOCO_VERSION

    def update(self, canonical):
        matrices = [_csc(a) for a in (sparse.triu(canonical.P, format="csc"), canonical.A, canonical.G)]
        same = all(a.shape == b.shape and np.array_equal(a.indptr, b.indptr)
                   and np.array_equal(a.indices, b.indices)
                   for a, b in zip(matrices, (self.P, self.A, self.G)))
        same = same and np.array_equal(canonical.cone_sizes, self.cones)
        if not same:
            self.close()
            self.__init__(canonical, tolerance=self.tolerance,
                          max_iterations=self.max_iterations, library=self.library)
            return
        self.P, self.A, self.G = matrices
        self.c, self.b, self.h = (np.ascontiguousarray(a, dtype=np.float64)
                                 for a in (canonical.c, canonical.b, canonical.h))
        if not all(np.isfinite(a).all() for a in
                   (self.P.data, self.A.data, self.G.data, self.c, self.b, self.h)):
            raise ValueError("QOCO input data must be finite.")
        null = ct.cast(None, _ICORE)
        self.lib.benchmark_qoco_update(self.handle, _icore(self.P.data), _icore(self.c),
            _icore(self.A.data) if self.p else null, _icore(self.b) if self.p else null,
            _icore(self.G.data), _icore(self.h))

    def solve(self):
        if not self.handle:
            raise RuntimeError("QOCO workspace is closed.")
        x, y, s, z = (np.empty(size) for size in (self.n, self.p, self.m, self.m))
        timing, info = np.empty(2), np.empty(4, dtype=np.int32)
        self.lib.benchmark_qoco_solve(self.handle, *map(_icore, (x, y, s, z, timing)), _ip(info))
        return dict(x=x, y=y, s=s, z=z, status=int(info[0]), status_ok=bool(info[0] == 1),
                    iterations=int(info[1]), refinements=int(info[2]),
                    native_seconds=float(timing[0]), reported_seconds=float(timing[1]),
                    native_return=int(info[3]))

    def close(self):
        if getattr(self, "handle", None):
            self.lib.benchmark_qoco_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
