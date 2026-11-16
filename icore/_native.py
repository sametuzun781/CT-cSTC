"""Array validation and the single native ICORE entry point."""

import ctypes as ct
from functools import lru_cache
from time import perf_counter
import numpy as np
from ._build import build
from ._blasfeo_build import enable_file_checks

_ICORE = ct.POINTER(ct.c_double)
_IP = ct.POINTER(ct.c_int)


class _Model(ct.Structure):
    _fields_ = [(k, ct.c_int) for k in ("N", "nx", "nu", "nlp", "nsoc", "nf", "ne")]
    _fields_ += [("soc_sizes", ct.c_void_p)]
    _fields_ += [(k, ct.c_void_p) for k in ("H", "c", "A", "B", "b", "G", "h", "x0", "F", "f")]


class Options(ct.Structure):
    _fields_ = [(k, ct.c_int) for k in ("max_iterations", "refinement_steps", "riccati_mode")]
    _fields_ += [
        (k, ct.c_double)
        for k in ("tolerance", "step_fraction", "regularization", "refinement_tolerance")
    ]


class Info(ct.Structure):
    _fields_ = [(k, ct.c_int) for k in ("status", "iterations", "refinements", "factor_retries")]
    _fields_ += [
        (k, ct.c_double)
        for k in ("objective", "primal_residual", "dual_residual", "gap", "seconds")
    ]

    def as_dict(self):
        return {name: getattr(self, name) for name, _ in self._fields_}


def _ptr(a):
    # The C pointer representation is unchanged. Prepared.arrays/dims and
    # Native.buffers retain the owners; a ctypes cast per array would create
    # redundant pointer objects and reference cycles on every real update.
    return a.ctypes.data


def _prepare(data, nonnegative=None, soc_sizes=None, epigraph_controls=0, validate_finite=True):
    """Accept H/c directly, or the equivalent Q/R/S/q/r block representation."""
    p = {key: np.asarray(value) for key, value in data.items()}
    if p["B"].ndim != 3 or p["G"].ndim != 3:
        raise ValueError("B and G must have a stage axis and two matrix axes")
    nx, nu = p["B"].shape[-2:]
    N = p["G"].shape[0] - 1
    nv = nx + nu
    if N < 1 or nx < 1:
        raise ValueError("N and nx must be positive")

    def shape(key, expected, extra_stage=False):
        a = p[key]
        allowed = (expected, (expected[0] + 1, *expected[1:])) if extra_stage else (expected,)
        if a.shape not in allowed:
            raise ValueError(f"{key} has shape {a.shape}; expected {allowed}")
        return a[: expected[0]] if extra_stage else a

    for key, expected in (("A", (N, nx, nx)), ("B", (N, nx, nu)), ("b", (N, nx))):
        p[key] = shape(key, expected, extra_stage=True)
    p.setdefault("F", np.empty((0, nx)))
    p.setdefault("f", np.empty(0))
    if "H" not in p:
        for key, expected, extra in (
            ("R", (N, nu, nu), True),
            ("S", (N, nu, nx), True),
            ("Q", (N + 1, nx, nx), False),
            ("r", (N, nu), True),
            ("q", (N + 1, nx), False),
        ):
            p[key] = shape(key, expected, extra_stage=extra)
        # Every interior block is assigned below. Only terminal control
        # padding is absent from the block input and needs explicit zeros.
        H = np.empty((N + 1, nv, nv))
        H[N, :nu, :] = 0.0
        H[N, nu:, :nu] = 0.0
        H[:N, :nu, :nu] = p["R"][:N]
        H[:N, :nu, nu:] = p["S"][:N]
        H[:N, nu:, :nu] = p["S"][:N].transpose(0, 2, 1)
        H[:, nu:, nu:] = p["Q"][: N + 1]
        p["H"] = H
        p["c"] = np.zeros((N + 1, nv))
        p["c"][:N, :nu] = p["r"][:N]
        p["c"][:, nu:] = p["q"][: N + 1]
    if soc_sizes is None:
        soc_sizes = p.get("soc_dims", [])
    raw_dims = np.asarray(soc_sizes)
    if raw_dims.ndim != 1:
        raise ValueError("soc_sizes must be a one-dimensional integer sequence")
    if raw_dims.size:
        if not np.all(np.isfinite(raw_dims)) or not np.all(raw_dims == np.floor(raw_dims)):
            raise ValueError("soc_sizes must be a one-dimensional integer sequence")
        if np.any(raw_dims < 2) or np.any(raw_dims > np.iinfo(np.int32).max):
            raise ValueError("each SOC must have at least two entries")
    dims = np.ascontiguousarray(raw_dims, dtype=np.int32)
    soc_rows = int(dims.sum()) if dims.size else 0
    raw_nlp = (
        p.get("nonnegative", p["G"].shape[1] - soc_rows) if nonnegative is None else nonnegative
    )
    nlp = int(raw_nlp)
    if nlp != raw_nlp or nlp < 0 or nlp > np.iinfo(np.int32).max:
        raise ValueError("nonnegative must be a nonnegative integer")
    if epigraph_controls is not None and (
        int(epigraph_controls) != epigraph_controls or not 0 <= epigraph_controls <= nu // 2
    ):
        raise ValueError("epigraph_controls must be an integer between zero and nu//2")
    nf = p["F"].shape[0]
    m = nlp + soc_rows
    for key, expected in (
        ("H", (N + 1, nv, nv)),
        ("c", (N + 1, nv)),
        ("G", (N + 1, m, nv)),
        ("h", (N + 1, m)),
        ("x0", (nx,)),
        ("F", (nf, nx)),
        ("f", (nf,)),
    ):
        shape(key, expected)
    arrays = {
        k: np.ascontiguousarray(p[k], dtype=np.float64)
        for k in ("H", "c", "A", "B", "b", "G", "h", "x0", "F", "f")
    }
    if validate_finite and not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Model coefficients must be finite")
    if epigraph_controls is None:
        from .presolve import infer_epigraph_controls

        epigraph_controls = infer_epigraph_controls(arrays, nlp)
    nf = arrays["F"].shape[0]
    model = _Model(
        N,
        nx,
        nu,
        nlp,
        len(dims),
        nf,
        int(epigraph_controls),
        dims.ctypes.data,
        *(_ptr(arrays[k]) for k in ("H", "c", "A", "B", "b", "G", "h", "x0", "F", "f")),
    )
    return model, arrays, dims


CHAIN_FIELDS = (
    "left",
    "right",
    "offset",
    "pair_left",
    "pair_right",
    "pair_offset",
    "local_matrix",
    "local_offset",
    "center",
    "hessian",
    "linear",
    "l1_weight",
    "free",
    "fixed",
)


class _Chain(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in CHAIN_FIELDS]


class _Input(ct.Structure):
    _fields_ = [("format", ct.c_int), ("data", ct.c_void_p)]
    _fields_ += [(name, ct.c_int) for name in ("K", "n", "d", "p", "l")]


class _Output(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in ("v", "y", "nu", "t", "s", "z")]


class Prepared:
    """Validated array buffers plus their C layout; no solver selection."""

    def __init__(
        self, data, *, nonnegative=None, soc_sizes=None, epigraph_controls=0, validate_finite=True
    ):
        self.chain = all(name in data for name in CHAIN_FIELDS)
        self._owned_hc = not self.chain and "H" not in data
        if self.chain:
            self._finite_checked = True
            K, n = np.shape(data["center"])
            d, p, l = (
                np.shape(data[name])[1] for name in ("offset", "pair_offset", "local_offset")
            )
            if K < 1 or n < 1:
                raise ValueError("Chain stage count and width must be positive")
            shapes = (
                (K - 1, d, n),
                (K - 1, d, n),
                (K - 1, d),
                (K - 1, p, n),
                (K - 1, p, n),
                (K - 1, p),
                (K, l, n),
                (K, l),
                (K, n),
                (K, n),
                (K, n),
                (K - 1, d),
                (K, n),
                (K, n),
            )
            arrays = {}
            for name, shape in zip(CHAIN_FIELDS, shapes):
                value = np.asarray(data[name], dtype=np.float64)
                if name == "l1_weight":
                    value = np.broadcast_to(value, shape)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"{name} must be finite with shape {shape}")
                arrays[name] = np.ascontiguousarray(value)
            if np.any(arrays["hessian"] <= 0) or np.any(arrays["l1_weight"] <= 0):
                raise ValueError("Chain curvature and L1 weights must be positive")
            if not np.all((arrays["free"] == 0) | (arrays["free"] == 1)):
                raise ValueError("free must contain only zero and one")
            self.arrays, self.dims = arrays, None
            self.layout = _Chain(*(_ptr(arrays[name]) for name in CHAIN_FIELDS))
            self.model = _Input(1, ct.addressof(self.layout), K, n, d, p, l)
            self.signature = (1, K, n, d, p, l)
            count = 2 * (K - 1) * d + (K - 1) * p + K * l
            self.shapes = ((K, n), (0,), (0,), (K - 1, d), (count,), (count,))
            self.cone_shapes = ((K - 1, d), (K - 1, d), (K - 1, p), (K, l))
            nd, npair = (K - 1) * d, (K - 1) * p
            self.cuts = (nd, 2 * nd, 2 * nd + npair)
        else:
            self.layout, self.arrays, self.dims = _prepare(
                data, nonnegative, soc_sizes, epigraph_controls, validate_finite
            )
            self._finite_checked = validate_finite
            m = self.layout
            self.model = _Input(0, ct.addressof(m), 0, 0, 0, 0, 0)
            self.signature = (0, m.N, m.nx, m.nu, m.nlp, tuple(self.dims), m.nf, m.ne)
            rows = m.nlp + (int(self.dims.sum()) if m.nsoc else 0)
            self.shapes = (
                (m.N + 1, m.nx + m.nu),
                (m.N, m.nx),
                (m.nf,),
                (0,),
                (m.N + 1, rows),
                (m.N + 1, rows),
            )

    @classmethod
    def refresh(cls, data, previous, spare=None, *, nonnegative=None, soc_sizes=None,
                epigraph_controls=0):
        """Refresh an unchanged orthant layout, reading every current coefficient.

        Shape/type/cone changes use the general validator. Only an inactive
        bank's owned H/c conversion buffers may be recycled; the
        currently committed model is never modified before native acceptance.
        """
        if previous.chain or previous.layout.nsoc:
            return None
        if any(name in data for name in CHAIN_FIELDS):
            return None
        cone_dims = data.get("soc_dims", ()) if soc_sizes is None else soc_sizes
        if not isinstance(cone_dims, (list, tuple, np.ndarray)) or np.shape(cone_dims) != (0,):
            return None
        m = previous.layout
        N, nx, nu, nf = m.N, m.nx, m.nu, m.nf
        nv, rows = nx + nu, m.nlp
        raw_nlp = data.get("nonnegative", rows) if nonnegative is None else nonnegative
        if not isinstance(raw_nlp, (int, float, np.integer, np.floating)) or raw_nlp != rows:
            return None
        if epigraph_controls is not None and (
            not isinstance(epigraph_controls, (int, float, np.integer, np.floating)) or
            not np.isfinite(epigraph_controls) or int(epigraph_controls) != epigraph_controls or
            not 0 <= epigraph_controls <= nu // 2
        ):
            return None
        arrays = {}
        for key, shape in (("A", (N,nx,nx)), ("B", (N,nx,nu)), ("b", (N,nx)),
                           ("G", (N+1,rows,nv)), ("h", (N+1,rows)), ("x0", (nx,)),
                           ("F", (nf,nx)), ("f", (nf,))):
            a = data.get(key)
            if a is None and key in ("F", "f") and nf == 0 and key not in data:
                a = previous.arrays[key]
            if type(a) is not np.ndarray or a.dtype != np.float64 or not a.flags.c_contiguous:
                return None
            if a.shape != shape:
                if key not in ("A", "B", "b") or a.shape != (N+1, *shape[1:]):
                    return None
                a = a[:N]
            arrays[key] = a
        if "H" in data:
            for key, shape in (("H", (N+1,nv,nv)), ("c", (N+1,nv))):
                a = data.get(key)
                if type(a) is not np.ndarray or a.dtype != np.float64 or a.shape != shape or not a.flags.c_contiguous:
                    return None
                arrays[key] = a
            owned_hc = False
        else:
            blocks = {}
            for key, shape in (("R", (N,nu,nu)), ("S", (N,nu,nx)), ("Q", (N+1,nx,nx)),
                               ("r", (N,nu)), ("q", (N+1,nx))):
                a = data.get(key)
                if type(a) is not np.ndarray or a.dtype != np.float64:
                    return None
                if a.shape != shape:
                    if key not in ("R", "S", "r") or a.shape != (N+1, *shape[1:]):
                        return None
                    a = a[:N]
                blocks[key] = a
            # Borrow only the inactive conversion bank, never user-owned H/c.
            reusable = spare is not None and spare[0].shape == (N+1,nv,nv)
            H = spare[0] if reusable else np.empty((N+1,nv,nv))
            c = spare[1] if reusable else np.empty((N+1,nv))
            H[N,:nu,:] = 0.0
            H[N,nu:,:nu] = 0.0
            H[:N,:nu,:nu] = blocks["R"]
            H[:N,:nu,nu:] = blocks["S"]
            H[:N,nu:,:nu] = blocks["S"].transpose(0,2,1)
            H[:,nu:,nu:] = blocks["Q"]
            c[N,:nu] = 0.0
            c[:N,:nu] = blocks["r"]
            c[:,nu:] = blocks["q"]
            arrays["H"], arrays["c"] = H, c
            owned_hc = True
        if epigraph_controls is None:
            from .presolve import infer_epigraph_controls
            epigraph_controls = infer_epigraph_controls(arrays, rows)
        result = cls.__new__(cls)
        result.chain, result._finite_checked, result._owned_hc = False, False, owned_hc
        result.arrays, result.dims = arrays, previous.dims
        result.layout = _Model(N,nx,nu,rows,0,nf,int(epigraph_controls),result.dims.ctypes.data,
            *(_ptr(arrays[k]) for k in ("H","c","A","B","b","G","h","x0","F","f")))
        result.model = _Input(0,ct.addressof(result.layout),0,0,0,0,0)
        result.signature = (0,N,nx,nu,rows,(),nf,int(epigraph_controls))
        result.shapes = previous.shapes
        return result

    def ensure_finite(self):
        """Validate before any reduction that could discard input entries."""
        if not self._finite_checked:
            if not all(np.isfinite(a).all() for a in self.arrays.values()):
                raise ValueError("Model coefficients must be finite")
            self._finite_checked = True


@lru_cache(maxsize=None)
def _load_library(path):
    """Bind the ABI once; every solver still owns its separate workspace."""
    lib = ct.CDLL(path)
    enable_file_checks(lib)
    lib.icore_default_options.restype = Options
    lib.icore_create.argtypes = [ct.POINTER(_Input)]
    lib.icore_create.restype = ct.c_void_p
    lib.icore_destroy.argtypes = [ct.c_void_p]
    lib.icore_destroy.restype = None
    lib.icore_update.argtypes = [ct.c_void_p, ct.POINTER(_Input)]
    lib.icore_update.restype = ct.c_int
    lib.icore_set_step_policy.argtypes = [ct.c_void_p, ct.c_int]
    lib.icore_set_step_policy.restype = ct.c_int
    lib.icore_solve.argtypes = [
        ct.c_void_p,
        ct.POINTER(Options),
        ct.POINTER(_Output),
        ct.POINTER(Info),
    ]
    lib.icore_solve.restype = ct.c_int
    for name in ("last_kernel", "dense_face_attempts", "dense_face_steps", "dense_face_accepted", "dense_fallbacks", "dense_packed", "ocp_packed_stages",
                 "ocp_packed_rejected", "ocp_sqrt_factors", "ocp_sqrt_rejected"):
        function = getattr(lib, "icore_" + name)
        function.argtypes = [ct.c_void_p]
        function.restype = ct.c_int
    return lib


class Native:
    """One native library and icore_solve call for either input layout."""

    def __init__(self, prepared, backend, settings):
        self._handle = None
        self.lib = lib = _load_library(build(backend))
        self.options = lib.icore_default_options()
        self.step_policy = 1
        self.configure(settings)
        self.info = Info()
        self.prepared = None
        self.update(prepared)

    def configure(self, settings):
        policy = settings["step_policy"]
        if self._handle and policy != self.step_policy:
            status = self.lib.icore_set_step_policy(self._handle, policy)
            if status:
                raise ValueError(f"Step policy rejected (status {status})")
        self.step_policy = policy
        for key, value in settings.items():
            if key != "step_policy":
                setattr(self.options, key, value)

    def update(self, prepared):
        if self.prepared is None or prepared.signature != self.prepared.signature:
            handle = self.lib.icore_create(ct.byref(prepared.model))
            if not handle:
                prepared.ensure_finite()
                raise ValueError("Invalid convex model or workspace allocation failed")
            try:
                status = self.lib.icore_set_step_policy(handle, self.step_policy)
                if status:
                    raise ValueError(f"Step policy rejected (status {status})")
                buffers = tuple(np.empty(shape) for shape in prepared.shapes)
                output = _Output(*(_ptr(value) for value in buffers))
            except BaseException:
                self.lib.icore_destroy(handle)
                raise
            old = self._handle
            self._handle = handle
            self.buffers, self.output = buffers, output
            if old:
                self.lib.icore_destroy(old)
        else:
            status = self.lib.icore_update(self._handle, ct.byref(prepared.model))
            if status:
                prepared.ensure_finite()
                raise ValueError(f"Coefficient update rejected (status {status})")
        self.prepared = prepared

    def solve(self):
        if not self._handle:
            raise RuntimeError("Solver is closed")
        started = perf_counter()
        status = self.lib.icore_solve(
            self._handle, ct.byref(self.options), ct.byref(self.output), ct.byref(self.info)
        )
        elapsed = perf_counter() - started
        info = self.info.as_dict()
        result = dict(
            status=int(status),
            iterations=info["iterations"],
            info=info,
            objective=info["objective"],
            native_seconds=elapsed,
            residuals=np.array(
                [info[name] for name in ("primal_residual", "dual_residual", "gap")]
            ),
            refinements=info["refinements"],
            factor_retries=info["factor_retries"],
        )
        v, y, nu, t, s, z = self.buffers
        p = self.prepared
        if p.chain:
            split = lambda a: tuple(
                value.reshape(shape)
                for value, shape in zip(np.split(a.copy(), p.cuts), p.cone_shapes)
            )
            result.update(
                stage_y=v.copy(),
                epigraph=t.copy(),
                s=split(s),
                z=split(z),
                qr_fallbacks=info["factor_retries"],
            )
        else:
            n = p.layout.nu
            result.update(
                x=v[:, n:].copy(),
                u=v[:-1, :n].copy(),
                y=y.copy(),
                nu=nu.copy(),
                s=s.copy(),
                z=z.copy(),
            )
        return result

    def diagnostics(self):
        """Read the last solve's actual matrix path without solving again."""
        if not self._handle:
            raise RuntimeError("Solver is closed")
        return {name: int(getattr(self.lib, "icore_" + name)(self._handle))
                for name in ("last_kernel", "dense_face_attempts", "dense_face_steps", "dense_face_accepted", "dense_fallbacks", "dense_packed", "ocp_packed_stages",
                             "ocp_packed_rejected", "ocp_sqrt_factors", "ocp_sqrt_rejected")}

    def close(self):
        if getattr(self, "_handle", None):
            self.lib.icore_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()
