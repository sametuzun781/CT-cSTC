"""Pinned, native HPIPM QP baselines with original-coordinate solution outputs.

All numerical solves, matrix packing, updates and solver timers live in bridge.c.
There is no Python optimizer, no feasibility repair, and no result-selected retry.
Generic second-order cones are explicitly outside this QP adapter's domain.
"""
from __future__ import annotations

import ctypes as ct
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from collections.abc import Mapping
import numpy as np

HERE = Path(__file__).resolve().parent
OCP_KEYS = ("Q", "R", "S", "q", "r", "A", "B", "b", "G", "h", "x0", "F", "f")
PROFILES = {"BALANCE": 2, "ROBUST": 3}
STATUS = {0: "SUCCESS", 1: "MAX_ITER", 2: "MIN_STEP", 3: "NAN_SOL",
          4: "INCONS_EQ", 5: "FAILED_REG", 6: "IN_PROGRESS"}
DP = ct.POINTER(ct.c_double)
IP = ct.POINTER(ct.c_int)


class UnsupportedProblem(ValueError):
    """The mathematical model is outside this adapter's declared support."""


class _OCPData(ct.Structure):
    _fields_ = [(k, ct.c_int) for k in ("N", "nx", "nu", "m", "stages", "nf")]
    _fields_ += [(k, DP) for k in OCP_KEYS]


class _DenseData(ct.Structure):
    _fields_ = [(k, ct.c_int) for k in ("n", "m", "ne")]
    _fields_ += [(k, DP) for k in ("P", "q", "G", "h", "A", "b")]


class _Metrics(ct.Structure):
    _fields_ = [(k, ct.c_int) for k in (
        "status", "iterations", "stat_m", "native_bound_count", "actual_ric_alg",
        "actual_lq_fact", "npd_hess", "npd_reg_hess")]
    _fields_ += [(k, ct.c_int) for k in ("box_variables", "box_constraint_rows",
                                       "general_constraint_rows", "terminal_equality_rows",
                                       "inactive_constraint_rows")]
    _fields_ += [("memory_bytes", ct.c_uint64)]
    _fields_ += [(k, ct.c_double) for k in (
        "setup_s", "update_s", "solve_s", "extract_s", "residual_stat", "residual_eq",
        "residual_ineq", "residual_comp", "dual_gap", "objective", "tolerance", "tol_comp")]


def _library():
    from ..._comparators import build_comparators
    return _load_library(str(build_comparators()["hpipm"]))


@lru_cache(maxsize=None)
def _load_library(path):
    """Keep each ABI binding with its exact build, including live old solvers."""
    lib = ct.CDLL(path)
    for kind, data_type, outputs in (("ocp", _OCPData, 6), ("dense", _DenseData, 4)):
        create = getattr(lib, f"hp_{kind}_create")
        create.argtypes = [ct.POINTER(data_type), ct.c_int, ct.c_double, ct.c_int]
        create.restype = ct.c_void_p
        update = getattr(lib, f"hp_{kind}_update")
        update.argtypes = [ct.c_void_p, ct.POINTER(data_type)]
        update.restype = ct.c_int
        solve = getattr(lib, f"hp_{kind}_solve")
        solve.argtypes = [ct.c_void_p] + [DP] * outputs + [ct.POINTER(_Metrics)]
        solve.restype = ct.c_int
        destroy = getattr(lib, f"hp_{kind}_destroy")
        destroy.argtypes = [ct.c_void_p]
        destroy.restype = None
        settings = getattr(lib, f"hp_{kind}_settings")
        settings.argtypes = [ct.c_void_p, DP, IP]
        settings.restype = None
        history = getattr(lib, f"hp_{kind}_history")
        history.argtypes = [ct.c_void_p, DP, ct.c_int]
        history.restype = ct.c_int
        metrics = getattr(lib, f"hp_{kind}_metrics")
        metrics.argtypes = [ct.c_void_p, ct.POINTER(_Metrics)]
        metrics.restype = None
        budget = getattr(lib, f"hp_{kind}_complementarity_budget")
        budget.argtypes = [ct.c_void_p, ct.c_int]
        budget.restype = ct.c_int
    return lib


def _ptr(a):
    return a.ctypes.data_as(DP)


def _array(a):
    if hasattr(a, "toarray"):
        a = a.toarray()
    value = np.ascontiguousarray(a, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("HPIPM input must be finite; encode absent bounds by omitting rows")
    return value


def _check_shape(p, name, shape):
    if p[name].shape != shape:
        raise ValueError(f"{name}: expected {shape}, got {p[name].shape}")


def _ocp_data(problem):
    if tuple(problem.get("soc_sizes", ())) or tuple(problem.get("soc_dims", ())):
        raise UnsupportedProblem("Generic SOC rows are outside the native OCP-QP adapter")
    p = {key: _array(problem[key]) for key in OCP_KEYS}
    if p["Q"].ndim != 3 or p["R"].ndim != 3 or p["h"].ndim != 2:
        raise ValueError("Q/R must have stage axes and h must have shape (stages, constraints)")
    N, nx, nu = len(p["Q"]) - 1, p["Q"].shape[1], p["R"].shape[1]
    stages, m = p["h"].shape
    nf = len(p["f"])
    if N < 1 or nx < 1 or nu < 1 or stages not in (N, N + 1):
        raise UnsupportedProblem("OCP adapter requires N,nx,nu>=1 and N or N+1 constraint stages")
    _check_shape(p, "Q", (N + 1, nx, nx))
    _check_shape(p, "q", (N + 1, nx))
    for key, shape in (("R", (nu, nu)), ("S", (nu, nx)), ("r", (nu,)),
                       ("A", (nx, nx)), ("B", (nx, nu)), ("b", (nx,))):
        if len(p[key]) not in (N, N + 1) or p[key].shape[1:] != shape:
            raise ValueError(f"Unexpected {key} shape: {p[key].shape}")
    _check_shape(p, "G", (stages, m, nu + nx))
    _check_shape(p, "x0", (nx,))
    _check_shape(p, "F", (nf, nx))
    _check_shape(p, "f", (nf,))
    shape = (N, nx, nu, m, stages, nf)
    return p, _OCPData(*shape, *(_ptr(p[k]) for k in OCP_KEYS)), shape


def _dense_data(problem):
    if not isinstance(problem, Mapping):
        if tuple(getattr(problem, "soc_sizes", ())):
            raise UnsupportedProblem("Native dense HPIPM QP does not accept generic SOC constraints")
        problem = {k: getattr(problem, k) for k in ("P", "c", "A", "b", "G", "h")}
    if tuple(problem.get("soc_sizes", ())):
        raise UnsupportedProblem("Native dense HPIPM QP does not accept generic SOC constraints")
    P = _array(problem["P"])
    n = P.shape[0]
    q = _array(problem["q"] if "q" in problem else problem["c"])
    A = _array(problem.get("A", np.empty((0, n))))
    G = _array(problem.get("G", np.empty((0, n))))
    p = dict(P=P, q=q, A=A, b=_array(problem.get("b", np.empty(0))),
             G=G, h=_array(problem.get("h", np.empty(0))))
    ne, m = len(p["b"]), len(p["h"])
    if "nonnegative" in problem and int(problem["nonnegative"]) != m:
        raise UnsupportedProblem("All dense inequality rows must be scalar nonnegative-cone rows")
    for key, shape in (("P", (n, n)), ("q", (n,)), ("A", (ne, n)),
                       ("b", (ne,)), ("G", (m, n)), ("h", (m,))):
        _check_shape(p, key, shape)
    return p, _DenseData(n, m, ne, *(_ptr(p[k]) for k in ("P", "q", "G", "h", "A", "b"))), (n, m, ne)


class _Base:
    def _init_native(self, profile, tolerance, max_iter, complementarity_budget):
        self.profile = str(profile).upper()
        if self.profile not in PROFILES:
            raise ValueError("Choose one globally declared profile: BALANCE or ROBUST")
        self.tolerance = float(tolerance)
        self.max_iter = int(max_iter)
        self.complementarity_budget = str(complementarity_budget)
        if self.complementarity_budget not in ("count", "component"):
            raise ValueError("complementarity_budget must be 'count' or 'component'")
        self._lib = _library()
        self.library = self._lib._name
        self._handle = getattr(self._lib, f"hp_{self.kind}_create")(
            ct.byref(self._data), PROFILES[self.profile], self.tolerance, self.max_iter)
        if not self._handle:
            raise RuntimeError("HPIPM native workspace allocation/model initialization failed")
        code = getattr(self._lib, f"hp_{self.kind}_complementarity_budget")(
            self._handle, int(self.complementarity_budget == "component"))
        if code:
            self.close()
            raise RuntimeError("HPIPM complementarity profile configuration failed")
        self.settings = self._settings()
        self.last_native_update_seconds = self.native_metrics()["update_s"]
        self.last_conversion_seconds = self.python_prepare_s

    def _settings(self):
        d, i = np.empty(15), np.empty(21, dtype=np.int32)
        getattr(self._lib, f"hp_{self.kind}_settings")(self._handle, _ptr(d), i.ctypes.data_as(IP))
        dk = ("tol_stat", "tol_eq", "tol_ineq", "tol_comp", "tol_dual_gap", "mu0",
              "alpha_min", "reg_prim", "reg_dual", "lam_min", "t_min", "tau_min",
              "lam0_min", "t0_min", "m_safe")
        ik = ("mode", "iter_max", "stat_max", "warm_start", "ric_alg", "lq_fact",
              "itref_pred_max", "itref_corr_max", "pred_corr", "cond_pred_corr",
              "abs_form", "comp_dual_sol_eq", "comp_res_exit", "comp_res_pred", "split_step",
              "var_init_scheme", "t_lam_min", "t0_init", "update_fact_exit", "kkt_fact_alg",
              "remove_lin_dep_eq")
        counts = self.native_metrics()
        return {**dict(zip(dk, map(float, d))), **dict(zip(ik, map(int, i))),
                "profile": self.profile, "native_form": self.kind,
                "native_box_variables": counts["box_variables"],
                "native_box_constraint_rows": counts["box_constraint_rows"],
                "native_general_constraint_rows": counts["general_constraint_rows"],
                "native_equality_rows": counts["terminal_equality_rows"],
                "inactive_constant_rows": counts["inactive_constraint_rows"],
                "constant_row_elimination": "exact zero active coefficients after x0 substitution and nonnegative RHS; restored s=RHS,z=0",
                "bound_extraction": "exact single-variable scalar rows; duplicate same-side bounds stay general",
                "bound_inverse_map": "s_original=abs(a)*t_native; z_original=lambda_native/abs(a)",
                "complementarity_budget": self.complementarity_budget,
                "tolerance_rule": ("tol_comp = tolerance / max(1, active native scalar bounds)"
                    if self.complementarity_budget == "count" else
                    "tol_comp = tolerance; tol_dual_gap = tolerance; independent total-gap audit authoritative"),
                "source_revision": "16a8db59f595bda642f460aa0f62e308f4b76db3"}

    def _check_open(self):
        if not getattr(self, "_handle", None):
            raise RuntimeError("HPIPM workspace is closed")

    def native_metrics(self):
        self._check_open()
        metrics = _Metrics()
        getattr(self._lib, f"hp_{self.kind}_metrics")(self._handle, ct.byref(metrics))
        return {key: getattr(metrics, key) for key, _ in _Metrics._fields_}

    def close(self):
        if getattr(self, "_handle", None):
            getattr(self._lib, f"hp_{self.kind}_destroy")(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    def update(self, problem):
        """Repack all coefficients into the same dimensions; every solve stays cold.

        Dimensions and extracted box/constant-row patterns must remain valid.
        Invalid pattern changes are rejected before native model mutation.
        No numeric factor is reused across independent IPMs.
        """
        self._check_open()
        start = perf_counter()
        p, data, shape = self._prepare(problem)
        if shape != self.shape:
            raise ValueError("Same-pattern update requires unchanged stage/model dimensions")
        prep_s = perf_counter() - start
        code = getattr(self._lib, f"hp_{self.kind}_update")(self._handle, ct.byref(data))
        if code == -2:
            raise ValueError("Update changes an extracted bound's variable or side; recreate the native workspace")
        if code == -3:
            raise ValueError("Update makes an eliminated constant row nonconstant or infeasible; recreate the native workspace")
        if code:
            raise RuntimeError(f"HPIPM update returned {code}")
        self.problem, self._data = p, data
        self.python_prepare_s = prep_s
        self.last_conversion_seconds = prep_s
        self.last_native_update_seconds = self.native_metrics()["update_s"]

    def _solve_outputs(self, outputs, return_history):
        self._check_open()
        metrics = _Metrics()
        getattr(self._lib, f"hp_{self.kind}_solve")(
            self._handle, *(_ptr(a) for a in outputs.values()), ct.byref(metrics))
        native = {key: getattr(metrics, key) for key, _ in _Metrics._fields_}
        result = {**outputs, "status": native["status"], "status_text": STATUS.get(native["status"], "UNKNOWN"),
                  "iterations": native["iterations"], "native": native, "settings": self.settings.copy(),
                  "python_prepare_s": self.python_prepare_s,
                  "native_cold_total_s": native["setup_s"] + native["solve_s"] + native["extract_s"],
                  "native_update_total_s": native["update_s"] + native["solve_s"] + native["extract_s"]}
        if return_history:
            shape = (native["iterations"] + 1, native["stat_m"])
            trace = np.empty(shape)
            getattr(self._lib, f"hp_{self.kind}_history")(self._handle, _ptr(trace), trace.size)
            result["native_history"] = trace
        return result


class HPIPMOCP(_Base):
    """Native OCP-QP for the existing 13-array fixed-initial-state problem.

    Terminal F*x=f is exact equal lower/upper general bounds, with recovered
    free multiplier lam_upper-lam_lower. No conic or quadratic approximation.
    """
    kind = "ocp"
    _prepare = staticmethod(_ocp_data)

    def __init__(self, problem, nonnegative=None, soc_sizes=(), *, profile="BALANCE",
                 tolerance=1e-8, max_iter=200, complementarity_budget="count"):
        self._handle = None
        if tuple(soc_sizes):
            raise UnsupportedProblem(
                "HPIPM OCP-QP cannot represent generic second-order cones. "
                "QCQP is not a general SOCP interface; affine-head cones are not squared here.")
        start = perf_counter()
        self.problem, self._data, self.shape = self._prepare(problem)
        if nonnegative is not None and int(nonnegative) != self.shape[3]:
            raise UnsupportedProblem("All OCP constraint rows must be scalar nonnegative-cone rows")
        self.python_prepare_s = perf_counter() - start
        self._init_native(profile, tolerance, max_iter, complementarity_budget)

    def solve(self, *, return_history=False):
        N, nx, nu, m, stages, nf = self.shape
        out = dict(x=np.empty((N + 1, nx)), u=np.empty((N, nu)), y=np.empty((N, nx)),
                   nu=np.empty(nf), s=np.empty((stages, m)), z=np.empty((stages, m)))
        return self._solve_outputs(out, return_history)


class HPIPMDense(_Base):
    """Native dense QP: .5*v.T@P@v + q.T@v, A@v=b, G@v<=h.

    The result's y satisfies P*v+q+A.T*y+G.T*z=0. Preserves native slack s.
    Intended for already-condensed public QPs, with that input track disclosed.
    """
    kind = "dense"
    _prepare = staticmethod(_dense_data)

    def __init__(self, problem, *, profile="BALANCE", tolerance=1e-8, max_iter=200,
                 complementarity_budget="count"):
        self._handle = None
        start = perf_counter()
        self.problem, self._data, self.shape = self._prepare(problem)
        self.python_prepare_s = perf_counter() - start
        self._init_native(profile, tolerance, max_iter, complementarity_budget)

    def solve(self, *, return_history=False):
        n, m, ne = self.shape
        out = dict(v=np.empty(n), y=np.empty(ne), s=np.empty(m), z=np.empty(m))
        return self._solve_outputs(out, return_history)


def dense_audit(problem, result):
    """Small original-data audit for adapter smoke tests; main harness audits independently."""
    p, _, _ = _dense_data(problem)
    v, y, s, z = (result[k] for k in ("v", "y", "s", "z"))
    if not all(np.isfinite(a).all() for a in (v, y, s, z)):
        return dict(primal=float("inf"), dual=float("inf"), gap=float("inf"), objective=float("nan"))
    P = np.triu(p["P"]) + np.triu(p["P"], 1).T
    norm = lambda a: float(np.max(np.abs(a), initial=0.))
    return dict(primal=max(norm(p["A"] @ v - p["b"]), norm(p["G"] @ v + s - p["h"]),
                           float(np.max(-s, initial=0.)), float(np.max(p["G"] @ v - p["h"], initial=0.))),
                dual=max(norm(P @ v + p["q"] + p["A"].T @ y + p["G"].T @ z),
                         float(np.max(-z, initial=0.))),
                gap=float(np.sum(np.abs(s * z))), objective=float(.5 * v @ P @ v + p["q"] @ v))
