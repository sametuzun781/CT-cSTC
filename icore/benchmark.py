"""One problem, four native solvers, three cold solves, one accuracy check.

This module contains only model conversion, timing and independent residual
checks. Optimization stays in ICORE, HPIPM and QOCO's native libraries.
"""
from dataclasses import dataclass
import json
import ctypes as ct
from pathlib import Path
import platform
import re
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from . import Solver, __version__
from ._build import build, build_info
from ._comparators import build_comparators, build_info as comparator_build_info, PINS
from .vendor.hpipm.adapter import HPIPMDense, HPIPMOCP
from .vendor.qoco.api import NativeQoco


PROBLEMS = Path(__file__).resolve().parent.parent / "problems"
METHODS = ("ICORE C", "ICORE C++", "HPIPM", "QOCO")
# Across eight repetitions, positions and preceding methods are balanced,
# including transitions between repetitions. Three repeats use a partial block.
# Complete linear blocks omit one closing self-pair; full balance is cyclic.
_METHOD_ORDERS = (
    (0, 1, 3, 2), (2, 3, 0, 1), (1, 2, 0, 3), (3, 0, 2, 1),
    (1, 0, 3, 2), (2, 3, 1, 0), (0, 2, 1, 3), (3, 1, 2, 0),
)
OCP_FIELDS = ("Q", "R", "S", "q", "r", "A", "B", "b", "G", "h", "x0", "F", "f")


def list_problems(directory=PROBLEMS):
    """List names, families and dimensions without loading every coefficient."""
    rows = []
    for path in sorted(Path(directory).glob("*.npz")):
        with np.load(path, allow_pickle=False) as arrays:
            row = json.loads(str(arrays["metadata"]))
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No problem files found in {directory}")
    return rows


@dataclass
class Problem:
    """Original stage coefficients plus an equivalent sparse QP for checking."""
    metadata: dict
    ocp: dict
    qp: SimpleNamespace
    stage_indices: list

    @property
    def name(self):
        return self.metadata["name"]

    @property
    def dense(self):
        return self.metadata["representation"] == "dense"

    def vectors(self, result):
        """Return independently owned x/y/s/z in the stored QP's coordinates."""
        if self.dense:
            return (result["u"][0].copy(), np.empty(0),
                    result["s"][0].copy(), result["z"][0].copy())
        N = self.metadata["N"]
        middle = np.concatenate((result["u"][1:], result["x"][1:N]), axis=1)
        x = np.concatenate((result["u"][0], middle.ravel(), result["x"][N]))
        y = np.concatenate((result["y"].ravel(), result["nu"].ravel()))
        return x, y, result["s"].ravel().copy(), result["z"].ravel().copy()


def _canonical_ocp(p):
    """Eliminate fixed x0, then assemble the same QP in stage order [u,x].

    Dynamics use A*x + B*u - x_next = -b, so their multiplier signs match
    the ICORE and HPIPM public stage outputs. Terminal controls do not exist.
    """
    N, nx, nu = len(p["Q"]) - 1, p["Q"].shape[1], p["R"].shape[1]
    stage_count, row_count = p["h"].shape
    blocks, linear, indices = [], [], []
    cursor = 0
    for k in range(N + 1):
        index = np.full(nu + nx, -1, dtype=int)
        if k == 0:
            index[:nu] = np.arange(cursor, cursor + nu)
            cursor += nu
            blocks.append(p["R"][k])
            linear.append(p["r"][k] + p["S"][k] @ p["x0"])
        elif k == N:
            index[nu:] = np.arange(cursor, cursor + nx)
            cursor += nx
            blocks.append(p["Q"][k])
            linear.append(p["q"][k])
        else:
            index[:] = np.arange(cursor, cursor + nu + nx)
            cursor += nu + nx
            blocks.append(np.block([[p["R"][k], p["S"][k]], [p["S"][k].T, p["Q"][k]]]))
            linear.append(np.r_[p["r"][k], p["q"][k]])
        indices.append(index)
    P = sparse.block_diag(blocks, format="csc")
    c = np.concatenate(linear)
    constant = float(.5 * p["x0"] @ p["Q"][0] @ p["x0"] + p["q"][0] @ p["x0"])
    A = sparse.lil_matrix((N * nx + len(p["f"]), cursor))
    b = np.r_[-p["b"][:N].ravel(), p["f"]]
    for k in range(N):
        rows = slice(k * nx, (k + 1) * nx)
        A[rows, indices[k][:nu]] = p["B"][k]
        if k:
            A[rows, indices[k][nu:]] = p["A"][k]
        else:
            b[rows] -= p["A"][k] @ p["x0"]
        A[rows, indices[k + 1][nu:]] = -np.eye(nx)
    if len(p["f"]):
        A[N * nx:, indices[N][nu:]] = p["F"]
    G = sparse.lil_matrix((stage_count * row_count, cursor))
    h = p["h"].copy()
    for k in range(stage_count):
        rows = slice(k * row_count, (k + 1) * row_count)
        active = indices[k] >= 0
        G[rows, indices[k][active]] = p["G"][k][:, active]
        if k == 0:
            h[k] -= p["G"][k, :, nu:] @ p["x0"]
    return _qp(P, c, A, b, G, h.ravel(), constant), indices


def _qp(P, c, A, b, G, h, constant=0.):
    matrices = [sparse.csc_matrix(matrix) for matrix in (P, A, G)]
    for matrix in matrices:
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
    return SimpleNamespace(P=matrices[0], c=c, A=matrices[1], b=b,
                           G=matrices[2], h=h, constant=float(constant),
                           l=len(h), cone_sizes=())


def load_problem(name, directory=PROBLEMS):
    """Read a selected .npz; reference objectives never initialize an optimizer."""
    path = Path(directory) / (str(name).removesuffix(".npz") + ".npz")
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"]))
        p = {key: np.array(arrays[key], copy=True) for key in OCP_FIELDS}
        if metadata["representation"] == "dense":
            values = [np.array(arrays["dense_" + key], copy=True)
                      for key in ("P", "c", "A", "b", "G", "h")]
            qp, indices = _qp(*values, float(arrays["dense_constant"])), []
        else:
            qp, indices = _canonical_ocp(p)
    for array in p.values():
        array.flags.writeable = False
    return Problem(metadata, p, qp, indices)


def prepare():
    """Compile both ICORE languages and comparators, without solving a problem."""
    return {"c": str(build("c")), "cpp": str(build("cpp")),
            **{name: str(path) for name, path in build_comparators().items()}}


def environment_info(libraries=None):
    """Show the actual compiled targets so a portable fallback is visible."""
    libraries = prepare() if libraries is None else libraries
    # Load every library during common preparation. Inspecting ICORE's target
    # must not leave only the competitors' first setup paying for dlopen.
    loaded = {name: ct.CDLL(path) for name, path in libraries.items()}
    solvers = []
    for name, backend in (("ICORE C", "c"), ("ICORE C++", "cpp")):
        info = build_info(backend, library=libraries[backend])
        library = loaded[backend]
        library.icore_la_blasfeo_available.restype = ct.c_int
        available = bool(library.icore_la_blasfeo_available())
        solvers.append(dict(solver=name, version=__version__, compiler=info["compiler"],
                            target=info["target"], blasfeo_available=available))
    header = Path(libraries["hpipm"]).parent / "install/include/blasfeo_target.h"
    targets = re.findall(r"^#define TARGET_([A-Z0-9_]+)\s*$", header.read_text(), re.MULTILINE)
    comparator = comparator_build_info(libraries["hpipm"])
    compiler = " ".join(comparator["commands"]["cc"])
    solvers.append(dict(solver="HPIPM", version="BALANCE", compiler=compiler,
                        target=targets[0] if targets else "unknown", blasfeo_available=True))
    solvers.append(dict(solver="QOCO", version="0.3.2", compiler=compiler,
                        target="builtin sparse LDL", blasfeo_available=False))
    return dict(platform=platform.platform(), machine=platform.machine(),
                python=sys.version.split()[0], solvers=solvers)


def audit(problem, vectors, *, status_ok, tolerance=1e-8):
    """Recompute unscaled original-QP residuals; no solver-reported pass flag.

    Gap is the SUM of absolute s_i*z_i, not its average. The equality/slack
    equations, actual inequalities, dual signs and reference objective are all
    checked. Different native stopping rules can therefore fail this same gate.
    """
    p, (x, y, slack, dual) = problem.qp, vectors
    if not all(np.isfinite(a).all() for a in vectors):
        return dict(primal=np.inf, dual=np.inf, gap=np.inf, objective=np.nan,
                    objective_error=np.inf, accepted=False)
    maximum = lambda a: float(np.max(np.abs(a), initial=0.))
    actual_slack = p.h - p.G @ x
    stationarity = p.P @ x + p.c + p.A.T @ y + p.G.T @ dual
    primal = max(maximum(p.A @ x - p.b), maximum(actual_slack - slack),
                 float(np.max(-slack, initial=0.)), float(np.max(-actual_slack, initial=0.)))
    dual_error = max(maximum(stationarity), float(np.max(-dual, initial=0.)))
    gap = float(np.sum(np.abs(slack * dual)))
    objective = float(.5 * x @ (p.P @ x) + p.c @ x + p.constant)
    reference = problem.metadata["reference_objective"]
    error = abs(objective - reference)
    finite = all(np.isfinite(v) for v in (primal, dual_error, gap, objective, error))
    accepted = bool(status_ok and finite and max(primal, dual_error, gap) <= tolerance
                    and error <= 10 * tolerance * (1 + abs(reference)))
    return dict(primal=primal, dual=dual_error, gap=gap, objective=objective,
                objective_error=error, accepted=accepted)


class _Run:
    """One fresh native workspace; no optimizer is called during construction."""
    def __init__(self, name, problem, tolerance):
        self.name, self.problem = name, problem
        p = problem.qp
        if name.startswith("ICORE"):
            self.solver = Solver(problem.ocp, backend="cpp" if name.endswith("C++") else "c",
                                 tolerance=tolerance, max_iterations=200)
        elif name == "HPIPM":
            options = dict(profile="BALANCE", tolerance=tolerance, max_iter=200,
                           complementarity_budget="component")
            self.solver = (HPIPMDense(dict(P=p.P, q=p.c, A=p.A, b=p.b, G=p.G, h=p.h), **options)
                           if problem.dense else HPIPMOCP(problem.ocp, **options))
        elif name == "QOCO":
            self.solver = NativeQoco(p, tolerance=tolerance, max_iterations=200,
                                     library=build_comparators()["qoco"])
        else:
            raise ValueError(name)

    def configuration(self):
        """Record the version and effective profile used by this workspace."""
        if self.name.startswith("ICORE"):
            return dict(version=__version__, language="C++17" if self.name.endswith("C++") else "C11",
                        profile="default", settings=self.solver.settings.copy())
        if self.name == "HPIPM":
            return dict(revision=PINS["hpipm"], profile="BALANCE",
                        form="dense QP" if self.problem.dense else "OCP QP",
                        settings=self.solver.settings.copy())
        return dict(version="0.3.2", revision=PINS["qoco"], profile="upstream defaults",
                    settings=dict(absolute_tolerance=self.solver.tolerance, relative_tolerance=0.,
                                  max_iterations=200, ruiz_iterations=0,
                                  refinement_tolerance=1e-6, refinement_steps=5))

    def solve(self):
        result = self.solver.solve()
        if self.name == "QOCO":
            vectors = tuple(result[key] for key in ("x", "y", "s", "z"))
        elif self.name == "HPIPM" and self.problem.dense:
            vectors = tuple(result[key] for key in ("v", "y", "s", "z"))
        else:
            vectors = self.problem.vectors(result)
        return result, vectors

    def close(self):
        self.solver.close()


def run_comparison(problem, *, repetitions=3, tolerance=1e-8):
    """Balance method order; retain every attempt and failure.

    Compilation and the common canonical-QP conversion are outside both timers.
    Solver-specific conversion/packing and workspace construction are measured
    by setup_ms. solve_ms includes the cold solve and owned canonical x/y/s/z.
    Accuracy checks happen afterwards. There are no warmup solves, retries or
    discarded runs.
    """
    if isinstance(problem, str):
        problem = load_problem(problem)
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    environment = environment_info(prepare())
    rows, solutions, methods = [], [], {}
    for repetition in range(repetitions):
        indices = _METHOD_ORDERS[repetition % len(_METHOD_ORDERS)]
        order = tuple(METHODS[index] for index in indices)
        for position, method in enumerate(order):
            row = dict(problem=problem.name, solver=method, repetition=repetition + 1,
                       order=position + 1, setup_ms=np.nan, solve_ms=np.nan, total_ms=np.nan,
                       status=None, iterations=None, accepted=False, error="")
            run = None
            try:
                started = perf_counter()
                run = _Run(method, problem, tolerance)
                row["setup_ms"] = (perf_counter() - started) * 1e3
                methods[method] = run.configuration()
                started = perf_counter()
                result, vectors = run.solve()
                row["solve_ms"] = (perf_counter() - started) * 1e3
                row["total_ms"] = row["setup_ms"] + row["solve_ms"]
                row["status"] = int(result["status"])
                row["iterations"] = int(result.get("iterations", result.get("info", {}).get("iterations", -1)))
                if method.startswith("ICORE"):
                    diagnostics = run.solver.diagnostics
                    packed = diagnostics["dense_packed"] or diagnostics["ocp_packed_stages"]
                    row["linear_algebra"] = ("dense, 0 IPM steps" if diagnostics["last_kernel"] == 1
                                              and row["iterations"] == 0 else
                                              "chain" if diagnostics["last_kernel"] == 2 else
                                              "BLASFEO" if packed else "portable")
                    row["kernel_diagnostics"] = diagnostics
                    row["refinements"] = int(result["refinements"])
                    row["factor_retries"] = int(result["factor_retries"])
                else:
                    row["linear_algebra"] = "BLASFEO" if method == "HPIPM" else "sparse LDL"
                row.update(audit(problem, vectors, status_ok=row["status"] == (1 if method == "QOCO" else 0),
                                 tolerance=tolerance))
                solutions.append(dict(solver=method, repetition=repetition + 1, vectors=vectors))
            except Exception as error:
                row["error"] = type(error).__name__ + ": " + str(error)
            finally:
                if run is not None:
                    try:
                        run.close()
                    except Exception as error:
                        row["accepted"] = False
                        row["error"] += ("; " if row["error"] else "") + "close: " + type(error).__name__ + ": " + str(error)
            rows.append(row)
    return dict(problem=problem.metadata, tolerance=tolerance, repetitions=repetitions,
                methods=methods, rows=rows, solutions=solutions, environment=environment)


def tables(result):
    """Two readable notebook tables; every timing and failure stays visible."""
    import pandas as pd
    details = pd.DataFrame(result["rows"])
    summary = []
    for method in METHODS:
        rows = details.loc[details.solver == method].sort_values("repetition")
        times = rows.solve_ms.to_numpy()
        item = {"Solver": method}
        item.update({f"{i + 1}. run (ms)": value for i, value in enumerate(times)})
        item["Median (ms)"] = float(np.median(times)) if np.isfinite(times).all() else np.nan
        setup = rows.setup_ms.to_numpy()
        item["Setup median (ms)"] = float(np.median(setup)) if np.isfinite(setup).all() else np.nan
        total = rows.total_ms.to_numpy()
        item["Setup + solve median (ms)"] = float(np.median(total)) if np.isfinite(total).all() else np.nan
        item["Accuracy"] = f"{int(rows.accepted.sum())}/{len(rows)}"
        summary.append(item)
    return pd.DataFrame(summary), details
