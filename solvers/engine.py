"""Application selector for native QOCO and the public ICORE Solver."""
from time import perf_counter
from icore import Solver
from ctcstc.utils import to_host


class ChainSolver:
    def __init__(self, backend, *, tolerance=1e-8, qoco_options=None):
        if backend not in ("qoco_native", "jax_icore", "icore_c", "icore_cpp"):
            raise ValueError("Select qoco_native, jax_icore, icore_c or icore_cpp")
        self.backend, self.tolerance = backend, float(tolerance)
        options = dict(qoco_options or {})
        self.max_iterations = int(options.pop("max_iters", 200))
        expected = dict(abstol=self.tolerance, abstol_inacc=self.tolerance,
                        reltol=0., reltol_inacc=0.)
        for key, value in options.items():
            if key not in expected or value != expected[key]:
                raise ValueError(f"Unsupported or inconsistent QOCO setting: {key}={value}")
        self.compile_seconds = self.setup_seconds = 0.
        self.prepared = False
        self.native = self.layout = None
        self.last_result = self.last_model = None

    @staticmethod
    def _host(qp):
        return to_host(qp)

    def prepare(self, qp):
        if self.prepared:
            return
        template, _ = self._host(qp)
        started = perf_counter()
        if self.backend == "qoco_native":
            from .qoco_native.chain import ChainCanonicalizer
            from .qoco_native import Solver as Qoco
            self.layout = ChainCanonicalizer(template)
            self.native = Qoco(self.layout.build(template), tolerance=self.tolerance,
                               max_iterations=self.max_iterations)
        else:
            backend = {"icore_c": "c", "icore_cpp": "cpp", "jax_icore": "jax"}[self.backend]
            self.native = Solver(template, backend=backend, tolerance=self.tolerance,
                                 max_iterations=self.max_iterations)
            self.compile_seconds = self.native.compile_seconds
        self.setup_seconds += perf_counter() - started - self.compile_seconds
        self.prepared = True

    def solve(self, qp):
        before_compile, before_setup = self.compile_seconds, self.setup_seconds
        self.prepare(qp)
        host_qp, download = (qp, 0.) if self.backend == 'jax_icore' else self._host(qp)
        started = perf_counter()
        if self.backend == "qoco_native":
            model = self.layout.build(host_qp)
            converted = perf_counter()
            self.native.update(model)
            updated = perf_counter()
            result = self.native.solve()
            y = self.layout.decode(result, model)
            conversion, update = converted - started, updated - converted
        else:
            model = host_qp
            timing = self.native.update(model)
            conversion = timing["model_conversion_seconds"]
            update = timing["workspace_update_seconds"]
            result = self.native.solve()
            y = result["stage_y"]
            self.compile_seconds = self.native.compile_seconds
        seconds = perf_counter() - started
        status, iterations = int(result["status"]), int(result["iterations"])
        retries = result.get("factor_retries", result.get("qr_fallbacks"))
        refinements = result.get("refinements")
        self.last_result, self.last_model = result, model
        success = status == (1 if self.backend == "qoco_native" else 0)
        record = dict(solver=self.backend, status=status, iterations=iterations,
                      status_ok=success, seconds=seconds, native_seconds=result.get("native_seconds"),
                      model_conversion_seconds=conversion, workspace_update_seconds=update,
                      factor_retries=None if retries is None else int(retries),
                      refinements=None if refinements is None else int(refinements),
                      jax_to_numpy_seconds=download, numpy_to_jax_seconds=0.,
                      compilation_seconds=self.compile_seconds - before_compile,
                      setup_seconds=self.setup_seconds - before_setup)
        if not success:
            raise RuntimeError(f"Convex subproblem did not converge: {record}")
        return y, record

    def close(self):
        if getattr(self, "native", None) is not None:
            self.native.close()
            self.native = None
        self.prepared = False

    def __del__(self):
        self.close()
