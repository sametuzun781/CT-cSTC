"""Public solve/update API shared by C, C++ and JAX backends."""

from collections.abc import Mapping
from math import isfinite
from numbers import Integral
from time import perf_counter

from ._native import Prepared, Native, CHAIN_FIELDS


def _mapping(problem):
    if hasattr(problem, "_asdict"):
        return problem._asdict()
    if isinstance(problem, Mapping):
        return problem
    raise TypeError("Provide a coefficient dictionary or ChainQP")


def _settings(
    tolerance,
    max_iterations,
    refinement_steps,
    regularization,
    step_fraction,
    refinement_tolerance,
    riccati_mode,
    step_policy,
):
    for name, value in (("max_iterations", max_iterations), ("refinement_steps", refinement_steps)):
        if not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not isinstance(riccati_mode, Integral) or riccati_mode not in (0, 1, 2):
        raise ValueError("riccati_mode must be 0, 1 or 2")
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not isfinite(step_fraction) or not 0 < step_fraction < 1:
        raise ValueError("step_fraction must be strictly between zero and one")
    if not isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    if not isfinite(refinement_tolerance) or refinement_tolerance < 0:
        raise ValueError("refinement_tolerance must be finite and nonnegative")
    policies = {"auto": 0, "fixed": 1, "adaptive": 2}
    if not isinstance(step_policy, str) or step_policy not in policies:
        raise ValueError("step_policy must be auto, fixed or adaptive")
    return dict(
        tolerance=float(tolerance),
        max_iterations=int(max_iterations),
        refinement_steps=int(refinement_steps),
        regularization=float(regularization),
        step_fraction=float(step_fraction),
        refinement_tolerance=float(refinement_tolerance),
        riccati_mode=int(riccati_mode),
        step_policy=policies[step_policy],
    )


class Solver:
    """Solve convex stage models; select matrix kernels from their coefficients.

    ``backend`` selects only the execution language: ``c``, ``cpp`` or ``jax``.
    Input dictionaries use H/c/A/B/b/G/h/x0 (and optional F/f) in [u,x]
    order, or the fourteen adjacent-stage ChainQP arrays. For OCP inputs,
    ``nonnegative`` and ``soc_sizes`` describe each stage's product cone.
    Without cone metadata, all inequality rows are nonnegative-cone rows.
    ``solve(problem)`` updates coefficients and solves; ``solve()`` reuses
    the prepared coefficients. Each solve starts from the same cold start.
    ``step_policy='auto'`` uses adaptive boundary fractions for orthant-only
    models and a fixed fraction for models with SOCs. ``'fixed'`` preserves
    the previous step rule; ``'adaptive'`` explicitly enables it for any cone.
    For fixed steps, ``step_fraction`` multiplies the admissible step. For
    adaptive steps it is the conservative endpoint of an interpolation toward
    ``max(step_fraction, 1-1e-7)`` as the admissible step approaches one.
    """

    def __init__(
        self,
        problem,
        *,
        backend="c",
        tolerance=1e-8,
        max_iterations=200,
        nonnegative=None,
        soc_sizes=None,
        epigraph_controls=None,
        refinement_steps=5,
        regularization=1e-11,
        step_fraction=0.99,
        refinement_tolerance=0.0,
        riccati_mode=0,
        structural_presolve=True,
        step_policy="auto",
    ):
        if backend not in ("c", "cpp", "jax"):
            raise ValueError("backend must be c, cpp or jax")
        self.backend = backend
        if not isinstance(structural_presolve, bool):
            raise TypeError("structural_presolve must be a bool")
        self.structural_presolve = structural_presolve
        self.settings = _settings(
            tolerance,
            max_iterations,
            refinement_steps,
            regularization,
            step_fraction,
            refinement_tolerance,
            riccati_mode,
            step_policy,
        )
        self.cones = dict(
            nonnegative=nonnegative, soc_sizes=soc_sizes, epigraph_controls=epigraph_controls
        )
        self._engine = None
        self._prepare_spare = None
        self._reduction = None
        self._presolve_witness = None
        self._curved_cones = 0
        self.closed = False
        self.compile_seconds = 0.0
        started = perf_counter()
        self.last_update = self.update(problem)
        self.setup_seconds = perf_counter() - started - self.compile_seconds

    def _prepare(self, problem):
        data = _mapping(problem)
        # Native create/update validates every original coefficient. A reduction
        # also needs host validation because it can discard input entries; defer
        # that scan until the current coefficients actually permit a reduction.
        defer = self.backend in ("c", "cpp")
        prepared = None
        if defer and self._engine is not None and self._reduction is None:
            prepared = Prepared.refresh(data, self._engine.prepared, self._prepare_spare, **self.cones)
        if prepared is None:
            prepared = Prepared(data, validate_finite=not defer, **self.cones)
        reduction = None
        witness = None
        if not prepared.chain and self.structural_presolve:
            from .presolve import (
                try_chain_presolve,
                _chain_rejection_still_valid,
                _chain_rejection_witness,
            )

            if _chain_rejection_still_valid(prepared.arrays, self._presolve_witness):
                witness = self._presolve_witness
            else:
                witness = _chain_rejection_witness(prepared.arrays)
                if witness is None:
                    prepared.ensure_finite()
                    parameters = dict(nonnegative=prepared.layout.nlp, soc_sizes=prepared.dims)
                    if self._reduction is not None and hasattr(self._reduction, "refresh"):
                        reduction = self._reduction.refresh(prepared.arrays, **parameters)
                    if reduction is None:
                        reduction = try_chain_presolve(prepared.arrays, **parameters)
                    if reduction is not None:
                        prepared = Prepared(reduction.problem)
        return prepared, reduction, witness

    def update(self, problem):
        """Validate new coefficients, reduce their structure and update storage."""
        if self.closed:
            raise RuntimeError("Solver is closed")
        started = perf_counter()
        data = _mapping(problem)
        if (
            self.backend == "jax"
            and self._engine is not None
            and all(k in data for k in CHAIN_FIELDS)
        ):
            self._engine.settings = self.settings.copy()
            self._engine.update_chain(data)
            self._reduction = None
            self._presolve_witness = None
            self._curved_cones = 0
            self.kernel = "affine_chain"
            self.compile_seconds = self._engine.compile_seconds
            self.last_update = dict(
                model_conversion_seconds=0.0, workspace_update_seconds=perf_counter() - started
            )
            return self.last_update.copy()
        previous = self._engine.prepared if self._engine is not None and self.backend in ("c", "cpp") else None
        prepared, reduction, witness = self._prepare(data)
        converted = perf_counter()
        settings = self.settings.copy()
        if reduction is not None:
            settings["tolerance"] /= reduction.tolerance_scale
        if self._engine is None:
            if self.backend == "jax":
                from ._jax import Jax

                self._engine = Jax(prepared, settings)
            else:
                self._engine = Native(prepared, self.backend, settings)
        else:
            if self.backend == "jax":
                self._engine.settings = settings
            self._engine.update(prepared)
            if self.backend != "jax":
                self._engine.configure(settings)
        # Retain only owned conversion buffers, not a previous input model's
        # potentially large A/B/G arrays. This bank is inactive after commit.
        self._prepare_spare = ((previous.arrays["H"], previous.arrays["c"])
                               if previous is not None and previous._owned_hc and reduction is None
                               else None)
        self._reduction = reduction
        # Commit only after the real native coefficient update succeeds.
        self._presolve_witness = witness
        self._curved_cones = 0 if prepared.chain else prepared.layout.nsoc
        self.kernel = "affine_chain" if prepared.chain else "riccati"
        self.compile_seconds = getattr(self._engine, "compile_seconds", 0.0)
        self.last_update = dict(
            model_conversion_seconds=converted - started,
            workspace_update_seconds=perf_counter() - converted,
        )
        return self.last_update.copy()

    def solve(self, problem=None):
        """Return a solution in the coordinates and constraint order of the input."""
        if self.closed:
            raise RuntimeError("Solver is closed")
        if problem is not None:
            self.update(problem)
        result = self._engine.solve()
        if self._reduction is not None:
            reduced_residuals = result["residuals"]
            if self.backend == "jax":
                import jax.numpy as xp
            else:
                import numpy as xp
            result = self._reduction.decode(result, xp=xp)
            residual = result["residuals"]
            valid = xp.all(xp.isfinite(residual)) & (xp.max(residual) <= self.settings["tolerance"])
            status = xp.where((result["status"] == 0) & ~valid, 2, result["status"])
            result["status"] = status if self.backend == "jax" else int(status)
            result["info"]["status"] = result["status"]
            result["reduced_residuals"] = reduced_residuals
        result["kernel"] = self.kernel
        return result

    @property
    def diagnostics(self):
        """Actual matrix-kernel counters for the most recent native solve."""
        if self.closed:
            raise RuntimeError("Solver is closed")
        if self.backend == "jax":
            return dict(kernel="jax")
        return self._engine.diagnostics()

    @property
    def factorization_plan(self):
        """Explain the coefficient-derived numerical path without executing it.

        The plan contains no application or benchmark identifiers. Both paths
        use the same native outer IPM driver and original stopping tolerances.
        ``structural_presolve=False`` disables only optional exact OCP reduction;
        an explicit ChainQP still uses the chain kernel.
        """
        chain = self.kernel == "affine_chain"
        reduced = self._reduction is not None
        curved = self._curved_cones > 0
        policy = ("auto", "fixed", "adaptive")[self.settings["step_policy"]]
        return dict(
            input_format="stage_ocp" if reduced or not chain else "affine_chain",
            kernel=self.kernel,
            selection_reason=(
                "exact_coefficient_eligible_ocp_reduction"
                if reduced
                else "explicit_adjacent_stage_chain" if chain else "general_stage_ocp"
            ),
            outer_iteration="shared_mehrotra_driver",
            factorization=(
                ((
                    "chain_classical_with_augmented_ldl_and_qr_recovery",
                    "chain_classical_cholesky",
                    "chain_orthogonal_qr",
                )[self.settings["riccati_mode"]] if self.backend in ("c", "cpp")
                 else "block_cholesky_with_qr_recovery")
                if chain
                else (
                    "riccati_auto_with_equality_capable_qr",
                    "riccati_classical",
                    "riccati_square_root",
                )[self.settings["riccati_mode"]]
            ),
            structural_presolve=self.structural_presolve,
            step_policy=policy,
            resolved_step_policy=(
                "adaptive" if policy == "adaptive" or (policy == "auto" and not curved) else "fixed"
            ),
            final_absolute_tolerance=self.settings["tolerance"],
        )

    @property
    def function(self):
        """Pure JAX function on prepared dynamic arrays, for composition in jit."""
        if self.backend != "jax":
            raise AttributeError("function is available with backend='jax'")
        return self._engine.function

    @property
    def compiled(self):
        if self.backend != "jax":
            raise AttributeError("compiled is available with backend='jax'")
        return self._engine.compiled

    @property
    def arguments(self):
        """Prepared coefficient pytree consumed by function/compiled (JAX only)."""
        if self.backend != "jax":
            raise AttributeError("arguments is available with backend='jax'")
        return self._engine.arguments

    def close(self):
        if self._engine is not None:
            self._engine.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
