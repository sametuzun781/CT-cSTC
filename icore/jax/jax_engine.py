"""One JAX Mehrotra iteration for all supported matrix layouts."""

from math import isfinite
from numbers import Integral

import jax
import jax.numpy as jnp
from jax import lax


def validate_options(
    tolerance,
    max_iterations,
    refinement_steps,
    regularization,
    step_fraction,
    refinement_tolerance,
    riccati_mode,
    step_policy=1,
):
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    if not isfinite(step_fraction) or not 0 < step_fraction < 1:
        raise ValueError("step_fraction must lie strictly between zero and one")
    if not isfinite(refinement_tolerance) or refinement_tolerance < 0:
        raise ValueError("refinement_tolerance must be finite and nonnegative")
    if not isinstance(max_iterations, Integral) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    if not isinstance(refinement_steps, Integral) or refinement_steps < 0:
        raise ValueError("refinement_steps must be a nonnegative integer")
    if not isinstance(riccati_mode, Integral) or riccati_mode not in (0, 1, 2):
        raise ValueError("riccati_mode must be zero, one, or two")
    if not isinstance(step_policy, Integral) or step_policy not in (0, 1, 2):
        raise ValueError("step_policy must be zero, one, or two")
    return refinement_tolerance or min(1e-6, 0.1 * tolerance)


def run_ipm(
    initial,
    *,
    valid,
    cone_degree,
    tolerance,
    max_iterations,
    step_fraction,
    evaluate,
    prepare,
    predictor_rhs,
    corrector_rhs,
    direction,
    maximum_step,
    cone_pair,
    cone_interior,
    step_policy=1,
    curved_cones=0
):
    """Run the common IPM loop; array coefficients stay in traced closures.

    An iterate and its Newton direction are matching pytrees whose final
    two entries are primal slacks and cone duals. Kernels own matrix
    preparation, cone algebra and linear solves. This function owns all
    nonlinear stopping, predictor/corrector, centering and step policy.
    """
    error = evaluate(initial)
    state = (
        jnp.array(0),
        initial,
        error,
        valid & jnp.all(jnp.isfinite(error)),
        (jnp.array(0), jnp.array(0)),
    )

    def add(values, increments, alpha):
        return jax.tree.map(lambda value, delta: value + alpha * delta, values, increments)

    def iteration(state):
        it, values, error, healthy, statistics = state
        context = prepare(values)
        context, affine, first_stats, first_good = direction(
            context, values, predictor_rhs(context, values)
        )
        s, z = values[-2:]
        if cone_degree:
            ads, adz = affine[-2:]
            alpha_s, alpha_z = maximum_step(s, ads), maximum_step(z, adz)
            mu = error[2] / cone_degree
            affine_gap = cone_pair(add(s, ads, alpha_s), add(z, adz, alpha_z))
            mu_aff = jnp.maximum(0.0, affine_gap) / cone_degree
            sigma = jnp.clip(mu_aff / mu, 0.0, 1.0) ** 3
            context, step, last_stats, last_good = direction(
                context, values, corrector_rhs(context, values, affine, sigma * mu)
            )
            ds, dz = step[-2:]
            admissible = jnp.minimum(maximum_step(s, ds), maximum_step(z, dz))
            if step_policy == 2 or (step_policy == 0 and curved_cones == 0):
                ceiling = max(step_fraction, 0.9999999)
                alpha = admissible * ((1.0 - admissible) * step_fraction + admissible * ceiling)
            else:
                alpha = step_fraction * admissible
            increment_stats = (last_stats[0], first_stats[1] + last_stats[1])
        else:
            step, last_good, increment_stats = affine, first_good, first_stats
            ds, dz = step[-2:]
            alpha = jnp.array(1.0)

        def inside(a):
            return cone_interior(add(s, ds, a), add(z, dz, a))

        alpha, backtracks = lax.while_loop(
            lambda st: (~inside(st[0])) & (st[1] < 20),
            lambda st: (0.5 * st[0], st[1] + 1),
            (alpha, jnp.array(0)),
        )
        candidate = add(values, step, alpha)
        next_error = evaluate(candidate)
        good = (
            first_good
            & last_good
            & jnp.all(jnp.isfinite(next_error))
            & jnp.isfinite(alpha)
            & (alpha > 0)
            & (backtracks < 20)
            & inside(alpha)
        )
        output = jax.tree.map(lambda new, old: jnp.where(good, new, old), candidate, values)
        statistics = tuple(a + b for a, b in zip(statistics, increment_stats))
        return it + 1, output, jnp.where(good, next_error, error), good, statistics

    final = lax.while_loop(
        lambda st: (st[0] < max_iterations) & (jnp.max(st[2]) > tolerance) & st[3], iteration, state
    )
    it, values, error, healthy, statistics = final
    status = jnp.where(jnp.max(error) <= tolerance, 0, jnp.where(healthy, 1, 2))
    status = jnp.where(valid, status, 3)
    return values, dict(
        status=status,
        iterations=it,
        residuals=error,
        qr_fallbacks=statistics[0],
        refinements=statistics[1],
    )
