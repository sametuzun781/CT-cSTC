"""Physical-chain Newton kernels for the shared JAX primal-dual IPM driver.

The Newton factorization eliminates L1 epigraphs, then factors the physical
block-tridiagonal system. Coefficients remain dynamic
arguments of the returned jitted function. Outputs use the physical ChainQP
constraint order: (positive defect, negative defect, pair, local).
"""

from functools import lru_cache
from .jax_engine import run_ipm, validate_options
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import solve_triangular

jax.config.update("jax_enable_x64", True)


def _mv(a, x):
    return jnp.einsum("...ij,...j->...i", a, x)


def _max(a):
    return jnp.max(jnp.abs(a), initial=0.0)


def _tri(a, b, transpose=False):
    return solve_triangular(a, b, lower=True, trans="T" if transpose else "N")


def _sym(a):
    return 0.5 * (a + a.swapaxes(-1, -2))


def _check_shapes(qp):
    """Check only static metadata; never copy coefficient values to the host."""
    if len(qp.center.shape) != 2 or min(qp.center.shape) < 1:
        raise ValueError("center must have shape (K,n) with K,n positive")
    K, n = qp.center.shape
    for name in ("left", "pair_left", "local_matrix"):
        if len(getattr(qp, name).shape) != 3:
            raise ValueError(name + " must have three dimensions")
    d, p, l = qp.left.shape[1], qp.pair_left.shape[1], qp.local_matrix.shape[1]
    expected = dict(
        left=(K - 1, d, n),
        right=(K - 1, d, n),
        offset=(K - 1, d),
        pair_left=(K - 1, p, n),
        pair_right=(K - 1, p, n),
        pair_offset=(K - 1, p),
        local_matrix=(K, l, n),
        local_offset=(K, l),
        hessian=(K, n),
        linear=(K, n),
        free=(K, n),
        fixed=(K, n),
    )
    for name, shape in expected.items():
        if getattr(qp, name).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if jnp.broadcast_shapes(qp.l1_weight.shape, (K - 1, d)) != (K - 1, d):
        raise ValueError("l1_weight must broadcast to (K-1,d)")


@lru_cache(maxsize=32)
def make_chain_solver(
    *,
    tolerance=1e-8,
    max_iterations=200,
    refinement_steps=5,
    regularization=1e-11,
    step_fraction=0.99,
    refinement_tolerance=0.0,
    riccati_mode=0,
    step_policy=1,
):
    """Return a pure JAX solver; convergence uses absolute KKT and total gap."""
    refinement_limit = validate_options(
        tolerance,
        max_iterations,
        refinement_steps,
        regularization,
        step_fraction,
        refinement_tolerance,
        riccati_mode,
        step_policy,
    )

    def solve(qp):
        _check_shapes(qp)
        K, n = qp.center.shape
        free = qp.free
        fixed = qp.fixed * (1 - free)
        diagonal = qp.hessian * free + (1 - free)
        linear = free * (qp.linear + qp.hessian * (fixed - qp.center))
        L, R = qp.left * free[:-1, None, :], qp.right * free[1:, None, :]
        A, B = qp.pair_left * free[:-1, None, :], qp.pair_right * free[1:, None, :]
        G = qp.local_matrix * free[:, None, :]
        a = qp.offset + _mv(qp.left, fixed[:-1]) + _mv(qp.right, fixed[1:])
        original_b = qp.pair_offset + _mv(qp.pair_left, fixed[:-1]) + _mv(qp.pair_right, fixed[1:])
        original_c = qp.local_offset + _mv(qp.local_matrix, fixed)
        constant_pair = jnp.all(A == 0, axis=-1) & jnp.all(B == 0, axis=-1) & (original_b <= 0)
        constant_local = jnp.all(G == 0, axis=-1) & (original_c <= 0)
        b = jnp.where(constant_pair, -1.0, original_b)
        c = jnp.where(constant_local, -1.0, original_c)
        weight = jnp.broadcast_to(qp.l1_weight, a.shape)
        edge_left, edge_right = jnp.concatenate((L, A), axis=1), jnp.concatenate((R, B), axis=1)
        eye = jnp.eye(n)
        count = 2 * a.size + b.size + c.size

        def edge_apply(left, right, x):
            return _mv(left, x[:-1]) + _mv(right, x[1:])

        def edge_transpose(left, right, z):
            out = jnp.zeros((K, n), dtype=diagonal.dtype)
            return (
                out.at[:-1]
                .add(_mv(left.swapaxes(-1, -2), z))
                .at[1:]
                .add(_mv(right.swapaxes(-1, -2), z))
            )

        def constraints(x, t):
            residual = edge_apply(L, R, x)
            return residual - t, -residual - t, edge_apply(A, B, x), _mv(G, x)

        def transpose(z):
            zp, zm, za, zg = z
            return (
                edge_transpose(L, R, zp - zm)
                + edge_transpose(A, B, za)
                + _mv(G.swapaxes(-1, -2), zg)
            )

        def factor(edge_weights, local_weights, qr_only=False):
            local_root = jnp.concatenate(
                (
                    jnp.sqrt(diagonal + regularization)[:, :, None] * eye,
                    jnp.sqrt(local_weights)[:, :, None] * G,
                ),
                axis=1,
            )

            def classical(_):
                diagonal = local_root.swapaxes(-1, -2) @ local_root
                diagonal = diagonal.at[:-1].add(
                    edge_left.swapaxes(-1, -2) @ (edge_weights[:, :, None] * edge_left)
                )
                diagonal = diagonal.at[1:].add(
                    edge_right.swapaxes(-1, -2) @ (edge_weights[:, :, None] * edge_right)
                )
                upper = edge_left.swapaxes(-1, -2) @ (edge_weights[:, :, None] * edge_right)
                first = jnp.linalg.cholesky(_sym(diagonal[0]))

                def step(previous, blocks):
                    matrix, cross = blocks
                    link = _tri(previous, cross)
                    root = jnp.linalg.cholesky(_sym(matrix - link.T @ link))
                    return root, (root, link)

                _, (roots, links) = lax.scan(step, first, (diagonal[1:], upper))
                return jnp.concatenate((first[None], roots)), links, jnp.array(False)

            def square_root(_):
                roots_left = jnp.sqrt(edge_weights)[:, :, None] * edge_left
                roots_right = jnp.sqrt(edge_weights)[:, :, None] * edge_right

                def step(carry, blocks):
                    Qroot, Lroot, Rroot = blocks
                    upper = jnp.concatenate((Qroot, carry, Lroot), axis=0)
                    lower = jnp.concatenate((jnp.zeros((Qroot.shape[0] + n, n)), Rroot), axis=0)
                    triangular = jnp.linalg.qr(jnp.concatenate((upper, lower), axis=1), mode="r")
                    return triangular[n:, n:], (triangular[:n, :n].T, triangular[:n, n:])

                carry, (roots, links) = lax.scan(
                    step, jnp.zeros((n, n)), (local_root[:-1], roots_left, roots_right)
                )
                last = jnp.linalg.qr(jnp.concatenate((local_root[-1], carry), axis=0), mode="r").T
                return jnp.concatenate((roots, last[None])), links, jnp.array(True)

            if qr_only or riccati_mode == 2:
                return square_root(None)
            result = classical(None)
            if riccati_mode == 1:
                return result
            valid = jnp.all(jnp.isfinite(result[0])) & jnp.all(jnp.isfinite(result[1]))
            return lax.cond(valid, lambda _: result, square_root, None)

        def linear_solve(factors, rhs):
            roots, links, _ = factors
            first = _tri(roots[0], rhs[0])

            def advance(previous, blocks):
                root, link, bk = blocks
                value = _tri(root, bk - link.T @ previous)
                return value, value

            _, middle = lax.scan(advance, first, (roots[1:], links, rhs[1:]))
            middle = jnp.concatenate((first[None], middle))
            last = _tri(roots[-1], middle[-1], True)

            def backward(following, blocks):
                root, link, bk = blocks
                value = _tri(root, bk - link @ following, True)
                return value, value

            _, preceding = lax.scan(backward, last, (roots[:-1], links, middle[:-1]), reverse=True)
            return jnp.concatenate((preceding, last[None]))

        def residual(x, t, s, z):
            values = constraints(x, t)
            offsets = (a, -a, b, c)
            primal = tuple(value + offset + sk for value, offset, sk in zip(values, offsets, s))
            dual_x = diagonal * x + linear + transpose(z)
            dual_t = weight - z[0] - z[1]
            primal_norm = jnp.max(
                jnp.stack(
                    [_max(v) for v in primal]
                    + [jnp.maximum(0.0, -jnp.min(v, initial=0.0)) for v in s]
                )
            )
            dual_norm = jnp.max(
                jnp.stack(
                    [_max(dual_x), _max(dual_t)]
                    + [jnp.maximum(0.0, -jnp.min(v, initial=0.0)) for v in z]
                )
            )
            gap = sum(jnp.sum(jnp.abs(sk * zk)) for sk, zk in zip(s, z))
            return (dual_x, dual_t, primal), jnp.stack((primal_norm, dual_norm, gap))

        # Same least-squares start as an unscaled primal-dual KKT initialization.
        initial_factor = factor(
            jnp.concatenate((2 * jnp.ones_like(a), jnp.ones_like(b)), axis=1), jnp.ones_like(c)
        )
        initial_rhs = (
            -linear
            - edge_transpose(L, R, 2 * a)
            - edge_transpose(A, B, b)
            - _mv(G.swapaxes(-1, -2), c)
        )
        x = linear_solve(initial_factor, initial_rhs)
        t = -0.5 * weight
        raw = tuple(-v - o for v, o in zip(constraints(x, t), (a, -a, b, c)))
        minimum = jnp.min(jnp.stack([jnp.min(v, initial=jnp.inf) for v in raw]))
        maximum = jnp.max(jnp.stack([jnp.max(v, initial=-jnp.inf) for v in raw]))
        s = tuple(v + jnp.maximum(0.0, 1.0 - minimum) for v in raw)
        z = tuple(-v + jnp.maximum(0.0, 1.0 + maximum) for v in raw)
        _, error = residual(x, t, s, z)
        valid = (
            jnp.all((free == 0) | (free == 1))
            & jnp.all(qp.hessian > 0)
            & jnp.all(weight > 0)
            & jnp.all(jnp.isfinite(jnp.concatenate([v.reshape(-1) for v in qp])))
        )

        def maximum_step(values, directions):
            return jnp.minimum(
                1.0,
                jnp.min(
                    jnp.stack(
                        [
                            jnp.min(jnp.where(dv < 0, -v / dv, jnp.inf), initial=1.0)
                            for v, dv in zip(values, directions)
                        ]
                    )
                ),
            )

        def prepare(values):
            x, t, s, z = values
            (rdx, rdt, rp), error = residual(x, t, s, z)
            weights = tuple(zk / sk for sk, zk in zip(s, z))
            wp, wm, wa, wg = weights
            inverse = 1.0 / (wp + wm)
            ratio = (wm - wp) * inverse
            small, large = jnp.minimum(wp, wm), jnp.maximum(wp, wm)
            curvature = 4 * small / (1 + small / large)
            edge_weights = jnp.concatenate((curvature, wa), axis=1)
            fac = factor(edge_weights, wg)
            target = (-rdx, -rdt, tuple(-v for v in rp))

            return target, weights, inverse, ratio, curvature, edge_weights, fac

        def solve_direction(context, values, complementarity):
            target, weights, inverse, ratio, curvature, edge_weights, fac = context
            x, t, s, z = values
            wp, wm, wa, wg = weights

            def direction(fac, complementarity):
                bx, bt, bp = target
                scales = (
                    1 + _max(bx),
                    1 + _max(bt),
                    1 + jnp.max(jnp.stack([_max(v) for v in bp])),
                    1 + jnp.max(jnp.stack([_max(v) for v in complementarity])),
                )

                def solve_rhs(rhs):
                    bx, bt, bp, bc = rhs
                    q = tuple((ck - zk * pk) / sk for ck, zk, pk, sk in zip(bc, z, bp, s))
                    trhs = bt + q[0] + q[1]
                    force = 2 * (wm * inverse) * q[0] - 2 * (wp * inverse) * q[1] + ratio * bt
                    reduced = (
                        bx
                        - edge_transpose(L, R, force)
                        - edge_transpose(A, B, q[2])
                        - _mv(G.swapaxes(-1, -2), q[3])
                    )
                    dx = linear_solve(fac, reduced)
                    defect_step = edge_apply(L, R, dx)
                    dt = inverse * trhs - ratio * defect_step
                    gv = constraints(dx, dt)
                    ds = tuple(pk - gk for pk, gk in zip(bp, gv))
                    difference = force + curvature * defect_step
                    dz = (
                        0.5 * (difference - bt),
                        0.5 * (-difference - bt),
                        q[2] + wa * gv[2],
                        q[3] + wg * gv[3],
                    )
                    return dx, dt, ds, dz

                original = (*target, complementarity)

                def errors(step):
                    dx, dt, ds, dz = step
                    gv = constraints(dx, dt)
                    ex = bx - diagonal * dx - transpose(dz)
                    et = bt + dz[0] + dz[1]
                    ep = tuple(pk - gk - sk for pk, gk, sk in zip(bp, gv, ds))
                    ec = tuple(
                        ck - zk * dsk - sk * dzk
                        for ck, zk, dsk, sk, dzk in zip(complementarity, z, ds, s, dz)
                    )
                    norm = jnp.max(
                        jnp.stack(
                            (
                                _max(ex) / scales[0],
                                _max(et) / scales[1],
                                jnp.max(jnp.stack([_max(v) for v in ep])) / scales[2],
                                jnp.max(jnp.stack([_max(v) for v in ec])) / scales[3],
                            )
                        )
                    )
                    return (ex, et, ep, ec), norm

                step = solve_rhs(original)
                defect, norm = errors(step)

                def refine(carry):
                    count, step, defect, norm, improving = carry
                    correction = solve_rhs(defect)
                    candidate = jax.tree.map(jnp.add, step, correction)
                    candidate_defect, candidate_norm = errors(candidate)
                    improve = jnp.isfinite(candidate_norm) & (candidate_norm < norm)
                    chosen = jax.tree.map(
                        lambda new, old: jnp.where(improve, new, old), candidate, step
                    )
                    chosen_defect = jax.tree.map(
                        lambda new, old: jnp.where(improve, new, old), candidate_defect, defect
                    )
                    return (
                        count + 1,
                        chosen,
                        chosen_defect,
                        jnp.where(improve, candidate_norm, norm),
                        improve,
                    )

                result = lax.while_loop(
                    lambda st: (st[0] < refinement_steps) & (st[3] > refinement_limit) & st[4],
                    refine,
                    (jnp.array(0), step, defect, norm, jnp.array(True)),
                )
                return result[1], result[3], result[0]

            result = direction(fac, complementarity)
            if riccati_mode == 0:

                def retry_direction(_):
                    new = factor(edge_weights, wg, qr_only=True)
                    return new, direction(new, complementarity)

                retry = ((~jnp.isfinite(result[1])) | (result[1] > refinement_limit)) & (~fac[-1])
                fac, result = lax.cond(retry, retry_direction, lambda _: (fac, result), None)
            context = target, weights, inverse, ratio, curvature, edge_weights, fac
            return (
                context,
                result[0],
                (fac[-1].astype(jnp.int32), result[2]),
                jnp.isfinite(result[1]),
            )

        def predictor_rhs(context, values):
            s, z = values[-2:]
            return tuple(-sk * zk for sk, zk in zip(s, z))

        def corrector_rhs(context, values, affine, target):
            s, z = values[-2:]
            ds, dz = affine[-2:]
            return tuple(-sk * zk - dsk * dzk + target for sk, zk, dsk, dzk in zip(s, z, ds, dz))

        def cone_pair(s, z):
            return sum(jnp.sum(jnp.abs(sk * zk)) for sk, zk in zip(s, z))

        def cone_interior(s, z):
            return jnp.all(jnp.stack([jnp.all(v > 0.0) for v in (*s, *z)]))

        values, run = run_ipm(
            (x, t, s, z),
            valid=valid,
            cone_degree=count,
            tolerance=tolerance,
            max_iterations=max_iterations,
            step_fraction=step_fraction,
            step_policy=step_policy,
            curved_cones=0,
            evaluate=lambda values: residual(*values)[1],
            prepare=prepare,
            predictor_rhs=predictor_rhs,
            corrector_rhs=corrector_rhs,
            direction=solve_direction,
            maximum_step=maximum_step,
            cone_pair=cone_pair,
            cone_interior=cone_interior,
        )
        x, t, s, z = values
        it, error, status = run["iterations"], run["residuals"], run["status"]
        nqr, nref = run["qr_fallbacks"], run["refinements"]
        y = fixed + free * x
        # Undo harmless normalization of constant feasible inequalities.
        s = (
            s[0],
            s[1],
            jnp.where(constant_pair, -original_b, s[2]),
            jnp.where(constant_local, -original_c, s[3]),
        )
        z = (z[0], z[1], jnp.where(constant_pair, 0.0, z[2]), jnp.where(constant_local, 0.0, z[3]))
        objective = (
            0.5 * jnp.sum(qp.hessian * (y - qp.center) ** 2)
            + jnp.sum(qp.linear * y)
            + jnp.sum(weight * jnp.abs(edge_apply(qp.left, qp.right, y) + qp.offset))
        )
        return dict(
            stage_y=y,
            epigraph=t,
            s=s,
            z=z,
            status=status,
            iterations=it,
            residuals=error,
            objective=objective,
            qr_fallbacks=nqr,
            refinements=nref,
        )

    return jax.jit(solve)
