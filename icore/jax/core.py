"""OCP Newton kernels with Riccati recursion and Nesterov--Todd cone scaling.

``make_solver(orthant, cones)(pack(data))`` is a pure array function. Its
coefficients remain dynamic arguments when composed inside ``jax.jit``.
Stage dimensions and cone layout determine the executable shape. Arithmetic
uses float64; the backward/forward sweeps use ``lax.scan`` and the IPM and
refinement iterations use ``lax.while_loop``. The nonlinear iteration is
shared with the physical-chain layout in ``jax_engine.run_ipm``.
"""

from functools import lru_cache
from .jax_engine import run_ipm, validate_options

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import solve_triangular

jax.config.update("jax_enable_x64", True)


def pack(data):
    """Build dynamic stage arrays from Q/R/S/q/r or H/c coefficients."""
    p = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in data.items()}
    N, nx, nu = p["G"].shape[0] - 1, p["B"].shape[-2], p["B"].shape[-1]
    if "H" not in p:
        H = jnp.zeros((N + 1, nu + nx, nu + nx), dtype=jnp.float64)
        H = H.at[:N, :nu, :nu].set(p["R"][:N])
        H = H.at[:N, :nu, nu:].set(p["S"][:N])
        H = H.at[:N, nu:, :nu].set(p["S"][:N].swapaxes(-1, -2))
        H = H.at[:, nu:, nu:].set(p["Q"])
        c = jnp.zeros((N + 1, nu + nx), dtype=jnp.float64)
        c = c.at[:N, :nu].set(p["r"][:N]).at[:, nu:].set(p["q"])
    else:
        H, c = p["H"], p["c"]
    H = jnp.triu(H) + jnp.triu(H, 1).swapaxes(-1, -2)
    return dict(
        H=H,
        c=c,
        A=p["A"][:N],
        B=p["B"][:N],
        b=p["b"][:N],
        G=p["G"],
        h=p["h"],
        x0=p["x0"],
        F=p.get("F", jnp.empty((0, nx))),
        f=p.get("f", jnp.empty((0,))),
    )


def _mv(a, x):
    return jnp.einsum("...ij,...j->...i", a, x)


def _sym(a):
    return (a + a.swapaxes(-1, -2)) * 0.5


def _inf(a):
    return jnp.max(jnp.abs(a), initial=0.0)


def _tri(a, b, transpose=False):
    return solve_triangular(a, b, lower=True, trans="T" if transpose else "N")


@lru_cache(maxsize=64)
def make_solver(
    orthant,
    cones=(),
    *,
    epigraph_controls=0,
    tolerance=1e-7,
    max_iterations=200,
    refinement_steps=5,
    regularization=1e-11,
    step_fraction=0.99,
    refinement_tolerance=0.0,
    riccati_mode=0,
    step_policy=1
):
    """Return a cached jitted solve function for a static cone layout.

    ``orthant`` is the number of nonnegative coordinates per stage and
    ``cones`` contains Lorentz-cone sizes. A solve returns physical arrays
    x, u, y, nu, s, z, plus status (0 solved, 1 iteration limit, 2 numerical
    failure, 3 invalid model), iteration count and KKT residuals.

    With ``epigraph_controls=ne``, the first ne controls are t variables
    with zero cost-Hessian and dynamics columns, each constrained only by
    two orthant rows -t+e and -t-e with unit coefficients. Their terminal
    padded columns are zero. This contract is checked on dynamic arrays.
    """
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
    cones = tuple(map(int, cones))
    orthant, ne = int(orthant), int(epigraph_controls)
    offsets, offset = [], orthant
    for size in cones:
        if size < 2:
            raise ValueError("A Lorentz cone needs at least two coordinates")
        offsets.append((offset, offset + size))
        offset += size
    if orthant < 0 or tolerance <= 0 or max_iterations < 0 or ne < 0:
        raise ValueError("Invalid cone layout or solver settings")

    def cone_min(v):
        out = jnp.min(v[:, :orthant], initial=jnp.inf)
        for a, b in offsets:
            out = jnp.minimum(out, jnp.min(v[:, a] - jnp.linalg.norm(v[:, a + 1 : b], axis=1)))
        return out

    def cone_pair(s, z):
        total = jnp.sum(jnp.abs(s[:, :orthant] * z[:, :orthant]))
        for a, b in offsets:
            st, zt = s[:, a + 1 : b], z[:, a + 1 : b]
            rs, rz = jnp.linalg.norm(st, axis=1), jnp.linalg.norm(zt, axis=1)
            ss, zz = jnp.maximum(rs, 1e-300), jnp.maximum(rz, 1e-300)
            angle = jnp.sum((st / ss[:, None] + zt / zz[:, None]) ** 2, axis=1)
            value = (s[:, a] - rs) * z[:, a] + rs * (z[:, a] - rz) + 0.5 * rs * rz * angle
            total += jnp.sum(jnp.abs(value))
        return total

    def cone_shift(v):
        delta = jnp.maximum(0.0, 1.0 - cone_min(v))
        v = v.at[:, :orthant].add(delta)
        for a, _ in offsets:
            v = v.at[:, a].add(delta)
        return v

    def scaling(s, z):
        diagonal = jnp.sqrt(s[:, :orthant]) / jnp.sqrt(z[:, :orthant])
        blocks = []
        for a, b in offsets:
            st, zt = s[:, a + 1 : b], z[:, a + 1 : b]
            rs, rz = jnp.linalg.norm(st, axis=1), jnp.linalg.norm(zt, axis=1)
            ds = jnp.sqrt(s[:, a] - rs) * jnp.sqrt(s[:, a]) * jnp.sqrt(1.0 + rs / s[:, a])
            dz = jnp.sqrt(z[:, a] - rz) * jnp.sqrt(z[:, a]) * jnp.sqrt(1.0 + rz / z[:, a])
            ss, zz = jnp.maximum(rs, 1e-300), jnp.maximum(rz, 1e-300)
            angle = jnp.sum((st / ss[:, None] + zt / zz[:, None]) ** 2, axis=1)
            sp, zp = (s[:, a] + rs) / ds, (z[:, a] + rz) / dz
            inner = 0.5 * (sp / zp + zp / sp) + 0.5 * (rs / ds) * (rz / dz) * angle
            inner = jnp.where((rs > 0) & (rz > 0), inner, (s[:, a] / ds) * (z[:, a] / dz))
            axis = (
                0.5 * (st / ds[:, None] - zt / dz[:, None]) / jnp.sqrt(0.5 * (1.0 + inner))[:, None]
            )
            radius = jnp.linalg.norm(axis, axis=1)
            large = jnp.hypot(1.0, radius) + radius
            axis = axis / jnp.maximum(radius, 1e-300)[:, None]
            blocks.append((large, axis, jnp.sqrt(ds) / jnp.sqrt(dz)))
        return diagonal, tuple(blocks)

    def cone_apply(scale, value, inverse=False):
        diagonal, blocks = scale
        matrix = value.ndim == 3
        factor = 1.0 / diagonal if inverse else diagonal
        out = value.at[:, :orthant].set(
            value[:, :orthant] * (factor[..., None] if matrix else factor)
        )
        for (a, b), (large, axis, eta) in zip(offsets, blocks):
            v = value[:, a:b]
            if matrix:
                parallel = jnp.einsum("ki,kij->kj", axis, v[:, 1:])
                head, large, eta = v[:, 0], large[:, None], eta[:, None]
            else:
                parallel = jnp.sum(axis * v[:, 1:], axis=1)
                head = v[:, 0]
            high, low = (1.0 / large, large) if inverse else (large, 1.0 / large)
            plus, minus = high * (head + parallel), low * (head - parallel)
            factor = 1.0 / eta if inverse else eta
            transformed_head = factor * 0.5 * (plus + minus)
            correction = 0.5 * (plus - minus) - parallel
            if matrix:
                transformed_tail = factor[:, None, :] * (
                    v[:, 1:] + axis[..., None] * correction[:, None, :]
                )
            else:
                transformed_tail = factor[:, None] * (v[:, 1:] + axis * correction[:, None])
            out = out.at[:, a].set(transformed_head).at[:, a + 1 : b].set(transformed_tail)
        return out

    def corrector(scale, lam, ads, adz, target):
        a, b = cone_apply(scale, ads, True), cone_apply(scale, adz)
        q = -lam
        q = q.at[:, :orthant].add((target - ads[:, :orthant] * adz[:, :orthant]) / lam[:, :orthant])
        for lo, hi in offsets:
            l, av, bv = lam[:, lo:hi], a[:, lo:hi], b[:, lo:hi]
            head = target - jnp.sum(av * bv, axis=1)
            tail = -(av[:, :1] * bv[:, 1:] + bv[:, :1] * av[:, 1:])
            radius = jnp.linalg.norm(l[:, 1:], axis=1)
            denominator = (l[:, 0] - radius) * (1.0 + radius / l[:, 0])
            q0 = (head - jnp.sum((l[:, 1:] / l[:, :1]) * tail, axis=1)) / denominator
            qt = (tail - l[:, 1:] * q0[:, None]) / l[:, :1]
            q = q.at[:, lo].add(q0).at[:, lo + 1 : hi].add(qt)
        return q

    def cone_step(x, d):
        alpha = jnp.minimum(
            1.0,
            jnp.min(
                jnp.where(d[:, :orthant] < 0, -x[:, :orthant] / d[:, :orthant], jnp.inf),
                initial=1.0,
            ),
        )
        for lo, hi in offsets:
            xv, dv = x[:, lo:hi], d[:, lo:hi]
            scale = jnp.maximum(jnp.max(jnp.abs(xv), axis=1), jnp.max(jnp.abs(dv), axis=1))
            xv, dv = xv / scale[:, None], dv / scale[:, None]
            xt, dt = xv[:, 0], dv[:, 0]
            rx, rd = jnp.linalg.norm(xv[:, 1:], axis=1), jnp.linalg.norm(dv[:, 1:], axis=1)
            aa, bb, cc = (
                (dt - rd) * (dt + rd),
                2 * (xt * dt - jnp.sum(xv[:, 1:] * dv[:, 1:], axis=1)),
                (xt - rx) * (xt + rx),
            )
            disc = jnp.maximum(0.0, bb * bb - 4 * aa * cc)
            qq = -0.5 * (bb + jnp.where(bb >= 0, 1.0, -1.0) * jnp.sqrt(disc))
            r1, r2 = qq / aa, cc / qq
            limit = jnp.minimum(
                jnp.where(dt < 0, -xt / dt, jnp.inf),
                jnp.minimum(jnp.where(r1 > 0, r1, jnp.inf), jnp.where(r2 > 0, r2, jnp.inf)),
            )
            trial = xv + dv
            inside = trial[:, 0] > jnp.linalg.norm(trial[:, 1:], axis=1)
            alpha = jnp.minimum(alpha, jnp.min(jnp.where(inside, 1.0, limit)))
        return jnp.clip(alpha, 0.0, 1.0)

    def solve(p):
        H, c, A, B, b, G, h, x0, F, f = (
            p[k] for k in ("H", "c", "A", "B", "b", "G", "h", "x0", "F", "f")
        )
        N, nx, nu, nf = A.shape[0], A.shape[-1], B.shape[-1], F.shape[0]
        nv, nr, K, m = nx + nu, nu - ne, N + 1, G.shape[1]
        if m != offset or ne > nu // 2:
            raise ValueError("Cone layout or epigraph count does not match coefficients")
        D = jnp.concatenate((B, A), axis=2)
        eye = jnp.eye(nv - ne, dtype=H.dtype)
        # Orthogonalize terminal rows, retaining the map to physical duals.
        if nf:
            fq, fr = jnp.linalg.qr(F.T, mode="reduced")
            Fn = fq.T
            fn = solve_triangular(fr.T, f, lower=True)
        else:
            Fn, fn, fr = F, f, jnp.empty((0, 0), dtype=H.dtype)
        evals, evecs = jnp.linalg.eigh(H)
        roots = (evecs * jnp.sqrt(jnp.maximum(evals, 0.0))[:, None, :]).swapaxes(-1, -2)
        model_valid = jnp.array(True)
        for array in p.values():
            model_valid = model_valid & jnp.isfinite(array).all()
        model_valid = model_valid & jnp.all(
            jnp.min(evals, axis=1) >= -1e-11 * jnp.maximum(1.0, jnp.max(jnp.abs(evals), axis=1))
        )
        if ne:
            epimask = jnp.any(G[:, :, :ne] != 0, axis=2)
            epicol = G[:N, :orthant, :ne].swapaxes(1, 2)
            # Each epigraph has the constraints -t+e <= h and -t-e <= h.
            remaining = G[:N, :orthant, ne:]
            pairmask = epicol != 0
            positive = jnp.any(remaining > 0, axis=2)[:, None, :] & pairmask
            negative = jnp.any(remaining < 0, axis=2)[:, None, :] & pairmask
            plus, minus = jnp.argmax(positive, axis=2), jnp.argmax(negative, axis=2)
            selected = jnp.take_along_axis(remaining, plus[:, :, None], axis=1)
            eindex = jnp.argmax(jnp.abs(selected), axis=2)
            emap = jax.nn.one_hot(eindex, nv - ne, dtype=H.dtype)
            minus_selected = jnp.take_along_axis(remaining, minus[:, :, None], axis=1)
            tplus = jnp.take_along_axis(G[:N, :orthant, :ne], plus[:, :, None], axis=1)
            tminus = jnp.take_along_axis(G[:N, :orthant, :ne], minus[:, :, None], axis=1)
            model_valid = model_valid & jnp.all(H[:, :ne] == 0.0) & jnp.all(D[:, :, :ne] == 0.0)
            model_valid = model_valid & jnp.all(jnp.sum(pairmask, axis=2) == 2)
            model_valid = model_valid & jnp.all(selected == emap) & jnp.all(minus_selected == -emap)
            model_valid = (
                model_valid & jnp.all(tplus == -jnp.eye(ne)) & jnp.all(tminus == -jnp.eye(ne))
            )
            model_valid = model_valid & jnp.all(eindex < nr)
            model_valid = (
                model_valid & jnp.all(G[:N, orthant:, :ne] == 0.0) & jnp.all(G[-1, :, :ne] == 0.0)
            )
        else:
            epimask = jnp.zeros((K, m), dtype=bool)
            emap = jnp.zeros((N, 0, nv), dtype=H.dtype)

        def factor(scale):
            gh = cone_apply(scale, G, True)
            gh0 = jnp.where(epimask[..., None], 0.0, gh)
            hb = H[:, ne:, ne:] + gh0[:, :, ne:].swapaxes(-1, -2) @ gh0[:, :, ne:]
            if ne:
                weight = 1.0 / scale[0][:N] ** 2
                wp = jnp.take_along_axis(weight, plus, axis=1)
                wm = jnp.take_along_axis(weight, minus, axis=1)
                total = wp + wm
                invsum, ratio = 1.0 / total, (wm - wp) / total
                small, large = jnp.minimum(wp, wm), jnp.maximum(wp, wm)
                curvature = 4.0 * small / (1.0 + small / large)
                hb = hb.at[:N].add(jnp.einsum("kei,ke,kej->kij", emap, curvature, emap))
                eroot = jnp.sqrt(curvature)[..., None] * emap
            else:
                invsum, ratio = jnp.empty((N, 0)), jnp.empty((N, 0))
                eroot = jnp.empty((N, 0, nv))
            hb = _sym(hb)
            dp = D[:, :, ne:]

            def terminal_map(Cn, L, T, dk):
                U = Cn @ dk
                Z = _tri(L, U[:, :nr].T)
                Ck = U[:, nr:] - Z.T @ T.T
                return Ck, Z

            def classical(_):
                PN = hb[-1, nr:, nr:] + regularization * jnp.eye(nx)
                carry = (PN, Fn, jnp.zeros((nf, nf)), jnp.array(False))

                def backward(carry, data):
                    Pn, Cn, schur, poor = carry
                    hk, dk = data
                    M = _sym(hk + dk.T @ Pn @ dk) + regularization * eye
                    L = jnp.linalg.cholesky(M[:nr, :nr])
                    T = _tri(L, M[:nr, nr:]).T
                    Pk = _sym(M[nr:, nr:] - T @ T.T)
                    Ck, Z = terminal_map(Cn, L, T, dk)
                    poor = poor | ~jnp.all(jnp.isfinite(L)) | ~jnp.all(jnp.isfinite(Pk))
                    poor = poor | jnp.any(
                        jnp.diag(Pk) < -1e-8 * jnp.maximum(1.0, jnp.abs(jnp.diag(M[nr:, nr:])))
                    )
                    poor = poor | (
                        jnp.min(jnp.diag(L) ** 2, initial=jnp.inf)
                        < 1e-12 * jnp.max(jnp.abs(jnp.diag(M[:nr, :nr])), initial=0.0)
                    )
                    return (Pk, Ck, schur + Z.T @ Z, poor), (Pk, Ck, L, T, Z)

                end, arr = lax.scan(backward, carry, (hb[:-1], dp), reverse=True)
                P, C, L, T, Z = arr
                return (
                    jnp.concatenate((P, PN[None])),
                    jnp.concatenate((C, Fn[None])),
                    L,
                    T,
                    Z,
                    end[2],
                ), end[3]

            def square_root(_):
                initial = jnp.concatenate(
                    (roots[-1, :, nu:], gh0[-1, :, nu:], jnp.sqrt(regularization) * jnp.eye(nx)),
                    axis=0,
                )
                RN = jnp.linalg.qr(initial, mode="r")
                PN = RN.T @ RN
                carry = (RN, Fn, jnp.zeros((nf, nf)))

                def backward(carry, data):
                    Rn, Cn, schur = carry
                    root, gk, ek, dk = data
                    stack = jnp.concatenate(
                        (root, gk, ek, Rn @ dk, jnp.sqrt(regularization) * eye), axis=0
                    )
                    Rk = jnp.linalg.qr(stack, mode="r")
                    L, T, R = Rk[:nr, :nr].T, Rk[:nr, nr:].T, Rk[nr:, nr:]
                    Ck, Z = terminal_map(Cn, L, T, dk)
                    return (R, Ck, schur + Z.T @ Z), (R.T @ R, Ck, L, T, Z)

                end, arr = lax.scan(
                    backward, carry, (roots[:-1, :, ne:], gh0[:-1, :, ne:], eroot, dp), reverse=True
                )
                P, C, L, T, Z = arr
                return (
                    jnp.concatenate((P, PN[None])),
                    jnp.concatenate((C, Fn[None])),
                    L,
                    T,
                    Z,
                    end[2],
                )

            if riccati_mode == 2:
                P, C, L, T, Z, schur = square_root(None)
            else:
                classical_factors, poor = classical(None)
                if riccati_mode == 1:
                    P, C, L, T, Z, schur = classical_factors
                else:
                    P, C, L, T, Z, schur = lax.cond(
                        poor, square_root, lambda _: classical_factors, None
                    )
            SL = jnp.linalg.cholesky(_sym(schur) + 1e-16 * jnp.eye(nf))
            return P, C, L, T, Z, SL, invsum, ratio, gh

        def equality(fac, bd, d, target, fixed):
            P, C, L, T, Z, SL, invsum, ratio, gh = fac
            pN = -bd[-1, nu:]

            def sweep(carry, data):
                pn, work = carry
                dk, Dk, Pn, Cn, Lk, Tk, Zk, bdk, rk, ek = data
                linear = -bdk + Dk.T @ (pn + Pn @ dk)
                reduced = linear[ne:] + ek.T @ (rk * bdk[:ne])
                ak = _tri(Lk, reduced[:nr])
                pk = reduced[nr:] - Tk @ ak
                return (pk, work + Cn @ dk - Zk.T @ ak), (pk, ak)

            end, (pp, aa) = lax.scan(
                sweep,
                (pN, jnp.zeros(nf)),
                (d, D, P[1:], C[1:], L, T, Z, bd[:-1], ratio, emap),
                reverse=True,
            )
            pp = jnp.concatenate((pp, pN[None]))
            nt = _tri(SL, _tri(SL, end[1] + C[0] @ fixed - target), True)

            def forward(x, data):
                Dk, dk, Lk, Tk, Zk, ak, Pn, pn, Cn, bd0, inv, rat, ek = data
                ur = _tri(Lk, -ak - Tk.T @ x - Zk @ nt, True)
                reduced = jnp.concatenate((ur, x))
                epi = inv * bd0 - rat * (ek @ reduced)
                v = jnp.concatenate((epi, reduced))
                xn = Dk @ v + dk
                y = Pn @ xn + pn + Cn.T @ nt
                return xn, (v, y)

            xn, (v, y) = lax.scan(
                forward,
                fixed,
                (D, d, L, T, Z, aa, P[1:], pp[1:], C[1:], bd[:-1, :ne], invsum, ratio, emap),
            )
            v = jnp.concatenate((v, jnp.concatenate((jnp.zeros(nu), xn))[None]))
            return v, y, nt

        def stationarity(v, y, nt, z, affine=True):
            rd = _mv(H, v) + _mv(G.swapaxes(1, 2), z)
            if affine:
                rd += c
            rd = rd.at[:-1].add(_mv(D.swapaxes(1, 2), y))
            rd = rd.at[1:, nu:].add(-y).at[-1, nu:].add(Fn.T @ nt)
            return rd.at[0, nu:].set(0.0).at[-1, :nu].set(0.0)

        def residual(v, y, nt, s, z):
            rd = stationarity(v, y, nt, z)
            re = _mv(D, v[:-1]) + b - v[1:, nu:]
            rf = Fn @ v[-1, nu:] - fn
            rp = _mv(G, v) + s - h
            primal = jnp.maximum(
                jnp.maximum(_inf(re), _inf(F @ v[-1, nu:] - f)), jnp.maximum(_inf(rp), -cone_min(s))
            )
            dual = jnp.maximum(_inf(rd), -cone_min(z))
            gap = cone_pair(s, z)
            return (rd, re, rf, rp), jnp.stack((primal, dual, gap))

        identity = (
            jnp.ones((K, orthant)),
            tuple((jnp.ones(K), jnp.zeros((K, b - a - 1)), jnp.ones(K)) for a, b in offsets),
        )
        fac = factor(identity)
        v, y, nt = equality(fac, -c + _mv(G.swapaxes(1, 2), h), b, fn, x0)
        slack = h - _mv(G, v)
        s, z = cone_shift(slack), cone_shift(-slack)
        _, error = residual(v, y, nt, s, z)

        def prepare(values):
            v, y, nt, s, z = values
            (rd, re, rf, rp), _ = residual(*values)
            scale = scaling(s, z)
            lam = 0.5 * (cone_apply(scale, s, True) + cone_apply(scale, z))
            fac = factor(scale)
            return rd, re, rf, rp, scale, lam, fac

        def direction(context, values, q):
            rd, re, rf, rp, scale, lam, fac = context
            gh = fac[-1]

            def block(bd, de, tf, bc):
                pc = -cone_apply(scale, bc, True)
                dv, dy, dn = equality(fac, bd - _mv(gh.swapaxes(1, 2), pc), de, tf, jnp.zeros(nx))
                dz = cone_apply(scale, _mv(gh, dv) + pc, True)
                return dv, dy, dn, dz

            bc = -rp - cone_apply(scale, q)
            rhs_scales = jnp.stack((1 + _inf(rd), 1 + _inf(re), 1 + _inf(rf), 1 + _inf(bc)))

            def defect(values):
                dv, dy, dn, dz = values
                gv = _mv(G, dv)
                brd = -rd - stationarity(dv, dy, dn, dz, False)
                bre = re + _mv(D, dv[:-1]) - dv[1:, nu:]
                brf = -rf - Fn @ dv[-1, nu:]
                brc = bc - gv + cone_apply(scale, cone_apply(scale, dz))
                err = jnp.max(jnp.stack((_inf(brd), _inf(bre), _inf(brf), _inf(brc))) / rhs_scales)
                return (brd, bre, brf, brc), err

            values = block(-rd, re, -rf, bc)
            defects, err = defect(values)
            limit = refinement_limit

            def refine_cond(st):
                return (st[0] < refinement_steps) & (st[3] > limit) & st[4]

            def refine(st):
                count, values, defects, err, _ = st
                correction = block(*defects)
                candidate = tuple(v + d for v, d in zip(values, correction))
                next_defects, next_err = defect(candidate)
                improve = next_err < err
                chosen = tuple(jnp.where(improve, a, b) for a, b in zip(candidate, values))
                next_defects = tuple(
                    jnp.where(improve, a, b) for a, b in zip(next_defects, defects)
                )
                return count + 1, chosen, next_defects, jnp.where(improve, next_err, err), improve

            refined = lax.while_loop(
                refine_cond, refine, (jnp.array(0), values, defects, err, jnp.array(True))
            )
            dv, dy, dn, dz = refined[1]
            ds = -rp - _mv(G, dv)
            return (
                context,
                (dv, dy, dn, ds, dz),
                (jnp.array(0), refined[0]),
                jnp.isfinite(refined[3]),
            )

        def predictor_rhs(context, values):
            return -context[5]

        def corrector_rhs(context, values, affine, target):
            return corrector(context[4], context[5], affine[-2], affine[-1], target)

        values, run = run_ipm(
            (v, y, nt, s, z),
            valid=model_valid,
            cone_degree=K * (orthant + len(cones)),
            tolerance=tolerance,
            max_iterations=max_iterations,
            step_fraction=step_fraction,
            step_policy=step_policy,
            curved_cones=len(cones),
            evaluate=lambda values: residual(*values)[1],
            prepare=prepare,
            predictor_rhs=predictor_rhs,
            corrector_rhs=corrector_rhs,
            direction=direction,
            maximum_step=cone_step,
            cone_pair=cone_pair,
            cone_interior=lambda s, z: (cone_min(s) > 0.0) & (cone_min(z) > 0.0),
        )
        v, y, nt, s, z = values
        it, error, status = run["iterations"], run["residuals"], run["status"]
        physical_nu = solve_triangular(fr, nt, lower=False) if nf else nt
        return dict(
            x=v[:, nu:],
            u=v[:-1, :nu],
            y=y,
            nu=physical_nu,
            s=s,
            z=z,
            status=status,
            iterations=it,
            residuals=error,
            objective=0.5 * jnp.sum(v * _mv(H, v)) + jnp.sum(c * v),
        )

    return jax.jit(solve)
