"""Exact elimination of copy states and scalar absolute-value epigraphs.

This is a coefficient-based presolve. A nonmatching OCP returns ``None``;
no application names, model tags, or approximate zero tests select a kernel.
"""

from dataclasses import dataclass
import numpy as np


def _has_off_diagonal(matrix, diagonal=None):
    """Exact zero test without constructing a dense diagonal/identity array."""
    if diagonal is None:
        diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    return np.count_nonzero(matrix) != np.count_nonzero(diagonal)


def _chain_rejection_still_valid(arrays, witness):
    """Recheck one current coefficient; never reuse its old value.

    This witness only bypasses reduction. Complete finite validation may still
    be pending and remains mandatory before the native model is accepted.
    """
    if witness is None:
        return False
    key, shape, index = witness
    array = arrays[key]
    return array.shape == shape and array[index] != 0.0


def _chain_rejection_witness(arrays):
    """Retain only a key, shape and index proving exact reduction impossible.

    Canonical shapes are required, but finite validation may still be pending.
    A nonzero A entry or H off-diagonal rules out this reduction. It does not
    prove the model valid; native create/update must still check every original
    coefficient. No buffer, solved iterate or cached value is kept.
    """
    A, H = arrays["A"], arrays["H"]
    if A.flat[0] != 0.0:
        return "A", A.shape, (0, 0, 0)
    if H.shape[1] > 1 and H[0, 0, 1] != 0.0:
        return "H", H.shape, (0, 0, 1)
    if np.any(A):
        index = tuple(int(i) for i in np.unravel_index(np.argmax(A != 0.0), A.shape))
        return "A", A.shape, index
    if _has_off_diagonal(H):
        for stage in range(H.shape[0]):
            for row in range(H.shape[1]):
                columns = np.flatnonzero(H[stage, row])
                if columns.size:
                    column = int(columns[0])
                    if column == row:
                        if columns.size == 1:
                            continue
                        column = int(columns[1])
                    return "H", H.shape, (stage, row, column)
    return None


def _canonical(data):
    """Read either [u,x] H/c arrays or Q/R/S/q/r cost blocks."""
    if np.any(np.asarray(data["A"]) != 0.0):
        return None
    p = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in data.items()
        if key in ("H", "c", "Q", "R", "S", "q", "r", "A", "B", "b", "G", "h", "x0", "F", "f")
    }
    if p["B"].ndim != 3 or p["G"].ndim != 3:
        return None
    N, nx, nu = p["G"].shape[0] - 1, *p["B"].shape[1:]
    nv = nu + nx
    if N < 1 or nx < 1 or p["A"].shape not in ((N, nx, nx), (N + 1, nx, nx)):
        return None
    if np.any(p["A"][:N] != 0.0):
        return None
    for key, shape in (("B", (N, nx, nu)), ("b", (N, nx))):
        if p[key].shape not in (shape, (shape[0] + 1, *shape[1:])):
            return None
        p[key] = p[key][:N]
    p["A"] = p["A"][:N]
    p.setdefault("F", np.empty((0, nx)))
    p.setdefault("f", np.empty(0))
    if p["F"].shape != (0, nx) or p["f"].shape != (0,):
        return None
    if "H" not in p:
        shapes = (
            ("Q", (N + 1, nx, nx), False),
            ("q", (N + 1, nx), False),
            ("R", (N, nu, nu), True),
            ("S", (N, nu, nx), True),
            ("r", (N, nu), True),
        )
        for key, shape, extra in shapes:
            allowed = (shape, (shape[0] + 1, *shape[1:])) if extra else (shape,)
            if p[key].shape not in allowed:
                return None
        qr = np.diagonal(p["Q"], axis1=1, axis2=2)
        rr = np.diagonal(p["R"][:N], axis1=1, axis2=2)
        if (
            np.any(p["S"][:N] != 0.0)
            or _has_off_diagonal(p["Q"], qr)
            or _has_off_diagonal(p["R"][:N], rr)
        ):
            return None
        diagonal, c = np.zeros((N + 1, nv)), np.zeros((N + 1, nv))
        diagonal[:N, :nu], diagonal[:, nu:] = rr, qr
        c[:N, :nu], c[:, nu:] = p["r"][:N], p["q"]
        p["c"] = c
    else:
        if p["H"].shape != (N + 1, nv, nv):
            return None
        diagonal = np.diagonal(p["H"], axis1=1, axis2=2).copy()
        if _has_off_diagonal(p["H"], diagonal):
            return None
    m = p["G"].shape[1]
    for key, shape in (("c", (N + 1, nv)), ("G", (N + 1, m, nv)), ("h", (N + 1, m)), ("x0", (nx,))):
        if p[key].shape != shape:
            return None
    p = {key: p[key] for key in ("c", "A", "B", "b", "G", "h", "x0", "F", "f")}
    if not all(np.isfinite(v).all() for v in p.values()) or not np.isfinite(diagonal).all():
        return None
    if np.any(diagonal < 0.0):
        return None
    if (
        np.any(diagonal[N, :nu] != 0.0)
        or np.any(p["c"][N, :nu] != 0.0)
        or np.any(p["G"][N, :, :nu] != 0.0)
    ):
        return None
    return p, diagonal, N, nx, nu, m


@dataclass
class PresolvedChain:
    problem: dict
    original: dict
    diagonal: np.ndarray
    control_base: np.ndarray
    epigraph_map: np.ndarray
    row_map: np.ndarray
    constant_slack: np.ndarray
    objective_offset: float
    pattern: dict
    tolerance_scale: float

    def refresh(self, data, *, nonnegative=None, soc_sizes=None):
        """Re-read numerical arrays; return None if cached structure changed."""
        return try_chain_presolve(data, nonnegative=nonnegative, soc_sizes=soc_sizes, _cached=self)

    def decode(self, result, xp=np):
        """Restore the original OCP primal/dual variables, including fixed rows.

        ``xp=jax.numpy`` makes this affine reconstruction traceable; structure
        detection itself is a host operation performed when coefficients update.
        """
        p = {key: xp.asarray(value) for key, value in self.original.items()}
        Y, t = xp.asarray(result["stage_y"]), xp.asarray(result["epigraph"])
        N, nx, nu = p["B"].shape
        d = t.shape[1]
        padded_t = xp.concatenate((xp.zeros((1, d), dtype=Y.dtype), t))
        u = xp.asarray(self.control_base) + xp.einsum("kij,ki->kj", p["B"], Y)
        u += xp.einsum("kij,kj->ki", xp.asarray(self.epigraph_map), padded_t)
        x = xp.concatenate((p["x0"][None, :], Y))
        index = xp.asarray(self.row_map)
        mapped = []
        for name in ("s", "z"):
            flat = xp.concatenate(
                tuple(xp.ravel(value) for value in result[name]) + (xp.zeros(1, dtype=Y.dtype),)
            )
            selected = flat[xp.where(index >= 0, index, flat.size - 1)]
            fallback = xp.asarray(self.constant_slack) if name == "s" else 0.0
            mapped.append(xp.where(index >= 0, selected, fallback))
        s, z = mapped
        diagonal = xp.asarray(self.diagonal)
        y = diagonal[1:, nu:] * x[1:] + p["c"][1:, nu:]
        y += xp.einsum("kmi,km->ki", p["G"][1:, :, nu:], z[1:])
        v = xp.concatenate(
            (
                xp.concatenate((u, x[:-1]), axis=1),
                xp.concatenate((xp.zeros((1, nu)), x[-1:]), axis=1),
            )
        )
        rd = diagonal * v + p["c"] + xp.einsum("kmi,km->ki", p["G"], z)
        control_residual = rd[:-1, :nu] + xp.einsum("kij,ki->kj", p["B"], y)
        state_residual = rd[1:, nu:] - y
        dynamics = xp.einsum("kij,kj->ki", p["B"], u) + p["b"] - x[1:]
        inequalities = xp.einsum("kmi,ki->km", p["G"], v) + s - p["h"]

        def maximum(value):
            return xp.max(xp.abs(value), initial=0.0)

        primal = xp.maximum(maximum(dynamics), maximum(inequalities))
        primal = xp.maximum(primal, xp.maximum(0.0, xp.max(-s, initial=0.0)))
        dual = xp.maximum(maximum(control_residual), maximum(state_residual))
        dual = xp.maximum(dual, xp.maximum(0.0, xp.max(-z, initial=0.0)))
        gap = xp.sum(xp.abs(s * z))
        objective = xp.sum(0.5 * diagonal * v * v + p["c"] * v)
        info = dict(result.get("info", {}))
        info.update(objective=objective, primal_residual=primal, dual_residual=dual, gap=gap)
        decoded = dict(result)
        decoded.update(
            x=x,
            u=u,
            y=y,
            nu=xp.zeros(0),
            s=s,
            z=z,
            info=info,
            objective=objective,
            residuals=xp.stack((primal, dual, gap)),
        )
        return decoded


def try_chain_presolve(data, *, nonnegative=None, soc_sizes=None, _cached=None):
    """Return an exact copy-state reduction, or ``None`` for the general kernel.

    Accepted dynamics have A=0 and B a partial +1 permutation. Costs are
    diagonal; each uncopied control is either independent or a positive-cost
    affine |.| epigraph with two opposite rows. Every interior stage has the
    same number of such epigraphs. No terminal equalities or SOCs are removed.
    """
    dims = data.get("soc_dims", ()) if soc_sizes is None else soc_sizes
    if np.asarray(dims).size:
        return None
    try:
        parsed = _canonical(data)
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if parsed is None:
        return None
    p, diagonal, N, nx, nu, m = parsed
    nlp = data.get("nonnegative", m) if nonnegative is None else nonnegative
    if nlp != m:
        return None
    B, G, c, h, b = (p[key] for key in ("B", "G", "c", "h", "b"))
    cache = _cached.pattern if _cached is not None else None
    if cache is not None and (
        G.shape != _cached.original["G"].shape or not np.array_equal(B, cache["B"])
    ):
        return None
    if (
        np.any((B != 0.0) & (B != 1.0))
        or np.any(np.sum(B != 0.0, axis=1) > 1)
        or np.any(np.sum(B != 0.0, axis=2) > 1)
    ):
        return None
    free = np.sum(B, axis=2)
    fixed = b * (1.0 - free)
    base = -np.einsum("kij,ki->kj", B, b)
    unused = np.all(B == 0.0, axis=1)
    if cache is None:
        presence = G[:-1, :, :nu] != 0.0
        counts = np.count_nonzero(presence, axis=1)
        independent = unused & (counts == 0)
        epi = unused & ~independent
    else:
        nk, nj = cache["null_index"]
        if not np.array_equal(np.count_nonzero(G[nk, :, nj], axis=1), cache["null_count"]):
            return None
        independent, epi, counts = cache["independent"], cache["epi"], cache["counts"]
    stage_counts = np.count_nonzero(epi, axis=1)
    d = int(stage_counts[1]) if N > 1 else 0
    if (
        stage_counts[0] != 0
        or np.any(stage_counts[1:] != d)
        or np.any(counts[epi] != 2)
        or np.any(diagonal[:-1, :nu][epi] != 0.0)
        or np.any(c[:-1, :nu][epi] <= 0.0)
        or np.any(independent & (diagonal[:-1, :nu] == 0.0) & (c[:-1, :nu] != 0.0))
    ):
        return None
    unconstrained = np.divide(
        -c[:-1, :nu], diagonal[:-1, :nu], out=np.zeros_like(base), where=diagonal[:-1, :nu] > 0.0
    )
    base = np.where(independent, unconstrained, base)
    constant = 0.5 * np.dot(diagonal[0, nu:] * p["x0"], p["x0"]) + c[0, nu:] @ p["x0"]
    constant += np.sum(
        np.where(independent, 0.5 * diagonal[:-1, :nu] * base * base + c[:-1, :nu] * base, 0.0)
    )
    stages, controls = np.nonzero(epi) if cache is None else cache["epi_index"]
    if d:
        first = np.argmax(presence, axis=1)[stages, controls] if cache is None else cache["first"]
        last = (
            m - 1 - np.argmax(presence[:, ::-1], axis=1)[stages, controls]
            if cache is None
            else cache["last"]
        )
        alpha = -G[stages, first, controls]
        if np.any(alpha <= 0.0) or np.any(G[stages, last, controls] != -alpha):
            return None
        total = G[stages, first] + G[stages, last]
        total[np.arange(len(stages)), controls] = 0.0
        occupied = (
            np.concatenate((stages * m + first, stages * m + last)) if cache is None else None
        )
        if np.any(total != 0.0) or (cache is None and len(np.unique(occupied)) != len(occupied)):
            return None
    else:
        first, last, alpha = np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0)
    if cache is None:
        used = np.zeros((N + 1, m), dtype=bool)
        used[stages, first], used[stages, last] = True, True
    else:
        used = cache["used"]
    rdiag = np.einsum("kij,kj->ki", B, diagonal[:-1, :nu])
    rlinear = np.einsum("kij,kj->ki", B, c[:-1, :nu])
    hessian = rdiag + diagonal[1:, nu:]
    linear = rlinear - rdiag * b + c[1:, nu:]
    if np.any(hessian[free != 0.0] <= 0.0):
        return None
    constant += np.sum((0.5 * diagonal[1:, nu:] * fixed + c[1:, nu:]) * fixed)
    constant += np.sum(0.5 * rdiag * b * b - rlinear * b)
    hessian = np.where(free != 0.0, hessian, 1.0)
    linear = np.where(free != 0.0, linear, 0.0)
    # Transform every row in one batch. B is a selection, so gather columns
    # instead of multiplying a dense matrix containing mostly zeros.
    previous_free = np.concatenate((np.zeros((1, nx)), free))
    previous_fixed = np.concatenate((p["x0"][None, :], fixed))
    L = G[:, :, nu:] * previous_free[:, None, :]
    R = np.zeros_like(L)
    if cache is None:
        segments = []
        for k in range(N):
            indices, columns = np.nonzero(B[k])
            starts = np.r_[0, np.flatnonzero((np.diff(indices) != 1) | (np.diff(columns) != 1)) + 1]
            for a, z in zip(starts, np.r_[starts[1:], len(indices)]):
                if a < z:
                    segments.append((k, int(indices[a]), int(indices[z - 1]) + 1, int(columns[a])))
    else:
        segments = cache["segments"]
    for k, a, z, j in segments:
        R[k, :, a:z] = G[k, :, j : j + z - a]
    values = -h + np.einsum("kmi,ki->km", G[:, :, nu:], previous_fixed)
    values[:-1] += np.einsum("kmj,kj->km", G[:-1, :, :nu], base)
    left = L[stages, first].reshape(N - 1, d, nx)
    right = R[stages, first].reshape(N - 1, d, nx)
    offset = (0.5 * (values[stages, first] - values[stages, last])).reshape(N - 1, d)
    weight = (c[stages, controls] / alpha).reshape(N - 1, d)
    base[stages, controls] = 0.5 * (values[stages, first] + values[stages, last]) / alpha
    epimap = np.zeros((N, nu, d))
    epi_number = np.tile(np.arange(d), N - 1)
    epimap[stages, controls, epi_number] = 1.0 / alpha
    constant += np.sum(c[stages, controls] * base[stages, controls])
    row_map = np.full((N + 1, m), -1, dtype=np.int64)
    nd = (N - 1) * d
    row_map[stages, first], row_map[stages, last] = np.arange(nd), nd + np.arange(nd)
    have_left, have_right = np.any(L != 0.0, axis=2), np.any(R != 0.0, axis=2)
    constant_rows = ~used & ~have_left & ~have_right
    if np.any(values[constant_rows] > 0.0):
        return None
    constant_slack = np.where(constant_rows, -values, 0.0)
    pair_rows = ~used & have_left & have_right
    local_left = ~used & have_left & ~have_right
    local_right = ~used & ~have_left & have_right
    if cache is not None and (
        not np.array_equal(have_left, cache["have_left"])
        or not np.array_equal(have_right, cache["have_right"])
    ):
        return None
    npair = int(np.max(np.sum(pair_rows, axis=1), initial=0))
    nlocal = int(
        np.max(np.sum(local_left[1:], axis=1) + np.sum(local_right[:-1], axis=1), initial=0)
    )
    PL, PR = np.zeros((N - 1, npair, nx)), np.zeros((N - 1, npair, nx))
    PO = -np.ones((N - 1, npair))
    LM, LO = np.zeros((N, nlocal, nx)), -np.ones((N, nlocal))
    if cache is None:
        for k in range(1, N):
            rows = np.flatnonzero(pair_rows[k])
            count = len(rows)
            row_map[k, rows] = 2 * nd + (k - 1) * npair + np.arange(count)
        for k in range(N):
            rr, rl = np.flatnonzero(local_right[k]), np.flatnonzero(local_left[k + 1])
            nr, nl = len(rr), len(rl)
            start = 2 * nd + (N - 1) * npair + k * nlocal
            row_map[k, rr], row_map[k + 1, rl] = start + np.arange(nr), start + nr + np.arange(nl)
        pk, pr = np.nonzero(pair_rows)
        pi = (row_map[pk, pr] - 2 * nd) % npair if npair else np.empty(0, dtype=int)
        lk, lr = np.nonzero(local_left | local_right)
        ls = local_left[lk, lr]
        node = np.where(ls, lk - 1, lk)
        li = (
            (row_map[lk, lr] - 2 * nd - (N - 1) * npair) % nlocal
            if nlocal
            else np.empty(0, dtype=int)
        )
        null_index = np.nonzero(unused)
        cache = dict(
            B=B.copy(),
            independent=independent,
            epi=epi,
            counts=counts,
            null_index=null_index,
            null_count=counts[null_index],
            epi_index=(stages, controls),
            first=first,
            last=last,
            segments=segments,
            used=used,
            have_left=have_left,
            have_right=have_right,
            pair_index=(pk, pr, pi),
            local_index=(lk, lr, node, li, ls),
            row_map=row_map,
        )
    else:
        row_map = cache["row_map"]
        pk, pr, pi = cache["pair_index"]
        lk, lr, node, li, ls = cache["local_index"]
    PL[pk - 1, pi], PR[pk - 1, pi], PO[pk - 1, pi] = L[pk, pr], R[pk, pr], values[pk, pr]
    LM[node, li] = np.where(ls[:, None], L[lk, lr], R[lk, lr])
    LO[node, li] = values[lk, lr]
    problem = dict(
        left=left,
        right=right,
        offset=offset,
        pair_left=PL,
        pair_right=PR,
        pair_offset=PO,
        local_matrix=LM,
        local_offset=LO,
        center=fixed,
        hessian=hessian,
        linear=linear,
        l1_weight=weight,
        free=free,
        fixed=fixed,
    )
    if not (
        np.isfinite(constant)
        and all(np.isfinite(v).all() for v in problem.values())
        and np.isfinite(base).all()
        and np.isfinite(epimap).all()
    ):
        return None
    # The native workspace owns its coefficients. Keep the reconstruction at
    # the same snapshot even if the caller later mutates the input arrays.
    original = {key: np.array(value, copy=True, order="C") for key, value in p.items()}
    # All restored residuals preserve units except epigraph stationarity,
    # whose residual is alpha times the reduced one. Floating-point effects
    # are checked again in original coordinates by the public solver.
    scale = max(1.0, float(np.max(alpha, initial=0.0)))
    return PresolvedChain(
        problem,
        original,
        diagonal,
        base,
        epimap,
        row_map,
        constant_slack,
        float(constant),
        cache,
        scale,
    )


def infer_epigraph_controls(arrays, nonnegative):
    """Return the largest safe leading prefix, using exact coefficients.

    Inputs have canonical shapes; complete finite validation may be deferred
    to the native validator. This recognizer does not eliminate entries. Each
    inferred t is absent from H and B and appears only in the orthant rows
    -t+e <= 0 and -t-e <= 0. Its partner e is a retained control appearing
    in no other inequality. No application metadata is consulted.
    """
    H, B, G, h, c = (arrays[key] for key in ("H", "B", "G", "h", "c"))
    N, _, nu = B.shape
    if nu < 2 or nonnegative < 2 or H[0, 0, 0] != 0.0 or np.any(H[:-1, 0, 0] != 0.0):
        return 0
    # Terminal control columns are padding in the generic OCP API.
    if (
        np.any(H[-1, :nu, :] != 0.0)
        or np.any(H[-1, :, :nu] != 0.0)
        or np.any(c[-1, :nu] != 0.0)
        or np.any(G[-1, :, :nu] != 0.0)
    ):
        return 0
    partners = [set() for _ in range(N)]
    minimum_partner = nu
    for i in range(nu // 2):
        if (
            minimum_partner <= i
            or np.any(H[:-1, i, :] != 0.0)
            or np.any(H[:-1, :, i] != 0.0)
            or np.any(B[:, :, i] != 0.0)
        ):
            return i
        new_partners = []
        for k in range(N):
            rows = np.flatnonzero(G[k, :, i])
            if (
                len(rows) != 2
                or rows[-1] >= nonnegative
                or np.any(G[k, rows, i] != -1.0)
                or np.any(h[k, rows] != 0.0)
            ):
                return i
            a, b = map(int, rows)
            ca, cb = np.flatnonzero(G[k, a]), np.flatnonzero(G[k, b])
            if len(ca) != 2 or len(cb) != 2 or not np.array_equal(ca, cb):
                return i
            j = int(ca[ca != i][0])
            if (
                j <= i
                or j >= nu
                or j in partners[k]
                or abs(G[k, a, j]) != 1.0
                or G[k, b, j] != -G[k, a, j]
                or np.count_nonzero(G[k, :, j]) != 2
            ):
                return i
            new_partners.append(j)
        for k, j in enumerate(new_partners):
            partners[k].add(j)
            minimum_partner = min(minimum_partner, j)
    return nu // 2
