/* Cold initialization and the default native options.
 * A cached initializer belongs to the current coefficients and factorization options.
 * It is never the previous solved iterate; every solve remains a cold solve.
 * Private implementation: included exactly once by ocp.c. */

/* Retain the exact auxiliary-QP factor, separately from any IPM iterate.
 * Coefficient changes invalidate the initializer; right-hand-side changes
 * retain only the auxiliary factor. Storage is never allocated during solve. */
/* The live allocations retain their established layout. An optional aligned
 * auxiliary bank holds the other factor. Exchanges move only slot pointers;
 * its owning base pointer never changes, and destruction restores the original
 * live pointers before releasing the individual allocations. */
static void auxiliary_factor_storage_create(icore_ocp_workspace *w)
{
    size_t N = w->N, K = w->K, nx = w->nx, nr = w->nr, nf = w->nf, ne = w->ne, count = 0, i;
    size_t sizes[9] = {N * nr * nr, N * nx * nr, K * nx * nx, K * nf * nx, N * nr * nf,
                       nf * nf,     nf * nf,     N * ne,      N * ne};
    double **bank[9] = {&w->aux_L, &w->aux_T,  &w->aux_P,       &w->aux_C,        &w->aux_Z,
                        &w->aux_S, &w->aux_SL, &w->aux_epi_inv, &w->aux_epi_ratio};
    double *next;
    int swappable = !w->ocp_la && !w->qp_eligible;
    for (i = 0; i < 9; ++i) {
        if (swappable) {
            if (sizes[i] > SIZE_MAX - 7u)
                return;
            sizes[i] = sizes[i] ? (sizes[i] + 7u) & ~(size_t)7u : 8u;
        }
        if (sizes[i] > SIZE_MAX - count)
            return;
        count += sizes[i];
    }
    if (count > (SIZE_MAX - 63u) / sizeof(double))
        return;
    next = (double *)(swappable ? array_new(count, sizeof(double))
                                : array_overwritten(count, sizeof(double)));
    if (!next)
        return;
    w->aux_factor_storage = next;
    w->aux_factor_bytes = (count * sizeof(double) + 63u) & ~(size_t)63u;
    w->aux_factor_swappable = swappable;
    for (i = 0; i < 9; ++i) {
        *bank[i] = next;
        next += sizes[i];
    }
}

static void auxiliary_factor_exchange(icore_ocp_workspace *w)
{
    double **live[9] = {&w->L, &w->T,  &w->P,       &w->C,        &w->Z,
                        &w->S, &w->SL, &w->epi_inv, &w->epi_ratio};
    double **bank[9] = {&w->aux_L, &w->aux_T,  &w->aux_P,       &w->aux_C,        &w->aux_Z,
                        &w->aux_S, &w->aux_SL, &w->aux_epi_inv, &w->aux_epi_ratio};
    for (int i = 0; i < 9; ++i) {
        double *p = *live[i];
        *live[i] = *bank[i];
        *bank[i] = p;
    }
    w->factor_bank_exchanged = !w->factor_bank_exchanged;
}

static void auxiliary_factor_preserve(icore_ocp_workspace *w)
{
    if (w->aux_factor_live) {
        auxiliary_factor_exchange(w);
        w->aux_factor_live = 0;
    }
}

static size_t auxiliary_factor_copy(icore_ocp_workspace *w, int save, int packed)
{
    size_t N = (size_t)w->N, K = (size_t)w->K, nx = (size_t)w->nx, nr = (size_t)w->nr;
    size_t nf = (size_t)w->nf, ne = (size_t)w->ne, i, bytes = 0;
    const size_t sizes[9] = {N * nr * nr, N * nx * nr, K * nx * nx, K * nf * nx, N * nr * nf,
                             nf * nf,     nf * nf,     N * ne,      N * ne};
    double *live[9] = {w->L, w->T, w->P, w->C, w->Z, w->S, w->SL, w->epi_inv, w->epi_ratio};
    double *bank[9] = {w->aux_L, w->aux_T,  w->aux_P,       w->aux_C,        w->aux_Z,
                       w->aux_S, w->aux_SL, w->aux_epi_inv, w->aux_epi_ratio};
    for (i = 0; i < 9; ++i)
        if (!(packed && (i == 2 || (i < 2 && icore_ocp_la_sqrt_owns_factors(w->ocp_la))))) {
            size_t part = sizes[i] * sizeof(double);
            memcpy(save ? bank[i] : live[i], save ? live[i] : bank[i], part);
            bytes += part;
        }
    return bytes;
}

static void auxiliary_factor_capture(icore_ocp_workspace *w, const icore_ocp_options *opt, int row_scaled)
{
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    size_t bytes;
    w->aux_factor_valid = 0;
    if (!w->aux_factor_storage)
        return;
    if (w->classical_initial_zero ||
        (w->ocp_sqrt_active && icore_ocp_la_sqrt_initial_zero_active(w->ocp_la)))
        return;
    if (w->ocp_sqrt_active && icore_ocp_la_sqrt_checkpoint(w->ocp_la) != ICORE_LA_OK)
        return;
    bytes = w->aux_factor_swappable ? 0 : auxiliary_factor_copy(w, 1, w->ocp_sqrt_active);
    w->aux_factor_live = w->aux_factor_swappable;
    if (w->ocp_sqrt_active)
        bytes += icore_ocp_la_sqrt_checkpoint_copy_bytes(w->ocp_la);
    w->aux_factor_mode = opt->riccati_mode;
    w->aux_factor_row_scaled = row_scaled;
    w->aux_factor_reg = opt->regularization;
    w->aux_factor_sqrt = w->ocp_sqrt_active;
    w->aux_factor_poor = w->factor_poor;
    w->aux_factor_copy_bytes = bytes;
    w->aux_factor_valid = 1;
    ++w->aux_factor_captures;
}

static int auxiliary_factor_restore(icore_ocp_workspace *w, const icore_ocp_options *opt, int row_scaled)
{
    w->classical_initial_zero = 0;
    w->mixed_sweeps_active = 0;
    icore_ocp_la_mixed_sweeps_invalidate(w->ocp_la);
    w->mixed_factor_active = 0;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
    w->orthant_roots_lazy = 0;
    size_t bytes;
    if (!w->aux_factor_valid || w->aux_factor_mode != opt->riccati_mode ||
        w->aux_factor_reg != opt->regularization || w->aux_factor_row_scaled != row_scaled)
        return 0;
    if (w->aux_factor_sqrt &&
        (icore_ocp_la_sqrt_restore(w->ocp_la) != ICORE_LA_OK || icore_ocp_la_sqrt_initial_zero_active(w->ocp_la))) {
        w->aux_factor_valid = 0;
        return 0;
    }
    bytes = w->aux_factor_swappable ? 0 : auxiliary_factor_copy(w, 0, w->aux_factor_sqrt);
    if (w->aux_factor_swappable && !w->aux_factor_live) {
        auxiliary_factor_exchange(w);
        w->aux_factor_live = 1;
    }
    if (w->aux_factor_sqrt)
        bytes += icore_ocp_la_sqrt_checkpoint_copy_bytes(w->ocp_la);
    w->reg = fmax(0.0, opt->regularization);
    w->ocp_sqrt_active = w->aux_factor_sqrt;
    w->factor_poor = w->aux_factor_poor;
    w->sweep_pd_valid = 0;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->aux_factor_copy_bytes = bytes;
    ++w->aux_factor_restores;
    return 1;
}

/* A cold, infeasible-start initialization for orthant OCPs consisting only
 * of single-variable bounds. It uses actual coefficients and takes the
 * tightest lower/upper bound when multiple rows constrain one variable.
 * The scalar floor is in variable units; slack and dual are mapped back to
 * the original row units. General rows, SOCs, epigraphs and terminal
 * equalities retain the auxiliary-QP initializer. No solved iterate is used.
 * If finite input cannot represent this initialization, the caller falls
 * back to the auxiliary problem before entering the IPM iteration. */
static int initialize_single_variable_bounds(icore_ocp_workspace *w)
{
    const double floor = 0.1, mu = 1.0;
    size_t nvcount = (size_t)w->K * w->nv, mcount = (size_t)w->K * w->m;
    int k, i, j;
    if (w->nsoc || w->ne || w->nf || mcount <= (size_t)w->constant_count)
        return 0;
    for (k = 0; k < w->K; ++k)
        for (i = 0; i < w->m; ++i) {
            size_t row = (size_t)k * w->m + i;
            if (!w->constant_orthant[row] && w->nnz[row] != 1)
                return 0;
        }
    memset(w->v, 0, nvcount * sizeof(double));
    memset(w->y, 0, (size_t)w->N * w->nx * sizeof(double));
    memcpy(w->v + w->nu, w->x0, (size_t)w->nx * sizeof(double));
    for (k = 0; k < w->K; ++k) {
        double *v = w->v + (size_t)k * w->nv;
        int start = k == w->N ? w->nu : 0;
        int end = k == 0 ? w->nu : w->nv;
        for (j = 0; j < w->nv; ++j) {
            w->tmp[j] = -INFINITY;
            w->tmp2[j] = INFINITY;
        }
        for (i = 0; i < w->m; ++i) {
            size_t row = (size_t)k * w->m + i;
            double a, bound;
            if (w->constant_orthant[row])
                continue;
            j = w->single_index[row];
            a = w->single_value[row];
            bound = w->h[row] / a;
            if (!isfinite(bound))
                return 0;
            if (a > 0.0) {
                if (bound < w->tmp2[j])
                    w->tmp2[j] = bound;
                else if (bound == w->tmp2[j])
                    w->tmp2[j] = fmin(w->tmp2[j], bound);
            } else {
                if (bound > w->tmp[j])
                    w->tmp[j] = bound;
                else if (bound == w->tmp[j])
                    w->tmp[j] = fmax(w->tmp[j], bound);
            }
        }
        for (j = start; j < end; ++j) {
            double lower = w->tmp[j], upper = w->tmp2[j];
            if (-lower < floor) {
                if (upper < floor)
                    v[j] = 0.5 * lower + 0.5 * upper;
                else
                    v[j] = lower + floor;
            } else if (upper < floor)
                v[j] = upper - floor;
            if (!isfinite(v[j]))
                return 0;
        }
        for (i = 0; i < w->m; ++i) {
            size_t row = (size_t)k * w->m + i;
            double a, slack, minimum;
            if (w->constant_orthant[row]) {
                w->s[row] = w->z[row] = 1.0;
                continue;
            }
            j = w->single_index[row];
            a = w->single_value[row];
            minimum = fabs(a) * floor;
            slack = w->h[row] - a * v[j];
            if (!(minimum > 0.0) || !isfinite(minimum) || !isfinite(slack))
                return 0;
            w->s[row] = slack > minimum ? slack : minimum;
            w->z[row] = mu / w->s[row];
            if (!(w->z[row] > 0.0) || !isfinite(w->z[row]))
                return 0;
        }
    }
    return 1;
}

/* Limit large constraint coefficients only in the auxiliary QP. Small rows
 * are left alone: amplifying them can make a loose constraint dominate the
 * cold start. With d=max(1,maxabs(g)), use G/d and h/d for initialization;
 * all IPM residuals and returned values retain the original row units. */
static int initialize_row_scaling(icore_ocp_workspace *w)
{
    size_t row, count = (size_t)w->K * w->m;
    double ceiling = sqrt(DBL_MAX);
    int changed = 0, i;
    if (w->nsoc || w->ne)
        return 0;
    /* Proposals use dead scratch. Commit no scale before the complete check. */
    for (row = 0; row < count; ++row) {
        const double *g = w->G + row * w->nv;
        const int *index = w->indices + w->index_start[row];
        double scale = 1.0, scaled_h, coefficient;
        int entries = w->nnz[row];
        if (entries && index[entries - 1] - index[0] + 1 == entries) {
            double magnitude = norm_inf(g + index[0], (size_t)entries);
            if (magnitude > scale)
                scale = magnitude;
        } else {
            /* Canonical coefficients have already passed finite validation.
             * Positive magnitudes need no libm fmax call per sparse entry. */
            for (i = 0; i < entries; ++i) {
                double magnitude = fabs(g[index[i]]);
                if (magnitude > scale)
                    scale = magnitude;
            }
        }
        if (!(scale <= ceiling))
            return 0;
        scaled_h = w->h[row] / scale;
        coefficient = scaled_h / scale;
        if (!isfinite(scaled_h) || !isfinite(coefficient) ||
            (w->h[row] != 0.0 && (scaled_h == 0.0 || coefficient == 0.0)))
            return 0;
        w->lambda[row] = scale;
        w->bcone[row] = coefficient;
        changed |= scale > 1.0;
    }
    if (!changed)
        return 0;
    for (row = 0; row < count; ++row) {
        w->boost[row] = w->lambda[row];
        w->orthant_inv[row] = 1.0 / w->lambda[row];
    }
    return 1;
}

static int initialize_with_row_scaling(icore_ocp_workspace *w, const icore_ocp_options *opt, int normalize,
                                       int *used_scaling)
{
    w->classical_initial_zero = 0;
    w->mixed_sweeps_active = 0;
    icore_ocp_la_mixed_sweeps_invalidate(w->ocp_la);
    w->mixed_factor_active = 0;
    int k, i, j, normalized = 0;
    size_t vcount = (size_t)w->K * w->nv, ycount = (size_t)w->N * w->nx;
    size_t mcount = (size_t)w->K * w->m, fcount = (size_t)w->nf;
    if (used_scaling)
        *used_scaling = 0;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
    w->orthant_direct = 0;
    w->orthant_roots_lazy = 0;
    /* A restored initial point has no current Newton factor. In particular,
     * a short zero-direction chain cannot become an auxiliary factor. */
    w->ocp_sqrt_active = 0;
    w->sweep_pd_valid = 0;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    /* This cache contains only the cold initializer for these coefficients.
     * A solved IPM iterate is never saved here. Coefficient changes and
     * factorization-option changes invalidate this entry. */
    if (w->initial_valid && w->initial_mode == opt->riccati_mode &&
        w->initial_reg == opt->regularization) {
        memcpy(w->v, w->initial_v, vcount * sizeof(double));
        memcpy(w->y, w->initial_y, ycount * sizeof(double));
        memcpy(w->nu_t, w->initial_nu, fcount * sizeof(double));
        memcpy(w->s, w->initial_s, mcount * sizeof(double));
        memcpy(w->z, w->initial_z, mcount * sizeof(double));
        return 1;
    }
    w->initial_valid = 0;
    if (initialize_single_variable_bounds(w))
        goto store_initial;
    scaling_identity(w);
    normalized = normalize && initialize_row_scaling(w);
    if (used_scaling)
        *used_scaling = normalized;
    if (!auxiliary_factor_restore(w, opt, normalized)) {
        if (!factor_problem(w, opt))
            return 0;
        auxiliary_factor_capture(w, opt, normalized);
    }
    for (k = 0; k < w->K; ++k) {
        const double *G = w->G + (size_t)k * w->m * w->nv;
        double *rhs = w->rhs + (size_t)k * w->nv;
        for (i = 0; i < w->nv; ++i)
            rhs[i] = -w->c[(size_t)k * w->nv + i];
        const double *target = (normalized ? w->bcone : w->h) + (size_t)k * w->m;
        if (constraint_transpose_grouped(w, k))
            constraint_transpose_add(w, k, G, target, rhs);
        else
            for (j = 0; j < w->m; ++j)
                supported_axpy(w, (size_t)k * w->m + j, rhs, G + (size_t)j * w->nv, target[j],
                               w->single_value);
    }
    if (!equality_solve(w, w->rhs, w->b, w->f, w->x0, w->v, w->y, w->nu_t))
        return 0;
    for (k = 0; k < w->K; ++k) {
        const double *G = w->G + (size_t)k * w->m * w->nv;
        for (i = 0; i < w->m; ++i) {
            size_t p = (size_t)k * w->m + i;
            w->s[p] = w->h[p] - supported_dot(w, (size_t)k * w->m + i, G + (size_t)i * w->nv,
                                              w->v + (size_t)k * w->nv, w->single_value);
            if (normalized)
                w->s[p] *= w->orthant_inv[p];
            w->z[p] = -w->s[p];
        }
    }
    cone_shift(w, w->s);
    cone_shift(w, w->z);
    if (normalized)
        for (size_t row = 0; row < mcount; ++row) {
            w->s[row] *= w->boost[row];
            w->z[row] *= w->orthant_inv[row];
            /* A large cone shift can cancel to zero in floating point. */
            if (!(w->s[row] > 0.0) || !(w->z[row] > 0.0))
                return 0;
        }
    if (!finite_array(w->s, mcount) || !finite_array(w->z, mcount))
        return 0;
store_initial:
    memcpy(w->initial_v, w->v, vcount * sizeof(double));
    memcpy(w->initial_y, w->y, ycount * sizeof(double));
    memcpy(w->initial_nu, w->nu_t, fcount * sizeof(double));
    memcpy(w->initial_s, w->s, mcount * sizeof(double));
    memcpy(w->initial_z, w->z, mcount * sizeof(double));
    w->initial_mode = opt->riccati_mode;
    w->initial_reg = opt->regularization;
    w->initial_valid = 1;
    return 1;
}

static int initialize(icore_ocp_workspace *w, const icore_ocp_options *opt)
{
    int used_scaling;
    if (w->nsoc || w->ne)
        return initialize_with_row_scaling(w, opt, 0, NULL);
    if (initialize_with_row_scaling(w, opt, 1, &used_scaling))
        return 1;
    if (!used_scaling)
        return 0;
    /* Report the extra auxiliary factorization in the existing retry count.
     * Recovery retains the original initializer and the same IPM budget. */
    ++w->factor_retries;
    w->initial_valid = 0;
    w->aux_factor_valid = 0;
    return initialize_with_row_scaling(w, opt, 0, NULL);
}

icore_ocp_options icore_ocp_default_options(void)
{
    icore_ocp_options o;
    o.max_iterations = 200;
    o.refinement_steps = 5;
    o.riccati_mode = 0;
    o.tolerance = 1e-7;
    o.step_fraction = 0.99;
    o.regularization = 1e-11;
    o.refinement_tolerance = 0.0;
    return o;
}
