/* Equality sweeps, original-system residuals, and Newton correction.
 * Predictor and corrector share the current factorization.
 * Refinement checks the original equations before accepting a correction.
 * Direction-product tokens are consumed once and cleared before incompatible writes.
 * Private implementation: included exactly once by ocp.c. */

/* Add D' x using exact column supports; the existing output is the
 * accumulation seed, preserving the row order of the dense AXPY path. */
static void sparse_dynamics_transpose_add_columns(const icore_ocp_workspace *w, int stage,
                                                  const double *x, double *out, int end)
{
    int j, a, nv = w->nv, nx = w->nx;
    for (j = w->ne; j < end; ++j) {
        size_t column = (size_t)stage * nv + j;
        const int *index = w->dyn_index + column * nx;
        const double *value = w->dyn_value + column * nx;
        int count = w->dyn_count[column];
        double sum = out[j];
        if (count == 3) {
            sum += value[0] * x[index[0]];
            sum += value[1] * x[index[1]];
            sum += value[2] * x[index[2]];
        } else if (count == 2) {
            sum += value[0] * x[index[0]];
            sum += value[1] * x[index[1]];
        } else
            for (a = 0; a < count; ++a)
                sum += value[a] * x[index[a]];
        out[j] = sum;
    }
}

static void sparse_dynamics_transpose_add(const icore_ocp_workspace *w, int stage, const double *x,
                                          double *out)
{
    sparse_dynamics_transpose_add_columns(w, stage, x, out, w->nv);
}

/* Solve a fixed factored equality QP.  bd is the stationarity RHS;
 * x_next=D v+d, F x_N=target.  The fixed initial state can be nonzero.
 * The terminal dual returned here is always in the original F row units. */
/* SP-only equality dispatcher. Failure returns to ocp_direction, where the
 * full DP chain and canonical RHS are rebuilt at the same iterate. No stage
 * can fall through to an ordinary helper that consumes double C. */
static int mixed_equality_forward(icore_ocp_workspace *w, const double *bd, const double *d,
                                  const double *target, const double *fixed, double *v, double *y,
                                  double *nu)
{
    int k, cache_rhs = d == w->re;
    (void)bd;
    (void)target;
    (void)nu;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
    if (!w->mixed_sweeps_active || fixed || w->nf || w->ne || w->nsoc ||
        !icore_ocp_la_mixed_sweeps_active(w->ocp_la))
        return 0;
    memset(v + w->nu, 0, (size_t)w->nx * sizeof(double));
    memset(v + (size_t)w->N * w->nv, 0, (size_t)w->nu * sizeof(double));
    for (k = 0; k < w->N; ++k) {
        double *vk = v + (size_t)k * w->nv;
        ++w->mixed_sweep_counts[3];
        if (icore_ocp_la_mixed_sweeps_forward(
                w->ocp_la, k, w->sweep_a + (size_t)k * w->nr, k ? vk + w->nu : NULL,
                d + (size_t)k * w->nx, w->sweep_p + (size_t)(k + 1) * w->nx, vk,
                v + (size_t)(k + 1) * w->nv + w->nu, y + (size_t)k * w->nx) != ICORE_LA_OK) {
            ++w->mixed_sweep_counts[7];
            return 0;
        }
    }
    if (cache_rhs)
        w->sweep_pd_valid = 1;
    /* The SP dynamics product is approximate. It never supplies the cache
     * consumed by the original-DP residual computation. */
    w->dynamics_cache_direction = NULL;
    return 1;
}

static int mixed_equality_solve(icore_ocp_workspace *w, const double *bd, const double *d,
                                const double *target, const double *fixed, double *v, double *y,
                                double *nu)
{
    int k, i, cache_rhs = d == w->re, cached = cache_rhs && w->sweep_pd_valid;
    double *pN = w->sweep_p + (size_t)w->N * w->nx;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
    if (!w->mixed_sweeps_active || fixed || w->nf || w->ne || w->nsoc ||
        !icore_ocp_la_mixed_sweeps_active(w->ocp_la))
        return 0;
    for (i = 0; i < w->nx; ++i)
        pN[i] = -bd[(size_t)w->N * w->nv + w->nu + i];
    for (k = w->N - 1; k >= 0; --k) {
        ++w->mixed_sweep_counts[2];
        if (icore_ocp_la_mixed_sweeps_backward(w->ocp_la, k, bd + (size_t)k * w->nv, d + (size_t)k * w->nx,
                                      w->sweep_p + (size_t)(k + 1) * w->nx, cached,
                                      cache_rhs ? w->sweep_pd + (size_t)k * w->nx : NULL,
                                      w->sweep_a + (size_t)k * w->nr,
                                      k ? w->sweep_p + (size_t)k * w->nx : NULL, 0) != ICORE_LA_OK) {
            ++w->mixed_sweep_counts[7];
            return 0;
        }
    }
    return mixed_equality_forward(w, bd, d, target, fixed, v, y, nu);
}

static int equality_forward(icore_ocp_workspace *w, const double *bd, const double *d,
                            const double *target, const double *fixed, double *v, double *y,
                            double *nu)
{
    if (w->mixed_sweeps_active)
        return mixed_equality_forward(w, bd, d, target, fixed, v, y, nu);
    int k, i, j, a, nx = w->nx, nr = w->nr, nv = w->nv, nf = w->nf;
    int initial_zero = w->ocp_sqrt_active && icore_ocp_la_sqrt_initial_zero_active(w->ocp_la);
    int classical_initial_zero = !w->ocp_sqrt_active && w->classical_initial_zero;
    int packed_sweeps = w->ocp_sqrt_active && !fixed && !w->ne && icore_ocp_la_sqrt_owns_factors(w->ocp_la);
    int packed_forward_complete = packed_sweeps;
    int cache_rhs = d == w->re && !fixed;
    /* A packed allocation may be unused, or only some stages may use it.
     * The row-dot producer below certifies its own entries independently. */
    int cache_row_major = ICORE_OCP_DYNAMICS_CACHE && cache_rhs && v == w->dv && !packed_sweeps;
    int cache_dynamics = cache_row_major || (ICORE_OCP_DYNAMICS_CACHE && packed_sweeps && cache_rhs &&
                                             v == w->dv && icore_ocp_la_dynamics_cache_available());
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
    if (initial_zero && (fixed || nf || w->ne))
        return 0;
    if (classical_initial_zero && fixed)
        return 0;
    for (a = 0; a < nf; ++a) {
        w->term_work[a] -= target[a] / w->Fnorm[a];
        if (fixed)
            w->term_work[a] += dot(w->C + (size_t)a * nx, fixed, nx);
    }
    triangular_forward(w->SL, w->term_work, nf);
    triangular_backward(w->SL, w->term_work, nf);
    for (a = 0; a < nf; ++a)
        nu[a] = w->term_work[a] / w->Fnorm[a];
    if (packed_sweeps && !finite_array(nu, (size_t)nf))
        return 0;
    if (fixed) {
        /* The auxiliary initializer may consume a failed partial sweep. */
        memset(v, 0, (size_t)w->K * nv * sizeof(double));
        memcpy(v + w->nu, fixed, (size_t)nx * sizeof(double));
    } else {
        /* Successful stages replace every other state and control entry. */
        memset(v + w->nu, 0, (size_t)nx * sizeof(double));
        memset(v + (size_t)w->N * nv, 0, (size_t)w->nu * sizeof(double));
    }
    for (k = 0; k < w->N; ++k) {
        const double *D = w->D + (size_t)k * nx * nv, *L = w->L + (size_t)k * nr * nr;
        const double *T = w->T + (size_t)k * nx * nr, *ak = w->sweep_a + (size_t)k * nr;
        const double *Z = w->Z + (size_t)k * nr * nf, *Cn = w->C + (size_t)(k + 1) * nf * nx;
        const double *Pn = w->P + (size_t)(k + 1) * nx * nx,
                     *pn = w->sweep_p + (size_t)(k + 1) * nx;
        double *vk = v + (size_t)k * nv, *next = v + (size_t)(k + 1) * nv + w->nu;
        double *yk = y + (size_t)k * nx;
        if (packed_sweeps) {
            const double *effective_a = ak;
            if (nf) {
                /* Solve the same bordered system. Fold Z*lambda into the
                 * control RHS before the fused triangular/dynamics pass.
                 * Dual terminal corrections are added afterwards, retaining
                 * the original pn + Pn*x + Cn'*lambda summation order. */
                for (i = 0; i < nr; ++i)
                    w->tmp3[i] = ak[i] + dot(Z + (size_t)i * nf, w->term_work, nf);
                effective_a = w->tmp3;
            }
            int status = cache_dynamics && k > 0
                             ? icore_ocp_la_packed_forward_stage_cached(
                                   w->ocp_la, k, effective_a, vk + w->nu, d + (size_t)k * nx, pn,
                                   vk, next, yk, w->bre + (size_t)k * nx)
                             : icore_ocp_la_packed_forward_stage(w->ocp_la, k, effective_a,
                                                        k == 0 && initial_zero ? NULL : vk + w->nu,
                                                        d + (size_t)k * nx, pn, vk, next, yk);
            if (status == ICORE_LA_OK) {
                for (i = 0; i < nx; ++i)
                    for (a = 0; a < nf; ++a)
                        yk[i] += Cn[(size_t)a * nx + i] * w->term_work[a];
                if (nf && !finite_array(yk, (size_t)nx))
                    return 0;
                continue;
            }
            /* No caller output changed, so ordinary stage recovery sees
             * the same state and RHS. Its final whole-vector check remains. */
            packed_forward_complete = 0;
            cache_dynamics = 0;
        }
        for (i = 0; i < nr; ++i)
            vk[w->ne + i] = -ak[i] - dot(Z + (size_t)i * nf, w->term_work, nf);
        if (k == 0 && initial_zero) {
            if (icore_ocp_la_initial_zero_sweep(w->ocp_la, 1, vk, vk) != ICORE_LA_OK)
                return 0;
        } else if (w->ocp_sqrt_active && icore_ocp_la_sqrt_owns_factors(w->ocp_la)) {
            if (icore_ocp_la_sweep_forward(w->ocp_la, k, vk + w->ne, vk + w->nu, vk + w->ne) != ICORE_LA_OK)
                return 0;
        } else {
            if (!(k == 0 && classical_initial_zero))
                for (j = 0; j < nx; ++j)
                    axpy(vk + w->ne, T + (size_t)j * nr, -vk[w->nu + j], nr);
            triangular_backward(L, vk + w->ne, nr);
        }
        for (i = 0; i < w->ne; ++i) {
            size_t epos = (size_t)k * w->ne + i;
            vk[i] =
                w->epi_inv[epos] * bd[(size_t)k * nv + i] - w->epi_ratio[epos] * vk[w->epi_e[epos]];
        }
        if (k == 0 && initial_zero) {
            if (icore_ocp_la_initial_zero_dynamics_product(w->ocp_la, 0, 1.0, vk, 0.0, next) == ICORE_LA_OK)
                for (i = 0; i < nx; ++i)
                    next[i] += d[i];
            else
                for (i = 0; i < nx; ++i)
                    next[i] = dot(D + (size_t)i * nv, vk, nr) + d[i];
        } else if (!w->ne && w->ocp_packed_ready[k] &&
                   icore_ocp_la_dynamics_product(w->ocp_la, k, 0, 1.0, vk, 0.0, next) == ICORE_LA_OK) {
            for (i = 0; i < nx; ++i)
                next[i] += d[(size_t)k * nx + i];
        } else if (row_dots_grouped(nx, nv)) {
            double *product = cache_row_major && k > 0 ? w->bre + (size_t)k * nx : next;
            row_dots(D, nx, nv, vk, nv, product);
            for (i = 0; i < nx; ++i)
                next[i] = product[i] + d[(size_t)k * nx + i];
        } else
            for (i = 0; i < nx; ++i) {
                double product = dot(D + (size_t)i * nv, vk, nv);
                /* The first original-system residual uses this exact product.
                 * Cache before adding d: recovering it from next-d loses bits. */
                if (cache_row_major && k > 0)
                    w->bre[(size_t)k * nx + i] = product;
                next[i] = product + d[(size_t)k * nx + i];
            }
        if (w->ocp_sqrt_active && icore_ocp_la_sqrt_apply(w->ocp_la, k + 1, next, yk) != ICORE_LA_OK)
            return 0;
        if (!w->ocp_sqrt_active && !row_dots_grouped(nx, nx)) {
            for (i = 0; i < nx; ++i) {
                double value =
                    pn[i] + (w->ocp_sqrt_active ? yk[i] : dot(Pn + (size_t)i * nx, next, nx));
                for (a = 0; a < nf; ++a)
                    value += Cn[(size_t)a * nx + i] * w->term_work[a];
                yk[i] = value;
            }
        } else {
            if (!w->ocp_sqrt_active)
                row_dots(Pn, nx, nx, next, nx, yk);
            for (i = 0; i < nx; ++i) {
                double value = pn[i] + yk[i];
                for (a = 0; a < nf; ++a)
                    value += Cn[(size_t)a * nx + i] * w->term_work[a];
                yk[i] = value;
            }
        }
    }
    /* Every successful fused forward stage certifies its controls, next
     * state and dual before committing. Initial state and terminal control
     * padding came from the zero fill above. Terminal duals and their
     * forward corrections have separate finite checks.
     * Any ordinary-stage fallback retains the original complete scan. */
    if (!packed_forward_complete &&
        (!finite_array(v, (size_t)w->K * nv) || !finite_array(y, (size_t)w->N * nx) ||
         !finite_array(nu, (size_t)nf)))
        return 0;
    if (cache_rhs)
        w->sweep_pd_valid = 1;
    if (cache_dynamics && (packed_forward_complete || cache_row_major)) {
        w->dynamics_cache_direction = v;
        w->dynamics_cache_kind = cache_row_major ? ICORE_OCP_DYNAMICS_ROW_MAJOR : ICORE_OCP_DYNAMICS_PACKED;
    }
    return 1;
}

static int equality_solve(icore_ocp_workspace *w, const double *bd, const double *d,
                          const double *target, const double *fixed, double *v, double *y,
                          double *nu)
{
    if (w->mixed_sweeps_active)
        return mixed_equality_solve(w, bd, d, target, fixed, v, y, nu);
    int k, i, j, a;
    int nx = w->nx, nr = w->nr, nv = w->nv, nf = w->nf;
    int initial_zero = w->ocp_sqrt_active && icore_ocp_la_sqrt_initial_zero_active(w->ocp_la);
    int classical_initial_zero = !w->ocp_sqrt_active && w->classical_initial_zero;
    int packed_sweeps = w->ocp_sqrt_active && !fixed && !w->ne && icore_ocp_la_sqrt_owns_factors(w->ocp_la);
    /* re is the owned nonlinear dynamics residual: unchanged
     * across the predictor/corrector pair. Refinement uses bre;
     * initialization uses b and a nonzero fixed-state input.
     * Both a residual refresh and a new factor invalidate it. */
    int cache_rhs = d == w->re && !fixed;
    int cached = cache_rhs && w->sweep_pd_valid;
    double *pN = w->sweep_p + (size_t)w->N * nx;
    /* A short stage-zero factor is valid only for this Newton boundary.
     * This check also prevents a stale/custom checkpoint from being used by
     * the nonzero auxiliary initializer or terminal/epigraph algebra. */
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
    if (initial_zero && (fixed || nf || w->ne))
        return 0;
    if (classical_initial_zero && fixed)
        return 0;
    for (i = 0; i < nx; ++i)
        pN[i] = -bd[(size_t)w->N * nv + w->nu + i];
    memset(w->term_work, 0, (size_t)nf * sizeof(double));
    for (k = w->N - 1; k >= 0; --k) {
        const double *D = w->D + (size_t)k * nx * nv, *Pn = w->P + (size_t)(k + 1) * nx * nx;
        const double *pn = w->sweep_p + (size_t)(k + 1) * nx, *dk = d + (size_t)k * nx;
        const double *L = w->L + (size_t)k * nr * nr, *T = w->T + (size_t)k * nx * nr;
        double *ak = w->sweep_a + (size_t)k * nr, *pk = w->sweep_p + (size_t)k * nx;
        double *pd = w->sweep_pd + (size_t)k * nx;
        if (packed_sweeps &&
            icore_ocp_la_packed_backward_stage(w->ocp_la, k, bd + (size_t)k * nv, dk, pn, cached,
                                      cache_rhs ? pd : NULL, ak,
                                      k == 0 && initial_zero ? NULL : pk) == ICORE_LA_OK) {
            if (nf) {
                const double *Cn = w->C + (size_t)(k + 1) * nf * nx;
                const double *Z = w->Z + (size_t)k * nr * nf;
                for (a = 0; a < nf; ++a) {
                    double value = dot(Cn + (size_t)a * nx, dk, nx);
                    for (i = 0; i < nr; ++i)
                        value -= Z[(size_t)i * nf + a] * ak[i];
                    w->term_work[a] += value;
                }
            }
            continue;
        }
        /* A rejected fused call has not written Pd/a/p. Keep the complete
         * established stage path, including its scalar-product fallback. */
        if (cached) {
            for (i = 0; i < nx; ++i)
                w->tmp[i] = pn[i] + pd[i];
        } else if (w->ocp_sqrt_active) {
            double *product = cache_rhs ? pd : w->tmp;
            if (icore_ocp_la_sqrt_apply(w->ocp_la, k + 1, dk, product) != ICORE_LA_OK)
                return 0;
            for (i = 0; i < nx; ++i)
                w->tmp[i] = pn[i] + product[i];
        } else if (row_dots_grouped(nx, nx)) {
            double *product = cache_rhs ? pd : w->tmp;
            row_dots(Pn, nx, nx, dk, nx, product);
            for (i = 0; i < nx; ++i)
                w->tmp[i] = pn[i] + product[i];
        } else
            for (i = 0; i < nx; ++i) {
                double product = dot(Pn + (size_t)i * nx, dk, nx);
                if (cache_rhs)
                    pd[i] = product;
                w->tmp[i] = pn[i] + product;
            }
        if (k == 0 && initial_zero) {
            for (i = 0; i < nr; ++i)
                w->tmp3[i] = -bd[i];
            if (w->dyn_sparse[0])
                sparse_dynamics_transpose_add_columns(w, 0, w->tmp, w->tmp3, nr);
            else if (icore_ocp_la_initial_zero_dynamics_product(w->ocp_la, 1, 1.0, w->tmp, 1.0, w->tmp3) !=
                     ICORE_LA_OK)
                dense_transpose_add(D, nx, nv, w->tmp, w->tmp3, nr);
            if (icore_ocp_la_initial_zero_sweep(w->ocp_la, 0, w->tmp3, ak) != ICORE_LA_OK)
                return 0;
            /* p0 has no predecessor and nf=0, so no consumer exists. */
            continue;
        }
        int columns = k == 0 && classical_initial_zero ? (nr + 3) / 4 * 4 : nv;
        if (columns > nv)
            columns = nv;
        for (i = 0; i < columns; ++i)
            w->tmp3[i] = -bd[(size_t)k * nv + i];
        if (w->dyn_sparse[k])
            sparse_dynamics_transpose_add_columns(w, k, w->tmp, w->tmp3, columns);
        else if (columns < nv || w->ne || !w->ocp_packed_ready[k] ||
                 icore_ocp_la_dynamics_product(w->ocp_la, k, 1, 1.0, w->tmp, 1.0, w->tmp3) != ICORE_LA_OK)
            dense_transpose_add(D, nx, nv, w->tmp, w->tmp3, columns);
        for (i = 0; i < nr; ++i)
            ak[i] = w->tmp3[w->ne + i];
        for (i = 0; i < w->ne; ++i) {
            size_t epos = (size_t)k * w->ne + i;
            ak[w->epi_e[epos] - w->ne] += w->epi_ratio[epos] * bd[(size_t)k * nv + i];
        }
        if (w->ocp_sqrt_active && icore_ocp_la_sqrt_owns_factors(w->ocp_la)) {
            if (icore_ocp_la_sweep_backward(w->ocp_la, k, ak, w->tmp3 + w->nu, ak, pk) != ICORE_LA_OK)
                return 0;
        } else {
            triangular_forward(L, ak, nr);
            if (!(k == 0 && classical_initial_zero)) {
                if (row_dots_grouped(nx, nr)) {
                    row_dots(T, nx, nr, ak, nr, pk);
                    for (i = 0; i < nx; ++i)
                        pk[i] = w->tmp3[w->nu + i] - pk[i];
                } else
                    for (i = 0; i < nx; ++i)
                        pk[i] = w->tmp3[w->nu + i] - dot(T + (size_t)i * nr, ak, nr);
            }
        }
        if (nf) {
            const double *Cn = w->C + (size_t)(k + 1) * nf * nx, *Z = w->Z + (size_t)k * nr * nf;
            for (a = 0; a < nf; ++a) {
                double value = dot(Cn + (size_t)a * nx, dk, nx);
                for (i = 0; i < nr; ++i)
                    value -= Z[(size_t)i * nf + a] * ak[i];
                w->term_work[a] += value;
            }
        }
    }
    return equality_forward(w, bd, d, target, fixed, v, y, nu);
}

static double stationarity(const icore_ocp_workspace *w, const double *v, const double *y,
                           const double *nu, const double *z, double *out, int affine,
                           double *dynamics, int cached_dynamics)
{
    int k, i, j, a;
    int nx = w->nx, nv = w->nv;
    double objective = 0.0;
    for (k = 0; k < w->K; ++k) {
        const double *H = w->H + (size_t)k * nv * nv, *G = w->G + (size_t)k * w->m * nv;
        const double *vk = v + (size_t)k * nv, *zk = z + (size_t)k * w->m;
        double *r = out + (size_t)k * nv;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
        /* In a committed model, this root-storage flag was installed only
         * after exact original-H diagonal certification. An invalid inactive
         * bank may retain zero root storage but cannot certify its current H.
         * Use canonical H, including accepted tiny negative diagonals. */
        if (w->coefficient_cache_valid && w->cost_root_diagonal[k])
            for (i = 0; i < nv; ++i) {
                size_t row = (size_t)k * nv + i;
                double coefficient = H[(size_t)i * nv + i], product = 0.0;
                /* A zero-length dot must not evaluate 0*Inf or 0*NaN. */
                if (coefficient != 0.0)
                    product += coefficient * vk[i];
                r[i] = product + (affine ? w->c[row] : 0.0);
                if (affine)
                    objective += 0.5 * vk[i] * product + w->c[row] * vk[i];
            }
        else
#endif
            for (i = 0; i < nv; ++i) {
                size_t row = (size_t)k * nv + i;
                int start = w->cost_start[row], length = w->cost_length[row];
                double product = dot(H + (size_t)i * nv + start, vk + start, length);
                r[i] = product + (affine ? w->c[row] : 0.0);
                if (affine)
                    objective += 0.5 * vk[i] * product + w->c[row] * vk[i];
            }
        j = 0;
#if ICORE_OCP_SINGLETON_AXPY4
        if (w->ocp_diagonal_cost_ready)
            for (; w->m - j >= 4; j += 4) {
                size_t row = (size_t)k * w->m + j;
                if (!singleton_axpy4(w, row, r, _mm256_loadu_pd(zk + j))) {
                    int lane;
                    for (lane = 0; lane < 4; ++lane)
                        supported_axpy(w, row + (size_t)lane, r, G + (size_t)(j + lane) * nv,
                                       zk[j + lane], w->single_value);
                }
            }
#endif
        if (j == 0 && constraint_transpose_grouped(w, k))
            constraint_transpose_add(w, k, G, zk, r);
        else
            for (; j < w->m; ++j)
                supported_axpy(w, (size_t)k * w->m + j, r, G + (size_t)j * nv, zk[j],
                               w->single_value);
        if (k < w->N) {
            const double *D = w->D + (size_t)k * nx * nv, *yk = y + (size_t)k * nx;
            double *dk = dynamics + (size_t)k * nx;
            /* A row-dot token covers exactly stages taking the row-major
             * producer. Packed GEMV_T and GEMV_NT need not round identically,
             * so a classical sweep's packed stages still recompute D*v. */
            int cached_row_major = cached_dynamics && k > 0 &&
                                   w->dynamics_cache_kind == ICORE_OCP_DYNAMICS_ROW_MAJOR &&
                                   (w->ne || !w->ocp_packed_ready[k]);
            int cached = cached_dynamics && k > 0 &&
                         w->dynamics_cache_kind == ICORE_OCP_DYNAMICS_PACKED && w->ocp_la &&
                         icore_ocp_la_dynamics_nt_add(w->ocp_la, k, yk, r) == ICORE_LA_OK;
            if (!cached && (w->ne || !w->ocp_packed_ready[k] ||
                            icore_ocp_la_dynamics_pair(w->ocp_la, k, vk, yk, r, dk) != ICORE_LA_OK)) {
                if (w->dyn_sparse[k])
                    sparse_dynamics_transpose_add(w, k, yk, r);
                else if (w->ne || !w->ocp_packed_ready[k] ||
                         icore_ocp_la_dynamics_product(w->ocp_la, k, 1, 1.0, yk, 1.0, r) != ICORE_LA_OK)
                    dense_transpose_add(D, nx, nv, yk, r, nv);
                if (!cached_row_major &&
                    (w->ne || !w->ocp_packed_ready[k] ||
                     icore_ocp_la_dynamics_product(w->ocp_la, k, 0, 1.0, vk, 0.0, dk) != ICORE_LA_OK))
                    for (j = 0; j < nx; ++j)
                        dk[j] = dot(D + (size_t)j * nv, vk, nv);
            }
        }
        if (k > 0)
            for (i = 0; i < nx; ++i)
                r[w->nu + i] -= y[(size_t)(k - 1) * nx + i];
        if (k == w->N)
            for (a = 0; a < w->nf; ++a)
                axpy(r + w->nu, w->F + (size_t)a * nx, nu[a], nx);
    }
    for (i = 0; i < nx; ++i)
        out[w->nu + i] = 0.0;
    for (i = 0; i < w->nu; ++i)
        out[(size_t)w->N * nv + i] = 0.0;
    return objective;
}

/* Removed cone outputs are s=z=0. Their original row equations therefore
 * have residual G_original*v (the certified h is exactly zero). Evaluate at
 * the actual approximate state: a small F residual can be amplified by G. */
/* Keep optional original-model verification outside the ordinary residual
 * code body. Compilers without an outlining attribute retain identical math. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
static double
constant_soc_original_defect(const icore_ocp_workspace *w)
{
    double result = 0.0;
    if (!w->constant_soc_count)
        return result;
    for (int k = 0; k < w->K; ++k)
        for (int c = 0; c < w->nsoc; ++c) {
            size_t offset = constant_soc_offset(w, k, c);
            if (offset == SIZE_MAX)
                continue;
            const double *G = w->constant_soc_G + offset * w->nv;
            const double *v = w->v + (size_t)k * w->nv;
            for (int i = 0; i < w->sizes[c]; ++i) {
                double value = fabs(dot(G + (size_t)i * w->nv, v, w->nv));
                if (!isfinite(value))
                    return INFINITY;
                result = fmax(result, value);
            }
        }
    return result;
}

static void residuals(icore_ocp_workspace *w, icore_ocp_info *info)
{
    int k, i;
    int nx = w->nx, nv = w->nv;
    double primal = w->constant_primal_defect, objective = 0.0;
    double smin, zmin, gap, rp_norm;
    w->sweep_pd_valid = 0;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
    objective = stationarity(w, w->v, w->y, w->nu_t, w->z, w->rd, 1, w->re, 0);
    for (k = 0; k < w->N; ++k)
        for (i = 0; i < nx; ++i) {
            size_t p = (size_t)k * nx + i;
            w->re[p] = (w->re[p] + w->b[p]) - w->v[(size_t)(k + 1) * nv + w->nu + i];
        }
    for (i = 0; i < w->nf; ++i)
        w->rf[i] = dot(w->F + (size_t)i * nx, w->v + (size_t)w->N * nv + w->nu, nx) - w->f[i];
#if ICORE_OCP_SINGLETON_GV4
    if (w->ocp_diagonal_cost_ready)
        rp_norm = singleton_primal_residuals(w);
    else
#endif
    {
        for (k = 0; k < w->K; ++k) {
            const double *G = w->G + (size_t)k * w->m * nv, *v = w->v + (size_t)k * nv;
            for (i = 0; i < w->m; ++i)
                w->rp[(size_t)k * w->m + i] =
                    supported_dot(w, (size_t)k * w->m + i, G + (size_t)i * nv, v, w->single_value) +
                    w->s[(size_t)k * w->m + i] - w->h[(size_t)k * w->m + i];
        }
        rp_norm = norm_inf(w->rp, (size_t)w->K * w->m);
    }
    cone_statistics(w, &smin, &zmin, &gap);
    w->rhs_norm[0] = norm_inf(w->rd, (size_t)w->K * nv);
    w->rhs_norm[1] = norm_inf(w->re, (size_t)w->N * nx);
    w->rhs_norm[2] = norm_inf(w->rf, (size_t)w->nf);
    primal = fmax(primal, w->rhs_norm[1]);
    primal = fmax(primal, w->rhs_norm[2]);
    primal = fmax(primal, rp_norm);
    if (w->constant_soc_count)
        primal = fmax(primal, constant_soc_original_defect(w));
    primal = fmax(primal, -smin);
    info->primal_residual = primal;
    info->dual_residual = fmax(w->rhs_norm[0], -zmin);
    info->gap = gap;
    info->objective = objective;
}

/* Solve [H E' G'; E 0 0; G 0 -W^2] for arbitrary original RHS.
 * p=-W^-1 bc; the reduced stationarity RHS is bd-Ghat' p. */
static void form_reduced_rhs(icore_ocp_workspace *w, const double *bd, const double *bc,
                             int direct_cone_rhs)
{
    int k, i, nv = w->nv;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
    memcpy(w->rhs, bd, (size_t)w->K * nv * sizeof(double));
    w->orthant_gv_direction = NULL;
    if (w->orthant_direct) {
        for (k = 0; k < w->K; ++k) {
            const double *G = w->G + (size_t)k * w->m * nv;
            double *p = w->pcone + (size_t)k * w->m;
            for (i = 0; i < w->m; ++i) {
                size_t row = (size_t)k * w->m + i;
                p[i] = -w->lambda[row] * bc[row];
                supported_axpy(w, row, w->rhs + (size_t)k * nv, G + (size_t)i * nv, -p[i],
                               w->single_value);
            }
        }
        return;
    }
    for (k = 0; k < w->K; ++k) {
        const double *Gh = w->Gh + (size_t)k * w->m * nv;
        double *p = w->pcone + (size_t)k * w->m;
        if (direct_cone_rhs) {
            cone_apply(w, k, w->rp + (size_t)k * w->m, p, 1);
            for (i = 0; i < w->m; ++i) {
                p[i] += w->q[(size_t)k * w->m + i];
                supported_axpy(w, (size_t)k * w->m + i, w->rhs + (size_t)k * nv,
                               Gh + (size_t)i * nv, -p[i], w->single_scaled);
            }
        } else {
            cone_apply(w, k, bc + (size_t)k * w->m, p, 1);
            for (i = 0; i < w->m; ++i) {
                p[i] = -p[i];
                supported_axpy(w, (size_t)k * w->m + i, w->rhs + (size_t)k * nv,
                               Gh + (size_t)i * nv, -p[i], w->single_scaled);
            }
        }
    }
}

static int recover_cone_dual(icore_ocp_workspace *w, const double *v, double *z)
{
    int k, j, nv = w->nv;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    if (w->orthant_direct) {
        int valid = 1;
        w->orthant_gv_direction = NULL;
        for (k = 0; k < w->K; ++k) {
            const double *G = w->G + (size_t)k * w->m * nv;
            const double *vk = v + (size_t)k * nv;
            j = 0;
#if ICORE_OCP_SINGLETON_GV4
            if (w->ocp_diagonal_cost_ready) {
                const __m256d sign = _mm256_set1_pd(-0.0), limit = _mm256_set1_pd(DBL_MAX);
                for (; w->m - j >= 4; j += 4) {
                    size_t row = (size_t)k * w->m + j;
                    __m256d gv = singleton_gv4(w, row, vk);
                    __m256d zi = _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(w->lambda + row), gv),
                                               _mm256_loadu_pd(w->pcone + row));
                    _mm256_storeu_pd(z + row, zi);
                    _mm256_storeu_pd(w->pcone + row, gv);
                    valid &= _mm256_movemask_pd(_mm256_cmp_pd(_mm256_andnot_pd(sign, zi), limit,
                                                              _CMP_LE_OQ)) == 15;
                }
            }
#endif
            for (; j < w->m; ++j) {
                size_t row = (size_t)k * w->m + j;
                double gv = supported_dot(w, row, G + (size_t)j * nv, vk, w->single_value);
                z[row] = w->lambda[row] * gv + w->pcone[row];
                /* Preserve original Gv for the next residual check only. */
                w->pcone[row] = gv;
                valid &= isfinite(z[row]);
            }
        }
        if (!valid) {
            w->dynamics_cache_direction = NULL;
            return 0;
        }
        w->orthant_gv_direction = v;
        return 1;
    }
    for (k = 0; k < w->K; ++k) {
        const double *Gh = w->Gh + (size_t)k * w->m * nv;
        const double *vk = v + (size_t)k * nv;
        if (!w->nsoc) {
            const double *inverse = w->orthant_inv + (size_t)k * w->nlp;
            for (j = 0; j < w->m; ++j) {
                double value = supported_dot(w, (size_t)k * w->m + j, Gh + (size_t)j * nv, vk,
                                             w->single_scaled) +
                               w->pcone[(size_t)k * w->m + j];
                z[(size_t)k * w->m + j] = value * inverse[j];
            }
        } else {
            for (j = 0; j < w->m; ++j)
                w->tmp[j] = supported_dot(w, (size_t)k * w->m + j, Gh + (size_t)j * nv, vk,
                                          w->single_scaled) +
                            w->pcone[(size_t)k * w->m + j];
            cone_apply(w, k, w->tmp, z + (size_t)k * w->m, 1);
        }
    }
    if (!finite_array(z, (size_t)w->K * w->m)) {
        w->dynamics_cache_direction = NULL;
        return 0;
    }
    return 1;
}

static int block_solve(icore_ocp_workspace *w, const double *bd, const double *d, const double *tf,
                       const double *bc, int direct_cone_rhs, double *v, double *y, double *nu,
                       double *z)
{
    form_reduced_rhs(w, bd, bc, direct_cone_rhs);
    if (!equality_solve(w, w->rhs, d, tf, NULL, v, y, nu))
        return 0;
    return recover_cone_dual(w, v, z);
}

/* Original, unregularized three-block residual; each block has its own scale.
 * RHS norms are unchanged during refinement and cached when each RHS is made.
 * Equality residual storage uses x_next-Dv, matching equality_solve's input.
 * Slack recovery reuses this same Gv, including when a trial is accepted. */
static double block_residual(icore_ocp_workspace *w, const double *v, const double *y, const double *nu,
                             const double *z, double *ds)
{
    int k, i;
    int nx = w->nx, nv = w->nv;
    double md = 0.0, me = 0.0, mf = 0.0, mc = 0.0;
    int reuse_gv = w->orthant_direct && w->orthant_gv_direction == v;
    int reuse_dynamics = w->dynamics_cache_direction == v && v == w->dv;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->orthant_gv_direction = NULL; /* consume a one-use owned-direction token */
    w->dynamics_cache_direction = NULL;
    stationarity(w, v, y, nu, z, w->brd, 0, w->bre, reuse_dynamics);
    for (k = 0; k < w->K; ++k)
        for (i = 0; i < nv; ++i) {
            size_t p = (size_t)k * nv + i;
            w->brd[p] = -w->rd[p] - w->brd[p];
        }
    for (k = 0; k < w->N; ++k)
        for (i = 0; i < nx; ++i) {
            size_t p = (size_t)k * nx + i;
            w->bre[p] = (w->re[p] + w->bre[p]) - v[(size_t)(k + 1) * nv + w->nu + i];
        }
    for (i = 0; i < w->nf; ++i)
        w->brf[i] = -w->rf[i] - dot(w->F + (size_t)i * nx, v + (size_t)w->N * nv + w->nu, nx);
#if ICORE_OCP_SINGLETON_GV4
    if (reuse_gv)
        mc = direct_cached_residual(w, z, ds);
    else
#endif
    {
        for (k = 0; k < w->K; ++k) {
            const double *G = w->G + (size_t)k * w->m * nv, *vk = v + (size_t)k * nv;
            if (w->orthant_direct) {
                const double *delta = w->boost + (size_t)k * w->m;
                const double *zk = z + (size_t)k * w->m;
                for (i = 0; i < w->m; ++i)
                    w->tmp2[i] = zk[i] * delta[i];
            } else if (!w->nsoc) {
                const double *boost = w->boost + (size_t)k * w->m;
                const double *zk = z + (size_t)k * w->m;
                /* Preserve the two successive products, including their range;
                 * pre-squaring boost would change overflow/underflow behavior. */
                for (i = 0; i < w->m; ++i)
                    w->tmp2[i] = (zk[i] * boost[i]) * boost[i];
            } else {
                cone_apply(w, k, z + (size_t)k * w->m, w->tmp, 0);
                cone_apply(w, k, w->tmp, w->tmp2, 0);
            }
            for (i = 0; i < w->m; ++i) {
                size_t p = (size_t)k * w->m + i;
                double gv = reuse_gv ? w->pcone[p]
                                     : supported_dot(w, p, G + (size_t)i * nv, vk, w->single_value);
                w->brc[p] = w->bcone[p] - gv + w->tmp2[i];
                ds[p] = -w->rp[p] - gv;
            }
        }
        mc = norm_inf(w->brc, (size_t)w->K * w->m);
    }
    md = norm_inf(w->brd, (size_t)w->K * nv) / (1.0 + w->rhs_norm[0]);
    me = norm_inf(w->bre, (size_t)w->N * nx) / (1.0 + w->rhs_norm[1]);
    mf = norm_inf(w->brf, (size_t)w->nf) / (1.0 + w->rhs_norm[2]);
    mc /= 1.0 + w->rhs_norm[3];
    return fmax(fmax(md, me), fmax(mf, mc));
}

static void form_affine_rhs(icore_ocp_workspace *w)
{
    size_t i, count = (size_t)w->K * w->m;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    for (i = 0; i < count; ++i)
        w->q[i] = w->constant_orthant[i] ? 0.0 : -(w->orthant_direct ? w->s[i] : w->lambda[i]);
    if (w->constant_soc_count)
        for (int k = 0; k < w->K; ++k)
            for (int c = 0; c < w->nsoc; ++c)
                if (constant_soc(w, k, c))
                    memset(w->q + (size_t)k * w->m + w->offsets[c], 0,
                           (size_t)w->sizes[c] * sizeof(double));
}

static void prepare_newton_rhs(icore_ocp_workspace *w)
{
    int k, i;
    size_t nvcount = (size_t)w->K * w->nv, mcount = (size_t)w->K * w->m;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
    if (w->orthant_direct) {
        size_t row;
        for (row = 0; row < mcount; ++row)
            w->bcone[row] = -w->rp[row] - w->q[row];
    } else
        for (k = 0; k < w->K; ++k) {
            cone_apply(w, k, w->q + (size_t)k * w->m, w->tmp, 0);
            for (i = 0; i < w->m; ++i)
                w->bcone[(size_t)k * w->m + i] = -w->rp[(size_t)k * w->m + i] - w->tmp[i];
        }
    w->rhs_norm[3] = norm_inf(w->bcone, mcount);
    for (i = 0; (size_t)i < nvcount; ++i)
        w->brd[i] = -w->rd[i];
    for (i = 0; i < w->nf; ++i)
        w->brf[i] = -w->rf[i];
}

/* Assemble the main Newton RHS and its norm in one pass. This is equivalent
 * to prepare_newton_rhs followed by form_reduced_rhs with bd=brd, bc=bcone
 * and direct_cone_rhs=1. Arbitrary refinement RHS uses the separate helpers.
 * No cache token is needed; G' updates follow ascending row order. */
static void prepare_main_reduced_rhs(icore_ocp_workspace *w)
{
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    if (w->orthant_direct) {
        size_t p, nvcount = (size_t)w->K * w->nv;
        int k, i, j, nv = w->nv, tail_finite = 1;
        double tail_maximum = 0.0, maximum_value;
        const __m256d sign = _mm256_set1_pd(-0.0), ceiling = _mm256_set1_pd(DBL_MAX);
        __m256d maximum = _mm256_setzero_pd();
        __m256d valid = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));
        __m128d pair;
        w->affine_state = ICORE_OCP_AFFINE_EMPTY;
        w->dynamics_cache_direction = NULL;
        for (p = 0; p < nvcount; ++p)
            w->brd[p] = -w->rd[p];
        for (i = 0; i < w->nf; ++i)
            w->brf[i] = -w->rf[i];
        memcpy(w->rhs, w->brd, nvcount * sizeof(double));
        w->orthant_gv_direction = NULL;
        for (k = 0; k < w->K; ++k) {
            const double *G = w->G + (size_t)k * w->m * nv;
            double *rhs = w->rhs + (size_t)k * nv;
            int tiled = constraint_transpose_grouped(w, k);
            for (i = 0; w->m - i >= 4; i += 4) {
                size_t row = (size_t)k * w->m + i;
                __m256d bc = _mm256_sub_pd(_mm256_xor_pd(_mm256_loadu_pd(w->rp + row), sign),
                                           _mm256_loadu_pd(w->q + row));
                __m256d pc =
                    _mm256_mul_pd(_mm256_xor_pd(_mm256_loadu_pd(w->lambda + row), sign), bc);
                __m256d magnitude = _mm256_andnot_pd(sign, bc);
                _mm256_storeu_pd(w->bcone + row, bc);
                _mm256_storeu_pd(w->pcone + row, pc);
                valid = _mm256_and_pd(valid, _mm256_cmp_pd(magnitude, ceiling, _CMP_LE_OQ));
                maximum = _mm256_max_pd(maximum, magnitude);
#if ICORE_OCP_SINGLETON_AXPY4
                if (w->ocp_diagonal_cost_ready &&
                    singleton_axpy4(w, row, rhs, _mm256_xor_pd(pc, sign)))
                    continue;
#endif
                if (!tiled)
                    for (j = 0; j < 4; ++j)
                        supported_axpy(w, row + (size_t)j, rhs, G + (size_t)(i + j) * nv,
                                       -w->pcone[row + (size_t)j], w->single_value);
            }
            for (; i < w->m; ++i) {
                size_t row = (size_t)k * w->m + i;
                double magnitude;
                w->bcone[row] = -w->rp[row] - w->q[row];
                w->pcone[row] = -w->lambda[row] * w->bcone[row];
                magnitude = fabs(w->bcone[row]);
                if (!isfinite(magnitude))
                    tail_finite = 0;
                else if (magnitude > tail_maximum)
                    tail_maximum = magnitude;
                if (!tiled)
                    supported_axpy(w, row, rhs, G + (size_t)i * nv, -w->pcone[row],
                                   w->single_value);
            }
            if (tiled) {
                for (i = 0; i < w->m; ++i)
                    w->tmp[i] = -w->pcone[(size_t)k * w->m + i];
                constraint_transpose_add(w, k, G, w->tmp, rhs);
            }
        }
        pair = _mm_max_pd(_mm256_castpd256_pd128(maximum), _mm256_extractf128_pd(maximum, 1));
        maximum_value = _mm_cvtsd_f64(_mm_max_sd(pair, _mm_unpackhi_pd(pair, pair)));
        if (tail_maximum > maximum_value)
            maximum_value = tail_maximum;
        /* Nonfinite bcone still produces the same infinite RHS norm, but
         * does not reject or truncate pcone/rhs writes earlier than before. */
        w->rhs_norm[3] = tail_finite && _mm256_movemask_pd(valid) == 15 ? maximum_value : INFINITY;
        return;
    }
#endif
    prepare_newton_rhs(w);
    form_reduced_rhs(w, w->brd, w->bcone, 1);
}

static int newton_direction(icore_ocp_workspace *w, const icore_ocp_options *opt)
{
    int iter, scheduled, accepted_refinement = 0;
    size_t nvcount = (size_t)w->K * w->nv, ycount = (size_t)w->N * w->nx,
           mcount = (size_t)w->K * w->m;
    double error, limit = ipm_refinement_limit(opt);
    if (w->mixed_factor_active && !(opt->refinement_tolerance > 0.0))
        limit = w->mixed_forcing;
    scheduled = w->affine_state == ICORE_OCP_AFFINE_PREDICTOR_ARMED && w->orthant_direct &&
                w->ocp_sqrt_active && !w->ne && !w->nf &&
                (w->mixed_sweeps_active || icore_ocp_la_sqrt_initial_zero_active(w->ocp_la));
    if (w->mixed_factor_active) {
        w->mixed_last_error = INFINITY;
        w->mixed_limit = limit;
    }
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    if (scheduled && (!equality_forward(w, w->rhs, w->re, w->brf, NULL, w->dv, w->dy, w->dnu) ||
                      !recover_cone_dual(w, w->dv, w->dz))) {
        /* Partial forward/cone output cannot lend a Pd or Gv cache to the
         * complete ordinary retry. RHS assembly below replaces pcone too. */
        if (w->mixed_sweeps_active)
            return 0;
        scheduled = 0;
        w->sweep_pd_valid = 0;
        w->affine_state = ICORE_OCP_AFFINE_EMPTY;
        w->orthant_gv_direction = NULL;
        w->dynamics_cache_direction = NULL;
    }
    if (!scheduled) {
        prepare_main_reduced_rhs(w);
        if (!equality_solve(w, w->rhs, w->re, w->brf, NULL, w->dv, w->dy, w->dnu) ||
            !recover_cone_dual(w, w->dv, w->dz))
            return 0;
    }
    error = block_residual(w, w->dv, w->dy, w->dnu, w->dz, w->ds);
    for (iter = 0; iter < opt->refinement_steps && error > limit; ++iter) {
        double next_error;
        size_t p;
        if (iter == 0 && !w->mixed_factor_active && w->dual_recovery_valid &&
            equality_dual_residual_eligible(w, limit)) {
            ++w->dual_recovery_attempts;
            if (equality_dual_propose(w, w->dv, w->dz, w->ty, w->tnu)) {
                /* Proposal construction has finished with cv. Save every
                 * original correction RHS before the full trial residual.
                 * Rejection restores the same RHS for this ordinary slot. */
                memcpy(w->cv, w->brd, nvcount * sizeof(double));
                memcpy(w->cy, w->bre, ycount * sizeof(double));
                memcpy(w->cnu, w->brf, (size_t)w->nf * sizeof(double));
                memcpy(w->tz, w->brc, mcount * sizeof(double));
                next_error = block_residual(w, w->dv, w->ty, w->tnu, w->dz, w->cz);
                if (isfinite(next_error) && next_error <= limit) {
                    memcpy(w->dy, w->ty, ycount * sizeof(double));
                    memcpy(w->dnu, w->tnu, (size_t)w->nf * sizeof(double));
                    /* dx/dz/ds are unchanged. This is one accepted correction
                     * inside the existing refinement budget, never an extra
                     * slot or a changed original-equation acceptance limit. */
                    error = next_error;
                    accepted_refinement = 1;
                    ++w->refinements;
                    ++w->dual_recovery_accepted;
                    continue;
                }
                memcpy(w->brd, w->cv, nvcount * sizeof(double));
                memcpy(w->bre, w->cy, ycount * sizeof(double));
                memcpy(w->brf, w->cnu, (size_t)w->nf * sizeof(double));
                memcpy(w->brc, w->tz, mcount * sizeof(double));
            }
        }
        if (!block_solve(w, w->brd, w->bre, w->brf, w->brc, 0, w->cv, w->cy, w->cnu, w->cz))
            break;
        for (p = 0; p < nvcount; ++p)
            w->tv[p] = w->dv[p] + w->cv[p];
        for (p = 0; p < ycount; ++p)
            w->ty[p] = w->dy[p] + w->cy[p];
        for (p = 0; p < (size_t)w->nf; ++p)
            w->tnu[p] = w->dnu[p] + w->cnu[p];
        for (p = 0; p < mcount; ++p)
            w->tz[p] = w->dz[p] + w->cz[p];
        /* tz has consumed the correction dual cz. Reuse that dead storage
         * for trial slack, preserving affine ads for a possible NT retry of
         * the corrector. Trial z lives in the disjoint tz array. */
        next_error = block_residual(w, w->tv, w->ty, w->tnu, w->tz, w->cz);
        if (!(next_error < error))
            break;
        memcpy(w->dv, w->tv, nvcount * sizeof(double));
        memcpy(w->dy, w->ty, ycount * sizeof(double));
        memcpy(w->dnu, w->tnu, (size_t)w->nf * sizeof(double));
        memcpy(w->dz, w->tz, mcount * sizeof(double));
        memcpy(w->ds, w->cz, mcount * sizeof(double));
        accepted_refinement = 1;
        error = next_error;
        ++w->refinements;
        if (w->mixed_factor_active)
            ++w->mixed_counts[5];
        if (w->mixed_sweeps_active)
            ++w->mixed_sweep_counts[6];
    }
    /* Initial dz already passed recover_cone_dual's full finite check.
     * Residuals and rejected/failed refinement trials do not write it.
     * Any accepted trial invalidates that proof for the rest of this call. */
    if (w->mixed_factor_active) {
        w->mixed_last_error = error;
        /* The automatic policy changes the mixed inner limit only.
         * This is still the original unregularized residual, checked against
         * the one forcing value captured before this iterate's factor. */
        if (!isfinite(error) || !(error <= limit)) {
            ++w->mixed_policy_counts[5];
            return 0;
        }
        if (!finite_array(w->ds, mcount) || (accepted_refinement && !finite_array(w->dz, mcount)))
            return 0;
        if (error > w->mixed_max_error)
            w->mixed_max_error = error;
        ++w->mixed_counts[4];
        if (w->mixed_sweeps_active)
            ++w->mixed_sweep_counts[5];
        if (limit > ipm_refinement_limit(opt))
            ++w->mixed_policy_counts[4];
        if (w->mixed_counts[4] == 1 || limit < w->mixed_policy_values[4])
            w->mixed_policy_values[4] = limit;
        if (error / limit > w->mixed_policy_values[5])
            w->mixed_policy_values[5] = error / limit;
        w->mixed_policy_values[7] = limit;
        return 1;
    }
    return finite_array(w->ds, mcount) && (!accepted_refinement || finite_array(w->dz, mcount));
}

static int corrector_rhs(icore_ocp_workspace *w, double target)
{
    int k, j, i;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
    if (w->orthant_direct) {
        size_t row = 0, count = (size_t)w->K * w->m;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
        {
            const __m256d zero = _mm256_setzero_pd(), one = _mm256_set1_pd(1.0);
            const __m256d tv = _mm256_set1_pd(target), sign = _mm256_set1_pd(-0.0);
            const __m256d ceiling = _mm256_set1_pd(DBL_MAX);
            __m256d valid = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));
            int tail_valid = 1;
            for (; count - row >= 4; row += 4) {
                const unsigned char *constant = w->constant_orthant + row;
                __m256d s, z, ads, adz, t = tv, q, mask = zero;
                if (constant[0] && constant[1] && constant[2] && constant[3]) {
                    _mm256_storeu_pd(w->q + row, zero);
                    continue;
                }
                s = _mm256_loadu_pd(w->s + row);
                z = _mm256_loadu_pd(w->z + row);
                ads = _mm256_loadu_pd(w->ads + row);
                adz = _mm256_loadu_pd(w->adz + row);
                if (constant[0] || constant[1] || constant[2] || constant[3]) {
                    mask = _mm256_castsi256_pd(
                        _mm256_set_epi64x(-(long long)!!constant[3], -(long long)!!constant[2],
                                          -(long long)!!constant[1], -(long long)!!constant[0]));
                    s = _mm256_andnot_pd(mask, s);
                    ads = _mm256_andnot_pd(mask, ads);
                    adz = _mm256_andnot_pd(mask, adz);
                    t = _mm256_andnot_pd(mask, t);
                    z = _mm256_blendv_pd(z, one, mask);
                }
                /* Keep the scalar expression's division and ordinary compiler
                 * contraction policy.  Constant rows never consume an invalid
                 * operand, and their result is the original positive zero. */
                q = _mm256_sub_pd(_mm256_div_pd(_mm256_sub_pd(t, _mm256_mul_pd(ads, adz)), z), s);
                q = _mm256_andnot_pd(mask, q);
                _mm256_storeu_pd(w->q + row, q);
                valid = _mm256_and_pd(
                    valid, _mm256_cmp_pd(_mm256_andnot_pd(sign, q), ceiling, _CMP_LE_OQ));
            }
            for (; row < count; ++row) {
                w->q[row] = w->constant_orthant[row]
                                ? 0.0
                                : -w->s[row] + (target - w->ads[row] * w->adz[row]) / w->z[row];
                tail_valid &= isfinite(w->q[row]) != 0;
            }
            /* Write the complete q before testing finiteness.
             * Preserve that contract even when an early lane is nonfinite. */
            return tail_valid && _mm256_movemask_pd(valid) == 15;
        }
#else
        for (row = 0; row < count; ++row)
            w->q[row] = w->constant_orthant[row]
                            ? 0.0
                            : -w->s[row] + (target - w->ads[row] * w->adz[row]) / w->z[row];
        return finite_array(w->q, count);
#endif
    }
    for (k = 0; k < w->K; ++k) {
        const double *lam = w->lambda + (size_t)k * w->m;
        double *q = w->q + (size_t)k * w->m;
        /* Orthant correctors use the unscaled affine product below.
         * Only SOC Jordan products consume these transformed directions. */
        if (w->nsoc) {
            cone_apply(w, k, w->ads + (size_t)k * w->m, w->tmp, 1);
            cone_apply(w, k, w->adz + (size_t)k * w->m, w->tmp2, 0);
        }
        for (j = 0; j < w->nlp; ++j)
            q[j] = w->constant_orthant[(size_t)k * w->m + j]
                       ? 0.0
                       : -lam[j] + (target -
                                    w->ads[(size_t)k * w->m + j] * w->adz[(size_t)k * w->m + j]) /
                                       lam[j];
        for (j = 0; j < w->nsoc; ++j) {
            int o = w->offsets[j], n = w->sizes[j];
            if (constant_soc(w, k, j)) {
                memset(q + o, 0, (size_t)n * sizeof(double));
                continue;
            }
            double a = lam[o], r = norm2(lam + o + 1, n - 1), den = (a - r) * (1.0 + r / a), q0;
            long double numerator;
            w->tmp3[0] = target - dot(w->tmp + o, w->tmp2 + o, n);
            for (i = 1; i < n; ++i)
                w->tmp3[i] = -(w->tmp[o] * w->tmp2[o + i] + w->tmp2[o] * w->tmp[o + i]);
            numerator = w->tmp3[0];
            for (i = 1; i < n; ++i)
                numerator -= (long double)(lam[o + i] / a) * w->tmp3[i];
            if (!(den > 0.0))
                return 0;
            q0 = (double)numerator / den;
            q[o] = -a + q0;
            for (i = 1; i < n; ++i)
                q[o + i] = -lam[o + i] + (w->tmp3[i] - lam[o + i] * q0) / a;
        }
    }
    return finite_array(w->q, (size_t)w->K * w->m);
}
