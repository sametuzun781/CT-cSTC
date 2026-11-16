/* Guarded equality-dual recovery for bordered conic Newton directions.
 * The constant terminal-control map depends only on original D and F.
 * It proposes no primal/cone change and never replaces the original
 * residual/refinement gate. Private implementation included once by ocp.c. */

/* A dy/dnu-only proposal cannot change terminal-primal or cone residuals.
 * Reuse their already computed original residuals before any proposal work.
 * Keep the same divisions/comparisons as block_residual; multiplying the
 * threshold by its scale can round to a different acceptance boundary.
 * Dynamics is deliberately excluded: its first residual may use a consumed
 * packed D*v token, whereas a trial recomputes that product. The full trial
 * still checks every original block, including dynamics and stationarity. */
static int equality_dual_residual_eligible(const icore_ocp_workspace *w, double limit)
{
    if (!isfinite(w->rhs_norm[2]) || !isfinite(w->rhs_norm[3]))
        return 0;
    return norm_inf(w->brf, (size_t)w->nf) / (1.0 + w->rhs_norm[2]) <= limit &&
           norm_inf(w->brc, (size_t)w->K * w->m) / (1.0 + w->rhs_norm[3]) <= limit;
}

static void equality_dual_storage_create(icore_ocp_workspace *w)
{
    size_t rows = (size_t)w->N * w->nu, nf = (size_t)w->nf, count;
    if (!w->nsoc || !nf || !w->nu || rows < nf || rows > SIZE_MAX / nf)
        return;
    count = rows * nf;
    if (nf > (SIZE_MAX - count) / (nf + 1u))
        return;
    count += nf * nf + nf;
    w->dual_recovery_storage = (double *)array_new(count, sizeof(double));
    if (w->dual_recovery_storage)
        w->dual_recovery_bytes = (count * sizeof(double) + 63u) & ~(size_t)63u;
}

/* A_K'=A_k^T propagates terminal multiplier columns backward. The control
 * rows J_k=B_k^T A_{k+1}^T ... A_{N-1}^T Fn^T are constant across IPM steps.
 * Normalize each column before the small Gram factor. No floored pivot is
 * admitted, and an infinity-norm condition estimate guards normal equations. */
static void equality_dual_prepare(icore_ocp_workspace *w)
{
    size_t rows = (size_t)w->N * w->nu;
    int nx = w->nx, nv = w->nv, nf = w->nf, k, i, j, a, b, poor = 0;
    double *J = w->dual_recovery_storage;
    double *L, *scale, *basis = w->PD, *next = w->M;
    double anorm = 0.0, invnorm = 0.0;
    w->dual_recovery_valid = 0;
    w->dual_recovery_condition = INFINITY;
    if (!J || !nf || rows < (size_t)nf)
        return;
    L = J + rows * nf;
    scale = L + (size_t)nf * nf;
    for (i = 0; i < nx; ++i)
        for (a = 0; a < nf; ++a)
            basis[(size_t)i * nf + a] = w->Fn[(size_t)a * nx + i];
    for (k = w->N - 1; k >= 0; --k) {
        const double *D = w->D + (size_t)k * nx * nv;
        for (j = 0; j < w->nu; ++j)
            for (a = 0; a < nf; ++a) {
                long double value = 0.0L;
                for (i = 0; i < nx; ++i)
                    value += (long double)D[(size_t)i * nv + j] * basis[(size_t)i * nf + a];
                if (!isfinite((double)value))
                    return;
                J[((size_t)k * w->nu + j) * nf + a] = (double)value;
            }
        if (k) {
            for (j = 0; j < nx; ++j)
                for (a = 0; a < nf; ++a) {
                    long double value = 0.0L;
                    for (i = 0; i < nx; ++i)
                        value +=
                            (long double)D[(size_t)i * nv + w->nu + j] * basis[(size_t)i * nf + a];
                    if (!isfinite((double)value))
                        return;
                    next[(size_t)j * nf + a] = (double)value;
                }
            {
                double *swap = basis;
                basis = next;
                next = swap;
            }
        }
    }
    for (a = 0; a < nf; ++a) {
        long double norm = 0.0L;
        for (size_t row = 0; row < rows; ++row)
            norm = hypotl(norm, (long double)J[row * nf + a]);
        scale[a] = (double)norm;
        if (!(scale[a] > 0.0) || !isfinite(scale[a]))
            return;
        for (size_t row = 0; row < rows; ++row)
            J[row * nf + a] /= scale[a];
    }
    /* Both temporary basis banks are dead; M now holds the original Gram. */
    for (a = 0; a < nf; ++a)
        for (b = 0; b <= a; ++b) {
            long double value = 0.0L;
            for (size_t row = 0; row < rows; ++row)
                value += (long double)J[row * nf + a] * J[row * nf + b];
            if (!isfinite((double)value))
                return;
            w->M[(size_t)a * nf + b] = w->M[(size_t)b * nf + a] = (double)value;
        }
    memcpy(L, w->M, (size_t)nf * nf * sizeof(double));
    if (!cholesky(L, nf, 0.0, &poor, w->qr) || poor)
        return;
    memset(w->tmp2, 0, (size_t)nf * sizeof(double));
    for (a = 0; a < nf; ++a) {
        double row_sum = 0.0;
        for (b = 0; b < nf; ++b)
            row_sum += fabs(w->M[(size_t)a * nf + b]);
        if (row_sum > anorm)
            anorm = row_sum;
        memset(w->tmp, 0, (size_t)nf * sizeof(double));
        w->tmp[a] = 1.0;
        triangular_forward(L, w->tmp, nf);
        triangular_backward(L, w->tmp, nf);
        for (b = 0; b < nf; ++b)
            w->tmp2[b] += fabs(w->tmp[b]);
    }
    invnorm = norm_inf(w->tmp2, (size_t)nf);
    w->dual_recovery_condition = anorm * invnorm;
    /* This only gates an optional proposal; it is not a relaxed KKT bound. */
    if (!isfinite(w->dual_recovery_condition) || w->dual_recovery_condition > 1e6)
        return;
    w->dual_recovery_valid = 1;
}

/* Write a proposed dy/dnu using dead correction-gradient scratch cv.
 * Main Newton bd=-rd. The primal/cone direction and original equations
 * stay fixed. The caller must validate every original residual block. */
static int equality_dual_propose(icore_ocp_workspace *w, const double *v, const double *z, double *y,
                                 double *nu)
{
    int nx = w->nx, nv = w->nv, nf = w->nf, k, i, j, a;
    size_t rows = (size_t)w->N * w->nu;
    const double *J = w->dual_recovery_storage;
    const double *L, *scale;
    double *gradient = w->cv;
    if (!w->dual_recovery_valid || !J)
        return 0;
    L = J + rows * nf;
    scale = L + (size_t)nf * nf;
    for (k = 0; k < w->K; ++k) {
        const double *H = w->H + (size_t)k * nv * nv, *G = w->G + (size_t)k * w->m * nv;
        for (i = 0; i < nv; ++i) {
            size_t row = (size_t)k * nv + i;
            int start = w->cost_start[row], end = start + w->cost_length[row];
            long double value = -(long double)w->rd[row];
            for (j = start; j < end; ++j)
                value -= (long double)H[(size_t)i * nv + j] * v[(size_t)k * nv + j];
            for (j = 0; j < w->m; ++j) {
                double g = G[(size_t)j * nv + i];
                if (g != 0.0)
                    value -= (long double)g * z[(size_t)k * w->m + j];
            }
            gradient[row] = (double)value;
            if (!isfinite(gradient[row]))
                return 0;
        }
    }
    /* Particular equality-dual recurrence with terminal multiplier zero. */
    for (k = w->N; k > 0; --k)
        for (i = 0; i < nx; ++i) {
            long double value = -(long double)gradient[(size_t)k * nv + w->nu + i];
            if (k < w->N)
                for (j = 0; j < nx; ++j)
                    value += (long double)w->D[(size_t)k * nx * nv + (size_t)j * nv + w->nu + i] *
                             y[(size_t)k * nx + j];
            y[(size_t)(k - 1) * nx + i] = (double)value;
            if (!isfinite((double)value))
                return 0;
        }
    /* Control entries of cv may now hold the least-squares RHS. State
     * entries remain available for the final backward recurrence. */
    for (k = 0; k < w->N; ++k)
        for (i = 0; i < w->nu; ++i) {
            long double value = gradient[(size_t)k * nv + i];
            for (j = 0; j < nx; ++j)
                value -= (long double)w->D[(size_t)k * nx * nv + (size_t)j * nv + i] *
                         y[(size_t)k * nx + j];
            gradient[(size_t)k * nv + i] = (double)value;
            if (!isfinite((double)value))
                return 0;
        }
    for (a = 0; a < nf; ++a) {
        long double value = 0.0L;
        for (k = 0; k < w->N; ++k)
            for (i = 0; i < w->nu; ++i)
                value +=
                    (long double)J[((size_t)k * w->nu + i) * nf + a] * gradient[(size_t)k * nv + i];
        nu[a] = (double)value;
        if (!isfinite(nu[a]))
            return 0;
    }
    triangular_forward(L, nu, nf);
    triangular_backward(L, nu, nf);
    for (a = 0; a < nf; ++a) {
        nu[a] = (double)((long double)nu[a] / scale[a] / w->Fnorm[a]);
        if (!isfinite(nu[a]))
            return 0;
    }
    for (k = w->N; k > 0; --k)
        for (i = 0; i < nx; ++i) {
            long double value = -(long double)gradient[(size_t)k * nv + w->nu + i];
            if (k < w->N)
                for (j = 0; j < nx; ++j)
                    value += (long double)w->D[(size_t)k * nx * nv + (size_t)j * nv + w->nu + i] *
                             y[(size_t)k * nx + j];
            else
                for (a = 0; a < nf; ++a)
                    value += (long double)w->F[(size_t)a * nx + i] * nu[a];
            y[(size_t)(k - 1) * nx + i] = (double)value;
            if (!isfinite((double)value))
                return 0;
        }
    return 1;
}
