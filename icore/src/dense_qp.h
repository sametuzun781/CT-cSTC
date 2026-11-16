/*
 * Direct dense orthant kernel for one-stage QPs with a neutral terminal cost.
 * It shares ipm_run, original-coordinate stopping tests, and user options.
 * Eligibility is algebraic and recomputed after every coefficient update.
 * H/G remain untouched; unsafe direct arithmetic falls back to the OCP kernel.
 *
 * This internal header is included after ocp_context and the OCP kernel.
 */
#ifndef ICORE_OCP_DENSE_AUX_INITIALIZER
#define ICORE_OCP_DENSE_AUX_INITIALIZER 0
#endif

#include "dense_bound_pairs.h"

/* Dense callbacks for the same driver_loop.h algorithm.
 * Model certification runs during updates; numeric sections follow below. */
#include "dense_qp_model.h"

/* ---- Certified active ranges and original-coefficient matrix products. ---- */

/* The active row list is strictly increasing. Its last index certifies an
 * exact prefix, even if constant constraints follow it in the model. */
ICORE_OCP_INLINE int dense_qp_active_prefix(const icore_ocp_workspace *w)
{
    return w->qp_active == 0 || w->qp_rows[w->qp_active - 1] == w->qp_active - 1;
}

/* Constant rows can precede an otherwise contiguous active range. The
 * strictly increasing row list certifies the whole slice by its endpoints;
 * zero-length slices use offset zero. No inactive row is read or changed. */
ICORE_OCP_INLINE int dense_qp_active_begin(const icore_ocp_workspace *w)
{
    if (!w->qp_active)
        return 0;
    int begin = w->qp_rows[0];
    return w->qp_rows[w->qp_active - 1] - begin == w->qp_active - 1 ? begin : -1;
}

static int dense_qp_h_product(icore_ocp_workspace *w, const double *x, double *hx)
{
    int i, n = w->nu;
    /* Exact original-H structure, certified at model commit, not a numerical
     * diagonal approximation. The original Newton residual still uses H. */
    if (w->coefficient_cache_valid && w->cost_root_diagonal[0]) {
        if (!finite_array(x, (size_t)n))
            return 0;
        for (i = 0; i < n; ++i)
            hx[i] = w->qp_H[(size_t)i * n + i] * x[i];
        return finite_array(hx, (size_t)n);
    }
    /* Small products use the existing row-major H directly. This avoids
     * packed-vector copies and a module call; the original H and the full
     * finite-output test are retained. Larger products keep BLASFEO. */
    if (w->qp_packed_ready && n >= 32)
        return icore_la_hmv(w->qp_la, 1.0, x, 0.0, hx) == ICORE_LA_OK;
    for (i = 0; i < n; ++i)
        hx[i] = dot(w->qp_H + (size_t)i * n, x, n);
    return finite_array(hx, (size_t)n);
}

/* General rows are strictly increasing. If last-first == count-1, their
 * values already occupy one exact contiguous slice, at any model offset.
 * Borrow that slice only until ICORE_LA validates and copies it to its own padded
 * vector. Gapped models retain the indexed gather; no caller storage is
 * passed directly to a BLASFEO kernel. */
ICORE_OCP_INLINE const double *dense_qp_general_values(icore_ocp_workspace *w, const double *values)
{
    int count = w->qp_general_count;
    if (count == 0)
        return values;
    int first = w->qp_general_rows[0];
    if (w->qp_general_rows[count - 1] - first == count - 1)
        return values + first;
    for (int a = 0; a < count; ++a)
        w->qp_weights[a] = values[w->qp_general_rows[a]];
    return w->qp_weights;
}

static int dense_qp_g_pair(icore_ocp_workspace *w, const double *x, const double *z)
{
    return icore_la_gmv_pair(w->qp_la, x, dense_qp_general_values(w, z), w->qp_gx, w->qp_gt) == ICORE_LA_OK;
}

/* Preserve exact diagonal products. Non-diagonal H and general G share
 * the same validated x pack, including small dimensions where G already
 * needs that pack. Standalone H products retain their existing dispatch. */
static int dense_qp_hg_product(icore_ocp_workspace *w, const double *x, const double *z, double *hx,
                               int forward)
{
    if (!w->qp_packed_ready || w->qp_general_count <= 0 ||
        (w->coefficient_cache_valid && w->cost_root_diagonal[0]))
        return 0;
    int status = icore_la_hg_products(w->qp_la, x, dense_qp_general_values(w, z), hx, w->qp_gx,
                                  w->qp_gt, forward);
    return status == ICORE_LA_OK ? 1 : status == ICORE_LA_UNAVAILABLE ? 0 : -1;
}

ICORE_OCP_INLINE double dense_qp_row_dot(const icore_ocp_workspace *w, int row, const double *x)
{
    int count = w->qp_nnz[row], j;
    if (count == 1)
        return w->qp_box_value[row] * x[w->qp_box_index[row]];
    const int *idx = w->indices + w->index_start[row];
    const double *g = w->G + (size_t)row * w->nv;
    double value = 0.0;
    if (count == 2)
        return g[idx[0]] * x[idx[0]] + g[idx[1]] * x[idx[1]];
    if (count && idx[count - 1] - idx[0] + 1 == count)
        return dot(g + idx[0], x + idx[0], count);
    for (j = 0; j < count; ++j)
        value += g[idx[j]] * x[idx[j]];
    return value;
}

ICORE_OCP_INLINE void dense_qp_row_add(const icore_ocp_workspace *w, int row, double *x, double scale)
{
    int count = w->qp_nnz[row], j;
    if (count == 1) {
        x[w->qp_box_index[row]] += scale * w->qp_box_value[row];
        return;
    }
    const int *idx = w->indices + w->index_start[row];
    const double *g = w->G + (size_t)row * w->nv;
    if (count == 2) {
        x[idx[0]] += scale * g[idx[0]];
        x[idx[1]] += scale * g[idx[1]];
        return;
    }
    if (count && idx[count - 1] - idx[0] + 1 == count) {
        axpy(x + idx[0], g + idx[0], scale, count);
        return;
    }
    for (j = 0; j < count; ++j)
        x[idx[j]] += scale * g[idx[j]];
}

static void dense_qp_expand(icore_ocp_workspace *w)
{
    int i;
    memcpy(w->v + w->nu, w->x0, (size_t)w->nx * sizeof(double));
    for (i = 0; i < w->nx; ++i)
        w->v[w->nv + w->nu + i] = dot(w->D + (size_t)i * w->nv, w->v, w->nv) + w->b[i];
}

/* ---- Cold initialization and original-coordinate residual evaluation. ---- */

/* Treat a singleton row in bound units only when the positive control
 * cost already supplies curvature. Preserve the established cold point for
 * semidefinite costs and for rows near the requested absolute roundoff floor.
 * Decide for the complete model before touching s/z, so an unsafe row keeps
 * the whole original initializer. */
static int dense_qp_normalize_initial_bounds(const icore_ocp_workspace *w)
{
    int i, a, scaled_bound = 0;
    const double safe_min = sqrt(DBL_MIN), safe_max = 1.0 / safe_min;
    if (!w->qp_reuse_forward)
        return 0;
    for (i = 0; i < w->nu; ++i)
        if (!(w->Hs[(size_t)i * w->nv + i] > 0.0))
            return 0;
    for (a = 0; a < w->qp_active; ++a) {
        int r = w->qp_rows[a];
        if (w->qp_nnz[r] == 1) {
            double unit = fabs(w->qp_box_value[r]);
            double slack = w->qp_h[r] > unit ? w->qp_h[r] : unit;
            if (!(slack >= safe_min && slack <= safe_max))
                return 0;
            scaled_bound |= unit != 1.0;
        }
    }
    return scaled_bound;
}

static int dense_qp_initialize(void *context)
{
    ocp_context *c = (ocp_context *)context;
    icore_ocp_workspace *w = c->w;
    int a;
    size_t p;
    /* All QPs use the common IPM driver. No direct optimizer is run here. */
    w->qp_reuse_forward = w->qp_row_magnitude < c->options->tolerance / (16.0 * DBL_EPSILON);
#if ICORE_OCP_DENSE_AUX_INITIALIZER
    if (!initialize(w, c->options))
        return 0;
#else
    int normalize_bounds = dense_qp_normalize_initial_bounds(w);
    /* A cold infeasible-start point: no solved iterate, no auxiliary factor. */
    memset(w->v, 0, c->nvcount * sizeof(double));
    memset(w->y, 0, c->ycount * sizeof(double));
    for (p = 0; p < c->mcount; ++p)
        w->s[p] = w->z[p] = 1.0;
    dense_qp_expand(w);
    for (a = 0; a < w->qp_active; ++a) {
        int r = w->qp_rows[a];
        double slack = w->qp_h[r] - dense_qp_row_dot(w, r, w->v);
        if (!isfinite(slack))
            return 0;
        w->s[r] = slack > 1.0 ? slack : 1.0;
        if (normalize_bounds && w->qp_nnz[r] == 1) {
            /* In bound units, s=max(h/|g|-sign(g)*u, 1), z=1/s.
             * Map both back to the original row: s_original=|g|*s and
             * z_original=z/|g|. This also leaves unit rows unchanged.
             * Extreme scales retain the original initializer before any
             * reciprocal or barrier weight could overflow or underflow. */
            double floor = fabs(w->qp_box_value[r]);
            w->s[r] = slack > floor ? slack : floor;
        }
        w->z[r] = 1.0 / w->s[r];
        if (!(w->s[r] > 0.0 && w->z[r] > 0.0) || !isfinite(w->s[r]) || !isfinite(w->z[r]))
            return 0;
    }
#endif
    memset(w->dv, 0, c->nvcount * sizeof(double));
    memset(w->dy, 0, c->ycount * sizeof(double));
    memset(w->ds, 0, c->mcount * sizeof(double));
    memset(w->dz, 0, c->mcount * sizeof(double));
    memset(w->ads, 0, c->mcount * sizeof(double));
    memset(w->adz, 0, c->mcount * sizeof(double));
    memset(w->rd, 0, c->nvcount * sizeof(double));
    memset(w->rp, 0, c->mcount * sizeof(double));
    return finite_array(w->v, c->nvcount);
}

static int dense_qp_evaluate_rows(void *context, ipm_residual *result, int box_only)
{
    ocp_context *c = (ocp_context *)context;
    icore_ocp_workspace *w = c->w;
    int i, a, n = w->nu;
    double primal = w->constant_primal_defect, dual = 0.0, gap = 0.0;
    double objective = w->qp_constant, rpnorm = 0.0;
    if (!finite_array(w->v, c->nvcount))
        return 0;
    int fused = dense_qp_hg_product(w, w->v, w->z, w->rd, 1);
    if (fused < 0 || (!fused && !dense_qp_h_product(w, w->v, w->rd)))
        return 0;
    int general = w->qp_packed_ready && w->qp_general_count > 0;
    if (general && !fused && !dense_qp_g_pair(w, w->v, w->z))
        return 0;
    for (i = 0; i < n; ++i) {
        double hx = w->rd[i];
        w->rd[i] = hx + w->qp_c[i];
        if (general)
            w->rd[i] += w->qp_gt[i];
        objective += w->v[i] * (0.5 * hx + w->qp_c[i]);
    }
    a = 0;
#if ICORE_OCP_BOUND_PAIRS
    if (box_only && w->qp_bound_pairs) {
        if (!bound_pairs_evaluate(w, &primal, &dual, &gap, &rpnorm))
            return 0;
        a = w->qp_active;
    }
#endif
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
    if (w->qp_packed_ready && w->qp_general_count == w->qp_active && dense_qp_active_prefix(w)) {
        const __m256d sign = _mm256_set1_pd(-0.0), ceiling = _mm256_set1_pd(DBL_MAX);
        __m256d maxrp = _mm256_setzero_pd(), mins = _mm256_setzero_pd(), minz = _mm256_setzero_pd();
        for (; a + 3 < w->qp_active; a += 4) {
            __m256d sv = _mm256_loadu_pd(w->s + a), zv = _mm256_loadu_pd(w->z + a);
            __m256d rp = _mm256_sub_pd(_mm256_add_pd(_mm256_loadu_pd(w->qp_gx + a), sv),
                                       _mm256_loadu_pd(w->qp_h + a));
            __m256d mag = _mm256_andnot_pd(sign, rp);
            __m256d valid = _mm256_and_pd(
                _mm256_cmp_pd(mag, ceiling, _CMP_LE_OQ),
                _mm256_and_pd(_mm256_cmp_pd(_mm256_andnot_pd(sign, sv), ceiling, _CMP_LE_OQ),
                              _mm256_cmp_pd(_mm256_andnot_pd(sign, zv), ceiling, _CMP_LE_OQ)));
            if (_mm256_movemask_pd(valid) != 15)
                return 0;
            _mm256_storeu_pd(w->rp + a, rp);
            maxrp = _mm256_max_pd(maxrp, mag);
            mins = _mm256_min_pd(mins, sv);
            minz = _mm256_min_pd(minz, zv);
            double products[4];
            _mm256_storeu_pd(products, _mm256_andnot_pd(sign, _mm256_mul_pd(sv, zv)));
            for (int lane = 0; lane < 4; ++lane)
                gap += products[lane];
        }
        double pr[4], ss[4], zz[4];
        _mm256_storeu_pd(pr, maxrp);
        _mm256_storeu_pd(ss, mins);
        _mm256_storeu_pd(zz, minz);
        for (int lane = 0; lane < 4; ++lane) {
            if (pr[lane] > rpnorm)
                rpnorm = pr[lane];
            if (-ss[lane] > primal)
                primal = -ss[lane];
            if (-zz[lane] > dual)
                dual = -zz[lane];
        }
    }
#endif
    for (; a < w->qp_active; ++a) {
        int r = w->qp_rows[a], g = !box_only && w->qp_packed_ready ? w->qp_gindex[r] : -1;
        double gv = g >= 0 ? w->qp_gx[g]
                           : (box_only ? w->qp_box_value[r] * w->v[w->qp_box_index[r]]
                                       : dense_qp_row_dot(w, r, w->v));
        double rp = gv + w->s[r] - w->qp_h[r];
        if (!isfinite(rp) || !isfinite(w->s[r]) || !isfinite(w->z[r]))
            return 0;
        w->rp[r] = rp;
        if (g < 0) {
            if (box_only)
                w->rd[w->qp_box_index[r]] += w->z[r] * w->qp_box_value[r];
            else
                dense_qp_row_add(w, r, w->rd, w->z[r]);
        }
        /* Values were checked above. Comparisons keep this reduction in the
         * vector loop instead of calling libm fmax for every scalar row. */
        if (fabs(rp) > rpnorm)
            rpnorm = fabs(rp);
        if (-w->s[r] > primal)
            primal = -w->s[r];
        if (-w->z[r] > dual)
            dual = -w->z[r];
        gap += fabs(w->s[r] * w->z[r]);
    }
    w->rhs_norm[0] = norm_inf(w->rd, (size_t)n);
    w->rhs_norm[1] = rpnorm;
    primal = fmax(primal, rpnorm);
    dual = fmax(dual, w->rhs_norm[0]);
    result->primal = c->info->primal_residual = primal;
    result->dual = c->info->dual_residual = dual;
    result->gap = c->info->gap = gap;
    c->info->objective = objective;
    if (primal <= c->options->tolerance && dual <= c->options->tolerance &&
        gap <= c->options->tolerance) {
        /* Elimination may change cancellation. Acceptance is always in the
         * original OCP coordinates, including the reconstructed dynamics. */
        residuals(w, c->info);
        result->primal = c->info->primal_residual;
        result->dual = c->info->dual_residual;
        result->gap = c->info->gap;
        return isfinite(c->info->objective) &&
               isfinite(result->primal + result->dual + result->gap) &&
               result->primal <= c->options->tolerance && result->dual <= c->options->tolerance &&
               result->gap <= c->options->tolerance;
    }
    return isfinite(objective) && isfinite(primal + dual + gap);
}

static int dense_qp_evaluate(void *context, ipm_residual *result)
{
    if (((ocp_context *)context)->w->qp_bound_pairs ||
        (((ocp_context *)context)->w->qp_packed_ready &&
         ((ocp_context *)context)->w->qp_general_count == 0))
        return dense_qp_evaluate_rows(context, result, 1);
    return dense_qp_evaluate_rows(context, result, 0);
}

/* ---- Dense Hessian assembly, Newton directions, and refinement. ---- */

static int dense_qp_assemble(icore_ocp_workspace *w, double *L, double regularization)
{
    int i, a, b, n = w->nu;
    for (i = 0; i < n; ++i) {
        memcpy(L + (size_t)i * n, w->qp_H + (size_t)i * n, (size_t)(i + 1) * sizeof(double));
        L[(size_t)i * n + i] += regularization;
    }
    for (a = 0; a < w->qp_active; ++a) {
        int r = w->qp_rows[a], count = w->qp_nnz[r];
        const int *idx = w->indices + w->index_start[r];
        const double *g = w->G + (size_t)r * w->nv;
        if (count == 1 && w->qp_unit_singletons)
            L[(size_t)idx[0] * n + idx[0]] += w->orthant_inv[r];
        else if (count == 1) {
            double value = sqrt(w->orthant_inv[r]) * g[idx[0]];
            L[(size_t)idx[0] * n + idx[0]] += value * value;
        } else {
            double scale = sqrt(w->orthant_inv[r]);
            for (i = 0; i < count; ++i) {
                w->tmp2[i] = scale * g[idx[i]];
                if (!isfinite(w->tmp2[i]))
                    return 0;
            }
            if (count && idx[count - 1] - idx[0] + 1 == count) {
                int first = idx[0];
                for (i = 0; i < count; ++i)
                    axpy(L + (size_t)(first + i) * n + first, w->tmp2, w->tmp2[i], i + 1);
            } else
                for (i = 0; i < count; ++i) {
                    for (b = 0; b <= i; ++b)
                        L[(size_t)idx[i] * n + idx[b]] += w->tmp2[i] * w->tmp2[b];
                }
        }
    }
    return 1;
}

static int dense_qp_factor(void *context)
{
    ocp_context *c = (ocp_context *)context;
    icore_ocp_workspace *w = c->w;
    int a, poor = 0, ok;
    a = 0;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    /* A contiguous active slice may start after constant constraints. */
    int begin = dense_qp_active_begin(w);
    if (begin >= 0) {
        const __m256d zero = _mm256_setzero_pd(), maximum = _mm256_set1_pd(DBL_MAX);
        for (; a + 3 < w->qp_active; a += 4) {
            __m256d weight =
                _mm256_div_pd(_mm256_loadu_pd(w->z + begin + a), _mm256_loadu_pd(w->s + begin + a));
            if (_mm256_movemask_pd(_mm256_and_pd(_mm256_cmp_pd(weight, zero, _CMP_GT_OQ),
                                                 _mm256_cmp_pd(weight, maximum, _CMP_LE_OQ))) != 15)
                return 0;
            _mm256_storeu_pd(w->orthant_inv + begin + a, weight);
        }
    }
#endif
    for (; a < w->qp_active; ++a) {
        int r = w->qp_rows[a];
        double weight = w->z[r] / w->s[r];
        if (!(weight > 0.0) || !isfinite(weight))
            return 0;
        w->orthant_inv[r] = weight;
    }
    w->reg = fmax(0.0, c->options->regularization);
    w->qp_packed_factor = 0;
    w->qp_diagonal_factor = 0;
    if (w->qp_diagonal_eligible) {
        int i, safe = 1;
        double maximum = 0.0, floor;
        for (i = 0; i < w->nu; ++i)
            w->qp_diag[i] = w->qp_H[(size_t)i * w->nu + i] + w->reg;
        for (a = 0; a < w->qp_active; ++a) {
            int row = w->qp_rows[a], j = w->qp_box_index[row];
            if (w->qp_unit_singletons)
                w->qp_diag[j] += w->orthant_inv[row];
            else {
                double value = sqrt(w->orthant_inv[row]) * w->qp_box_value[row];
                w->qp_diag[j] += value * value;
            }
        }
        for (i = 0; i < w->nu; ++i) {
            double d = w->qp_diag[i];
            if (!isfinite(d))
                safe = 0;
            if (fabs(d) > maximum)
                maximum = fabs(d);
        }
        floor = fmax(fmax(w->reg, 1e-12 * maximum), 1024.0 * DBL_MIN);
        for (i = 0; i < w->nu && safe; ++i) {
            double d = w->qp_diag[i];
            if (!(d >= floor))
                safe = 0;
            else
                w->L[(size_t)i * w->nu + i] = sqrt(d);
        }
        if (safe) {
            w->qp_diagonal_factor = 1;
            return 1;
        }
        /* The original equilibrated/floored recovery remains responsible
         * for unsafe curvature, small pivots and extreme range. */
    }
    if (w->qp_packed_ready) {
        const double *weights = w->qp_weights;
        memset(w->qp_diag, 0, (size_t)w->nu * sizeof(double));
        if (w->qp_general_count == w->qp_active && dense_qp_active_prefix(w))
            weights = w->orthant_inv;
        else
#if ICORE_OCP_BOUND_PAIRS
            if (w->qp_bound_pairs)
            bound_pairs_diagonal(w);
        else
#endif
            for (a = 0; a < w->qp_active; ++a) {
                int r = w->qp_rows[a], g = w->qp_gindex[r];
                if (g >= 0)
                    w->qp_weights[g] = w->orthant_inv[r];
                else {
                    int j = w->qp_box_index[r];
                    if (w->qp_unit_singletons)
                        w->qp_diag[j] += w->orthant_inv[r];
                    else {
                        double value = sqrt(w->orthant_inv[r]) * w->qp_box_value[r];
                        w->qp_diag[j] += value * value;
                    }
                }
            }
        ok = icore_la_factor(w->qp_la, w->qp_diag, weights, w->reg, 0) == ICORE_LA_OK;
        if (ok && icore_la_pivots_acceptable(w->qp_la, fmax(w->reg, 1024.0 * DBL_MIN), 1e-12)) {
            w->qp_packed_factor = 1;
            return 1;
        }
        ok = icore_la_factor(w->qp_la, w->qp_diag, weights, w->reg, 1) == ICORE_LA_OK;
        if (ok && icore_la_pivots_acceptable(w->qp_la, fmax(w->reg, 1024.0 * DBL_MIN), 1e-12)) {
            w->qp_packed_factor = 1;
            return 1;
        }
    }
    if (!dense_qp_assemble(w, w->L, w->reg))
        return 0;
    ok = cholesky(w->L, w->nu, w->reg, &poor, w->qr);
    if (ok && !poor)
        return 1;
    /* Re-form only on trouble; no ordinary Hb->M->L matrix-copy chain. */
    if (!dense_qp_assemble(w, w->M, w->reg))
        return 0;
    return cholesky_equilibrated(w->M, w->nu, w->L, w->nu, w->reg, w->tmp, w->qr);
}

static int dense_qp_form_rhs(void *context, double target, int corrector)
{
    ocp_context *c = (ocp_context *)context;
    icore_ocp_workspace *w = c->w;
    int a;
    double maximum = 0.0;
    a = 0;
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
    {
        const __m256d sign = _mm256_set1_pd(-0.0), ceiling = _mm256_set1_pd(DBL_MAX);
        const __m256d centering = _mm256_set1_pd(target);
        __m256d max4 = _mm256_setzero_pd();
        for (; a + 3 < w->qp_active && w->qp_rows[a + 3] == w->qp_rows[a] + 3; a += 4) {
            int r = w->qp_rows[a];
            __m256d rc = _mm256_mul_pd(_mm256_loadu_pd(w->s + r), _mm256_loadu_pd(w->z + r));
            if (corrector)
                rc = _mm256_add_pd(rc, _mm256_fmsub_pd(_mm256_loadu_pd(w->ads + r),
                                                       _mm256_loadu_pd(w->adz + r), centering));
            __m256d magnitude = _mm256_andnot_pd(sign, rc);
            if (_mm256_movemask_pd(_mm256_cmp_pd(magnitude, ceiling, _CMP_LE_OQ)) != 15)
                return 0;
            _mm256_storeu_pd(w->q + r, rc);
            max4 = _mm256_max_pd(max4, magnitude);
        }
        __m128d pair = _mm_max_pd(_mm256_castpd256_pd128(max4), _mm256_extractf128_pd(max4, 1));
        maximum = _mm_cvtsd_f64(_mm_max_sd(pair, _mm_unpackhi_pd(pair, pair)));
    }
#endif
    for (; a < w->qp_active; ++a) {
        int r = w->qp_rows[a];
        double rc = w->s[r] * w->z[r];
        if (corrector)
            rc += w->ads[r] * w->adz[r] - target;
        if (!isfinite(rc))
            return 0;
        w->q[r] = rc;
        if (fabs(rc) > maximum)
            maximum = fabs(rc);
    }
    w->rhs_norm[3] = maximum;
    return 1;
}

/* H dx+G' dz=bd, G dx+ds=bp, Z ds+S dz=bc. */
static int dense_qp_linear_solve_rows(icore_ocp_workspace *w, const double *bd, const double *bp,
                                      const double *bc, double *dx, double *ds, double *dz,
                                      int box_only, int recover, int negate_rhs)
{
    int a, n = w->nu, contiguous_gamma = 0;
    const double *general_gamma = w->qp_weights;
    /* The first direction consumes -rd/-rp/-q directly. Apply the exact
     * sign change where each entry is used, avoiding three temporary
     * negative vectors. Refinement consumes already formed residuals. */
    if (negate_rhs)
        for (int i = 0; i < n; ++i)
            w->rhs[i] = -bd[i];
    else
        memcpy(w->rhs, bd, (size_t)n * sizeof(double));
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
    /* Independent divisions can execute in SIMD even when subsequent G'
     * contributions target the same variable. Keep those accumulations in
     * their original row order. The active-count certificate excludes holes. */
    {
        const __m256d sign = _mm256_set1_pd(-0.0), ceiling = _mm256_set1_pd(DBL_MAX);
        a = 0;
        while (a < w->qp_active) {
            int r = w->qp_rows[a];
            if (a + 3 < w->qp_active && w->qp_rows[a + 3] == r + 3) {
                __m256d pv = _mm256_loadu_pd(bp + r), cv = _mm256_loadu_pd(bc + r);
                if (negate_rhs) {
                    pv = _mm256_xor_pd(sign, pv);
                    cv = _mm256_xor_pd(sign, cv);
                }
                __m256d gamma = _mm256_div_pd(_mm256_fnmadd_pd(_mm256_loadu_pd(w->z + r), pv, cv),
                                              _mm256_loadu_pd(w->s + r));
                if (_mm256_movemask_pd(
                        _mm256_cmp_pd(_mm256_andnot_pd(sign, gamma), ceiling, _CMP_LE_OQ)) != 15)
                    return 0;
                _mm256_storeu_pd(w->pcone + r, gamma);
                a += 4;
            } else {
                double pv = negate_rhs ? -bp[r] : bp[r];
                double cv = negate_rhs ? -bc[r] : bc[r];
                double gamma = (cv - w->z[r] * pv) / w->s[r];
                if (!isfinite(gamma))
                    return 0;
                w->pcone[r] = gamma;
                ++a;
            }
        }
        contiguous_gamma = 1;
    }
#endif
#if ICORE_OCP_BOUND_PAIRS
    if (box_only && w->qp_bound_pairs && contiguous_gamma)
        bound_pairs_rhs(w);
    else
#endif
        if (contiguous_gamma && w->qp_packed_ready && w->qp_general_count == w->qp_active &&
            dense_qp_active_prefix(w))
        general_gamma = w->pcone;
    else
        for (a = 0; a < w->qp_active; ++a) {
            int r = w->qp_rows[a];
            double pv = negate_rhs ? -bp[r] : bp[r];
            double cv = negate_rhs ? -bc[r] : bc[r];
            double gamma = contiguous_gamma ? w->pcone[r] : (cv - w->z[r] * pv) / w->s[r];
            if (!contiguous_gamma && !isfinite(gamma))
                return 0;
            w->pcone[r] = gamma;
            if (!box_only && w->qp_packed_ready && w->qp_gindex[r] >= 0)
                w->qp_weights[w->qp_gindex[r]] = gamma;
            else {
                if (box_only)
                    w->rhs[w->qp_box_index[r]] += -gamma * w->qp_box_value[r];
                else
                    dense_qp_row_add(w, r, w->rhs, -gamma);
            }
        }
    int fused_direction = 0;
    if (w->qp_packed_ready && w->qp_general_count > 0 && w->qp_packed_factor &&
        !w->qp_diagonal_factor) {
        int status = icore_la_direction(w->qp_la, w->rhs, general_gamma, dx, w->qp_gx);
        if (status == ICORE_LA_OK)
            fused_direction = 1;
        else if (status != ICORE_LA_UNAVAILABLE)
            return 0;
    }
    if (!fused_direction && w->qp_packed_ready && w->qp_general_count > 0) {
        if (icore_la_gtmv(w->qp_la, general_gamma, w->qp_gt) != ICORE_LA_OK)
            return 0;
        axpy(w->rhs, w->qp_gt, -1.0, n);
    }
    if (!fused_direction) {
        if (w->qp_diagonal_factor) {
            for (int i = 0; i < n; ++i) {
                double diagonal = w->L[(size_t)i * n + i];
                dx[i] = (w->rhs[i] / diagonal) / diagonal;
            }
            if (!finite_array(dx, (size_t)n))
                return 0;
        } else if (w->qp_packed_factor) {
            if (icore_la_solve(w->qp_la, w->rhs, dx) != ICORE_LA_OK)
                return 0;
        } else {
            memcpy(dx, w->rhs, (size_t)n * sizeof(double));
            triangular_forward(w->L, dx, n);
            triangular_backward(w->L, dx, n);
            if (!finite_array(dx, (size_t)n))
                return 0;
        }
        if (w->qp_packed_ready && w->qp_general_count > 0 &&
            icore_la_gmv(w->qp_la, dx, w->qp_gx) != ICORE_LA_OK)
            return 0;
    }
    if (!recover)
        return 1;
#if ICORE_OCP_BOUND_PAIRS
    if (box_only && w->qp_bound_pairs && !negate_rhs)
        return bound_pairs_recover(w, dx, bp, ds, dz);
#endif
    a = 0;
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
    if (w->qp_packed_ready && w->qp_general_count == w->qp_active && dense_qp_active_prefix(w)) {
        const __m256d sign = _mm256_set1_pd(-0.0), ceiling = _mm256_set1_pd(DBL_MAX);
        for (; a + 3 < w->qp_active; a += 4) {
            __m256d gv = _mm256_loadu_pd(w->qp_gx + a);
            __m256d pv = _mm256_loadu_pd(bp + a);
            if (negate_rhs)
                pv = _mm256_xor_pd(sign, pv);
            __m256d sv = _mm256_sub_pd(pv, gv);
            __m256d zv = _mm256_fmadd_pd(_mm256_loadu_pd(w->orthant_inv + a), gv,
                                         _mm256_loadu_pd(w->pcone + a));
            __m256d valid =
                _mm256_and_pd(_mm256_cmp_pd(_mm256_andnot_pd(sign, sv), ceiling, _CMP_LE_OQ),
                              _mm256_cmp_pd(_mm256_andnot_pd(sign, zv), ceiling, _CMP_LE_OQ));
            if (_mm256_movemask_pd(valid) != 15)
                return 0;
            _mm256_storeu_pd(ds + a, sv);
            _mm256_storeu_pd(dz + a, zv);
        }
    }
#endif
    for (; a < w->qp_active; ++a) {
        int r = w->qp_rows[a], g = !box_only && w->qp_packed_ready ? w->qp_gindex[r] : -1;
        double gv = g >= 0 ? w->qp_gx[g]
                           : (box_only ? w->qp_box_value[r] * dx[w->qp_box_index[r]]
                                       : dense_qp_row_dot(w, r, dx));
        ds[r] = (negate_rhs ? -bp[r] : bp[r]) - gv;
        dz[r] = w->orthant_inv[r] * gv + w->pcone[r];
        if (!isfinite(ds[r]) || !isfinite(dz[r]))
            return 0;
    }
    return 1;
}

static int dense_qp_linear_solve_impl(icore_ocp_workspace *w, const double *bd, const double *bp,
                                      const double *bc, double *dx, double *ds, double *dz,
                                      int recover)
{
    if (w->qp_bound_pairs || (w->qp_packed_ready && w->qp_general_count == 0))
        return dense_qp_linear_solve_rows(w, bd, bp, bc, dx, ds, dz, 1, recover, 0);
    return dense_qp_linear_solve_rows(w, bd, bp, bc, dx, ds, dz, 0, recover, 0);
}

static int dense_qp_solve_first_rhs(icore_ocp_workspace *w)
{
    int box_only = w->qp_bound_pairs || (w->qp_packed_ready && w->qp_general_count == 0);
    return dense_qp_linear_solve_rows(w, w->rd, w->rp, w->q, w->dv, w->ds, w->dz, box_only, 0, 1);
}

static int dense_qp_linear_solve(icore_ocp_workspace *w, const double *bd, const double *bp,
                                 const double *bc, double *dx, double *ds, double *dz)
{
    return dense_qp_linear_solve_impl(w, bd, bp, bc, dx, ds, dz, 1);
}

/* The original unregularized dual equation, before singleton-row terms. */
ICORE_OCP_INLINE int dense_qp_linear_dual(icore_ocp_workspace *w, const double *dx, const double *dz,
                                     int cached_forward)
{
    int i, n = w->nu;
    if (w->qp_packed_ready && w->qp_general_count > 0 &&
        !(w->coefficient_cache_valid && w->cost_root_diagonal[0])) {
        int status = icore_la_hg_residual(w->qp_la, dx, dense_qp_general_values(w, dz), w->rd, w->brd,
                                      w->qp_gx, !cached_forward);
        if (status == ICORE_LA_OK)
            return 1;
        if (status != ICORE_LA_UNAVAILABLE && status != ICORE_LA_NONFINITE)
            return 0;
        /* Preserve the previous nonfinite residual/refinement sequence:
         * separate products and caller-side subtraction are the fallback. */
    }
    int fused = dense_qp_hg_product(w, dx, dz, w->brd, !cached_forward);
    if (fused < 0 || (!fused && !dense_qp_h_product(w, dx, w->brd)))
        return 0;
    if (w->qp_packed_ready && w->qp_general_count > 0) {
        /* The first residual follows linear_solve without changing dx or G.
         * Its G*dx is still in qp_gx; only G'*dz must be formed again.
         * A refined candidate has a new dx and recomputes both products. */
        if (!fused && cached_forward) {
            if (icore_la_gtmv(w->qp_la, dense_qp_general_values(w, dz), w->qp_gt) != ICORE_LA_OK)
                return 0;
        } else if (!fused && !dense_qp_g_pair(w, dx, dz))
            return 0;
    }
    /* Both original matrix products are ready. Retain (-rd-H*dx)-G'*dz
     * in one vector traversal; singleton rows are still accumulated later
     * in their original physical order. */
    int general = w->qp_packed_ready && w->qp_general_count > 0;
    for (i = 0; i < n; ++i) {
        w->brd[i] = -w->rd[i] - w->brd[i];
        if (general)
            w->brd[i] -= w->qp_gt[i];
    }
    return 1;
}

/* Residuals of the original unregularized Newton equations. */
static double dense_qp_linear_residual_rows(icore_ocp_workspace *w, const double *dx, const double *ds,
                                            const double *dz, int cached_forward, int box_only)
{
    int a, n = w->nu;
    double mp = 0.0, mc = 0.0, md;
    if (!dense_qp_linear_dual(w, dx, dz, cached_forward))
        return INFINITY;
    a = 0;
#if ICORE_OCP_BOUND_PAIRS
    if (box_only && w->qp_bound_pairs) {
        if (!bound_pairs_residual(w, dx, ds, dz, &mp, &mc))
            return INFINITY;
        a = w->qp_active;
    }
#endif
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
    if (w->qp_packed_ready && w->qp_general_count == w->qp_active && dense_qp_active_prefix(w)) {
        const __m256d sign = _mm256_set1_pd(-0.0), ceiling = _mm256_set1_pd(DBL_MAX);
        __m256d maxp = _mm256_setzero_pd(), maxc = _mm256_setzero_pd();
        for (; a + 3 < w->qp_active; a += 4) {
            __m256d rp =
                _mm256_sub_pd(_mm256_sub_pd(_mm256_xor_pd(sign, _mm256_loadu_pd(w->rp + a)),
                                            _mm256_loadu_pd(w->qp_gx + a)),
                              _mm256_loadu_pd(ds + a));
            __m256d rc = _mm256_fnmadd_pd(_mm256_loadu_pd(w->z + a), _mm256_loadu_pd(ds + a),
                                          _mm256_xor_pd(sign, _mm256_loadu_pd(w->q + a)));
            rc = _mm256_fnmadd_pd(_mm256_loadu_pd(w->s + a), _mm256_loadu_pd(dz + a), rc);
            __m256d ap = _mm256_andnot_pd(sign, rp), ac = _mm256_andnot_pd(sign, rc);
            __m256d valid = _mm256_and_pd(_mm256_cmp_pd(ap, ceiling, _CMP_LE_OQ),
                                          _mm256_cmp_pd(ac, ceiling, _CMP_LE_OQ));
            if (_mm256_movemask_pd(valid) != 15)
                return INFINITY;
            _mm256_storeu_pd(w->bcone + a, rp);
            _mm256_storeu_pd(w->brc + a, rc);
            maxp = _mm256_max_pd(maxp, ap);
            maxc = _mm256_max_pd(maxc, ac);
        }
        double pp[4], cc[4];
        _mm256_storeu_pd(pp, maxp);
        _mm256_storeu_pd(cc, maxc);
        for (int lane = 0; lane < 4; ++lane) {
            if (pp[lane] > mp)
                mp = pp[lane];
            if (cc[lane] > mc)
                mc = cc[lane];
        }
    }
#endif
    for (; a < w->qp_active; ++a) {
        int r = w->qp_rows[a], g = !box_only && w->qp_packed_ready ? w->qp_gindex[r] : -1;
        double gv = g >= 0 ? w->qp_gx[g]
                           : (box_only ? w->qp_box_value[r] * dx[w->qp_box_index[r]]
                                       : dense_qp_row_dot(w, r, dx));
        double rp = -w->rp[r] - gv - ds[r];
        double rc = -w->q[r] - w->z[r] * ds[r] - w->s[r] * dz[r];
        if (g < 0) {
            if (box_only)
                w->brd[w->qp_box_index[r]] += -dz[r] * w->qp_box_value[r];
            else
                dense_qp_row_add(w, r, w->brd, -dz[r]);
        }
        w->bcone[r] = rp;
        w->brc[r] = rc;
        if (!isfinite(rp) || !isfinite(rc))
            return INFINITY;
        if (fabs(rp) > mp)
            mp = fabs(rp);
        if (fabs(rc) > mc)
            mc = fabs(rc);
    }
    md = norm_inf(w->brd, (size_t)n) / (1.0 + w->rhs_norm[0]);
    mp /= 1.0 + w->rhs_norm[1];
    mc /= 1.0 + w->rhs_norm[3];
    return fmax(md, fmax(mp, mc));
}

static double dense_qp_linear_residual(icore_ocp_workspace *w, const double *dx, const double *ds,
                                       const double *dz, int cached_forward)
{
    if (w->qp_bound_pairs || (w->qp_packed_ready && w->qp_general_count == 0))
        return dense_qp_linear_residual_rows(w, dx, ds, dz, cached_forward, 1);
    return dense_qp_linear_residual_rows(w, dx, ds, dz, cached_forward, 0);
}

/* A first direction has unchanged dx and original right-hand sides. Recover
 * general rows before their transpose product, then recover singleton rows
 * together with all three original Newton-equation checks. Thus the dual
 * arithmetic remains H, G_general' dz, singleton rows in physical order.
 * No residual is inferred from the factorized/regularized system. Refinement
 * candidates and roundoff-sensitive models retain the separate original path.
 * On a rejected intermediate the caller rebuilds the RHS and runs that path. */
static int dense_qp_finish_first_direction_rows(icore_ocp_workspace *w, double *error, int box_only)
{
    int a = 0, n = w->nu;
    double mp = 0.0, mc = 0.0;
    const double *dx = w->dv;
    double *ds = w->ds, *dz = w->dz;
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
    int count = w->qp_general_count;
    int begin = count ? w->qp_general_rows[0] : 0;
    if (count && w->qp_general_rows[count - 1] - begin == count - 1) {
        const __m256d sign = _mm256_set1_pd(-0.0), ceiling = _mm256_set1_pd(DBL_MAX);
        __m256d maxp = _mm256_setzero_pd(), maxc = maxp;
        for (; a + 3 < count; a += 4) {
            __m256d gv = _mm256_loadu_pd(w->qp_gx + a);
            __m256d sv = _mm256_sub_pd(_mm256_xor_pd(sign, _mm256_loadu_pd(w->rp + begin + a)), gv);
            __m256d zv = _mm256_fmadd_pd(_mm256_loadu_pd(w->orthant_inv + begin + a), gv,
                                         _mm256_loadu_pd(w->pcone + begin + a));
            __m256d valid =
                _mm256_and_pd(_mm256_cmp_pd(_mm256_andnot_pd(sign, sv), ceiling, _CMP_LE_OQ),
                              _mm256_cmp_pd(_mm256_andnot_pd(sign, zv), ceiling, _CMP_LE_OQ));
            if (_mm256_movemask_pd(valid) != 15)
                return 0;
            __m256d rp = _mm256_sub_pd(
                _mm256_sub_pd(_mm256_xor_pd(sign, _mm256_loadu_pd(w->rp + begin + a)), gv), sv);
            __m256d rc = _mm256_fnmadd_pd(_mm256_loadu_pd(w->z + begin + a), sv,
                                          _mm256_xor_pd(sign, _mm256_loadu_pd(w->q + begin + a)));
            rc = _mm256_fnmadd_pd(_mm256_loadu_pd(w->s + begin + a), zv, rc);
            __m256d ap = _mm256_andnot_pd(sign, rp), ac = _mm256_andnot_pd(sign, rc);
            valid = _mm256_and_pd(_mm256_cmp_pd(ap, ceiling, _CMP_LE_OQ),
                                  _mm256_cmp_pd(ac, ceiling, _CMP_LE_OQ));
            if (_mm256_movemask_pd(valid) != 15)
                return 0;
            _mm256_storeu_pd(ds + begin + a, sv);
            _mm256_storeu_pd(dz + begin + a, zv);
            _mm256_storeu_pd(w->bcone + begin + a, rp);
            _mm256_storeu_pd(w->brc + begin + a, rc);
            maxp = _mm256_max_pd(maxp, ap);
            maxc = _mm256_max_pd(maxc, ac);
        }
        double pp[4], cc[4];
        _mm256_storeu_pd(pp, maxp);
        _mm256_storeu_pd(cc, maxc);
        for (int lane = 0; lane < 4; ++lane) {
            if (pp[lane] > mp)
                mp = pp[lane];
            if (cc[lane] > mc)
                mc = cc[lane];
        }
    }
#endif
    for (; a < w->qp_general_count; ++a) {
        int r = w->qp_general_rows[a];
        double gv = w->qp_gx[a], sv = -w->rp[r] - gv;
        double zv = w->orthant_inv[r] * gv + w->pcone[r];
        if (!isfinite(sv) || !isfinite(zv))
            return 0;
        double rp = -w->rp[r] - gv - sv;
        double rc = -w->q[r] - w->z[r] * sv - w->s[r] * zv;
        if (!isfinite(rp) || !isfinite(rc))
            return 0;
        ds[r] = sv;
        dz[r] = zv;
        w->bcone[r] = rp;
        w->brc[r] = rc;
        if (fabs(rp) > mp)
            mp = fabs(rp);
        if (fabs(rc) > mc)
            mc = fabs(rc);
    }
    if (!dense_qp_linear_dual(w, dx, dz, 1))
        return 0;
#if ICORE_OCP_BOUND_PAIRS
    if (w->qp_bound_pairs) {
        if (!bound_pairs_finish(w, dx, ds, dz, &mp, &mc))
            return 0;
    } else
#endif
        for (a = 0; a < w->qp_active; ++a) {
            int r = w->qp_rows[a];
            if (w->qp_gindex[r] >= 0)
                continue;
            int j = w->qp_box_index[r];
            double gv = w->qp_box_value[r] * dx[j], sv = -w->rp[r] - gv;
            double zv = w->orthant_inv[r] * gv + w->pcone[r];
            if (!isfinite(sv) || !isfinite(zv))
                return 0;
            const double *rg = w->qp_box_value, *rx = dx;
#if defined(__GNUC__) || defined(__clang__)
            /* The separate residual can contract g*dx into -rp. Sharing the
         * rounded recovery product would instead cancel with ds. Preserve
         * that independent original-equation arithmetic without a row pass. */
            __asm__ __volatile__("" : "+r"(rg), "+r"(rx));
#endif
            double rgv = box_only ? rg[r] * rx[j] : dense_qp_row_dot(w, r, rx);
            double rp = -w->rp[r] - rgv - sv;
            double rc = -w->q[r] - w->z[r] * sv - w->s[r] * zv;
            if (!isfinite(rp) || !isfinite(rc))
                return 0;
            ds[r] = sv;
            dz[r] = zv;
            w->brd[j] += -zv * w->qp_box_value[r];
            w->bcone[r] = rp;
            w->brc[r] = rc;
            if (fabs(rp) > mp)
                mp = fabs(rp);
            if (fabs(rc) > mc)
                mc = fabs(rc);
        }
    double md = norm_inf(w->brd, (size_t)n) / (1.0 + w->rhs_norm[0]);
    mp /= 1.0 + w->rhs_norm[1];
    mc /= 1.0 + w->rhs_norm[3];
    *error = fmax(md, fmax(mp, mc));
    return isfinite(*error);
}

static int dense_qp_finish_first_direction(icore_ocp_workspace *w, double *error)
{
    if (w->qp_bound_pairs || w->qp_general_count == 0)
        return dense_qp_finish_first_direction_rows(w, error, 1);
    return dense_qp_finish_first_direction_rows(w, error, 0);
}

ICORE_OCP_INLINE void dense_qp_first_rhs(icore_ocp_workspace *w)
{
    for (int i = 0; i < w->nu; ++i)
        w->brd[i] = -w->rd[i];
    int begin = dense_qp_active_begin(w);
    if (begin >= 0) {
        for (int a = begin; a < begin + w->qp_active; ++a) {
            w->bcone[a] = -w->rp[a];
            w->brc[a] = -w->q[a];
        }
    } else {
        for (int a = 0; a < w->qp_active; ++a) {
            int r = w->qp_rows[a];
            w->bcone[r] = -w->rp[r];
            w->brc[r] = -w->q[r];
        }
    }
}

static int dense_qp_direction(void *context, int corrector)
{
    ocp_context *c = (ocp_context *)context;
    icore_ocp_workspace *w = c->w;
    int i, a, iteration, n = w->nu;
    double error, limit = ipm_refinement_limit(c->options);
    int fused = w->qp_packed_ready && w->qp_reuse_forward;
    if (fused) {
        if (!dense_qp_solve_first_rhs(w))
            return 0;
        if (!dense_qp_finish_first_direction(w, &error)) {
            /* Fused scratch may be partial. Rebuild every original RHS and
             * retry the existing complete direction/residual, never a stale
             * residual or a modified stopping/refinement condition. */
            fused = 0;
        }
    }
    if (!fused) {
        dense_qp_first_rhs(w);
        if (!dense_qp_linear_solve(w, w->brd, w->bcone, w->brc, w->dv, w->ds, w->dz))
            return 0;
        error = dense_qp_linear_residual(w, w->dv, w->ds, w->dz, w->qp_reuse_forward);
    }
    for (iteration = 0; iteration < c->options->refinement_steps && error > limit; ++iteration) {
        double next_error;
        if (!dense_qp_linear_solve(w, w->brd, w->bcone, w->brc, w->cv, w->tmp, w->cz))
            break;
        for (i = 0; i < n; ++i)
            w->tv[i] = w->dv[i] + w->cv[i];
        for (a = 0; a < w->qp_active; ++a) {
            int r = w->qp_rows[a];
            w->ads[r] = w->ds[r] + w->tmp[r];
            w->tz[r] = w->dz[r] + w->cz[r];
        }
        next_error = dense_qp_linear_residual(w, w->tv, w->ads, w->tz, 0);
        if (!(next_error < error))
            break;
        memcpy(w->dv, w->tv, (size_t)n * sizeof(double));
        for (a = 0; a < w->qp_active; ++a) {
            int r = w->qp_rows[a];
            w->ds[r] = w->ads[r];
            w->dz[r] = w->tz[r];
        }
        error = next_error;
        ++w->refinements;
    }
    if (!corrector) {
        int begin = dense_qp_active_begin(w);
        if (begin >= 0) {
            memcpy(w->ads + begin, w->ds + begin, (size_t)w->qp_active * sizeof(double));
            memcpy(w->adz + begin, w->dz + begin, (size_t)w->qp_active * sizeof(double));
        } else
            for (a = 0; a < w->qp_active; ++a) {
                int r = w->qp_rows[a];
                w->ads[r] = w->ds[r];
                w->adz[r] = w->dz[r];
            }
    }
    return 1;
}

/* ---- Step bounds, interior checks, updates, and the immutable callback table. ---- */

static void dense_qp_step_bounds(void *context, int corrector, double *primal, double *dual)
{
    ocp_context *c = (ocp_context *)context;
    icore_ocp_workspace *w = c->w;
    const double *ds = corrector ? w->ds : w->ads, *dz = corrector ? w->dz : w->adz;
    int begin = dense_qp_active_begin(w);
    const int *rows = begin >= 0 ? NULL : w->qp_rows;
    int offset = begin >= 0 ? begin : 0;
    *primal = icore_ocp_orthant_step(w->s + offset, ds + offset, rows, w->qp_active);
    *dual = icore_ocp_orthant_step(w->z + offset, dz + offset, rows, w->qp_active);
}

static double dense_qp_affine_pair(void *context, double primal, double dual)
{
    ocp_context *c = (ocp_context *)context;
    icore_ocp_workspace *w = c->w;
    int a;
    double value = 0.0;
    for (a = 0; a < w->qp_active; ++a) {
        int r = w->qp_rows[a];
        value += (w->s[r] + primal * w->ads[r]) * (w->z[r] + dual * w->adz[r]);
    }
    return value;
}

static int dense_qp_interior(void *context, double alpha)
{
    ocp_context *c = (ocp_context *)context;
    icore_ocp_workspace *w = c->w;
    int a, i;
    for (i = 0; i < w->nu; ++i)
        if (!isfinite(w->v[i] + alpha * w->dv[i]))
            return 0;
    a = 0;
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
    int begin = dense_qp_active_begin(w);
    if (begin >= 0) {
        const __m256d av = _mm256_set1_pd(alpha), zero = _mm256_setzero_pd();
        const __m256d maximum = _mm256_set1_pd(DBL_MAX);
        for (; a + 3 < w->qp_active; a += 4) {
            __m256d sv = _mm256_fmadd_pd(av, _mm256_loadu_pd(w->ds + begin + a),
                                         _mm256_loadu_pd(w->s + begin + a));
            __m256d zv = _mm256_fmadd_pd(av, _mm256_loadu_pd(w->dz + begin + a),
                                         _mm256_loadu_pd(w->z + begin + a));
            __m256d valid = _mm256_and_pd(_mm256_cmp_pd(sv, zero, _CMP_GT_OQ),
                                          _mm256_cmp_pd(zv, zero, _CMP_GT_OQ));
            valid = _mm256_and_pd(valid, _mm256_and_pd(_mm256_cmp_pd(sv, maximum, _CMP_LE_OQ),
                                                       _mm256_cmp_pd(zv, maximum, _CMP_LE_OQ)));
            if (_mm256_movemask_pd(valid) != 15)
                return 0;
        }
    }
#endif
    for (; a < w->qp_active; ++a) {
        int r = w->qp_rows[a];
        double s = w->s[r] + alpha * w->ds[r], z = w->z[r] + alpha * w->dz[r];
        if (!(s > 0.0 && z > 0.0) || !isfinite(s) || !isfinite(z))
            return 0;
    }
    return 1;
}

static void dense_qp_update(void *context, double alpha)
{
    ocp_context *c = (ocp_context *)context;
    icore_ocp_workspace *w = c->w;
    int a;
    axpy(w->v, w->dv, alpha, w->nu);
    int begin = dense_qp_active_begin(w);
    if (begin >= 0) {
        axpy(w->s + begin, w->ds + begin, alpha, w->qp_active);
        axpy(w->z + begin, w->dz + begin, alpha, w->qp_active);
        dense_qp_expand(w);
        return;
    }
    for (a = 0; a < w->qp_active; ++a) {
        int r = w->qp_rows[a];
        w->s[r] += alpha * w->ds[r];
        w->z[r] += alpha * w->dz[r];
    }
    dense_qp_expand(w);
}

static int dense_qp_accept_affine(void *context, double alpha, ipm_residual *result)
{
    return accept_affine_trial((ocp_context *)context, alpha, dense_qp_evaluate, result, 1);
}

static const ipm_kernel dense_qp_kernel = {
    dense_qp_initialize, dense_qp_evaluate,     dense_qp_factor,      dense_qp_form_rhs,
    dense_qp_direction,  dense_qp_step_bounds,  dense_qp_affine_pair, dense_qp_interior,
    dense_qp_update,     dense_qp_accept_affine};
