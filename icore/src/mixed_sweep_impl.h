/* Experimental SP factor/sweep bank. Public state and original residuals stay
 * double. Its token never certifies the unrelated double C storage. */
#if ICORE_OCP_LA_MIXED_SUPPORTED
int icore_ocp_la_mixed_sweeps_available(const icore_ocp_la_workspace *w)
{
    return w && w->mixed && w->mixed->sweep_storage;
}

void icore_ocp_la_mixed_sweeps_invalidate(icore_ocp_la_workspace *w)
{
    if (w && w->mixed)
        w->mixed->sweep_next = -2;
}

int icore_ocp_la_mixed_sweeps_active(const icore_ocp_la_workspace *w)
{
    return w && w->mixed && w->mixed->sweep_storage && w->mixed->sweep_next == -1 &&
           w->sqrt_next == -2 && mixed_environment();
}

static int mixed_sweep_root_offset(const icore_ocp_la_workspace *w, int stage)
{
    return stage < w->N ? w->nr : 0;
}

static int mixed_sweep_reject(icore_ocp_la_workspace *w, int status)
{
    icore_ocp_la_mixed_sweeps_invalidate(w);
    return sqrt_reject(w, status);
}

/* Same fresh complete-square certificate as the converted-factor path.
 * No DP factor is written. A failed certificate rejects the whole SP chain. */
static ICORE_OCP_LA_AVX2_TARGET int mixed_sweep_check_fresh(struct blasfeo_smat *S, int n, int nr, double cf,
                                                   double sf)
{
    int i = 0;
    const __m128i zero = _mm_setzero_si128(), exponent = _mm_set1_epi32(0x7f800000);
    const __m128i offsets = _mm_set_epi32(27, 18, 9, 0);
    if (!mixed_environment() || S->use_dA != 1)
        return ICORE_LA_UNAVAILABLE;
    for (; i + 3 < n; i += 4) {
        const float *panel = S->pA + (size_t)(i & ~7) * S->cn + (i & 7) + 8 * (size_t)i;
        __m128 diagonal = _mm_i32gather_ps(panel, offsets, 4);
        __m128i dbits = _mm_castps_si128(diagonal),
                ibits = _mm_castps_si128(_mm_loadu_ps(S->dA + i));
        __m128i bad = _mm_or_si128(_mm_cmpeq_epi32(_mm_and_si128(dbits, exponent), exponent),
                                   _mm_cmpeq_epi32(_mm_and_si128(ibits, exponent), exponent));
        __m256d d, pivot, floors;
        if (_mm_movemask_epi8(bad) || _mm_movemask_epi8(_mm_cmpgt_epi32(dbits, zero)) != 65535 ||
            _mm_movemask_epi8(_mm_cmpgt_epi32(ibits, zero)) != 65535)
            return ICORE_LA_NOT_SPD;
        d = _mm256_cvtps_pd(diagonal);
        pivot = _mm256_mul_pd(d, d);
        floors = _mm256_set_pd(i + 3 < nr ? cf : sf, i + 2 < nr ? cf : sf, i + 1 < nr ? cf : sf,
                               i < nr ? cf : sf);
        if (_mm256_movemask_pd(_mm256_cmp_pd(pivot, floors, _CMP_LT_OQ)))
            return ICORE_OCP_LA_PIVOT_REJECTED;
    }
    for (; i < n; ++i) {
        uint32_t dbits, ibits;
        double d;
        memcpy(&dbits, &BLASFEO_SMATEL(S, i, i), sizeof(dbits));
        memcpy(&ibits, S->dA + i, sizeof(ibits));
        if ((dbits & UINT32_C(0x80000000)) || !(dbits & UINT32_C(0x7fffffff)) ||
            (dbits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000) ||
            (ibits & UINT32_C(0x80000000)) || !(ibits & UINT32_C(0x7fffffff)) ||
            (ibits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000))
            return ICORE_LA_NOT_SPD;
        d = (double)BLASFEO_SMATEL(S, i, i);
        if (d * d < (i < nr ? cf : sf))
            return ICORE_OCP_LA_PIVOT_REJECTED;
    }
    return ICORE_LA_OK;
}

int icore_ocp_la_mixed_sweeps_begin(icore_ocp_la_workspace *w, const double *diagonal, double reg, double absolute,
                           double relative)
{
    int i, status;
    double maximum = 0.0, floor;
    struct icore_ocp_la_mixed *m;
    if (!icore_ocp_la_mixed_sweeps_available(w) || !icore_ocp_la_mixed_ready(w) || !diagonal ||
        !sqrt_arguments(reg, absolute, relative))
        return mixed_sweep_reject(w, ICORE_LA_UNAVAILABLE);
    m = w->mixed;
    m->terminal_diagonal = 0;
    w->sqrt_next = -2;
    w->sqrt_equilibrated = 0;
    w->sqrt_root_omitted = 0;
    w->sqrt_initial_zero = 0;
    sqrt_recovery_clear(w);
    for (i = 0; i < w->nx; ++i) {
        double d = diagonal[i] + reg;
        float f;
        if (!mixed_cast(d, &f))
            return mixed_sweep_reject(w, ICORE_LA_NONFINITE);
        BLASFEO_SMATEL(m->cost + w->N, i, i) = f;
        if (fabs(d) > maximum)
            maximum = fabs(d);
    }
    floor = fmax(absolute, relative * maximum);
    if (!isfinite(floor))
        return mixed_sweep_reject(w, ICORE_LA_INVALID);
    m->cost[w->N].use_dA = 0;
    if (w->cost_loaded[w->N] == 2) {
        struct blasfeo_smat *H = m->cost + w->N, *F = m->C + w->N;
        int positive = 1;
        for (i = 0; i < w->nx; ++i)
            positive &= BLASFEO_SMATEL(H, i, i) > 0.0f;
        if (positive) {
            blasfeo_sgese(w->nx, w->nx, 0.0f, F, 0, 0);
            for (i = 0; i < w->nx; ++i) {
                float d = BLASFEO_SMATEL(H, i, i), inverse = 1.0f / sqrtf(d);
                F->dA[i] = inverse;
                BLASFEO_SMATEL(F, i, i) = d * inverse;
            }
            F->use_dA = 1;
            m->terminal_diagonal = 1;
        } else
            blasfeo_spotrf_l(w->nx, H, 0, 0, F, 0, 0);
    } else
        blasfeo_spotrf_l(w->nx, m->cost + w->N, 0, 0, m->C + w->N, 0, 0);
    status = mixed_sweep_check_fresh(m->C + w->N, w->nx, 0, floor, floor);
    if (status != ICORE_LA_OK)
        return mixed_sweep_reject(w, status);
    m->sweep_next = w->N - 1;
    return ICORE_LA_OK;
}

ICORE_OCP_LA_AVX2_TARGET int icore_ocp_la_mixed_sweeps_stage(icore_ocp_la_workspace *w, int stage, const double *diagonal,
                                           double reg, double absolute, double relative)
{
    int status, rows, offset;
    struct icore_ocp_la_mixed *m;
    if (!w || !w->mixed || !w->mixed->sweep_storage || stage < 0 || stage >= w->N || !diagonal ||
        w->mixed->sweep_next != stage || w->sqrt_next != -2 || !mixed_environment() ||
        !sqrt_arguments(reg, absolute, relative))
        return mixed_sweep_reject(w, ICORE_LA_INVALID);
    m = w->mixed;
    rows = stage == 0 ? w->nr : w->n;
    offset = mixed_sweep_root_offset(w, stage + 1);
    if ((offset & 3) || !m->dvalid[stage] || !m->hvalid[stage])
        return mixed_sweep_reject(w, ICORE_LA_UNAVAILABLE);
    if (stage == w->N - 1 && m->terminal_diagonal) {
        int i, j;
        for (i = 0; i + 7 < rows; i += 8) {
            const float *source = m->DT[stage].pA + (size_t)i * m->DT[stage].cn;
            float *target = m->DTP.pA + (size_t)i * m->DTP.cn;
            for (j = 0; j < w->nx; ++j)
                _mm256_storeu_ps(target + 8 * (size_t)j,
                                 _mm256_mul_ps(_mm256_loadu_ps(source + 8 * (size_t)j),
                                               _mm256_set1_ps(BLASFEO_SMATEL(m->C + w->N, j, j))));
        }
        for (; i < rows; ++i)
            for (j = 0; j < w->nx; ++j)
                BLASFEO_SMATEL(&m->DTP, i, j) =
                    BLASFEO_SMATEL(m->DT + stage, i, j) * BLASFEO_SMATEL(m->C + w->N, j, j);
    } else
        blasfeo_strmm_rlnn(rows, w->nx, 1.0f, m->C + stage + 1, offset, offset, m->DT + stage, 0, 0,
                           &m->DTP, 0, 0);
    status = mixed_stage_factor_checked(w, stage, diagonal, reg, absolute, relative, rows, 1);
    if (status != ICORE_LA_OK)
        return mixed_sweep_reject(w, status);
    m->sweep_next = stage - 1;
    return ICORE_LA_OK;
}

static ICORE_OCP_LA_AVX2_TARGET int mixed_sweep_cast_vector(const double *x, float *out, int n)
{
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 value;
        if (!mixed_cache_cast4(_mm256_loadu_pd(x + i), &value))
            return 0;
        _mm_storeu_ps(out + i, value);
    }
    for (; i < n; ++i)
        if (!mixed_cast(x[i], out + i))
            return 0;
    memset(out + n, 0, 8 * sizeof(float));
    return 1;
}

static ICORE_OCP_LA_AVX2_TARGET int mixed_sweep_finite(const float *x, int n)
{
    int i = 0;
    const __m256i exponent = _mm256_set1_epi32(0x7f800000);
    for (; i + 8 <= n; i += 8) {
        __m256i bits = _mm256_castps_si256(_mm256_loadu_ps(x + i));
        if (_mm256_movemask_epi8(_mm256_cmpeq_epi32(_mm256_and_si256(bits, exponent), exponent)))
            return 0;
    }
    for (; i < n; ++i)
        if (!isfinite(x[i]))
            return 0;
    return 1;
}

static ICORE_OCP_LA_AVX2_TARGET void mixed_sweep_export(const float *x, double *out, int n)
{
    int i = 0;
    for (; i + 4 <= n; i += 4)
        _mm256_storeu_pd(out + i, _mm256_cvtps_pd(_mm_loadu_ps(x + i)));
    for (; i < n; ++i)
        out[i] = (double)x[i];
}

static int mixed_sweep_add(float *out, const float *a, const float *b, int n)
{
    int i;
    for (i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
    return mixed_sweep_finite(out, n);
}

static ICORE_OCP_LA_AVX2_TARGET void mixed_terminal_root_multiply(icore_ocp_la_workspace *w, const float *x, float *y)
{
    int i;
    struct blasfeo_smat *F = w->mixed->C + w->N;
    const __m256i offsets = _mm256_set_epi32(63, 54, 45, 36, 27, 18, 9, 0);
    /* SP eligibility already requires nx to be a multiple of eight. */
    for (i = 0; i < w->nx; i += 8) {
        const float *panel = F->pA + (size_t)i * F->cn + 8 * (size_t)i;
        __m256 diagonal = _mm256_i32gather_ps(panel, offsets, 4);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(diagonal, _mm256_loadu_ps(x + i)));
    }
}

static int mixed_sweep_root_apply(icore_ocp_la_workspace *w, int stage, struct blasfeo_svec *x,
                                  struct blasfeo_svec *tmp, struct blasfeo_svec *out)
{
    struct icore_ocp_la_mixed *m = w->mixed;
    int offset = mixed_sweep_root_offset(w, stage);
    memset(x->pa + w->nx, 0, 8 * sizeof(float));
    if (stage == w->N && m->terminal_diagonal)
        mixed_terminal_root_multiply(w, x->pa, tmp->pa);
    else
        blasfeo_strmv_ltn(w->nx, m->C + stage, offset, offset, x, 0, tmp, 0);
    if (!mixed_sweep_finite(tmp->pa, w->nx))
        return 0;
    memset(tmp->pa + w->nx, 0, 8 * sizeof(float));
    if (stage == w->N && m->terminal_diagonal)
        mixed_terminal_root_multiply(w, tmp->pa, out->pa);
    else
        blasfeo_strmv_lnn(w->nx, m->C + stage, offset, offset, tmp, 0, out, 0);
    return mixed_sweep_finite(out->pa, w->nx);
}

int icore_ocp_la_mixed_sweeps_backward(icore_ocp_la_workspace *w, int stage, const double *bd, const double *d,
                              const double *pnext, int reuse_pd, double *pd, double *a, double *p,
                              int suffix_ready)
{
    int i, rows, zero;
    struct icore_ocp_la_mixed *m;
    if (!w || stage < 0 || stage >= w->N || !bd || !pnext || !a ||
        (reuse_pd != 0 && reuse_pd != 1) || (reuse_pd ? !pd : !d) ||
        (suffix_ready != 0 && suffix_ready != 1))
        return ICORE_LA_INVALID;
    m = w->mixed;
    if (!m || !m->sweep_storage || w->sqrt_next != -2 ||
        m->sweep_next != (suffix_ready ? stage - 1 : -1) || !mixed_environment())
        return ICORE_LA_NOT_READY;
    zero = stage == 0;
    rows = zero ? w->nr : w->n;
    if (zero ? p != NULL : p == NULL)
        return ICORE_LA_INVALID;
    if ((!zero && vector_spans_overlap(a, w->nr, p, w->nx)) ||
        (pd && (vector_spans_overlap(a, w->nr, pd, w->nx) ||
                (!zero && vector_spans_overlap(p, w->nx, pd, w->nx)))))
        return ICORE_LA_INVALID;
    if (reuse_pd) {
        if (!mixed_sweep_cast_vector(pd, m->V.pa, w->nx))
            return ICORE_LA_NONFINITE;
    } else {
        if (!mixed_sweep_cast_vector(d, m->X.pa, w->nx) ||
            !mixed_sweep_root_apply(w, stage + 1, &m->X, &m->Y, &m->V))
            return ICORE_LA_NONFINITE;
    }
    if (!mixed_sweep_cast_vector(pnext, m->X.pa, w->nx) ||
        !mixed_sweep_add(m->X.pa, m->X.pa, m->V.pa, w->nx) ||
        !mixed_sweep_cast_vector(bd, m->Z.pa, rows))
        return ICORE_LA_NONFINITE;
    for (i = 0; i < rows; ++i)
        m->Z.pa[i] = -m->Z.pa[i];
    blasfeo_sgemv_n(rows, w->nx, 1.0f, m->DT + stage, 0, 0, &m->X, 0, 1.0f, &m->Z, 0, &m->Y, 0);
    if (!mixed_sweep_finite(m->Y.pa, rows))
        return ICORE_LA_NONFINITE;
    blasfeo_strsv_lnn_mn(rows, w->nr, m->C + stage, 0, 0, &m->Y, 0, &m->Y, 0);
    if (!mixed_sweep_finite(m->Y.pa, rows))
        return ICORE_LA_NONFINITE;
    mixed_sweep_export(m->Y.pa, a, w->nr);
    if (!zero)
        mixed_sweep_export(m->Y.pa + w->nr, p, w->nx);
    if (!reuse_pd && pd)
        mixed_sweep_export(m->V.pa, pd, w->nx);
    return ICORE_LA_OK;
}

int icore_ocp_la_mixed_sweeps_forward(icore_ocp_la_workspace *w, int stage, const double *a, const double *x,
                             const double *d, const double *pnext, double *u, double *next,
                             double *dual)
{
    int i, rows, zero;
    struct icore_ocp_la_mixed *m;
    if (!w || stage < 0 || stage >= w->N || !a || !d || !pnext || !u || !next || !dual)
        return ICORE_LA_INVALID;
    if (!icore_ocp_la_mixed_sweeps_active(w))
        return ICORE_LA_NOT_READY;
    m = w->mixed;
    zero = stage == 0;
    rows = zero ? w->nr : w->n;
    if (zero ? x != NULL : x == NULL)
        return ICORE_LA_INVALID;
    if (vector_spans_overlap(u, w->nr, next, w->nx) ||
        vector_spans_overlap(u, w->nr, dual, w->nx) ||
        vector_spans_overlap(next, w->nx, dual, w->nx))
        return ICORE_LA_INVALID;
    if (!mixed_sweep_cast_vector(a, m->X.pa, w->nr))
        return ICORE_LA_NONFINITE;
    for (i = 0; i < w->nr; ++i)
        m->X.pa[i] = -m->X.pa[i];
    if (!zero && !mixed_sweep_cast_vector(x, m->X.pa + w->nr, w->nx))
        return ICORE_LA_NONFINITE;
    blasfeo_strsv_ltn_mn(rows, w->nr, m->C + stage, 0, 0, &m->X, 0, &m->X, 0);
    if (!mixed_sweep_finite(m->X.pa, rows))
        return ICORE_LA_NONFINITE;
    memset(m->Z.pa, 0, (size_t)w->nx * sizeof(float));
    blasfeo_sgemv_t(rows, w->nx, 1.0f, m->DT + stage, 0, 0, &m->X, 0, 0.0f, &m->Z, 0, &m->Y, 0);
    if (!mixed_sweep_cast_vector(d, m->Z.pa, w->nx) ||
        !mixed_sweep_add(m->Y.pa, m->Y.pa, m->Z.pa, w->nx) ||
        !mixed_sweep_root_apply(w, stage + 1, &m->Y, &m->Z, &m->V))
        return ICORE_LA_NONFINITE;
    if (!mixed_sweep_cast_vector(pnext, m->Z.pa, w->nx) ||
        !mixed_sweep_add(m->V.pa, m->V.pa, m->Z.pa, w->nx))
        return ICORE_LA_NONFINITE;
    mixed_sweep_export(m->X.pa, u, w->nr);
    mixed_sweep_export(m->Y.pa, next, w->nx);
    mixed_sweep_export(m->V.pa, dual, w->nx);
    return ICORE_LA_OK;
}
#else
#include "mixed_sweep_stubs.h"
#endif
