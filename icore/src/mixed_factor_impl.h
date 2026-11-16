/* Optional FP32 factors and equality sweeps. FP64 state and original-system
 * residuals are owned by ocp.c. The caller's original-system residual gate
 * accepts each mixed direction; rejected chains are rebuilt in DP. */
#if ICORE_OCP_LA_MIXED_SUPPORTED
#include <blasfeo_s_aux.h>
#include <blasfeo_s_blasfeo_api.h>
#include <float.h>
#include <fenv.h>

struct icore_ocp_la_mixed {
    void *storage, *sweep_storage;
    int sweep_next;
    int terminal_diagonal;
    struct blasfeo_svec X, Y, Z, V;
    struct blasfeo_smat *DT, *C, *cost, DTP, M;
    unsigned char *dvalid, *hvalid;
};

static void mixed_destroy(icore_ocp_la_workspace *w)
{
    struct icore_ocp_la_mixed *m = w->mixed;
    if (!m)
        return;
    free(m->sweep_storage);
    free(m->storage);
    free(m->DT);
    free(m->C);
    free(m->cost);
    free(m->dvalid);
    free(m->hvalid);
    free(m);
    w->mixed = NULL;
}

static void mixed_create(icore_ocp_la_workspace *w)
{
    size_t hbytes, dbytes, tbytes, total;
    int k;
    unsigned char *next;
    struct icore_ocp_la_mixed *m;
    if (!w->owned_factors || (w->nr & 3) || (w->nx & 7))
        return;
    hbytes = align64(blasfeo_memsize_smat(w->n, w->n));
    dbytes = align64(blasfeo_memsize_smat(w->n, w->nx));
    tbytes = align64(blasfeo_memsize_smat(w->nx, w->nx));
    if ((size_t)w->N > (SIZE_MAX - dbytes - hbytes - 2 * tbytes - 63u) / (dbytes + 2 * hbytes))
        return;
    total = (size_t)w->N * (dbytes + 2 * hbytes) + dbytes + hbytes + 2 * tbytes;
    m = (struct icore_ocp_la_mixed *)calloc(1, sizeof(*m));
    if (!m)
        return;
    w->mixed = m;
    m->DT = (struct blasfeo_smat *)calloc((size_t)w->N, sizeof(*m->DT));
    m->C = (struct blasfeo_smat *)calloc((size_t)w->N + 1, sizeof(*m->C));
    m->cost = (struct blasfeo_smat *)calloc((size_t)w->N + 1, sizeof(*m->cost));
    m->dvalid = (unsigned char *)calloc((size_t)w->N, 1);
    m->hvalid = (unsigned char *)calloc((size_t)w->N + 1, 1);
    m->storage = calloc(1, total + 63u);
    if (!m->DT || !m->C || !m->cost || !m->dvalid || !m->hvalid || !m->storage) {
        mixed_destroy(w);
        return;
    }
    next = (unsigned char *)(((uintptr_t)m->storage + 63u) & ~(uintptr_t)63u);
    for (k = 0; k < w->N; ++k) {
        blasfeo_create_smat(w->n, w->nx, m->DT + k, take(&next, dbytes));
        blasfeo_create_smat(w->n, w->n, m->C + k, take(&next, hbytes));
        blasfeo_create_smat(w->n, w->n, m->cost + k, take(&next, hbytes));
    }
    blasfeo_create_smat(w->nx, w->nx, m->C + w->N, take(&next, tbytes));
    blasfeo_create_smat(w->nx, w->nx, m->cost + w->N, take(&next, tbytes));
    blasfeo_create_smat(w->n, w->nx, &m->DTP, take(&next, dbytes));
    blasfeo_create_smat(w->n, w->n, &m->M, take(&next, hbytes));
    m->sweep_next = -2;
    /* Optional SP sweep vectors do not affect converted-factor availability. */
    {
        size_t vbytes = align64(blasfeo_memsize_svec(w->n + 8));
        if (vbytes <= (SIZE_MAX - 63u) / 4u) {
            m->sweep_storage = calloc(1, 4u * vbytes + 63u);
            if (m->sweep_storage) {
                next = (unsigned char *)(((uintptr_t)m->sweep_storage + 63u) & ~(uintptr_t)63u);
                blasfeo_create_svec(w->n + 8, &m->X, take(&next, vbytes));
                blasfeo_create_svec(w->n + 8, &m->Y, take(&next, vbytes));
                blasfeo_create_svec(w->n + 8, &m->Z, take(&next, vbytes));
                blasfeo_create_svec(w->n + 8, &m->V, take(&next, vbytes));
            }
        }
    }
}

static int mixed_cast(double value, float *out)
{
    double magnitude = fabs(value);
    if (!isfinite(value) || magnitude > FLT_MAX || (magnitude != 0.0 && magnitude < FLT_MIN))
        return 0;
    *out = (float)value;
    return isfinite(*out) && (*out != 0.0f || value == 0.0);
}

static int mixed_environment(void)
{
    unsigned int csr;
    unsigned short cw;
    if (fegetround() != FE_TONEAREST)
        return 0;
    __asm__ volatile("stmxcsr %0" : "=m"(csr));
    __asm__ volatile("fnstcw %0" : "=m"(cw));
    return !(csr & 0xe040u) && (csr & 0x1f80u) == 0x1f80u && (cw & 0x3fu) == 0x3fu;
}

/* Validate by IEEE magnitude bits before converting. This is the same
 * finite, normal-or-zero FP32 range as mixed_cast, without scalar calls. */
static ICORE_OCP_LA_AVX2_TARGET int mixed_cache_cast4(__m256d value, __m128 *out)
{
    const __m256i zero = _mm256_setzero_si256();
    const __m256i magnitude =
        _mm256_and_si256(_mm256_castpd_si256(value), _mm256_set1_epi64x(INT64_MAX));
    const __m256i maximum = _mm256_castpd_si256(_mm256_set1_pd((double)FLT_MAX));
    const __m256i minimum = _mm256_castpd_si256(_mm256_set1_pd((double)FLT_MIN));
    const __m256i below = _mm256_cmpgt_epi64(minimum, magnitude);
    const __m256i nonzero_below = _mm256_andnot_si256(_mm256_cmpeq_epi64(magnitude, zero), below);
    const __m256i invalid = _mm256_or_si256(_mm256_cmpgt_epi64(magnitude, maximum), nonzero_below);
    if (_mm256_movemask_epi8(invalid))
        return 0;
    *out = _mm256_cvtpd_ps(value);
    return 1;
}

static ICORE_OCP_LA_AVX2_TARGET void mixed_load_dynamics(icore_ocp_la_workspace *w, int stage)
{
    struct icore_ocp_la_mixed *m = w->mixed;
    int i, j;
    struct blasfeo_dmat *D;
    struct blasfeo_smat *S;
    if (!m)
        return;
    m->dvalid[stage] = 0;
    if (!mixed_environment())
        return;
    D = w->DT + stage;
    S = m->DT + stage;
    for (i = 0; i + 8 <= w->n; i += 8) {
        const double *lo = D->pA + (size_t)i * D->cn;
        const double *hi = lo + 4 * (size_t)D->cn;
        float *dst = S->pA + (size_t)i * S->cn;
        for (j = 0; j < w->nx; ++j) {
            __m128 a, b;
            if (!mixed_cache_cast4(_mm256_loadu_pd(lo + 4 * (size_t)j), &a) ||
                !mixed_cache_cast4(_mm256_loadu_pd(hi + 4 * (size_t)j), &b))
                return;
            _mm256_storeu_ps(dst + 8 * (size_t)j,
                             _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1));
        }
    }
    /* Eligibility makes the only possible remainder four valid rows.
     * Do not read DP padding or overwrite the SP half-panel padding. */
    if (i < w->n) {
        const double *src = D->pA + (size_t)i * D->cn;
        float *dst = S->pA + (size_t)i * S->cn;
        for (j = 0; j < w->nx; ++j) {
            __m128 a;
            if (!mixed_cache_cast4(_mm256_loadu_pd(src + 4 * (size_t)j), &a))
                return;
            _mm_storeu_ps(dst + 8 * (size_t)j, a);
        }
    }
    S->use_dA = 0;
    m->dvalid[stage] = 1;
}

static void mixed_load_cost(icore_ocp_la_workspace *w, int stage)
{
    struct icore_ocp_la_mixed *m = w->mixed;
    int i, j, n = stage == w->N ? w->nx : w->n;
    unsigned char previous;
    if (!m)
        return;
    previous = m->hvalid[stage];
    m->hvalid[stage] = 0;
    if (!mixed_environment())
        return;
    if (w->cost_loaded[stage] == 2) {
        /* Both caches certify zero off-diagonals. Factor calls only replace
         * the diagonal, so repeated diagonal loads need no zero casts. */
        if (previous != 2)
            blasfeo_sgese(n, n, 0.0f, m->cost + stage, 0, 0);
        for (i = 0; i < n; ++i) {
            float f;
            if (!mixed_cast(BLASFEO_DMATEL(w->cost + stage, i, i), &f))
                return;
            BLASFEO_SMATEL(m->cost + stage, i, i) = f;
        }
        m->cost[stage].use_dA = 0;
        m->hvalid[stage] = 2;
        return;
    }
    for (i = 0; i < n; ++i)
        for (j = 0; j <= i; ++j) {
            float f;
            if (!mixed_cast(BLASFEO_DMATEL(w->cost + stage, i, j), &f))
                return;
            BLASFEO_SMATEL(m->cost + stage, i, j) = f;
        }
    m->cost[stage].use_dA = 0;
    m->hvalid[stage] = 1;
}

int icore_ocp_la_mixed_ready(const icore_ocp_la_workspace *w)
{
    int k;
    if (!w || !w->mixed || !w->owned_factors || (w->nr & 3) || (w->nx & 7))
        return 0;
    if (!mixed_environment())
        return 0;
    for (k = 0; k <= w->N; ++k)
        if (!w->mixed->hvalid[k])
            return 0;
    for (k = 0; k < w->N; ++k)
        if (!w->mixed->dvalid[k])
            return 0;
    return 1;
}

/* Full validation while converting. The DP fresh-square certificate does not
 * apply to an SP factor. Rebuild reciprocal doubles from converted diagonals. */
static int mixed_convert_ordered(struct blasfeo_smat *S, struct blasfeo_dmat *D, int n, int nr,
                                 double control_floor, double state_floor, int terminal)
{
    int i, j;
    for (i = 0; i < n; ++i) {
        double diagonal = (double)BLASFEO_SMATEL(S, i, i), inverse, pivot;
        float sp_inverse = S->dA[i];
        if (!isfinite(diagonal) || !(diagonal > 0.0) || !isfinite(sp_inverse) ||
            !(sp_inverse > 0.0f))
            return ICORE_LA_NOT_SPD;
        inverse = 1.0 / diagonal;
        pivot = diagonal * diagonal;
        if (!isfinite(inverse) || !(inverse > 0.0) || !isfinite(pivot) || !(pivot > 0.0) ||
            pivot < (i < nr ? control_floor : state_floor))
            return ICORE_OCP_LA_PIVOT_REJECTED;
        for (j = 0; j <= i; ++j) {
            double value = (double)BLASFEO_SMATEL(S, i, j);
            if (!isfinite(value))
                return ICORE_LA_NONFINITE;
            BLASFEO_DMATEL(D, i, j) = value;
        }
        D->dA[i] = inverse;
    }
    /* Pinned DP POTRF records n; fused SYRK/POTRF records one. */
    D->use_dA = terminal ? n : 1;
    return ICORE_LA_OK;
}

/* Every consumed lower entry is checked, using only integer operations.
 * A failed fast preflight returns to the original ordered converter before
 * any destination write. This is not a fresh-factor diagonal certificate. */
static ICORE_OCP_LA_AVX2_TARGET int mixed_lower_finite4(const struct blasfeo_smat *S, int n)
{
    int row = 0, j, a;
    const __m128i exponent = _mm_set1_epi32(0x7f800000);
    for (; row + 3 < n; row += 4) {
        const float *panel = S->pA + (size_t)(row & ~7) * S->cn + (row & 7);
        __m128i bad = _mm_setzero_si128();
        for (j = 0; j < row; ++j) {
            __m128i bits = _mm_loadu_si128((const __m128i *)(panel + (size_t)8 * j));
            bad = _mm_or_si128(bad, _mm_cmpeq_epi32(_mm_and_si128(bits, exponent), exponent));
        }
        if (_mm_movemask_epi8(bad))
            return 0;
        for (a = 0; a < 4; ++a)
            for (j = row; j <= row + a; ++j) {
                uint32_t bits;
                memcpy(&bits, panel + (size_t)8 * j + a, sizeof(bits));
                if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000))
                    return 0;
            }
    }
    for (; row < n; ++row)
        for (j = 0; j <= row; ++j) {
            uint32_t bits;
            memcpy(&bits, &BLASFEO_SMATEL(S, row, j), sizeof(bits));
            if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000))
                return 0;
        }
    return 1;
}

static ICORE_OCP_LA_AVX2_TARGET int mixed_copy_lower4(struct blasfeo_smat *S, struct blasfeo_dmat *D, int n,
                                             int terminal)
{
    int i, j, row;
    for (row = 0; row + 3 < n; row += 4) {
        const float *input = S->pA + (size_t)(row & ~7) * S->cn + (row & 7);
        double *output = D->pA + (size_t)row * D->cn;
        for (j = 0; j < row; ++j)
            _mm256_storeu_pd(output + (size_t)4 * j,
                             _mm256_cvtps_pd(_mm_loadu_ps(input + (size_t)8 * j)));
        /* Preserve unconsumed upper entries and all padding. */
        for (i = row; i < row + 4; ++i) {
            for (j = row; j <= i; ++j)
                BLASFEO_DMATEL(D, i, j) = (double)BLASFEO_SMATEL(S, i, j);
            D->dA[i] = 1.0 / (double)BLASFEO_SMATEL(S, i, i);
        }
    }
    for (; row < n; ++row) {
        for (j = 0; j <= row; ++j)
            BLASFEO_DMATEL(D, row, j) = (double)BLASFEO_SMATEL(S, row, j);
        D->dA[row] = 1.0 / (double)BLASFEO_SMATEL(S, row, row);
    }
    D->use_dA = terminal ? n : 1;
    return ICORE_LA_OK;
}

static ICORE_OCP_LA_AVX2_TARGET int mixed_convert(struct blasfeo_smat *S, struct blasfeo_dmat *D, int n,
                                         int nr, double control_floor, double state_floor,
                                         int terminal)
{
    int i;
    if (!mixed_lower_finite4(S, n))
        return mixed_convert_ordered(S, D, n, nr, control_floor, state_floor, terminal);
    for (i = 0; i < n; ++i) {
        uint32_t dbits, ibits;
        memcpy(&dbits, &BLASFEO_SMATEL(S, i, i), sizeof(dbits));
        memcpy(&ibits, S->dA + i, sizeof(ibits));
        /* Positive finite SP values are exactly representable in double;
         * their squares and reciprocals are within double's normal range.
         * Keep the actual squared-pivot predicates and ordered rejection. */
        if ((dbits & UINT32_C(0x80000000)) || !(dbits & UINT32_C(0x7fffffff)) ||
            (ibits & UINT32_C(0x80000000)) || !(ibits & UINT32_C(0x7fffffff)) ||
            (ibits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000))
            return mixed_convert_ordered(S, D, n, nr, control_floor, state_floor, terminal);
        double d = (double)BLASFEO_SMATEL(S, i, i), pivot = d * d;
        if (!isfinite(pivot) || !(pivot > 0.0) || pivot < (i < nr ? control_floor : state_floor))
            return mixed_convert_ordered(S, D, n, nr, control_floor, state_floor, terminal);
    }
    return mixed_copy_lower4(S, D, n, terminal);
}

/* Private certificate for an immediately preceding complete square factor
 * from the repaired bundled Haswell SP kernels. Every stored lower Lij is
 * used in row i's later pivot, including x4/x8 inner blocks and all tails.
 * A nonfinite Lij prevents a positive finite diagonal and SP reciprocal;
 * unordered/nonpositive pivots explicitly store a zero reciprocal. This
 * does not apply to arbitrary, partial, or subsequently modified factors.
 * Preserve all squared-pivot floors and use the full converter on rejection. */
static ICORE_OCP_LA_AVX2_TARGET int mixed_convert_fresh_square(struct blasfeo_smat *S,
                                                      struct blasfeo_dmat *D, int n, int nr,
                                                      double control_floor, double state_floor,
                                                      int terminal)
{
    int i = 0;
    unsigned saved_mxcsr = _mm_getcsr();
    const __m128i zero = _mm_setzero_si128(), exponent = _mm_set1_epi32(0x7f800000);
    const __m128i offsets = _mm_set_epi32(27, 18, 9, 0);
    if (!mixed_environment())
        return mixed_convert(S, D, n, nr, control_floor, state_floor, terminal);
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
            goto full;
        d = _mm256_cvtps_pd(diagonal);
        pivot = _mm256_mul_pd(d, d);
        floors = _mm256_set_pd(
            i + 3 < nr ? control_floor : state_floor, i + 2 < nr ? control_floor : state_floor,
            i + 1 < nr ? control_floor : state_floor, i < nr ? control_floor : state_floor);
        /* Finite positive FP32 diagonals have positive finite, exactly
         * representable squares in FP64 and finite positive reciprocals. */
        if (_mm256_movemask_pd(_mm256_cmp_pd(pivot, floors, _CMP_LT_OQ)))
            goto full;
    }
    for (; i < n; ++i) {
        uint32_t dbits, ibits;
        double d, pivot;
        memcpy(&dbits, &BLASFEO_SMATEL(S, i, i), sizeof(dbits));
        memcpy(&ibits, S->dA + i, sizeof(ibits));
        if ((dbits & UINT32_C(0x80000000)) || !(dbits & UINT32_C(0x7fffffff)) ||
            (dbits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000) ||
            (ibits & UINT32_C(0x80000000)) || !(ibits & UINT32_C(0x7fffffff)) ||
            (ibits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000))
            goto full;
        d = (double)BLASFEO_SMATEL(S, i, i);
        pivot = d * d;
        if (pivot < (i < nr ? control_floor : state_floor))
            goto full;
    }
    return mixed_copy_lower4(S, D, n, terminal);
full:
    /* Preflight may inspect later masked-exception lanes. No destination
     * write occurred; replay the full converter with its original flags. */
    _mm_setcsr(saved_mxcsr);
    return mixed_convert(S, D, n, nr, control_floor, state_floor, terminal);
}

static int mixed_stage_floors_ordered(icore_ocp_la_workspace *w, int stage, const double *diagonal,
                                      double reg, int rows, double *cm, double *sm)
{
    int i, j;
    struct icore_ocp_la_mixed *m = w->mixed;
    *cm = *sm = 0.0;
    for (i = 0; i < rows; ++i) {
        double d = diagonal[i] + reg, physical = d;
        float f;
        if (!mixed_cast(d, &f))
            return ICORE_LA_NONFINITE;
        BLASFEO_SMATEL(m->cost + stage, i, i) = f;
        for (j = 0; j < w->nx; ++j) {
            double v = (double)BLASFEO_SMATEL(&m->DTP, i, j);
            if (!isfinite(v))
                return ICORE_LA_NONFINITE;
            physical += v * v;
        }
        if (!isfinite(physical))
            return ICORE_LA_NONFINITE;
        if (i < w->nr) {
            if (fabs(physical) > *cm)
                *cm = fabs(physical);
        } else if (fabs(physical) > *sm)
            *sm = fabs(physical);
    }
    return ICORE_LA_OK;
}

static ICORE_OCP_LA_FMA_TARGET int mixed_stage_floors4(icore_ocp_la_workspace *w, int stage, const double *diagonal,
                                              double reg, int rows, double *cm, double *sm)
{
    int i, j, a;
    struct icore_ocp_la_mixed *m = w->mixed;
    unsigned saved_mxcsr = _mm_getcsr();
    if (rows & 3)
        return mixed_stage_floors_ordered(w, stage, diagonal, reg, rows, cm, sm);
    *cm = *sm = 0.0;
    for (i = 0; i < rows; i += 4) {
        double physical[4];
        float cast_diagonal[4];
        const float *panel = m->DTP.pA + (size_t)(i & ~7) * m->DTP.cn + (i & 7);
        for (a = 0; a < 4; ++a) {
            physical[a] = diagonal[i + a] + reg;
            if (!mixed_cast(physical[a], cast_diagonal + a))
                goto ordered;
        }
        __m256d sum = _mm256_loadu_pd(physical);
        /* Finite FP32 squares have at most 48 significant bits and fit
         * exactly in FP64. FMA therefore gives the same rounded addition
         * as the ordered scalar product-plus-sum, column by column. A
         * nonfinite squared term cannot return to a finite running sum. */
        for (j = 0; j < w->nx; ++j) {
            __m256d value = _mm256_cvtps_pd(_mm_loadu_ps(panel + (size_t)8 * j));
            sum = _mm256_fmadd_pd(value, value, sum);
        }
        _mm256_storeu_pd(physical, sum);
        for (a = 0; a < 4; ++a)
            if (!isfinite(physical[a]))
                goto ordered;
        /* No current-block cost write precedes its finite check. Ordered
         * fallback can reproduce the scalar partial-write/status behavior. */
        for (a = 0; a < 4; ++a) {
            BLASFEO_SMATEL(m->cost + stage, i + a, i + a) = cast_diagonal[a];
            if (i + a < w->nr) {
                if (fabs(physical[a]) > *cm)
                    *cm = fabs(physical[a]);
            } else if (fabs(physical[a]) > *sm)
                *sm = fabs(physical[a]);
        }
    }
    return ICORE_LA_OK;
ordered:
    /* A rejected vector block may have evaluated later masked-exception
     * lanes. Restore its entry flags before replaying the ordered path. */
    _mm_setcsr(saved_mxcsr);
    return mixed_stage_floors_ordered(w, stage, diagonal, reg, rows, cm, sm);
}

typedef struct mixed_lazy_floor_bound {
    double cm, sm, cf, sf;
    int reason;
} mixed_lazy_floor_bound;

enum {
    ICORE_OCP_LA_LAZY_READY = 0,
    ICORE_OCP_LA_LAZY_CLEAN_PRECISION = 1,
    ICORE_OCP_LA_LAZY_ENVIRONMENT = 2,
    ICORE_OCP_LA_LAZY_PARAMETERS = 3,
    ICORE_OCP_LA_LAZY_DTP_RANGE = 4,
    ICORE_OCP_LA_LAZY_DIAGONAL = 5,
    ICORE_OCP_LA_LAZY_BOUND_RANGE = 6
};

static uint64_t mixed_lazy_dbits(double x)
{
    uint64_t b;
    memcpy(&b, &x, sizeof(b));
    return b & UINT64_C(0x7fffffffffffffff);
}

static double mixed_lazy_double(uint64_t b)
{
    double x;
    memcpy(&x, &b, sizeof(x));
    return x;
}

static int mixed_lazy_dceil(uint64_t b)
{
    return (int)(b >> 52) - 1023 + ((b & UINT64_C(0x000fffffffffffff)) != 0);
}

static int mixed_lazy_sceil(uint32_t b)
{
    return (int)(b >> 23) - 127 + ((b & UINT32_C(0x007fffff)) != 0);
}

static int mixed_lazy_log2ceil(unsigned n)
{
    int result = 0;
    --n;
    while (n) {
        ++result;
        n >>= 1;
    }
    return result;
}

static uint32_t mixed_lazy_reduce4(__m128i v)
{
    uint32_t x[4], m;
    int i;
    _mm_storeu_si128((__m128i *)x, v);
    m = x[0];
    for (i = 1; i < 4; ++i)
        if (x[i] > m)
            m = x[i];
    return m;
}

static int mixed_lazy_diagonal_overlaps(const double *d, int rows, const struct blasfeo_smat *S)
{
    uintptr_t a = (uintptr_t)d, b = (uintptr_t)S->pA;
    size_t na = (size_t)rows * sizeof(double), nb = (size_t)S->memsize;
    return (a <= b && b - a < na) || (b < a && a - b < nb);
}

/* Integer-only scan. Invalid SP values and SP subnormals are left to the
 * original ordered path, including its conversion exception semantics. */
static ICORE_OCP_LA_AVX2_TARGET int mixed_lazy_dtp_maxima(icore_ocp_la_workspace *w, int rows, uint32_t *control,
                                                 uint32_t *state)
{
    struct blasfeo_smat *S = &w->mixed->DTP;
    const __m256i sign = _mm256_set1_epi32(INT32_MAX);
    const __m256i exponent = _mm256_set1_epi32(0x7f800000);
    const __m256i zero = _mm256_setzero_si256();
    __m128i cm = _mm_setzero_si128(), sm = _mm_setzero_si128();
    __m256i subnormal = _mm256_setzero_si256();
    int row = 0, j;
    if (rows & 3)
        return 0;
    for (; row + 8 <= rows; row += 8) {
        const float *panel = S->pA + (size_t)row * S->cn;
        __m256i maximum = zero;
        for (j = 0; j < w->nx; ++j) {
            __m256i mag = _mm256_and_si256(
                _mm256_loadu_si256((const __m256i *)(panel + 8 * (size_t)j)), sign);
            __m256i small = _mm256_cmpeq_epi32(_mm256_and_si256(mag, exponent), zero);
            maximum = _mm256_max_epu32(maximum, mag);
            subnormal = _mm256_or_si256(subnormal, _mm256_and_si256(mag, small));
        }
        __m128i lo = _mm256_castsi256_si128(maximum), hi = _mm256_extracti128_si256(maximum, 1);
        if (row + 8 <= w->nr)
            cm = _mm_max_epu32(cm, _mm_max_epu32(lo, hi));
        else if (row >= w->nr)
            sm = _mm_max_epu32(sm, _mm_max_epu32(lo, hi));
        else { /* nr%4==0 makes this exactly the half-panel split. */
            cm = _mm_max_epu32(cm, lo);
            sm = _mm_max_epu32(sm, hi);
        }
    }
    if (row < rows) {
        const float *panel = S->pA + (size_t)row * S->cn;
        __m128i maximum = _mm_setzero_si128(), sub = _mm_setzero_si128();
        for (j = 0; j < w->nx; ++j) {
            __m128i mag = _mm_and_si128(_mm_loadu_si128((const __m128i *)(panel + 8 * (size_t)j)),
                                        _mm_set1_epi32(INT32_MAX));
            __m128i small = _mm_cmpeq_epi32(_mm_and_si128(mag, _mm_set1_epi32(0x7f800000)),
                                            _mm_setzero_si128());
            maximum = _mm_max_epu32(maximum, mag);
            sub = _mm_or_si128(sub, _mm_and_si128(mag, small));
        }
        if (!_mm_testz_si128(sub, sub))
            return 0;
        if (row < w->nr)
            cm = _mm_max_epu32(cm, maximum);
        else
            sm = _mm_max_epu32(sm, maximum);
    }
    if (!_mm256_testz_si256(subnormal, subnormal))
        return 0;
    *control = mixed_lazy_reduce4(cm);
    *state = mixed_lazy_reduce4(sm);
    return *control < UINT32_C(0x7f800000) && *state < UINT32_C(0x7f800000);
}

/* For nx<2^31 and u=2^-53, (1+u)^nx<2. Exact finite SP squares
 * fit in normal FP64. Bound the sum of |d| and nx squares by 2^(M+1),
 * then allow one further exponent for every ordered RN accumulation.
 * The zero-DTP case has no rounded accumulation and needs no margin. */
static int mixed_lazy_energy_exponent(uint64_t d, uint32_t v, int nx, int *nonzero)
{
    int ed = d ? mixed_lazy_dceil(d) : -4096;
    *nonzero = d != 0 || v != 0;
    if (!v)
        return ed;
    int ev = 2 * mixed_lazy_sceil(v) + mixed_lazy_log2ceil((unsigned)nx);
    return (ed > ev ? ed : ev) + 2;
}

static int mixed_lazy_floor_bits(uint64_t absolute, uint64_t relative, int energy_exponent,
                                 int energy_nonzero, double *energy, double *floor)
{
    uint64_t eb = 0, fb = 0;
    if (energy_nonzero) {
        if (energy_exponent < -1022 || energy_exponent > 1023)
            return 0;
        eb = (uint64_t)(energy_exponent + 1023) << 52;
        if (relative) {
            int total = energy_exponent + mixed_lazy_dceil(relative);
            if (total < -1022 || total > 1023)
                return 0;
            fb = (uint64_t)(total + 1023) << 52;
        }
    }
    *energy = mixed_lazy_double(eb);
    *floor = mixed_lazy_double(absolute > fb ? absolute : fb);
    return 1;
}

/* No cost write precedes complete validation. On refusal restore entry
 * MXCSR and let the unchanged helper reproduce its original partial writes.
 * The successful preflight's diagonal operations already occur in the old
 * helper. Omitting finite DP square-sums may only omit precision exceptions;
 * require that sticky flag on entry. Relative>=2^-724 prevents the original
 * relative*max product from underflow: a nonzero ordered energy is >=2^-298.
 * Its overflow is excluded by the finite conservative product bound. */
static int mixed_lazy_prepare(icore_ocp_la_workspace *w, int stage, const double *diagonal, double reg,
                              int rows, double absolute, double relative,
                              mixed_lazy_floor_bound *out)
{
#if FLT_EVAL_METHOD != 0
    /* The skipped-operation flag proof requires ordinary FP64 evaluation. */
    (void)w;
    (void)stage;
    (void)diagonal;
    (void)reg;
    (void)rows;
    (void)absolute;
    (void)relative;
    out->reason = ICORE_OCP_LA_LAZY_ENVIRONMENT;
    return 0;
#endif
    unsigned csr = _mm_getcsr();
    int i, ec, es, nc, ns;
    uint64_t dc = 0, ds = 0, ab = mixed_lazy_dbits(absolute), rb = mixed_lazy_dbits(relative);
    uint32_t vc = 0, vs = 0;
    out->reason = ICORE_OCP_LA_LAZY_CLEAN_PRECISION;
    if (!(csr & 0x20u))
        return 0;
    out->reason = ICORE_OCP_LA_LAZY_ENVIRONMENT;
    if (!mixed_environment())
        return 0;
    out->reason = ICORE_OCP_LA_LAZY_PARAMETERS;
    if ((ab && ab < UINT64_C(0x0010000000000000)) || (rb && rb < ((uint64_t)(1023 - 724) << 52)) ||
        ab >= UINT64_C(0x7ff0000000000000) || rb >= UINT64_C(0x7ff0000000000000))
        return 0;
    /* The engine owns a disjoint diagonal vector. Keep old ordering if an
     * internal caller aliases storage that preparation or factorization writes. */
    if (mixed_lazy_diagonal_overlaps(diagonal, rows, w->mixed->cost + stage) ||
        mixed_lazy_diagonal_overlaps(diagonal, rows, w->mixed->C + stage))
        return 0;
    out->reason = ICORE_OCP_LA_LAZY_DTP_RANGE;
    if (!mixed_lazy_dtp_maxima(w, rows, &vc, &vs))
        return 0;
    out->reason = ICORE_OCP_LA_LAZY_DIAGONAL;
    for (i = 0; i < rows; ++i) {
        double d = diagonal[i] + reg;
        float f;
        uint64_t b;
        if (!mixed_cast(d, &f))
            goto restore;
        b = mixed_lazy_dbits(d);
        if (i < w->nr) {
            if (b > dc)
                dc = b;
        } else if (b > ds)
            ds = b;
    }
    ec = mixed_lazy_energy_exponent(dc, vc, w->nx, &nc);
    es = mixed_lazy_energy_exponent(ds, vs, w->nx, &ns);
    out->reason = ICORE_OCP_LA_LAZY_BOUND_RANGE;
    if (!mixed_lazy_floor_bits(ab, rb, ec, nc, &out->cm, &out->cf) ||
        !mixed_lazy_floor_bits(ab, rb, es, ns, &out->sm, &out->sf))
        goto restore;
    if (stage == 0)
        out->sf = out->cf;
    for (i = 0; i < rows; ++i) {
        /* Exactly the same input addition and cast as the validated pass. */
        float f = (float)(diagonal[i] + reg);
        BLASFEO_SMATEL(w->mixed->cost + stage, i, i) = f;
    }
    out->reason = ICORE_OCP_LA_LAZY_READY;
    return 1;
restore:
    _mm_setcsr(csr);
    return 0;
}

/* Defined by the SP-sweep bridge below; this read-only fresh certificate is
 * also valid before conversion of this same immediately produced factor. */
static ICORE_OCP_LA_AVX2_TARGET int mixed_sweep_check_fresh(struct blasfeo_smat *S, int n, int nr, double cf,
                                                   double sf);

/* A sufficient upper floor only skips work when it proves the old floor.
 * Otherwise replay the original exact floor and certificate on the unchanged
 * fresh factor. The successful preflight proves that replay cannot fail in
 * energy formation and that moving it here changes no exception flags. */
static int mixed_stage_factor_checked(icore_ocp_la_workspace *w, int stage, const double *diagonal,
                                      double reg, double absolute, double relative, int rows,
                                      int sweeps)
{
    struct icore_ocp_la_mixed *m = w->mixed;
    mixed_lazy_floor_bound bound;
    int status, ready;
    double cm = 0.0, sm = 0.0, cf, sf;
    ready = mixed_lazy_prepare(w, stage, diagonal, reg, rows, absolute, relative, &bound);
    if (!ready) {
        status = mixed_stage_floors4(w, stage, diagonal, reg, rows, &cm, &sm);
        if (status != ICORE_LA_OK)
            return status;
        cf = fmax(absolute, relative * cm);
        sf = stage == 0 ? cf : fmax(absolute, relative * sm);
        if (!isfinite(cf) || !isfinite(sf))
            return ICORE_LA_INVALID;
    } else {
        cf = bound.cf;
        sf = bound.sf;
    }
    m->cost[stage].use_dA = 0;
    blasfeo_ssyrk_spotrf_ln(rows, w->nx, &m->DTP, 0, 0, &m->DTP, 0, 0, m->cost + stage, 0, 0,
                            m->C + stage, 0, 0);
    if (ready) {
        unsigned csr = _mm_getcsr();
        status = mixed_sweep_check_fresh(m->C + stage, rows, w->nr, cf, sf);
        if (status == ICORE_LA_OK) {
            return sweeps ? ICORE_LA_OK : mixed_copy_lower4(m->C + stage, w->C + stage, rows, 0);
        }
        _mm_setcsr(csr);
        status = mixed_stage_floors4(w, stage, diagonal, reg, rows, &cm, &sm);
        if (status != ICORE_LA_OK)
            return status;
        cf = fmax(absolute, relative * cm);
        sf = stage == 0 ? cf : fmax(absolute, relative * sm);
        if (!isfinite(cf) || !isfinite(sf))
            return ICORE_LA_INVALID;
    }
    return sweeps ? mixed_sweep_check_fresh(m->C + stage, rows, w->nr, cf, sf)
                  : mixed_convert_fresh_square(m->C + stage, w->C + stage, rows, w->nr, cf, sf, 0);
}

int icore_ocp_la_mixed_begin(icore_ocp_la_workspace *w, const double *diagonal, double reg, double absolute,
                    double relative)
{
    int i, status;
    double maximum = 0.0, floor;
    struct icore_ocp_la_mixed *m;
    if (!icore_ocp_la_mixed_ready(w) || !diagonal || !sqrt_arguments(reg, absolute, relative))
        return ICORE_LA_UNAVAILABLE;
    m = w->mixed;
    w->sqrt_next = -2;
    w->sqrt_equilibrated = 0;
    w->sqrt_root_omitted = 0;
    w->sqrt_initial_zero = 0;
    sqrt_recovery_clear(w);
    for (i = 0; i < w->nx; ++i) {
        double d = diagonal[i] + reg;
        float f;
        if (!mixed_cast(d, &f))
            return sqrt_reject(w, ICORE_LA_NONFINITE);
        BLASFEO_SMATEL(m->cost + w->N, i, i) = f;
        if (fabs(d) > maximum)
            maximum = fabs(d);
    }
    floor = fmax(absolute, relative * maximum);
    if (!isfinite(floor))
        return sqrt_reject(w, ICORE_LA_INVALID);
    m->cost[w->N].use_dA = 0;
    blasfeo_spotrf_l(w->nx, m->cost + w->N, 0, 0, m->C + w->N, 0, 0);
    status = mixed_convert_fresh_square(m->C + w->N, w->C + w->N, w->nx, 0, floor, floor, 1);
    if (status != ICORE_LA_OK)
        return sqrt_reject(w, status);
    w->sqrt_next = w->N - 1;
    return ICORE_LA_OK;
}

int icore_ocp_la_mixed_stage(icore_ocp_la_workspace *w, int stage, const double *diagonal, double reg,
                    double absolute, double relative)
{
    int status, rows, offset;
    struct icore_ocp_la_mixed *m;
    if (!w || !w->mixed || stage < 0 || stage >= w->N || !diagonal || w->sqrt_next != stage ||
        !sqrt_arguments(reg, absolute, relative))
        return sqrt_reject(w, ICORE_LA_INVALID);
    m = w->mixed;
    rows = stage == 0 ? w->nr : w->n;
    offset = stage + 1 < w->N ? w->nr : 0;
    /* Pinned S_PS=8 implementation has the aligned and half-panel roots.
     * Both nontriangular inputs and the destination start at row zero. */
    if ((offset & 3) || !m->dvalid[stage] || !m->hvalid[stage])
        return sqrt_reject(w, ICORE_LA_UNAVAILABLE);
    blasfeo_strmm_rlnn(rows, w->nx, 1.0f, m->C + stage + 1, offset, offset, m->DT + stage, 0, 0,
                       &m->DTP, 0, 0);
    status = mixed_stage_factor_checked(w, stage, diagonal, reg, absolute, relative, rows, 0);
    if (status != ICORE_LA_OK)
        return sqrt_reject(w, status);
    w->sqrt_next = stage - 1;
    if (stage == 0) {
        w->sqrt_root_omitted = 1;
        w->sqrt_initial_zero = 1;
    }
    return ICORE_LA_OK;
}
#else
static void mixed_create(icore_ocp_la_workspace *w)
{
    (void)w;
}

static void mixed_destroy(icore_ocp_la_workspace *w)
{
    (void)w;
}

static void mixed_load_dynamics(icore_ocp_la_workspace *w, int stage)
{
    (void)w;
    (void)stage;
}

static void mixed_load_cost(icore_ocp_la_workspace *w, int stage)
{
    (void)w;
    (void)stage;
}

int icore_ocp_la_mixed_ready(const icore_ocp_la_workspace *w)
{
    (void)w;
    return 0;
}

int icore_ocp_la_mixed_begin(icore_ocp_la_workspace *w, const double *d, double r, double a, double t)
{
    (void)w;
    (void)d;
    (void)r;
    (void)a;
    (void)t;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_mixed_stage(icore_ocp_la_workspace *w, int k, const double *d, double r, double a, double t)
{
    (void)w;
    (void)k;
    (void)d;
    (void)r;
    (void)a;
    (void)t;
    return ICORE_LA_UNAVAILABLE;
}
#endif
