#include "dense_la.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__FAST_MATH__)
#error "dense_la requires finite-value checks: compile without fast-math"
#endif

#if defined(ICORE_LA_WITH_BLASFEO) && ICORE_LA_WITH_BLASFEO
#define ICORE_LA_BF 1
#include <blasfeo_common.h>
#include <blasfeo_d_aux.h>
#include <blasfeo_d_blasfeo_api.h>
#ifdef __cplusplus
extern "C" {
#endif
#include <blasfeo_processor_features.h>
#ifdef __cplusplus
}
#endif
#if !defined(MF_PANELMAJ)
#error "This optional backend requires BLASFEO panel-major headers/library"
#endif
#else
#define ICORE_LA_BF 0
#endif

/* A packed workspace has already passed the backend's CPU/OS probe.  The
 * local target also keeps standalone portable builds free of AVX2 code. */
#if ICORE_LA_BF && D_PS == 4 && defined(TARGET_NEED_FEATURE_AVX2) &&                                   \
    (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define ICORE_LA_AVX2 1
#define ICORE_LA_AVX2_TARGET __attribute__((target("avx2")))
#else
#define ICORE_LA_AVX2 0
#define ICORE_LA_AVX2_TARGET
#endif

#ifndef ICORE_LA_FUSED_WEIGHTED_GRAM
#define ICORE_LA_FUSED_WEIGHTED_GRAM ICORE_LA_AVX2
#endif

struct icore_la_workspace {
    int n, capacity, limit, ng, h_loaded, factor_ready, factor_equilibrated;
    icore_la_backend backend;
    double min_physical_pivot, min_scaled_pivot, max_factored_diagonal;
    void *storage;
    double *h, *l, *g;            /* Portable matrices only. */
    double *scale, *vx, *vy, *vz; /* n-vectors. */
    double *sqrtw, *gx, *gz;      /* capacity-vectors. */
#if ICORE_LA_BF
    struct blasfeo_dmat H, L, GT, WG;
    struct blasfeo_dvec D, X, Y, Z, W, GX, GZ;
#endif
};

static int finite_vector(const double *x, int n)
{
    int i;
    if (n && !x)
        return 0;
    for (i = 0; i < n; ++i)
        if (!isfinite(x[i]))
            return 0;
    return 1;
}

#if ICORE_LA_AVX2 && (!defined(ICORE_LA_SCALAR_VALIDATION) || !ICORE_LA_SCALAR_VALIDATION)
static ICORE_LA_AVX2_TARGET int finite_vector_avx2(const double *x, int n)
{
    int i = 0;
    const __m256i exponent = _mm256_set1_epi64x((long long)UINT64_C(0x7ff0000000000000));
    for (; i + 4 <= n; i += 4) {
        __m256i bits = _mm256_loadu_si256((const __m256i *)(x + i));
        __m256i bad = _mm256_cmpeq_epi64(_mm256_and_si256(bits, exponent), exponent);
        if (_mm256_movemask_pd(_mm256_castsi256_pd(bad)))
            return 0;
    }
    for (; i < n; ++i)
        if (!isfinite(x[i]))
            return 0;
    return 1;
}
#endif

static int finite_workspace_vector(const icore_la_workspace *w, const double *x, int n)
{
    if (n && !x)
        return 0;
#if ICORE_LA_AVX2 && (!defined(ICORE_LA_SCALAR_VALIDATION) || !ICORE_LA_SCALAR_VALIDATION)
    if (w->backend == ICORE_LA_BLASFEO)
        return finite_vector_avx2(x, n);
#else
    (void)w;
#endif
    return finite_vector(x, n);
}

/* Validate and populate owned input scratch in one pass. Partial scratch
 * writes on a rejected input are internal; caller outputs stay transactional. */
#if ICORE_LA_AVX2 && (!defined(ICORE_LA_SCALAR_VALIDATION) || !ICORE_LA_SCALAR_VALIDATION)
static ICORE_LA_AVX2_TARGET int finite_copy_avx2(double *dst, const double *src, int n)
{
    int i = 0;
    const __m256i exponent = _mm256_set1_epi64x((long long)UINT64_C(0x7ff0000000000000));
    for (; i + 4 <= n; i += 4) {
        __m256i bits = _mm256_loadu_si256((const __m256i *)(src + i));
        if (_mm256_movemask_pd(_mm256_castsi256_pd(
                _mm256_cmpeq_epi64(_mm256_and_si256(bits, exponent), exponent))))
            return 0;
        _mm256_storeu_si256((__m256i *)(dst + i), bits);
    }
    for (; i < n; ++i) {
        if (!isfinite(src[i]))
            return 0;
        dst[i] = src[i];
    }
    return 1;
}
#endif
static int finite_workspace_copy(const icore_la_workspace *w, double *dst, const double *src, int n)
{
    if (n && !src)
        return 0;
#if ICORE_LA_AVX2 && (!defined(ICORE_LA_SCALAR_VALIDATION) || !ICORE_LA_SCALAR_VALIDATION)
    if (w->backend == ICORE_LA_BLASFEO)
        return finite_copy_avx2(dst, src, n);
#else
    (void)w;
#endif
    for (int i = 0; i < n; ++i) {
        if (!isfinite(src[i]))
            return 0;
        dst[i] = src[i];
    }
    return 1;
}

static int valid_dimensions(int n, int capacity)
{
    size_t rows, cols;
    if (n <= 0 || capacity < 0 || n > INT_MAX - 64 || capacity > INT_MAX - 64)
        return 0;
    rows = (size_t)n + 64;
    cols = (size_t)(capacity > n ? capacity : n) + 64;
    /* BLASFEO stores each matrix's memory size in an int. This conservative
     * padded bound also prevents overflow in the portable allocation math. */
    return rows <= ((size_t)INT_MAX - 4096) / sizeof(double) / cols;
}

#if ICORE_LA_BF
static int probe_blasfeo_cpu(void)
{
    int features = 0;
    /* GCC/Clang's probe includes OS support for extended vector state,
     * whereas upstream's CPUID probe only checks the hardware bits. */
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
#if defined(TARGET_NEED_FEATURE_AVX2)
    if (!__builtin_cpu_supports("avx2"))
        return 0;
#endif
#if defined(TARGET_NEED_FEATURE_FMA)
    if (!__builtin_cpu_supports("fma"))
        return 0;
#endif
#if defined(TARGET_NEED_FEATURE_AVX)
    if (!__builtin_cpu_supports("avx"))
        return 0;
#endif
#endif
    return blasfeo_processor_cpu_features(&features) != 0;
}
#endif

int icore_la_blasfeo_available(void)
{
#if ICORE_LA_BF
#if defined(__GNUC__) || defined(__clang__)
    /* A loaded library has one CPU target. Retain the complete CPU/OS probe
     * result rather than repeating CPUID for every workspace. Concurrent
     * first calls may repeat the probe; only this integer is shared. */
    static int cached = 0; /* 0: unchecked, 1: unavailable, 2: available. */
    int result = __atomic_load_n(&cached, __ATOMIC_RELAXED);
    if (!result) {
        result = 1 + probe_blasfeo_cpu();
        __atomic_store_n(&cached, result, __ATOMIC_RELAXED);
    }
    return result - 1;
#else
    return probe_blasfeo_cpu();
#endif
#else
    return 0;
#endif
}

#if ICORE_LA_BF
static size_t aligned_size(size_t bytes)
{
    return (bytes + 63u) & ~(size_t)63u;
}

static unsigned char *take_memory(unsigned char **next, size_t size)
{
    unsigned char *result = *next;
    *next += aligned_size(size);
    return result;
}

static int create_packed(icore_la_workspace *w)
{
    size_t hbytes = blasfeo_memsize_dmat(w->n, w->n);
    size_t gbytes = blasfeo_memsize_dmat(w->n, w->capacity);
    size_t nbytes = blasfeo_memsize_dvec(w->n);
    size_t mbytes = blasfeo_memsize_dvec(w->capacity);
    size_t bytes = 2 * aligned_size(hbytes) + 2 * aligned_size(gbytes) + 4 * aligned_size(nbytes) +
                   3 * aligned_size(mbytes);
    unsigned char *next;
    w->storage = calloc(1, bytes + 63u);
    if (!w->storage)
        return ICORE_LA_NOMEM;
    next = (unsigned char *)(((uintptr_t)w->storage + 63u) & ~(uintptr_t)63u);
    blasfeo_create_dmat(w->n, w->n, &w->H, take_memory(&next, hbytes));
    blasfeo_create_dmat(w->n, w->n, &w->L, take_memory(&next, hbytes));
    blasfeo_create_dmat(w->n, w->capacity, &w->GT, take_memory(&next, gbytes));
    blasfeo_create_dmat(w->n, w->capacity, &w->WG, take_memory(&next, gbytes));
    blasfeo_create_dvec(w->n, &w->D, take_memory(&next, nbytes));
    blasfeo_create_dvec(w->n, &w->X, take_memory(&next, nbytes));
    blasfeo_create_dvec(w->n, &w->Y, take_memory(&next, nbytes));
    blasfeo_create_dvec(w->n, &w->Z, take_memory(&next, nbytes));
    blasfeo_create_dvec(w->capacity, &w->W, take_memory(&next, mbytes));
    blasfeo_create_dvec(w->capacity, &w->GX, take_memory(&next, mbytes));
    blasfeo_create_dvec(w->capacity, &w->GZ, take_memory(&next, mbytes));
    w->scale = w->D.pa;
    w->vx = w->X.pa;
    w->vy = w->Y.pa;
    w->vz = w->Z.pa;
    w->sqrtw = w->W.pa;
    w->gx = w->GX.pa;
    w->gz = w->GZ.pa;
    return ICORE_LA_OK;
}
#endif

static int create_portable(icore_la_workspace *w)
{
    size_t n = (size_t)w->n, m = (size_t)w->capacity;
    size_t count = 2 * n * n + m * n + 4 * n + 3 * m;
    double *next;
    if (count > SIZE_MAX / sizeof(double))
        return ICORE_LA_NOMEM;
    next = (double *)calloc(count, sizeof(double));
    if (!next)
        return ICORE_LA_NOMEM;
    w->storage = next;
    w->h = next;
    next += n * n;
    w->l = next;
    next += n * n;
    w->g = next;
    next += m * n;
    w->scale = next;
    next += n;
    w->vx = next;
    next += n;
    w->vy = next;
    next += n;
    w->vz = next;
    next += n;
    w->sqrtw = next;
    next += m;
    w->gx = next;
    next += m;
    w->gz = next;
    return ICORE_LA_OK;
}

int icore_la_create(int n, int max_general, icore_la_backend requested, icore_la_workspace **out)
{
    icore_la_workspace *w;
    int status;
    if (!out)
        return ICORE_LA_INVALID;
    *out = NULL;
    if (!valid_dimensions(n, max_general) || requested < ICORE_LA_AUTO || requested > ICORE_LA_BLASFEO)
        return ICORE_LA_INVALID;
    if (requested == ICORE_LA_AUTO)
        requested = icore_la_blasfeo_available() ? ICORE_LA_BLASFEO : ICORE_LA_PORTABLE;
    if (requested == ICORE_LA_BLASFEO && !icore_la_blasfeo_available())
        return ICORE_LA_UNAVAILABLE;
    w = (icore_la_workspace *)calloc(1, sizeof(*w));
    if (!w)
        return ICORE_LA_NOMEM;
    w->n = n;
    w->capacity = max_general ? max_general : 1;
    w->limit = max_general;
    w->backend = requested;
#if ICORE_LA_BF
    if (requested == ICORE_LA_BLASFEO)
        status = create_packed(w);
    else
#endif
        status = create_portable(w);
    if (status != ICORE_LA_OK) {
        icore_la_destroy(w);
        return status;
    }
    *out = w;
    return ICORE_LA_OK;
}

void icore_la_destroy(icore_la_workspace *w)
{
    if (w) {
        free(w->storage);
        free(w);
    }
}

icore_la_backend icore_la_backend_used(const icore_la_workspace *w)
{
    return w ? w->backend : ICORE_LA_AUTO;
}

const char *icore_la_backend_name(const icore_la_workspace *w)
{
    if (!w)
        return "none";
    return w->backend == ICORE_LA_BLASFEO ? "blasfeo-panel-major" : "portable-c99";
}

const char *icore_la_status_string(int status)
{
    switch (status) {
    case ICORE_LA_OK:
        return "ok";
    case ICORE_LA_INVALID:
        return "invalid argument";
    case ICORE_LA_NOMEM:
        return "allocation failed";
    case ICORE_LA_NOT_READY:
        return "model or factor not ready";
    case ICORE_LA_NOT_SPD:
        return "matrix is not positive definite at working precision";
    case ICORE_LA_NONFINITE:
        return "nonfinite input or numerical result";
    case ICORE_LA_UNAVAILABLE:
        return "requested backend unavailable";
    default:
        return "unknown status";
    }
}

int icore_la_load_h(icore_la_workspace *w, const double *h, size_t row_stride)
{
    int i, j;
    if (!w || !h || row_stride < (size_t)w->n || row_stride > (size_t)INT_MAX)
        return ICORE_LA_INVALID;
    for (i = 0; i < w->n; ++i)
        for (j = 0; j <= i; ++j)
            if (!isfinite(h[(size_t)i * row_stride + j]))
                return ICORE_LA_NONFINITE;
#if ICORE_LA_BF
    if (w->backend == ICORE_LA_BLASFEO) {
        for (i = 0; i < w->n; ++i)
            for (j = 0; j <= i; ++j)
                BLASFEO_DMATEL(&w->H, i, j) = h[(size_t)i * row_stride + j];
        w->H.use_dA = 0;
    } else
#endif
    {
        for (i = 0; i < w->n; ++i)
            memcpy(w->h + (size_t)i * w->n, h + (size_t)i * row_stride,
                   ((size_t)i + 1) * sizeof(double));
    }
    w->h_loaded = 1;
    w->factor_ready = 0;
    return ICORE_LA_OK;
}

int icore_la_load_g(icore_la_workspace *w, int general_rows, const double *g, size_t row_stride)
{
    int i;
    if (!w || general_rows < 0 || general_rows > w->limit)
        return ICORE_LA_INVALID;
    if (general_rows && (!g || row_stride < (size_t)w->n || row_stride > (size_t)INT_MAX))
        return ICORE_LA_INVALID;
    for (i = 0; i < general_rows; ++i)
        if (!finite_vector(g + (size_t)i * row_stride, w->n))
            return ICORE_LA_NONFINITE;
#if ICORE_LA_BF
    if (w->backend == ICORE_LA_BLASFEO) {
        /* A row-major m by n G is exactly a column-major n by m G^T.
         * BLASFEO's pack API lacks const; it does not modify its input. */
        if (general_rows)
            blasfeo_pack_dmat(w->n, general_rows, (double *)g, (int)row_stride, &w->GT, 0, 0);
        w->GT.use_dA = 0;
    } else
#endif
    {
        for (i = 0; i < general_rows; ++i)
            memcpy(w->g + (size_t)i * w->n, g + (size_t)i * row_stride,
                   (size_t)w->n * sizeof(double));
    }
    w->ng = general_rows;
    w->factor_ready = 0;
    return ICORE_LA_OK;
}

static double original_hdiag(const icore_la_workspace *w, int i)
{
#if ICORE_LA_BF
    if (w->backend == ICORE_LA_BLASFEO)
        return BLASFEO_DMATEL(&w->H, i, i);
#endif
    return w->h[(size_t)i * w->n + i];
}

static double original_g(const icore_la_workspace *w, int row, int col)
{
#if ICORE_LA_BF
    if (w->backend == ICORE_LA_BLASFEO)
        return BLASFEO_DMATEL(&w->GT, col, row);
#endif
    return w->g[(size_t)row * w->n + col];
}

#if ICORE_LA_AVX2 && ICORE_LA_FUSED_WEIGHTED_GRAM
/* Form G^T*sqrt(W) and its squared row norms in one panel traversal.  The
 * old path read the same packed coefficients once with a strided diagonal
 * scan and once in dgemm_nd.  Four independent accumulators preserve each
 * row's original constraint order.  No padded coefficient is consumed. */
static ICORE_LA_AVX2_TARGET int prepare_packed_gram(icore_la_workspace *w, const double *diagonal,
                                                double reg, int equilibrate)
{
    int i, j, lane;
    for (i = 0; i + 4 <= w->n; i += 4) {
        double base[4], total[4];
        int status[4] = {ICORE_LA_OK, ICORE_LA_OK, ICORE_LA_OK, ICORE_LA_OK};
        __m256d sum;
        const double *g = w->GT.pA + (size_t)i * w->GT.cn;
        double *wg = w->WG.pA + (size_t)i * w->WG.cn;
        for (lane = 0; lane < 4; ++lane) {
            double b = original_hdiag(w, i + lane) + reg;
            if (diagonal) {
                if (!isfinite(diagonal[i + lane]))
                    status[lane] = ICORE_LA_NONFINITE;
                else if (diagonal[i + lane] < 0.0)
                    status[lane] = ICORE_LA_INVALID;
                b += diagonal[i + lane];
            }
            if (status[lane] == ICORE_LA_OK && !isfinite(b))
                status[lane] = ICORE_LA_NONFINITE;
            base[lane] = b;
            w->vy[i + lane] = b;
        }
        sum = _mm256_loadu_pd(base);
        for (j = 0; j < w->ng; ++j) {
            __m256d a =
                _mm256_mul_pd(_mm256_loadu_pd(g + 4 * (size_t)j), _mm256_set1_pd(w->sqrtw[j]));
            _mm256_storeu_pd(wg + 4 * (size_t)j, a);
            sum = _mm256_add_pd(sum, _mm256_mul_pd(a, a));
        }
        _mm256_storeu_pd(total, sum);
        /* Retain the old per-row failure precedence, including a malformed
         * later diagonal after an overflowing earlier row. */
        for (lane = 0; lane < 4; ++lane) {
            double t = total[lane];
            if (status[lane] != ICORE_LA_OK)
                return status[lane];
            if (!isfinite(t))
                return ICORE_LA_NONFINITE;
            if (equilibrate) {
                if (!(t > 0.0))
                    return ICORE_LA_NOT_SPD;
                w->scale[i + lane] = 1.0 / sqrt(t);
            } else
                w->scale[i + lane] = 1.0;
            t = fabs((t * w->scale[i + lane]) * w->scale[i + lane]);
            if (t > w->max_factored_diagonal)
                w->max_factored_diagonal = t;
        }
    }
    for (; i < w->n; ++i) {
        double base = original_hdiag(w, i) + reg, total;
        if (diagonal) {
            if (!isfinite(diagonal[i]))
                return ICORE_LA_NONFINITE;
            if (diagonal[i] < 0.0)
                return ICORE_LA_INVALID;
            base += diagonal[i];
        }
        if (!isfinite(base))
            return ICORE_LA_NONFINITE;
        w->vy[i] = base;
        total = base;
        for (j = 0; j < w->ng; ++j) {
            double a = original_g(w, j, i) * w->sqrtw[j];
            BLASFEO_DMATEL(&w->WG, i, j) = a;
            total += a * a;
        }
        if (!isfinite(total))
            return ICORE_LA_NONFINITE;
        if (equilibrate) {
            if (!(total > 0.0))
                return ICORE_LA_NOT_SPD;
            w->scale[i] = 1.0 / sqrt(total);
        } else
            w->scale[i] = 1.0;
        total = fabs((total * w->scale[i]) * w->scale[i]);
        if (total > w->max_factored_diagonal)
            w->max_factored_diagonal = total;
    }
    w->WG.use_dA = 0;
    return ICORE_LA_OK;
}
#endif

#if ICORE_LA_AVX2 && (!defined(ICORE_LA_SCALAR_VALIDATION) || !ICORE_LA_SCALAR_VALIDATION)
/* Independent nonnegative square roots share one SIMD instruction. Invalid
 * blocks retain the original scalar row-order status and scratch writes. */
static ICORE_LA_AVX2_TARGET int prepare_weights_avx2(icore_la_workspace *w, const double *weights)
{
    int j = 0;
    const __m256i exponent = _mm256_set1_epi64x((long long)UINT64_C(0x7ff0000000000000));
    const __m256d zero = _mm256_setzero_pd();
    for (; j + 4 <= w->ng; j += 4) {
        __m256d value = _mm256_loadu_pd(weights + j);
        __m256i bits = _mm256_castpd_si256(value);
        int bad = _mm256_movemask_pd(
            _mm256_castsi256_pd(_mm256_cmpeq_epi64(_mm256_and_si256(bits, exponent), exponent)));
        if (bad || _mm256_movemask_pd(_mm256_cmp_pd(value, zero, _CMP_LT_OQ))) {
            for (int k = 0; k < 4; ++k) {
                if (!isfinite(weights[j + k]))
                    return ICORE_LA_NONFINITE;
                if (weights[j + k] < 0.0)
                    return ICORE_LA_INVALID;
                w->sqrtw[j + k] = sqrt(weights[j + k]);
            }
        } else
            _mm256_storeu_pd(w->sqrtw + j, _mm256_sqrt_pd(value));
    }
    for (; j < w->ng; ++j) {
        if (!isfinite(weights[j]))
            return ICORE_LA_NONFINITE;
        if (weights[j] < 0.0)
            return ICORE_LA_INVALID;
        w->sqrtw[j] = sqrt(weights[j]);
    }
    return ICORE_LA_OK;
}
#endif

static int prepare_scaling(icore_la_workspace *w, const double *diagonal, const double *weights,
                           double reg, int equilibrate)
{
    int i, j;
    w->max_factored_diagonal = 0.0;
#if ICORE_LA_AVX2 && (!defined(ICORE_LA_SCALAR_VALIDATION) || !ICORE_LA_SCALAR_VALIDATION)
    if (w->backend == ICORE_LA_BLASFEO) {
        int status = prepare_weights_avx2(w, weights);
        if (status != ICORE_LA_OK)
            return status;
    } else
#endif
        for (j = 0; j < w->ng; ++j) {
            if (!isfinite(weights[j]))
                return ICORE_LA_NONFINITE;
            if (weights[j] < 0.0)
                return ICORE_LA_INVALID;
            w->sqrtw[j] = sqrt(weights[j]);
        }
#if ICORE_LA_AVX2 && ICORE_LA_FUSED_WEIGHTED_GRAM
    if (w->backend == ICORE_LA_BLASFEO && w->ng)
        return prepare_packed_gram(w, diagonal, reg, equilibrate);
#endif
    for (i = 0; i < w->n; ++i) {
        double base = original_hdiag(w, i) + reg, total;
        if (diagonal) {
            if (!isfinite(diagonal[i]))
                return ICORE_LA_NONFINITE;
            if (diagonal[i] < 0.0)
                return ICORE_LA_INVALID;
            base += diagonal[i];
        }
        if (!isfinite(base))
            return ICORE_LA_NONFINITE;
        w->vy[i] = base;
        total = base;
        for (j = 0; j < w->ng; ++j) {
            double a = original_g(w, j, i) * w->sqrtw[j];
            total += a * a;
        }
        if (!isfinite(total))
            return ICORE_LA_NONFINITE;
        if (equilibrate) {
            if (!(total > 0.0))
                return ICORE_LA_NOT_SPD;
            w->scale[i] = 1.0 / sqrt(total);
        } else {
            w->scale[i] = 1.0;
        }
        total = fabs((total * w->scale[i]) * w->scale[i]);
        if (total > w->max_factored_diagonal)
            w->max_factored_diagonal = total;
    }
    return ICORE_LA_OK;
}

#if ICORE_LA_BF
static int factor_packed(icore_la_workspace *w, int equilibrate)
{
    int i, j;
    if (equilibrate) {
        for (i = 0; i < w->n; ++i) {
            for (j = 0; j < i; ++j)
                BLASFEO_DMATEL(&w->L, i, j) =
                    (BLASFEO_DMATEL(&w->H, i, j) * w->scale[i]) * w->scale[j];
            BLASFEO_DMATEL(&w->L, i, i) = (w->vy[i] * w->scale[i]) * w->scale[i];
        }
    } else {
        blasfeo_dtrcp_l(w->n, &w->H, 0, 0, &w->L, 0, 0);
        blasfeo_ddiain(w->n, 1.0, &w->Y, 0, &w->L, 0, 0);
    }
    w->L.use_dA = 0;
    if (w->ng) {
        /* WG = D*G^T*sqrt(W). Keep all matrices packed, then fuse the
         * symmetric rank update and Cholesky; no assembled A or unpack. */
#if !(ICORE_LA_AVX2 && ICORE_LA_FUSED_WEIGHTED_GRAM)
        blasfeo_dgemm_nd(w->n, w->ng, 1.0, &w->GT, 0, 0, &w->W, 0, 0.0, &w->WG, 0, 0, &w->WG, 0, 0);
#endif
        if (equilibrate)
            blasfeo_dgemm_dn(w->n, w->ng, 1.0, &w->D, 0, &w->WG, 0, 0, 0.0, &w->WG, 0, 0, &w->WG, 0,
                             0);
        blasfeo_dsyrk_dpotrf_ln(w->n, w->ng, &w->WG, 0, 0, &w->WG, 0, 0, &w->L, 0, 0, &w->L, 0, 0);
    } else {
        blasfeo_dpotrf_l(w->n, &w->L, 0, 0, &w->L, 0, 0);
    }
    /* Inspect each packed diagonal once. Original L/reciprocal failures
     * retain precedence over the later squared-pivot policy: defer the
     * latter status until all original L checks have completed. */
    double min_pivot = INFINITY, min_scaled = INFINITY;
    int pivot_status = ICORE_LA_OK;
    for (i = 0; i < w->n; ++i) {
        double d = BLASFEO_DMATEL(&w->L, i, i), p;
        if (!isfinite(d))
            return ICORE_LA_NONFINITE;
        if (!(d > 0.0))
            return ICORE_LA_NOT_SPD;
        if (!isfinite(w->L.dA[i]) || !(w->L.dA[i] > 0.0))
            return ICORE_LA_NONFINITE;
        if (pivot_status != ICORE_LA_OK)
            continue;
        p = d * d;
        if (!isfinite(p)) {
            pivot_status = ICORE_LA_NONFINITE;
            continue;
        }
        if (!(p > 0.0)) {
            pivot_status = ICORE_LA_NOT_SPD;
            continue;
        }
        if (p < min_scaled)
            min_scaled = p;
        if (equilibrate) {
            d /= w->scale[i];
            p = d * d;
            if (!isfinite(p)) {
                pivot_status = ICORE_LA_NONFINITE;
                continue;
            }
            if (!(p > 0.0)) {
                pivot_status = ICORE_LA_NOT_SPD;
                continue;
            }
        }
        if (p < min_pivot)
            min_pivot = p;
    }
    if (pivot_status != ICORE_LA_OK)
        return pivot_status;
    w->min_physical_pivot = min_pivot;
    w->min_scaled_pivot = min_scaled;
    return ICORE_LA_OK;
}
#endif

static int factor_portable(icore_la_workspace *w, int equilibrate)
{
    int i, j, k, r, n = w->n;
    double *l = w->l;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < i; ++j)
            l[(size_t)i * n + j] = (w->h[(size_t)i * n + j] * w->scale[i]) * w->scale[j];
        l[(size_t)i * n + i] = (w->vy[i] * w->scale[i]) * w->scale[i];
    }
    for (r = 0; r < w->ng; ++r) {
        for (i = 0; i < n; ++i) {
            w->vx[i] = w->g[(size_t)r * n + i] * w->sqrtw[r];
            if (equilibrate)
                w->vx[i] *= w->scale[i];
        }
        for (i = 0; i < n; ++i)
            for (j = 0; j <= i; ++j)
                l[(size_t)i * n + j] += w->vx[i] * w->vx[j];
    }
    for (i = 0; i < n; ++i)
        for (j = 0; j <= i; ++j) {
            double v = l[(size_t)i * n + j];
            for (k = 0; k < j; ++k)
                v -= l[(size_t)i * n + k] * l[(size_t)j * n + k];
            if (!isfinite(v))
                return ICORE_LA_NONFINITE;
            if (i == j) {
                if (!(v > 0.0))
                    return ICORE_LA_NOT_SPD;
                l[(size_t)i * n + i] = sqrt(v);
            } else {
                v /= l[(size_t)j * n + j];
                if (!isfinite(v))
                    return ICORE_LA_NONFINITE;
                l[(size_t)i * n + j] = v;
            }
        }
    return ICORE_LA_OK;
}

int icore_la_factor(icore_la_workspace *w, const double *diagonal, const double *weights,
                double regularization, int equilibrate)
{
    int status, i;
    if (!w)
        return ICORE_LA_INVALID;
    w->factor_ready = 0;
    if (!w->h_loaded)
        return ICORE_LA_NOT_READY;
    if ((w->ng && !weights) || regularization < 0.0 || (equilibrate != 0 && equilibrate != 1))
        return ICORE_LA_INVALID;
    if (!isfinite(regularization))
        return ICORE_LA_NONFINITE;
    status = prepare_scaling(w, diagonal, weights, regularization, equilibrate);
    if (status != ICORE_LA_OK)
        return status;
#if ICORE_LA_BF
    if (w->backend == ICORE_LA_BLASFEO)
        status = factor_packed(w, equilibrate);
    else
#endif
        status = factor_portable(w, equilibrate);
    if (status == ICORE_LA_OK) {
#if ICORE_LA_BF
        if (w->backend != ICORE_LA_BLASFEO)
#endif
        {
            double min_pivot = INFINITY, min_scaled = INFINITY;
            for (i = 0; i < w->n; ++i) {
                double d, p;
#if ICORE_LA_BF
                if (w->backend == ICORE_LA_BLASFEO)
                    d = BLASFEO_DMATEL(&w->L, i, i);
                else
#endif
                    d = w->l[(size_t)i * w->n + i];
                p = d * d;
                if (!isfinite(p))
                    return ICORE_LA_NONFINITE;
                if (!(p > 0.0))
                    return ICORE_LA_NOT_SPD;
                if (p < min_scaled)
                    min_scaled = p;
                /* With no equilibration D is exactly I: the physical pivot is
             * the already checked scaled pivot. Avoid n divisions by one
             * and duplicate squares without changing either pivot policy. */
                if (equilibrate) {
                    d /= w->scale[i];
                    p = d * d;
                    if (!isfinite(p))
                        return ICORE_LA_NONFINITE;
                    if (!(p > 0.0))
                        return ICORE_LA_NOT_SPD;
                }
                if (p < min_pivot)
                    min_pivot = p;
            }
            w->min_physical_pivot = min_pivot;
            w->min_scaled_pivot = min_scaled;
        }
        w->factor_equilibrated = equilibrate;
        w->factor_ready = 1;
    }
    return status;
}

double icore_la_min_physical_pivot(const icore_la_workspace *w)
{
    return w && w->factor_ready ? w->min_physical_pivot : NAN;
}

double icore_la_min_scaled_pivot(const icore_la_workspace *w)
{
    return w && w->factor_ready ? w->min_scaled_pivot : NAN;
}

int icore_la_pivots_acceptable(const icore_la_workspace *w, double absolute, double relative)
{
    if (!w || !w->factor_ready || !isfinite(absolute) || !isfinite(relative) || absolute < 0.0 ||
        relative < 0.0)
        return 0;
    return w->min_physical_pivot >= absolute &&
           w->min_scaled_pivot >= relative * w->max_factored_diagonal;
}

int icore_la_solve(icore_la_workspace *w, const double *rhs, double *x)
{
    int i, j, n;
    if (!w || !rhs || !x)
        return ICORE_LA_INVALID;
    if (!w->factor_ready)
        return ICORE_LA_NOT_READY;
    n = w->n;
    if (w->factor_equilibrated) {
        if (!finite_workspace_vector(w, rhs, n))
            return ICORE_LA_NONFINITE;
        for (i = 0; i < n; ++i)
            w->vx[i] = rhs[i] * w->scale[i];
        if (!finite_workspace_vector(w, w->vx, n))
            return ICORE_LA_NONFINITE;
    } else if (!finite_workspace_copy(w, w->vx, rhs, n))
        return ICORE_LA_NONFINITE;
#if ICORE_LA_BF
    if (w->backend == ICORE_LA_BLASFEO) {
        blasfeo_dtrsv_lnn(n, &w->L, 0, 0, &w->X, 0, &w->Y, 0);
        blasfeo_dtrsv_ltn(n, &w->L, 0, 0, &w->Y, 0, &w->Z, 0);
    } else
#endif
    {
        for (i = 0; i < n; ++i) {
            double v = w->vx[i];
            for (j = 0; j < i; ++j)
                v -= w->l[(size_t)i * n + j] * w->vy[j];
            w->vy[i] = v / w->l[(size_t)i * n + i];
        }
        for (i = n - 1; i >= 0; --i) {
            double v = w->vy[i];
            for (j = i + 1; j < n; ++j)
                v -= w->l[(size_t)j * n + i] * w->vz[j];
            w->vz[i] = v / w->l[(size_t)i * n + i];
        }
    }
    if (w->factor_equilibrated)
        for (i = 0; i < n; ++i)
            w->vz[i] *= w->scale[i];
    if (!finite_workspace_vector(w, w->vz, n))
        return ICORE_LA_NONFINITE;
    memcpy(x, w->vz, (size_t)n * sizeof(double));
    return ICORE_LA_OK;
}

int icore_la_direction(icore_la_workspace *w, const double *rhs, const double *z, double *x, double *gx)
{
#if ICORE_LA_BF
    int i, n, m;
    if (!w || !rhs || !x)
        return ICORE_LA_INVALID;
    if (w->backend != ICORE_LA_BLASFEO)
        return ICORE_LA_UNAVAILABLE;
    if (!w->factor_ready)
        return ICORE_LA_NOT_READY;
    n = w->n;
    m = w->ng;
    if (m && (!z || !gx))
        return ICORE_LA_INVALID;
    if (!finite_workspace_copy(w, w->gz, z, m))
        return ICORE_LA_NONFINITE;
    if (m) {
        blasfeo_dgemv_n(n, m, 1.0, &w->GT, 0, 0, &w->GZ, 0, 0.0, &w->Y, 0, &w->Y, 0);
        if (!finite_workspace_vector(w, w->vy, n))
            return ICORE_LA_NONFINITE;
    } else
        memset(w->vy, 0, (size_t)n * sizeof(double));
    /* Subtract before scaling, exactly as in the separate product/solve. */
    for (i = 0; i < n; ++i)
        w->vx[i] = rhs[i] - w->vy[i];
    if (!finite_workspace_vector(w, w->vx, n))
        return ICORE_LA_NONFINITE;
    if (w->factor_equilibrated) {
        for (i = 0; i < n; ++i)
            w->vx[i] *= w->scale[i];
        if (!finite_workspace_vector(w, w->vx, n))
            return ICORE_LA_NONFINITE;
    }
    blasfeo_dtrsv_lnn(n, &w->L, 0, 0, &w->X, 0, &w->Y, 0);
    blasfeo_dtrsv_ltn(n, &w->L, 0, 0, &w->Y, 0, &w->Z, 0);
    if (w->factor_equilibrated)
        for (i = 0; i < n; ++i)
            w->vz[i] *= w->scale[i];
    if (!finite_workspace_vector(w, w->vz, n))
        return ICORE_LA_NONFINITE;
    if (m) {
        /* G*x consumes the already validated solution in Z directly. */
        blasfeo_dgemv_t(n, m, 1.0, &w->GT, 0, 0, &w->Z, 0, 0.0, &w->GX, 0, &w->GX, 0);
        if (!finite_workspace_vector(w, w->gx, m))
            return ICORE_LA_NONFINITE;
    }
    memcpy(x, w->vz, (size_t)n * sizeof(double));
    if (m)
        memcpy(gx, w->gx, (size_t)m * sizeof(double));
    return ICORE_LA_OK;
#else
    (void)w;
    (void)rhs;
    (void)z;
    (void)x;
    (void)gx;
    return ICORE_LA_UNAVAILABLE;
#endif
}

int icore_la_hmv(icore_la_workspace *w, double alpha, const double *x, double beta, double *y)
{
    int i, j, n;
    if (!w || !x || !y)
        return ICORE_LA_INVALID;
    if (!w->h_loaded)
        return ICORE_LA_NOT_READY;
    n = w->n;
    if (!isfinite(alpha) || !isfinite(beta) || !finite_workspace_copy(w, w->vx, x, n) ||
        (beta != 0.0 && !finite_workspace_copy(w, w->vy, y, n)))
        return ICORE_LA_NONFINITE;
    if (alpha == 0.0 && beta == 0.0) {
        memset(y, 0, (size_t)n * sizeof(double));
        return ICORE_LA_OK;
    }
    if (beta == 0.0)
        memset(w->vy, 0, (size_t)n * sizeof(double));
#if ICORE_LA_BF
    if (w->backend == ICORE_LA_BLASFEO) {
        blasfeo_dsymv_l(n, alpha, &w->H, 0, 0, &w->X, 0, beta, &w->Y, 0, &w->Z, 0);
    } else
#endif
    {
        for (i = 0; i < n; ++i)
            w->vz[i] = beta * w->vy[i];
        for (i = 0; i < n; ++i) {
            w->vz[i] += alpha * w->h[(size_t)i * n + i] * w->vx[i];
            for (j = 0; j < i; ++j) {
                double a = alpha * w->h[(size_t)i * n + j];
                w->vz[i] += a * w->vx[j];
                w->vz[j] += a * w->vx[i];
            }
        }
    }
    if (!finite_workspace_vector(w, w->vz, n))
        return ICORE_LA_NONFINITE;
    memcpy(y, w->vz, (size_t)n * sizeof(double));
    return ICORE_LA_OK;
}

/* Some pinned GEMV_NT kernels read beta_t's input even when beta_t is
 * zero. W is a certified finite seed: calloc initializes its full padded
 * extent, and prepare_scaling stores only sqrt of validated finite,
 * nonnegative weights. GX can contain a rejected overflow result. Using W
 * prevents that rejected scratch from contaminating the next valid product;
 * the original matrices, kernels, finite tests and public commits remain. */
static int hg_products(icore_la_workspace *w, const double *x, const double *z, double *hx, double *gx,
                       double *gtz, int forward, const double *rd, int residual)
{
#if ICORE_LA_BF
    if (!w || !x || !hx || (!residual && !gtz) || (residual && !rd) ||
        (forward != 0 && forward != 1))
        return ICORE_LA_INVALID;
    if (w->backend != ICORE_LA_BLASFEO)
        return ICORE_LA_UNAVAILABLE;
    if (!w->h_loaded)
        return ICORE_LA_NOT_READY;
    const int n = w->n, m = w->ng;
    if (m && (!z || (forward && !gx)))
        return ICORE_LA_INVALID;
    if (!finite_workspace_copy(w, w->vx, x, n) || !finite_workspace_copy(w, w->gz, z, m))
        return ICORE_LA_NONFINITE;
    memset(w->vy, 0, (size_t)n * sizeof(double));
    blasfeo_dsymv_l(n, 1.0, &w->H, 0, 0, &w->X, 0, 0.0, &w->Y, 0, &w->Z, 0);
    if (!finite_workspace_vector(w, w->vz, n))
        return ICORE_LA_NONFINITE;
    if (m) {
        if (forward)
            blasfeo_dgemv_nt(n, m, 1.0, 1.0, &w->GT, 0, 0, &w->GZ, 0, &w->X, 0, 0.0, 0.0, &w->Y, 0,
                             &w->W, 0, &w->Y, 0, &w->GX, 0);
        else
            blasfeo_dgemv_n(n, m, 1.0, &w->GT, 0, 0, &w->GZ, 0, 0.0, &w->Y, 0, &w->Y, 0);
        if (!finite_workspace_vector(w, w->vy, n) ||
            (forward && !finite_workspace_vector(w, w->gx, m)))
            return ICORE_LA_NONFINITE;
    }
    if (residual) {
        /* Preserve the caller's original two rounded subtractions, after
         * independently validating both original matrix products. X is no
         * longer needed after GEMV, and is owned, padded export scratch. */
        for (int i = 0; i < n; ++i)
            w->vx[i] = (-rd[i] - w->vz[i]) - w->vy[i];
        if (!finite_workspace_vector(w, w->vx, n))
            return ICORE_LA_NONFINITE;
        memcpy(hx, w->vx, (size_t)n * sizeof(double));
    } else {
        memcpy(hx, w->vz, (size_t)n * sizeof(double));
        memcpy(gtz, w->vy, (size_t)n * sizeof(double));
    }
    if (forward && m)
        memcpy(gx, w->gx, (size_t)m * sizeof(double));
    return ICORE_LA_OK;
#else
    (void)w;
    (void)x;
    (void)z;
    (void)hx;
    (void)gx;
    (void)gtz;
    (void)forward;
    (void)rd;
    (void)residual;
    return ICORE_LA_UNAVAILABLE;
#endif
}

int icore_la_hg_products(icore_la_workspace *w, const double *x, const double *z, double *hx, double *gx,
                     double *gtz, int forward)
{
    return hg_products(w, x, z, hx, gx, gtz, forward, NULL, 0);
}

int icore_la_hg_residual(icore_la_workspace *w, const double *x, const double *z, const double *rd,
                     double *residual, double *gx, int forward)
{
    return hg_products(w, x, z, residual, gx, NULL, forward, rd, 1);
}

static int g_products(icore_la_workspace *w, const double *x, const double *z, double *gx, double *gtz,
                      int forward, int transpose)
{
    int i, j, n, m;
    if (!w || (forward && !x) || (transpose && !gtz))
        return ICORE_LA_INVALID;
    n = w->n;
    m = w->ng;
    if (n <= 0 || m < 0)
        return ICORE_LA_INVALID;
    if ((forward && m && !gx) || (transpose && m && !z))
        return ICORE_LA_INVALID;
    if ((forward && !finite_workspace_copy(w, w->vx, x, n)) ||
        (transpose && !finite_workspace_copy(w, w->gz, z, m)))
        return ICORE_LA_NONFINITE;
    if (!m) {
        if (transpose)
            memset(gtz, 0, (size_t)n * sizeof(double));
        return ICORE_LA_OK;
    }
#if ICORE_LA_BF
    if (w->backend == ICORE_LA_BLASFEO) {
        if (forward && transpose) {
            blasfeo_dgemv_nt(n, m, 1.0, 1.0, &w->GT, 0, 0, &w->GZ, 0, &w->X, 0, 0.0, 0.0, &w->Y, 0,
                             &w->W, 0, &w->Y, 0, &w->GX, 0);
        } else if (forward) {
            blasfeo_dgemv_t(n, m, 1.0, &w->GT, 0, 0, &w->X, 0, 0.0, &w->GX, 0, &w->GX, 0);
        } else {
            blasfeo_dgemv_n(n, m, 1.0, &w->GT, 0, 0, &w->GZ, 0, 0.0, &w->Y, 0, &w->Y, 0);
        }
    } else
#endif
    {
        if (forward)
            memset(w->gx, 0, (size_t)m * sizeof(double));
        if (transpose)
            memset(w->vy, 0, (size_t)n * sizeof(double));
        for (i = 0; i < m; ++i)
            for (j = 0; j < n; ++j) {
                double a = w->g[(size_t)i * n + j];
                if (forward)
                    w->gx[i] += a * w->vx[j];
                if (transpose)
                    w->vy[j] += a * w->gz[i];
            }
    }
    if ((forward && !finite_workspace_vector(w, w->gx, m)) ||
        (transpose && !finite_workspace_vector(w, w->vy, n)))
        return ICORE_LA_NONFINITE;
    if (forward)
        memcpy(gx, w->gx, (size_t)m * sizeof(double));
    if (transpose)
        memcpy(gtz, w->vy, (size_t)n * sizeof(double));
    return ICORE_LA_OK;
}

int icore_la_gmv(icore_la_workspace *w, const double *x, double *gx)
{
    return g_products(w, x, NULL, gx, NULL, 1, 0);
}

int icore_la_gtmv(icore_la_workspace *w, const double *z, double *gtz)
{
    return g_products(w, NULL, z, NULL, gtz, 0, 1);
}

int icore_la_gmv_pair(icore_la_workspace *w, const double *x, const double *z, double *gx, double *gtz)
{
    return g_products(w, x, z, gx, gtz, 1, 1);
}
