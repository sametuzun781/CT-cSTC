#include "ocp_la.h"
#include "mixed_factor.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__FAST_MATH__)
#error "ocp_la requires finite-value checks: compile without fast-math"
#endif

#if defined(ICORE_LA_WITH_BLASFEO) && ICORE_LA_WITH_BLASFEO
#include <blasfeo_common.h>
#include <blasfeo_d_aux.h>
#include <blasfeo_d_blasfeo_api.h>
#if !defined(MF_PANELMAJ)
#error "The optional OCP stage backend requires panel-major BLASFEO"
#endif

/* Every live packed workspace passed icore_la_blasfeo_available(), including
 * the CPU/OS AVX2 check. A local target keeps the standalone build portable
 * without enabling AVX2 globally. Other targets retain the scalar checks. */
#if (!defined(ICORE_OCP_LA_SCALAR_VALIDATION) || !ICORE_OCP_LA_SCALAR_VALIDATION) && D_PS == 4 &&                    \
    defined(TARGET_NEED_FEATURE_AVX2) && (defined(__x86_64__) || defined(__i386__)) &&             \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define ICORE_OCP_LA_SIMD_VALIDATION 1
#define ICORE_OCP_LA_AVX2_TARGET __attribute__((target("avx2")))
#else
#define ICORE_OCP_LA_SIMD_VALIDATION 0
#define ICORE_OCP_LA_AVX2_TARGET
#endif

#if ICORE_OCP_LA_SIMD_VALIDATION && defined(TARGET_X64_INTEL_HASWELL) && defined(LA_HIGH_PERFORMANCE) &&    \
    D_PS == 4
#define ICORE_OCP_LA_DYNAMICS_CACHE 1
#define ICORE_OCP_LA_FMA_TARGET __attribute__((target("avx2,fma")))
#else
#define ICORE_OCP_LA_DYNAMICS_CACHE 0
#endif

#if defined(ICORE_BUNDLED_SP_ROUTINES) && ICORE_OCP_LA_SIMD_VALIDATION &&                                   \
    defined(TARGET_X64_INTEL_HASWELL) && defined(LA_HIGH_PERFORMANCE) && D_PS == 4 && S_PS == 8 && \
    !defined(ICORE_OCP_SCALAR) && !defined(ICORE_OCP_LA_DISABLE_MIXED)
#define ICORE_OCP_LA_MIXED_SUPPORTED 1
#else
#define ICORE_OCP_LA_MIXED_SUPPORTED 0
#endif
/* Packed implementation, ordered as ownership, factorization, and sweeps.
 * The unavailable-backend stubs are kept out of this numeric path. */
/* ---- Packed storage, ownership, and dynamics coefficient loading. ---- */

struct icore_ocp_la_mixed;
static void mixed_create(icore_ocp_la_workspace *w);
static void mixed_destroy(icore_ocp_la_workspace *w);
static void mixed_load_dynamics(icore_ocp_la_workspace *w, int stage);
static void mixed_load_cost(icore_ocp_la_workspace *w, int stage);

struct icore_ocp_la_workspace {
    struct icore_ocp_la_mixed *mixed;
    int N, nx, nr, n;
    int owned_factors;
    int sqrt_next;
    int sqrt_equilibrated;
    int sqrt_root_omitted;
    int sqrt_initial_zero;
    int sqrt_terminal_diagonal;
    int sqrt_recovery_stage, sqrt_recovery_initial_fixed;
    double sqrt_recovery_reg;
    void *storage;
    double *scale;
    struct blasfeo_dmat *DT, *C, *cost;
    unsigned char *loaded, *cost_loaded;
    struct blasfeo_dmat PN, H, M, DTP, F, P;
    struct blasfeo_dvec X, Y, Z, V;
    struct blasfeo_dmat *checkpoint_C;
    void *checkpoint_storage;
    size_t checkpoint_bytes;
    int checkpoint_valid, checkpoint_equilibrated, checkpoint_root_omitted;
    int checkpoint_initial_zero;
};

static void sqrt_recovery_clear(icore_ocp_la_workspace *w)
{
    if (w) {
        icore_ocp_la_mixed_sweeps_invalidate(w);
        w->sqrt_recovery_stage = -1;
        w->sqrt_recovery_initial_fixed = 0;
        w->sqrt_recovery_reg = 0.0;
    }
}

static size_t align64(size_t n)
{
    return (n + 63u) & ~(size_t)63u;
}

static void *take(unsigned char **next, size_t bytes)
{
    void *result = *next;
    *next += align64(bytes);
    return result;
}

static int create_impl(int N, int nx, int nr, int owned, icore_ocp_la_workspace **out)
{
    icore_ocp_la_workspace *w;
    size_t pbytes, hbytes, cbytes, dbytes, fbytes, vbytes, sbytes, total, rows;
    unsigned char *next;
    int k, n;
    if (!out)
        return ICORE_LA_INVALID;
    *out = NULL;
    if (N <= 0 || nx <= 0 || nr <= 0 || nx > INT_MAX - nr - 64)
        return ICORE_LA_INVALID;
    n = nx + nr;
    rows = (size_t)n + 64;
    if (rows > ((size_t)INT_MAX - 4096) / sizeof(double) / rows)
        return ICORE_LA_INVALID;
    if (!icore_la_blasfeo_available())
        return ICORE_LA_UNAVAILABLE;
    pbytes = align64(blasfeo_memsize_dmat(nx, nx));
    hbytes = align64(blasfeo_memsize_dmat(n, n));
    dbytes = align64(blasfeo_memsize_dmat(n, nx));
    fbytes = hbytes;
    cbytes = owned ? hbytes : pbytes;
    vbytes = align64(blasfeo_memsize_dvec(n));
    sbytes = align64((size_t)n * sizeof(double));
    total = 4 * pbytes + 2 * hbytes + dbytes + fbytes + 4 * vbytes + sbytes;
    if ((size_t)N > (SIZE_MAX - total - 63u) / (dbytes + cbytes + hbytes))
        return ICORE_LA_NOMEM;
    total += (size_t)N * (dbytes + cbytes + hbytes);
    w = (icore_ocp_la_workspace *)calloc(1, sizeof(*w));
    if (!w)
        return ICORE_LA_NOMEM;
    w->N = N;
    w->nx = nx;
    w->nr = nr;
    w->n = n;
    w->owned_factors = owned;
    w->sqrt_next = -2;
    sqrt_recovery_clear(w);
    w->DT = (struct blasfeo_dmat *)calloc((size_t)N, sizeof(*w->DT));
    w->C = (struct blasfeo_dmat *)calloc((size_t)N + 1u, sizeof(*w->C));
    w->cost = (struct blasfeo_dmat *)calloc((size_t)N + 1u, sizeof(*w->cost));
    w->loaded = (unsigned char *)calloc((size_t)N, 1);
    w->cost_loaded = (unsigned char *)calloc((size_t)N + 1u, 1);
    w->storage = calloc(1, total + 63u);
    if (!w->DT || !w->C || !w->cost || !w->loaded || !w->cost_loaded || !w->storage) {
        icore_ocp_la_destroy(w);
        return ICORE_LA_NOMEM;
    }
    next = (unsigned char *)(((uintptr_t)w->storage + 63u) & ~(uintptr_t)63u);
    for (k = 0; k < N; ++k)
        blasfeo_create_dmat(n, nx, w->DT + k, take(&next, dbytes));
    blasfeo_create_dmat(nx, nx, &w->PN, take(&next, pbytes));
    blasfeo_create_dmat(n, n, &w->H, take(&next, hbytes));
    blasfeo_create_dmat(n, n, &w->M, take(&next, hbytes));
    blasfeo_create_dmat(n, nx, &w->DTP, take(&next, dbytes));
    blasfeo_create_dmat(n, n, &w->F, take(&next, fbytes));
    blasfeo_create_dmat(nx, nx, &w->P, take(&next, pbytes));
    for (k = 0; k < N; ++k)
        blasfeo_create_dmat(owned ? n : nx, owned ? n : nx, w->C + k, take(&next, cbytes));
    blasfeo_create_dmat(nx, nx, w->C + N, take(&next, pbytes));
    for (k = 0; k < N; ++k)
        blasfeo_create_dmat(n, n, w->cost + k, take(&next, hbytes));
    blasfeo_create_dmat(nx, nx, w->cost + N, take(&next, pbytes));
    blasfeo_create_dvec(n, &w->X, take(&next, vbytes));
    blasfeo_create_dvec(n, &w->Y, take(&next, vbytes));
    blasfeo_create_dvec(n, &w->Z, take(&next, vbytes));
    blasfeo_create_dvec(n, &w->V, take(&next, vbytes));
    w->scale = (double *)take(&next, sbytes);
    /* This performance-only bank is optional. Failure must not turn a
     * previously usable packed workspace into a creation failure. */
    if ((size_t)N <= (SIZE_MAX - 63u - pbytes) / cbytes) {
        size_t bytes = (size_t)N * cbytes + pbytes + 63u;
        if ((size_t)N + 1u <= (SIZE_MAX - bytes) / sizeof(*w->checkpoint_C)) {
            w->checkpoint_C =
                (struct blasfeo_dmat *)calloc((size_t)N + 1u, sizeof(*w->checkpoint_C));
            w->checkpoint_storage = calloc(1, bytes);
            if (w->checkpoint_C && w->checkpoint_storage) {
                next =
                    (unsigned char *)(((uintptr_t)w->checkpoint_storage + 63u) & ~(uintptr_t)63u);
                for (k = 0; k < N; ++k)
                    blasfeo_create_dmat(owned ? n : nx, owned ? n : nx, w->checkpoint_C + k,
                                        take(&next, cbytes));
                blasfeo_create_dmat(nx, nx, w->checkpoint_C + N, take(&next, pbytes));
                w->checkpoint_bytes = bytes + ((size_t)N + 1u) * sizeof(*w->checkpoint_C);
            } else {
                free(w->checkpoint_C);
                free(w->checkpoint_storage);
                w->checkpoint_C = NULL;
                w->checkpoint_storage = NULL;
            }
        }
    }
    mixed_create(w);
    *out = w;
    return ICORE_LA_OK;
}

int icore_ocp_la_create(int N, int nx, int nr, icore_ocp_la_workspace **out)
{
    return create_impl(N, nx, nr, 0, out);
}

/* Owned chains factor directly into persistent packed stage storage. If the
 * larger optional allocation fails, preserve the legacy packed backend. */
int icore_ocp_la_create_owned(int N, int nx, int nr, icore_ocp_la_workspace **out)
{
    int status = create_impl(N, nx, nr, 1, out);
    if (status == ICORE_LA_NOMEM)
        status = create_impl(N, nx, nr, 0, out);
    return status;
}

int icore_ocp_la_sqrt_owns_factors(const icore_ocp_la_workspace *w)
{
    return w && w->owned_factors;
}

static int root_offset(const icore_ocp_la_workspace *w, int stage)
{
    return w->owned_factors && stage < w->N ? w->nr : 0;
}

void icore_ocp_la_destroy(icore_ocp_la_workspace *w)
{
    if (w) {
        mixed_destroy(w);
        free(w->storage);
        free(w->loaded);
        free(w->cost_loaded);
        free(w->checkpoint_storage);
        free(w->checkpoint_C);
        free(w->DT);
        free(w->C);
        free(w->cost);
        free(w);
    }
}

static ICORE_OCP_LA_AVX2_TARGET int finite_prefix(const double *values, int count);

static int load_dynamics_input(icore_ocp_la_workspace *w, int stage, const double *D, size_t stride,
                               int finite_certified)
{
    int i;
    if (!w || stage < 0 || stage >= w->N || !D || stride < (size_t)w->n || stride > INT_MAX)
        return ICORE_LA_INVALID;
    if (!finite_certified)
        for (i = 0; i < w->nx; ++i)
            if (!finite_prefix(D + (size_t)i * stride, w->n))
                return ICORE_LA_NONFINITE;
    /* Row-major nx*n input is column-major n*nx D transpose. */
    blasfeo_pack_dmat(w->n, w->nx, (double *)D, (int)stride, w->DT + stage, 0, 0);
    w->loaded[stage] = 1;
    mixed_load_dynamics(w, stage);
    w->checkpoint_valid = 0;
    w->sqrt_next = -2;
    w->sqrt_equilibrated = 0;
    w->sqrt_root_omitted = 0;
    sqrt_recovery_clear(w);
    return ICORE_LA_OK;
}

int icore_ocp_la_load_dynamics(icore_ocp_la_workspace *w, int stage, const double *D, size_t stride)
{
    return load_dynamics_input(w, stage, D, stride, 0);
}

int icore_ocp_la_load_dynamics_finite(icore_ocp_la_workspace *w, int stage, const double *D, size_t stride)
{
    return load_dynamics_input(w, stage, D, stride, 1);
}

/* ---- Packed classical Riccati products and stage factorization. ---- */

static int form_products(icore_ocp_la_workspace *w, int stage, const double *Pn, size_t ldP, const double *H,
                         size_t ldH, double reg)
{
    int i, j;
    if (!w || stage < 0 || stage >= w->N || !Pn || !H || ldP < (size_t)w->nx ||
        ldH < (size_t)w->n || ldP > INT_MAX || ldH > INT_MAX || reg < 0.0)
        return ICORE_LA_INVALID;
    if (!w->loaded[stage])
        return ICORE_LA_NOT_READY;
    if (!isfinite(reg))
        return ICORE_LA_NONFINITE;
    for (i = 0; i < w->nx; ++i)
        for (j = 0; j <= i; ++j) {
            double v = Pn[(size_t)i * ldP + j];
            if (!isfinite(v))
                return ICORE_LA_NONFINITE;
            BLASFEO_DMATEL(&w->PN, i, j) = v;
            BLASFEO_DMATEL(&w->PN, j, i) = v;
        }
    for (i = 0; i < w->n; ++i)
        for (j = 0; j <= i; ++j) {
            double v = H[(size_t)i * ldH + j];
            if (!isfinite(v))
                return ICORE_LA_NONFINITE;
            BLASFEO_DMATEL(&w->H, i, j) = v;
        }
    w->PN.use_dA = 0;
    w->H.use_dA = 0;
    /* DTP = D^T P; M lower = H lower + DTP D. */
    blasfeo_dgemm_nn(w->n, w->nx, w->nx, 1.0, w->DT + stage, 0, 0, &w->PN, 0, 0, 0.0, &w->DTP, 0, 0,
                     &w->DTP, 0, 0);
    blasfeo_dsyrk_ln(w->n, w->nx, 1.0, &w->DTP, 0, 0, w->DT + stage, 0, 0, 1.0, &w->H, 0, 0, &w->M,
                     0, 0);
    for (i = 0; i < w->n; ++i) {
        BLASFEO_DMATEL(&w->M, i, i) += reg;
        for (j = 0; j <= i; ++j)
            if (!isfinite(BLASFEO_DMATEL(&w->M, i, j)))
                return ICORE_LA_NONFINITE;
    }
    w->M.use_dA = 0;
    return ICORE_LA_OK;
}

static void export_symmetric(struct blasfeo_dmat *S, int n, double *out)
{
    int i, j;
    for (i = 0; i < n; ++i)
        for (j = 0; j <= i; ++j) {
            double v = BLASFEO_DMATEL(S, i, j);
            out[(size_t)i * n + j] = v;
            out[(size_t)j * n + i] = v;
        }
}

int icore_ocp_la_products(icore_ocp_la_workspace *w, int stage, const double *P_next, size_t ldP, const double *H,
                 size_t ldH, double reg, double *PD, double *M)
{
    int i, j, status;
    sqrt_recovery_clear(w);
    status = form_products(w, stage, P_next, ldP, H, ldH, reg);
    if (status != ICORE_LA_OK)
        return status;
    if (PD) {
        for (i = 0; i < w->n; ++i)
            for (j = 0; j < w->nx; ++j)
                if (!isfinite(BLASFEO_DMATEL(&w->DTP, i, j)))
                    return ICORE_LA_NONFINITE;
    }
    /* No writes before the complete numerical/argument check succeeds. */
    if (PD)
        blasfeo_unpack_dmat(w->n, w->nx, &w->DTP, 0, 0, PD, w->n);
    if (M)
        export_symmetric(&w->M, w->n, M);
    return ICORE_LA_OK;
}

int icore_ocp_la_factor_stage(icore_ocp_la_workspace *w, int stage, const double *P_next, size_t ldP, const double *H,
                     size_t ldH, double reg, double absolute, double relative, double *L, double *T,
                     double *P)
{
    int i, j, status;
    double maxdiag = 0.0, floor;
    sqrt_recovery_clear(w);
    if (!L || !T || !P || !isfinite(absolute) || !isfinite(relative) || absolute < 0.0 ||
        relative < 0.0)
        return ICORE_LA_INVALID;
    /* Pinned BLASFEO K_MAX_STACK=300: unaligned Schur input beyond that
     * limit would require a disabled upstream heap/reference fallback. */
    if (w && w->nr > 300 && (w->nr & 3))
        return ICORE_LA_UNAVAILABLE;
    status = form_products(w, stage, P_next, ldP, H, ldH, reg);
    if (status != ICORE_LA_OK)
        return status;
    for (i = 0; i < w->nr; ++i) {
        double d = fabs(BLASFEO_DMATEL(&w->M, i, i));
        if (d > maxdiag)
            maxdiag = d;
    }
    floor = fmax(absolute, relative * maxdiag);
    /* Factor only the control columns. The bottom block becomes
     * T=M_xu L^-T; no full-stage SPD assumption is made. */
    blasfeo_dpotrf_l_mn(w->n, w->nr, &w->M, 0, 0, &w->F, 0, 0);
    for (i = 0; i < w->nr; ++i) {
        double d = BLASFEO_DMATEL(&w->F, i, i), pivot;
        if (!isfinite(d))
            return ICORE_LA_NONFINITE;
        if (!(d > 0.0))
            return ICORE_LA_NOT_SPD;
        if (!isfinite(w->F.dA[i]) || !(w->F.dA[i] > 0.0))
            return ICORE_LA_NONFINITE;
        pivot = d * d;
        if (!isfinite(pivot))
            return ICORE_LA_NONFINITE;
        if (!(pivot >= floor))
            return ICORE_OCP_LA_PIVOT_REJECTED;
    }
    for (i = 0; i < w->nx; ++i)
        for (j = 0; j < w->nr; ++j)
            if (!isfinite(BLASFEO_DMATEL(&w->F, w->nr + i, j)))
                return ICORE_LA_NONFINITE;
    blasfeo_dsyrk_ln(w->nx, w->nr, -1.0, &w->F, w->nr, 0, &w->F, w->nr, 0, 1.0, &w->M, w->nr, w->nr,
                     &w->P, 0, 0);
    for (i = 0; i < w->nx; ++i) {
        for (j = 0; j <= i; ++j)
            if (!isfinite(BLASFEO_DMATEL(&w->P, i, j)))
                return ICORE_LA_NONFINITE;
        if (BLASFEO_DMATEL(&w->P, i, i) <
            -1e-8 * fmax(1.0, fabs(BLASFEO_DMATEL(&w->M, w->nr + i, w->nr + i))))
            return ICORE_OCP_LA_SCHUR_REJECTED;
    }
    /* Export only after all checks; fallback may reuse the caller's original
     * arrays without undoing partial writes. Original H/P inputs may alias. */
    for (i = 0; i < w->nr; ++i) {
        for (j = 0; j <= i; ++j)
            L[(size_t)i * w->nr + j] = BLASFEO_DMATEL(&w->F, i, j);
        for (j = i + 1; j < w->nr; ++j)
            L[(size_t)i * w->nr + j] = 0.0;
    }
    blasfeo_unpack_tran_dmat(w->nx, w->nr, &w->F, w->nr, 0, T, w->nr);
    export_symmetric(&w->P, w->nx, P);
    return ICORE_LA_OK;
}

/* ---- Packed square-root factors, finite checks, and recovery. ---- */

static int sqrt_reject(icore_ocp_la_workspace *w, int status)
{
    if (w) {
        w->sqrt_next = -2;
        w->sqrt_initial_zero = 0;
        sqrt_recovery_clear(w);
    }
    return status;
}

/* Only the guarded numerical rejection at a valid pending stage can retain
 * a suffix token. Ordinary apply/checkpoint still see an invalid chain. */
static int sqrt_recoverable_reject(icore_ocp_la_workspace *w, int status, int stage, int initial_fixed,
                                   double reg)
{
    sqrt_reject(w, status);
    if (status == ICORE_OCP_LA_PIVOT_REJECTED) {
        w->sqrt_recovery_stage = stage;
        w->sqrt_recovery_initial_fixed = initial_fixed;
        w->sqrt_recovery_reg = reg;
    }
    return status;
}

static int sqrt_arguments(double reg, double absolute, double relative)
{
    return isfinite(reg) && isfinite(absolute) && isfinite(relative) && reg >= 0.0 &&
           absolute >= 0.0 && relative >= 0.0;
}

static ICORE_OCP_LA_AVX2_TARGET int finite_prefix(const double *values, int count)
{
    int j = 0;
#if ICORE_OCP_LA_SIMD_VALIDATION
    const __m256i exponent = _mm256_set1_epi64x((long long)UINT64_C(0x7ff0000000000000));
    __m256i bad = _mm256_setzero_si256();
    for (; j + 3 < count; j += 4) {
        __m256i bits = _mm256_castpd_si256(_mm256_loadu_pd(values + j));
        bad = _mm256_or_si256(bad, _mm256_cmpeq_epi64(_mm256_and_si256(bits, exponent), exponent));
    }
    if (!_mm256_testz_si256(bad, bad))
        return 0;
#endif
    for (; j < count; ++j)
        if (!isfinite(values[j]))
            return 0;
    return 1;
}

/* Every output element is written before reporting failure. This helper is
 * used only on private scratch, with disjoint inputs or exact in-place alias. */
static ICORE_OCP_LA_AVX2_TARGET int add_finite(double *out, const double *a, const double *b, int count)
{
    int i = 0, finite = 1;
#if ICORE_OCP_LA_SIMD_VALIDATION
    const __m256i exponent = _mm256_set1_epi64x((long long)UINT64_C(0x7ff0000000000000));
    __m256i bad = _mm256_setzero_si256();
    for (; i + 3 < count; i += 4) {
        __m256d sum = _mm256_add_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i));
        __m256i bits = _mm256_castpd_si256(sum);
        _mm256_storeu_pd(out + i, sum);
        bad = _mm256_or_si256(bad, _mm256_cmpeq_epi64(_mm256_and_si256(bits, exponent), exponent));
    }
    finite = _mm256_testz_si256(bad, bad);
#endif
    for (; i < count; ++i) {
        double sum = a[i] + b[i];
        out[i] = sum;
        finite &= isfinite(sum) != 0;
    }
    return finite;
}

#if ICORE_OCP_LA_SIMD_VALIDATION
/* One bit per row of a panel. The prefix j<row is contiguous panel storage;
 * only true lower-triangle lanes in its diagonal block are inspected.
 * Padded rows may be loaded, but their bits are masked and never rejected. */
static ICORE_OCP_LA_AVX2_TARGET unsigned lower_panel_bad_rows(const struct blasfeo_dmat *F, int n, int row)
{
    const double *panel = F->pA + (size_t)row * F->cn;
    const __m256i exponent = _mm256_set1_epi64x((long long)UINT64_C(0x7ff0000000000000));
    __m256i bad = _mm256_setzero_si256();
    int j, a, height = n - row < 4 ? n - row : 4;
    unsigned mask;
    for (j = 0; j < row; ++j) {
        __m256i bits = _mm256_castpd_si256(_mm256_loadu_pd(panel + 4 * (size_t)j));
        bad = _mm256_or_si256(bad, _mm256_cmpeq_epi64(_mm256_and_si256(bits, exponent), exponent));
    }
    mask = (unsigned)_mm256_movemask_pd(_mm256_castsi256_pd(bad));
    mask &= (1u << height) - 1u;
    for (a = 0; a < height; ++a)
        for (j = 0; j <= a; ++j) {
            uint64_t bits;
            memcpy(&bits, panel + 4 * (size_t)(row + j) + a, sizeof(bits));
            if ((bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000))
                mask |= 1u << a;
        }
    return mask;
}
#endif

/* Input is row-major: transposed pack preserves its lower triangle.
 * The upper triangle is allowed to contain NaNs and is never used by the
 * triangular factor kernels. Checking raw rows avoids packed scalar stores. */
static ICORE_OCP_LA_AVX2_TARGET int pack_lower_input(struct blasfeo_dmat *dst, int n, const double *src,
                                            size_t stride, int finite_certified)
{
    int i;
    if (!src || stride < (size_t)n || stride > INT_MAX)
        return ICORE_LA_INVALID;
    if (!finite_certified)
        for (i = 0; i < n; ++i)
            if (!finite_prefix(src + (size_t)i * stride, i + 1))
                return ICORE_LA_NONFINITE;
    blasfeo_pack_tran_dmat(n, n, (double *)src, (int)stride, dst, 0, 0);
    return ICORE_LA_OK;
}

static ICORE_OCP_LA_AVX2_TARGET int pack_lower_checked(struct blasfeo_dmat *dst, int n, const double *src,
                                              size_t stride)
{
    return pack_lower_input(dst, n, src, stride, 0);
}

static int load_cost_input(icore_ocp_la_workspace *w, int stage, const double *H, size_t stride,
                           int finite_certified)
{
    int status, n;
    if (!w || stage < 0 || stage > w->N)
        return ICORE_LA_INVALID;
    n = stage == w->N ? w->nx : w->n;
    /* Public calls validate before writes. Engine-only calls supply the
     * same finite proof from their canonical coefficient load. */
    status = pack_lower_input(w->cost + stage, n, H, stride, finite_certified);
    if (status != ICORE_LA_OK)
        return status;
    w->cost_loaded[stage] = 1;
    mixed_load_cost(w, stage);
    w->checkpoint_valid = 0;
    w->sqrt_next = -2;
    w->sqrt_equilibrated = 0;
    w->sqrt_root_omitted = 0;
    sqrt_recovery_clear(w);
    return ICORE_LA_OK;
}

int icore_ocp_la_load_cost(icore_ocp_la_workspace *w, int stage, const double *H, size_t stride)
{
    return load_cost_input(w, stage, H, stride, 0);
}

int icore_ocp_la_load_cost_finite(icore_ocp_la_workspace *w, int stage, const double *H, size_t stride)
{
    return load_cost_input(w, stage, H, stride, 1);
}

int icore_ocp_la_load_diagonal_cost(icore_ocp_la_workspace *w, int stage, const double *diagonal)
{
    int i, n;
    struct blasfeo_dmat *cost;
    if (!w || stage < 0 || stage > w->N || !diagonal)
        return ICORE_LA_INVALID;
    n = stage == w->N ? w->nx : w->n;
    if (!finite_prefix(diagonal, n))
        return ICORE_LA_NONFINITE;
    /* Stage the complete diagonal before any cost write. This also permits
     * an input that aliases packed storage. The existing vector is scratch
     * after a successful load invalidates the previous factor chain. */
    memmove(w->X.pa, diagonal, (size_t)n * sizeof(double));
    cost = w->cost + stage;
    /* Value 2 certifies only the off-diagonal zeros. Cached factor calls
     * may have replaced every diagonal entry since the previous load. */
    if (w->cost_loaded[stage] != 2)
        blasfeo_dgese(n, n, 0.0, cost, 0, 0);
    for (i = 0; i < n; ++i)
        BLASFEO_DMATEL(cost, i, i) = w->X.pa[i];
    cost->use_dA = 0;
    w->cost_loaded[stage] = 2;
    mixed_load_cost(w, stage);
    w->checkpoint_valid = 0;
    w->sqrt_next = -2;
    w->sqrt_equilibrated = 0;
    w->sqrt_root_omitted = 0;
    sqrt_recovery_clear(w);
    return ICORE_LA_OK;
}

static int prepare_cached_diagonal(icore_ocp_la_workspace *w, int stage, const double *diagonal,
                                   struct blasfeo_dmat *destination)
{
    int i, n = stage == w->N ? w->nx : w->n;
    if (!diagonal)
        return ICORE_LA_INVALID;
    if (!w->cost_loaded[stage])
        return ICORE_LA_NOT_READY;
    if (!finite_prefix(diagonal, n))
        return ICORE_LA_NONFINITE;
    /* Cached off-diagonals are immutable, while every caller supplies the
     * complete current diagonal. Reuse the stage's packed matrix directly;
     * no original-H product may consume this mutable diagonal afterward. */
    if (destination != w->cost + stage)
        blasfeo_dtrcp_l(n, w->cost + stage, 0, 0, destination, 0, 0);
    for (i = 0; i < n; ++i)
        BLASFEO_DMATEL(destination, i, i) = diagonal[i];
    destination->use_dA = 0;
    return ICORE_LA_OK;
}

static ICORE_OCP_LA_AVX2_TARGET int sqrt_check_factor(struct blasfeo_dmat *F, int n, int nr,
                                             double control_floor, double state_floor)
{
    int i;
#if ICORE_OCP_LA_SIMD_VALIDATION
    unsigned bad_rows = 0;
#else
    int j;
#endif
    for (i = 0; i < n; ++i) {
        double d = BLASFEO_DMATEL(F, i, i), pivot;
#if ICORE_OCP_LA_SIMD_VALIDATION
        if (!(i & 3))
            bad_rows = lower_panel_bad_rows(F, n, i);
#endif
        if (!isfinite(d))
            return ICORE_LA_NONFINITE;
        if (!(d > 0.0))
            return ICORE_LA_NOT_SPD;
        if (!isfinite(F->dA[i]) || !(F->dA[i] > 0.0))
            return ICORE_LA_NONFINITE;
        pivot = d * d;
        if (!isfinite(pivot))
            return ICORE_LA_NONFINITE;
        if (!(pivot > 0.0) || pivot < (i < nr ? control_floor : state_floor))
            return ICORE_OCP_LA_PIVOT_REJECTED;
#if ICORE_OCP_LA_SIMD_VALIDATION
        /* Delay the off-diagonal failure until after this same row's
         * diagonal/reciprocal/pivot checks, preserving scalar error order. */
        if (bad_rows & (1u << (i & 3)))
            return ICORE_LA_NONFINITE;
#else
        for (j = 0; j < i; ++j)
            if (!isfinite(BLASFEO_DMATEL(F, i, j)))
                return ICORE_LA_NONFINITE;
#endif
    }
    return ICORE_LA_OK;
}

/* Private certificate for an immediately preceding, complete square
 * POTRF/SYRK-POTRF from the bundled Haswell backend. Every consumed lower
 * entry in row i contributes its square to that row's pivot, including
 * within the 12x4/8x8 blocks and their variable-size tails. A nonfinite
 * entry therefore prevents a positive finite diagonal and reciprocal:
 * the kernels' rejected-pivot branch writes a zero reciprocal, never a
 * positive replacement. Keep every original diagonal/pivot/floor test.
 *
 * This is not a validator for an arbitrary or subsequently modified
 * factor. Partial state cross blocks have no such diagonal certificate;
 * recovery and physical unscaling also retain their complete checks.
 * If any certificate test fails, restart the unchanged row-ordered full
 * validator so its exact status and recovery policy remain authoritative. */
static ICORE_OCP_LA_AVX2_TARGET int sqrt_check_fresh_square_factor(struct blasfeo_dmat *F, int n, int nr,
                                                          double control_floor, double state_floor)
{
#if defined(TARGET_X64_INTEL_HASWELL) && defined(LA_HIGH_PERFORMANCE) && D_PS == 4
    int i = 0;
#if ICORE_OCP_LA_SIMD_VALIDATION
    const __m256i exponent = _mm256_set1_epi64x((long long)UINT64_C(0x7ff0000000000000));
    const __m256i diagonal_offsets = _mm256_set_epi64x(15, 10, 5, 0);
    const __m256d zero = _mm256_setzero_pd();
    for (; i + 3 < n; i += 4) {
        const double *panel = F->pA + (size_t)i * F->cn + 4 * (size_t)i;
        __m256d diagonal = _mm256_i64gather_pd(panel, diagonal_offsets, 8);
        __m256d inverse = _mm256_loadu_pd(F->dA + i);
        __m256i dbits = _mm256_castpd_si256(diagonal), ibits = _mm256_castpd_si256(inverse);
        __m256i bad =
            _mm256_or_si256(_mm256_cmpeq_epi64(_mm256_and_si256(dbits, exponent), exponent),
                            _mm256_cmpeq_epi64(_mm256_and_si256(ibits, exponent), exponent));
        __m256d pivot, floors;
        if (!_mm256_testz_si256(bad, bad) ||
            _mm256_movemask_pd(_mm256_cmp_pd(diagonal, zero, _CMP_GT_OQ)) != 15 ||
            _mm256_movemask_pd(_mm256_cmp_pd(inverse, zero, _CMP_GT_OQ)) != 15)
            return sqrt_check_factor(F, n, nr, control_floor, state_floor);
        pivot = _mm256_mul_pd(diagonal, diagonal);
        dbits = _mm256_castpd_si256(pivot);
        bad = _mm256_cmpeq_epi64(_mm256_and_si256(dbits, exponent), exponent);
        floors = _mm256_set_pd(
            i + 3 < nr ? control_floor : state_floor, i + 2 < nr ? control_floor : state_floor,
            i + 1 < nr ? control_floor : state_floor, i < nr ? control_floor : state_floor);
        if (!_mm256_testz_si256(bad, bad) ||
            _mm256_movemask_pd(_mm256_cmp_pd(pivot, zero, _CMP_GT_OQ)) != 15 ||
            _mm256_movemask_pd(_mm256_cmp_pd(pivot, floors, _CMP_LT_OQ)))
            return sqrt_check_factor(F, n, nr, control_floor, state_floor);
    }
#endif
    for (; i < n; ++i) {
        double d = BLASFEO_DMATEL(F, i, i), pivot;
        if (!isfinite(d) || !(d > 0.0) || !isfinite(F->dA[i]) || !(F->dA[i] > 0.0))
            break;
        pivot = d * d;
        if (!isfinite(pivot) || !(pivot > 0.0) || pivot < (i < nr ? control_floor : state_floor))
            break;
    }
    if (i == n)
        return ICORE_LA_OK;
#endif
    return sqrt_check_factor(F, n, nr, control_floor, state_floor);
}

/* Only the first nr columns exist after a partial factor. In particular,
 * state diagonals and their inverse-diagonal slots must never be read. */
static int sqrt_check_control_factor(struct blasfeo_dmat *F, int n, int nr, double floor)
{
    int i, j, status = sqrt_check_factor(F, nr, nr, floor, floor);
    if (status != ICORE_LA_OK)
        return status;
    for (i = nr; i < n; ++i)
        for (j = 0; j < nr; ++j)
            if (!isfinite(BLASFEO_DMATEL(F, i, j)))
                return ICORE_LA_NONFINITE;
    return ICORE_LA_OK;
}

static int vector_spans_overlap(const double *a, int na, const double *b, int nb);
#include "mixed_factor_impl.h"
#include "mixed_sweep_impl.h"

/* Retry only a positive but undersized pivot in diagonally equilibrated
 * coordinates. H already includes reg exactly once; V is the original D^T*C.
 * The physical matrix remains H+V*V^T: no ridge or pivot clipping is added.
 * scale initially holds its diagonal, collected by the ordinary stage pass. */
static int sqrt_equilibrated_retry(icore_ocp_la_workspace *w, double absolute, double relative,
                                   int initial_fixed, struct blasfeo_dmat *F)
{
    int i, j, status;
    double maximum = 0.0, floor;
    for (i = 0; i < w->n; ++i) {
        double diagonal = w->scale[i];
        if (initial_fixed && i >= w->nr) {
            w->scale[i] = 1.0;
            continue;
        }
        if (!isfinite(diagonal))
            return ICORE_LA_NONFINITE;
        if (!(diagonal > 0.0))
            return ICORE_LA_NOT_SPD;
        w->scale[i] = 1.0 / sqrt(diagonal);
        if (!isfinite(w->scale[i]) || !(w->scale[i] > 0.0))
            return ICORE_LA_NONFINITE;
    }
    for (i = 0; i < w->n; ++i) {
        double diagonal;
        for (j = 0; j <= i; ++j) {
            double v = (BLASFEO_DMATEL(&w->H, i, j) * w->scale[i]) * w->scale[j];
            if (!isfinite(v))
                return ICORE_LA_NONFINITE;
            BLASFEO_DMATEL(&w->H, i, j) = v;
        }
        diagonal = BLASFEO_DMATEL(&w->H, i, i);
        for (j = 0; j < w->nx; ++j) {
            double v = BLASFEO_DMATEL(&w->DTP, i, j) * w->scale[i];
            if (!isfinite(v))
                return ICORE_LA_NONFINITE;
            BLASFEO_DMATEL(&w->DTP, i, j) = v;
            diagonal += v * v;
        }
        if (!isfinite(diagonal))
            return ICORE_LA_NONFINITE;
        if (!initial_fixed || i < w->nr) {
            if (!(diagonal > 0.0))
                return ICORE_LA_NOT_SPD;
            if (diagonal > maximum)
                maximum = diagonal;
        }
    }
    floor = relative * maximum;
    if (!isfinite(floor))
        return ICORE_LA_INVALID;
    w->H.use_dA = 0;
    w->DTP.use_dA = 0;
    if (initial_fixed) {
        blasfeo_dsyrk_dpotrf_ln_mn(w->n, w->nr, w->nx, &w->DTP, 0, 0, &w->DTP, 0, 0, &w->H, 0, 0, F,
                                   0, 0);
        status = sqrt_check_control_factor(F, w->n, w->nr, floor);
    } else {
        blasfeo_dsyrk_dpotrf_ln(w->n, w->nx, &w->DTP, 0, 0, &w->DTP, 0, 0, &w->H, 0, 0, F, 0, 0);
        status = sqrt_check_factor(F, w->n, w->nr, floor, floor);
    }
    if (status != ICORE_LA_OK)
        return status;
    for (i = 0; i < w->n; ++i) {
        int columns = initial_fixed ? w->nr : w->n;
        if (i < columns) {
            double d = BLASFEO_DMATEL(F, i, i) / w->scale[i], pivot, inverse;
            if (!isfinite(d) || !(d > 0.0))
                return ICORE_LA_NONFINITE;
            pivot = d * d;
            inverse = 1.0 / d;
            if (!isfinite(pivot) || !isfinite(inverse) || !(inverse > 0.0))
                return ICORE_LA_NONFINITE;
            if (!(pivot > 0.0) || pivot < absolute)
                return ICORE_OCP_LA_PIVOT_REJECTED;
            F->dA[i] = inverse;
        }
        /* D^-1*F is the Cholesky factor in physical coordinates, including
         * both the control/cross blocks and the propagated state root. */
        for (j = 0; j <= i && j < columns; ++j) {
            double v = BLASFEO_DMATEL(F, i, j) / w->scale[i];
            if (!isfinite(v))
                return ICORE_LA_NONFINITE;
            BLASFEO_DMATEL(F, i, j) = v;
        }
    }
    F->use_dA = 0;
    ++w->sqrt_equilibrated;
    return ICORE_LA_OK;
}

static int sqrt_begin_impl(icore_ocp_la_workspace *w, const double *H_terminal, size_t row_stride,
                           const double *diagonal, double reg, double absolute, double relative)
{
    int i, status;
    double maximum = 0.0, floor;
    struct blasfeo_dmat *H;
    if (!w)
        return ICORE_LA_INVALID;
    w->sqrt_next = -2;
    w->sqrt_equilibrated = 0;
    w->sqrt_root_omitted = 0;
    w->sqrt_initial_zero = 0;
    w->sqrt_terminal_diagonal = 0;
    sqrt_recovery_clear(w);
    if (!sqrt_arguments(reg, absolute, relative))
        return ICORE_LA_INVALID;
    H = diagonal ? w->cost + w->N : &w->PN;
    status = diagonal ? prepare_cached_diagonal(w, w->N, diagonal, H)
                      : pack_lower_checked(&w->PN, w->nx, H_terminal, row_stride);
    if (status != ICORE_LA_OK)
        return status;
    for (i = 0; i < w->nx; ++i) {
        double d = BLASFEO_DMATEL(H, i, i) + reg;
        if (!isfinite(d))
            return ICORE_LA_NONFINITE;
        BLASFEO_DMATEL(H, i, i) = d;
        if (fabs(d) > maximum)
            maximum = fabs(d);
    }
    floor = fmax(absolute, relative * maximum);
    if (!isfinite(floor))
        return ICORE_LA_INVALID;
    H->use_dA = 0;
    if (diagonal && w->cost_loaded[w->N] == 2) {
        /* Exact diagonal structure is certified when the cost is loaded.
         * Preserve the packed kernel's reciprocal-first rounding: its
         * stored diagonal is d*(1/sqrt(d)), not a fresh sqrt(d). */
        struct blasfeo_dmat *F = w->C + w->N;
        int positive = 1;
        for (i = 0; i < w->nx; ++i)
            positive &= BLASFEO_DMATEL(H, i, i) > 0.0;
        if (positive) {
            blasfeo_dgese(w->nx, w->nx, 0.0, F, 0, 0);
            for (i = 0; i < w->nx; ++i) {
                double d = BLASFEO_DMATEL(H, i, i), inverse = 1.0 / sqrt(d);
                F->dA[i] = inverse;
                BLASFEO_DMATEL(F, i, i) = d * inverse;
            }
            F->use_dA = w->nx;
            w->sqrt_terminal_diagonal = 1;
        } else
            blasfeo_dpotrf_l(w->nx, H, 0, 0, F, 0, 0);
    } else
        blasfeo_dpotrf_l(w->nx, H, 0, 0, w->C + w->N, 0, 0);
    status = sqrt_check_fresh_square_factor(w->C + w->N, w->nx, 0, floor, floor);
    if (status != ICORE_LA_OK) {
        w->sqrt_terminal_diagonal = 0;
        return sqrt_recoverable_reject(w, status, w->N, 0, reg);
    }
    w->sqrt_next = w->N - 1;
    return ICORE_LA_OK;
}

/* Multiplication by the terminal diagonal root needs no triangular sum.
 * Both matrices keep their existing packed layout and storage ownership. */
static ICORE_OCP_LA_AVX2_TARGET void terminal_diagonal_product(icore_ocp_la_workspace *w, int stage, int rows)
{
    int i = 0, j;
    struct blasfeo_dmat *D = w->DT + stage, *P = &w->DTP, *F = w->C + w->N;
#if ICORE_OCP_LA_SIMD_VALIDATION
    for (; i + 3 < rows; i += 4) {
        const double *source = D->pA + (size_t)i * D->cn;
        double *target = P->pA + (size_t)i * P->cn;
        for (j = 0; j < w->nx; ++j)
            _mm256_storeu_pd(target + 4 * (size_t)j,
                             _mm256_mul_pd(_mm256_loadu_pd(source + 4 * (size_t)j),
                                           _mm256_set1_pd(BLASFEO_DMATEL(F, j, j))));
    }
#endif
    for (; i < rows; ++i)
        for (j = 0; j < w->nx; ++j)
            BLASFEO_DMATEL(P, i, j) = BLASFEO_DMATEL(D, i, j) * BLASFEO_DMATEL(F, j, j);
}

static ICORE_OCP_LA_AVX2_TARGET void terminal_root_multiply(icore_ocp_la_workspace *w, const double *x, double *y)
{
    int i = 0;
    struct blasfeo_dmat *F = w->C + w->N;
#if ICORE_OCP_LA_SIMD_VALIDATION
    const __m256i offsets = _mm256_set_epi64x(15, 10, 5, 0);
    for (; i + 3 < w->nx; i += 4) {
        const double *panel = F->pA + (size_t)i * F->cn + 4 * (size_t)i;
        __m256d diagonal = _mm256_i64gather_pd(panel, offsets, 8);
        _mm256_storeu_pd(y + i, _mm256_mul_pd(diagonal, _mm256_loadu_pd(x + i)));
    }
#endif
    for (; i < w->nx; ++i)
        y[i] = BLASFEO_DMATEL(F, i, i) * x[i];
}

/* Interleave independent panels without changing any row's column order.
 * The checked-finite initial diagonal receives only squared terms: a NaN
 * or infinite product cannot cancel back to a finite sum. The completed
 * diagonal check therefore also validates every consumed DTP entry.
 * No padded row is read by this path. */
static ICORE_OCP_LA_AVX2_TARGET int sqrt_stage_diagonal_rows(icore_ocp_la_workspace *w, struct blasfeo_dmat *H,
                                                    double reg, int rows, double *control_max,
                                                    double *state_max)
{
    int i = 0, j;
    *control_max = *state_max = 0.0;
#if ICORE_OCP_LA_SIMD_VALIDATION && (!defined(ICORE_OCP_LA_SCALAR_STAGE_SCAN) || !ICORE_OCP_LA_SCALAR_STAGE_SCAN)
    for (; i + 15 < rows; i += 16) {
        double diagonal[16];
        const double *panel0 = w->DTP.pA + (size_t)i * w->DTP.cn;
        const double *panel1 = panel0 + (size_t)4 * w->DTP.cn;
        const double *panel2 = panel0 + (size_t)8 * w->DTP.cn;
        const double *panel3 = panel0 + (size_t)12 * w->DTP.cn;
        __m256d sum0, sum1, sum2, sum3;
        int lane;
        for (lane = 0; lane < 16; ++lane) {
            double d = BLASFEO_DMATEL(H, i + lane, i + lane) + reg;
            if (!isfinite(d))
                return ICORE_LA_NONFINITE;
            BLASFEO_DMATEL(H, i + lane, i + lane) = d;
            diagonal[lane] = d;
        }
        sum0 = _mm256_loadu_pd(diagonal);
        sum1 = _mm256_loadu_pd(diagonal + 4);
        sum2 = _mm256_loadu_pd(diagonal + 8);
        sum3 = _mm256_loadu_pd(diagonal + 12);
        for (j = 0; j < w->nx; ++j) {
            __m256d v0 = _mm256_loadu_pd(panel0 + (size_t)4 * j);
            __m256d v1 = _mm256_loadu_pd(panel1 + (size_t)4 * j);
            __m256d v2 = _mm256_loadu_pd(panel2 + (size_t)4 * j);
            __m256d v3 = _mm256_loadu_pd(panel3 + (size_t)4 * j);
            v0 = _mm256_mul_pd(v0, v0);
            v1 = _mm256_mul_pd(v1, v1);
            v2 = _mm256_mul_pd(v2, v2);
            v3 = _mm256_mul_pd(v3, v3);
            /* C++ SLP may otherwise contract only this four-panel form
             * into FMA. Keep the original separately rounded products;
             * this register-only barrier emits no instruction. */
            __asm__("" : "+x"(v0), "+x"(v1), "+x"(v2), "+x"(v3));
            sum0 = _mm256_add_pd(sum0, v0);
            sum1 = _mm256_add_pd(sum1, v1);
            sum2 = _mm256_add_pd(sum2, v2);
            sum3 = _mm256_add_pd(sum3, v3);
        }
        _mm256_storeu_pd(diagonal, sum0);
        _mm256_storeu_pd(diagonal + 4, sum1);
        _mm256_storeu_pd(diagonal + 8, sum2);
        _mm256_storeu_pd(diagonal + 12, sum3);
        for (lane = 0; lane < 16; ++lane) {
            double d = diagonal[lane];
            if (!isfinite(d))
                return ICORE_LA_NONFINITE;
            w->scale[i + lane] = d;
            if (i + lane < w->nr) {
                if (fabs(d) > *control_max)
                    *control_max = fabs(d);
            } else if (fabs(d) > *state_max)
                *state_max = fabs(d);
        }
    }
    for (; i + 3 < rows; i += 4) {
        double diagonal[4];
        const double *panel = w->DTP.pA + (size_t)i * w->DTP.cn;
        __m256d sum;
        int lane;
        for (lane = 0; lane < 4; ++lane) {
            double d = BLASFEO_DMATEL(H, i + lane, i + lane) + reg;
            if (!isfinite(d))
                return ICORE_LA_NONFINITE;
            BLASFEO_DMATEL(H, i + lane, i + lane) = d;
            diagonal[lane] = d;
        }
        sum = _mm256_loadu_pd(diagonal);
        for (j = 0; j < w->nx; ++j) {
            __m256d value = _mm256_loadu_pd(panel + (size_t)4 * j);
            sum = _mm256_add_pd(sum, _mm256_mul_pd(value, value));
        }
        _mm256_storeu_pd(diagonal, sum);
        for (lane = 0; lane < 4; ++lane) {
            double d = diagonal[lane];
            if (!isfinite(d))
                return ICORE_LA_NONFINITE;
            w->scale[i + lane] = d;
            if (i + lane < w->nr) {
                if (fabs(d) > *control_max)
                    *control_max = fabs(d);
            } else if (fabs(d) > *state_max)
                *state_max = fabs(d);
        }
    }
#endif
    for (; i < rows; ++i) {
        double d = BLASFEO_DMATEL(H, i, i) + reg, diag = d;
        if (!isfinite(d))
            return ICORE_LA_NONFINITE;
        BLASFEO_DMATEL(H, i, i) = d;
        for (j = 0; j < w->nx; ++j) {
            double v = BLASFEO_DMATEL(&w->DTP, i, j);
            /* Retain the scalar remainder's original operation boundary;
             * removing this branch enables C++ tail FMA contraction. */
            if (!isfinite(v))
                return ICORE_LA_NONFINITE;
            diag += v * v;
        }
        if (!isfinite(diag))
            return ICORE_LA_NONFINITE;
        w->scale[i] = diag;
        if (i < w->nr) {
            if (fabs(diag) > *control_max)
                *control_max = fabs(diag);
        } else if (fabs(diag) > *state_max)
            *state_max = fabs(diag);
    }
    return ICORE_LA_OK;
}

static ICORE_OCP_LA_AVX2_TARGET int sqrt_stage_diagonal(icore_ocp_la_workspace *w, struct blasfeo_dmat *H, double reg,
                                               double *control_max, double *state_max)
{
    return sqrt_stage_diagonal_rows(w, H, reg, w->n, control_max, state_max);
}

static int sqrt_stage_impl(icore_ocp_la_workspace *w, int stage, const double *H, size_t row_stride,
                           const double *diagonal, double reg, double absolute, double relative,
                           double *L, double *T, int initial_fixed)
{
    int i, j, status;
    struct blasfeo_dmat *F, *physical_H;
    double control_max = 0.0, state_max = 0.0, control_floor, state_floor;
    sqrt_recovery_clear(w);
    if (!w || stage < 0 || stage >= w->N || ((!L || !T) && !w->owned_factors) ||
        ((L == NULL) != (T == NULL)) || !sqrt_arguments(reg, absolute, relative))
        return sqrt_reject(w, ICORE_LA_INVALID);
    if (w->sqrt_next != stage || !w->loaded[stage])
        return sqrt_reject(w, ICORE_LA_NOT_READY);
    F = w->owned_factors ? w->C + stage : &w->F;
    physical_H = diagonal ? w->cost + stage : &w->H;
    status = diagonal ? prepare_cached_diagonal(w, stage, diagonal, physical_H)
                      : pack_lower_checked(&w->H, w->n, H, row_stride);
    if (status != ICORE_LA_OK)
        return sqrt_reject(w, status);
    /* BLASFEO puts the triangular operand first for this TRMM API. */
    if (stage == w->N - 1 && w->sqrt_terminal_diagonal)
        terminal_diagonal_product(w, stage, w->n);
    else
        blasfeo_dtrmm_rlnn(w->n, w->nx, 1.0, w->C + stage + 1, root_offset(w, stage + 1),
                           root_offset(w, stage + 1), w->DT + stage, 0, 0, &w->DTP, 0, 0);
    status = sqrt_stage_diagonal(w, physical_H, reg, &control_max, &state_max);
    if (status != ICORE_LA_OK)
        return sqrt_reject(w, status);
    control_floor = fmax(absolute, relative * control_max);
    state_floor = initial_fixed ? 0.0 : fmax(absolute, relative * state_max);
    if (!isfinite(control_floor) || !isfinite(state_floor))
        return sqrt_reject(w, ICORE_LA_INVALID);
    physical_H->use_dA = 0;
    if (initial_fixed) {
        /* Zero-offset partial fused factor has no K_MAX_STACK/heap branch,
         * including nr>300 when nr is not a multiple of the panel size. */
        blasfeo_dsyrk_dpotrf_ln_mn(w->n, w->nr, w->nx, &w->DTP, 0, 0, &w->DTP, 0, 0, physical_H, 0,
                                   0, F, 0, 0);
        status = sqrt_check_control_factor(F, w->n, w->nr, control_floor);
    } else {
        blasfeo_dsyrk_dpotrf_ln(w->n, w->nx, &w->DTP, 0, 0, &w->DTP, 0, 0, physical_H, 0, 0, F, 0,
                                0);
        status = sqrt_check_fresh_square_factor(F, w->n, w->nr, control_floor, state_floor);
    }
    if (status == ICORE_OCP_LA_PIVOT_REJECTED) {
        /* Equilibration modifies off-diagonals. Preserve the cached cost
         * by copying this physical matrix, with reg already included once,
         * to the existing recovery scratch only on the rejection path. */
        if (physical_H != &w->H)
            blasfeo_dtrcp_l(w->n, physical_H, 0, 0, &w->H, 0, 0);
        status = sqrt_equilibrated_retry(w, absolute, relative, initial_fixed, F);
    }
    if (status != ICORE_LA_OK)
        return sqrt_recoverable_reject(w, status, stage, initial_fixed, reg);
    /* Retain the state root; no Schur matrix construction or P export. */
    if (!initial_fixed && !w->owned_factors)
        blasfeo_dtrcp_l(w->nx, F, w->nr, w->nr, w->C + stage, 0, 0);
    if (L)
        for (i = 0; i < w->nr; ++i) {
            for (j = 0; j <= i; ++j)
                L[(size_t)i * w->nr + j] = BLASFEO_DMATEL(F, i, j);
            for (j = i + 1; j < w->nr; ++j)
                L[(size_t)i * w->nr + j] = 0.0;
        }
    if (T)
        blasfeo_unpack_tran_dmat(w->nx, w->nr, F, w->nr, 0, T, w->nr);
    w->sqrt_next = stage - 1;
    if (stage == 0) {
        w->sqrt_root_omitted = initial_fixed;
        w->sqrt_initial_zero = 0;
    }
    return ICORE_LA_OK;
}

/* The initial Newton direction has x0=0. Its state cross block is neither
 * needed by the control solve nor propagated to an earlier stage. Restrict
 * this opt-in factor to canonical owned storage and commit a separate token
 * only after validating the complete control factor. All original model
 * inputs retain their ordinary dimensions and finite-validation contract.
 * A numerical rejection restarts the full established initial-fixed stage,
 * from the unchanged inputs, before its equilibration or local QR recovery.
 * Thus a rejected short factor never masquerades as a complete L0/T0 pair. */
static int sqrt_initial_zero_impl(icore_ocp_la_workspace *w, const double *H, size_t row_stride,
                                  const double *diagonal, double reg, double absolute,
                                  double relative)
{
    int status;
    double control_max, state_max, floor;
    struct blasfeo_dmat *physical_H, *F;
    sqrt_recovery_clear(w);
    if (!w || !sqrt_arguments(reg, absolute, relative))
        return sqrt_reject(w, ICORE_LA_INVALID);
    if (!w->owned_factors)
        return sqrt_reject(w, ICORE_LA_UNAVAILABLE);
    if (w->sqrt_next != 0 || !w->loaded[0])
        return sqrt_reject(w, ICORE_LA_NOT_READY);
    physical_H = diagonal ? w->cost : &w->H;
    status = diagonal ? prepare_cached_diagonal(w, 0, diagonal, physical_H)
                      : pack_lower_checked(&w->H, w->n, H, row_stride);
    if (status != ICORE_LA_OK)
        return sqrt_reject(w, status);
    if (w->N == 1 && w->sqrt_terminal_diagonal)
        terminal_diagonal_product(w, 0, w->nr);
    else
        blasfeo_dtrmm_rlnn(w->nr, w->nx, 1.0, w->C + 1, root_offset(w, 1), root_offset(w, 1), w->DT,
                           0, 0, &w->DTP, 0, 0);
    status = sqrt_stage_diagonal_rows(w, physical_H, reg, w->nr, &control_max, &state_max);
    if (status != ICORE_LA_OK)
        return sqrt_reject(w, status);
    floor = fmax(absolute, relative * control_max);
    if (!isfinite(floor))
        return sqrt_reject(w, ICORE_LA_INVALID);
    F = w->C;
    blasfeo_dsyrk_dpotrf_ln_mn(w->nr, w->nr, w->nx, &w->DTP, 0, 0, &w->DTP, 0, 0, physical_H, 0, 0,
                               F, 0, 0);
    status = sqrt_check_fresh_square_factor(F, w->nr, w->nr, floor, floor);
    if (status != ICORE_LA_OK)
        return sqrt_stage_impl(w, 0, H, row_stride, diagonal, reg, absolute, relative, NULL, NULL,
                               1);
    w->sqrt_next = -1;
    w->sqrt_root_omitted = 1;
    w->sqrt_initial_zero = 1;
    return ICORE_LA_OK;
}

int icore_ocp_la_sqrt_initial_zero(icore_ocp_la_workspace *w, const double *H, size_t row_stride, double reg,
                          double absolute, double relative)
{
    return sqrt_initial_zero_impl(w, H, row_stride, NULL, reg, absolute, relative);
}

int icore_ocp_la_sqrt_initial_zero_cached_diagonal(icore_ocp_la_workspace *w, const double *diagonal, double reg,
                                          double absolute, double relative)
{
    if (!diagonal)
        return sqrt_reject(w, ICORE_LA_INVALID);
    return sqrt_initial_zero_impl(w, NULL, 0, diagonal, reg, absolute, relative);
}

int icore_ocp_la_sqrt_initial_zero_active(const icore_ocp_la_workspace *w)
{
    return w && w->owned_factors && w->sqrt_next == -1 && w->sqrt_initial_zero;
}

int icore_ocp_la_sqrt_begin(icore_ocp_la_workspace *w, const double *H_terminal, size_t row_stride, double reg,
                   double absolute, double relative)
{
    return sqrt_begin_impl(w, H_terminal, row_stride, NULL, reg, absolute, relative);
}

int icore_ocp_la_sqrt_stage(icore_ocp_la_workspace *w, int stage, const double *H, size_t row_stride, double reg,
                   double absolute, double relative, double *L, double *T)
{
    return sqrt_stage_impl(w, stage, H, row_stride, NULL, reg, absolute, relative, L, T, 0);
}

int icore_ocp_la_sqrt_initial_fixed(icore_ocp_la_workspace *w, const double *H, size_t row_stride, double reg,
                           double absolute, double relative, double *L, double *T)
{
    return sqrt_stage_impl(w, 0, H, row_stride, NULL, reg, absolute, relative, L, T, 1);
}

int icore_ocp_la_sqrt_begin_cached_diagonal(icore_ocp_la_workspace *w, const double *diagonal, double reg,
                                   double absolute, double relative)
{
    return sqrt_begin_impl(w, NULL, 0, diagonal, reg, absolute, relative);
}

int icore_ocp_la_sqrt_stage_cached_diagonal(icore_ocp_la_workspace *w, int stage, const double *diagonal, double reg,
                                   double absolute, double relative, double *L, double *T)
{
    if (!diagonal)
        return sqrt_reject(w, ICORE_LA_INVALID);
    return sqrt_stage_impl(w, stage, NULL, 0, diagonal, reg, absolute, relative, L, T, 0);
}

int icore_ocp_la_sqrt_initial_fixed_cached_diagonal(icore_ocp_la_workspace *w, const double *diagonal, double reg,
                                           double absolute, double relative, double *L, double *T)
{
    if (!diagonal)
        return sqrt_reject(w, ICORE_LA_INVALID);
    return sqrt_stage_impl(w, 0, NULL, 0, diagonal, reg, absolute, relative, L, T, 1);
}

int icore_ocp_la_sqrt_equilibrated_stages(const icore_ocp_la_workspace *w)
{
    return w ? w->sqrt_equilibrated : 0;
}

int icore_ocp_la_sqrt_recovery_pending(const icore_ocp_la_workspace *w, int stage, int initial_fixed, double reg)
{
    if (!w || stage < 0 || stage > w->N || (initial_fixed != 0 && initial_fixed != 1) ||
        (initial_fixed && stage != 0) || !isfinite(reg) || reg < 0.0)
        return ICORE_LA_INVALID;
    if (w->sqrt_next != -2 || w->sqrt_recovery_stage != stage ||
        w->sqrt_recovery_initial_fixed != initial_fixed || w->sqrt_recovery_reg != reg)
        return ICORE_LA_NOT_READY;
    return ICORE_LA_OK;
}

int icore_ocp_la_sqrt_recovery_product(icore_ocp_la_workspace *w, int stage, double *RD, size_t row_stride)
{
    int i, j;
    if (!w || stage < 0 || stage >= w->N || !RD || row_stride < (size_t)w->n ||
        row_stride > INT_MAX || (size_t)w->nx > SIZE_MAX / sizeof(double) / row_stride)
        return sqrt_reject(w, ICORE_LA_INVALID);
    if (w->sqrt_next != -2 || w->sqrt_recovery_stage != stage || !w->loaded[stage])
        return sqrt_reject(w, ICORE_LA_NOT_READY);
    /* A failed equilibrated factor scaled H/DTP scratch. Recompute the
     * original propagation block from unchanged physical C_next and D. */
    blasfeo_dtrmm_rlnn(w->n, w->nx, 1.0, w->C + stage + 1, root_offset(w, stage + 1),
                       root_offset(w, stage + 1), w->DT + stage, 0, 0, &w->DTP, 0, 0);
    for (i = 0; i < w->n; ++i)
        for (j = 0; j < w->nx; ++j)
            if (!isfinite(BLASFEO_DMATEL(&w->DTP, i, j)))
                return sqrt_reject(w, ICORE_LA_NONFINITE);
    /* Packed DTP is n*nx; its column-major export is row-major C_next^T D. */
    blasfeo_unpack_dmat(w->n, w->nx, &w->DTP, 0, 0, RD, (int)row_stride);
    return ICORE_LA_OK;
}

int icore_ocp_la_sqrt_recovery_accept(icore_ocp_la_workspace *w, int stage, int initial_fixed, double reg,
                             const double *R, size_t row_stride, double *L, double *T)
{
    int i, j, n, terminal, status;
    struct blasfeo_dmat *F;
    status = icore_ocp_la_sqrt_recovery_pending(w, stage, initial_fixed, reg);
    if (status != ICORE_LA_OK)
        return sqrt_reject(w, status);
    terminal = stage == w->N;
    n = terminal ? w->nx : w->n;
    if (!R || row_stride < (size_t)n || row_stride > INT_MAX ||
        (size_t)n > SIZE_MAX / sizeof(double) / row_stride ||
        (!terminal && ((!L || !T) && !w->owned_factors)) ||
        (!terminal && ((L == NULL) != (T == NULL))) || (terminal && (L || T)))
        return sqrt_reject(w, ICORE_LA_INVALID);
    /* This is the existing robust QR acceptance, not a claim that the
     * rejected Cholesky pivot passed its unchanged relative floor. */
    for (i = 0; i < n; ++i) {
        double d = R[(size_t)i * row_stride + i];
        if (!isfinite(d))
            return sqrt_reject(w, ICORE_LA_NONFINITE);
        if (!(d > 0.0))
            return sqrt_reject(w, ICORE_LA_NOT_SPD);
        for (j = i + 1; j < n; ++j)
            if (!isfinite(R[(size_t)i * row_stride + j]))
                return sqrt_reject(w, ICORE_LA_NONFINITE);
    }
    /* Copy all input before committing outputs, so R may alias L/T.
     * The row-major upper R becomes the physical lower R^T in F. */
    F = w->owned_factors && !terminal ? w->C + stage : &w->F;
    blasfeo_pack_dmat(n, n, (double *)R, (int)row_stride, F, 0, 0);
    F->use_dA = 0;
    if (terminal) {
        blasfeo_dtrcp_l(w->nx, F, 0, 0, w->C + stage, 0, 0);
        w->C[stage].use_dA = 0;
    } else {
        if (!initial_fixed && !w->owned_factors) {
            blasfeo_dtrcp_l(w->nx, F, w->nr, w->nr, w->C + stage, 0, 0);
            w->C[stage].use_dA = 0;
        }
        if (L)
            for (i = 0; i < w->nr; ++i) {
                for (j = 0; j <= i; ++j)
                    L[(size_t)i * w->nr + j] = BLASFEO_DMATEL(F, i, j);
                for (j = i + 1; j < w->nr; ++j)
                    L[(size_t)i * w->nr + j] = 0.0;
            }
        if (T)
            blasfeo_unpack_tran_dmat(w->nx, w->nr, F, w->nr, 0, T, w->nr);
    }
    w->sqrt_next = stage - 1;
    if (stage == 0) {
        w->sqrt_root_omitted = initial_fixed;
        w->sqrt_initial_zero = 0;
    }
    sqrt_recovery_clear(w);
    return ICORE_LA_OK;
}

/* ---- Packed matrix-vector operations, sweeps, and factor checkpoints. ---- */

int icore_ocp_la_sqrt_apply(icore_ocp_la_workspace *w, int stage, const double *x, double *y)
{
    if (!w || stage < 0 || stage > w->N || !x || !y)
        return ICORE_LA_INVALID;
    if (w->sqrt_next != -1)
        return ICORE_LA_NOT_READY;
    if (stage == 0 && w->sqrt_root_omitted)
        return ICORE_LA_NOT_READY;
    if (!finite_prefix(x, w->nx))
        return ICORE_LA_NONFINITE;
    memcpy(w->X.pa, x, (size_t)w->nx * sizeof(double));
    if (stage == w->N && w->sqrt_terminal_diagonal)
        terminal_root_multiply(w, w->X.pa, w->Y.pa);
    else
        blasfeo_dtrmv_ltn(w->nx, w->C + stage, root_offset(w, stage), root_offset(w, stage), &w->X,
                          0, &w->Y, 0);
    if (!finite_prefix(w->Y.pa, w->nx))
        return ICORE_LA_NONFINITE;
    if (stage == w->N && w->sqrt_terminal_diagonal)
        terminal_root_multiply(w, w->Y.pa, w->Z.pa);
    else
        blasfeo_dtrmv_lnn(w->nx, w->C + stage, root_offset(w, stage), root_offset(w, stage), &w->Y,
                          0, &w->Z, 0);
    if (!finite_prefix(w->Z.pa, w->nx))
        return ICORE_LA_NONFINITE;
    memcpy(y, w->Z.pa, (size_t)w->nx * sizeof(double));
    return ICORE_LA_OK;
}

int icore_ocp_la_dynamics_product(icore_ocp_la_workspace *w, int stage, int transpose, double alpha, const double *x,
                         double beta, double *y)
{
    int input, output;
    if (!w || stage < 0 || stage >= w->N || !x || !y || (transpose != 0 && transpose != 1) ||
        !isfinite(alpha) || !isfinite(beta))
        return ICORE_LA_INVALID;
    if (!w->loaded[stage])
        return ICORE_LA_NOT_READY;
    input = transpose ? w->nx : w->n;
    output = transpose ? w->n : w->nx;
    if (!finite_prefix(x, input) || (beta != 0.0 && !finite_prefix(y, output)))
        return ICORE_LA_NONFINITE;
    /* Copy both inputs before calculation: aliases and failed products never
     * leave a partial result in the caller's vector. DT is the original,
     * immutable dynamics matrix, independent of current factor state. */
    memcpy(w->X.pa, x, (size_t)input * sizeof(double));
    if (beta != 0.0)
        memcpy(w->Y.pa, y, (size_t)output * sizeof(double));
    else
        memset(w->Y.pa, 0, (size_t)output * sizeof(double));
    if (transpose)
        blasfeo_dgemv_n(w->n, w->nx, alpha, w->DT + stage, 0, 0, &w->X, 0, beta, &w->Y, 0, &w->Z,
                        0);
    else
        blasfeo_dgemv_t(w->n, w->nx, alpha, w->DT + stage, 0, 0, &w->X, 0, beta, &w->Y, 0, &w->Z,
                        0);
    if (!finite_prefix(w->Z.pa, output))
        return ICORE_LA_NONFINITE;
    memcpy(y, w->Z.pa, (size_t)output * sizeof(double));
    return ICORE_LA_OK;
}

/* The residual needs both original dynamics products. Keep independent
 * input/output validation and commit both outputs only after both are finite.
 * DT is immutable factor-independent storage, so one GEMV_NT traverses it
 * for gradient+=D' dual and dynamics=D primal together. */
int icore_ocp_la_dynamics_pair(icore_ocp_la_workspace *w, int stage, const double *primal, const double *dual,
                      double *gradient, double *dynamics)
{
    uintptr_t g, d;
    size_t gb, db;
    if (!w || stage < 0 || stage >= w->N || !primal || !dual || !gradient || !dynamics)
        return ICORE_LA_INVALID;
    if (!w->loaded[stage])
        return ICORE_LA_NOT_READY;
    g = (uintptr_t)gradient;
    d = (uintptr_t)dynamics;
    gb = (size_t)w->n * sizeof(double);
    db = (size_t)w->nx * sizeof(double);
    if ((g <= d && d - g < gb) || (d < g && g - d < db))
        return ICORE_LA_INVALID;
    if (!finite_prefix(primal, w->n) || !finite_prefix(dual, w->nx) ||
        !finite_prefix(gradient, w->n))
        return ICORE_LA_NONFINITE;
    memset(w->V.pa, 0, (size_t)w->nx * sizeof(double));
#if defined(TARGET_X64_INTEL_HASWELL) && defined(LA_HIGH_PERFORMANCE) && D_PS == 4
    {
        /* The pinned GEMV_NT kernels only read logical input elements,
         * including non-panel tails. Descriptors are call-local views;
         * both results still use private storage until the joint commit. */
        struct blasfeo_dvec dual_view = w->X, primal_view = w->Y, gradient_view = w->Z;
        dual_view.pa = (double *)dual;
        primal_view.pa = (double *)primal;
        gradient_view.pa = gradient;
        blasfeo_dgemv_nt(w->n, w->nx, 1.0, 1.0, w->DT + stage, 0, 0, &dual_view, 0, &primal_view, 0,
                         1.0, 0.0, &gradient_view, 0, &w->V, 0, &w->Z, 0, &w->V, 0);
    }
#else
    memcpy(w->X.pa, dual, (size_t)w->nx * sizeof(double));
    memcpy(w->Y.pa, primal, (size_t)w->n * sizeof(double));
    memcpy(w->Z.pa, gradient, (size_t)w->n * sizeof(double));
    blasfeo_dgemv_nt(w->n, w->nx, 1.0, 1.0, w->DT + stage, 0, 0, &w->X, 0, &w->Y, 0, 1.0, 0.0,
                     &w->Z, 0, &w->V, 0, &w->Z, 0, &w->V, 0);
#endif
    if (!finite_prefix(w->Z.pa, w->n) || !finite_prefix(w->V.pa, w->nx))
        return ICORE_LA_NONFINITE;
    memcpy(gradient, w->Z.pa, (size_t)w->n * sizeof(double));
    memcpy(dynamics, w->V.pa, (size_t)w->nx * sizeof(double));
    return ICORE_LA_OK;
}

int icore_ocp_la_dynamics_cache_available(void)
{
    return ICORE_OCP_LA_DYNAMICS_CACHE;
}

#if ICORE_OCP_LA_DYNAMICS_CACHE
static inline ICORE_OCP_LA_FMA_TARGET __attribute__((always_inline)) void
dynamics_nt_add_panels(const struct blasfeo_dmat *A, int nx, int columns, const double *dual,
                       const double *gradient, double *out, int first, int panels, __m256d one)
{
    __m256d sums[8];
    int j, p;
    for (p = 0; p < panels; ++p)
        sums[p] = _mm256_mul_pd(one, _mm256_loadu_pd(gradient + first + 4 * p));
    for (j = 0; j < nx; ++j) {
        __m256d x = _mm256_mul_pd(one, _mm256_set1_pd(dual[j]));
        for (p = 0; p < panels; ++p)
            sums[p] = _mm256_fmadd_pd(
                _mm256_load_pd(A->pA + (size_t)(first + 4 * p) * A->cn + 4 * j), x, sums[p]);
    }
    for (; j < columns; ++j) {
        __m256d zero = _mm256_setzero_pd();
        for (p = 0; p < panels; ++p)
            sums[p] = _mm256_fmadd_pd(
                _mm256_load_pd(A->pA + (size_t)(first + 4 * p) * A->cn + 4 * j), zero, sums[p]);
    }
    for (p = 0; p < panels; ++p)
        _mm256_storeu_pd(out + first + 4 * p, sums[p]);
}

/* GEMV_NT seeds each gradient entry before its ascending-column FMAs.
 * Ordinary GEMV_N computes a dot first, so it is not interchangeable here.
 * Independent row panels hide the dependency latency without regrouping
 * any row. Keep the backend's multiply-by-one boundaries for FTZ/DAZ. */
static ICORE_OCP_LA_FMA_TARGET void dynamics_nt_add_kernel(icore_ocp_la_workspace *w, int stage, const double *dual,
                                                  const double *gradient)
{
    const struct blasfeo_dmat *A = w->DT + stage;
    double *out = w->Z.pa;
    int i = 0, j, n = w->n, nx = w->nx, remainder = nx % 6, columns = nx;
    __m256d one = _mm256_set1_pd(1.0);
    __asm__("" : "+x"(one));
    /* The pinned NT tail always executes four column FMAs, zeroing the
     * missing multipliers. Its packed-matrix reads are owned storage and
     * must remain present: adding signed zero can change a zero result. */
    if (remainder > 0 && remainder < 4)
        columns += 4 - remainder;
    else if (remainder == 5)
        columns += 3;
    for (; n - i >= 32; i += 32)
        dynamics_nt_add_panels(A, nx, columns, dual, gradient, out, i, 8, one);
    if (n - i >= 16) {
        dynamics_nt_add_panels(A, nx, columns, dual, gradient, out, i, 4, one);
        i += 16;
    }
    if (n - i >= 8) {
        dynamics_nt_add_panels(A, nx, columns, dual, gradient, out, i, 2, one);
        i += 8;
    }
    if (n - i >= 4) {
        dynamics_nt_add_panels(A, nx, columns, dual, gradient, out, i, 1, one);
        i += 4;
    }
    if (i < n) {
        __m256i mask = _mm256_set_epi64x(0, n - i > 2 ? -1LL : 0, n - i > 1 ? -1LL : 0, -1LL);
        __m256d sum = _mm256_mul_pd(one, _mm256_maskload_pd(gradient + i, mask));
        for (j = 0; j < nx; ++j) {
            __m256d x = _mm256_mul_pd(one, _mm256_set1_pd(dual[j]));
            sum = _mm256_fmadd_pd(_mm256_maskload_pd(A->pA + (size_t)i * A->cn + 4 * j, mask), x,
                                  sum);
        }
        for (; j < columns; ++j)
            sum = _mm256_fmadd_pd(_mm256_maskload_pd(A->pA + (size_t)i * A->cn + 4 * j, mask),
                                  _mm256_setzero_pd(), sum);
        _mm256_maskstore_pd(out + i, mask, sum);
    }
}
#endif

int icore_ocp_la_dynamics_nt_add(icore_ocp_la_workspace *w, int stage, const double *dual, double *gradient)
{
    if (!w || stage < 0 || stage >= w->N || !dual || !gradient)
        return ICORE_LA_INVALID;
    if (!w->loaded[stage])
        return ICORE_LA_NOT_READY;
#if ICORE_OCP_LA_DYNAMICS_CACHE
    if (!finite_prefix(dual, w->nx) || !finite_prefix(gradient, w->n))
        return ICORE_LA_NONFINITE;
    dynamics_nt_add_kernel(w, stage, dual, gradient);
    if (!finite_prefix(w->Z.pa, w->n))
        return ICORE_LA_NONFINITE;
    memcpy(gradient, w->Z.pa, (size_t)w->n * sizeof(double));
    return ICORE_LA_OK;
#else
    return ICORE_LA_UNAVAILABLE;
#endif
}

int icore_ocp_la_sweep_backward(icore_ocp_la_workspace *w, int stage, const double *q_u, const double *q_x, double *a,
                       double *p)
{
    if (!w || stage < 0 || stage >= w->N || !q_u || !q_x || !a || !p)
        return ICORE_LA_INVALID;
    if (!w->owned_factors || w->sqrt_next != -1)
        return ICORE_LA_NOT_READY;
    if (stage == 0 && w->sqrt_initial_zero)
        return ICORE_LA_NOT_READY;
    if (!finite_prefix(q_u, w->nr) || !finite_prefix(q_x, w->nx))
        return ICORE_LA_NONFINITE;
    memcpy(w->X.pa, q_u, (size_t)w->nr * sizeof(double));
    memcpy(w->X.pa + w->nr, q_x, (size_t)w->nx * sizeof(double));
    blasfeo_dtrsv_lnn_mn(w->n, w->nr, w->C + stage, 0, 0, &w->X, 0, &w->Y, 0);
    if (!finite_prefix(w->Y.pa, w->n))
        return ICORE_LA_NONFINITE;
    memcpy(a, w->Y.pa, (size_t)w->nr * sizeof(double));
    memcpy(p, w->Y.pa + w->nr, (size_t)w->nx * sizeof(double));
    return ICORE_LA_OK;
}

int icore_ocp_la_sweep_forward(icore_ocp_la_workspace *w, int stage, const double *q_u, const double *x, double *u)
{
    if (!w || stage < 0 || stage >= w->N || !q_u || !x || !u)
        return ICORE_LA_INVALID;
    if (!w->owned_factors || w->sqrt_next != -1)
        return ICORE_LA_NOT_READY;
    if (stage == 0 && w->sqrt_initial_zero)
        return ICORE_LA_NOT_READY;
    if (!finite_prefix(q_u, w->nr) || !finite_prefix(x, w->nx))
        return ICORE_LA_NONFINITE;
    memcpy(w->X.pa, q_u, (size_t)w->nr * sizeof(double));
    memcpy(w->X.pa + w->nr, x, (size_t)w->nx * sizeof(double));
    blasfeo_dtrsv_ltn_mn(w->n, w->nr, w->C + stage, 0, 0, &w->X, 0, &w->Y, 0);
    if (!finite_prefix(w->Y.pa, w->nr))
        return ICORE_LA_NONFINITE;
    memcpy(u, w->Y.pa, (size_t)w->nr * sizeof(double));
    return ICORE_LA_OK;
}

int icore_ocp_la_initial_zero_sweep(icore_ocp_la_workspace *w, int transpose, const double *q, double *out)
{
    if (!w || !q || !out || (transpose != 0 && transpose != 1))
        return ICORE_LA_INVALID;
    if (!icore_ocp_la_sqrt_initial_zero_active(w))
        return ICORE_LA_NOT_READY;
    if (!finite_prefix(q, w->nr))
        return ICORE_LA_NONFINITE;
    memcpy(w->X.pa, q, (size_t)w->nr * sizeof(double));
    if (transpose)
        blasfeo_dtrsv_ltn_mn(w->nr, w->nr, w->C, 0, 0, &w->X, 0, &w->Y, 0);
    else
        blasfeo_dtrsv_lnn_mn(w->nr, w->nr, w->C, 0, 0, &w->X, 0, &w->Y, 0);
    if (!finite_prefix(w->Y.pa, w->nr))
        return ICORE_LA_NONFINITE;
    memcpy(out, w->Y.pa, (size_t)w->nr * sizeof(double));
    return ICORE_LA_OK;
}

int icore_ocp_la_initial_zero_dynamics_product(icore_ocp_la_workspace *w, int transpose, double alpha,
                                      const double *x, double beta, double *y)
{
    int input, output;
    if (!w || !x || !y || (transpose != 0 && transpose != 1) || !isfinite(alpha) || !isfinite(beta))
        return ICORE_LA_INVALID;
    if (!icore_ocp_la_sqrt_initial_zero_active(w))
        return ICORE_LA_NOT_READY;
    input = transpose ? w->nx : w->nr;
    output = transpose ? w->nr : w->nx;
    if (!finite_prefix(x, input) || (beta != 0.0 && !finite_prefix(y, output)))
        return ICORE_LA_NONFINITE;
    memcpy(w->X.pa, x, (size_t)input * sizeof(double));
    if (beta != 0.0)
        memcpy(w->Y.pa, y, (size_t)output * sizeof(double));
    else
        memset(w->Y.pa, 0, (size_t)output * sizeof(double));
    if (transpose)
        blasfeo_dgemv_n(w->nr, w->nx, alpha, w->DT, 0, 0, &w->X, 0, beta, &w->Y, 0, &w->Z, 0);
    else
        blasfeo_dgemv_t(w->nr, w->nx, alpha, w->DT, 0, 0, &w->X, 0, beta, &w->Y, 0, &w->Z, 0);
    if (!finite_prefix(w->Z.pa, output))
        return ICORE_LA_NONFINITE;
    memcpy(y, w->Z.pa, (size_t)output * sizeof(double));
    return ICORE_LA_OK;
}

static int vector_spans_overlap(const double *a, int na, const double *b, int nb)
{
    uintptr_t pa = (uintptr_t)a, pb = (uintptr_t)b;
    size_t sa = (size_t)na * sizeof(double), sb = (size_t)nb * sizeof(double);
    return (pa <= pb && pb - pa < sa) || (pb < pa && pa - pb < sb);
}

static int packed_backward_stage_impl(icore_ocp_la_workspace *w, int stage, const double *bd,
                                      const double *d, const double *pnext, int reuse_pd,
                                      double *pd, double *a, double *p, int suffix_ready)
{
    int i, rows, zero, offset;
    if (!w || stage < 0 || stage >= w->N || !bd || !pnext || !a ||
        (reuse_pd != 0 && reuse_pd != 1) || (reuse_pd ? !pd : !d))
        return ICORE_LA_INVALID;
    if (!w->owned_factors || !w->loaded[stage] || w->sqrt_next != (suffix_ready ? stage - 1 : -1))
        return ICORE_LA_NOT_READY;
    zero = stage == 0 && w->sqrt_initial_zero;
    if (zero ? p != NULL : p == NULL)
        return ICORE_LA_INVALID;
    rows = zero ? w->nr : w->n;
    if ((!zero && vector_spans_overlap(a, w->nr, p, w->nx)) ||
        (pd && (vector_spans_overlap(a, w->nr, pd, w->nx) ||
                (!zero && vector_spans_overlap(p, w->nx, pd, w->nx)))))
        return ICORE_LA_INVALID;
    if (reuse_pd) {
        if (!add_finite(w->X.pa, pnext, pd, w->nx))
            return ICORE_LA_NONFINITE;
    } else {
        if (!finite_prefix(d, w->nx))
            return ICORE_LA_NONFINITE;
        memcpy(w->X.pa, d, (size_t)w->nx * sizeof(double));
        offset = root_offset(w, stage + 1);
        if (stage == w->N - 1 && w->sqrt_terminal_diagonal) {
            terminal_root_multiply(w, w->X.pa, w->Y.pa);
            terminal_root_multiply(w, w->Y.pa, w->V.pa);
        } else {
            blasfeo_dtrmv_ltn(w->nx, w->C + stage + 1, offset, offset, &w->X, 0, &w->Y, 0);
            blasfeo_dtrmv_lnn(w->nx, w->C + stage + 1, offset, offset, &w->Y, 0, &w->V, 0);
        }
        /* The accepted triangular root has a finite nonzero diagonal, so
         * a nonfinite Y cannot produce a finite V. The finite sum below
         * also certifies both V and pnext. V retains Pd until commit. */
        if (!add_finite(w->X.pa, pnext, w->V.pa, w->nx))
            return ICORE_LA_NONFINITE;
    }
    if (!finite_prefix(bd, rows))
        return ICORE_LA_NONFINITE;
    for (i = 0; i < rows; ++i)
        w->Z.pa[i] = -bd[i];
    blasfeo_dgemv_n(rows, w->nx, 1.0, w->DT + stage, 0, 0, &w->X, 0, 1.0, &w->Z, 0, &w->Y, 0);
    if (!finite_prefix(w->Y.pa, rows))
        return ICORE_LA_NONFINITE;
    /* Pinned BLASFEO's triangular solve explicitly supports x==z. */
    blasfeo_dtrsv_lnn_mn(rows, w->nr, w->C + stage, 0, 0, &w->Y, 0, &w->Y, 0);
    if (!finite_prefix(w->Y.pa, rows))
        return ICORE_LA_NONFINITE;
    memcpy(a, w->Y.pa, (size_t)w->nr * sizeof(double));
    if (!zero)
        memcpy(p, w->Y.pa + w->nr, (size_t)w->nx * sizeof(double));
    if (!reuse_pd && pd)
        memcpy(pd, w->V.pa, (size_t)w->nx * sizeof(double));
    return ICORE_LA_OK;
}

int icore_ocp_la_packed_backward_stage(icore_ocp_la_workspace *w, int stage, const double *bd, const double *d,
                              const double *pnext, int reuse_pd, double *pd, double *a, double *p)
{
    return packed_backward_stage_impl(w, stage, bd, d, pnext, reuse_pd, pd, a, p, 0);
}

int icore_ocp_la_factor_backward_stage(icore_ocp_la_workspace *w, int stage, const double *bd, const double *d,
                              const double *pnext, double *pd, double *a, double *p)
{
    /* Accepted factor progress proves this stage and its contiguous suffix
     * are complete. The ordinary public sweep still requires the full chain.
     * A new factor cannot reuse a Pd value from any previous factor. */
    return packed_backward_stage_impl(w, stage, bd, d, pnext, 0, pd, a, p, 1);
}

static int packed_forward_stage_impl(icore_ocp_la_workspace *w, int stage, const double *a, const double *x,
                                     const double *d, const double *pnext, double *u, double *next,
                                     double *dual, double *unshifted_dynamics)
{
    int i, rows, zero, offset;
    if (!w || stage < 0 || stage >= w->N || !a || !d || !pnext || !u || !next || !dual)
        return ICORE_LA_INVALID;
    if (!w->owned_factors || w->sqrt_next != -1 || !w->loaded[stage])
        return ICORE_LA_NOT_READY;
    zero = stage == 0 && w->sqrt_initial_zero;
    if (zero ? x != NULL : x == NULL)
        return ICORE_LA_INVALID;
    rows = zero ? w->nr : w->n;
    if (vector_spans_overlap(u, w->nr, next, w->nx) ||
        vector_spans_overlap(u, w->nr, dual, w->nx) ||
        vector_spans_overlap(next, w->nx, dual, w->nx))
        return ICORE_LA_INVALID;
    if (unshifted_dynamics && (vector_spans_overlap(unshifted_dynamics, w->nx, u, w->nr) ||
                               vector_spans_overlap(unshifted_dynamics, w->nx, next, w->nx) ||
                               vector_spans_overlap(unshifted_dynamics, w->nx, dual, w->nx)))
        return ICORE_LA_INVALID;
    if (!finite_prefix(a, w->nr) || (!zero && !finite_prefix(x, w->nx)))
        return ICORE_LA_NONFINITE;
    for (i = 0; i < w->nr; ++i)
        w->X.pa[i] = -a[i];
    if (!zero)
        memcpy(w->X.pa + w->nr, x, (size_t)w->nx * sizeof(double));
    /* Preserve [u;x] in X through the dynamics product and final commit. */
    blasfeo_dtrsv_ltn_mn(rows, w->nr, w->C + stage, 0, 0, &w->X, 0, &w->X, 0);
    if (!finite_prefix(w->X.pa, w->nr))
        return ICORE_LA_NONFINITE;
    memset(w->Z.pa, 0, (size_t)w->nx * sizeof(double));
    blasfeo_dgemv_t(rows, w->nx, 1.0, w->DT + stage, 0, 0, &w->X, 0, 0.0, &w->Z, 0, &w->Y, 0);
    /* Scale is factor scratch and has no consumer while the completed
     * chain is being solved. Keep the unshifted product private until all
     * forward outputs have passed their existing finite checks. */
    if (unshifted_dynamics)
        memcpy(w->scale, w->Y.pa, (size_t)w->nx * sizeof(double));
    if (!add_finite(w->Y.pa, w->Y.pa, d, w->nx))
        return ICORE_LA_NONFINITE;
    offset = root_offset(w, stage + 1);
    if (stage == w->N - 1 && w->sqrt_terminal_diagonal) {
        terminal_root_multiply(w, w->Y.pa, w->Z.pa);
        terminal_root_multiply(w, w->Z.pa, w->V.pa);
    } else {
        blasfeo_dtrmv_ltn(w->nx, w->C + stage + 1, offset, offset, &w->Y, 0, &w->Z, 0);
        blasfeo_dtrmv_lnn(w->nx, w->C + stage + 1, offset, offset, &w->Z, 0, &w->V, 0);
    }
    /* As in the backward sweep, the complete finite sum certifies the
     * preceding triangular product and both of its sum operands. */
    if (!add_finite(w->V.pa, w->V.pa, pnext, w->nx))
        return ICORE_LA_NONFINITE;
    memcpy(u, w->X.pa, (size_t)w->nr * sizeof(double));
    memcpy(next, w->Y.pa, (size_t)w->nx * sizeof(double));
    memcpy(dual, w->V.pa, (size_t)w->nx * sizeof(double));
    if (unshifted_dynamics)
        memcpy(unshifted_dynamics, w->scale, (size_t)w->nx * sizeof(double));
    return ICORE_LA_OK;
}

int icore_ocp_la_packed_forward_stage(icore_ocp_la_workspace *w, int stage, const double *a, const double *x,
                             const double *d, const double *pnext, double *u, double *next,
                             double *dual)
{
    return packed_forward_stage_impl(w, stage, a, x, d, pnext, u, next, dual, NULL);
}

int icore_ocp_la_packed_forward_stage_cached(icore_ocp_la_workspace *w, int stage, const double *a, const double *x,
                                    const double *d, const double *pnext, double *u, double *next,
                                    double *dual, double *unshifted_dynamics)
{
    if (stage <= 0 || !unshifted_dynamics)
        return ICORE_LA_INVALID;
    return packed_forward_stage_impl(w, stage, a, x, d, pnext, u, next, dual, unshifted_dynamics);
}

int icore_ocp_la_export_stage_factor(icore_ocp_la_workspace *w, int stage, double *L, double *T)
{
    int i, j;
    if (!w || stage < 0 || stage >= w->N || !L || !T)
        return ICORE_LA_INVALID;
    if (!w->owned_factors || w->sqrt_next != stage - 1 || (stage == 0 && w->sqrt_initial_zero))
        return ICORE_LA_NOT_READY;
    for (i = 0; i < w->nr; ++i) {
        for (j = 0; j <= i; ++j)
            L[(size_t)i * w->nr + j] = BLASFEO_DMATEL(w->C + stage, i, j);
        for (j = i + 1; j < w->nr; ++j)
            L[(size_t)i * w->nr + j] = 0.0;
    }
    blasfeo_unpack_tran_dmat(w->nx, w->nr, w->C + stage, w->nr, 0, T, w->nr);
    return ICORE_LA_OK;
}

int icore_ocp_la_terminal_border(icore_ocp_la_workspace *w, int stage, int nf, const double *Cn, double *C, double *Z,
                        double *S)
{
    int i, a, b;
    if (!w || stage < 0 || stage >= w->N || nf < 1 || nf > w->nx || nf > INT_MAX / w->nx || !Cn ||
        !C || !Z || !S)
        return ICORE_LA_INVALID;
    if (!w->owned_factors || w->sqrt_next != stage - 1 || (stage == 0 && w->sqrt_initial_zero))
        return ICORE_LA_NOT_READY;
    if (!finite_prefix(Cn, nf * w->nx))
        return ICORE_LA_NONFINITE;
    /* Row-major Cn is column-major Cn'. All matrices below are preallocated
     * factor scratch; accepted owned roots and dynamics stay untouched. */
    blasfeo_pack_dmat(w->nx, nf, (double *)Cn, w->nx, &w->PN, 0, 0);
    blasfeo_dgemm_nn(w->n, nf, w->nx, 1.0, w->DT + stage, 0, 0, &w->PN, 0, 0, 0.0, &w->M, 0, 0,
                     &w->M, 0, 0);
    blasfeo_dtrsm_llnn(w->nr, nf, 1.0, w->C + stage, 0, 0, &w->M, 0, 0, &w->F, 0, 0);
    blasfeo_dgemm_nn(w->nx, nf, w->nr, -1.0, w->C + stage, w->nr, 0, &w->F, 0, 0, 1.0, &w->M, w->nr,
                     0, &w->M, w->nr, 0);
    for (a = 0; a < nf; ++a) {
        for (i = 0; i < w->nr; ++i)
            if (!isfinite(BLASFEO_DMATEL(&w->F, i, a)))
                return ICORE_LA_NONFINITE;
        for (i = 0; i < w->nx; ++i)
            if (!isfinite(BLASFEO_DMATEL(&w->M, w->nr + i, a)))
                return ICORE_LA_NONFINITE;
    }
    /* Preserve the original ascending dot/accumulation order for S. */
    for (a = 0; a < nf; ++a)
        for (b = 0; b <= a; ++b) {
            double sum = 0.0, value;
            for (i = 0; i < w->nr; ++i)
                sum += BLASFEO_DMATEL(&w->F, i, a) * BLASFEO_DMATEL(&w->F, i, b);
            value = S[(size_t)a * nf + b] + sum;
            if (!isfinite(value))
                return ICORE_LA_NONFINITE;
            BLASFEO_DMATEL(&w->P, a, b) = value;
        }
    blasfeo_unpack_tran_dmat(w->nr, nf, &w->F, 0, 0, Z, nf);
    blasfeo_unpack_dmat(w->nx, nf, &w->M, w->nr, 0, C, w->nx);
    for (a = 0; a < nf; ++a)
        for (b = 0; b <= a; ++b)
            S[(size_t)a * nf + b] = BLASFEO_DMATEL(&w->P, a, b);
    return ICORE_LA_OK;
}

int icore_ocp_la_sqrt_checkpoint(icore_ocp_la_workspace *w)
{
    int k;
    if (!w)
        return ICORE_LA_INVALID;
    if (!w->checkpoint_C)
        return ICORE_LA_UNAVAILABLE;
    if (w->sqrt_next != -1)
        return ICORE_LA_NOT_READY;
    for (k = w->sqrt_root_omitted && !w->owned_factors ? 1 : 0; k <= w->N; ++k) {
        int n = w->owned_factors && k < w->N ? w->n : w->nx;
        blasfeo_dtrcp_l(n, w->C + k, 0, 0, w->checkpoint_C + k, 0, 0);
        if (w->owned_factors) {
            /* Factor kernels may compute reciprocal diagonals with a
             * different rounding path than a later 1/F_ii. These cached
             * values belong to the factor, not merely its packed matrix. */
            memcpy(w->checkpoint_C[k].dA, w->C[k].dA, (size_t)n * sizeof(double));
            w->checkpoint_C[k].use_dA = w->C[k].use_dA;
        } else
            w->checkpoint_C[k].use_dA = 0;
    }
    w->checkpoint_equilibrated = w->sqrt_equilibrated;
    w->checkpoint_root_omitted = w->sqrt_root_omitted;
    w->checkpoint_initial_zero = w->sqrt_initial_zero;
    w->checkpoint_valid = 1;
    return ICORE_LA_OK;
}

int icore_ocp_la_sqrt_restore(icore_ocp_la_workspace *w)
{
    int k;
    if (!w)
        return ICORE_LA_INVALID;
    sqrt_recovery_clear(w);
    if (!w->checkpoint_C)
        return ICORE_LA_UNAVAILABLE;
    if (!w->checkpoint_valid)
        return ICORE_LA_NOT_READY;
    /* A checkpoint may belong to a full, nondiagonal auxiliary factor. */
    w->sqrt_terminal_diagonal = 0;
    for (k = w->checkpoint_root_omitted && !w->owned_factors ? 1 : 0; k <= w->N; ++k) {
        int n = w->owned_factors && k < w->N ? w->n : w->nx;
        blasfeo_dtrcp_l(n, w->checkpoint_C + k, 0, 0, w->C + k, 0, 0);
        if (w->owned_factors) {
            memcpy(w->C[k].dA, w->checkpoint_C[k].dA, (size_t)n * sizeof(double));
            w->C[k].use_dA = w->checkpoint_C[k].use_dA;
        } else
            w->C[k].use_dA = 0;
    }
    w->sqrt_equilibrated = w->checkpoint_equilibrated;
    w->sqrt_root_omitted = w->checkpoint_root_omitted;
    w->sqrt_initial_zero = w->checkpoint_initial_zero;
    w->sqrt_next = -1;
    return ICORE_LA_OK;
}

void icore_ocp_la_sqrt_discard_checkpoint(icore_ocp_la_workspace *w)
{
    if (w)
        w->checkpoint_valid = 0;
}

size_t icore_ocp_la_sqrt_checkpoint_bytes(const icore_ocp_la_workspace *w)
{
    return w ? w->checkpoint_bytes : 0;
}

size_t icore_ocp_la_sqrt_checkpoint_copy_bytes(const icore_ocp_la_workspace *w)
{
    size_t count;
    if (!w || !w->checkpoint_valid)
        return 0;
    if (w->owned_factors)
        return ((size_t)w->N * (size_t)w->n * ((size_t)w->n + 1u) / 2u +
                (size_t)w->nx * ((size_t)w->nx + 1u) / 2u + (size_t)w->N * (size_t)w->n +
                (size_t)w->nx) *
               sizeof(double);
    count = (size_t)w->N + (w->checkpoint_root_omitted ? 0u : 1u);
    return count * ((size_t)w->nx * ((size_t)w->nx + 1u) / 2u) * sizeof(double);
}

/* ---- Local packed QR workspace and checked factorization. ---- */

struct icore_ocp_la_qr_workspace {
    int max_rows, max_columns;
    size_t allocated_bytes;
    void *storage;
    double *sign;
    struct blasfeo_dmat AT;
};

int icore_ocp_la_qr_create(int max_rows, int max_columns, icore_ocp_la_qr_workspace **out)
{
    icore_ocp_la_qr_workspace *w;
    unsigned char *next;
    size_t padded_rows, padded_columns, matrix_bytes, sign_bytes, total;
    int raw_bytes;
    if (!out)
        return ICORE_LA_INVALID;
    *out = NULL;
    if (max_columns <= 0 || max_rows < max_columns || max_rows > INT_MAX - 64 ||
        max_columns > INT_MAX - 64)
        return ICORE_LA_INVALID;
    /* The pinned library's matrix-size arithmetic uses int. Bound its
     * padded matrix plus diagonal workspace before querying that routine. */
    padded_rows = (size_t)max_rows + 64u;
    padded_columns = (size_t)max_columns + 64u;
    if (padded_rows > ((size_t)INT_MAX - 4096u) / sizeof(double) / padded_columns)
        return ICORE_LA_UNAVAILABLE;
    if (!icore_la_blasfeo_available())
        return ICORE_LA_UNAVAILABLE;
    raw_bytes = blasfeo_memsize_dmat(max_columns, max_rows);
    if (raw_bytes <= 0)
        return ICORE_LA_UNAVAILABLE;
    matrix_bytes = align64((size_t)raw_bytes);
    sign_bytes = align64((size_t)max_columns * sizeof(double));
    if (matrix_bytes > SIZE_MAX - sign_bytes - 63u - sizeof(*w))
        return ICORE_LA_NOMEM;
    total = matrix_bytes + sign_bytes + 63u;
    w = (icore_ocp_la_qr_workspace *)calloc(1, sizeof(*w));
    if (!w)
        return ICORE_LA_NOMEM;
    w->storage = calloc(1, total);
    if (!w->storage) {
        free(w);
        return ICORE_LA_NOMEM;
    }
    next = (unsigned char *)(((uintptr_t)w->storage + 63u) & ~(uintptr_t)63u);
    blasfeo_create_dmat(max_columns, max_rows, &w->AT, take(&next, matrix_bytes));
    w->sign = (double *)take(&next, sign_bytes);
    w->max_rows = max_rows;
    w->max_columns = max_columns;
    w->allocated_bytes = sizeof(*w) + total;
    *out = w;
    return ICORE_LA_OK;
}

void icore_ocp_la_qr_destroy(icore_ocp_la_qr_workspace *w)
{
    if (w) {
        free(w->storage);
        free(w);
    }
}

size_t icore_ocp_la_qr_bytes(const icore_ocp_la_qr_workspace *w)
{
    return w ? w->allocated_bytes : 0;
}

int icore_ocp_la_qr_upper(icore_ocp_la_qr_workspace *w, int rows, int columns, double *A, size_t stride)
{
    int i, j;
    int extreme = 0, packed_rows = 0, run_start = -1;
    if (!w || !A || columns <= 0 || rows < columns || rows > w->max_rows ||
        columns > w->max_columns || stride < (size_t)columns || stride > INT_MAX ||
        (size_t)rows > SIZE_MAX / sizeof(double) / stride)
        return ICORE_LA_INVALID;
    /* Upstream LQ uses sum-of-squares norms. Keep substantial exponent
     * headroom for products/Householder updates; extreme input takes the
     * original hypot-based QR without mutating A. Nonfinite has priority. */
    for (i = 0; i < rows; ++i) {
        uint64_t row_bits = 0;
        for (j = 0; j < columns; ++j) {
            double value = A[(size_t)i * stride + j], magnitude;
            uint64_t bits;
            if (!isfinite(value))
                return ICORE_LA_NONFINITE;
            magnitude = fabs(value);
            if (magnitude != 0.0 && (magnitude < 0x1p-450 || magnitude > 0x1p450))
                extreme = 1;
            /* Exact binary64 zero, independent of the caller's DAZ/FTZ.
             * Both signed zeros qualify; no subnormal value does. */
            memcpy(&bits, &value, sizeof(bits));
            row_bits |= bits;
        }
        if (row_bits & UINT64_C(0x7fffffffffffffff)) {
            if (run_start < 0)
                run_start = i;
        } else if (run_start >= 0) {
            /* Removing an exact-zero row preserves A' A. Pack contiguous
             * nonzero runs in their original order; only scratch changes.
             * In the transpose, these rows are adjacent columns of AT. */
            int count = i - run_start;
            blasfeo_pack_dmat(columns, count, A + (size_t)run_start * stride, (int)stride, &w->AT,
                              0, packed_rows);
            packed_rows += count;
            run_start = -1;
        }
    }
    if (extreme)
        return ICORE_LA_UNAVAILABLE;
    if (run_start >= 0) {
        int count = rows - run_start;
        blasfeo_pack_dmat(columns, count, A + (size_t)run_start * stride, (int)stride, &w->AT, 0,
                          packed_rows);
        packed_rows += count;
    }
    if (packed_rows < columns) {
        /* Preserve the generic API's established rank-deficient behavior.
         * Solver stacks append a positive regularization diagonal, so their
         * n regularization rows always survive exact-zero compaction. */
        packed_rows = rows;
        blasfeo_pack_dmat(columns, rows, A, (int)stride, &w->AT, 0, 0);
    }
    /* Only the n by n R is returned; no Q or row map leaves this routine. */
    blasfeo_dgelqf(columns, packed_rows, &w->AT, 0, 0, &w->AT, 0, 0, NULL);
    for (i = 0; i < columns; ++i) {
        double diagonal = BLASFEO_DMATEL(&w->AT, i, i);
        if (!isfinite(diagonal))
            return ICORE_LA_NONFINITE;
        if (diagonal == 0.0)
            return ICORE_LA_NOT_SPD;
        w->sign[i] = diagonal < 0.0 ? -1.0 : 1.0;
        for (j = 0; j < i; ++j)
            if (!isfinite(BLASFEO_DMATEL(&w->AT, i, j)))
                return ICORE_LA_NONFINITE;
    }
    /* AT=L*Q implies A=Q^T*L^T. Column signs of L become row signs of R;
     * normalizing them preserves R^T*R and the parent's positive diagonal. */
    for (i = 0; i < columns; ++i)
        for (j = i; j < columns; ++j)
            A[(size_t)i * stride + j] = w->sign[i] * BLASFEO_DMATEL(&w->AT, j, i);
    return ICORE_LA_OK;
}

#else
#include "ocp_la_stubs.h"
#endif
