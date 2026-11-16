/* Physical-chain matrix kernels for the shared primal-dual IPM driver. */
#include "chain.h"
#include "driver.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__) && !defined(CHAIN_SCALAR)
#include <immintrin.h>
#endif
#if defined(__GNUC__) || defined(__clang__)
#define INLINE static inline __attribute__((always_inline))
#define RESTRICT __restrict__
#else
#define INLINE static inline
#define RESTRICT
#endif

/* Read driver_loop.h for the shared algorithm. Model ownership and callback
 * adapters are separated below; local algebra remains in this translation unit. */
/* ---- Owned chain state and sparse-row metadata. ---- */

typedef struct {
    int stage, left, count, constant;
} row;
typedef struct chain_saddle chain_saddle;

struct chain_workspace {
    int K, n, d, p, l, nv, nd, np, ng, m, nr, dim, valid, use_qr;
    row *rows;
    int *columns, *active, *position, *active_count;
    uint64_t *patterns;
    unsigned char *sparse_stage;
    double *arena, *values, *offset, *original_offset, *h, *c, *fixed, *free;
    double *center, *original_h, *original_c, *lambda, *D, *link, *qr, *work;
    double *state, *target, *affine, *step, *error, *correction, *trial, *trial_error;
    double *weight, *inverse, *ratio, *curvature, *rhs, *q, *force;
    chain_saddle *saddle;
    /* Only chain_update changes row structure.  A retained graph is merely a
     * cache until an actual augmented request validates that exact structure. */
    int saddle_prepare_pending;
};

static chain_saddle *chain_saddle_prepare(chain_workspace *, chain_saddle *);
static void chain_saddle_destroy(chain_saddle *);
static void chain_saddle_invalidate(chain_saddle *);

/* ---- Vector products and local Cholesky / QR primitives. ---- */

INLINE double dot(const double *a, const double *b, int n)
{
#if defined(__AVX2__) && !defined(CHAIN_SCALAR)
    __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
    int i = 0;
    double sum;
    if (n < 4) {
        sum = 0.;
        for (i = 0; i < n; ++i)
            sum += a[i] * b[i];
        return sum;
    }
    for (; i + 7 < n; i += 8) {
#if defined(__FMA__)
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i), s0);
        s1 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4), s1);
#else
        s0 = _mm256_add_pd(s0, _mm256_mul_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
        s1 = _mm256_add_pd(s1,
                           _mm256_mul_pd(_mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4)));
#endif
    }
    if (i + 3 < n) {
#if defined(__FMA__)
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i), s0);
#else
        s0 = _mm256_add_pd(s0, _mm256_mul_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
#endif
        i += 4;
    }
    s0 = _mm256_add_pd(s0, s1);
    {
        __m128d p = _mm_add_pd(_mm256_castpd256_pd128(s0), _mm256_extractf128_pd(s0, 1));
        sum = _mm_cvtsd_f64(_mm_add_sd(p, _mm_unpackhi_pd(p, p)));
    }
    for (; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
#else
    double a0 = 0., a1 = 0., a2 = 0., a3 = 0.;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        a0 += a[i] * b[i];
        a1 += a[i + 1] * b[i + 1];
        a2 += a[i + 2] * b[i + 2];
        a3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; ++i)
        a0 += a[i] * b[i];
    return (a0 + a1) + (a2 + a3);
#endif
}

INLINE void axpy(double *RESTRICT y, const double *RESTRICT x, double a, int n)
{
    int i;
    for (i = 0; i < n; ++i)
        y[i] += a * x[i];
}

INLINE double maximum(const double *x, int n)
{
    double v = 0.;
    int i = 0;
#if defined(__AVX2__) && !defined(CHAIN_SCALAR)
    __m256d high = _mm256_setzero_pd(), sign = _mm256_set1_pd(-0.), limit = _mm256_set1_pd(DBL_MAX);
    for (; i + 3 < n; i += 4) {
        __m256d a = _mm256_andnot_pd(sign, _mm256_loadu_pd(x + i));
        if (_mm256_movemask_pd(_mm256_cmp_pd(a, limit, _CMP_NLE_UQ)))
            return INFINITY;
        high = _mm256_max_pd(high, a);
    }
    {
        __m128d p = _mm_max_pd(_mm256_castpd256_pd128(high), _mm256_extractf128_pd(high, 1));
        v = _mm_cvtsd_f64(_mm_max_pd(p, _mm_unpackhi_pd(p, p)));
    }
#endif
    for (; i < n; ++i) {
        double a = fabs(x[i]);
        if (!isfinite(a))
            return INFINITY;
        if (a > v)
            v = a;
    }
    return v;
}

INLINE int first_bit(uint64_t bits)
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(bits);
#else
    int index = 0;
    while (!(bits & 1)) {
        ++index;
        bits >>= 1;
    }
    return index;
#endif
}

static int cholesky(double *L, int n, const uint64_t *pattern)
{
    int i, j;
    for (i = 0; i < n; ++i)
        for (j = 0; j <= i; ++j) {
            double sum = 0., v;
            if (pattern) {
                uint64_t bits;
                if (!(pattern[i] & (UINT64_C(1) << j))) {
                    L[i * n + j] = 0.;
                    continue;
                }
                bits = pattern[i] & pattern[j] & ((UINT64_C(1) << j) - 1);
                while (bits) {
                    int k = first_bit(bits);
                    sum += L[i * n + k] * L[j * n + k];
                    bits &= bits - 1;
                }
            } else
                sum = dot(L + i * n, L + j * n, j);
            v = L[i * n + j] - sum;
            if (i == j) {
                if (!(v > 0.) || !isfinite(v))
                    return 0;
                L[i * n + j] = sqrt(v);
            } else
                L[i * n + j] = v / L[j * n + j];
        }
    return 1;
}

static int cholesky_active(chain_workspace *w, int stage)
{
    int i, j, k, n = w->n, count = w->active_count[stage];
    const int *idx = w->active + stage * n, *pos = w->position + stage * n;
    double *L = w->D + (size_t)stage * n * n;
    for (i = 0; i < n; ++i)
        if (pos[i] < 0) {
            double value = L[i * n + i];
            if (!(value > 0.) || !isfinite(value))
                return 0;
            L[i * n + i] = sqrt(value);
        }
    for (i = 0; i < count; ++i)
        for (j = 0; j <= i; ++j) {
            int ii = idx[i], jj = idx[j];
            double sum = 0., value;
            for (k = 0; k < j; ++k)
                sum += L[ii * n + idx[k]] * L[jj * n + idx[k]];
            value = L[ii * n + jj] - sum;
            if (i == j) {
                if (!(value > 0.) || !isfinite(value))
                    return 0;
                L[ii * n + jj] = sqrt(value);
            } else
                L[ii * n + jj] = value / L[jj * n + jj];
        }
    return 1;
}

static void schur_active(chain_workspace *w, int stage)
{
    int i, j, k, n = w->n, c = w->active_count[stage], next_count = w->active_count[stage + 1];
    const int *idx = w->active + stage * n, *next_idx = idx + n;
    double *L = w->D + (size_t)stage * n * n, *next = L + n * n;
    double *link = w->link + (size_t)stage * n * n, *u = w->qr;
    for (i = 0; i < c; ++i) {
        int ii = idx[i];
        double inv = 1. / L[ii * n + ii];
        for (j = 0; j < next_count; ++j) {
            int jj = next_idx[j];
            double value = link[jj * n + ii];
            for (k = 0; k < i; ++k)
                value -= L[ii * n + idx[k]] * u[idx[k] * n + jj];
            u[ii * n + jj] = value * inv;
        }
    }
    for (i = 0; i < next_count; ++i)
        for (j = 0; j <= i; ++j) {
            int ii = next_idx[i], jj = next_idx[j];
            double value = next[ii * n + jj];
            for (k = 0; k < c; ++k)
                value -= u[idx[k] * n + ii] * u[idx[k] * n + jj];
            next[ii * n + jj] = value;
        }
    for (i = 0; i < next_count; ++i)
        for (j = 0; j < c; ++j)
            link[next_idx[i] * n + idx[j]] = u[idx[j] * n + next_idx[i]];
}

static void forward(const double *L, double *x, int n)
{
    int i;
    for (i = 0; i < n; ++i)
        x[i] = (x[i] - dot(L + i * n, x, i)) / L[i * n + i];
}

static void backward(const double *L, double *x, int n)
{
    int i;
    for (i = n - 1; i >= 0; --i) {
        x[i] /= L[i * n + i];
        axpy(x, L + i * n, -x[i], i);
    }
}

/* Insert sparse rows with Givens rotations. Exact zero entries need no
 * rotation; only the first required diagonal pivots must be nonzero. */
static int qr(double *a, int rows, int cols, int stride, int required, double *work)
{
    int i, j, k;
    (void)work;
    for (i = 1; i < rows; ++i)
        for (j = 0; j < cols && j < i; ++j) {
            double x = a[i * stride + j], head, square, norm, c, sn;
            if (x == 0.)
                continue;
            head = a[j * stride + j];
            square = head * head + x * x;
            norm = square >= DBL_MIN && isfinite(square) ? sqrt(square) : hypot(head, x);
            if (!(norm > 0.) || !isfinite(norm))
                return 0;
            c = head / norm;
            sn = x / norm;
            for (k = j + 1; k < cols; ++k) {
                double top = a[j * stride + k], bottom = a[i * stride + k];
                a[j * stride + k] = c * top + sn * bottom;
                a[i * stride + k] = -sn * top + c * bottom;
            }
            a[j * stride + j] = norm;
            a[i * stride + j] = 0.;
        }
    for (i = 0; i < cols; ++i) {
        if (i < required && !(fabs(a[i * stride + i]) > 0.))
            return 0;
        for (j = i; j < cols; ++j)
            if (!isfinite(a[i * stride + j]))
                return 0;
    }
    return 1;
}

#include "chain_model.h"

/* ---- Chain products, factorization, Newton solves, and refinement. ---- */

INLINE double row_dot(const chain_workspace *w, int r, const double *x)
{
    row a = w->rows[r];
    const int *col = w->columns + (size_t)r * 2 * w->n;
    const double *v = w->values + (size_t)r * 2 * w->n;
    double value = 0.;
    int j;
    x += a.stage * w->n;
    for (j = 0; j < a.left; ++j)
        value += v[j] * x[col[j]];
    for (; j < a.count; ++j)
        value += v[j] * x[w->n + col[j]];
    return value;
}

INLINE void row_add(const chain_workspace *w, int r, double value, double *x)
{
    row a = w->rows[r];
    const int *col = w->columns + (size_t)r * 2 * w->n;
    const double *v = w->values + (size_t)r * 2 * w->n;
    int j;
    x += a.stage * w->n;
    for (j = 0; j < a.left; ++j)
        x[col[j]] += value * v[j];
    for (; j < a.count; ++j)
        x[w->n + col[j]] += value * v[j];
}

static void apply(const chain_workspace *w, const double *x, const double *t, double *g,
                  int offsets)
{
    int i;
    double value;
    for (i = 0; i < w->nd; ++i) {
        value = row_dot(w, i, x) + (offsets ? w->offset[i] : 0.);
        g[i] = value - t[i];
        g[w->nd + i] = -value - t[i];
    }
    for (i = w->nd; i < w->nr; ++i)
        g[w->nd + i] = row_dot(w, i, x) + (offsets ? w->offset[i] : 0.);
}

static void transpose(const chain_workspace *w, const double *z, double *x, double sign)
{
    int i;
    for (i = 0; i < w->nd; ++i)
        row_add(w, i, sign * (z[i] - z[w->nd + i]), x);
    for (i = w->nd; i < w->nr; ++i)
        row_add(w, i, sign * z[w->nd + i], x);
}

/* Private sparse augmented recovery for a physical chain.
 *
 * [ H+reg*I  (sqrt(C) A)' ] [dx ] = [ bx          ]
 * [ sqrt(C) A      -I    ] [eta]   [ -f/sqrt(C)  ]
 *
 * The combined dual is sqrt(C)*eta.  No normal RHS or cancelled
 * f+C*A*dx is formed.  Original Newton checks and QR remain authoritative.
 * Included after the chain workspace and row kernels are defined.
 */
#ifndef ICORE_CHAIN_SADDLE_H
#define ICORE_CHAIN_SADDLE_H
#include "chain_sparse/amd/amd.h"
#include "chain_sparse/qdldl/qdldl.h"

struct chain_saddle {
    int n, nnz, lnz, nr, edges, valid;
    int *Ap, *Ai, *P, *Lp, *Li, *Lnz, *etree, *iwork, *signature;
    size_t *source;
    unsigned char *bwork;
    double *Ax, *Lx, *D, *Dinv, *fwork, *b, *root;
    double regularization;
};

typedef struct {
    int row;
    size_t source;
} chain_saddle_entry;

static int chain_saddle_entry_compare(const void *aa, const void *bb)
{
    const chain_saddle_entry *a = (const chain_saddle_entry *)aa;
    const chain_saddle_entry *b = (const chain_saddle_entry *)bb;
    return (a->row > b->row) - (a->row < b->row);
}

static void chain_saddle_invalidate(chain_saddle *a)
{
    if (a)
        a->valid = 0;
}

static void chain_saddle_destroy(chain_saddle *a)
{
    if (!a)
        return;
    free(a->Ap);
    free(a->Ai);
    free(a->P);
    free(a->Lp);
    free(a->Li);
    free(a->Lnz);
    free(a->etree);
    free(a->iwork);
    free(a->signature);
    free(a->source);
    free(a->bwork);
    free(a->Ax);
    free(a->Lx);
    free(a->D);
    free(a->Dinv);
    free(a->fwork);
    free(a->b);
    free(a->root);
    free(a);
}

static void *chain_saddle_allocate(size_t count, size_t width)
{
    if (count > SIZE_MAX / width)
        return NULL;
    return calloc(count ? count : 1, width);
}

/* Exact structural identity, not a hash or a numeric rank estimate.
 * Coefficients and cost values are read fresh from w during every factor. */
static int chain_saddle_structure_same(const chain_workspace *w, const chain_saddle *a)
{
    int r, j;
    size_t k = 0;
    if (!a || a->n != w->nv + w->nr || a->nr != w->nr)
        return 0;
    for (r = 0; r < w->nr; ++r) {
        row q = w->rows[r];
        if (a->signature[k++] != q.stage || a->signature[k++] != q.left ||
            a->signature[k++] != q.count)
            return 0;
        for (j = 0; j < q.count; ++j) {
            int col =
                q.stage * w->n + w->columns[(size_t)r * 2 * w->n + j] + (j >= q.left ? w->n : 0);
            if (a->signature[k++] != col)
                return 0;
        }
    }
    return k == (size_t)3 * w->nr + a->edges;
}

static chain_saddle *chain_saddle_prepare(chain_workspace *w, chain_saddle *old)
{
    chain_saddle *a = NULL;
    int *Op = NULL, *Oi = NULL, *inv = NULL, *next = NULL;
    size_t *Os = NULL;
    chain_saddle_entry *entries = NULL;
    size_t edges = 0, total, spos = 0;
    int n = w->nv + w->nr, nnz, r, j, k, pos, status;
    if (chain_saddle_structure_same(w, old)) {
        chain_saddle_invalidate(old);
        return old;
    }
    /* QDLDL's three n-element work slices use its signed integer index type. */
    if (n > INT_MAX / 3)
        goto done;
    for (r = 0; r < w->nr; ++r) {
        if (edges > (size_t)INT_MAX - (size_t)w->rows[r].count)
            goto done;
        edges += (size_t)w->rows[r].count;
    }
    if (edges > (size_t)INT_MAX - (size_t)n)
        goto done;
    total = (size_t)n + edges;
    nnz = (int)total;
    a = (chain_saddle *)calloc(1, sizeof(*a));
    if (!a)
        goto done;
    a->n = n;
    a->nnz = nnz;
    a->nr = w->nr;
    a->edges = (int)edges;
#define S_ALLOC(field, count, type)                                                                \
    a->field = (type *)chain_saddle_allocate((size_t)(count), sizeof(type));                       \
    if (!a->field)                                                                                 \
    goto failure
    S_ALLOC(Ap, (size_t)n + 1, int);
    S_ALLOC(Ai, nnz, int);
    S_ALLOC(P, n, int);
    S_ALLOC(Lp, (size_t)n + 1, int);
    S_ALLOC(Lnz, n, int);
    S_ALLOC(etree, n, int);
    S_ALLOC(iwork, (size_t)3 * n, int);
    S_ALLOC(signature, (size_t)3 * w->nr + edges, int);
    S_ALLOC(source, nnz, size_t);
    S_ALLOC(bwork, n, unsigned char);
    S_ALLOC(Ax, nnz, double);
    S_ALLOC(D, n, double);
    S_ALLOC(Dinv, n, double);
    S_ALLOC(fwork, n, double);
    S_ALLOC(b, n, double);
    S_ALLOC(root, w->nr, double);
    Op = (int *)chain_saddle_allocate((size_t)n + 1, sizeof(int));
    Oi = (int *)chain_saddle_allocate(nnz, sizeof(int));
    Os = (size_t *)chain_saddle_allocate(nnz, sizeof(size_t));
    inv = (int *)chain_saddle_allocate(n, sizeof(int));
    next = (int *)chain_saddle_allocate(n, sizeof(int));
    entries = (chain_saddle_entry *)chain_saddle_allocate(nnz, sizeof(chain_saddle_entry));
    if (!Op || !Oi || !Os || !inv || !next || !entries)
        goto failure;
    pos = 0;
    for (j = 0; j < w->nv; ++j) {
        Op[j] = pos;
        Oi[pos] = j;
        Os[pos++] = SIZE_MAX;
    }
    for (r = 0; r < w->nr; ++r) {
        row q = w->rows[r];
        Op[w->nv + r] = pos;
        a->signature[spos++] = q.stage;
        a->signature[spos++] = q.left;
        a->signature[spos++] = q.count;
        for (j = 0; j < q.count; ++j) {
            size_t offset = (size_t)r * 2 * w->n + j;
            int col = q.stage * w->n + w->columns[offset] + (j >= q.left ? w->n : 0);
            Oi[pos] = col;
            Os[pos++] = offset;
            a->signature[spos++] = col;
        }
        Oi[pos] = w->nv + r;
        Os[pos++] = SIZE_MAX;
    }
    Op[n] = pos;
    if (pos != nnz)
        goto failure;
    status = amd_order(n, Op, Oi, a->P, NULL, NULL);
    if (status != AMD_OK && status != AMD_OK_BUT_JUMBLED)
        goto failure;
    for (j = 0; j < n; ++j)
        inv[a->P[j]] = j;
    for (j = 0; j < n; ++j)
        for (k = Op[j]; k < Op[j + 1]; ++k) {
            int c = inv[j] > inv[Oi[k]] ? inv[j] : inv[Oi[k]];
            ++a->Ap[c + 1];
        }
    for (j = 0; j < n; ++j) {
        a->Ap[j + 1] += a->Ap[j];
        next[j] = a->Ap[j];
    }
    for (j = 0; j < n; ++j)
        for (k = Op[j]; k < Op[j + 1]; ++k) {
            int p = inv[j], q = inv[Oi[k]], c = p > q ? p : q, at = next[c]++;
            entries[at].row = p < q ? p : q;
            entries[at].source = Os[k];
        }
    for (j = 0; j < n; ++j) {
        qsort(entries + a->Ap[j], (size_t)(a->Ap[j + 1] - a->Ap[j]), sizeof(*entries),
              chain_saddle_entry_compare);
        for (k = a->Ap[j]; k < a->Ap[j + 1]; ++k) {
            a->Ai[k] = entries[k].row;
            a->source[k] = entries[k].source;
        }
    }
    a->lnz = QDLDL_etree(n, a->Ap, a->Ai, a->iwork, a->Lnz, a->etree);
    if (a->lnz < 0)
        goto failure;
    S_ALLOC(Li, a->lnz, int);
    S_ALLOC(Lx, a->lnz, double);
#undef S_ALLOC
    goto done;
failure:
    chain_saddle_destroy(a);
    a = NULL;
done:
    free(Op);
    free(Oi);
    free(Os);
    free(inv);
    free(next);
    free(entries);
    chain_saddle_destroy(old);
    return a;
}

static int chain_saddle_factor(chain_workspace *w, double reg)
{
    chain_saddle *a;
    int i, j, k, status;
    if (w->saddle_prepare_pending) {
        /* Preparation is optional recovery work, charged to this solve.  The
         * old graph cannot be used before exact current-structure validation.
         * Allocation failure returns to the existing classical/QR route. */
        w->saddle = chain_saddle_prepare(w, w->saddle);
        /* At most one preparation attempt per accepted update, including
         * unavailable optional storage. A later accepted update can retry. */
        w->saddle_prepare_pending = 0;
    }
    a = w->saddle;
    if (!a)
        return 0;
    a->valid = 0;
    a->regularization = reg;
    for (i = 0; i < w->nr; ++i) {
        double curvature = w->curvature[i];
        if (!(curvature > 0.) || !isfinite(curvature))
            return 0;
        a->root[i] = sqrt(curvature);
    }
    for (j = 0; j < a->n; ++j)
        for (k = a->Ap[j]; k < a->Ap[j + 1]; ++k) {
            int r = a->P[a->Ai[k]], c = a->P[j];
            double value;
            if (r == c)
                value = r < w->nv ? w->h[r] + reg : -1.;
            else
                value = w->values[a->source[k]] * a->root[(r >= w->nv ? r : c) - w->nv];
            if (!isfinite(value))
                return 0;
            a->Ax[k] = value;
        }
    status = QDLDL_factor(a->n, a->Ap, a->Ai, a->Ax, a->Lp, a->Li, a->Lx, a->D, a->Dinv, a->Lnz,
                          a->etree, a->bwork, a->iwork, a->fwork, a->P, w->nv, 0.);
    if (status != w->nv)
        return 0;
    for (i = 0; i < a->n; ++i)
        if (!isfinite(a->D[i]) || a->D[i] == 0. || !isfinite(a->Dinv[i]) || a->Dinv[i] == 0. ||
            (a->P[i] < w->nv ? !(a->D[i] > 0.) : !(a->D[i] < 0.)))
            return 0;
    for (i = 0; i < a->lnz; ++i)
        if (!isfinite(a->Lx[i]))
            return 0;
    a->valid = 1;
    w->use_qr = 0;
    return 1;
}

/* Generic reduced primal RHS used only by initialization.  In this call
 * the dual RHS is zero, giving exactly (H+reg I+A'C A)x=b. */
static int chain_saddle_reduced(chain_workspace *w, const double *b, double *x)
{
    chain_saddle *a = w->saddle;
    int i;
    if (!a || !a->valid || w->saddle_prepare_pending)
        return 0;
    for (i = 0; i < a->n; ++i)
        a->b[i] = a->P[i] < w->nv ? b[a->P[i]] : 0.;
    QDLDL_solve(a->n, a->Lp, a->Li, a->Lx, a->Dinv, a->b);
    for (i = 0; i < a->n; ++i)
        if (!isfinite(a->b[i]))
            return 0;
    for (i = 0; i < a->n; ++i)
        if (a->P[i] < w->nv)
            x[a->P[i]] = a->b[i];
    return 1;
}

static int chain_saddle_rhs(chain_workspace *w, const double *rhs, double *step)
{
    chain_saddle *a = w->saddle;
    int i, nv = w->nv, nd = w->nd, m = w->m;
    const double *bx = rhs, *bt = bx + nv, *bp = bt + nd, *bc = bp + m;
    const double *s = w->state + nv + nd, *z = s + m;
    double *dx = step, *dt = dx + nv, *ds = dt + nd, *dz = ds + m, *q = w->q;
    if (!a || !a->valid || w->saddle_prepare_pending)
        return 0;
    for (i = 0; i < m; ++i)
        q[i] = (bc[i] - z[i] * bp[i]) / s[i];
    for (i = 0; i < nd; ++i) {
        double inv = w->inverse[i], wp = w->weight[i], wm = w->weight[nd + i];
        w->force[i] = 2. * (wm * inv) * q[i] - 2. * (wp * inv) * q[nd + i] + w->ratio[i] * bt[i];
    }
    for (i = 0; i < a->n; ++i) {
        int j = a->P[i];
        double value;
        if (j < nv)
            value = bx[j];
        else {
            int r = j - nv;
            value = -(r < nd ? w->force[r] : q[nd + r]) / a->root[r];
        }
        if (!isfinite(value))
            return 0;
        a->b[i] = value;
    }
    QDLDL_solve(a->n, a->Lp, a->Li, a->Lx, a->Dinv, a->b);
    for (i = 0; i < a->n; ++i) {
        int j = a->P[i];
        if (!isfinite(a->b[i]))
            return 0;
        if (j < nv)
            dx[j] = a->b[i];
        else {
            int r = j - nv;
            double nu = a->root[r] * a->b[i];
            if (r < nd) {
                dz[r] = .5 * (nu - bt[r]);
                dz[nd + r] = .5 * (-nu - bt[r]);
            } else
                dz[nd + r] = nu;
        }
    }
    for (i = 0; i < nd; ++i) {
        double value = row_dot(w, i, dx);
        dt[i] = w->inverse[i] * (bt[i] + q[i] + q[nd + i]) - w->ratio[i] * value;
        ds[i] = bp[i] - (value - dt[i]);
        ds[nd + i] = bp[nd + i] - (-value - dt[i]);
    }
    for (i = nd; i < w->nr; ++i)
        ds[nd + i] = bp[nd + i] - row_dot(w, i, dx);
    return isfinite(maximum(step, w->dim));
}
#endif

static int factor_qr(chain_workspace *w, double reg)
{
    chain_saddle_invalidate(w->saddle);
    int k, j, i, a, r, n = w->n, stride = 2 * n, rows, cols, cur, next;
    double *matrix = w->qr, *carry = w->link + (size_t)(w->K - 1) * n * n;
    memset(carry, 0, (size_t)n * n * sizeof(double));
    for (k = 0; k < w->K; ++k) {
        const int *idx = w->active + k * n, *pos = w->position + k * n;
        const int *next_pos = pos + n;
        cur = w->active_count[k];
        next = k < w->K - 1 ? w->active_count[k + 1] : 0;
        if (cur == n && (k == w->K - 1 || next == n)) {
            rows = 2 * n + w->l + (k < w->K - 1 ? w->d + w->p : 0);
            memset(matrix, 0, (size_t)rows * stride * sizeof(double));
            for (i = 0; i < n; ++i)
                matrix[i * stride + i] = sqrt(w->h[k * n + i] + reg);
            for (i = 0; i < n; ++i)
                for (j = 0; j < n; ++j)
                    matrix[(n + i) * stride + j] = carry[i * n + j];
            a = 2 * n;
            for (j = 0; j < w->l; ++j) {
                r = w->nd + w->np + k * w->l + j;
                row q = w->rows[r];
                double scale = sqrt(w->curvature[r]);
                for (i = 0; i < q.count; ++i)
                    matrix[a * stride + w->columns[(size_t)r * 2 * n + i]] =
                        scale * w->values[(size_t)r * 2 * n + i];
                ++a;
            }
            if (k < w->K - 1)
                for (j = 0; j < w->d + w->p; ++j) {
                    r = j < w->d ? k * w->d + j : w->nd + k * w->p + j - w->d;
                    row q = w->rows[r];
                    double scale = sqrt(w->curvature[r]);
                    for (i = 0; i < q.count; ++i)
                        matrix[a * stride + (i < q.left ? 0 : n) +
                               w->columns[(size_t)r * 2 * n + i]] =
                            scale * w->values[(size_t)r * 2 * n + i];
                    ++a;
                }
            if (!qr(matrix, rows, k < w->K - 1 ? 2 * n : n, stride, n, w->work))
                return 0;
            for (i = 0; i < n; ++i)
                for (j = 0; j <= i; ++j)
                    w->D[(size_t)k * n * n + i * n + j] = matrix[j * stride + i];
            if (k < w->K - 1) {
                for (i = 0; i < n; ++i)
                    for (j = 0; j < n; ++j) {
                        w->link[(size_t)k * n * n + i * n + j] = matrix[j * stride + n + i];
                        carry[i * n + j] = j >= i ? matrix[(n + i) * stride + n + j] : 0.;
                    }
            }
            continue;
        }
        cols = cur + next;
        rows = 2 * cur + w->l + (k < w->K - 1 ? w->d + w->p : 0);
        if (rows < cols)
            rows = cols;
        memset(matrix, 0, (size_t)rows * stride * sizeof(double));
        for (i = 0; i < cur; ++i)
            matrix[i * stride + i] = sqrt(w->h[k * n + idx[i]] + reg);
        for (i = 0; i < cur; ++i)
            for (j = 0; j < cur; ++j)
                matrix[(cur + i) * stride + j] = carry[i * n + j];
        a = 2 * cur;
        for (j = 0; j < w->l; ++j) {
            row q;
            double scale;
            r = w->nd + w->np + k * w->l + j;
            q = w->rows[r];
            scale = sqrt(w->curvature[r]);
            for (i = 0; i < q.count; ++i)
                matrix[a * stride + pos[w->columns[(size_t)r * 2 * n + i]]] =
                    scale * w->values[(size_t)r * 2 * n + i];
            ++a;
        }
        if (k < w->K - 1)
            for (j = 0; j < w->d + w->p; ++j) {
                row q;
                double scale;
                r = j < w->d ? k * w->d + j : w->nd + k * w->p + j - w->d;
                q = w->rows[r];
                scale = sqrt(w->curvature[r]);
                for (i = 0; i < q.count; ++i) {
                    int col = w->columns[(size_t)r * 2 * n + i];
                    col = i < q.left ? pos[col] : cur + next_pos[col];
                    matrix[a * stride + col] = scale * w->values[(size_t)r * 2 * n + i];
                }
                ++a;
            }
        if (!qr(matrix, rows, cols, stride, cur, w->work))
            return 0;
        for (i = 0; i < n; ++i)
            for (j = 0; j <= i; ++j) {
                double value = 0.;
                if (pos[i] >= 0 && pos[j] >= 0)
                    value = matrix[pos[j] * stride + pos[i]];
                else if (i == j) {
                    double pivot = w->h[k * n + i] + reg;
                    if (!(pivot > 0.) || !isfinite(pivot))
                        return 0;
                    value = sqrt(pivot);
                }
                w->D[(size_t)k * n * n + i * n + j] = value;
            }
        if (k < w->K - 1) {
            for (i = 0; i < n; ++i)
                for (j = 0; j < n; ++j)
                    w->link[(size_t)k * n * n + i * n + j] =
                        next_pos[i] >= 0 && pos[j] >= 0
                            ? matrix[pos[j] * stride + cur + next_pos[i]]
                            : 0.;
            for (i = 0; i < next; ++i)
                for (j = 0; j < next; ++j)
                    carry[i * n + j] = j >= i ? matrix[(cur + i) * stride + cur + j] : 0.;
        }
    }
    w->use_qr = 1;
    return 1;
}

/* AUTO has already requested its numerically robust recovery here.  The
 * exact augmented solve is tried once; original QR remains the fallback. */
static int factor_recovery(chain_workspace *w, double reg)
{
    if (chain_saddle_factor(w, reg))
        return 1;
    return factor_qr(w, reg);
}

static int factor(chain_workspace *w, double reg, int mode)
{
    chain_saddle_invalidate(w->saddle);
    int r, i, j, k, n = w->n;
    double *D = w->D, *T = w->link;
    if (mode == ICORE_OCP_SQUARE_ROOT)
        return factor_qr(w, reg);
    memset(D, 0, (size_t)w->K * n * n * sizeof(double));
    memset(T, 0, (size_t)w->K * n * n * sizeof(double));
    for (k = 0; k < w->K; ++k)
        for (i = 0; i < n; ++i)
            D[(size_t)k * n * n + i * n + i] = w->h[k * n + i] + reg;
    for (r = 0; r < w->nr; ++r) {
        row a = w->rows[r];
        const int *col = w->columns + (size_t)r * 2 * n;
        const double *v = w->values + (size_t)r * 2 * n;
        double weight = w->curvature[r], *left = D + (size_t)a.stage * n * n, *right = left + n * n,
               *cross = T + (size_t)a.stage * n * n;
        for (i = 0; i < a.left; ++i) {
            double value = weight * v[i];
            for (j = 0; j <= i; ++j)
                left[col[i] * n + col[j]] += value * v[j];
            for (j = a.left; j < a.count; ++j)
                cross[col[j] * n + col[i]] += value * v[j];
        }
        for (i = a.left; i < a.count; ++i) {
            double value = weight * v[i];
            for (j = a.left; j <= i; ++j)
                right[col[i] * n + col[j]] += value * v[j];
        }
    }
    for (k = 0; k < w->K; ++k) {
        double *root = D + (size_t)k * n * n;
        if (!(w->active_count[k] == n
                  ? cholesky(root, n,
                             w->patterns && w->sparse_stage[k] ? w->patterns + k * n : NULL)
                  : cholesky_active(w, k)))
            return mode == ICORE_OCP_AUTO ? factor_recovery(w, reg) : 0;
        if (k < w->K - 1) {
            if (w->active_count[k] != n || w->active_count[k + 1] != n) {
                schur_active(w, k);
                continue;
            }
            double *link = T + (size_t)k * n * n, *u = w->qr, *next = root + n * n;
            for (i = 0; i < n; ++i)
                for (j = 0; j < n; ++j)
                    u[i * n + j] = link[j * n + i];
            for (i = 0; i < n; ++i) {
                double inverse = 1. / root[i * n + i];
                for (j = 0; j < i; ++j)
                    if (root[i * n + j] != 0.)
                        axpy(u + i * n, u + j * n, -root[i * n + j], n);
                for (j = 0; j < n; ++j)
                    u[i * n + j] *= inverse;
            }
            for (i = 0; i < n; ++i)
                for (j = 0; j < n; ++j) {
                    if (u[i * n + j] != 0.)
                        axpy(next + j * n, u + i * n, -u[i * n + j], j + 1);
                    link[j * n + i] = u[i * n + j];
                }
        }
    }
    w->use_qr = 0;
    return 1;
}
#if defined(__AVX2__) && !defined(CHAIN_SCALAR) && (defined(__GNUC__) || defined(__clang__))
/* Match the scalar C/C++ expression's contraction policy. */
INLINE __m256d chain_lane_madd(__m256d a, __m256d b, __m256d c)
{
    return _mm256_setr_pd(c[0] + a[0] * b[0], c[1] + a[1] * b[1], c[2] + a[2] * b[2],
                          c[3] + a[3] * b[3]);
}
#endif
/* y -= A^T*x, keeping small output blocks in registers across all rows.
 * Each component sees the same row order and multiply-add as the AXPY path.
 * Larger blocks and portable builds retain the original operation. */
static void transpose_subtract(const double *RESTRICT A, const double *RESTRICT x,
                               double *RESTRICT y, int n)
{
    int i, j;
    /* At least two vector accumulators are needed to hide dependency latency;
     * retain the original kernel for shorter blocks. */
    if (n < 8) {
        for (i = 0; i < n; ++i)
            axpy(y, A + (size_t)i * n, -x[i], n);
        return;
    }
#if defined(__AVX2__) && !defined(CHAIN_SCALAR) && (defined(__GNUC__) || defined(__clang__))
#define CHAIN_ACC(acc, v, a) acc = chain_lane_madd(v, a, acc)
#define CHAIN_LOAD_1 __m256d s0 = _mm256_loadu_pd(y + 0);
#define CHAIN_UPDATE_1 CHAIN_ACC(s0, v, _mm256_loadu_pd(a + 0));
#define CHAIN_STORE_1 _mm256_storeu_pd(y + 0, s0);
#define CHAIN_LOAD_2 CHAIN_LOAD_1 __m256d s1 = _mm256_loadu_pd(y + 4);
#define CHAIN_UPDATE_2 CHAIN_UPDATE_1 CHAIN_ACC(s1, v, _mm256_loadu_pd(a + 4));
#define CHAIN_STORE_2 CHAIN_STORE_1 _mm256_storeu_pd(y + 4, s1);
#define CHAIN_LOAD_3 CHAIN_LOAD_2 __m256d s2 = _mm256_loadu_pd(y + 8);
#define CHAIN_UPDATE_3 CHAIN_UPDATE_2 CHAIN_ACC(s2, v, _mm256_loadu_pd(a + 8));
#define CHAIN_STORE_3 CHAIN_STORE_2 _mm256_storeu_pd(y + 8, s2);
#define CHAIN_LOAD_4 CHAIN_LOAD_3 __m256d s3 = _mm256_loadu_pd(y + 12);
#define CHAIN_UPDATE_4 CHAIN_UPDATE_3 CHAIN_ACC(s3, v, _mm256_loadu_pd(a + 12));
#define CHAIN_STORE_4 CHAIN_STORE_3 _mm256_storeu_pd(y + 12, s3);
#define CHAIN_LOAD_5 CHAIN_LOAD_4 __m256d s4 = _mm256_loadu_pd(y + 16);
#define CHAIN_UPDATE_5 CHAIN_UPDATE_4 CHAIN_ACC(s4, v, _mm256_loadu_pd(a + 16));
#define CHAIN_STORE_5 CHAIN_STORE_4 _mm256_storeu_pd(y + 16, s4);
#define CHAIN_LOAD_6 CHAIN_LOAD_5 __m256d s5 = _mm256_loadu_pd(y + 20);
#define CHAIN_UPDATE_6 CHAIN_UPDATE_5 CHAIN_ACC(s5, v, _mm256_loadu_pd(a + 20));
#define CHAIN_STORE_6 CHAIN_STORE_5 _mm256_storeu_pd(y + 20, s5);
#define CHAIN_LOAD_7 CHAIN_LOAD_6 __m256d s6 = _mm256_loadu_pd(y + 24);
#define CHAIN_UPDATE_7 CHAIN_UPDATE_6 CHAIN_ACC(s6, v, _mm256_loadu_pd(a + 24));
#define CHAIN_STORE_7 CHAIN_STORE_6 _mm256_storeu_pd(y + 24, s6);
#define CHAIN_LOAD_8 CHAIN_LOAD_7 __m256d s7 = _mm256_loadu_pd(y + 28);
#define CHAIN_UPDATE_8 CHAIN_UPDATE_7 CHAIN_ACC(s7, v, _mm256_loadu_pd(a + 28));
#define CHAIN_STORE_8 CHAIN_STORE_7 _mm256_storeu_pd(y + 28, s7);
#define CHAIN_CASE(k)                                                                              \
    case k: {                                                                                      \
        CHAIN_LOAD_##k for (i = 0; i < n; ++i)                                                     \
        {                                                                                          \
            const double *a = A + (size_t)i * n;                                                   \
            __m256d v = _mm256_set1_pd(-x[i]);                                                     \
            CHAIN_UPDATE_##k                                                                       \
        }                                                                                          \
        CHAIN_STORE_##k break;                                                                     \
    }
    switch (n / 4) {
        CHAIN_CASE(1)
        CHAIN_CASE(2)
        CHAIN_CASE(3)
        CHAIN_CASE(4) CHAIN_CASE(5) CHAIN_CASE(6) CHAIN_CASE(7) CHAIN_CASE(8) default
            : for (i = 0; i < n; ++i) axpy(y, A + (size_t)i * n, -x[i], n);
        return;
    }
    for (j = n - (n % 4); j < n; ++j) {
        double sum = y[j];
        for (i = 0; i < n; ++i) {
            sum += (-x[i]) * A[(size_t)i * n + j];
            /* Keep each scalar update at the original contraction boundary;
             * do not vectorize products first and then reduce them. */
            __asm__ __volatile__("" : "+x"(sum));
        }
        y[j] = sum;
    }
#undef CHAIN_CASE
#undef CHAIN_ACC
#undef CHAIN_LOAD_1
#undef CHAIN_UPDATE_1
#undef CHAIN_STORE_1
#undef CHAIN_LOAD_2
#undef CHAIN_UPDATE_2
#undef CHAIN_STORE_2
#undef CHAIN_LOAD_3
#undef CHAIN_UPDATE_3
#undef CHAIN_STORE_3
#undef CHAIN_LOAD_4
#undef CHAIN_UPDATE_4
#undef CHAIN_STORE_4
#undef CHAIN_LOAD_5
#undef CHAIN_UPDATE_5
#undef CHAIN_STORE_5
#undef CHAIN_LOAD_6
#undef CHAIN_UPDATE_6
#undef CHAIN_STORE_6
#undef CHAIN_LOAD_7
#undef CHAIN_UPDATE_7
#undef CHAIN_STORE_7
#undef CHAIN_LOAD_8
#undef CHAIN_UPDATE_8
#undef CHAIN_STORE_8
#else
    (void)j;
    for (i = 0; i < n; ++i)
        axpy(y, A + (size_t)i * n, -x[i], n);
#endif
}

static void linear_solve(chain_workspace *w, const double *b, double *x)
{
    int k, i, j, n = w->n;
    if (w->saddle && w->saddle->valid) {
        double reg = w->saddle->regularization;
        if (chain_saddle_reduced(w, b, x))
            return;
        if (!factor_qr(w, reg)) {
            for (i = 0; i < w->nv; ++i)
                x[i] = NAN;
            return;
        }
    }
    memmove(x, b, (size_t)w->nv * sizeof(double));
    for (k = 0; k < w->K; ++k) {
        int count = w->active_count[k];
        const int *idx = w->active + k * n;
        const double *L = w->D + (size_t)k * n * n;
        if (k) {
            const double *T = w->link + (size_t)(k - 1) * n * n;
            const int *prev_idx = idx - n;
            int prev_count = w->active_count[k - 1];
            if (count == n && prev_count == n) {
                for (i = 0; i < n; ++i)
                    x[k * n + i] -= dot(T + i * n, x + (k - 1) * n, n);
            } else
                for (i = 0; i < count; ++i) {
                    double sum = 0.;
                    int ii = idx[i];
                    for (j = 0; j < prev_count; ++j)
                        sum += T[ii * n + prev_idx[j]] * x[(k - 1) * n + prev_idx[j]];
                    x[k * n + ii] -= sum;
                }
        }
        if (count == n)
            forward(L, x + k * n, n);
        else {
            const int *pos = w->position + k * n;
            for (i = 0; i < n; ++i)
                if (pos[i] < 0)
                    x[k * n + i] /= L[i * n + i];
            for (i = 0; i < count; ++i) {
                int ii = idx[i];
                double sum = 0.;
                for (j = 0; j < i; ++j)
                    sum += L[ii * n + idx[j]] * x[k * n + idx[j]];
                x[k * n + ii] = (x[k * n + ii] - sum) / L[ii * n + ii];
            }
        }
    }
    for (k = w->K - 1; k >= 0; --k) {
        int count = w->active_count[k];
        const int *idx = w->active + k * n;
        const double *L = w->D + (size_t)k * n * n;
        if (k < w->K - 1) {
            const double *T = w->link + (size_t)k * n * n;
            const int *next_idx = idx + n;
            int next_count = w->active_count[k + 1];
            if (count == n && next_count == n) {
                transpose_subtract(T, x + (k + 1) * n, x + k * n, n);
            } else if (count == n) {
                for (i = 0; i < next_count; ++i) {
                    int ii = next_idx[i];
                    axpy(x + k * n, T + ii * n, -x[(k + 1) * n + ii], n);
                }
            } else
                for (i = 0; i < next_count; ++i) {
                    int ii = next_idx[i];
                    double value = -x[(k + 1) * n + ii];
                    for (j = 0; j < count; ++j)
                        x[k * n + idx[j]] += value * T[ii * n + idx[j]];
                }
        }
        if (count == n)
            backward(L, x + k * n, n);
        else {
            const int *pos = w->position + k * n;
            for (i = 0; i < n; ++i)
                if (pos[i] < 0)
                    x[k * n + i] /= L[i * n + i];
            for (i = count - 1; i >= 0; --i) {
                int ii = idx[i];
                x[k * n + ii] /= L[ii * n + ii];
                for (j = 0; j < i; ++j)
                    x[k * n + idx[j]] -= x[k * n + ii] * L[ii * n + idx[j]];
            }
        }
    }
}

static void residual(chain_workspace *w, double *error)
{
    int i, nv = w->nv, nd = w->nd, m = w->m;
    double *x = w->state, *t = x + nv, *s = t + nd, *z = s + m;
    double *bx = w->target, *bt = bx + nv, *bp = bt + nd;
    for (i = 0; i < nv; ++i)
        bx[i] = -w->h[i] * x[i] - w->c[i];
    transpose(w, z, bx, -1.);
    for (i = 0; i < nd; ++i)
        bt[i] = z[i] + z[nd + i] - w->lambda[i];
    apply(w, x, t, bp, 1);
    error[0] = 0.;
    error[1] = fmax(maximum(bx, nv), maximum(bt, nd));
    error[2] = 0.;
    for (i = 0; i < m; ++i) {
        double a;
        bp[i] = -bp[i] - s[i];
        a = fabs(bp[i]);
        if (!isfinite(a))
            error[0] = INFINITY;
        else if (a > error[0])
            error[0] = a;
        if (-s[i] > error[0])
            error[0] = -s[i];
        if (-z[i] > error[1])
            error[1] = -z[i];
        error[2] += fabs(s[i] * z[i]);
    }
    if (!isfinite(error[2]) || !isfinite(maximum(w->state, w->dim)))
        error[0] = error[1] = error[2] = INFINITY;
}

static void solve_rhs(chain_workspace *w, const double *rhs, double *step)
{
    int i, nv = w->nv, nd = w->nd, m = w->m;
    if (w->saddle && w->saddle->valid) {
        double reg = w->saddle->regularization;
        if (chain_saddle_rhs(w, rhs, step))
            return;
        /* A failed augmented solve may have partly written the direction.
         * D/link are not its factors: rebuild original QR before reuse. */
        if (!factor_qr(w, reg)) {
            for (i = 0; i < w->dim; ++i)
                step[i] = NAN;
            return;
        }
    }
    const double *bx = rhs, *bt = bx + nv, *bp = bt + nd, *bc = bp + m;
    const double *s = w->state + nv + nd, *z = s + m;
    double *dx = step, *dt = dx + nv, *ds = dt + nd, *dz = ds + m, *q = w->q, *reduced = w->rhs;
    memcpy(reduced, bx, (size_t)nv * sizeof(double));
    for (i = 0; i < m; ++i)
        q[i] = (bc[i] - z[i] * bp[i]) / s[i];
    for (i = 0; i < nd; ++i) {
        double inv = w->inverse[i], wp = w->weight[i], wm = w->weight[nd + i];
        w->force[i] = 2. * (wm * inv) * q[i] - 2. * (wp * inv) * q[nd + i] + w->ratio[i] * bt[i];
        row_add(w, i, -w->force[i], reduced);
    }
    for (i = nd; i < w->nr; ++i)
        row_add(w, i, -q[nd + i], reduced);
    linear_solve(w, reduced, dx);
    for (i = 0; i < nd; ++i) {
        double value = row_dot(w, i, dx), difference = w->force[i] + w->curvature[i] * value;
        dt[i] = w->inverse[i] * (bt[i] + q[i] + q[nd + i]) - w->ratio[i] * value;
        ds[i] = bp[i] - (value - dt[i]);
        ds[nd + i] = bp[nd + i] - (-value - dt[i]);
        dz[i] = .5 * (difference - bt[i]);
        dz[nd + i] = .5 * (-difference - bt[i]);
    }
    for (i = nd; i < w->nr; ++i) {
        double value = row_dot(w, i, dx);
        int j = nd + i;
        ds[j] = bp[j] - value;
        dz[j] = q[j] + w->weight[j] * value;
    }
}

static double newton_error(chain_workspace *w, const double *rhs, const double *step, double *e,
                           const double scales[4])
{
    int i, nv = w->nv, nd = w->nd, m = w->m;
    const double *dx = step, *dt = dx + nv, *ds = dt + nd, *dz = ds + m, *s = w->state + nv + nd,
                 *z = s + m;
    double *ex = e, *et = ex + nv, *ep = et + nd, *ec = ep + m;
    for (i = 0; i < nv; ++i)
        ex[i] = rhs[i] - w->h[i] * dx[i];
    transpose(w, dz, ex, -1.);
    for (i = 0; i < nd; ++i)
        et[i] = rhs[nv + i] + dz[i] + dz[nd + i];
    apply(w, dx, dt, ep, 0);
    for (i = 0; i < m; ++i) {
        ep[i] = rhs[nv + nd + i] - ep[i] - ds[i];
        ec[i] = rhs[nv + nd + m + i] - z[i] * ds[i] - s[i] * dz[i];
    }
    return fmax(fmax(maximum(ex, nv) / scales[0], maximum(et, nd) / scales[1]),
                fmax(maximum(ep, m) / scales[2], maximum(ec, m) / scales[3]));
}

/* Refine the current best step/error pair.  This helper deliberately does
 * not compute another initial solution of the original RHS. */
static double refine_direction(chain_workspace *w, const double *rhs, double *step, int refinements,
                               double limit, int *count, const double scales[4], double norm)
{
    int j, i;
    double candidate;
    for (j = 0; j < refinements && norm > limit; ++j) {
        solve_rhs(w, w->error, w->correction);
        for (i = 0; i < w->dim; ++i)
            w->trial[i] = step[i] + w->correction[i];
        candidate = newton_error(w, rhs, w->trial, w->trial_error, scales);
        ++*count;
        if (!(candidate < norm))
            break;
        memcpy(step, w->trial, (size_t)w->dim * sizeof(double));
        memcpy(w->error, w->trial_error, (size_t)w->dim * sizeof(double));
        norm = candidate;
    }
    return norm;
}

static double direction(chain_workspace *w, const double *rhs, double *step, int refinements,
                        double limit, int *count)
{
    double scales[4], norm;
    scales[0] = 1. + maximum(rhs, w->nv);
    scales[1] = 1. + maximum(rhs + w->nv, w->nd);
    scales[2] = 1. + maximum(rhs + w->nv + w->nd, w->m);
    scales[3] = 1. + maximum(rhs + w->nv + w->nd + w->m, w->m);
    solve_rhs(w, rhs, step);
    norm = newton_error(w, rhs, step, w->error, scales);
    return refine_direction(w, rhs, step, refinements, limit, count, scales, norm);
}

#include "chain_driver.h"
