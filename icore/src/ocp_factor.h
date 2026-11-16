/* Stage cost assembly and Riccati factorization.
 * Classical, square-root, packed, and mixed-precision paths share one model.
 * Local equilibration and QR are recovery paths, not changes to the final KKT test.
 * Private implementation: included exactly once by ocp.c. */

/* Local algebra -> stage assembly -> Riccati factorization.
 * Numeric recovery and the original definition order are unchanged. */
/* ---- Cost roots, guarded Cholesky, equilibration, and triangular solves. ---- */

/* Reference semidefinite Cholesky, retained for difficult cancellation. */
static int cost_root_reference(const double *H, double *R, int n)
{
    double scale = norm_inf(H, (size_t)n * n), tolerance = 128.0 * DBL_EPSILON * fmax(1.0, scale);
    int i, j, k;
    if (!isfinite(scale))
        return 0;
    memset(R, 0, (size_t)n * n * sizeof(double));
    for (j = 0; j < n; ++j) {
        double pivot = H[(size_t)j * n + j];
        for (k = 0; k < j; ++k)
            pivot -= R[(size_t)k * n + j] * R[(size_t)k * n + j];
        if (!isfinite(pivot) || pivot < -tolerance)
            return 0;
        if (pivot <= 0.0) {
            for (i = j + 1; i < n; ++i) {
                double r = H[(size_t)j * n + i];
                for (k = 0; k < j; ++k)
                    r -= R[(size_t)k * n + j] * R[(size_t)k * n + i];
                if (!isfinite(r) || fabs(r) > 8.0 * tolerance)
                    return 0;
            }
            continue;
        }
        R[(size_t)j * n + j] = sqrt(pivot);
        for (i = j + 1; i < n; ++i) {
            double r = H[(size_t)j * n + i];
            for (k = 0; k < j; ++k)
                r -= R[(size_t)k * n + j] * R[(size_t)k * n + i];
            if (!isfinite(r))
                return 0;
            R[(size_t)j * n + i] = r / R[(size_t)j * n + j];
            if (!isfinite(R[(size_t)j * n + i]))
                return 0;
        }
    }
    return 1;
}

/* The lower triangle holds the evolving Schur complement; completed upper
 * rows hold the cost square root. Contiguous rank-one updates replace the
 * reference's strided inner products, without flooring positive pivots. */
static int cost_root(const double *H, double *R, int n)
{
    double magnitude = norm_inf(H, (size_t)n * n);
    double tolerance = 128.0 * DBL_EPSILON * fmax(1.0, magnitude);
    int i, j, first = 0, diagonal = 1;
    if (!isfinite(magnitude))
        return 0;
    for (i = 0; i < n && diagonal; ++i)
        if (structural_first_nonzero(H + (size_t)i * n + i + 1, n - i - 1) != n - i - 1)
            diagonal = 0;
    if (diagonal) {
        memset(R, 0, (size_t)n * n * sizeof(double));
        for (i = 0; i < n; ++i) {
            double pivot = H[(size_t)i * n + i];
            if (pivot < -tolerance)
                return 0;
            if (pivot > 0.0)
                R[(size_t)i * n + i] = sqrt(pivot);
        }
        return 1;
    }
#if defined(ICORE_OCP_REFERENCE_COST_ROOT) && ICORE_OCP_REFERENCE_COST_ROOT
    return cost_root_reference(H, R, n);
#endif
    if (n < 16)
        return cost_root_reference(H, R, n);
    for (i = 0; i < n; ++i)
        for (j = 0; j <= i; ++j)
            R[(size_t)i * n + j] = H[(size_t)j * n + i];
    /* Exact zero leading rows leave no Schur update. Keep the original
     * whole-matrix magnitude and semidefinite tolerance above. */
    while (first < n &&
           structural_first_nonzero(H + (size_t)first * n + first, n - first) == n - first)
        ++first;
    for (i = 0; i < first; ++i)
        memset(R + (size_t)i * n, 0, (size_t)n * sizeof(double));
    for (j = first; j < n; ++j) {
        double pivot = R[(size_t)j * n + j];
        if (!isfinite(pivot) || pivot < -tolerance)
            return cost_root_reference(H, R, n);
        if (pivot <= 0.0) {
            R[(size_t)j * n + j] = 0.0;
            for (i = j + 1; i < n; ++i) {
                if (!isfinite(R[(size_t)i * n + j]) || fabs(R[(size_t)i * n + j]) > 8.0 * tolerance)
                    return cost_root_reference(H, R, n);
                R[(size_t)j * n + i] = 0.0;
            }
            continue;
        }
        pivot = sqrt(pivot);
        R[(size_t)j * n + j] = pivot;
        for (i = j + 1; i < n; ++i)
            R[(size_t)j * n + i] = R[(size_t)i * n + j] / pivot;
        for (i = j + 1; i < n; ++i)
            axpy(R + (size_t)i * n + j + 1, R + (size_t)j * n + j + 1, -R[(size_t)j * n + i],
                 i - j);
    }
    for (i = 1; i < n; ++i)
        memset(R + (size_t)i * n, 0, (size_t)i * sizeof(double));
    if (!finite_array(R, (size_t)n * n))
        return cost_root_reference(H, R, n);
    return 1;
}

/* Current exact row spans certify every off-diagonal coefficient is zero.
 * This reuses the descriptor already required for original-H products. */
static int stage_cost_is_diagonal(const icore_ocp_workspace *w, int stage)
{
    int i;
    size_t first = (size_t)stage * w->nv;
    for (i = 0; i < w->nv; ++i) {
        int length = w->cost_length[first + i];
        if (length && (length != 1 || w->cost_start[first + i] != i))
            return 0;
    }
    return 1;
}

/* known_zero certifies this owned root's off-diagonal bytes, not its
 * coefficients. Every diagonal is replaced, including zeros and negative
 * values accepted by the unchanged semidefinite tolerance. */
static int diagonal_cost_root_storage(const double *H, double *R, int n, int known_zero)
{
    double magnitude = 0.0, tolerance;
    int i;
    for (i = 0; i < n; ++i) {
        double pivot = H[(size_t)i * n + i];
        if (!isfinite(pivot))
            return 0;
        magnitude = fmax(magnitude, fabs(pivot));
    }
    tolerance = 128.0 * DBL_EPSILON * fmax(1.0, magnitude);
    if (!known_zero)
        memset(R, 0, (size_t)n * n * sizeof(double));
    else
        for (i = 0; i < n; ++i)
            R[(size_t)i * n + i] = 0.0;
    for (i = 0; i < n; ++i) {
        double pivot = H[(size_t)i * n + i];
        if (pivot < -tolerance)
            return 0;
        if (pivot > 0.0)
            R[(size_t)i * n + i] = sqrt(pivot);
    }
    return 1;
}

/* Standalone checked construction starts without an owned-zero certificate. */
static int diagonal_cost_root(const double *H, double *R, int n)
{
    return diagonal_cost_root_storage(H, R, n, 0);
}

/* Four-pivot panels reuse the trailing tile in registers for four rank-one
 * updates. The packed panel occupies 4*n doubles in existing QR scratch;
 * QR and Cholesky use that scratch at disjoint times. */
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
static int cholesky_panel4(double *A, int n, double floor, int *poor, double *panel)
{
    int b, i, j, k;
    for (b = 0; b < n; b += 4) {
        int end = b + 4 < n ? b + 4 : n;
        for (j = b; j < end; ++j) {
            double pivot = A[(size_t)j * n + j];
            for (k = b; k < j; ++k)
                pivot -= A[(size_t)j * n + k] * A[(size_t)j * n + k];
            if (!isfinite(pivot))
                return 0;
            if (pivot < floor) {
                pivot = floor;
                if (poor)
                    *poor = 1;
            }
            pivot = sqrt(pivot);
            A[(size_t)j * n + j] = pivot;
            for (i = j + 1; i < n; ++i) {
                double value = A[(size_t)i * n + j];
                for (k = b; k < j; ++k)
                    value -= A[(size_t)i * n + k] * A[(size_t)j * n + k];
                if (!isfinite(value))
                    return 0;
                A[(size_t)i * n + j] = value / pivot;
            }
        }
        for (k = b; k < end; ++k)
            for (i = end; i < n; ++i)
                panel[(size_t)(k - b) * n + i] = A[(size_t)i * n + k];
        i = end;
        for (; i + 3 < n; i += 4) {
            j = end;
            for (; j <= i && j + 7 < n; j += 8) {
                __m256d r00 = _mm256_loadu_pd(A + (size_t)i * n + j);
                __m256d r01 = _mm256_loadu_pd(A + (size_t)i * n + j + 4);
                __m256d r10 = _mm256_loadu_pd(A + (size_t)(i + 1) * n + j);
                __m256d r11 = _mm256_loadu_pd(A + (size_t)(i + 1) * n + j + 4);
                __m256d r20 = _mm256_loadu_pd(A + (size_t)(i + 2) * n + j);
                __m256d r21 = _mm256_loadu_pd(A + (size_t)(i + 2) * n + j + 4);
                __m256d r30 = _mm256_loadu_pd(A + (size_t)(i + 3) * n + j);
                __m256d r31 = _mm256_loadu_pd(A + (size_t)(i + 3) * n + j + 4);
                for (k = 0; k < end - b; ++k) {
                    const double *p = panel + (size_t)k * n;
                    __m256d v0 = _mm256_loadu_pd(p + j), v1 = _mm256_loadu_pd(p + j + 4), x;
                    x = _mm256_set1_pd(p[i]);
                    r00 = _mm256_fnmadd_pd(x, v0, r00);
                    r01 = _mm256_fnmadd_pd(x, v1, r01);
                    x = _mm256_set1_pd(p[i + 1]);
                    r10 = _mm256_fnmadd_pd(x, v0, r10);
                    r11 = _mm256_fnmadd_pd(x, v1, r11);
                    x = _mm256_set1_pd(p[i + 2]);
                    r20 = _mm256_fnmadd_pd(x, v0, r20);
                    r21 = _mm256_fnmadd_pd(x, v1, r21);
                    x = _mm256_set1_pd(p[i + 3]);
                    r30 = _mm256_fnmadd_pd(x, v0, r30);
                    r31 = _mm256_fnmadd_pd(x, v1, r31);
                }
                _mm256_storeu_pd(A + (size_t)i * n + j, r00);
                _mm256_storeu_pd(A + (size_t)i * n + j + 4, r01);
                _mm256_storeu_pd(A + (size_t)(i + 1) * n + j, r10);
                _mm256_storeu_pd(A + (size_t)(i + 1) * n + j + 4, r11);
                _mm256_storeu_pd(A + (size_t)(i + 2) * n + j, r20);
                _mm256_storeu_pd(A + (size_t)(i + 2) * n + j + 4, r21);
                _mm256_storeu_pd(A + (size_t)(i + 3) * n + j, r30);
                _mm256_storeu_pd(A + (size_t)(i + 3) * n + j + 4, r31);
            }
            {
                int r, c;
                for (r = i; r < i + 4; ++r)
                    for (c = j; c <= r; ++c) {
                        double value = A[(size_t)r * n + c];
                        for (k = 0; k < end - b; ++k)
                            value -= panel[(size_t)k * n + r] * panel[(size_t)k * n + c];
                        A[(size_t)r * n + c] = value;
                    }
            }
        }
        for (; i < n; ++i)
            for (k = 0; k < end - b; ++k)
                axpy(A + (size_t)i * n + end, panel + (size_t)k * n + end,
                     -panel[(size_t)k * n + i], i - end + 1);
    }
    return 1;
}
#endif

/* In-place lower Cholesky with a local, scale-aware positive pivot floor.
 * column is scratch with at least 4*n doubles when n>=32. */
static int cholesky(double *A, int n, double absolute, int *poor, double *column)
{
    double maxdiag = 0.0, floor;
    int i, j, k;
    for (i = 0; i < n; ++i)
        maxdiag = fmax(maxdiag, fabs(A[(size_t)i * n + i]));
    floor = fmax(absolute, 1e-12 * maxdiag);
    floor = fmax(floor, DBL_MIN * 1024.0);
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
    if (n >= 32)
        return cholesky_panel4(A, n, floor, poor, column);
#endif
    if (n >= 16) {
        /* Right-looking rank-one updates use contiguous SIMD rows and avoid
         * one horizontal reduction for every off-diagonal entry. Each entry
         * still receives its subtractions in increasing pivot order. */
        for (j = 0; j < n; ++j) {
            double pivot = A[(size_t)j * n + j];
            if (!isfinite(pivot))
                return 0;
            if (pivot < floor) {
                pivot = floor;
                if (poor)
                    *poor = 1;
            }
            pivot = sqrt(pivot);
            A[(size_t)j * n + j] = pivot;
            for (i = j + 1; i < n; ++i) {
                double value = A[(size_t)i * n + j];
                if (!isfinite(value))
                    return 0;
                column[i] = A[(size_t)i * n + j] = value / pivot;
            }
            for (i = j + 1; i < n; ++i)
                axpy(A + (size_t)i * n + j + 1, column + j + 1, -column[i], i - j);
        }
        return 1;
    }
    for (i = 0; i < n; ++i) {
        for (j = 0; j <= i; ++j) {
            double v = A[(size_t)i * n + j];
            for (k = 0; k < j; ++k)
                v -= A[(size_t)i * n + k] * A[(size_t)j * n + k];
            if (!isfinite(v))
                return 0;
            if (i == j) {
                if (v < floor) {
                    v = floor;
                    if (poor)
                        *poor = 1;
                }
                A[(size_t)i * n + i] = sqrt(v);
            } else
                A[(size_t)i * n + j] = v / A[(size_t)j * n + j];
        }
    }
    return 1;
}

/* Retry a poor positive control block in diagonally equilibrated coordinates.
 * M is the unchanged, already regularized physical matrix.  Nothing is added
 * in scaled coordinates.  A successful retry must use no pivot flooring and
 * must preserve the absolute physical pivot floor after L=D*Lhat is restored.
 * No workspace is allocated: scale has n entries and column has 4*n entries.
 * On rejection L is scratch; the caller reconstructs its classical factor. */
static int cholesky_equilibrated(const double *M, int stride, double *L, int n, double absolute,
                                 double *scale, double *column)
{
    double physical_floor = sqrt(fmax(absolute, 1024.0 * DBL_MIN));
    int i, j, poor = 0;
    for (i = 0; i < n; ++i) {
        double diagonal = M[(size_t)i * stride + i];
        if (!(diagonal > 0.0) || !isfinite(diagonal))
            return 0;
        scale[i] = sqrt(diagonal);
        if (!(scale[i] > 0.0) || !isfinite(scale[i]))
            return 0;
    }
    for (i = 0; i < n; ++i)
        for (j = 0; j <= i; ++j) {
            double value = (M[(size_t)i * stride + j] / scale[i]) / scale[j];
            if (!isfinite(value))
                return 0;
            L[(size_t)i * n + j] = value;
        }
    if (!cholesky(L, n, 0.0, &poor, column) || poor)
        return 0;
    for (i = 0; i < n; ++i) {
        for (j = 0; j <= i; ++j) {
            double value = L[(size_t)i * n + j] * scale[i];
            if (!isfinite(value))
                return 0;
            L[(size_t)i * n + j] = value;
        }
        if (!(L[(size_t)i * n + i] >= physical_floor))
            return 0;
    }
    return 1;
}

static void triangular_forward(const double *L, double *x, int n)
{
    int i, j;
    for (i = 0; i < n; ++i) {
        double v = x[i];
        if (n >= 16)
            v -= dot(L + (size_t)i * n, x, i);
        else
            for (j = 0; j < i; ++j)
                v -= L[(size_t)i * n + j] * x[j];
        x[i] = v / L[(size_t)i * n + i];
    }
}

static void triangular_backward(const double *L, double *x, int n)
{
    int i, j;
    if (n >= 16) {
        /* Row updates avoid the strided column reads of a wide L' solve. */
        for (i = n - 1; i >= 0; --i) {
            x[i] /= L[(size_t)i * n + i];
            axpy(x, L + (size_t)i * n, -x[i], i);
        }
        return;
    }
    for (i = n - 1; i >= 0; --i) {
        double v = x[i];
        for (j = i + 1; j < n; ++j)
            v -= L[(size_t)j * n + i] * x[j];
        x[i] = v / L[(size_t)i * n + i];
    }
}

/* ---- Barrier-weighted stage Hessians and diagonal structure handling. ---- */

/* Cache W^-1 G and H + G' W^-2 G.  Epigraph pairs are condensed here.
 * Their e curvature is 4 min(w+,w-)/(1+min/max), with no large subtraction. */

static void dense_constraint_gram(icore_ocp_workspace *w, int stage, const double *Gh, double *H)
{
    int nv = w->nv, start = stage == w->N ? w->nu : w->ne, n = nv - start;
    int count = w->dense_count[stage], i = 0, r;
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
    int j;
#endif
    const int *rows = w->dense_rows + (size_t)stage * w->m;
    if (!count)
        return;
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
    for (; i + 3 < n; i += 4)
        for (j = 0; j <= i; j += 4) {
            __m256d h0 = _mm256_loadu_pd(H + (size_t)(start + i) * nv + start + j);
            __m256d h1 = _mm256_loadu_pd(H + (size_t)(start + i + 1) * nv + start + j);
            __m256d h2 = _mm256_loadu_pd(H + (size_t)(start + i + 2) * nv + start + j);
            __m256d h3 = _mm256_loadu_pd(H + (size_t)(start + i + 3) * nv + start + j);
            for (r = 0; r < count; ++r) {
                const double *g = Gh + (size_t)rows[r] * nv + start;
                __m256d v = _mm256_loadu_pd(g + j);
                h0 = _mm256_fmadd_pd(_mm256_set1_pd(g[i]), v, h0);
                h1 = _mm256_fmadd_pd(_mm256_set1_pd(g[i + 1]), v, h1);
                h2 = _mm256_fmadd_pd(_mm256_set1_pd(g[i + 2]), v, h2);
                h3 = _mm256_fmadd_pd(_mm256_set1_pd(g[i + 3]), v, h3);
            }
            _mm256_storeu_pd(H + (size_t)(start + i) * nv + start + j, h0);
            _mm256_storeu_pd(H + (size_t)(start + i + 1) * nv + start + j, h1);
            _mm256_storeu_pd(H + (size_t)(start + i + 2) * nv + start + j, h2);
            _mm256_storeu_pd(H + (size_t)(start + i + 3) * nv + start + j, h3);
        }
#endif
    for (; i < n; ++i)
        for (r = 0; r < count; ++r) {
            const double *g = Gh + (size_t)rows[r] * nv + start;
            axpy(H + (size_t)(start + i) * nv + start, g, g[i], i + 1);
        }
}

/* Direct singleton curvature needs no square roots in the normal path.
 * Full-Hessian recovery materializes the original roots before using Gh. */
static void materialize_orthant_roots(icore_ocp_workspace *w)
{
    size_t i = 0, count = (size_t)w->K * w->m;
    if (!w->orthant_roots_lazy)
        return;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    for (; count - i >= 4; i += 4)
        _mm256_storeu_pd(w->orthant_inv + i, _mm256_sqrt_pd(_mm256_loadu_pd(w->lambda + i)));
#endif
    for (; i < count; ++i)
        w->orthant_inv[i] = sqrt(w->lambda[i]);
    w->orthant_roots_lazy = 0;
}

/* Each sparse row contributes one rank-one update in ascending row order.
 * Dense rows are accumulated together by dense_constraint_gram. */
ICORE_OCP_INLINE void add_sparse_row_curvature(icore_ocp_workspace *w, int stage, size_t row, const double *g,
                                          int count, const int *idx, double *H)
{
    int a, b, nv = w->nv;
    if (w->epi_row[row] >= 0 || (count >= 4 && count * 2 >= nv - (stage == w->N ? w->nu : w->ne)))
        return;
    if (count && idx[count - 1] - idx[0] + 1 == count) {
        int first = idx[0];
        for (a = 0; a < count; ++a)
            axpy(H + (size_t)(first + a) * nv + first, g + first, g[first + a], a + 1);
    } else {
        for (a = 0; a < count; ++a) {
            int ia = idx[a];
            double ga = g[ia];
            for (b = 0; b <= a; ++b)
                H[(size_t)ia * nv + idx[b]] += ga * g[idx[b]];
        }
    }
}

/* Scaling only writes the cached supports. Check those same entries while
 * they are still in cache; all other Gh entries remain owned exact zeros. */
ICORE_OCP_INLINE int scaled_row_finite(icore_ocp_workspace *w, size_t row, const double *g, int count,
                                  const int *idx)
{
    int a;
    if (count == 1) {
        w->single_scaled[row] = g[idx[0]];
        return isfinite(w->single_scaled[row]);
    }
    if (count && idx[count - 1] - idx[0] + 1 == count)
        return finite_array(g + idx[0], (size_t)count);
    for (a = 0; a < count; ++a)
        if (!isfinite(g[idx[a]]))
            return 0;
    return 1;
}

/* Exact current singleton pairs share scaling/validation. Writes to Gh and
 * Hb retain the ordinary scalar row order and contraction policy. A negative
 * return rejects nonfinite scaling; zero leaves an unsupported layout alone. */
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR) && (defined(__GNUC__) || defined(__clang__))
ICORE_OCP_INLINE int scale_singleton_pairs8(icore_ocp_workspace *w, size_t row, double *Gh, double *Hb)
{
    int first = w->single_index[row], nv = w->nv;
    if (first < 0 || nv < 4 || first > nv - 4)
        return 0;
    __m256i expected = _mm256_setr_epi32(first, first, first + 1, first + 1, first + 2, first + 2,
                                         first + 3, first + 3);
    __m256i indices = _mm256_loadu_si256((const __m256i *)(w->single_index + row));
    __m256i epi = _mm256_loadu_si256((const __m256i *)(w->epi_row + row));
    if (_mm256_movemask_epi8(_mm256_cmpeq_epi32(indices, expected)) != -1 ||
        _mm256_movemask_epi8(_mm256_cmpeq_epi32(epi, _mm256_set1_epi32(-1))) != -1)
        return 0;
    __m256d g0 = _mm256_mul_pd(_mm256_loadu_pd(w->single_value + row),
                               _mm256_loadu_pd(w->orthant_inv + row));
    __m256d g1 = _mm256_mul_pd(_mm256_loadu_pd(w->single_value + row + 4),
                               _mm256_loadu_pd(w->orthant_inv + row + 4));
    __m256d sign = _mm256_set1_pd(-0.0), maximum = _mm256_set1_pd(DBL_MAX);
    __m256d valid = _mm256_and_pd(_mm256_cmp_pd(_mm256_andnot_pd(sign, g0), maximum, _CMP_LE_OQ),
                                  _mm256_cmp_pd(_mm256_andnot_pd(sign, g1), maximum, _CMP_LE_OQ));
    if (_mm256_movemask_pd(valid) != 15)
        return -1;
    _mm256_storeu_pd(w->single_scaled + row, g0);
    _mm256_storeu_pd(w->single_scaled + row + 4, g1);
    for (int i = 0; i < 8; ++i) {
        int j = first + i / 2;
        double g = w->single_scaled[row + i];
        Gh[(size_t)i * nv + j] = g;
        Hb[(size_t)j * nv + j] += g * g;
    }
    return 1;
}
#endif

static int form_stage_hessians(icore_ocp_workspace *w)
{
    int k, i, j, a;
    size_t nv = w->nv, m = w->m;
    materialize_orthant_roots(w);
    for (k = 0; k < w->K; ++k) {
        const double *G = w->G + (size_t)k * m * nv;
        double *Gh = w->Gh + (size_t)k * m * nv, *Hb = w->Hb + (size_t)k * nv * nv;
        const double *boost = w->boost + (size_t)k * m;
        memcpy(Hb, w->H + (size_t)k * nv * nv, nv * nv * sizeof(double));
        for (i = 0; i < w->nlp; ++i) {
            size_t row = (size_t)k * m + i;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR) && (defined(__GNUC__) || defined(__clang__))
            if (!w->nsoc && w->nlp - i >= 8) {
                int paired = scale_singleton_pairs8(w, row, Gh + (size_t)i * nv, Hb);
                if (paired < 0)
                    return 0;
                if (paired) {
                    i += 7;
                    continue;
                }
            }
#endif
            int count = w->nnz[row];
            const int *idx = w->indices + w->index_start[row];
            double r = w->orthant_inv[(size_t)k * w->nlp + i];
            if (!w->nsoc && count <= 1) {
                if (count) {
                    double g;
                    j = w->single_index[row];
                    g = w->single_value[row] * r;
                    Gh[(size_t)i * nv + j] = w->single_scaled[row] = g;
                    if (!isfinite(g))
                        return 0;
                    if (w->epi_row[row] < 0)
                        Hb[(size_t)j * nv + j] += g * g;
                }
                continue;
            }
            if (count && idx[count - 1] - idx[0] + 1 == count) {
                int first = idx[0];
                a = 0;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
                __m256d scale = _mm256_set1_pd(r);
                for (; a + 3 < count; a += 4)
                    _mm256_storeu_pd(
                        Gh + (size_t)i * nv + first + a,
                        _mm256_mul_pd(_mm256_loadu_pd(G + (size_t)i * nv + first + a), scale));
#endif
                for (; a < count; ++a)
                    Gh[(size_t)i * nv + first + a] = G[(size_t)i * nv + first + a] * r;
            } else
                for (a = 0; a < count; ++a) {
                    j = idx[a];
                    Gh[(size_t)i * nv + j] = G[(size_t)i * nv + j] * r;
                }
            if (!w->nsoc) {
                const double *g = Gh + (size_t)i * nv;
                if (!scaled_row_finite(w, row, g, count, idx))
                    return 0;
                add_sparse_row_curvature(w, k, row, g, count, idx, Hb);
            }
        }
        for (i = 0; i < w->nsoc; ++i) {
            int o = w->offsets[i], n = w->sizes[i];
            size_t row = (size_t)k * m + o;
            int count = w->nnz[row], active;
            const int *idx = w->indices + w->index_start[row];
            double large = boost[o], small = 1.0 / large;
            double fac = 1.0 / w->scale[(size_t)k * w->nsoc + i];
            /* Apply the same spectral boost to the active matrix columns
             * together.  This avoids a gather/apply/scatter call per column. */
            for (active = 0; active < count; ++active)
                w->tmp[active] = 0.0;
            for (a = 1; a < n; ++a) {
                const double *gr = G + (size_t)(o + a) * nv;
                double axis = boost[o + a];
                if (count && idx[count - 1] - idx[0] + 1 == count)
                    axpy(w->tmp, gr + idx[0], axis, count);
                else
                    for (active = 0; active < count; ++active)
                        w->tmp[active] += axis * gr[idx[active]];
            }
            for (active = 0; active < count; ++active) {
                double p = w->tmp[active], plus, minus;
                j = idx[active];
                plus = small * (G[(size_t)o * nv + j] + p);
                minus = large * (G[(size_t)o * nv + j] - p);
                Gh[(size_t)o * nv + j] = fac * 0.5 * (plus + minus);
                w->tmp2[active] = 0.5 * (plus - minus) - p;
            }
            for (a = 1; a < n; ++a) {
                const double *gr = G + (size_t)(o + a) * nv;
                double *gh = Gh + (size_t)(o + a) * nv;
                double axis = boost[o + a];
                if (count && idx[count - 1] - idx[0] + 1 == count) {
                    int first = idx[0];
                    for (active = 0; active < count; ++active)
                        gh[first + active] = fac * (gr[first + active] + axis * w->tmp2[active]);
                } else
                    for (active = 0; active < count; ++active) {
                        j = idx[active];
                        gh[j] = fac * (gr[j] + axis * w->tmp2[active]);
                    }
            }
        }
        if (w->nsoc)
            for (i = 0; i < w->m; ++i) {
                size_t row = (size_t)k * m + i;
                add_sparse_row_curvature(w, k, row, Gh + (size_t)i * nv, w->nnz[row],
                                         w->indices + w->index_start[row], Hb);
            }
        dense_constraint_gram(w, k, Gh, Hb);
        if (k < w->N)
            for (i = 0; i < w->ne; ++i) {
                size_t epos = (size_t)k * w->ne + i;
                int p = w->epi_plus[epos], n = w->epi_minus[epos], e = w->epi_e[epos];
                double wp = w->orthant_inv[(size_t)k * w->nlp + p],
                       wn = w->orthant_inv[(size_t)k * w->nlp + n];
                wp *= wp;
                wn *= wn;
                double large = fmax(wp, wn), small = fmin(wp, wn), ratio = small / large;
                double sum = wp + wn;
                if (!(sum > 0.0) || !isfinite(sum))
                    return 0;
                w->epi_inv[epos] = 1.0 / sum;
                w->epi_ratio[epos] = (wn - wp) / sum;
                w->epi_curv[epos] = 4.0 * small / (1.0 + ratio);
                Hb[(size_t)e * nv + e] += w->epi_curv[epos];
            }
        /* Control-stage factor kernels consume the lower triangle. Only the
         * terminal block is copied as a full P matrix by classical Riccati. */
        if (k == w->N)
            for (i = 0; i < w->nv; ++i)
                for (j = 0; j < i; ++j)
                    Hb[(size_t)j * nv + i] = Hb[(size_t)i * nv + j];
        /* SOC rows are coupled during scaling; validate after the block pass. */
        if (w->nsoc)
            for (i = 0; i < w->m; ++i) {
                size_t row = (size_t)k * m + i;
                if (!scaled_row_finite(w, row, Gh + (size_t)i * nv, w->nnz[row],
                                       w->indices + w->index_start[row]))
                    return 0;
            }
    }
    return 1;
}

/* Exact one-variable rows add only diagonal curvature. The packed cost
 * cache preserves general H; direct ratios can avoid scaled rows. The
 * guarded fallback restores the original square-root accumulation and Gh.
 * Eligibility follows the validated supports after each model update. */
static int form_stage_diagonal(icore_ocp_workspace *w)
{
    int k, i, j, nv = w->nv, m = w->m;
    if (w->orthant_direct && w->orthant_roots_lazy) {
        /* The packed cost cache contains the full original H. Only the
         * singleton barrier contribution is diagonal. A rejected range
         * restarts every diagonal through the original square-root path. */
        int safe = 1;
        for (k = 0; k < w->K && safe; ++k) {
            const double *H = w->H + (size_t)k * nv * nv;
            double *diagonal = w->ocp_diagonal + (size_t)k * nv;
            for (j = 0; j < nv; ++j)
                diagonal[j] = H[(size_t)j * nv + j];
            for (i = 0; i < m; ++i) {
                size_t row = (size_t)k * m + i;
                double g, curvature;
                if (!w->nnz[row])
                    continue;
                j = w->single_index[row];
                g = w->single_value[row];
                curvature = (w->lambda[row] * g) * g;
                if (!(curvature >= 16.0 * DBL_MIN && curvature <= DBL_MAX / 16.0)) {
                    safe = 0;
                    break;
                }
                diagonal[j] += curvature;
                if (!isfinite(diagonal[j])) {
                    safe = 0;
                    break;
                }
            }
        }
        if (safe)
            return 1;
        materialize_orthant_roots(w);
    }
    for (k = 0; k < w->K; ++k) {
        const double *H = w->H + (size_t)k * nv * nv;
        double *Gh = w->Gh + (size_t)k * m * nv;
        double *diagonal = w->ocp_diagonal + (size_t)k * nv;
        for (j = 0; j < nv; ++j)
            diagonal[j] = H[(size_t)j * nv + j];
        for (i = 0; i < m; ++i) {
            size_t row = (size_t)k * m + i;
            if (w->nnz[row]) {
                double g;
                j = w->indices[w->index_start[row]];
                g = w->single_value[row] * w->orthant_inv[(size_t)k * w->nlp + i];
                if (!isfinite(g))
                    return 0;
                Gh[(size_t)i * nv + j] = g;
                w->single_scaled[row] = g;
                diagonal[j] += g * g;
                if (!isfinite(diagonal[j]))
                    return 0;
            }
        }
    }
    return 1;
}

/* ---- Riccati products and classical / square-root factorization recovery. ---- */

/* Reuse PD scratch (nf<=nx) for C_next D and solve all terminal RHS together.
 * Four-row products share each dynamics load; the scalar path is complete. */

static void factor_terminal_constraints(icore_ocp_workspace *w, int k)
{
    int i, j, a, b;
    int nx = w->nx, nr = w->nr, nf = w->nf, nv = w->nv;
    int initial_zero = k == 0 && w->classical_initial_zero;
    int columns = initial_zero ? (w->nu + 3) / 4 * 4 : nv;
    if (columns > nv)
        columns = nv;
    const double *D = w->D + (size_t)k * nx * nv;
    const double *L = w->L + (size_t)k * nr * nr, *T = w->T + (size_t)k * nx * nr;
    const double *Cn = w->C + (size_t)(k + 1) * nf * nx;
    double *C = w->C + (size_t)k * nf * nx, *Z = w->Z + (size_t)k * nr * nf;
    double *U = w->PD;
    a = 0;
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
    for (; a + 3 < nf; a += 4) {
        j = 0;
        for (; j + 3 < columns; j += 4) {
            __m256d r0 = _mm256_setzero_pd(), r1 = r0, r2 = r0, r3 = r0;
            for (i = 0; i < nx; ++i) {
                __m256d d = _mm256_loadu_pd(D + (size_t)i * nv + j);
                r0 = _mm256_fmadd_pd(_mm256_set1_pd(Cn[(size_t)a * nx + i]), d, r0);
                r1 = _mm256_fmadd_pd(_mm256_set1_pd(Cn[(size_t)(a + 1) * nx + i]), d, r1);
                r2 = _mm256_fmadd_pd(_mm256_set1_pd(Cn[(size_t)(a + 2) * nx + i]), d, r2);
                r3 = _mm256_fmadd_pd(_mm256_set1_pd(Cn[(size_t)(a + 3) * nx + i]), d, r3);
            }
            _mm256_storeu_pd(U + (size_t)a * nv + j, r0);
            _mm256_storeu_pd(U + (size_t)(a + 1) * nv + j, r1);
            _mm256_storeu_pd(U + (size_t)(a + 2) * nv + j, r2);
            _mm256_storeu_pd(U + (size_t)(a + 3) * nv + j, r3);
        }
        for (; j < columns; ++j) {
            double r0 = 0.0, r1 = 0.0, r2 = 0.0, r3 = 0.0;
            for (i = 0; i < nx; ++i) {
                double d = D[(size_t)i * nv + j];
                r0 += Cn[(size_t)a * nx + i] * d;
                r1 += Cn[(size_t)(a + 1) * nx + i] * d;
                r2 += Cn[(size_t)(a + 2) * nx + i] * d;
                r3 += Cn[(size_t)(a + 3) * nx + i] * d;
            }
            U[(size_t)a * nv + j] = r0;
            U[(size_t)(a + 1) * nv + j] = r1;
            U[(size_t)(a + 2) * nv + j] = r2;
            U[(size_t)(a + 3) * nv + j] = r3;
        }
    }
#endif
    for (; a < nf; ++a) {
        memset(U + (size_t)a * nv, 0, (size_t)columns * sizeof(double));
        dense_transpose_add(D, nx, nv, Cn + (size_t)a * nx, U + (size_t)a * nv, columns);
    }
    for (i = 0; i < nr; ++i)
        for (a = 0; a < nf; ++a)
            Z[(size_t)i * nf + a] = U[(size_t)a * nv + w->ne + i];
    /* Same ascending FMA/division order; retain terminal RHS in registers
     * instead of storing them after each preceding triangular row. */
    lower_solve_multiple(L, nr, Z, nf);
    for (a = 0; a < nf && !initial_zero; ++a) {
        for (i = 0; i < nr; ++i)
            w->tmp[i] = Z[(size_t)i * nf + a];
        for (j = 0; j < nx; ++j)
            C[(size_t)a * nx + j] =
                U[(size_t)a * nv + w->nu + j] - dot(w->tmp, T + (size_t)j * nr, nr);
    }
    for (a = 0; a < nf; ++a)
        for (b = 0; b <= a; ++b) {
            double s = 0.0;
            for (i = 0; i < nr; ++i)
                s += Z[(size_t)i * nf + a] * Z[(size_t)i * nf + b];
            w->S[(size_t)a * nf + b] += s;
        }
}

/* Exact column supports combine contiguous rows of symmetric P. */
static void sparse_riccati_products(icore_ocp_workspace *w, int k, const double *Pn, int columns)
{
    int nx = w->nx, n = w->nvr, nv = w->nv, start = w->ne;
    double *PDT = w->PDT, *PD = w->PD, *M = w->M;
    int j, i, a;
    for (j = 0; j < columns; ++j) {
        size_t column = (size_t)k * nv + start + j;
        int count = w->dyn_count[column];
        const int *index = w->dyn_index + column * nx;
        const double *value = w->dyn_value + column * nx;
        double *out = PDT + (size_t)j * nx;
        memset(out, 0, (size_t)nx * sizeof(double));
        for (a = 0; a < count; ++a)
            axpy(out, Pn + (size_t)index[a] * nx, value[a], nx);
    }
    i = 0;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    for (; i + 3 < nx; i += 4) {
        j = 0;
        for (; j + 3 < columns; j += 4) {
            __m256d a0 = _mm256_loadu_pd(PDT + (size_t)j * nx + i);
            __m256d a1 = _mm256_loadu_pd(PDT + (size_t)(j + 1) * nx + i);
            __m256d a2 = _mm256_loadu_pd(PDT + (size_t)(j + 2) * nx + i);
            __m256d a3 = _mm256_loadu_pd(PDT + (size_t)(j + 3) * nx + i);
            __m256d t0 = _mm256_unpacklo_pd(a0, a1), t1 = _mm256_unpackhi_pd(a0, a1);
            __m256d t2 = _mm256_unpacklo_pd(a2, a3), t3 = _mm256_unpackhi_pd(a2, a3);
            _mm256_storeu_pd(PD + (size_t)i * n + j, _mm256_permute2f128_pd(t0, t2, 0x20));
            _mm256_storeu_pd(PD + (size_t)(i + 1) * n + j, _mm256_permute2f128_pd(t1, t3, 0x20));
            _mm256_storeu_pd(PD + (size_t)(i + 2) * n + j, _mm256_permute2f128_pd(t0, t2, 0x31));
            _mm256_storeu_pd(PD + (size_t)(i + 3) * n + j, _mm256_permute2f128_pd(t1, t3, 0x31));
        }
        for (; j < columns; ++j) {
            PD[(size_t)i * n + j] = PDT[(size_t)j * nx + i];
            PD[(size_t)(i + 1) * n + j] = PDT[(size_t)j * nx + i + 1];
            PD[(size_t)(i + 2) * n + j] = PDT[(size_t)j * nx + i + 2];
            PD[(size_t)(i + 3) * n + j] = PDT[(size_t)j * nx + i + 3];
        }
    }
#endif
    for (; i < nx; ++i)
        for (j = 0; j < columns; ++j)
            PD[(size_t)i * n + j] = PDT[(size_t)j * nx + i];
    for (i = 0; i < columns; ++i) {
        size_t column = (size_t)k * nv + start + i;
        int count = w->dyn_count[column];
        const int *index = w->dyn_index + column * nx;
        const double *value = w->dyn_value + column * nx;
        for (a = 0; a < count; ++a)
            axpy(M + (size_t)i * n, PD + (size_t)index[a] * n, value[a], i + 1);
    }
}

/* A wide dense tile reuses each left-hand broadcast across three SIMD
 * vectors. The inner products retain their original ascending summation
 * order. Strides describe either a matrix or its transpose without packing. */
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
ICORE_OCP_INLINE void dense_product_tile12(double *output, int output_stride, const double *left,
                                      int left_row, int left_column, const double *right,
                                      int right_stride, int inner, int accumulate)
{
    __m256d r00 = accumulate ? _mm256_loadu_pd(output) : _mm256_setzero_pd();
    __m256d r01 = accumulate ? _mm256_loadu_pd(output + 4) : _mm256_setzero_pd();
    __m256d r02 = accumulate ? _mm256_loadu_pd(output + 8) : _mm256_setzero_pd();
    __m256d r10 = accumulate ? _mm256_loadu_pd(output + output_stride) : _mm256_setzero_pd();
    __m256d r11 = accumulate ? _mm256_loadu_pd(output + output_stride + 4) : _mm256_setzero_pd();
    __m256d r12 = accumulate ? _mm256_loadu_pd(output + output_stride + 8) : _mm256_setzero_pd();
    __m256d r20 =
        accumulate ? _mm256_loadu_pd(output + 2 * (size_t)output_stride) : _mm256_setzero_pd();
    __m256d r21 =
        accumulate ? _mm256_loadu_pd(output + 2 * (size_t)output_stride + 4) : _mm256_setzero_pd();
    __m256d r22 =
        accumulate ? _mm256_loadu_pd(output + 2 * (size_t)output_stride + 8) : _mm256_setzero_pd();
    __m256d r30 =
        accumulate ? _mm256_loadu_pd(output + 3 * (size_t)output_stride) : _mm256_setzero_pd();
    __m256d r31 =
        accumulate ? _mm256_loadu_pd(output + 3 * (size_t)output_stride + 4) : _mm256_setzero_pd();
    __m256d r32 =
        accumulate ? _mm256_loadu_pd(output + 3 * (size_t)output_stride + 8) : _mm256_setzero_pd();
    int a;
    for (a = 0; a < inner; ++a) {
        const double *lp = left + (size_t)a * left_column;
        const double *rp = right + (size_t)a * right_stride;
        __m256d d0 = _mm256_loadu_pd(rp), d1 = _mm256_loadu_pd(rp + 4);
        __m256d d2 = _mm256_loadu_pd(rp + 8), b;
        b = _mm256_set1_pd(lp[0]);
        r00 = _mm256_fmadd_pd(b, d0, r00);
        r01 = _mm256_fmadd_pd(b, d1, r01);
        r02 = _mm256_fmadd_pd(b, d2, r02);
        b = _mm256_set1_pd(lp[left_row]);
        r10 = _mm256_fmadd_pd(b, d0, r10);
        r11 = _mm256_fmadd_pd(b, d1, r11);
        r12 = _mm256_fmadd_pd(b, d2, r12);
        b = _mm256_set1_pd(lp[2 * (size_t)left_row]);
        r20 = _mm256_fmadd_pd(b, d0, r20);
        r21 = _mm256_fmadd_pd(b, d1, r21);
        r22 = _mm256_fmadd_pd(b, d2, r22);
        b = _mm256_set1_pd(lp[3 * (size_t)left_row]);
        r30 = _mm256_fmadd_pd(b, d0, r30);
        r31 = _mm256_fmadd_pd(b, d1, r31);
        r32 = _mm256_fmadd_pd(b, d2, r32);
    }
    _mm256_storeu_pd(output, r00);
    _mm256_storeu_pd(output + 4, r01);
    _mm256_storeu_pd(output + 8, r02);
    _mm256_storeu_pd(output + output_stride, r10);
    _mm256_storeu_pd(output + output_stride + 4, r11);
    _mm256_storeu_pd(output + output_stride + 8, r12);
    _mm256_storeu_pd(output + 2 * (size_t)output_stride, r20);
    _mm256_storeu_pd(output + 2 * (size_t)output_stride + 4, r21);
    _mm256_storeu_pd(output + 2 * (size_t)output_stride + 8, r22);
    _mm256_storeu_pd(output + 3 * (size_t)output_stride, r30);
    _mm256_storeu_pd(output + 3 * (size_t)output_stride + 4, r31);
    _mm256_storeu_pd(output + 3 * (size_t)output_stride + 8, r32);
}
#endif

#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
/* Medium stages (12<=nx<32, n>=16) use eight independent accumulator chains
 * to reuse each loaded input pair across
 * four output rows. Each output retains its original seed and ascending FMA
 * order; only the number of simultaneously computed columns changes. */
static void dense_product_tile8(double *output, int output_stride, const double *left, int left_row,
                                int left_column, const double *right, int right_stride, int inner,
                                int accumulate)
{
    __m256d r00, r01, r10, r11, r20, r21, r30, r31;
    int a;
    if (accumulate) {
        r00 = _mm256_loadu_pd(output);
        r01 = _mm256_loadu_pd(output + 4);
        r10 = _mm256_loadu_pd(output + output_stride);
        r11 = _mm256_loadu_pd(output + output_stride + 4);
        r20 = _mm256_loadu_pd(output + 2 * (size_t)output_stride);
        r21 = _mm256_loadu_pd(output + 2 * (size_t)output_stride + 4);
        r30 = _mm256_loadu_pd(output + 3 * (size_t)output_stride);
        r31 = _mm256_loadu_pd(output + 3 * (size_t)output_stride + 4);
    } else {
        r00 = r01 = r10 = r11 = r20 = r21 = r30 = r31 = _mm256_setzero_pd();
    }
    for (a = 0; a < inner; ++a) {
        const double *lp = left + (size_t)a * left_column;
        const double *rp = right + (size_t)a * right_stride;
        __m256d d0 = _mm256_loadu_pd(rp), d1 = _mm256_loadu_pd(rp + 4), b;
        b = _mm256_set1_pd(lp[0]);
        r00 = _mm256_fmadd_pd(b, d0, r00);
        r01 = _mm256_fmadd_pd(b, d1, r01);
        b = _mm256_set1_pd(lp[left_row]);
        r10 = _mm256_fmadd_pd(b, d0, r10);
        r11 = _mm256_fmadd_pd(b, d1, r11);
        b = _mm256_set1_pd(lp[2 * (size_t)left_row]);
        r20 = _mm256_fmadd_pd(b, d0, r20);
        r21 = _mm256_fmadd_pd(b, d1, r21);
        b = _mm256_set1_pd(lp[3 * (size_t)left_row]);
        r30 = _mm256_fmadd_pd(b, d0, r30);
        r31 = _mm256_fmadd_pd(b, d1, r31);
    }
    _mm256_storeu_pd(output, r00);
    _mm256_storeu_pd(output + 4, r01);
    _mm256_storeu_pd(output + output_stride, r10);
    _mm256_storeu_pd(output + output_stride + 4, r11);
    _mm256_storeu_pd(output + 2 * (size_t)output_stride, r20);
    _mm256_storeu_pd(output + 2 * (size_t)output_stride + 4, r21);
    _mm256_storeu_pd(output + 3 * (size_t)output_stride, r30);
    _mm256_storeu_pd(output + 3 * (size_t)output_stride + 4, r31);
}
#endif

/* Four-row tiles keep small dense products in registers across their inner
 * dimension.  The Schur product visits one block triangle; diagonal tiles
 * may include unused entries above each diagonal. Wide tiles are used only
 * for nx>=32 and reduced stage width>=48; small and sparse products keep
 * their existing kernels. Selection depends only on matrix structure. */
static void riccati_products_columns(icore_ocp_workspace *w, const double *Pn, const double *D,
                                     const double *H, int columns)
{
    int nx = w->nx, n = w->nvr, nv = w->nv, start = w->ne;
    double *PD = w->PD, *M = w->M;
    int i = 0, a;
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
    int j;
#endif
    for (a = 0; a < columns; ++a) {
        memcpy(M + (size_t)a * n, H + (size_t)(start + a) * nv + start,
               (size_t)columns * sizeof(double));
        M[(size_t)a * n + a] += w->reg;
    }
    {
        int stage = (int)((D - w->D) / ((size_t)nx * nv));
        if (w->dyn_sparse[stage]) {
            sparse_riccati_products(w, stage, Pn, columns);
            return;
        }
    }
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
    for (; i + 3 < nx; i += 4) {
        j = 0;
        if (nx >= 32 && n >= 48)
            for (; j + 11 < columns; j += 12)
                dense_product_tile12(PD + (size_t)i * n + j, n, Pn + (size_t)i * nx, nx, 1,
                                     D + start + j, nv, nx, 0);
        if (nx >= 12 && nx < 32 && n >= 16)
            for (; j + 7 < columns; j += 8)
                dense_product_tile8(PD + (size_t)i * n + j, n, Pn + (size_t)i * nx, nx, 1,
                                    D + start + j, nv, nx, 0);
        for (; j + 3 < columns; j += 4) {
            __m256d r0 = _mm256_setzero_pd(), r1 = r0, r2 = r0, r3 = r0;
            for (a = 0; a < nx; ++a) {
                __m256d d = _mm256_loadu_pd(D + (size_t)a * nv + start + j);
                r0 = _mm256_fmadd_pd(_mm256_set1_pd(Pn[(size_t)i * nx + a]), d, r0);
                r1 = _mm256_fmadd_pd(_mm256_set1_pd(Pn[(size_t)(i + 1) * nx + a]), d, r1);
                r2 = _mm256_fmadd_pd(_mm256_set1_pd(Pn[(size_t)(i + 2) * nx + a]), d, r2);
                r3 = _mm256_fmadd_pd(_mm256_set1_pd(Pn[(size_t)(i + 3) * nx + a]), d, r3);
            }
            _mm256_storeu_pd(PD + (size_t)i * n + j, r0);
            _mm256_storeu_pd(PD + (size_t)(i + 1) * n + j, r1);
            _mm256_storeu_pd(PD + (size_t)(i + 2) * n + j, r2);
            _mm256_storeu_pd(PD + (size_t)(i + 3) * n + j, r3);
        }
        for (; j < columns; ++j) {
            double r0 = 0.0, r1 = 0.0, r2 = 0.0, r3 = 0.0;
            for (a = 0; a < nx; ++a) {
                double d = D[(size_t)a * nv + start + j];
                r0 += Pn[(size_t)i * nx + a] * d;
                r1 += Pn[(size_t)(i + 1) * nx + a] * d;
                r2 += Pn[(size_t)(i + 2) * nx + a] * d;
                r3 += Pn[(size_t)(i + 3) * nx + a] * d;
            }
            PD[(size_t)i * n + j] = r0;
            PD[(size_t)(i + 1) * n + j] = r1;
            PD[(size_t)(i + 2) * n + j] = r2;
            PD[(size_t)(i + 3) * n + j] = r3;
        }
    }
#endif
    for (; i < nx; ++i) {
        memset(PD + (size_t)i * n, 0, (size_t)columns * sizeof(double));
        for (a = 0; a < nx; ++a)
            axpy(PD + (size_t)i * n, D + (size_t)a * nv + start, Pn[(size_t)i * nx + a], columns);
    }
    i = 0;
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
    for (; i + 3 < columns; i += 4) {
        j = 0;
        if (nx >= 32 && n >= 48)
            for (; j <= i && j + 11 < columns; j += 12)
                dense_product_tile12(M + (size_t)i * n + j, n, D + start + i, 1, nv, PD + j, n, nx,
                                     1);
        if (nx >= 12 && nx < 32 && n >= 16)
            for (; j + 7 <= i; j += 8)
                dense_product_tile8(M + (size_t)i * n + j, n, D + start + i, 1, nv, PD + j, n, nx,
                                    1);
        for (; j <= i; j += 4) {
            __m256d r0 = _mm256_loadu_pd(M + (size_t)i * n + j);
            __m256d r1 = _mm256_loadu_pd(M + (size_t)(i + 1) * n + j);
            __m256d r2 = _mm256_loadu_pd(M + (size_t)(i + 2) * n + j);
            __m256d r3 = _mm256_loadu_pd(M + (size_t)(i + 3) * n + j);
            for (a = 0; a < nx; ++a) {
                __m256d p = _mm256_loadu_pd(PD + (size_t)a * n + j);
                const double *d = D + (size_t)a * nv + start + i;
                r0 = _mm256_fmadd_pd(_mm256_set1_pd(d[0]), p, r0);
                r1 = _mm256_fmadd_pd(_mm256_set1_pd(d[1]), p, r1);
                r2 = _mm256_fmadd_pd(_mm256_set1_pd(d[2]), p, r2);
                r3 = _mm256_fmadd_pd(_mm256_set1_pd(d[3]), p, r3);
            }
            _mm256_storeu_pd(M + (size_t)i * n + j, r0);
            _mm256_storeu_pd(M + (size_t)(i + 1) * n + j, r1);
            _mm256_storeu_pd(M + (size_t)(i + 2) * n + j, r2);
            _mm256_storeu_pd(M + (size_t)(i + 3) * n + j, r3);
        }
    }
#endif
    for (; i < columns; ++i)
        for (a = 0; a < nx; ++a) {
            double d = D[(size_t)a * nv + start + i];
            if (d != 0.0)
                axpy(M + (size_t)i * n, PD + (size_t)a * n, d, i + 1);
        }
}

static void riccati_products(icore_ocp_workspace *w, const double *Pn, const double *D, const double *H)
{
    riccati_products_columns(w, Pn, D, H, w->nvr);
}

static int factor_classical_qr_suffix(icore_ocp_workspace *w, int k, int *qr_first, int *mixed_eligible);

static int factor_classical(icore_ocp_workspace *w, int recover_poor)
{
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
    int k, i, j;
    int nx = w->nx, nr = w->nr, nv = w->nv, n = w->nvr;
    double *PN = w->P + (size_t)w->N * nx * nx;
    const double *HN = w->Hb + (size_t)w->N * nv * nv;
    for (i = 0; i < nx; ++i)
        for (j = 0; j < nx; ++j)
            PN[(size_t)i * nx + j] =
                HN[(size_t)(w->nu + i) * nv + w->nu + j] + (i == j ? w->reg : 0.0);
    w->factor_poor = 0;
    int qr_first = w->N + 1, mixed_eligible = -1;
    for (k = w->N - 1; k >= 0; --k) {
        int cross_nonzero = 0, block_poor = 0;
        const double *Pn = w->P + (size_t)(k + 1) * nx * nx;
        const double *D = w->D + (size_t)k * nx * nv, *H = w->Hb + (size_t)k * nv * nv;
        double *L = w->L + (size_t)k * nr * nr, *T = w->T + (size_t)k * nx * nr;
        double *P = w->P + (size_t)k * nx * nx, *M = w->M;
        if (w->ocp_la && w->ocp_packed_ready[k]) {
            int packed = icore_ocp_la_factor_stage(w->ocp_la, k, Pn, (size_t)nx,
                                          H + (size_t)w->ne * nv + w->ne, (size_t)nv, w->reg,
                                          fmax(w->reg, 1024.0 * DBL_MIN), 1e-12, L, T, P);
            if (packed == ICORE_LA_OK) {
                ++w->ocp_packed_stages;
                if (w->nf)
                    factor_terminal_constraints(w, k);
                continue;
            }
            /* No packed output was committed. Retain the original stage's
             * local equilibrated retry and the complete outer QR fallback. */
            ++w->ocp_packed_rejected;
        }
        /* At the Newton boundary dx0=0, only the control factor is used.
         * P0 and T0 have no predecessor. Keep four-column tile boundaries
         * so the retained control block follows the ordinary arithmetic. */
        int initial_zero = k == 0 && w->classical_initial_zero;
        if (initial_zero) {
            int columns = (nr + 3) / 4 * 4;
            if (columns > n)
                columns = n;
            riccati_products_columns(w, Pn, D, H, columns);
        } else
            riccati_products(w, Pn, D, H);
        for (i = 0; i < nr; ++i)
            for (j = 0; j <= i; ++j)
                L[(size_t)i * nr + j] = M[(size_t)i * n + j];
        if (!cholesky(L, nr, w->reg, &block_poor, w->qr))
            return 0;
        if (block_poor) {
            if (cholesky_equilibrated(M, n, L, nr, w->reg, w->tmp, w->qr)) {
                block_poor = 0;
            } else {
                /* Reproduce the original floored factor before any classical
                 * continuation or outer QR fallback.  Earlier stage warnings
                 * must not be erased by a successful local retry. */
                for (i = 0; i < nr; ++i)
                    for (j = 0; j <= i; ++j)
                        L[(size_t)i * nr + j] = M[(size_t)i * n + j];
                block_poor = 0;
                if (!cholesky(L, nr, w->reg, &block_poor, w->qr))
                    return 0;
            }
        }
        w->factor_poor |= block_poor;
        /* Repair an unrecovered block using original QR stacks through
         * this stage. A monotonically advancing suffix boundary prevents
         * any stage from being QR-factored twice in this classical attempt.
         * Explicit CLASSICAL retains its established floored continuation. */
        if (recover_poor && block_poor) {
            if (!factor_classical_qr_suffix(w, k, &qr_first, &mixed_eligible))
                return 0;
            w->factor_poor = 0;
            continue;
        }
        if (initial_zero) {
            if (w->nf)
                factor_terminal_constraints(w, k);
            continue;
        }
        /* Solve the cross-cost RHS together in transposed PD scratch. */
        for (j = 0; j < nr; ++j)
            for (i = 0; i < nx; ++i) {
                double value = M[(size_t)(nr + i) * n + j];
                w->PD[(size_t)j * nx + i] = value;
                if (value != 0.0)
                    cross_nonzero = 1;
            }
        if (!cross_nonzero) {
            /* The exact zero RHS gives T=0 and no Schur subtraction. This
             * also handles models whose state is an independent fixed dummy. */
            memset(T, 0, (size_t)nx * nr * sizeof(double));
            for (i = 0; i < nx; ++i)
                for (j = 0; j <= i; ++j)
                    P[(size_t)i * nx + j] = M[(size_t)(nr + i) * n + nr + j];
        } else {
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
            if (nx >= 16 && nr >= 4)
                lower_solve_multiple(L, nr, w->PD, nx);
            else
#endif
            {
                for (i = 0; i < nr; ++i) {
                    for (j = 0; j < i; ++j)
                        axpy(w->PD + (size_t)i * nx, w->PD + (size_t)j * nx, -L[(size_t)i * nr + j],
                             nx);
                    for (j = 0; j < nx; ++j)
                        w->PD[(size_t)i * nx + j] /= L[(size_t)i * nr + i];
                }
            }
            for (i = 0; i < nx; ++i)
                for (j = 0; j < nr; ++j)
                    T[(size_t)i * nr + j] = w->PD[(size_t)j * nx + i];
            i = 0;
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
            for (; i + 3 < nx; i += 4)
                for (j = 0; j <= i; j += 4) {
                    __m256d r0 = _mm256_setzero_pd(), r1 = r0, r2 = r0, r3 = r0;
                    int a;
                    for (a = 0; a < nr; ++a) {
                        const double *t = w->PD + (size_t)a * nx;
                        __m256d v = _mm256_loadu_pd(t + j);
                        r0 = _mm256_fmadd_pd(_mm256_set1_pd(t[i]), v, r0);
                        r1 = _mm256_fmadd_pd(_mm256_set1_pd(t[i + 1]), v, r1);
                        r2 = _mm256_fmadd_pd(_mm256_set1_pd(t[i + 2]), v, r2);
                        r3 = _mm256_fmadd_pd(_mm256_set1_pd(t[i + 3]), v, r3);
                    }
                    _mm256_storeu_pd(
                        P + (size_t)i * nx + j,
                        _mm256_sub_pd(_mm256_loadu_pd(M + (size_t)(nr + i) * n + nr + j), r0));
                    _mm256_storeu_pd(
                        P + (size_t)(i + 1) * nx + j,
                        _mm256_sub_pd(_mm256_loadu_pd(M + (size_t)(nr + i + 1) * n + nr + j), r1));
                    _mm256_storeu_pd(
                        P + (size_t)(i + 2) * nx + j,
                        _mm256_sub_pd(_mm256_loadu_pd(M + (size_t)(nr + i + 2) * n + nr + j), r2));
                    _mm256_storeu_pd(
                        P + (size_t)(i + 3) * nx + j,
                        _mm256_sub_pd(_mm256_loadu_pd(M + (size_t)(nr + i + 3) * n + nr + j), r3));
                }
#endif
            for (; i < nx; ++i)
                for (j = 0; j <= i; ++j)
                    P[(size_t)i * nx + j] = M[(size_t)(nr + i) * n + nr + j] -
                                            dot(T + (size_t)i * nr, T + (size_t)j * nr, nr);
        }
        for (i = 0; i < nx; ++i) {
            for (j = 0; j <= i; ++j) {
                double value = P[(size_t)i * nx + j];
                if (!isfinite(value))
                    return 0;
                P[(size_t)j * nx + i] = value;
            }
            if (P[(size_t)i * nx + i] < -1e-8 * fmax(1.0, fabs(M[(size_t)(nr + i) * n + nr + i]))) {
                w->factor_poor = 1;
                if (recover_poor)
                    break;
            }
        }
        if (recover_poor && w->factor_poor) {
            if (!factor_classical_qr_suffix(w, k, &qr_first, &mixed_eligible))
                return 0;
            w->factor_poor = 0;
            continue;
        }
        if (w->nf)
            factor_terminal_constraints(w, k);
    }
    return 1;
}

/* Householder QR of a tall array, retaining its upper triangular factor. */
static int qr_upper(double *A, int rows, int n, int stride)
{
    int i, j, k;
    for (j = 0; j < n; ++j) {
        double norm = 0.0, head = A[(size_t)j * stride + j], alpha, tau, den;
        for (i = j; i < rows; ++i)
            norm = hypot(norm, A[(size_t)i * stride + j]);
        if (!(norm > 0.0) || !isfinite(norm))
            return 0;
        alpha = -copysign(norm, head);
        den = head - alpha;
        tau = (alpha - head) / alpha;
        for (i = j + 1; i < rows; ++i)
            A[(size_t)i * stride + j] /= den;
        for (k = j + 1; k < n; ++k) {
            double s = A[(size_t)j * stride + k];
            for (i = j + 1; i < rows; ++i)
                s += A[(size_t)i * stride + j] * A[(size_t)i * stride + k];
            s *= tau;
            A[(size_t)j * stride + k] -= s;
            for (i = j + 1; i < rows; ++i)
                A[(size_t)i * stride + k] -= A[(size_t)i * stride + j] * s;
        }
        A[(size_t)j * stride + j] = alpha;
    }
    for (i = 0; i < n; ++i)
        if (A[(size_t)i * stride + i] < 0.0)
            for (j = i; j < n; ++j)
                A[(size_t)i * stride + j] = -A[(size_t)i * stride + j];
    return 1;
}

/* Form the established QR stack in its original row/column order. A local
 * recovery uses the validated physical packed suffix C_next^T D instead of
 * the equivalent upper-root R_next D from a whole-chain QR recomputation. */
static int form_square_root_stack(icore_ocp_workspace *w, int k, int packed_suffix)
{
    int i, a, row = 0, nx = w->nx, nv = w->nv;
    int terminal = k == w->N, n = terminal ? nx : w->nvr, start = terminal ? w->nu : w->ne;
    const double *Hs = w->Hs + (size_t)k * nv * nv, *Gh = w->Gh + (size_t)k * w->m * nv;
    double *Q = w->qr;
    double rootreg = sqrt(fmax(w->reg, 1e-16));
    memset(Q, 0, (size_t)w->qr_rows * nv * sizeof(double));
    for (i = 0; i < nv; ++i, ++row)
        memcpy(Q + (size_t)row * nv, Hs + (size_t)i * nv + start, (size_t)n * sizeof(double));
    for (i = 0; i < w->m; ++i) {
        if (w->epi_row[(size_t)k * w->m + i] >= 0)
            continue;
        if (w->orthant_roots_lazy) {
            size_t index = (size_t)k * w->m + i;
            int column = w->single_index[index];
            if (column >= start && column < start + n) {
                double scaled = w->single_value[index] * sqrt(w->lambda[index]);
                if (!isfinite(scaled))
                    return 0;
                Q[(size_t)row * nv + column - start] = scaled;
            }
        } else
            memcpy(Q + (size_t)row * nv, Gh + (size_t)i * nv + start, (size_t)n * sizeof(double));
        ++row;
    }
    if (!terminal) {
        for (i = 0; i < w->ne; ++i, ++row) {
            size_t epos = (size_t)k * w->ne + i;
            Q[(size_t)row * nv + w->epi_e[epos] - start] = sqrt(w->epi_curv[epos]);
        }
        if (packed_suffix) {
            if (icore_ocp_la_sqrt_recovery_product(w->ocp_la, k, Q + (size_t)row * nv, (size_t)nv) !=
                ICORE_LA_OK)
                return 0;
            row += nx;
        } else {
            const double *Rn = w->R + (size_t)(k + 1) * nx * nx, *D = w->D + (size_t)k * nx * nv;
            for (i = 0; i < nx; ++i, ++row)
                for (a = i; a < nx; ++a)
                    axpy(Q + (size_t)row * nv, D + (size_t)a * nv + start, Rn[(size_t)i * nx + a],
                         n);
        }
    }
    for (i = 0; i < n; ++i, ++row)
        Q[(size_t)row * nv + i] = rootreg;
    return row;
}

/* Factor original QR stacks over an inclusive backward stage range.
 * For first<N, R[first+1] and the terminal border must already describe an
 * accepted QR suffix. The full-QR entry passes N..0, as before. */
static int factor_square_root_range(icore_ocp_workspace *w, int first, int last)
{
    int k, i, j, a, row;
    int nx = w->nx, nr = w->nr, nv = w->nv;
    for (k = first; k >= last; --k) {
        int terminal = k == w->N, n = terminal ? nx : w->nvr;
        double *Q = w->qr, *R = w->R + (size_t)k * nx * nx, *P = w->P + (size_t)k * nx * nx;
        row = form_square_root_stack(w, k, 0);
        if (!row)
            return 0;
        /* The packed Householder path factors this exact unchanged stack.
         * Rejection leaves Q intact for the existing range-safe fallback. */
        if ((!w->ocp_qr_la || icore_ocp_la_qr_upper(w->ocp_qr_la, row, n, Q, (size_t)nv) != ICORE_LA_OK) &&
            !qr_upper(Q, row, n, nv))
            return 0;
        if (!terminal) {
            double *L = w->L + (size_t)k * nr * nr, *T = w->T + (size_t)k * nx * nr;
            for (i = 0; i < nr; ++i)
                for (j = 0; j <= i; ++j)
                    L[(size_t)i * nr + j] = Q[(size_t)j * nv + i];
            for (i = 0; i < nx; ++i)
                for (j = 0; j < nr; ++j)
                    T[(size_t)i * nr + j] = Q[(size_t)j * nv + nr + i];
        }
        memset(R, 0, (size_t)nx * nx * sizeof(double));
        for (i = 0; i < nx; ++i)
            for (j = i; j < nx; ++j)
                R[(size_t)i * nx + j] = Q[(size_t)(n - nx + i) * nv + n - nx + j];
        for (i = 0; i < nx; ++i)
            for (j = 0; j <= i; ++j) {
                double v = 0.0;
                for (a = 0; a <= j; ++a)
                    v += R[(size_t)a * nx + i] * R[(size_t)a * nx + j];
                P[(size_t)i * nx + j] = P[(size_t)j * nx + i] = v;
            }
        if (!terminal && w->nf)
            factor_terminal_constraints(w, k);
    }
    return 1;
}

/* Replace the intervening classical suffix from original coefficients,
 * without constructing a root of a cancellation-damaged P matrix. Each
 * stage enters the QR suffix at most once in a factor attempt. The dead
 * terminal-factor buffer SL checkpoints the already accepted QR suffix's
 * accumulated border S. Thus overwritten classical C/Z terms are removed
 * before their QR replacements are accumulated, including multiple repairs.
 * Failed recovery retains the existing complete QR fallback. */
static int factor_classical_qr_suffix(icore_ocp_workspace *w, int k, int *qr_first, int *mixed_eligible)
{
    int stage, i, j;
    if (w->reg < 1e-16)
        return 0;
    if (*mixed_eligible < 0) {
        *mixed_eligible = 1;
        /* QR roots use upper H and classical factors use lower H. The
         * mixed chain must describe the same operator at every stage. */
        for (stage = 0; stage < w->K; ++stage) {
            const double *H = w->H + (size_t)stage * w->nv * w->nv;
            for (i = 0; i < w->nv; ++i)
                for (j = 0; j < i; ++j)
                    if (H[(size_t)i * w->nv + j] != H[(size_t)j * w->nv + i])
                        *mixed_eligible = 0;
        }
    }
    if (!*mixed_eligible)
        return 0;
    if (w->nf) {
        if (*qr_first == w->N + 1)
            memset(w->S, 0, (size_t)w->nf * w->nf * sizeof(double));
        else
            memcpy(w->S, w->SL, (size_t)w->nf * w->nf * sizeof(double));
    }
    if (!factor_square_root_range(w, *qr_first - 1, k))
        return 0;
    if (w->nf)
        memcpy(w->SL, w->S, (size_t)w->nf * w->nf * sizeof(double));
    *qr_first = k;
    ++w->factor_retries;
    return 1;
}

static int factor_square_root(icore_ocp_workspace *w)
{
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
    return factor_square_root_range(w, w->N, 0);
}

#ifndef ICORE_OCP_PACKED_SQRT
#define ICORE_OCP_PACKED_SQRT 1
#endif

#ifndef ICORE_OCP_PACKED_LOCAL_QR
#define ICORE_OCP_PACKED_LOCAL_QR 1
#endif

static int factor_local_square_root(icore_ocp_workspace *w, int k, int status, int *mixed_eligible)
{
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
#if ICORE_OCP_PACKED_LOCAL_QR
    int i, j, stage, row, n = k == w->N ? w->nx : w->nvr;
    int initial_fixed = k == 0;
    double *L = k == w->N ? NULL : w->L + (size_t)k * w->nr * w->nr;
    double *T = k == w->N ? NULL : w->T + (size_t)k * w->nx * w->nr;
    if (status != ICORE_OCP_LA_PIVOT_REJECTED || w->reg < 1e-16 ||
        icore_ocp_la_sqrt_recovery_pending(w->ocp_la, k, initial_fixed, w->reg) != ICORE_LA_OK)
        return 0;
    if (*mixed_eligible < 0) {
        *mixed_eligible = 1;
        /* Cost roots use upper H, whereas packed Cholesky uses lower H.
         * Preserve whole-chain policy for tolerated asymmetric input. Check
         * every retained/potential stage, not only the one being recovered. */
        for (stage = 0; stage < w->K; ++stage) {
            const double *H = w->H + (size_t)stage * w->nv * w->nv;
            for (i = 0; i < w->nv; ++i)
                for (j = 0; j < i; ++j)
                    if (H[(size_t)i * w->nv + j] != H[(size_t)j * w->nv + i])
                        *mixed_eligible = 0;
        }
    }
    if (!*mixed_eligible)
        return 0;
    row = form_square_root_stack(w, k, 1);
    if (!row)
        return 0;
    if ((!w->ocp_qr_la || icore_ocp_la_qr_upper(w->ocp_qr_la, row, n, w->qr, (size_t)w->nv) != ICORE_LA_OK) &&
        !qr_upper(w->qr, row, n, w->nv))
        return 0;
    if (icore_ocp_la_sqrt_recovery_accept(w->ocp_la, k, initial_fixed, w->reg, w->qr, (size_t)w->nv, L, T) !=
        ICORE_LA_OK)
        return 0;
    ++w->factor_retries;
    return 1;
#else
    (void)w;
    (void)k;
    (void)status;
    (void)mixed_eligible;
    return 0;
#endif
}

#include "mixed_solver_impl.h"

/* Propagate state Cholesky factors directly. Free state stages require a
 * positive-definite full factor. At stage zero x0 is fixed, so only the
 * control factor and cross block are needed. A guarded rejected stage may
 * use the existing QR stack while preserving its accepted downstream roots.
 * Unrecovered chains retain the complete classical/QR fallback.
 * P is applied from its factor in equality_solve, never materialized here. */
static int factor_packed_square_root(icore_ocp_workspace *w, int diagonal, int zero_initial_direction,
                                     int affine_predictor)
{
    if (factor_mixed_square_root(w, diagonal, zero_initial_direction, affine_predictor))
        return 1;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
#if ICORE_OCP_PACKED_SQRT
    int k, i, status, equilibrated, affine = 0, mixed_eligible = -1, nx = w->nx, nv = w->nv,
                                    nr = w->nr;
    double floor = fmax(w->reg, 1024.0 * DBL_MIN);
    const double *HN = w->Hb + (size_t)w->N * nv * nv + (size_t)w->nu * nv + w->nu;
    if (!w->ocp_la)
        return 0;
    zero_initial_direction =
        zero_initial_direction && !w->ne && !w->nf && icore_ocp_la_sqrt_owns_factors(w->ocp_la);
    for (k = 0; k < w->N; ++k)
        if (!w->ocp_packed_ready[k])
            return 0;
    affine = affine_predictor && w->orthant_direct && zero_initial_direction;
    if (affine) {
        /* Hbar/Gh and the direct scaling are complete. These shared helpers
         * perform exactly the RHS work otherwise done after factorization.
         * Factor/recovery scratch is disjoint from rhs/pcone/bcone/brd/brf. */
        form_affine_rhs(w);
        prepare_main_reduced_rhs(w);
        for (i = 0; i < nx; ++i)
            w->sweep_p[(size_t)w->N * nx + i] = -w->rhs[(size_t)w->N * nv + w->nu + i];
    }
    status = diagonal ? icore_ocp_la_sqrt_begin_cached_diagonal(w->ocp_la,
                                                       w->ocp_diagonal + (size_t)w->N * nv + w->nu,
                                                       w->reg, floor, 1e-12)
                      : icore_ocp_la_sqrt_begin(w->ocp_la, HN, (size_t)nv, w->reg, floor, 1e-12);
    if (status != ICORE_LA_OK)
        affine = 0;
    if (status != ICORE_LA_OK && !factor_local_square_root(w, w->N, status, &mixed_eligible)) {
        ++w->ocp_sqrt_rejected;
        return 0;
    }
    for (k = w->N - 1; k >= 0; --k) {
        const double *H = w->Hb + (size_t)k * nv * nv + (size_t)w->ne * nv + w->ne;
        double *L = w->L + (size_t)k * nr * nr, *T = w->T + (size_t)k * nx * nr;
        if (icore_ocp_la_sqrt_owns_factors(w->ocp_la))
            L = T = NULL;
        if (k == 0 && zero_initial_direction) {
            status = diagonal
                         ? icore_ocp_la_sqrt_initial_zero_cached_diagonal(w->ocp_la, w->ocp_diagonal, w->reg,
                                                                 floor, 1e-12)
                         : icore_ocp_la_sqrt_initial_zero(w->ocp_la, H, (size_t)nv, w->reg, floor, 1e-12);
        } else if (diagonal) {
            const double *d = w->ocp_diagonal + (size_t)k * nv + w->ne;
            status = k == 0 ? icore_ocp_la_sqrt_initial_fixed_cached_diagonal(w->ocp_la, d, w->reg, floor,
                                                                     1e-12, L, T)
                            : icore_ocp_la_sqrt_stage_cached_diagonal(w->ocp_la, k, d, w->reg, floor, 1e-12,
                                                             L, T);
        } else {
            status =
                k == 0
                    ? icore_ocp_la_sqrt_initial_fixed(w->ocp_la, H, (size_t)nv, w->reg, floor, 1e-12, L, T)
                    : icore_ocp_la_sqrt_stage(w->ocp_la, k, H, (size_t)nv, w->reg, floor, 1e-12, L, T);
        }
        if (status != ICORE_LA_OK)
            affine = 0;
        if (status != ICORE_LA_OK && !factor_local_square_root(w, k, status, &mixed_eligible)) {
            w->ocp_sqrt_equilibrated += icore_ocp_la_sqrt_equilibrated_stages(w->ocp_la);
            ++w->ocp_sqrt_rejected;
            return 0;
        }
        if (w->nf && icore_ocp_la_terminal_border(w->ocp_la, k, w->nf, w->C + (size_t)(k + 1) * w->nf * nx,
                                         w->C + (size_t)k * w->nf * nx,
                                         w->Z + (size_t)k * nr * w->nf, w->S) != ICORE_LA_OK) {
            if (icore_ocp_la_sqrt_owns_factors(w->ocp_la) &&
                icore_ocp_la_export_stage_factor(w->ocp_la, k, w->L + (size_t)k * nr * nr,
                                        w->T + (size_t)k * nx * nr) != ICORE_LA_OK)
                return 0;
            factor_terminal_constraints(w, k);
        }
        if (affine) {
            /* An internally retried initial stage must use the established
             * complete equality solve, even if its full factor succeeded. */
            if (k == 0 && !icore_ocp_la_sqrt_initial_zero_active(w->ocp_la))
                affine = 0;
            else if (icore_ocp_la_factor_backward_stage(
                         w->ocp_la, k, w->rhs + (size_t)k * nv, w->re + (size_t)k * nx,
                         w->sweep_p + (size_t)(k + 1) * nx, w->sweep_pd + (size_t)k * nx,
                         w->sweep_a + (size_t)k * nr,
                         k ? w->sweep_p + (size_t)k * nx : NULL) != ICORE_LA_OK)
                affine = 0;
        }
    }
    w->ocp_sqrt_active = 1;
    w->factor_poor = 0;
    equilibrated = icore_ocp_la_sqrt_equilibrated_stages(w->ocp_la);
    w->ocp_sqrt_equilibrated += equilibrated;
    /* Only a wholly accepted, unrecovered chain can publish this one-use
     * RHS token. Pd validity still waits for the complete forward solve. */
    w->affine_state = affine && !equilibrated ? ICORE_OCP_AFFINE_BACKWARD_READY : ICORE_OCP_AFFINE_EMPTY;
    ++w->ocp_sqrt_factors;
    w->ocp_packed_stages += w->N;
    return 1;
#else
    (void)w;
    (void)diagonal;
    (void)zero_initial_direction;
    (void)affine_predictor;
    return 0;
#endif
}

static int factor_problem_requested(icore_ocp_workspace *w, const icore_ocp_options *opt,
                                    int zero_initial_direction, int affine_predictor)
{
    auxiliary_factor_preserve(w);
    w->mixed_sweeps_active = 0;
    icore_ocp_la_mixed_sweeps_invalidate(w->ocp_la);
    w->mixed_factor_active = 0;
    int ok, poor = 0;
    int diagonal = opt->riccati_mode == 0 && w->ocp_diagonal_cost_ready;
    w->mixed_sweeps_requested = !(opt->refinement_tolerance > 0.0);
    w->ocp_sqrt_active = 0;
    w->classical_initial_zero = 0;
    /* A new factor invalidates every cached P[k+1]*d[k]. */
    w->sweep_pd_valid = 0;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
    if (diagonal ? !form_stage_diagonal(w) : !form_stage_hessians(w))
        return 0;
    w->reg = fmax(0.0, opt->regularization);
    memset(w->S, 0, (size_t)w->nf * w->nf * sizeof(double));
    if (w->nf)
        memcpy(w->C + (size_t)w->N * w->nf * w->nx, w->Fn, (size_t)w->nf * w->nx * sizeof(double));
    if (opt->riccati_mode == 2)
        ok = factor_square_root(w);
    else {
        ok = opt->riccati_mode == 0 &&
             factor_packed_square_root(w, diagonal, zero_initial_direction, affine_predictor);
        if (!ok) {
            /* The diagonal-only route did not materialize Hb. Rebuild it
             * before any classical/QR recovery can consume the matrices. */
            if (diagonal && !form_stage_hessians(w))
                return 0;
            /* A partial packed chain may already have accumulated terminal
             * constraint terms. Recompute every stage and this sum together. */
            memset(w->S, 0, (size_t)w->nf * w->nf * sizeof(double));
            w->classical_initial_zero = zero_initial_direction && !w->ne;
            ok = factor_classical(w, opt->riccati_mode == ICORE_OCP_AUTO);
        }
        if (opt->riccati_mode == 0 && (!ok || w->factor_poor)) {
            ++w->factor_retries;
            w->classical_initial_zero = 0;
            memset(w->S, 0, (size_t)w->nf * w->nf * sizeof(double));
            ok = factor_square_root(w);
        }
    }
    if (!ok)
        return 0;
    if (w->nf) {
        memcpy(w->SL, w->S, (size_t)w->nf * w->nf * sizeof(double));
        if (!cholesky(w->SL, w->nf, fmax(w->reg * 1e-3, 1e-16), &poor, w->qr))
            return 0;
        if (poor)
            ++w->factor_retries;
    }
    return 1;
}

/* All refactor/recovery callers retain the ordinary RHS path. Only the
 * driver's predictor-factor callback requests interleaving. */
static int factor_problem_context(icore_ocp_workspace *w, const icore_ocp_options *opt,
                                  int zero_initial_direction)
{
    return factor_problem_requested(w, opt, zero_initial_direction, 0);
}

/* Auxiliary initialization can solve for a nonzero fixed x0. Keep this
 * established entry point full; only Newton callbacks opt into dx0=0. */
static int factor_problem(icore_ocp_workspace *w, const icore_ocp_options *opt)
{
    return factor_problem_context(w, opt, 0);
}
