/* Small vector kernels and exact structure checks.
 * The SIMD and scalar paths retain their own summation and rounding order.
 * Structural tests inspect coefficients; they do not solve an optimization problem.
 * Private implementation: included exactly once by ocp.c. */

static void *array_storage(size_t n, size_t element, int clear)
{
    size_t bytes;
    void *p;
    if (n && element > SIZE_MAX / n)
        return NULL;
    bytes = (n ? n : 1) * element;
    if (bytes > SIZE_MAX - 63)
        return NULL;
    bytes = (bytes + 63) & ~(size_t)63;
    p = aligned_alloc(64, bytes);
    if (p && (clear || !n))
        memset(p, 0, bytes);
    return p;
}

static void *array_new(size_t n, size_t element)
{
    return array_storage(n, element, 1);
}

/* Only for arrays whose every used entry is assigned before its first read.
 * Empty arrays retain the original zero-filled sentinel allocation. */
static void *array_overwritten(size_t n, size_t element)
{
    return array_storage(n, element, 0);
}

ICORE_OCP_INLINE double dot(const double *a, const double *b, int n)
{
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
    __m128d pair;
    double sum;
    int i = 0;
    if (n < 4) {
        sum = 0.0;
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
    pair = _mm_add_pd(_mm256_castpd256_pd128(s0), _mm256_extractf128_pd(s0, 1));
    sum = _mm_cvtsd_f64(_mm_add_sd(pair, _mm_unpackhi_pd(pair, pair)));
    for (; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
#else
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    int i;
    for (i = 0; i + 3 < n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; ++i)
        s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
#endif
}

/* Independent row dots share the input-vector loads. Each output keeps
 * dot()'s two SIMD accumulators, horizontal reduction and scalar tail order.
 * No matrix packing, structure dispatch, or new floating-point expression. */
static void row_dots_vector(const double *A, int rows, int stride, const double *x, int n,
                            double *out)
{
    int r = 0;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    if (n >= 4)
        for (; r + 3 < rows; r += 4) {
            const double *a0 = A + (size_t)r * stride;
            const double *a1 = a0 + stride, *a2 = a1 + stride, *a3 = a2 + stride;
            __m256d s00 = _mm256_setzero_pd(), s01 = s00;
            __m256d s10 = s00, s11 = s00, s20 = s00, s21 = s00, s30 = s00, s31 = s00;
            int i = 0;
            for (; i + 7 < n; i += 8) {
                __m256d x0 = _mm256_loadu_pd(x + i), x1 = _mm256_loadu_pd(x + i + 4);
#if defined(__FMA__)
                s00 = _mm256_fmadd_pd(_mm256_loadu_pd(a0 + i), x0, s00);
                s01 = _mm256_fmadd_pd(_mm256_loadu_pd(a0 + i + 4), x1, s01);
                s10 = _mm256_fmadd_pd(_mm256_loadu_pd(a1 + i), x0, s10);
                s11 = _mm256_fmadd_pd(_mm256_loadu_pd(a1 + i + 4), x1, s11);
                s20 = _mm256_fmadd_pd(_mm256_loadu_pd(a2 + i), x0, s20);
                s21 = _mm256_fmadd_pd(_mm256_loadu_pd(a2 + i + 4), x1, s21);
                s30 = _mm256_fmadd_pd(_mm256_loadu_pd(a3 + i), x0, s30);
                s31 = _mm256_fmadd_pd(_mm256_loadu_pd(a3 + i + 4), x1, s31);
#else
                s00 = _mm256_add_pd(s00, _mm256_mul_pd(_mm256_loadu_pd(a0 + i), x0));
                s01 = _mm256_add_pd(s01, _mm256_mul_pd(_mm256_loadu_pd(a0 + i + 4), x1));
                s10 = _mm256_add_pd(s10, _mm256_mul_pd(_mm256_loadu_pd(a1 + i), x0));
                s11 = _mm256_add_pd(s11, _mm256_mul_pd(_mm256_loadu_pd(a1 + i + 4), x1));
                s20 = _mm256_add_pd(s20, _mm256_mul_pd(_mm256_loadu_pd(a2 + i), x0));
                s21 = _mm256_add_pd(s21, _mm256_mul_pd(_mm256_loadu_pd(a2 + i + 4), x1));
                s30 = _mm256_add_pd(s30, _mm256_mul_pd(_mm256_loadu_pd(a3 + i), x0));
                s31 = _mm256_add_pd(s31, _mm256_mul_pd(_mm256_loadu_pd(a3 + i + 4), x1));
#endif
            }
            if (i + 3 < n) {
                __m256d x0 = _mm256_loadu_pd(x + i);
#if defined(__FMA__)
                s00 = _mm256_fmadd_pd(_mm256_loadu_pd(a0 + i), x0, s00);
                s10 = _mm256_fmadd_pd(_mm256_loadu_pd(a1 + i), x0, s10);
                s20 = _mm256_fmadd_pd(_mm256_loadu_pd(a2 + i), x0, s20);
                s30 = _mm256_fmadd_pd(_mm256_loadu_pd(a3 + i), x0, s30);
#else
                s00 = _mm256_add_pd(s00, _mm256_mul_pd(_mm256_loadu_pd(a0 + i), x0));
                s10 = _mm256_add_pd(s10, _mm256_mul_pd(_mm256_loadu_pd(a1 + i), x0));
                s20 = _mm256_add_pd(s20, _mm256_mul_pd(_mm256_loadu_pd(a2 + i), x0));
                s30 = _mm256_add_pd(s30, _mm256_mul_pd(_mm256_loadu_pd(a3 + i), x0));
#endif
                i += 4;
            }
            __m256d t0 = _mm256_add_pd(s00, s01), t1 = _mm256_add_pd(s10, s11);
            __m256d t2 = _mm256_add_pd(s20, s21), t3 = _mm256_add_pd(s30, s31);
            __m128d p0 = _mm_add_pd(_mm256_castpd256_pd128(t0), _mm256_extractf128_pd(t0, 1));
            __m128d p1 = _mm_add_pd(_mm256_castpd256_pd128(t1), _mm256_extractf128_pd(t1, 1));
            __m128d p2 = _mm_add_pd(_mm256_castpd256_pd128(t2), _mm256_extractf128_pd(t2, 1));
            __m128d p3 = _mm_add_pd(_mm256_castpd256_pd128(t3), _mm256_extractf128_pd(t3, 1));
            double v0 = _mm_cvtsd_f64(_mm_add_sd(p0, _mm_unpackhi_pd(p0, p0)));
            double v1 = _mm_cvtsd_f64(_mm_add_sd(p1, _mm_unpackhi_pd(p1, p1)));
            double v2 = _mm_cvtsd_f64(_mm_add_sd(p2, _mm_unpackhi_pd(p2, p2)));
            double v3 = _mm_cvtsd_f64(_mm_add_sd(p3, _mm_unpackhi_pd(p3, p3)));
            /* Keep scalar tails as independent reductions. Interleaving
             * rows here permits a different compiler contraction boundary. */
            int tail = i;
            for (i = tail; i < n; ++i)
                v0 += a0[i] * x[i];
            for (i = tail; i < n; ++i)
                v1 += a1[i] * x[i];
            for (i = tail; i < n; ++i)
                v2 += a2[i] * x[i];
            for (i = tail; i < n; ++i)
                v3 += a3[i] * x[i];
            out[r] = v0;
            out[r + 1] = v1;
            out[r + 2] = v2;
            out[r + 3] = v3;
        }
#endif
    for (; r < rows; ++r)
        out[r] = dot(A + (size_t)r * stride, x, n);
}

/* Existing machine-width eligibility, shared by the ordinary sweep
 * producer and its affine consumer. Ineligible work keeps their original
 * fused scalar loop instead of materializing an intermediate vector. */
ICORE_OCP_INLINE int row_dots_grouped(int rows, int n)
{
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    return n >= 4 && rows >= 4;
#else
    (void)rows;
    (void)n;
    return 0;
#endif
}

/* The no-SIMD case keeps the original inlined dot loop. Calling the
 * larger vector kernel only when it can group four rows avoids a full
 * out-of-line call for short controls or fewer than four output rows.
 * These are the existing machine-width requirements, not a size crossover. */
ICORE_OCP_INLINE void row_dots(const double *A, int rows, int stride, const double *x, int n,
                          double *out)
{
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    if (row_dots_grouped(rows, n)) {
        row_dots_vector(A, rows, stride, x, n, out);
        return;
    }
#endif
    for (int r = 0; r < rows; ++r)
        out[r] = dot(A + (size_t)r * stride, x, n);
}

/* Ordinary magnitudes use a vector dot; the scaled path preserves range
 * when squaring would overflow or underflow. */
static double norm2(const double *a, int n)
{
    double scale = 0.0, sum = 1.0;
    int i;
    double square = dot(a, a, n);
    if (square >= DBL_MIN && isfinite(square))
        return sqrt(square);
    for (i = 0; i < n; ++i) {
        double x = fabs(a[i]);
        if (x > 0.0) {
            if (scale < x) {
                double r = scale / x;
                sum = 1.0 + sum * r * r;
                scale = x;
            } else {
                double r = x / scale;
                sum += r * r;
            }
        }
    }
    return scale == 0.0 ? 0.0 : scale * sqrt(sum);
}

static double norm_inf(const double *a, size_t n)
{
    double s = 0.0;
    size_t i = 0;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    if (n >= 4) {
        __m256d maximum = _mm256_setzero_pd(), sign = _mm256_set1_pd(-0.0);
        __m256d finite = _mm256_set1_pd(DBL_MAX);
        __m128d pair;
        for (; i + 3 < n; i += 4) {
            __m256d value = _mm256_andnot_pd(sign, _mm256_loadu_pd(a + i));
            if (_mm256_movemask_pd(_mm256_cmp_pd(value, finite, _CMP_LE_OQ)) != 15)
                return INFINITY;
            maximum = _mm256_max_pd(maximum, value);
        }
        pair = _mm_max_pd(_mm256_castpd256_pd128(maximum), _mm256_extractf128_pd(maximum, 1));
        s = _mm_cvtsd_f64(_mm_max_sd(pair, _mm_unpackhi_pd(pair, pair)));
    }
#endif
    for (; i < n; ++i) {
        double x = fabs(a[i]);
        if (!isfinite(x))
            return INFINITY;
        if (x > s)
            s = x;
    }
    return s;
}

ICORE_OCP_INLINE void axpy(double *ICORE_OCP_RESTRICT a, const double *ICORE_OCP_RESTRICT b, double s, int n)
{
    int i = 0;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    __m256d scale = _mm256_set1_pd(s);
    for (; i + 3 < n; i += 4) {
#if defined(__FMA__)
        __m256d value = _mm256_fmadd_pd(scale, _mm256_loadu_pd(b + i), _mm256_loadu_pd(a + i));
#else
        __m256d value =
            _mm256_add_pd(_mm256_mul_pd(scale, _mm256_loadu_pd(b + i)), _mm256_loadu_pd(a + i));
#endif
        _mm256_storeu_pd(a + i, value);
    }
#endif
    for (; i < n; ++i)
        a[i] += s * b[i];
}

/* Row-major triangular solve with simultaneous right-hand sides. Keep
 * output tiles in registers across all preceding triangular rows, instead
 * of storing the entire row after each AXPY. The FMA and division order of
 * every retained scalar is unchanged. */
static void lower_solve_multiple(const double *L, int n, double *B, int columns)
{
    int i, j, c;
    for (i = 0; i < n; ++i) {
        c = 0;
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
        __m256d diagonal = _mm256_set1_pd(L[(size_t)i * n + i]);
        for (; c + 15 < columns; c += 16) {
            __m256d r0 = _mm256_loadu_pd(B + (size_t)i * columns + c);
            __m256d r1 = _mm256_loadu_pd(B + (size_t)i * columns + c + 4);
            __m256d r2 = _mm256_loadu_pd(B + (size_t)i * columns + c + 8);
            __m256d r3 = _mm256_loadu_pd(B + (size_t)i * columns + c + 12);
            for (j = 0; j < i; ++j) {
                __m256d l = _mm256_set1_pd(-L[(size_t)i * n + j]);
                const double *b = B + (size_t)j * columns + c;
                r0 = _mm256_fmadd_pd(l, _mm256_loadu_pd(b), r0);
                r1 = _mm256_fmadd_pd(l, _mm256_loadu_pd(b + 4), r1);
                r2 = _mm256_fmadd_pd(l, _mm256_loadu_pd(b + 8), r2);
                r3 = _mm256_fmadd_pd(l, _mm256_loadu_pd(b + 12), r3);
            }
            _mm256_storeu_pd(B + (size_t)i * columns + c, _mm256_div_pd(r0, diagonal));
            _mm256_storeu_pd(B + (size_t)i * columns + c + 4, _mm256_div_pd(r1, diagonal));
            _mm256_storeu_pd(B + (size_t)i * columns + c + 8, _mm256_div_pd(r2, diagonal));
            _mm256_storeu_pd(B + (size_t)i * columns + c + 12, _mm256_div_pd(r3, diagonal));
        }
        for (; c + 7 < columns; c += 8) {
            __m256d r0 = _mm256_loadu_pd(B + (size_t)i * columns + c),
                    r1 = _mm256_loadu_pd(B + (size_t)i * columns + c + 4);
            for (j = 0; j < i; ++j) {
                __m256d l = _mm256_set1_pd(-L[(size_t)i * n + j]);
                const double *b = B + (size_t)j * columns + c;
                r0 = _mm256_fmadd_pd(l, _mm256_loadu_pd(b), r0);
                r1 = _mm256_fmadd_pd(l, _mm256_loadu_pd(b + 4), r1);
            }
            _mm256_storeu_pd(B + (size_t)i * columns + c, _mm256_div_pd(r0, diagonal));
            _mm256_storeu_pd(B + (size_t)i * columns + c + 4, _mm256_div_pd(r1, diagonal));
        }
        for (; c + 3 < columns; c += 4) {
            __m256d r = _mm256_loadu_pd(B + (size_t)i * columns + c);
            for (j = 0; j < i; ++j)
                r = _mm256_fmadd_pd(_mm256_set1_pd(-L[(size_t)i * n + j]),
                                    _mm256_loadu_pd(B + (size_t)j * columns + c), r);
            _mm256_storeu_pd(B + (size_t)i * columns + c, _mm256_div_pd(r, diagonal));
        }
#endif
        /* Preserve the original scalar/portable tail. */
        for (j = 0; j < i; ++j)
            axpy(B + (size_t)i * columns + c, B + (size_t)j * columns + c, -L[(size_t)i * n + j],
                 columns - c);
        for (; c < columns; ++c)
            B[(size_t)i * columns + c] /= L[(size_t)i * n + i];
    }
}

/* Keep each output tile in registers while traversing the rows. Each lane
 * retains the existing AXPY seed and ascending multiply-add order. */
static void dense_transpose_add(const double *D, int rows, int stride, const double *x, double *out,
                                int columns)
{
    int j = 0, i;
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR)
    /* Independent output columns hide the FMA dependency latency without
     * regrouping a single dot product or changing its initial seed. */
    for (; j + 15 < columns; j += 16) {
        __m256d v0 = _mm256_loadu_pd(out + j);
        __m256d v1 = _mm256_loadu_pd(out + j + 4);
        __m256d v2 = _mm256_loadu_pd(out + j + 8);
        __m256d v3 = _mm256_loadu_pd(out + j + 12);
        for (i = 0; i < rows; ++i) {
            const double *row = D + (size_t)i * stride + j;
            __m256d a = _mm256_set1_pd(x[i]);
            v0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(row), v0);
            v1 = _mm256_fmadd_pd(a, _mm256_loadu_pd(row + 4), v1);
            v2 = _mm256_fmadd_pd(a, _mm256_loadu_pd(row + 8), v2);
            v3 = _mm256_fmadd_pd(a, _mm256_loadu_pd(row + 12), v3);
        }
        _mm256_storeu_pd(out + j, v0);
        _mm256_storeu_pd(out + j + 4, v1);
        _mm256_storeu_pd(out + j + 8, v2);
        _mm256_storeu_pd(out + j + 12, v3);
    }
    for (; j + 7 < columns; j += 8) {
        __m256d v0 = _mm256_loadu_pd(out + j);
        __m256d v1 = _mm256_loadu_pd(out + j + 4);
        for (i = 0; i < rows; ++i) {
            const double *row = D + (size_t)i * stride + j;
            __m256d a = _mm256_set1_pd(x[i]);
            v0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(row), v0);
            v1 = _mm256_fmadd_pd(a, _mm256_loadu_pd(row + 4), v1);
        }
        _mm256_storeu_pd(out + j, v0);
        _mm256_storeu_pd(out + j + 4, v1);
    }
    for (; j + 3 < columns; j += 4) {
        __m256d value = _mm256_loadu_pd(out + j);
        for (i = 0; i < rows; ++i)
            value = _mm256_fmadd_pd(_mm256_set1_pd(x[i]),
                                    _mm256_loadu_pd(D + (size_t)i * stride + j), value);
        _mm256_storeu_pd(out + j, value);
    }
#endif
    /* Keep the existing scalar-tail expression and compiler contraction. */
    for (i = 0; i < rows; ++i)
        axpy(out + j, D + (size_t)i * stride + j, x[i], columns - j);
}

/* Exact supports are shared by an SOC's transformed rows.  A contiguous
 * support takes the ordinary vector path, including dense and scalar rows. */
ICORE_OCP_INLINE double supported_dot(const icore_ocp_workspace *w, size_t row, const double *a,
                                 const double *v, const double *single)
{
    int count = w->nnz[row], i;
    const int *idx = w->indices + w->index_start[row];
    double sum = 0.0;
    if (!count)
        return 0.0;
    if (count == 1)
        return single[row] * v[w->single_index[row]];
    if (count == 2)
        return a[idx[0]] * v[idx[0]] + a[idx[1]] * v[idx[1]];
    if (idx[count - 1] - idx[0] + 1 == count)
        return dot(a + idx[0], v + idx[0], count);
    for (i = 0; i < count; ++i)
        sum += a[idx[i]] * v[idx[i]];
    return sum;
}

/* Singleton rows have one rounded product. The register boundary keeps
 * that product separate from the following residual or dual accumulation.
 * Mask both indices and coefficients so constant rows return positive zero. */
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR) && (defined(__GNUC__) || defined(__clang__))
#define ICORE_OCP_SINGLETON_GV4 1

ICORE_OCP_INLINE __m256d singleton_gv4(const icore_ocp_workspace *w, size_t row, const double *v)
{
    __m128i indices = _mm_loadu_si128((const __m128i *)(w->single_index + row));
    __m128i active32 = _mm_cmpgt_epi32(indices, _mm_set1_epi32(-1));
    __m256d active = _mm256_castsi256_pd(_mm256_cvtepi32_epi64(active32));
    __m256d values = _mm256_and_pd(active, _mm256_loadu_pd(w->single_value + row));
    __m256d state = _mm256_mask_i32gather_pd(_mm256_setzero_pd(), v, indices, active, 8);
    __m256d product = _mm256_mul_pd(values, state);
    __asm__ __volatile__("" : "+x"(product));
    return product;
}
#else
#define ICORE_OCP_SINGLETON_GV4 0
#endif

ICORE_OCP_INLINE void supported_axpy(const icore_ocp_workspace *w, size_t row, double *out, const double *a,
                                double value, const double *single)
{
    int count = w->nnz[row], i;
    const int *idx = w->indices + w->index_start[row];
    if (!count)
        return;
    if (count == 1) {
        out[w->single_index[row]] += value * single[row];
        return;
    }
    if (count == 2) {
        out[idx[0]] += value * a[idx[0]];
        out[idx[1]] += value * a[idx[1]];
        return;
    }
    if (idx[count - 1] - idx[0] + 1 == count) {
        axpy(out + idx[0], a + idx[0], value, count);
        return;
    }
    /* A nearly dense row is faster as contiguous vector FMAs than indexed
     * scalar updates. Off-support entries are owned exact zeros. Only use
     * this route for a finite multiplier so zero*Inf cannot poison unrelated
     * coordinates. Nonfinite multipliers retain the support-only path. */
    if (count >= 4 && count * 2 >= w->nv && isfinite(value)) {
        axpy(out, a, value, w->nv);
        return;
    }
    for (i = 0; i < count; ++i)
        out[idx[i]] += value * a[idx[i]];
}

/* Eight singleton rows can update four consecutive coordinates in row
 * order. The exact current indices certify the layout, including mixed
 * bound/general stages. No gather, allocation or coefficient approximation. */
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR) && (defined(__GNUC__) || defined(__clang__))
#define ICORE_OCP_SINGLETON_PAIRS8 1

ICORE_OCP_INLINE int singleton_pairs8(const icore_ocp_workspace *w, size_t row, const double *x, double *out)
{
    int first = w->single_index[row];
    if (first < 0 || w->nv < 4 || first > w->nv - 4)
        return 0;
    __m256i indices = _mm256_loadu_si256((const __m256i *)(w->single_index + row));
    __m256i expected = _mm256_setr_epi32(first, first, first + 1, first + 1, first + 2, first + 2,
                                         first + 3, first + 3);
    if (_mm256_movemask_epi8(_mm256_cmpeq_epi32(indices, expected)) != -1)
        return 0;
    const double *g = w->single_value + row;
    __m256d value = _mm256_setr_pd(out[first] + x[0] * g[0], out[first + 1] + x[2] * g[2],
                                   out[first + 2] + x[4] * g[4], out[first + 3] + x[6] * g[6]);
    value = _mm256_setr_pd(value[0] + x[1] * g[1], value[1] + x[3] * g[3], value[2] + x[5] * g[5],
                           value[3] + x[7] * g[7]);
    _mm256_storeu_pd(out + first, value);
    return 1;
}
#else
#define ICORE_OCP_SINGLETON_PAIRS8 0
#endif

/* Gate grouping cheaply from the first two dense rows. Irregular sparse
 * models keep their direct support loop without a second structure scan. */
static int constraint_transpose_grouped(const icore_ocp_workspace *w, int stage)
{
#if ICORE_OCP_SINGLETON_PAIRS8
    if (w->m >= 8) {
        size_t row = (size_t)stage * w->m;
        int first = w->single_index[row];
        if (first >= 0 && w->nv >= 4 && first <= w->nv - 4) {
            __m256i indices = _mm256_loadu_si256((const __m256i *)(w->single_index + row));
            __m256i expected = _mm256_setr_epi32(first, first, first + 1, first + 1, first + 2,
                                                 first + 2, first + 3, first + 3);
            if (_mm256_movemask_epi8(_mm256_cmpeq_epi32(indices, expected)) == -1)
                return 1;
        }
    }
#endif
    const int *rows = w->dense_rows + (size_t)stage * w->m;
    return w->dense_count[stage] >= 2 && rows[1] == rows[0] + 1 &&
           w->constraint_group_end[(size_t)stage * w->m + rows[0]] >= rows[0] + 2;
}

/* Consecutive rows sharing one contiguous support can accumulate their
 * transpose product entirely in registers. Row order and each output's
 * original seed are unchanged; irregular and singleton rows keep the
 * support-aware path. No structure assumptions survive a model update. */
static void constraint_transpose_add(const icore_ocp_workspace *w, int stage, const double *G,
                                     const double *x, double *out)
{
    int row = 0, m = w->m, nv = w->nv;
    const size_t base = (size_t)stage * m;
    while (row < m) {
        size_t pos = base + row;
        int end = w->constraint_group_end[pos];
        if (end - row >= 2) {
            int count = w->nnz[pos], first = w->indices[w->index_start[pos]];
            dense_transpose_add(G + (size_t)row * nv + first, end - row, nv, x + row, out + first,
                                count);
            row = end;
        } else {
#if ICORE_OCP_SINGLETON_PAIRS8
            if (m - row >= 8 && singleton_pairs8(w, pos, x + row, out)) {
                row += 8;
                continue;
            }
#endif
            supported_axpy(w, pos, out, G + (size_t)row * nv, x[row], w->single_value);
            ++row;
        }
    }
}

/* Four independent singleton rows can share one contiguous output update.
 * The committed structure cache sets a nonnegative single_index exactly for
 * singleton rows. Callers keep ascending row order and gate the singleton
 * model path. Model coefficients and working outputs are separately owned.
 * Nonfinite operands retain the scalar instruction order and NaN behavior. */
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR) &&                              \
    (defined(__GNUC__) || defined(__clang__))
#define ICORE_OCP_SINGLETON_AXPY4 1

ICORE_OCP_INLINE int singleton_axpy4(const icore_ocp_workspace *w, size_t row, double *out, __m256d values)
{
    int first = w->single_index[row];
    __m128i indices, expected;
    __m256d coefficients, previous;
    __m256i exponent, nonfinite;
    if (first < 0 || w->nv < 4 || first > w->nv - 4)
        return 0;
    indices = _mm_loadu_si128((const __m128i *)(w->single_index + row));
    expected = _mm_setr_epi32(first, first + 1, first + 2, first + 3);
    if (_mm_movemask_epi8(_mm_cmpeq_epi32(indices, expected)) != 65535)
        return 0;
    coefficients = _mm256_loadu_pd(w->single_value + row);
    previous = _mm256_loadu_pd(out + first);
    exponent = _mm256_set1_epi64x(INT64_C(0x7ff0000000000000));
    nonfinite = _mm256_or_si256(
        _mm256_cmpeq_epi64(_mm256_and_si256(_mm256_castpd_si256(values), exponent), exponent),
        _mm256_cmpeq_epi64(_mm256_and_si256(_mm256_castpd_si256(coefficients), exponent),
                           exponent));
    nonfinite = _mm256_or_si256(
        nonfinite,
        _mm256_cmpeq_epi64(_mm256_and_si256(_mm256_castpd_si256(previous), exponent), exponent));
    if (_mm256_movemask_pd(_mm256_castsi256_pd(nonfinite)))
        return 0;
    /* Spell the four scalar expressions exactly as supported_axpy does.
     * This retains the compiler's C/C++ contraction boundary: the default
     * GCC C11 path rounds the product; C++17 can contract each expression. */
    _mm256_storeu_pd(out + first, _mm256_setr_pd(previous[0] + values[0] * coefficients[0],
                                                 previous[1] + values[1] * coefficients[1],
                                                 previous[2] + values[2] * coefficients[2],
                                                 previous[3] + values[3] * coefficients[3]));
    return 1;
}
#else
#define ICORE_OCP_SINGLETON_AXPY4 0
#endif

static int finite_array(const double *a, size_t n)
{
    size_t i = 0;
    if (n && !a)
        return 0;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    __m256d sign = _mm256_set1_pd(-0.0), finite = _mm256_set1_pd(DBL_MAX);
    for (; i + 3 < n; i += 4) {
        __m256d value = _mm256_andnot_pd(sign, _mm256_loadu_pd(a + i));
        if (_mm256_movemask_pd(_mm256_cmp_pd(value, finite, _CMP_LE_OQ)) != 15)
            return 0;
    }
#endif
    for (; i < n; ++i)
        if (!isfinite(a[i]))
            return 0;
    return 1;
}

/* Structural scans perform no floating-point arithmetic.  The vector path
 * preserves the exact zero/nonzero classification (including signed zero),
 * and emits supports in the original ascending column order. */
ICORE_OCP_INLINE int structural_first_nonzero(const double *a, int n)
{
    int i = 0;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    const __m256d zero = _mm256_setzero_pd();
    for (; n - i >= 4; i += 4) {
        unsigned mask =
            (unsigned)_mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(a + i), zero, _CMP_NEQ_UQ));
        if (mask)
            return i + __builtin_ctz(mask);
    }
#endif
    for (; i < n; ++i)
        if (a[i] != 0.0)
            break;
    return i;
}

ICORE_OCP_INLINE int structural_last_nonzero(const double *a, int n)
{
    int i = n;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    const __m256d zero = _mm256_setzero_pd();
    while (i >= 4) {
        unsigned mask = (unsigned)_mm256_movemask_pd(
            _mm256_cmp_pd(_mm256_loadu_pd(a + i - 4), zero, _CMP_NEQ_UQ));
        if (mask)
            return i - 4 + (int)(32 - __builtin_clz(mask));
        i -= 4;
    }
#endif
    while (i > 0 && a[i - 1] == 0.0)
        --i;
    return i;
}

ICORE_OCP_INLINE int structural_support(const double *a, int n, int *indices)
{
    int i = 0, count = 0;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    const __m256d zero = _mm256_setzero_pd();
    for (; n - i >= 4; i += 4) {
        unsigned mask =
            (unsigned)_mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(a + i), zero, _CMP_NEQ_UQ));
        while (mask) {
            indices[count++] = i + __builtin_ctz(mask);
            mask &= mask - 1u;
        }
    }
#endif
    for (; i < n; ++i)
        if (a[i] != 0.0)
            indices[count++] = i;
    return count;
}

/* The sparse kernels are used only below the existing one-quarter density
 * threshold. Stop counting as soon as a stage is provably dense, without
 * constructing the unused column-index/value representation. */
static int dynamics_sparse_structure(const double *D, int nx, int nv, int first)
{
    size_t entries = (size_t)nx * (size_t)(nv - first), count = 0;
    size_t threshold = (entries - 1u) / 4u + 1u;
    int i, j;
    for (i = 0; i < nx; ++i) {
        const double *row = D + (size_t)i * nv;
        j = first;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
        const __m256d zero = _mm256_setzero_pd();
        for (; nv - j >= 4; j += 4) {
            unsigned mask = (unsigned)_mm256_movemask_pd(
                _mm256_cmp_pd(_mm256_loadu_pd(row + j), zero, _CMP_NEQ_UQ));
            count += (unsigned)__builtin_popcount(mask);
            if (count >= threshold)
                return 0;
        }
#endif
        for (; j < nv; ++j)
            if (row[j] != 0.0 && ++count >= threshold)
                return 0;
    }
    return 1;
}

ICORE_OCP_INLINE int symmetric_pair_valid(double a, double b)
{
    return a == b || fabs(a - b) <= 1e-12 * fmax(1.0, fmax(fabs(a), fabs(b)));
}

/* Validate four-by-four off-diagonal tiles in contiguous rows.  An exact
 * vector equality succeeds without scalar gathers; differing tiles still
 * use precisely the existing relative symmetry rule for every entry. */
static int symmetric_coefficients(const double *H, int n)
{
    int i = 0, j, a, b;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    for (; n - i >= 4; i += 4) {
        for (j = 0; j < i; j += 4) {
            __m256d q0 = _mm256_loadu_pd(H + (size_t)j * n + i);
            __m256d q1 = _mm256_loadu_pd(H + (size_t)(j + 1) * n + i);
            __m256d q2 = _mm256_loadu_pd(H + (size_t)(j + 2) * n + i);
            __m256d q3 = _mm256_loadu_pd(H + (size_t)(j + 3) * n + i);
            __m256d t0 = _mm256_unpacklo_pd(q0, q1), t1 = _mm256_unpackhi_pd(q0, q1);
            __m256d t2 = _mm256_unpacklo_pd(q2, q3), t3 = _mm256_unpackhi_pd(q2, q3);
            __m256d e0 = _mm256_cmp_pd(_mm256_loadu_pd(H + (size_t)i * n + j),
                                       _mm256_permute2f128_pd(t0, t2, 0x20), _CMP_EQ_OQ);
            __m256d e1 = _mm256_cmp_pd(_mm256_loadu_pd(H + (size_t)(i + 1) * n + j),
                                       _mm256_permute2f128_pd(t1, t3, 0x20), _CMP_EQ_OQ);
            __m256d e2 = _mm256_cmp_pd(_mm256_loadu_pd(H + (size_t)(i + 2) * n + j),
                                       _mm256_permute2f128_pd(t0, t2, 0x31), _CMP_EQ_OQ);
            __m256d e3 = _mm256_cmp_pd(_mm256_loadu_pd(H + (size_t)(i + 3) * n + j),
                                       _mm256_permute2f128_pd(t1, t3, 0x31), _CMP_EQ_OQ);
            if (_mm256_movemask_pd(_mm256_and_pd(_mm256_and_pd(e0, e1), _mm256_and_pd(e2, e3))) ==
                15)
                continue;
            for (a = 0; a < 4; ++a)
                for (b = 0; b < 4; ++b)
                    if (!symmetric_pair_valid(H[(size_t)(i + a) * n + j + b],
                                              H[(size_t)(j + b) * n + i + a]))
                        return 0;
        }
        for (a = 1; a < 4; ++a)
            for (b = 0; b < a; ++b)
                if (!symmetric_pair_valid(H[(size_t)(i + a) * n + i + b],
                                          H[(size_t)(i + b) * n + i + a]))
                    return 0;
    }
#else
    (void)a;
    (void)b;
#endif
    for (; i < n; ++i)
        for (j = 0; j < i; ++j)
            if (!symmetric_pair_valid(H[(size_t)i * n + j], H[(size_t)j * n + i]))
                return 0;
    return 1;
}
