/* Orthant boundary and ratio-scaling primitives shared by the OCP kernel.
 * Keep the arithmetic order and original finite guards unchanged.
 */

#ifndef ICORE_ORTHANT_STEP_H
#define ICORE_ORTHANT_STEP_H

#include <math.h>
#include <stddef.h>
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
#include <immintrin.h>
#endif

/* Exact orthant boundary min(1, min_{d_i<0} -x_i/d_i).  An optional index
 * array selects active constraints without reading inactive storage.  This
 * changes neither the fraction-to-boundary rule nor the backtracking test.
 * Inactive lanes use denominator one, avoiding spurious zero divisions.
 * A negative-direction lane with x >= -d has quotient >= 1 and cannot
 * improve any current lane minimum. Ordered comparison leaves NaNs on the
 * original division path; the certified +Inf/-Inf case also preserves best
 * because the original MINPD ignores a NaN first operand. The same negative
 * mask handles DAZ-subnormal directions, and no rounded product is tested. */
static inline double icore_ocp_orthant_step(const double *x, const double *d, const int *index,
                                       int count)
{
    int i = 0;
    double alpha = 1.0;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    __m256d best = _mm256_set1_pd(1.0);
    const __m256d one = _mm256_set1_pd(1.0), zero = _mm256_setzero_pd();
    const __m256d sign = _mm256_set1_pd(-0.0);
    if (index) {
        for (; i + 4 <= count; i += 4) {
            __m128i ids = _mm_loadu_si128((const __m128i *)(index + i));
            __m256d xi = _mm256_i32gather_pd(x, ids, 8);
            __m256d di = _mm256_i32gather_pd(d, ids, 8);
            __m256d negative = _mm256_cmp_pd(di, zero, _CMP_LT_OQ);
            __m256d nonlimiting = _mm256_cmp_pd(xi, _mm256_xor_pd(di, sign), _CMP_GE_OQ);
            if (!_mm256_movemask_pd(_mm256_andnot_pd(nonlimiting, negative)))
                continue;
            __m256d denominator = _mm256_blendv_pd(one, di, negative);
            __m256d bound = _mm256_div_pd(_mm256_xor_pd(xi, sign), denominator);
            bound = _mm256_blendv_pd(one, bound, negative);
            best = _mm256_min_pd(bound, best);
        }
    } else {
        for (; i + 4 <= count; i += 4) {
            __m256d xi = _mm256_loadu_pd(x + i), di = _mm256_loadu_pd(d + i);
            __m256d negative = _mm256_cmp_pd(di, zero, _CMP_LT_OQ);
            __m256d nonlimiting = _mm256_cmp_pd(xi, _mm256_xor_pd(di, sign), _CMP_GE_OQ);
            if (!_mm256_movemask_pd(_mm256_andnot_pd(nonlimiting, negative)))
                continue;
            __m256d denominator = _mm256_blendv_pd(one, di, negative);
            __m256d bound = _mm256_div_pd(_mm256_xor_pd(xi, sign), denominator);
            bound = _mm256_blendv_pd(one, bound, negative);
            best = _mm256_min_pd(bound, best);
        }
    }
    {
        /* best starts at one, and MINPD(bound,best) cannot install a
         * NaN. Reduce these certified non-NaN lanes in registers rather
         * than making three scalar libm calls. Preserve pair ordering. */
        __m128d lo = _mm256_castpd256_pd128(best), hi = _mm256_extractf128_pd(best, 1);
        lo = _mm_min_sd(lo, _mm_unpackhi_pd(lo, lo));
        hi = _mm_min_sd(hi, _mm_unpackhi_pd(hi, hi));
        alpha = _mm_cvtsd_f64(_mm_min_sd(lo, hi));
    }
#endif
    for (; i < count; ++i) {
        int row = index ? index[i] : i;
        if (d[row] < 0.0)
            alpha = fmin(alpha, -x[row] / d[row]);
    }
    return alpha;
}

#endif

#ifndef ICORE_ORTHANT_SCALING_H
#define ICORE_ORTHANT_SCALING_H

#include <math.h>
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
#include <immintrin.h>
#endif

/* Independent orthant NT entries, retaining the scalar operation order:
 * rs=sqrt(s), rz=sqrt(z), b=rs/rz, inverse=1/b, lambda=rs*rz.
 * Do not replace these by sqrt(s/z) or sqrt(s*z): those expressions have
 * different range and rounding. Constant rows use exactly one, even if their
 * unused inputs contain nonpositive or nonfinite values. Outputs are disjoint
 * workspace arrays, not aliases of either input or one another.
 *
 * An invalid group resumes scalar evaluation at its first entry, preserving
 * the scalar success/failure result and the exact output prefix on failure.
 * No allocation, reciprocal approximation, or fast-math assumption is used.
 */
static inline int icore_ocp_orthant_scaling(const double *s, const double *z,
                                       const unsigned char *constant, double *b, double *inverse,
                                       double *lambda, int count)
{
    int i = 0;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    const __m256d one = _mm256_set1_pd(1.0), zero = _mm256_setzero_pd();
    for (; count - i >= 4; i += 4) {
        __m256d si = _mm256_loadu_pd(s + i), zi = _mm256_loadu_pd(z + i);
        __m256d rs, rz, bi;
        if (constant[i] || constant[i + 1] || constant[i + 2] || constant[i + 3]) {
            __m256d mask = _mm256_castsi256_pd(
                _mm256_set_epi64x(-(long long)!!constant[i + 3], -(long long)!!constant[i + 2],
                                  -(long long)!!constant[i + 1], -(long long)!!constant[i]));
            si = _mm256_blendv_pd(si, one, mask);
            zi = _mm256_blendv_pd(zi, one, mask);
        }
        if (_mm256_movemask_pd(_mm256_and_pd(_mm256_cmp_pd(si, zero, _CMP_GT_OQ),
                                             _mm256_cmp_pd(zi, zero, _CMP_GT_OQ))) != 15)
            break;
        rs = _mm256_sqrt_pd(si);
        rz = _mm256_sqrt_pd(zi);
        bi = _mm256_div_pd(rs, rz);
        _mm256_storeu_pd(b + i, bi);
        _mm256_storeu_pd(inverse + i, _mm256_div_pd(one, bi));
        _mm256_storeu_pd(lambda + i, _mm256_mul_pd(rs, rz));
    }
#endif
    for (; i < count; ++i) {
        double rs, rz;
        if (constant[i]) {
            b[i] = lambda[i] = inverse[i] = 1.0;
            continue;
        }
        if (!(s[i] > 0.0 && z[i] > 0.0))
            return 0;
        rs = sqrt(s[i]);
        rz = sqrt(z[i]);
        b[i] = rs / rz;
        inverse[i] = 1.0 / b[i];
        lambda[i] = rs * rz;
    }
    return 1;
}

#endif
