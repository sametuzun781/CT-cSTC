/* Certified consecutive singleton pairs for coordinates 0..n-1.
 * SIMD lanes are independent coordinates. The two contributions to each
 * coordinate retain physical row order, as does the complementarity sum. */
#if defined(__AVX2__) && defined(__FMA__) && !defined(ICORE_OCP_SCALAR) &&                              \
    (defined(__GNUC__) || defined(__clang__))
#define ICORE_OCP_BOUND_PAIRS 1

/* Explicit independent-lane FMA; original residual checks remain active. */
ICORE_OCP_INLINE __m256d bound_madd(__m256d a, __m256d b, __m256d c)
{
    return _mm256_fmadd_pd(a, b, c);
}

ICORE_OCP_INLINE __m256d bound_nmadd(__m256d a, __m256d b, __m256d c)
{
    return _mm256_fnmadd_pd(a, b, c);
}

ICORE_OCP_INLINE void bound_pair_load(const double *p, __m256d *a, __m256d *b)
{
    __m256d x = _mm256_loadu_pd(p), y = _mm256_loadu_pd(p + 4);
    *a = _mm256_permute4x64_pd(_mm256_unpacklo_pd(x, y), 0xd8);
    *b = _mm256_permute4x64_pd(_mm256_unpackhi_pd(x, y), 0xd8);
}

ICORE_OCP_INLINE void bound_pair_store(double *p, __m256d a, __m256d b)
{
    __m256d x = _mm256_unpacklo_pd(a, b), y = _mm256_unpackhi_pd(a, b);
    _mm256_storeu_pd(p, _mm256_permute2f128_pd(x, y, 0x20));
    _mm256_storeu_pd(p + 4, _mm256_permute2f128_pd(x, y, 0x31));
}

ICORE_OCP_INLINE int bound_finite4(__m256d value)
{
    return _mm256_movemask_pd(_mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), value),
                                            _mm256_set1_pd(DBL_MAX), _CMP_LE_OQ)) == 15;
}

static int bound_pairs_evaluate(icore_ocp_workspace *w, double *primal, double *dual, double *gap,
                                double *rpnorm)
{
    int i = 0, n = w->nu;
    const __m256d sign = _mm256_set1_pd(-0.0);
    __m256d maxrp = _mm256_setzero_pd(), mins = maxrp, minz = maxrp;
    for (; i + 3 < n; i += 4) {
        int r = 2 * i;
        __m256d g[2], s[2], z[2], h[2], rp[2], products[2];
        __m256d x = _mm256_loadu_pd(w->v + i), rd = _mm256_loadu_pd(w->rd + i);
        bound_pair_load(w->qp_box_value + r, &g[0], &g[1]);
        bound_pair_load(w->s + r, &s[0], &s[1]);
        bound_pair_load(w->z + r, &z[0], &z[1]);
        bound_pair_load(w->qp_h + r, &h[0], &h[1]);
        for (int b = 0; b < 2; ++b) {
            __m256d gv = _mm256_mul_pd(g[b], x);
            __asm__ __volatile__("" : "+x"(gv));
            rp[b] = _mm256_sub_pd(_mm256_add_pd(gv, s[b]), h[b]);
            if (!bound_finite4(rp[b]) || !bound_finite4(s[b]) || !bound_finite4(z[b]))
                return 0;
            rd = bound_madd(z[b], g[b], rd);
            maxrp = _mm256_max_pd(maxrp, _mm256_andnot_pd(sign, rp[b]));
            mins = _mm256_min_pd(mins, s[b]);
            minz = _mm256_min_pd(minz, z[b]);
            products[b] = _mm256_andnot_pd(sign, _mm256_mul_pd(s[b], z[b]));
        }
        bound_pair_store(w->rp + r, rp[0], rp[1]);
        _mm256_storeu_pd(w->rd + i, rd);
        double p[8];
        bound_pair_store(p, products[0], products[1]);
        for (int j = 0; j < 8; ++j)
            *gap += p[j];
    }
    for (; i < n; ++i)
        for (int b = 0; b < 2; ++b) {
            int r = 2 * i + b;
            double gv = w->qp_box_value[r] * w->v[i], rp = gv + w->s[r] - w->qp_h[r];
            if (!isfinite(rp) || !isfinite(w->s[r]) || !isfinite(w->z[r]))
                return 0;
            w->rp[r] = rp;
            w->rd[i] += w->z[r] * w->qp_box_value[r];
            if (fabs(rp) > *rpnorm)
                *rpnorm = fabs(rp);
            if (-w->s[r] > *primal)
                *primal = -w->s[r];
            if (-w->z[r] > *dual)
                *dual = -w->z[r];
            *gap += fabs(w->s[r] * w->z[r]);
        }
    double p[4], s[4], z[4];
    _mm256_storeu_pd(p, maxrp);
    _mm256_storeu_pd(s, mins);
    _mm256_storeu_pd(z, minz);
    for (int j = 0; j < 4; ++j) {
        if (p[j] > *rpnorm)
            *rpnorm = p[j];
        if (-s[j] > *primal)
            *primal = -s[j];
        if (-z[j] > *dual)
            *dual = -z[j];
    }
    return 1;
}

static void bound_pairs_diagonal(icore_ocp_workspace *w)
{
    int i = 0, n = w->nu;
    if (w->qp_unit_singletons) {
        /* g^2=1 exactly. Do not round through sqrt(W)^2, and retain
         * the original physical-row order of the two additions. */
        for (; i + 3 < n; i += 4) {
            __m256d first, second;
            bound_pair_load(w->orthant_inv + 2 * i, &first, &second);
            __m256d d = _mm256_add_pd(_mm256_loadu_pd(w->qp_diag + i), first);
            _mm256_storeu_pd(w->qp_diag + i, _mm256_add_pd(d, second));
        }
        for (; i < n; ++i)
            for (int b = 0; b < 2; ++b)
                w->qp_diag[i] += w->orthant_inv[2 * i + b];
        return;
    }
    for (; i + 3 < n; i += 4) {
        __m256d a = _mm256_mul_pd(_mm256_sqrt_pd(_mm256_loadu_pd(w->orthant_inv + 2 * i)),
                                  _mm256_loadu_pd(w->qp_box_value + 2 * i));
        __m256d b = _mm256_mul_pd(_mm256_sqrt_pd(_mm256_loadu_pd(w->orthant_inv + 2 * i + 4)),
                                  _mm256_loadu_pd(w->qp_box_value + 2 * i + 4));
        __m256d u = _mm256_permute4x64_pd(_mm256_unpacklo_pd(a, b), 0xd8);
        __m256d l = _mm256_permute4x64_pd(_mm256_unpackhi_pd(a, b), 0xd8);
        __m256d d = bound_madd(u, u, _mm256_loadu_pd(w->qp_diag + i));
        _mm256_storeu_pd(w->qp_diag + i, bound_madd(l, l, d));
    }
    for (; i < n; ++i)
        for (int b = 0; b < 2; ++b) {
            int r = 2 * i + b;
            double v = sqrt(w->orthant_inv[r]) * w->qp_box_value[r];
            w->qp_diag[i] += v * v;
        }
}

static void bound_pairs_rhs(icore_ocp_workspace *w)
{
    int i = 0, n = w->nu;
    for (; i + 3 < n; i += 4) {
        __m256d a, b, g, h;
        bound_pair_load(w->pcone + 2 * i, &a, &b);
        bound_pair_load(w->qp_box_value + 2 * i, &g, &h);
        __m256d rhs = bound_nmadd(a, g, _mm256_loadu_pd(w->rhs + i));
        _mm256_storeu_pd(w->rhs + i, bound_nmadd(b, h, rhs));
    }
    for (; i < n; ++i)
        for (int b = 0; b < 2; ++b) {
            int r = 2 * i + b;
            w->rhs[i] += -w->pcone[r] * w->qp_box_value[r];
        }
}

static int bound_pairs_recover(icore_ocp_workspace *w, const double *dx, const double *bp, double *ds,
                               double *dz)
{
    int i = 0, n = w->nu;
    for (; i + 3 < n; i += 4) {
        __m256d x = _mm256_loadu_pd(dx + i);
        __m256d a = _mm256_unpacklo_pd(x, x), b = _mm256_unpackhi_pd(x, x);
        __m256d state[2] = {_mm256_permute2f128_pd(a, b, 0x20), _mm256_permute2f128_pd(a, b, 0x31)};
        for (int j = 0; j < 2; ++j) {
            int r = 2 * i + 4 * j;
            __m256d gv = _mm256_mul_pd(_mm256_loadu_pd(w->qp_box_value + r), state[j]);
            __asm__ __volatile__("" : "+x"(gv));
            __m256d s = _mm256_sub_pd(_mm256_loadu_pd(bp + r), gv);
            __m256d z =
                bound_madd(_mm256_loadu_pd(w->orthant_inv + r), gv, _mm256_loadu_pd(w->pcone + r));
            if (!bound_finite4(s) || !bound_finite4(z))
                return 0;
            _mm256_storeu_pd(ds + r, s);
            _mm256_storeu_pd(dz + r, z);
        }
    }
    for (; i < n; ++i)
        for (int b = 0; b < 2; ++b) {
            int r = 2 * i + b;
            double gv = w->qp_box_value[r] * dx[i];
            ds[r] = bp[r] - gv;
            dz[r] = w->orthant_inv[r] * gv + w->pcone[r];
            if (!isfinite(ds[r]) || !isfinite(dz[r]))
                return 0;
        }
    return 1;
}

/* Recover the first Newton direction and check the original equations in
 * the same row traversal. SIMD lanes retain the original rounded product;
 * scalar tails recompute the residual product independently to preserve the
 * original expression's permitted contraction and rounding.
 * The original first primal RHS is exactly -rp; consume that sign change
 * directly instead of first materializing it in bcone. The dual accumulator
 * already contains H and general G. */
static int bound_pairs_finish(icore_ocp_workspace *w, const double *dx, double *ds, double *dz,
                              double *mp, double *mc)
{
    int i = 0, n = w->nu;
    const __m256d sign = _mm256_set1_pd(-0.0);
    __m256d maxp = _mm256_setzero_pd(), maxc = maxp;
    for (; i + 3 < n; i += 4) {
        __m256d x = _mm256_loadu_pd(dx + i), g[2], z[2];
        __m256d a = _mm256_unpacklo_pd(x, x), b = _mm256_unpackhi_pd(x, x);
        __m256d state[2] = {_mm256_permute2f128_pd(a, b, 0x20), _mm256_permute2f128_pd(a, b, 0x31)};
        for (int j = 0; j < 2; ++j) {
            int r = 2 * i + 4 * j;
            g[j] = _mm256_loadu_pd(w->qp_box_value + r);
            __m256d gv = _mm256_mul_pd(g[j], state[j]);
            __asm__ __volatile__("" : "+x"(gv));
            __m256d sv = _mm256_sub_pd(_mm256_xor_pd(sign, _mm256_loadu_pd(w->rp + r)), gv);
            z[j] =
                bound_madd(_mm256_loadu_pd(w->orthant_inv + r), gv, _mm256_loadu_pd(w->pcone + r));
            if (!bound_finite4(sv) || !bound_finite4(z[j]))
                return 0;
            __m256d rp = _mm256_sub_pd(
                _mm256_sub_pd(_mm256_xor_pd(sign, _mm256_loadu_pd(w->rp + r)), gv), sv);
            __m256d rc = bound_nmadd(_mm256_loadu_pd(w->z + r), sv,
                                     _mm256_xor_pd(sign, _mm256_loadu_pd(w->q + r)));
            rc = bound_nmadd(_mm256_loadu_pd(w->s + r), z[j], rc);
            if (!bound_finite4(rp) || !bound_finite4(rc))
                return 0;
            _mm256_storeu_pd(ds + r, sv);
            _mm256_storeu_pd(dz + r, z[j]);
            _mm256_storeu_pd(w->bcone + r, rp);
            _mm256_storeu_pd(w->brc + r, rc);
            maxp = _mm256_max_pd(maxp, _mm256_andnot_pd(sign, rp));
            maxc = _mm256_max_pd(maxc, _mm256_andnot_pd(sign, rc));
        }
        __m256d gu = _mm256_permute4x64_pd(_mm256_unpacklo_pd(g[0], g[1]), 0xd8);
        __m256d gl = _mm256_permute4x64_pd(_mm256_unpackhi_pd(g[0], g[1]), 0xd8);
        __m256d zu = _mm256_permute4x64_pd(_mm256_unpacklo_pd(z[0], z[1]), 0xd8);
        __m256d zl = _mm256_permute4x64_pd(_mm256_unpackhi_pd(z[0], z[1]), 0xd8);
        __m256d rd = bound_nmadd(zu, gu, _mm256_loadu_pd(w->brd + i));
        _mm256_storeu_pd(w->brd + i, bound_nmadd(zl, gl, rd));
    }
    for (; i < n; ++i)
        for (int b = 0; b < 2; ++b) {
            int r = 2 * i + b;
            double gv = w->qp_box_value[r] * dx[i];
            double sv = -w->rp[r] - gv, zv = w->orthant_inv[r] * gv + w->pcone[r];
            if (!isfinite(sv) || !isfinite(zv))
                return 0;
            const double *rg = w->qp_box_value, *rx = dx;
            /* Keep the original residual product independent of the rounded
         * recovery product, including the compiler's permitted contraction. */
            __asm__ __volatile__("" : "+r"(rg), "+r"(rx));
            double rgv = rg[r] * rx[i];
            double rp = -w->rp[r] - rgv - sv, rc = -w->q[r] - w->z[r] * sv - w->s[r] * zv;
            if (!isfinite(rp) || !isfinite(rc))
                return 0;
            ds[r] = sv;
            dz[r] = zv;
            w->brd[i] += -zv * w->qp_box_value[r];
            w->bcone[r] = rp;
            w->brc[r] = rc;
            if (fabs(rp) > *mp)
                *mp = fabs(rp);
            if (fabs(rc) > *mc)
                *mc = fabs(rc);
        }
    double p[4], c[4];
    _mm256_storeu_pd(p, maxp);
    _mm256_storeu_pd(c, maxc);
    for (int j = 0; j < 4; ++j) {
        if (p[j] > *mp)
            *mp = p[j];
        if (c[j] > *mc)
            *mc = c[j];
    }
    return 1;
}

static int bound_pairs_residual(icore_ocp_workspace *w, const double *dx, const double *ds,
                                const double *dz, double *mp, double *mc)
{
    int i = 0, n = w->nu;
    const __m256d sign = _mm256_set1_pd(-0.0);
    __m256d maxp = _mm256_setzero_pd(), maxc = maxp;
    for (; i + 3 < n; i += 4) {
        __m256d x = _mm256_loadu_pd(dx + i), g[2], z[2];
        __m256d a = _mm256_unpacklo_pd(x, x), b = _mm256_unpackhi_pd(x, x);
        __m256d state[2] = {_mm256_permute2f128_pd(a, b, 0x20), _mm256_permute2f128_pd(a, b, 0x31)};
        for (int j = 0; j < 2; ++j) {
            int r = 2 * i + 4 * j;
            g[j] = _mm256_loadu_pd(w->qp_box_value + r);
            z[j] = _mm256_loadu_pd(dz + r);
            __m256d gv = _mm256_mul_pd(g[j], state[j]);
            __asm__ __volatile__("" : "+x"(gv));
            __m256d rp =
                _mm256_sub_pd(_mm256_sub_pd(_mm256_xor_pd(sign, _mm256_loadu_pd(w->rp + r)), gv),
                              _mm256_loadu_pd(ds + r));
            __m256d rc = bound_nmadd(_mm256_loadu_pd(w->z + r), _mm256_loadu_pd(ds + r),
                                     _mm256_xor_pd(sign, _mm256_loadu_pd(w->q + r)));
            rc = bound_nmadd(_mm256_loadu_pd(w->s + r), z[j], rc);
            if (!bound_finite4(rp) || !bound_finite4(rc))
                return 0;
            _mm256_storeu_pd(w->bcone + r, rp);
            _mm256_storeu_pd(w->brc + r, rc);
            maxp = _mm256_max_pd(maxp, _mm256_andnot_pd(sign, rp));
            maxc = _mm256_max_pd(maxc, _mm256_andnot_pd(sign, rc));
        }
        __m256d gu = _mm256_permute4x64_pd(_mm256_unpacklo_pd(g[0], g[1]), 0xd8);
        __m256d gl = _mm256_permute4x64_pd(_mm256_unpackhi_pd(g[0], g[1]), 0xd8);
        __m256d zu = _mm256_permute4x64_pd(_mm256_unpacklo_pd(z[0], z[1]), 0xd8);
        __m256d zl = _mm256_permute4x64_pd(_mm256_unpackhi_pd(z[0], z[1]), 0xd8);
        __m256d rd = bound_nmadd(zu, gu, _mm256_loadu_pd(w->brd + i));
        _mm256_storeu_pd(w->brd + i, bound_nmadd(zl, gl, rd));
    }
    for (; i < n; ++i)
        for (int b = 0; b < 2; ++b) {
            int r = 2 * i + b;
            double gv = w->qp_box_value[r] * dx[i];
            double rp = -w->rp[r] - gv - ds[r], rc = -w->q[r] - w->z[r] * ds[r] - w->s[r] * dz[r];
            w->brd[i] += -dz[r] * w->qp_box_value[r];
            w->bcone[r] = rp;
            w->brc[r] = rc;
            if (!isfinite(rp) || !isfinite(rc))
                return 0;
            if (fabs(rp) > *mp)
                *mp = fabs(rp);
            if (fabs(rc) > *mc)
                *mc = fabs(rc);
        }
    double p[4], c[4];
    _mm256_storeu_pd(p, maxp);
    _mm256_storeu_pd(c, maxc);
    for (int j = 0; j < 4; ++j) {
        if (p[j] > *mp)
            *mp = p[j];
        if (c[j] > *mc)
            *mc = c[j];
    }
    return 1;
}
#else
#define ICORE_OCP_BOUND_PAIRS 0
#endif
