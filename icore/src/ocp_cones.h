/* Cone arithmetic, residual statistics, scaling, and admissible steps.
 * Scalar inequalities use positive slacks; SOC blocks use radial and angular parts.
 * Direct orthant scaling and general cone scaling solve the same Newton equations.
 * Private implementation: included exactly once by ocp.c. */

/* The radial/angular expression stays nonnegative even at complementarity. */
static double soc_pair(const double *a, const double *b, int n)
{
    double ra = norm2(a + 1, n - 1), rb = norm2(b + 1, n - 1), angular = 0.0;
    int i;
    if (ra == 0.0 || rb == 0.0)
        return a[0] * b[0];
    for (i = 1; i < n; ++i) {
        double u = a[i] / ra + b[i] / rb;
        angular += u * u;
    }
    return (a[0] - ra) * b[0] + ra * (b[0] - rb) + 0.5 * ra * rb * angular;
}

static double cone_pair(const icore_ocp_workspace *w, const double *a, const double *b)
{
    double sum = 0.0;
    int k, j;
    for (k = 0; k < w->K; ++k) {
        const double *ak = a + (size_t)k * w->m, *bk = b + (size_t)k * w->m;
        for (j = 0; j < w->nlp; ++j)
            if (!w->constant_orthant[(size_t)k * w->m + j])
                sum += fabs(ak[j] * bk[j]);
        for (j = 0; j < w->nsoc; ++j) {
            int o = w->offsets[j];
            if (!constant_soc(w, k, j))
                sum += fabs(soc_pair(ak + o, bk + o, w->sizes[j]));
        }
    }
    return sum;
}

static double cone_min(const icore_ocp_workspace *w, const double *a)
{
    double low = DBL_MAX;
    int k, j;
    for (k = 0; k < w->K; ++k) {
        const double *ak = a + (size_t)k * w->m;
        for (j = 0; j < w->nlp; ++j)
            if (!w->constant_orthant[(size_t)k * w->m + j] && ak[j] < low)
                low = ak[j];
        for (j = 0; j < w->nsoc; ++j) {
            int o = w->offsets[j];
            if (constant_soc(w, k, j))
                continue;
            double t = ak[o] - norm2(ak + o + 1, w->sizes[j] - 1);
            if (t < low)
                low = t;
        }
    }
    return low;
}

/* The ordinary orthant statistics need the same two input streams.
 * Exceptional arithmetic retains the ordered cone routines. */
static void cone_statistics(const icore_ocp_workspace *w, double *smin, double *zmin, double *gap)
{
#if ICORE_OCP_SINGLETON_GV4
    if (!w->nsoc) {
        const size_t count = (size_t)w->K * w->m;
        const __m256d zero = _mm256_setzero_pd(), ceiling = _mm256_set1_pd(DBL_MAX);
        const __m256d sign = _mm256_set1_pd(-0.0);
        __m256d slo = ceiling, zlo = ceiling, total = zero;
        double sl = DBL_MAX, zl = DBL_MAX, tail = 0.0, lanes[4];
        size_t i = 0;
        for (; count - i >= 4; i += 4) {
            const unsigned char *c = w->constant_orthant + i;
            __m256d mask = _mm256_castsi256_pd(_mm256_set_epi64x(
                -(long long)!!c[3], -(long long)!!c[2], -(long long)!!c[1], -(long long)!!c[0]));
            __m256d s = _mm256_andnot_pd(mask, _mm256_loadu_pd(w->s + i));
            __m256d z = _mm256_andnot_pd(mask, _mm256_loadu_pd(w->z + i));
            __m256d ok =
                _mm256_and_pd(_mm256_cmp_pd(_mm256_andnot_pd(sign, s), ceiling, _CMP_LE_OQ),
                              _mm256_cmp_pd(_mm256_andnot_pd(sign, z), ceiling, _CMP_LE_OQ));
            __m256d product;
            if (_mm256_movemask_pd(ok) != 15)
                goto ordered_statistics;
            product = _mm256_andnot_pd(sign, _mm256_mul_pd(s, z));
            if (_mm256_movemask_pd(_mm256_cmp_pd(product, ceiling, _CMP_LE_OQ)) != 15)
                goto ordered_statistics;
            total = _mm256_add_pd(total, product);
            slo = _mm256_min_pd(slo, _mm256_blendv_pd(s, ceiling, mask));
            zlo = _mm256_min_pd(zlo, _mm256_blendv_pd(z, ceiling, mask));
        }
        for (; i < count; ++i)
            if (!w->constant_orthant[i]) {
                double s = w->s[i], z = w->z[i], product = fabs(s * z);
                if (!isfinite(s) || !isfinite(z) || !isfinite(product))
                    goto ordered_statistics;
                if (s < sl)
                    sl = s;
                if (z < zl)
                    zl = z;
                tail += product;
            }
        _mm256_storeu_pd(lanes, slo);
        for (i = 0; i < 4; ++i)
            if (lanes[i] < sl)
                sl = lanes[i];
        _mm256_storeu_pd(lanes, zlo);
        for (i = 0; i < 4; ++i)
            if (lanes[i] < zl)
                zl = lanes[i];
        _mm256_storeu_pd(lanes, total);
        tail += ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]));
        if (!isfinite(tail))
            goto ordered_statistics;
        *smin = sl;
        *zmin = zl;
        *gap = tail;
        return;
    }
ordered_statistics:
#endif
    *smin = cone_min(w, w->s);
    *zmin = cone_min(w, w->z);
    *gap = fabs(cone_pair(w, w->s, w->z));
}

#if ICORE_OCP_SINGLETON_GV4
static double singleton_primal_residuals(icore_ocp_workspace *w)
{
    const __m256d sign = _mm256_set1_pd(-0.0), ceiling = _mm256_set1_pd(DBL_MAX);
    __m256d maximum = _mm256_setzero_pd();
    int valid = 1, k, i;
    double result = 0.0, lanes[4];
    for (k = 0; k < w->K; ++k) {
        const double *v = w->v + (size_t)k * w->nv;
        for (i = 0; w->m - i >= 4; i += 4) {
            size_t row = (size_t)k * w->m + i;
            __m256d gv = singleton_gv4(w, row, v);
            __m256d rp = _mm256_sub_pd(_mm256_add_pd(gv, _mm256_loadu_pd(w->s + row)),
                                       _mm256_loadu_pd(w->h + row));
            __m256d magnitude = _mm256_andnot_pd(sign, rp);
            __m256d ok = _mm256_cmp_pd(magnitude, ceiling, _CMP_LE_OQ);
            _mm256_storeu_pd(w->rp + row, rp);
            valid &= _mm256_movemask_pd(ok) == 15;
            maximum = _mm256_max_pd(maximum, _mm256_and_pd(ok, magnitude));
        }
        for (; i < w->m; ++i) {
            size_t row = (size_t)k * w->m + i;
            int column = w->single_index[row];
            double gv = column < 0 ? 0.0 : w->single_value[row] * v[column], magnitude;
            __asm__ __volatile__("" : "+x"(gv));
            w->rp[row] = (gv + w->s[row]) - w->h[row];
            magnitude = fabs(w->rp[row]);
            if (isfinite(magnitude)) {
                if (magnitude > result)
                    result = magnitude;
            } else
                valid = 0;
        }
    }
    _mm256_storeu_pd(lanes, maximum);
    for (i = 0; i < 4; ++i)
        if (lanes[i] > result)
            result = lanes[i];
    return valid ? result : INFINITY;
}

static double direct_cached_residual(icore_ocp_workspace *w, const double *z, double *ds)
{
    const size_t count = (size_t)w->K * w->m;
    const __m256d sign = _mm256_set1_pd(-0.0), ceiling = _mm256_set1_pd(DBL_MAX);
    __m256d maximum = _mm256_setzero_pd();
    size_t i = 0;
    int valid = 1;
    double result = 0.0, lanes[4];
    for (; count - i >= 4; i += 4) {
        __m256d gv = _mm256_loadu_pd(w->pcone + i);
        __m256d product = _mm256_mul_pd(_mm256_loadu_pd(z + i), _mm256_loadu_pd(w->boost + i));
        __m256d brc, magnitude, ok;
        /* Round the product before adding it to the complementarity sum. */
        __asm__ __volatile__("" : "+x"(product));
        brc = _mm256_add_pd(_mm256_sub_pd(_mm256_loadu_pd(w->bcone + i), gv), product);
        _mm256_storeu_pd(w->brc + i, brc);
        _mm256_storeu_pd(ds + i,
                         _mm256_sub_pd(_mm256_xor_pd(_mm256_loadu_pd(w->rp + i), sign), gv));
        magnitude = _mm256_andnot_pd(sign, brc);
        ok = _mm256_cmp_pd(magnitude, ceiling, _CMP_LE_OQ);
        valid &= _mm256_movemask_pd(ok) == 15;
        /* MAXPD itself raises invalid on a quiet NaN. Keep it masked. */
        maximum = _mm256_max_pd(maximum, _mm256_and_pd(ok, magnitude));
    }
    for (; i < count; ++i) {
        double product = z[i] * w->boost[i], magnitude;
        __asm__ __volatile__("" : "+x"(product));
        w->brc[i] = (w->bcone[i] - w->pcone[i]) + product;
        ds[i] = -w->rp[i] - w->pcone[i];
        magnitude = fabs(w->brc[i]);
        if (isfinite(magnitude)) {
            if (magnitude > result)
                result = magnitude;
        } else
            valid = 0;
    }
    _mm256_storeu_pd(lanes, maximum);
    for (i = 0; i < 4; ++i)
        if (lanes[i] > result)
            result = lanes[i];
    return valid ? result : INFINITY;
}
#endif

static void cone_shift(icore_ocp_workspace *w, double *a)
{
    double delta = fmax(0.0, 1.0 - cone_min(w, a));
    int k, j;
    for (k = 0; k < w->K; ++k) {
        double *ak = a + (size_t)k * w->m;
        for (j = 0; j < w->nlp; ++j) {
            if (w->constant_orthant[(size_t)k * w->m + j])
                ak[j] = 1.0;
            else
                ak[j] += delta;
        }
        for (j = 0; j < w->nsoc; ++j) {
            int o = w->offsets[j];
            if (constant_soc(w, k, j)) {
                memset(ak + o, 0, (size_t)w->sizes[j] * sizeof(double));
                ak[o] = 1.0;
            } else
                ak[o] += delta;
        }
    }
}

/* SOC boost: b[0] is its large eigenvalue, b[1:] its unit axis.
 * Its other eigenvalue is computed as 1/b[0], not by subtraction.
 * Thus no rounded identity w0*w0-|wt|^2==1 is assumed. */
ICORE_OCP_INLINE void boost_apply(const double *b, double eta, const double *v, double *out, int n,
                             int inverse)
{
    double p = dot(b + 1, v + 1, n - 1), large = b[0], small = 1.0 / large;
    double plus, minus, head, parallel, fac = inverse ? 1.0 / eta : eta;
    int i;
    if (inverse) {
        double t = large;
        large = small;
        small = t;
    }
    plus = large * (v[0] + p);
    minus = small * (v[0] - p);
    head = 0.5 * (plus + minus);
    parallel = 0.5 * (plus - minus);
    out[0] = fac * head;
    for (i = 1; i < n; ++i)
        out[i] = fac * (v[i] + b[i] * (parallel - p));
}

static void cone_apply(const icore_ocp_workspace *w, int k, const double *v, double *out, int inverse)
{
    const double *b = w->boost + (size_t)k * w->m;
    int j;
    if (inverse)
        for (j = 0; j < w->nlp; ++j)
            out[j] = v[j] * w->orthant_inv[(size_t)k * w->nlp + j];
    else
        for (j = 0; j < w->nlp; ++j)
            out[j] = v[j] * b[j];
    for (j = 0; j < w->nsoc; ++j) {
        int o = w->offsets[j];
        boost_apply(b + o, w->scale[(size_t)k * w->nsoc + j], v + o, out + o, w->sizes[j], inverse);
    }
}

static void scaling_identity(icore_ocp_workspace *w)
{
    int k, j;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
    w->orthant_direct = 0;
    w->orthant_roots_lazy = 0;
    memset(w->boost, 0, (size_t)w->K * w->m * sizeof(double));
    for (k = 0; k < w->K; ++k) {
        double *b = w->boost + (size_t)k * w->m;
        for (j = 0; j < w->nlp; ++j) {
            b[j] = 1.0;
            w->orthant_inv[(size_t)k * w->nlp + j] = 1.0;
        }
        for (j = 0; j < w->nsoc; ++j) {
            b[w->offsets[j]] = 1.0;
            w->scale[(size_t)k * w->nsoc + j] = 1.0;
        }
    }
}

static int scaling_update(icore_ocp_workspace *w)
{
    int k, j, i;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
    w->orthant_direct = 0;
    w->orthant_roots_lazy = 0;
    for (k = 0; k < w->K; ++k) {
        const double *s = w->s + (size_t)k * w->m, *z = w->z + (size_t)k * w->m;
        double *b = w->boost + (size_t)k * w->m, *lam = w->lambda + (size_t)k * w->m;
        if (!icore_ocp_orthant_scaling(s, z, w->constant_orthant + (size_t)k * w->m, b,
                                  w->orthant_inv + (size_t)k * w->nlp, lam, w->nlp))
            return 0;
        for (j = 0; j < w->nsoc; ++j) {
            int o = w->offsets[j], n = w->sizes[j];
            if (constant_soc(w, k, j)) {
                memset(b + o, 0, (size_t)n * sizeof(double));
                memset(lam + o, 0, (size_t)n * sizeof(double));
                b[o] = lam[o] = 1.0;
                w->scale[(size_t)k * w->nsoc + j] = 1.0;
                continue;
            }
            const double *sk = s + o, *zk = z + o;
            double rs = norm2(sk + 1, n - 1), rz = norm2(zk + 1, n - 1);
            double ds, dz, angular = 0.0, inner, gamma, radius, eta;
            if (!(sk[0] > rs && zk[0] > rz))
                return 0;
            ds = sqrt(sk[0] - rs) * sqrt(sk[0]) * sqrt(1.0 + rs / sk[0]);
            dz = sqrt(zk[0] - rz) * sqrt(zk[0]) * sqrt(1.0 + rz / zk[0]);
            if (!(ds > 0.0 && dz > 0.0) || !isfinite(ds + dz))
                return 0;
            if (rs > 0.0 && rz > 0.0) {
                double sp = (sk[0] / ds) + (rs / ds), zp = (zk[0] / dz) + (rz / dz);
                for (i = 1; i < n; ++i) {
                    double u = sk[i] / rs + zk[i] / rz;
                    angular += u * u;
                }
                inner = 0.5 * (sp / zp + zp / sp) + 0.5 * (rs / ds) * (rz / dz) * angular;
            } else
                inner = (sk[0] / ds) * (zk[0] / dz);
            gamma = sqrt(0.5 * (1.0 + inner));
            for (i = 1; i < n; ++i)
                b[o + i] = 0.5 * ((sk[i] / ds) - (zk[i] / dz)) / gamma;
            radius = norm2(b + o + 1, n - 1);
            /* Recover the algebraic unit-determinant head from its tail.
             * This also keeps a zero-axis boost exactly equal to identity. */
            b[o] = hypot(1.0, radius) + radius;
            if (!(b[o] >= 1.0) || !isfinite(b[o]))
                return 0;
            if (radius > 0.0)
                for (i = 1; i < n; ++i)
                    b[o + i] /= radius;
            eta = sqrt(ds) / sqrt(dz);
            w->scale[(size_t)k * w->nsoc + j] = eta;
            boost_apply(b + o, eta, sk, w->tmp, n, 1);
            boost_apply(b + o, eta, zk, w->tmp2, n, 0);
            for (i = 0; i < n; ++i)
                lam[o + i] = 0.5 * (w->tmp[i] + w->tmp2[i]);
            if (!(lam[o] > norm2(lam + o + 1, n - 1)))
                return 0;
        }
    }
    return 1;
}

/* For an orthant-only Newton system, eliminate complementarity directly:
 * Gamma=Z/S, Delta=S/Z, dz=Gamma*G*dv-Gamma*bc.  The same Gram/QR
 * factor kernels consume sqrt(Gamma)*G.  SOC and epigraph eliminations retain
 * NT scaling. lambda/boost hold Gamma/Delta only while orthant_direct is set.
 * A nonrepresentable direct ratio leaves the full range-safe NT path available;
 * no regularization or pivot guard is altered. */
static int orthant_direct_scaling(icore_ocp_workspace *w)
{
    size_t i = 0, count = (size_t)w->K * w->m;
    int lazy = w->ocp_diagonal_cost_ready;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
    w->orthant_direct = 0;
    w->orthant_roots_lazy = 0;
    w->orthant_gv_direction = NULL;
    if (w->nsoc || w->ne || !count)
        return 0;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    {
        const __m256d one = _mm256_set1_pd(1.0), zero = _mm256_setzero_pd();
        const __m256d floor = _mm256_set1_pd(DBL_MIN), ceiling = _mm256_set1_pd(DBL_MAX);
        for (; count - i >= 4; i += 4) {
            __m256d si = _mm256_loadu_pd(w->s + i), zi = _mm256_loadu_pd(w->z + i), gamma, delta;
            if (w->constant_orthant[i] || w->constant_orthant[i + 1] ||
                w->constant_orthant[i + 2] || w->constant_orthant[i + 3]) {
                __m256d mask =
                    _mm256_castsi256_pd(_mm256_set_epi64x(-(long long)!!w->constant_orthant[i + 3],
                                                          -(long long)!!w->constant_orthant[i + 2],
                                                          -(long long)!!w->constant_orthant[i + 1],
                                                          -(long long)!!w->constant_orthant[i]));
                si = _mm256_blendv_pd(si, one, mask);
                zi = _mm256_blendv_pd(zi, one, mask);
            }
            if (_mm256_movemask_pd(_mm256_and_pd(_mm256_cmp_pd(si, zero, _CMP_GT_OQ),
                                                 _mm256_cmp_pd(zi, zero, _CMP_GT_OQ))) != 15)
                break;
            gamma = _mm256_div_pd(zi, si);
            delta = _mm256_div_pd(si, zi);
            if (_mm256_movemask_pd(
                    _mm256_and_pd(_mm256_and_pd(_mm256_cmp_pd(gamma, floor, _CMP_GE_OQ),
                                                _mm256_cmp_pd(gamma, ceiling, _CMP_LE_OQ)),
                                  _mm256_and_pd(_mm256_cmp_pd(delta, floor, _CMP_GE_OQ),
                                                _mm256_cmp_pd(delta, ceiling, _CMP_LE_OQ)))) != 15)
                break;
            _mm256_storeu_pd(w->lambda + i, gamma);
            _mm256_storeu_pd(w->boost + i, delta);
            if (!lazy)
                _mm256_storeu_pd(w->orthant_inv + i, _mm256_sqrt_pd(gamma));
        }
    }
#endif
    for (; i < count; ++i) {
        double gamma, delta;
        if (w->constant_orthant[i]) {
            w->lambda[i] = w->boost[i] = 1.0;
            if (!lazy)
                w->orthant_inv[i] = 1.0;
            continue;
        }
        if (!(w->s[i] > 0.0 && w->z[i] > 0.0))
            return 0;
        gamma = w->z[i] / w->s[i];
        delta = w->s[i] / w->z[i];
        if (!(gamma >= DBL_MIN && delta >= DBL_MIN) || !isfinite(gamma) || !isfinite(delta))
            return 0;
        w->lambda[i] = gamma;
        w->boost[i] = delta;
        if (!lazy)
            w->orthant_inv[i] = sqrt(gamma);
    }
    w->orthant_direct = 1;
    w->orthant_roots_lazy = lazy;
    return 1;
}

static double soc_step(const double *x, const double *d, int n)
{
    double scale = fmax(norm_inf(x, (size_t)n), norm_inf(d, (size_t)n));
    double xt, dt, rx, rd, a, b, c, disc, q, r1, r2, alpha = 1.0, trial;
    double cross = 0.0, nd = 0.0, nx = 0.0, trial_square = 0.0, cross_absolute = 0.0;
    int i;
    if (!(scale > 0.0) || !isfinite(scale))
        return 0.0;
    xt = x[0] / scale;
    dt = d[0] / scale;
    for (i = 1; i < n; ++i) {
        double u = x[i] / scale, v = d[i] / scale, uv = u * v, t = u + v;
        nx += u * u;
        nd += v * v;
        cross += uv;
        cross_absolute += fabs(uv);
        trial_square += t * t;
    }
    trial = xt + dt;
    if (trial > 0.0 && trial > sqrt(trial_square))
        return 1.0;
    rx = sqrt(nx);
    rd = sqrt(nd);
    c = (xt - rx) * (xt + rx);
    a = (dt - rd) * (dt + rd);
    b = 2.0 * (xt * dt - cross);
    /* Widen only a cancellation-dominated Lorentz inner product. */
    if (fabs(b) < 64.0 * DBL_EPSILON * (fabs(xt * dt) + cross_absolute)) {
        long double precise = (long double)xt * dt;
        for (i = 1; i < n; ++i)
            precise -= (long double)(x[i] / scale) * (d[i] / scale);
        b = 2.0 * (double)precise;
    }
    if (dt < 0.0)
        alpha = fmin(alpha, -xt / dt);
    disc = b * b - 4.0 * a * c;
    if (disc >= 0.0 && c > 0.0) {
        q = -0.5 * (b + copysign(sqrt(disc), b));
        r1 = (a != 0.0) ? q / a : -1.0;
        r2 = (q != 0.0) ? c / q : -1.0;
        if (r1 > 0.0)
            alpha = fmin(alpha, r1);
        if (r2 > 0.0)
            alpha = fmin(alpha, r2);
        /* The boundary may occur exactly at the requested unit cap. */
        if (alpha > 0.0 && alpha <= 1.0)
            return alpha;
    }
    /* Rare roundoff fallback: convexity makes a bracketed search unambiguous. */
    {
        double lo = 0.0, hi = alpha;
        for (i = 0; i < 55; ++i) {
            double mid = 0.5 * (lo + hi), rad = 0.0;
            int j;
            for (j = 1; j < n; ++j)
                rad = hypot(rad, x[j] + mid * d[j]);
            if (x[0] + mid * d[0] > rad)
                lo = mid;
            else
                hi = mid;
        }
        return lo;
    }
}

static double cone_step(const icore_ocp_workspace *w, const double *x, const double *d)
{
    double alpha = 1.0;
    int k, j;
    if (!w->nsoc && (size_t)w->K * w->m <= INT_MAX)
        return icore_ocp_orthant_step(x, d, NULL, w->K * w->m);
    for (k = 0; k < w->K; ++k) {
        const double *xk = x + (size_t)k * w->m, *dk = d + (size_t)k * w->m;
        alpha = fmin(alpha, icore_ocp_orthant_step(xk, dk, NULL, w->nlp));
        for (j = 0; j < w->nsoc; ++j) {
            int o = w->offsets[j];
            if (!constant_soc(w, k, j))
                alpha = fmin(alpha, soc_step(xk + o, dk + o, w->sizes[j]));
        }
    }
    return alpha;
}
