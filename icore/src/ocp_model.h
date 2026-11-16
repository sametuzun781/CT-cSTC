/* Model validation, exact structure caches, and workspace ownership.
 * A model update is validated in the inactive bank before a complete swap.
 * Failed updates leave the live model unchanged. Numeric equality certificates
 * belong to owned bytes and never depend on the caller reusing an address.
 * Private implementation: included exactly once by ocp.c. */

/* Model lifecycle: certify structure, validate/load owned coefficients,
 * then manage storage and commit updates. The live bank is never half-updated. */
/* ---- Exact structure certificates, supports, and cached model packs. ---- */

enum {
    CACHE_SAME_H = 1,
    CACHE_SAME_D = 2,
    CACHE_SAME_G = 4,
    CACHE_SAME_F = 8,
    CACHE_SAME_G_PATTERN = 16
};

static int same_zero_pattern(const double *a, const double *b, int n)
{
    int j = 0;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    const __m256d zero = _mm256_setzero_pd();
    for (; n - j >= 4; j += 4) {
        int am = _mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(a + j), zero, _CMP_EQ_OQ));
        int bm = _mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(b + j), zero, _CMP_EQ_OQ));
        if (am != bm)
            return 0;
    }
#endif
    for (; j < n; ++j)
        if ((a[j] == 0.0) != (b[j] == 0.0))
            return 0;
    return 1;
}

/* Only the previously identified singleton entry is excluded. A successful
 * integer proof means every other input byte is literal +0, including its
 * sign bit. The full original comparison remains available for all refusals. */
ICORE_OCP_INLINE int singleton_positive_zero_spans(const double *a, int n, int single)
{
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    __m256i bits = _mm256_setzero_si256();
    int i = 0, tile = single & ~3;
    for (; i < tile; i += 4)
        bits = _mm256_or_si256(bits, _mm256_loadu_si256((const __m256i *)(a + i)));
    if (n - i >= 4) {
        __m256i active = _mm256_cmpeq_epi64(_mm256_set1_epi64x(single),
                                            _mm256_set_epi64x(i + 3, i + 2, i + 1, i));
        bits = _mm256_or_si256(
            bits, _mm256_andnot_si256(active, _mm256_loadu_si256((const __m256i *)(a + i))));
        i += 4;
    }
    for (; n - i >= 4; i += 4)
        bits = _mm256_or_si256(bits, _mm256_loadu_si256((const __m256i *)(a + i)));
    if (!_mm256_testz_si256(bits, bits))
        return 0;
    for (; i < n; ++i)
        if (i != single) {
            uint64_t value;
            memcpy(&value, a + i, sizeof(value));
            if (value)
                return 0;
        }
    return 1;
#else
    int i;
    for (i = 0; i < n; ++i)
        if (i != single) {
            uint64_t value;
            memcpy(&value, a + i, sizeof(value));
            if (value)
                return 0;
        }
    return 1;
#endif
}

/* Gh is written only at the cached supports (the common support of all rows
 * in an SOC block). On a changed pattern, clear those old entries before
 * replacing the descriptors. A failed/uncommitted prior load has no valid
 * support certificate, so it always receives a complete clear. */
static void clear_scaled_supports(icore_ocp_workspace *w, int valid)
{
    size_t row, count = (size_t)w->K * w->m;
    if (!valid) {
        memset(w->Gh, 0, count * w->nv * sizeof(double));
        return;
    }
    for (row = 0; row < count; ++row) {
        double *g = w->Gh + row * w->nv;
        int i, n = w->nnz[row];
        const int *index = w->indices + w->index_start[row];
        if (n > w->nv / 2)
            memset(g, 0, (size_t)w->nv * sizeof(double));
        else
            for (i = 0; i < n; ++i)
                g[index[i]] = 0.0;
    }
}

/* Small stages benefit from packed kernels when scalar bounds add only
 * diagonal curvature. General rows keep the established size crossover.
 * Read the current exact support cache, so a changed constraint pattern
 * cannot retain a previous model's small-stage eligibility. */
static int packed_stage_size_eligible(const icore_ocp_workspace *w)
{
    if (w->nx >= 16 && w->nvr >= 32)
        return 1;
    /* A direct dense QP does not need a new stage-factor workspace. */
    if (w->qp_eligible)
        return 0;
    if (w->nx < 8 || w->nvr < 12 || w->nsoc || w->ne || w->nf)
        return 0;
    /* Below the normal packed crossover, require enough control columns
     * to amortize a full state/control square-root factor. */
    if (2 * (size_t)w->nr < (size_t)w->nx)
        return 0;
    /* Small workspaces pay off only when every stage can use packed dense
     * dynamics. Sparse stages retain their existing row-major cache. */
    for (int stage = 0; stage < w->N; ++stage)
        if (w->dyn_sparse[stage])
            return 0;
    for (size_t row = 0; row < (size_t)w->K * w->m; ++row)
        if (w->nnz[row] > 1)
            return 0;
    return 1;
}

static int structure_cache(icore_ocp_workspace *w, int same, const unsigned char *input_cost_diagonal)
{
    int k, i, j, a;
    int nv = w->nv, nx = w->nx;
    int exact_unchanged_h =
        (same & CACHE_SAME_H) && w->cost_validation_reusable == ICORE_OCP_COST_VALIDATION_REUSABLE;
    size_t p, index_cursor = 0;
    if (!(same & CACHE_SAME_D))
        for (k = 0; k < w->N; ++k) {
            w->dyn_sparse[k] = dynamics_sparse_structure(w->D + (size_t)k * nx * nv, nx, nv, w->ne);
            if (!w->dyn_sparse[k])
                continue;
            for (j = w->ne; j < nv; ++j) {
                size_t column = (size_t)k * nv + j;
                int count = 0;
                for (i = 0; i < nx; ++i) {
                    double d = w->D[(size_t)k * nx * nv + (size_t)i * nv + j];
                    if (d != 0.0) {
                        w->dyn_index[column * nx + count] = i;
                        w->dyn_value[column * nx + count++] = d;
                    }
                }
                w->dyn_count[column] = count;
            }
        }
    if (!(same & CACHE_SAME_G))
        for (p = 0; p < (size_t)w->K * w->m; ++p)
            w->epi_row[p] = -1;
    for (k = 0; k < w->K; ++k) {
        const double *G = w->G + (size_t)k * w->m * nv;
        const double *H = w->H + (size_t)k * nv * nv;
        /* Exact row bounds remove only leading/trailing zeros.  Interior
         * zeros remain in the contiguous vector dot, with no gather path. */
        if (!(same & CACHE_SAME_H)) {
            if (input_cost_diagonal && input_cost_diagonal[k]) {
                /* The incoming off-diagonal bytes were just proved zero.
                 * Reproduce the ordinary row descriptors without scanning
                 * those bytes again. This does not certify PSD or roots. */
                for (i = 0; i < nv; ++i) {
                    int nonzero = H[(size_t)i * nv + i] != 0.0;
                    w->cost_start[(size_t)k * nv + i] = nonzero ? i : nv;
                    w->cost_length[(size_t)k * nv + i] = nonzero ? 1 : 0;
                }
            } else
                for (i = 0; i < nv; ++i) {
                    const double *row = H + (size_t)i * nv;
                    int first = structural_first_nonzero(row, nv);
                    int last =
                        first == nv ? nv : first + structural_last_nonzero(row + first, nv - first);
                    if (last - first > nv / 2) {
                        first = 0;
                        last = nv;
                    }
                    w->cost_start[(size_t)k * nv + i] = first;
                    w->cost_length[(size_t)k * nv + i] = last - first;
                }
        }
        if (!(same & CACHE_SAME_G_PATTERN))
            for (i = 0; i < w->nlp; ++i) {
                size_t row = (size_t)k * w->m + i;
                int count;
                w->index_start[row] = index_cursor;
                count = structural_support(G + (size_t)i * nv, nv, w->indices + index_cursor);
                w->nnz[row] = count;
                index_cursor += (size_t)count;
            }
        if (!(same & CACHE_SAME_G_PATTERN))
            for (a = 0; a < w->nsoc; ++a) {
                int o = w->offsets[a], n = w->sizes[a], count = 0;
                for (j = 0; j < nv; ++j) {
                    int active = 0;
                    for (i = 0; i < n; ++i)
                        if (G[(size_t)(o + i) * nv + j] != 0.0)
                            active = 1;
                    if (active)
                        w->indices[index_cursor + count++] = j;
                }
                for (i = 0; i < n; ++i) {
                    size_t row = (size_t)k * w->m + o + i;
                    w->nnz[row] = count;
                    w->index_start[row] = index_cursor;
                }
                index_cursor += (size_t)count;
            }
        if (!(same & CACHE_SAME_G))
            for (i = 0; i < w->m; ++i) {
                size_t row = (size_t)k * w->m + i;
                int single = w->nnz[row] == 1 ? w->indices[w->index_start[row]] : -1;
                if (single < 0)
                    w->single_positive_zero[row] = ICORE_OCP_SINGLETON_UNKNOWN;
                else if (single != w->single_index[row] || !w->single_positive_zero[row])
                    w->single_positive_zero[row] =
                        (unsigned char)(singleton_positive_zero_spans(G + (size_t)i * nv, nv,
                                                                      single)
                                            ? ICORE_OCP_SINGLETON_POSITIVE_ZERO
                                            : ICORE_OCP_SINGLETON_COMPARE_BYTES);
                w->single_index[row] = single;
                w->single_value[row] = single >= 0 ? G[(size_t)i * nv + single] : 0.0;
                w->single_scaled[row] = 0.0;
            }
        if (!(same & CACHE_SAME_H)) {
            double *root = w->Hs + (size_t)k * nv * nv;
            int known_zero = w->cost_root_diagonal[k];
            /* Only root construction writes Hs. Clear its certificate before
             * any potentially partial write; install it only after success.
             * This storage property survives coefficient-bank rejection, but
             * never reuses an old diagonal value or a failed dense root. */
            w->cost_root_diagonal[k] = 0;
            if (stage_cost_is_diagonal(w, k)) {
                if (!(known_zero ? diagonal_cost_root_storage(H, root, nv, 1)
                                 : diagonal_cost_root(H, root, nv)))
                    return 0;
                w->cost_root_diagonal[k] = 1;
            } else if (!cost_root(H, root, nv))
                return 0;
        }
        if (k == w->N) {
            for (i = 0; i < w->nu; ++i) {
                if (w->c[(size_t)k * nv + i] != 0.0)
                    return 0;
                if (!exact_unchanged_h)
                    for (j = 0; j < nv; ++j)
                        if (w->H[(size_t)k * nv * nv + (size_t)i * nv + j] != 0.0 ||
                            w->H[(size_t)k * nv * nv + (size_t)j * nv + i] != 0.0)
                            return 0;
                /* A valid unchanged support pattern retains the exact zero
                 * control columns of the previously accepted terminal rows. */
                if (!(same & CACHE_SAME_G_PATTERN))
                    for (j = 0; j < w->m; ++j)
                        if (G[(size_t)j * nv + i] != 0.0)
                            return 0;
            }
            continue;
        }
        for (i = 0; i < w->ne; ++i) {
            int plus = -1, minus = -1, other = -1, count = 0;
            size_t epos = (size_t)k * w->ne + i;
            if (!exact_unchanged_h)
                for (j = 0; j < nv; ++j)
                    if (w->H[(size_t)k * nv * nv + (size_t)i * nv + j] != 0.0 ||
                        w->H[(size_t)k * nv * nv + (size_t)j * nv + i] != 0.0)
                        return 0;
            for (j = 0; j < nx; ++j)
                if (w->D[(size_t)k * nx * nv + (size_t)j * nv + i] != 0.0)
                    return 0;
            if (same & CACHE_SAME_G) {
                if (w->h[(size_t)k * w->m + w->epi_plus[epos]] != 0.0 ||
                    w->h[(size_t)k * w->m + w->epi_minus[epos]] != 0.0)
                    return 0;
                continue;
            }
            for (a = 0; a < w->m; ++a)
                if (G[(size_t)a * nv + i] != 0.0) {
                    int e = -1;
                    ++count;
                    if (a >= w->nlp || G[(size_t)a * nv + i] != -1.0 ||
                        w->h[(size_t)k * w->m + a] != 0.0)
                        return 0;
                    for (j = 0; j < nv; ++j)
                        if (j != i && G[(size_t)a * nv + j] != 0.0) {
                            if (e >= 0 || j < w->ne || j >= w->nu ||
                                fabs(G[(size_t)a * nv + j]) != 1.0)
                                return 0;
                            e = j;
                        }
                    if (e < 0 || (other >= 0 && e != other))
                        return 0;
                    other = e;
                    if (G[(size_t)a * nv + e] > 0.0) {
                        if (plus >= 0)
                            return 0;
                        plus = a;
                    } else {
                        if (minus >= 0)
                            return 0;
                        minus = a;
                    }
                }
            if (count != 2 || plus < 0 || minus < 0)
                return 0;
            for (a = 0; a < w->m; ++a)
                if (a != plus && a != minus && G[(size_t)a * nv + other] != 0.0)
                    return 0;
            for (a = 0; a < i; ++a)
                if (w->epi_e[(size_t)k * w->ne + a] == other)
                    return 0;
            w->epi_e[epos] = other;
            w->epi_plus[epos] = plus;
            w->epi_minus[epos] = minus;
            w->epi_row[(size_t)k * w->m + plus] = i;
            w->epi_row[(size_t)k * w->m + minus] = i;
        }
    }
    if (!(same & CACHE_SAME_G_PATTERN))
        for (k = 0; k < w->K; ++k) {
            int count = 0, active = w->nv - (k == w->N ? w->nu : w->ne);
            for (i = 0; i < w->m; ++i) {
                size_t row = (size_t)k * w->m + i;
                if (w->epi_row[row] < 0 && w->nnz[row] >= 4 && w->nnz[row] * 2 >= active)
                    w->dense_rows[(size_t)k * w->m + count++] = i;
            }
            w->dense_count[k] = count;
        }
    /* Group membership depends only on the owned exact support pattern.
     * Rebuild in the inactive model bank when that pattern changes. */
    if (!(same & CACHE_SAME_G_PATTERN))
        for (k = 0; k < w->K; ++k) {
            size_t base = (size_t)k * w->m;
            for (i = 0; i < w->m;) {
                int end = i + 1, count = w->nnz[base + i];
                const int *idx = w->indices + w->index_start[base + i];
                if (count >= 4 && idx[count - 1] - idx[0] + 1 == count) {
                    while (end < w->m) {
                        const int *next = w->indices + w->index_start[base + end];
                        if (w->nnz[base + end] != count || next[0] != idx[0] ||
                            next[count - 1] != idx[count - 1])
                            break;
                        ++end;
                    }
                }
                for (; i < end; ++i)
                    w->constraint_group_end[base + i] = end;
            }
        }
    if (!(same & CACHE_SAME_F))
        for (i = 0; i < w->nf; ++i) {
            double r = norm2(w->F + (size_t)i * nx, nx);
            if (!(r > 0.0))
                return 0;
            w->Fnorm[i] = r;
            for (j = 0; j < nx; ++j)
                w->Fn[(size_t)i * nx + j] = w->F[(size_t)i * nx + j] / r;
        }
    dense_qp_plan(w, same);
    int packed_size_eligible = packed_stage_size_eligible(w);
    int packed_created = 0;
    /* Allocate the smaller packed workspace only after the exact support
     * cache proves that this model can use it. A later update may establish
     * eligibility, so force its first dynamics load even when A/B match. */
    if (!w->ocp_la && w->nvr < 32 && w->nr && packed_size_eligible && icore_la_blasfeo_available())
        packed_created = icore_ocp_la_create_owned(w->N, w->nx, w->nr, &w->ocp_la) == ICORE_LA_OK;
    if (packed_created || !(same & CACHE_SAME_D) || (w->nvr < 32 && !(same & CACHE_SAME_G_PATTERN)))
        for (k = 0; k < w->N; ++k) {
            w->ocp_packed_ready[k] = 0;
            /* Canonical D contains only copies of A/B already fully validated
             * by coefficients_valid; no arithmetic can invalidate that proof. */
            if (packed_size_eligible && w->ocp_la && !w->dyn_sparse[k] &&
                icore_ocp_la_load_dynamics_finite(w->ocp_la, k, w->D + (size_t)k * nx * nv + w->ne,
                                         (size_t)nv) == ICORE_LA_OK)
                w->ocp_packed_ready[k] = 1;
        }
    if (w->ocp_la && !w->nsoc && !w->ne) {
        int eligible = 1;
        int cached_cost = (same & CACHE_SAME_H) && w->ocp_diagonal_cost_ready;
        w->ocp_diagonal_cost_ready = 0;
        for (k = 0; k < w->N; ++k)
            if (!w->ocp_packed_ready[k])
                eligible = 0;
        for (k = 0; k < w->K; ++k)
            for (i = 0; i < w->m; ++i)
                if (w->nnz[(size_t)k * w->m + i] > 1)
                    eligible = 0;
        /* The exact H match is valid only for a previously committed model
         * cache. A first eligible model or any changed H repacks all costs.
         * Dynamics' independent exact-reuse flags remain unchanged. */
        if (eligible && !cached_cost)
            for (k = 0; k < w->K; ++k) {
                int first = k == w->N ? w->nu : w->ne;
                const double *H = w->H + (size_t)k * nv * nv + (size_t)first * nv + first;
                int status;
                if (stage_cost_is_diagonal(w, k)) {
                    double *diagonal = w->ocp_diagonal + (size_t)k * nv;
                    for (i = 0; i < nv - first; ++i)
                        diagonal[i] = H[(size_t)i * nv + i];
                    status = icore_ocp_la_load_diagonal_cost(w->ocp_la, k, diagonal);
                } else {
                    /* Canonical H is unchanged after complete finite/symmetry/PSD
                     * validation. Transformed Hessians retain their own checks. */
                    status = icore_ocp_la_load_cost_finite(w->ocp_la, k, H, (size_t)nv);
                }
                if (status != ICORE_LA_OK) {
                    eligible = 0;
                    break;
                }
            }
        w->ocp_diagonal_cost_ready = eligible;
    } else
        w->ocp_diagonal_cost_ready = 0;
    return 1;
}

/* ---- Coefficient validation, equality certificates, and transactional loading. ---- */

/* A fresh binary64 zero certificate. Integer tests accept both signed zeros
 * and do not classify subnormal coefficients as zero under DAZ/FTZ modes. */

ICORE_OCP_INLINE int exact_zero_span(const double *a, int n)
{
    int i = 0;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    const __m256i magnitude = _mm256_set1_epi64x(INT64_MAX);
    for (; n - i >= 4; i += 4) {
        __m256i bits = _mm256_and_si256(_mm256_loadu_si256((const __m256i *)(a + i)), magnitude);
        if (!_mm256_testz_si256(bits, bits))
            return 0;
    }
#endif
    for (; i < n; ++i) {
        uint64_t bits;
        memcpy(&bits, a + i, sizeof(bits));
        if (bits & UINT64_C(0x7fffffffffffffff))
            return 0;
    }
    return 1;
}

/* Exact diagonal structure and finite diagonal entries jointly prove every
 * coefficient finite and symmetric. This is a fresh scan, not a cache token.
 * Any unproved entry takes the unchanged general validation below. The
 * separate cost-root/PSD acceptance policy is unaffected. */
static int diagonal_coefficients_finite(const double *H, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        const double *row = H + (size_t)i * n;
        if (!isfinite(row[i]) || !exact_zero_span(row, i) ||
            !exact_zero_span(row + i + 1, n - i - 1))
            return 0;
    }
    return 1;
}

/* Preserve the sign bits in the input certificate: 0 is unproved, 1 permits
 * either sign of off-diagonal zero, and 2 proves all those bytes are +0.
 * Integer loads neither flush subnormals nor raise floating-point flags. */
ICORE_OCP_INLINE int exact_zero_span_class(const double *a, int n)
{
    uint64_t tail = 0;
    int i = 0;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    const __m256i magnitude = _mm256_set1_epi64x(INT64_MAX);
    __m256i bits = _mm256_setzero_si256();
    /* Most dense rows differ in their first tile. Avoid reading the rest of
     * that row before taking the full finite/symmetry validation path. */
    if (n >= 4) {
        bits = _mm256_loadu_si256((const __m256i *)a);
        if (!_mm256_testz_si256(bits, magnitude))
            return 0;
        i = 4;
    }
    for (; n - i >= 4; i += 4)
        bits = _mm256_or_si256(bits, _mm256_loadu_si256((const __m256i *)(a + i)));
    if (!_mm256_testz_si256(bits, magnitude))
        return 0;
#endif
    for (; i < n; ++i) {
        uint64_t value;
        memcpy(&value, a + i, sizeof(value));
        tail |= value;
    }
    if (tail & UINT64_C(0x7fffffffffffffff))
        return 0;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    if (!_mm256_testz_si256(bits, bits))
        return 1;
#endif
    return tail ? 1 : 2;
}

static int diagonal_coefficients_class(const double *H, int n)
{
    int i, kind = 2;
    for (i = 0; i < n; ++i) {
        const double *row = H + (size_t)i * n;
        int before, after;
        if (!isfinite(row[i]))
            return 0;
        before = exact_zero_span_class(row, i);
        if (!before)
            return 0;
        after = exact_zero_span_class(row + i + 1, n - i - 1);
        if (!after)
            return 0;
        if (before == 1 || after == 1)
            kind = 1;
    }
    return kind;
}

static int coefficients_valid_structure_known_h(const icore_ocp_model *model,
                                                unsigned char *cost_diagonal, int known_h)
{
    size_t N = (size_t)model->N, K = N + 1, nx = (size_t)model->nx, nu = (size_t)model->nu;
    size_t nv = nx + nu, nf = (size_t)model->nf, m = (size_t)model->nlp;
    int i, k;
    for (i = 0; i < model->nsoc; ++i)
        m += (size_t)model->soc_sizes[i];
    if (!model->H)
        return 0;
    if (!known_h)
        for (k = 0; k <= model->N; ++k) {
            const double *H = model->H + (size_t)k * nv * nv;
            int diagonal = cost_diagonal ? diagonal_coefficients_class(H, (int)nv)
                                         : diagonal_coefficients_finite(H, (int)nv);
            if (cost_diagonal)
                cost_diagonal[k] = (unsigned char)diagonal;
            if (!diagonal && (!finite_array(H, nv * nv) || !symmetric_coefficients(H, (int)nv)))
                return 0;
        }
    if (!finite_array(model->c, K * nv) || !finite_array(model->A, N * nx * nx) ||
        !finite_array(model->B, N * nx * nu) || !finite_array(model->b, N * nx) ||
        (m && !model->G) || !finite_array(model->h, K * m) || !finite_array(model->x0, nx) ||
        !finite_array(model->F, nf * nx) || !finite_array(model->f, nf))
        return 0;
    return 1;
}

static int coefficients_valid_structure(const icore_ocp_model *model, unsigned char *cost_diagonal)
{
    return coefficients_valid_structure_known_h(model, cost_diagonal, 0);
}

static int coefficients_valid(const icore_ocp_model *model)
{
    return coefficients_valid_structure(model, NULL);
}

/* A byte-equal owned H can provide its previous finite/symmetry proof only
 * when validation performs no environment-dependent arithmetic. Certify the
 * stronger property once: every entry is normal or signed zero, and opposite
 * off-diagonal entries are exactly equal (either sign of zero is allowed).
 * Integer loads leave rounding modes and floating-point flags untouched.
 * Asymmetric or subnormal data always retain the original validation. */
static int cost_validation_certificate(const double *H, int K, int nv)
{
    size_t p, count = (size_t)K * nv * nv;
    int k, i, j;
    for (p = 0; p < count; ++p) {
        uint64_t bits, absolute, exponent;
        memcpy(&bits, H + p, sizeof(bits));
        absolute = bits & UINT64_C(0x7fffffffffffffff);
        exponent = absolute & UINT64_C(0x7ff0000000000000);
        if (exponent == UINT64_C(0x7ff0000000000000) || (!exponent && absolute))
            return 0;
    }
    for (k = 0; k < K; ++k) {
        const double *stage = H + (size_t)k * nv * nv;
        for (i = 1; i < nv; ++i)
            for (j = 0; j < i; ++j) {
                uint64_t a, b;
                memcpy(&a, stage + (size_t)i * nv + j, sizeof(a));
                memcpy(&b, stage + (size_t)j * nv + i, sizeof(b));
                if (a != b && ((a | b) & UINT64_C(0x7fffffffffffffff)))
                    return 0;
            }
    }
    return 1;
}

static int cost_validation_matches(icore_ocp_workspace *w, const icore_ocp_model *model)
{
    size_t count = (size_t)w->K * w->nv * w->nv;
    if (!w->coefficient_cache_valid || !model->H || memcmp(w->H, model->H, count * sizeof(double)))
        return 0;
    if (!w->cost_validation_reusable)
        w->cost_validation_reusable = cost_validation_certificate(w->H, w->K, w->nv)
                                          ? ICORE_OCP_COST_VALIDATION_REUSABLE
                                          : ICORE_OCP_COST_VALIDATION_FULL;
    return w->cost_validation_reusable == ICORE_OCP_COST_VALIDATION_REUSABLE;
}

/* Decouple finite scalar inequalities whose variable coefficients are exactly
 * zero.  At stage zero x0 is fixed, so only control columns can vary.  The
 * original signed slack is retained, and every negative slack contributes its
 * unavoidable defect to the original-coordinate primal residual.  Classification
 * is tolerance independent; every solve applies its current tolerance.  Output
 * has s=max(original_slack,0), z=0.  Internally a decoupled (s,z)=(1,1) row has
 * zero Newton RHS and contributes neither barrier degree nor gap. */
static void eliminate_constant_orthant(icore_ocp_workspace *w)
{
    int k, i, j, nv = w->nv;
    w->constant_count = 0;
    w->constant_primal_defect = 0.0;
    memset(w->constant_orthant, 0, (size_t)w->K * w->m * sizeof(unsigned char));
    for (k = 0; k < w->K; ++k)
        for (i = 0; i < w->nlp; ++i) {
            size_t row = (size_t)k * w->m + i;
            double *g = w->G + row * nv;
            double slack = w->h[row];
            int variable = k == 0 ? w->nu : nv, constant = 1;
            for (j = 0; j < variable; ++j)
                if (g[j] != 0.0) {
                    constant = 0;
                    break;
                }
            if (!constant)
                continue;
            if (k == 0)
                slack -= dot(g + w->nu, w->x0, w->nx);
            if (!isfinite(slack))
                continue;
            w->constant_orthant[row] = 1;
            w->constant_slack[row] = slack;
            w->constant_primal_defect = fmax(w->constant_primal_defect, -slack);
            ++w->constant_count;
            memset(g, 0, (size_t)nv * sizeof(double));
            w->h[row] = 1.0;
        }
}

/* Compare the owned combined dynamics with the incoming separate A/B rows.
 * Pointer identity is never a substitute for exact coefficient contents. */
static int dynamics_equal(const icore_ocp_workspace *w, const icore_ocp_model *model)
{
    int k, i;
    size_t nx = (size_t)w->nx, nu = (size_t)w->nu, nv = (size_t)w->nv;
    for (k = 0; k < w->N; ++k)
        for (i = 0; i < w->nx; ++i) {
            const double *row = w->D + ((size_t)k * nx + i) * nv;
            if ((nu && memcmp(row, model->B + ((size_t)k * nx + i) * nu, nu * sizeof(double))) ||
                memcmp(row + nu, model->A + ((size_t)k * nx + i) * nx, nx * sizeof(double)))
                return 0;
        }
    return 1;
}

/* Compare all bytes except the one previously certified singleton entry.
 * Integer lanes preserve signed-zero ownership and never classify NaNs.
 * Every vector load stays within the row, including its final partial tile. */
ICORE_OCP_INLINE int singleton_spans_equal(const double *a, const double *b, int n, int single)
{
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
    __m256i different = _mm256_setzero_si256();
    int i = 0, tile = single & ~3;
    for (; i < tile; i += 4)
        different = _mm256_or_si256(different,
                                    _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)),
                                                     _mm256_loadu_si256((const __m256i *)(b + i))));
    if (n - i >= 4) {
        __m256i active = _mm256_cmpeq_epi64(_mm256_set1_epi64x(single),
                                            _mm256_set_epi64x(i + 3, i + 2, i + 1, i));
        __m256i delta = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)),
                                         _mm256_loadu_si256((const __m256i *)(b + i)));
        different = _mm256_or_si256(different, _mm256_andnot_si256(active, delta));
        i += 4;
    }
    for (; n - i >= 4; i += 4)
        different = _mm256_or_si256(different,
                                    _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)),
                                                     _mm256_loadu_si256((const __m256i *)(b + i))));
    if (!_mm256_testz_si256(different, different))
        return 0;
    for (; i < n; ++i)
        if (i != single && memcmp(a + i, b + i, sizeof(double)))
            return 0;
    return 1;
#else
    return !memcmp(a, b, (size_t)single * sizeof(double)) &&
           !memcmp(a + single + 1, b + single + 1, (size_t)(n - single - 1) * sizeof(double));
#endif
}

/* A sufficient exact certificate: an entire SOC has h=0 and every nonzero
 * dependency is an explicit fixed zero at stage zero, or a terminal variable
 * proved zero by a singleton homogeneous equality. The empty dependency set
 * is valid at any stage. Integer zero tests retain subnormal dependencies. */
static int constant_soc_certificate(const icore_ocp_workspace *w, const icore_ocp_model *model, int k,
                                    int cone, const double *terminal_zero)
{
    int i, j, n = w->sizes[cone], o = w->offsets[cone], nv = w->nv;
    size_t row = (size_t)k * w->m + o;
    const double *G = model->G + row * nv;
    if (!exact_zero_span(model->h + row, n))
        return 0;
    for (i = 0; i < n; ++i) {
        const double *g = G + (size_t)i * nv;
        if (!exact_zero_span(g, w->nu))
            return 0;
        for (j = 0; j < w->nx; ++j)
            if (!exact_zero_span(g + w->nu + j, 1)) {
                int fixed_zero =
                    k == 0 ? exact_zero_span(w->x0 + j, 1) : k == w->N && terminal_zero[j] != 0.0;
                if (!fixed_zero)
                    return 0;
            }
    }
    /* A NaN on an otherwise fixed coordinate must never disappear. */
    return finite_array(G, (size_t)n * nv);
}

/* Classification is rebuilt from current coefficients on every inactive-bank
 * update. No caller pointers or previously accepted proof are retained. */
static int prepare_constant_soc(icore_ocp_workspace *w, const icore_ocp_model *model)
{
    int k, c, r, j;
    size_t count = 0, rows = 0, slots, values, cursor = 0;
    size_t *new_map = NULL;
    double *new_G = NULL;
    w->constant_soc_count = w->constant_soc_rows = 0;
    if (!w->nsoc)
        return 1;
    for (j = 0; j < w->nx; ++j)
        w->tmp[j] = 0.0;
    for (r = 0; r < w->nf; ++r) {
        const double *row = w->F + (size_t)r * w->nx;
        if (!exact_zero_span(w->f + r, 1))
            continue;
        for (j = 0; j < w->nx && exact_zero_span(row + j, 1); ++j)
            ;
        if (j < w->nx && exact_zero_span(row + j + 1, w->nx - j - 1))
            w->tmp[j] = 1.0;
    }
    for (k = 0; k < w->K; ++k)
        for (c = 0; c < w->nsoc; ++c)
            if (constant_soc_certificate(w, model, k, c, w->tmp)) {
                if (rows > SIZE_MAX - (size_t)w->sizes[c])
                    return 0;
                rows += (size_t)w->sizes[c];
                ++count;
            }
    if (!count)
        return 1;
    if ((size_t)w->K > SIZE_MAX / (size_t)w->nsoc || rows > SIZE_MAX / (size_t)w->nv)
        return 0;
    slots = (size_t)w->K * w->nsoc;
    values = rows * (size_t)w->nv;
    if (slots > w->constant_soc_map_capacity) {
        new_map = (size_t *)array_new(slots, sizeof(size_t));
        if (!new_map)
            return 0;
    }
    if (values > w->constant_soc_G_capacity) {
        new_G = (double *)array_new(values, sizeof(double));
        if (!new_G) {
            free(new_map);
            return 0;
        }
    }
    if (new_map) {
        free(w->constant_soc_offset);
        w->constant_soc_offset = new_map;
        w->constant_soc_map_capacity = slots;
    }
    if (new_G) {
        free(w->constant_soc_G);
        w->constant_soc_G = new_G;
        w->constant_soc_G_capacity = values;
    }
    for (size_t slot = 0; slot < slots; ++slot)
        w->constant_soc_offset[slot] = SIZE_MAX;
    for (k = 0; k < w->K; ++k)
        for (c = 0; c < w->nsoc; ++c)
            if (constant_soc_certificate(w, model, k, c, w->tmp)) {
                size_t source = ((size_t)k * w->m + w->offsets[c]) * w->nv;
                w->constant_soc_offset[(size_t)k * w->nsoc + c] = cursor;
                memcpy(w->constant_soc_G + cursor * w->nv, model->G + source,
                       (size_t)w->sizes[c] * w->nv * sizeof(double));
                cursor += (size_t)w->sizes[c];
            }
    w->constant_soc_count = count;
    w->constant_soc_rows = rows;
    return cursor == rows;
}

/* Load the same canonical scalar rows as eliminate_constant_orthant, while
 * comparing each nonconstant row before overwriting the owned G. Constant
 * classification and signed original slack are recomputed for every update,
 * including changed x0/h and finite-input substitution overflow. The retained
 * cache is for canonical G only; it never treats changed raw data as identical.
 * The complete incoming row is proved finite before any floating-point
 * classification. Exact equality with a committed finite row, or equality
 * of both singleton zero spans plus validation of its active entry, supplies
 * that proof without a second full matrix scan. All other rows are validated
 * in full before copying. A negative result rejects the inactive load.
 * No new coefficient shadow, hash collision, or caller-pointer lifetime is used. */
static int load_constraint_coefficients(icore_ocp_workspace *w, const icore_ocp_model *model, int cache_valid)
{
    int k, i, nv = w->nv, cone = 0;
    int same = cache_valid ? CACHE_SAME_G | CACHE_SAME_G_PATTERN : 0;
    if (!prepare_constant_soc(w, model))
        return -1;
    w->constant_count = 0;
    w->constant_primal_defect = 0.0;
    for (k = 0; k < w->K; ++k)
        for (i = 0; i < w->m; ++i) {
            size_t row = (size_t)k * w->m + i;
            const double *input = model->G + row * nv;
            double *g = w->G + row * nv;
            double slack = model->h[row];
            int constant = i < w->nlp;
            int inactive_soc = 0;
            if (i == 0)
                cone = 0;
            if (i >= w->nlp && w->constant_soc_count) {
                while (cone + 1 < w->nsoc && i >= w->offsets[cone + 1])
                    ++cone;
                inactive_soc = constant_soc(w, k, cone);
            }
            int variable = k == 0 ? w->nu : nv;
            int single = cache_valid ? w->single_index[row] : -1;
            int single_spans = 0, row_equal = 0;
            /* Byte comparisons cannot classify a NaN as a zero. Only a committed
             * canonical row can provide a finite-value certificate; an invalid
             * or partially loaded bank always takes the complete validation path.
             * This includes the state tail of a stage-zero constant inequality. */
            if (single >= 0 && ((w->single_positive_zero[row] == ICORE_OCP_SINGLETON_POSITIVE_ZERO &&
                                 singleton_positive_zero_spans(input, nv, single)) ||
                                singleton_spans_equal(g, input, nv, single))) {
                if (!finite_array(input + single, 1))
                    return -1;
                single_spans = 1;
                row_equal = !memcmp(g + single, input + single, sizeof(double));
            } else if (cache_valid && !memcmp(g, input, (size_t)nv * sizeof(double))) {
                row_equal = 1;
            } else if (!finite_array(input, (size_t)nv))
                return -1;
            if (constant) {
                /* Reuse only the exact support proof that validated this row.
                 * Orthant supports are sorted; a byte-equal canonical row has no
                 * variable coefficient iff its first support lies past them.
                 * The singleton coefficient may itself have become signed zero. */
                if (single_spans)
                    constant = input[single] == 0.0 || single >= variable;
                else if (row_equal)
                    constant = !w->nnz[row] || w->indices[w->index_start[row]] >= variable;
                else
                    constant = structural_first_nonzero(input, variable) == variable;
                if (constant && k == 0)
                    slack -= dot(input + w->nu, w->x0, w->nx);
                if (!isfinite(slack))
                    constant = 0;
            }
            if (inactive_soc) {
                if (!cache_valid || !exact_zero_span(g, nv)) {
                    same &= ~(CACHE_SAME_G | CACHE_SAME_G_PATTERN);
                    w->single_positive_zero[row] = ICORE_OCP_SINGLETON_UNKNOWN;
                }
                memset(g, 0, (size_t)nv * sizeof(double));
                w->h[row] = i == w->offsets[cone] ? 1.0 : 0.0;
            } else if (constant) {
                if (!cache_valid || !w->constant_orthant[row]) {
                    if (!cache_valid || w->nnz[row])
                        same &= ~CACHE_SAME_G_PATTERN;
                    w->single_positive_zero[row] = ICORE_OCP_SINGLETON_UNKNOWN;
                    memset(g, 0, (size_t)nv * sizeof(double));
                    same &= ~CACHE_SAME_G;
                }
                w->constant_slack[row] = slack;
                w->constant_primal_defect = fmax(w->constant_primal_defect, -slack);
                ++w->constant_count;
                w->h[row] = 1.0;
            } else {
                /* A singleton row often changes only its one coefficient. Exact
                 * byte comparisons of both zero spans preserve signed-zero
                 * ownership as well as detecting every new support. The earlier
                 * validation already performed those comparisons. */
                if (single_spans && input[single] != 0.0) {
                    if (g[single] != input[single]) {
                        g[single] = input[single];
                        same &= ~CACHE_SAME_G;
                    }
                } else if (!row_equal) {
                    if ((same & CACHE_SAME_G_PATTERN) && !same_zero_pattern(g, input, nv))
                        same &= ~CACHE_SAME_G_PATTERN;
                    w->single_positive_zero[row] = ICORE_OCP_SINGLETON_UNKNOWN;
                    memcpy(g, input, (size_t)nv * sizeof(double));
                    same &= ~CACHE_SAME_G;
                }
                w->h[row] = model->h[row];
            }
            w->constant_orthant[row] = (unsigned char)constant;
        }
    return same;
}

/* Only this loader writes canonical H. A committed bank can therefore retain
 * an exact +0 off-diagonal byte certificate. Combining it with the fresh input
 * certificate permits diagonal-only copies without losing signed-zero bytes.
 * The ordinary cache-valid token also guards these certificates: a rejected
 * partial load never makes them reusable. Root and packed reuse still depend
 * on the same all-stages, exact-byte CACHE_SAME_H condition as before. */
static int load_cost_coefficients(icore_ocp_workspace *w, const double *input, int valid,
                                  const unsigned char *diagonal)
{
    size_t nv = (size_t)w->nv, K = (size_t)w->K;
    size_t stage_count = nv * nv, stage_bytes = stage_count * sizeof(double);
    unsigned char *owned_positive_zero = w->cost_root_diagonal + 2 * K;
    int k, reuse = 0, same = valid;
    if (valid && diagonal)
        for (k = 0; k < w->K; ++k)
            if (diagonal[k] == 2 && owned_positive_zero[k]) {
                reuse = 1;
                break;
            }
    if (!reuse) {
        same = valid && !memcmp(w->H, input, K * stage_bytes);
        if (!same)
            w->cost_validation_reusable = ICORE_OCP_COST_VALIDATION_UNKNOWN;
        if (!same)
            memcpy(w->H, input, K * stage_bytes);
        for (k = 0; k < w->K; ++k)
            owned_positive_zero[k] = (unsigned char)(diagonal && diagonal[k] == 2);
        return same ? CACHE_SAME_H : 0;
    }
    for (k = 0; k < w->K; ++k) {
        double *target = w->H + (size_t)k * stage_count;
        const double *source = input + (size_t)k * stage_count;
        int positive_zero = diagonal[k] == 2;
        int sparse_copy = positive_zero && owned_positive_zero[k];
        owned_positive_zero[k] = 0;
        if (sparse_copy) {
            size_t i;
            for (i = 0; i < nv; ++i) {
                uint64_t before, after;
                size_t index = i * (nv + 1);
                memcpy(&before, target + index, sizeof(before));
                memcpy(&after, source + index, sizeof(after));
                if (before != after) {
                    memcpy(target + index, &after, sizeof(after));
                    same = 0;
                }
            }
        } else if (memcmp(target, source, stage_bytes)) {
            memcpy(target, source, stage_bytes);
            same = 0;
        }
        owned_positive_zero[k] = (unsigned char)positive_zero;
    }
    if (!same)
        w->cost_validation_reusable = ICORE_OCP_COST_VALIDATION_UNKNOWN;
    return same ? CACHE_SAME_H : 0;
}

/* Dimensions, cone layout, and finite/symmetric coefficients other than G
 * have already been checked; the constraint loader proves each G row finite
 * before classification or copying. Reuse only caches whose owned source
 * coefficients match. Clear
 * the validity token before any write so a rejected partial load cannot lend
 * stale cost roots, supports or packed dynamics to the next update. The cold
 * initializer is retained only when every owned coefficient matches exactly;
 * its factor can also survive changes confined to the right-hand side. */
static int load_coefficients_validated(icore_ocp_workspace *w, const icore_ocp_model *model,
                                       const unsigned char *input_cost_diagonal, int known_same_h)
{
    w->mixed_sweeps_active = 0;
    icore_ocp_la_mixed_sweeps_invalidate(w->ocp_la);
    w->mixed_factor_active = 0;
    size_t K = (size_t)w->K, N = (size_t)w->N, nx = (size_t)w->nx, nu = (size_t)w->nu;
    size_t nv = (size_t)w->nv, m = (size_t)w->m, nf = (size_t)w->nf;
    int k, i, j, same = 0, constraint_same, valid = w->coefficient_cache_valid;
    /* The cache contains the cold initializer, never the last solved iterate.
     * Preserve it only when every owned input coefficient matches exactly.
     * Eliminated zero SOCs have canonical h=(1,0,...) rather than their raw
     * zero h. Those bytes can match an incoming active constant cone even
     * though its classification, and therefore the cold cone shift, changed.
     * Recompute the initializer whenever this bank contains such a reduction.
     * Matrix equality is completed below, after successful validation. */
    int initial_valid = valid && !w->constant_soc_count && w->initial_valid &&
                        (!K || !nv || !memcmp(w->c, model->c, K * nv * sizeof(double))) &&
                        (!N || !nx || !memcmp(w->b, model->b, N * nx * sizeof(double))) &&
                        (!nx || !memcmp(w->x0, model->x0, nx * sizeof(double))) &&
                        (!K || !m || !memcmp(w->h, model->h, K * m * sizeof(double))) &&
                        (!nf || !memcmp(w->f, model->f, nf * sizeof(double)));
    int auxiliary_valid = w->aux_factor_valid;
    int dual_recovery_valid = w->dual_recovery_valid;
    w->dual_recovery_valid = 0;
    w->coefficient_cache_valid = 0;
    w->orthant_roots_lazy = 0;
    w->initial_valid = 0;
    w->aux_factor_valid = 0;
    w->sweep_pd_valid = 0;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
    if (valid) {
        if (dynamics_equal(w, model))
            same |= CACHE_SAME_D;
        if (!nf || !memcmp(w->F, model->F, nf * nx * sizeof(double)))
            same |= CACHE_SAME_F;
    }
    same |= known_same_h ? CACHE_SAME_H
                         : load_cost_coefficients(w, model->H, valid, input_cost_diagonal);
    memcpy(w->c, model->c, K * nv * sizeof(double));
    memcpy(w->b, model->b, N * nx * sizeof(double));
    memcpy(w->x0, model->x0, nx * sizeof(double));
    if (nf) {
        if (!(same & CACHE_SAME_F))
            memcpy(w->F, model->F, nf * nx * sizeof(double));
        memcpy(w->f, model->f, nf * sizeof(double));
    }
    if (!(same & CACHE_SAME_D))
        for (k = 0; k < w->N; ++k)
            for (i = 0; i < w->nx; ++i) {
                for (j = 0; j < w->nu; ++j)
                    w->D[(size_t)k * nx * nv + (size_t)i * nv + j] =
                        model->B[(size_t)k * nx * nu + (size_t)i * nu + j];
                for (j = 0; j < w->nx; ++j)
                    w->D[(size_t)k * nx * nv + (size_t)i * nv + nu + j] =
                        model->A[(size_t)k * nx * nx + (size_t)i * nx + j];
            }
    constraint_same = load_constraint_coefficients(w, model, valid);
    if (constraint_same < 0)
        return 0;
    same |= constraint_same;
    /* Scaling writes only structural entries. A changed G may leave old
     * nonzeros between the new supported entries used by dense Gram tiles. */
    if (!(same & CACHE_SAME_G_PATTERN))
        clear_scaled_supports(w, valid);
    if (!structure_cache(w, same, input_cost_diagonal))
        return 0;
    if (dual_recovery_valid &&
        (same & (CACHE_SAME_D | CACHE_SAME_F)) == (CACHE_SAME_D | CACHE_SAME_F))
        w->dual_recovery_valid = 1;
    else
        equality_dual_prepare(w);
    /* A structural update can introduce a packed or dense-QP workspace.
     * Restore the established live allocation layout and retire pointer-bank
     * reuse for this bank before any new solve can use those kernels. */
    if (w->aux_factor_swappable && (w->ocp_la || w->qp_eligible)) {
        if (w->factor_bank_exchanged)
            auxiliary_factor_exchange(w);
        w->aux_factor_live = 0;
        w->aux_factor_swappable = 0;
        auxiliary_valid = 0;
    }
    w->coefficient_cache_valid = 1;
    if ((same & (CACHE_SAME_H | CACHE_SAME_D | CACHE_SAME_G | CACHE_SAME_F)) ==
        (CACHE_SAME_H | CACHE_SAME_D | CACHE_SAME_G | CACHE_SAME_F)) {
        w->aux_factor_valid = auxiliary_valid;
        w->initial_valid = initial_valid;
    }
    return 1;
}

static int load_coefficients_with_diagonal(icore_ocp_workspace *w, const icore_ocp_model *model,
                                           const unsigned char *input_cost_diagonal)
{
    return load_coefficients_validated(w, model, input_cost_diagonal, 0);
}

static int load_coefficients(icore_ocp_workspace *w, const icore_ocp_model *model)
{
    return load_coefficients_with_diagonal(w, model, NULL);
}

/* ---- Workspace allocation, destruction, and committed model updates. ---- */

void icore_ocp_destroy(icore_ocp_workspace *w)
{
    if (!w)
        return;
    if (w->factor_bank_exchanged)
        auxiliary_factor_exchange(w);
    if (w->update_storage) {
        icore_ocp_workspace *spare = w->update_storage;
        w->update_storage = NULL;
        icore_ocp_destroy(spare);
    }
#define RELEASE(name) free(w->name)
    RELEASE(sizes);
    RELEASE(offsets);
    RELEASE(nnz);
    RELEASE(indices);
    RELEASE(index_start);
    RELEASE(epi_row);
    RELEASE(single_index);
    RELEASE(single_positive_zero);
    RELEASE(single_value);
    RELEASE(single_scaled);
    RELEASE(epi_e);
    RELEASE(epi_plus);
    RELEASE(epi_minus);
    RELEASE(dense_count);
    RELEASE(dense_rows);
    RELEASE(constraint_group_end);
    RELEASE(cost_start);
    RELEASE(cost_length);
    RELEASE(cost_root_diagonal);
    RELEASE(dyn_count);
    RELEASE(dyn_index);
    RELEASE(dyn_sparse);
    RELEASE(dyn_value);
    RELEASE(PDT);
    RELEASE(constant_orthant);
    RELEASE(constant_slack);
    RELEASE(constant_soc_offset);
    RELEASE(constant_soc_G);
    RELEASE(qp_rows);
    RELEASE(qp_nnz);
    RELEASE(qp_H);
    RELEASE(qp_c);
    RELEASE(qp_h);
    RELEASE(qp_box_index);
    RELEASE(qp_box_value);
    icore_la_destroy(w->qp_la);
    RELEASE(qp_general_rows);
    RELEASE(qp_gindex);
    RELEASE(qp_G);
    RELEASE(qp_diag);
    RELEASE(qp_weights);
    RELEASE(qp_gx);
    RELEASE(qp_gt);
    icore_ocp_la_destroy(w->ocp_la);
    icore_ocp_la_qr_destroy(w->ocp_qr_la);
    RELEASE(ocp_packed_ready);
    RELEASE(ocp_diagonal);
    RELEASE(H);
    RELEASE(c);
    RELEASE(D);
    RELEASE(b);
    RELEASE(G);
    RELEASE(h);
    RELEASE(x0);
    RELEASE(F);
    RELEASE(f);
    RELEASE(Fn);
    RELEASE(Fnorm);
    RELEASE(Hs);
    RELEASE(v);
    RELEASE(y);
    RELEASE(nu_t);
    RELEASE(s);
    RELEASE(z);
    RELEASE(rd);
    RELEASE(re);
    RELEASE(rf);
    RELEASE(rp);
    RELEASE(initial_v);
    RELEASE(initial_y);
    RELEASE(initial_nu);
    RELEASE(initial_s);
    RELEASE(initial_z);
    RELEASE(aux_factor_storage);
    RELEASE(boost);
    RELEASE(orthant_inv);
    RELEASE(scale);
    RELEASE(lambda);
    RELEASE(Gh);
    RELEASE(Hb);
    RELEASE(epi_inv);
    RELEASE(epi_ratio);
    RELEASE(epi_curv);
    RELEASE(L);
    RELEASE(T);
    RELEASE(P);
    RELEASE(R);
    RELEASE(C);
    RELEASE(Z);
    RELEASE(S);
    RELEASE(SL);
    RELEASE(sweep_a);
    RELEASE(sweep_p);
    RELEASE(sweep_pd);
    RELEASE(term_work);
    RELEASE(rhs);
    RELEASE(pcone);
    RELEASE(bcone);
    RELEASE(q);
    RELEASE(brd);
    RELEASE(bre);
    RELEASE(brf);
    RELEASE(brc);
    RELEASE(dv);
    RELEASE(dy);
    RELEASE(dnu);
    RELEASE(ds);
    RELEASE(dz);
    RELEASE(ads);
    RELEASE(adz);
    RELEASE(cv);
    RELEASE(cy);
    RELEASE(cnu);
    RELEASE(cz);
    RELEASE(tv);
    RELEASE(ty);
    RELEASE(tnu);
    RELEASE(tz);
    RELEASE(M);
    RELEASE(PD);
    RELEASE(qr);
    RELEASE(tmp);
    RELEASE(tmp2);
    RELEASE(tmp3);
    RELEASE(dual_recovery_storage);
#undef RELEASE
    free(w);
}

icore_ocp_workspace *icore_ocp_create(const icore_ocp_model *model)
{
    icore_ocp_workspace *w;
    size_t K, N, nx, nu, nv, nr, m, nf, ne, maxdim, qn, qm;
    int i, rows;
    if (!model || model->N < 1 || model->N == INT_MAX || model->nx < 1 || model->nu < 0 ||
        model->nlp < 0 || model->nsoc < 0 || model->nf < 0 || model->nf > model->nx ||
        model->ne < 0 || model->ne > model->nu / 2 || model->nx > INT_MAX - model->nu ||
        (model->nsoc && !model->soc_sizes))
        return NULL;
    rows = model->nlp;
    for (i = 0; i < model->nsoc; ++i) {
        if (model->soc_sizes[i] < 2 || rows > INT_MAX - model->soc_sizes[i])
            return NULL;
        rows += model->soc_sizes[i];
    }
    N = (size_t)model->N;
    K = N + 1;
    nx = (size_t)model->nx;
    nu = (size_t)model->nu;
    nv = nx + nu;
    nr = nu - model->ne;
    m = (size_t)rows;
    nf = (size_t)model->nf;
    ne = (size_t)model->ne;
    /* This also prevents overflow in every subsequently used array product. */
    if (nv > SIZE_MAX / nv || K > SIZE_MAX / (nv * nv) || (m && K > SIZE_MAX / m) ||
        (m && K * m > SIZE_MAX / nv) || K * nv * nv > SIZE_MAX / sizeof(double) ||
        (m && K * m * nv > SIZE_MAX / sizeof(double)) || K * nv > (size_t)INT_MAX)
        return NULL;
    w = (icore_ocp_workspace *)calloc(1, sizeof(*w));
    if (!w)
        return NULL;
    w->N = model->N;
    w->K = model->N + 1;
    w->nx = model->nx;
    w->nu = model->nu;
    w->nv = (int)nv;
    w->nr = (int)nr;
    w->nvr = (int)(nv - ne);
    w->m = rows;
    w->nlp = model->nlp;
    w->nsoc = model->nsoc;
    w->nf = model->nf;
    w->ne = model->ne;
    /* Keep the exact diagonal proof from initial validation, as updates do.
     * The loader can then construct row descriptors without rescanning H.
     * This does not replace the separate positive-semidefinite/root checks. */
    w->cost_root_diagonal = (unsigned char *)array_new(3 * K, sizeof(unsigned char));
    if (!w->cost_root_diagonal || !coefficients_valid_structure(model, w->cost_root_diagonal + K)) {
        icore_ocp_destroy(w);
        return NULL;
    }
    if (2 * nv + m + nx + ne > (size_t)INT_MAX) {
        icore_ocp_destroy(w);
        return NULL;
    }
    w->qr_rows = (int)(2 * nv + m + nx + ne);
    maxdim = nv;
    if (m > maxdim)
        maxdim = m;
    if (nf > maxdim)
        maxdim = nf;
    qn = (N == 1 && !model->nsoc && !nf && !ne) ? nu : 0;
    qm = qn ? m : 0;
#define ALLOCATE(name, count, type, allocator)                                                     \
    do {                                                                                           \
        w->name = (type *)allocator((count), sizeof(type));                                        \
        if (!w->name) {                                                                            \
            icore_ocp_destroy(w);                                                                       \
            return NULL;                                                                           \
        }                                                                                          \
    } while (0)
#define ALLOC(name, count, type) ALLOCATE(name, count, type, array_new)
#define OVERWRITTEN(name, count, type) ALLOCATE(name, count, type, array_overwritten)
    /* Coefficients, roots and scaled constraints are fully assigned by the
     * initial load. Sparse-index tails are never read beyond their counts. */
    ALLOC(sizes, (size_t)w->nsoc, int);
    ALLOC(offsets, (size_t)w->nsoc, int);
    ALLOC(nnz, K * m, int);
    OVERWRITTEN(indices, K * m * nv, int);
    ALLOC(index_start, K * m, size_t);
    ALLOC(epi_row, K * m, int);
    ALLOC(single_index, K * m, int);
    ALLOC(single_positive_zero, K * m, unsigned char);
    ALLOC(single_value, K * m, double);
    ALLOC(single_scaled, K * m, double);
    ALLOC(epi_e, N * ne, int);
    ALLOC(epi_plus, N * ne, int);
    ALLOC(epi_minus, N * ne, int);
    ALLOC(dense_count, K, int);
    ALLOC(dense_rows, K * m, int);
    ALLOC(cost_start, K * nv, int);
    ALLOC(cost_length, K * nv, int);
    /* cost_root_diagonal was allocated for validation above: first K bytes
     * describe Hs; next K are fresh input proofs; final K describe owned H. */
    ALLOC(dyn_count, N * nv, int);
    OVERWRITTEN(dyn_index, N * nv * nx, int);
    ALLOC(dyn_sparse, N, int);
    OVERWRITTEN(dyn_value, N * nv * nx, double);
    ALLOC(PDT, nv * nx, double);
    ALLOC(constant_orthant, K * m, unsigned char);
    ALLOC(constant_slack, K * m, double);
    ALLOC(qp_rows, qm, int);
    ALLOC(qp_nnz, qm, int);
    ALLOC(qp_box_index, qm, int);
    ALLOC(qp_box_value, qm, double);
    ALLOC(qp_H, qn * qn, double);
    ALLOC(qp_c, qn, double);
    ALLOC(qp_h, qm, double);
    ALLOC(qp_general_rows, qm, int);
    ALLOC(qp_gindex, qm, int);
    ALLOC(qp_G, qn * qm, double);
    ALLOC(qp_diag, qn, double);
    ALLOC(qp_weights, qm, double);
    ALLOC(qp_gx, qm, double);
    ALLOC(qp_gt, qn, double);
    ALLOC(ocp_packed_ready, N, unsigned char);
    ALLOC(ocp_diagonal, K * nv, double);
    OVERWRITTEN(H, K * nv * nv, double);
    OVERWRITTEN(c, K * nv, double);
    OVERWRITTEN(D, N * nx * nv, double);
    OVERWRITTEN(b, N * nx, double);
    OVERWRITTEN(G, K * m * nv, double);
    OVERWRITTEN(h, K * m, double);
    OVERWRITTEN(x0, nx, double);
    OVERWRITTEN(F, nf * nx, double);
    OVERWRITTEN(f, nf, double);
    OVERWRITTEN(Fn, nf * nx, double);
    OVERWRITTEN(Fnorm, nf, double);
    OVERWRITTEN(Hs, K * nv * nv, double);
    ALLOC(v, K * nv, double);
    ALLOC(y, N * nx, double);
    ALLOC(nu_t, nf, double);
    ALLOC(s, K * m, double);
    ALLOC(z, K * m, double);
    ALLOC(initial_v, K * nv, double);
    ALLOC(initial_y, N * nx, double);
    ALLOC(initial_nu, nf, double);
    ALLOC(initial_s, K * m, double);
    ALLOC(initial_z, K * m, double);
    ALLOC(rd, K * nv, double);
    ALLOC(re, N * nx, double);
    ALLOC(rf, nf, double);
    ALLOC(rp, K * m, double);
    ALLOC(boost, K * m, double);
    ALLOC(orthant_inv, K * (size_t)w->nlp, double);
    ALLOC(scale, K * (size_t)w->nsoc, double);
    ALLOC(lambda, K * m, double);
    OVERWRITTEN(Gh, K * m * nv, double);
    ALLOC(Hb, K * nv * nv, double);
    ALLOC(epi_inv, N * ne, double);
    ALLOC(epi_ratio, N * ne, double);
    ALLOC(epi_curv, N * ne, double);
    ALLOC(L, N * nr * nr, double);
    ALLOC(T, N * nx * nr, double);
    ALLOC(P, K * nx * nx, double);
    ALLOC(R, K * nx * nx, double);
    ALLOC(C, K * nf * nx, double);
    ALLOC(Z, N * nr * nf, double);
    ALLOC(S, nf * nf, double);
    ALLOC(SL, nf * nf, double);
    ALLOC(sweep_a, N * nr, double);
    ALLOC(sweep_p, K * nx, double);
    ALLOC(sweep_pd, N * nx, double);
    ALLOC(term_work, nf, double);
    ALLOC(rhs, K * nv, double);
    ALLOC(pcone, K * m, double);
    ALLOC(bcone, K * m, double);
    ALLOC(q, K * m, double);
    ALLOC(brd, K * nv, double);
    ALLOC(bre, N * nx, double);
    ALLOC(brf, nf, double);
    ALLOC(brc, K * m, double);
    ALLOC(dv, K * nv, double);
    ALLOC(dy, N * nx, double);
    ALLOC(dnu, nf, double);
    ALLOC(ds, K * m, double);
    ALLOC(dz, K * m, double);
    ALLOC(ads, K * m, double);
    ALLOC(adz, K * m, double);
    ALLOC(cv, K * nv, double);
    ALLOC(cy, N * nx, double);
    ALLOC(cnu, nf, double);
    ALLOC(cz, K * m, double);
    ALLOC(tv, K * nv, double);
    ALLOC(ty, N * nx, double);
    ALLOC(tnu, nf, double);
    ALLOC(tz, K * m, double);
    ALLOC(M, nv * nv, double);
    ALLOC(PD, nx * nv, double);
    ALLOC(qr, (size_t)w->qr_rows * nv, double);
    ALLOC(tmp, maxdim, double);
    ALLOC(tmp2, maxdim, double);
    ALLOC(tmp3, maxdim, double);
#undef OVERWRITTEN
#undef ALLOC
#undef ALLOCATE
    rows = w->nlp;
    for (i = 0; i < w->nsoc; ++i) {
        w->sizes[i] = model->soc_sizes[i];
        w->offsets[i] = rows;
        rows += w->sizes[i];
    }
    if (qn >= 12 && icore_la_blasfeo_available())
        (void)icore_la_create((int)qn, (int)qm, ICORE_LA_BLASFEO, &w->qp_la);
    if (nx >= 16 && nv - ne >= 32 && nr && icore_la_blasfeo_available())
        (void)icore_ocp_la_create_owned((int)N, (int)nx, (int)nr, &w->ocp_la);
    if (icore_la_blasfeo_available())
        (void)icore_ocp_la_qr_create(w->qr_rows, (int)nv, &w->ocp_qr_la);
    w->constraint_group_end = (int *)array_new(K * m, sizeof(int));
    if (!w->constraint_group_end) {
        icore_ocp_destroy(w);
        return NULL;
    }
    equality_dual_storage_create(w);
    if (!load_coefficients_with_diagonal(w, model, w->cost_root_diagonal + K)) {
        icore_ocp_destroy(w);
        return NULL;
    }
    auxiliary_factor_storage_create(w);
    return w;
}

int icore_ocp_update(icore_ocp_workspace *w, const icore_ocp_model *model)
{
    icore_ocp_workspace *replacement;
    icore_ocp_workspace old;
    int i;
    if (!w || !model || model->N != w->N || model->nx != w->nx || model->nu != w->nu ||
        model->nlp != w->nlp || model->nsoc != w->nsoc || model->nf != w->nf ||
        model->ne != w->ne || (w->nsoc && !model->soc_sizes))
        return ICORE_OCP_INVALID_MODEL;
    for (i = 0; i < w->nsoc; ++i)
        if (model->soc_sizes[i] != w->sizes[i])
            return ICORE_OCP_INVALID_MODEL;
    replacement = w->update_storage;
    if (replacement) {
        unsigned char *input_cost_diagonal = replacement->cost_root_diagonal + replacement->K;
        int known_same_h = cost_validation_matches(replacement, model);
        if (!coefficients_valid_structure_known_h(model, input_cost_diagonal, known_same_h) ||
            !load_coefficients_validated(replacement, model, input_cost_diagonal, known_same_h))
            return ICORE_OCP_INVALID_MODEL;
    } else {
        replacement = icore_ocp_create(model);
        if (!replacement)
            return ICORE_OCP_INVALID_MODEL;
    }
    /* Commit only after complete validation in inactive storage. Retaining the
     * previous workspace removes allocation from subsequent same-layout
     * updates. Peak storage is two workspaces, as in a create/swap/destroy
     * update; both now remain allocated until destruction or a layout change.
     * The inactive workspace never owns another inactive workspace. */
    w->update_storage = NULL;
    replacement->update_storage = NULL;
    old = *w;
    *w = *replacement;
    *replacement = old;
    w->update_storage = replacement;
    return ICORE_OCP_SOLVED;
}
