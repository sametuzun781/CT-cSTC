/* Chain model validation, ownership, and structural preparation.
 * Private implementation included by chain.c; not a separate translation unit.
 */

/* ---- Validated creation, coefficient updates, and sparsity certificates. ---- */

static int size_product(size_t a, size_t b, size_t *result)
{
    if (a && b > SIZE_MAX / a)
        return 0;
    *result = a * b;
    return 1;
}

static int size_accumulate(size_t *sum, size_t count, size_t width)
{
    size_t value;
    if (!size_product(count, width, &value) || value > SIZE_MAX - *sum)
        return 0;
    *sum += value;
    return 1;
}

chain_workspace *chain_create(int K, int n, int d, int p, int l)
{
    chain_workspace *w;
    size_t total = 0, nn, qrsize, indices, bytes;
    double *next;
    if (K < 1 || n < 1 || d < 0 || p < 0 || l < 0 || (double)n * n > INT_MAX ||
        (double)K * n > INT_MAX / 16. || (double)K * (2. * d + p + l) > INT_MAX / 16.)
        return NULL;
    w = (chain_workspace *)calloc(1, sizeof(*w));
    if (!w)
        return NULL;
    w->K = K;
    w->n = n;
    w->d = d;
    w->p = p;
    w->l = l;
    w->nv = K * n;
    w->nd = (K - 1) * d;
    w->np = (K - 1) * p;
    w->ng = K * l;
    w->m = 2 * w->nd + w->np + w->ng;
    w->nr = w->nd + w->np + w->ng;
    w->dim = w->nv + w->nd + 2 * w->m;
    if (!size_product((size_t)K * n, n, &nn) ||
        !size_product((size_t)2 * n + l + d + p + 1, 2 * n, &qrsize) || qrsize > INT_MAX ||
        !size_product(w->nr, 2 * n, &indices) || !size_product(indices, sizeof(int), &bytes) ||
        !size_accumulate(&total, w->nr, 2 * n + 3) || !size_accumulate(&total, w->nv, 7) ||
        !size_accumulate(&total, w->nd, 4) || !size_accumulate(&total, nn, 2) ||
        !size_accumulate(&total, qrsize, 1) || !size_accumulate(&total, 2 * n, 1) ||
        !size_accumulate(&total, w->dim, 8) || !size_accumulate(&total, w->m, 1) ||
        !size_product(total, sizeof(double), &bytes) || !size_product(w->nr, sizeof(row), &bytes)) {
        chain_destroy(w);
        return NULL;
    }
    if (!size_product((size_t)2 * w->nv + w->K, sizeof(int), &bytes)) {
        chain_destroy(w);
        return NULL;
    }
    w->active = (int *)calloc(1, bytes);
    if (w->active) {
        w->position = w->active + w->nv;
        w->active_count = w->position + w->nv;
    }
    w->arena = (double *)calloc(total ? total : 1, sizeof(double));
    w->rows = (row *)calloc(w->nr ? w->nr : 1, sizeof(row));
    w->columns = (int *)calloc(indices ? indices : 1, sizeof(int));
#if !defined(CHAIN_DENSE_PATTERN)
    if (n <= 64) {
        w->patterns = (uint64_t *)calloc((size_t)2 * w->nv, sizeof(uint64_t));
        w->sparse_stage = (unsigned char *)calloc(K, sizeof(unsigned char));
    }
    if (n <= 64 && (!w->patterns || !w->sparse_stage)) {
        chain_destroy(w);
        return NULL;
    }
#endif
    if (!w->arena || !w->rows || !w->columns || !w->active) {
        chain_destroy(w);
        return NULL;
    }
    next = w->arena;
#define TAKE(field, count)                                                                         \
    w->field = next;                                                                               \
    next += (size_t)(count)
    TAKE(values, (size_t)w->nr * 2 * n);
    TAKE(offset, w->nr);
    TAKE(original_offset, w->nr);
    TAKE(curvature, w->nr);
    TAKE(h, w->nv);
    TAKE(c, w->nv);
    TAKE(fixed, w->nv);
    TAKE(free, w->nv);
    TAKE(center, w->nv);
    TAKE(original_h, w->nv);
    TAKE(original_c, w->nv);
    TAKE(lambda, w->nd);
    TAKE(inverse, w->nd);
    TAKE(ratio, w->nd);
    TAKE(force, w->nd);
    TAKE(D, nn);
    TAKE(link, nn);
    TAKE(qr, qrsize);
    TAKE(work, 2 * n);
    TAKE(state, w->dim);
    TAKE(target, w->dim);
    TAKE(affine, w->dim);
    TAKE(step, w->dim);
    TAKE(error, w->dim);
    TAKE(correction, w->dim);
    TAKE(trial, w->dim);
    TAKE(trial_error, w->dim);
    TAKE(weight, w->m);
#undef TAKE
    /* rhs and q alias unused initial-direction storage during a solve. */
    w->rhs = w->trial_error;
    w->q = w->trial_error + w->nv;
    return w;
}

void chain_destroy(chain_workspace *w)
{
    if (w) {
        chain_saddle_destroy(w->saddle);
        free(w->active);
        free(w->arena);
        free(w->rows);
        free(w->columns);
        free(w->patterns);
        free(w->sparse_stage);
        free(w);
    }
}

/* Symbolic fill uses unions, so numerical cancellation never removes an entry. */
static void prepare_patterns(chain_workspace *w)
{
    int i, j, r, k, n = w->n;
    uint64_t *D = w->patterns, *U;
    if (!D)
        return;
    U = D + w->nv;
    memset(D, 0, (size_t)2 * w->nv * sizeof(uint64_t));
    for (k = 0; k < w->K; ++k)
        for (i = 0; i < n; ++i)
            D[k * n + i] = UINT64_C(1) << i;
    for (r = 0; r < w->nr; ++r) {
        row a = w->rows[r];
        const int *col = w->columns + (size_t)r * 2 * n;
        uint64_t left = 0, right = 0;
        for (i = 0; i < a.left; ++i)
            left |= UINT64_C(1) << col[i];
        for (; i < a.count; ++i)
            right |= UINT64_C(1) << col[i];
        for (i = 0; i < a.left; ++i) {
            D[a.stage * n + col[i]] |= left;
            U[a.stage * n + col[i]] |= right;
        }
        for (; i < a.count; ++i)
            D[(a.stage + 1) * n + col[i]] |= right;
    }
    for (k = 0; k < w->K; ++k) {
        uint64_t *stage = D + k * n, *cross = U + k * n;
        for (j = 0; j < n; ++j) {
            uint64_t below = j == 63 ? 0 : stage[j] & ~((UINT64_C(1) << (j + 1)) - 1), bits = below;
            while (bits) {
                i = first_bit(bits);
                stage[i] |= below;
                bits &= bits - 1;
            }
        }
        {
            int entries = 0;
            for (i = 0; i < n; ++i) {
                uint64_t bits;
                stage[i] &= i == 63 ? UINT64_MAX : (UINT64_C(1) << (i + 1)) - 1;
                bits = stage[i];
                while (bits) {
                    ++entries;
                    bits &= bits - 1;
                }
            }
            /* Dense stages retain the contiguous dot kernel. */
            w->sparse_stage[k] = (unsigned char)(4 * entries <= 3 * n * (n + 1) / 2);
        }
        if (k == w->K - 1)
            break;
        for (i = 0; i < n; ++i) {
            uint64_t bits = stage[i] & ((UINT64_C(1) << i) - 1);
            while (bits) {
                j = first_bit(bits);
                cross[i] |= cross[j];
                bits &= bits - 1;
            }
        }
        for (i = 0; i < n; ++i) {
            uint64_t bits = cross[i];
            while (bits) {
                j = first_bit(bits);
                D[(k + 1) * n + j] |= cross[i];
                bits &= bits - 1;
            }
        }
    }
}

static int set_row(chain_workspace *w, int index, int stage, const double *left,
                   const double *right, double offset, int normalize)
{
    int side, j, n = w->n, count = 0;
    row *r = w->rows + index;
    int *col = w->columns + (size_t)index * 2 * n;
    double *v = w->values + (size_t)index * 2 * n;
    r->stage = stage;
    r->left = 0;
    for (side = 0; side < 2; ++side) {
        const double *a = side ? right : left;
        int base = (stage + side) * n;
        if (!a)
            break;
        for (j = 0; j < n; ++j) {
            if (!isfinite(a[j]))
                return 0;
            offset += a[j] * w->fixed[base + j];
            if (a[j] != 0. && w->free[base + j] != 0.) {
                col[count] = j;
                v[count++] = a[j];
            }
        }
        if (side == 0)
            r->left = count;
    }
    if (!isfinite(offset))
        return 0;
    r->count = count;
    r->constant = normalize && count == 0 && offset <= 0.;
    w->original_offset[index] = offset;
    w->offset[index] = r->constant ? -1. : offset;
    return 1;
}

/* Validate both raw arrays and the derived fixed-coordinate offsets before
 * touching the active model. This makes a rejected native update transactional
 * without allocating a second chain workspace. */
static int row_valid(const chain_workspace *w, const chain_problem *q, int stage,
                     const double *left, const double *right, double offset)
{
    int side, j, n = w->n;
    for (side = 0; side < 2; ++side) {
        const double *a = side ? right : left;
        int base = (stage + side) * n;
        if (!a)
            break;
        for (j = 0; j < n; ++j) {
            if (!isfinite(a[j]))
                return 0;
            offset += a[j] * ((1. - q->free[base + j]) * q->fixed[base + j]);
        }
    }
    return isfinite(offset);
}

static int chain_coefficients_valid(const chain_workspace *w, const chain_problem *q)
{
    int i, k, j, n = w->n;
    if (!q->center || !q->hessian || !q->linear || !q->free || !q->fixed ||
        (w->nd && (!q->left || !q->right || !q->offset || !q->l1_weight)) ||
        (w->np && (!q->pair_left || !q->pair_right || !q->pair_offset)) ||
        (w->ng && (!q->local_matrix || !q->local_offset)))
        return 0;
    for (i = 0; i < w->nv; ++i) {
        double fixed, c;
        if (!(q->hessian[i] > 0.) || !isfinite(q->hessian[i]) || !isfinite(q->center[i]) ||
            !isfinite(q->linear[i]) || !isfinite(q->fixed[i]) ||
            (q->free[i] != 0. && q->free[i] != 1.))
            return 0;
        fixed = (1. - q->free[i]) * q->fixed[i];
        c = q->free[i] * (q->linear[i] + q->hessian[i] * (fixed - q->center[i]));
        if (!isfinite(c))
            return 0;
    }
    for (k = 0; k < w->K - 1; ++k) {
        for (j = 0; j < w->d; ++j) {
            i = k * w->d + j;
            if (!(q->l1_weight[i] > 0.) || !isfinite(q->l1_weight[i]) ||
                !row_valid(w, q, k, q->left + (size_t)i * n, q->right + (size_t)i * n,
                           q->offset[i]))
                return 0;
        }
        for (j = 0; j < w->p; ++j) {
            i = k * w->p + j;
            if (!row_valid(w, q, k, q->pair_left + (size_t)i * n, q->pair_right + (size_t)i * n,
                           q->pair_offset[i]))
                return 0;
        }
    }
    for (k = 0; k < w->K; ++k)
        for (j = 0; j < w->l; ++j) {
            i = k * w->l + j;
            if (!row_valid(w, q, k, q->local_matrix + (size_t)i * n, NULL, q->local_offset[i]))
                return 0;
        }
    return 1;
}

int chain_update(chain_workspace *w, const chain_problem *q)
{
    int i, k, j, n;
    if (!w || !q || !chain_coefficients_valid(w, q))
        return 3;
    w->valid = 0;
    n = w->n;
    if (!q->center || !q->hessian || !q->linear || !q->free || !q->fixed ||
        (w->nd && (!q->left || !q->right || !q->offset || !q->l1_weight)) ||
        (w->np && (!q->pair_left || !q->pair_right || !q->pair_offset)) ||
        (w->ng && (!q->local_matrix || !q->local_offset)))
        return 3;
    for (i = 0; i < w->nv; ++i) {
        if (!(q->hessian[i] > 0.) || !isfinite(q->hessian[i]) || !isfinite(q->center[i]) ||
            !isfinite(q->linear[i]) || !isfinite(q->fixed[i]) ||
            (q->free[i] != 0. && q->free[i] != 1.))
            return 3;
        w->free[i] = q->free[i];
        w->fixed[i] = (1. - q->free[i]) * q->fixed[i];
        w->h[i] = q->free[i] ? q->hessian[i] : 1.;
        w->c[i] = q->free[i] * (q->linear[i] + q->hessian[i] * (w->fixed[i] - q->center[i]));
        w->original_h[i] = q->hessian[i];
        w->original_c[i] = q->linear[i];
        w->center[i] = q->center[i];
    }
    for (k = 0; k < w->K - 1; ++k) {
        for (j = 0; j < w->d; ++j) {
            i = k * w->d + j;
            if (!(q->l1_weight[i] > 0.) || !isfinite(q->l1_weight[i]))
                return 3;
            w->lambda[i] = q->l1_weight[i];
            if (!set_row(w, i, k, q->left + (size_t)i * n, q->right + (size_t)i * n, q->offset[i],
                         0))
                return 3;
        }
        for (j = 0; j < w->p; ++j) {
            i = k * w->p + j;
            if (!set_row(w, w->nd + i, k, q->pair_left + (size_t)i * n,
                         q->pair_right + (size_t)i * n, q->pair_offset[i], 1))
                return 3;
        }
    }
    for (k = 0; k < w->K; ++k)
        for (j = 0; j < w->l; ++j) {
            i = k * w->l + j;
            if (!set_row(w, w->nd + w->np + i, k, q->local_matrix + (size_t)i * n, NULL,
                         q->local_offset[i], 1))
                return 3;
        }
    prepare_patterns(w);
    for (k = 0; k < w->K; ++k) {
        int count = 0;
        for (j = 0; j < n; ++j) {
            int index = k * n + j;
            w->position[index] = -1;
            if (w->free[index]) {
                w->position[index] = count;
                w->active[k * n + count++] = j;
            }
        }
        /* Gather kernels are useful for a substantial dimensional reduction.
         * For modest masks retain all computational coordinates and the
         * original dense/symbolic kernels, including independent fixed ones.
         * The physical free mask and all constraint rows are unchanged. */
        if (2 * count > n) {
            count = n;
            for (j = 0; j < n; ++j) {
                w->position[k * n + j] = j;
                w->active[k * n + j] = j;
            }
        }
        w->active_count[k] = count;
    }
    chain_saddle_invalidate(w->saddle);
    w->saddle_prepare_pending = 1;
    w->valid = 1;
    return 0;
}
