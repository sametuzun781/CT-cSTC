/* Dense-QP model certificate and immutable coefficient packs.
 * Private implementation included by dense_qp.h; not a separate translation unit.
 */

/* ---- Dense-QP eligibility and immutable coefficient packs. ---- */

static void dense_qp_plan(icore_ocp_workspace *w, int same)
{
    int i, j, r;
    int n = w->nu, nv = w->nv;
    int reuse_h = w->qp_eligible && (same & CACHE_SAME_H);
    int reuse_g = w->qp_eligible && (same & CACHE_SAME_G);
    int packed_h = w->qp_packed_ready && reuse_h;
    int packed_g = w->qp_packed_ready && reuse_g;
    size_t p;
    w->qp_eligible = 0;
    w->qp_diagonal_eligible = w->qp_diagonal_factor = 0;
    w->qp_reuse_forward = 0;
    w->qp_row_magnitude = 0.0;
    w->qp_active = 0;
    w->qp_bound_pairs = 0;
    w->qp_unit_singletons = 0;
    w->qp_constant = 0.0;
    w->qp_packed_ready = 0;
    w->qp_packed_factor = 0;
    if (!packed_g)
        w->qp_general_count = 0;
    if (w->N != 1 || n < 1 || w->nsoc || w->nf || w->ne)
        return;
    if (!reuse_h)
        for (p = (size_t)nv * nv; p < 2 * (size_t)nv * nv; ++p)
            if (w->H[p] != 0.0)
                return;
    for (i = 0; i < nv; ++i)
        if (w->c[nv + i] != 0.0)
            return;
    if (!reuse_g)
        for (p = (size_t)w->m * nv; p < 2 * (size_t)w->m * nv; ++p)
            if (w->G[p] != 0.0)
                return;
    for (i = 0; i < n; ++i) {
        if (!reuse_h)
            for (j = 0; j < n; ++j)
                w->qp_H[(size_t)i * n + j] = w->H[(size_t)i * nv + j];
        w->qp_c[i] = w->c[i] + dot(w->H + (size_t)i * nv + n, w->x0, w->nx);
    }
    for (i = 0; i < w->nx; ++i)
        w->qp_constant +=
            w->x0[i] * (w->c[n + i] + 0.5 * dot(w->H + (size_t)(n + i) * nv + n, w->x0, w->nx));
    for (r = 0; r < w->m; ++r) {
        const int *idx = w->indices + w->index_start[r];
        int count = w->qp_nnz[r];
        if (!reuse_g) {
            count = 0;
            while (count < w->nnz[r] && idx[count] < n)
                ++count;
            w->qp_nnz[r] = count;
            w->qp_box_index[r] = count == 1 ? idx[0] : -1;
            w->qp_box_value[r] = count == 1 ? w->G[(size_t)r * nv + idx[0]] : 0.0;
        }
        w->qp_h[r] = w->h[r] - dot(w->G + (size_t)r * nv + n, w->x0, w->nx);
        if (!w->constant_orthant[r]) {
            if (!count)
                return;
            w->qp_rows[w->qp_active++] = r;
        }
    }
    if (!finite_array(w->qp_c, (size_t)n) || !finite_array(w->qp_h, (size_t)w->m) ||
        !isfinite(w->qp_constant))
        return;
    w->qp_eligible = 1;
    w->qp_unit_singletons = 1;
    for (i = 0; i < w->qp_active; ++i) {
        r = w->qp_rows[i];
        if (w->qp_nnz[r] == 1 && fabs(w->qp_box_value[r]) != 1.0)
            w->qp_unit_singletons = 0;
    }
    /* Exact model structure, refreshed after every update; this is a
     * diagonal Newton factor inside the same IPM, not a direct optimizer. */
    /* structure_cache runs before coefficient_cache_valid is committed. */
    w->qp_diagonal_eligible = w->cost_root_diagonal[0];
    for (i = 0; i < w->qp_active && w->qp_diagonal_eligible; ++i)
        if (w->qp_nnz[w->qp_rows[i]] != 1)
            w->qp_diagonal_eligible = 0;
#if ICORE_OCP_BOUND_PAIRS
    /* Exact, complete coordinate-pair certificate; no sign/scale assumptions.
     * Rebuilt on every model update, including changes of row supports. */
    if ((size_t)w->qp_active == 2 * (size_t)n && w->qp_rows[w->qp_active - 1] == w->qp_active - 1) {
        int eligible = 1;
        for (r = 0; r < w->qp_active; ++r)
            if (w->qp_nnz[r] != 1 || w->qp_box_index[r] != r / 2) {
                eligible = 0;
                break;
            }
        w->qp_bound_pairs = eligible;
    }
#endif
    /* Unit-coordinate row magnitudes screen roundoff-sensitive arithmetic.
     * Retain the original contraction order there. This owned-coefficient
     * cache is rebuilt on every numeric update, including x0 and h changes. */
    for (i = 0; i < w->qp_active; ++i) {
        r = w->qp_rows[i];
        const int *idx = w->indices + w->index_start[r];
        const double *g = w->G + (size_t)r * nv;
        double magnitude = fabs(w->qp_h[r]);
        for (j = 0; j < w->qp_nnz[r]; ++j)
            magnitude += fabs(g[idx[j]]);
        for (j = 0; j < w->nx; ++j)
            magnitude += fabs(g[n + j] * w->x0[j]);
        if (!isfinite(magnitude))
            magnitude = INFINITY;
        if (magnitude > w->qp_row_magnitude)
            w->qp_row_magnitude = magnitude;
    }
    if (w->qp_la) {
        /* The symmetric packed product must represent the exact original H.
         * Tolerated asymmetric inputs retain the full row-major products. */
        if (!packed_h)
            for (i = 0; i < n; ++i)
                for (j = 0; j < i; ++j)
                    if (w->qp_H[(size_t)i * n + j] != w->qp_H[(size_t)j * n + i])
                        return;
        if (!packed_g)
            for (r = 0; r < w->m; ++r)
                w->qp_gindex[r] = -1;
        if (!packed_g)
            for (i = 0; i < w->qp_active; ++i) {
                r = w->qp_rows[i];
                if (w->qp_nnz[r] > 1) {
                    int g = w->qp_general_count++;
                    w->qp_general_rows[g] = r;
                    w->qp_gindex[r] = g;
                    memcpy(w->qp_G + (size_t)g * n, w->G + (size_t)r * nv,
                           (size_t)n * sizeof(double));
                }
            }
        /* same is derived from exact owned-coefficient comparisons after
         * full input validation. Only immutable model packs are retained;
         * qp_packed_factor is always cleared for the new cold solve. */
        if ((!packed_h && icore_la_load_h(w->qp_la, w->qp_H, (size_t)n) != ICORE_LA_OK) ||
            (!packed_g &&
             icore_la_load_g(w->qp_la, w->qp_general_count, w->qp_G, (size_t)n) != ICORE_LA_OK))
            return;
        w->qp_packed_ready = 1;
    }
}
