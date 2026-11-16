#if ICORE_OCP_MIXED_DIAGNOSTICS
/* Optional diagnostic counters; they do not change the iteration or options. */
static __thread uint64_t mixed_last_counts[8];
static __thread uint64_t mixed_last_sweep_counts[8];

int icore_mixed_sweeps_read(uint64_t *counts, size_t ncounts)
{
    if (!counts || ncounts != 8)
        return 0;
    memcpy(counts, mixed_last_sweep_counts, sizeof(mixed_last_sweep_counts));
    return 1;
}

static __thread double mixed_last_error, mixed_last_limit, mixed_max_accepted_error;
static __thread int mixed_failure_stage = -1, mixed_failure_phase = -1, mixed_failure_status = 0;
static __thread uint64_t mixed_policy_last_counts[6];
static __thread double mixed_policy_last_values[8];

double icore_mixed_max_accepted_error(void)
{
    return mixed_max_accepted_error;
}

int icore_mixed_last_factor(int *stage, int *phase, int *status)
{
    if (!stage || !phase || !status)
        return 0;
    *stage = mixed_failure_stage;
    *phase = mixed_failure_phase;
    *status = mixed_failure_status;
    return 1;
}

int icore_mixed_read(uint64_t *counts, size_t count, double *error, double *limit)
{
    if (!counts || count != 8 || !error || !limit)
        return 0;
    memcpy(counts, mixed_last_counts, sizeof(mixed_last_counts));
    *error = mixed_last_error;
    *limit = mixed_last_limit;
    return 1;
}

int icore_mixed_policy_read(uint64_t *counts, size_t ncounts, double *values, size_t nvalues)
{
    if (!counts || ncounts != 6 || !values || nvalues != 8)
        return 0;
    memcpy(counts, mixed_policy_last_counts, sizeof(mixed_policy_last_counts));
    memcpy(values, mixed_policy_last_values, sizeof(mixed_policy_last_values));
    return 1;
}
#else
double icore_mixed_max_accepted_error(void)
{
    return 0.0;
}

int icore_mixed_last_factor(int *stage, int *phase, int *status)
{
    (void)stage;
    (void)phase;
    (void)status;
    return 0;
}

int icore_mixed_read(uint64_t *counts, size_t count, double *error, double *limit)
{
    (void)counts;
    (void)count;
    (void)error;
    (void)limit;
    return 0;
}

int icore_mixed_policy_read(uint64_t *counts, size_t ncounts, double *values, size_t nvalues)
{
    (void)counts;
    (void)ncounts;
    (void)values;
    (void)nvalues;
    return 0;
}

int icore_mixed_sweeps_read(uint64_t *counts, size_t ncounts)
{
    (void)counts;
    (void)ncounts;
    return 0;
}
#endif
/* The automatic mixed Newton limit is captured once for both directions at
 * this iterate. The shared driver and final original KKT test are unchanged.
 * An explicitly positive inner tolerance keeps the double-precision route. */
static void mixed_select_inner_policy(icore_ocp_workspace *w, const icore_ocp_options *opt, double gap)
{
    double degree, mu, limit;
    if (w->mixed_disabled || opt->riccati_mode != 0 || !w->orthant_direct ||
        !w->ocp_diagonal_cost_ready || w->ne || w->nf || w->nsoc || !icore_ocp_la_mixed_ready(w->ocp_la))
        return;
    limit = ipm_refinement_limit(opt);
    w->mixed_forcing = limit;
    w->mixed_policy_values[2] = limit;
    if (opt->refinement_tolerance > 0.0) {
        ++w->mixed_policy_counts[1];
        if (limit > w->mixed_policy_values[3])
            w->mixed_policy_values[3] = limit;
        /* An explicit inner tolerance keeps the double-precision route. */
        w->mixed_disabled = 1;
        return;
    }
    ++w->mixed_policy_counts[0];
    /* Match exactly the active barrier degree supplied to ipm_run. */
    degree = w->K * (double)(w->nlp + w->nsoc) - w->constant_count;
    w->mixed_policy_values[1] = degree;
    if (!(degree > 0.0) || !isfinite(degree) || !isfinite(gap) || gap < 0.0) {
        ++w->mixed_policy_counts[3];
        w->mixed_disabled = 1;
        return;
    }
    mu = gap / degree;
    w->mixed_policy_values[0] = mu;
    if (!isfinite(mu)) {
        ++w->mixed_policy_counts[3];
        w->mixed_disabled = 1;
        return;
    }
    if (mu < 1e-5) {
        ++w->mixed_policy_counts[2];
        w->mixed_policy_values[6] = mu;
        w->mixed_disabled = 1;
        return;
    }
    limit = fmax(limit, fmin(1e-4, 0.01 * mu));
    w->mixed_forcing = limit;
    w->mixed_policy_values[2] = limit;
    if (limit > w->mixed_policy_values[3])
        w->mixed_policy_values[3] = limit;
}

static void mixed_clear_directions(icore_ocp_workspace *w)
{
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->sweep_pd_valid = 0;
    w->orthant_gv_direction = NULL;
    w->dynamics_cache_direction = NULL;
}

static int factor_mixed_square_root(icore_ocp_workspace *w, int diagonal, int zero_initial_direction,
                                    int affine_predictor)
{
#if ICORE_OCP_PACKED_SQRT
    int i, k, status, failed_stage = w->N, phase = 0;
    int sp_sweeps;
    double floor = fmax(w->reg, 1024.0 * DBL_MIN);
    if (w->mixed_disabled || !diagonal || !zero_initial_direction || !affine_predictor ||
        !w->orthant_direct || w->ne || w->nf || w->nsoc || !icore_ocp_la_mixed_ready(w->ocp_la))
        return 0;
    for (k = 0; k < w->N; ++k)
        if (!w->ocp_packed_ready[k])
            return 0;
    sp_sweeps = w->mixed_sweeps_requested && icore_ocp_la_mixed_sweeps_available(w->ocp_la);
    ++w->mixed_counts[0];
    if (sp_sweeps)
        ++w->mixed_sweep_counts[0];
    mixed_clear_directions(w);
    form_affine_rhs(w);
    prepare_main_reduced_rhs(w);
    for (i = 0; i < w->nx; ++i)
        w->sweep_p[(size_t)w->N * w->nx + i] = -w->rhs[(size_t)w->N * w->nv + w->nu + i];
    status = sp_sweeps
                 ? icore_ocp_la_mixed_sweeps_begin(w->ocp_la, w->ocp_diagonal + (size_t)w->N * w->nv + w->nu,
                                          w->reg, floor, 1e-12)
                 : icore_ocp_la_mixed_begin(w->ocp_la, w->ocp_diagonal + (size_t)w->N * w->nv + w->nu,
                                   w->reg, floor, 1e-12);
    for (k = w->N - 1; status == ICORE_LA_OK && k >= 0; --k) {
        failed_stage = k;
        phase = 1;
        status = sp_sweeps
                     ? icore_ocp_la_mixed_sweeps_stage(w->ocp_la, k, w->ocp_diagonal + (size_t)k * w->nv,
                                              w->reg, floor, 1e-12)
                     : icore_ocp_la_mixed_stage(w->ocp_la, k, w->ocp_diagonal + (size_t)k * w->nv, w->reg,
                                       floor, 1e-12);
        if (status == ICORE_LA_OK) {
            phase = 2;
            if (sp_sweeps) {
                ++w->mixed_sweep_counts[2];
                status = icore_ocp_la_mixed_sweeps_backward(
                    w->ocp_la, k, w->rhs + (size_t)k * w->nv, w->re + (size_t)k * w->nx,
                    w->sweep_p + (size_t)(k + 1) * w->nx, 0, w->sweep_pd + (size_t)k * w->nx,
                    w->sweep_a + (size_t)k * w->nr, k ? w->sweep_p + (size_t)k * w->nx : NULL, 1);
            } else
                status = icore_ocp_la_factor_backward_stage(
                    w->ocp_la, k, w->rhs + (size_t)k * w->nv, w->re + (size_t)k * w->nx,
                    w->sweep_p + (size_t)(k + 1) * w->nx, w->sweep_pd + (size_t)k * w->nx,
                    w->sweep_a + (size_t)k * w->nr, k ? w->sweep_p + (size_t)k * w->nx : NULL);
        }
    }
    if (status != ICORE_LA_OK) {
#if ICORE_OCP_MIXED_DIAGNOSTICS
        mixed_failure_stage = failed_stage;
        mixed_failure_phase = phase;
        mixed_failure_status = status;
#endif
        w->mixed_sweeps_active = 0;
        icore_ocp_la_mixed_sweeps_invalidate(w->ocp_la);
        w->mixed_disabled = 1;
        w->mixed_factor_active = 0;
        ++w->mixed_counts[2];
        mixed_clear_directions(w);
        return 0;
    }
    w->mixed_factor_active = 1;
    ++w->mixed_counts[1];
    w->mixed_sweeps_active = sp_sweeps;
    if (sp_sweeps)
        ++w->mixed_sweep_counts[1];
    w->ocp_sqrt_active = 1;
    w->factor_poor = 0;
    w->affine_state = ICORE_OCP_AFFINE_BACKWARD_READY;
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
