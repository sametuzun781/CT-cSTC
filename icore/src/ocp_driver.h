/* Callbacks connecting OCP arithmetic to the common Mehrotra driver.
 * This file also owns public solve adapters and diagnostic accessors.
 * driver_loop.h owns the outer iteration and stopping rule; driver.c exports the generic ABI.
 * Private implementation: included exactly once by ocp.c. */

typedef struct {
    icore_ocp_workspace *w;
    const icore_ocp_options *options;
    icore_ocp_info *info;
    size_t nvcount, ycount, mcount;
} ocp_context;

static int ocp_initialize(void *context)
{
    ocp_context *c = (ocp_context *)context;
    if (!initialize(c->w, c->options))
        return 0;
    if (c->w->constant_primal_defect > c->options->tolerance) {
        /* The fixed row cannot be repaired by an IPM step.  Report every
         * residual from the finite cold initializer rather than a stale
         * iterate or a misleading zero-filled information structure. */
        residuals(c->w, c->info);
        return 0;
    }
    return 1;
}

static int ocp_evaluate(void *context, ipm_residual *result)
{
    ocp_context *c = (ocp_context *)context;
    residuals(c->w, c->info);
    result->primal = c->info->primal_residual;
    result->dual = c->info->dual_residual;
    result->gap = c->info->gap;
    return isfinite(c->info->objective);
}

static int ocp_factor(void *context)
{
    ocp_context *c = (ocp_context *)context;
    if (orthant_direct_scaling(c->w)) {
        mixed_select_inner_policy(c->w, c->options, c->info->gap);
        if (factor_problem_requested(c->w, c->options, 1, 1))
            return 1;
    }
    return scaling_update(c->w) && factor_problem_context(c->w, c->options, 1);
}

static int ocp_form_rhs(void *context, double target, int corrector)
{
    ocp_context *c = (ocp_context *)context;
    c->w->orthant_target = target;
    if (corrector) {
        if (corrector_rhs(c->w, target))
            return 1;
        if (!c->w->orthant_direct || !scaling_update(c->w) ||
            !factor_problem_context(c->w, c->options, 1))
            return 0;
        return corrector_rhs(c->w, target);
    }
    if (c->w->affine_state == ICORE_OCP_AFFINE_BACKWARD_READY && c->w->orthant_direct &&
        c->w->ocp_sqrt_active) {
        c->w->affine_state = ICORE_OCP_AFFINE_PREDICTOR_ARMED;
        return 1;
    }
    form_affine_rhs(c->w);
    return 1;
}

static int ocp_direction(void *context, int corrector)
{
    ocp_context *c = (ocp_context *)context;
    icore_ocp_workspace *w = c->w;
    if (corrector)
        w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    int direction_ok = newton_direction(w, c->options);
    if (!direction_ok && w->mixed_factor_active) {
        /* Failed refinement overwrote brd/bre/brc. Rebuild the full DP chain
         * at this same iterate, then reconstruct q/RHS from rd/re/rp and the
         * preserved affine ads/adz for a corrector. Try this before NT retry. */
        if (w->mixed_sweeps_active)
            ++w->mixed_sweep_counts[4];
        w->mixed_sweeps_active = 0;
        icore_ocp_la_mixed_sweeps_invalidate(w->ocp_la);
        w->mixed_disabled = 1;
        w->mixed_factor_active = 0;
        ++w->mixed_counts[3];
        mixed_clear_directions(w);
        if (factor_problem_context(w, c->options, 1)) {
            int rhs_ok;
            if (corrector)
                rhs_ok = corrector_rhs(w, w->orthant_target);
            else {
                form_affine_rhs(w);
                rhs_ok = 1;
            }
            if (rhs_ok)
                direction_ok = newton_direction(w, c->options);
        }
    }
    if (!direction_ok) {
        /* Retry the general cone-scaling equations if direct ratio/RHS arithmetic
         * encounters an unsafe intermediate.  The iterate is unchanged. */
        if (!w->orthant_direct || !scaling_update(w) || !factor_problem_context(w, c->options, 1) ||
            !ocp_form_rhs(context, w->orthant_target, corrector) ||
            !newton_direction(w, c->options))
            return 0;
    }
    if (!corrector) {
        memcpy(w->ads, w->ds, c->mcount * sizeof(double));
        memcpy(w->adz, w->dz, c->mcount * sizeof(double));
    }
    return 1;
}

static void ocp_step_bounds(void *context, int corrector, double *primal, double *dual)
{
    ocp_context *c = (ocp_context *)context;
    icore_ocp_workspace *w = c->w;
    *primal = cone_step(w, w->s, corrector ? w->ds : w->ads);
    *dual = cone_step(w, w->z, corrector ? w->dz : w->adz);
}

static double ocp_affine_pair(void *context, double primal, double dual)
{
    ocp_context *c = (ocp_context *)context;
    icore_ocp_workspace *w = c->w;
    size_t i;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR) && (defined(__GNUC__) || defined(__clang__))
    if (!w->nsoc) {
        double sum = 0.0;
        for (i = 0; i < c->mcount; ++i)
            if (!w->constant_orthant[i]) {
                double s = w->s[i] + primal * w->ads[i], z = w->z[i] + dual * w->adz[i];
                double product;
                /* These empty register barriers preserve the two rounded trial
                 * values and their separately rounded product. They emit no
                 * instruction or memory store. */
                __asm__("" : "+x"(s), "+x"(z));
                product = s * z;
                __asm__("" : "+x"(product));
                sum += fabs(product);
            }
        return sum;
    }
#endif
    for (i = 0; i < c->mcount; ++i) {
        w->cz[i] = w->s[i] + primal * w->ads[i];
        w->tz[i] = w->z[i] + dual * w->adz[i];
    }
    return cone_pair(w, w->cz, w->tz);
}

static int ocp_interior(void *context, double alpha)
{
    ocp_context *c = (ocp_context *)context;
    icore_ocp_workspace *w = c->w;
    size_t i;
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR) && (defined(__GNUC__) || defined(__clang__))
    if (!w->nsoc) {
        int interior = 1;
        for (i = 0; i < c->mcount; ++i) {
            double s = w->s[i] + alpha * w->ds[i], z = w->z[i] + alpha * w->dz[i];
            __asm__("" : "+x"(s), "+x"(z));
            /* Constant rows are omitted only from positivity, exactly as
             * cone_min did. Their finite trial values are still required. */
            if (!isfinite(s) || !isfinite(z))
                return 0;
            if (!w->constant_orthant[i] && !(s > 0.0 && z > 0.0))
                interior = 0;
        }
        return interior;
    }
#endif
    for (i = 0; i < c->mcount; ++i) {
        w->cz[i] = w->s[i] + alpha * w->ds[i];
        w->tz[i] = w->z[i] + alpha * w->dz[i];
        if (!isfinite(w->cz[i]) || !isfinite(w->tz[i]))
            return 0;
    }
    return cone_min(w, w->cz) > 0.0 && cone_min(w, w->tz) > 0.0;
}

static void ocp_update(void *context, double alpha)
{
    ocp_context *c = (ocp_context *)context;
    icore_ocp_workspace *w = c->w;
    size_t i;
    w->mixed_sweeps_active = 0;
    icore_ocp_la_mixed_sweeps_invalidate(w->ocp_la);
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
    for (i = 0; i < c->nvcount; ++i)
        w->v[i] += alpha * w->dv[i];
    for (i = 0; i < c->ycount; ++i)
        w->y[i] += alpha * w->dy[i];
    for (i = 0; i < (size_t)w->nf; ++i)
        w->nu_t[i] += alpha * w->dnu[i];
    for (i = 0; i < c->mcount; ++i) {
        w->s[i] += alpha * w->ds[i];
        w->z[i] += alpha * w->dz[i];
    }
}

/* The predictor sometimes already solves the original QP. Test that claim
 * in disjoint trial arrays instead of computing an unnecessary corrector.
 * The same evaluator (including its absolute KKT, positivity, and finite
 * checks) certifies the trial. A rejected trial restores the original
 * residuals and leaves the iterate, affine direction, and factors intact. */
static int accept_affine_trial(ocp_context *c, double alpha,
                               int (*evaluate)(void *, ipm_residual *), ipm_residual *result,
                               int dense)
{
    icore_ocp_workspace *w = c->w;
    double *v = w->v, *y = w->y, *nu = w->nu_t, *s = w->s, *z = w->z;
    ipm_residual trial;
    size_t i;
    int valid;
    for (i = 0; i < c->nvcount; ++i)
        w->tv[i] = v[i] + alpha * w->dv[i];
    for (i = 0; i < c->ycount; ++i)
        w->ty[i] = y[i] + alpha * w->dy[i];
    for (i = 0; i < (size_t)w->nf; ++i)
        w->tnu[i] = nu[i] + alpha * w->dnu[i];
    for (i = 0; i < c->mcount; ++i) {
        w->cz[i] = s[i] + alpha * w->ds[i];
        w->tz[i] = z[i] + alpha * w->dz[i];
    }
    w->v = w->tv;
    w->y = w->ty;
    w->nu_t = w->tnu;
    w->s = w->cz;
    w->z = w->tz;
    if (dense) {
        memcpy(w->v + w->nu, w->x0, (size_t)w->nx * sizeof(double));
        for (i = 0; i < (size_t)w->nx; ++i)
            w->v[w->nv + w->nu + i] = dot(w->D + i * w->nv, w->v, w->nv) + w->b[i];
    }
    valid = evaluate(c, &trial) && isfinite(trial.primal + trial.dual + trial.gap) &&
            trial.primal <= 0.5 * c->options->tolerance &&
            trial.dual <= 0.5 * c->options->tolerance && trial.gap <= 0.5 * c->options->tolerance;
    w->v = v;
    w->y = y;
    w->nu_t = nu;
    w->s = s;
    w->z = z;
    if (valid) {
        memcpy(v, w->tv, c->nvcount * sizeof(double));
        memcpy(y, w->ty, c->ycount * sizeof(double));
        memcpy(nu, w->tnu, (size_t)w->nf * sizeof(double));
        memcpy(s, w->cz, c->mcount * sizeof(double));
        memcpy(z, w->tz, c->mcount * sizeof(double));
        *result = trial;
        return 1;
    }
    /* Evaluation invalidates direction-product tokens. Recompute only the
     * original residuals; the ordinary corrector rebuilds its own RHS. */
    evaluate(c, result);
    return 0;
}

static int ocp_accept_affine(void *context, double alpha, ipm_residual *result)
{
    return accept_affine_trial((ocp_context *)context, alpha, ocp_evaluate, result, 0);
}

static const ipm_kernel ocp_kernel = {
    ocp_initialize,  ocp_evaluate,    ocp_factor,   ocp_form_rhs, ocp_direction,
    ocp_step_bounds, ocp_affine_pair, ocp_interior, ocp_update,   ocp_accept_affine};

#include "dense_qp.h"

int icore_ocp_dense_face_attempts(const icore_ocp_workspace *w)
{
    return w ? w->qp_face_attempts : 0;
}

int icore_ocp_dense_face_steps(const icore_ocp_workspace *w)
{
    return w ? w->qp_face_steps : 0;
}

int icore_ocp_dense_face_accepted(const icore_ocp_workspace *w)
{
    return w ? w->qp_face_accepted : 0;
}

int icore_ocp_last_kernel(const icore_ocp_workspace *w)
{
    return w ? w->qp_used : 0;
}

int icore_ocp_dense_fallbacks(const icore_ocp_workspace *w)
{
    return w ? w->qp_fallbacks : 0;
}

int icore_ocp_dense_packed(const icore_ocp_workspace *w)
{
    return w && w->qp_used && w->qp_packed_ready;
}

int icore_ocp_ocp_packed_stages(const icore_ocp_workspace *w)
{
    return w ? w->ocp_packed_stages : 0;
}

int icore_ocp_ocp_packed_rejected(const icore_ocp_workspace *w)
{
    return w ? w->ocp_packed_rejected : 0;
}

int icore_ocp_ocp_sqrt_factors(const icore_ocp_workspace *w)
{
    return w ? w->ocp_sqrt_factors : 0;
}

int icore_ocp_ocp_sqrt_rejected(const icore_ocp_workspace *w)
{
    return w ? w->ocp_sqrt_rejected : 0;
}

int icore_ocp_ocp_sqrt_equilibrated(const icore_ocp_workspace *w)
{
    return w ? w->ocp_sqrt_equilibrated : 0;
}

/* Compatibility adapter: validation/output only; ipm_run owns the iteration. */
int icore_ocp_solve_policy(icore_ocp_workspace *w, const icore_ocp_options *options, int step_policy, double *v,
                      double *y, double *terminal_dual, double *s, double *z,
                      icore_ocp_info *information)
{
    icore_ocp_options defaults = icore_ocp_default_options();
    const icore_ocp_options *opt = options ? options : &defaults;
    icore_ocp_info local, *info = information ? information : &local;
    ipm_run_info run;
    ocp_context context;
    clock_t begin = clock();
    int status;
    memset(info, 0, sizeof(*info));
    if (!w || !ipm_options_valid(opt) || step_policy < ICORE_OCP_STEP_AUTO ||
        step_policy > ICORE_OCP_STEP_ADAPTIVE) {
        info->status = ICORE_OCP_INVALID_MODEL;
        return info->status;
    }
    w->mixed_sweeps_active = 0;
    icore_ocp_la_mixed_sweeps_invalidate(w->ocp_la);
    w->mixed_factor_active = 0;
    w->mixed_disabled = 0;
#if ICORE_OCP_MIXED_DIAGNOSTICS
    mixed_failure_stage = -1;
    mixed_failure_phase = -1;
    mixed_failure_status = 0;
#endif
    memset(w->mixed_counts, 0, sizeof(w->mixed_counts));
    memset(w->mixed_sweep_counts, 0, sizeof(w->mixed_sweep_counts));
    w->mixed_last_error = 0.0;
    w->mixed_limit = ipm_refinement_limit(opt);
    w->mixed_max_error = 0.0;
    w->mixed_forcing = w->mixed_limit;
    memset(w->mixed_policy_counts, 0, sizeof(w->mixed_policy_counts));
    memset(w->mixed_policy_values, 0, sizeof(w->mixed_policy_values));
    w->mixed_counts[7] = (uint64_t)icore_ocp_la_mixed_ready(w->ocp_la);
    context.w = w;
    context.options = opt;
    context.info = info;
    context.nvcount = (size_t)w->K * w->nv;
    context.ycount = (size_t)w->N * w->nx;
    context.mcount = (size_t)w->K * w->m;
    w->factor_retries = 0;
    w->refinements = 0;
    w->dual_recovery_attempts = 0;
    w->dual_recovery_accepted = 0;
    w->qp_used = 0;
    w->qp_face_attempts = w->qp_face_steps = w->qp_face_accepted = 0;
    w->qp_fallbacks = 0;
    w->ocp_packed_stages = 0;
    w->ocp_packed_rejected = 0;
    w->affine_state = ICORE_OCP_AFFINE_EMPTY;
    w->dynamics_cache_direction = NULL;
    w->ocp_sqrt_active = 0;
    w->classical_initial_zero = 0;
    w->ocp_sqrt_factors = 0;
    w->ocp_sqrt_rejected = 0;
    w->ocp_sqrt_equilibrated = 0;
    if (w->qp_eligible && opt->riccati_mode != ICORE_OCP_SQUARE_ROOT &&
        w->constant_primal_defect <= opt->tolerance) {
        w->qp_used = 1;
        status = ipm_run_inline(&context, &dense_qp_kernel, (double)w->qp_active, opt, step_policy,
                                0, &run);
        if (status == ICORE_OCP_NUMERICAL_FAILURE && run.iterations < opt->max_iterations) {
            int spent = run.iterations;
            icore_ocp_options fallback = *opt;
            fallback.max_iterations -= spent;
            context.options = &fallback;
            w->initial_valid = 0;
            w->qp_used = 0;
            ++w->qp_fallbacks;
            status = ipm_run_inline(&context, &ocp_kernel,
                                    w->K * (double)(w->nlp + w->nsoc) - w->constant_count -
                                        (double)w->constant_soc_count,
                                    &fallback, step_policy, w->nsoc, &run);
            run.iterations += spent;
            context.options = opt;
        }
    } else {
        status = ipm_run_inline(&context, &ocp_kernel,
                                w->K * (double)(w->nlp + w->nsoc) - w->constant_count -
                                    (double)w->constant_soc_count,
                                opt, step_policy, w->nsoc, &run);
    }
    if (v)
        memcpy(v, w->v, context.nvcount * sizeof(double));
    if (y)
        memcpy(y, w->y, context.ycount * sizeof(double));
    if (terminal_dual)
        memcpy(terminal_dual, w->nu_t, (size_t)w->nf * sizeof(double));
    if (s)
        memcpy(s, w->s, context.mcount * sizeof(double));
    if (z)
        memcpy(z, w->z, context.mcount * sizeof(double));
    if (s || z) {
        size_t row;
        for (row = 0; row < context.mcount; ++row)
            if (w->constant_orthant[row]) {
                if (s)
                    s[row] = fmax(w->constant_slack[row], 0.0);
                if (z)
                    z[row] = 0.0;
            }
    }
    if ((s || z) && w->constant_soc_count)
        for (int k = 0; k < w->K; ++k)
            for (int c = 0; c < w->nsoc; ++c)
                if (constant_soc(w, k, c)) {
                    size_t row = (size_t)k * w->m + w->offsets[c];
                    if (s)
                        memset(s + row, 0, (size_t)w->sizes[c] * sizeof(double));
                    if (z)
                        memset(z + row, 0, (size_t)w->sizes[c] * sizeof(double));
                }
    info->status = status;
    info->iterations = run.iterations;
    info->refinements = w->refinements;
    info->factor_retries = w->factor_retries;
    info->seconds = (double)(clock() - begin) / (double)CLOCKS_PER_SEC;
    w->mixed_counts[6] = (uint64_t)w->mixed_disabled;
#if ICORE_OCP_MIXED_DIAGNOSTICS
    memcpy(mixed_last_counts, w->mixed_counts, sizeof(mixed_last_counts));
    memcpy(mixed_last_sweep_counts, w->mixed_sweep_counts, sizeof(mixed_last_sweep_counts));
    mixed_last_error = w->mixed_last_error;
    mixed_last_limit = w->mixed_limit;
    mixed_max_accepted_error = w->mixed_max_error;
    memcpy(mixed_policy_last_counts, w->mixed_policy_counts, sizeof(mixed_policy_last_counts));
    memcpy(mixed_policy_last_values, w->mixed_policy_values, sizeof(mixed_policy_last_values));
#endif
    return status;
}

int icore_ocp_solve(icore_ocp_workspace *w, const icore_ocp_options *options, double *v, double *y,
               double *terminal_dual, double *s, double *z, icore_ocp_info *information)
{
    return icore_ocp_solve_policy(w, options, ICORE_OCP_STEP_FIXED, v, y, terminal_dual, s, z, information);
}
