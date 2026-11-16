/* Chain callbacks for the shared IPM loop and public solve adapters.
 * Private implementation included by chain.c; not a separate translation unit.
 */

/* ---- Chain callbacks and public adapters for the common IPM loop. ---- */

INLINE double fraction(const double *x, const double *dx, int n)
{
    double a = 1.;
    int i;
    for (i = 0; i < n; ++i)
        if (dx[i] < 0.) {
            double b = -x[i] / dx[i];
            if (b < a)
                a = b;
        }
    return a;
}

typedef struct {
    chain_workspace *w;
    const icore_ocp_options *options;
    int qr_count, ref_count;
} chain_context;

static int chain_initialize(void *context)
{
    chain_context *c = (chain_context *)context;
    chain_workspace *w = c->w;
    int i, nv = w->nv, nd = w->nd, m = w->m;
    double minimum = INFINITY, maximum_raw = -INFINITY;
    double *x = w->state, *t = x + nv, *s = t + nd, *z = s + m;
    for (i = 0; i < w->nr; ++i)
        w->curvature[i] = i < nd ? 2. : 1.;
    if (!factor(w, c->options->regularization, c->options->riccati_mode))
        return 0;
    for (i = 0; i < nv; ++i)
        w->rhs[i] = -w->c[i];
    for (i = 0; i < w->nr; ++i)
        row_add(w, i, -w->curvature[i] * w->offset[i], w->rhs);
    linear_solve(w, w->rhs, x);
    for (i = 0; i < nd; ++i)
        t[i] = -.5 * w->lambda[i];
    apply(w, x, t, s, 1);
    for (i = 0; i < m; ++i) {
        s[i] = -s[i];
        if (s[i] < minimum)
            minimum = s[i];
        if (s[i] > maximum_raw)
            maximum_raw = s[i];
    }
    {
        double zs = maximum_raw > -1. ? 1. + maximum_raw : 0.,
               ss = minimum < 1. ? 1. - minimum : 0.;
        for (i = 0; i < m; ++i) {
            z[i] = -s[i] + zs;
            s[i] += ss;
        }
    }
    return 1;
}

static int chain_evaluate(void *context, ipm_residual *result)
{
    chain_context *c = (chain_context *)context;
    double error[3];
    residual(c->w, error);
    result->primal = error[0];
    result->dual = error[1];
    result->gap = error[2];
    return 1;
}

static int chain_factor(void *context)
{
    chain_context *c = (chain_context *)context;
    chain_workspace *w = c->w;
    int i, nd = w->nd, m = w->m;
    const double *s = w->state + w->nv + nd, *z = s + m;
    for (i = 0; i < m; ++i)
        w->weight[i] = z[i] / s[i];
    for (i = 0; i < nd; ++i) {
        double wp = w->weight[i], wm = w->weight[nd + i];
        double small = wp < wm ? wp : wm, large = wp > wm ? wp : wm;
        w->inverse[i] = 1. / (wp + wm);
        w->ratio[i] = (wm - wp) * w->inverse[i];
        w->curvature[i] = 4. * small / (1. + small / large);
    }
    for (i = nd; i < w->nr; ++i)
        w->curvature[i] = w->weight[nd + i];
    if (c->options->riccati_mode == ICORE_OCP_AUTO && w->saddle && w->saddle->valid) {
        if (chain_saddle_factor(w, c->options->regularization))
            return 1;
        /* Range or factor failure cannot discard the original recovery. */
        if (factor(w, c->options->regularization, ICORE_OCP_CLASSICAL))
            return 1;
        return factor_qr(w, c->options->regularization);
    }
    /* Once the current cold solve requires the square-root factor, avoid
     * repeatedly rebuilding the cancellation-prone normal equations.
     * Initialization of every new solve still starts from the requested mode. */
    if (c->options->riccati_mode == ICORE_OCP_AUTO && w->use_qr) {
        if (factor(w, c->options->regularization, ICORE_OCP_SQUARE_ROOT))
            return 1;
        /* A preferred QR failure must not remove the original AUTO recovery
         * path: rebuild from coefficients before trying classical/QR again. */
        return factor(w, c->options->regularization, ICORE_OCP_AUTO);
    }
    return factor(w, c->options->regularization, c->options->riccati_mode);
}

static int chain_form_rhs(void *context, double target, int corrector)
{
    chain_context *c = (chain_context *)context;
    chain_workspace *w = c->w;
    int i, m = w->m;
    const double *s = w->state + w->nv + w->nd, *z = s + m;
    const double *ads = w->affine + w->nv + w->nd, *adz = ads + m;
    double *bc = w->target + w->nv + w->nd + m;
    if (corrector)
        for (i = 0; i < m; ++i)
            bc[i] = -s[i] * z[i] - ads[i] * adz[i] + target;
    else
        for (i = 0; i < m; ++i)
            bc[i] = -s[i] * z[i];
    return 1;
}

static int chain_direction(void *context, int corrector)
{
    chain_context *c = (chain_context *)context;
    chain_workspace *w = c->w;
    const icore_ocp_options *o = c->options;
    double limit = ipm_refinement_limit(o), norm, *step = corrector ? w->step : w->affine;
    int started_augmented = w->saddle && w->saddle->valid;
    int before = c->ref_count;
    norm = direction(w, w->target, step, o->refinement_steps, limit, &c->ref_count);
    if ((!isfinite(norm) || norm > limit) && !w->use_qr && o->riccati_mode == ICORE_OCP_AUTO) {
        int remaining = o->refinement_steps;
        /* Preserve all successful ordinary refinements.  A classical-start
         * direction reaches here only at its original QR recovery trigger. */
        if (started_augmented)
            remaining -= c->ref_count - before;
        else if (o->refinement_steps > 1 && chain_saddle_factor(w, o->regularization)) {
            double scales[4], candidate = INFINITY;
            int refined_before;
            scales[0] = 1. + maximum(w->target, w->nv);
            scales[1] = 1. + maximum(w->target + w->nv, w->nd);
            scales[2] = 1. + maximum(w->target + w->nv + w->nd, w->m);
            scales[3] = 1. + maximum(w->target + w->nv + w->nd + w->m, w->m);
            /* This original-RHS proposal spends one existing QR correction
             * slot, including a failed or rejected RHS.  Factor failure alone
             * spends none. The original QR initial solve remains available. */
            ++c->ref_count;
            --remaining;
            if (chain_saddle_rhs(w, w->target, w->trial))
                candidate = newton_error(w, w->target, w->trial, w->trial_error, scales);
            if (isfinite(candidate) && candidate < norm) {
                memcpy(step, w->trial, (size_t)w->dim * sizeof(double));
                memcpy(w->error, w->trial_error, (size_t)w->dim * sizeof(double));
                norm = candidate;
                refined_before = c->ref_count;
                norm = refine_direction(w, w->target, step, remaining, limit, &c->ref_count, scales,
                                        norm);
                remaining -= c->ref_count - refined_before;
            } else {
                chain_saddle_invalidate(w->saddle);
                w->use_qr = 0;
            }
        }
        /* Augmented corrections and QR corrections share the old remaining
         * budget. A raw augmented-to-QR fallback in solve_rhs sets use_qr,
         * so this outer path cannot repeat that reserved QR initial attempt. */
        if ((!isfinite(norm) || norm > limit) && !w->use_qr) {
            if (!factor_qr(w, o->regularization))
                return 0;
            norm = direction(w, w->target, step, remaining, limit, &c->ref_count);
        }
    }
    if (!corrector && !w->m)
        memcpy(w->step, w->affine, (size_t)w->dim * sizeof(double));
    return isfinite(norm);
}

static void chain_step_bounds(void *context, int corrector, double *primal, double *dual)
{
    chain_context *c = (chain_context *)context;
    chain_workspace *w = c->w;
    const double *s = w->state + w->nv + w->nd, *z = s + w->m;
    const double *ds = (corrector ? w->step : w->affine) + w->nv + w->nd, *dz = ds + w->m;
    *primal = fraction(s, ds, w->m);
    *dual = fraction(z, dz, w->m);
}

static double chain_affine_pair(void *context, double primal, double dual)
{
    chain_context *c = (chain_context *)context;
    chain_workspace *w = c->w;
    int i;
    double value = 0.;
    const double *s = w->state + w->nv + w->nd, *z = s + w->m;
    const double *ds = w->affine + w->nv + w->nd, *dz = ds + w->m;
    for (i = 0; i < w->m; ++i)
        value += fabs((s[i] + primal * ds[i]) * (z[i] + dual * dz[i]));
    return value;
}

static int chain_interior(void *context, double alpha)
{
    chain_context *c = (chain_context *)context;
    chain_workspace *w = c->w;
    int i;
    const double *s = w->state + w->nv + w->nd, *z = s + w->m;
    const double *ds = w->step + w->nv + w->nd, *dz = ds + w->m;
    for (i = 0; i < w->m; ++i)
        if (!(s[i] + alpha * ds[i] > 0.) || !(z[i] + alpha * dz[i] > 0.))
            return 0;
    return 1;
}

static void chain_increment(void *context, double alpha)
{
    chain_context *c = (chain_context *)context;
    axpy(c->w->state, c->w->step, alpha, c->w->dim);
    c->qr_count += c->w->use_qr;
}

/* Certify a predictor that already solves the original chain.  Residual
 * evaluation writes the RHS workspace, so rejection restores that RHS while
 * leaving the iterate, factorization and affine direction intact. */
static int chain_accept_affine(void *context, double alpha, ipm_residual *result)
{
    chain_context *c = (chain_context *)context;
    chain_workspace *w = c->w;
    double *state = w->state;
    ipm_residual candidate;
    int i, valid;
    for (i = 0; i < w->dim; ++i)
        w->trial[i] = state[i] + alpha * w->affine[i];
    w->state = w->trial;
    valid = chain_evaluate(context, &candidate) &&
            isfinite(candidate.primal + candidate.dual + candidate.gap) &&
            candidate.primal <= 0.5 * c->options->tolerance &&
            candidate.dual <= 0.5 * c->options->tolerance &&
            candidate.gap <= 0.5 * c->options->tolerance;
    w->state = state;
    if (valid) {
        memcpy(state, w->trial, (size_t)w->dim * sizeof(double));
        c->qr_count += w->use_qr;
        *result = candidate;
        return 1;
    }
    chain_evaluate(context, result);
    return 0;
}

static const ipm_kernel chain_kernel = {
    chain_initialize,  chain_evaluate,    chain_factor,   chain_form_rhs,  chain_direction,
    chain_step_bounds, chain_affine_pair, chain_interior, chain_increment, chain_accept_affine};

int chain_solve_options_policy(chain_workspace *w, const icore_ocp_options *options, int step_policy,
                               double *out_y, double *out_t, double *out_s, double *out_z,
                               chain_info *information)
{
    icore_ocp_options defaults = icore_ocp_default_options();
    const icore_ocp_options *opt = options ? options : &defaults;
    chain_info local, *info = information ? information : &local;
    ipm_run_info run;
    chain_context context;
    double errors[3], *x, *t, *s, *z;
    int i, status, nv, nd, m;
    memset(info, 0, sizeof(*info));
    info->status = ICORE_OCP_INVALID_MODEL;
    if (!w || !out_y || !w->valid || (w->nd && !out_t) || (w->m && (!out_s || !out_z)) ||
        !ipm_options_valid(opt) || step_policy < ICORE_OCP_STEP_AUTO || step_policy > ICORE_OCP_STEP_ADAPTIVE)
        return info->status;
    nv = w->nv;
    nd = w->nd;
    m = w->m;
    x = w->state;
    t = x + nv;
    s = t + nd;
    z = s + m;
    context.w = w;
    context.options = opt;
    context.qr_count = 0;
    context.ref_count = 0;
    status = ipm_run_inline(&context, &chain_kernel, (double)m, opt, step_policy, 0, &run);
    residual(w, errors);
    for (i = 0; i < nv; ++i)
        out_y[i] = w->fixed[i] + w->free[i] * x[i];
    if (nd)
        memcpy(out_t, t, (size_t)nd * sizeof(double));
    if (m) {
        memcpy(out_s, s, (size_t)m * sizeof(double));
        memcpy(out_z, z, (size_t)m * sizeof(double));
    }
    for (i = nd; i < w->nr; ++i)
        if (w->rows[i].constant) {
            out_s[nd + i] = -w->original_offset[i];
            out_z[nd + i] = 0.;
        }
    info->status = status;
    info->iterations = run.iterations;
    info->qr_fallbacks = context.qr_count;
    info->refinements = context.ref_count;
    info->primal = errors[0];
    info->dual = errors[1];
    info->gap = errors[2];
    for (i = 0; i < nv; ++i) {
        double delta = out_y[i] - w->center[i];
        info->objective += .5 * w->original_h[i] * delta * delta + w->original_c[i] * out_y[i];
    }
    for (i = 0; i < nd; ++i)
        info->objective += w->lambda[i] * fabs(row_dot(w, i, x) + w->offset[i]);
    return status;
}

int chain_solve_options(chain_workspace *w, const icore_ocp_options *options, double *y, double *t,
                        double *s, double *z, chain_info *info)
{
    return chain_solve_options_policy(w, options, ICORE_OCP_STEP_FIXED, y, t, s, z, info);
}

/* Legacy signature is an options adapter, with no separate solver loop. */
int chain_solve(chain_workspace *w, double tolerance, int max_iterations, int refinement_steps,
                double regularization, double *y, double *t, double *s, double *z, chain_info *info)
{
    icore_ocp_options options = icore_ocp_default_options();
    if (!info)
        return ICORE_OCP_INVALID_MODEL;
    options.tolerance = tolerance;
    options.max_iterations = max_iterations;
    options.refinement_steps = refinement_steps;
    options.regularization = regularization;
    return chain_solve_options(w, &options, y, t, s, z, info);
}
