#ifndef ICORE_DRIVER_LOOP_H
#define ICORE_DRIVER_LOOP_H
#include <math.h>
#include <string.h>
/* One maintained algorithm body. Internal callers expose their immutable
 * kernel tables to the compiler; the generic exported entry remains valid.
 * No model-specific tolerance, numerical policy, or runtime selector. */
/* The single primal-dual Mehrotra iteration for every model layout. */
#if defined(__GNUC__) || defined(__clang__)
static inline
    __attribute__((always_inline))
#else
static inline
#endif
    int
    ipm_run_inline(void *context, const ipm_kernel *kernel, double degree, const icore_ocp_options *o,
                   int step_policy, int curved_cones, ipm_run_info *info)
{
    int iteration = 0, status = ICORE_OCP_NUMERICAL_FAILURE;
    memset(info, 0, sizeof(*info));
    if (!context || !kernel || !ipm_options_valid(o) || !(degree >= 0.0) || !isfinite(degree) ||
        curved_cones < 0 || step_policy < ICORE_OCP_STEP_AUTO || step_policy > ICORE_OCP_STEP_ADAPTIVE) {
        info->status = ICORE_OCP_INVALID_MODEL;
        return info->status;
    }
    /* Start from the model's cold initializer, never from a previous solved iterate. */
    if (!kernel->initialize(context))
        goto done;
    for (iteration = 0; iteration <= o->max_iterations; ++iteration) {
        double alpha = 1.0;
        int backtracks;
        if (!kernel->evaluate(context, &info->residual) ||
            !isfinite(info->residual.primal + info->residual.dual + info->residual.gap))
            break;
        if (info->residual.primal <= o->tolerance && info->residual.dual <= o->tolerance &&
            info->residual.gap <= o->tolerance) {
            status = ICORE_OCP_SOLVED;
            break;
        }
        if (iteration == o->max_iterations) {
            status = ICORE_OCP_ITERATION_LIMIT;
            break;
        }
        /* Predictor: one factorization, then the affine Newton direction. */
        if (!kernel->factor(context) || !kernel->form_rhs(context, 0.0, 0) ||
            !kernel->direction(context, 0))
            break;
        if (degree > 0.0) {
            double primal, dual, mu, mu_aff, sigma;
            kernel->step_bounds(context, 0, &primal, &dual);
            mu = info->residual.gap / degree;
            mu_aff = fmax(0.0, kernel->affine_pair(context, primal, dual)) / degree;
            if (kernel->accept_affine && curved_cones == 0 && step_policy != ICORE_OCP_STEP_FIXED &&
                fmin(primal, dual) >= 0.99 && mu_aff <= 0.5 * o->tolerance / degree &&
                kernel->accept_affine(context, fmin(primal, dual), &info->residual)) {
                status = ICORE_OCP_SOLVED;
                ++iteration;
                break;
            }
            /* Corrector: reuse the factorization with the centering/correction RHS. */
            sigma = fmin(1.0, fmax(0.0, mu_aff / mu));
            sigma = sigma * sigma * sigma;
            if (!kernel->form_rhs(context, sigma * mu, 1) || !kernel->direction(context, 1))
                break;
            kernel->step_bounds(context, 1, &primal, &dual);
            {
                double admissible = fmin(primal, dual);
                if (step_policy == ICORE_OCP_STEP_ADAPTIVE ||
                    (step_policy == ICORE_OCP_STEP_AUTO && curved_cones == 0)) {
                    /* Keep conservative damping for a small admissible step;
                     * approach a full step when the Newton direction allows it.
                     * The existing interior check remains the final guard. */
                    double ceiling = fmax(o->step_fraction, 0.9999999);
                    alpha =
                        admissible * ((1.0 - admissible) * o->step_fraction + admissible * ceiling);
                } else
                    alpha = o->step_fraction * admissible;
            }
        }
        if (!(alpha > 0.0) || !isfinite(alpha))
            break;
        /* Interior acceptance remains the final numerical guard on every path. */
        for (backtracks = 0; backtracks < 20; ++backtracks) {
            if (kernel->interior(context, alpha))
                break;
            alpha *= 0.5;
        }
        if (backtracks == 20 || !(alpha > 0.0))
            break;
        kernel->update(context, alpha);
    }
done:
    info->status = status;
    info->iterations = iteration;
    return status;
}

#endif
