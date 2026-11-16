#include "driver.h"
#include <math.h>
#include <string.h>

int ipm_options_valid(const icore_ocp_options *o)
{
    return o && o->max_iterations >= 0 && o->refinement_steps >= 0 &&
           o->riccati_mode >= ICORE_OCP_AUTO && o->riccati_mode <= ICORE_OCP_SQUARE_ROOT &&
           o->tolerance > 0.0 && isfinite(o->tolerance) && o->step_fraction > 0.0 &&
           o->step_fraction < 1.0 && o->regularization >= 0.0 && isfinite(o->regularization) &&
           o->refinement_tolerance >= 0.0 && isfinite(o->refinement_tolerance);
}

double ipm_refinement_limit(const icore_ocp_options *o)
{
    return o->refinement_tolerance > 0.0 ? o->refinement_tolerance : fmin(1e-6, 0.1 * o->tolerance);
}

/* Generic ABI entry; statically known internal kernels use the same body. */
int ipm_run(void *context, const ipm_kernel *kernel, double degree, const icore_ocp_options *o,
            int step_policy, int curved_cones, ipm_run_info *info)
{
    return ipm_run_inline(context, kernel, degree, o, step_policy, curved_cones, info);
}
