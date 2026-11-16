#ifndef ICORE_DRIVER_H
#define ICORE_DRIVER_H

#include "icore_ocp.h"

/* Internal model operations. Newton solves may use different matrix layouts;
 * all nonlinear iteration policy lives in ipm_run. */
typedef struct {
    double primal, dual, gap;
} ipm_residual;

typedef struct {
    int status, iterations;
    ipm_residual residual;
} ipm_run_info;

typedef struct {
    int (*initialize)(void *context);
    int (*evaluate)(void *context, ipm_residual *residual);
    int (*factor)(void *context);
    int (*form_rhs)(void *context, double target, int corrector);
    int (*direction)(void *context, int corrector);
    void (*step_bounds)(void *context, int corrector, double *primal, double *dual);
    double (*affine_pair)(void *context, double primal, double dual);
    int (*interior)(void *context, double alpha);
    void (*update)(void *context, double alpha);
    /* Optional transactional trial. Commit only an independently checked
     * affine trial satisfying the original-coordinate stopping conditions. */
    int (*accept_affine)(void *context, double alpha, ipm_residual *residual);
} ipm_kernel;

int ipm_options_valid(const icore_ocp_options *options);
double ipm_refinement_limit(const icore_ocp_options *options);
int ipm_run(void *context, const ipm_kernel *kernel, double cone_degree,
            const icore_ocp_options *options, int step_policy, int curved_cones, ipm_run_info *info);

#include "driver_loop.h"

#endif
