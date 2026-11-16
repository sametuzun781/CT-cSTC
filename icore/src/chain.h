#ifndef ICORE_CHAIN_H
#define ICORE_CHAIN_H
#include "icore_ocp.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct chain_workspace chain_workspace;

/* Row-major arrays. l1_weight has (K-1)*d entries after broadcasting. */
typedef struct {
    const double *left, *right, *offset, *pair_left, *pair_right, *pair_offset;
    const double *local_matrix, *local_offset, *center, *hessian, *linear;
    const double *l1_weight, *free, *fixed;
} chain_problem;

typedef struct {
    int status, iterations, qr_fallbacks, refinements;
    double primal, dual, gap, objective;
} chain_info;

chain_workspace *chain_create(int K, int n, int d, int p, int l);
void chain_destroy(chain_workspace *w);
int chain_update(chain_workspace *w, const chain_problem *q);
/* Outputs s,z use (positive L1, negative L1, pair, local) order. */
int chain_solve(chain_workspace *w, double tolerance, int max_iterations, int refinement_steps,
                double regularization, double *y, double *t, double *s, double *z,
                chain_info *info);
/* Same IPM options as the OCP layout. riccati_mode chooses automatic,
 * Cholesky-only, or square-root factorization of the physical blocks. */
int chain_solve_options(chain_workspace *w, const icore_ocp_options *options, double *y, double *t,
                        double *s, double *z, chain_info *info);
/* Explicit policy adapter; legacy entry points keep the FIXED policy. */
int chain_solve_options_policy(chain_workspace *w, const icore_ocp_options *options, int step_policy,
                               double *y, double *t, double *s, double *z, chain_info *info);
#ifdef __cplusplus
}
#endif
#endif
