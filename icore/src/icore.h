#ifndef ICORE_PUBLIC_H
#define ICORE_PUBLIC_H
#include "icore_ocp.h"
#include "chain.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Input array layout. Both layouts use the same primal-dual iteration. */
enum { ICORE_STAGE_OCP = 0, ICORE_AFFINE_CHAIN = 1 };

enum {
    ICORE_STEP_AUTO = ICORE_OCP_STEP_AUTO,
    ICORE_STEP_FIXED = ICORE_OCP_STEP_FIXED,
    ICORE_STEP_ADAPTIVE = ICORE_OCP_STEP_ADAPTIVE
};

typedef struct {
    int format;
    const void *data;  /* icore_ocp_model or chain_problem */
    int K, n, d, p, l; /* array dimensions for the affine-chain layout */
} icore_model;

typedef icore_ocp_options icore_options;
typedef icore_ocp_info icore_info;
typedef struct icore_workspace icore_workspace;

typedef struct {
    double *v, *y, *nu, *t, *s, *z;
} icore_output;

icore_options icore_default_options(void);
icore_workspace *icore_create(const icore_model *model);
/* Workspaces default to AUTO: adaptive for orthant-only models,
 * fixed for models containing SOCs. FIXED uses a constant boundary fraction.
 * The setting survives coefficient updates; invalid values leave it intact.
 * The options struct and existing solve ABI are unchanged. */
int icore_set_step_policy(icore_workspace *workspace, int step_policy);
int icore_update(icore_workspace *workspace, const icore_model *model);
int icore_solve(icore_workspace *workspace, const icore_options *options,
                const icore_output *output, icore_info *info);
void icore_destroy(icore_workspace *workspace);
/* Last solve: -1=invalid, 0=general OCP, 1=direct dense QP, 2=affine chain. */
int icore_dense_face_attempts(const icore_workspace *workspace);
int icore_dense_face_steps(const icore_workspace *workspace);
int icore_dense_face_accepted(const icore_workspace *workspace);
int icore_last_kernel(const icore_workspace *workspace);
int icore_dense_fallbacks(const icore_workspace *workspace);
int icore_dense_packed(const icore_workspace *workspace);
int icore_ocp_packed_stages(const icore_workspace *workspace);
int icore_ocp_packed_rejected(const icore_workspace *workspace);
int icore_ocp_sqrt_factors(const icore_workspace *workspace);
int icore_ocp_sqrt_rejected(const icore_workspace *workspace);
int icore_ocp_sqrt_equilibrated(const icore_workspace *workspace);
#ifdef __cplusplus
}
#endif
#endif
