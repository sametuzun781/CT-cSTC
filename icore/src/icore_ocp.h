#ifndef ICORE_OCP_H
#define ICORE_OCP_H

#ifdef __cplusplus
extern "C" {
#endif

/* Every array is contiguous row-major; stages are the outermost index.
   nv=nu+nx, m=nlp+sum(soc_sizes). v[k]=[u[k],x[k]].
   H,c,G,h contain N+1 stages; A,B,b contain N stages.
   Stage N has no control; its control coefficients must be zero.
   H[k] is nv by nv, A[k] nx by nx, B[k] nx by nu,
   G[k] m by nv, F nf by nx. The library owns a copy after create/update.
   Optional ne: the first ne controls are independent epigraph variables t.
   Each t must occur only in two orthant rows e-t<=0 and -e-t<=0, with a
   unique control e inferred from exact row supports. H and B have zero t
   columns; neither t nor e occurs in any other cone row. Set ne=0 to use
   the general formulation without this reduction. */
typedef struct {
    int N, nx, nu, nlp, nsoc, nf, ne;
    const int *soc_sizes;
    const double *H, *c, *A, *B, *b, *G, *h, *x0, *F, *f;
} icore_ocp_model;

typedef struct {
    int max_iterations, refinement_steps, riccati_mode;
    double tolerance, step_fraction, regularization, refinement_tolerance;
} icore_ocp_options;

typedef struct {
    int status, iterations, refinements, factor_retries;
    /* seconds is process CPU time. For latency, time the call with a wall clock. */
    double objective, primal_residual, dual_residual, gap, seconds;
} icore_ocp_info;

typedef struct icore_ocp_workspace icore_ocp_workspace;

enum { ICORE_OCP_AUTO = 0, ICORE_OCP_CLASSICAL = 1, ICORE_OCP_SQUARE_ROOT = 2 };

/* Step policy is separate from icore_ocp_options to preserve its ABI. */
enum { ICORE_OCP_STEP_AUTO = 0, ICORE_OCP_STEP_FIXED = 1, ICORE_OCP_STEP_ADAPTIVE = 2 };

enum {
    ICORE_OCP_SOLVED = 0,
    ICORE_OCP_ITERATION_LIMIT = 1,
    ICORE_OCP_NUMERICAL_FAILURE = 2,
    ICORE_OCP_INVALID_MODEL = 3,
    ICORE_OCP_ALLOCATION_FAILURE = 4
};

icore_ocp_options icore_ocp_default_options(void);
icore_ocp_workspace *icore_ocp_create(const icore_ocp_model *model);
/* Transactional numeric/model update. Same dimensions/cone layout required. */
int icore_ocp_update(icore_ocp_workspace *workspace, const icore_ocp_model *model);
void icore_ocp_destroy(icore_ocp_workspace *workspace);
/* Cold solve. Output buffers, when non-null, receive independently owned values.
   v:(N+1)*nv; y:N*nx; terminal_dual:nf; s,z:(N+1)*m.
   Equality convention: A*x+B*u+b-x_next=0 and F*x_N-f=0.
   No memory allocation is performed by solve. */
int icore_ocp_solve(icore_ocp_workspace *workspace, const icore_ocp_options *options, double *v, double *y,
               double *terminal_dual, double *s, double *z, icore_ocp_info *info);
/* Explicit step policy; the legacy icore_ocp_solve entry point remains FIXED. */
int icore_ocp_solve_policy(icore_ocp_workspace *workspace, const icore_ocp_options *options, int step_policy,
                      double *v, double *y, double *terminal_dual, double *s, double *z,
                      icore_ocp_info *info);
/* Last completed solve's kernel: 0=general OCP, 1=direct dense orthant.
   Dense fallback count is separate from QR factorization retry statistics. */
int icore_ocp_dense_face_attempts(const icore_ocp_workspace *workspace);
int icore_ocp_dense_face_steps(const icore_ocp_workspace *workspace);
int icore_ocp_dense_face_accepted(const icore_ocp_workspace *workspace);
int icore_ocp_last_kernel(const icore_ocp_workspace *workspace);
int icore_ocp_dense_fallbacks(const icore_ocp_workspace *workspace);
int icore_ocp_dense_packed(const icore_ocp_workspace *workspace);
int icore_ocp_ocp_packed_stages(const icore_ocp_workspace *workspace);
int icore_ocp_ocp_packed_rejected(const icore_ocp_workspace *workspace);
int icore_ocp_ocp_sqrt_factors(const icore_ocp_workspace *workspace);
int icore_ocp_ocp_sqrt_rejected(const icore_ocp_workspace *workspace);
int icore_ocp_ocp_sqrt_equilibrated(const icore_ocp_workspace *workspace);

#ifdef __cplusplus
}
#endif
#endif
