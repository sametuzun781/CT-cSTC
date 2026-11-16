#ifndef ICORE_DENSE_LA_H
#define ICORE_DENSE_LA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Standalone linear algebra only. No IPM policy, stopping test or HPIPM call.
 * Opaque workspaces are independent; one workspace must not be used concurrently.
 * C99. Define ICORE_LA_WITH_BLASFEO=1 when compiling dense_la.c for the optional
 * packed backend. With no definition, there is no BLASFEO dependency at all.
 */
typedef struct icore_la_workspace icore_la_workspace;

typedef enum {
    ICORE_LA_AUTO = 0, /* BLASFEO if compiled and supported; otherwise portable. */
    ICORE_LA_PORTABLE = 1,
    ICORE_LA_BLASFEO = 2 /* Fails explicitly if unavailable on this CPU/build. */
} icore_la_backend;

enum {
    ICORE_LA_OK = 0,
    ICORE_LA_INVALID = -1,
    ICORE_LA_NOMEM = -2,
    ICORE_LA_NOT_READY = -3,
    ICORE_LA_NOT_SPD = -4,
    ICORE_LA_NONFINITE = -5,
    ICORE_LA_UNAVAILABLE = -6
};

/* Fixed dimension n>0, capacity max_general>=0. All allocation is here;
 * every subsequent load, factor, solve and product is allocation-free.
 * On failure *out is NULL. destroy(NULL) is valid.
 */
int icore_la_create(int n, int max_general, icore_la_backend requested, icore_la_workspace **out);
void icore_la_destroy(icore_la_workspace *w);
int icore_la_blasfeo_available(void);
icore_la_backend icore_la_backend_used(const icore_la_workspace *w);
const char *icore_la_backend_name(const icore_la_workspace *w);
const char *icore_la_status_string(int status);

/* Load original model data only when it changes. Strides are in doubles.
 * H: n by n row-major; ONLY the lower triangle is read and validated.
 * G: general_rows by n row-major; represents G*x, not -G*x.
 * Accepted loads invalidate the current factor. Rejected loads leave the
 * previous model and factor untouched. G may be cleared with rows=0, G=NULL.
 * The packed backend retains H and G^T in panel-major layout between calls.
 */
int icore_la_load_h(icore_la_workspace *w, const double *h, size_t row_stride);
int icore_la_load_g(icore_la_workspace *w, int general_rows, const double *g, size_t row_stride);

/* Factor A = H + diag(diagonal) + G^T diag(weights) G + regularization*I.
 * diagonal: n nonnegative entries, or NULL for zero.
 * weights: general_rows nonnegative entries; may be NULL only if rows=0.
 * These are already-formed finite barrier contributions. The caller owns
 * exact indexed-box aggregation (including signs, scales, duplicate rows),
 * range-safe z/s calculation, regularization and fallback policy.
 *
 * equilibrate=1 factors D*A*D, D_i=1/sqrt(A_ii), and solve applies the
 * corresponding RHS/solution scaling. equilibrate=0 factors A directly.
 * No pivot clipping, hidden regularization, or solver-tolerance adjustment.
 * Every factor attempt invalidates the old factor, including failed attempts.
 * A failed factor/solve must trigger the caller's robust linear algebra path.
 */
int icore_la_factor(icore_la_workspace *w, const double *diagonal, const double *weights,
                double regularization, int equilibrate);

/* Smallest successful factor pivot in ORIGINAL matrix units:
 * min_i (L_scaled[i,i]/D[i])^2; D=1 for the unscaled path.
 * This is the pre-square-root pivot, so compare it with the caller's physical
 * pivot floor, not with its square root. Returns NaN without a valid factor.
 * Rejecting a too-small result and selecting the caller's robust path preserves
 * its existing pivot policy without silently clipping the BLASFEO factor.
 */
double icore_la_min_physical_pivot(const icore_la_workspace *w);

/* Actual factor-coordinate pivot and a combined policy check. The latter
 * requires min_actual >= relative*max_diag_of_factored_input AND
 * min_physical >= absolute. Both thresholds must be finite and nonnegative.
 * It returns 0 without a valid factor. No matrix or factor is changed.
 * Computing the original input diagonal adds O(n*general_rows) work for the
 * unscaled path; the equilibrated path already needs exactly this diagonal.
 */
double icore_la_min_scaled_pivot(const icore_la_workspace *w);
int icore_la_pivots_acceptable(const icore_la_workspace *w, double absolute, double relative);

/* Solve the last successfully factored A*x=rhs. rhs and x may alias.
 * Output remains unchanged on failure. Factor is reused across all RHSs.
 */
int icore_la_solve(icore_la_workspace *w, const double *rhs, double *x);

/* Engine-internal packed direction. Solve with rhs-G^T*z, then return G*x.
 * It uses the same packed kernels, equilibration and finite checks as the
 * three separate products/solve. Caller outputs commit only after all checks.
 * Portable workspaces return ICORE_LA_UNAVAILABLE, without changing outputs. */
int icore_la_direction(icore_la_workspace *workspace, const double *rhs, const double *z, double *x,
                   double *gx);

/* y <- alpha*H*x + beta*y using the ORIGINAL H, with x/y aliasing allowed.
 * beta=0 does not read y. Used for original-coordinate stationarity/KKT.
 */
int icore_la_hmv(icore_la_workspace *w, double alpha, const double *x, double beta, double *y);

/* Products using original G. Outputs remain unchanged on failure.
 * Input/output aliasing is allowed; the two outputs of pair must be disjoint.
 * pair performs both products in one BLASFEO gemv_nt call.
 */
int icore_la_gmv(icore_la_workspace *w, const double *x, double *gx);
int icore_la_gtmv(icore_la_workspace *w, const double *z, double *gtz);
/* Packed H*x and G products with a shared validated x. gx is untouched
 * when forward==0. The three output arrays must be disjoint; inputs may
 * alias outputs. Failure leaves all caller outputs unchanged. */
int icore_la_hg_products(icore_la_workspace *w, const double *x, const double *z, double *hx, double *gx,
                     double *gtz, int forward);
/* Original dual residual (-rd-H*x)-G'*z. Each matrix product is evaluated
 * and checked separately, retaining subtraction order before exporting one
 * vector. Inputs may alias outputs; on failure outputs remain unchanged.
 * Nonfinite subtraction reports NONFINITE; a caller that needs its prior
 * overflow/recovery sequence can retry through icore_la_hg_products.
 * Portable backend returns UNAVAILABLE, as for icore_la_hg_products. */
int icore_la_hg_residual(icore_la_workspace *w, const double *x, const double *z, const double *rd,
                     double *residual, double *gx, int forward);

int icore_la_gmv_pair(icore_la_workspace *w, const double *x, const double *z, double *gx, double *gtz);

#ifdef __cplusplus
}
#endif
#endif
