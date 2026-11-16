#ifndef ICORE_OCP_LA_H
#define ICORE_OCP_LA_H

#include "dense_la.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct icore_ocp_la_workspace icore_ocp_la_workspace;

enum { ICORE_OCP_LA_PIVOT_REJECTED = -7, ICORE_OCP_LA_SCHUR_REJECTED = -8 };

/* Optional BLASFEO stage operations; no optimizer, epigraph elimination,
 * termination tests or fallback policy. With no BLASFEO build,
 * create returns ICORE_LA_UNAVAILABLE so the caller keeps its existing kernels.
 * Dimensions are constant across stages; nx>0, nr>0, N>0, n=nr+nx.
 * All allocation happens here. One workspace is not reentrant.
 */
int icore_ocp_la_create(int N, int nx, int nr, icore_ocp_la_workspace **out);
void icore_ocp_la_destroy(icore_ocp_la_workspace *w);
/* Optional canonical packed factors. The ordinary create/export contract
 * remains available; owned chains may omit both L/T output pointers. */
int icore_ocp_la_create_owned(int N, int nx, int nr, icore_ocp_la_workspace **out);
int icore_ocp_la_sqrt_owns_factors(const icore_ocp_la_workspace *w);
int icore_ocp_la_sweep_backward(icore_ocp_la_workspace *w, int stage, const double *q_u, const double *q_x, double *a,
                       double *p);
int icore_ocp_la_sweep_forward(icore_ocp_la_workspace *w, int stage, const double *q_u, const double *x, double *u);

/* Pack reduced dynamics D=[B A], nx by n row-major, only on accepted data
 * updates. For the existing epigraph elimination use D_original+ne with
 * row_stride=nv. Failed loads preserve that stage's previous D. Accepted
 * loads invalidate every cached square-root factor.
 */
int icore_ocp_la_load_dynamics(icore_ocp_la_workspace *w, int stage, const double *D, size_t row_stride);

/* Optional cost cache, independent of dynamics. Stage 0..N-1
 * has width n, terminal N has width nx. Only the input's lower triangle
 * is read for validation and subsequent arithmetic. A successful load
 * owns a copy and invalidates the factor chain; a failed load preserves
 * the old cost and chain. Off-diagonals remain immutable until the next
 * successful load; cached factor calls replace the complete diagonal and
 * use it as factor scratch. No original-H product may consume that scratch
 * diagonal. No allocation occurs in this call.
 */
int icore_ocp_la_load_cost(icore_ocp_la_workspace *w, int stage, const double *H, size_t row_stride);

/* Engine-internal finite-certified loads. The caller must already have proved
 * every source entry inspected by the corresponding public loader finite,
 * before any mutation, and retain that proof through copy-only ownership.
 * ocp.c uses these only for canonical D/H after coefficients_valid and
 * structure checks. They must not receive transformed arithmetic results.
 * Pointer/dimension/stride checks, packing, ownership and chain invalidation
 * are identical to the public loaders. No new alias or allocation contract
 * is introduced. Ordinary module callers use the validating loaders above. */
int icore_ocp_la_load_dynamics_finite(icore_ocp_la_workspace *w, int stage, const double *D, size_t row_stride);
int icore_ocp_la_load_cost_finite(icore_ocp_la_workspace *w, int stage, const double *H, size_t row_stride);

/* Load an exactly diagonal cost, supplied as n (terminal: nx) contiguous
 * finite entries. Owns all entries and establishes zero off-diagonals.
 * Input may alias packed storage. Repeated diagonal loads preserve the
 * certified off-diagonal zeros while replacing every diagonal entry.
 * Factor/checkpoint invalidation and failed-load preservation match
 * load_cost(). No positivity assumption or allocation is introduced. */
int icore_ocp_la_load_diagonal_cost(icore_ocp_la_workspace *w, int stage, const double *diagonal);

/* Products for the caller's unchanged classical Riccati code:
 * PD = P_next*D, M = H + D^T*P_next*D + reg*I.
 * P_next: nx by nx symmetric, H: n by n symmetric; lower triangles only.
 * P_next and H are packed once per call; D stays packed between iterations.
 * Optional outputs PD and M are contiguous row-major nx*n and n*n; M is
 * exported fully symmetric. If both outputs are supplied, they must be
 * disjoint. No output changes on a failed call.
 */
int icore_ocp_la_products(icore_ocp_la_workspace *w, int stage, const double *P_next, size_t ldP, const double *H,
                 size_t ldH, double reg, double *PD, double *M);

/* Full classical stage using the same M construction:
 * L*L^T=M_uu; T=M_xu*L^-T; P=M_xx-T*T^T.
 * Partial POTRF(m=n,n=nr) fuses the control factor and cross solve, followed
 * by a packed Schur SYRK. P_next need not be positive definite. The leading
 * control block must pass the caller-provided pivot policy.
 *
 * Require every physical pre-sqrt control pivot >=
 * max(absolute, relative*max(abs(diag(M_uu)))). No clipping or scaled retry.
 * If the existing negative-Schur warning condition is met, return
 * ICORE_OCP_LA_SCHUR_REJECTED so the caller recomputes its existing classical/QR path.
 *
 * All three outputs are required and must be disjoint: L[nr*nr] (upper zero),
 * T[nx*nr], P[nx*nx] (symmetric), contiguous row-major. Outputs can alias input
 * matrices because input packing finishes before writes. No output changes
 * on failure. Caller must redo the complete old stage after rejection.
 */
int icore_ocp_la_factor_stage(icore_ocp_la_workspace *w, int stage, const double *P_next, size_t ldP, const double *H,
                     size_t ldH, double reg, double absolute, double relative, double *L, double *T,
                     double *P);

/* Packed square-root chain. Begin starts a new chain and invalidates every
 * earlier chain, including on failure. H_terminal is the original nx*nx
 * terminal state block, lower triangle, before adding reg. Begin factors
 * H_terminal+reg*I. Stage calls must then occur in order N-1,...,0.
 *
 * Stage forms V=D^T*C_next and factors H+reg*I+V*V^T, retaining C_stage
 * from the bottom-right Cholesky block. Outputs L and T have exactly the
 * same physical row-major meaning as the classical stage; no P is formed.
 * The first factor uses max(absolute,relative*maxabsdiag(M_uu)) for control
 * pivots and the analogous state-block criterion. Only ICORE_OCP_LA_PIVOT_REJECTED
 * triggers one stage-local diagonal equilibration retry, with D_ii defined
 * by 1/sqrt(M_ii). That retry requires each scaled pivot to meet
 * relative*maxdiag(D*M*D) and each restored physical pivot to meet absolute.
 * Every factor row, including the retained state root, is unscaled before
 * use. Terminal begin keeps its original absolute/relative criterion.
 * All full-stage factor entries and reciprocal diagonals must be finite.
 * There is no clipping, extra shift or second addition of regularization.
 *
 * Any failed begin/stage invalidates the chain for apply/checkpoint. An
 * opt-in recovery API below can retain a guarded, validated suffix. L/T must
 * be disjoint but may alias H. No output writes occur on failure. The
 * caller retains its complete classical/QR fallback for rejected chains.
 */
int icore_ocp_la_sqrt_begin(icore_ocp_la_workspace *w, const double *H_terminal, size_t row_stride, double reg,
                   double absolute, double relative);
int icore_ocp_la_sqrt_stage(icore_ocp_la_workspace *w, int stage, const double *H, size_t row_stride, double reg,
                   double absolute, double relative, double *L, double *T);

/* Optional last call for a caller whose initial state is fixed. Valid only
 * while stage zero is pending. A zero-offset partial fused factor computes
 * L0/T0 and omits C0; it checks only control pivots and all L0/T0 entries.
 * The local retry equilibrates only controls, using unit state row scales.
 * No state pivot or positive state-diagonal requirement is imposed.
 *
 * This opt-in chain permits apply(1..N); apply(0) returns ICORE_LA_NOT_READY
 * without writes. Calling the ordinary stage(0) instead retains the full
 * generic apply(0) contract. Failure invalidation and output ownership are
 * identical to stage(). No terminal-constraint algebra is performed here. */
int icore_ocp_la_sqrt_initial_fixed(icore_ocp_la_workspace *w, const double *H, size_t row_stride, double reg,
                           double absolute, double relative, double *L, double *T);

/* Explicit zero-initial-direction variant, owned factors only. Ordinary
 * input dimensions/finite checks are unchanged, but a successful short
 * factor retains only L0 for H_uu+B0' P1 B0; neither T0 nor C0 is formed.
 * On control-factor rejection, this call recomputes the full initial-fixed
 * stage, including its existing equilibration and QR-recovery contract.
 * Query initial_zero_active() to distinguish a short accepted factor from
 * that complete L0/T0 fallback. Begin/failure/accepted loads invalidate the
 * token; checkpoint/restore carries it with the corresponding factors.
 * Legacy stage-zero backward/forward sweeps return ICORE_LA_NOT_READY while
 * the token is active. Other stages and apply(1..N) remain usable.
 * The caller must never use a short chain for a nonzero fixed initial state.
 */
int icore_ocp_la_sqrt_initial_zero(icore_ocp_la_workspace *w, const double *H, size_t row_stride, double reg,
                          double absolute, double relative);
int icore_ocp_la_sqrt_initial_zero_cached_diagonal(icore_ocp_la_workspace *w, const double *diagonal, double reg,
                                          double absolute, double relative);
int icore_ocp_la_sqrt_initial_zero_active(const icore_ocp_la_workspace *w);

/* Token-gated initial-stage operations. sweep computes L0^-1*q when
 * transpose=0 and L0^-T*q when transpose=1; q/out contain nr controls.
 * dynamics_product uses original B0 only: input/output sizes nr/nx when
 * transpose=0 and nx/nr when transpose=1. It computes alpha*op(B0)*x+beta*y.
 * Inputs/outputs may alias. A zero beta ignores incoming y. Every failure
 * preserves the complete output; no operation allocates or consumes T0.
 */
int icore_ocp_la_initial_zero_sweep(icore_ocp_la_workspace *w, int transpose, const double *q, double *out);
int icore_ocp_la_initial_zero_dynamics_product(icore_ocp_la_workspace *w, int transpose, double alpha,
                                      const double *x, double beta, double *y);

/* Internal owned-factor stage sweeps for a caller with no epigraph or
 * terminal-equality corrections. A completed packed chain is required.
 * They use the same kernels and intermediate finite checks as the separate
 * apply/product/triangular APIs, retaining vectors in preallocated storage.
 * The caller retains its ordinary stage fallback if an operation rejects.
 *
 * Backward forms Pd=P_next*d, q=-bd+D'*(pnext+Pd), then a=L^-1*q_u and
 * p=q_x-T*a. bd has n entries; d/pnext/Pd/p have nx; a has nr. With
 * reuse_pd=1, pd is a required input and d is ignored. Otherwise d is
 * required and pd is an optional output committed only with a/p.
 * At short initial-zero stage 0, bd has nr entries and p MUST be NULL;
 * only B0' is used and p0 is omitted. Ordinary/full stage calls require p.
 *
 * Forward forms u=-L^-T*(a+T'*x), next=D*[u;x]+d, and
 * dual=P_next*next+pnext. a/u have nr entries; x/d/pnext/next/dual have nx.
 * At short initial-zero stage 0, x MUST be NULL and only B0 is used.
 * Other stages, including a full initial-fixed retry, require x.
 *
 * All caller outputs must be pairwise disjoint. Pd must remain disjoint
 * from a/p even when reused, preserving the caller's cache. Other inputs
 * may alias outputs because all inputs are consumed before any commit.
 * Every failed call
 * preserves all caller outputs and the factor/checkpoint readiness. */
int icore_ocp_la_packed_backward_stage(icore_ocp_la_workspace *w, int stage, const double *bd, const double *d,
                              const double *pnext, int reuse_pd, double *pd, double *a, double *p);
/* Engine-private scheduling variant. Call immediately after an accepted
 * owned factor at this stage, while the completed suffix is still current:
 * factor progress must equal stage-1. It performs the identical fresh-Pd
 * backward operations above; pointer, alias, finite and atomic-output
 * contracts are unchanged. It does not mark the complete chain ready or
 * commit any engine RHS cache. Ordinary callers use the full-chain API. */
int icore_ocp_la_factor_backward_stage(icore_ocp_la_workspace *w, int stage, const double *bd, const double *d,
                              const double *pnext, double *pd, double *a, double *p);
int icore_ocp_la_packed_forward_stage(icore_ocp_la_workspace *w, int stage, const double *a, const double *x,
                             const double *d, const double *pnext, double *u, double *next,
                             double *dual);
/* Engine-only forward variant for stage>0. Also commits the original D*[u;x]
 * before d is added. The extra nx-vector must be disjoint from all three
 * outputs; it may alias inputs. No output changes on failure. */
int icore_ocp_la_packed_forward_stage_cached(icore_ocp_la_workspace *w, int stage, const double *a, const double *x,
                                    const double *d, const double *pnext, double *u, double *next,
                                    double *dual, double *unshifted_dynamics);

/* Same square-root operations and pivot policy, with cached off-diagonal
 * cost. The supplied diagonal is the COMPLETE current unregularized
 * diagonal (cost plus current diagonal curvature), not a diagonal increment.
 * It has nx entries at begin and n entries at a control stage. This permits
 * the caller to preserve its original ordered H_ii += g_i*g_i accumulation.
 * Regularization is added exactly once by the existing factor operation.
 * Every diagonal must be finite. Missing cost returns ICORE_LA_NOT_READY.
 * Begin/stage failure invalidates the chain and leaves L/T unchanged,
 * exactly like the ordinary APIs. Ordinary and cached stages may be mixed.
 */
int icore_ocp_la_sqrt_begin_cached_diagonal(icore_ocp_la_workspace *w, const double *diagonal, double reg,
                                   double absolute, double relative);
int icore_ocp_la_sqrt_stage_cached_diagonal(icore_ocp_la_workspace *w, int stage, const double *diagonal, double reg,
                                   double absolute, double relative, double *L, double *T);
int icore_ocp_la_sqrt_initial_fixed_cached_diagonal(icore_ocp_la_workspace *w, const double *diagonal, double reg,
                                           double absolute, double relative, double *L, double *T);

/* Diagnostic count of successful equilibrated stages in the latest begun
 * chain attempt. Reset by begin and accepted dynamics loads; a subsequent
 * stage failure does not erase the count. Zero for null/unavailable backend. */
int icore_ocp_la_sqrt_equilibrated_stages(const icore_ocp_la_workspace *w);

/* Optional caller-managed QR recovery after an actual ICORE_OCP_LA_PIVOT_REJECTED
 * from the currently pending begin/stage, including its equilibration retry.
 * A private token retains only validated downstream physical roots, while
 * ordinary apply/checkpoint remain unavailable. The caller must reconstruct
 * its exact current tall QR stack, with the same regularization and stage
 * equations; this API does not create a shift or relax a Cholesky pivot test.
 *
 * pending() is a read-only query matching stage, initial-fixed flag, and the
 * exact failed regularization. Stage N denotes terminal begin and requires
 * initial_fixed=0. A new begin/stage or classical product/factor, an accepted
 * model load, or checkpoint restore invalidates an older recovery token.
 * Failed model loads preserve the old model and token.
 *
 * product() is only for control stages. It recomputes and exports the
 * row-major nx*n propagation block C_next^T D from untouched physical roots
 * and original dynamics, never from retry-scaled scratch. All output entries
 * stay unchanged on failure, including padding. A failed recovery operation
 * invalidates its token; the caller retains complete classical/QR fallback.
 *
 * accept() requires the positive-finite-diagonal upper factor R of that exact
 * tall QR stack, row-major n*n (terminal nx*nx); lower/padding values are
 * ignored. It validates all upper entries and copies before committing, so
 * R may alias L/T. Nonterminal L[nr*nr] and T[nx*nr] must be disjoint; terminal
 * L/T must be NULL. Interior C_stage=R_xx^T is retained; initial_fixed=1 at
 * stage 0 omits C0 and leaves apply(0) unavailable. Success resumes stage-1;
 * no partial output writes occur on failure. No operation allocates.
 */
int icore_ocp_la_sqrt_recovery_pending(const icore_ocp_la_workspace *w, int stage, int initial_fixed, double reg);
int icore_ocp_la_sqrt_recovery_product(icore_ocp_la_workspace *w, int stage, double *RD, size_t row_stride);
int icore_ocp_la_sqrt_recovery_accept(icore_ocp_la_workspace *w, int stage, int initial_fixed, double reg,
                             const double *R, size_t row_stride, double *L, double *T);

/* Multiply the last successfully loaded, original stage dynamics. With
 * transpose=0, x has nr+nx entries and y has nx; transpose=1 reverses them.
 * Compute y=alpha*op(D)*x+beta*y. Input/output overlap is supported. A zero
 * beta does not inspect the incoming y. Failed calls leave all y unchanged.
 * This needs loaded dynamics, not a completed or current factor chain. */
int icore_ocp_la_dynamics_product(icore_ocp_la_workspace *w, int stage, int transpose, double alpha, const double *x,
                         double beta, double *y);

/* Paired original dynamics products: gradient += D' dual, dynamics = D primal.
 * primal/gradient have nr+nx entries; dual/dynamics have nx. Inputs may alias
 * outputs, but the two output spans must be disjoint. Incoming dynamics is
 * ignored. Every failure preserves both complete caller outputs. No factors
 * are needed, and no call allocates. */
int icore_ocp_la_dynamics_pair(icore_ocp_la_workspace *w, int stage, const double *primal, const double *dual,
                      double *gradient, double *dynamics);
/* Availability of the pinned GEMV_NT accumulation order used below. */
int icore_ocp_la_dynamics_cache_available(void);
/* Engine-only N half of dynamics_pair, with the same seeded FMA order.
 * The caller separately owns a current original D*primal product. Inputs
 * may alias the output. Every failure leaves gradient unchanged. */
int icore_ocp_la_dynamics_nt_add(icore_ocp_la_workspace *w, int stage, const double *dual, double *gradient);

/* Optional exact checkpoint of one completed physical state-root chain.
 * Storage is attempted during create; failure leaves ordinary operations
 * usable and makes checkpoint/restore return ICORE_LA_UNAVAILABLE. Capture only
 * accepts a completed chain and preserves an older checkpoint on failure.
 * Normal begin/stage calls do not erase the checkpoint. A successful cost or
 * dynamics load invalidates it; caller owns the additional G/F/mode/reg key.
 * Restore reinstates root-omitted/equilibration metadata and apply readiness,
 * without touching cost/dynamics or other stage scratch. No call allocates.
 */
/* Internal bordered Riccati matrix operations on an accepted owned stage.
 * Cn/C are nf*nx, Z is nr*nf, and S is nf*nf, all row-major and disjoint.
 * Form U=Cn*D, solve L*Z=Uu', C=Ux-(T*Z)', and add Z'*Z to lower S.
 * Caller retains the existing full row-major fallback. No writes on error.
 * Only immediate accepted factor progress (sqrt_next=stage-1) is eligible.
 * All scratch was allocated by create; no solve-time allocation occurs. */
/* Export L/T only for immediate accepted owned factor progress. L[nr*nr]
 * and T[nx*nr] must be disjoint; no output changes on rejection. */
int icore_ocp_la_export_stage_factor(icore_ocp_la_workspace *w, int stage, double *L, double *T);
int icore_ocp_la_terminal_border(icore_ocp_la_workspace *w, int stage, int nf, const double *Cn, double *C, double *Z,
                        double *S);

int icore_ocp_la_sqrt_checkpoint(icore_ocp_la_workspace *w);
int icore_ocp_la_sqrt_restore(icore_ocp_la_workspace *w);
void icore_ocp_la_sqrt_discard_checkpoint(icore_ocp_la_workspace *w);
size_t icore_ocp_la_sqrt_checkpoint_bytes(const icore_ocp_la_workspace *w);
size_t icore_ocp_la_sqrt_checkpoint_copy_bytes(const icore_ocp_la_workspace *w);

/* Once the complete chain is accepted, compute P_stage*x=C*(C^T*x).
 * stage may be 0..N, except 0 after opting into initial_fixed().
 * Input/output aliasing is supported. The output stays
 * unchanged on any failure, and the cached factors are not modified.
 */
int icore_ocp_la_sqrt_apply(icore_ocp_la_workspace *w, int stage, const double *x, double *y);

/* Optional packed Householder acceleration for the caller's existing tall
 * QR stack. This is a separate workspace; it does not alter Riccati roots,
 * checkpoint state, model data, regularization, or factor-quality policies.
 * Creation failure leaves *out NULL and the caller's portable QR usable.
 */
typedef struct icore_ocp_la_qr_workspace icore_ocp_la_qr_workspace;
int icore_ocp_la_qr_create(int max_rows, int max_columns, icore_ocp_la_qr_workspace **out);
void icore_ocp_la_qr_destroy(icore_ocp_la_qr_workspace *w);
size_t icore_ocp_la_qr_bytes(const icore_ocp_la_qr_workspace *w);

/* Factor the complete row-major rows-by-columns input, rows >= columns.
 * On success only the leading upper triangular R is committed, with positive
 * diagonal. All other input/padding entries remain unchanged. Every failure
 * leaves the entire input unchanged for the caller's original QR recovery.
 * No shift, pivot clipping, or numerical acceptance tolerance is added.
 * The input is transposed into packed storage and factored by zero-offset
 * LQ. Nonfinite data reject; extreme nonzero input magnitudes outside
 * [2^-450,2^450] return ICORE_LA_UNAVAILABLE before factorization so the caller
 * retains its hypot-based range-safe QR. No call allocates after create.
 */
int icore_ocp_la_qr_upper(icore_ocp_la_qr_workspace *w, int rows, int columns, double *A, size_t row_stride);

#ifdef __cplusplus
}
#endif
#endif
