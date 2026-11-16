/* Owned workspace and internal declarations.
 * Dimensions: K=N+1 stages; nv=nu+nx; nr=nu-ne; nvr=nv-ne.
 * Each stage stores [u,x]; ne counts exactly eliminated epigraph controls.
 * m=nlp+sum(SOC sizes), nf is the number of terminal equalities.
 * All pointers are owned by the workspace unless explicitly marked as tokens.
 * Allocation and transactional model updates live in ocp_model.h.
 * Private implementation: included exactly once by ocp.c. */

/* Internal state tags, not user options or build switches.
 * Factor -> RHS -> predictor: EMPTY -> BACKWARD_READY -> PREDICTOR_ARMED -> EMPTY.
 * A residual refresh, model update, retry, or competing solve invalidates the token.
 * Literal expansions preserve the existing integer representation and code. */
#define ICORE_OCP_AFFINE_EMPTY 0
#define ICORE_OCP_AFFINE_BACKWARD_READY 1
#define ICORE_OCP_AFFINE_PREDICTOR_ARMED 2

/* Valid only while dynamics_cache_direction identifies the current direction. */
#define ICORE_OCP_DYNAMICS_ROW_MAJOR 1
#define ICORE_OCP_DYNAMICS_PACKED 2

/* Exact original-coefficient certificates. These never relax input validation. */
#define ICORE_OCP_SINGLETON_UNKNOWN 0
#define ICORE_OCP_SINGLETON_POSITIVE_ZERO 1
#define ICORE_OCP_SINGLETON_COMPARE_BYTES 2
#define ICORE_OCP_COST_VALIDATION_UNKNOWN 0
#define ICORE_OCP_COST_VALIDATION_REUSABLE 1
#define ICORE_OCP_COST_VALIDATION_FULL -1

struct icore_ocp_workspace {
    /* Inactive transaction bank; it never owns another bank. */
    struct icore_ocp_workspace *update_storage;
    int N, K, nx, nu, nv, nr, nvr, m, nlp, nsoc, nf, ne;
    /* Constraint supports and exactly eliminated epigraph pairs. */
    int *sizes, *offsets, *nnz, *indices, *epi_row, *epi_e, *epi_plus, *epi_minus;
    size_t *index_start;
    int *single_index;
    /* 0 unknown, 1 literal +0 outside singleton, 2 requires byte comparison. */
    unsigned char *single_positive_zero;
    double *single_value, *single_scaled;
    int *dense_count, *dense_rows;
    int *cost_start, *cost_length;
    unsigned char *cost_root_diagonal;
    int *dyn_count, *dyn_index, *dyn_sparse;
    double *dyn_value, *PDT;
    unsigned char *constant_orthant;
    double *constant_slack;
    int constant_count;
    double constant_primal_defect;
    /* Direct dense-QP route for the eligible one-interval model. */
    int qp_diagonal_eligible, qp_diagonal_factor;
    int qp_eligible, qp_active, qp_used, qp_fallbacks;
    int qp_reuse_forward;
    int qp_bound_pairs;
    int qp_face_attempts, qp_face_steps, qp_face_accepted;
    double qp_row_magnitude;
    int *qp_rows, *qp_nnz, *qp_box_index;
    double qp_constant, *qp_H, *qp_c, *qp_h, *qp_box_value;
    icore_la_workspace *qp_la;
    int qp_packed_ready, qp_packed_factor, qp_general_count;
    int *qp_general_rows, *qp_gindex;
    double *qp_G, *qp_diag, *qp_weights, *qp_gx, *qp_gt;
    /* Optional packed Riccati and local QR workspaces. */
    icore_ocp_la_workspace *ocp_la;
    icore_ocp_la_qr_workspace *ocp_qr_la;
    unsigned char *ocp_packed_ready;
    int ocp_diagonal_cost_ready;
    double *ocp_diagonal;
    int ocp_packed_stages, ocp_packed_rejected;
    int ocp_sqrt_active, ocp_sqrt_factors, ocp_sqrt_rejected, ocp_sqrt_equilibrated;
    /* Owned coefficients. D=[B,A]; Hs is the cached cost root. */
    double *H, *c, *D, *b, *G, *h, *x0, *F, *f, *Fn, *Fnorm, *Hs;
    /* Iterate and original residuals: stationarity, dynamics, terminal, cone. */
    double *v, *y, *nu_t, *s, *z, *rd, *re, *rf, *rp;
    /* Cold-start cache, separate from the current iterate and Newton factors. */
    double *initial_v, *initial_y, *initial_nu, *initial_s, *initial_z;
    double *aux_factor_storage, *aux_L, *aux_T, *aux_P, *aux_C, *aux_Z;
    double *aux_S, *aux_SL, *aux_epi_inv, *aux_epi_ratio;
    size_t aux_factor_bytes, aux_factor_copy_bytes;
    size_t aux_factor_captures, aux_factor_restores;
    int aux_factor_valid, aux_factor_mode, aux_factor_sqrt, aux_factor_poor;
    int aux_factor_row_scaled;
    double aux_factor_reg;
    /* Cone scaling and barrier-adjusted stage costs. */
    double *boost, *orthant_inv, *scale, *lambda, *Gh, *Hb, *epi_inv, *epi_ratio, *epi_curv;
    /* Riccati factors, reduced terminal constraints, and triangular scratch. */
    double *L, *T, *P, *R, *C, *Z, *S, *SL;
    double *sweep_a, *sweep_p, *sweep_pd, *term_work;
    int sweep_pd_valid;
    /* 0 empty; 1 complete factor/RHS/backward; 2 armed by affine form_rhs.
     * The forward predictor consumes this token before writing outputs. */
    int affine_state;
    /* Newton right-hand sides and original-system correction residuals. */
    double *rhs, *pcone, *bcone, *q, *brd, *bre, *brf, *brc;
    /* Current direction; ads/adz preserve the affine predictor for the corrector. */
    double *dv, *dy, *dnu, *ds, *dz, *ads, *adz;
    /* Refinement correction (c*) and trial direction (t*). */
    double *cv, *cy, *cnu, *cz, *tv, *ty, *tnu, *tz;
    double *M, *PD, *qr, *tmp, *tmp2, *tmp3;
    int qr_rows, factor_poor, factor_retries, refinements;
    int classical_initial_zero;
    int initial_valid, initial_mode;
    int coefficient_cache_valid;
    /* 0 unknown, 1 exact normal/zero symmetric, -1 requires full validation. */
    int cost_validation_reusable;
    /* Per-solve mixed-precision state, counters, and acceptance limits. */
    int mixed_factor_active, mixed_disabled;
    int mixed_sweeps_active, mixed_sweeps_requested;
    uint64_t mixed_sweep_counts[8];
    uint64_t mixed_counts[8];
    double mixed_last_error, mixed_limit, mixed_max_error;
    double mixed_forcing;
    uint64_t mixed_policy_counts[6];
    double mixed_policy_values[8];
    int orthant_direct;
    int orthant_roots_lazy;
    double orthant_target;
    const double *orthant_gv_direction;
    /* Original D*dv is retained in bre only until the next block residual. */
    const double *dynamics_cache_direction;
    double reg, initial_reg, rhs_norm[4];
    /* Append optimization metadata so established field offsets stay fixed. */
    int *constraint_group_end;
    int factor_bank_exchanged, aux_factor_live, aux_factor_swappable;
    /* Consumed only with dynamics_cache_direction: 1=row-dot stages,
     * 2=the fully accepted packed forward chain. No iterate is cached. */
    int dynamics_cache_kind;
    /* Optional constant equality-dual operator, owned by this model bank. */
    double *dual_recovery_storage;
    size_t dual_recovery_bytes;
    double dual_recovery_condition;
    int dual_recovery_valid;
    uint64_t dual_recovery_attempts, dual_recovery_accepted;
    /* Exact equality-implied zero-apex SOCs. Public cone slots are retained;
     * only removed original rows receive an owned compact coefficient copy. */
    size_t *constant_soc_offset;
    double *constant_soc_G;
    size_t constant_soc_count, constant_soc_rows;
    size_t constant_soc_map_capacity, constant_soc_G_capacity;
    /* Exact model certificate: every active singleton has |g|=1. */
    int qp_unit_singletons;
};

ICORE_OCP_INLINE size_t constant_soc_offset(const icore_ocp_workspace *w, int k, int cone)
{
    return w->constant_soc_count ? w->constant_soc_offset[(size_t)k * w->nsoc + cone] : SIZE_MAX;
}

ICORE_OCP_INLINE int constant_soc(const icore_ocp_workspace *w, int k, int cone)
{
    return constant_soc_offset(w, k, cone) != SIZE_MAX;
}

static void dense_qp_plan(icore_ocp_workspace *w, int same);
static void form_affine_rhs(icore_ocp_workspace *w);
static void prepare_newton_rhs(icore_ocp_workspace *w);
static void form_reduced_rhs(icore_ocp_workspace *w, const double *bd, const double *bc,
                             int direct_cone_rhs);
static void prepare_main_reduced_rhs(icore_ocp_workspace *w);

static void auxiliary_factor_preserve(icore_ocp_workspace *w);
