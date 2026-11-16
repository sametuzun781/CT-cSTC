/* Unavailable packed backend: preserve the public return/status contract.
 * Private implementation included only by the #else branch in ocp_la.c.
 */

/* ---- Unavailable-backend ABI stubs ---- */

struct icore_ocp_la_workspace {
    int unused;
};

struct icore_ocp_la_qr_workspace {
    int unused;
};

int icore_ocp_la_qr_create(int max_rows, int max_columns, icore_ocp_la_qr_workspace **out)
{
    (void)max_rows;
    (void)max_columns;
    if (!out)
        return ICORE_LA_INVALID;
    *out = NULL;
    return ICORE_LA_UNAVAILABLE;
}

void icore_ocp_la_qr_destroy(icore_ocp_la_qr_workspace *w)
{
    (void)w;
}

size_t icore_ocp_la_qr_bytes(const icore_ocp_la_qr_workspace *w)
{
    (void)w;
    return 0;
}

int icore_ocp_la_qr_upper(icore_ocp_la_qr_workspace *w, int rows, int columns, double *A, size_t stride)
{
    (void)w;
    (void)rows;
    (void)columns;
    (void)A;
    (void)stride;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_mixed_sweeps_available(const icore_ocp_la_workspace *w)
{
    (void)w;
    return 0;
}

int icore_ocp_la_mixed_sweeps_active(const icore_ocp_la_workspace *w)
{
    (void)w;
    return 0;
}

void icore_ocp_la_mixed_sweeps_invalidate(icore_ocp_la_workspace *w)
{
    (void)w;
}

int icore_ocp_la_mixed_sweeps_begin(icore_ocp_la_workspace *w, const double *d, double r, double a, double t)
{
    (void)w;
    (void)d;
    (void)r;
    (void)a;
    (void)t;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_mixed_sweeps_stage(icore_ocp_la_workspace *w, int k, const double *d, double r, double a, double t)
{
    (void)w;
    (void)k;
    (void)d;
    (void)r;
    (void)a;
    (void)t;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_mixed_sweeps_backward(icore_ocp_la_workspace *w, int k, const double *bd, const double *d,
                              const double *pnext, int reuse_pd, double *pd, double *a, double *p,
                              int suffix_ready)
{
    (void)w;
    (void)k;
    (void)bd;
    (void)d;
    (void)pnext;
    (void)reuse_pd;
    (void)pd;
    (void)a;
    (void)p;
    (void)suffix_ready;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_mixed_sweeps_forward(icore_ocp_la_workspace *w, int k, const double *a, const double *x,
                             const double *d, const double *pnext, double *u, double *next,
                             double *dual)
{
    (void)w;
    (void)k;
    (void)a;
    (void)x;
    (void)d;
    (void)pnext;
    (void)u;
    (void)next;
    (void)dual;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_mixed_ready(const icore_ocp_la_workspace *w)
{
    (void)w;
    return 0;
}

int icore_ocp_la_mixed_begin(icore_ocp_la_workspace *w, const double *d, double r, double a, double t)
{
    (void)w;
    (void)d;
    (void)r;
    (void)a;
    (void)t;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_mixed_stage(icore_ocp_la_workspace *w, int k, const double *d, double r, double a, double t)
{
    (void)w;
    (void)k;
    (void)d;
    (void)r;
    (void)a;
    (void)t;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_create(int N, int nx, int nr, icore_ocp_la_workspace **out)
{
    (void)N;
    (void)nx;
    (void)nr;
    if (!out)
        return ICORE_LA_INVALID;
    *out = NULL;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_create_owned(int N, int nx, int nr, icore_ocp_la_workspace **out)
{
    return icore_ocp_la_create(N, nx, nr, out);
}

int icore_ocp_la_sqrt_owns_factors(const icore_ocp_la_workspace *w)
{
    (void)w;
    return 0;
}

int icore_ocp_la_sqrt_initial_zero_active(const icore_ocp_la_workspace *w)
{
    (void)w;
    return 0;
}

int icore_ocp_la_sqrt_initial_zero(icore_ocp_la_workspace *w, const double *H, size_t stride, double reg,
                          double absolute, double relative)
{
    (void)w;
    (void)H;
    (void)stride;
    (void)reg;
    (void)absolute;
    (void)relative;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_sqrt_initial_zero_cached_diagonal(icore_ocp_la_workspace *w, const double *diagonal, double reg,
                                          double absolute, double relative)
{
    (void)w;
    (void)diagonal;
    (void)reg;
    (void)absolute;
    (void)relative;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_initial_zero_sweep(icore_ocp_la_workspace *w, int transpose, const double *q, double *out)
{
    (void)w;
    (void)transpose;
    (void)q;
    (void)out;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_initial_zero_dynamics_product(icore_ocp_la_workspace *w, int transpose, double alpha,
                                      const double *x, double beta, double *y)
{
    (void)w;
    (void)transpose;
    (void)alpha;
    (void)x;
    (void)beta;
    (void)y;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_packed_backward_stage(icore_ocp_la_workspace *w, int stage, const double *bd, const double *d,
                              const double *pnext, int reuse_pd, double *pd, double *a, double *p)
{
    (void)w;
    (void)stage;
    (void)bd;
    (void)d;
    (void)pnext;
    (void)reuse_pd;
    (void)pd;
    (void)a;
    (void)p;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_factor_backward_stage(icore_ocp_la_workspace *w, int stage, const double *bd, const double *d,
                              const double *pnext, double *pd, double *a, double *p)
{
    (void)w;
    (void)stage;
    (void)bd;
    (void)d;
    (void)pnext;
    (void)pd;
    (void)a;
    (void)p;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_packed_forward_stage(icore_ocp_la_workspace *w, int stage, const double *a, const double *x,
                             const double *d, const double *pnext, double *u, double *next,
                             double *dual)
{
    (void)w;
    (void)stage;
    (void)a;
    (void)x;
    (void)d;
    (void)pnext;
    (void)u;
    (void)next;
    (void)dual;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_packed_forward_stage_cached(icore_ocp_la_workspace *w, int stage, const double *a, const double *x,
                                    const double *d, const double *pnext, double *u, double *next,
                                    double *dual, double *unshifted_dynamics)
{
    (void)w;
    (void)stage;
    (void)a;
    (void)x;
    (void)d;
    (void)pnext;
    (void)u;
    (void)next;
    (void)dual;
    (void)unshifted_dynamics;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_dynamics_cache_available(void)
{
    return 0;
}

int icore_ocp_la_dynamics_nt_add(icore_ocp_la_workspace *w, int stage, const double *dual, double *gradient)
{
    (void)w;
    (void)stage;
    (void)dual;
    (void)gradient;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_dynamics_pair(icore_ocp_la_workspace *w, int stage, const double *primal, const double *dual,
                      double *gradient, double *dynamics)
{
    (void)w;
    (void)stage;
    (void)primal;
    (void)dual;
    (void)gradient;
    (void)dynamics;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_sweep_backward(icore_ocp_la_workspace *w, int stage, const double *q_u, const double *q_x, double *a,
                       double *p)
{
    (void)w;
    (void)stage;
    (void)q_u;
    (void)q_x;
    (void)a;
    (void)p;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_sweep_forward(icore_ocp_la_workspace *w, int stage, const double *q_u, const double *x, double *u)
{
    (void)w;
    (void)stage;
    (void)q_u;
    (void)x;
    (void)u;
    return ICORE_LA_UNAVAILABLE;
}

void icore_ocp_la_destroy(icore_ocp_la_workspace *w)
{
    (void)w;
}

int icore_ocp_la_load_dynamics(icore_ocp_la_workspace *w, int stage, const double *D, size_t stride)
{
    (void)w;
    (void)stage;
    (void)D;
    (void)stride;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_load_dynamics_finite(icore_ocp_la_workspace *w, int stage, const double *D, size_t stride)
{
    (void)w;
    (void)stage;
    (void)D;
    (void)stride;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_load_cost(icore_ocp_la_workspace *w, int stage, const double *H, size_t stride)
{
    (void)w;
    (void)stage;
    (void)H;
    (void)stride;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_load_cost_finite(icore_ocp_la_workspace *w, int stage, const double *H, size_t stride)
{
    (void)w;
    (void)stage;
    (void)H;
    (void)stride;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_load_diagonal_cost(icore_ocp_la_workspace *w, int stage, const double *diagonal)
{
    (void)w;
    (void)stage;
    (void)diagonal;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_products(icore_ocp_la_workspace *w, int stage, const double *P_next, size_t ldP, const double *H,
                 size_t ldH, double reg, double *PD, double *M)
{
    (void)w;
    (void)stage;
    (void)P_next;
    (void)ldP;
    (void)H;
    (void)ldH;
    (void)reg;
    (void)PD;
    (void)M;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_factor_stage(icore_ocp_la_workspace *w, int stage, const double *P_next, size_t ldP, const double *H,
                     size_t ldH, double reg, double absolute, double relative, double *L, double *T,
                     double *P)
{
    (void)w;
    (void)stage;
    (void)P_next;
    (void)ldP;
    (void)H;
    (void)ldH;
    (void)reg;
    (void)absolute;
    (void)relative;
    (void)L;
    (void)T;
    (void)P;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_sqrt_begin(icore_ocp_la_workspace *w, const double *H, size_t stride, double reg, double absolute,
                   double relative)
{
    (void)w;
    (void)H;
    (void)stride;
    (void)reg;
    (void)absolute;
    (void)relative;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_sqrt_stage(icore_ocp_la_workspace *w, int stage, const double *H, size_t stride, double reg,
                   double absolute, double relative, double *L, double *T)
{
    (void)w;
    (void)stage;
    (void)H;
    (void)stride;
    (void)reg;
    (void)absolute;
    (void)relative;
    (void)L;
    (void)T;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_sqrt_apply(icore_ocp_la_workspace *w, int stage, const double *x, double *y)
{
    (void)w;
    (void)stage;
    (void)x;
    (void)y;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_sqrt_initial_fixed(icore_ocp_la_workspace *w, const double *H, size_t stride, double reg,
                           double absolute, double relative, double *L, double *T)
{
    (void)w;
    (void)H;
    (void)stride;
    (void)reg;
    (void)absolute;
    (void)relative;
    (void)L;
    (void)T;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_sqrt_equilibrated_stages(const icore_ocp_la_workspace *w)
{
    (void)w;
    return 0;
}

int icore_ocp_la_sqrt_recovery_pending(const icore_ocp_la_workspace *w, int stage, int initial_fixed, double reg)
{
    (void)w;
    (void)stage;
    (void)initial_fixed;
    (void)reg;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_sqrt_recovery_product(icore_ocp_la_workspace *w, int stage, double *RD, size_t row_stride)
{
    (void)w;
    (void)stage;
    (void)RD;
    (void)row_stride;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_sqrt_recovery_accept(icore_ocp_la_workspace *w, int stage, int initial_fixed, double reg,
                             const double *R, size_t row_stride, double *L, double *T)
{
    (void)w;
    (void)stage;
    (void)initial_fixed;
    (void)reg;
    (void)R;
    (void)row_stride;
    (void)L;
    (void)T;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_dynamics_product(icore_ocp_la_workspace *w, int stage, int transpose, double alpha, const double *x,
                         double beta, double *y)
{
    (void)w;
    (void)stage;
    (void)transpose;
    (void)alpha;
    (void)x;
    (void)beta;
    (void)y;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_export_stage_factor(icore_ocp_la_workspace *w, int stage, double *L, double *T)
{
    (void)w;
    (void)stage;
    (void)L;
    (void)T;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_terminal_border(icore_ocp_la_workspace *w, int stage, int nf, const double *Cn, double *C, double *Z,
                        double *S)
{
    (void)w;
    (void)stage;
    (void)nf;
    (void)Cn;
    (void)C;
    (void)Z;
    (void)S;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_sqrt_checkpoint(icore_ocp_la_workspace *w)
{
    (void)w;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_sqrt_restore(icore_ocp_la_workspace *w)
{
    (void)w;
    return ICORE_LA_UNAVAILABLE;
}

void icore_ocp_la_sqrt_discard_checkpoint(icore_ocp_la_workspace *w)
{
    (void)w;
}

size_t icore_ocp_la_sqrt_checkpoint_bytes(const icore_ocp_la_workspace *w)
{
    (void)w;
    return 0;
}

size_t icore_ocp_la_sqrt_checkpoint_copy_bytes(const icore_ocp_la_workspace *w)
{
    (void)w;
    return 0;
}

int icore_ocp_la_sqrt_begin_cached_diagonal(icore_ocp_la_workspace *w, const double *diagonal, double reg,
                                   double absolute, double relative)
{
    (void)w;
    (void)diagonal;
    (void)reg;
    (void)absolute;
    (void)relative;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_sqrt_stage_cached_diagonal(icore_ocp_la_workspace *w, int stage, const double *diagonal, double reg,
                                   double absolute, double relative, double *L, double *T)
{
    (void)w;
    (void)stage;
    (void)diagonal;
    (void)reg;
    (void)absolute;
    (void)relative;
    (void)L;
    (void)T;
    return ICORE_LA_UNAVAILABLE;
}

int icore_ocp_la_sqrt_initial_fixed_cached_diagonal(icore_ocp_la_workspace *w, const double *diagonal, double reg,
                                           double absolute, double relative, double *L, double *T)
{
    (void)w;
    (void)diagonal;
    (void)reg;
    (void)absolute;
    (void)relative;
    (void)L;
    (void)T;
    return ICORE_LA_UNAVAILABLE;
}
