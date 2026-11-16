/* Portable mixed-precision sweep fallback.
 *
 * Included by mixed_sweep_impl.h when ICORE_OCP_LA_MIXED_SUPPORTED is 0,
 * including the BLASFEO GENERIC target selected on Apple Silicon.
 * The optional FP32 sweep path reports unavailable; the caller retains
 * the existing FP64 factorization and solve paths. No buffer is modified.
 *
 * These definitions preserve the same unavailable-backend contract as
 * ocp_la_stubs.h. This is a private implementation fragment, not a public
 * header, and is included only from ocp_la.c.
 */

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
