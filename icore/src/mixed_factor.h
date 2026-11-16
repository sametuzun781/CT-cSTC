#ifndef ICORE_MIXED_FACTOR_H
#define ICORE_MIXED_FACTOR_H
#include "ocp_la.h"
#include <stdint.h>
/* Test builds may enable last-solve thread diagnostics. Default builds
 * contain no diagnostic TLS and the optional readers return zero. */
#ifndef ICORE_OCP_MIXED_DIAGNOSTICS
#define ICORE_OCP_MIXED_DIAGNOSTICS 0
#endif
#ifdef __cplusplus
extern "C" {
#endif
int icore_ocp_la_mixed_ready(const icore_ocp_la_workspace *w);
int icore_ocp_la_mixed_begin(icore_ocp_la_workspace *w, const double *diagonal, double reg, double absolute,
                    double relative);
int icore_ocp_la_mixed_stage(icore_ocp_la_workspace *w, int stage, const double *diagonal, double reg,
                    double absolute, double relative);
/* Separate SP-only factor/sweep chain. The double C bank remains invalid. */
int icore_ocp_la_mixed_sweeps_available(const icore_ocp_la_workspace *w);
int icore_ocp_la_mixed_sweeps_active(const icore_ocp_la_workspace *w);
void icore_ocp_la_mixed_sweeps_invalidate(icore_ocp_la_workspace *w);
int icore_ocp_la_mixed_sweeps_begin(icore_ocp_la_workspace *w, const double *diagonal, double reg, double absolute,
                           double relative);
int icore_ocp_la_mixed_sweeps_stage(icore_ocp_la_workspace *w, int stage, const double *diagonal, double reg,
                           double absolute, double relative);
int icore_ocp_la_mixed_sweeps_backward(icore_ocp_la_workspace *w, int stage, const double *bd, const double *d,
                              const double *pnext, int reuse_pd, double *pd, double *a, double *p,
                              int suffix_ready);
int icore_ocp_la_mixed_sweeps_forward(icore_ocp_la_workspace *w, int stage, const double *a, const double *x,
                             const double *d, const double *pnext, double *u, double *next,
                             double *dual);
int icore_mixed_sweeps_read(uint64_t *counts, size_t ncounts);
/* Experimental diagnostic only: the last completed solve on this thread. */
int icore_mixed_read(uint64_t *counts, size_t count, double *last_error, double *limit);
double icore_mixed_max_accepted_error(void);
int icore_mixed_last_factor(int *stage, int *phase, int *status);
/* Experimental inexact-policy counters/values; fixed lengths six/eight. */
int icore_mixed_policy_read(uint64_t *counts, size_t ncounts, double *values, size_t nvalues);
#ifdef __cplusplus
}
#endif
#endif
