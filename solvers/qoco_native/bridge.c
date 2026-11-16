/* Public-API and timing adapter; all optimization is in upstream QOCO. */
#define _POSIX_C_SOURCE 200809L
#include <qoco.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

_Static_assert(sizeof(QOCOFloat) == sizeof(double), "QOCO must use doubles");
_Static_assert(sizeof(QOCOInt) == sizeof(int), "QOCO must use 32-bit integers");

static double seconds(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

void *benchmark_qoco_create(int n, int m, int p,
    int pn, int *pp, int *pi, double *pv, double *c,
    int an, int *ap, int *ai, double *av, double *b,
    int gn, int *gp, int *gi, double *gv, double *h,
    int l, int nsoc, int *cones, double tolerance, int max_iterations, int *error)
{
    QOCOSolver *solver = calloc(1, sizeof(*solver));
    QOCOSettings settings;
    QOCOCscMatrix P, A, G;
    if (!solver) { *error = -1; return NULL; }
    set_default_settings(&settings);
    settings.abstol = settings.abstol_inacc = tolerance;
    settings.reltol = settings.reltol_inacc = 0.0;
    settings.max_iters = max_iterations;
    settings.verbose = 0;
    qoco_set_csc(&P, n, n, pn, pv, pp, pi);
    qoco_set_csc(&A, p, n, an, av, ap, ai);
    qoco_set_csc(&G, m, n, gn, gv, gp, gi);
    *error = qoco_setup(solver, n, m, p, pn ? &P : NULL, c,
        p ? &A : NULL, p ? b : NULL, m ? &G : NULL, m ? h : NULL,
        l, nsoc, nsoc ? cones : NULL, &settings);
    if (*error) { free(solver); return NULL; }
    return solver;
}

void benchmark_qoco_solve(void *handle, double *x, double *y, double *s,
                          double *z, double *timing, int *info)
{
    QOCOSolver *solver = handle;
    QOCOProblemData *data = solver->work->data;
    double start = seconds();
    /* Upstream qoco_solve initializes its IPM iterates on every call. */
    info[3] = qoco_solve(solver);
    timing[0] = seconds() - start;
    timing[1] = solver->sol->solve_time_sec;
    info[0] = solver->sol->status;
    info[1] = solver->sol->iters;
    info[2] = solver->sol->ir_iters;
    memcpy(x, solver->sol->x, (size_t)data->n * sizeof(double));
    if (data->p) memcpy(y, solver->sol->y, (size_t)data->p * sizeof(double));
    if (data->m) {
        memcpy(s, solver->sol->s, (size_t)data->m * sizeof(double));
        memcpy(z, solver->sol->z, (size_t)data->m * sizeof(double));
    }
}

void benchmark_qoco_destroy(void *handle)
{
    /* QOCO cleanup owns and frees the outer solver allocation as well. */
    if (handle) qoco_cleanup(handle);
}

void benchmark_qoco_update(void *handle, double *pv, double *c,
                           double *av, double *b, double *gv, double *h)
{
    QOCOSolver *solver = handle;
    qoco_update_matrix_data(solver, pv, av, gv);
    qoco_update_vector_data(solver, c, b, h);
}
