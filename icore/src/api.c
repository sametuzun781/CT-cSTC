#include "icore.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct icore_workspace {
    int format, K, n, d, p, l, step_policy;
    icore_ocp_workspace *ocp;
    chain_workspace *chain;
};

icore_options icore_default_options(void)
{
    return icore_ocp_default_options();
}

int icore_dense_face_attempts(const icore_workspace *w)
{
    return w && w->format == ICORE_STAGE_OCP ? icore_ocp_dense_face_attempts(w->ocp) : 0;
}

int icore_dense_face_steps(const icore_workspace *w)
{
    return w && w->format == ICORE_STAGE_OCP ? icore_ocp_dense_face_steps(w->ocp) : 0;
}

int icore_dense_face_accepted(const icore_workspace *w)
{
    return w && w->format == ICORE_STAGE_OCP ? icore_ocp_dense_face_accepted(w->ocp) : 0;
}

int icore_last_kernel(const icore_workspace *w)
{
    if (!w)
        return -1;
    return w->format == ICORE_STAGE_OCP ? icore_ocp_last_kernel(w->ocp) : 2;
}

int icore_dense_fallbacks(const icore_workspace *w)
{
    return w && w->format == ICORE_STAGE_OCP ? icore_ocp_dense_fallbacks(w->ocp) : 0;
}

int icore_dense_packed(const icore_workspace *w)
{
    return w && w->format == ICORE_STAGE_OCP ? icore_ocp_dense_packed(w->ocp) : 0;
}

int icore_ocp_packed_stages(const icore_workspace *w)
{
    return w && w->format == ICORE_STAGE_OCP ? icore_ocp_ocp_packed_stages(w->ocp) : 0;
}

int icore_ocp_packed_rejected(const icore_workspace *w)
{
    return w && w->format == ICORE_STAGE_OCP ? icore_ocp_ocp_packed_rejected(w->ocp) : 0;
}

int icore_ocp_sqrt_factors(const icore_workspace *w)
{
    return w && w->format == ICORE_STAGE_OCP ? icore_ocp_ocp_sqrt_factors(w->ocp) : 0;
}

int icore_ocp_sqrt_rejected(const icore_workspace *w)
{
    return w && w->format == ICORE_STAGE_OCP ? icore_ocp_ocp_sqrt_rejected(w->ocp) : 0;
}

int icore_ocp_sqrt_equilibrated(const icore_workspace *w)
{
    return w && w->format == ICORE_STAGE_OCP ? icore_ocp_ocp_sqrt_equilibrated(w->ocp) : 0;
}

void icore_destroy(icore_workspace *w)
{
    if (!w)
        return;
    icore_ocp_destroy(w->ocp);
    chain_destroy(w->chain);
    free(w);
}

icore_workspace *icore_create(const icore_model *model)
{
    icore_workspace *w;
    if (!model || !model->data ||
        (model->format != ICORE_STAGE_OCP && model->format != ICORE_AFFINE_CHAIN))
        return NULL;
    w = (icore_workspace *)calloc(1, sizeof(*w));
    if (!w)
        return NULL;
    w->format = model->format;
    w->step_policy = ICORE_OCP_STEP_AUTO;
    if (w->format == ICORE_STAGE_OCP) {
        w->ocp = icore_ocp_create((const icore_ocp_model *)model->data);
        if (!w->ocp) {
            icore_destroy(w);
            return NULL;
        }
    } else {
        w->K = model->K;
        w->n = model->n;
        w->d = model->d;
        w->p = model->p;
        w->l = model->l;
        w->chain = chain_create(w->K, w->n, w->d, w->p, w->l);
        if (!w->chain || chain_update(w->chain, (const chain_problem *)model->data)) {
            icore_destroy(w);
            return NULL;
        }
    }
    return w;
}

int icore_set_step_policy(icore_workspace *w, int step_policy)
{
    if (!w || step_policy < ICORE_OCP_STEP_AUTO || step_policy > ICORE_OCP_STEP_ADAPTIVE)
        return ICORE_OCP_INVALID_MODEL;
    w->step_policy = step_policy;
    return ICORE_OCP_SOLVED;
}

int icore_update(icore_workspace *w, const icore_model *model)
{
    if (!w || !model || !model->data || model->format != w->format)
        return ICORE_OCP_INVALID_MODEL;
    if (w->format == ICORE_STAGE_OCP)
        return icore_ocp_update(w->ocp, (const icore_ocp_model *)model->data);
    if (model->K != w->K || model->n != w->n || model->d != w->d || model->p != w->p ||
        model->l != w->l)
        return ICORE_OCP_INVALID_MODEL;
    return chain_update(w->chain, (const chain_problem *)model->data);
}

int icore_solve(icore_workspace *w, const icore_options *options, const icore_output *out,
                icore_info *information)
{
    icore_info local, *info = information ? information : &local;
    if (!w || !out) {
        memset(info, 0, sizeof(*info));
        info->status = ICORE_OCP_INVALID_MODEL;
        return info->status;
    }
    if (w->format == ICORE_STAGE_OCP)
        return icore_ocp_solve_policy(w->ocp, options, w->step_policy, out->v, out->y, out->nu, out->s,
                                 out->z, info);
    else {
        chain_info native;
        clock_t begin = clock();
        int status = chain_solve_options_policy(w->chain, options, w->step_policy, out->v, out->t,
                                                out->s, out->z, &native);
        memset(info, 0, sizeof(*info));
        info->status = status;
        info->iterations = native.iterations;
        info->refinements = native.refinements;
        info->factor_retries = native.qr_fallbacks;
        info->objective = native.objective;
        info->primal_residual = native.primal;
        info->dual_residual = native.dual;
        info->gap = native.gap;
        info->seconds = (double)(clock() - begin) / (double)CLOCKS_PER_SEC;
        return status;
    }
}
