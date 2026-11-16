/* Native, allocation-accounted HPIPM adapter. Upstream HPIPM is unmodified.
 * Original OCP convention: Ax + Bu + b - x_next = 0, G[u;x] + s = h.
 * Matrices passed here are row-major; symmetric costs use their upper triangle.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <blasfeo_d_aux.h>
#include <hpipm_common.h>
#include <hpipm_d_ocp_qp_dim.h>
#include <hpipm_d_ocp_qp.h>
#include <hpipm_d_ocp_qp_sol.h>
#include <hpipm_d_ocp_qp_ipm.h>
#include <hpipm_d_dense_qp_dim.h>
#include <hpipm_d_dense_qp.h>
#include <hpipm_d_dense_qp_sol.h>
#include <hpipm_d_dense_qp_ipm.h>

typedef struct {
    int N, nx, nu, m, stages, nf;
    const double *Q, *R, *S, *q, *r, *A, *B, *b, *G, *h, *x0, *F, *f;
} ocp_data;

typedef struct {
    int n, m, ne;
    const double *P, *q, *G, *h, *A, *b;
} dense_data;

typedef struct {
    int status, iterations, stat_m, native_bound_count;
    int actual_ric_alg, actual_lq_fact, npd_hess, npd_reg_hess;
    int box_variables, box_constraint_rows, general_constraint_rows, terminal_equality_rows;
    int inactive_constraint_rows;
    uint64_t memory_bytes;
    double setup_s, update_s, solve_s, extract_s;
    double residual_stat, residual_eq, residual_ineq, residual_comp;
    double dual_gap, objective, tolerance, tol_comp;
} hp_metrics;

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}
static size_t maxz(size_t a, size_t b) { return a > b ? a : b; }
static void *allocate(size_t bytes, uint64_t *total) {
    if (bytes == 0) bytes = 64;
    void *p = calloc(1, bytes);
    if (p) *total += bytes;
    return p;
}
static void colmaj(double *dst, const double *src, int rows, int cols) {
    for (int i=0; i<rows; ++i)
        for (int j=0; j<cols; ++j) dst[i+j*rows] = src[i*cols+j];
}
static void symmetric_upper(double *dst, const double *src, int n) {
    for (int i=0; i<n; ++i)
        for (int j=0; j<n; ++j) dst[i+j*n] = i<=j ? src[i*n+j] : src[j*n+i];
}

/* Identify a true single-variable row after fixed coordinates are eliminated.
 * No threshold, approximate sparsification, or outcome-based classification. */
static int single_variable(const double *row,int offset,int count,double *coefficient) {
    int column=-1;
    for (int j=0;j<count;++j) if (row[offset+j]!=0.) {
        if (column>=0) return -2;
        column=j;*coefficient=row[offset+j];
    }
    return column;
}

/* Duplicate same-side bounds stay general, preserving every original row.
 * Opposite sides can share a native box variable; no bound is discarded. */
static void classify_bounds(const double *G,const double *rhs,int m,int width,int offset,int nv,
    int *kind,int *index,int *columns,int *lower_row,int *upper_row,int *scratch,
    int *nb,int *ng) {
    int *positive=scratch,*negative=scratch+nv,*map=scratch+2*nv;
    memset(scratch,0,(size_t)3*nv*sizeof(int));
    for (int j=0;j<nv;++j) {map[j]=-1;lower_row[j]=-1;upper_row[j]=-1;}
    for (int row=0;row<m;++row) {
        double coefficient=0.;int col=single_variable(G+(size_t)row*width,offset,nv,&coefficient);
        if (col==-1 && rhs[row]>=0.) {kind[row]=3;index[row]=-1;continue;}
        kind[row]=col<0 ? 0 : (coefficient>0. ? 1 : 2);index[row]=col;
        if (col>=0) {if (coefficient>0.) positive[col]++;else negative[col]++;}
    }
    *nb=0;*ng=0;
    for (int j=0;j<nv;++j) if (positive[j]<=1 && negative[j]<=1 && positive[j]+negative[j]>0) {
        map[j]=*nb;columns[(*nb)++]=j;
    }
    for (int row=0;row<m;++row) {
        if (kind[row]==3) continue;
        if (kind[row] && map[index[row]]>=0) {
            index[row]=map[index[row]];
            if (kind[row]==1) upper_row[index[row]]=row;else lower_row[index[row]]=row;
        } else {kind[row]=0;index[row]=(*ng)++;}
    }
}

typedef struct {
    int N, nx, nu, m, stages, nf;
    struct d_ocp_qp_dim dim;
    struct d_ocp_qp qp;
    struct d_ocp_qp_sol sol;
    struct d_ocp_qp_ipm_arg arg;
    struct d_ocp_qp_ipm_ws ws;
    void *dim_mem, *qp_mem, *sol_mem, *arg_mem, *ws_mem;
    double *tmp, *vec, *lower, *mask, *x0, *lam_lower, *lam_upper, *row_scale, *row_constant;
    int *row_kind,*row_index,*box_columns,*box_lower_row,*box_upper_row,*scratch_i;
    hp_metrics metrics;
} ocp_handle;

void hp_ocp_destroy(void *ptr) {
    ocp_handle *h = (ocp_handle *)ptr;
    if (!h) return;
    free(h->dim_mem); free(h->qp_mem); free(h->sol_mem);
    free(h->arg_mem); free(h->ws_mem);
    free(h->tmp); free(h->vec); free(h->lower); free(h->mask);
    free(h->x0); free(h->lam_lower); free(h->lam_upper);free(h->row_scale);free(h->row_constant);
    free(h->row_kind);free(h->row_index);free(h->box_columns);
    free(h->box_lower_row);free(h->box_upper_row);free(h->scratch_i);free(h);
}

int hp_ocp_update(void *ptr, const ocp_data *p) {
    ocp_handle *h = (ocp_handle *)ptr;
    if (!h || !p || h->N!=p->N || h->nx!=p->nx || h->nu!=p->nu ||
        h->m!=p->m || h->stages!=p->stages || h->nf!=p->nf) return -1;
    const double t = seconds();
    const int N=h->N, nx=h->nx, nu=h->nu, m=h->m, nf=h->nf;
    /* Validate extraction structure before touching any retained model data. */
    for (int k=0;k<h->stages;++k) for (int i=0;i<m;++i) {
        int row=k*m+i,kind=h->row_kind[row];
        if (!kind) continue;
        int offset=k==N ? nu : 0,nv=(k==0 ? nu : (k==N ? nx : nu+nx));
        double coefficient=0.;
        int col=single_variable(p->G+(size_t)row*(nu+nx),offset,nv,&coefficient);
        if (kind==3) {
            double rhs=p->h[row];
            if (k==0) for (int c=0;c<nx;++c) rhs-=p->G[(size_t)i*(nu+nx)+nu+c]*p->x0[c];
            if (col!=-1 || !(rhs>=0.) || !isfinite(rhs)) return -3;
            continue;
        }
        if (col!=h->box_columns[k*(nu+nx)+h->row_index[row]] ||
            (kind==1 ? coefficient<=0. : coefficient>=0.)) return -2;
    }
    memcpy(h->x0,p->x0,(size_t)nx*sizeof(double));
    for (int k=0;k<h->stages;++k) for (int i=0;i<m;++i) if (h->row_kind[k*m+i]==3) {
        double rhs=p->h[k*m+i];
        if (k==0) for (int c=0;c<nx;++c) rhs-=p->G[(size_t)i*(nu+nx)+nu+c]*p->x0[c];
        h->row_constant[k*m+i]=rhs;
    }
    for (int k=0; k<=N; ++k) {
        if (k>0) {
            symmetric_upper(h->tmp,p->Q+(size_t)k*nx*nx,nx);
            d_ocp_qp_set_Q(k,h->tmp,&h->qp);
            d_ocp_qp_set_q(k,(double *)p->q+(size_t)k*nx,&h->qp);
        }
        if (k<N) {
            symmetric_upper(h->tmp,p->R+(size_t)k*nu*nu,nu);
            d_ocp_qp_set_R(k,h->tmp,&h->qp);
            if (k) {
                colmaj(h->tmp,p->S+(size_t)k*nu*nx,nu,nx);
                d_ocp_qp_set_S(k,h->tmp,&h->qp);
                d_ocp_qp_set_r(k,(double *)p->r+(size_t)k*nu,&h->qp);
                colmaj(h->tmp,p->A+(size_t)k*nx*nx,nx,nx);
                d_ocp_qp_set_A(k,h->tmp,&h->qp);
                d_ocp_qp_set_b(k,(double *)p->b+(size_t)k*nx,&h->qp);
            } else {
                for (int i=0; i<nu; ++i) {
                    double v=p->r[i];
                    for (int j=0; j<nx; ++j) v+=p->S[i*nx+j]*p->x0[j];
                    h->vec[i]=v;
                }
                d_ocp_qp_set_r(0,h->vec,&h->qp);
                for (int i=0; i<nx; ++i) {
                    double v=p->b[i];
                    for (int j=0; j<nx; ++j) v+=p->A[i*nx+j]*p->x0[j];
                    h->vec[i]=v;
                }
                d_ocp_qp_set_b(0,h->vec,&h->qp);
            }
            colmaj(h->tmp,p->B+(size_t)k*nx*nu,nx,nu);
            d_ocp_qp_set_B(k,h->tmp,&h->qp);
        }
        const int mi=k<h->stages ? m : 0;
        const int nb=h->dim.nb[k],ng=h->dim.ng[k],original_ng=ng-(k==N ? nf : 0);
        int *kind=h->row_kind+(size_t)k*m,*index=h->row_index+(size_t)k*m;
        /* Reconstruct native box bounds from original row RHS and coefficient.
           Missing sides are masked, never replaced with a large number. */
        if (nb) {
            int *columns=h->box_columns+k*(nu+nx);
            int *lr=h->box_lower_row+k*(nu+nx),*ur=h->box_upper_row+k*(nu+nx);
            d_ocp_qp_set_idxb(k,columns,&h->qp);
            for (int side=0;side<2;++side) {
                for (int j=0;j<nb;++j) {
                    int i=side ? ur[j] : lr[j];
                    h->mask[j]=i>=0 ? 1. : 0.;h->vec[j]=0.;
                    if (i<0) continue;
                    double rhs=p->h[k*m+i];
                    if (k==0) for (int c=0;c<nx;++c) rhs-=p->G[i*(nu+nx)+nu+c]*p->x0[c];
                    int offset=k==N ? nu : 0;
                    double coefficient=p->G[((size_t)k*m+i)*(nu+nx)+offset+columns[j]];
                    h->vec[j]=rhs/coefficient;h->row_scale[k*m+i]=fabs(coefficient);
                }
                if (side) {d_ocp_qp_set_ub(k,h->vec,&h->qp);d_ocp_qp_set_ub_mask(k,h->mask,&h->qp);}
                else {d_ocp_qp_set_lb(k,h->vec,&h->qp);d_ocp_qp_set_lb_mask(k,h->mask,&h->qp);}
            }
        }
        if (!ng) continue;
        if (k>0) {
            for (int j=0; j<nx; ++j) {
                for (int i=0; i<mi; ++i) if (!kind[i])
                    h->tmp[index[i]+j*ng]=p->G[((size_t)k*m+i)*(nx+nu)+nu+j];
                if (k==N) for (int i=0; i<nf; ++i) h->tmp[original_ng+i+j*ng]=p->F[i*nx+j];
            }
            d_ocp_qp_set_C(k,h->tmp,&h->qp);
        }
        if (k<N) {
            for (int j=0; j<nu; ++j)
                for (int i=0; i<mi; ++i) if (!kind[i])
                    h->tmp[index[i]+j*ng]=p->G[((size_t)k*m+i)*(nx+nu)+j];
            d_ocp_qp_set_D(k,h->tmp,&h->qp);
        }
        for (int i=0; i<mi; ++i) if (!kind[i]) {
            double v=p->h[k*m+i];
            if (k==0) for (int j=0; j<nx; ++j)
                v-=p->G[i*(nx+nu)+nu+j]*p->x0[j];
            h->vec[index[i]]=v; h->lower[index[i]]=0.; h->mask[index[i]]=0.;
        }
        if (k==N) for (int i=0; i<nf; ++i) {
            h->vec[original_ng+i]=p->f[i]; h->lower[original_ng+i]=p->f[i]; h->mask[original_ng+i]=1.;
        }
        /* Exact equality F*x=f is represented by equal finite lower/upper
           bounds. Its signed multiplier is lam_upper-lam_lower. */
        d_ocp_qp_set_ug(k,h->vec,&h->qp);
        d_ocp_qp_set_lg(k,h->lower,&h->qp);
        d_ocp_qp_set_lg_mask(k,h->mask,&h->qp);
        for (int i=0; i<ng; ++i) h->mask[i]=1.;
        d_ocp_qp_set_ug_mask(k,h->mask,&h->qp);
    }
    h->metrics.update_s=seconds()-t;
    return 0;
}

void *hp_ocp_create(const ocp_data *p, int profile, double tolerance, int max_iter) {
    if (!p || p->N<1 || p->nx<1 || p->nu<1 || p->m<0 || p->nf<0 ||
        (p->stages!=p->N && p->stages!=p->N+1) ||
        (profile!=BALANCE && profile!=ROBUST) || !(tolerance>0) || max_iter<1) return NULL;
    const double t=seconds();
    ocp_handle *h=calloc(1,sizeof(*h));
    if (!h) return NULL;
    h->metrics.memory_bytes=sizeof(*h);
    h->N=p->N;h->nx=p->nx;h->nu=p->nu;h->m=p->m;h->stages=p->stages;h->nf=p->nf;
    uint64_t *bytes=&h->metrics.memory_bytes;
    const size_t row_count=(size_t)(p->N+1)*p->m,box_slots=(size_t)(p->N+1)*(p->nu+p->nx);
    h->row_kind=allocate(row_count*sizeof(int),bytes);
    h->row_index=allocate(row_count*sizeof(int),bytes);
    h->row_scale=allocate(row_count*sizeof(double),bytes);
    h->row_constant=allocate(row_count*sizeof(double),bytes);
    h->box_columns=allocate(box_slots*sizeof(int),bytes);
    h->box_lower_row=allocate(box_slots*sizeof(int),bytes);
    h->box_upper_row=allocate(box_slots*sizeof(int),bytes);
    h->scratch_i=allocate((size_t)3*(p->nu+p->nx)*sizeof(int),bytes);
    if (!h->row_kind || !h->row_index || !h->row_scale || !h->row_constant || !h->box_columns ||
        !h->box_lower_row || !h->box_upper_row || !h->scratch_i) goto fail;
    h->dim_mem=allocate(d_ocp_qp_dim_memsize(p->N),bytes);
    if (!h->dim_mem) goto fail;
    d_ocp_qp_dim_create(p->N,&h->dim,h->dim_mem);
    for (int k=0;k<=p->N;++k) {
        int local_nx=k ? p->nx : 0,local_nu=k<p->N ? p->nu : 0;
        int nb=0,ng=0,nbu=0,width=p->nu+p->nx;
        int rows=k<p->stages ? p->m : 0;
        for (int i=0;i<rows;++i) {
            double rhs=p->h[k*p->m+i];
            if (k==0) for (int c=0;c<p->nx;++c) rhs-=p->G[(size_t)i*width+p->nu+c]*p->x0[c];
            if (!isfinite(rhs)) goto fail;
            h->row_constant[k*p->m+i]=rhs;
        }
        classify_bounds(p->G+(size_t)k*p->m*width,h->row_constant+(size_t)k*p->m,rows,width,k==p->N ? p->nu : 0,
            local_nx+local_nu,h->row_kind+(size_t)k*p->m,h->row_index+(size_t)k*p->m,
            h->box_columns+k*width,h->box_lower_row+k*width,h->box_upper_row+k*width,
            h->scratch_i,&nb,&ng);
        for (int j=0;j<nb;++j) if (h->box_columns[k*width+j]<local_nu) nbu++;
        d_ocp_qp_dim_set_nx(k,local_nx,&h->dim);
        d_ocp_qp_dim_set_nu(k,local_nu,&h->dim);
        d_ocp_qp_dim_set_nbu(k,nbu,&h->dim);
        d_ocp_qp_dim_set_nbx(k,nb-nbu,&h->dim);
        d_ocp_qp_dim_set_ng(k,ng+(k==p->N ? p->nf : 0),&h->dim);
        int inactive=0;
        for (int i=0;i<rows;++i) if (h->row_kind[k*p->m+i]==3) inactive++;
        h->metrics.box_variables+=nb;
        h->metrics.box_constraint_rows+=rows-ng-inactive;
        h->metrics.general_constraint_rows+=ng;
        h->metrics.inactive_constraint_rows+=inactive;
    }
    h->metrics.terminal_equality_rows=p->nf;
    h->qp_mem=allocate(d_ocp_qp_memsize(&h->dim),bytes);
    h->sol_mem=allocate(d_ocp_qp_sol_memsize(&h->dim),bytes);
    h->arg_mem=allocate(d_ocp_qp_ipm_arg_memsize(&h->dim),bytes);
    if (!h->qp_mem || !h->sol_mem || !h->arg_mem) goto fail;
    d_ocp_qp_create(&h->dim,&h->qp,h->qp_mem);
    d_ocp_qp_set_all_zero(&h->qp);
    d_ocp_qp_sol_create(&h->dim,&h->sol,h->sol_mem);
    d_ocp_qp_ipm_arg_create(&h->dim,&h->arg,h->arg_mem);
    d_ocp_qp_ipm_arg_set_default(profile,&h->arg);
    h->metrics.native_bound_count=p->stages*p->m-h->metrics.inactive_constraint_rows+2*p->nf;
    h->metrics.tolerance=tolerance;
    double comp=tolerance/(h->metrics.native_bound_count>0 ? h->metrics.native_bound_count : 1);
    h->metrics.tol_comp=comp;
    d_ocp_qp_ipm_arg_set_tol_stat(&tolerance,&h->arg);
    d_ocp_qp_ipm_arg_set_tol_eq(&tolerance,&h->arg);
    d_ocp_qp_ipm_arg_set_tol_ineq(&tolerance,&h->arg);
    d_ocp_qp_ipm_arg_set_tol_comp(&comp,&h->arg);
    d_ocp_qp_ipm_arg_set_tol_dual_gap(&tolerance,&h->arg);
    d_ocp_qp_ipm_arg_set_iter_max(&max_iter,&h->arg);
    h->arg.stat_max=max_iter;
    h->arg.warm_start=0;
    h->ws_mem=allocate(d_ocp_qp_ipm_ws_memsize(&h->dim,&h->arg),bytes);
    if (!h->ws_mem) goto fail;
    d_ocp_qp_ipm_ws_create(&h->dim,&h->arg,&h->ws,h->ws_mem);
    const size_t ng=(size_t)p->m+p->nf;
    const size_t v=maxz(maxz((size_t)p->nx,(size_t)p->nu),ng)+1;
    const size_t nn=maxz((size_t)p->nx,(size_t)p->nu);
    h->tmp=allocate(maxz(nn*nn,ng*nn)*sizeof(double)+64,bytes);
    h->vec=allocate(v*sizeof(double),bytes);
    h->lower=allocate(v*sizeof(double),bytes);
    h->mask=allocate(v*sizeof(double),bytes);
    h->x0=allocate((size_t)p->nx*sizeof(double),bytes);
    h->lam_lower=allocate(v*sizeof(double),bytes);
    h->lam_upper=allocate(v*sizeof(double),bytes);
    if (!h->tmp || !h->vec || !h->lower || !h->mask || !h->x0 ||
        !h->lam_lower || !h->lam_upper) goto fail;
    if (hp_ocp_update(h,p)) goto fail;
    h->metrics.setup_s=seconds()-t;
    return h;
fail:
    hp_ocp_destroy(h);
    return NULL;
}

int hp_ocp_solve(void *ptr,double *x,double *u,double *y,double *nu,double *s,double *z,hp_metrics *out) {
    ocp_handle *h=(ocp_handle *)ptr;
    if (!h) return -1;
    double t=seconds();
    d_ocp_qp_sol_set_zero(&h->sol);
    d_ocp_qp_ipm_solve(&h->qp,&h->sol,&h->arg,&h->ws);
    h->metrics.solve_s=seconds()-t;
    t=seconds();
    d_ocp_qp_ipm_get_status(&h->ws,&h->metrics.status);
    d_ocp_qp_ipm_get_iter(&h->ws,&h->metrics.iterations);
    d_ocp_qp_ipm_get_stat_m(&h->ws,&h->metrics.stat_m);
    d_ocp_qp_ipm_get_max_res_stat(&h->ws,&h->metrics.residual_stat);
    d_ocp_qp_ipm_get_max_res_eq(&h->ws,&h->metrics.residual_eq);
    d_ocp_qp_ipm_get_max_res_ineq(&h->ws,&h->metrics.residual_ineq);
    d_ocp_qp_ipm_get_max_res_comp(&h->ws,&h->metrics.residual_comp);
    d_ocp_qp_ipm_get_dual_gap(&h->ws,&h->metrics.dual_gap);
    d_ocp_qp_ipm_get_obj(&h->ws,&h->metrics.objective);
    h->metrics.actual_ric_alg=h->ws.last_square_root_alg;
    h->metrics.actual_lq_fact=h->ws.last_lq_fact;
    h->metrics.npd_hess=h->ws.npd_hess;
    h->metrics.npd_reg_hess=h->ws.npd_reg_hess;
    memcpy(x,h->x0,(size_t)h->nx*sizeof(double));
    for (int k=0;k<=h->N;++k) {
        if (k) d_ocp_qp_sol_get_x(k,&h->sol,x+(size_t)k*h->nx);
        if (k<h->N) {
            d_ocp_qp_sol_get_u(k,&h->sol,u+(size_t)k*h->nu);
            d_ocp_qp_sol_get_pi(k,&h->sol,y+(size_t)k*h->nx);
        }
        const int mi=k<h->stages ? h->m : 0;
        const int nb=h->dim.nb[k],ng=h->dim.ng[k];
        for (int i=0;i<mi;++i) {
            const int row=k*h->m+i,kind=h->row_kind[row],index=h->row_index[row];
            if (kind==3) {s[row]=h->row_constant[row];z[row]=0.;continue;}
            int native_index=kind==2 ? index : (kind==1 ? nb+ng+index : 2*nb+ng+index);
            double scale=kind ? h->row_scale[row] : 1.;
            /* Preserve actual native slack/dual with exact scalar inverse map. */
            s[row]=h->sol.t[k].pa[native_index]*scale;
            z[row]=h->sol.lam[k].pa[native_index]/scale;
        }
        if (k==h->N && h->nf) {
            int original_ng=ng-h->nf;
            for (int j=0;j<h->nf;++j)
                nu[j]=h->sol.lam[k].pa[2*nb+ng+original_ng+j]-h->sol.lam[k].pa[nb+original_ng+j];
        }
    }
    h->metrics.extract_s=seconds()-t;
    if (out) *out=h->metrics;
    return h->metrics.status;
}

int hp_ocp_history(void *ptr,double *out,int capacity) {
    ocp_handle *h=(ocp_handle *)ptr;
    if (!h) return -1;
    int count=(h->metrics.iterations+1)*h->ws.stat_m;
    if (!out || capacity<count) return count;
    memcpy(out,h->ws.stat,(size_t)count*sizeof(double));
    return count;
}
void hp_ocp_metrics(void *ptr,hp_metrics *out) {
    if (ptr && out) *out=((ocp_handle *)ptr)->metrics;
}
int hp_ocp_complementarity_budget(void *ptr,int component) {
    ocp_handle *h=ptr;if (!h || (component!=0 && component!=1)) return -1;
    const double start=seconds();
    double divisor=component ? 1. : (h->metrics.native_bound_count>0 ? h->metrics.native_bound_count : 1);
    double tol=h->metrics.tolerance/divisor;
    d_ocp_qp_ipm_arg_set_tol_comp(&tol,&h->arg);h->metrics.tol_comp=tol;
    h->metrics.setup_s+=seconds()-start;
    return 0;
}

/* Query actual effective native settings, rather than a Python shadow copy. */
void hp_ocp_settings(void *ptr,double *d,int *i) {
    ocp_handle *h=(ocp_handle *)ptr;
    struct d_ocp_qp_ipm_arg *a=&h->arg;
    double v[]={a->res_g_max,a->res_b_max,a->res_d_max,a->res_m_max,a->dual_gap_max,
        a->mu0,a->alpha_min,a->reg_prim,0.,a->lam_min,a->t_min,a->tau_min,a->lam0_min,a->t0_min,a->m_safe};
    int w[]={a->mode,a->iter_max,a->stat_max,a->warm_start,a->square_root_alg,a->lq_fact,
        a->itref_pred_max,a->itref_corr_max,a->pred_corr,a->cond_pred_corr,a->abs_form,
        a->comp_dual_sol_eq,a->comp_res_exit,a->comp_res_pred,a->split_step,
        a->var_init_scheme,a->t_lam_min,a->t0_init,a->update_fact_exit,-1,-1};
    memcpy(d,v,sizeof(v));memcpy(i,w,sizeof(w));
}

typedef struct {
    int n,m,ne;
    struct d_dense_qp_dim dim;
    struct d_dense_qp qp;
    struct d_dense_qp_sol sol;
    struct d_dense_qp_ipm_arg arg;
    struct d_dense_qp_ipm_ws ws;
    void *dim_mem,*qp_mem,*sol_mem,*arg_mem,*ws_mem;
    double *tmp,*zero,*one,*mask,*row_scale,*row_constant;
    int *row_kind,*row_index,*box_columns,*box_lower_row,*box_upper_row,*scratch_i;
    hp_metrics metrics;
} dense_handle;

void hp_dense_destroy(void *ptr) {
    dense_handle *h=ptr;if (!h) return;
    free(h->dim_mem);free(h->qp_mem);free(h->sol_mem);free(h->arg_mem);free(h->ws_mem);
    free(h->tmp);free(h->zero);free(h->one);free(h->mask);free(h->row_scale);free(h->row_constant);
    free(h->row_kind);free(h->row_index);free(h->box_columns);
    free(h->box_lower_row);free(h->box_upper_row);free(h->scratch_i);free(h);
}
int hp_dense_update(void *ptr,const dense_data *p) {
    dense_handle *h=ptr;
    if (!h || h->n!=p->n || h->m!=p->m || h->ne!=p->ne) return -1;
    const double t=seconds();
    for (int row=0;row<h->m;++row) if (h->row_kind[row]) {
        double coefficient=0.;
        int col=single_variable(p->G+(size_t)row*h->n,0,h->n,&coefficient);
        if (h->row_kind[row]==3) {
            if (col!=-1 || !(p->h[row]>=0.)) return -3;
            continue;
        }
        if (col!=h->box_columns[h->row_index[row]] ||
            (h->row_kind[row]==1 ? coefficient<=0. : coefficient>=0.)) return -2;
    }
    for (int row=0;row<h->m;++row) if (h->row_kind[row]==3) h->row_constant[row]=p->h[row];
    symmetric_upper(h->tmp,p->P,h->n);d_dense_qp_set_H(h->tmp,&h->qp);
    d_dense_qp_set_g((double *)p->q,&h->qp);
    if (h->ne) {
        colmaj(h->tmp,p->A,h->ne,h->n);d_dense_qp_set_A(h->tmp,&h->qp);
        d_dense_qp_set_b((double *)p->b,&h->qp);
    }
    const int nb=h->dim.nb,ng=h->dim.ng;
    if (nb) {
        d_dense_qp_set_idxb(h->box_columns,&h->qp);
        for (int side=0;side<2;++side) {
            for (int j=0;j<nb;++j) {
                int row=side ? h->box_upper_row[j] : h->box_lower_row[j];
                h->mask[j]=row>=0 ? 1. : 0.;h->tmp[j]=0.;
                if (row<0) continue;
                double coefficient=p->G[(size_t)row*h->n+h->box_columns[j]];
                h->tmp[j]=p->h[row]/coefficient;h->row_scale[row]=fabs(coefficient);
            }
            if (side) {d_dense_qp_set_ub(h->tmp,&h->qp);d_dense_qp_set_ub_mask(h->mask,&h->qp);}
            else {d_dense_qp_set_lb(h->tmp,&h->qp);d_dense_qp_set_lb_mask(h->mask,&h->qp);}
        }
    }
    if (ng) {
        for (int j=0;j<h->n;++j) for (int row=0;row<h->m;++row) if (!h->row_kind[row])
            h->tmp[h->row_index[row]+j*ng]=p->G[(size_t)row*h->n+j];
        d_dense_qp_set_C(h->tmp,&h->qp);
        for (int row=0;row<h->m;++row) if (!h->row_kind[row]) h->tmp[h->row_index[row]]=p->h[row];
        d_dense_qp_set_lg(h->zero,&h->qp);d_dense_qp_set_lg_mask(h->zero,&h->qp);
        d_dense_qp_set_ug(h->tmp,&h->qp);d_dense_qp_set_ug_mask(h->one,&h->qp);
    }
    h->metrics.update_s=seconds()-t;
    return 0;
}
void *hp_dense_create(const dense_data *p,int profile,double tolerance,int max_iter) {
    if (!p || p->n<1 || p->m<0 || p->ne<0 || !(tolerance>0) || max_iter<1 ||
        (profile!=BALANCE && profile!=ROBUST)) return NULL;
    const double t=seconds();
    dense_handle *h=calloc(1,sizeof(*h));if (!h) return NULL;
    h->metrics.memory_bytes=sizeof(*h);uint64_t *bytes=&h->metrics.memory_bytes;
    h->n=p->n;h->m=p->m;h->ne=p->ne;
    h->row_kind=allocate((size_t)p->m*sizeof(int),bytes);
    h->row_index=allocate((size_t)p->m*sizeof(int),bytes);
    h->row_scale=allocate((size_t)p->m*sizeof(double),bytes);
    h->row_constant=allocate((size_t)p->m*sizeof(double),bytes);
    h->box_columns=allocate((size_t)p->n*sizeof(int),bytes);
    h->box_lower_row=allocate((size_t)p->n*sizeof(int),bytes);
    h->box_upper_row=allocate((size_t)p->n*sizeof(int),bytes);
    h->scratch_i=allocate((size_t)3*p->n*sizeof(int),bytes);
    if (!h->row_kind || !h->row_index || !h->row_scale || !h->row_constant || !h->box_columns ||
        !h->box_lower_row || !h->box_upper_row || !h->scratch_i) goto fail;
    int nb=0,ng=0;
    classify_bounds(p->G,p->h,p->m,p->n,0,p->n,h->row_kind,h->row_index,h->box_columns,
        h->box_lower_row,h->box_upper_row,h->scratch_i,&nb,&ng);
    for (int i=0;i<p->m;++i) if (h->row_kind[i]==3) h->metrics.inactive_constraint_rows++;
    h->metrics.box_variables=nb;h->metrics.box_constraint_rows=p->m-ng-h->metrics.inactive_constraint_rows;
    h->metrics.general_constraint_rows=ng;h->metrics.terminal_equality_rows=p->ne;
    h->dim_mem=allocate(d_dense_qp_dim_memsize(),bytes);if (!h->dim_mem) goto fail;
    d_dense_qp_dim_create(&h->dim,h->dim_mem);
    d_dense_qp_dim_set_all(p->n,p->ne,nb,ng,0,&h->dim);
    h->qp_mem=allocate(d_dense_qp_memsize(&h->dim),bytes);
    h->sol_mem=allocate(d_dense_qp_sol_memsize(&h->dim),bytes);
    h->arg_mem=allocate(d_dense_qp_ipm_arg_memsize(&h->dim),bytes);
    if (!h->qp_mem || !h->sol_mem || !h->arg_mem) goto fail;
    d_dense_qp_create(&h->dim,&h->qp,h->qp_mem);d_dense_qp_set_all_zero(&h->qp);
    d_dense_qp_sol_create(&h->dim,&h->sol,h->sol_mem);
    d_dense_qp_ipm_arg_create(&h->dim,&h->arg,h->arg_mem);
    d_dense_qp_ipm_arg_set_default(profile,&h->arg);
    h->metrics.native_bound_count=p->m-h->metrics.inactive_constraint_rows;h->metrics.tolerance=tolerance;
    double comp=tolerance/(h->metrics.native_bound_count ? h->metrics.native_bound_count : 1);h->metrics.tol_comp=comp;
    d_dense_qp_ipm_arg_set_tol_stat(&tolerance,&h->arg);
    d_dense_qp_ipm_arg_set_tol_eq(&tolerance,&h->arg);
    d_dense_qp_ipm_arg_set_tol_ineq(&tolerance,&h->arg);
    d_dense_qp_ipm_arg_set_tol_comp(&comp,&h->arg);
    d_dense_qp_ipm_arg_set_tol_dual_gap(&tolerance,&h->arg);
    d_dense_qp_ipm_arg_set_iter_max(&max_iter,&h->arg);
    h->arg.stat_max=max_iter;h->arg.warm_start=0;
    h->ws_mem=allocate(d_dense_qp_ipm_ws_memsize(&h->dim,&h->arg),bytes);
    if (!h->ws_mem) goto fail;
    d_dense_qp_ipm_ws_create(&h->dim,&h->arg,&h->ws,h->ws_mem);
    h->tmp=allocate(maxz(maxz((size_t)p->n*p->n,(size_t)p->m*p->n),(size_t)p->ne*p->n)*sizeof(double)+64,bytes);
    h->zero=allocate((size_t)p->m*sizeof(double)+64,bytes);
    h->one=allocate((size_t)p->m*sizeof(double)+64,bytes);
    h->mask=allocate((size_t)p->m*sizeof(double)+64,bytes);
    if (!h->tmp || !h->zero || !h->one || !h->mask) goto fail;
    for (int j=0;j<p->m;++j) h->one[j]=1.;
    if (hp_dense_update(h,p)) goto fail;
    h->metrics.setup_s=seconds()-t;
    return h;
fail: hp_dense_destroy(h);return NULL;
}
int hp_dense_solve(void *ptr,double *v,double *y,double *s,double *z,hp_metrics *out) {
    dense_handle *h=ptr;if (!h) return -1;
    double t=seconds();
    blasfeo_dvecse(h->n,0.,h->sol.v,0);blasfeo_dvecse(h->ne,0.,h->sol.pi,0);
    const int nb=h->dim.nb,ng=h->dim.ng,nc=nb+ng;
    blasfeo_dvecse(2*nc,0.,h->sol.t,0);blasfeo_dvecse(2*nc,0.,h->sol.lam,0);
    d_dense_qp_ipm_solve(&h->qp,&h->sol,&h->arg,&h->ws);
    h->metrics.solve_s=seconds()-t;t=seconds();
    d_dense_qp_ipm_get_status(&h->ws,&h->metrics.status);
    d_dense_qp_ipm_get_iter(&h->ws,&h->metrics.iterations);
    d_dense_qp_ipm_get_stat_m(&h->ws,&h->metrics.stat_m);
    d_dense_qp_ipm_get_max_res_stat(&h->ws,&h->metrics.residual_stat);
    d_dense_qp_ipm_get_max_res_eq(&h->ws,&h->metrics.residual_eq);
    d_dense_qp_ipm_get_max_res_ineq(&h->ws,&h->metrics.residual_ineq);
    d_dense_qp_ipm_get_max_res_comp(&h->ws,&h->metrics.residual_comp);
    d_dense_qp_ipm_get_dual_gap(&h->ws,&h->metrics.dual_gap);
    d_dense_qp_ipm_get_obj(&h->ws,&h->metrics.objective);
    h->metrics.actual_ric_alg=-1;h->metrics.actual_lq_fact=h->ws.last_lq_fact;
    h->metrics.npd_hess=-1;h->metrics.npd_reg_hess=-1;
    d_dense_qp_sol_get_v(&h->sol,v);
    if (h->ne) {
        d_dense_qp_sol_get_pi(&h->sol,y);
        /* Dense HPIPM uses b-A*v for its equality Lagrangian. */
        for (int j=0;j<h->ne;++j) y[j]=-y[j];
    }
    for (int row=0;row<h->m;++row) {
        int kind=h->row_kind[row],index=h->row_index[row];
        if (kind==3) {s[row]=h->row_constant[row];z[row]=0.;continue;}
        int native_index=kind==2 ? index : (kind==1 ? nb+ng+index : 2*nb+ng+index);
        double scale=kind ? h->row_scale[row] : 1.;
        s[row]=h->sol.t->pa[native_index]*scale;
        z[row]=h->sol.lam->pa[native_index]/scale;
    }
    h->metrics.extract_s=seconds()-t;if (out) *out=h->metrics;
    return h->metrics.status;
}
int hp_dense_history(void *ptr,double *out,int capacity) {
    dense_handle *h=ptr;if (!h) return -1;
    int count=(h->metrics.iterations+1)*h->ws.stat_m;
    if (!out || capacity<count) return count;
    memcpy(out,h->ws.stat,(size_t)count*sizeof(double));return count;
}
void hp_dense_metrics(void *ptr,hp_metrics *out) {
    if (ptr && out) *out=((dense_handle *)ptr)->metrics;
}
int hp_dense_complementarity_budget(void *ptr,int component) {
    dense_handle *h=ptr;if (!h || (component!=0 && component!=1)) return -1;
    const double start=seconds();
    double divisor=component ? 1. : (h->metrics.native_bound_count>0 ? h->metrics.native_bound_count : 1);
    double tol=h->metrics.tolerance/divisor;
    d_dense_qp_ipm_arg_set_tol_comp(&tol,&h->arg);h->metrics.tol_comp=tol;
    h->metrics.setup_s+=seconds()-start;
    return 0;
}
void hp_dense_settings(void *ptr,double *d,int *i) {
    dense_handle *h=ptr;struct d_dense_qp_ipm_arg *a=&h->arg;
    double v[]={a->res_g_max,a->res_b_max,a->res_d_max,a->res_m_max,a->dual_gap_max,
        a->mu0,a->alpha_min,a->reg_prim,a->reg_dual,a->lam_min,a->t_min,a->tau_min,a->lam0_min,a->t0_min,a->m_safe};
    int w[]={a->mode,a->iter_max,a->stat_max,a->warm_start,-1,a->lq_fact,
        a->itref_pred_max,a->itref_corr_max,a->pred_corr,a->cond_pred_corr,a->abs_form,
        1,a->comp_res_exit,a->comp_res_pred,a->split_step,
        -1,a->t_lam_min,a->t0_init,a->update_fact_exit,a->kkt_fact_alg,a->remove_lin_dep_eq};
    memcpy(d,v,sizeof(v));memcpy(i,w,sizeof(w));
}
