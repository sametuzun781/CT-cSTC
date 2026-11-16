/* Fixed-grid RK4 with analytic sparse variational equations.
 * No heap allocation, finite differences, reduced step count or fast-math.
 * State order and physical equations match the delivered rocket notebook.
 */
#include <math.h>
#include <string.h>
#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif
#include "rocket_field.c"
#include "sensitivity_plan.h"

int icore_rocket_discretize(int nodes, int points, const double *X, const double *U,
                           const double *duration, const double *raw_parameters, double *output)
{
    if (nodes<2 || points<2) return 1;
    double parameters[ROCKET_NPARAM+ROCKET_NINVARIANT];
    memcpy(parameters,raw_parameters,ROCKET_NPARAM*sizeof(double));
    rocket_prepare(raw_parameters,parameters+ROCKET_NPARAM);
    const double h=1.0/(points-1);
    for (int interval=0;interval<nodes-1;++interval) {
        double v[NV]={0}, tmp[NV], k1[NV], k2[NV], k3[NV], k4[NV];
        const double *x0=X+interval*NX, *u0=U+interval*NU, *u1=u0+NU;
        const double scale=duration[interval];
        if (!isfinite(scale) || scale<=0.0) return 4;
        initialize(v,x0);
        for (int step=0;step<points-1;++step) {
            double t=step*h;
            int error=rhs(v,u0,u1,scale,t,parameters,k1);
            if (error) return error;
            for (int j=0;j<NV;++j) {k1[j]*=h;tmp[j]=v[j]+0.5*k1[j];}
            error=rhs(tmp,u0,u1,scale,t+0.5*h,parameters,k2);
            if (error) return error;
            for (int j=0;j<NV;++j) {k2[j]*=h;tmp[j]=v[j]+0.5*k2[j];}
            error=rhs(tmp,u0,u1,scale,t+0.5*h,parameters,k3);
            if (error) return error;
            for (int j=0;j<NV;++j) {k3[j]*=h;tmp[j]=v[j]+k3[j];}
            error=rhs(tmp,u0,u1,scale,t+h,parameters,k4);
            if (error) return error;
            for (int j=0;j<NV;++j) {
                k4[j]*=h;
                v[j]+=(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6.0;

            }
            int invalid=0;
            for(int j=0;j<NV;++j) invalid|=!isfinite(v[j]);
            if(invalid) return 5;
        }
        unpack(v,x0,u0,u1,scale,1,output+interval*OUT);
    }
    return 0;
}
