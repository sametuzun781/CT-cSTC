"""Generate a conservative static sensitivity graph; no numeric zero thresholds."""
from pathlib import Path
import json, numpy as np, sys

def generate(root, model, nx, nu):
    src=Path(root)/'native_discretization'
    layout=json.loads((src/'field_layout.json').read_text())
    indices=layout['jacobian_indices'];nd=nx+2*nu+1
    A=np.zeros((nx,nx),bool);B=np.zeros((nx,nu),bool)
    for r,c in indices:
        if c<nx:A[r,c]=True
        else:B[r,c-nx]=True
    inject=np.zeros((nx,nd),bool)
    inject[:,nx:nx+nu]=B;inject[:,nx+nu:nx+2*nu]=B
    # Conservative: all f_i may be nonzero, including at changing parameters.
    inject[:,-1]=True
    support=np.zeros_like(inject);support[:,:nx]=np.eye(nx,dtype=bool);support|=inject
    for _ in range(nx+1):
        new=support|((A.astype(int)@support.astype(int))!=0)
        if np.array_equal(new,support):break
        support=new
    dynamic=((A.astype(int)@support.astype(int))!=0)|inject
    mapping=-np.ones((nx,nd),int)
    mapping[dynamic]=nx+np.arange(np.count_nonzero(dynamic))
    nv=nx+np.count_nonzero(dynamic)
    lines=['/* Generated structural sensitivity transport. No approximate sparsity. */',
           f'#define NX {nx}',f'#define NU {nu}',f'#define ND {nd}',f'#define NV {nv}',
           '#define OUT (NX + NX*NX + 2*NX*NU + 2*NX)',
           'static void initialize(double *v, const double *x0) {',
           '    memset(v,0,NV*sizeof(double)); memcpy(v,x0,NX*sizeof(double));']
    for i in range(nx):
        if mapping[i,i]>=0:lines.append(f'    v[{mapping[i,i]}]=1.0;')
    lines+=['}',
    'static int rhs(const double *restrict v, const double *u0, const double *u1,',
    '               double scale, double beta, const double *parameters, double *restrict d) {',
    f'    double u[NU], f[NX], jac[{model.upper()}_NNZ];',
    '    for(int i=0;i<NU;++i) u[i]=u0[i]+beta*(u1[i]-u0[i]);',
    f'    {model}_field_prepared(v,u,parameters,f,jac);',
    '    memset(d,0,NV*sizeof(double));',
    '    for(int i=0;i<NX;++i) {if(!isfinite(f[i])) return 2; d[i]=scale*f[i];}',
    f'    for(int i=0;i<{model.upper()}_NNZ;++i) if(!isfinite(jac[i])) return 3;']
    for r in range(nx):lines.append(f'    d[{mapping[r,nd-1]}]=f[{r}];')
    products=0
    for n,(r,c) in enumerate(indices):
        lines.append(f'    {{ const double a=scale*jac[{n}];')
        if c<nx:
            terms=[]
            for j in range(nd):
                if not support[c,j]:continue
                dst=mapping[r,j];source=mapping[c,j]
                assert dst>=0
                if source>=0:terms.append((dst,source))
                elif c==j:lines.append(f'      d[{dst}]+=a;');products+=1
                else:raise RuntimeError('Nonzero support with no stored value')
            # Express contiguous copies as constant-bound loops, without indirect indexing.
            while terms:
                d,s=terms.pop(0);count=1
                while terms and terms[0]==(d+count,s+count):terms.pop(0);count+=1
                products+=count
                if count==1:lines.append(f'      d[{d}]+=a*v[{s}];')
                else:lines.append(f'      for(int j=0;j<{count};++j) d[{d}+j]+=a*v[{s}+j];')
        else:
            j=c-nx
            lines.append(f'      d[{mapping[r,nx+j]}]+=a*(1.0-beta);')
            lines.append(f'      d[{mapping[r,nx+nu+j]}]+=a*beta;');products+=2
        lines.append('    }')
    lines+=['    return 0;','}',
    'static void unpack(const double *v,const double *x0,const double *u0,const double *u1,',
    '                   double scale,int free_time,double *out) {',
    '    memset(out,0,OUT*sizeof(double));memcpy(out,v,NX*sizeof(double));']
    # Output matches row-major A,B,C,S,z; Rocket wrapper converts to column-major.
    for i in range(nx):
        for j in range(nd):
            pos=(nx+i*nx+j if j<nx else nx+nx*nx+i*nu+j-nx if j<nx+nu else
                nx+nx*nx+nx*nu+i*nu+j-nx-nu if j<nx+2*nu else nx+nx*nx+2*nx*nu+i)
            if mapping[i,j]>=0:lines.append(f'    out[{pos}]=v[{mapping[i,j]}];')
            elif i==j:lines.append(f'    out[{pos}]=1.0;')
    lines+=[
    '    for(int i=0;i<NX;++i) {',
    '        double affine=v[i];',
    '        for(int j=0;j<NX;++j) affine-=out[NX+i*NX+j]*x0[j];',
    '        for(int j=0;j<NU;++j) affine-=out[NX+NX*NX+i*NU+j]*u0[j];',
    '        for(int j=0;j<NU;++j) affine-=out[NX+NX*NX+NX*NU+i*NU+j]*u1[j];',
    '        if(free_time) affine-=out[NX+NX*NX+2*NX*NU+i]*scale;',
    '        out[OUT-NX+i]=affine;',
    '    }','}']
    (src/'sensitivity_plan.h').write_text('\n'.join(lines)+'\n')
    stats=dict(nx=nx,nu=nu,tangent_columns=nd,dense_tangent_elements=nx*nd,
      dynamic_tangent_elements=int(np.count_nonzero(dynamic)),
      invariant_identity_elements=sum(mapping[i,i]<0 for i in range(nx)),
      structural_zero_elements=int((~support).sum()),
      dense_sparse_product_terms=int(A.sum()*nd+2*B.sum()),compressed_product_terms=products,
      state_dependency_edges=int(A.sum()),input_dependency_edges=int(B.sum()))
    stats['invariant_identity_elements']=int(stats['invariant_identity_elements'])
    (src/'transport_layout.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(model,stats)
if __name__=='__main__':generate(sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]))
