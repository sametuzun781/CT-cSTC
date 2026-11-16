"""Independent KKT check in the original quadratic-plus-L1 chain variables."""
import numpy as np

def audit_chain(qp,result,tolerance=1e-8,reference=None):
    qp = type(qp)(*(np.asarray(value) for value in qp))
    y,t=np.asarray(result['stage_y']),np.asarray(result['epigraph'])
    s=tuple(np.asarray(v) for v in result['s'])
    z=tuple(np.asarray(v) for v in result['z'])
    maximum=lambda a:float(np.max(np.abs(a),initial=0.))
    if not all(np.isfinite(a).all() for a in (y,t,*s,*z)):
        return dict(primal=float('inf'),dual=float('inf'),gap=float('inf'),accepted=False)
    defect=np.einsum('kij,kj->ki',qp.left,y[:-1])+np.einsum('kij,kj->ki',qp.right,y[1:])+qp.offset
    pair=np.einsum('kij,kj->ki',qp.pair_left,y[:-1])+np.einsum('kij,kj->ki',qp.pair_right,y[1:])+qp.pair_offset
    local=np.einsum('kij,kj->ki',qp.local_matrix,y)+qp.local_offset
    constraints=(defect-t,-defect-t,pair,local)
    gradient=qp.hessian*(y-qp.center)+qp.linear
    gradient[:-1]+=np.einsum('kij,ki->kj',qp.left,z[0]-z[1])+np.einsum('kij,ki->kj',qp.pair_left,z[2])
    gradient[1:]+=np.einsum('kij,ki->kj',qp.right,z[0]-z[1])+np.einsum('kij,ki->kj',qp.pair_right,z[2])
    gradient+=np.einsum('kij,ki->kj',qp.local_matrix,z[3])
    primal=max(maximum((1-qp.free)*(y-qp.fixed)),*(maximum(g+v) for g,v in zip(constraints,s)),*(float(np.max(-v,initial=0.)) for v in s))
    dual=max(maximum(qp.free*gradient),maximum(qp.l1_weight-z[0]-z[1]),*(float(np.max(-v,initial=0.)) for v in z))
    gap=sum(float(np.sum(np.abs(a*b))) for a,b in zip(s,z))
    objective=float(.5*np.sum(qp.hessian*(y-qp.center)**2)+np.sum(qp.linear*y)+np.sum(qp.l1_weight*np.abs(defect)))
    hard=max(maximum((1-qp.free)*(y-qp.fixed)),float(np.max(pair,initial=0.)),float(np.max(local,initial=0.)))
    error=None if reference is None else abs(objective-reference)
    return dict(primal=primal,dual=dual,gap=gap,objective=objective,objective_error=error,
                hard_constraint_violation=hard,accepted=bool(int(result['status'])==0 and max(primal,dual,gap)<=tolerance
                and (error is None or error<=10*tolerance*(1+abs(reference)))))
