"""Application array model and its physical objective."""
import numpy as np
from icore import ChainQP


def chain_objective(qp, y, xp=np):
    defect = (xp.einsum("kij,kj->ki",qp.left,y[:-1])
              + xp.einsum("kij,kj->ki",qp.right,y[1:]) + qp.offset)
    return (.5*xp.sum(qp.hessian*(y-qp.center)**2)
            + xp.sum(qp.linear*y) + xp.sum(qp.l1_weight*xp.abs(defect)))
