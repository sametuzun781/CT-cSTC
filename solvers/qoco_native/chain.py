"""Direct sparse assembly of the same quadratic/L1 chain model."""
from types import SimpleNamespace
import numpy as np
from scipy import sparse


class ChainCanonicalizer:
    def __init__(self, qp):
        free = np.asarray(qp.free)
        if not np.all((free == 0) | (free == 1)):
            raise ValueError("The free-coordinate mask must contain only zero and one.")
        self.free = free.astype(bool, copy=True)
        self.nodes, self.width = self.free.shape
        self.indices = np.flatnonzero(self.free.ravel())
        self.nfree = len(self.indices)
        self.stage_columns = []
        cursor = 0
        for mask in self.free:
            count = np.count_nonzero(mask)
            self.stage_columns.append(np.arange(cursor, cursor + count))
            cursor += count

    def _operator(self, left, right=None):
        rows, columns, values = [], [], []
        count = left.shape[1]
        for k in range(len(left)):
            for stage, block in ((k, left[k]),) if right is None else ((k, left[k]), (k + 1, right[k])):
                cols = self.stage_columns[stage]
                rows.append(np.repeat(np.arange(k * count, (k + 1) * count), len(cols)))
                columns.append(np.tile(cols, count))
                values.append(block[:, self.free[stage]].ravel())
        shape = (len(left) * count, self.nfree)
        if not rows:
            return sparse.csc_matrix(shape)
        matrix = sparse.coo_matrix((np.concatenate(values),
                                   (np.concatenate(rows), np.concatenate(columns))), shape=shape).tocsc()
        matrix.eliminate_zeros()
        return matrix

    def build(self, qp):
        free = np.asarray(qp.free)
        if not np.all((free == 0) | (free == 1)) or not np.array_equal(free, self.free):
            raise ValueError("The fixed-coordinate pattern changed; construct a new solver.")
        if not np.all(np.asarray(qp.hessian) > 0) or not np.all(np.asarray(qp.l1_weight) > 0):
            raise ValueError("ChainQP needs positive diagonal curvature and L1 weights.")
        fixed = np.where(self.free, 0., qp.fixed)
        defect = self._operator(qp.left, qp.right)
        offset = (qp.offset + np.einsum("kij,kj->ki", qp.left, fixed[:-1])
                  + np.einsum("kij,kj->ki", qp.right, fixed[1:])).ravel()
        ne = len(offset)
        pair = self._operator(qp.pair_left, qp.pair_right)
        pair_offset = (qp.pair_offset + np.einsum("kij,kj->ki", qp.pair_left, fixed[:-1])
                       + np.einsum("kij,kj->ki", qp.pair_right, fixed[1:])).ravel()
        local = self._operator(qp.local_matrix)
        local_offset = (qp.local_offset + np.einsum("kij,kj->ki", qp.local_matrix, fixed)).ravel()
        # Variables are [free physical coordinates, residuals, epigraphs].
        # Keeping D y - v = -offset explicit avoids duplicating D inside
        # increasingly ill-conditioned barrier rows near zero L1 residuals.
        eye = sparse.eye(ne, format="csc")
        zero = sparse.csc_matrix((ne, ne))
        A = sparse.hstack((defect, -eye, zero), format="csc")
        G = sparse.vstack((
            sparse.hstack((sparse.csc_matrix((ne, self.nfree)), eye, -eye)),
            sparse.hstack((sparse.csc_matrix((ne, self.nfree)), -eye, -eye)),
            sparse.hstack((pair, sparse.csc_matrix((pair.shape[0], 2*ne)))),
            sparse.hstack((local, sparse.csc_matrix((local.shape[0], 2*ne))))), format="csr")
        h = np.concatenate((np.zeros(2*ne), -pair_offset, -local_offset))
        G.eliminate_zeros()
        keep = (np.diff(G.indptr) > 0) | (h < 0.)
        G, h = G[keep].tocsc(), h[keep]
        diagonal = np.r_[np.asarray(qp.hessian).ravel()[self.indices], np.zeros(2*ne)]
        # Keep explicit zero diagonal slots so public matrix updates retain P's pattern.
        P = sparse.csc_matrix((diagonal, np.arange(len(diagonal)), np.arange(len(diagonal) + 1)),
                              shape=(len(diagonal), len(diagonal)))
        linear = np.asarray(qp.linear) - np.asarray(qp.hessian) * np.asarray(qp.center)
        c = np.r_[linear.ravel()[self.indices], np.zeros(ne),
                  np.broadcast_to(qp.l1_weight, qp.offset.shape).ravel()]
        return SimpleNamespace(P=P, c=c, A=A, b=-offset,
                               G=G, h=h, l=len(h), cone_sizes=(), keep=keep, epigraphs=ne,
                               fixed=fixed, shape=self.free.shape)

    def decode(self, result, canonical):
        y = canonical.fixed.copy().ravel()
        y[self.indices] = result["x"][:self.nfree]
        return y.reshape(self.free.shape)
