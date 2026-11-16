"""Rocket model adapter for native QOCO and the three ICORE execution paths."""
from time import perf_counter
import numpy as np
import jax
import jax.numpy as jnp
from solvers.chain_model import ChainQP
from ctcstc.array_io import to_device

COST_SCALE = 4096.0


class OptionalConvexProblem:
    """Expose the rocket notebook's .solve(X, U, sigma, weight, params) interface."""

    def __init__(self, params, backend='jax_icore', *, tolerance=1e-8,
                 qoco_options=None):
        from solvers import ChainSolver
        if not (params['n_x'] == 15 and params['n_u'] == 5 and
                params['free_final_time'] and params['time_dil'] and
                params['add_minmax_sig'] and params['inp_param'] == 'FOH'):
            raise NotImplementedError('This adapter implements the supplied FOH rocket model.')
        if params['min_sigma'] < 0:
            raise ValueError('The time-cost adapter requires nonnegative durations.')
        self.params = params.copy()
        self.backend = backend
        self.records = []
        self._compile_seconds = 0.0
        self.engine = ChainSolver(backend, tolerance=tolerance, qoco_options=qoco_options)
        self.build = jax.jit(lambda values: build_rocket_qp(*values, self.params))
        self.compiled_build = None
        self.compiled_decode = None

    @property
    def compile_seconds(self):
        return self._compile_seconds + self.engine.compile_seconds

    @property
    def setup_seconds(self):
        return self.engine.setup_seconds

    def decode(self, y, qp):
        n_x, n_u = self.params['n_x'], self.params['n_u']
        defect = (jnp.einsum('kij,kj->ki', qp.left, y[:-1])
                  + jnp.einsum('kij,kj->ki', qp.right, y[1:]) + qp.offset)
        return y[:, :n_x].T, y[:, n_x:n_x+n_u].T, y[:-1, -1][None], defect.T

    @staticmethod
    def _values(X, U, sigma, weight, params):
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError('A positive finite proximal weight is required.')
        matrices = tuple(params[name] for name in ('A_bar', 'B_bar', 'C_bar', 'S_bar', 'z_bar'))
        return X, U, sigma, float(weight), matrices

    def prepare(self, X, U, sigma, weight, params):
        """Build workspaces and compile the selected path, without a warm-up solve."""
        values, _ = to_device(self._values(X, U, sigma, weight, params), dtype=jnp.float64)
        if self.compiled_build is None:
            started = perf_counter()
            self.compiled_build = self.build.lower(values).compile()
            self._compile_seconds += perf_counter() - started
        qp = self.compiled_build(values)
        jax.block_until_ready(qp)
        self.engine.prepare(qp)
        if self.compiled_decode is None:
            started = perf_counter()
            self.compiled_decode = jax.jit(self.decode).lower(qp.center, qp).compile()
            self._compile_seconds += perf_counter() - started

    def solve(self, X, U, sigma, weight, params):
        """Solve one convex subproblem and return device-resident X, U, sigma, nu."""
        compilation_before = self.compile_seconds
        setup_before = self.setup_seconds
        if self.compiled_build is None:
            self.prepare(X, U, sigma, weight, params)
        values, upload_seconds = to_device(
            self._values(X, U, sigma, weight, params), dtype=jnp.float64)
        started = perf_counter()
        qp = self.compiled_build(values)
        jax.block_until_ready(qp)
        build_seconds = perf_counter() - started
        y, record = self.engine.solve(qp)
        started = perf_counter()
        # Compiled JAX accepts NumPy input; input conversion and decode are timed together.
        solution = self.compiled_decode(y, qp)
        jax.block_until_ready(solution)
        decode_seconds = perf_counter() - started
        record = dict(record)
        record['seconds'] += build_seconds + decode_seconds
        record['numpy_to_jax_seconds'] += upload_seconds
        record['solution_decode_seconds'] = decode_seconds
        record['compilation_seconds'] = self.compile_seconds - compilation_before
        record['setup_seconds'] = self.setup_seconds - setup_before
        record['objective_scale'] = COST_SCALE
        self.records.append(record)
        return solution

    def close(self):
        self.engine.close()


def build_rocket_qp(X, U, sigma, weight, linearization, params):
    """Translate the original FOH subproblem to ChainQP; do not change its cost."""
    nodes = params['K']
    n_x = params['n_x']
    n_u = params['n_u']
    width = n_x + n_u + 1                 # one dummy terminal duration, fixed at zero
    A_bar, B_bar, C_bar, S_bar, z_bar = linearization

    # The discretizer stores each matrix column-major.
    A = A_bar.T.reshape(nodes - 1, n_x, n_x).transpose(0, 2, 1)
    B = B_bar.T.reshape(nodes - 1, n_u, n_x).transpose(0, 2, 1)
    C = C_bar.T.reshape(nodes - 1, n_u, n_x).transpose(0, 2, 1)
    left = -jnp.concatenate((A, B, S_bar.T[:, :, None]), axis=2)
    right = jnp.concatenate((jnp.broadcast_to(jnp.eye(n_x), (nodes - 1, n_x, n_x)),
                             -C, jnp.zeros((nodes - 1, n_x, 1))), axis=2)

    # CTCS accumulator: x_ctcs[k+1] - x_ctcs[k] <= beta_ctcs.
    pair_left = jnp.zeros((nodes - 1, 1, width)).at[:, 0, n_x - 1].set(-1.)
    pair_right = -pair_left
    pair_offset = jnp.full((nodes - 1, 1), -params['beta_ctcs'])

    # Each row is a scalar affine constraint: G[k] @ y[k] + g[k] <= 0.
    G = jnp.zeros((nodes, 9, width))
    g = jnp.zeros((nodes, 9))
    G = G.at[:, 0, 0].set(-1.)                 # minimum mass
    g = g.at[:, 0].set(params['m_dry'])
    G = G.at[:, 1, n_x].set(1.)                # thrust upper/lower bounds
    G = G.at[:, 2, n_x].set(-1.)
    g = g.at[:, 1].set(-params['T_max'])
    g = g.at[:, 2].set(params['T_min_aft'])
    G = G.at[:, 3, n_x + 1].set(1.)            # engine gimbal angle
    G = G.at[:, 4, n_x + 1].set(-1.)
    g = g.at[:, 3:5].set(-np.deg2rad(params['delta_engine_max']))
    G = G.at[:, 5, n_x + 3].set(1.)            # camera boresight angle
    G = G.at[:, 6, n_x + 3].set(-1.)
    g = g.at[:, 5:7].set(-np.deg2rad(params['delta_boresight_max']))
    G = G.at[:-1, 7, -1].set(1.)               # interval duration bounds
    G = G.at[:-1, 8, -1].set(-1.)
    g = g.at[:-1, 7].set(-params['max_sigma'])
    g = g.at[:-1, 8].set(params['min_sigma'])
    g = g.at[-1, 7:9].set(-1.)                 # no bound on the dummy duration

    center = jnp.concatenate((X.T, U.T, jnp.pad(sigma.T, ((0, 1), (0, 0)))), axis=1)
    hessian = jnp.full_like(center, 2. * weight / COST_SCALE)
    hessian = hessian.at[:, -1].multiply((params['w_ds'] / params['t_f']) ** 2)
    linear = jnp.zeros_like(center).at[:-1, -1].set(
        params['w_sigma'] / params['t_f'] / COST_SCALE
    )
    free = jnp.ones_like(center).at[0, :n_x].set(0.)
    free = free.at[-1, 1:n_x - 1].set(0.).at[-1, -1].set(0.)
    fixed = jnp.zeros_like(center).at[0, :n_x].set(params['x_init'])
    fixed = fixed.at[-1, 1:n_x - 1].set(params['x_final'][1:-1])

    return ChainQP(left, right, -z_bar.T, pair_left, pair_right, pair_offset,
                   G, g, center, hessian, linear, params['w_con_dyn'] / COST_SCALE,
                   free, fixed)


def evaluate_rocket_costs(candidate, qp, params, integrate, linear_defect=None):
    """Shared scalar costs for both SCP loops; all trajectory work stays in JAX."""
    nx, nu = params['n_x'], params['n_u']
    X, U, S = candidate[:, :nx].T, candidate[:, nx:nx + nu].T, candidate[:-1, -1][None]
    nonlinear_defect = X[:, 1:] - integrate(X, U, S)
    if linear_defect is None:
        linear_defect = (jnp.einsum('kij,kj->ki', qp.left, candidate[:-1])
                         + jnp.einsum('kij,kj->ki', qp.right, candidate[1:]) + qp.offset)
    time_cost = params['w_sigma'] * jnp.sum(jnp.abs(S)) / params['t_f']
    lcd = params['w_con_dyn'] * jnp.sum(jnp.abs(linear_defect))
    nlcd = params['w_con_dyn'] * jnp.sum(jnp.abs(nonlinear_defect))
    proximal = .5 * COST_SCALE * jnp.sum(qp.hessian * (candidate - qp.center) ** 2)
    return dict(sigma_cost=time_cost, dyn_cost=nlcd, ptr_cost=proximal,
                lcd=lcd, nlcd=nlcd, lc=time_cost + lcd + proximal, nlc=time_cost + nlcd)
