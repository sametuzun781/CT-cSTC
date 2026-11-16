"""Load the notebook model without running optimization or plotting cells."""
from pathlib import Path
import copy
import json

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ('cvxpy_qoco', 'qoco_native', 'jax_icore', 'icore_c', 'icore_cpp')


def load_application(backend='icore_cpp', autodiff='forward', discretization='fast'):
    if backend not in BACKENDS:
        raise ValueError(f'Unknown solver: {backend}')
    notebook = json.loads((ROOT / 'CT-cSTC.ipynb').read_text())
    ns = {'__name__': '__trajectory_example__'}
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = '\n'.join(line for line in ''.join(cell['source']).splitlines()
                           if not line.lstrip().startswith('%'))
        exec(compile(source, f'CT-cSTC.ipynb:cell{index}', 'exec'), ns)
        if 'SOLVER_BACKEND =' in source:
            ns.update(SOLVER_BACKEND=backend, DYNAMICS_AD=autodiff,
                      ORIGINAL_PATH=backend == 'cvxpy_qoco', DISCRETIZATION_MODE=discretization)
        if 'rl_params_sc' in ns:
            break
    params = ns['rl_params_sc']
    disc = ns['make_calc_fn_cd'](params, compile=False)
    integrate = ns['make_calc_fn_im'](params, compile=False)
    ns.update(calculate_discretization_jitted=disc, integrate_multiple_jitted=integrate,
              optional_problem=None, original_solver_names=set())
    return ns, params, {'disc': disc, 'integrate': integrate}


def prepare_solver(ns, params, functions, backend='icore_cpp', tolerance=None):
    if backend == 'cvxpy_qoco':
        return None
    import jax
    from solvers.ipm_adapter import OptionalConvexProblem
    tolerance = ns['SOLVER_TOLERANCE'] if tolerance is None else tolerance
    matrices = functions['disc'](params['X_last'], params['U_last'], params['sigma_last'])
    jax.block_until_ready(matrices)
    problem = OptionalConvexProblem(params, backend, tolerance=tolerance,
                                    qoco_options=ns['QOCO_OPTIONS'])
    problem.prepare(params['X_last'], params['U_last'], params['sigma_last'],
                    params['w_ptr'], dict(params, **matrices))
    ns['optional_problem'] = problem
    return problem


def run_scp(ns, params, functions, problem, iterations=None):
    params = copy.deepcopy(params)
    if iterations is not None:
        params['ite'] = iterations
    return ns['prox_linear'](params)
