"""Check all five solvers against the notebook's original CVXPY subproblem."""
import json
from importlib.metadata import version
from pathlib import Path
import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from ctcstc.application import ROOT, load_application
from solvers.ipm_adapter import OptionalConvexProblem
from tests.audit_chain import audit_chain


def test_installed_icore_runtime():
    import icore
    from tools.install_icore import VERSION
    from solvers.qoco_native.build import QOCO_SHA256, _sha
    package = Path(icore.__file__).resolve().parent
    assert package != ROOT/'icore'
    assert version('icore') == VERSION
    assert not (package/'benchmark.py').exists()
    assert not (package/'vendor/hpipm').exists()
    for name in ('LICENSE', 'LICENSE-NOTE.md', 'THIRD_PARTY_NOTICES.md'):
        assert (package/name).is_file()
    assert _sha(package/'vendor/qoco/qoco-source.tar.gz') == QOCO_SHA256


def test_five_solvers_match_original_problem(monkeypatch):
    ns, p, functions = load_application(backend='cvxpy_qoco')
    X, U, S = (jnp.asarray(p[key]) for key in ('X_last', 'U_last', 'sigma_last'))
    matrices = functions['disc'](X, U, S)
    p.update(matrices)
    captured = []
    original = cp.Problem
    def capture(*args, **kwargs):
        problem = original(*args, **kwargs)
        captured.append(problem)
        return problem
    with monkeypatch.context() as patch:
        patch.setattr(cp, 'Problem', capture)
        ns['solve_convex_problem'](np.asarray(X), np.asarray(U), np.asarray(S), p['w_ptr'],
                                  dict(p, **jax.device_get(matrices)))
    reference = captured[0]
    objective = float(reference.value)
    for backend in ('qoco_native', 'jax_icore', 'icore_c', 'icore_cpp'):
        problem = OptionalConvexProblem(p, backend)
        try:
            result = problem.solve(X, U, S, p['w_ptr'], p)
            saved = tuple(np.array(a) for a in result)
            problem.solve(X, U, S, p['w_ptr'], p)
            for value, expected in zip(result, saved):
                np.testing.assert_array_equal(value, expected)
            if backend != 'qoco_native':
                assert audit_chain(problem.engine.last_model, problem.engine.last_result)['accepted']
            if backend == 'jax_icore':
                assert all(isinstance(a, jax.Array) for a in problem.engine.last_model)
                assert problem.records[-1]['jax_to_numpy_seconds'] == 0.
            by_shape = {a.shape: a for a in saved}
            for variable in reference.variables():
                variable.value = by_shape[variable.shape]
            violation = max(float(np.max(c.violation(), initial=0.)) for c in reference.constraints)
            assert violation < 1e-7, (backend, violation)
            assert abs(float(reference.objective.value)-objective)/(1+abs(objective)) < 1e-7
        finally:
            problem.close()


def test_notebook_is_clean_and_root_is_small():
    notebook = json.loads((ROOT/'CT-cSTC.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == [] and cell['execution_count'] is None
    assert {p.name for p in ROOT.glob('*.py')} == {'run_example.py'}
