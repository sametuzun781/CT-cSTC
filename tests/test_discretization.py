"""Native RK4 must preserve the notebook's original JAX equations."""
import copy
from concurrent.futures import ThreadPoolExecutor
import jax
import numpy as np
import pytest
from ctcstc.application import load_application
import native_discretization as native


def serial_matrices(discretizer, X, U, S):
    """Run the unchanged serial C kernel without the parallel FFI boundary."""
    arrays = [np.ascontiguousarray(np.asarray(a).T) for a in (X, U, S)]
    out = np.empty((discretizer.nodes-1, 420))
    status = discretizer.library.icore_rocket_discretize(
        discretizer.nodes, discretizer.points, *arrays, discretizer.parameters, out)
    assert status == 0
    result = dict(f_bar=out[:, :15].T, S_bar=out[:, 390:405].T, z_bar=out[:, 405:].T)
    for key, start, cols in (('A_bar', 15, 15), ('B_bar', 240, 5), ('C_bar', 315, 5)):
        result[key] = out[:, start:start+15*cols].reshape(-1, 15, cols).transpose(0, 2, 1).reshape(-1, 15*cols).T
    return result


@pytest.mark.parametrize('autodiff', ['forward', 'reverse'])
@pytest.mark.parametrize('nodes,points', [(5, 2), (9, 7), (15, 20), (21, 11)])
def test_native_matches_reference(autodiff, nodes, points):
    ns, base, _ = load_application(autodiff=autodiff, discretization='reference')
    p = copy.deepcopy(base)
    p.update(K=nodes, rk4_steps_J=points)
    p['g_I'] = np.asarray(p['g_I']) * .97
    p['rho_air'] *= 1.03
    p = ns['rl_dynamics'](p)
    before, after = np.linspace(0, 1, base['K']), np.linspace(0, 1, nodes)
    X = np.array([np.interp(after, before, v) for v in base['X_last']])
    U = np.array([np.interp(after, before, v) for v in base['U_last']])
    S = np.full((1, nodes-1), np.sum(base['sigma_last']) / (nodes-1))
    reference = ns['make_reference_calc_fn_cd'](p, compile=False)(X, U, S)
    prepared = native.NativeRocketDiscretizer(p)
    result = prepared(X, U, S)
    serial = serial_matrices(prepared, X, U, S)
    for key in reference:
        a, b = np.asarray(reference[key]), np.asarray(result[key])
        assert np.isfinite(b).all()
        assert np.max(np.abs(a-b)/(1+np.abs(a))) < 1e-9, key
        np.testing.assert_array_equal(b, serial[key])


def test_parallel_calls_own_their_buffers():
    _, p, functions = load_application(discretization='native')
    prepared = functions['disc']
    X, U = p['X_last'], p['U_last']
    durations = [np.asarray(p['sigma_last'])*(1+0.01*k) for k in range(8)]
    expected = [serial_matrices(prepared, X, U, S) for S in durations]
    with ThreadPoolExecutor(max_workers=4) as workers:
        actual = list(workers.map(lambda S: prepared(X, U, S.T), durations))
    for reference, result in zip(expected, actual):
        for key in reference:
            np.testing.assert_array_equal(result[key], reference[key])
    bad = durations[0].copy()
    bad[0, -1] = np.nan
    with pytest.raises((ValueError, RuntimeError)):
        prepared(X, U, bad)
    for key, value in prepared(X, U, durations[0]).items():
        np.testing.assert_array_equal(value, expected[0][key])


def test_routing_and_input_rejection(monkeypatch):
    ns, p, _ = load_application()
    factory = ns['make_reference_calc_fn_cd']
    assert native.eligibility(p)[0]
    modified = dict(p, f_func=lambda x, u: p['f_func'](x, u)*1.001)
    assert not native.eligibility(modified)[0]
    assert native.make_discretizer(modified, factory, 'fast', False).discretization_backend == 'jax'
    with pytest.raises(ValueError, match='callbacks'):
        native.make_discretizer(modified, factory, 'native')
    for mode in ('jax', 'reference'):
        assert native.make_discretizer(p, factory, mode, False).discretization_backend == mode
    prepared = native.NativeRocketDiscretizer(p)
    for durations in (-np.asarray(p['sigma_last']), np.full_like(p['sigma_last'], np.nan)):
        with pytest.raises((ValueError, RuntimeError)):
            prepared(p['X_last'], p['U_last'], durations)
    def unavailable():
        raise RuntimeError('compiler unavailable')
    monkeypatch.setattr(native, 'build_library', unavailable)
    with pytest.warns(RuntimeWarning, match='using JAX'):
        assert native.make_discretizer(p, factory, 'fast', False).discretization_backend == 'jax'
    with pytest.raises(RuntimeError, match='compiler'):
        native.make_discretizer(p, factory, 'native')


def test_field_and_jacobians():
    _, p, _ = load_application()
    prepared = native.NativeRocketDiscretizer(p)
    reference = jax.jit(lambda x, u: (p['f_func'](x, u), *jax.jacfwd(p['f_func'], argnums=(0, 1))(x, u)))
    rng = np.random.default_rng(37)
    for _ in range(32):
        k = rng.integers(p['K'])
        x = p['X_last'][:, k] + rng.normal(size=p['n_x'])*1e-4
        u = p['U_last'][:, k] + rng.normal(size=p['n_u'])*1e-4
        value, a, b = reference(x, u)
        actual, jacobian = prepared.field_and_jacobian(x, u)
        for expected, candidate in ((np.asarray(value), actual), (np.c_[a, b], jacobian)):
            assert np.max(np.abs(expected-candidate)/(1+np.abs(expected))) < 1e-9
