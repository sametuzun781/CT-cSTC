"""Iteration formatting: timing records are seconds; display is milliseconds."""
import numbers
import numpy as np


def print_run_summary(result, backend, discretizer):
    trials = result['trial_timings']
    calls = result['convex_solve_count']
    count = len({row['iteration'] for row in trials})
    print('Solver:', backend, '| Discretization:', discretizer.discretization_backend)
    print('Linearizations / convex calls:', count, '/', calls, '| Converged:', result['converged'])
    for label, seconds, n in (
        ('Discretization', result['total_discretization_seconds'], count),
        ('Convex calls (incl. solution input/decode)', result['total_convex_seconds'], calls),
        ('Explicit array transitions (loop)', sum(t['T_J_Np'] + t['T_Np_J'] for t in trials), calls),
    ):
        print(f'{label}: {1000*seconds:.3f} ms total; {1000*seconds/max(n,1):.4f} ms/call')
    startup = result['initial_jax_to_numpy_seconds'] + result['initial_numpy_to_jax_seconds']
    print(f"Startup / final transitions [ms]: {1000*startup:.3f} / {1000*result['final_jax_to_numpy_seconds']:.3f}")
    print('Flight time [s]:', float(np.sum(result['sigma_new'])))


def print_ite(log_ite_all):
    columns = list(log_ite_all)
    labels = [key + " [ms]" if key.startswith("T-") else key for key in columns]
    width = max(14, max(map(len, labels), default=0) + 2)
    if log_ite_all.get("Note", [None])[-1] == "Start":
        print("".join(label.ljust(width) for label in labels))
    values = []
    for key in columns:
        value = log_ite_all[key]
        if isinstance(value, list):
            value = value[-1]
        if hasattr(value, "shape") and np.size(value) == 1:
            value = float(value)
        if isinstance(value, numbers.Real):
            if key.startswith("T-"):
                value = f"{1000.0 * float(value):.3f}"
            elif not isinstance(value, numbers.Integral):
                value = f"{value:.2f}"
        values.append(str(value).ljust(width))
    print("".join(values))
