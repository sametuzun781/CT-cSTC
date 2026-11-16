"""Run the notebook's fixed and adaptive phases without Jupyter."""
import argparse
import copy
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import jax
from ctcstc.application import BACKENDS, load_application, prepare_solver, run_scp
from ctcstc.utils import print_run_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=BACKENDS, default='icore_cpp')
    parser.add_argument('--ad', choices=('forward', 'reverse'), default='forward')
    parser.add_argument('--discretization', choices=('fast', 'native', 'jax', 'reference'), default='fast')
    parser.add_argument('--iterations', type=int, help='Override each phase budget.')
    parser.add_argument('--single-phase', action='store_true', help='Omit the second adaptive phase.')
    parser.add_argument('--audit', action='store_true', help='Check each ICORE subproblem against the original ChainQP KKT equations.')
    parser.add_argument('--output', type=Path, default=Path('runs/result.json'))
    args = parser.parse_args()
    if args.iterations is not None and args.iterations < 1:
        parser.error('--iterations must be positive')
    if args.audit and args.backend not in ('icore_c', 'icore_cpp', 'jax_icore'):
        parser.error('--audit requires an ICORE backend')
    started = perf_counter()
    ns, params, functions = load_application(args.backend, args.ad, args.discretization)
    problem = prepare_solver(ns, params, functions, args.backend)
    preparation = perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audits = []
    if args.audit:
        from tests.audit_chain import audit_chain
        solve = problem.engine.solve
        def checked(qp):
            value = solve(qp)
            audit = audit_chain(problem.engine.last_model, problem.engine.last_result,
                                tolerance=problem.engine.tolerance)
            audits.append(audit)
            if not audit['accepted']:
                raise RuntimeError(f'Original-model KKT check failed: {audit}')
            return value
        problem.engine.solve = checked
    receipt = dict(backend=args.backend, dynamics_ad=args.ad,
                   discretization=functions['disc'].discretization_backend,
                   preparation_seconds=preparation, phases=[], status='running')
    print(f'Preparation/build/JIT: {1000*preparation:.3f} ms', flush=True)
    try:
        result = None
        for phase in (('fixed',) if args.single_phase else ('fixed', 'adaptive')):
            p = copy.deepcopy(params)
            if phase == 'adaptive':
                p.update(X_last=result['X_new'], U_last=result['U_new'],
                         sigma_last=result['sigma_new'], adaptive_step=True, ite=15)
            result = run_scp(ns, p, functions, problem, args.iterations)
            output = dict(X=result['X_new'], U=result['U_new'], S=result['sigma_new'])
            snapshot = args.output.with_name(args.output.stem + '-' + phase + '.npz')
            np.savez_compressed(snapshot, **output)
            predicted = functions['integrate'](output['X'], output['U'], output['S'])
            defect = output['X'][:, 1:] - np.asarray(jax.device_get(predicted))
            receipt['phases'].append(dict(
                phase=phase, converged=bool(result['converged']), convex_calls=result['convex_solve_count'],
                final_nonlinear_cost=float(result['nlc'][-1]), final_linear_cost=float(result['lc'][-1]),
                final_nonlinear_defect_max=float(np.max(np.abs(defect))),
                total_scp_seconds=result['total_scp_seconds'],
                total_discretization_seconds=result['total_discretization_seconds'],
                total_convex_seconds=result['total_convex_seconds'],
                trial_timings=result['trial_timings'], snapshot=snapshot.name))
            print('Phase:', phase)
            print_run_summary(result, args.backend, functions['disc'])
        receipt['status'] = 'completed'
    except Exception as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        receipt['audits'] = dict(count=len(audits), all_accepted=all(a['accepted'] for a in audits))
        receipt['solver_records'] = [] if problem is None else problem.records
        args.output.write_text(json.dumps(receipt, indent=2) + '\n')
        if audits:
            args.output.with_suffix('.audits.jsonl').write_text(''.join(json.dumps(a) + '\n' for a in audits))
        if problem is not None:
            problem.close()
    print('Result:', args.output)


if __name__ == '__main__':
    main()
