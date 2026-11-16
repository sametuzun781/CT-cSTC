# Convex solver backends

`icore_c`, `icore_cpp`, and `jax_icore` use the separately installed ICORE
Solver/ChainQP interface (`python tools/install_icore.py`). `qoco_native` uses
the pinned QOCO source archive included in that installation. `cvxpy_qoco`
remains an optional modeling comparator.
Compilation is excluded from solve timing. No warm-start solution is injected.
