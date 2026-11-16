# Convex solver backends

`icore_c`, `icore_cpp`, and `jax_icore` use the bundled ICORE 0.20.0 public
Solver/ChainQP interface. `qoco_native` uses the pinned original QOCO source,
bundled offline with ICORE. `cvxpy_qoco` remains an optional modeling comparator.
Compilation is excluded from solve timing. No warm-start solution is injected.
