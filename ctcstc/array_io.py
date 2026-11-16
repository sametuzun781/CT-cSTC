"""Explicit array boundaries; times are seconds, not asynchronous dispatch times.

Call block_until_ready at the end of the producing computation BEFORE to_host.
The defensive wait below is outside the transfer timer. No copy is requested
on CPU unless the library needs one. NumPy results must be treated as read-only.
"""
from time import perf_counter
import jax
import jax.numpy as jnp
import numpy as np

ARRAY_FLOW_VERSION = "compiled-solution-input-r2"


def to_host(values):
    """Fetch a ready pytree once and time only host materialization."""
    jax.block_until_ready(values)
    started = perf_counter()
    result = (jax.tree.map(np.asarray, values) if jax.default_backend() == 'cpu'
              else jax.device_get(values))
    return result, perf_counter() - started


def to_device(values, *, dtype=None):
    """Keep device arrays; time any required upload or explicit dtype conversion."""
    leaves = jax.tree.leaves(values)
    ready = all(isinstance(value, jax.Array) and
                (dtype is None or value.dtype == dtype) for value in leaves)
    if ready:
        return values, 0.0
    started = perf_counter()
    result = (jax.device_put(values) if dtype is None else
              jax.tree.map(lambda value: jnp.asarray(value, dtype=dtype), values))
    jax.block_until_ready(result)
    return result, perf_counter() - started
