"""JAX preparation and timing; the IPM iteration lives in jax/jax_engine.py."""

from time import perf_counter
import jax
import jax.numpy as jnp
from ._native import CHAIN_FIELDS
from .model import ChainQP


def _chain_arguments(data):
    K, n = data["center"].shape
    d, p, l = (data[name].shape[1] for name in ("offset", "pair_offset", "local_offset"))
    shapes = (
        (K - 1, d, n),
        (K - 1, d, n),
        (K - 1, d),
        (K - 1, p, n),
        (K - 1, p, n),
        (K - 1, p),
        (K, l, n),
        (K, l),
        (K, n),
        (K, n),
        (K, n),
        (K - 1, d),
        (K, n),
        (K, n),
    )
    values = []
    for name, shape in zip(CHAIN_FIELDS, shapes):
        value = jnp.asarray(data[name], dtype=jnp.float64)
        if name == "l1_weight":
            value = jnp.broadcast_to(value, shape)
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        values.append(value)
    return ChainQP(*values), (1, K, n, d, p, l)


class Jax:
    def __init__(self, prepared, settings):
        jax.config.update("jax_enable_x64", True)
        self.settings = settings
        self.signature = None
        self._cache_key = None
        self.compile_seconds = 0.0
        self.update(prepared)

    def _compile(self, signature, function, arguments):
        key = signature, tuple(self.settings.items())
        if key != self._cache_key:
            started = perf_counter()
            self.function = function
            self.compiled = function.lower(arguments).compile()
            self.compile_seconds += perf_counter() - started
            self.signature = signature
            self._cache_key = key
        self.arguments = arguments

    def update_chain(self, data):
        from .jax.chain import make_chain_solver

        arguments, signature = _chain_arguments(data)
        function = make_chain_solver(**self.settings)
        self._compile(signature, function, arguments)

    def update(self, prepared):
        if prepared.chain:
            self.update_chain(prepared.arrays)
        else:
            from .jax.core import make_solver, pack

            model = prepared.layout
            function = make_solver(
                model.nlp,
                tuple(map(int, prepared.dims)),
                epigraph_controls=model.ne,
                **self.settings,
            )
            self._compile(prepared.signature, function, pack(prepared.arrays))

    def solve(self):
        started = perf_counter()
        result = self.compiled(self.arguments)
        jax.block_until_ready(result)
        elapsed = perf_counter() - started
        result = dict(result)
        result.update(native_seconds=None, jax_seconds=elapsed)
        if "factor_retries" not in result:
            result["factor_retries"] = result.get("qr_fallbacks", 0)
        return result

    def close(self):
        self.arguments = None
