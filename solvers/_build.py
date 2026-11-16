"""Serialize the optional official QOCO dependency build."""
from contextlib import contextmanager


@contextmanager
def build_lock(cache):
    import fcntl
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
