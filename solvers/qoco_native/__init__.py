"""Pinned official QOCO C API, with reusable sparse data updates."""
from .api import NativeQoco
from .build import build_qoco


class Solver(NativeQoco):
    def __init__(self, canonical, **options):
        if "library" not in options:
            options["library"] = build_qoco()["library"]
        super().__init__(canonical, **options)
