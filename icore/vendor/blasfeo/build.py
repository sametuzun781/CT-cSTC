#!/usr/bin/env python3
"""Build the private DP+SP BLASFEO dependency and/or the optional icore_la module.

The pinned archive and kernel repairs are bundled; rebuilding needs no network.
No system install, HPIPM link, fast-math, native-CPU flag, or timing runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import pinned_source

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / "src"
PIN = pinned_source.PIN
FLAGS = ["-O3", "-DNDEBUG", "-fno-fast-math", "-fPIC"]
DEPENDENCY_CONFIG = pinned_source.dependency_configuration("X64_INTEL_HASWELL", FLAGS)


def installed_dependency(install, receipt_path):
    return pinned_source.installed_dependency(install, receipt_path, DEPENDENCY_CONFIG)


def install_dependency(private_source, build, install):
    """Publish headers and archive together after a successful private build."""
    if install.is_symlink():
        raise ValueError("Private BLASFEO install directory must not be a symlink")
    with tempfile.TemporaryDirectory(prefix="blasfeo-install-", dir=build) as temporary:
        temporary = Path(temporary)
        staging = temporary / "new"
        shutil.copytree(private_source / "include", staging / "include")
        (staging / "lib").mkdir()
        shutil.copy2(private_source / "lib" / "libblasfeo.a", staging / "lib")
        with (staging / "include" / "blasfeo_target.h").open("a") as target:
            target.write("\n#define ICORE_BUNDLED_SP_ROUTINES 1\n")
        previous = temporary / "previous"
        if install.exists():
            install.rename(previous)
        try:
            staging.rename(install)
        except OSError:
            if previous.exists():
                previous.rename(install)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dependency-only", action="store_true")
    parser.add_argument("--portable-only", action="store_true")
    parser.add_argument("--cpp-proof", action="store_true")
    parser.add_argument("--rebuild-dependency", action="store_true")
    parser.add_argument("--cpus", default="4")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    build = ROOT / "build"
    install = ROOT / "install"
    build.mkdir(exist_ok=True)
    affinity = ["taskset", "-c", args.cpus] if args.cpus else []

    def run(command, label, **kwargs):
        command = list(map(str, command))
        with (build / f"{label}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, **kwargs)
        print(label, "ok", flush=True)

    dependency_receipt = None
    if not args.portable_only:
        pinned_source.validate_inputs(ROOT)
        private_source = build / "blasfeo"
        dependency_path = ROOT / "dependency_receipt.json"
        dependency_receipt = installed_dependency(install, dependency_path)
        if args.rebuild_dependency or dependency_receipt is None:
            pinned_source.prepare(ROOT, build)
            # The archive remains unmodified. Only this private copy carries
            # the reviewed SP/LQ repairs and local Make configuration.
            (private_source / "Makefile.local").write_text("CFLAGS = " + " ".join(FLAGS) + "\n")
            run(
                affinity
                + [
                    "make",
                    "-C",
                    private_source,
                    f"-j{args.jobs}",
                    "static_library",
                    "TARGET=X64_INTEL_HASWELL",
                    "LA=HIGH_PERFORMANCE",
                    "MF=PANELMAJ",
                    "BLAS_API=0",
                    "BLASFEO_REF_API=0",
                    "DP_ROUTINES=1",
                    "SP_ROUTINES=1",
                    "EXT_DEP=0",
                    "EXT_DEP_MALLOC=0",
                    "CC=gcc",
                ],
                "blasfeo-build",
            )
            install_dependency(private_source, build, install)
            dependency_receipt = {
                "configuration": DEPENDENCY_CONFIG,
                "installed_files": pinned_source.inventory(install),
            }
            dependency_path.write_text(json.dumps(dependency_receipt, indent=2) + "\n")
        else:
            print("blasfeo-dependency current", flush=True)
    if not args.dependency_only:
        common = ["gcc", *FLAGS, "-std=c99", "-Wall", "-Wextra", "-Werror"]
        names = ["dense_la", "ocp_la"]
        sources = [SOURCE / (name + ".c") for name in names]
        portable_objects = [build / (name + "_portable.o") for name in names]
        for name, source, obj in zip(names, sources, portable_objects):
            run(affinity + common + ["-c", source, "-o", obj], name + "-portable-compile")
        run(
            ["ar", "rcs", build / "libicore_la_portable.a", *portable_objects], "module-portable-static"
        )
        run(
            affinity + common + ["-shared", *sources, "-lm", "-o", build / "libicore_la_portable.so"],
            "module-portable-shared",
        )
        if args.cpp_proof:
            run(
                affinity
                + ["g++", *FLAGS, "-std=c++17", "-Wall", "-Wextra", "-Werror"]
                + ["-shared", *sources, "-lm", "-o", build / "libicore_la_portable_cpp.so"],
                "module-portable-cpp-proof",
            )
        if not args.portable_only:
            bf = ["-DICORE_LA_WITH_BLASFEO=1", "-I", install / "include"]
            packed_objects = [build / (name + ".o") for name in names]
            for name, source, obj in zip(names, sources, packed_objects):
                run(affinity + common + bf + ["-c", source, "-o", obj], name + "-blasfeo-compile")
            run(["ar", "rcs", build / "libicore_la.a", *packed_objects], "module-blasfeo-static")
            run(
                affinity
                + common
                + bf
                + [
                    "-shared",
                    *sources,
                    install / "lib" / "libblasfeo.a",
                    "-Wl,--exclude-libs,ALL",
                    "-lm",
                    "-o",
                    build / "libicore_la.so",
                ],
                "module-blasfeo-shared",
            )
            if args.cpp_proof:
                run(
                    affinity
                    + ["g++", *FLAGS, "-std=c++17", "-Wall", "-Wextra", "-Werror"]
                    + bf
                    + [
                        "-shared",
                        *sources,
                        install / "lib" / "libblasfeo.a",
                        "-Wl,--exclude-libs,ALL",
                        "-lm",
                        "-o",
                        build / "libicore_la_cpp.so",
                    ],
                    "module-cpp-proof",
                )
    print("build complete", flush=True)


if __name__ == "__main__":
    main()
