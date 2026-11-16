"""Build the same sources as C11 or C++17 into one native library."""

import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import threading
import tempfile

from ._blasfeo_build import (COMPILER_ENV, FileCheck, command_arguments, dependency,
                            file_fingerprint as _fingerprint, resolve_executable)


# A native workspace must not rehash the entire source/dependency tree or
# launch the compiler merely to discover that its library is already built.
# Keep the content-checked result while its inputs retain the same metadata.
# ctime and inode also catch replacement and edits that restore an old mtime.
_BUILDS = {}
_BUILD_LOCK = threading.Lock()
_COMPILER_ENV = (*COMPILER_ENV, "CC", "ICORE_BLASFEO_TARGET", "ICORE_LTO")


def build_info(backend="c", *, library=None):
    """Describe the library actually selected, outside a timed solve."""
    path = Path(build(backend) if library is None else library)
    receipt = json.loads(path.with_suffix(".json").read_text())
    identity = json.loads(receipt["identity"])
    return dict(compiler=identity["compiler"], target=identity["target"],
                blasfeo=bool(identity["packed_files"]),
                lto="-flto" in identity["command"])


def build(backend="c"):
    if backend not in ("c", "cpp"):
        raise ValueError("Native backend must be c or cpp")
    compiler = os.environ.get("CC" if backend == "c" else "CXX", "cc" if backend == "c" else "c++")
    arguments = command_arguments(compiler)
    executable = resolve_executable(arguments[0]) if arguments else None
    dependency_compiler = command_arguments(os.environ.get("CC", "cc"))
    dependency_executable = (executable if dependency_compiler == arguments else
                             resolve_executable(dependency_compiler[0]) if dependency_compiler else None)
    packed = os.environ.get("ICORE_BLASFEO", "auto")
    environment = tuple((name, os.environ.get(name)) for name in _COMPILER_ENV)
    key = (backend, compiler, executable, dependency_executable, packed, environment,
           sys.platform, platform.machine())
    with _BUILD_LOCK:
        cached = _BUILDS.get(key)
        if cached is not None:
            output, check = cached
            if check.unchanged():
                return output
        output, paths = _build(backend, compiler, packed, environment)
        signature = _fingerprint(paths)
        if signature is not None:
            _BUILDS[key] = output, FileCheck(paths, signature)
        return output


def _lto_flags(compiler, backend, cache, mac):
    """Enable cross-module optimization only when the actual linker supports it.

    This probe runs only on a build-cache miss. AUTO retains the ordinary
    compiler path on toolchains without an LTO linker/plugin. Floating-point
    reassociation and fast-math are never enabled.
    """
    mode = os.environ.get("ICORE_LTO", "auto")
    if mode not in ("auto", "on", "off"):
        raise ValueError("ICORE_LTO must be auto, on or off")
    if mode == "off":
        return []
    flags = ["-flto", "-fno-semantic-interposition"]
    with tempfile.TemporaryDirectory(prefix="lto-probe-", dir=cache) as directory:
        output = Path(directory) / ("probe.dylib" if mac else "probe.so")
        command = [*compiler, *flags, "-fPIC", "-dynamiclib" if mac else "-shared",
                   "-x", "c" if backend == "c" else "c++", "-", "-o", str(output)]
        result = subprocess.run(command, input="int icore_lto_probe(void){return 0;}\n",
                                text=True, capture_output=True)
    if result.returncode == 0:
        return flags
    if mode == "on":
        raise RuntimeError("ICORE_LTO=on requires compiler/linker LTO support:\n" + result.stderr)
    return []


def _build(backend, compiler_text, packed, environment):
    root = Path(__file__).resolve().parent
    cache = root.parent / ".build" / ("icore_" + backend)
    cache.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (cache / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        compiler = shlex.split(compiler_text)
        executable = shutil.which(compiler[0]) if compiler else None
        if executable is None:
            raise RuntimeError("Install a C/C++ compiler before using a native backend")
        source = root / "src"
        files = sorted(p for p in source.rglob("*") if p.suffix in (".c", ".h"))
        mac = sys.platform == "darwin"
        output = cache / ("libicore.dylib" if mac else "libicore.so")
        if packed not in ("auto", "off", "on"):
            raise ValueError("ICORE_BLASFEO must be auto, off or on")
        vendor = root / "vendor" / "blasfeo" / "install"
        target = "portable"
        if packed != "off":
            vendor, target = dependency(root)
        static = vendor / "lib" / "libblasfeo.a"
        packed_files = [static, *sorted((vendor / "include").glob("*.h"))] if packed != "off" else []
        dependency_source = root / "vendor" / "blasfeo"
        build_files = [Path(__file__), root / "_blasfeo_build.py", dependency_source / "pinned_source.py"]
        if packed != "off":
            build_files += [*sorted((dependency_source / "vendor").glob("*.tar.gz")),
                            *sorted((dependency_source / "patches").glob("*.patch"))]
            receipt = dependency_source / "dependency_receipt.json"
            if receipt.is_file():
                build_files.append(receipt)
        # Include directory entries so adding/removing headers also invalidates
        # the cached input inventory, including a newly installed dependency.
        directories = [source, *sorted(p for p in source.rglob("*") if p.is_dir())]
        # Watch the pinned source directory, not a generated build parent.
        # The installed archive, headers and their directories remain tracked.
        directories += [path for path in (root / "vendor", dependency_source, vendor,
                                          vendor / "lib", vendor / "include") if path.is_dir()]
        inputs = tuple(map(str, [*files, *packed_files, *directories,
                                Path(executable), *build_files]))
        dependency_compiler = shlex.split(os.environ.get("CC", "cc"))
        dependency_executable = shutil.which(dependency_compiler[0]) if dependency_compiler else None
        if packed != "off" and dependency_executable:
            inputs += (dependency_executable,)
        before = _fingerprint(inputs)
        arch = (
            "-mcpu=native"
            if platform.machine().lower() in ("arm64", "aarch64")
            else "-march=native"
        )
        lto_flags = _lto_flags(compiler, backend, cache, mac)
        command = [
            *compiler,
            *lto_flags,
            "-O3",
            arch,
            "-fPIC",
            "-dynamiclib" if mac else "-shared",
            "-x",
            "c" if backend == "c" else "c++",
            "-std=c11" if backend == "c" else "-std=c++17",
            "-I" + str(source),
            *(["-DICORE_LA_WITH_BLASFEO=1", "-I" + str(vendor / "include")] if packed_files else []),
            *[str(p) for p in files if p.suffix == ".c"],
            *(["-x", "none", str(static), *([] if mac else ["-Wl,--exclude-libs,ALL"])]
              if packed_files else []),
            "-lm",
            "-o",
            str(output),
        ]
        identity = dict(
            files={
                str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files
            },
            packed_files={
                os.path.relpath(p, root): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in packed_files
            },
            build_files={os.path.relpath(p, root): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in build_files},
            target=target,
            command=command,
            compiler=subprocess.check_output([*compiler, "--version"], text=True).splitlines()[0],
            compiler_file=(os.path.realpath(executable), _fingerprint((executable,))),
            environment=environment,
        )
        # dlopen may return a previously loaded library for the same pathname.
        # A distinct filename per build keeps existing solvers usable and makes
        # a new source/compiler configuration load the matching machine code.
        encoded = json.dumps(identity, sort_keys=True)
        tag = hashlib.sha256(encoded.encode()).hexdigest()[:20]
        output = output.with_stem(output.stem + "_" + tag)
        stamp = output.with_suffix(".json")
        try:
            receipt = json.loads(stamp.read_text())
            valid = (receipt["identity"] == encoded
                     and receipt["library"] == hashlib.sha256(output.read_bytes()).hexdigest())
        except (OSError, ValueError, KeyError, TypeError):
            valid = False
        if not valid:
            temporary = output.with_name(output.name + ".tmp")
            command[-1] = str(temporary)
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode:
                temporary.unlink(missing_ok=True)
                raise RuntimeError("ICORE build failed:\n" + result.stdout + result.stderr)
            if before is None or before != _fingerprint(inputs):
                temporary.unlink(missing_ok=True)
                raise RuntimeError("Native build inputs changed while building; call build again")
            os.replace(temporary, output)
            stamp.write_text(json.dumps(dict(identity=encoded,
                library=hashlib.sha256(output.read_bytes()).hexdigest())))
        if before is None or before != _fingerprint(inputs):
            raise RuntimeError("Native build inputs changed while building; call build again")
        paths = (*inputs, str(output), str(stamp))
        return str(output), paths
