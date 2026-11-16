"""Select or build the bundled BLASFEO dependency for this machine.

The distributed archive is for Linux/AVX2. Other hosts compile the same
pinned, repaired sources locally; no solver or dependency is downloaded.
"""

from contextlib import contextmanager
import ctypes as ct
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile

from .vendor.blasfeo import pinned_source


FLAGS = ("-O3", "-DNDEBUG", "-fno-fast-math", "-fPIC")
COMPILER_ENV = (
    "PATH", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH",
    "COMPILER_PATH", "GCC_EXEC_PREFIX", "SDKROOT", "MACOSX_DEPLOYMENT_TARGET",
    "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "MAKEFLAGS",
    "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "SOURCE_DATE_EPOCH",
    "CMAKE_GENERATOR", "CMAKE_TOOLCHAIN_FILE", "CMAKE_PREFIX_PATH",
)


@lru_cache(maxsize=128)
def command_arguments(text):
    """Parsing is reusable; compiler files and search results are not."""
    return tuple(shlex.split(text))


@lru_cache(maxsize=128)
def _executable_candidates(program, search_path):
    if os.path.dirname(program):
        return (program,)
    if not search_path:
        return ()
    return tuple(dict.fromkeys(os.path.join(directory, program)
                               for directory in search_path.split(os.pathsep)))


def resolve_executable(program):
    """Search the current PATH, including newly added higher-priority tools.

    Only parsed path strings are cached. Every candidate is checked again;
    one stat supplies the existence and directory checks performed by which.
    """
    search_path = os.environ.get("PATH")
    if search_path is None:
        try:
            search_path = os.confstr("CS_PATH")
        except (AttributeError, ValueError):
            search_path = os.defpath
    candidates = _executable_candidates(program, search_path)
    if _native_executable_check is not None:
        result = _native_executable_check(len(candidates), _encoded_candidates(candidates))
        if result >= -1:
            return None if result == -1 else candidates[result]
    for candidate in candidates:
        try:
            mode = os.stat(candidate).st_mode
        except OSError:
            continue
        if not stat.S_ISDIR(mode) and os.access(candidate, os.X_OK):
            return candidate
    return None


@lru_cache(maxsize=128)
def _fingerprint_groups(paths):
    groups = {}
    for index, path in enumerate(paths):
        parent, name = os.path.split(path)
        groups.setdefault(parent or ".", []).append((index, name))
    return tuple((parent, tuple(entries)) for parent, entries in groups.items())


def file_fingerprint(paths):
    """Check every file, resolving shared parent directories only once.

    Directory handles live only for this check and are always closed. Files
    keep the same inode, mode, size and nanosecond timestamp checks; only the
    grouping of path strings is cached. Small groups use ordinary stat calls.
    """
    paths = tuple(map(os.fspath, paths))
    values = [None] * len(paths)
    relative = os.stat in os.supports_dir_fd
    try:
        for parent, entries in _fingerprint_groups(paths):
            descriptor = None
            if relative and len(entries) >= 8:
                try:
                    descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                except OSError:
                    # Search-only directory permissions can still permit stat.
                    pass
            try:
                for index, name in entries:
                    s = (os.stat(paths[index]) if descriptor is None
                         else os.stat(name, dir_fd=descriptor))
                    values[index] = (s.st_dev, s.st_ino, s.st_mode, s.st_size,
                                     s.st_mtime_ns, s.st_ctime_ns)
            finally:
                if descriptor is not None:
                    os.close(descriptor)
        return tuple(values)
    except OSError:
        return None



class _FileGroup(ct.Structure):
    _fields_ = [("parent", ct.c_char_p), ("count", ct.c_size_t),
                ("paths", ct.POINTER(ct.c_char_p)), ("names", ct.POINTER(ct.c_char_p)),
                ("expected", ct.POINTER(ct.c_uint64))]


_native_file_check = None
_native_executable_check = None


@lru_cache(maxsize=128)
def _encoded_candidates(candidates):
    # Only path strings are retained. Existence, mode and executable access
    # are checked again on every lookup, including earlier PATH entries.
    encoded = tuple(os.fsencode(p) for p in candidates)
    if any(b"\0" in p for p in encoded):
        raise ValueError("embedded null byte")
    return (ct.c_char_p * len(candidates))(*encoded)


def enable_file_checks(library):
    """Use an already loaded native library for batched stat calls."""
    global _native_file_check, _native_executable_check
    function = getattr(library, "icore_files_unchanged", None)
    if function is not None:
        function.argtypes = [ct.c_size_t, ct.POINTER(_FileGroup)]
        function.restype = ct.c_int
        _native_file_check = function
    function = getattr(library, "icore_executable_candidate", None)
    if function is not None:
        function.argtypes = [ct.c_size_t, ct.POINTER(ct.c_char_p)]
        function.restype = ct.c_ssize_t
        _native_executable_check = function


class FileCheck:
    """Retain paths and expected metadata, then recheck every file each time.

    The C helper avoids Python objects per stat result. It follows the same
    symlinks and compares the same six fields as file_fingerprint. Seconds and
    nanoseconds travel separately to avoid overflowing signed 64-bit time.
    """
    def __init__(self, paths, signature):
        self.paths = tuple(map(os.fspath, paths))
        self.signature = signature
        groups = []
        for parent, entries in _fingerprint_groups(self.paths):
            full = (ct.c_char_p * len(entries))(*(os.fsencode(self.paths[i]) for i, _ in entries))
            names = (ct.c_char_p * len(entries))(*(os.fsencode(name) for _, name in entries))
            values = []
            for i, _ in entries:
                device, inode, mode, size, modified, changed = signature[i]
                values.extend((device, inode, mode, size, *divmod(modified, 10**9),
                               *divmod(changed, 10**9)))
            expected = (ct.c_uint64 * len(values))(*values)
            groups.append(_FileGroup(os.fsencode(parent), len(entries), full, names, expected))
        # ctypes retains all pointed-to arrays and encoded strings through this
        # structure array; no directory handle or coefficient buffer is kept.
        self.groups = (_FileGroup * len(groups))(*groups)

    def unchanged(self):
        if _native_file_check is not None:
            result = _native_file_check(len(self.groups), self.groups)
            if result >= 0:
                return bool(result)
        return file_fingerprint(self.paths) == self.signature


def compiler_environment():
    return tuple((name, os.environ.get(name)) for name in COMPILER_ENV)


def _has_avx2():
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    try:
        if sys.platform == "darwin":
            features = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"],
                text=True, stderr=subprocess.DEVNULL).lower()
        else:
            features = Path("/proc/cpuinfo").read_text().lower()
    except (OSError, subprocess.CalledProcessError):
        return False
    return "avx2" in features and "fma" in features


def host_target():
    """Return the automatic target shared by ICORE and its comparators."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        selected = "X64_INTEL_HASWELL" if _has_avx2() else "GENERIC"
    elif machine in ("aarch64", "arm64"):
        # ARM assembly is not validated here; upstream reports a triangular
        # solve defect: https://github.com/giaf/blasfeo/issues/215
        selected = "GENERIC"
    else:
        selected = "GENERIC"
    return selected


def target():
    """A GENERIC override is useful for portability checks, never for HPIPM."""
    selected = host_target()
    requested = os.environ.get("ICORE_BLASFEO_TARGET", "auto")
    if requested not in ("auto", "GENERIC", selected):
        raise ValueError(f"ICORE_BLASFEO_TARGET must be auto, GENERIC, or {selected} on this host")
    return selected if requested == "auto" else requested


@contextmanager
def _lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def dependency(root):
    """Return (install directory, target); callers cache validated builds."""
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        raise RuntimeError("Native builds support Linux and macOS; use WSL on Windows.")
    root = Path(root)
    vendor = root / "vendor" / "blasfeo"
    selected = target()
    bundled = vendor / "install"
    pinned_source.validate_inputs(vendor)
    if (sys.platform.startswith("linux") and selected == "X64_INTEL_HASWELL"
            and pinned_source.installed_dependency(
                bundled, vendor / "dependency_receipt.json",
                pinned_source.dependency_configuration(selected, FLAGS)) is not None):
        return bundled, selected

    compiler = shlex.split(os.environ.get("CC", "cc"))
    executable = shutil.which(compiler[0]) if compiler else None
    if executable is None:
        raise RuntimeError("A C compiler is needed to build BLASFEO for this machine.")
    make = shutil.which("make")
    if make is None:
        raise RuntimeError("Install make to build BLASFEO, or set ICORE_BLASFEO=off.")
    source_files = (Path(__file__), vendor / "pinned_source.py",
                    vendor / "vendor" / f"blasfeo-{pinned_source.PIN}.tar.gz",
                    *sorted((vendor / "patches").glob("*.patch")))
    inputs = (*source_files, executable, make)
    before = file_fingerprint(inputs)
    configuration = dict(
        commit=pinned_source.PIN, target=selected, platform=sys.platform,
        machine=platform.machine(), compiler=compiler,
        compiler_version=subprocess.check_output([*compiler, "--version"], text=True),
        compiler_file=(os.path.realpath(executable), file_fingerprint((executable,))),
        make_file=(os.path.realpath(make), file_fingerprint((make,))),
        environment=compiler_environment(),
        flags=FLAGS, source=pinned_source.ARCHIVE_SHA256, repair=pinned_source.PATCH_SHA256,
        lq_repair=pinned_source.LQ_PATCH_SHA256,
        lq_pointer_repair=pinned_source.LQ_POINTER_PATCH_SHA256,
    )
    # Include actual input bytes: editing the archive/repair must enter the
    # source validator instead of accepting a previously compiled library.
    identity = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode())
    for path in source_files:
        identity.update(path.read_bytes())
    build = root.parent / ".build" / "blasfeo" / identity.hexdigest()[:20]
    install = build / "install"
    with _lock(build / "build.lock"):
        receipt = build / "built.json"
        try:
            recorded = json.loads(receipt.read_text())
            current = pinned_source.inventory(install)
            if (current and recorded == current
                    and (install / "lib/libblasfeo.a").is_file()):
                if before is None or before != file_fingerprint(inputs):
                    raise RuntimeError("BLASFEO build inputs changed; call build again")
                return install, selected
        except (OSError, ValueError):
            pass
        pinned_source.prepare(vendor, build)
        source = build / "blasfeo"
        (source / "Makefile.local").write_text("CFLAGS = " + " ".join(FLAGS) + "\n")
        command = [make, "-C", str(source), "-j2", "static_library",
                   f"TARGET={selected}", "OS=" + ("MAC" if sys.platform == "darwin" else "LINUX"),
                   "LA=HIGH_PERFORMANCE", "MF=PANELMAJ", "BLAS_API=0", "BLASFEO_REF_API=0",
                   "DP_ROUTINES=1", "SP_ROUTINES=1", "EXT_DEP=0", "EXT_DEP_MALLOC=0",
                   "CC=" + shlex.join(compiler)]
        logfile = build / "build.log"
        with logfile.open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"BLASFEO build failed. See {logfile}. "
                               "Set ICORE_BLASFEO=off to use the portable implementation.")
        with tempfile.TemporaryDirectory(prefix="install-", dir=build) as temporary:
            staging = Path(temporary) / "new"
            shutil.copytree(source / "include", staging / "include")
            (staging / "lib").mkdir()
            shutil.copy2(source / "lib/libblasfeo.a", staging / "lib")
            with (staging / "include/blasfeo_target.h").open("a") as header:
                header.write("\n#define ICORE_BUNDLED_SP_ROUTINES 1\n")
            previous = Path(temporary) / "previous"
            if before is None or before != file_fingerprint(inputs):
                raise RuntimeError("BLASFEO build inputs changed while building; call build again")
            if install.exists():
                install.rename(previous)
            try:
                staging.rename(install)
            except OSError:
                if previous.exists():
                    previous.rename(install)
                raise
        receipt.write_text(json.dumps(pinned_source.inventory(install), sort_keys=True) + "\n")
    return install, selected
