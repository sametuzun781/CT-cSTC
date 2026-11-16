"""Build unmodified, pinned upstream QOCO once, outside benchmark timing.

The pinned dependency retains its upstream licenses in .build/qoco_native/source.
The source archive is included in the separately installed ICORE runtime.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
from urllib.request import urlopen
from contextlib import contextmanager
from importlib.resources import files

QOCO_VERSION = "0.3.2"
QOCO_COMMIT = "598de4e1aeaa9c522b3bc5bb1a56f3d7287dc14f"
QOCO_URL = f"https://codeload.github.com/qoco-org/qoco/tar.gz/{QOCO_COMMIT}"
QOCO_SHA256 = "13e6481fc547450bd8872bbc867e977c11c9ac624e2a2703929c0d73daf04fb5"
ROOT = Path(__file__).resolve().parents[2]


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


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def library_path(root=None):
    extension = ".dylib" if sys.platform == "darwin" else ".so"
    return Path(root or ROOT) / ".build/qoco_native/build/out" / ("libbenchmark_qoco" + extension)


def _run(command, logfile):
    result = subprocess.run(command, text=True, capture_output=True)
    with logfile.open("a") as stream:
        stream.write(" ".join(map(str, command)) + "\n" + result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"QOCO build failed; see {logfile}.\n" + result.stderr[-1800:])


def build_qoco(*, root=None, native_arch=True, force=False):
    """Build once while holding the repository's native-QOCO cache lock."""
    root = Path(root or ROOT).resolve()
    with build_lock(root / ".build/qoco_native"):
        return _build_qoco(root=root, native_arch=native_arch, force=force)


def _build_qoco(*, root=None, native_arch=True, force=False):
    """Return build metadata; reuse a checked binary on subsequent notebook runs.

    Linux and macOS require C/C++ compilers and CMake >=3.18. Windows users can
    run the same notebook in WSL. native_arch should match the Riccati build.
    """
    if sys.platform not in ("linux", "darwin"):
        raise RuntimeError("The native benchmark build supports Linux/macOS; use WSL on Windows.")
    root = Path(root or ROOT).resolve()
    cache = root / ".build/qoco_native"
    cache.mkdir(parents=True, exist_ok=True)
    local_cmake = Path(sys.executable).with_name("cmake")
    cmake = str(local_cmake) if local_cmake.is_file() else shutil.which("cmake")
    cc = shutil.which(os.environ.get("CC", "cc"))
    cxx = shutil.which(os.environ.get("CXX", "c++"))
    if not (cmake and cc and cxx):
        raise RuntimeError("Install C/C++ compilers (Linux build-essential or macOS Command Line Tools) and CMake.")
    flags = ["-O3", "-DNDEBUG"]
    if native_arch:
        flags.append("-mcpu=native" if platform.machine().lower() in ("arm64", "aarch64") else "-march=native")
    bridge = root / "solvers/qoco_native/bridge.c"
    inputs = dict(version=QOCO_VERSION, commit=QOCO_COMMIT, archive_sha256=QOCO_SHA256,
                  bridge_sha256=_sha(bridge), build_recipe_sha256=_sha(__file__),
                  build_root=str(cache), platform=sys.platform, flags=flags, machine=platform.machine(),
                  compiler=subprocess.check_output([cc, "--version"], text=True).splitlines()[0],
                  cxx_compiler=subprocess.check_output([cxx, "--version"], text=True).splitlines()[0])
    binary, manifest = library_path(root), cache / "build.json"
    if not force and manifest.is_file() and binary.is_file():
        previous = json.loads(manifest.read_text())
        if previous.get("inputs") == inputs and previous.get("binary_sha256") == _sha(binary):
            return {**previous, "library": str(binary), "cache_hit": True}
    if (cache / "build").is_dir():
        shutil.rmtree(cache / "build")
    archive = cache / "qoco-598de4e1.tar.gz"
    if not archive.is_file():
        bundled = files("icore").joinpath("vendor/qoco/qoco-source.tar.gz")
        if bundled.is_file():
            archive.write_bytes(bundled.read_bytes())
    if not archive.is_file():
        try:
            with urlopen(QOCO_URL, timeout=60) as response:
                downloaded = response.read()
        except OSError as error:
            raise RuntimeError("Downloading the pinned public QOCO dependency requires internet access.") from error
        if hashlib.sha256(downloaded).hexdigest() != QOCO_SHA256:
            raise RuntimeError("QOCO source archive checksum does not match the pinned release.")
        archive.write_bytes(downloaded)
    if _sha(archive) != QOCO_SHA256:
        raise RuntimeError(f"QOCO archive checksum mismatch: remove {archive} and rerun the build.")
    source_parent = cache / "source"
    source = source_parent / ("qoco-" + QOCO_COMMIT)
    if source.is_dir():
        shutil.rmtree(source)
    source_parent.mkdir(exist_ok=True)
    with tarfile.open(archive) as package:
        # Python >=3.11.8 supports the safe extraction filter.
        package.extractall(source_parent, filter="data")
    project = cache / "project"
    project.mkdir(exist_ok=True)
    (project / "CMakeLists.txt").write_text(
        'cmake_minimum_required(VERSION 3.18)\nproject(benchmark_qoco LANGUAGES C CXX)\n'
        'set(CMAKE_POSITION_INDEPENDENT_CODE ON)\n'
        f'add_subdirectory("{source.as_posix()}" upstream)\n'
        f'add_library(benchmark_qoco SHARED "{bridge.as_posix()}")\n'
        'target_compile_features(benchmark_qoco PRIVATE c_std_11)\n'
        'target_link_libraries(benchmark_qoco PRIVATE qocostatic)\n'
        'set_target_properties(benchmark_qoco PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/out")\n')
    logfile = cache / "build.log"
    logfile.write_text("")
    configure = [cmake, "-S", str(project), "-B", str(cache / "build"),
        "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_C_COMPILER={cc}", f"-DCMAKE_CXX_COMPILER={cxx}",
        f"-DCMAKE_C_FLAGS_RELEASE={' '.join(flags)}", f"-DCMAKE_CXX_FLAGS_RELEASE={' '.join(flags)}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DQOCO_ALGEBRA_BACKEND=builtin",
        "-DBUILD_QOCO_DEMO=OFF", "-DBUILD_QOCO_BENCHMARK_RUNNER=OFF", "-DENABLE_TESTING=OFF"]
    _run(configure, logfile)
    _run([cmake, "--build", str(cache / "build"), "--target", "benchmark_qoco", "--parallel", "2"], logfile)
    commands = json.loads((cache / "build/compile_commands.json").read_text())
    if native_arch and any(flags[-1] not in item["command"] for item in commands):
        raise RuntimeError("The upstream build did not preserve the requested CPU optimization flag.")
    result = dict(inputs=inputs, library=str(binary), binary_sha256=_sha(binary),
                  source_url=QOCO_URL, source_directory=str(source),
                  license_directory=str(source), cache_hit=False)
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    return result
