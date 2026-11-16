"""Build the notebook's pinned HPIPM and QOCO dependencies, without downloads."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading

from ._blasfeo_build import (FileCheck, command_arguments, compiler_environment, file_fingerprint,
                            host_target, resolve_executable)


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
PINS = {
    "blasfeo": "0a3df8683c6bf9d98168d616e0c5d05938c9d51c",
    "hpipm": "16a8db59f595bda642f460aa0f62e308f4b76db3",
    "qoco": "598de4e1aeaa9c522b3bc5bb1a56f3d7287dc14f",
}
_BUILT = None
_THREAD_LOCK = threading.Lock()


@contextmanager
def _build_lock(path):
    """Prevent two notebook kernels from writing the same native build."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def build_info(library):
    """Read the configuration belonging to an actual loaded library path."""
    return json.loads((Path(library).parent / "configuration.json").read_text())


def build_comparators():
    """Reuse a checked build while its compiler, inputs and outputs are unchanged.

    Requires Linux or macOS, C/C++ compilers, CMake, and make. Windows users
    can run this notebook in WSL. Archives are included in the package.
    """
    global _BUILT
    if sys.platform not in ("linux", "darwin"):
        raise RuntimeError("Use Linux, macOS, or Windows with WSL for native builds.")
    commands = {}
    for name, variable in (("cc", "CC"), ("c++", "CXX"), ("cmake", "CMAKE")):
        arguments = command_arguments(os.environ.get(variable, name))
        executable = resolve_executable(arguments[0]) if arguments else None
        if executable is None:
            raise RuntimeError("Native build tool is missing: " + name)
        commands[name] = [executable, *arguments[1:]]
    environment = compiler_environment()
    key = (str(ROOT), sys.platform, platform.machine(), environment,
           tuple((name, tuple(command)) for name, command in commands.items()))
    with _THREAD_LOCK:
        if _BUILT is not None:
            previous_key, libraries, check = _BUILT
            if key == previous_key and check.unchanged():
                return libraries.copy()
        libraries, paths = _build_comparators(commands, environment)
        signature = file_fingerprint(paths)
        if signature is not None:
            _BUILT = key, libraries, FileCheck(paths, signature)
        return libraries.copy()


def _build_comparators(commands, environment):
    blasfeo_target = host_target()
    avx2 = blasfeo_target == "X64_INTEL_HASWELL"
    architecture = "-mcpu=native" if platform.machine().lower() in ("arm64", "aarch64") else "-march=native"
    flags = "-O3 -DNDEBUG -fno-fast-math " + architecture
    archives = {
        "blasfeo": VENDOR / "blasfeo/vendor" / ("blasfeo-" + PINS["blasfeo"] + ".tar.gz"),
        "hpipm": VENDOR / "hpipm/hpipm-16a8db59.tar.gz",
        "qoco": VENDOR / "qoco/qoco-598de4e1.tar.gz",
    }
    inputs = [Path(__file__), ROOT / "_blasfeo_build.py", *archives.values(),
              VENDOR / "hpipm/bridge.c", VENDOR / "qoco/bridge.c"]
    tool_files = [command[0] for command in commands.values()]
    tracked_inputs = (*inputs, *tool_files)
    before = file_fingerprint(tracked_inputs)
    compiler_versions = {name: subprocess.check_output(commands[name] + ["--version"], text=True)
                         for name in ("cc", "c++", "cmake")}
    configuration = dict(platform=sys.platform, machine=platform.machine(),
                         blasfeo_target=blasfeo_target, flags=flags, environment=environment,
                         commands=commands, versions=compiler_versions,
                         tool_files=file_fingerprint(tool_files))
    identity = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode())
    for path in inputs:
        identity.update(path.read_bytes())
    identity.update(repr((str(ROOT), sys.platform, platform.machine(), blasfeo_target, flags)).encode())
    cache = ROOT / ".build/comparators" / identity.hexdigest()[:20]
    extension = ".dylib" if sys.platform == "darwin" else ".so"
    libraries = {name: cache / ("lib" + name + "_bridge" + extension) for name in ("hpipm", "qoco")}
    with _build_lock(cache / "build.lock"):
        receipt = cache / "built.json"
        configuration_file = cache / "configuration.json"
        tracked = (*tracked_inputs, *libraries.values(), receipt, configuration_file)
        try:
            expected = json.loads(receipt.read_text())
        except (OSError, ValueError):
            expected = {}
        if not isinstance(expected, dict):
            expected = {}
        if all(path.is_file() and expected.get(name) == hashlib.sha256(path.read_bytes()).hexdigest()
               for name, path in libraries.items()):
            if before is None or before != file_fingerprint(tracked_inputs):
                raise RuntimeError("Comparator build inputs changed; call prepare again")
            configuration_file.write_text(json.dumps(configuration, sort_keys=True) + "\n")
            return libraries, tracked
        logfile = cache / "build.log"
        logfile.write_text("")

        def run(command):
            with logfile.open("a") as log:
                log.write(" ".join(map(str, command)) + "\n")
                log.flush()
                result = subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Native comparator build failed. Details: {logfile}")

        sources = cache / "source"
        sources.mkdir(exist_ok=True)
        for archive in archives.values():
            with tarfile.open(archive) as content:
                # Upstream experiments include author-local links; none is a
                # build input. Extract regular source files and directories.
                members = [m for m in content.getmembers() if m.isfile() or m.isdir()]
                content.extractall(sources, members=members, filter="data")
        install = cache / "install"
        cmake = commands["cmake"]
        common = ["-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
                  "-DCMAKE_C_COMPILER=" + ";".join(commands["cc"]),
                  "-DCMAKE_CXX_COMPILER=" + ";".join(commands["c++"]),
                  f"-DCMAKE_C_FLAGS_RELEASE={flags}", f"-DCMAKE_CXX_FLAGS_RELEASE={flags}",
                  f"-DCMAKE_INSTALL_PREFIX={install}", "-DBUILD_SHARED_LIBS=OFF"]
        for name in ("blasfeo", "hpipm"):
            source, build = sources / (name + "-" + PINS[name]), cache / ("build_" + name)
            options = (["-DBLASFEO_EXAMPLES=OFF", "-DBLASFEO_TESTING=OFF", "-DLA=HIGH_PERFORMANCE",
                        "-DTARGET=" + blasfeo_target]
                       if name == "blasfeo" else
                       [f"-DBLASFEO_PATH={install}", "-DHPIPM_TESTING=OFF",
                        "-DTARGET=" + ("AVX" if avx2 else "GENERIC")])
            run([*cmake, "-S", source, "-B", build, *common, *options])
            run([*cmake, "--build", build, "--parallel", "2"])
            run([*cmake, "--install", build])
        run([*commands["cc"], *flags.split(), "-std=c99", "-fPIC",
             "-dynamiclib" if sys.platform == "darwin" else "-shared",
             "-I", install / "include", VENDOR / "hpipm/bridge.c",
             install / "lib/libhpipm.a", install / "lib/libblasfeo.a", "-lm", "-o", libraries["hpipm"]])
        project = cache / "qoco_project"
        project.mkdir(exist_ok=True)
        source = sources / ("qoco-" + PINS["qoco"])
        (project / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.18)\nproject(qoco_bridge LANGUAGES C CXX)\n"
            "set(CMAKE_POSITION_INDEPENDENT_CODE ON)\n"
            f'add_subdirectory("{source.as_posix()}" upstream)\n'
            f'add_library(qoco_bridge SHARED "{(VENDOR / "qoco/bridge.c").as_posix()}")\n'
            "target_compile_features(qoco_bridge PRIVATE c_std_11)\n"
            "target_link_libraries(qoco_bridge PRIVATE qocostatic)\n"
            f'set_target_properties(qoco_bridge PROPERTIES LIBRARY_OUTPUT_DIRECTORY "{cache.as_posix()}")\n')
        build = cache / "build_qoco"
        run([*cmake, "-S", project, "-B", build, *common, "-DQOCO_ALGEBRA_BACKEND=builtin",
             "-DBUILD_QOCO_DEMO=OFF", "-DBUILD_QOCO_BENCHMARK_RUNNER=OFF", "-DENABLE_TESTING=OFF"])
        run([*cmake, "--build", build, "--target", "qoco_bridge", "--parallel", "2"])
        if before is None or before != file_fingerprint(tracked_inputs):
            raise RuntimeError("Comparator build inputs changed while building; call prepare again")
        receipt.write_text(json.dumps({name: hashlib.sha256(path.read_bytes()).hexdigest()
                                       for name, path in libraries.items()}))
        configuration_file.write_text(json.dumps(configuration, sort_keys=True) + "\n")
    return libraries, tracked
