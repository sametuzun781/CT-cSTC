"""Install the pinned ICORE runtime without copying solver sources into CT-cSTC."""
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import urlopen

COMMIT = 'a63ebffb005725a02d2c3e4247fb73aec453e590'
ARCHIVE_SHA256 = 'f0d3da4ab6fe13d36b5547607787dfb15472272e29e6f3863bea4af825dd1d49'
VERSION = '0.20.0+' + COMMIT[:12]
ROOT = Path(__file__).resolve().parents[1]


def stage_runtime(archive, destination):
    """Keep solver kernels, build inputs and licenses; omit benchmark-only code."""
    prefix = f'ICORE-{COMMIT}/'
    omitted = ('icore/benchmark.py', 'icore/_comparators.py', 'icore/vendor/README.md',
               'icore/vendor/hpipm/', 'icore/vendor/qoco/__init__.py',
               'icore/vendor/qoco/api.py', 'icore/vendor/qoco/bridge.c')
    licenses = ('LICENSE', 'LICENSE-NOTE.md', 'THIRD_PARTY_NOTICES.md')
    with tarfile.open(archive) as source:
        for member in source:
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            name = member.name[len(prefix):]
            if name in licenses:
                name = 'icore/' + name
            elif not name.startswith('icore/') or name.startswith(omitted):
                continue
            if '..' in PurePosixPath(name).parts:
                raise ValueError('Invalid source archive path')
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as stream, target.open('wb') as output:
                shutil.copyfileobj(stream, output)


def main():
    try:
        if version('icore') == VERSION:
            print(f'ICORE {VERSION} is already installed in {sys.prefix}')
            return
    except PackageNotFoundError:
        pass
    cache = ROOT / '.build' / 'dependencies'
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f'ICORE-{COMMIT}.tar.gz'
    if not archive.is_file():
        url = f'https://codeload.github.com/sametuzun781/ICORE/tar.gz/{COMMIT}'
        print(f'Downloading ICORE {COMMIT[:12]}', flush=True)
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if sha256(data).hexdigest() != ARCHIVE_SHA256:
            raise RuntimeError('ICORE archive checksum mismatch')
        archive.write_bytes(data)
    if sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError(f'ICORE archive checksum mismatch: remove {archive} and retry')
    with tempfile.TemporaryDirectory(dir=cache) as directory:
        project = Path(directory)
        stage_runtime(archive, project)
        (project / 'pyproject.toml').write_text(f'''[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "icore"
version = "{VERSION}"
description = "ICORE runtime pinned for CT-cSTC"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "scipy>=1.11"]

[project.urls]
Source = "https://github.com/sametuzun781/ICORE/tree/{COMMIT}"

[tool.setuptools.packages.find]
include = ["icore", "icore.*"]

[tool.setuptools.package-data]
icore = ["**/*"]
''')
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps',
                        '--wheel-dir', str(cache), str(project)], check=True)
    wheel = cache / f'icore-{VERSION}-py3-none-any.whl'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', str(wheel)], check=True)
    print(f'Installed ICORE {VERSION} in {sys.prefix}')


if __name__ == '__main__':
    main()
