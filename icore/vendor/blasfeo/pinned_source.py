"""Prepare the pinned BLASFEO source and its reviewed private kernel repairs.

This module reads the bundled archive only. It neither downloads source nor
changes an extracted upstream checkout.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile

PIN = "0a3df8683c6bf9d98168d616e0c5d05938c9d51c"
ARCHIVE_SHA256 = "731544773c6b1a4d638049a1d79d6dbd0e51a4df5d86467e36abfd67dcb95762"
SOURCE_SHA256 = "0d3e9704364153691b941584fedca70690e4f8f3595d39dc6aac526663da6877"
KERNEL = Path("kernel/avx2/kernel_sgemm_8x8_lib8.S")
KERNEL_BEFORE = "3be2fc0399ef59feaa489304ce41c585d26f59706fe8cd3cef8a591411a74864"
KERNEL_AFTER = "54b9c9fdcd804ff8a1622801d97ff488b4de68ab9401a4627221336dd564833c"
PATCH_SHA256 = "efea364688932ad909851a882defe7fc464f67a00332e8934ae114f2760bdaca"
LQ_KERNEL = Path("kernel/avx2/kernel_dgelqf_4_lib4.S")
LQ_KERNEL_BEFORE = "d6f42a02fb2424ab6298dcd15daadee6fba32fe5e0eead3badec708a3813273a"
LQ_KERNEL_AFTER = "bfa94114cc04a22153cee369c4c9b00daa477ea09a8deb411d32c51cf55880e6"
LQ_PATCH_SHA256 = "8a6f6ac26af052a604dbae1d0a523ad4ae516af57d4de59053b06949e10fe670"
LQ_POINTER_AFTER = "101bc95e721c80fe58c27f7b96a74ec59cc824aa85b6a7987062f7003aacd101"
LQ_POINTER_PATCH_SHA256 = "91bbb5471a38c79e47e7c5252b7f48900caaa68f20c08b244879c481420865ef"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(directory):
    return {
        str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob("*")) if p.is_file()
    }


def source_digest(source):
    result = hashlib.sha256()
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        result.update(str(path.relative_to(source)).encode() + b"\0")
        result.update(hashlib.sha256(path.read_bytes()).digest())
    return result.hexdigest()


def dependency_configuration(target, flags):
    """The bundled install receipt must describe every retained source repair."""
    return dict(commit=PIN, target=target, layout="PANELMAJ", la="HIGH_PERFORMANCE",
                dp_routines=1, sp_routines=1, blas_api=0, reference_api=0,
                ext_dep=0, ext_dep_malloc=0, flags=list(flags),
                archive_sha256=ARCHIVE_SHA256, sp_repair_sha256=PATCH_SHA256,
                lq_zero_tau_repair_sha256=LQ_PATCH_SHA256,
                lq_zero_tail_pointer_repair_sha256=LQ_POINTER_PATCH_SHA256)


def installed_dependency(install, receipt_path, configuration):
    """An old archive or a marker alone cannot certify the current repairs."""
    try:
        receipt = json.loads(receipt_path.read_text())
        if receipt["configuration"] != configuration:
            return None
        if receipt["installed_files"] != inventory(install):
            return None
        if not (install / "lib/libblasfeo.a").is_file():
            return None
        if "#define ICORE_BUNDLED_SP_ROUTINES 1" not in (install / "include/blasfeo_target.h").read_text():
            return None
        return receipt
    except (OSError, ValueError, KeyError, TypeError):
        return None


def validate_inputs(root):
    """Validate actual input bytes before accepting any compiled dependency."""
    expected = ((root / "vendor" / f"blasfeo-{PIN}.tar.gz", ARCHIVE_SHA256),
                (root / "patches/sp_8x8_negation.patch", PATCH_SHA256),
                (root / "patches/lq_12_zero_tau.patch", LQ_PATCH_SHA256),
                (root / "patches/lq_zero_tail_pointer.patch", LQ_POINTER_PATCH_SHA256))
    for path, digest in expected:
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"Pinned BLASFEO source input differs from the expected version: {path.name}")


def repair_sp_kernel(source, patch_file):
    """Apply exactly the retained NEG_ACC repair, without a patch dependency."""
    target = source / KERNEL
    if sha(target) != KERNEL_BEFORE or sha(patch_file) != PATCH_SHA256:
        raise ValueError("Pinned SP source or repair patch differs from the expected version")
    before = target.read_text()
    after = before
    for register in range(3, 8):
        old_source = 2 if register < 7 else 3
        old = f"\tvxorps\t\t%ymm15, %ymm{old_source}, %ymm{register}"
        new = f"\tvxorps\t\t%ymm15, %ymm{register}, %ymm{register}"
        if after.count(old) != 3:
            raise ValueError("Unexpected SP NEG_ACC register sequence")
        after = after.replace(old, new)
    old = "\t.float -1.0\n" * 8
    if after.count(old) != 1:
        raise ValueError("Unexpected SP NEG_ACC sign mask")
    after = after.replace(old, "\t.float -0.0\n" * 8)
    after = after.replace("// { -1 -1 -1 -1 -1 -1 -1 -1 }", "// sign bits for NEG_ACC")
    actual_patch = "".join(
        difflib.unified_diff(
            before.splitlines(True),
            after.splitlines(True),
            fromfile="a/" + KERNEL.as_posix(),
            tofile="b/" + KERNEL.as_posix(),
        )
    )
    if actual_patch != patch_file.read_text():
        raise ValueError("SP repair does not match the retained patch")
    if hashlib.sha256(after.encode()).hexdigest() != KERNEL_AFTER:
        raise ValueError("Repaired SP source differs from the expected version")
    target.write_text(after)


def repair_lq_kernel(source, patch_file):
    """Write T's diagonal when a zero Householder tail sets tau to zero.

    The 12-reflector kernels clear only the first 4-by-4 block of T. A later
    zero-tail reflector must overwrite its own diagonal, or the previous
    panel's tau can be reused by the trailing block update. Four-reflector
    kernels already clear their full T and are unchanged.
    """
    target = source / LQ_KERNEL
    if sha(target) != LQ_KERNEL_BEFORE or sha(patch_file) != LQ_PATCH_SHA256:
        raise ValueError("Pinned LQ source or repair patch differs from the expected version")
    before = target.read_text()
    lines, output, function = before.splitlines(True), [], ""
    counts = {"kernel_dgelqf_dlarft12_12_lib4": 0,
              "kernel_dgelqf_pd_dlarft12_12_lib4": 0}
    for index, line in enumerate(lines):
        declaration = re.match(r"// void (\w+)\(", line)
        if declaration:
            function = declaration.group(1)
        output.append(line)
        if function in counts and re.match(r"\s*vmovsd\s+%xmm14,\s*\d+\(%r12\)\s*$", line):
            end = next(i for i in range(index + 1, len(lines)) if lines[i].strip() == "102:")
            stores = [value for value in lines[index + 1:end]
                      if re.search(r"vmovsd\s+%xmm\d+,.*%r13", value)]
            if len(stores) != 1:
                raise ValueError("Unexpected LQ reflector diagonal sequence")
            address = stores[0].split(",", 1)[1].split("//")[0].strip()
            output.append("\tvmovsd\t\t\t%xmm14, " + address + " // zero T diagonal when tau is zero\n")
            counts[function] += 1
    if set(counts.values()) != {12}:
        raise ValueError("Unexpected LQ zero-tail reflector count")
    after = "".join(output)
    actual_patch = "".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile="a/" + LQ_KERNEL.as_posix(), tofile="b/" + LQ_KERNEL.as_posix()))
    if actual_patch != patch_file.read_text() or hashlib.sha256(after.encode()).hexdigest() != LQ_KERNEL_AFTER:
        raise ValueError("LQ repair differs from the retained patch or expected source")
    target.write_text(after)


def repair_lq_pointer(source, patch_file):
    """Reset the advanced matrix pointer on every Haswell zero-tail branch.

    The nonzero branch already resets r11 before the shared reflector work.
    The zero branch must use the identical origin (including later 4-column
    blocks), even though tau is zero: its preliminary dot products otherwise
    access unrelated capacity and can read outside a tight allocation.
    """
    target = source / LQ_KERNEL
    if sha(target) != LQ_KERNEL_AFTER or sha(patch_file) != LQ_POINTER_PATCH_SHA256:
        raise ValueError("Pinned LQ pointer source or repair differs from the expected version")
    before = target.read_text()
    lines, output, function = before.splitlines(True), [], ""
    expected = {"kernel_dgelqf_dlarft12_12_lib4": 12,
                "kernel_dgelqf_dlarft4_12_lib4": 4,
                "kernel_dgelqf_dlarft4_8_lib4": 4,
                "kernel_dgelqf_dlarft4_4_lib4": 4,
                "kernel_dgelqf_pd_dlarft12_12_lib4": 12,
                "kernel_dgelqf_pd_dlarft4_8_lib4": 4,
                "kernel_dgelqf_pd_dlarft4_4_lib4": 4}
    counts = dict.fromkeys(expected, 0)
    for index, line in enumerate(lines):
        declaration = re.match(r"// void (\w+)\(", line)
        if declaration:
            function = declaration.group(1)
        if function in expected and re.match(r"\s*jmp\s+102f\s*$", line):
            start = next(i for i in range(index + 1, len(lines)) if lines[i].strip() == "101:") + 1
            reset = [lines[start]]
            if lines[start + 1].lstrip().startswith("addq"):
                reset.append(lines[start + 1])
            wanted = ["\tmovq\tARG2, %r11 // D\n"]
            block = counts[function] // 4
            if expected[function] == 12 and block:
                wanted.append(f"\taddq\t$ {128 * block}, %r11\n")
            if reset != wanted:
                raise ValueError("Unexpected LQ reflector pointer origin")
            output.extend(reset)
            counts[function] += 1
        output.append(line)
    if counts != expected:
        raise ValueError("Unexpected LQ zero-tail pointer count")
    after = "".join(output)
    actual_patch = "".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile="a/" + LQ_KERNEL.as_posix(), tofile="b/" + LQ_KERNEL.as_posix()))
    if actual_patch != patch_file.read_text() or hashlib.sha256(after.encode()).hexdigest() != LQ_POINTER_AFTER:
        raise ValueError("LQ pointer repair differs from the retained patch or expected source")
    target.write_text(after)


def prepare(root, build):
    """Extract a clean private build tree; never reuse possibly stale objects."""
    archive = root / "vendor" / f"blasfeo-{PIN}.tar.gz"
    patch = root / "patches" / "sp_8x8_negation.patch"
    lq_patch = root / "patches/lq_12_zero_tau.patch"
    pointer_patch = root / "patches/lq_zero_tail_pointer.patch"
    if not archive.is_file():
        raise RuntimeError(
            "Bundled BLASFEO archive is missing; this build does not download source"
        )
    if sha(archive) != ARCHIVE_SHA256:
        raise ValueError("Bundled BLASFEO archive differs from the pinned source")
    validate_inputs(root)
    private_source = build / "blasfeo"
    if private_source.is_symlink():
        raise ValueError("Private BLASFEO build directory must not be a symlink")
    with tempfile.TemporaryDirectory(prefix="blasfeo-source-", dir=build) as temporary:
        temporary = Path(temporary)
        expected_root = f"blasfeo-{PIN}"
        with tarfile.open(archive, mode="r:gz") as bundle:
            for member in bundle.getmembers():
                path = Path(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not path.parts
                    or path.parts[0] != expected_root
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError("Unexpected member in pinned BLASFEO archive")
            bundle.extractall(temporary, filter="data")
        source = temporary / expected_root
        if source_digest(source) != SOURCE_SHA256:
            raise ValueError("Extracted BLASFEO source differs from the pinned tree")
        repair_sp_kernel(source, patch)
        repair_lq_kernel(source, lq_patch)
        repair_lq_pointer(source, pointer_patch)
        if private_source.exists():
            shutil.rmtree(private_source)
        source.rename(private_source)
    return {
        "commit": PIN,
        "archive_sha256": ARCHIVE_SHA256,
        "upstream_source_sha256": SOURCE_SHA256,
        "changed_numeric_source": [KERNEL.as_posix(), LQ_KERNEL.as_posix()],
        "patch": str(patch.relative_to(root)),
        "patch_sha256": PATCH_SHA256,
        "repaired_kernel_sha256": KERNEL_AFTER,
        "lq_patch": str(lq_patch.relative_to(root)),
        "lq_patch_sha256": LQ_PATCH_SHA256,
        "repaired_lq_tau_kernel_sha256": LQ_KERNEL_AFTER,
        "lq_pointer_patch": str(pointer_patch.relative_to(root)),
        "lq_pointer_patch_sha256": LQ_POINTER_PATCH_SHA256,
        "repaired_lq_kernel_sha256": LQ_POINTER_AFTER,
        "offline": True,
    }
