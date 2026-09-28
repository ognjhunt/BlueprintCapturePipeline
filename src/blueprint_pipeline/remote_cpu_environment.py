"""Record the interpreter environment an episode compile depends on (plan 14 §5, the host census).

One module measures both sides of remote-worker parity: the control-plane host, through the chain
preflight, and later the worker.  ``environment_digest`` covers behaviour, never build strings:

- the CPython release (``sys.version_info``);
- golden digests of level-6 zlib and raw deflate over a fixed corpus (a compiled packet's zip
  members are deflated at level 6, after a level-6 compressibility probe);
- a golden digest of the float32 ``exp`` and sigmoid and float64 quaternion norms that NuRec
  conversion computes, which can differ by SIMD path;
- the versions of the distributions a compile loads;
- the CPU class: the machine, numpy's SIMD view and the CPU flags.

``sys.version``, ``libc_ver`` and the zlib version strings are recorded as informational only: two
zlib builds that deflate identically must not look like different environments.  This module
measures; it decides nothing and writes nothing.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

ENVIRONMENT_SCHEMA_VERSION = "remote_cpu_environment.v1"
CPUINFO_PATH = Path("/proc/cpuinfo")
# The distributions a compile loads, measured on 2026-09-28 by running the compiler and worker
# tests under an audit hook (plan 14, Facts).  pxr and msgpack load lazily, so plan 14 task 4.2
# re-derives this set from a real fixture compile.
COMPILE_DISTRIBUTIONS = (
    "attrs", "defusedxml", "jsonschema", "msgpack", "numpy", "packaging", "pillow", "pydantic",
    "pydantic-core", "referencing", "rpds-py", "usd-core", "webcolors",
)
DIGESTED_FIELDS = ("python_version_info", "golden_deflate", "golden_simd", "distributions", "cpu_class")


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def golden_corpus() -> bytes:
    """A fixed corpus of manifest-like rows and hash-derived noise, identical on every machine."""

    rows = [
        json.dumps({
            "index": index,
            "member": f"members/{index % 97:03d}.bin",
            "sha256": hashlib.sha256(b"remote-cpu-golden-row:%d" % index).hexdigest(),
            "size_bytes": (index * 7919) % 65536,
        }, sort_keys=True)
        for index in range(2048)
    ]
    noise = b"".join(hashlib.sha256(b"remote-cpu-golden-noise:%d" % index).digest() for index in range(1024))
    return ("\n".join(rows) + "\n").encode("ascii") + noise


def golden_deflate() -> dict[str, str]:
    """Digests of the level-6 zlib probe and of level-6 raw deflate (what a zip member holds)."""

    corpus = golden_corpus()
    compressor = zlib.compressobj(6, zlib.DEFLATED, -15)
    return {
        "corpus_digest": _digest(corpus),
        "zlib_level6_digest": _digest(zlib.compress(corpus, 6)),
        "raw_deflate_level6_digest": _digest(compressor.compress(corpus) + compressor.flush()),
    }


def golden_simd() -> dict[str, str] | None:
    """Digests of a fixed input and of NuRec's float math over it; ``None`` without numpy.

    The input is built from integer arithmetic and one correctly rounded division, so it is the
    same everywhere; the output follows ``particlefield_usd``: float32 ``exp`` of scales, the
    float32 sigmoid of opacities, and float64 quaternion norms cast to float32.
    """

    try:
        import numpy as np
    except ImportError:
        return None
    steps = np.arange(4096, dtype=np.int64)
    values = ((steps * 7919) % 40001 - 20000).astype(np.float32) / np.float32(997.0)
    quaternions = ((steps * 104729) % 2001 / 1000.0 + 0.5).astype(np.float64).reshape(1024, 4)
    exponentials = np.exp(values).astype(np.float32)
    sigmoid = (1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))).astype(np.float32)
    unit = (quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)).astype(np.float32)
    return {
        "input_digest": _digest(values.tobytes() + quaternions.tobytes()),
        "output_digest": _digest(exponentials.tobytes() + sigmoid.tobytes() + unit.tobytes()),
    }


def numpy_simd() -> dict[str, list[str]] | None:
    """numpy's compiled baseline, its dispatch targets, and the features this CPU enables."""

    try:
        from numpy._core import _multiarray_umath as umath
    except ImportError:
        try:  # numpy 1.x
            from numpy.core import _multiarray_umath as umath
        except ImportError:
            return None
    return {
        "baseline": sorted(getattr(umath, "__cpu_baseline__", [])),
        "dispatch": sorted(getattr(umath, "__cpu_dispatch__", [])),
        "enabled": sorted(name for name, on in getattr(umath, "__cpu_features__", {}).items() if on),
    }


def cpu_flags(cpuinfo_path: Path) -> tuple[list[str], str | None]:
    """The first processor's flag set from ``/proc/cpuinfo`` (``Features`` on arm64); unread is ``None``."""

    try:
        text = Path(cpuinfo_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], None
    for line in text.splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip() in {"flags", "Features"}:
            return sorted(set(value.split())), str(cpuinfo_path)
    return [], str(cpuinfo_path)


def distribution_versions(names: Sequence[str] = COMPILE_DISTRIBUTIONS) -> list[dict[str, str | None]]:
    rows = []
    for name in sorted(set(names)):
        try:
            version: str | None = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        rows.append({"name": name, "version": version})
    return rows


def environment_digest(record: Mapping[str, Any]) -> str:
    """The digest dispatch compares: ``DIGESTED_FIELDS`` of a record, never its informational fields."""

    return canonical_digest({name: record[name] for name in DIGESTED_FIELDS})


def environment_record(*, cpuinfo_path: Path | None = None,
                       distributions: Sequence[str] = COMPILE_DISTRIBUTIONS) -> dict[str, Any]:
    """Measure this interpreter.

    ``cpu_class`` digests the machine, numpy's SIMD view and the CPU flags; it is ``None`` when
    either is unmeasured, and an unmeasured class is never qualified.
    """

    flags, source = cpu_flags(CPUINFO_PATH if cpuinfo_path is None else Path(cpuinfo_path))
    simd = numpy_simd()
    record: dict[str, Any] = {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "python_version_info": list(sys.version_info),
        "golden_deflate": golden_deflate(),
        "golden_simd": golden_simd(),
        "distributions": distribution_versions(distributions),
        "cpu_class": canonical_digest({"machine": platform.machine(), "numpy_simd": simd, "flags": flags})
        if simd is not None and source is not None else None,
        "informational": {
            "python_version": sys.version, "implementation": platform.python_implementation(),
            "machine": platform.machine(), "libc": list(platform.libc_ver()),
            "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION, "zlib_compile_version": zlib.ZLIB_VERSION,
            "numpy_simd": simd, "cpu_flags": flags, "cpu_flags_source": source, "executable": sys.executable,
        },
    }
    record["environment_digest"] = environment_digest(record)
    return record
