"""Record the interpreter environment an episode compile depends on (plan 14 §5, the host census).

One module measures both sides of remote-worker parity: the control-plane host, through the chain
preflight, and later the worker.  It records CPython; the zlib build with golden level-6 deflate
digests over a fixed corpus (a compiled packet's zip members are deflated at level 6, after a
level-6 compressibility probe); the numpy SIMD dispatch features and CPU flags that together form a
CPU class; and the versions of the distributions a compile loads.  Dispatch will require equality on
everything but the CPU class.  This module only measures: it decides nothing and writes nothing.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import zlib
from collections.abc import Sequence
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


def environment_record(*, cpuinfo_path: Path | None = None,
                       distributions: Sequence[str] = COMPILE_DISTRIBUTIONS) -> dict[str, Any]:
    """Measure this interpreter.

    ``environment_digest`` covers the interpreter, zlib and distribution versions: everything
    dispatch compares for equality.  ``cpu_class`` digests the machine, numpy's SIMD view and the CPU
    flags; it is ``None`` when either is unmeasured, and an unmeasured class is never qualified.
    ``executable`` is informational and in neither digest.
    """

    parity = {
        "interpreter": {
            "implementation": platform.python_implementation(), "version": sys.version,
            "machine": platform.machine(), "libc": list(platform.libc_ver()),
        },
        "zlib": {"runtime_version": zlib.ZLIB_RUNTIME_VERSION, "compile_version": zlib.ZLIB_VERSION,
                 **golden_deflate()},
        "distributions": distribution_versions(distributions),
    }
    flags, source = cpu_flags(CPUINFO_PATH if cpuinfo_path is None else Path(cpuinfo_path))
    cpu = {"numpy_simd": numpy_simd(), "flags": flags, "flags_source": source}
    measured = cpu["numpy_simd"] is not None and source is not None
    return {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        **parity,
        "environment_digest": canonical_digest(parity),
        "cpu": cpu,
        "cpu_class": canonical_digest({"machine": platform.machine(), "numpy_simd": cpu["numpy_simd"],
                                       "flags": flags}) if measured else None,
        "executable": sys.executable,
    }
