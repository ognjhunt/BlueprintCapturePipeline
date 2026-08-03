#!/usr/bin/env python3
"""Create an isolated, exact PyChrono development runtime.

PyChrono's official binary route is conda, and the Python package does not
publish normal ``importlib.metadata`` distribution metadata.  This bootstrap
therefore binds the exact conda record and verifies a real ``pychrono.core``
import.  On platforms where the package's OpenMP runtime is not loaded by the
dynamic loader, it is preloaded process-globally before the import.

The receipt is development evidence only.  It does not establish a stable
granular operating point, qualification, production eligibility, or R7.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PYCHRONO_VERSION = "10.0.0"
PYTHON_VERSION = "3.12"
PYCHRONO_CHANNEL = "https://conda.anaconda.org/projectchrono/label/release"
CONDA_FORGE_CHANNEL = "https://conda.anaconda.org/conda-forge"
SCHEMA_VERSION = "measurement_chrono_development_environment.v1"
ROOT = Path(__file__).resolve().parents[1]


class ChronoBootstrapError(RuntimeError):
    pass


def _run(argv: Sequence[str]) -> None:
    completed = subprocess.run(  # nosec B603 - explicit argv, no shell
        list(argv),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "command_failed"
        raise ChronoBootstrapError(detail)


def _environment_python(environment: Path) -> Path:
    candidates = (environment / "bin/python", environment / "python.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.absolute()
    raise ChronoBootstrapError("chrono_environment_python_missing")


def _conda_record(environment: Path) -> dict[str, Any]:
    records = sorted((environment / "conda-meta").glob("pychrono-*.json"))
    if len(records) != 1:
        raise ChronoBootstrapError("chrono_conda_record_not_unique")
    try:
        record = json.loads(records[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ChronoBootstrapError("chrono_conda_record_invalid") from exc
    if not isinstance(record, Mapping):
        raise ChronoBootstrapError("chrono_conda_record_invalid")
    required = {"name", "version", "build", "channel", "subdir"}
    if not required <= set(record):
        raise ChronoBootstrapError("chrono_conda_record_incomplete")
    if record["name"] != "pychrono" or record["version"] != PYCHRONO_VERSION:
        raise ChronoBootstrapError("chrono_conda_record_version_mismatch")
    subdir = str(record["subdir"])
    channel = str(record["channel"]).rstrip("/")
    if subdir and channel.endswith(f"/{subdir}"):
        channel = channel[: -(len(subdir) + 1)]
    if channel != PYCHRONO_CHANNEL:
        raise ChronoBootstrapError("chrono_conda_record_channel_mismatch")
    return {
        "package_build": str(record["build"]),
        "package_channel": channel,
        "package_subdir": subdir,
    }


def _verify_import(worker_python: Path) -> dict[str, Any]:
    verification = subprocess.run(  # nosec B603 - exact interpreter and source
        [
            str(worker_python),
            "-c",
            (
                "import ctypes,json,platform,sys; from pathlib import Path; "
                "p=Path(sys.prefix); "
                "cs=[p/'lib/libiomp5.dylib',p/'lib/libiomp5.so',"
                "p/'Library/bin/libiomp5md.dll']; "
                "o=next((x for x in cs if x.is_file()),None); "
                "o and ctypes.CDLL(str(o),mode=getattr(ctypes,'RTLD_GLOBAL',0)); "
                "import pychrono.core as chrono; chrono.ChSystemSMC(); "
                "print(json.dumps({'python_version':platform.python_version(),"
                "'openmp_library':str(o) if o else None,'import_verified':True}))"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if verification.returncode != 0:
        raise ChronoBootstrapError(
            verification.stderr.strip() or "chrono_bootstrap_import_verification_failed"
        )
    try:
        observed = json.loads(verification.stdout)
    except json.JSONDecodeError as exc:
        raise ChronoBootstrapError("chrono_bootstrap_import_output_invalid") from exc
    if observed.get("import_verified") is not True:
        raise ChronoBootstrapError("chrono_bootstrap_import_not_verified")
    if not str(observed.get("python_version", "")).startswith(f"{PYTHON_VERSION}."):
        raise ChronoBootstrapError("chrono_bootstrap_python_version_mismatch")
    openmp = observed.get("openmp_library")
    if openmp is not None:
        openmp_path = Path(str(openmp)).absolute()
        try:
            openmp_path.relative_to(worker_python.parents[1])
        except ValueError as exc:
            raise ChronoBootstrapError("chrono_bootstrap_openmp_outside_environment") from exc
        openmp = str(openmp_path)
    return {
        "python_version": str(observed["python_version"]),
        "openmp_library": openmp,
        "openmp_preload_used": openmp is not None,
    }


def inspect_environment(*, environment: Path, conda: Path) -> dict[str, object]:
    environment = environment.expanduser().absolute()
    conda = conda.expanduser().absolute()
    if not conda.is_file():
        raise ChronoBootstrapError("chrono_bootstrap_conda_invalid")
    worker_python = _environment_python(environment)
    record = _conda_record(environment)
    observed = _verify_import(worker_python)
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "pychrono_version": PYCHRONO_VERSION,
        **record,
        **observed,
        "worker_python": str(worker_python),
        "conda_executable": str(conda),
        "environment": str(environment),
        "runtime_scope": "isolated_external_conda",
        "package_metadata_source": "conda-meta",
        "development_only": True,
        "granular_benchmark_established_by_environment": False,
        "production_route_eligible": False,
        "r7_admission": False,
    }
    result["bootstrap_receipt_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return result


def bootstrap(*, environment: Path, conda: Path) -> dict[str, object]:
    environment = environment.expanduser().absolute()
    conda = conda.expanduser().absolute()
    if not conda.is_file():
        raise ChronoBootstrapError("chrono_bootstrap_conda_invalid")
    protected = {Path("/"), Path.home().resolve(), ROOT.resolve(), ROOT.parent.resolve()}
    if environment.resolve() in protected or environment.exists():
        raise ChronoBootstrapError("chrono_bootstrap_environment_must_be_new_explicit_path")
    environment.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(conda),
            "create",
            "--yes",
            "--prefix",
            str(environment),
            "--override-channels",
            "--strict-channel-priority",
            "--channel",
            PYCHRONO_CHANNEL,
            "--channel",
            CONDA_FORGE_CHANNEL,
            f"python={PYTHON_VERSION}",
            f"pychrono={PYCHRONO_VERSION}",
        ]
    )
    return inspect_environment(environment=environment, conda=conda)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--conda", type=Path, required=True)
    parser.add_argument("--inspect-existing", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = (
            inspect_environment(environment=args.environment, conda=args.conda)
            if args.inspect_existing
            else bootstrap(environment=args.environment, conda=args.conda)
        )
    except (ChronoBootstrapError, OSError) as exc:
        print(f"[measurement-chrono-bootstrap] ERROR {exc}", file=sys.stderr)
        return 1
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
