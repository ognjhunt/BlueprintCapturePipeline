"""Immutable input-bundle compiler for the ADP-009D native Isaac micro-check."""

from __future__ import annotations

import hashlib
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Any, Mapping

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest


PROBE_KIND = "adp009d-franka-native-microcheck"
SCHEMA_VERSION = "adp009d_native_microcheck_bundle.v1"
DEFAULT_IMAGE = (
    "nvcr.io/nvidia/isaac-sim:6.0.0-dev2@"
    "sha256:c3e7bef5b2bfdb9972807c34195206078372bf8c6cff79716be130a3fe3e9ce9"
)
ARENA_REVISION = "8b4a3a47fc53de23e8205089d71109a2e2348acd"
ARENA_TREE = "03f31f3dd56c56d00f24dbfb09711ec0ab345de8"
ISAAC_LAB_REVISION = "e57379c634b42db5a0fe9f754341be6e2a7c7c43"
ISAAC_LAB_TREE = "454115265327a80acabd07cbd36e10071fc0c065"
ASSET_BINDINGS = {
    "approved_can.usda": "sha256:61c2a03bef425803d82cc5ef24ced5b2ccb4160923c53bb10c6ad0e3f52532ec",
    "sage_collision.usd": "sha256:b265706c24f6a8ace3ee6743fd138583c4e21d83f61b99a06fd435e6ac2d6b41",
}
TARGET_COLLIDER_PRIM = "/Root/ZHQYGJJVAJYEYPTUKY888888"
ENTRYPOINT = """#!/usr/bin/env bash
set +e
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${BLUEPRINT_ADP_ARENA_OUTPUT_DIR:-$RUNTIME_DIR/../runtime_output}"
export BLUEPRINT_ADP009D_OUTPUT_DIR="$OUT_DIR"
mkdir -p "$OUT_DIR"
/isaac-sim/python.sh "$RUNTIME_DIR/adp_arena_provider_runner.py"
runner_rc=$?
if [ $runner_rc -ne 0 ] && [ ! -f "$OUT_DIR/adp009d_native_microcheck.json" ]; then
/isaac-sim/python.sh - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / "adp009d_native_microcheck.json").write_text(json.dumps({
    "schema_version": "adp009d_native_microcheck.v1",
    "status": "blocked",
    "blockers": ["adp009d_worker_failed_without_runtime_result"],
    "candidate_policy_queried": False,
    "candidate_outcomes_accessed": False
}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
fi
exit $runner_rc
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _overlay_text() -> str:
    target_name = TARGET_COLLIDER_PRIM.rsplit("/", 1)[-1]
    return f'''#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Root" (
    prepend references = @sage_collision.usd@</Root>
)
{{
    over "{target_name}" (
        active = false
    )
    {{
    }}
}}
'''


def _copy_bound_asset(source: Path, destination: Path, expected_digest: str) -> dict[str, Any]:
    if not source.is_file():
        raise ValueError(f"adp009d_bound_asset_missing:{destination.name}")
    observed = _sha256(source)
    if observed != expected_digest:
        raise ValueError(f"adp009d_bound_asset_digest_mismatch:{destination.name}")
    shutil.copy2(source, destination)
    return {
        "filename": destination.name,
        "sha256": observed,
        "size_bytes": destination.stat().st_size,
    }


def build_native_microcheck_bundle(
    *,
    job_dir: str | Path,
    approved_can_path: str | Path,
    sage_collision_path: str | Path,
    harness_manifest_path: str | Path,
    implementation_commit: str,
    generated_at: str | None = None,
    expected_asset_bindings: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compile a deterministic bundle from materialized, digest-verified bytes."""

    if len(implementation_commit) != 40 or any(ch not in "0123456789abcdef" for ch in implementation_commit):
        raise ValueError("adp009d_implementation_commit_invalid")
    job = Path(job_dir).expanduser().resolve()
    if job.exists():
        shutil.rmtree(job)
    runtime = job / "provider_runtime"
    assets = runtime / "assets"
    ensure_dir(assets)
    bindings = dict(expected_asset_bindings or ASSET_BINDINGS)
    sources = {
        "approved_can.usda": Path(approved_can_path).expanduser().resolve(),
        "sage_collision.usd": Path(sage_collision_path).expanduser().resolve(),
    }
    asset_rows = [
        _copy_bound_asset(sources[name], assets / name, bindings[name]) for name in sorted(bindings)
    ]
    overlay_path = assets / "sage_collision_overlay.usda"
    overlay_path.write_text(_overlay_text(), encoding="utf-8")
    asset_rows.append(
        {
            "filename": overlay_path.name,
            "sha256": _sha256(overlay_path),
            "size_bytes": overlay_path.stat().st_size,
            "composition_only": True,
            "sealed_source_mutated": False,
            "deactivated_source_prim": TARGET_COLLIDER_PRIM,
        }
    )

    source_dir = Path(__file__).resolve().parent
    shutil.copy2(source_dir / "adp009d_native_microcheck_worker.py", runtime / "adp_arena_provider_runner.py")
    shutil.copy2(source_dir / "adp009d_isaac_runtime.py", runtime / "adp009d_isaac_runtime.py")
    harness_source = Path(harness_manifest_path).expanduser().resolve()
    shutil.copy2(harness_source, runtime / "adp009d_franka_eval_harness_manifest.v1.json")
    _write_executable(runtime / "run_adp_arena_provider_runtime.sh", ENTRYPOINT)
    generated = generated_at or utc_now_iso()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated,
        "status": "ready",
        "program_id": "arm-decision-proof-v1",
        "probe_kind": PROBE_KIND,
        "implementation_commit": implementation_commit,
        "container_image": DEFAULT_IMAGE,
        "official_sources": {
            "isaac_lab_arena": {
                "repository": "https://github.com/isaac-sim/IsaacLab-Arena",
                "revision": ARENA_REVISION,
                "tree": ARENA_TREE,
                "version": "release/0.2.1",
            },
            "isaac_lab": {
                "repository": "https://github.com/isaac-sim/IsaacLab",
                "revision": ISAAC_LAB_REVISION,
                "tree": ISAAC_LAB_TREE,
                "version": "3.0.0 nested by Arena",
            },
        },
        "asset_bindings": asset_rows,
        "harness_manifest_sha256": _sha256(harness_source),
        "runtime_entrypoint": "provider_runtime/run_adp_arena_provider_runtime.sh",
        "expected_output_filename": "adp009d_native_microcheck.json",
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "private_data_uploaded": False,
        "retry_cap": 0,
        "provider_zero_required_after_return": True,
        "blockers": [],
    }
    manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
    write_json(runtime / "adp_arena_provider_manifest.json", manifest)
    bundle_path = job / "adp009d_native_microcheck_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", allowZip64=True) as archive:
        for path in sorted(runtime.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.relative_to(job).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_STORED)
    receipt = {
        **manifest,
        "bundle_path": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
    }
    write_json(job / "adp009d_native_microcheck_bundle_receipt.json", receipt)
    return receipt


__all__ = [
    "DEFAULT_IMAGE",
    "PROBE_KIND",
    "build_native_microcheck_bundle",
]
