"""Build the immutable native-Isaac provider packet for ADP-009B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
from typing import Any, Sequence
import zipfile

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .provider_runtime_bundle_contract import provider_runtime_contract_blockers


BUNDLE_SCHEMA_VERSION = "adp009b_simready_isaac_provider_bundle.v1"
DEFAULT_IMAGE = (
    "nvcr.io/nvidia/isaac-sim:6.0.1@"
    "sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9"
)
REQUIRED_NATIVE_FILES = (
    "adp009b_simready_native_probe_manifest.json",
    "drop_stage.usda",
    "isaac_gripper_stage.usda",
    "isaac_probe_spec.json",
    "isaac_slide_stage.usda",
    "isaac_tip_stage.usda",
    "scene/collision_and_replacement.usda",
)


ENTRYPOINT = r'''#!/usr/bin/env bash
set +e
BUNDLE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${BLUEPRINT_ISAAC_OUTPUT_DIR:-$BUNDLE_DIR/runtime_output}"
RESULT="$OUTPUT_DIR/isaac_runtime_result.json"
mkdir -p "$OUTPUT_DIR"
/isaac-sim/python.sh "$BUNDLE_DIR/provider_runtime/isaac_realistic_runtime_runner.py" \
  --spec "$BUNDLE_DIR/provider_runtime/native/isaac_probe_spec.json" \
  --output "$RESULT"
runner_rc=$?
write_missing_result() {
  if [ -s "$RESULT" ]; then return 0; fi
  /isaac-sim/python.sh - "$RESULT" "$runner_rc" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "adp009b_simready_isaac_result.v1",
    "status": "blocked_isaac_process_exited_without_result",
    "blockers": [f"isaac_runner_process_exited_without_runtime_result:{sys.argv[2]}"],
    "native_isaac_executed": False,
    "physical_success_established": False,
    "provider_zero_required_after_return": True
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}
write_missing_result
if [ ! -s "$RESULT" ]; then
  echo "blocked_isaac_process_exited_without_result" >&2
  exit 2
fi
exit "$runner_rc"
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("simready_isaac_bundle_json_not_object")
    return value


def _file_record(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _deterministic_zip(source_root: Path, output: Path) -> None:
    output_resolved = output.resolve()
    with zipfile.ZipFile(output, "w") as archive:
        for path in sorted(source_root.rglob("*")):
            if not path.is_file() or path.resolve() == output_resolved:
                continue
            info = zipfile.ZipInfo(
                path.relative_to(source_root).as_posix(),
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.create_system = 3
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def build_simready_isaac_bundle(
    *,
    probe_root: str | Path,
    job_dir: str | Path,
    worker_source: str | Path,
    source_commit_sha: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    source = Path(probe_root).expanduser().resolve()
    job = Path(job_dir).expanduser().resolve()
    worker = Path(worker_source).expanduser().resolve()
    if not source.is_dir() or not worker.is_file():
        raise ValueError("simready_isaac_bundle_source_missing")
    if len(source_commit_sha) != 40:
        raise ValueError("simready_isaac_bundle_source_commit_invalid")
    for relative in REQUIRED_NATIVE_FILES:
        if not (source / relative).is_file():
            raise ValueError(f"simready_isaac_bundle_native_file_missing:{relative}")
    manifest = _read_json(source / "adp009b_simready_native_probe_manifest.json")
    probe_spec = _read_json(source / "isaac_probe_spec.json")
    if (
        manifest.get("status") != "ready"
        or (manifest.get("isaac") or {}).get("status") != "frozen_not_executed"
        or probe_spec.get("status") != "frozen_before_execution"
        or (manifest.get("isaac") or {}).get("probe_spec_sha256")
        != _sha256(source / "isaac_probe_spec.json")
    ):
        raise ValueError("simready_isaac_bundle_probe_binding_invalid")
    if job.exists():
        shutil.rmtree(job)
    runtime = job / "provider_runtime"
    native = runtime / "native"
    ensure_dir(native)
    shutil.copytree(source, native, dirs_exist_ok=True)
    shutil.copy2(worker, runtime / "isaac_realistic_runtime_runner.py")
    _write_executable(runtime / "run_isaac_realistic_runtime.sh", ENTRYPOINT)
    shutil.copy2(source / "drop_stage.usda", runtime / "generated_site_scene.usda")
    shutil.copy2(source / "drop_stage.usda", runtime / "generated_site_scene.usd")
    generated = generated_at or utc_now_iso()
    common = {
        "schema_version": "adp009b_simready_isaac_placeholder_contract.v1",
        "generated_at": generated,
        "source_commit_sha": source_commit_sha,
        "probe_spec_sha256": _sha256(source / "isaac_probe_spec.json"),
        "status": "bounded_exact_scene_probe",
    }
    for name in (
        "scenario_eval_matrix.json",
        "camera_manifest.json",
        "episode_spec_manifest.json",
    ):
        write_json(runtime / name, {**common, "artifact": name.removesuffix(".json")})
    eval_manifest = {
        "schema_version": "isaac_provider_eval_manifest.v1",
        "generated_at": generated,
        "job_id": "adp009b-exact-simready-840313-ins160",
        "relative_paths": {
            "generated_site_scene_usda": "generated_site_scene.usda",
            "generated_site_scene_usd": "generated_site_scene.usd",
            "scenario_eval_matrix": "scenario_eval_matrix.json",
            "camera_manifest": "camera_manifest.json",
            "episode_spec_manifest": "episode_spec_manifest.json",
            "runtime_runner": "isaac_realistic_runtime_runner.py",
            "entrypoint": "run_isaac_realistic_runtime.sh",
        },
        "proof_boundaries": {
            "exact_simready_simulation_only": True,
            "bounded_gripper_proxy_is_not_robot_task_success": True,
            "physical_success_established": False,
        },
    }
    write_json(runtime / "isaac_provider_eval_manifest.json", eval_manifest)
    input_records = [
        _file_record(path, root=runtime)
        for path in sorted(native.rglob("*"))
        if path.is_file()
    ]
    readiness: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "ready",
        "source_commit_sha": source_commit_sha,
        "container_image": DEFAULT_IMAGE,
        "probe_spec_sha256": _sha256(source / "isaac_probe_spec.json"),
        "native_probe_manifest_sha256": _sha256(
            source / "adp009b_simready_native_probe_manifest.json"
        ),
        "input_files": input_records,
        "local_bundle_ready_for_remote_staging": True,
        "provider_zero_required_after_return": True,
        "retry_cap": 0,
        "blockers": [],
        "claim_ceiling": "native_isaac_exact_scene_simulation_only",
    }
    readiness["bundle_manifest_digest"] = canonical_digest(
        readiness, digest_field="bundle_manifest_digest"
    )
    write_json(job / "isaac_provider_bundle_readiness.json", readiness)
    write_json(runtime / "isaac_provider_bundle_readiness.json", readiness)
    contract_blockers = provider_runtime_contract_blockers(
        provider_bundle_kind="isaac",
        entrypoint_text=ENTRYPOINT,
        runner_text=(runtime / "isaac_realistic_runtime_runner.py").read_text(
            encoding="utf-8"
        ),
    )
    if contract_blockers:
        raise ValueError("simready_isaac_bundle_runtime_contract_invalid")
    bundle_path = job / "isaac_provider_runtime_bundle.zip"
    _deterministic_zip(job, bundle_path)
    with zipfile.ZipFile(bundle_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("simready_isaac_bundle_zip_invalid")
    receipt = {
        **readiness,
        "bundle_path": str(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "bundle_sha256": _sha256(bundle_path),
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(job / "adp009b_simready_isaac_bundle_receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--worker-source", type=Path, required=True)
    parser.add_argument("--source-commit-sha", required=True)
    args = parser.parse_args(argv)
    receipt = build_simready_isaac_bundle(
        probe_root=args.probe_root,
        job_dir=args.job_dir,
        worker_source=args.worker_source,
        source_commit_sha=args.source_commit_sha,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_IMAGE", "build_simready_isaac_bundle"]
