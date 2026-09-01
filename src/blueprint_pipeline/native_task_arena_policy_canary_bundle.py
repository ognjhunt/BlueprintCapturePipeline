"""Build one provider bundle for the paired internal policy canary session."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import shutil
import stat
import tempfile
import zipfile
from typing import Any

from .adp009d_policy_provisioning import (
    CHECKPOINT_INVENTORY_STAGED_NAME,
    POLICY_EXECUTION_SPEC_STAGED_NAME,
    build_provisioning_script,
)
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .native_task_arena_bundle import (
    POLICY_RUNTIME_ROOT_MODULE_NAMES,
    _sha256,
    _write_zip_file,
    build_native_task_arena_bundle,
)
from .native_task_arena_execution_contract import POLICY_RUNTIME_MODULE_NAMES
from .native_task_arena_policy_canary_session import (
    CANDIDATE_IDS,
    CLAIM_CEILING,
    EPISODES_PER_POLICY,
    LEARNED_ROLLOUT_COUNT,
    PROVIDER_BUNDLE_SCHEMA_VERSION,
    PROVIDER_RESULT_FILENAME,
    RUN_KIND,
    validate_provider_bundle,
    validate_runtime_input_manifest,
    validate_session_authority,
)
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from .task_evaluation_canary_hotfix_overlay import (
    apply_canary_hotfix_overlay,
    canary_hotfix_execution_release,
    verify_canary_hotfix_overlay,
)


EXECUTION_AUTHORITY = "internal_policy_canary_unqualified"


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("policy_canary_bundle_input_invalid")
    return value


def _bound_record_path(record: Mapping[str, Any], *, code: str) -> Path:
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise ValueError(code)
    return path


def _validate_spec(value: Mapping[str, Any], *, candidate: str) -> dict[str, Any]:
    spec = json.loads(json.dumps(value, allow_nan=False))
    rights = spec.get("candidate_rights_binding")
    if (
        spec.get("schema_version")
        != "native_task_arena_policy_canary_execution_spec.v1"
        or spec.get("candidate_id") != candidate
        or spec.get("execution_authority") != EXECUTION_AUTHORITY
        or spec.get("claim_ceiling") != CLAIM_CEILING
        or spec.get("ranking_permitted") is not False
        or spec.get("qualification_permitted") is not False
        or spec.get("scene_promotion_permitted") is not False
        or not isinstance(spec.get("policy_endpoint"), Mapping)
        or not isinstance(spec.get("policy_spec"), Mapping)
        or not isinstance(rights, Mapping)
        or not str(spec.get("checkpoint_digest") or "").startswith("sha256:")
        or not str(spec.get("runtime_identity_digest") or "").startswith("sha256:")
        or spec.get("execution_spec_digest")
        != canonical_digest(spec, digest_field="execution_spec_digest")
    ):
        raise ValueError("policy_canary_execution_spec_invalid")
    return spec


def _entrypoint() -> str:
    return f'''#!/usr/bin/env bash
set +e
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${{BLUEPRINT_ADP_ARENA_OUTPUT_DIR:-$RUNTIME_DIR/../runtime_output}}"
export RUNTIME_DIR OUT_DIR
mkdir -p "$OUT_DIR"
SOURCE_RECEIPT="$RUNTIME_DIR/native_task_runtime_sources/native_task_runtime_source_packet.v1.json"
SOURCE_PACKET="$RUNTIME_DIR/native_task_runtime_sources/native_task_runtime_sources.zip"
cd "$RUNTIME_DIR"
/isaac-sim/python.sh -m blueprint_pipeline.native_task_runtime_source_provision \
  --source-receipt "$SOURCE_RECEIPT" --source-packet "$SOURCE_PACKET" \
  --extraction-dir "$RUNTIME_DIR/provisioned_runtime_sources" \
  --output "$OUT_DIR/native_task_runtime_source_provisioning.v1.json" \
  --simulator-root /isaac-sim
if [ $? -ne 0 ]; then exit 2; fi

teardown_servers() {{
  original_rc=$?
  trap - EXIT INT TERM HUP
  for candidate in pi05_droid groot_n17_droid; do
    /isaac-sim/python.sh "$RUNTIME_DIR/adp009d_policy_server_worker.py" \
      --terminate-ready-server \
      --receipt "$OUT_DIR/adp009d_policy_server_receipt.$candidate.json" \
      --result "$OUT_DIR/{PROVIDER_RESULT_FILENAME}" \
      --result-schema native_task_arena_policy_canary_session_result.v1 || true
  done
  exit $original_rc
}}
trap teardown_servers EXIT INT TERM HUP

cp "$RUNTIME_DIR/runtime_inputs/policy_execution_spec.pi05_droid.json" \
  "$RUNTIME_DIR/{POLICY_EXECUTION_SPEC_STAGED_NAME}"
bash "$RUNTIME_DIR/adp009d_policy_provisioning.pi05_droid.sh" \
  >"$OUT_DIR/policy_provisioning.pi05_droid.log" 2>&1 || exit $?
cp "$RUNTIME_DIR/runtime_inputs/policy_execution_spec.groot_n17_droid.json" \
  "$RUNTIME_DIR/{POLICY_EXECUTION_SPEC_STAGED_NAME}"
bash "$RUNTIME_DIR/adp009d_policy_provisioning.groot_n17_droid.sh" \
  >"$OUT_DIR/policy_provisioning.groot_n17_droid.log" 2>&1 || exit $?

/isaac-sim/python.sh "$RUNTIME_DIR/adp_arena_provider_runner.py"
runner_rc=$?
if [ ! -f "$OUT_DIR/{PROVIDER_RESULT_FILENAME}" ]; then
  /isaac-sim/python.sh - "$OUT_DIR/{PROVIDER_RESULT_FILENAME}" "$runner_rc" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({{
  "schema_version": "native_task_arena_policy_canary_session_result.v1",
  "status": "blocked",
  "run_kind": "internal_policy_canary",
  "claim_ceiling": "diagnostic_policy_execution",
  "candidate_policy_queried": False,
  "blockers": ["policy_canary_worker_failed_without_result"],
  "worker_exit_code": int(sys.argv[2]),
}}, sort_keys=True) + "\\n", encoding="utf-8")
PY
fi
exit $runner_rc
'''


def build_policy_canary_session_bundle(
    *,
    job_dir: str | Path,
    packet_dir: str | Path,
    runtime_source_packet_receipt: str | Path,
    runtime_input_manifest_path: str | Path,
    session_authority_path: str | Path,
    pi05_execution_spec_path: str | Path,
    groot_execution_spec_path: str | Path,
    pi05_checkpoint_inventory_path: str | Path,
    implementation_commit: str,
    hotfix_overlay_path: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Package both frozen policy servers and the warm 20-episode worker."""

    inputs = validate_runtime_input_manifest(_read(runtime_input_manifest_path))
    authority = validate_session_authority(_read(session_authority_path))
    hotfix_manifest = (
        verify_canary_hotfix_overlay(hotfix_overlay_path)
        if hotfix_overlay_path is not None
        else None
    )
    execution_release = (
        canary_hotfix_execution_release(hotfix_manifest)
        if hotfix_manifest is not None
        else None
    )
    if authority.get("execution_release") != execution_release:
        raise ValueError("policy_canary_bundle_execution_release_mismatch")
    if authority["runtime_inputs_digest"] != inputs["runtime_inputs_digest"]:
        raise ValueError("policy_canary_bundle_authority_input_mismatch")
    packet_receipt_path = (
        Path(packet_dir).expanduser().resolve()
        / "native_task_arena_packet_receipt.v1.json"
    )
    if _bound_record_path(
        inputs["base_native_packet"], code="policy_canary_base_packet_record_invalid"
    ) != packet_receipt_path:
        raise ValueError("policy_canary_base_packet_record_mismatch")
    runtime_source_path = _bound_record_path(
        inputs["runtime_source"], code="policy_canary_runtime_source_record_invalid"
    )
    if runtime_source_path != Path(runtime_source_packet_receipt).expanduser().resolve():
        raise ValueError("policy_canary_runtime_source_record_mismatch")
    construction_result_path = _bound_record_path(
        inputs["construction_result"],
        code="policy_canary_construction_result_record_invalid",
    )
    specs = {
        "pi05_droid": _validate_spec(
            _read(pi05_execution_spec_path), candidate="pi05_droid"
        ),
        "groot_n17_droid": _validate_spec(
            _read(groot_execution_spec_path), candidate="groot_n17_droid"
        ),
    }
    package = Path(__file__).resolve().parent
    runtime_modules = [package / name for name in POLICY_RUNTIME_MODULE_NAMES]
    runtime_modules.extend(
        (
            package / "native_task_arena_policy_worker.py",
            package / "native_task_arena_policy_canary_session.py",
        )
    )
    job = Path(job_dir).expanduser().resolve()
    base = build_native_task_arena_bundle(
        job_dir=job / "base",
        packet_dir=packet_dir,
        worker_source=package / "native_task_arena_policy_canary_worker.py",
        runtime_module_sources=sorted(set(runtime_modules)),
        implementation_commit=implementation_commit,
        execution_mode="construction_canary",
        expected_output_filename=PROVIDER_RESULT_FILENAME,
        container_image=NATIVE_TASK_ARENA_IMAGE,
        runtime_source_packet_receipt=runtime_source_packet_receipt,
        bound_runtime_inputs={
            "policy_canary_runtime_inputs.json": runtime_input_manifest_path,
            "policy_canary_session_authority.json": session_authority_path,
            "policy_execution_spec.pi05_droid.json": pi05_execution_spec_path,
            "policy_execution_spec.groot_n17_droid.json": groot_execution_spec_path,
            "native_task_arena_construction_result.v1.json": construction_result_path,
        },
        generated_at=generated_at,
    )
    with tempfile.TemporaryDirectory(prefix="policy-canary-session-bundle-") as raw:
        root = Path(raw)
        with zipfile.ZipFile(base["bundle_path"]) as archive:
            archive.extractall(root)
        runtime = root / "provider_runtime"
        for candidate in CANDIDATE_IDS:
            script = runtime / f"adp009d_policy_provisioning.{candidate}.sh"
            script.write_text(build_provisioning_script(candidate), encoding="utf-8")
            script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        for name in POLICY_RUNTIME_ROOT_MODULE_NAMES:
            shutil.copy2(package / name, runtime / name)
        hotfix_application = (
            apply_canary_hotfix_overlay(
                archive_path=hotfix_overlay_path,
                provider_runtime_root=runtime,
            )
            if hotfix_overlay_path is not None
            else None
        )
        shutil.copy2(
            pi05_checkpoint_inventory_path,
            runtime / CHECKPOINT_INVENTORY_STAGED_NAME,
        )
        entrypoint = runtime / "run_adp_arena_provider_runtime.sh"
        entrypoint.write_text(_entrypoint(), encoding="utf-8")
        entrypoint.chmod(entrypoint.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        manifest_path = runtime / "adp_arena_provider_manifest.json"
        manifest = _read(manifest_path)
        manifest.update(
            {
                "schema_version": PROVIDER_BUNDLE_SCHEMA_VERSION,
                "execution_mode": "internal_policy_canary_paired_session",
                "run_kind": RUN_KIND,
                "claim_ceiling": CLAIM_CEILING,
                "candidate_ids": list(CANDIDATE_IDS),
                "episodes_per_policy": EPISODES_PER_POLICY,
                "learned_policy_rollout_count": LEARNED_ROLLOUT_COUNT,
                "maximum_provider_allocations": 1,
                "retry_cap": 0,
                "authority_digest": authority["authority_digest"],
                "runtime_inputs_digest": inputs["runtime_inputs_digest"],
                "candidate_policy_queried": False,
                "expected_output_filename": PROVIDER_RESULT_FILENAME,
                "scene_promotion_authorized": False,
                "official_ranking_authorized": False,
                "execution_release": execution_release,
                "hotfix_application": hotfix_application,
                "execution_spec_digests": {
                    candidate: spec["execution_spec_digest"]
                    for candidate, spec in specs.items()
                },
                "input_digest": "",
            }
        )
        manifest["input_digest"] = canonical_digest(
            manifest, digest_field="input_digest"
        )
        write_json(manifest_path, manifest)
        job.mkdir(parents=True, exist_ok=True)
        bundle_path = job / "native_task_arena_policy_canary_session_bundle.zip"
        with zipfile.ZipFile(bundle_path, "w", allowZip64=True) as archive:
            for source in sorted(root.rglob("*")):
                if source.is_file():
                    _write_zip_file(
                        archive,
                        source=source,
                        archive_path=source.relative_to(root).as_posix(),
                    )
    receipt = {
        **manifest,
        "bundle_path": str(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "bundle_sha256": _sha256(bundle_path),
    }
    receipt_path = job / "native_task_arena_policy_canary_session_bundle_receipt.v1.json"
    write_json(receipt_path, receipt)
    validate_provider_bundle(receipt, authority=authority)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--runtime-source-packet-receipt", required=True)
    parser.add_argument("--runtime-input-manifest-path", required=True)
    parser.add_argument("--session-authority-path", required=True)
    parser.add_argument("--pi05-execution-spec-path", required=True)
    parser.add_argument("--groot-execution-spec-path", required=True)
    parser.add_argument("--pi05-checkpoint-inventory-path", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--hotfix-overlay")
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    receipt = build_policy_canary_session_bundle(
        job_dir=args.job_dir,
        packet_dir=args.packet_dir,
        runtime_source_packet_receipt=args.runtime_source_packet_receipt,
        runtime_input_manifest_path=args.runtime_input_manifest_path,
        session_authority_path=args.session_authority_path,
        pi05_execution_spec_path=args.pi05_execution_spec_path,
        groot_execution_spec_path=args.groot_execution_spec_path,
        pi05_checkpoint_inventory_path=args.pi05_checkpoint_inventory_path,
        implementation_commit=args.implementation_commit,
        hotfix_overlay_path=args.hotfix_overlay,
        generated_at=args.generated_at,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EXECUTION_AUTHORITY", "build_policy_canary_session_bundle", "main"]
