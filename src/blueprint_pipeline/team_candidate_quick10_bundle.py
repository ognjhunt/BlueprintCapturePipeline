"""Seal the provider bundle for ONE admitted team candidate's Quick-10 session.

Beta journey step: L5 team policy evaluation. The bundle reuses the warm
Quick-10 runtime, worker and static preflight of the frozen-pair canary
(``native_task_arena_policy_canary_bundle``) with three differences:

- one sealed execution spec, for the admitted team candidate, whose policy is
  the team's own service over authenticated HTTPS (``openpi_droid_https.v1``);
- no provisioning scripts, checkpoint inventory or local policy server: the
  entrypoint starts none and downloads no weights;
- the manifest names the environment variable that will hold the path of the
  bearer-token file, never the token. The token reaches the GPU host only
  through the provider's private runtime-secret channel at launch, and
  :func:`assert_secret_absent` refuses a bundle whose bytes contain it.

Strict per-cell controls are refused by name (they are paused by the owner,
and the admission refuses them too). The result's claim ceiling stays
``diagnostic_policy_execution``.
"""

from __future__ import annotations

import json
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .native_task_arena_bundle import POLICY_RUNTIME_ROOT_MODULE_NAMES, _sha256, _write_zip_file, build_native_task_arena_bundle
from .native_task_arena_execution_contract import POLICY_RUNTIME_MODULE_NAMES
from .native_task_arena_policy_canary_bundle import (
    CANARY_SHIPPED_PACKAGE_FILES,
    TEAM_RUNTIME_MODULE_NAMES,
    _bound_record_path,
    _read,
    stage_policy_contract_dependencies,
)
from .native_task_arena_policy_canary_session import (
    CLAIM_CEILING,
    EXECUTION_MODES,
    PROVIDER_BUNDLE_SCHEMA_VERSION,
    PROVIDER_RESULT_FILENAME,
    TEAM_RUN_KIND,
    validate_provider_bundle,
    validate_runtime_input_manifest,
    validate_session_authority,
)
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from .policy_canary_session_bundle_identity import TEAM_ENTRYPOINT_MARKER
from .policy_interface_binding import TEAM_HTTPS_INTERFACES
from .provider_runtime_import_closure import assert_provider_runtime_import_closure

EXECUTION_AUTHORITY = "team_candidate_evaluation_unqualified"
_SECRET_FILE_ENV = re.compile(r"^[A-Z][A-Z0-9_]{2,58}_FILE$")


class TeamCandidateBundleError(ValueError):
    """The team bundle cannot be sealed for these inputs."""


def _validate_team_spec(value: Mapping[str, Any], *, inputs: Mapping[str, Any]) -> dict[str, Any]:
    spec = json.loads(json.dumps(value, allow_nan=False))
    endpoint = spec.get("policy_endpoint")
    binding = inputs.get("team_candidate_admission") or {}
    if (
        spec.get("schema_version") != "native_task_arena_policy_canary_execution_spec.v1"
        or spec.get("candidate_id") != inputs["candidate_ids"][0]
        or spec.get("execution_authority") != EXECUTION_AUTHORITY
        or spec.get("claim_ceiling") != CLAIM_CEILING
        or spec.get("policy_interface_id") not in TEAM_HTTPS_INTERFACES
        or any(spec.get(key) is not False for key in (
            "ranking_permitted", "qualification_permitted", "scene_promotion_permitted"))
        or not isinstance(endpoint, Mapping) or set(endpoint) != {"credential_env"}
        or _SECRET_FILE_ENV.fullmatch(str(endpoint.get("credential_env") or "")) is None
        or spec.get("team_candidate_admission_digest") != binding.get("admission_digest")
        or spec.get("task_success_contract_digest") != inputs["task_success_contract_digest"]
        or spec.get("task_success_contract") != inputs["task_success_contract"]
        or spec.get("execution_spec_digest") != canonical_digest(spec, digest_field="execution_spec_digest")
    ):
        raise TeamCandidateBundleError("team_candidate_execution_spec_invalid")
    return spec


def _entrypoint(credential_env: str) -> str:
    if _SECRET_FILE_ENV.fullmatch(credential_env) is None:
        raise TeamCandidateBundleError("team_candidate_credential_env_invalid")
    return f'''#!/usr/bin/env bash
# {TEAM_ENTRYPOINT_MARKER}: the team's policy is reached over authenticated
# HTTPS. No checkpoint is downloaded and no local policy server starts.
set +e
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${{BLUEPRINT_ADP_ARENA_OUTPUT_DIR:-$RUNTIME_DIR/../runtime_output}}"
export RUNTIME_DIR OUT_DIR
mkdir -p "$OUT_DIR"
SOURCE_RECEIPT="$RUNTIME_DIR/native_task_runtime_sources/native_task_runtime_source_packet.v1.json"
SOURCE_PACKET="$RUNTIME_DIR/native_task_runtime_sources/native_task_runtime_sources.zip"
cd "$RUNTIME_DIR"

write_fallback_result() {{
  fallback_reason="$1"
  fallback_rc="$2"
  if [ ! -f "$OUT_DIR/{PROVIDER_RESULT_FILENAME}" ]; then
    /isaac-sim/python.sh - "$OUT_DIR/{PROVIDER_RESULT_FILENAME}" "$fallback_rc" "$fallback_reason" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({{
  "schema_version": "native_task_arena_policy_canary_session_result.v1",
  "status": "blocked",
  "run_kind": "{TEAM_RUN_KIND}",
  "claim_ceiling": "{CLAIM_CEILING}",
  "candidate_policy_queried": False,
  "blockers": [sys.argv[3]],
  "worker_exit_code": int(sys.argv[2]),
}}, sort_keys=True) + "\\n", encoding="utf-8")
PY
  fi
}}

finish_session() {{
  original_rc=$?
  trap - EXIT INT TERM HUP
  write_fallback_result policy_canary_entrypoint_failed_without_result "$original_rc"
  exit $original_rc
}}
trap finish_session EXIT INT TERM HUP

# This entrypoint replaces the base arena entrypoint, including its media
# bootstrap. Both tools must exist before any isolated simulator cell starts.
echo "BLUEPRINT_WAM_RUNTIME_PHASE:native_task_arena:media_toolchain:started"
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive apt-get update -qq >"$OUT_DIR/media_toolchain_install.log" 2>&1 && \\
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >>"$OUT_DIR/media_toolchain_install.log" 2>&1
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "BLUEPRINT_WAM_RUNTIME_PHASE:native_task_arena:media_toolchain:blocked"
  write_fallback_result policy_canary_media_toolchain_unavailable 86
  exit 86
fi
echo "BLUEPRINT_WAM_RUNTIME_PHASE:native_task_arena:media_toolchain:completed"

/isaac-sim/python.sh -m blueprint_pipeline.native_task_runtime_source_provision \\
  --source-receipt "$SOURCE_RECEIPT" --source-packet "$SOURCE_PACKET" \\
  --extraction-dir "$RUNTIME_DIR/provisioned_runtime_sources" \\
  --output "$OUT_DIR/native_task_runtime_source_provisioning.v1.json" \\
  --simulator-root /isaac-sim
source_rc=$?
if [ $source_rc -ne 0 ]; then
  write_fallback_result policy_canary_runtime_source_provision_failed "$source_rc"
  exit $source_rc
fi

/isaac-sim/python.sh - <<'PYINPUT'
import json, os
from pathlib import Path
from adp_arena_provider_runner import preflight_policy_canary_runtime_directory
receipt = preflight_policy_canary_runtime_directory(Path(os.environ["RUNTIME_DIR"]))
print(json.dumps(receipt, sort_keys=True))
raise SystemExit(0 if receipt["status"] == "passed" else 2)
PYINPUT
input_rc=$?
if [ $input_rc -ne 0 ]; then
  write_fallback_result policy_canary_input_preflight_failed "$input_rc"
  exit $input_rc
fi

strict_controls=$(/isaac-sim/python.sh - <<'PYSTRICT'
import json, os
from pathlib import Path
from blueprint_pipeline.native_policy_canary_control_gate import controls_required
inputs = json.loads((Path(os.environ["RUNTIME_DIR"]) / "runtime_inputs/policy_canary_runtime_inputs.json").read_text())
print("required" if controls_required(inputs["task_success_contract"]) else "legacy")
PYSTRICT
)
if [ "$strict_controls" != legacy ]; then
  write_fallback_result team_candidate_strict_controls_runtime_unavailable 87
  exit 87
fi

# The provider bootstrap wrote the bearer token to a private file and exported its path.
if [ -z "${{{credential_env}:-}}" ] || [ ! -f "${{{credential_env}}}" ]; then
  write_fallback_result team_candidate_credential_unavailable 88
  exit 88
fi

/isaac-sim/python.sh "$RUNTIME_DIR/adp_arena_provider_runner.py"
runner_rc=$?
write_fallback_result policy_canary_worker_failed_without_result "$runner_rc"
exit $runner_rc
'''


def assert_secret_absent(bundle_path: str | Path, secret: str) -> None:
    """Refuse a bundle whose archive bytes, or any member's bytes, contain the secret."""

    needle = secret.strip().encode("utf-8")
    if len(needle) < 8:
        raise TeamCandidateBundleError("team_candidate_secret_probe_too_short")
    path = Path(bundle_path)
    if needle in path.read_bytes():
        raise TeamCandidateBundleError("team_candidate_secret_in_bundle")
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if needle in archive.read(member):
                raise TeamCandidateBundleError("team_candidate_secret_in_bundle")


def build_team_candidate_session_bundle(
    *,
    job_dir: str | Path,
    packet_dir: str | Path,
    runtime_source_packet_receipt: str | Path,
    runtime_input_manifest_path: str | Path,
    session_authority_path: str | Path,
    execution_spec_path: str | Path,
    implementation_commit: str,
    generated_at: str | None = None,
    team_robot_model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Package the warm Quick-10 worker for one admitted team candidate (no secret material).

    A candidate on its own robot ships its verified robot USD at the path its
    sealed binding names inside the scene packet.
    """

    inputs = validate_runtime_input_manifest(_read(runtime_input_manifest_path))
    authority = validate_session_authority(_read(session_authority_path))
    if inputs["run_kind"] != TEAM_RUN_KIND or authority["run_kind"] != TEAM_RUN_KIND:
        raise TeamCandidateBundleError("team_candidate_bundle_run_kind_invalid")
    if authority.get("execution_release") is not None:
        raise TeamCandidateBundleError("team_candidate_bundle_hotfix_overlay_unsupported")
    if authority["runtime_inputs_digest"] != inputs["runtime_inputs_digest"]:
        raise TeamCandidateBundleError("team_candidate_bundle_authority_input_mismatch")
    packet_receipt_path = Path(packet_dir).expanduser().resolve() / "native_task_arena_packet_receipt.v1.json"
    if _bound_record_path(inputs["base_native_packet"], code="policy_canary_base_packet_record_invalid") \
            != packet_receipt_path:
        raise TeamCandidateBundleError("team_candidate_base_packet_record_mismatch")
    if _bound_record_path(inputs["runtime_source"], code="policy_canary_runtime_source_record_invalid") \
            != Path(runtime_source_packet_receipt).expanduser().resolve():
        raise TeamCandidateBundleError("team_candidate_runtime_source_record_mismatch")
    construction_result_path = _bound_record_path(
        inputs["construction_result"], code="policy_canary_construction_result_record_invalid")
    spec = _validate_team_spec(_read(execution_spec_path), inputs=inputs)
    candidate = spec["candidate_id"]
    credential_env = str(spec["policy_endpoint"]["credential_env"])
    from .native_task_arena_policy_canary_worker import preflight_policy_canary_static_inputs
    packet_root = packet_receipt_path.parent
    static_preflight = preflight_policy_canary_static_inputs(
        inputs=inputs, authority=authority,
        base_scene_plan=_read(packet_root / "native_task_arena_scene_plan.v1.json"),
        construction=_read(construction_result_path),
        packet_request=_read(packet_root / "native_task_arena_packet_request.v1.json"),
        specs={candidate: spec},
    )
    job = Path(job_dir).expanduser().resolve()
    job.mkdir(parents=True, exist_ok=True)
    write_json(job / "policy_canary_static_startup_preflight.v1.json", static_preflight)
    if static_preflight["status"] != "passed":
        raise TeamCandidateBundleError(
            "policy_canary_static_startup_preflight_failed:" + ",".join(static_preflight["blockers"]))
    package = Path(__file__).resolve().parent
    runtime_modules = sorted({package / name for name in POLICY_RUNTIME_MODULE_NAMES} | {
        package / name for name in (
            "native_task_arena_policy_worker.py", "native_task_arena_policy_canary_session.py",
            "policy_canary_interrupted_cell_recovery.py", "native_policy_canary_diagnostic_continuation.py",
            "native_task_arena_policy_canary_worker.py", *TEAM_RUNTIME_MODULE_NAMES)})
    assert_provider_runtime_import_closure(
        package_source_dir=package, shipped_module_names=sorted(path.name for path in runtime_modules),
        shipped_package_files=CANARY_SHIPPED_PACKAGE_FILES, code="team_candidate_bundle_import_closure_incomplete")
    base = build_native_task_arena_bundle(
        job_dir=job / "base", packet_dir=packet_dir, team_embodiment=spec.get("team_embodiment"),
        team_robot_model=team_robot_model_path if spec.get("team_embodiment") is not None else None,
        worker_source=package / "native_task_arena_policy_canary_worker.py",
        runtime_module_sources=runtime_modules, implementation_commit=implementation_commit,
        execution_mode="construction_canary", expected_output_filename=PROVIDER_RESULT_FILENAME,
        container_image=NATIVE_TASK_ARENA_IMAGE, runtime_source_packet_receipt=runtime_source_packet_receipt,
        runtime_extra_files={f"blueprint_pipeline/{name}": package / name for name in CANARY_SHIPPED_PACKAGE_FILES},
        bound_runtime_inputs={
            "policy_canary_runtime_inputs.json": runtime_input_manifest_path,
            "policy_canary_session_authority.json": session_authority_path,
            f"policy_execution_spec.{candidate}.json": execution_spec_path,
            "native_task_arena_construction_result.v1.json": construction_result_path,
        },
        generated_at=generated_at,
    )
    with tempfile.TemporaryDirectory(prefix="team-candidate-session-bundle-") as raw:
        root = Path(raw)
        with zipfile.ZipFile(base["bundle_path"]) as archive:
            archive.extractall(root)
        runtime = root / "provider_runtime"
        contract_dependency = stage_policy_contract_dependencies(runtime)
        for name in POLICY_RUNTIME_ROOT_MODULE_NAMES:
            if name != "adp009d_policy_server_worker.py":  # no local policy server in a team session
                shutil.copy2(package / name, runtime / name)
        binding = spec.get("team_embodiment")
        if (binding is None) != (team_robot_model_path is None):
            raise TeamCandidateBundleError("team_embodiment_model_binding_mismatch")
        if binding is not None:
            from .team_embodiment import verify_model_file

            record = verify_model_file(binding["manifest"], team_robot_model_path)
            if record["size_bytes"] != binding["model"]["size_bytes"]:
                raise TeamCandidateBundleError("team_embodiment_model_binding_mismatch")
            target = runtime / "native_task_packet" / binding["model"]["bundle_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            closure = binding["model"].get("dependency_closure")
            if closure is not None:
                from .team_robot_model_delivery import copy_prepared_closure

                if Path(team_robot_model_path).name != closure["model_path"]:
                    raise TeamCandidateBundleError("team_embodiment_model_binding_mismatch")
                copy_prepared_closure(source=Path(team_robot_model_path).parent,
                                      destination=target.parent, closure=closure)
            else:
                shutil.copyfile(team_robot_model_path, target)
        entrypoint = runtime / "run_adp_arena_provider_runtime.sh"
        entrypoint.write_text(_entrypoint(credential_env), encoding="utf-8")
        entrypoint.chmod(entrypoint.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        manifest_path = runtime / "adp_arena_provider_manifest.json"
        manifest = _read(manifest_path)
        manifest.update({
            "schema_version": PROVIDER_BUNDLE_SCHEMA_VERSION, "execution_mode": EXECUTION_MODES[TEAM_RUN_KIND],
            "run_kind": TEAM_RUN_KIND, "claim_ceiling": CLAIM_CEILING, "candidate_ids": [candidate],
            "team_policy_interface_id": spec["policy_interface_id"],
            "episodes_per_policy": authority["episodes_per_policy"],
            "learned_policy_rollout_count": authority["learned_policy_rollout_count"],
            "maximum_provider_allocations": 1, "retry_cap": 0, "authority_digest": authority["authority_digest"],
            "runtime_inputs_digest": inputs["runtime_inputs_digest"],
            "task_success_contract_digest": inputs["task_success_contract_digest"],
            "candidate_policy_queried": False, "expected_output_filename": PROVIDER_RESULT_FILENAME,
            "scene_promotion_authorized": False, "official_ranking_authorized": False,
            "execution_release": None, "hotfix_application": None,
            "execution_spec_digests": {candidate: spec["execution_spec_digest"]},
            "team_candidate_admission_digest": inputs["team_candidate_admission"]["admission_digest"],
            "team_credential_env": credential_env, "secret_material_bundled": False,
            "contract_python_dependencies": [contract_dependency], "static_startup_preflight": static_preflight,
            "input_digest": "",
        })
        manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
        write_json(manifest_path, manifest)
        bundle_path = job / "team_candidate_quick10_session_bundle.zip"
        with zipfile.ZipFile(bundle_path, "w", allowZip64=True) as archive:
            for source in sorted(root.rglob("*")):
                if source.is_file():
                    _write_zip_file(archive, source=source, archive_path=source.relative_to(root).as_posix())
    receipt = {**manifest, "bundle_path": str(bundle_path), "bundle_size_bytes": bundle_path.stat().st_size,
               "bundle_sha256": _sha256(bundle_path)}
    write_json(job / "team_candidate_quick10_session_bundle_receipt.v1.json", receipt)
    validate_provider_bundle(receipt, authority=authority)
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Rebuild the sealed team bundle at the deployed commit from its sealed inputs."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("--job-dir", "--packet-dir", "--runtime-source-packet-receipt", "--runtime-input-manifest-path",
                 "--session-authority-path", "--execution-spec-path", "--implementation-commit"):
        parser.add_argument(name, required=True)
    parser.add_argument("--generated-at")
    parser.add_argument("--team-robot-model", help="the team's staged USD model, for a bring-your-own-arm run")
    args = parser.parse_args(argv)
    receipt = build_team_candidate_session_bundle(
        job_dir=args.job_dir, packet_dir=args.packet_dir,
        runtime_source_packet_receipt=args.runtime_source_packet_receipt,
        runtime_input_manifest_path=args.runtime_input_manifest_path,
        session_authority_path=args.session_authority_path, execution_spec_path=args.execution_spec_path,
        implementation_commit=args.implementation_commit, generated_at=args.generated_at,
        team_robot_model_path=args.team_robot_model)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXECUTION_AUTHORITY",
    "TeamCandidateBundleError",
    "assert_secret_absent",
    "build_team_candidate_session_bundle",
    "main",
]
