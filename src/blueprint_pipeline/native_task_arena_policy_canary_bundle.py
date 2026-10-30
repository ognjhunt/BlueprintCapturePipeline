"""Build one provider bundle for the paired internal policy canary session."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from importlib.metadata import distribution
import json
import os
import shlex
import re
from pathlib import Path
import shutil
import stat
import subprocess
import sys
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
from .provider_runtime_import_closure import assert_provider_runtime_import_closure
from .native_task_arena_policy_canary_session import (
    CLAIM_CEILING,
    EXECUTION_MODES,
    PROVIDER_BUNDLE_SCHEMA_VERSION,
    PROVIDER_RESULT_FILENAME,
    RUN_KIND,
    validate_provider_bundle,
    validate_runtime_input_manifest,
    validate_session_authority,
)
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from .openwam_piper_bundle import RUNTIME_MODULES as OPENWAM_RUNTIME_MODULES
from .task_evaluation_canary_hotfix_overlay import (
    apply_canary_hotfix_overlay,
    canary_hotfix_execution_release,
    verify_canary_hotfix_overlay,
)

# The bounded HTTPS boundary the team candidate's client uses (stdlib only).
CANARY_SHIPPED_PACKAGE_FILES = ("core/__init__.py", "core/common.py", "core/security_controls.py")
# The team path the shared worker imports: the team's HTTPS clients and its own robot.
TEAM_RUNTIME_MODULE_NAMES = (
    "team_candidate_policy_client.py", "droid_https_policy_client.py", "team_policy_https_client.py",
    "task_family_policy_https_client.py",
    "dreamlabs_policy_client.py", "team_arm_kinematics.py",
    "team_policy_episode_outcome.py", "team_evaluation_worker_retention.py",
    "native_task_arena_execution_contract.py",
    "team_embodiment.py", "team_embodiment_runtime.py", "native_task_robot_contact_topology.py",
    # The first session on an undecided rigid scene records and scores its robot-free probe.
    "rigid_robot_free_probe_recorder.py", "rigid_robot_free_probe_plan.py",
    "rigid_robot_free_readiness_score.py",
)
TEAM_RUNTIME_MODULE_NAMES += OPENWAM_RUNTIME_MODULES


EXECUTION_AUTHORITY = "internal_policy_canary_unqualified"


def stage_policy_contract_dependencies(runtime: Path) -> dict[str, Any]:
    """Ship the pinned pure-Python owner-contract canonicalizer with its license.

    The retained simulator archive predates this dependency. Bundling these small
    files makes validation independent of the provider's site-packages and avoids
    rebuilding or downloading an unchanged multi-gigabyte simulator archive.
    """
    dependency = distribution("rfc8785")
    if dependency.version != "0.1.4":
        raise ValueError("policy_canary_rfc8785_version_mismatch")
    paths = ("rfc8785/__init__.py", "rfc8785/_impl.py", "rfc8785/py.typed",
             "rfc8785-0.1.4.dist-info/LICENSE")
    rows = []
    for relative in paths:
        source = Path(dependency.locate_file(relative))
        if not source.is_file() or source.is_symlink():
            raise ValueError("policy_canary_rfc8785_source_missing")
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        rows.append({"relative_path": relative, "sha256": _sha256(target),
                     "size_bytes": target.stat().st_size})
    return {"distribution": "rfc8785", "version": "0.1.4", "files": rows}


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("policy_canary_bundle_input_invalid")
    return value


def preflight_sealed_policy_canary_bundle(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Run the sealed worker's own static preflight, without assets or providers."""
    blocked = {"schema_version": "policy_canary_static_startup_preflight.v1", "status": "blocked",
               "provider_mutation_performed": False, "candidate_policy_queried": False}
    archive_path = Path(str(receipt.get("bundle_path") or ""))
    if (not archive_path.is_file() or archive_path.stat().st_size != receipt.get("bundle_size_bytes")
        or _sha256(archive_path) != receipt.get("bundle_sha256")):
        return {**blocked, "blockers": ["policy_canary_preflight_bundle_bytes_mismatch"]}
    try:
        with tempfile.TemporaryDirectory(prefix="policy-canary-static-preflight-") as raw:
            root = Path(raw)
            with zipfile.ZipFile(archive_path) as archive:
                manifest = json.loads(archive.read("provider_runtime/adp_arena_provider_manifest.json"))
                from .task_evaluation_control_stage_policy import CONTROLS_PAUSED
                from .native_policy_canary_control_gate import controls_required
                inputs = json.loads(archive.read("provider_runtime/runtime_inputs/policy_canary_runtime_inputs.json"))
                if CONTROLS_PAUSED and controls_required(inputs.get("task_success_contract") or {}):
                    return {**blocked, "blockers": ["task_evaluation_controls_paused_by_owner"]}
                dependencies = manifest.get("contract_python_dependencies") or []
                canonicalizer = next((item for item in dependencies if item.get("distribution") == "rfc8785"), {})
                required = {"rfc8785/__init__.py", "rfc8785/_impl.py", "rfc8785/py.typed", "rfc8785-0.1.4.dist-info/LICENSE"}
                if canonicalizer.get("version") != "0.1.4" or {row.get("relative_path") for row in canonicalizer.get("files", [])} != required:
                    return {**blocked, "blockers": ["policy_canary_preflight_contract_dependency_missing"]}
                for row in canonicalizer["files"]:
                    content = archive.read("provider_runtime/" + row["relative_path"])
                    if len(content) != row["size_bytes"] or "sha256:" + hashlib.sha256(content).hexdigest() != row["sha256"]:
                        return {**blocked, "blockers": ["policy_canary_preflight_contract_dependency_digest_mismatch"]}
                for member in archive.infolist():
                    path = Path(member.filename)
                    if (not path.is_absolute() and ".." not in path.parts and path.parts
                        and path.parts[0] == "provider_runtime" and path.suffix in {".py", ".json"}):
                        target = root / path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.read(member))
            output = root / "preflight.json"
            code = ("import json,sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
                    "from adp_arena_provider_runner import preflight_policy_canary_runtime_directory;"
                    "r=preflight_policy_canary_runtime_directory(Path(sys.argv[1]));"
                    "Path(sys.argv[2]).write_text(json.dumps(r));sys.exit(0 if r['status']=='passed' else 2)")
            child = subprocess.run([sys.executable, "-I", "-c", code, str(root / "provider_runtime"), str(output)],
                                   capture_output=True, text=True, timeout=60, check=False)
            if output.is_file():
                result = _read(output)
                result["sealed_bundle_sha256"] = receipt["bundle_sha256"]
                return result
            return {**blocked, "blockers": ["policy_canary_sealed_static_preflight_unavailable"],
                    "diagnostic": child.stderr[-3000:]}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, subprocess.TimeoutExpired) as exc:
        return {**blocked, "blockers": ["policy_canary_sealed_static_preflight_failed:" + str(exc)]}


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


def _family_operator_inputs(
    specs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Path], dict[str, Any] | None]:
    """Stage only the operator-configured catalog and its sealed qualifications."""
    from .task_family_policy_execution import FAMILY_BACKEND

    family_specs = [spec for spec in specs.values() if spec.get("evaluation_backend") == FAMILY_BACKEND]
    if not family_specs:
        return {}, None
    configured = os.environ.get("BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH")
    catalog_path = Path(configured) if configured else None
    if (catalog_path is None or not catalog_path.is_absolute() or not catalog_path.is_file()
            or catalog_path.is_symlink() or any(parent.is_symlink() for parent in catalog_path.parents)):
        raise ValueError("task_family_qualified_runtime_catalog_not_configured")
    catalog = _read(catalog_path)
    if (catalog.get("schema_version") != "task_family_native_runtime_catalog.v1"
            or catalog.get("catalog_digest") != canonical_digest(catalog, digest_field="catalog_digest")):
        raise ValueError("task_family_operator_runtime_catalog_invalid")
    files = {"runtime_operator_inputs/catalog.json": catalog_path}
    for spec in family_specs:
        selection = spec["task_family_runtime_selection"]
        rows = [row for row in catalog.get("runtimes", []) if isinstance(row, Mapping)
                and row.get("runtime_id") == selection["runtime_id"]]
        if selection["runtime_catalog_digest"] != catalog["catalog_digest"] or len(rows) != 1:
            raise ValueError("task_family_operator_runtime_catalog_binding_mismatch")
        reference = rows[0].get("qualification")
        if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
            raise ValueError("task_family_operator_runtime_qualification_required")
        relative = Path(str(reference["path"]))
        source = catalog_path.parent / relative
        if (relative.is_absolute() or ".." in relative.parts or not source.is_file()
                or source.is_symlink() or any(parent.is_symlink() for parent in source.parents)
                or not source.resolve().is_relative_to(catalog_path.parent.resolve())
                or _sha256(source) != reference["sha256"]):
            raise ValueError("task_family_operator_runtime_qualification_changed")
        destination = "runtime_operator_inputs/" + relative.as_posix()
        if destination in files and files[destination] != source:
            raise ValueError("task_family_operator_runtime_qualification_path_collision")
        files[destination] = source
    records = [{"relative_path": relative, "sha256": _sha256(source), "size_bytes": source.stat().st_size}
               for relative, source in sorted(files.items())]
    return files, {"catalog_relative_path": "runtime_operator_inputs/catalog.json",
                   "runtime_catalog_digest": catalog["catalog_digest"], "files": records}


def _entrypoint(*, local_candidates: tuple[str, ...] = ("pi05_droid", "groot_n17_droid"),
                run_kind: str = RUN_KIND, credential_env: str | None = None,
                family_runtime_catalog_staged: bool = False, openwam_local: bool = False) -> str:
    if credential_env is not None and re.fullmatch(r"[A-Z][A-Z0-9_]{2,58}_FILE", credential_env) is None:
        raise ValueError("beta_robolab_https_credential_binding_invalid")
    credential_check = "" if credential_env is None else f'''/isaac-sim/python.sh - {shlex.quote(credential_env)} <<'PYCREDENTIAL'
import os, stat, sys
from pathlib import Path
raw = os.environ.get(sys.argv[1])
try:
    path = Path(raw) if raw else None
    info = path.lstat() if path is not None else None
    valid = (path is not None and path.is_absolute() and not path.is_symlink()
             and stat.S_ISREG(info.st_mode) and not info.st_mode & 0o077
             and 0 < info.st_size <= 65536)
except OSError:
    valid = False
raise SystemExit(0 if valid else 86)
PYCREDENTIAL
credential_rc=$?
if [ $credential_rc -ne 0 ]; then
  write_fallback_result beta_robolab_private_credential_file_invalid "$credential_rc"
  exit $credential_rc
fi
'''
    names = " ".join(shlex.quote(candidate) for candidate in local_candidates)
    provisioning = "\n".join(
        f'cp "$RUNTIME_DIR/runtime_inputs/policy_execution_spec.{candidate}.json" '
        f'"$RUNTIME_DIR/{POLICY_EXECUTION_SPEC_STAGED_NAME}"\n'
        f'bash "$RUNTIME_DIR/adp009d_policy_provisioning.{candidate}.sh" '
        f'>"$OUT_DIR/policy_provisioning.{candidate}.log" 2>&1 || exit $?'
        for candidate in local_candidates
    )
    catalog_config = ('export BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH="$RUNTIME_DIR/runtime_operator_inputs/catalog.json"'
                      if family_runtime_catalog_staged else "")
    local_client_config = ('export PYTHONDONTWRITEBYTECODE=1\nexport PYTHONPATH="$RUNTIME_DIR/openwam_client:$RUNTIME_DIR${PYTHONPATH:+:$PYTHONPATH}"'
                           if openwam_local else "")
    return f'''#!/usr/bin/env bash
set +e
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${{BLUEPRINT_ADP_ARENA_OUTPUT_DIR:-$RUNTIME_DIR/../runtime_output}}"
export RUNTIME_DIR OUT_DIR
{catalog_config}
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
  "run_kind": "{run_kind}",
  "claim_ceiling": "diagnostic_policy_execution",
  "candidate_policy_queried": False,
  "blockers": [sys.argv[3]],
  "worker_exit_code": int(sys.argv[2]),
}}, sort_keys=True) + "\\n", encoding="utf-8")
PY
  fi
}}

teardown_servers() {{
  original_rc=$?
  trap - EXIT INT TERM HUP
  write_fallback_result policy_canary_entrypoint_failed_without_result "$original_rc"
  for candidate in {names}; do
    /isaac-sim/python.sh "$RUNTIME_DIR/adp009d_policy_server_worker.py" \
      --terminate-ready-server \
      --receipt "$OUT_DIR/adp009d_policy_server_receipt.$candidate.json" \
      --result "$OUT_DIR/{PROVIDER_RESULT_FILENAME}" \
      --result-schema native_task_arena_policy_canary_session_result.v1 || true
  done
  exit $original_rc
}}
trap teardown_servers EXIT INT TERM HUP

{credential_check}
/isaac-sim/python.sh -m blueprint_pipeline.native_task_runtime_source_provision \
  --source-receipt "$SOURCE_RECEIPT" --source-packet "$SOURCE_PACKET" \
  --extraction-dir "$RUNTIME_DIR/provisioned_runtime_sources" \
  --output "$OUT_DIR/native_task_runtime_source_provisioning.v1.json" \
  --simulator-root /isaac-sim
source_rc=$?
if [ $source_rc -ne 0 ]; then
  write_fallback_result policy_canary_runtime_source_provision_failed "$source_rc"
  exit $source_rc
fi

# Validate the actual immutable input before downloading or starting either
# policy. Lazy contract dependencies must be present at this boundary.
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

# Strict controls execute before either checkpoint is loaded. The second stage
# validates and reuses these exact cell receipts after ordinary provisioning.
strict_controls=$(/isaac-sim/python.sh - <<'PYSTRICT'
import json, os
from pathlib import Path
from blueprint_pipeline.native_policy_canary_control_gate import controls_required
inputs = json.loads((Path(os.environ["RUNTIME_DIR"]) / "runtime_inputs/policy_canary_runtime_inputs.json").read_text())
print("required" if controls_required(inputs["task_success_contract"]) else "legacy")
PYSTRICT
)
strict_rc=$?
if [ $strict_rc -ne 0 ]; then exit $strict_rc; fi
if [ "$strict_controls" = required ]; then
  BLUEPRINT_POLICY_CANARY_MATRIX_STAGE=controls /isaac-sim/python.sh "$RUNTIME_DIR/adp_arena_provider_runner.py"
  controls_rc=$?
  if [ $controls_rc -ne 0 ]; then exit $controls_rc; fi
  export BLUEPRINT_POLICY_CANARY_MATRIX_STAGE=policies
elif [ "$strict_controls" != legacy ]; then
  exit 86
fi

{provisioning}
{local_client_config}

/isaac-sim/python.sh "$RUNTIME_DIR/adp_arena_provider_runner.py"
runner_rc=$?
write_fallback_result policy_canary_worker_failed_without_result "$runner_rc"
exit $runner_rc
'''


def build_policy_canary_session_bundle(
    *,
    job_dir: str | Path,
    packet_dir: str | Path,
    runtime_source_packet_receipt: str | Path,
    runtime_input_manifest_path: str | Path,
    session_authority_path: str | Path,
    pi05_execution_spec_path: str | Path | None = None,
    groot_execution_spec_path: str | Path | None = None,
    pi05_checkpoint_inventory_path: str | Path | None = None,
    execution_spec_paths: Mapping[str, str | Path] | None = None,
    checkpoint_inventory_paths: Mapping[str, str | Path] | None = None,
    family_control_result_path: str | Path | None = None,
    team_robot_model_path: str | Path | None = None,
    minimum_gpu_memory_gb: float | None = None,
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
    from .native_task_arena_policy_canary_worker import appearance_render_backend_from_plan
    appearance_render_backend_from_plan(
        _read(packet_receipt_path.parent / "native_task_arena_scene_plan.v1.json"),
        packet_request=_read(packet_receipt_path.parent / "native_task_arena_packet_request.v1.json"),
    )
    runtime_source_path = _bound_record_path(
        inputs["runtime_source"], code="policy_canary_runtime_source_record_invalid"
    )
    if runtime_source_path != Path(runtime_source_packet_receipt).expanduser().resolve():
        raise ValueError("policy_canary_runtime_source_record_mismatch")
    family_pending = ((inputs.get("beta_screening_admission") or {}).get("backend") == "task_family_native.v1"
                      and inputs.get("pre_policy_site_qualification_required") is True)
    construction_result_path = (
        _bound_record_path(inputs["construction_result"], code="policy_canary_construction_result_record_invalid")
        if inputs.get("construction_result") is not None else None
    )
    if construction_result_path is None and not family_pending:
        raise ValueError("policy_canary_construction_result_record_invalid")
    from .policy_interface_binding import TEAM_HTTPS_INTERFACES, resolve_policy_interface
    from .beta_robolab_runtime import stage_robolab_source
    from .beta_robolab_policy_execution import validate_robolab_policy_execution_spec
    from .task_family_policy_execution import FAMILY_BACKEND, validate_task_family_policy_execution_spec

    spec_paths = dict(execution_spec_paths or {})
    if not spec_paths:
        if pi05_execution_spec_path is None or groot_execution_spec_path is None:
            raise ValueError("policy_canary_execution_specs_missing")
        spec_paths = {"pi05_droid": pi05_execution_spec_path, "groot_n17_droid": groot_execution_spec_path}
    specs = {candidate: _validate_spec(_read(path), candidate=candidate) for candidate, path in spec_paths.items()}
    family_scene = any(spec.get("evaluation_backend") == FAMILY_BACKEND for spec in specs.values())
    family_controls_path = None
    if family_scene:
        if family_control_result_path is not None:
            family_controls_path = Path(family_control_result_path).expanduser().resolve()
        elif isinstance(inputs.get("control_result"), Mapping):
            family_controls_path = _bound_record_path(
                inputs["control_result"], code="task_family_qualified_site_controls_receipt_required")
        if family_controls_path is not None:
            if not family_controls_path.is_file() or family_controls_path.is_symlink():
                raise ValueError("task_family_qualified_site_controls_receipt_required")
            from .native_task_arena_controls_admission import validate_native_task_controls_admission

            validate_native_task_controls_admission(
                scene_plan=_read(packet_receipt_path.parent / "native_task_arena_scene_plan.v1.json"),
                construction_result=_read(construction_result_path) if construction_result_path is not None else {},
                control_result=_read(family_controls_path),
            )
    local_candidates = []
    for candidate, spec in specs.items():
        interface = resolve_policy_interface(spec)
        if interface not in TEAM_HTTPS_INTERFACES:
            if interface == "openwam_piper_eef20.v1":
                from .openwam_piper_runtime import validate_execution_spec as validate_openwam_spec
                validate_openwam_spec(spec)
                if inputs.get("run_kind") != "blueprint_beta_screening":
                    raise ValueError("openwam_native_execution_requires_beta_run")
            elif candidate not in ("pi05_droid", "groot_n17_droid", "cosmos3_edge_policy_droid"):
                raise ValueError("policy_canary_local_provisioning_identity_not_admitted")
            if candidate == "cosmos3_edge_policy_droid":
                from .cosmos_edge_beta_runtime import validate_execution_spec as validate_cosmos_spec
                validate_cosmos_spec(spec)
                if inputs.get("run_kind") != "blueprint_beta_screening":
                    raise ValueError("cosmos_edge_native_execution_requires_beta_run")
            local_candidates.append(candidate)
        if inputs.get("run_kind") == "blueprint_beta_screening":
            if spec.get("evaluation_backend") == FAMILY_BACKEND:
                validate_task_family_policy_execution_spec(
                    spec, candidate_id=candidate,
                    scene_plan=_read(packet_receipt_path.parent / "native_task_arena_scene_plan.v1.json"),
                )
            else:
                validate_robolab_policy_execution_spec(spec, candidate_id=candidate)
    remote_specs = [spec for spec in specs.values() if resolve_policy_interface(spec) in TEAM_HTTPS_INTERFACES]
    credential_env = None
    if remote_specs:
        if inputs.get("run_kind") != "blueprint_beta_screening" or len(remote_specs) != 1 or len(specs) != 1:
            raise ValueError("policy_canary_remote_candidate_session_invalid")
        credential_env = str(remote_specs[0]["policy_endpoint"]["credential_env"])
    inventory_paths = dict(checkpoint_inventory_paths or {})
    if pi05_checkpoint_inventory_path is not None:
        inventory_paths.setdefault("pi05_droid", pi05_checkpoint_inventory_path)
    if "pi05_droid" in local_candidates and "pi05_droid" not in inventory_paths:
        raise ValueError("policy_canary_openpi_inventory_missing")
    if set(specs) != set(inputs["candidate_ids"]) or set(specs) != set(authority["candidate_ids"]):
        raise ValueError("policy_canary_provider_candidate_interfaces_not_admitted")
    if any(
        spec.get("task_success_contract") != inputs["task_success_contract"]
        or spec.get("task_success_contract_digest")
        != inputs["task_success_contract_digest"]
        for spec in specs.values()
    ):
        raise ValueError("policy_canary_execution_spec_task_success_contract_mismatch")
    from .native_task_arena_policy_canary_worker import preflight_policy_canary_static_inputs
    static_preflight = preflight_policy_canary_static_inputs(
        inputs=inputs, authority=authority,
        base_scene_plan=_read(packet_receipt_path.parent / "native_task_arena_scene_plan.v1.json"),
        construction=_read(construction_result_path) if construction_result_path is not None else {},
        packet_request=_read(packet_receipt_path.parent / "native_task_arena_packet_request.v1.json"),
        specs=specs,
    )
    write_json(Path(job_dir) / "policy_canary_static_startup_preflight.v1.json", static_preflight)
    if static_preflight["status"] != "passed":
        raise ValueError("policy_canary_static_startup_preflight_failed:" + ",".join(static_preflight["blockers"]))
    operator_files, operator_inputs = _family_operator_inputs(specs)
    package = Path(__file__).resolve().parent
    runtime_modules = [package / name for name in POLICY_RUNTIME_MODULE_NAMES]
    runtime_modules.extend(
        (
            package / "native_task_arena_policy_worker.py",
            package / "native_task_arena_policy_canary_session.py",
            # Recovery depends on paired-canary authority, absent from generic policy bundles.
            package / "policy_canary_interrupted_cell_recovery.py",
            package / "native_policy_canary_diagnostic_continuation.py",
            # Interruption recovery reuses the worker's pure input validators.
            package / "native_task_arena_policy_canary_worker.py",
        )
    )
    # Beta L5: one admitted team candidate's authenticated HTTPS client, and its own robot.
    runtime_modules.extend(package / name for name in TEAM_RUNTIME_MODULE_NAMES)
    # Refuse to seal a bundle whose shipped package cannot satisfy its own
    # imports; the alternative is discovering it on a rented GPU after both
    # policy servers are up.  The worker itself ships at the runtime root, so
    # its ``blueprint_pipeline.*`` imports are checked against the same set.
    assert_provider_runtime_import_closure(
        package_source_dir=package,
        shipped_module_names=sorted(
            {path.name for path in runtime_modules}
            | {"native_task_arena_policy_canary_worker.py"}
        ),
        shipped_package_files=CANARY_SHIPPED_PACKAGE_FILES,
        code="policy_canary_bundle_import_closure_incomplete",
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
        runtime_extra_files={**{f"blueprint_pipeline/{name}": package / name for name in CANARY_SHIPPED_PACKAGE_FILES},
                             **operator_files},
        bound_runtime_inputs={
            "policy_canary_runtime_inputs.json": runtime_input_manifest_path,
            "policy_canary_session_authority.json": session_authority_path,
            **{f"policy_execution_spec.{candidate}.json": path for candidate, path in spec_paths.items()},
            **({"native_task_arena_construction_result.v1.json": construction_result_path}
               if construction_result_path is not None else {}),
            **({"native_task_arena_control_result.v1.json": family_controls_path}
               if family_controls_path is not None else {}),
        },
        generated_at=generated_at,
    )
    with tempfile.TemporaryDirectory(prefix="policy-canary-session-bundle-") as raw:
        root = Path(raw)
        with zipfile.ZipFile(base["bundle_path"]) as archive:
            archive.extractall(root)
        runtime = root / "provider_runtime"
        team_bindings = [spec["team_embodiment"] for spec in specs.values() if spec.get("team_embodiment") is not None]
        if (bool(team_bindings) != (team_robot_model_path is not None)
                or (team_bindings and len(specs) != 1)):
            raise ValueError("beta_screening_team_robot_model_binding_mismatch")
        if team_bindings:
            from .team_embodiment import verify_model_file
            from .team_robot_model_delivery import copy_prepared_closure

            binding = team_bindings[0]
            record = verify_model_file(binding["manifest"], team_robot_model_path)
            if (record["sha256"] != binding["model"]["sha256"]
                    or record["size_bytes"] != binding["model"]["size_bytes"]):
                raise ValueError("beta_screening_team_robot_model_binding_mismatch")
            relative = Path(binding["model"]["bundle_path"])
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "team_embodiment":
                raise ValueError("beta_screening_team_robot_bundle_path_invalid")
            target = runtime / "native_task_packet" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            closure = binding["model"].get("dependency_closure")
            if closure is not None:
                if Path(team_robot_model_path).name != closure["model_path"]:
                    raise ValueError("beta_screening_team_robot_model_binding_mismatch")
                copy_prepared_closure(source=Path(team_robot_model_path).parent,
                                      destination=target.parent, closure=closure)
            else:
                shutil.copyfile(team_robot_model_path, target)
        contract_dependency = stage_policy_contract_dependencies(runtime)
        robolab_dependency = (stage_robolab_source(runtime)
                              if any(spec.get("evaluation_backend") == "robolab.v1" for spec in specs.values()) else None)
        openwam_contract = None
        for candidate in local_candidates:
            script = runtime / f"adp009d_policy_provisioning.{candidate}.sh"
            if specs[candidate].get("policy_interface_id") == "openwam_piper_eef20.v1":
                from .openwam_piper_bundle import stage_runtime, provisioning_script as openwam_script
                openwam_contract = stage_runtime(runtime)
                provisioning_script = openwam_script()
            elif candidate == "cosmos3_edge_policy_droid":
                from .cosmos_edge_beta_provisioning import build_provisioning_script as build_cosmos_provisioning
                from .cosmos_edge_beta_runtime import (
                    SNAPSHOT_STAGED_NAME, snapshot_manifest_path, verified_snapshot_manifest,
                )
                snapshot_path = snapshot_manifest_path()
                verified_snapshot_manifest(snapshot_path)
                shutil.copy2(snapshot_path, runtime / SNAPSHOT_STAGED_NAME)
                provisioning_script = build_cosmos_provisioning()
            else:
                provisioning_script = build_provisioning_script(candidate)
            script.write_text(provisioning_script, encoding="utf-8")
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
        if "pi05_droid" in local_candidates:
            shutil.copy2(inventory_paths["pi05_droid"], runtime / CHECKPOINT_INVENTORY_STAGED_NAME)
        entrypoint = runtime / "run_adp_arena_provider_runtime.sh"
        entrypoint.write_text(_entrypoint(local_candidates=tuple(local_candidates), run_kind=inputs["run_kind"],
                                         credential_env=credential_env,
                                         family_runtime_catalog_staged=operator_inputs is not None,
                                         openwam_local=openwam_contract is not None), encoding="utf-8")
        entrypoint.chmod(entrypoint.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        manifest_path = runtime / "adp_arena_provider_manifest.json"
        manifest = _read(manifest_path)
        if minimum_gpu_memory_gb is not None:
            if type(minimum_gpu_memory_gb) not in (int, float) or not 0 < minimum_gpu_memory_gb <= 1024:
                raise ValueError("beta_screening_gpu_memory_bound_invalid")
            manifest["minimum_gpu_memory_gb"] = minimum_gpu_memory_gb
        if openwam_contract is not None:
            from .openwam_piper_runtime import MINIMUM_GPU_MEMORY_GB
            manifest["minimum_gpu_memory_gb"] = max(MINIMUM_GPU_MEMORY_GB, manifest.get("minimum_gpu_memory_gb", 0))
            manifest.update(openwam_runtime=openwam_contract, team_policy_interface_id="openwam_piper_eef20.v1",
                            secret_material_bundled=False)
        manifest.update(
            {
                "schema_version": PROVIDER_BUNDLE_SCHEMA_VERSION,
                "execution_mode": EXECUTION_MODES[inputs["run_kind"]],
                "run_kind": inputs["run_kind"],
                **({"program_id": "blueprint-free-beta-evaluated-pilot"}
                   if family_scene and inputs.get("run_kind") == "blueprint_beta_screening" else {}),
                "claim_ceiling": CLAIM_CEILING,
                "candidate_ids": list(inputs["candidate_ids"]),
                "episodes_per_policy": len(inputs["cells"]),
                "learned_policy_rollout_count": len(inputs["cells"]) * len(inputs["candidate_ids"]),
                "maximum_provider_allocations": 1,
                "retry_cap": 0,
                "authority_digest": authority["authority_digest"],
                "runtime_inputs_digest": inputs["runtime_inputs_digest"],
                "task_success_contract_digest": inputs[
                    "task_success_contract_digest"
                ],
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
                "contract_python_dependencies": [contract_dependency] + ([robolab_dependency] if robolab_dependency else []),
                **({"task_family_operator_inputs": operator_inputs} if operator_inputs is not None else {}),
                **({"pre_policy_site_qualification_required": True,
                    "family_static_readiness_digest": inputs["family_static_readiness_digest"]}
                   if family_pending else {}),
                **({"beta_screening_admission": inputs["beta_screening_admission"],
                    "evaluation_backend": next(iter(specs.values())).get("evaluation_backend")}
                   if inputs.get("run_kind") == "blueprint_beta_screening" else {}),
                **({"team_credential_env": credential_env,
                    "team_policy_interface_id": resolve_policy_interface(remote_specs[0]),
                    "secret_material_bundled": False} if credential_env is not None else {}),
                "static_startup_preflight": static_preflight,
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
    parser.add_argument("--pi05-execution-spec-path")
    parser.add_argument("--groot-execution-spec-path")
    parser.add_argument("--pi05-checkpoint-inventory-path")
    parser.add_argument("--execution-spec-path", action="append", dest="execution_spec_paths",
                        metavar="CANDIDATE=PATH", help="Repeat for each admitted candidate; replaces the legacy pair flags.")
    parser.add_argument("--checkpoint-inventory-path", action="append", dest="checkpoint_inventory_paths",
                        metavar="CANDIDATE=PATH", help="Repeat for each candidate needing a local checkpoint inventory.")
    parser.add_argument("--family-control-result", dest="family_control_result_path",
                        help="Optional retained native-family control result; existing receipt validation applies.")
    parser.add_argument("--team-robot-model-path",
                        help="Prepared robot model matching the sealed team binding; stages its verified USD closure.")
    parser.add_argument("--minimum-gpu-memory-gb", type=float,
                        help="Candidate runtime memory requirement, retained for allocator admission.")
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--hotfix-overlay")
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)

    def named_paths(values: list[str] | None, flag: str) -> dict[str, str] | None:
        if values is None:
            return None
        result = {}
        for value in values:
            candidate, separator, path = value.partition("=")
            if (not separator or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", candidate) is None
                    or not path.strip() or any(ord(char) < 32 or ord(char) == 127 for char in path)):
                parser.error(f"{flag} requires CANDIDATE=PATH with a valid candidate ID and nonempty path")
            if candidate in result:
                parser.error(f"{flag} repeats candidate {candidate}")
            result[candidate] = path
        return result

    args.execution_spec_paths = named_paths(args.execution_spec_paths, "--execution-spec-path")
    args.checkpoint_inventory_paths = named_paths(args.checkpoint_inventory_paths, "--checkpoint-inventory-path")
    if args.execution_spec_paths and (args.pi05_execution_spec_path or args.groot_execution_spec_path):
        parser.error("use --execution-spec-path or the legacy pair spec flags, not both")
    if args.checkpoint_inventory_paths and args.pi05_checkpoint_inventory_path:
        parser.error("use --checkpoint-inventory-path or --pi05-checkpoint-inventory-path, not both")
    if not args.execution_spec_paths and not (args.pi05_execution_spec_path and args.groot_execution_spec_path):
        parser.error("provide --execution-spec-path or both legacy pair spec flags")
    receipt = build_policy_canary_session_bundle(
        job_dir=args.job_dir,
        packet_dir=args.packet_dir,
        runtime_source_packet_receipt=args.runtime_source_packet_receipt,
        runtime_input_manifest_path=args.runtime_input_manifest_path,
        session_authority_path=args.session_authority_path,
        pi05_execution_spec_path=args.pi05_execution_spec_path,
        groot_execution_spec_path=args.groot_execution_spec_path,
        pi05_checkpoint_inventory_path=args.pi05_checkpoint_inventory_path,
        execution_spec_paths=args.execution_spec_paths,
        checkpoint_inventory_paths=args.checkpoint_inventory_paths,
        family_control_result_path=args.family_control_result_path,
        team_robot_model_path=args.team_robot_model_path,
        minimum_gpu_memory_gb=args.minimum_gpu_memory_gb,
        implementation_commit=args.implementation_commit,
        hotfix_overlay_path=args.hotfix_overlay,
        generated_at=args.generated_at,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EXECUTION_AUTHORITY", "build_policy_canary_session_bundle", "main"]
