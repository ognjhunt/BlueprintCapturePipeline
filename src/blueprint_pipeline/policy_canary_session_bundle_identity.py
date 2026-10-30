"""What the Vast adapter requires of a Quick-10 session bundle, per run kind.

Beta journey step: L5 team policy evaluation. One provider bundle kind
(``native_task_arena_policy_canary_session``) carries two run kinds:

- ``internal_policy_canary``: the frozen pi0.5/GR00T pair, both served by
  local policy servers the entrypoint provisions and tears down (unchanged);
- ``team_candidate_evaluation``: ONE admitted team candidate whose policy is
  reached over authenticated HTTPS. No checkpoint is downloaded and no local
  server starts; the bearer token arrives only through the provider's private
  runtime-secret channel, under the environment name the manifest records.

The adapter validates the sealed manifest member, then requires the members
and entrypoint teardown that run kind needs. A manifest that claims neither
shape is refused before any provider mutation.
"""

from __future__ import annotations

import re
import hashlib
import math
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest

SCHEMA = "native_task_arena_policy_canary_provider_bundle.v1"
RESULT_FILENAME = "native_task_arena_policy_canary_session_result.v1.json"
ENTRYPOINT = "provider_runtime/run_adp_arena_provider_runtime.sh"
PAIR = ["pi05_droid", "groot_n17_droid"]
PAIR_SHAPE = ("internal_policy_canary_paired_session", "internal_policy_canary", 20)
TEAM_SHAPE = ("team_candidate_evaluation_session", "team_candidate_evaluation", 10)
BETA_SHAPE = ("blueprint_beta_screening_session", "blueprint_beta_screening", 10)
# Mirrors native_task_arena_policy_canary_session.TEAM_MAX_CELLS_PER_SESSION (kept equal by a test).
TEAM_MAX_CELLS_PER_SESSION = 200
# Written by the team entrypoint in place of the pair's local-server teardown.
TEAM_ENTRYPOINT_MARKER = "blueprint_team_candidate_remote_policy_session"
_TEAM_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SECRET_FILE_ENV = re.compile(r"^[A-Z][A-Z0-9_]{2,58}_FILE$")
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PAIR_ENTRIES = frozenset({
    "provider_runtime/runtime_inputs/policy_execution_spec.pi05_droid.json",
    "provider_runtime/runtime_inputs/policy_execution_spec.groot_n17_droid.json",
    "provider_runtime/blueprint_pipeline/openpi_droid_policy_runtime.py",
    "provider_runtime/blueprint_pipeline/groot_n17_droid_policy_runtime.py",
    "provider_runtime/adp009d_policy_server_worker.py",
})


def _shape(payload: Mapping[str, Any]) -> tuple[str, str, int] | None:
    for shape in (PAIR_SHAPE, TEAM_SHAPE, BETA_SHAPE):
        if (payload.get("execution_mode"), payload.get("run_kind")) == shape[:2]:
            return shape
    return None


def _episode_shape_valid(payload: Mapping[str, Any], shape: tuple[str, str, int]) -> bool:
    """The pair runs its ten cells twice; a team session runs its own 1-200 cells once."""

    episodes, rollouts = payload.get("episodes_per_policy"), payload.get("learned_policy_rollout_count")
    if shape is BETA_SHAPE:
        return type(episodes) is int and 1 <= episodes <= 10 and rollouts == episodes
    if shape is TEAM_SHAPE:
        return (type(episodes) is int and 1 <= episodes <= TEAM_MAX_CELLS_PER_SESSION and rollouts == episodes)
    return episodes == 10 and rollouts == shape[2]


def _family_operator_inputs_valid(value: Any) -> bool:
    if (not isinstance(value, Mapping)
            or value.get("catalog_relative_path") != "runtime_operator_inputs/catalog.json"
            or not _SHA.fullmatch(str(value.get("runtime_catalog_digest") or ""))):
        return False
    rows = value.get("files")
    if not isinstance(rows, list) or not rows:
        return False
    names = []
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        name = str(row.get("relative_path") or "")
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or not name.startswith("runtime_operator_inputs/")
                or not _SHA.fullmatch(str(row.get("sha256") or ""))
                or type(row.get("size_bytes")) is not int or row["size_bytes"] < 1):
            return False
        names.append(name)
    return len(names) == len(set(names)) and value["catalog_relative_path"] in names


def validate_policy_canary_provider_manifest(payload: object) -> dict[str, Any]:
    """Return the readiness record for a sealed session manifest, or raise."""

    if not isinstance(payload, Mapping):
        raise TypeError("native_task_arena_policy_canary_manifest_not_mapping")
    minimum_gpu_ram_mb(payload)
    shape = _shape(payload)
    candidates = payload.get("candidate_ids")
    openwam = payload.get("team_policy_interface_id") == "openwam_piper_eef20.v1"
    if openwam:
        from .openwam_piper_bundle import validate_runtime_contract
        validate_runtime_contract(payload.get("openwam_runtime") or {})
        if minimum_gpu_ram_mb(payload) < 80_000:
            raise ValueError("openwam_gpu_memory_floor_invalid")
    molmoact2 = payload.get("team_policy_interface_id") == "molmoact2_yam_joint14.v1"
    if molmoact2:
        from .molmoact2_yam_bundle import validate_runtime_contract
        validate_runtime_contract(payload.get("molmoact2_runtime") or {})
        if minimum_gpu_ram_mb(payload) < 48_000:
            raise ValueError("molmoact2_gpu_memory_floor_invalid")
    molmoact2_droid = payload.get("team_policy_interface_id") == "molmoact2_droid_joint8.v1"
    if molmoact2_droid:
        from .molmoact2_droid_bundle import validate_runtime_contract
        validate_runtime_contract(payload.get("molmoact2_droid_runtime") or {})
        if minimum_gpu_ram_mb(payload) < 48_000:
            raise ValueError("molmoact2_droid_gpu_memory_floor_invalid")
    if shape is BETA_SHAPE:
        dependencies = payload.get("contract_python_dependencies") or []
        robolab = next((row for row in dependencies if row.get("distribution") == "robolab"), {})
        family = payload.get("evaluation_backend") == "task_family_native.v1"
        delivery_valid = (
            payload.get("team_policy_interface_id") == "task_family_policy_https.v1"
            and _SECRET_FILE_ENV.fullmatch(str(payload.get("team_credential_env") or "")) is not None
            and payload.get("secret_material_bundled") is False
            and _family_operator_inputs_valid(payload.get("task_family_operator_inputs"))
            and payload.get("pre_policy_site_qualification_required") is True
            and _SHA.fullmatch(str(payload.get("family_static_readiness_digest") or "")) is not None
        ) if family else (
            (("team_credential_env" not in payload and isinstance(candidates, list) and candidates
              and candidates[0] in [*PAIR, "cosmos3_edge_policy_droid"])
             or ((openwam or molmoact2 or molmoact2_droid) and "team_credential_env" not in payload and payload.get("secret_material_bundled") is False)
             or (payload.get("team_policy_interface_id") in {
                     "openpi_droid_https.v1", "blueprint_policy_https.v1", "dreamlabs_dream1.v1"}
                 and _SECRET_FILE_ENV.fullmatch(str(payload.get("team_credential_env") or "")) is not None
                 and payload.get("secret_material_bundled") is False))
            and payload.get("evaluation_backend") == "robolab.v1"
            and robolab.get("source_commit") == "b6cfcf6ccd2113c4cd1216d35b33024a7d5691da"
            and robolab.get("archive_sha256") == "sha256:16092baa43c5a2eb78f6dc8a330330e8e8bfc6a8cafce59efff560b26d6af278"
        )
        candidates_valid = (
            isinstance(candidates, list) and len(candidates) == 1
            and _TEAM_ID.fullmatch(str(candidates[0] or "")) is not None
            and delivery_valid
        )
    elif shape is TEAM_SHAPE:
        candidates_valid = (
            isinstance(candidates, list) and len(candidates) == 1
            and _TEAM_ID.fullmatch(str(candidates[0] or "")) is not None and candidates[0] not in PAIR
            and _SECRET_FILE_ENV.fullmatch(str(payload.get("team_credential_env") or "")) is not None
            and payload.get("secret_material_bundled") is False
        )
    else:
        candidates_valid = candidates == PAIR and "team_credential_env" not in payload
    if (
        shape is None
        or not candidates_valid
        or payload.get("schema_version") != SCHEMA
        or payload.get("status") != "ready"
        or payload.get("blockers") != []
        or payload.get("input_digest") != canonical_digest(payload, digest_field="input_digest")
        or payload.get("claim_ceiling") != "diagnostic_policy_execution"
        or not _episode_shape_valid(payload, shape)
        or payload.get("maximum_provider_allocations") != 1
        or payload.get("retry_cap") != 0
        or payload.get("expected_output_filename") != RESULT_FILENAME
        or payload.get("runtime_entrypoint") != ENTRYPOINT
        or payload.get("candidate_policy_queried") is not False
        or payload.get("provider_zero_required_after_return") is not True
    ):
        raise ValueError("native_task_arena_policy_canary_manifest_invalid")
    readiness = dict(payload)
    readiness["local_bundle_ready_for_remote_staging"] = True
    return readiness


def minimum_gpu_ram_mb(payload: Mapping[str, Any]) -> int:
    """Respect the sealed candidate memory floor without lowering legacy bounds."""
    memory = payload.get("minimum_gpu_memory_gb", 46)
    if type(memory) not in (int, float) or not 0 < memory <= 1024:
        raise ValueError("beta_screening_gpu_memory_bound_invalid")
    return max(46_000, math.ceil(memory * 1000))


def run_kind_required_entries(manifest: Mapping[str, Any]) -> set[str]:
    """Archive members this run kind needs beyond the shared session runtime."""

    if _shape(manifest) is BETA_SHAPE:
        candidate = str((manifest.get("candidate_ids") or [""])[0])
        required = {f"provider_runtime/runtime_inputs/policy_execution_spec.{candidate}.json"}
        if manifest.get("team_policy_interface_id") == "openwam_piper_eef20.v1":
            from .openwam_piper_bundle import RUNTIME_MODULES, validate_runtime_contract
            contract = validate_runtime_contract(manifest["openwam_runtime"])
            required.update("provider_runtime/blueprint_pipeline/" + name for name in RUNTIME_MODULES)
            required.update("provider_runtime/" + name for name in contract["required_entries"])
            required.update({f"provider_runtime/adp009d_policy_provisioning.{candidate}.sh",
                             "provider_runtime/adp009d_policy_server_worker.py",
                             "provider_runtime/blueprint_pipeline/team_candidate_policy_client.py"})
        if manifest.get("team_policy_interface_id") == "molmoact2_yam_joint14.v1":
            from .molmoact2_yam_bundle import RUNTIME_MODULES, validate_runtime_contract
            contract = validate_runtime_contract(manifest["molmoact2_runtime"])
            required.update("provider_runtime/blueprint_pipeline/" + name for name in RUNTIME_MODULES)
            required.update("provider_runtime/" + name for name in contract["required_entries"])
            required.update({f"provider_runtime/adp009d_policy_provisioning.{candidate}.sh",
                             "provider_runtime/adp009d_policy_server_worker.py",
                             "provider_runtime/blueprint_pipeline/team_candidate_policy_client.py"})
        if manifest.get("team_policy_interface_id") == "molmoact2_droid_joint8.v1":
            from .molmoact2_droid_bundle import RUNTIME_MODULES, validate_runtime_contract
            contract = validate_runtime_contract(manifest["molmoact2_droid_runtime"])
            required.update("provider_runtime/blueprint_pipeline/" + name for name in RUNTIME_MODULES)
            required.update("provider_runtime/" + name for name in contract["required_entries"])
            required.update({f"provider_runtime/adp009d_policy_provisioning.{candidate}.sh",
                             "provider_runtime/adp009d_policy_server_worker.py"})
        if candidate == "cosmos3_edge_policy_droid":
            required.update({"provider_runtime/cosmos_edge_policy_snapshot_manifest.json",
                             "provider_runtime/adp009d_policy_provisioning.cosmos3_edge_policy_droid.sh",
                             "provider_runtime/blueprint_pipeline/cosmos_edge_beta_runtime.py",
                             "provider_runtime/blueprint_pipeline/cosmos_edge_droid_policy_server.py",
                             "provider_runtime/blueprint_pipeline/cosmos_edge_droid_policy_runtime.py"})
        if manifest.get("evaluation_backend") == "task_family_native.v1":
            required.update("provider_runtime/blueprint_pipeline/" + name for name in (
                "task_family_native_runtime.py", "task_family_native_contracts.py", "task_family_native_controls.py",
                "task_family_control_omission.py",
                "task_family_policy_execution.py", "task_family_policy_worker_execution.py", "task_family_policy_canary_execution.py",
                "task_family_policy_https_client.py", "task_family_result_admission.py",
                "task_evaluation_task_family_native_adapter.py", "task_family_scoring.py", "core/common.py"))
            required.update("provider_runtime/" + row["relative_path"]
                            for row in manifest["task_family_operator_inputs"]["files"])
        else:
            required.update({"provider_runtime/runtime_inputs/native_task_arena_construction_result.v1.json",
                "provider_runtime/blueprint_pipeline/beta_robolab_runtime.py", "provider_runtime/robolab/eval/episode.py",
                "provider_runtime/robolab/eval/base_client.py", "provider_runtime/robolab-source.v1.json"})
        if manifest.get("team_credential_env"):
            required.update({"provider_runtime/blueprint_pipeline/team_candidate_policy_client.py",
                "provider_runtime/blueprint_pipeline/droid_https_policy_client.py",
                "provider_runtime/blueprint_pipeline/core/security_controls.py"})
        return required
    if _shape(manifest) is not TEAM_SHAPE:
        return set(_PAIR_ENTRIES) | {"provider_runtime/runtime_inputs/native_task_arena_construction_result.v1.json"}
    candidate = str((manifest.get("candidate_ids") or [""])[0])
    required = {
        f"provider_runtime/runtime_inputs/policy_execution_spec.{candidate}.json",
        "provider_runtime/blueprint_pipeline/team_candidate_policy_client.py",
        "provider_runtime/blueprint_pipeline/droid_https_policy_client.py",
        "provider_runtime/blueprint_pipeline/core/security_controls.py",
        "provider_runtime/runtime_inputs/native_task_arena_construction_result.v1.json",
    }
    if manifest.get("team_policy_interface_id") == "dreamlabs_dream1.v1":
        required.update({"provider_runtime/blueprint_pipeline/dreamlabs_policy_client.py",
                         "provider_runtime/blueprint_pipeline/team_arm_kinematics.py"})
    return required


def run_kind_entrypoint_blockers(manifest: Mapping[str, Any], entrypoint_text: str) -> list[str]:
    """The pair must tear down its local servers; a team run must start none."""

    if _shape(manifest) is BETA_SHAPE:
        valid = ("blueprint_beta_screening" in entrypoint_text
                 and "adp_arena_provider_runner.py" in entrypoint_text
                 and "trap teardown_servers EXIT INT TERM HUP" in entrypoint_text)
        if manifest.get("team_policy_interface_id") == "openwam_piper_eef20.v1":
            valid = valid and ('openwam_client:$RUNTIME_DIR' in entrypoint_text
                               and '--terminate-ready-server' in entrypoint_text)
        if manifest.get("team_policy_interface_id") in {"molmoact2_yam_joint14.v1", "molmoact2_droid_joint8.v1"}:
            valid = valid and '--terminate-ready-server' in entrypoint_text
        if manifest.get("evaluation_backend") == "task_family_native.v1":
            valid = valid and 'export BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH="$RUNTIME_DIR/runtime_operator_inputs/catalog.json"' in entrypoint_text
    elif _shape(manifest) is not TEAM_SHAPE:
        valid = "adp009d_policy_server_worker.py" in entrypoint_text
    else:
        valid = (TEAM_ENTRYPOINT_MARKER in entrypoint_text
                 and "adp009d_policy_provisioning" not in entrypoint_text
                 and "adp009d_policy_server_worker.py" not in entrypoint_text)
    return [] if valid else ["provider_runner_missing_policy_canary_session_runtime_contract"]


def verify_family_operator_input_bytes(manifest: Mapping[str, Any], read_member: Any) -> None:
    """Reprove the operator config retained inside a family provider archive."""
    if manifest.get("evaluation_backend") != "task_family_native.v1":
        return
    for row in manifest["task_family_operator_inputs"]["files"]:
        raw = read_member("provider_runtime/" + row["relative_path"])
        if len(raw) != row["size_bytes"] or "sha256:" + hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("task_family_operator_input_bytes_changed")


__all__ = [
    "TEAM_ENTRYPOINT_MARKER",
    "run_kind_entrypoint_blockers",
    "run_kind_required_entries",
    "validate_policy_canary_provider_manifest",
]
