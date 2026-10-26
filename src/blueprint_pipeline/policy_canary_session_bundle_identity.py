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
from collections.abc import Mapping
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


def validate_policy_canary_provider_manifest(payload: object) -> dict[str, Any]:
    """Return the readiness record for a sealed session manifest, or raise."""

    if not isinstance(payload, Mapping):
        raise TypeError("native_task_arena_policy_canary_manifest_not_mapping")
    shape = _shape(payload)
    candidates = payload.get("candidate_ids")
    if shape is BETA_SHAPE:
        dependencies = payload.get("contract_python_dependencies") or []
        robolab = next((row for row in dependencies if row.get("distribution") == "robolab"), {})
        candidates_valid = (
            isinstance(candidates, list) and len(candidates) == 1
            and _TEAM_ID.fullmatch(str(candidates[0] or "")) is not None
            and (("team_credential_env" not in payload and candidates[0] in PAIR)
                 or (payload.get("team_policy_interface_id") == "openpi_droid_https.v1"
                     and _SECRET_FILE_ENV.fullmatch(str(payload.get("team_credential_env") or "")) is not None
                     and payload.get("secret_material_bundled") is False))
            and payload.get("evaluation_backend") == "robolab.v1"
            and robolab.get("source_commit") == "b6cfcf6ccd2113c4cd1216d35b33024a7d5691da"
            and robolab.get("archive_sha256") == "sha256:16092baa43c5a2eb78f6dc8a330330e8e8bfc6a8cafce59efff560b26d6af278"
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


def run_kind_required_entries(manifest: Mapping[str, Any]) -> set[str]:
    """Archive members this run kind needs beyond the shared session runtime."""

    if _shape(manifest) is BETA_SHAPE:
        candidate = str((manifest.get("candidate_ids") or [""])[0])
        required = {
            f"provider_runtime/runtime_inputs/policy_execution_spec.{candidate}.json",
            "provider_runtime/blueprint_pipeline/beta_robolab_runtime.py",
            "provider_runtime/robolab/eval/episode.py",
            "provider_runtime/robolab/eval/base_client.py",
            "provider_runtime/robolab-source.v1.json",
        }
        if manifest.get("team_credential_env"):
            required.update({"provider_runtime/blueprint_pipeline/team_candidate_policy_client.py",
                "provider_runtime/blueprint_pipeline/droid_https_policy_client.py",
                "provider_runtime/blueprint_pipeline/core/security_controls.py"})
        return required
    if _shape(manifest) is not TEAM_SHAPE:
        return set(_PAIR_ENTRIES)
    candidate = str((manifest.get("candidate_ids") or [""])[0])
    required = {
        f"provider_runtime/runtime_inputs/policy_execution_spec.{candidate}.json",
        "provider_runtime/blueprint_pipeline/team_candidate_policy_client.py",
        "provider_runtime/blueprint_pipeline/droid_https_policy_client.py",
        "provider_runtime/blueprint_pipeline/core/security_controls.py",
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
    elif _shape(manifest) is not TEAM_SHAPE:
        valid = "adp009d_policy_server_worker.py" in entrypoint_text
    else:
        valid = (TEAM_ENTRYPOINT_MARKER in entrypoint_text
                 and "adp009d_policy_provisioning" not in entrypoint_text
                 and "adp009d_policy_server_worker.py" not in entrypoint_text)
    return [] if valid else ["provider_runner_missing_policy_canary_session_runtime_contract"]


__all__ = [
    "TEAM_ENTRYPOINT_MARKER",
    "run_kind_entrypoint_blockers",
    "run_kind_required_entries",
    "validate_policy_canary_provider_manifest",
]
