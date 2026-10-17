"""Stage one team candidate's runtime inputs on a scene's half of the evaluation.

Beta journey step: L5 team policy evaluation. A team run needs only the
scene's half: its compiled packet, the runtime source, the sealed success
contract and the per-cell control binding. That half comes from a prepared
scene's own compiled packet (``team_scene_packet``) or from the scene's frozen
internal Quick-10 inputs (``task_evaluation_policy_canary_runtime_inputs.v1``,
run kind ``internal_policy_canary``), read by :func:`scene_inputs_from_frozen_pair`.
For one shard of the scored matrix (``team_evaluation_plan``) it writes:

- a team activation manifest (the session's cells and seeds, run kind
  ``team_candidate_evaluation``, a run id of its own);
- team runtime inputs carrying the admission's runtime binding;
- a session authority bounded by the funding cap and the profile's hard TTL;
- the candidate's sealed execution spec (no secret material).

Nothing here touches a network, a provider or a secret.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_policy_canary_session import (
    CLAIM_CEILING,
    RUN_KIND,
    TEAM_RUN_KIND,
    build_session_authority,
    validate_runtime_input_manifest,
)
from .team_candidate_quick10_admission import execution_spec, runtime_binding, validate_admission

INPUTS_SCHEMA = "task_evaluation_policy_canary_runtime_inputs.v1"
ACTIVATION_SCHEMA = "task_evaluation_policy_campaign_activation.v1"
# The scene's half of a run's inputs: everything but the run, its candidates and its cells.
SCENE_INPUT_FIELDS = ("scene_revision_digest", "configuration_digest", "plan_digest", "task_success_contract",
                      "task_success_contract_digest", "base_native_packet", "runtime_source", "construction_result",
                      "execution_authority", "capture_contract")
CELL_KEYS = ("cell_id", "seed", "cell_spec_digest", "family")


class TeamCandidateStagingError(ValueError):
    """The frozen Quick-10 inputs cannot carry this admitted candidate."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TeamCandidateStagingError("team_candidate_staging_record_invalid")
    return value


def _record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "size_bytes": len(data), "sha256": "sha256:" + hashlib.sha256(data).hexdigest()}


def _write(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def frozen_quick10_inputs(inputs_path: Path, activation_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The scene's sealed internal Quick-10 inputs and activation, or raise."""

    inputs = validate_runtime_input_manifest(_read(inputs_path))
    activation = _read(activation_path)
    if (inputs["run_kind"] != RUN_KIND or activation.get("run_kind") != RUN_KIND
            or activation.get("activation_digest") != inputs["activation_digest"]
            or activation.get("activation_digest") != canonical_digest(activation, digest_field="activation_digest")):
        raise TeamCandidateStagingError("team_candidate_frozen_quick10_inputs_invalid")
    return inputs, activation


def scene_inputs_from_frozen_pair(inputs_path: Path, activation_path: Path) -> dict[str, Any]:
    """The scene half of a scene's frozen internal Quick-10 inputs, with its per-cell control binding."""

    pair, _activation = frozen_quick10_inputs(inputs_path, activation_path)
    return {**{key: pair[key] for key in SCENE_INPUT_FIELDS if key in pair},
            "control_diagnostic": dict(pair["cells"][0]["control_diagnostic"])}


def stage_team_runtime(*, root: Path, run_id: str, scene_inputs: Mapping[str, Any],
                       admission: Mapping[str, Any], prompt: str, max_policy_queries: int,
                       hard_cap_usd: float, hard_ttl_seconds: int, cells: list[Mapping[str, Any]],
                       team_embodiment: Mapping[str, Any] | None = None,
                       robot_free_probe: Mapping[str, Any] | None = None) -> dict[str, Path]:
    """Write the team activation, inputs, authority and spec under ``root``; return their paths.

    ``cells`` are this session's cells in the runtime-input shape
    (``team_evaluation_plan.runtime_cells``). ``team_embodiment`` is the staged
    binding of a candidate on its own robot.
    """

    admitted = validate_admission(admission)
    binding = runtime_binding(admitted)
    if (binding["scene_revision_digest"] != scene_inputs.get("scene_revision_digest")
            or binding["task_success_contract_digest"] != scene_inputs["task_success_contract_digest"]):
        raise TeamCandidateStagingError("team_candidate_admission_scene_binding_mismatch")
    candidate = [admitted["candidate_id"]]
    session_cells = [dict(cell) for cell in cells]
    units = [{"campaign_unit_id": f"{run_id}-{cell['cell_id']}", "candidate_ids": candidate,
              **{key: cell[key] for key in CELL_KEYS}} for cell in session_cells]
    team_activation = {"schema_version": ACTIVATION_SCHEMA, "run_id": run_id, "run_kind": TEAM_RUN_KIND,
                       "claim_ceiling": CLAIM_CEILING, "candidate_ids": candidate,
                       "task_success_contract": scene_inputs["task_success_contract"],
                       "task_success_contract_digest": scene_inputs["task_success_contract_digest"],
                       "campaign_unit_count": len(units), "campaign_units": units, "activation_digest": ""}
    team_activation["activation_digest"] = canonical_digest(team_activation, digest_field="activation_digest")
    inputs = {"schema_version": INPUTS_SCHEMA, "run_id": run_id, "run_kind": TEAM_RUN_KIND,
              "claim_ceiling": CLAIM_CEILING,
              **{key: scene_inputs[key] for key in SCENE_INPUT_FIELDS if key in scene_inputs},
              "candidate_ids": candidate, "cells": session_cells,
              "matrix_digest": canonical_digest({"ordered_cells": [cell["cell_id"] for cell in session_cells]}),
              "activation_digest": team_activation["activation_digest"], "team_candidate_admission": binding,
              "runtime_inputs_digest": ""}
    inputs["runtime_inputs_digest"] = canonical_digest(inputs, digest_field="runtime_inputs_digest")
    inputs = validate_runtime_input_manifest(inputs)
    directory = Path(root) / "runtime_inputs"
    activation_path = _write(directory / "team_candidate_activation.json", team_activation)
    inputs_path = _write(directory / "policy_canary_runtime_inputs.json", inputs)
    authority = build_session_authority(
        activation_manifest=team_activation, activation_record=_record(activation_path), runtime_inputs=inputs,
        runtime_input_record=_record(inputs_path),
        resource_name="blueprint-native-task-policy-canary-" + canonical_digest({"run_id": run_id})[7:39],
        hard_cap_usd=float(hard_cap_usd), hard_ttl_seconds=int(hard_ttl_seconds),
        hard_deadline_epoch=(datetime.fromisoformat(admitted["execution_deadline_at_iso"].replace("Z", "+00:00")).timestamp()
                             if admitted.get("execution_deadline_at_iso") else None))
    spec = execution_spec(admitted, task_success_contract=inputs["task_success_contract"], prompt=prompt,
                          max_policy_queries=max_policy_queries, team_embodiment=team_embodiment,
                          robot_free_probe=robot_free_probe)
    return {"activation": activation_path, "runtime_inputs": inputs_path,
            "session_authority": _write(directory / "policy_canary_session_authority.json", authority),
            "execution_spec": _write(directory / f"policy_execution_spec.{candidate[0]}.json", spec)}


__all__ = ["SCENE_INPUT_FIELDS", "TeamCandidateStagingError", "frozen_quick10_inputs",
           "scene_inputs_from_frozen_pair", "stage_team_runtime"]
