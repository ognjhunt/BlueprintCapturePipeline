"""Derive one team candidate's runtime inputs from the scene's frozen Quick-10 inputs.

Beta journey step: L5 team policy evaluation. The scene's internal Quick-10
inputs (``task_evaluation_policy_canary_runtime_inputs.v1``, run kind
``internal_policy_canary``) freeze the scene packet, ten cells, their seeds and
the sealed success contract. A team run keeps the scene and contract, replaces
the frozen pair with the one admitted candidate, and runs either those ten
published cells (practice) or one shard of the scored matrix
(``team_evaluation_plan``):

- a team activation manifest (the session's cells and seeds, run kind
  ``team_candidate_evaluation``, a run id of its own);
- team runtime inputs carrying the admission's runtime binding, with the
  pair-only records (policy readiness, diagnostic continuation, the pair's
  resource authority) removed;
- a session authority bounded by the funding cap and the profile's hard TTL;
- the candidate's sealed execution spec (no secret material).

Nothing here touches a network, a provider or a secret.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_policy_canary_session import (
    RUN_KIND,
    TEAM_RUN_KIND,
    build_session_authority,
    validate_runtime_input_manifest,
)
from .team_candidate_quick10_admission import execution_spec, runtime_binding, validate_admission

PAIR_ONLY_INPUT_FIELDS = ("policy_readiness", "diagnostic_continuation_protocol", "resource_authority")


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


def stage_team_runtime(*, root: Path, run_id: str, pair_inputs_path: Path, pair_activation_path: Path,
                       admission: Mapping[str, Any], prompt: str, max_policy_queries: int,
                       hard_cap_usd: float, hard_ttl_seconds: int,
                       cells: list[Mapping[str, Any]] | None = None,
                       team_embodiment: Mapping[str, Any] | None = None,
                       robot_free_probe: Mapping[str, Any] | None = None) -> dict[str, Path]:
    """Write the team activation, inputs, authority and spec under ``root``; return their paths.

    ``cells`` are this session's cells in the runtime-input shape; None runs the
    scene's ten frozen Quick-10 cells. ``team_embodiment`` is the staged binding
    of a candidate on its own robot.
    """

    admitted = validate_admission(admission)
    pair, activation = frozen_quick10_inputs(pair_inputs_path, pair_activation_path)
    binding = runtime_binding(admitted)
    if (binding["scene_revision_digest"] != pair.get("scene_revision_digest")
            or binding["task_success_contract_digest"] != pair["task_success_contract_digest"]):
        raise TeamCandidateStagingError("team_candidate_admission_scene_binding_mismatch")
    candidate = [admitted["candidate_id"]]
    session_cells = [dict(cell) for cell in (pair["cells"] if cells is None else cells)]
    template = dict((activation.get("campaign_units") or [{}])[0])
    units = [{**template, "campaign_unit_id": f"{run_id}-{cell['cell_id']}", "candidate_ids": candidate,
              **{key: cell[key] for key in ("cell_id", "seed", "cell_spec_digest", "family")}}
             for cell in session_cells]
    team_activation = {**activation, "run_id": run_id, "run_kind": TEAM_RUN_KIND, "candidate_ids": candidate,
                       "campaign_unit_count": len(units), "campaign_units": units, "activation_digest": ""}
    if isinstance(activation.get("paired_session_request"), Mapping):
        team_activation["paired_session_request"] = {
            **activation["paired_session_request"], "candidate_ids": candidate, "policy_count": 1,
            "episodes_per_policy": len(units), "learned_policy_rollout_count": len(units),
            "cells": [{key: cell[key] for key in ("cell_id", "seed", "cell_spec_digest", "family")}
                      for cell in session_cells]}
    team_activation["activation_digest"] = canonical_digest(team_activation, digest_field="activation_digest")
    inputs = {key: value for key, value in pair.items() if key not in PAIR_ONLY_INPUT_FIELDS}
    inputs.update(run_id=run_id, run_kind=TEAM_RUN_KIND, candidate_ids=candidate, cells=session_cells,
                  matrix_digest=canonical_digest({"ordered_cells": [cell["cell_id"] for cell in session_cells]}),
                  activation_digest=team_activation["activation_digest"], team_candidate_admission=binding,
                  runtime_inputs_digest="")
    inputs["runtime_inputs_digest"] = canonical_digest(inputs, digest_field="runtime_inputs_digest")
    inputs = validate_runtime_input_manifest(inputs)
    directory = Path(root) / "runtime_inputs"
    activation_path = _write(directory / "team_candidate_activation.json", team_activation)
    inputs_path = _write(directory / "policy_canary_runtime_inputs.json", inputs)
    authority = build_session_authority(
        activation_manifest=team_activation, activation_record=_record(activation_path), runtime_inputs=inputs,
        runtime_input_record=_record(inputs_path),
        resource_name="blueprint-native-task-policy-canary-" + canonical_digest({"run_id": run_id})[7:23],
        hard_cap_usd=float(hard_cap_usd), hard_ttl_seconds=int(hard_ttl_seconds))
    spec = execution_spec(admitted, task_success_contract=inputs["task_success_contract"], prompt=prompt,
                          max_policy_queries=max_policy_queries, team_embodiment=team_embodiment,
                          robot_free_probe=robot_free_probe)
    return {"activation": activation_path, "runtime_inputs": inputs_path,
            "session_authority": _write(directory / "policy_canary_session_authority.json", authority),
            "execution_spec": _write(directory / f"policy_execution_spec.{candidate[0]}.json", spec)}


__all__ = ["TeamCandidateStagingError", "frozen_quick10_inputs", "stage_team_runtime"]
