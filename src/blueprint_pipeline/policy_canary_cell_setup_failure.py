"""Name a matrix cell that stopped before any policy ran, and stop the matrix there.

Each cell runs in its own interpreter and opens its own simulator. A cell that
could not open the runtime, pass the pre-policy observation gate, or load a
policy measured nothing about the policy, and the next cell would fail the same
way on the same image and scene. The parent checks each cell as it finishes,
names the stage from the cell's own progress markers and sealed failure record,
seals a blocked session result that says so, and starts no further cell.

A gate refusal in the retained pre-policy shape keeps its existing handling:
it is per cell, and the matrix continues. The diagnostic continuation's
first-cell gate keeps cell 0 of its own runs. A beta gate that the session
continued past with observation-quality flags did not stop the cell: it is
never named as the observation-gate stage, and its verdict reads ``flagged``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from blueprint_pipeline.decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "policy_canary_cell_setup_failure.v1"
# "session" places the code in the website's launch category ("Could not start
# in our simulator. This is ours to fix.").
BLOCKER_PREFIX = "policy_canary_session_setup_failed"
STAGES = ("runtime_open", "observation_gate", "policy_load")
SETUP_FAILURE_BLOCKERS = frozenset(f"{BLOCKER_PREFIX}:{stage}" for stage in STAGES)
PROGRESS_FILENAME = "cell_progress.log"
# The fixed milestones a cell reports; nothing else is ever written or read.
CELL_PROGRESS_STAGES = frozenset({
    "static_preflight_passed", "isaac_launch_started", "isaac_launch_completed",
    "observation_gate_passed", "policy_loaded", "episode_completed",
})
_MARKER = re.compile(r"BLUEPRINT_POLICY_CANARY_PROGRESS:cell=(\d{1,4}):stage=([a-z_]{1,64})")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,159}")
_FILE = re.compile(r"[A-Za-z0-9_.-]{1,160}")
_CODE = re.compile(r"[a-z][a-z0-9_]*(?::[A-Za-z0-9_.-]{1,80})?")
_GATE_REASON = re.compile(r"[A-Za-z0-9_.:+-]{1,256}")
_MAX_PROGRESS_BYTES = 64 * 1024
_MAX_FRAMES = 24
_MAX_CODES = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file_record(output_root: Path, path: Path, result_digest: Any) -> dict[str, Any]:
    return {"relative_path": path.relative_to(output_root).as_posix(), "size_bytes": path.stat().st_size,
            "sha256": _sha256(path), "result_digest": result_digest}


def cell_progress_stages(child_root: Path, index: int) -> list[str]:
    """The fixed milestones this cell recorded, in order, each once."""
    path = Path(child_root) / PROGRESS_FILENAME
    try:
        if path.is_symlink() or not path.is_file():
            return []
        with path.open("rb") as stream:
            raw = stream.read(_MAX_PROGRESS_BYTES)
    except OSError:
        return []
    stages: list[str] = []
    for line in raw.decode("ascii", "replace").splitlines():
        match = _MARKER.fullmatch(line.strip())
        if match and int(match[1]) == index and match[2] in CELL_PROGRESS_STAGES and match[2] not in stages:
            stages.append(match[2])
    return stages


def _source_trace(value: Any) -> list[dict[str, Any]]:
    frames = []
    for frame in value if isinstance(value, list) else []:
        if (isinstance(frame, Mapping) and isinstance(frame.get("file"), str) and _FILE.fullmatch(frame["file"])
                and isinstance(frame.get("function"), str) and _IDENTIFIER.fullmatch(frame["function"])
                and type(frame.get("line")) is int and frame["line"] >= 0):
            frames.append({"file": frame["file"], "function": frame["function"], "line": frame["line"]})
    return frames[:_MAX_FRAMES]


def _gate_flags(child: Mapping[str, Any]) -> list[str] | None:
    """The flags a session continued with past its gate, re-derived from the retained gate; else ``None``."""
    from blueprint_pipeline.native_task_arena_policy_canary_session import observation_quality_flags

    flags = observation_quality_flags(child.get("preload_observation_gate"), run_kind=child.get("run_kind"))
    return flags if (flags and child.get("observation_integrity_flagged") is True
                     and child.get("observation_quality_flags") == flags) else None


def _stage(child: Mapping[str, Any], reached: list[str], functions: set[str]) -> str:
    gate = child.get("preload_observation_gate")
    if "isaac_launch_started" in reached and "isaac_launch_completed" not in reached:
        return "runtime_open"
    if "load_policy" in functions or "policy_loaded" in reached:
        return "policy_load"
    # The session continued past a flagged gate, so whatever stopped it came after the gate.
    if _gate_flags(child) is not None:
        return "policy_load"
    # The gate returned a refusal, raised, or its site qualification was refused.
    if gate is not None or "prepolicy_observation_gate" in functions:
        return "observation_gate"
    if "observation_gate_passed" in reached:
        return "policy_load"
    # No gate verdict and no gate frame: the session never opened.
    return "runtime_open"


def cell_setup_failure(*, child: Any, inputs: Mapping[str, Any], index: int, child_root: Path,
                       output_root: Path) -> dict[str, Any] | None:
    """The named setup failure of a finished cell that loaded and queried no policy, else ``None``.

    ``None`` leaves every other child to its existing validation: a completed
    cell, a cell that loaded or queried a policy, a control stop, and the
    retained pre-policy observation-gate refusal.
    """
    from blueprint_pipeline.native_task_arena_policy_canary_session import (
        PROVIDER_RESULT_FILENAME, RESULT_SCHEMA_VERSION,
    )
    from blueprint_pipeline.team_evaluation_worker_retention import _validate_isolated_cell_result

    if (not isinstance(child, Mapping) or child.get("schema_version") != RESULT_SCHEMA_VERSION
            or type(child.get("selected_cell_index")) is not int or child["selected_cell_index"] != index
            or child.get("status") != "blocked" or child.get("episodes") != [] or child.get("policy_loads") != []
            or child.get("candidate_policy_queried") is not False or child.get("execution_control_stop") is not None
            or not isinstance(child.get("session_failure_type"), str)
            or not _IDENTIFIER.fullmatch(child["session_failure_type"])
            or child.get("task_success_contract") != inputs.get("task_success_contract")
            or child.get("task_success_contract_digest") != inputs.get("task_success_contract_digest")
            or child.get("result_digest") != canonical_digest(child, digest_field="result_digest")
            or (index == 0 and inputs.get("diagnostic_continuation_protocol") is not None)):
        return None
    try:
        if _validate_isolated_cell_result(child=child, inputs=inputs, index=index) is False:
            return None
    except RuntimeError:
        pass
    else:  # pragma: no cover - a blocked child with no episode never validates as completed
        return None
    reached = cell_progress_stages(child_root, index)
    trace = _source_trace(child.get("session_failure_source_trace"))
    stage = _stage(child, reached, {frame["function"] for frame in trace})
    gate = child.get("preload_observation_gate")
    flags = _gate_flags(child)
    codes = child.get("session_failure_codes")
    cell = inputs["cells"][index]
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "stage": stage, "blocker": f"{BLOCKER_PREFIX}:{stage}",
        "cell_index": index, "cell_id": str(cell["cell_id"]), "seed": cell["seed"],
        "failure_type": child["session_failure_type"],
        "failure_codes": sorted({code for code in codes if isinstance(code, str) and _CODE.fullmatch(code)})[:_MAX_CODES]
        if isinstance(codes, list) else [],
        "source_trace": trace,
        "progress_stages": reached,
        "observation_gate_verdict": None if gate is None else (
            "passed" if isinstance(gate, Mapping) and gate.get("policy_observation_integrity_passed") is True
            else "flagged" if flags else "refused"),
        **({"observation_quality_flags": flags} if flags else {}),
        "observation_gate_blockers": sorted({
            item for item in (gate.get("blockers") if isinstance(gate, Mapping) else None) or []
            if isinstance(item, str) and _GATE_REASON.fullmatch(item)})[:_MAX_CODES],
        "candidate_policy_loaded": False, "candidate_policy_queried": False,
        "child_result": _file_record(Path(output_root), Path(child_root) / PROVIDER_RESULT_FILENAME,
                                     child["result_digest"]),
        "record_digest": "",
    }
    record["record_digest"] = canonical_digest(record, digest_field="record_digest")
    return record


def seal_cell_setup_failure(*, output_root: Path, inputs: Mapping[str, Any], authority: Mapping[str, Any],
                            children: list[Mapping[str, Any]], failure: Mapping[str, Any],
                            construction_lineage_mode: str, aggregate: Callable[..., dict[str, Any]],
                            seal_result: Callable[..., None]) -> dict[str, Any]:
    """Seal a blocked session result naming the failed cell; no further cell starts.

    Cells before it keep their measured episodes when they aggregate; any that
    do not stay referenced by their exact bytes, never re-scored or invented.
    """
    from blueprint_pipeline.native_task_arena_policy_canary_session import (
        BETA_RUN_KIND, CLAIM_CEILING, PROVIDER_RESULT_FILENAME, RESULT_SCHEMA_VERSION, RUN_KIND,
    )
    from blueprint_pipeline.policy_execution_evidence import candidate_execution_kinds, retained_rollout_counts
    from blueprint_pipeline.task_family_policy_canary_execution import beta_result_binding

    output_root = Path(output_root)
    prefix = list(children[:-1])
    result: dict[str, Any] | None = None
    if prefix:
        try:
            result = aggregate(inputs={**inputs, "cells": inputs["cells"][:len(prefix)]}, authority=authority,
                               child_results=prefix, output_root=output_root,
                               construction_lineage_mode=construction_lineage_mode)
        except (RuntimeError, ValueError):
            result = None
    if result is None:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION, "run_kind": inputs.get("run_kind") or RUN_KIND,
            "claim_ceiling": CLAIM_CEILING, "candidate_ids": list(inputs["candidate_ids"]),
            "task_success_contract": inputs["task_success_contract"],
            "task_success_contract_digest": inputs["task_success_contract_digest"],
            "episodes_per_policy": 0, **retained_rollout_counts(candidate_execution_kinds(inputs), []),
            "episodes": [], "policy_loads": [], "candidate_policy_queried": False,
            "artifact_inventory": [], "artifact_inventory_digest": canonical_digest({"value": []}),
            "provider_allocations_observed": None, "retry_cap": 0, "warm_session_open_count": 1,
            "session_closeout": {"status": "runtime_closed_pending_provider_teardown", "runtime_closed": True,
                                 "provider_closeout_pending": True},
            "session_failure_type": None, "scene_promotion_performed": False,
            "official_ranking_performed": False, "provider_zero_required_after_return": True,
            "construction_lineage_mode": construction_lineage_mode,
            "matrix_digest": inputs.get("matrix_digest") or canonical_digest({"value": inputs["cells"]}),
            **(beta_result_binding(inputs) if inputs.get("run_kind") == BETA_RUN_KIND else {}),
            **({"execution_release": authority["execution_release"]}
               if authority.get("execution_release") is not None else {}),
        }
        if prefix:
            result["retained_prefix_cell_results"] = [
                _file_record(output_root, output_root / "cell_runs" / f"{index:02d}" / PROVIDER_RESULT_FILENAME,
                             child.get("result_digest"))
                for index, child in enumerate(prefix)]
    result.update(status="blocked", blockers=[failure["blocker"]],
                  planned_episodes_per_policy=len(inputs["cells"]),
                  isolated_simulation_process_count=len(children), execution_stopped_after_cell=len(children),
                  stopped_before_next_cell=True, automatic_retry_performed=False,
                  authority_digest=authority["authority_digest"],
                  runtime_inputs_digest=inputs["runtime_inputs_digest"], cell_setup_failure=dict(failure))
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    seal_result(result_path=output_root / PROVIDER_RESULT_FILENAME, result=result)
    return result


def sealed_setup_failure_blocker(result: Any) -> str | None:
    """The one named setup blocker of a sealed provider session result, else ``None``.

    Only a code from the closed vocabulary is returned; nothing in the record
    is echoed. The caller authenticates the result's binding to its attempt.
    """
    from blueprint_pipeline.native_task_arena_policy_canary_session import CLAIM_CEILING, RESULT_SCHEMA_VERSION

    if not isinstance(result, Mapping):
        return None
    failure, blockers = result.get("cell_setup_failure"), result.get("blockers")
    planned, stopped = result.get("planned_episodes_per_policy"), result.get("execution_stopped_after_cell")
    if (result.get("schema_version") != RESULT_SCHEMA_VERSION or result.get("status") != "blocked"
            or result.get("claim_ceiling") != CLAIM_CEILING or result.get("stopped_before_next_cell") is not True
            or result.get("official_ranking_performed") is not False
            or result.get("scene_promotion_performed") is not False
            or not isinstance(blockers, list) or len(blockers) != 1 or blockers[0] not in SETUP_FAILURE_BLOCKERS
            or not isinstance(failure, Mapping) or failure.get("schema_version") != SCHEMA_VERSION
            or failure.get("stage") not in STAGES or failure.get("blocker") != f"{BLOCKER_PREFIX}:{failure['stage']}"
            or failure["blocker"] != blockers[0]
            or failure.get("candidate_policy_loaded") is not False or failure.get("candidate_policy_queried") is not False
            or failure.get("record_digest") != canonical_digest(failure, digest_field="record_digest")
            or type(planned) is not int or type(stopped) is not int or type(failure.get("cell_index")) is not int
            or not 0 <= failure["cell_index"] == stopped - 1 < planned
            or result.get("result_digest") != canonical_digest(result, digest_field="result_digest")):
        return None
    return f"{BLOCKER_PREFIX}:{failure['stage']}"


__all__ = [
    "BLOCKER_PREFIX", "CELL_PROGRESS_STAGES", "SETUP_FAILURE_BLOCKERS", "STAGES",
    "cell_progress_stages", "cell_setup_failure", "seal_cell_setup_failure", "sealed_setup_failure_blocker",
]
