"""Robot-free readiness: score the native probe with the exact configured success contract.

Owner decision 2026-10-12: robots never run in construction. A configured
articulated scene is ready for robot teams' evaluation when, under the same
scorer their episodes are judged by, the scene left untouched does not succeed
and the task joint driven into its goal interval and then released does.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .adp_task_scoring import (
    JOINT_HARD_LIMIT_SOLVER_RESIDUAL_RAD,
    TaskNeutralScoringError,
    score_articulated_task_episode,
    validate_articulated_task_spec,
)
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_rigid_relocation_native_adapter import (
    NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH,
    NATIVE_PHYSICS_FREQUENCY_HZ,
)
from .task_evaluation_scene_configuration_native_import_driver import ROBOT_FREE_PROBE_SCHEMA

SCHEMA = "task_evaluation_configured_scene_robot_free_readiness.v1"


def _scorer_trace(rows: Sequence[Mapping[str, Any]], *, spec: Mapping[str, Any],
                  native_names: Mapping[str, str | None]) -> list[dict[str, Any]]:
    """Probe readings as scorer samples. A fixed joint has no DOF and holds its reset."""

    resets, limits = spec["joint_reset_positions"], spec["joint_hard_limits"]
    trace = []
    for index, row in enumerate(rows):
        positions, velocities = {}, {}
        for joint_id, reset in resets.items():
            name = native_names.get(joint_id)
            position, velocity = (float(reset), 0.0) if name is None else (float(v) for v in row["dof"][name])
            positions[joint_id], velocities[joint_id] = position, velocity
        contact = row["phase"] == "drive"
        trace.append({
            "step_index": index, "joint_positions": positions, "joint_velocities_per_s": velocities,
            "task_contact_active": contact,
            "joint_limit_violation": any(
                not limits[joint_id][0] - JOINT_HARD_LIMIT_SOLVER_RESIDUAL_RAD <= value
                <= limits[joint_id][1] + JOINT_HARD_LIMIT_SOLVER_RESIDUAL_RAD
                for joint_id, value in positions.items()),
            # Nothing but the joint driver ever touches the asset, and it lets go.
            "containment_violation": False, "robot_collision_failure": False,
            "scene_collision_failure": False, "retreat_completed": not contact,
        })
    return trace


def _check(spec: Mapping[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        report = score_articulated_task_episode(task_spec=spec, samples=samples)
    except TaskNeutralScoringError as exc:
        return {"status": "unscorable", "task_succeeded": None, "errors": list(exc.args[0])[:20],
                "sample_count": len(samples)}
    return {"status": report["status"], "task_succeeded": report["task_succeeded"],
            "outcome": report["outcome"], "sample_count": len(samples)}


def score_robot_free_readiness(*, task_spec: Mapping[str, Any], task_joint_bindings: Sequence[Mapping[str, Any]],
                               native_import_result: Mapping[str, Any],
                               configured_scene_revision_digest: str | None = None) -> dict[str, Any]:
    """Return a sealed receipt; only both checks passing admits learned-policy evaluation."""

    spec = dict(task_spec)
    normalized = validate_articulated_task_spec(spec)
    probe = native_import_result.get("robot_free_success_probe")
    base = {"schema_version": SCHEMA, "configured_scene_revision_digest": configured_scene_revision_digest,
            "native_import_result_digest": native_import_result.get("result_digest"),
            "task_spec_digest": canonical_digest(spec)}
    boundary = {"robot_embodiment_used": False, "scene_context": "replacement_asset_on_ground_plane",
                "scene_collisions_modeled": False, "containment_modeled": False,
                "joint_limit_violation_source": "readback_against_qualified_limits",
                "simulator_evidence_is_physical_truth": False}
    if (not isinstance(probe, Mapping) or probe.get("schema_version") != ROBOT_FREE_PROBE_SCHEMA
            or probe.get("status") != "recorded"):
        reason = probe.get("error") if isinstance(probe, Mapping) else None
        receipt = {**base, "status": "probe_unavailable", "probe_error": reason, "proof_boundary": boundary,
                   "learned_policy_evaluation_admitted": False, "receipt_digest": ""}
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        return receipt
    control_hz = int(spec["control_frequency_hz"])
    decimation = round(NATIVE_PHYSICS_FREQUENCY_HZ / control_hz)
    if abs(round(1.0 / float(probe["timestep_seconds"])) - NATIVE_PHYSICS_FREQUENCY_HZ) > 0:
        raise ValueError("robot_free_probe_physics_rate_mismatch")
    native_names = {str(row["joint_id"]): row.get("native_joint_name") for row in task_joint_bindings}
    samples = list(probe["samples"])
    untouched = [row for row in samples if row["phase"] == "zero_action"]
    driven = [row for row in samples if row["phase"] in {"drive", "release"}]
    # One reading per control step, as a robot team's episode is sampled; the
    # driven episode starts from the untouched scene's final reset reading.
    zero_action = _check(spec, _scorer_trace(untouched[decimation - 1::decimation], spec=normalized,
                                             native_names=native_names))
    direct_drive = _check(spec, _scorer_trace(untouched[-1:] + driven[decimation - 1::decimation],
                                              spec=normalized, native_names=native_names))
    checks = {"zero_action_not_success": zero_action["task_succeeded"] is False and zero_action["status"] == "scored",
              "direct_drive_success": direct_drive["task_succeeded"] is True}
    passed = all(checks.values())
    receipt = {**base, "status": "passed" if passed else "failed",
               "probe": {"physics_hz": NATIVE_PHYSICS_FREQUENCY_HZ, "control_frequency_hz": control_hz,
                         "plan": dict(probe["plan"])},
               "zero_action": zero_action, "direct_drive": direct_drive, "checks": checks,
               "proof_boundary": boundary, "learned_policy_evaluation_admitted": passed, "receipt_digest": ""}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def robot_free_readiness_for_revision(*, revision: Mapping[str, Any], output_root: str | Path,
                                      fetcher: Callable[[Mapping[str, Any]], bytes] | None = None) -> dict[str, Any]:
    """Fetch the configured revision's exact documents, adapt its task as an evaluation would, and score."""

    from . import task_evaluation_configured_controls_deferred_inputs as inputs
    from .task_evaluation_articulated_open_close_native_adapter import adapt_articulated_open_close_task_template

    root = Path(output_root) / "documents"
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    documents: dict[str, Path] = {}
    for contract_path, declared in inputs._document_references(revision).items():
        reference = inputs._reference(declared, blocker=f"robot_free_readiness_reference_invalid:{contract_path}")
        path = root / inputs._document_name(contract_path)
        if not inputs._retained_matches(path, reference=reference):
            inputs._write_immutable_bytes(path, inputs._fetched(reference, fetcher=fetcher or inputs.default_reference_fetcher),
                                          conflict="robot_free_readiness_document_conflict")
        documents[contract_path] = path
    references = inputs._materialized_references(revision=revision, documents=documents)
    definition = adapt_articulated_open_close_task_template(
        configured_revision=revision, materialized_references=references)["native_task_definition"]
    return score_robot_free_readiness(
        task_spec=definition["task_spec"], task_joint_bindings=definition["task_joint_bindings"],
        native_import_result=json.loads(documents[NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH].read_text()),
        configured_scene_revision_digest=revision.get("revision_digest"))


def main(argv: list[str] | None = None) -> int:
    """Score a published configured revision's robot-free probe; writes the sealed receipt."""

    import argparse

    from .task_evaluation_configured_scene_revision import validate_configured_scene_revision

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--revision", required=True, help="configured_scene_revision.v1.json")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    revision = validate_configured_scene_revision(json.loads(Path(args.revision).read_text()))
    receipt = robot_free_readiness_for_revision(revision=revision, output_root=args.output_root)
    path = Path(args.output_root) / "robot_free_readiness.v1.json"
    text = json.dumps(receipt, sort_keys=True, indent=1) + "\n"
    if path.exists() and path.read_text() != text:
        raise SystemExit("robot_free_readiness_receipt_conflict")
    path.write_text(text)
    print(json.dumps({key: receipt.get(key) for key in ("status", "checks", "learned_policy_evaluation_admitted",
                                                       "receipt_digest")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "main", "robot_free_readiness_for_revision", "score_robot_free_readiness"]
