"""Robot-free readiness scores the native probe with the exact configured success contract."""
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_native_import_driver import ROBOT_FREE_PROBE_SCHEMA
from blueprint_pipeline.task_evaluation_scene_robot_free_readiness import score_robot_free_readiness

_DRIVE = {"drive_type": "none", "stiffness": 0.0, "damping": 0.0, "maximum_force": 0.0}


def _spec() -> dict:
    graph = {
        "schema_version": "adp_articulation_graph.v1",
        "links": [{"link_id": name, "is_root": name == "tub", "semantic_role": name} for name in ("tub", "door", "rack")],
        "joints": [
            {"joint_id": "door_hinge", "role": "target", "joint_type": "revolute", "parent_link_id": "tub",
             "child_link_id": "door", "axis": [0.0, 1.0, 0.0], "limits": [0.0, 1.57], "reset_position": 0.0,
             "reset_tolerance": 0.01, "drive": dict(_DRIVE)},
            {"joint_id": "rack_fixed", "role": "locked", "joint_type": "fixed", "parent_link_id": "tub",
             "child_link_id": "rack", "axis": [0.0, 0.0, 0.0], "limits": [0.0, 0.0], "reset_position": 0.0,
             "reset_tolerance": 0.005, "drive": dict(_DRIVE)},
        ],
        "collision_pairs": [],
        "success_predicate": {"combination": "all", "joint_intervals": {"door_hinge": [1.2, 1.57]}},
    }
    return {"schema_version": "adp_task_spec.v2", "task_kind": "articulated_open_close", "articulation_graph": graph,
            "settle_window_samples": 30, "maximum_settled_target_speed": 0.05, "locked_joint_motion_tolerance": 0.01,
            "movement_epsilon": 0.0157, "control_frequency_hz": 30}


_BINDINGS = [{"joint_id": "door_hinge", "native_joint_name": "door_hinge", "readback_kind": "native_coordinate"},
             {"joint_id": "rack_fixed", "readback_kind": "fixed_joint_static"}]


def _probe(*, rests_at: float) -> dict:
    rows = [{"step_index": i, "phase": "zero_action", "dof": {"door_hinge": [0.0, 0.0]}} for i in range(360)]
    for k in range(120):
        rows.append({"step_index": 360 + k, "phase": "drive", "dof": {"door_hinge": [1.385 * (k + 1) / 120, 0.0]}})
    for k in range(600):
        rows.append({"step_index": 480 + k, "phase": "release", "dof": {"door_hinge": [rests_at, 0.0]}})
    return {"schema_version": ROBOT_FREE_PROBE_SCHEMA, "status": "recorded", "timestep_seconds": 1.0 / 120.0,
            "plan": {"target_joint_id": "door_hinge", "drive_target_position": 1.385}, "samples": rows}


def _result(probe) -> dict:
    value = {"schema_version": "task_evaluation_replacement_native_import_result.v1", "status": "qualified",
             **({"robot_free_success_probe": probe} if probe is not None else {}), "result_digest": ""}
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


def test_door_left_alone_fails_and_driven_open_succeeds_so_the_scene_is_ready() -> None:
    receipt = score_robot_free_readiness(task_spec=_spec(), task_joint_bindings=_BINDINGS,
                                         native_import_result=_result(_probe(rests_at=1.57)))
    assert receipt["status"] == "passed" and receipt["learned_policy_evaluation_admitted"] is True
    assert receipt["checks"] == {"zero_action_not_success": True, "direct_drive_success": True}
    assert receipt["zero_action"]["task_succeeded"] is False and receipt["zero_action"]["status"] == "scored"
    # 3 s untouched, then 5 s driven-and-released, each read once per 30 Hz control step.
    assert (receipt["zero_action"]["sample_count"], receipt["direct_drive"]["sample_count"]) == (90, 181)
    assert receipt["proof_boundary"]["robot_embodiment_used"] is False
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")


def test_a_door_that_swings_shut_after_release_is_not_ready() -> None:
    receipt = score_robot_free_readiness(task_spec=_spec(), task_joint_bindings=_BINDINGS,
                                         native_import_result=_result(_probe(rests_at=0.0)))
    assert receipt["status"] == "failed" and receipt["learned_policy_evaluation_admitted"] is False
    assert receipt["checks"]["direct_drive_success"] is False
    assert receipt["direct_drive"]["outcome"] != "opened_and_settled"


def test_a_door_that_opens_by_itself_is_not_ready() -> None:
    probe = _probe(rests_at=1.57)
    for row in probe["samples"]:
        if row["phase"] == "zero_action" and row["step_index"] >= 60:
            row["dof"]["door_hinge"] = [1.57, 0.0]
    receipt = score_robot_free_readiness(task_spec=_spec(), task_joint_bindings=_BINDINGS,
                                         native_import_result=_result(probe))
    assert receipt["checks"]["zero_action_not_success"] is False and receipt["status"] == "failed"


@pytest.mark.parametrize("probe", [None, {"schema_version": ROBOT_FREE_PROBE_SCHEMA, "status": "failed",
                                          "error": "RuntimeError:isaac_handle_lost"}])
def test_without_a_recorded_probe_nothing_is_admitted(probe) -> None:
    receipt = score_robot_free_readiness(task_spec=_spec(), task_joint_bindings=_BINDINGS,
                                         native_import_result=_result(probe))
    assert receipt["status"] == "probe_unavailable" and receipt["learned_policy_evaluation_admitted"] is False


def test_the_real_adapted_drawer_contract_scores_its_recorded_probe(tmp_path) -> None:
    """End to end on CPU: the configured revision's own task spec scores the probe."""
    from blueprint_pipeline.task_evaluation_articulated_open_close_native_adapter import (
        adapt_articulated_open_close_task_template,
    )
    from tests.test_task_evaluation_articulated_open_close_native_adapter import _case

    case = _case(tmp_path)
    definition = adapt_articulated_open_close_task_template(
        configured_revision=case["configured"], materialized_references=case["references"],
    )["native_task_definition"]
    spec, bindings = definition["task_spec"], definition["task_joint_bindings"]
    graph = spec["articulation_graph"]
    target = next(row for row in graph["joints"] if row["role"] == "target")
    interval = graph["success_predicate"]["joint_intervals"][target["joint_id"]]
    goal = (interval[0] + interval[1]) / 2
    dofs = {row["joint_id"]: row["native_joint_name"] for row in bindings if row.get("native_joint_name")}

    def row(step, phase, target_position):
        values = {name: [float(next(j["reset_position"] for j in graph["joints"] if j["joint_id"] == joint_id)), 0.0]
                  for joint_id, name in dofs.items()}
        values[dofs[target["joint_id"]]] = [target_position, 0.0]
        return {"step_index": step, "phase": phase, "dof": values}

    samples = [row(i, "zero_action", float(target["reset_position"])) for i in range(360)]
    samples += [row(360 + k, "drive", goal * (k + 1) / 120) for k in range(120)]
    samples += [row(480 + k, "release", goal) for k in range(600)]
    probe = {"schema_version": ROBOT_FREE_PROBE_SCHEMA, "status": "recorded", "timestep_seconds": 1.0 / 120.0,
             "plan": {"target_joint_id": target["joint_id"], "drive_target_position": goal}, "samples": samples}
    receipt = score_robot_free_readiness(task_spec=spec, task_joint_bindings=bindings,
                                         native_import_result=_result(probe))
    assert receipt["checks"] == {"zero_action_not_success": True, "direct_drive_success": True}, receipt
    assert receipt["probe"]["control_frequency_hz"] == spec["control_frequency_hz"]


def test_a_published_revision_is_fetched_adapted_and_scored_by_its_digests(tmp_path) -> None:
    import copy
    import hashlib
    import json
    from pathlib import Path

    from blueprint_pipeline.task_evaluation_articulated_open_close_native_adapter import (
        adapt_articulated_open_close_task_template,
    )
    from blueprint_pipeline.task_evaluation_scene_robot_free_readiness import robot_free_readiness_for_revision
    from tests.test_task_evaluation_articulated_open_close_native_adapter import NATIVE_IMPORT, _case

    case = _case(tmp_path)
    graph = adapt_articulated_open_close_task_template(
        configured_revision=case["configured"], materialized_references=case["references"],
    )["native_task_definition"]["task_spec"]["articulation_graph"]
    target = next(row for row in graph["joints"] if row["role"] == "target")
    goal = sum(graph["success_predicate"]["joint_intervals"][target["joint_id"]]) / 2
    rows = [{"step_index": i, "phase": "zero_action", "dof": {target["joint_id"]: [0.0, 0.0]}} for i in range(360)]
    rows += [{"step_index": 360 + k, "phase": "drive", "dof": {target["joint_id"]: [goal * (k + 1) / 120, 0.0]}}
             for k in range(120)]
    rows += [{"step_index": 480 + k, "phase": "release", "dof": {target["joint_id"]: [goal, 0.0]}} for k in range(600)]
    native = {key: value for key, value in case["documents"][NATIVE_IMPORT].items()}
    native["robot_free_success_probe"] = {"schema_version": ROBOT_FREE_PROBE_SCHEMA, "status": "recorded",
                                          "timestep_seconds": 1.0 / 120.0, "plan": {"drive_target_position": goal},
                                          "samples": rows}
    native["result_digest"] = canonical_digest(native, digest_field="result_digest")
    payload = (json.dumps(native, sort_keys=True) + "\n").encode()
    uri = "s3://blueprint-production-inputs/native-with-probe.json"
    revision = copy.deepcopy(case["configured"])
    revision["replacement"]["native_import_qualification"] = {
        "uri": uri, "digest": "sha256:" + hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    revision["revision_digest"] = ""
    revision["revision_digest"] = canonical_digest(revision, digest_field="revision_digest")
    blobs = {row["uri"]: Path(row["materialized_path"]).read_bytes() for row in case["references"].values()}
    blobs[uri] = payload
    receipt = robot_free_readiness_for_revision(revision=revision, output_root=tmp_path / "readiness",
                                                fetcher=lambda reference: blobs[reference["uri"]])
    assert receipt["status"] == "passed" and receipt["configured_scene_revision_digest"] == revision["revision_digest"]
    assert receipt["native_import_result_digest"] == native["result_digest"]
    # A byte that does not match its published digest is refused before any scoring.
    blobs[uri] = payload + b" "
    with pytest.raises(Exception, match="fetch_mismatch"):
        robot_free_readiness_for_revision(revision=revision, output_root=tmp_path / "tampered",
                                          fetcher=lambda reference: blobs[reference["uri"]])
