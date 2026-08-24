from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_controls_worker import (
    _RigidScoringEnvironment,
    _canonical_digest,
    _contact_close_sweep_minimum_force_n,
    _construction_global_ik_joint_targets,
    _control_plan_global_ik_joint_targets,
    _input_binding_mismatches,
    _load_and_verify_manifest,
    _normalized_control_plan_for_execution,
    _parallel_jaw_equivalent_control_plan,
    _parallel_jaw_equivalent_quaternion_xyzw,
    _select_parallel_jaw_control_plan,
    _verified_runtime_inputs,
)


def test_close_sweep_uses_task_contact_force_before_plan_row_is_compiled() -> None:
    assert _contact_close_sweep_minimum_force_n(
        contact_close_row={"bilateral_task_contact_minimum_force_n": None},
        task_state_binding={"task_contact_minimum_force_n": 0.5},
    ) == pytest.approx(0.5)


def test_close_sweep_refuses_a_contact_force_gate_mismatch() -> None:
    with pytest.raises(
        RuntimeError, match="native_task_controls_contact_force_mismatch"
    ):
        _contact_close_sweep_minimum_force_n(
            contact_close_row={"bilateral_task_contact_minimum_force_n": 0.4},
            task_state_binding={"task_contact_minimum_force_n": 0.5},
        )


def test_controls_multistart_solves_missing_exact_pose_and_reuses_duplicates() -> None:
    class _Servo:
        def __init__(self):
            self.calls = []

        def read_arm_joint_positions(self):
            return [0.0] * 7

        def solve_grasp_target_multistart(self, **kwargs):
            self.calls.append(kwargs)
            joints = [float(kwargs["target_position_world_m"][0])] * 7
            return {
                "solved": True,
                "selected": {
                    "solved": True,
                    "joint_positions_rad": joints,
                },
                "attempts": [],
            }

    servo = _Servo()
    bound = [
        {
            "phase_id": "approach",
            "target_position_world_m": [1.0, 0.0, 0.0],
            "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "joint_positions_rad": [0.1] * 7,
        }
    ]
    targets, receipt = _control_plan_global_ik_joint_targets(
        servo=servo,
        control_plan={
            "scripted_positive_actions": [
                {
                    "phase_id": "prealign",
                    "mode": "ik_pose",
                    "position_only_arrival": True,
                    "target_position_world_m": [0.5, 0.0, 0.0],
                    "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "arrival_tolerance_m": 0.02,
                    "arrival_orientation_tolerance_rad": None,
                },
                {
                    "phase_id": "approach",
                    "mode": "ik_pose",
                    "target_position_world_m": [1.0, 0.0, 0.0],
                    "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                {
                    "phase_id": "contact_open",
                    "mode": "ik_pose",
                    "target_position_world_m": [2.0, 0.0, 0.0],
                    "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "arrival_tolerance_m": 0.005,
                    "arrival_orientation_tolerance_rad": 0.08,
                },
                {
                    "phase_id": "contact_close",
                    "mode": "ik_pose",
                    "target_position_world_m": [2.0, 0.0, 0.0],
                    "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            ]
        },
        bound_targets=bound,
        reference_seeds=[[0.5] * 7],
    )

    assert len(servo.calls) == 2
    assert servo.calls[0]["preferred_seeds"][0] == [0.0] * 7
    assert servo.calls[0]["position_tolerance_m"] == pytest.approx(0.02)
    assert servo.calls[0]["orientation_tolerance_rad"] == pytest.approx(0.08)
    assert servo.calls[1]["preferred_seeds"][0] == [0.1] * 7
    assert servo.calls[1]["position_tolerance_m"] == pytest.approx(0.005)
    assert servo.calls[1]["orientation_tolerance_rad"] == pytest.approx(0.08)
    assert servo.calls[1][
        "required_minimum_joint_limit_margin_rad"
    ] == pytest.approx(0.005)
    assert [row["phase_id"] for row in targets] == [
        "approach",
        "prealign",
        "contact_open",
    ]
    assert targets[1]["joint_positions_rad"] == [0.5] * 7
    assert targets[-1]["joint_positions_rad"] == [2.0] * 7
    assert receipt["status"] == "all_unique_poses_solved_or_bound"
    assert [row.get("status") for row in receipt["phases"]] == [
        None,
        "reused_bound_pose_solution",
        None,
        "reused_bound_pose_solution",
    ]
    assert receipt["phases"][0]["position_only_arrival_gate"] is True
    assert receipt["phases"][0][
        "full_pose_prepositioning_tolerance_rad"
    ] == pytest.approx(0.08)


def test_controls_approach_uses_path_margin_before_exact_contact() -> None:
    class _Servo:
        def __init__(self):
            self.calls = []

        def read_arm_joint_positions(self):
            return [0.0] * 7

        def solve_grasp_target_multistart(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "solved": True,
                "selected": {
                    "solved": True,
                    "joint_positions_rad": [0.1] * 7,
                },
                "attempts": [],
            }

    servo = _Servo()
    _control_plan_global_ik_joint_targets(
        servo=servo,
        control_plan={
            "scripted_positive_actions": [
                {
                    "phase_id": "approach",
                    "mode": "ik_pose",
                    "target_position_world_m": [1.0, 0.0, 0.0],
                    "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "arrival_tolerance_m": 0.02,
                    "arrival_orientation_tolerance_rad": 0.08,
                },
                {
                    "phase_id": "contact_open",
                    "mode": "ik_pose",
                    "target_position_world_m": [2.0, 0.0, 0.0],
                    "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "arrival_tolerance_m": 0.005,
                    "arrival_orientation_tolerance_rad": 0.08,
                },
            ]
        },
        bound_targets=[],
        reference_seeds=[],
    )

    assert servo.calls[0][
        "preferred_minimum_joint_limit_margin_rad"
    ] == pytest.approx(0.04)
    assert servo.calls[0][
        "required_minimum_joint_limit_margin_rad"
    ] == pytest.approx(0.0)
    assert servo.calls[1][
        "preferred_minimum_joint_limit_margin_rad"
    ] == pytest.approx(0.05)
    assert servo.calls[1][
        "required_minimum_joint_limit_margin_rad"
    ] == pytest.approx(0.005)


def test_parallel_jaw_equivalent_preserves_approach_and_reverses_jaw() -> None:
    nominal = [-2**-0.5, 0.0, 0.0, 2**-0.5]

    equivalent = _parallel_jaw_equivalent_quaternion_xyzw(nominal)
    plan = {
        "schema_version": "adp_task_control_plan.v1",
        "scripted_positive_actions": [
            {
                "phase_id": "contact_open",
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 2.0, 3.0],
                "target_quaternion_world_xyzw": nominal,
            }
        ],
        "plan_digest": "sha256:" + "a" * 64,
    }
    derived = _parallel_jaw_equivalent_control_plan(plan)
    evidence = derived["runtime_control_variant_equivalence"]

    assert equivalent == pytest.approx([0.0, 2**-0.5, 2**-0.5, 0.0])
    assert evidence["approach_axis_dot"] == pytest.approx(1.0)
    assert evidence["jaw_axis_dot"] == pytest.approx(-1.0)
    assert derived["scripted_positive_actions"][0][
        "target_position_world_m"
    ] == [1.0, 2.0, 3.0]
    assert derived["plan_digest"] == _canonical_digest(
        derived, field="plan_digest"
    )


def test_controls_selects_fully_solved_parallel_jaw_branch_with_more_margin() -> None:
    class _Servo:
        def read_arm_joint_positions(self):
            return [0.0] * 7

        def solve_grasp_target_multistart(self, **kwargs):
            quaternion = kwargs["target_grasp_frame_quaternion_world_xyzw"]
            equivalent = quaternion[1] > 0.5
            margin = 0.2 if equivalent else 0.002
            return {
                "solved": True,
                "selected": {
                    "solved": True,
                    "joint_positions_rad": [margin] * 7,
                    "minimum_joint_limit_margin_rad": margin,
                },
                "attempts": [],
            }

    nominal = [-2**-0.5, 0.0, 0.0, 2**-0.5]
    plan = {
        "schema_version": "adp_task_control_plan.v1",
        "scripted_positive_actions": [
            {
                "phase_id": "contact_open",
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 2.0, 3.0],
                "target_quaternion_world_xyzw": nominal,
                "arrival_tolerance_m": 0.005,
                "arrival_orientation_tolerance_rad": 0.08,
            },
            {
                "phase_id": "contact_close",
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 2.0, 3.0],
                "target_quaternion_world_xyzw": nominal,
                "arrival_tolerance_m": 0.005,
                "arrival_orientation_tolerance_rad": 0.08,
            },
        ],
        "plan_digest": "sha256:" + "a" * 64,
    }

    selected_plan, targets, receipt = _select_parallel_jaw_control_plan(
        servo=_Servo(),
        control_plan=plan,
        construction_bound_targets=[],
        reference_seeds=[[0.5] * 7],
    )

    assert receipt["selected_variant_id"] == "parallel_jaw_equivalent"
    assert receipt[
        "selected_contact_open_minimum_joint_limit_margin_rad"
    ] == pytest.approx(0.2)
    assert selected_plan["runtime_control_variant"].startswith("parallel_jaw")
    assert targets[0]["joint_positions_rad"] == pytest.approx([0.2] * 7)
    assert receipt["physics_steps_performed"] == 0


def test_controls_bind_construction_global_ik_branch_to_same_pose_phase() -> None:
    rows = _construction_global_ik_joint_targets(
        construction={
            "pink_global_ik_preflight": {
                "schema_version": "native_task_pink_global_ik_preflight.v1",
                "phases": [
                    {"phase_id": "prealign", "selected": None},
                    {
                        "phase_id": "approach",
                        "selected": {
                            "solved": True,
                            "joint_positions_rad": [0.1] * 7,
                        },
                    },
                ],
            },
            "phase_results": [
                {
                    "phase_id": "prealign",
                    "target_position_world_m": [1.0, 2.0, 2.5],
                    "target_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                {
                    "phase_id": "approach",
                    "target_position_world_m": [1.0, 2.0, 3.0],
                    "target_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            ],
        },
        control_plan={
            "scripted_positive_actions": [
                {
                    "phase_id": "prealign",
                    "mode": "ik_pose",
                    "target_position_world_m": [1.0, 2.0, 2.5],
                    "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                {
                    "phase_id": "approach",
                    "mode": "ik_pose",
                    "target_position_world_m": [1.0, 2.0, 3.0],
                    "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            ]
        },
    )

    assert rows == [
        {
            "phase_id": "approach",
            "target_position_world_m": [1.0, 2.0, 3.0],
            "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "joint_positions_rad": [0.1] * 7,
        }
    ]


def test_controls_reject_global_ik_phase_with_different_control_pose() -> None:
    with pytest.raises(
        RuntimeError,
        match="native_task_controls_global_ik_binding_invalid",
    ):
        _construction_global_ik_joint_targets(
            construction={
                "pink_global_ik_preflight": {
                    "schema_version": "native_task_pink_global_ik_preflight.v1",
                    "phases": [
                        {
                            "phase_id": "approach",
                            "selected": {
                                "solved": True,
                                "joint_positions_rad": [0.1] * 7,
                            },
                        }
                    ],
                },
                "phase_results": [
                    {
                        "phase_id": "approach",
                        "target_position_world_m": [1.0, 2.0, 3.0],
                        "target_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                ],
            },
            control_plan={
                "scripted_positive_actions": [
                    {
                        "phase_id": "approach",
                        "mode": "ik_pose",
                        "target_position_world_m": [1.0, 2.0, 3.1],
                        "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                ]
            },
        )


def test_controls_worker_source_has_no_scene_task_or_policy_identity() -> None:
    source = Path(
        __import__(
            "blueprint_pipeline.native_task_arena_controls_worker",
            fromlist=["x"],
        ).__file__
    ).read_text(encoding="utf-8")

    for forbidden in (
        "840313",
        "840796",
        "refrigerator",
        "approved_can",
        "pi05_droid",
        "groot_n17_droid",
    ):
        assert forbidden not in source


def test_controls_manifest_rejects_policy_or_construction_mode(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "native_task_arena_provider_bundle.v1",
        "execution_mode": "controls",
        "policy_candidate_id": None,
        "candidate_policy_queried": False,
        "input_digest": "",
    }
    manifest["input_digest"] = canonical_digest(
        manifest, digest_field="input_digest"
    )
    path = tmp_path / "adp_arena_provider_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _load_and_verify_manifest(tmp_path)["execution_mode"] == "controls"

    for mode in ("construction_canary", "policy"):
        manifest["execution_mode"] = mode
        manifest["input_digest"] = canonical_digest(
            manifest, digest_field="input_digest"
        )
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(RuntimeError, match="native_task_controls_manifest_invalid"):
            _load_and_verify_manifest(tmp_path)


def test_controls_runtime_inputs_reverify_every_byte(tmp_path: Path) -> None:
    inputs = tmp_path / "runtime_inputs"
    inputs.mkdir()
    rows = []
    for name in (
        "native_task_arena_construction_result.v1.json",
        "adp_task_control_plan.v1.json",
    ):
        path = inputs / name
        path.write_text("{}\n", encoding="utf-8")
        rows.append(
            {
                "relative_path": f"runtime_inputs/{name}",
                "size_bytes": path.stat().st_size,
                "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    verified = _verified_runtime_inputs(
        tmp_path, {"bound_runtime_inputs": rows}
    )
    assert set(verified) == {
        "native_task_arena_construction_result.v1.json",
        "adp_task_control_plan.v1.json",
    }

    (inputs / "adp_task_control_plan.v1.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity_mismatch"):
        _verified_runtime_inputs(tmp_path, {"bound_runtime_inputs": rows})


class _BaseRigidEnvironment:
    def reset(self) -> None:
        return None

    def read_object_sample(self) -> dict:
        return {
            "task_object_pose_world": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "gripper_width_m": 0.071,
            "grasp_frame_position_world_m": [1.0, 2.0, 3.0],
        }


class _ExactRigidReadback:
    def read_task_sample(self) -> dict:
        return {
            "asset_root_pose_world": [1.0, 2.0, 0.7, 0.0, 0.0, 0.0, 1.0],
            "task_scoring_pose_world": [1.02, 1.99, 0.73, 0.0, 0.0, 0.0, 1.0],
            "task_robot_contact_peak_force_n": 0.75,
            "task_support_contact_peak_force_n": 4.0,
            "task_scene_collision_peak_force_n": 0.2,
            "robot_scene_contact_peak_force_n": 0.1,
            "robot_task_forbidden_collision_peak_force_n": 0.0,
            "locked_joint_containment_violation": False,
        }


def _graph_rigid_task_spec() -> dict:
    return {
        "task_contact_minimum_force_n": 0.5,
        "collision_failure_minimum_force_n": 1.0,
        "workspace_position_bounds_world_m": {
            "minimum": [0.0, 0.0, 0.0],
            "maximum": [2.0, 3.0, 2.0],
        },
    }


def test_rigid_controls_environment_uses_scoring_frame_and_exact_contacts() -> None:
    environment = _RigidScoringEnvironment(
        environment=_BaseRigidEnvironment(),
        task_readback=_ExactRigidReadback(),
        task_spec=_graph_rigid_task_spec(),
    )

    sample = environment.read_object_sample()

    assert sample["task_object_pose_world"] == [
        1.02,
        1.99,
        0.73,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert sample["asset_root_pose_world"] != sample["task_object_pose_world"]
    assert sample["gripper_width_m"] == pytest.approx(0.071)
    assert sample["task_contact_active"] is True
    assert sample["support_contact_active"] is True
    assert sample["robot_collision_failure"] is False
    assert sample["scene_collision_failure"] is False
    assert sample["forbidden_robot_task_collision_failure"] is False
    assert sample["locked_joint_containment_violation"] is False
    assert sample["containment_violation"] is False
    environment.reset()


def test_rigid_controls_environment_fails_closed_on_missing_native_channel() -> None:
    readback = _ExactRigidReadback()
    readback.read_task_sample = lambda: {"task_scoring_pose_world": [0.0] * 7}
    environment = _RigidScoringEnvironment(
        environment=_BaseRigidEnvironment(),
        task_readback=readback,
        task_spec=_graph_rigid_task_spec(),
    )

    with pytest.raises(RuntimeError, match="rigid_sample_invalid"):
        environment.read_object_sample()


def _bundled_controls_inputs(tmp_path: Path, task_kind: str) -> dict[str, dict]:
    """Read back exactly what the worker reads on the provider, from real bytes.

    Nothing here is hand-written: the packet, the construction receipt, the
    control plan and the manifest all come from their real producers, are frozen
    into a real bundle, and are then read out of that bundle the way the worker
    reads them on the GPU.
    """

    import zipfile

    from tests.test_native_task_arena_bundle import (
        _packet,
        _runtime_source_packet,
        _sha,
        _articulated_packet,
        _qualified_construction,
    )
    from blueprint_pipeline.native_task_arena_controls_bundle import (
        build_native_task_arena_controls_bundle,
    )

    if task_kind == "articulated_open_close":
        packet, scene = _articulated_packet(tmp_path)
        construction_path = _qualified_construction(tmp_path, scene)
    else:
        from tests.test_native_task_control_plan import (
            _rigid_construction,
            _rigid_scene,
        )

        scene = _rigid_scene(scene_id="840313", asset_id="fixture_asset")
        packet = _packet(tmp_path, scene_id="840313")
        plan_path = packet / "native_task_arena_scene_plan.v1.json"
        plan_path.write_text(
            json.dumps(scene, sort_keys=True) + "\n", encoding="utf-8"
        )
        receipt_path = packet / "native_task_arena_packet_receipt.v1.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["arena_scene_plan_digest"] = scene["plan_digest"]
        artifact = next(
            row for row in receipt["artifacts"] if row["role"] == "arena_scene_plan"
        )
        artifact["size_bytes"] = plan_path.stat().st_size
        artifact["sha256"] = _sha(plan_path)
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        construction_path = tmp_path / "native_task_arena_construction_result.v1.json"
        construction_path.write_text(
            json.dumps(_rigid_construction(scene), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    bundle = build_native_task_arena_controls_bundle(
        job_dir=tmp_path / "controls-bundle",
        packet_dir=packet,
        construction_result_path=construction_path,
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="c" * 40,
        generated_at="fixed",
    )
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        archive.extractall(extracted)
    runtime = extracted / "provider_runtime"
    inner = runtime / "native_task_packet"
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))  # noqa: E731
    return {
        "manifest": read(runtime / "adp_arena_provider_manifest.json"),
        "packet_receipt": read(
            inner / "native_task_arena_packet_receipt.v1.json"
        ),
        "scene_plan": read(inner / "native_task_arena_scene_plan.v1.json"),
        "construction": read(
            runtime
            / "runtime_inputs/native_task_arena_construction_result.v1.json"
        ),
        "control_plan": read(
            runtime / "runtime_inputs/adp_task_control_plan.v1.json"
        ),
    }


@pytest.mark.parametrize(
    "task_kind", ["articulated_open_close", "rigid_pick_place"]
)
def test_real_producers_satisfy_every_controls_input_binding_relation(
    tmp_path: Path, task_kind: str
) -> None:
    """The gate must be satisfiable by the artifacts its own producers emit.

    A single disagreeing relation costs one full paid provider run, so this
    proves the whole chain agrees before any GPU is rented.
    """

    inputs = _bundled_controls_inputs(tmp_path, task_kind)

    assert _input_binding_mismatches(**inputs) == []
    assert inputs["scene_plan"]["task_kind"] == task_kind


def test_parallel_jaw_variant_is_accepted_by_real_control_plan_validator(
    tmp_path: Path,
) -> None:
    from blueprint_pipeline.adp009d_control_episode import (
        validate_task_control_plan,
    )

    inputs = _bundled_controls_inputs(tmp_path, "rigid_pick_place")
    normalized = _normalized_control_plan_for_execution(
        control_plan=inputs["control_plan"],
        task_spec=inputs["scene_plan"]["task_spec"],
    )
    derived = _parallel_jaw_equivalent_control_plan(normalized)

    checked = validate_task_control_plan(
        derived,
        task_spec=inputs["scene_plan"]["task_spec"],
    )

    assert checked["plan_digest"] == derived["plan_digest"]
    assert checked["plan_digest"] == _canonical_digest(
        checked, field="plan_digest"
    )
    assert checked["runtime_normalized_source_plan_digest"] == inputs[
        "control_plan"
    ]["plan_digest"]
    assert checked["runtime_control_variant"].startswith("parallel_jaw")
    assert checked["scripted_positive_actions"][0][
        "target_quaternion_world_xyzw"
    ] == pytest.approx(
        derived["scripted_positive_actions"][0][
            "target_quaternion_world_xyzw"
        ]
    )


def test_each_controls_binding_relation_reports_which_one_failed(
    tmp_path: Path,
) -> None:
    """One opaque blocker cannot be read; each relation must name itself."""

    inputs = _bundled_controls_inputs(tmp_path, "rigid_pick_place")
    other = "sha256:" + "0" * 64
    breakages = {
        "packet_receipt_digest_vs_manifest": (
            "manifest",
            "packet_receipt_digest",
            other,
        ),
        "scene_plan_digest_vs_manifest": (
            "manifest",
            "arena_scene_plan_digest",
            other,
        ),
        "construction_result_digest_vs_control_plan_planner_receipt": (
            "construction",
            "result_digest",
            other,
        ),
        "control_plan_construction_scene_plan_digest_vs_scene_plan": (
            "control_plan",
            "construction_scene_plan_digest",
            other,
        ),
        "control_plan_construction_clearance_plan_digest_vs_construction": (
            "control_plan",
            "construction_clearance_plan_digest",
            other,
        ),
        "control_plan_task_kind_vs_scene_plan_task_kind": (
            "control_plan",
            "task_kind",
            "articulated_open_close",
        ),
    }
    for relation, (artifact, field, value) in breakages.items():
        broken = {key: dict(item) for key, item in inputs.items()}
        broken[artifact][field] = value
        mismatches = _input_binding_mismatches(**broken)
        assert relation in mismatches, relation
        # Editing a control-plan field also breaks its self digest; nothing else
        # may be dragged in.
        expected = {relation}
        if artifact == "control_plan":
            expected.add("control_plan_plan_digest_vs_recomputed_canonical_digest")
        assert set(mismatches) == expected, relation

    tampered = {key: dict(item) for key, item in inputs.items()}
    tampered["control_plan"]["plan_digest"] = other
    assert _input_binding_mismatches(**tampered) == [
        "control_plan_plan_digest_vs_recomputed_canonical_digest"
    ]


def test_controls_binding_refuses_two_absent_digests(tmp_path: Path) -> None:
    """Two missing digests are two refusals, never one agreement.

    Comparing absent fields with `!=` alone admitted an unbound cell: every
    digest relation held vacuously as `None == None`, and only the control
    plan's self digest objected -- which a plan carrying nothing but its own
    digest satisfies.
    """

    empty_plan: dict = {}
    empty_plan["plan_digest"] = _canonical_digest(empty_plan, field="plan_digest")

    mismatches = _input_binding_mismatches(
        manifest={},
        packet_receipt={},
        scene_plan={},
        construction={},
        control_plan=empty_plan,
    )

    assert set(mismatches) == {
        "packet_receipt_digest_vs_manifest",
        "scene_plan_digest_vs_manifest",
        "construction_result_digest_vs_control_plan_planner_receipt",
        "control_plan_construction_scene_plan_digest_vs_scene_plan",
        "control_plan_construction_clearance_plan_digest_vs_construction",
    }


def test_persist_survives_values_json_cannot_encode() -> None:
    """A receipt that cannot be written destroys the diagnosis of a paid run.

    `_persist` is called from a `finally`. The digest is computed *before* the
    write, so passing `default=str` to the write alone left a stray warp array
    or Path raising inside the handler -- replacing the real exception and
    leaving a paid run with no receipt at all. The policy worker fixed this;
    the controls and construction workers still carried the defect.
    """

    from tempfile import TemporaryDirectory

    from blueprint_pipeline.native_task_arena_controls_worker import _persist

    class _Unencodable:
        def __repr__(self) -> str:
            return "<warp array>"

    with TemporaryDirectory() as directory:
        target = Path(directory) / "native_task_arena_control_result.v1.json"
        _persist(target, {"status": "blocked", "stray": _Unencodable()})

        written = json.loads(target.read_text(encoding="utf-8"))

    assert written["status"] == "blocked"
    assert written["stray"] == "<warp array>"
    assert written["result_digest"].startswith("sha256:")


def test_persisted_controls_digest_describes_the_bytes_on_disk() -> None:
    """The digest must be recomputable from the receipt a reviewer reads."""

    from tempfile import TemporaryDirectory

    from blueprint_pipeline.native_task_arena_controls_worker import _persist

    with TemporaryDirectory() as directory:
        target = Path(directory) / "native_task_arena_control_result.v1.json"
        _persist(target, {"status": "completed", "blockers": []})
        written = json.loads(target.read_text(encoding="utf-8"))

    assert written["result_digest"] == _canonical_digest(
        written, field="result_digest"
    )


def _branch_replay_task() -> dict:
    return {
        "schema_version": "adp_task_spec.v1",
        "task_kind": "articulated_open_close",
        "task_id": "washer_door_open_test",
        "target_joint_id": "door_hinge",
        "joint_reset_positions_rad": {"door_hinge": 0.0},
        "target_success_interval_rad": [0.7, 0.95],
        "joint_hard_limits_rad": {"door_hinge": [0.0, 1.2]},
        "settle_window_samples": 3,
        "maximum_settled_target_speed_rad_s": 0.05,
        "non_task_joint_motion_tolerance_rad": 0.001,
        "movement_epsilon_rad": 0.0001,
        "reset_tolerance_rad": 0.0001,
        "maximum_action_steps": 200,
    }


def _branch_replay_plan(task: dict) -> dict:
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    def pose(phase_id, position, gripper_state, *, hold=False):
        return {
            "phase_id": phase_id,
            "mode": "ik_pose",
            "target_position_world_m": position,
            "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "gripper_state": gripper_state,
            "minimum_steps": 1,
            "maximum_steps": 12,
            "arrival_tolerance_m": 0.005,
            "arrival_stability_steps": 2,
            "arrival_orientation_tolerance_rad": 0.08,
            "max_joint_delta_rad": 0.03,
            "max_joint_setpoint_lead_rad": 0.2,
            "hold_arm_joint_positions_during_gripper_transition": hold,
        }

    contact_position = [0.5, 0.1, 0.4]
    plan = {
        "schema_version": "adp_task_control_plan.v1",
        "cell_id": "branch-replay-test",
        "task_spec_digest": canonical_digest(task),
        "trajectory_source": "native_ik_preflight",
        "planner_receipt_digest": "sha256:" + "f" * 64,
        "zero_action_steps": 3,
        "scripted_positive_actions": [
            pose("prealign", [0.4, 0.0, 0.5], "open"),
            pose("approach", [0.45, 0.05, 0.45], "open"),
            pose("contact_open", contact_position, "open"),
            pose("contact_close", contact_position, "closed", hold=True),
            pose("retreat", [0.4, 0.0, 0.5], "open"),
        ],
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def _branch_replay_targets(plan: dict) -> list[dict]:
    rows = {row["phase_id"]: row for row in plan["scripted_positive_actions"]}
    return [
        {
            "phase_id": "approach",
            "target_position_world_m": rows["approach"]["target_position_world_m"],
            "target_quaternion_world_xyzw": rows["approach"][
                "target_quaternion_world_xyzw"
            ],
            "joint_positions_rad": [0.0] * 7,
        },
        {
            "phase_id": "contact_open",
            "target_position_world_m": rows["contact_open"][
                "target_position_world_m"
            ],
            "target_quaternion_world_xyzw": rows["contact_open"][
                "target_quaternion_world_xyzw"
            ],
            "joint_positions_rad": [0.3, -0.2, 0.1, 0.4, -0.5, 0.2, 0.05],
        },
    ]


def test_contact_entry_branch_replay_inserts_bounded_joint_path() -> None:
    """C28's sealed divergence: Cartesian servoing cannot land contact entry.

    Three measured-miss-biased attempts moved 15.4 -> 28.5 -> 38.6 mm away
    with the wrist swinging 0.264 rad, while the preflight held an exact
    interior-margin solution for the same pose.  The entry must replay the
    solved same-branch joint path; the contact_open arrival gate is unchanged.
    """

    from blueprint_pipeline.adp009d_control_episode import (
        validate_task_control_plan,
    )
    from blueprint_pipeline.native_task_arena_controls_worker import (
        CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID,
        CONTACT_ENTRY_BRANCH_REPLAY_SETTLE_ROWS,
        _with_contact_entry_branch_replay,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    targets = _branch_replay_targets(plan)

    rewritten, receipt = _with_contact_entry_branch_replay(
        control_plan=plan,
        scripted_pose_joint_targets=targets,
        task_spec=task,
    )

    assert receipt["status"] == "applied"
    actions = rewritten["scripted_positive_actions"]
    replay_rows = [
        row
        for row in actions
        if row.get("phase_id") == CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID
    ]
    assert receipt["interpolation_rows"] == 10
    assert len(replay_rows) == 10 + CONTACT_ENTRY_BRANCH_REPLAY_SETTLE_ROWS
    assert all(row["gripper_state"] == "open" for row in replay_rows)
    # Every row commands the same solved posture, and carries the bounds that
    # make the servo ramp toward it at a rate the joints can follow.  The
    # executor therefore cannot outrun a lagging joint the way an open-loop
    # interpolation did.
    for row in replay_rows:
        assert row["arm_joint_positions"] == pytest.approx(
            targets[1]["joint_positions_rad"]
        )
        assert row["max_joint_delta_rad"] > 0.0
        assert row["max_joint_setpoint_lead_rad"] >= row["max_joint_delta_rad"]
    # The rows sit immediately before the unchanged contact_open gate, and
    # contact_close still directly follows contact_open.
    phase_ids = [row["phase_id"] for row in actions]
    contact_index = phase_ids.index("contact_open")
    assert phase_ids[contact_index - 1] == CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID
    assert phase_ids[contact_index + 1] == "contact_close"
    # The rewritten plan re-digests, still satisfies the fail-closed
    # validator, and the validator preserves the bounds rather than dropping
    # the rows back onto the unbounded command path.
    validated = validate_task_control_plan(rewritten, task_spec=task)
    assert validated["plan_digest"] == rewritten["plan_digest"]
    validated_replay = [
        row
        for row in validated["scripted_positive_actions"]
        if row.get("phase_id") == CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID
    ]
    assert validated_replay
    assert all("max_joint_delta_rad" in row for row in validated_replay)
    assert all("max_joint_setpoint_lead_rad" in row for row in validated_replay)


def test_contact_entry_branch_replay_fails_open_without_solutions() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_contact_entry_branch_replay,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    targets = [row for row in _branch_replay_targets(plan) if row["phase_id"] != "contact_open"]

    rewritten, receipt = _with_contact_entry_branch_replay(
        control_plan=plan,
        scripted_pose_joint_targets=targets,
        task_spec=task,
    )

    assert receipt["status"] == "not_applied"
    assert receipt["reason"] == "approach_or_contact_solution_unsolved"
    assert rewritten["plan_digest"] == plan["plan_digest"]
    assert rewritten["scripted_positive_actions"] == plan["scripted_positive_actions"]


def test_contact_entry_branch_replay_respects_task_budget() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_contact_entry_branch_replay,
    )

    task = _branch_replay_task()
    # Below even a minimal replay once the confirm-only contact budget is
    # reclaimed, so there is genuinely nowhere to put the rows.
    task["maximum_action_steps"] = 56
    plan = _branch_replay_plan(task)
    targets = _branch_replay_targets(plan)

    rewritten, receipt = _with_contact_entry_branch_replay(
        control_plan=plan,
        scripted_pose_joint_targets=targets,
        task_spec=task,
    )

    assert receipt["status"] == "not_applied"
    assert receipt["reason"] == "task_step_budget_insufficient"
    assert rewritten["scripted_positive_actions"] == plan["scripted_positive_actions"]


def test_branch_replay_row_size_follows_the_slowest_actuator() -> None:
    """Replay rows bypass the servo, so they must be sized to the hardware.

    C30's replay drove straight into saturation by its eleventh row: these
    are commanded joint targets, so nothing bounds them, and 0.05 rad per
    15 Hz step is 0.75 rad/s against a wrist that clips above 0.15.
    """

    from blueprint_pipeline.native_task_arena_controls_worker import (
        CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID,
        _with_contact_entry_branch_replay,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    targets = _branch_replay_targets(plan)
    # Shoulders may step 0.036 rad; the wrist only 0.005.
    feasible = [0.036] * 4 + [0.005] * 3

    rewritten, receipt = _with_contact_entry_branch_replay(
        control_plan=plan,
        scripted_pose_joint_targets=targets,
        task_spec=task,
        actuator_feasible_step_rad=feasible,
    )

    assert receipt["status"] == "applied"
    assert receipt["actuator_feasible_step_rad"] == pytest.approx(0.005)
    rows = [
        row
        for row in rewritten["scripted_positive_actions"]
        if row.get("phase_id") == CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID
    ]
    # The slew handed to the servo is the slowest actuator's feasible step, so
    # the ramp it produces advances no faster than that joint can follow.
    assert all(
        row["max_joint_delta_rad"] == pytest.approx(0.005) for row in rows
    )
    assert all(
        row["arm_joint_positions"] == pytest.approx(targets[1]["joint_positions_rad"])
        for row in rows
    )
    # Enough rows to cover the traverse at that rate.
    assert len(rows) >= 0.5 / 0.005
    # The Cartesian phase behind a replayed entry only confirms arrival, and
    # that reclaimed budget is what pays for the extra rows.
    assert receipt["contact_phase_steps_reclaimed"] > 0
    assert receipt["budget_limited"] is False


def test_a_budget_limited_replay_seals_that_it_was_rushed() -> None:
    """A rushed replay will clip, so the receipt must say so."""

    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_contact_entry_branch_replay,
    )

    task = _branch_replay_task()
    task["maximum_action_steps"] = 90
    plan = _branch_replay_plan(task)
    targets = _branch_replay_targets(plan)

    _rewritten, receipt = _with_contact_entry_branch_replay(
        control_plan=plan,
        scripted_pose_joint_targets=targets,
        task_spec=task,
        actuator_feasible_step_rad=[0.001] * 7,
    )

    assert receipt["status"] == "applied"
    assert receipt["budget_limited"] is True
    assert receipt["per_row_joint_step_rad"] > receipt["actuator_feasible_step_rad"]


def test_adopting_a_calibrated_posture_requires_recompiling_the_plan() -> None:
    """C37's sealed trap: the episode executes the plan, not the posture list.

    The run adopted the calibration's best posture into the posture list after
    the plan -- replay rows included -- had already been compiled, so contact
    entry replayed the model's branch and missed by 70-114 mm while the
    calibrated posture itself measured 13 mm.  Adoption is only real when the
    replay generator is re-run over the updated postures: the recompiled rows
    must end at the adopted posture and the plan digest must change with them.
    """

    from blueprint_pipeline.native_task_arena_controls_worker import (
        CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID,
        _with_contact_entry_branch_replay,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    targets = _branch_replay_targets(plan)

    compiled, first_receipt = _with_contact_entry_branch_replay(
        control_plan=plan, scripted_pose_joint_targets=targets, task_spec=task
    )
    assert first_receipt["status"] == "applied"

    adopted = [0.35, -0.25, 0.12, 0.44, -0.55, 0.22, 0.01]
    updated_targets = [
        (
            {**row, "joint_positions_rad": list(adopted)}
            if row["phase_id"] == "contact_open"
            else row
        )
        for row in targets
    ]
    recompiled, receipt = _with_contact_entry_branch_replay(
        control_plan=plan,
        scripted_pose_joint_targets=updated_targets,
        task_spec=task,
    )

    assert receipt["status"] == "applied"
    replay_rows = [
        row
        for row in recompiled["scripted_positive_actions"]
        if row.get("phase_id") == CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID
    ]
    # The recompiled entry path lands on the adopted posture, not the model's.
    assert replay_rows[-1]["arm_joint_positions"] == pytest.approx(adopted)
    stale_rows = [
        row
        for row in compiled["scripted_positive_actions"]
        if row.get("phase_id") == CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID
    ]
    assert stale_rows[-1]["arm_joint_positions"] != pytest.approx(adopted)
    # And what ran is distinguishable from what would have run.
    assert recompiled["plan_digest"] != compiled["plan_digest"]


def _roll_plan(quaternion):
    """A plan whose grasp-holding phases all share one authored orientation."""

    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    rows = []
    for phase_id in (
        "prealign", "approach", "contact_open", "contact_close",
        "joint_path_01", "joint_path_02", "joint_path_03", "joint_path_04",
        "release", "retreat",
    ):
        rows.append(
            {
                "phase_id": phase_id,
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 0.0, 0.5],
                "target_quaternion_world_xyzw": list(quaternion),
                "arrival_tolerance_m": 0.005 if "contact" in phase_id else 0.02,
                "arrival_orientation_tolerance_rad": 0.08,
                "maximum_steps": 40,
                "minimum_steps": 2,
                "arrival_stability_steps": 2,
            }
        )
    plan = {"scripted_positive_actions": rows, "plan_digest": ""}
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


class _RollServo:
    """Strict signature: a stray keyword is a failure, as in production."""

    def __init__(self, margin_for):
        self._margin_for = margin_for
        self._pink_hand_pose_at_binding_base = [0.0] * 3 + [0.0, 0.0, 0.0, 1.0]
        self._pink_grasp_pose_at_binding_base = [0.13, 0.0, 0.0] + [0.0, 0.0, 0.0, 1.0]

    def grasp_approach_axis_body(self):
        return [1.0, 0.0, 0.0]

    def read_arm_joint_positions(self):
        return [0.0] * 7

    def solve_grasp_target_multistart(
        self,
        *,
        target_position_world_m,
        target_grasp_frame_quaternion_world_xyzw,
        preferred_seeds,
        reference_joint_positions_rad,
        position_tolerance_m,
        orientation_tolerance_rad,
        preferred_minimum_joint_limit_margin_rad,
        required_minimum_joint_limit_margin_rad,
    ):
        return {
            "solved": True,
            "selected": {
                "joint_positions_rad": [0.0] * 7,
                "minimum_joint_limit_margin_rad": self._margin_for(
                    target_grasp_frame_quaternion_world_xyzw
                ),
                "position_error_m": 0.001,
                "orientation_error_rad": 0.01,
            },
            "attempts": [],
        }


def test_the_chosen_roll_reaches_the_quaternion_the_controller_drives() -> None:
    """The defect this replaces: a rolled pose that was never commanded.

    Rolling inside the solver sealed a rolled quaternion in a receipt while the
    control-plan row kept the authored one.  The live differential-IK
    controller drives the plan's orientation, so the roll survived only as a
    null-space posture preference -- which by construction cannot move the
    primary six-dimensional pose objective.  This asserts on the plan the
    controller actually reads, by calling the real worker seam.
    """

    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_selected_grasp_roll,
    )

    authored = [0.0, 0.0, 0.0, 1.0]
    # The authored orientation cannot be held; rolling away from it can.
    def margin_for(q):
        deviation = sum(abs(a - b) for a, b in zip(q, authored))
        return 0.0 if deviation < 0.05 else 0.2

    derived, receipt = _with_selected_grasp_roll(
        servo=_RollServo(margin_for), control_plan=_roll_plan(authored)
    )

    assert receipt["status"] == "applied"
    assert receipt["selected_roll_rad"] != 0.0
    rows = {r["phase_id"]: r for r in derived["scripted_positive_actions"]}
    # Every grasp-holding phase now carries the rolled orientation...
    for phase_id in (
        "contact_open", "contact_close", "joint_path_01", "joint_path_02",
        "joint_path_03", "joint_path_04", "release",
    ):
        assert rows[phase_id]["target_quaternion_world_xyzw"] != authored
        assert rows[phase_id]["authored_target_quaternion_world_xyzw"] == authored
        assert rows[phase_id]["applied_grasp_roll_rad"] == receipt["selected_roll_rad"]
    # ...and every phase that is not holding the grasp is untouched.
    for phase_id in ("prealign", "approach", "retreat"):
        assert rows[phase_id]["target_quaternion_world_xyzw"] == authored
        assert "applied_grasp_roll_rad" not in rows[phase_id]
    # The derived plan is a different, digest-bound plan.
    assert derived["plan_digest"] != receipt["source_control_plan_digest"]
    assert derived["plan_digest"] == receipt["derived_control_plan_digest"]


def test_a_roll_is_refused_when_any_holding_phase_falls_below_the_floor() -> None:
    """Contact entry alone is not enough to admit a roll for the family."""

    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_selected_grasp_roll,
    )

    authored = [0.0, 0.0, 0.0, 1.0]
    plan = _roll_plan(authored)
    # Make one door-arc phase distinguishable, and starve every rolled pose
    # there: contact would be happy, the family must not be.
    for row in plan["scripted_positive_actions"]:
        if row["phase_id"] == "joint_path_03":
            row["target_position_world_m"] = [2.0, 0.0, 0.5]

    def margin_for(q):
        return 0.0 if sum(abs(a - b) for a, b in zip(q, authored)) < 0.05 else 0.2

    class _StarveOnePhase(_RollServo):
        def solve_grasp_target_multistart(self, **kwargs):
            result = super().solve_grasp_target_multistart(**kwargs)
            if kwargs["target_position_world_m"][0] == 2.0:
                result["selected"]["minimum_joint_limit_margin_rad"] = 0.001
            return result

    derived, receipt = _with_selected_grasp_roll(
        servo=_StarveOnePhase(margin_for), control_plan=plan
    )

    assert receipt["status"] == "not_applied"
    assert "no_roll_clears_required_margin_across_the_family" in receipt["reason"]
    # The plan is returned unchanged rather than half-rolled.
    rows = {r["phase_id"]: r for r in derived["scripted_positive_actions"]}
    assert all(
        row["target_quaternion_world_xyzw"] == authored for row in rows.values()
    )


def test_an_authored_orientation_that_holds_is_left_alone() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_selected_grasp_roll,
    )

    authored = [0.0, 0.0, 0.0, 1.0]
    derived, receipt = _with_selected_grasp_roll(
        servo=_RollServo(lambda q: 0.4), control_plan=_roll_plan(authored)
    )

    assert receipt["status"] == "not_applied"
    assert receipt["reason"] == "authored_roll_admissible"
    assert receipt["selected_roll_rad"] == 0.0
    rows = derived["scripted_positive_actions"]
    assert all(r["target_quaternion_world_xyzw"] == authored for r in rows)


def test_a_contact_phase_commands_the_posture_its_preflight_solved() -> None:
    """C42 and C43 measured the controller discarding the solved vector.

    The Cartesian controller re-derives a posture from scratch, and it walked
    0.19 to 0.53 rad away from the solved vector -- whose own forward
    kinematics sat inside the arrival gate -- onto one whose kinematics were
    already 20 mm outside it.  Tracking was never the problem: the arm reached
    what it was told within 0.008 rad.  It was told the wrong thing.
    """

    from blueprint_pipeline.adp009d_control_episode import (
        validate_task_control_plan,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_held_solved_contact_vectors,
    )

    solved = [0.11, 0.22, 0.33, -0.44, 0.55, 0.66, -0.77]
    rows = {r["phase_id"]: r for r in plan["scripted_positive_actions"]}
    derived, receipt = _with_held_solved_contact_vectors(
        control_plan=plan,
        scripted_pose_joint_targets=[
            {
                "phase_id": "contact_open",
                "target_position_world_m": rows["contact_open"][
                    "target_position_world_m"
                ],
                "target_quaternion_world_xyzw": rows["contact_open"][
                    "target_quaternion_world_xyzw"
                ],
                "joint_positions_rad": list(solved),
            }
        ],
    )
    assert receipt["status"] == "applied"
    assert receipt["held_phase_ids"] == ["contact_open"]
    # The plan is re-digested, so the validator accepts it.
    assert derived["plan_digest"] != receipt["source_control_plan_digest"]

    validated = validate_task_control_plan(derived, task_spec=task)

    held = [
        row
        for row in validated["scripted_positive_actions"]
        if row.get("hold_solved_arm_joint_positions_rad")
    ]
    assert held, "the solved vector must survive validation"
    assert held[0]["hold_solved_arm_joint_positions_rad"] == pytest.approx(solved)
    # The pose and its gate are untouched: a solved vector that does not put
    # the real fingertip on the target still fails honestly.
    assert held[0]["mode"] == "ik_pose"
    assert held[0]["arrival_tolerance_m"] > 0.0
    # Phases that did not carry one are unaffected.
    assert any(
        row.get("hold_solved_arm_joint_positions_rad") is None
        for row in validated["scripted_positive_actions"]
        if row.get("mode") == "ik_pose"
    )


def test_contact_close_compensates_measured_closed_pad_midpoint_travel() -> None:
    """C54 lost its global close solution and scored a moving TCP.

    The measured Robotiq pad midpoint advances along the grasp approach axis
    while closing.  Make contact_close a distinct, compensated global-IK pose
    so the *closed* measured midpoint lands on the authored grasp target.
    """

    from blueprint_pipeline.adp009d_control_episode import (
        validate_task_control_plan,
    )
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_closed_pad_midpoint_compensated_contact,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    source = next(
        row
        for row in plan["scripted_positive_actions"]
        if row["phase_id"] == "contact_close"
    )
    source["target_quaternion_world_xyzw"] = [
        0.0,
        0.7071067811865475,
        0.7071067811865476,
        0.0,
    ]
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    original = list(source["target_position_world_m"])
    derived, receipt = _with_closed_pad_midpoint_compensated_contact(
        control_plan=plan,
        gripper_convention={
            "open_command": 0.0,
            "closed_command": 1.0,
            "pad_midpoint_controlled_body_m": {
                "0.0": [
                    0.13009979642662167,
                    3.1868757355280053e-09,
                    -4.266703018673823e-09,
                ],
                "1.0": [
                    0.14365582057985699,
                    2.427310029085028e-07,
                    -2.635381418092386e-07,
                ],
            },
        },
        # C55's sealed reset frames. Controlled-body +X is not grasp-frame
        # +X, and at the authored grasp it maps almost entirely to world +Y.
        current_controlled_body_pose_world=[
            3.813486337661743,
            9.166367530822754,
            0.5625487565994263,
            -0.5000001490115236,
            0.5000003874300885,
            0.4999997019767144,
            0.4999997615813556,
        ],
        current_grasp_frame_pose_world=[
            3.8134863112111903,
            9.166367386974125,
            0.4324489601728867,
            0.7071061879139766,
            -0.7071072373633318,
            -0.0003500406233814535,
            0.0002671211765628578,
        ],
    )

    assert receipt["status"] == "applied"
    assert receipt["pad_midpoint_travel_m"] == pytest.approx(0.01355602415783117)
    checked = validate_task_control_plan(derived, task_spec=task)
    close = next(
        row
        for row in checked["scripted_positive_actions"]
        if row["phase_id"] == "contact_close"
    )
    # The compensation follows the measured controlled-body delta through the
    # rigid body-to-grasp transform. C55's discarded implementation compared
    # these differently expressed vectors directly and incorrectly no-opped.
    assert close["target_position_world_m"] == pytest.approx(
        [
            original[0] + 1.8634649404e-06,
            original[1] - 0.013556019075444,
            original[2] + 1.15897110396e-05,
        ]
    )
    assert close["arrival_target_position_world_m"] == pytest.approx(original)
    open_row = next(
        row
        for row in checked["scripted_positive_actions"]
        if row["phase_id"] == "contact_open"
    )
    assert open_row["target_position_world_m"] == pytest.approx(original)
    assert derived["plan_digest"] == canonical_digest(
        derived, digest_field="plan_digest"
    )


def test_a_malformed_solved_vector_is_refused_not_ignored() -> None:
    """Silently dropping it would restore the defect it exists to fix."""

    from blueprint_pipeline.adp009d_control_episode import (
        ControlEpisodeError,
        validate_task_control_plan,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    for row in plan["scripted_positive_actions"]:
        if row.get("phase_id") == "contact_open" and row.get("mode") == "ik_pose":
            row["hold_solved_arm_joint_positions_rad"] = [0.1, 0.2]  # not seven
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    with pytest.raises(ControlEpisodeError):
        validate_task_control_plan(plan, task_spec=task)


def test_measured_contact_frontier_starts_from_observed_success() -> None:
    from blueprint_pipeline.adp009d_control_episode import (
        validate_task_control_plan,
    )
    from blueprint_pipeline.native_task_arena_controls_worker import (
        MEASURED_CONTACT_ENTRY_PHASE_ID,
        MEASURED_CONTACT_FRONTIER_PHASE_PREFIX,
        _with_measured_contact_frontier,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    for row in plan["scripted_positive_actions"]:
        if row["phase_id"] == "contact_open":
            row["hold_solved_arm_joint_positions_rad"] = [0.9] * 7
        elif row["phase_id"] == "contact_close":
            row["hold_solved_arm_joint_positions_rad"] = [0.8] * 7
    contact_close_source = next(
        row
        for row in plan["scripted_positive_actions"]
        if row["phase_id"] == "contact_close"
    )
    path_row = copy.deepcopy(contact_close_source)
    path_row.update(
        {
            "phase_id": "joint_path_01",
            "target_position_world_m": [0.6, 0.2, 0.4],
            "hold_solved_arm_joint_positions_rad": None,
            "hold_arm_joint_positions_during_gripper_transition": False,
        }
    )
    plan["scripted_positive_actions"].insert(-1, path_row)
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    known_joints = [0.11, -0.22, 0.33, -0.44, 0.55, -0.66, 0.77]
    probe = {
        "status": "measured",
        "cells": [
            {
                "status": "measured",
                "offset_m": [0.0, 0.0, -0.04],
                "joint_positions_rad": known_joints,
                "joint_tracking_error_rad": 0.0002,
                "measured_distance_to_requested_m": 0.0042,
                "measured_grasp_frame_orientation_world_xyzw": [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "contact_steps": 0,
            },
            {
                "status": "measured",
                "offset_m": [0.0, 0.0, -0.02],
                "joint_positions_rad": [0.2] * 7,
                "joint_tracking_error_rad": 0.0003,
                "measured_distance_to_requested_m": 0.008,
                "measured_grasp_frame_orientation_world_xyzw": [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "contact_steps": 0,
            },
        ],
    }

    derived, receipt = _with_measured_contact_frontier(
        control_plan=plan,
        reachability_probe=probe,
        task_spec=task,
    )
    checked = validate_task_control_plan(derived, task_spec=task)
    rows = checked["scripted_positive_actions"]
    entry = next(row for row in rows if row["phase_id"] == MEASURED_CONTACT_ENTRY_PHASE_ID)
    contact = next(row for row in rows if row["phase_id"] == "contact_open")
    contact_close = next(row for row in rows if row["phase_id"] == "contact_close")
    joint_path = next(row for row in rows if row["phase_id"] == "joint_path_01")
    frontier = [
        row
        for row in rows
        if row["phase_id"].startswith(MEASURED_CONTACT_FRONTIER_PHASE_PREFIX)
    ]

    assert receipt["status"] == "applied"
    assert receipt["probe_measured_error_m"] == pytest.approx(0.0042)
    assert entry["hold_solved_arm_joint_positions_rad"] == pytest.approx(
        known_joints
    )
    assert entry["target_position_world_m"] == pytest.approx([0.5, 0.1, 0.36])
    assert frontier == []
    assert contact["hold_solved_arm_joint_positions_rad"] == pytest.approx(
        known_joints
    )
    assert contact["target_position_world_m"] == pytest.approx([0.5, 0.1, 0.36])
    assert contact_close["target_position_world_m"] == pytest.approx(
        [0.5, 0.1, 0.4]
    )
    assert contact_close["hold_solved_arm_joint_positions_rad"] == pytest.approx(
        [0.8] * 7
    )
    assert contact_close[
        "hold_arm_joint_positions_during_gripper_transition"
    ] is False
    assert contact_close["require_bilateral_task_contact"] is True
    assert contact_close[
        "bilateral_task_contact_minimum_force_n"
    ] == pytest.approx(0.5)
    assert contact_close["maximum_steps"] == 39
    assert joint_path["target_position_world_m"] == pytest.approx(
        [0.6, 0.2, 0.4]
    )
    assert receipt["frontier_phase_ids"] == [
        MEASURED_CONTACT_ENTRY_PHASE_ID,
        "contact_open",
    ]
    assert receipt["promoted_standoff_m"] == pytest.approx(0.04)
    assert receipt["probe_clearance_axis_alignment_dot"] == pytest.approx(1.0)
    assert receipt["rewritten_grasp_holding_phase_ids"] == [
        "contact_open",
    ]
    assert receipt["measured_joint_vector_bound_phase_ids"] == [
        "contact_open",
    ]
    assert receipt["contact_close_step_budget_added"] == 27
    assert receipt["contact_close_maximum_steps"] == 39
    assert receipt["synthetic_frontier_rows_inserted"] == 0
    assert receipt["replaced_branch_replay_rows"] == 0
    assert derived["plan_digest"] == canonical_digest(
        derived, digest_field="plan_digest"
    )


def test_measured_contact_frontier_refuses_unproven_probe_cells() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_measured_contact_frontier,
    )

    plan = _branch_replay_plan(_branch_replay_task())
    derived, receipt = _with_measured_contact_frontier(
        control_plan=plan,
        reachability_probe={
            "cells": [
                {
                    "status": "measured",
                    "offset_m": [0.0, 0.0, -0.02],
                    "joint_positions_rad": [0.2] * 7,
                    "joint_tracking_error_rad": 0.0002,
                    "measured_distance_to_requested_m": 0.008,
                    "measured_grasp_frame_orientation_world_xyzw": [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "contact_steps": 0,
                }
            ]
        },
    )

    assert receipt["status"] == "not_applied"
    assert receipt["reason"] == "no_noncontact_probe_cell_inside_arrival_gate"
    assert derived == plan


def test_contact_acquisition_adopts_only_physics_admitted_bilateral_cell() -> None:
    from blueprint_pipeline.adp009d_control_episode import (
        validate_task_control_plan,
    )
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_contact_acquisition_candidate,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    original_close = next(
        row
        for row in plan["scripted_positive_actions"]
        if row["phase_id"] == "contact_close"
    )
    original_close["require_bilateral_task_contact"] = True
    original_close["bilateral_task_contact_minimum_force_n"] = 0.5
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    reached_open = [0.11, -0.22, 0.33, -0.44, 0.55, -0.66, 0.77]

    derived, receipt = _with_contact_acquisition_candidate(
        control_plan=plan,
        sweep={
            "status": "measured",
            "best_cell": {
                "cell_index": 17,
                "admitted": True,
                "authored_target_gate_passed": True,
                "candidate_target_position_world_m": [0.501, 0.096, 0.402],
                "candidate_command_target_position_world_m": [
                    0.501,
                    0.0825,
                    0.402,
                ],
                "reached_open_joint_positions_rad": reached_open,
                "approach_offset_m": -0.005,
                "jaw_offset_m": 0.006,
                "lateral_offset_m": 0.0,
            },
        },
    )

    checked = validate_task_control_plan(derived, task_spec=task)
    rows = {
        row["phase_id"]: row for row in checked["scripted_positive_actions"]
    }
    contact_open = rows["contact_open"]
    contact_close = rows["contact_close"]

    assert receipt["status"] == "applied"
    assert receipt["adopted_cell_index"] == 17
    assert contact_open["target_position_world_m"] == pytest.approx(
        [0.501, 0.0825, 0.402]
    )
    assert contact_open["arrival_target_position_world_m"] == pytest.approx(
        [0.501, 0.0825, 0.402]
    )
    assert contact_open["hold_solved_arm_joint_positions_rad"] == pytest.approx(
        reached_open
    )
    assert contact_open["gripper_state"] == "open"
    assert contact_close[
        "hold_arm_joint_positions_during_gripper_transition"
    ] is True
    assert contact_close["target_position_world_m"] == pytest.approx(
        [0.501, 0.0825, 0.402]
    )
    assert contact_close["arrival_target_position_world_m"] == pytest.approx(
        [0.5, 0.1, 0.4]
    )
    assert receipt["authoritative_arrival_target_position_world_m"] == pytest.approx(
        [0.5, 0.1, 0.4]
    )
    assert contact_close["hold_solved_arm_joint_positions_rad"] == pytest.approx(
        reached_open
    )
    assert contact_close["require_bilateral_task_contact"] is True
    assert contact_close[
        "bilateral_task_contact_minimum_force_n"
    ] == pytest.approx(0.5)
    assert derived["plan_digest"] != plan["plan_digest"]
    assert derived["plan_digest"] == canonical_digest(
        derived, digest_field="plan_digest"
    )


def test_contact_acquisition_axes_use_authored_approach_when_open_and_close_share_pose() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _contact_acquisition_axes,
    )

    plan = _branch_replay_plan(_branch_replay_task())
    rows = {
        row["phase_id"]: row for row in plan["scripted_positive_actions"]
    }
    rows["approach"]["target_position_world_m"] = [0.5, 0.0, 0.4]
    shared_contact = [0.5, 0.1, 0.4]
    rows["contact_open"]["target_position_world_m"] = shared_contact
    rows["contact_close"]["target_position_world_m"] = shared_contact

    approach, jaw, lateral = _contact_acquisition_axes(
        control_plan=plan,
        authored_open_target=shared_contact,
        authored_close_target=shared_contact,
        pad_centers={
            "left": [0.0, 0.0, 0.05],
            "right": [0.0, 0.0, -0.05],
        },
    )

    assert approach == pytest.approx([0.0, 1.0, 0.0])
    assert jaw == pytest.approx([0.0, 0.0, 1.0])
    assert lateral == pytest.approx([1.0, 0.0, 0.0])


def test_contact_acquisition_progress_is_atomic_and_timeout_harvestable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _announce_contact_acquisition_cell,
        _persist_progress,
    )

    progress = {
        "schema_version": "native_task_arena_contact_acquisition_sweep.v1",
        "status": "running",
        "executed_cell_count": 8,
        "last_cell": {
            "cell_index": 7,
            "approach_offset_m": -0.005,
            "jaw_offset_m": 0.0,
            "lateral_offset_m": 0.006,
            "admitted": True,
            "maximum_consecutive_bilateral_steps": 2,
            "terminal_task_contact_pad_forces_n": {
                "left_inner_finger": 1.2,
                "right_inner_finger": 1.1,
            },
            "terminal_distance_to_candidate_target_m": 0.004,
            "terminal_distance_to_authored_target_m": 0.004,
            "terminal_orientation_error_rad": 0.05,
        },
    }
    output = tmp_path / "contact_acquisition_sweep.progress.v1.json"

    _persist_progress(output, progress)
    _announce_contact_acquisition_cell(progress)

    retained = json.loads(output.read_text(encoding="utf-8"))
    assert retained["executed_cell_count"] == 8
    assert retained["result_digest"].startswith("sha256:")
    assert not (tmp_path / f".{output.name}.tmp").exists()
    marker = capsys.readouterr().out.strip()
    assert marker.startswith("BLUEPRINT_CONTACT_ACQUISITION_PROGRESS:CELL:i=7:")
    assert ":ok=1:b=2:lf=1.2:rf=1.1:d=0.004:o=0.05" in marker


def test_contact_acquisition_refuses_nonadmitted_best_cell() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_contact_acquisition_candidate,
    )

    plan = _branch_replay_plan(_branch_replay_task())
    derived, receipt = _with_contact_acquisition_candidate(
        control_plan=plan,
        sweep={
            "status": "measured",
            "best_cell": {
                "cell_index": 0,
                "admitted": False,
                "candidate_target_position_world_m": [0.5, 0.1, 0.4],
                "reached_open_joint_positions_rad": [0.1] * 7,
            },
        },
    )

    assert receipt["status"] == "not_applied"
    assert receipt["reason"] == "no_physics_admitted_contact_acquisition_cell"
    assert derived == plan


def test_contact_acquisition_refuses_admitted_cell_without_authored_gate() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_contact_acquisition_candidate,
    )

    plan = _branch_replay_plan(_branch_replay_task())
    derived, receipt = _with_contact_acquisition_candidate(
        control_plan=plan,
        sweep={
            "status": "measured",
            "best_cell": {
                "cell_index": 8,
                "admitted": True,
                "candidate_target_position_world_m": [0.49, 0.1, 0.4],
                "candidate_command_target_position_world_m": [0.49, 0.1, 0.4],
                "reached_open_joint_positions_rad": [0.1] * 7,
                "approach_offset_m": -0.01,
                "jaw_offset_m": 0.0,
                "lateral_offset_m": 0.0,
            },
        },
    )

    assert receipt["status"] == "not_applied"
    assert receipt["reason"] == "physics_admitted_cell_authored_gate_unproven"
    assert derived == plan


def test_measured_anchor_replaces_the_old_contact_replay() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID,
        MEASURED_CONTACT_ENTRY_MAXIMUM_STEPS,
        _with_contact_entry_branch_replay,
        _with_measured_contact_frontier,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    original_contact = next(
        row
        for row in plan["scripted_positive_actions"]
        if row["phase_id"] == "contact_open"
    )
    original_contact["hold_solved_arm_joint_positions_rad"] = [0.9] * 7
    original_contact_maximum_steps = original_contact["maximum_steps"]
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    targets = [
        {
            "phase_id": row["phase_id"],
            "target_position_world_m": row["target_position_world_m"],
            "target_quaternion_world_xyzw": row["target_quaternion_world_xyzw"],
            "joint_positions_rad": [0.0] * 7
            if row["phase_id"] == "approach"
            else [0.4] * 7,
        }
        for row in plan["scripted_positive_actions"]
        if row["phase_id"] in {"approach", "contact_open"}
    ]
    replayed, replay_receipt = _with_contact_entry_branch_replay(
        control_plan=plan,
        scripted_pose_joint_targets=targets,
        task_spec=task,
        actuator_feasible_step_rad=[0.005] * 7,
    )
    assert replay_receipt["status"] == "applied"
    old_rows = [
        row
        for row in replayed["scripted_positive_actions"]
        if row["phase_id"] == CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID
    ]
    assert len(old_rows) > 1
    replayed_contact = next(
        row
        for row in replayed["scripted_positive_actions"]
        if row["phase_id"] == "contact_open"
    )

    derived, receipt = _with_measured_contact_frontier(
        control_plan=replayed,
        reclaimed_contact_steps=replay_receipt["contact_phase_steps_reclaimed"],
        reachability_probe={
            "cells": [
                {
                    "status": "measured",
                    "offset_m": [0.0, 0.0, -0.04],
                    "joint_positions_rad": [0.2] * 7,
                    "joint_tracking_error_rad": 0.0002,
                    "measured_distance_to_requested_m": 0.004,
                    "measured_grasp_frame_orientation_world_xyzw": [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "contact_steps": 0,
                }
            ]
        },
    )
    new_rows = [
        row
        for row in derived["scripted_positive_actions"]
        if row["phase_id"] == CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID
    ]
    assert len(new_rows) == 1
    assert new_rows[0]["mode"] == "ik_pose"
    assert new_rows[0]["hold_solved_arm_joint_positions_rad"] == [0.2] * 7
    assert new_rows[0]["maximum_steps"] == len(old_rows)
    assert new_rows[0]["max_joint_delta_rad"] == pytest.approx(0.005)
    assert receipt["replaced_branch_replay_rows"] == len(old_rows)
    assert receipt["preserved_branch_replay_step_rad"] == pytest.approx(0.005)
    assert receipt["measured_entry_maximum_steps"] == len(old_rows)
    contact = next(
        row
        for row in derived["scripted_positive_actions"]
        if row["phase_id"] == "contact_open"
    )
    assert contact["hold_solved_arm_joint_positions_rad"] == [0.2] * 7
    assert contact["target_position_world_m"] == pytest.approx([0.5, 0.1, 0.36])
    expected_anchor_steps = max(
        MEASURED_CONTACT_ENTRY_MAXIMUM_STEPS,
        replayed_contact["maximum_steps"],
        len(old_rows),
    )
    expected_restoration = min(
        replay_receipt["contact_phase_steps_reclaimed"],
        max(0, len(old_rows) - expected_anchor_steps),
    )
    assert contact["maximum_steps"] == (
        replayed_contact["maximum_steps"] + expected_restoration
    )
    assert contact["maximum_steps"] < original_contact_maximum_steps
    assert receipt["restored_contact_steps"] == expected_restoration
    assert receipt["restoration_limited_by_action_budget"] is True

    from blueprint_pipeline.adp009d_control_episode import (
        validate_task_control_plan,
    )

    validated = validate_task_control_plan(derived, task_spec=task)
    assert validated["plan_digest"] == derived["plan_digest"]


def test_contact_anchor_is_derived_from_the_approach_line() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _contact_approach_anchor_offset,
    )

    plan = _branch_replay_plan(_branch_replay_task())
    rows = {row["phase_id"]: row for row in plan["scripted_positive_actions"]}
    rows["approach"]["target_position_world_m"] = [0.5, -0.02, 0.4]
    rows["contact_open"]["target_position_world_m"] = [0.5, 0.1, 0.4]

    assert _contact_approach_anchor_offset(plan) == pytest.approx(
        [0.0, -0.04, 0.0]
    )


def test_contact_frontier_walks_from_the_proven_anchor_to_authored_contact() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _contact_frontier_offsets,
    )

    offsets = _contact_frontier_offsets([0.0, -0.04, 0.0], sample_count=5)
    expected = [
        [0.0, -0.04, 0.0],
        [0.0, -0.03, 0.0],
        [0.0, -0.02, 0.0],
        [0.0, -0.01, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert len(offsets) == len(expected)
    for observed, row in zip(offsets, expected, strict=True):
        assert observed == pytest.approx(row)
