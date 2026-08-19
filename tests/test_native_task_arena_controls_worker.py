from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_controls_worker import (
    _RigidScoringEnvironment,
    _canonical_digest,
    _input_binding_mismatches,
    _load_and_verify_manifest,
    _verified_runtime_inputs,
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
