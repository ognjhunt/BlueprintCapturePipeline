"""Native orchestration port rehearsal; Isaac/provider/authority outputs are doubles.

The real episode implementations have separate deterministic tests. This file
checks exact native signatures, all4 ordered subjects and teardown/error wiring.
It does not establish simulator or policy competence.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import create_autospec

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.exact_workcell_native_session import run_native_evaluation_cell
from blueprint_pipeline.policy_scientific_reset import REQUIRED_CHANNELS
from tests.test_exact_workcell_evaluation_session import _inputs


def test_native_cell_dispatches_both_controls_and_measured_droid_pair_with_exact_bindings(
    tmp_path, monkeypatch
):
    import blueprint_pipeline.adp009d_control_episode as controls
    import blueprint_pipeline.adp009d_policy_episode as policy_loop
    import blueprint_pipeline.native_task_control_plan as plans
    import blueprint_pipeline.native_task_arena_controls_worker as control_worker
    import blueprint_pipeline.native_task_arena_policy_bundle as policy_bundle
    import blueprint_pipeline.native_task_arena_policy_canary_worker as worker
    import blueprint_pipeline.exact_workcell_native_session as native
    import blueprint_pipeline.exact_workcell_native_admission as authority
    import blueprint_pipeline.policy_scientific_reset as resets

    args = _inputs(tmp_path)
    cell = args["matrix"]["cells"][0]
    rows = [row for row in args["schedule"]["rows"] if row["cell_id"] == cell["cell_id"]]
    plan = {
        "task_id": args["request"]["task_binding"]["task_id"],
        "task_kind": "rigid_pick_place",
        "robot": {"robot_id": "franka_panda"},
        "task_spec": {"settle_window_samples": 3},
        "plan_digest": args["source_binding"]["source_plan_digest"],
    }
    specs = {
        candidate: {
            "candidate_id": candidate,
            "scene_plan_digest": plan["plan_digest"],
            "task_id": plan["task_id"],
            "prompt": "Move the admitted object",
            "max_policy_queries": 3,
            "open_loop_horizon": 1,
        }
        for candidate in ("pi05_droid", "groot_n17_droid")
    }
    for candidate, spec in specs.items():
        spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
        args["schedule_request"]["candidate_set"]["candidate_identity_digests"][candidate] = spec[
            "execution_spec_digest"
        ]
    config = {
        **args,
        "source_scene_plan": plan,
        "construction_result": {"real_input": "replaced_in_unit_test"},
        "candidate_specs": specs,
        "native_packet_root": str(tmp_path / "packet"),
        "packet_request": {},
        "observation_integrity_authority": {},
        "policy_runtime_evidence_root": str(tmp_path / "measured-policy-runtime"),
        "policy_readiness": {},
        "scenario_suite": {},
    }
    events = []
    prepared = copy.deepcopy(plan)
    monkeypatch.setattr(native, "prepare_native_runtime_cell", lambda **_: prepared)
    monkeypatch.setattr(
        native, "read_native_runtime_cell", lambda **_: {"status": "applied_and_readback_verified"}
    )
    monkeypatch.setattr(
        plans, "materialize_native_task_control_plan", lambda **_: {"scripted_positive_actions": []}
    )
    monkeypatch.setattr(control_worker, "_construction_global_ik_joint_targets", lambda **_: [])
    monkeypatch.setattr(
        policy_bundle, "validate_native_task_policy_execution_spec", lambda spec: spec
    )
    monkeypatch.setattr(
        worker,
        "appearance_render_backend_from_plan",
        lambda *_, **__: {"launch_render_path": "native", "receipt_digest": "sha256:" + "d" * 64},
    )
    monkeypatch.setattr(
        authority,
        "validate_native_source_policy_candidates",
        lambda **_: events.append("source_authority"),
    )
    monkeypatch.setattr(
        authority, "derive_native_cell_policy_admission", lambda **_: {"candidate_specs": specs}
    )
    monkeypatch.setattr(
        resets,
        "read_native_reset_channels",
        lambda *_: {
            "observed": {key: {"measured": 1} for key in REQUIRED_CHANNELS},
            "sources": {key: "native_test_buffer" for key in REQUIRED_CHANNELS},
            "gaps": [],
        },
    )

    def receipt(kwargs, *, control=False):
        environment = kwargs["environment"]
        reset = environment.scientific_reset_readback(
            kwargs.get("control_id", kwargs.get("candidate_id"))
        )
        return {
            "episode_id": kwargs["episode_id"],
            "scientific_reset": reset,
            "control_passed": True if control else None,
        }

    control_call = create_autospec(
        controls.run_task_neutral_control,
        side_effect=lambda **kwargs: receipt(kwargs, control=True),
    )
    monkeypatch.setattr(controls, "run_task_neutral_control", control_call)
    policy_call = create_autospec(
        policy_loop.run_policy_episode, side_effect=lambda **kwargs: receipt(kwargs)
    )

    def launch(path, *, device, appearance_render_path):
        events.append("launch")
        return SimpleNamespace(close=lambda: events.append("close_app")), {}

    def camera_gate(*, simulation_app, built, packet_request, plan, output_root):
        events.append("camera_gate")
        return {"policy_observation_integrity_passed": True}

    def renderer_guard(*, observation_gate, output_path):
        events.append("renderer_guard")
        return {"status": "configured"}

    def identity(*, output_root, spec):
        assert output_root == tmp_path / "measured-policy-runtime"
        assert spec["candidate_id"] == "groot_n17_droid"
        events.append("groot_identity")
        return {"measured": True}, {}

    def client(spec, *, groot_worker_identity_receipt):
        events.append(spec["candidate_id"])
        assert (groot_worker_identity_receipt is not None) == (
            spec["candidate_id"] == "groot_n17_droid"
        )
        return SimpleNamespace(close=lambda: events.append("close_policy"))

    env = SimpleNamespace(
        reset=lambda **_: None,
        unwrapped=SimpleNamespace(scene={"robot": object()}),
        close=lambda: events.append("close_env"),
    )
    built = SimpleNamespace(env=env)
    runtime = SimpleNamespace(
        device="test",
        launch_isaac=launch,
        preflight_dependency_matrix=lambda **_: {"all_required_available": True},
        prepare_preconstruction=lambda **_: {"passed": True},
        build_environment=lambda *_, **__: built,
        prepare_appearance_renderer=lambda **_: {"passed": True},
        read_device_binding=lambda *_, **__: {"passed": True},
        gripper_probe=lambda **_: {
            "status": "measured",
            "open_command": 1.0,
            "closed_command": 0.0,
        },
        make_servo=lambda **_: object(),
        make_rigid_task_readback=lambda _: object(),
        to_tensor=lambda x: x,
        build_episode_environment=lambda **_: (object(), {}),
        wrap_rigid_scoring_environment=lambda **_: object(),
        prepolicy_camera_gate=camera_gate,
        configure_post_gate_renderer=renderer_guard,
        groot_worker_identity=identity,
        policy_client=client,
        run_policy_episode=policy_call,
    )
    begun, retained = [], []
    run_native_evaluation_cell(
        config=config,
        cell=cell,
        rows=rows,
        runtime=runtime,
        begin=lambda row: begun.append(row["episode_id"]),
        commit=lambda row, record: retained.append(record),
    )
    assert begun == [row["episode_id"] for row in rows]
    assert len(retained) == 4 and control_call.call_count == policy_call.call_count == 2
    assert events.index("camera_gate") < events.index("renderer_guard") < events.index("pi05_droid")
    assert events.index("groot_identity") < events.index("groot_n17_droid")
    assert events[-2:] == ["close_env", "close_app"]
    assert all(
        call.kwargs["require_complete_multicamera_media"] is True
        for call in policy_call.call_args_list
    )
    assert all(
        call.kwargs["require_prestart_readiness"] is True for call in policy_call.call_args_list
    )
    assert all((tmp_path / record["artifact"]["relative_path"]).is_file() for record in retained)
    for record in retained:
        receipt_path = tmp_path / record["artifact"]["relative_path"]
        receipt_path.relative_to(tmp_path / record["media_root_relative"])
        sidecar = (
            tmp_path
            / "receipts"
            / (canonical_digest({"episode_id": record["episode_id"]})[7:] + ".record.json")
        )
        assert json.loads(sidecar.read_text()) == record


def test_native_receipt_location_passes_actual_media_index_root_contract(tmp_path):
    from blueprint_pipeline.adp_episode_evidence_index import _episode_row
    from blueprint_pipeline.exact_workcell_native_session import _receipt_path
    from tests.test_adp009d_episode_evidence_index import _receipt

    episode_id = "cell-a.zero_action_negative"
    canonical = _receipt_path(tmp_path, episode_id, "cell-a")
    media_root = canonical.parent.parent
    media_root.mkdir(parents=True)
    produced = _receipt(
        media_root, episode_id=episode_id, subject_id="zero_action_negative", learned=False
    )
    canonical.parent.mkdir(exist_ok=True)
    canonical.write_bytes(produced.read_bytes())
    index = _episode_row(media_root, canonical)
    assert index["episode_id"] == episode_id
    assert index["receipt"]["relative_path"].startswith("receipts/")
