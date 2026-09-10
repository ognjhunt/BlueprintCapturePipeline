"""Offline interruption recovery from recorded wire inputs and existing lossless PNGs."""
from __future__ import annotations

import json

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline.adp009d_droid_observation import CANDIDATE_VIEW_SHAPES, resize_with_pad
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.episode_visual_evidence import finalize_failed_policy_visual_evidence, persist_observation_frame
from blueprint_pipeline.groot_n17_wire_client import encode_wire_message
from blueprint_pipeline.native_task_arena_policy_canary_worker import _resolved_scene_plan
from blueprint_pipeline.policy_canary_interrupted_cell_recovery import recover_interrupted_cell_result
from blueprint_pipeline.policy_request_evidence import capture_request, persist_request_evidence
from tests.test_native_task_arena_policy_canary_lifecycle_rehearsal import _stage_runtime_root


def _read(path):
    return json.loads(path.read_text())


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _stage(tmp_path):
    runtime, output = _stage_runtime_root(tmp_path)
    child = output / "cell_runs/00"
    child.mkdir(parents=True)
    inputs = _read(runtime / "runtime_inputs/policy_canary_runtime_inputs.json")
    base = _read(runtime / "native_task_packet/native_task_arena_scene_plan.v1.json")
    base["plan_digest"] = canonical_digest(base, digest_field="plan_digest")
    _write(runtime / "native_task_packet/native_task_arena_scene_plan.v1.json", base)
    manifest = _read(runtime / "adp_arena_provider_manifest.json")
    manifest["arena_scene_plan_digest"] = base["plan_digest"]
    for candidate in inputs["candidate_ids"]:
        path = runtime / f"runtime_inputs/policy_execution_spec.{candidate}.json"
        spec = _read(path)
        spec["max_policy_queries"] = 3
        spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
        _write(path, spec)
        manifest["execution_spec_digests"][candidate] = spec["execution_spec_digest"]
    manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
    _write(runtime / "adp_arena_provider_manifest.json", manifest)
    task = _resolved_scene_plan(base, inputs["cells"][0], task_success_contract=inputs["task_success_contract"])["task_spec"]
    return runtime, child, inputs, canonical_digest(task)


def _artifact(path, root, role):
    import hashlib
    return {"role": role, "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size, "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}


def _request(runtime, child, inputs, task_digest, candidate, index):
    spec = _read(runtime / f"runtime_inputs/policy_execution_spec.{candidate}.json")
    episode_id = f"{inputs['run_id']}--{inputs['cells'][0]['cell_id']}--{candidate}"
    raw = {camera: np.full((18, 32, 3), 35 + index * 50 + delta, dtype=np.uint8)
           for camera, delta in (("external", 0), ("wrist", 10), ("overview", 20))}
    height, width = CANDIDATE_VIEW_SHAPES[candidate]
    views = {camera: resize_with_pad(pixels, height=height, width=width) for camera, pixels in raw.items()}
    if candidate == "pi05_droid":
        request = {"observation/exterior_image_1_left": views["external"], "observation/wrist_image_left": views["wrist"],
                   "observation/joint_position": np.zeros(7), "observation/gripper_position": np.ones(1), "prompt": spec["prompt"]}
    else:
        request = {"video": {"exterior_image_1_left": views["external"][None, None], "wrist_image_left": views["wrist"][None, None]},
                   "state": {"joint_position": np.zeros((1, 1, 7), dtype=np.float32)},
                   "language": {"annotation.language.language_instruction": [[spec["prompt"].strip()]]}}
    receipt = capture_request(request, transport="groot_zmq" if candidate == "groot_n17_droid" else "openpi_websocket_msgpack_numpy",
        scientific_wire_bytes=encode_wire_message(request), decoded_wire_request=request)
    cell = inputs["cells"][0]
    request_ref = persist_request_evidence(receipt, root=child / "episodes", episode_id=episode_id, query_index=index,
        binding={"candidate_id": candidate, "cell_id": cell["cell_id"], "seed": cell["seed"],
                 "resolved_scenario_digest": cell["resolved_scenario_digest"], "task_spec_digest": task_digest})
    frame = persist_observation_frame(np.concatenate([views["external"], views["wrist"]], axis=1),
        output_dir=child / "episodes", episode_id=episode_id, frame_index=index, kind="policy-input")
    frame.update(candidate_exact_policy_input=True, candidate_id=candidate)
    frame["frame_manifest_digest"] = canonical_digest(frame, digest_field="frame_manifest_digest")
    for camera, pixels in raw.items():
        directory = child / "episodes/media" / episode_id / "frames" / camera
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(directory / f"{index * 2:06d}-policy-input.png")
        if index == 0:
            Image.fromarray(pixels).save(directory / "000001-review-sample.png")
    return episode_id, frame, request_ref


def _sealed_pi05_failure(runtime, child, inputs, task_digest):
    episode_id, frame, request_ref = _request(runtime, child, inputs, task_digest, "pi05_droid", 0)
    visual, artifacts = finalize_failed_policy_visual_evidence(output_dir=child / "episodes", episode_id=episode_id,
        identity={"candidate_id": "pi05_droid"}, exact_policy_input_frames=[frame], failure_reason="ConnectionClosedError:keepalive ping timeout")
    manifest = child / "episodes" / visual["frame_manifest"]["relative_path"]
    video = child / "episodes" / next(iter(visual["videos"].values()))["relative_path"]
    value = {"schema_version": "policy_canary_episode_failure_evidence.v2", "status": "blocked",
        "run_kind": "internal_policy_canary", "claim_ceiling": "diagnostic_policy_execution",
        "candidate_id": "pi05_droid", "cell_id": inputs["cells"][0]["cell_id"], "seed": inputs["cells"][0]["seed"],
        "candidate_policy_queried": False, "candidate_action_returned": False, "actions_reached_robot": False,
        "arm_moved": False, "first_observation_retained": True, "policy_outcome_interpretable": False,
        "failure_type": "ConnectionClosedError", "typed_harness_failure": "ConnectionClosedError",
        "failure_message": "sent 1011 (internal error) keepalive ping timeout; no close frame received",
        "visual_evidence": visual, "evidence_artifacts": {
            "frame_manifest": _artifact(manifest, child, "lossless_frame_manifest"),
            "review_video": _artifact(video, child, "review_video")},
        "episode": {"episode_id": episode_id, "visual_evidence": visual,
            "media_artifacts": [*artifacts, request_ref], "score": {"status": "not_scored"}}, "gap_digest": ""}
    value["gap_digest"] = canonical_digest(value, digest_field="gap_digest")
    path = child / "episodes" / f"{episode_id}.failure_evidence.json"
    _write(path, value)
    return path, value


def _snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_recovers_interrupted_pair_and_preserves_original_failure_and_pixels(tmp_path):
    runtime, child, inputs, task_digest = _stage(tmp_path)
    original_path, original = _sealed_pi05_failure(runtime, child, inputs, task_digest)
    for index in range(2):
        _request(runtime, child, inputs, task_digest, "groot_n17_droid", index)
    before = _snapshot(child)
    result = recover_interrupted_cell_result(runtime_root=runtime, child_root=child,
        selected_cell_index=0, reason="policy_canary_cell_timeout", timeout_seconds=900)
    assert result["status"] == "blocked"
    assert result["result_digest"] == canonical_digest(result, digest_field="result_digest")
    assert result["task_success_contract_digest"] == inputs["task_success_contract_digest"]
    assert result["session_closeout"]["runtime_closed"] is False
    assert result["automatic_retry_performed"] is False
    pi05, groot = result["episodes"]
    assert pi05["failure_type"] == original["failure_type"]
    assert pi05["failure_message"] == original["failure_message"]
    assert pi05["visual_evidence"] == original["visual_evidence"]
    assert pi05["evidence_artifacts"]["frame_manifest"] == original["evidence_artifacts"]["frame_manifest"]
    assert pi05["original_failure_receipt"]["relative_path"] == original_path.relative_to(child).as_posix()
    assert groot["status"] == "blocked"
    assert groot["typed_harness_failure"] == "interrupted_after_first_observation"
    assert groot["retained_policy_request_count"] == 2
    assert groot["candidate_policy_query_attempted"] is True
    assert groot["candidate_policy_queried"] is False
    assert groot["candidate_action_returned"] is False
    assert groot["actions_reached_robot"] is False
    assert groot["arm_moved"] is False
    assert groot["scientific_reset"] is None
    assert groot["reset_state_digest"] is None
    assert groot["episode"]["score"]["status"] == "not_scored"
    assert groot["candidate_policy_action_queries"] == []
    assert groot["commanded_actions"] == []
    assert set(groot["visual_evidence"]["videos"]) == {"external", "wrist", "overview", "policy_input_composite"}
    for camera, video in groot["visual_evidence"]["videos"].items():
        assert video["playback_only"] is True
        assert video["native_timestamps_recovered"] is False
        assert video["decode_round_trip_passed"] is True
        assert video["decoded_frame_count"] == (2 if camera == "policy_input_composite" else 3)
    manifest = _read(child / groot["evidence_artifacts"]["frame_manifest"]["relative_path"])
    assert manifest["frame_manifest_digest"] == canonical_digest(manifest, digest_field="frame_manifest_digest")
    assert [row["frame_index"] for row in manifest["camera_streams"]["external"]] == [0, 1, 2]
    assert all(row["exact_wire_pixels_verified"] for row in manifest["lossless_composite_frames"])
    assert all(row["native_timestamp_ns"] is None for row in manifest["camera_streams"]["external"])
    for row in result["episodes"]:
        spec = _read(runtime / f"runtime_inputs/policy_execution_spec.{row['candidate_id']}.json")
        assert row["checkpoint_digest"] == spec["checkpoint_digest"]
        assert row["runtime_identity_digest"] == spec["runtime_identity_digest"]
    for relative, content in before.items():
        assert (child / relative).read_bytes() == content
    inventory = result["artifact_inventory"]
    assert len({row["relative_path"] for row in inventory}) == len(inventory)
    for row in inventory:
        assert _artifact(child / row["relative_path"], child, row["role"])["sha256"] == row["sha256"]
    assert recover_interrupted_cell_result(runtime_root=runtime, child_root=child,
        selected_cell_index=0, reason="policy_canary_cell_timeout", timeout_seconds=900) == result


def test_no_first_observation_is_a_typed_gap_without_invented_frames_or_reset(tmp_path):
    runtime, child, _inputs, _task_digest = _stage(tmp_path)
    result = recover_interrupted_cell_result(runtime_root=runtime, child_root=child,
        selected_cell_index=0, reason="interrupted_during_startup")
    assert result["status"] == "blocked"
    for row in result["episodes"]:
        assert row["typed_harness_failure"] == "interrupted_before_first_observation"
        assert row["first_observation_retained"] is False
        assert row["retained_policy_request_count"] == 0
        assert row["candidate_policy_queried"] is False
        assert row["candidate_policy_query_attempted"] is False
        assert row["reset_state_digest"] is None
        assert row["evidence_artifacts"]["frame_manifest"] is None
        assert row["visual_evidence"]["videos"] == {}
    assert not list(child.rglob("*.png"))
    assert not list(child.rglob("*.mp4"))


@pytest.mark.parametrize("fault", ["seed", "candidate", "task", "request_digest", "wire_pixels", "raw_pixels", "source_failure", "spec", "plan", "symlink", "foreign_episode"])
def test_tampered_or_rebound_evidence_never_seals_a_recovered_result(tmp_path, fault):
    runtime, child, inputs, task_digest = _stage(tmp_path)
    original_path, _original = _sealed_pi05_failure(runtime, child, inputs, task_digest)
    episode_id, _frame, request_ref = _request(runtime, child, inputs, task_digest, "groot_n17_droid", 0)
    request_path = child / "episodes" / request_ref["relative_path"]
    if fault in {"seed", "candidate", "task", "request_digest"}:
        receipt = _read(request_path)
        if fault == "request_digest":
            receipt["request_digest"] = "sha256:" + "f" * 64
        else:
            key = {"seed": "seed", "candidate": "candidate_id", "task": "task_spec_digest"}[fault]
            receipt["episode_binding"][key] = 999 if fault == "seed" else "wrong"
            receipt["evidence_digest"] = canonical_digest(receipt, digest_field="evidence_digest")
        _write(request_path, receipt)
    elif fault in {"wire_pixels", "raw_pixels"}:
        frame_dir = child / "episodes/media" / episode_id / "frames"
        path = frame_dir / "000000-policy-input.png" if fault == "wire_pixels" else frame_dir / "external/000000-policy-input.png"
        with Image.open(path) as image:
            array = np.array(image)
        array[0, 0] = 255
        Image.fromarray(array).save(path)
    elif fault == "source_failure":
        value = _read(original_path)
        value["failure_message"] = "altered"
        _write(original_path, value)
    elif fault == "spec":
        path = runtime / "runtime_inputs/policy_execution_spec.groot_n17_droid.json"
        value = _read(path)
        value["checkpoint_digest"] = "sha256:" + "e" * 64
        value["execution_spec_digest"] = canonical_digest(value, digest_field="execution_spec_digest")
        _write(path, value)
    elif fault == "plan":
        path = runtime / "native_task_packet/native_task_arena_scene_plan.v1.json"
        value = _read(path)
        value["plan_digest"] = "sha256:" + "e" * 64
        _write(path, value)
    elif fault == "symlink":
        request_path.rename(request_path.with_suffix(".original"))
        request_path.symlink_to(request_path.with_suffix(".original"))
    else:
        directory = child / "episodes/media/other-run--other-cell--groot_n17_droid/policy-requests"
        directory.mkdir(parents=True)
        request_path.rename(directory / request_path.name)
    expected = {"seed": "request_binding_mismatch", "candidate": "request_binding_mismatch",
        "task": "request_binding_mismatch", "request_digest": "policy_request_evidence_digest_invalid",
        "wire_pixels": "policy_pixels_mismatch", "raw_pixels": "raw_camera_policy_pixels_mismatch",
        "source_failure": "original_failure_binding_mismatch", "spec": "execution_spec_binding_mismatch",
        "plan": "scene_plan_binding_mismatch", "symlink": "artifact_symlink_forbidden",
        "foreign_episode": "unbound_request_artifact"}
    with pytest.raises((ValueError, RuntimeError), match=expected[fault]):
        recover_interrupted_cell_result(runtime_root=runtime, child_root=child,
            selected_cell_index=0, reason="offline_interruption")
    assert not (child / "native_task_arena_policy_canary_session_result.v1.json").exists()


def test_existing_canonical_result_is_never_overwritten(tmp_path):
    runtime, child, _inputs, _task_digest = _stage(tmp_path)
    path = child / "native_task_arena_policy_canary_session_result.v1.json"
    _write(path, {"status": "already_completed", "result_digest": "original"})
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite_forbidden"):
        recover_interrupted_cell_result(runtime_root=runtime, child_root=child,
            selected_cell_index=0, reason="offline_interruption")
    assert path.read_bytes() == before


def test_new_helper_stays_inside_the_provider_bundle_import_closure():
    from blueprint_pipeline import policy_canary_interrupted_cell_recovery as recovery
    from blueprint_pipeline.provider_runtime_import_closure import provider_runtime_import_closure_blockers
    from tests.test_provider_runtime_import_closure import CANARY_SHIPPED_MODULES

    path = recovery.Path(recovery.__file__)
    assert provider_runtime_import_closure_blockers(package_source_dir=path.parent,
        shipped_module_names=[*CANARY_SHIPPED_MODULES, path.name]) == []


def test_recovery_replay_refuses_new_uninventoried_original_files(tmp_path):
    runtime, child, _inputs, _task_digest = _stage(tmp_path)
    recover_interrupted_cell_result(runtime_root=runtime, child_root=child,
        selected_cell_index=0, reason="offline_interruption")
    (child / "new-evidence.json").write_text('{}\n')
    with pytest.raises(FileExistsError, match="overwrite_forbidden"):
        recover_interrupted_cell_result(runtime_root=runtime, child_root=child,
            selected_cell_index=0, reason="offline_interruption")
