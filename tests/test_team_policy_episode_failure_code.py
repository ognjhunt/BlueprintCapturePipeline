"""A stopped episode's typed failure code: kept where its message never is."""
from __future__ import annotations

import os

import pytest

from blueprint_pipeline import policy_canary_worker_evidence as evidence
from blueprint_pipeline.team_policy_episode_outcome import FAILURE_CODE, episode_failure_fields, failure_code


@pytest.mark.parametrize("message, code", [
    ("policy_canary_gripper_unresolved", "policy_canary_gripper_unresolved"),
    ("policy_episode_prestart_readiness_failed:prepolicy_visual_frame_near_black:wrist;other:error",
     "policy_episode_prestart_readiness_failed:prepolicy_visual_frame_near_black:wrist"),
    ("No module named 'isaacsim.robot_motion.pink'", "module_not_found:isaacsim.robot_motion.pink"),
    ("a:b:c:d:e", "a:b:c:d"),
    ("scene_load_failed:<path>:detail", "scene_load_failed"),
    ("policy_canary_scientific_reset_pair_mismatch:object_pose,joint", "policy_canary_scientific_reset_pair_mismatch"),
    ("CUDA error: out of memory", None),
    ("cuda out of memory", None),
    ("/srv/scene/file.usd not found", None),
    ("", None),
    (None, None),
])
def test_failure_code_keeps_only_leading_typed_segments(message, code):
    assert failure_code(message) == code
    assert code is None or FAILURE_CODE.fullmatch(code)


def test_episode_failure_fields_only_for_a_stopped_episode():
    stopped = {"status": "blocked", "failure_type": "RuntimeError",
               "failure_message": "policy_canary_device_binding_failed"}
    assert episode_failure_fields(stopped) == {"failure_type": "RuntimeError",
                                               "failure_code": "policy_canary_device_binding_failed"}
    assert episode_failure_fields({**stopped, "status": "completed"}) == {}
    # A recorded code is kept when typed; an untyped one falls back to the message.
    assert episode_failure_fields({**stopped, "failure_code": "x:y"})["failure_code"] == "x:y"
    assert episode_failure_fields({**stopped, "failure_code": "Not A Code"})["failure_code"] == (
        "policy_canary_device_binding_failed")
    # An exception name that is not an identifier, or prose, is dropped.
    assert episode_failure_fields({"status": "blocked", "failure_type": "bad name", "failure_message": "prose here"}) == {}
    assert episode_failure_fields({"status": "blocked", "typed_harness_failure": "KeyError"}) == {
        "failure_type": "KeyError"}


def test_a_stopped_episode_names_its_typed_failure_on_the_console(monkeypatch):
    """2026-11-05: ten episodes stopped before the first policy query and the archive holding
    why was gone at settlement. Writing each failure record also writes one typed line to the
    console the host keeps; never the message."""
    read_fd, write_fd = os.pipe()
    try:
        monkeypatch.setenv(evidence._PROGRESS_FD_ENV, str(write_fd))
        monkeypatch.setenv(evidence._CELL_INDEX_ENV, "3")
        evidence._emit_episode_stop({
            "episode_failure_stage": "before_first_observation", "failure_type": "PolicyEpisodeError",
            "failure_message": "policy_episode_prestart_readiness_failed:reset_joint_state_out_of_bounds at /srv/x",
            "failure_code": "policy_episode_prestart_readiness_failed:reset_joint_state_out_of_bounds"})
        monkeypatch.setenv(evidence._CELL_INDEX_ENV, "not-an-index")
        evidence._emit_episode_stop({"failure_message": "free text with /a/path", "failure_type": "not a name",
                                     "episode_failure_stage": "made_up"})
        os.close(write_fd)
        write_fd = -1
        assert os.read(read_fd, 4096).decode().splitlines() == [
            "BLUEPRINT_POLICY_CANARY_EPISODE_STOPPED cell=3 stage=before_first_observation type=PolicyEpisodeError"
            " code=policy_episode_prestart_readiness_failed:reset_joint_state_out_of_bounds",
            "BLUEPRINT_POLICY_CANARY_EPISODE_STOPPED cell=unknown stage=unknown type=unknown code=untyped",
        ]
        # No channel, or a lost one, never changes execution.
        monkeypatch.setenv(evidence._PROGRESS_FD_ENV, "not-a-number")
        evidence._emit_episode_stop({})
        monkeypatch.delenv(evidence._PROGRESS_FD_ENV)
        evidence._emit_episode_stop({})
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def test_the_failure_record_carries_its_code_and_announces_itself(tmp_path, monkeypatch):
    read_fd, write_fd = os.pipe()
    try:
        monkeypatch.setenv(evidence._PROGRESS_FD_ENV, str(write_fd))
        monkeypatch.setenv(evidence._CELL_INDEX_ENV, "0")
        path = evidence._write_episode_failure_gap(
            output_root=tmp_path, run_id="run", context={"cell_id": "cell-0", "candidate_id": "c", "seed": 1},
            failure=RuntimeError("policy_canary_gripper_unresolved"), progress={})
        os.close(write_fd)
        write_fd = -1
        import json
        record = json.loads(path.read_text())
        assert record["failure_code"] == "policy_canary_gripper_unresolved"
        assert os.read(read_fd, 4096).decode() == (
            "BLUEPRINT_POLICY_CANARY_EPISODE_STOPPED cell=0 stage=before_first_observation type=RuntimeError"
            " code=policy_canary_gripper_unresolved\n")
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)
