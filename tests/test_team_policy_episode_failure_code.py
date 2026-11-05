"""A stopped episode's typed failure code: kept where its message never is."""
from __future__ import annotations

import pytest

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
    ("/home/user/scene/file.usd not found", None),
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
