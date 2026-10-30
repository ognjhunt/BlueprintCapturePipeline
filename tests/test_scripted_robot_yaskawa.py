"""Yaskawa software integration tests; simulator doubles never establish native task proof."""

from copy import deepcopy
import pytest
from blueprint_pipeline import scripted_robot_yaskawa_robotiq_runtime as runtime
from tests.test_scripted_robot_grasp_frames import development_manifest


@pytest.mark.parametrize("identity", sorted(runtime.BODIES))
def test_generic_loop_rebinds_current_cell_and_preserves_independent_score(
    tmp_path, monkeypatch, identity
):
    from tests import test_scripted_robot_robotiq as shared

    monkeypatch.setattr(shared, "runtime", runtime)
    shared.test_actual_loop_rebinds_current_cell_for_six_and_seven_dof_without_learned_claims(
        tmp_path, monkeypatch, identity
    )


@pytest.mark.parametrize("fault", ["closure", "contact", "dof", "policy"])
def test_changed_tool_or_profile_is_not_silently_accepted(fault):
    from blueprint_pipeline.scripted_robot_runtime_dispatch import runtime_for_policy

    manifest = deepcopy(development_manifest(next(iter(runtime.BODIES))))
    manifest.pop("embodiment_digest")
    if fault == "closure":
        manifest["grippers"][0]["joints"][0]["closed"] = 0
    elif fault == "contact":
        manifest["grippers"][0]["contact_links"][0] = "unreviewed_pad"
    elif fault == "dof":
        manifest["arms"][0]["joint_names"].pop()
    else:
        assert runtime_for_policy({"policy_id": runtime.PROFILE_ID}) is runtime
        with pytest.raises(ValueError):
            runtime_for_policy({"policy_id": "unreviewed_yaskawa"})
        return
    with pytest.raises(ValueError):
        runtime.build_policy_spec(manifest)
