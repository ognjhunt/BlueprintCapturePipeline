"""Public-arm stand-in software integration tests; simulator doubles never establish native task proof."""

from copy import deepcopy
import pytest
from blueprint_pipeline import scripted_robot_public_arm_robotiq_runtime as runtime
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


@pytest.mark.parametrize("identity", sorted(runtime.BODIES))
def test_each_stand_in_body_binds_its_own_arm_dof(identity):
    from blueprint_pipeline.deployment_embodiment_runtime_configuration import runtime_profiles

    source = next(row for row in runtime_profiles()["profiles"] if row["embodiment_id"] == identity)
    assert runtime.ARM_DOF[identity] == source["source_bindings"][0]["dof"]
    spec = runtime.build_policy_spec(development_manifest(identity))
    assert spec["policy_id"] == runtime.PROFILE_ID and spec["embodiment_id"] == identity
    assert list(spec["grasp_bindings"]) == ["arm"]
    # A body declaring another stand-in's arm cannot borrow its DoF binding.
    other = next(body for body in sorted(runtime.BODIES) if runtime.ARM_DOF[body] != runtime.ARM_DOF[identity])
    manifest = deepcopy(development_manifest(other))
    manifest.pop("embodiment_digest")
    manifest["embodiment_id"] = identity
    with pytest.raises(ValueError, match="scripted_grasp_source_tool_binding_mismatch"):
        runtime.build_policy_spec(manifest)


@pytest.mark.parametrize("fault", ["closure", "contact", "dof", "policy", "body"])
def test_changed_tool_or_profile_is_not_silently_accepted(fault):
    from blueprint_pipeline.scripted_robot_runtime_dispatch import runtime_for_manifest, runtime_for_policy

    manifest = deepcopy(development_manifest(next(iter(sorted(runtime.BODIES)))))
    manifest.pop("embodiment_digest")
    if fault == "closure":
        manifest["grippers"][0]["joints"][0]["closed"] = 0
    elif fault == "contact":
        manifest["grippers"][0]["contact_links"][0] = "unreviewed_pad"
    elif fault == "dof":
        manifest["arms"][0]["joint_names"].pop()
    elif fault == "body":
        # Bare bodies and the recorded three-finger Jaco2 have no scripted binding here.
        for identity in ("fanuc_m10id_12", "yaskawa_hc20", "agile_robots_diana_7", "kinova_jaco2_j2n6s300",
                         "kuka_kr50_r2100"):
            assert runtime_for_manifest({"embodiment_id": identity}) is not runtime
        return
    else:
        assert runtime_for_policy({"policy_id": runtime.PROFILE_ID}) is runtime
        with pytest.raises(ValueError):
            runtime_for_policy({"policy_id": "unreviewed_public_arm"})
        return
    with pytest.raises(ValueError):
        runtime.build_policy_spec(manifest)
