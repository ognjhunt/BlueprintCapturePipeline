from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.policy_episode_trace_evidence import episode_trace_evidence


def test_episode_trace_evidence_seals_scored_state_contacts_and_object_motion() -> None:
    state, contacts, trajectory = episode_trace_evidence(
        joint_trace=[[0.0] * 7, [0.1] * 7],
        task_samples=[
            {
                "step_index": 0,
                "can_pose_world": [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],
                "finger_contact_forces_n": [0.0, 0.0],
            },
            {
                "step_index": 1,
                "can_pose_world": [0.1, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],
                "finger_contact_forces_n": [1.2, 1.1],
            },
        ],
        task_pose_field="can_pose_world",
    )

    assert state["trace_digest"] == canonical_digest(
        state, digest_field="trace_digest"
    )
    assert contacts["typed_gap"] is None
    assert contacts["samples"][1]["finger_contact_forces_n"] == [1.2, 1.1]
    assert trajectory["typed_gap"] is None
    assert trajectory["samples"][1]["task_object_pose_world"][0] == 0.1


def test_episode_trace_evidence_preserves_typed_channel_gaps() -> None:
    _state, contacts, trajectory = episode_trace_evidence(
        joint_trace=[[0.0] * 7],
        task_samples=[{"step_index": 0, "joint_positions_rad": {"hinge": 0.0}}],
        task_pose_field="task_object_pose_world",
    )

    assert contacts["typed_gap"] == "contact_force_channels_unavailable_in_task_samples"
    assert trajectory["typed_gap"] == "task_object_pose_unavailable_in_task_samples"
