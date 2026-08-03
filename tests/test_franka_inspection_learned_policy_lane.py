from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.franka_inspection_learned_policy_lane import (
    IdentityBoundLearnedPolicyAdapter,
    LearnedPolicyLaneError,
    build_candidate_packet,
    build_execution_authorization,
    build_execution_bundle,
    build_identity_query_receipt,
    build_terminal_admission_artifact,
    execute_learned_policy_attempt,
    validate_attempt_evidence,
    validate_identity_query_receipt,
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _observation(value: int = 0) -> dict[str, Any]:
    return {
        "observation/exterior_image_1_left": np.full((224, 224, 3), value, dtype=np.uint8),
        "observation/wrist_image_left": np.full((224, 224, 3), value + 1, dtype=np.uint8),
        "observation/joint_position": np.zeros(7),
        "observation/gripper_position": np.zeros(1),
        "prompt": "Inspect the marked work-surface region and keep it visible in the wrist camera.",
    }


class _Policy:
    fixture_or_fake = True

    def __init__(self, identity: dict[str, Any]) -> None:
        self.policy_identity_digest = identity["policy_identity_digest"]
        self.rows = identity["native_action_chunk_rows"]
        self.calls = 0

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        assert "observation/exterior_image_1_left" in observation
        self.calls += 1
        actions = np.zeros((self.rows, 8), dtype=float)
        actions[0, 0] = self.calls / 10.0
        return {"actions": actions}


class _Simulator:
    fixture_or_fake = True

    def __init__(self, reset_digest: str) -> None:
        self.reset_digest = reset_digest
        self.steps = 0

    def reset(self, reset_contract: dict[str, Any]) -> dict[str, Any]:
        assert canonical_digest(reset_contract) == self.reset_digest
        self.steps = 0
        return {"matched_reset_digest": self.reset_digest}

    def observe(self) -> dict[str, Any]:
        return _observation(self.steps)

    def apply_action(self, action: dict[str, Any]) -> dict[str, Any]:
        assert action["control_hz"] == 15
        self.steps += 1
        return {
            "contacts": [{"pair": ["panda_hand", "air"], "force_n": 0.0}],
            "collisions": [],
            "observation": _observation(self.steps),
        }

    def terminal(self) -> bool:
        return self.steps >= 3

    def task_metric(self) -> dict[str, Any]:
        return {
            "metric_spec_digest": _sha("3"),
            "value": 0.5,
            "supported_for_ranking": False,
            "blockers": ["hermetic_fixture_not_real_execution"],
        }


def _reset() -> dict[str, Any]:
    return {
        "scene_digest": _sha("1"),
        "placement_digest": _sha("2"),
        "routing_decision_digest": _sha("4"),
        "target_binding_digest": _sha("5"),
        "random_seed": 7,
        "joint_position_rad": [0.0] * 7,
        "gripper_position": 0.0,
        "external_camera_extrinsics_digest": _sha("6"),
        "wrist_camera_extrinsics_digest": _sha("7"),
        "target_state_digest": _sha("8"),
    }


_FRANKA_LIMITS = [
    [-2.8973, 2.8973],
    [-1.7628, 1.7628],
    [-2.8973, 2.8973],
    [-3.0718, -0.0698],
    [-2.8973, 2.8973],
    [-0.0175, 3.7525],
    [-2.8973, 2.8973],
]


def test_official_five_are_distinct_but_rights_blocked() -> None:
    packet = build_candidate_packet()

    assert packet["candidate_count"] == 5
    assert packet["admitted_candidate_count"] == 0
    assert len({row["checkpoint_digest"] for row in packet["policy_candidates"]}) == 5
    assert {
        blocker
        for audit in packet["candidate_audits"]
        for blocker in audit["blockers"]
    } == {"checkpoint_specific_rights_missing"}


def test_hermetic_query_tests_mechanics_but_cannot_authorize_fleet() -> None:
    packet = build_candidate_packet()
    receipts = [
        build_identity_query_receipt(
            identity=identity,
            observation=_observation(index),
            policy_client=_Policy(identity),
        )
        for index, identity in enumerate(packet["policy_candidates"])
    ]

    authorization = build_execution_authorization(
        candidate_packet=packet,
        query_receipts=receipts,
        routing_decision_digest=_sha("1"),
        placement_digest=_sha("2"),
        metric_spec_digest=_sha("3"),
        matched_reset_digest=_sha("4"),
    )

    assert authorization["policy_execution_authorized"] is False
    assert "candidate_contract_or_rights_admission_incomplete" in authorization["blockers"]
    assert "real_identity_bound_query_missing" in authorization["blockers"]


def test_provider_neutral_adapter_preserves_exact_droid_observation() -> None:
    identity = build_candidate_packet()["policy_candidates"][0]
    backend = _Policy(identity)
    adapter = IdentityBoundLearnedPolicyAdapter(
        backend=backend,
        identity=identity,
        fixture_or_fake=True,
    )
    observation = _observation()

    output = adapter.infer(observation)

    assert set(observation) == {
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/gripper_position",
        "prompt",
    }
    assert output["actions"].shape == (15, 8)
    assert adapter.policy_identity_digest == identity["policy_identity_digest"]


def test_query_receipt_validator_rejects_native_output_tamper() -> None:
    identity = build_candidate_packet()["policy_candidates"][0]
    receipt = build_identity_query_receipt(
        identity=identity,
        observation=_observation(),
        policy_client=_Policy(identity),
    )
    receipt["native_policy_output"]["rows"][0][0] = 0.9

    blockers = validate_identity_query_receipt(receipt, identity=identity)

    assert "identity_query_receipt_digest_mismatch" in blockers
    assert "identity_query_native_output_digest_mismatch" in blockers


def test_runtime_requeries_every_step_and_retains_exact_evidence() -> None:
    identity = build_candidate_packet()["policy_candidates"][0]
    reset = _reset()
    reset_digest = canonical_digest(reset)
    authorization = {
        "policy_execution_authorized": True,
        "routing_decision_digest": reset["routing_decision_digest"],
        "placement_digest": reset["placement_digest"],
        "metric_spec_digest": _sha("3"),
        "matched_reset_digest": reset_digest,
    }
    policy = _Policy(identity)

    receipt = execute_learned_policy_attempt(
        identity=identity,
        authorization=authorization,
        reset_contract=reset,
        policy_client=policy,
        simulator=_Simulator(reset_digest),
        max_control_steps=5,
        joint_limits=_FRANKA_LIMITS,
    )

    assert policy.calls == 3
    assert receipt["fixture_or_fake"] is True
    assert receipt["learned_policy_action_proven"] is False
    assert receipt["fresh_policy_query_count"] == receipt["control_step_count"] == 3
    assert len(receipt["evidence"]["observation_trace"]) == 4
    assert all(len(row["native_policy_output"]["rows"]) == 15 for row in receipt["evidence"]["action_trace"])
    assert validate_attempt_evidence(receipt)["status"] == "validated"


def test_evidence_tamper_is_detected_and_fixture_bundle_is_refused() -> None:
    identity = build_candidate_packet()["policy_candidates"][0]
    reset = _reset()
    reset_digest = canonical_digest(reset)
    authorization = {
        "policy_execution_authorized": True,
        "routing_decision_digest": reset["routing_decision_digest"],
        "placement_digest": reset["placement_digest"],
        "metric_spec_digest": _sha("3"),
        "matched_reset_digest": reset_digest,
    }
    receipt = execute_learned_policy_attempt(
        identity=identity,
        authorization=authorization,
        reset_contract=reset,
        policy_client=_Policy(identity),
        simulator=_Simulator(reset_digest),
        max_control_steps=1,
        joint_limits=_FRANKA_LIMITS,
    )
    receipt["evidence"]["action_trace"][0]["native_policy_output"]["rows"][0][0] = 9.0

    validation = validate_attempt_evidence(receipt)

    assert validation["status"] == "blocked"
    assert "attempt_digest_mismatch" in validation["blockers"]
    with pytest.raises(LearnedPolicyLaneError) as caught:
        build_execution_bundle(
            policy_candidates=[identity] * 5,
            execution_authorization=authorization,
            task_metric={"metric_spec_digest": _sha("3")},
            attempts=[receipt] * 5,
        )
    assert "fixture_attempt_cannot_enter_real_execution_bundle" in caught.value.codes


def test_terminal_artifact_names_exact_no_spend_blocker() -> None:
    packet = build_candidate_packet()
    authorization = build_execution_authorization(
        candidate_packet=packet,
        query_receipts=[],
        routing_decision_digest=_sha("1"),
        placement_digest=_sha("2"),
        metric_spec_digest=_sha("3"),
        matched_reset_digest=_sha("4"),
    )

    terminal = build_terminal_admission_artifact(
        candidate_packet=packet, authorization=authorization
    )

    assert terminal["status"] == "blocked_before_fleet_execution"
    assert terminal["fleet_runnable"] is False
    assert "checkpoint-specific rights evidence" in terminal["exact_fifth_policy_blocker"]
    assert "no GPU/provider spend" in terminal["exact_fifth_policy_blocker"]
