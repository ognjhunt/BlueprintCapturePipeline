import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_visual_review_continuation as visual
from blueprint_pipeline import task_evaluation_visual_review_authority as approval
from blueprint_pipeline import task_evaluation_terminal_adoption_retirement as retirement
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_scene_execution_authority as execution
from blueprint_pipeline import task_evaluation_unstarted_controls_reservations as cancellations
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_task_evaluation_terminal_adoption_retirement import adopted as adopted
from tests.test_task_evaluation_unstarted_controls_reservations import reserved as reserved, put


@pytest.fixture
def failed_review(adopted, tmp_path):
    config, source, owner, old, old_path, reserve, run = adopted
    config["plan_root"] = str(tmp_path / "plans")
    root = Path(config["progression_root"]) / source["launch_id"] / "cpu-robot-binding"
    token = old["intent_digest"].removeprefix("sha256:")[:16]
    attempt = root / f"agent-placement-attempts-{token}" / "attempt_000"
    proposal = {
        "candidate_id": "candidate-1",
        "pose": {"position_world_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "support_surface_id": "table",
        "rationale": "Previously selected inventory member.",
        "addressed_blockers": [],
        "uncertainty": "Needs visual review.",
    }
    inventory = {
        "trajectory_digest": "sha256:" + "d" * 64,
        "candidate_inventory_digest": "sha256:" + "f" * 64,
        "candidates": [proposal],
    }
    inventory["checkpoint_digest"] = canonical_digest(inventory, digest_field="checkpoint_digest")
    put(attempt / "task_evaluation_robot_placement_candidate_inventory.v1.json", inventory)
    receipt = {
        "schema_version": "task_evaluation_robot_placement_receipt.v1",
        "status": "blocked",
        "accepted_pose": None,
        "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "native_attempt_count": 0,
        "model_grades_controls": False,
        "scene_binding_digest": "sha256:" + "a" * 64,
        "task_binding_digest": "sha256:" + "b" * 64,
        "task_trajectory_digest": inventory["trajectory_digest"],
        "candidate_inventory_digest": inventory["candidate_inventory_digest"],
        "rounds": [
            {
                "proposal": proposal,
                "proposal_model": "gpt-5.6-sol",
                "geometry_gate": {"status": "passed"},
                "visual_review": {"camera_views_are_sufficient": False, "status": "uncertain"},
                "native_attempt": None,
            }
        ],
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    put(attempt / "task_evaluation_robot_placement_receipt.v1.json", receipt)
    completion = {
        "provider_call_performed": True,
        "runtime_result_digest": receipt["receipt_digest"],
        "authorization_receipt_digest": old["intent_digest"],
    }
    completion["completion_receipt_digest"] = canonical_digest(
        completion, digest_field="completion_receipt_digest"
    )
    put(
        root
        / "agent-official-openai-cost"
        / attempt.parent.name
        / attempt.name
        / "openai_official_cost_run_completion.v1.json",
        completion,
    )
    return config, source, owner, old, reserve, receipt


def grant(failed_review):
    config, source, owner, old, reserve, receipt = failed_review
    result = approval.authorize(
        scene_root=config["scene_root"],
        intent_id=owner["intent_id"],
        source_attempt_id="controls-adopted-placement",
        source_placement_receipt_digest=receipt["receipt_digest"],
        authorization_reference="Explicit fixture owner approval: one $0.15 visual-only correction; total cap and GPU retries unchanged.",
        now=102,
    )
    return json.loads(Path(result["path"]).read_text())


def test_correction_needs_explicit_owner_approval(failed_review):
    config, source, owner, *_ = failed_review
    kwargs = dict(
        config=config, intent_id=owner["intent_id"], source=source, expected_commit="b" * 40
    )
    assert visual.discover(**kwargs) is None
    grant(failed_review)
    packet = visual.discover(**kwargs)
    assert visual.validate(packet)["proposal"]["candidate_id"] == "candidate-1"
    assert packet["maximum_reviewer_calls"] == 1


def test_only_unused_gpu_holds_move_and_spent_review_stays_reserved(failed_review):
    config, source, owner, old, reserve, receipt = failed_review
    authorized = grant(failed_review)
    kwargs = dict(
        config=config, intent_id=owner["intent_id"], source=source, expected_commit="b" * 40
    )
    packet = visual.discover(**kwargs)
    result = retirement.retire_unmaterialized_adoptions(
        config=config,
        intent_id=owner["intent_id"],
        source=source,
        expected_production_commit="b" * 40,
        visual_review_continuation=packet,
    )
    assert len(result) == 2
    directory = Path(config["scene_root"]) / owner["intent_id"]
    spent = intake._read(directory / "attempts/controls-adopted-placement.json", "attempt_digest")
    assert cancellations.validated_cancellation(directory, spent) is None
    assert (
        visual.discover(**kwargs) == packet
    )  # restart after cancellation preserves the exact input key
    for phase in ("destination", "construction", "controls"):
        reserve("controls-corrected-" + phase, 0.266666, commit="b")
    review = intake.reserve_scene_attempt(
        queue_root=config["scene_root"],
        intent_id=owner["intent_id"],
        attempt_id="controls-corrected-placement",
        source_commit="b" * 40,
        runtime_digest="sha256:" + "e" * 64,
        input_digest="sha256:" + "f" * 64,
        provider="openai",
        maximum_spend_usd=0.15,
        now=103,
        visual_review_authority=authorized,
    )
    assert (
        execution.scene_execution_authority_blockers(
            execution.bind_scene_attempt(review),
            source_commit="b" * 40,
            provider="openai",
            maximum_spend_usd=0.15,
            now=104,
        )
        == []
    )
    reserve("policies", 4, commit="b")
    live = [intake._read(p, "attempt_digest") for p in (directory / "attempts").glob("*.json")]
    live = [a for a in live if cancellations.validated_cancellation(directory, a) is None]
    assert len(live) == 7 and sum(a["maximum_spend_usd"] for a in live) == pytest.approx(19.509998)
    assert (
        intake._read(directory / "intent.json", "intent_digest")["request"]["execution"][
            "max_retries"
        ]
        == 0
    )
    with pytest.raises(ValueError, match="already_reserved"):
        intake.reserve_scene_attempt(
            queue_root=config["scene_root"],
            intent_id=owner["intent_id"],
            attempt_id="duplicate-review",
            source_commit="b" * 40,
            runtime_digest="sha256:" + "e" * 64,
            input_digest="sha256:" + "0" * 64,
            provider="openai",
            maximum_spend_usd=0.15,
            now=104,
            visual_review_authority=authorized,
        )


@pytest.mark.parametrize(
    "mutation", ["geometry_failed", "clear_views", "accepted", "native_plan", "wrong_receipt"]
)
def test_non_visual_failures_and_started_native_work_cannot_release_holds(failed_review, mutation):
    config, source, owner, old, reserve, receipt = failed_review
    grant(failed_review)
    root = Path(config["progression_root"]) / source["launch_id"] / "cpu-robot-binding"
    if mutation == "native_plan":
        put(
            Path(config["plan_root"]) / "plan.json",
            {"source_launch_id": source["launch_id"], "expected_production_commit": "c" * 40},
        )
    else:
        if mutation == "geometry_failed":
            receipt["rounds"][0]["geometry_gate"]["status"] = "rejected"
        if mutation == "clear_views":
            receipt["rounds"][0]["visual_review"]["camera_views_are_sufficient"] = True
        if mutation == "accepted":
            receipt["status"] = "accepted"
        if mutation == "wrong_receipt":
            receipt["extra"] = "changed"
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        put(next(root.rglob("task_evaluation_robot_placement_receipt.v1.json")), receipt)
    kwargs = dict(
        config=config, intent_id=owner["intent_id"], source=source, expected_commit="b" * 40
    )
    if mutation == "wrong_receipt":
        with pytest.raises(ValueError, match="owner_approved_other_review"):
            visual.discover(**kwargs)
    else:
        assert visual.discover(**kwargs) is None
    directory = Path(config["scene_root"]) / owner["intent_id"]
    for p in (directory / "attempts").glob("controls-adopted-*"):
        assert (
            cancellations.validated_cancellation(directory, intake._read(p, "attempt_digest"))
            is None
        )


def test_approval_never_authorizes_gpu_or_larger_cost(failed_review):
    config, source, owner, *_ = failed_review
    authorized = grant(failed_review)
    for provider, cost in [("vast", 0.15), ("openai", 0.16)]:
        with pytest.raises(ValueError, match="authority_invalid"):
            intake.reserve_scene_attempt(
                queue_root=config["scene_root"],
                intent_id=owner["intent_id"],
                attempt_id="bad",
                source_commit="b" * 40,
                runtime_digest="sha256:" + "e" * 64,
                input_digest="sha256:" + "f" * 64,
                provider=provider,
                maximum_spend_usd=cost,
                now=103,
                visual_review_authority=authorized,
            )
