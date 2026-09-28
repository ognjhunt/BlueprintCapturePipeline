"""The menu a paired task evaluation run is drawn from."""

from __future__ import annotations

import dataclasses

import pytest

from blueprint_pipeline import task_evaluation_policy_candidate_registry as registry
from blueprint_pipeline.task_evaluation_policy_canary_scene_setup import CANDIDATE_IDS


def test_the_registry_never_overstates_what_can_run() -> None:
    assert registry.registry_violations(CANDIDATE_IDS) == []
    assert registry.runnable_candidate_ids() == ("pi05_droid", "groot_n17_droid")
    pending = {row.candidate_id: row for row in registry.unavailable_candidates()}
    assert sorted(pending) == ["cosmos3_nano_policy_droid", "flux3_action_droid", "molmoact2_droid"]
    assert pending["cosmos3_nano_policy_droid"].status == "integration_pending"
    assert pending["cosmos3_nano_policy_droid"].commercial_use == "allowed"
    assert pending["molmoact2_droid"].status == "license_review_pending"
    assert pending["flux3_action_droid"].commercial_use == "not_allowed"
    assert all(row.checkpoint_digest_kind == "hub_tree_listing" for row in pending.values())
    assert all(len(row.checkpoint_revision) == 40 for row in pending.values())


def test_a_pair_is_two_distinct_runnable_candidates_in_registry_order() -> None:
    assert registry.validate_selected_pair(["groot_n17_droid", "pi05_droid"]) == ("pi05_droid", "groot_n17_droid")
    with pytest.raises(registry.PolicyCandidateRegistryError, match="two_distinct"):
        registry.validate_selected_pair(["pi05_droid", "pi05_droid"])
    with pytest.raises(registry.PolicyCandidateRegistryError) as held:
        registry.validate_selected_pair(["pi05_droid", "cosmos3_nano_policy_droid"])
    assert held.value.blockers == ("policy_candidate_not_runnable:cosmos3_nano_policy_droid:integration_pending",)
    with pytest.raises(registry.PolicyCandidateRegistryError, match="unknown:g05_droid"):
        registry.validate_selected_pair(["pi05_droid", "g05_droid"])


@pytest.mark.parametrize(
    ("change", "violation"),
    [
        ({"status": "runnable", "blockers": (), "reason": None}, "runnable_without_canary_runtime"),
        ({"commercial_use": "under_review", "status": "integration_pending"}, "license_blocked_but_not_marked"),
    ],
)
def test_flipping_a_label_cannot_make_a_policy_runnable(monkeypatch, change, violation) -> None:
    rows = tuple(
        dataclasses.replace(row, **change) if row.candidate_id == "cosmos3_nano_policy_droid" else row
        for row in registry.REGISTRY
    )
    monkeypatch.setattr(registry, "REGISTRY", rows)
    assert any(item.startswith(violation) for item in registry.registry_violations(CANDIDATE_IDS))


def test_a_runnable_policy_needs_verified_bytes_and_a_commercial_license(monkeypatch) -> None:
    rows = tuple(
        dataclasses.replace(row, commercial_use="not_allowed", checkpoint_digest_kind="hub_tree_listing")
        if row.candidate_id == "pi05_droid" else row
        for row in registry.REGISTRY
    )
    monkeypatch.setattr(registry, "REGISTRY", rows)
    violations = registry.registry_violations(CANDIDATE_IDS)
    assert "runnable_without_commercial_license:pi05_droid" in violations
    assert "runnable_without_verified_checkpoint:pi05_droid" in violations
