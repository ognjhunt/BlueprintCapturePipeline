from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.public_scene_task_selection import (
    PublicSceneTaskSelectionError,
    load_selection_preregistration,
    validate_selection_preregistration,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "docs"
    / "arm_decision_proof_v1"
    / "manifests"
    / "second_scene_selection_preregistration.v1.json"
)


def test_second_scene_selection_is_outcome_blind_and_excludes_first_scene() -> None:
    observed = load_selection_preregistration(MANIFEST)

    scene_ids = [row["publisher_scene_id"] for row in observed["candidate_order"]]
    assert scene_ids == sorted(scene_ids)
    assert "840313" not in scene_ids
    assert observed["previously_used_scene_ids"] == ["840313"]
    assert observed["selected_scene"] is None
    assert observed["learned_policy_outcomes_accessed"] is False
    assert observed["new_inpainting_outcomes_accessed"] is False
    assert all(row["method_outcomes_consulted"] is False for row in observed["candidate_order"])
    assert observed["claim_ceiling"] == "development_only_selection_rule"
    assert observed["task_family"] == "single_joint_articulated_open_or_close"
    assert observed["usd_content_agents_joint_agent"]["version"] == "0.5.2"
    assert observed["usd_content_agents_joint_agent"]["commit"] == (
        "36dbf3f274f8e256637230a05a085853f65cc175"
    )


def test_second_scene_selection_rejects_prior_scene_or_policy_leakage() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["candidate_order"][0]["publisher_scene_id"] = "840313"
    payload["learned_policy_outcomes_accessed"] = True

    with pytest.raises(PublicSceneTaskSelectionError) as caught:
        validate_selection_preregistration(payload)

    assert "selection_preregistration_prior_scene_reused" in caught.value.errors
    assert "selection_preregistration_policy_outcome_leakage" in caught.value.errors


def test_second_scene_selection_rejects_threshold_or_digest_mutation() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(payload)
    mutated["thresholds"]["minimum_target_visible_views"] = 1

    with pytest.raises(PublicSceneTaskSelectionError) as caught:
        validate_selection_preregistration(mutated)

    assert "selection_preregistration_thresholds_invalid" in caught.value.errors
    assert "selection_preregistration_digest_invalid" in caught.value.errors
