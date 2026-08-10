from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.registered_scene_components import (
    RegisteredSceneComponentError,
    build_registered_scene_components,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _checked_freeze() -> dict:
    return json.loads(
        (
            REPO_ROOT
            / "docs/arm_decision_proof_v1/manifests"
            / "second_scene_840796_scene_task_freeze.v1.json"
        ).read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    ("scene_id", "folder", "target_id", "target_label"),
    [
        ("840313", "0442_840313", "160", "canned_beverage"),
        ("840796", "0498_840796", "123", "refrigerator"),
    ],
)
def test_component_contract_is_scene_and_task_neutral(
    scene_id: str, folder: str, target_id: str, target_label: str
) -> None:
    freeze = _checked_freeze()
    freeze["scene"].update(
        {
            "publisher_scene_id": scene_id,
            "interiorgs_folder": folder,
            "target_instance_id": target_id,
            "target_semantic_label": target_label,
        }
    )
    components = build_registered_scene_components(freeze)

    appearance, receipt = components["interiorgs_appearance_scene"]
    assert appearance["scene_mapping"]["publisher_scene_id"] == scene_id
    assert appearance["target_binding"]["interiorgs_instance_id"] == target_id
    assert appearance["target_binding"]["semantic_label"] == target_label
    assert appearance["component_id"] == (
        f"public-scene-interiorgs-{scene_id}-{target_id}"
    )
    assert "support_instance_id" not in appearance["target_binding"]
    assert receipt["status"] == "admitted"
    assert receipt["checks"]["coordinate_frame_status_bound"] is True
    assert receipt["checks"]["coordinate_frame_qualification_status"] == (
        "legacy_verified"
    )
    assert receipt["component_manifest_digest"] == appearance["manifest_digest"]
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_component_contract_preserves_rights_and_articulated_claim_boundary() -> None:
    components = build_registered_scene_components(_checked_freeze())
    appearance, _ = components["interiorgs_appearance_scene"]
    collision, _ = components["sage3d_collision_companion"]

    assert appearance["rights"]["external_provider_upload_authorized"] is False
    assert appearance["rights"]["redistribution_allowed"] is False
    assert collision["claim_boundaries"]["source_joint_topology_claimed"] is False
    assert collision["target_binding"]["collision_prim_path"].startswith("/Root/")


def test_component_contract_rejects_scene_folder_cross_join() -> None:
    freeze = copy.deepcopy(_checked_freeze())
    freeze["scene"]["interiorgs_folder"] = "0498_wrong"

    with pytest.raises(RegisteredSceneComponentError, match="folder_mismatch"):
        build_registered_scene_components(freeze)
