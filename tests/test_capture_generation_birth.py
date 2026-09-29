"""ADP-009D/day28: original-owner capture birth stays separate from intent births."""

import hashlib
import json
from pathlib import Path

import pytest

from tests.test_capture_original_owner_observer import observation
from tests.test_scene_retirement_connected_acceptance import _sealed_file
from tests.test_scene_retirement_real_participants import access_fixture


def _fixture(tmp_path, monkeypatch):
    access, policy, root = access_fixture(tmp_path, monkeypatch)
    owner = observation()
    marker = owner["completion_marker"]
    video = owner["producer_delivery"]["raw_video"]
    prefix = f"scenes/{owner['scene_id']}/captures/{owner['capture_id']}"
    manifest_name = f"{prefix}/raw/manifest.json"
    delivery_key = hashlib.sha256(json.dumps(
        [owner["bucket"], marker["object_name"], marker["generation"]],
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    selector = {"object_name": f"{prefix}/deliveries/{delivery_key}/capture_delivery_membership.json",
                "generation": "17000000000000000003", "size_bytes": 100,
                "sha256": "sha256:" + "d" * 64}
    membership = {
        "schema_version": "capture_delivery_membership.v1", "delivery_key": delivery_key,
        "source_finalize": {"bucket": owner["bucket"], "object_name": marker["object_name"],
                            "generation": marker["generation"]},
        "producer_delivery": {"kind": owner["producer_delivery"]["kind"],
                              "receipt_object_name": owner["producer_delivery"]["server_record"]["object_name"],
                              "receipt_generation": owner["producer_delivery"]["server_record"]["generation"]},
        "raw": [
            {"object_name": marker["object_name"], "relative_path": "raw/capture_upload_complete.json",
             "generation": marker["generation"], "size_bytes": marker["size_bytes"],
             "crc32c": "AAAAAA==", "sha256": marker["sha256"]},
            {"object_name": manifest_name, "relative_path": "raw/manifest.json",
             "generation": "17000000000000000004", "size_bytes": 12,
             "crc32c": "AAAAAA==", "sha256": "sha256:" + "e" * 64},
            {"object_name": video["object_name"], "relative_path": "raw/walkthrough.mov",
             "generation": video["generation"], "size_bytes": video["size_bytes"],
             "crc32c": video["crc32c"], "sha256": "sha256:" + "f" * 64},
        ], "derived": [],
    }
    target = root / "capture"
    return access, policy, target, owner, selector, membership


def test_capture_birth_retains_original_proofs_before_empty_target(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import birth_capture_member

    _, policy, target, owner, selector, membership = _fixture(tmp_path, monkeypatch)
    born = birth_capture_member(target, observation=owner, membership_selector=selector,
                                membership=membership)
    assert born["schema_version"] == "scene_capture_generation.v1"
    assert born["capture_owner_user_id"] == "owner-1"
    assert "owner_intent_id" not in born and "birth_request_raw_ref" not in born
    assert born["pinned_marker"] == owner["completion_marker"]
    assert born["dev"] == target.stat().st_dev and born["ino"] == target.stat().st_ino
    assert list(target.iterdir()) == []
    for field in ("owner_observation_raw_ref", "birth_delivery_raw_ref"):
        ref = born[field]
        raw = Path(ref["path"]).read_bytes()
        assert Path(ref["path"]).parent == Path(policy["generation_store"])
        assert ref == {"path": ref["path"], "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
                       "size_bytes": len(raw)}
    assert birth_capture_member(target, observation=owner, membership_selector=selector,
                                membership=membership) == born


def test_capture_birth_rejects_missing_member_or_changed_delivery_without_target(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import birth_capture_member

    _, _, target, owner, selector, membership = _fixture(tmp_path, monkeypatch)
    membership["raw"] = membership["raw"][:1]
    with pytest.raises(ValueError):
        birth_capture_member(target, observation=owner, membership_selector=selector,
                             membership=membership)
    assert not target.exists()


def test_retired_capture_requires_new_raw_delivery_and_marker(tmp_path, monkeypatch):
    from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import birth_capture_member

    _, policy, target, owner, selector, membership = _fixture(tmp_path, monkeypatch)
    born = birth_capture_member(target, observation=owner, membership_selector=selector,
                                membership=membership)
    key = hashlib.sha256(str(target).encode()).hexdigest() + ".json"
    retired = dict(born, state="retired", state_sequence=born["state_sequence"] + 1)
    _sealed_file(Path(policy["generation_store"]) / key, retired, "state_digest", mode=0o600)
    target.rmdir()
    with pytest.raises(ValueError):
        birth_capture_member(target, observation=owner, membership_selector=selector,
                             membership=membership)
    assert not target.exists()
