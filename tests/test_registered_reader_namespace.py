"""Actual payload APIs must not interpret reserved aliases as legacy input."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_consumer.py
#   src/blueprint_pipeline/control_plane_registered_checkpoint_cache.py
#   src/blueprint_pipeline/native_g1_development_pair.py
#   src/blueprint_pipeline/wam_provider_object_store.py

import os

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_registered_experiment_issuer import encoded


def _alias(tmp_path, root, family, name, alias):
    if alias == "dotdot":
        (root / "ordinary").mkdir()
        return root / "ordinary" / ".." / family / name
    if alias == "symlink_parent":
        (root / "shortcut").symlink_to(root / family, target_is_directory=True)
        return root / "shortcut" / name
    link = tmp_path / "outside-shortcut"
    link.symlink_to(root / family / name, target_is_directory=True)
    return link


@pytest.mark.parametrize("alias", ["dotdot", "symlink_parent", "symlink_outside"])
def test_actual_native_result_reader_refuses_unadmitted_reserved_alias(
    tmp_path, monkeypatch, alias
):
    from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer
    from blueprint_pipeline import native_g1_development_pair as pair

    root = tmp_path / "lanes"
    name = "registered-" + "a" * 32
    target = root / "g1" / name
    target.mkdir(parents=True)
    value = dict(
        candidate_id="candidate",
        request_digest="sha256:" + "b" * 64,
        status="blocked",
        scene_plan_digest=None,
        ranking_eligible=False,
        physical_outcome_claimed=False,
    )
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    (target / "result.json").write_bytes(encoded(value))
    monkeypatch.setattr(consumer, "LANE_ROOTS", (root,))
    monkeypatch.setattr(consumer, "AUTHORITY_ROOT", tmp_path / "absent-authority")
    path = _alias(tmp_path, root, "g1", name, alias) / "result.json"
    assert os.path.samefile(path, target / "result.json")
    with pytest.raises(ValueError, match="experiment_consumer_path_unsafe"):
        pair._read_result(
            path,
            candidate_id="candidate",
            scene_plan_digest="sha256:" + "c" * 64,
            request_digest=value["request_digest"],
        )


@pytest.mark.parametrize("alias", ["dotdot", "symlink_parent", "symlink_outside"])
def test_actual_wam_hash_refuses_unadmitted_reserved_cache_alias(tmp_path, monkeypatch, alias):
    from blueprint_pipeline import control_plane_registered_checkpoint_cache as cache
    from blueprint_pipeline import wam_provider_object_store as wam

    root = tmp_path / "lanes"
    target = root / "g1-checkpoint" / "registered-cache"
    target.mkdir(parents=True)
    (target / "payload").write_bytes(b"tiny genuine cache payload")
    monkeypatch.setattr(cache, "_REGISTERED_ROOTS", frozenset((root,)))
    path = _alias(tmp_path, root, "g1-checkpoint", "registered-cache", alias) / "payload"
    assert os.path.samefile(path, target / "payload")
    with pytest.raises(cache.NeededCheckpointCacheError, match="needed_cache_path_unsafe"):
        wam._sha256_file(path)
