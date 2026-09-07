"""Account observation is read-only, fresh, and separate from immutable selection proof."""
import json

import pytest

from blueprint_pipeline import gpu_render_providers as providers
from blueprint_pipeline import task_evaluation_prefix_observation as observation
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_sam31_prefix_adoption import record


def inventory(at=1000):
    return dict(provider="vast", status="observed", api_confirmed=True, name_prefix="",
                live_resource_count=0, resources=[], http=200, observed_at_epoch=at,
                raw_provider_response_recorded=False)


def provider(monkeypatch, value):
    calls = []
    class ReadOnly:
        def billable_inventory(self, *, name_prefix):
            calls.append(name_prefix)
            return value
    def get(name):
        assert name == "vast"
        return ReadOnly()
    monkeypatch.setattr(providers, "get_render_provider", get)
    monkeypatch.setattr(observation.time, "time", lambda: 1001)
    return calls


@pytest.mark.parametrize("changes", [
    {"status": "blocked", "api_confirmed": False},
    {"observed_at_epoch": 100}, {"observed_at_epoch": 1002},
    {"observed_at_epoch": float("nan")}, {"observed_at_epoch": True},
    {"name_prefix": "blueprint-"}, {"http": 403}, {"provider": "runpod"},
    {"live_resource_count": False}, {"live_resource_count": 1, "resources": [{"instance_id": "7"}]},
    {"blockers": ["inventory_ambiguous"]},
])
def test_refuses_missing_stale_partial_or_ambiguous_global_zero(tmp_path, monkeypatch, changes):
    calls = provider(monkeypatch, {**inventory(), **changes})
    with pytest.raises(ValueError):
        observation.selection_observation(tmp_path)
    assert calls == [""]
    assert not (tmp_path / "prefix_selection.json").exists()


def test_raw_inventory_receipt_is_not_a_spend_guard(tmp_path, monkeypatch):
    calls = provider(monkeypatch, inventory())
    path, at = observation.selection_observation(tmp_path)
    saved = json.loads(path.read_text())
    assert saved["schema_version"] == observation.PREFIX_ZERO_SCHEMA
    assert saved["provider"] == "vast"
    assert saved["live_resource_count"] == 0 and saved["resources"] == []
    assert saved["observation_digest"] == canonical_digest(
        saved, digest_field="observation_digest"
    )
    assert at == 1001 and calls == [""]


def test_restart_preserves_selection_witness_but_observes_current_inventory(tmp_path, monkeypatch):
    provider(monkeypatch, inventory())
    path, at = observation.selection_observation(tmp_path)
    selected = {"schema_version": "task_evaluation_sam31_prefix_selection.v1",
                "status": "reusable_prefix_selected",
                "provider_zero_observation": record(path),
                "provider_zero_checked_at_epoch": at,
                "selection_digest": ""}
    selected["selection_digest"] = canonical_digest(
        selected, digest_field="selection_digest"
    )
    selection = tmp_path / "prefix_selection.json"
    selection.write_text(json.dumps(selected))
    original = selection.read_bytes()
    calls = provider(monkeypatch, inventory(3000))
    monkeypatch.setattr(observation.time, "time", lambda: 3001)
    fresh, fresh_at = observation.selection_observation(tmp_path)
    assert fresh != path and fresh_at == 3001
    assert calls == [""] and selection.read_bytes() == original
    assert len(list((tmp_path / "prefix-provider-observations").glob("*.json"))) == 2
    provider(monkeypatch, {**inventory(), "live_resource_count": 1, "resources": [{"instance_id": "7"}]})
    with pytest.raises(ValueError):
        observation.selection_observation(tmp_path)
    assert selection.read_bytes() == original


def test_interrupted_uncommitted_selection_can_refresh_after_expiry(tmp_path, monkeypatch):
    provider(monkeypatch, inventory())
    old, _ = observation.selection_observation(tmp_path)
    provider(monkeypatch, inventory(3000))
    monkeypatch.setattr(observation.time, "time", lambda: 3001)
    fresh, at = observation.selection_observation(tmp_path)
    assert fresh != old and old.exists() and at == 3001


def test_no_reuse_snapshot_does_not_pin_future_observation(tmp_path, monkeypatch):
    provider(monkeypatch, inventory())
    old, _ = observation.selection_observation(tmp_path)
    selection = tmp_path / "prefix_selection.json"
    value = {"schema_version": "task_evaluation_sam31_prefix_selection.v1",
             "status": "no_reusable_prefix", "rejected_candidates": [],
             "provider_zero_observation": record(old),
             "provider_zero_checked_at_epoch": 1001, "selection_digest": ""}
    value["selection_digest"] = canonical_digest(value, digest_field="selection_digest")
    selection.write_text(json.dumps(value))
    original = selection.read_bytes()
    provider(monkeypatch, inventory(2000))
    monkeypatch.setattr(observation.time, "time", lambda: 2001)
    fresh, at = observation.selection_observation(tmp_path)
    assert fresh != old and at == 2001 and selection.read_bytes() == original
