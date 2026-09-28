# Covers: src/blueprint_pipeline/control_plane_lane_scratch_retention.py
"""ADP-009D/day-28: lane observations grant no cleanup authority."""
from __future__ import annotations

import json
import os

import pytest

from blueprint_pipeline import control_plane_lane_scratch as producer
from blueprint_pipeline import control_plane_lane_scratch_retention as retention
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


@pytest.fixture
def roots(tmp_path, monkeypatch):
    root, pins = tmp_path.resolve() / "lanes", tmp_path.resolve() / "pins"
    root.mkdir()
    pins.mkdir()
    monkeypatch.setattr(retention, "_ALLOWED_ROOTS", frozenset({str(root)}))
    return root, pins


def lease(**changes):
    value = producer._creation_lease("lane-1", "folder-1", owner="owner-1", reason="fixture",
                                   class_intent="scratch", cleanup="delete", ttl_seconds=100,
                                   run_ref="run-1", now=lambda: 10)
    value.update(changes)
    value["lease_digest"] = canonical_digest(value, digest_field="lease_digest")
    return value


def folder(roots, value=None, raw=None):
    value = value or lease()
    path = roots[0] / value["lane"] / value["name"]
    path.mkdir(parents=True, exist_ok=True)
    (path / producer.LEASE_FILE).write_bytes(raw if raw is not None else json.dumps(value).encode())
    (path / "payload").write_bytes(b"tiny")
    return path


def observe(roots, **kwargs):
    return retention.observe_lane_scratch_retention((str(roots[0]),), pins_root=str(roots[1]),
        observed_at_epoch=50, enabled_requested=True, **kwargs)


def safe(result):
    assert result["status"] in {"report_only", "not_configured"}
    assert result["mutations"] == result["removed_bytes"] == 0
    assert result["candidate_bytes"] is None
    for key in ("execution_authorized", "apply_supported", "general_reference_inventory_complete",
                "queues_checked", "processes_checked", "consumer_fence_checked",
                "owner_approval_checked", "evidence_policy_checked", "restore_checked"):
        assert result[key] is False
    assert all(row["kept"] is True for row in result["rows"])
    assert all(row["bytes"] is None for row in result["retained_by_reason"].values())


@pytest.mark.parametrize("cleanup", ["delete", "offload", "owner_review"])
@pytest.mark.parametrize("state", ["live", "expired", "released"])
@pytest.mark.parametrize("reference", ["run_ref", "scene_ref"])
def test_valid_lease_states_and_intents_are_kept(roots, cleanup, state, reference):
    value = lease(cleanup=cleanup, expires_at_epoch=20 if state == "expired" else 110,
                  released_at_epoch=15 if state == "released" else None)
    if reference == "scene_ref":
        del value["run_ref"]
        value["scene_ref"] = "scene-1"
        value["lease_digest"] = canonical_digest(value, digest_field="lease_digest")
    path = folder(roots, value)
    result = observe(roots)
    safe(result)
    assert result["complete"] is True
    assert result["registered_count"] == result["observed_registered_count"] == 1
    row, = result["rows"]
    assert row["lease_status"] == state
    assert row["reference"] == {reference: value[reference]}
    assert row["logical_bytes"] == sum(p.stat().st_size for p in path.iterdir())
    assert row["pin_match"] == "no_pin_match_in_observed_ledger"
    assert "owner_approval_missing" in row["keep_reasons"]
    assert result["logical_bytes"] == row["logical_bytes"]


def test_valid_cache_and_renewal_producer_contract(roots):
    value = lease(class_intent="cache", size_budget_bytes=16, renewed_at_epoch=30,
                  expires_at_epoch=130)
    folder(roots, value)
    result = observe(roots)
    safe(result)
    assert result["complete"] and result["rows"][0]["class_intent"] == "cache"
    assert "live_lease" in result["rows"][0]["keep_reasons"]


@pytest.mark.parametrize("edit", [
    {"created_at_epoch": True}, {"expires_at_epoch": False}, {"renewed_at_epoch": True},
    {"released_at_epoch": True}, {"created_at_epoch": 10**400}, {"expires_at_epoch": float("inf")},
    {"schema_version": "foreign"}, {"lane": "foreign"}, {"owner": "bad/id"},
    {"size_budget_bytes": True}, {"class_intent": "cache", "size_budget_bytes": None},
    {"extra": "PRIVATE_MARKER"}, {"scene_ref": "scene-1"}, {"lease_digest": "sha256:" + "0" * 64},
])
def test_malformed_sealed_lease_is_unknown(roots, edit):
    value = lease()
    value.update(edit)
    if edit.keys() != {"lease_digest"}:
        try:
            value["lease_digest"] = canonical_digest(value, digest_field="lease_digest")
        except ValueError:
            pass
    path = folder(roots)
    (path / producer.LEASE_FILE).write_bytes(json.dumps(value).encode())
    result = observe(roots)
    safe(result)
    assert not result["complete"] and "lane_lease_invalid" in result["blockers"]
    assert result["registered_count"] is result["logical_bytes"] is result["allocated_bytes"] is None
    assert "PRIVATE_MARKER" not in json.dumps(result)


@pytest.mark.parametrize("raw", [b'{"owner":"one","owner":"two"}', b"[]", b"{", b"x" * 8193])
def test_bad_raw_lease_is_bounded_unknown(roots, raw):
    folder(roots, raw=raw)
    result = observe(roots)
    safe(result)
    assert not result["complete"]


def test_unregistered_folder_is_counted_without_descent(roots, monkeypatch):
    path = roots[0] / "lane-1" / "unregistered"
    path.mkdir(parents=True)
    (path / "not-observed").symlink_to("/private/foreign")
    result = observe(roots)
    safe(result)
    assert result["complete"]
    assert result["unregistered_count"] == 1 and result["registered_count"] == 0
    assert result["rows"] == [] and result["logical_bytes"] == 0


def test_missing_root_is_local_unknown_and_never_created(roots):
    roots[0].rmdir()
    result = observe(roots)
    safe(result)
    assert not result["complete"] and "lane_root_unavailable" in result["blockers"]
    assert not roots[0].exists()


def test_no_configured_roots_does_not_observe_host(roots, monkeypatch):
    monkeypatch.setattr(retention, "observe_storage_pins", lambda *a, **k: pytest.fail("no host scan"))
    result = retention.observe_lane_scratch_retention((), pins_root=str(roots[1]),
        observed_at_epoch=50, enabled_requested=False)
    safe(result)
    assert result["status"] == "not_configured" and not result["complete"]


def test_enabled_observation_uses_no_mutating_seam(roots, monkeypatch):
    folder(roots)
    opened = os.open
    def read_only(name, flags, *args, **kwargs):
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        return opened(name, flags, *args, **kwargs)
    monkeypatch.setattr(os, "open", read_only)
    for name in ("unlink", "mkdir", "rename", "replace", "rmdir", "write"):
        monkeypatch.setattr(os, name, lambda *a, **k: pytest.fail("mutation"))
    monkeypatch.setattr(producer, "list_lane_scratch", lambda *a, **k: pytest.fail("mutable reader"))
    result = observe(roots)
    safe(result)
    assert result["complete"] and not (roots[0] / ".lane-scratch.lock").exists()


@pytest.mark.parametrize("changes", [
    {"lane_roots": ["/mnt/blueprint-work/lanes"]},
    {"lane_roots": ("/foreign",)}, {"lane_roots": ("/mnt/blueprint-work/lanes/",)},
    {"lane_roots": ("/mnt/blueprint-work/lanes/../lanes",)},
    {"lane_roots": ("/mnt/blueprint-work/lanes",) * 3},
    {"enabled_requested": 1}, {"observed_at_epoch": True}, {"observed_at_epoch": 10**400},
    {"time_budget_seconds": 0}, {"time_budget_seconds": 11}, {"time_budget_seconds": True},
    {"pins_root": "/pins/../foreign"},
])
def test_parameters_fail_with_fixed_typed_error(roots, changes):
    arguments = dict(lane_roots=(str(roots[0]),), pins_root=str(roots[1]),
                     enabled_requested=False, observed_at_epoch=50)
    arguments.update(changes)
    with pytest.raises(retention.LaneScratchRetentionError, match="^lane_parameters_invalid$"):
        retention.observe_lane_scratch_retention(**arguments)
