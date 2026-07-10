"""Hermetic tests for the durable cross-run machine quarantine registry."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from blueprint_pipeline import machine_quarantine_registry as Q


_KEY = {
    "provider": "runpod",
    "machine_id": "machine-abc",
    "image_digest": "sha256:" + "a" * 64,
    "isaac_version": "6.0.0",
}


def _record(tmp_path, **overrides):
    kwargs = {
        **_KEY,
        "failure_class": "driver_incompatible",
        "phase": Q.PHASE_RUNTIME_CANARY,
        "registry_dir": tmp_path,
        "now_epoch": 1_000_000.0,
    }
    kwargs.update(overrides)
    return Q.record_machine_quarantine(**kwargs)


def test_record_and_find_active_entry(tmp_path):
    entry = _record(tmp_path, gpu_name="NVIDIA L40S", driver_version="570.124.06")
    assert entry["schema_version"] == Q.SCHEMA_VERSION
    assert entry["attempt_count"] == 1
    assert entry["provider_exclusion_supported"] is False
    assert entry["raw_provider_payload_recorded"] is False

    found = Q.find_active_quarantine(
        **_KEY, registry_dir=tmp_path, now_epoch=1_000_100.0
    )
    assert found is not None
    assert found["failure_class"] == "driver_incompatible"
    assert found["driver_version"] == "570.124.06"


def test_repeat_observation_increments_attempt_count(tmp_path):
    _record(tmp_path)
    entry = _record(tmp_path, now_epoch=1_000_500.0)
    assert entry["attempt_count"] == 2
    assert entry["first_observed_epoch"] == 1_000_000.0
    assert entry["last_observed_epoch"] == 1_000_500.0


def test_ttl_expiry_stops_matching(tmp_path):
    _record(tmp_path, ttl_seconds=120)
    expired_now = 1_000_000.0 + 121
    assert (
        Q.find_active_quarantine(**_KEY, registry_dir=tmp_path, now_epoch=expired_now)
        is None
    )
    # Still visible when explicitly asking for expired entries.
    entries = Q.load_quarantine_entries(
        registry_dir=tmp_path, include_expired=True, now_epoch=expired_now
    )
    assert len(entries) == 1 and entries[0]["expired"] is True


def test_expired_entry_restarts_attempt_count_on_rerecord(tmp_path):
    _record(tmp_path, ttl_seconds=60)
    entry = _record(tmp_path, now_epoch=1_000_000.0 + 3600)
    assert entry["attempt_count"] == 1
    assert entry["first_observed_epoch"] == 1_000_000.0 + 3600


def test_image_digest_change_gets_fresh_chance(tmp_path):
    _record(tmp_path)
    other_digest = dict(_KEY, image_digest="sha256:" + "b" * 64)
    assert (
        Q.find_active_quarantine(
            **other_digest, registry_dir=tmp_path, now_epoch=1_000_100.0
        )
        is None
    )


def test_isaac_version_change_gets_fresh_chance(tmp_path):
    _record(tmp_path)
    other = dict(_KEY, isaac_version="7.0.0")
    assert (
        Q.find_active_quarantine(**other, registry_dir=tmp_path, now_epoch=1_000_100.0)
        is None
    )


def test_same_machine_different_failure_class_gets_own_entry(tmp_path):
    _record(tmp_path, failure_class="driver_incompatible")
    _record(tmp_path, failure_class="no_runtime", phase=Q.PHASE_PRE_RUNTIME)
    entries = Q.load_quarantine_entries(registry_dir=tmp_path, now_epoch=1_000_100.0)
    classes = sorted(e["failure_class"] for e in entries)
    assert classes == ["driver_incompatible", "no_runtime"]
    assert all(e["attempt_count"] == 1 for e in entries)


@pytest.mark.parametrize(
    "failure_class",
    [
        "placement_gate_failed",
        "policy_rollout_failed",
        "kitchen_assets_missing",
        "task_validation_failed",
        "stance_rejected",
        "scene_load_failed",
    ],
)
def test_non_machine_failure_classes_are_refused(tmp_path, failure_class):
    with pytest.raises(Q.QuarantineRefused):
        _record(tmp_path, failure_class=failure_class)
    assert Q.load_quarantine_entries(registry_dir=tmp_path) == []


def test_invalid_phase_refused(tmp_path):
    with pytest.raises(Q.QuarantineRefused):
        _record(tmp_path, phase="kitchen_validation")


def test_secretlike_values_are_refused(tmp_path):
    with pytest.raises(Q.QuarantineRefused):
        _record(tmp_path, driver_version="Bearer abc123secret")
    with pytest.raises(Q.QuarantineRefused):
        _record(tmp_path, gpu_name="https://x?X-Amz-Signature=abc")


def test_corrupted_registry_file_is_skipped_not_fatal(tmp_path):
    _record(tmp_path)
    (tmp_path / "quarantine-corrupted.json").write_text("{not json", encoding="utf-8")
    entries = Q.load_quarantine_entries(registry_dir=tmp_path, now_epoch=1_000_100.0)
    assert len(entries) == 1
    health = Q.registry_health(registry_dir=tmp_path, now_epoch=1_000_100.0)
    assert health["active_count"] == 1
    assert len(health["corrupted_files"]) == 1


def test_concurrent_writers_do_not_corrupt_entry(tmp_path):
    def write(i: int):
        return _record(tmp_path, now_epoch=1_000_000.0 + i)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(write, range(16)))
    assert all(r["schema_version"] == Q.SCHEMA_VERSION for r in results)
    entries = Q.load_quarantine_entries(registry_dir=tmp_path, now_epoch=1_000_100.0)
    assert len(entries) == 1
    # Every update was serialized under the lock: no lost increments.
    assert entries[0]["attempt_count"] == 16


def test_no_raw_payload_or_secret_keys_persisted(tmp_path):
    entry = _record(tmp_path)
    text = json.dumps(json.loads(open(entry["path"], encoding="utf-8").read())).lower()
    for marker in ("api_key", "authorization", "x-amz", "raw_response"):
        assert marker not in text


def test_evidence_paths_recorded_with_checksums(tmp_path):
    evidence = tmp_path / "preflight.json"
    evidence.write_text('{"status": "blocked"}', encoding="utf-8")
    entry = _record(tmp_path, evidence_paths=[evidence])
    record = entry["evidence"][0]
    assert record["path"] == str(evidence)
    assert record["sha256"] and record["bytes"] > 0


def test_purge_expired_removes_only_expired(tmp_path):
    _record(tmp_path, ttl_seconds=60)
    _record(tmp_path, failure_class="no_runtime", phase=Q.PHASE_PRE_RUNTIME,
            ttl_seconds=100_000)
    report = Q.purge_expired(registry_dir=tmp_path, now_epoch=1_000_000.0 + 3600)
    assert len(report["removed"]) == 1
    remaining = Q.load_quarantine_entries(
        registry_dir=tmp_path, now_epoch=1_000_000.0 + 3600
    )
    assert len(remaining) == 1 and remaining[0]["failure_class"] == "no_runtime"


def test_cli_list_and_purge(tmp_path, capsys):
    import time

    _record(tmp_path, now_epoch=time.time())
    assert Q.main(["list", "--registry-dir", str(tmp_path)]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert len(listed) == 1
    assert Q.main(["purge-expired", "--registry-dir", str(tmp_path)]) == 0
