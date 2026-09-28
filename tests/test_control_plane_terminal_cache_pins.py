"""Archived evidence releases reproducible caches while live references win."""
import json
import os

import pytest

from blueprint_pipeline.control_plane_storage_pins import write_storage_pin, load_storage_pins
from blueprint_pipeline.control_plane_terminal_cache_pins import reconcile_terminal_cache_pins
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def setup_case(tmp_path):
    pins = tmp_path / "storage-pins"
    for kind, owner, deps in [("preparation", "prep", []), ("compilation", "prep", [{"kind":"preparation", "owner_id":"prep"}]),
        ("activation", "closed", [{"kind":"compilation", "owner_id":"prep"}, {"kind":"preparation", "owner_id":"prep"}])]:
        cache = tmp_path / "cache" / kind / owner
        cache.mkdir(parents=True)
        (cache / "payload.bin").write_bytes(b"preserved until the separate cache collector runs")
        write_storage_pin(pins_root=pins, kind=kind, owner_id=owner, paths=[cache], depends_on=deps, now=lambda: 0)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    pointer = {"schema_version":"control_plane_evidence_offload_pointer.v1", "status":"offloaded", "directory":"closed",
        "evidence_deleted":False, "terminal_receipt":"dispatch_receipt.json", "size_bytes":1024,
        "digest":"sha256:"+"a"*64, "uri":"s3://blueprint-task-evaluation-artifacts-prod/retained/evidence.tar"}
    pointer["pointer_digest"] = canonical_digest(pointer, digest_field="pointer_digest")
    path = evidence / "closed.offloaded.v1.json"
    path.write_text(json.dumps(pointer))
    return dict(pins_root=pins, queue_roots=[tmp_path / "queue"], evidence_roots=[evidence], now=30_000,
                reference_checker=lambda _:False, classifier=lambda *args, **kwargs:None), path


def test_archived_run_releases_dependency_pins_without_removing_evidence_or_bytes(tmp_path):
    args, path = setup_case(tmp_path)
    dry = reconcile_terminal_cache_pins(**args)
    assert len(dry["candidates"]) == 1 and not dry["released"]
    assert all(p["status"] == "live" for p in load_storage_pins(args["pins_root"], now=lambda:30_000))
    result = reconcile_terminal_cache_pins(**args, apply=True)
    assert len(result["released"][0]["released"]) == 3
    assert path.is_file() and len(list((tmp_path / "cache").rglob("payload.bin"))) == 3
    assert not reconcile_terminal_cache_pins(**args, apply=True)["released"]


def test_archived_website_launch_releases_its_activation_pin(tmp_path):
    owner = "website-example-20260920t204229z-activation-auto"
    cache = tmp_path / "cache" / owner
    cache.mkdir(parents=True)
    (cache / "payload.bin").write_bytes(b"reproducible input")
    pins = tmp_path / "storage-pins"
    write_storage_pin(pins_root=pins, kind="activation", owner_id=owner,
                      paths=[cache], now=lambda: 0)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    launch_id = owner + "-launch"
    pointer = {"schema_version": "control_plane_evidence_offload_pointer.v1",
               "status": "offloaded", "directory": launch_id, "evidence_deleted": False,
               "terminal_receipt": "launch_receipt.json", "size_bytes": 1024,
               "digest": "sha256:" + "a" * 64,
               "uri": "s3://blueprint-task-evaluation-artifacts-prod/retained/evidence.tar"}
    pointer["pointer_digest"] = canonical_digest(pointer, digest_field="pointer_digest")
    (evidence / (launch_id + ".offloaded.v1.json")).write_text(json.dumps(pointer))
    args = dict(pins_root=pins, queue_roots=[tmp_path / "queue"], evidence_roots=[evidence],
                now=30_000, reference_checker=lambda _: False,
                classifier=lambda *args, **kwargs: None)
    assert [row["owner_id"] for row in reconcile_terminal_cache_pins(**args)["candidates"]] == [owner]
    assert len(reconcile_terminal_cache_pins(**args, apply=True)["released"]) == 1
    assert cache.exists()


@pytest.mark.parametrize("reason", ["queue", "process", "tampered", "restored"])
def test_archived_run_pin_stays_when_reference_or_evidence_is_unsafe(tmp_path, reason):
    args, path = setup_case(tmp_path)
    if reason == "queue":
        pending = tmp_path / "queue/pending"
        pending.mkdir(parents=True)
        (pending / "active.json").write_text(json.dumps({"preparation_id":"prep"}))
    elif reason == "process":
        args["reference_checker"] = lambda _:True
    elif reason == "tampered":
        value = json.loads(path.read_text())
        value["size_bytes"] += 1
        path.write_text(json.dumps(value))
    else:
        (tmp_path / "evidence/closed").mkdir()
    result = reconcile_terminal_cache_pins(**args, apply=True)
    assert not result["released"]
    assert all(p["status"] == "live" for p in load_storage_pins(args["pins_root"], now=lambda:30_000))


def test_another_active_run_keeps_shared_inputs_pinned(tmp_path):
    args, _ = setup_case(tmp_path)
    write_storage_pin(pins_root=args["pins_root"], kind="activation", owner_id="active", paths=[tmp_path / "active"],
        depends_on=[{"kind":"compilation", "owner_id":"prep"}], now=lambda: 0)
    result = reconcile_terminal_cache_pins(**args, apply=True)
    assert result["released"][0]["released"] == [{"kind":"activation", "owner_id":"closed"}]
    states={(p["kind"],p["owner_id"]):p["status"] for p in load_storage_pins(args["pins_root"], now=lambda:30_000)}
    assert states[("compilation","prep")] == states[("preparation","prep")] == "live"


def _sealed_run(tmp_path, name, *, idle_seconds, now=30_000_000, receipt=True, registry=False):
    evidence = tmp_path / "evidence"
    run = evidence / name
    (run / "allocator").mkdir(parents=True)
    (run / "allocator" / "result.json").write_text("{}")
    if receipt:
        (run / "launch_receipt.json").write_text(json.dumps({"status": "blocked"}))
    if registry:
        (run / "artifacts/result_delivery").mkdir(parents=True)
        (run / "artifacts/result_delivery/artifact_registry.json").write_text("{}")
    stamp = now - idle_seconds
    for path in [run, run / "allocator", *run.rglob("*")]:
        os.utime(path, (stamp, stamp))
    return evidence, run


def _pin_on_run(tmp_path, run, *, created=30_000_000 - 7 * 86400):
    pins = tmp_path / "storage-pins"
    write_storage_pin(pins_root=pins, kind="activation", owner_id=run.name, paths=[run], now=lambda: created)
    return dict(pins_root=pins, queue_roots=[tmp_path / "queue"], evidence_roots=[run.parent], now=30_000_000,
                reference_checker=lambda _: False, classifier=lambda *args, **kwargs: None,
                hot_window_seconds=2 * 86400)


def test_a_sealed_cold_run_releases_its_own_activation_pin_so_it_can_be_offloaded(tmp_path):
    """A launch that ended blocked without releasing its pin must not hold its evidence forever.

    The collector refuses to offload a pinned run and used to release the pin
    only once the run was archived, which could never happen.
    """
    evidence, run = _sealed_run(tmp_path, "blocked-run", idle_seconds=5 * 86400)
    args = _pin_on_run(tmp_path, run)
    dry = reconcile_terminal_cache_pins(**args)
    assert [c["proof"]["kind"] for c in dry["candidates"]] == ["sealed_cold_run"]
    assert dry["candidates"][0]["proof"]["terminal_receipt"] == "launch_receipt.json"
    assert not dry["released"]
    result = reconcile_terminal_cache_pins(**args, apply=True)
    assert result["released"][0]["released"] == [{"kind": "activation", "owner_id": "blocked-run"}]
    assert run.is_dir() and (run / "launch_receipt.json").is_file()
    assert result["cache_or_evidence_bytes_removed"] is False


@pytest.mark.parametrize("reason", ["hot", "no_receipt", "result_registry", "queue", "touched"])
def test_a_run_still_in_use_keeps_its_activation_pin(tmp_path, reason):
    evidence, run = _sealed_run(
        tmp_path, "run-x", idle_seconds=3600 if reason == "hot" else 5 * 86400,
        receipt=reason != "no_receipt", registry=reason == "result_registry")
    args = _pin_on_run(tmp_path, run)
    if reason == "queue":
        pending = tmp_path / "queue/pending"
        pending.mkdir(parents=True)
        (pending / "active.json").write_text(json.dumps({"launch_id": "run-x"}))
    if reason == "touched":
        original = reconcile_terminal_cache_pins
        def touching(**kwargs):
            (run / "allocator" / "progress.json").write_text("{}")
            return original(**kwargs)
        dry = original(**args)
        assert len(dry["candidates"]) == 1
        result = touching(**args, apply=True)
    else:
        result = reconcile_terminal_cache_pins(**args, apply=True)
    assert not result["released"]
    assert all(p["status"] == "live" for p in load_storage_pins(args["pins_root"], now=lambda: 30_000_000))
