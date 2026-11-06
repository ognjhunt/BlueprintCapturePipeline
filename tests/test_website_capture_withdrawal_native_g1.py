"""Withdrawal deletion of a site's G1 registration and G1 scene packets; hermetic, synthetic bytes only.

Beta journey step: site withdrawal after the screening service registered the site's G1 candidates.
Nothing here reaches a provider, a bucket or the host; the packets are a few synthetic JSON records and
placeholder bytes, never a capture, mesh, USD or image.
"""
# Covers: src/blueprint_pipeline/website_capture_withdrawal_native_g1.py
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from blueprint_pipeline import native_g1_site_registration as registration
from blueprint_pipeline import team_candidate_registry_reader as reader
from blueprint_pipeline import website_capture_withdrawal as withdrawal
from blueprint_pipeline import website_capture_withdrawal_deletion as deletion
from blueprint_pipeline import website_capture_withdrawal_native_g1 as native_g1
from blueprint_pipeline.team_scene_packet_worker import capture_key
from tests import test_website_capture_withdrawal_deletion as base

world = base.world  # the deletion tests' site, runs, B2 and roots
PLACEHOLDER = b"#usda 1.0\n# development_only placeholder, not a scene\n"


def _namespace(scene: Path, *, scene_id: str | None = "site-req1", source: str = "a" * 64) -> Path:
    """One half's ``native-g1/<base>/<source>/`` packet, its staged inputs and its lock."""
    namespace = scene / native_g1.NAMESPACE
    packet = namespace / ("b" * 64) / source
    if scene_id is not None:
        base._write(packet / "native_task_arena_packet_request.v1.json",
                    {"schema_version": "native_task_arena_packet_request.v1", "scene_id": scene_id})
        base._write(packet / "native_beta_g1_preparation.v1.json",
                    {"site_id": scene_id, "source": {"capture_id": "walkthrough-" + scene_id[5:]}})
    base._write(packet / "scene/collision.usda", PLACEHOLDER)
    base._write(namespace / ("b" * 64) / f".{source}.inputs/robot/robot.usd", PLACEHOLDER)
    base._write(namespace / ("b" * 64) / f".{source}.lock", b"")
    return namespace


def _half(root: Path, key: str, attempt: str) -> Path:
    """A compiled team half laid out as the team worker lands it: ``<key>/<attempt>/scene``."""
    scene = root / key / attempt / "scene"
    base._write(scene / "team_scene_inputs.v1.json", {"schema_version": "blueprint.team_scene_inputs.v1"})
    base._write(scene / "base/collision.usda", PLACEHOLDER)
    return scene


@pytest.fixture
def g1(world):
    rows = Path(world["resolution"]["native_g1_registration_root"])
    halves = Path(world["resolution"]["team_scene_halves_root"])
    files = [base._write(rows / name, {"development_only": name}) for name in (
        "site-req1.json", "site-req1.status.json", ".site-req1.json.k3x9")]
    others = [base._write(rows / name, {"development_only": name}) for name in (
        "site-req11.json", "site-req2.json", "site-req2.status.json")]
    key = capture_key(world["capture"].resolve())
    # Under the half the team worker keys by the site's capture: one complete packet, and one killed
    # preparation with no record naming anything.
    by_key = [_namespace(_half(halves, key, "1111111111111111-aaaa0001")),
              _namespace(_half(halves, key, "2222222222222222-aaaa0002"), scene_id=None)]
    # Under a key this capture path does not give (a supplement capture, a retired workspace): its records.
    by_records = _namespace(_half(halves, "c" * 32, "3333333333333333-aaaa0003"))
    theirs = _namespace(_half(halves, "d" * 32, "4444444444444444-aaaa0004"), scene_id="site-req2")
    return {"rows": rows, "halves": halves, "files": files, "others": others, "key": key,
            "by_key": by_key, "by_records": by_records, "theirs": theirs}


def test_withdrawal_deletes_the_sites_g1_rows_and_packets_and_then_its_halves(world, g1, tmp_path):
    from tests.test_website_capture_withdrawal_deletion import FakeGcs, _apply, _plan, _snapshot

    before = _snapshot(tmp_path)
    section = _plan(world, uncovered_roots=())["native_g1"]
    assert _snapshot(tmp_path) == before  # The plan reads only.
    gcs = FakeGcs()
    plan = _plan(world, cloud_storage=gcs, uncovered_roots=())
    assert plan["unresolved"] == [] and plan["native_g1"] == section
    assert [row[0] for row in section["registration_files"]] == sorted(map(str, g1["files"]))
    assert {row["directory"]: row["bound_by"] for row in section["packet_namespaces"]} == {
        **{str(path): "capture_key" for path in g1["by_key"]}, str(g1["by_records"]): "records"}
    # The halves around the packets are the team scene section's: each is checked whole once its packets go.
    assert [row["directory"] for row in plan["team_scene"]["halves"]] == sorted([
        str(g1["halves"] / g1["key"]), str(g1["halves"] / ("c" * 32))])
    receipt = _apply(world, plan, cloud_storage=gcs, uncovered_roots=())
    assert not any(path.exists() for path in [*g1["files"], *g1["by_key"], g1["by_records"]])
    # Another site's rows (also one whose id this site's id prefixes) and packets are untouched.
    assert all(path.exists() for path in g1["others"]) and (g1["theirs"] / ("b" * 64)).is_dir()
    assert not (g1["halves"] / g1["key"]).exists() and not (g1["halves"] / ("c" * 32)).exists()
    deleted = receipt["deleted"]["native_g1"]
    assert deleted["registration_files"] == sorted(map(str, g1["files"]))
    assert sorted(deleted["packet_namespaces"]) == sorted(map(str, [*g1["by_key"], g1["by_records"]]))
    assert deleted["packet_files_deleted"] == 5 + 3 + 5  # two complete packets and one killed preparation
    assert receipt["deleted"]["team_scene"]["halves"] == [row["directory"] for row in plan["team_scene"]["halves"]]
    assert "team_scene_halves" not in {row["holder"] for row in receipt["remaining"]}
    assert receipt["pipeline_controlled_deletion_complete"] is True


def test_the_rows_and_packet_a_real_registration_writes_are_the_ones_found(tmp_path, monkeypatch):
    from tests.test_native_g1_site_registration import SITE, door_world, overlay_path

    retained, kwargs, _, _ = door_world(tmp_path, monkeypatch)
    assert registration.ensure_site_registration(**kwargs)["status"] == "registered"
    rows = overlay_path().parent
    capture = tmp_path / f"pubsub/source-bucket/scenes/{SITE}/captures/walkthrough-{SITE[5:]}"
    base._write(capture / "raw/video.mp4", b"synthetic-video")
    withdrawal.acknowledge_withdrawal(capture_root=capture, command={
        "schema_version": "website_capture_withdrawal.v1", "request_id": SITE[5:], "scene_id": SITE,
        "capture_id": capture.name, "withdrawal_id": "withdrawal-development",
        "requested_at_iso": "2026-11-06T00:00:00Z"})
    site, identity = deletion._site_identity(capture)
    unresolved: list = []
    section = native_g1.plan_native_g1(identity, [], site=site, registration_root=rows,
                                       halves_root=tmp_path / "halves", unresolved=unresolved)
    assert unresolved == []
    assert [row[0] for row in section["registration_files"]] == [
        str(rows / f"{SITE}.json"), str(rows / f"{SITE}.status.json")]
    [found] = section["packet_namespaces"]
    row = json.loads(overlay_path().read_text())
    assert all(item["native_g1"]["scene_packet"]["path"].startswith(found["directory"] + "/")
               for item in row["screening_candidates"].values())
    native_g1.apply_native_g1(section)
    assert not overlay_path().exists() and not Path(found["directory"]).exists()
    assert reader.site_registration_overlay(SITE) is None and retained.registry_path.exists()


@pytest.mark.parametrize("fault,reason", [
    ("mixed_sites", "native_g1_names_another_site"),
    ("link_in_namespace", "unsafe_entry"),
    ("linked_namespace", "unsafe_entry"),
    ("linked_row", "unsafe_entry"),
])
def test_a_g1_copy_it_cannot_judge_fails_closed_before_any_deletion(world, g1, tmp_path, fault, reason):
    from tests.test_website_capture_withdrawal_deletion import _apply, _plan, _snapshot

    if fault == "mixed_sites":
        base._write(g1["by_key"][0] / "stray.json", {"scene_id": "site-req9"})
    elif fault == "link_in_namespace":
        os.symlink("/etc", g1["theirs"] / "escape")  # another site's namespace: it could still hide this one's
    elif fault == "linked_namespace":
        moved = tmp_path / "elsewhere"
        shutil.move(str(g1["by_records"]), moved)
        os.symlink(moved, g1["by_records"])
    else:
        g1["files"][0].unlink()
        os.symlink(g1["others"][0], g1["files"][0])
    plan = _plan(world)
    assert any(row["reason"].endswith(reason) for row in plan["unresolved"]), plan["unresolved"]
    before = _snapshot(tmp_path)
    with pytest.raises(deletion.WithdrawalDeletionError, match="unresolved_copies"):
        _apply(world, plan)
    assert _snapshot(tmp_path) == before


def test_a_g1_copy_changed_after_review_is_not_deleted(world, g1):
    from tests.test_website_capture_withdrawal_deletion import _plan

    section = _plan(world)["native_g1"]
    base._write(g1["by_records"] / "late.json", {"scene_id": "site-req1"})
    with pytest.raises(deletion.WithdrawalDeletionError, match="native_g1_changed"):
        native_g1.apply_native_g1(section)
    section = _plan(world)["native_g1"]
    base._write(g1["files"][1], {"development_only": "rewritten after review"})
    with pytest.raises(deletion.WithdrawalDeletionError, match="native_g1_registration_changed"):
        native_g1.apply_native_g1(section)
    assert g1["files"][1].exists()


def test_without_the_g1_roots_the_receipt_says_what_was_not_scanned(world):
    from tests.test_website_capture_withdrawal_deletion import FakeGcs, _apply, _plan

    gcs = FakeGcs()
    roots = {"native_g1_registration_root": None, "team_scene_halves_root": None, "uncovered_roots": ()}
    receipt = _apply(world, _plan(world, cloud_storage=gcs, **roots), cloud_storage=gcs, **roots)
    holders = {row["holder"] for row in receipt["remaining"]}
    assert {"native_g1_registrations_not_scanned", "native_g1_packets_not_scanned"} <= holders
    assert receipt["pipeline_controlled_deletion_complete"] is False
