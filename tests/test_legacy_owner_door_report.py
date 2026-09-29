# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_legacy_owner_door.py
"""Public legacy census contains only current review labels and a linked digest."""

import hashlib
import json
import stat
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from blueprint_pipeline import control_plane_lane_legacy_owner as legacy
from blueprint_pipeline import control_plane_lane_legacy_owner_door as door
from blueprint_pipeline import control_plane_lane_owner_consents as owners
from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget


def observed(*, status="complete", references=None):
    active = bool(references)
    return dict(status=status, scan_errors=[], observed_owner_count=0 if active else 1,
                rows=[dict(path="/mnt/blueprint-work/lanes/old", family="other",
                           allocated_bytes=4096, age_seconds=1000,
                           owner=None if active else "owner",
                           approved_expiry=None if active else 2000,
                           classification=None if active else "legacy_owner_review",
                           gc_eligible=False, references_clear=False,
                           references=[] if references is None else references,
                           unreadable=0, private_process_cmdline="secret", raw_env="secret")])


def test_public_projection_never_leaks_private_scan_inputs_or_action_authority():
    result = door._public_report(observed())
    raw = json.dumps(result)
    assert result["status"] == "complete" and result["observed_owner_count"] == 1
    assert "secret" not in raw and "raw_env" not in raw and "process" not in raw
    assert result["rows"][0]["owner"] == "owner"
    assert result["gc_eligible"] is result["references_clear"] is False
    assert result["candidate_bytes"] is result["eta_seconds"] is None


def test_incomplete_or_active_snapshot_publishes_no_owner_label():
    assert door._public_report(observed(status="incomplete"))["rows"] == []
    assert door._public_report(observed(status="incomplete"))["status"] == "incomplete"
    active = door._public_report(observed(references=["active_run"]))
    assert len(active["rows"]) == 1
    assert active["rows"][0]["path"] == "/mnt/blueprint-work/lanes/old"
    assert active["rows"][0]["owner"] is None
    assert active["rows"][0]["classification"] == "unclassified"
    assert active["rows"][0]["gc_eligible"] is False


def test_unregistered_top_level_and_revoked_label_remain_visible_keep_rows():
    source = observed()
    source["rows"][0].update(owner=None, approved_expiry=None, classification=None)
    source["rows"].append(dict(path="/mnt/blueprint-work/old-experiment",
                               family="other", allocated_bytes=4096, age_seconds=1000,
                               references=[], unreadable=0, owner=None,
                               approved_expiry=None, classification=None))
    source["observed_owner_count"] = 0
    report = door._public_report(source)
    assert report["status"] == "complete" and len(report["rows"]) == 2
    assert report["observed_owner_count"] == 0
    assert {row["path"] for row in report["rows"]} == {
        "/mnt/blueprint-work/lanes/old", "/mnt/blueprint-work/old-experiment"}
    assert all(row["classification"] == "unclassified" and row["owner"] is None
               and row["gc_eligible"] is False and row["references_clear"] is False
               for row in report["rows"])


def test_public_projection_is_bounded_even_with_many_reviewed_rows():
    source = observed()
    source["rows"] *= 1025
    source["observed_owner_count"] = 1025
    result = door._public_report(source)
    assert result["status"] == "incomplete" and result["rows"] == []


@pytest.fixture
def publication(tmp_path, monkeypatch):
    spool = tmp_path / "requests"
    results = spool / "results"
    results.mkdir(parents=True, mode=0o755)
    results.chmod(0o755)
    real_security = owners._security

    def public_fixture_security(info):
        value = real_security(info)
        if stat.S_ISDIR(info.st_mode):
            return value[:2] + (stat.S_IFDIR | 0o755, 0, 0)
        return value[:3] + (0, 0)

    def protected_fixture(info, *, directory=False, mode=None):
        assert stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
        assert mode is None or stat.S_IMODE(info.st_mode) == mode

    monkeypatch.setattr(owners, "_security", public_fixture_security)
    monkeypatch.setattr(owners, "_protected", protected_fixture)

    @contextmanager
    def installed(_path, monotonic):
        budget = ReferenceCollectionBudget(monotonic=monotonic, values_limit=10_000)
        files = owners._Files(budget)
        try:
            yield files, budget, SimpleNamespace(spool_root=str(spool)), None
            files.verify()
        finally:
            files.finish()
            budget.close()

    monkeypatch.setattr(legacy, "_installed_session", installed)
    monkeypatch.setattr(legacy, "observe_owner_review", lambda **_kwargs: observed())
    return results


def test_connected_publication_links_exact_result_and_keeps_private_fields_out(publication):
    request_id = "20260929T000000Z-legacy-owner-census-deadbeef"
    summary = door.publish_current(installed_config_path="/fixture/door.json",
                                   results_dir=str(publication), request_id=request_id, now=1000)
    result_path = publication / (request_id + ".legacy-owner-census.json")
    outcome_path = publication / (request_id + ".outcome.json")
    raw = result_path.read_bytes()
    assert json.loads(outcome_path.read_bytes()) == summary
    assert summary["result"] == str(result_path)
    assert summary["result_sha256"] == "sha256:" + hashlib.sha256(raw).hexdigest()
    assert summary["result_size_bytes"] == len(raw)
    assert result_path.stat().st_mode & 0o777 == 0o644
    assert outcome_path.stat().st_mode & 0o777 == 0o644
    assert b"secret" not in raw and b"raw_env" not in raw
    assert summary["gc_eligible"] is summary["references_clear"] is False


def test_bad_request_id_or_result_path_never_publishes(publication):
    with pytest.raises(legacy.LegacyOwnerError):
        door.publish_current(installed_config_path="/fixture/door.json",
                             results_dir=str(publication), request_id="../foreign", now=1000)
    with pytest.raises(legacy.LegacyOwnerError):
        door.publish_current(installed_config_path="/fixture/door.json",
                             results_dir=str(publication.parent),
                             request_id="20260929T000000Z-legacy-owner-census-deadbeef", now=1000)
    assert list(publication.iterdir()) == []
