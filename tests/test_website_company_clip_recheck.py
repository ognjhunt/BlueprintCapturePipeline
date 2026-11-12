"""The website's push when a job's permission changes: the next pass reads the live rights again.

Hermetic like ``test_website_company_clip``: temporary capture roots, real local ffmpeg on synthetic
clips, fakes for the WebApp, the store and the bucket.
"""
# Covers: src/blueprint_pipeline/permission_recheck.py
# Covers: src/blueprint_pipeline/website_company_clip.py
# Covers: src/blueprint_pipeline/website_atlas_preview_http.py
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi import Request

from blueprint_pipeline import permission_recheck
from blueprint_pipeline import website_company_clip as clip
from tests.test_website_company_clip import (
    NOW, OLDER, SHARED, World, _Bucket, _capture, _manifest, _owned_list, _rights, _sha, _withdraw,
)

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                reason="needs local ffmpeg and ffprobe")


@pytest.fixture(autouse=True)
def _route_clock(monkeypatch):
    monkeypatch.setattr(clip, "_wall_clock", lambda: NOW + 60)
SIGNED = {"x-blueprint-pipeline-signature": "sha256=x", "x-test-client": "blueprint-webapp"}
REFUSED = ValueError("website_control_company-clip_http_409:website_company_clip_not_permitted")


def _client(storage_root: Path | None, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from blueprint_pipeline import website_atlas_preview_http as http

    def admission(request: Request):
        request.state.intake_client_id = request.headers.get("x-test-client", "")
    app = FastAPI()
    http.register_website_atlas_preview_routes(app, require_admission=admission)
    if storage_root is None:
        monkeypatch.delenv(http.HANDOFF_STORAGE_ROOT_ENV, raising=False)
    else:
        monkeypatch.setenv(http.HANDOFF_STORAGE_ROOT_ENV, str(storage_root))
    return TestClient(app)


def _recheck(client, capture_id: str = "walkthrough-req1", request_id: str = "req1", headers=SIGNED):
    return client.post(f"/api/live-pipeline/website-company-clips/{capture_id}/recheck?request_id={request_id}",
                       headers=headers)


def _owned_row(capture: Path) -> dict:
    return {"capture_id": "walkthrough-req1", "approved_by": "owner", "approved_on": "2026-11-12", "evidence": "t",
            "video_sha256_prefix": _sha(capture / "raw/walkthrough.mp4")[7:19]}


def test_a_recheck_rereads_the_rights_and_keeps_a_refusal_while_they_are_unchanged(tmp_path, monkeypatch):
    # A website that keeps asking for a recheck while it keeps refusing does not set the pace of uploads:
    # each recheck makes the pass read the live rights again, and unchanged rights keep the refusal.
    capture = _capture(tmp_path)
    world = World(delivery_error=REFUSED)
    assert world.publish(capture)["status"] == "refused"
    assert len(world.archived) == 1 and len(world.retracted) == 1
    clock = {"now": NOW + 60}
    monkeypatch.setattr(permission_recheck.time, "time", lambda: clock["now"])
    client = _client(tmp_path / "storage", monkeypatch)
    for minute in range(1, 5):
        clock["now"] = NOW + 61 * minute
        reads = world.rights_reads
        response = _recheck(client)
        assert response.status_code == 202
        assert response.json() == {"schema_version": "website_company_recheck.v1", "accepted": True}
        assert world.rights_reads == reads  # the request itself only marks: no read, encode, upload or delete
        assert world.publish(capture, now=lambda: clock["now"] + 1)["status"] == "refused"
        assert world.rights_reads == reads + 1
    assert len(world.archived) == 1 and len(world.retracted) == 1
    mark = capture.joinpath(*clip.DIRECTORY) / permission_recheck.MARK_FILE
    assert json.loads(mark.read_text())["requested_at_epoch"] == NOW + 61 * 4


def test_the_reference_capture_owned_then_refused_then_v4_after_a_recheck(tmp_path, monkeypatch):
    # The owner's reference job: shared as Blueprint's own, which the website does not accept; then the
    # owner accepts the sharing statement on the job page and the website asks for a recheck.
    capture = _capture(tmp_path, version=OLDER)
    monkeypatch.setattr(clip, "OWNED_CAPTURES_LIST", _owned_list(tmp_path, [_owned_row(capture)]))
    world = World(rights=_rights(OLDER), delivery_error=REFUSED)
    assert world.publish(capture)["status"] == "refused"
    assert _manifest(capture)["refused_by_webapp"] and world.objects == {}
    world.rights, world.delivery_error = _rights(SHARED), None
    monkeypatch.setattr(permission_recheck.time, "time", lambda: NOW + 30)
    assert _recheck(_client(tmp_path / "storage", monkeypatch)).status_code == 202
    assert world.publish(capture, now=lambda: NOW + 40)["status"] == "published"
    assert world.delivered[-1]["permission_basis"] == "site_sharing_statement"
    assert world.delivered[-1]["statement_version"] == SHARED and len(world.objects) == 1


def test_a_recheck_cuts_short_a_definite_no_within_the_window(tmp_path):
    # An older job that said no at its last live read is not asked again for 6 hours, unless the website
    # says its permission changed.
    capture = _capture(tmp_path, version=OLDER)
    world = World(rights=_rights(OLDER))
    assert world.publish(capture)["status"] == "not_permitted"
    world.rights = _rights(SHARED)
    assert world.publish(capture, now=lambda: NOW + 60)["status"] == "not_permitted"
    assert world.rights_reads == 1
    assert permission_recheck.request_recheck(capture, now=NOW + 70) == "accepted"
    assert world.publish(capture, now=lambda: NOW + 80)["status"] == "published"


def test_the_issuer_unknown_captures_and_a_second_ask_within_a_minute(tmp_path, monkeypatch):
    _capture(tmp_path)
    client = _client(tmp_path / "storage", monkeypatch)
    for headers in ({"x-test-client": "blueprint-webapp"}, {**SIGNED, "x-test-client": "robot-team"}):
        response = _recheck(client, headers=headers)
        assert response.status_code == 403
        assert response.json()["detail"] == "website company clip issuer not authorized"
    for capture_id, request_id in (("walkthrough-req2", "req2"), ("walkthrough-req1", "req2"),
                                   ("walkthrough-req1", ""), ("supplement-xyz", "req1"), ("walkthrough-req1x", "req1"),
                                   ("walkthrough-*", "*")):
        response = _recheck(client, capture_id, request_id)
        assert response.status_code == 404 and response.json() == {"detail": "website_company_clip_unavailable"}
    clock = {"now": NOW}
    monkeypatch.setattr(permission_recheck.time, "time", lambda: clock["now"])
    assert _recheck(client).status_code == 202
    clock["now"] = NOW + 59
    too_soon = _recheck(client)
    assert too_soon.status_code == 429 and too_soon.headers["retry-after"] == "60"
    assert set(too_soon.json()) == {"detail"}
    clock["now"] = NOW + 60
    assert _recheck(client).status_code == 202
    # The limit is per capture: another capture of the same job is not held back.
    _capture(tmp_path, capture_id="supplement-" + "a" * 40)
    assert _recheck(client, "supplement-" + "a" * 40, "req1").status_code == 202
    # No storage root configured: nothing can be found, so nothing is marked.
    assert _recheck(_client(None, monkeypatch)).status_code == 404


def test_a_recheck_for_a_withdrawn_capture_changes_nothing_a_company_can_get(tmp_path, monkeypatch):
    capture = _capture(tmp_path)
    world = World()
    assert world.publish(capture)["status"] == "published"
    digest = _manifest(capture)["archive"]["digest"]
    _withdraw(capture)
    monkeypatch.setattr(permission_recheck.time, "time", lambda: NOW + 30)
    assert _recheck(_client(tmp_path / "storage", monkeypatch)).status_code == 202
    assert world.publish(capture, now=lambda: NOW + 40) == {"status": "withdrawn"}
    assert len(world.archived) == 1 and len(world.delivered) == 1
    with pytest.raises(clip.CompanyClipUnavailable):
        clip.open_company_clip(request_id="req1", capture_id="walkthrough-req1", digest=digest,
                               storage_root=capture.parents[4], client=_Bucket(world.objects), bucket="b")


def test_the_mark_is_one_small_file_and_never_raises(tmp_path):
    capture = _capture(tmp_path)
    folder = capture.joinpath(*clip.DIRECTORY)
    assert not folder.exists()
    assert permission_recheck.request_recheck(capture, now=NOW) == "accepted"
    assert [path.name for path in folder.iterdir()] == [permission_recheck.MARK_FILE]
    assert (folder / permission_recheck.MARK_FILE).stat().st_size < 100
    assert permission_recheck.recheck_requested_since(capture, NOW) is True
    assert permission_recheck.recheck_requested_since(capture, NOW + 1) is False
    # A capture never handed off, a linked folder or a garbled mark: no mark, no error.
    assert permission_recheck.request_recheck(tmp_path / "nowhere", now=NOW) == "unavailable"
    (folder / permission_recheck.MARK_FILE).write_text("{not json")
    assert permission_recheck.requested_at(capture) is None
    assert permission_recheck.request_recheck(capture, now=NOW + 1) == "accepted"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    shutil.rmtree(folder)
    folder.symlink_to(elsewhere)
    assert permission_recheck.request_recheck(capture, now=NOW + 120) == "unavailable"
    assert list(elsewhere.iterdir()) == []
