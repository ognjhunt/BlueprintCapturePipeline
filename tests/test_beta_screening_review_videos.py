# Covers (for impacted-test selection):
#   src/blueprint_pipeline/beta_screening_review_videos.py
#   src/blueprint_pipeline/beta_screening_review_videos_http.py
#   src/blueprint_pipeline/beta_policy_screening_launch.py
#   src/blueprint_pipeline/beta_policy_screening_result.py
#   src/blueprint_pipeline/beta_policy_screening_coordinator.py
#   docs/schemas/beta_policy_screening_publication.v2.schema.json
"""A screening run's review videos are kept beside its capture, bounded and verified; keeping them never decides a run."""
from __future__ import annotations

import hashlib
import json
import stat
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import Request  # module level: the route test's dependency annotation must resolve

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_screening_review_videos as videos
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from tests.provider_output_fixtures import URL, RangeStore
from tests.test_beta_policy_screening_coordinator import Backend, native_result, plan, publication

RUN = "beta-" + "a" * 32
CAMERAS = ("external", "wrist", "overview")
PLENTY = SimpleNamespace(free=10**15)
ROOT = Path(__file__).resolve().parents[1]


def _mp4(milliseconds: int, filler: bytes, timescale: int = 1000) -> bytes:
    """A minimal MP4: ftyp, mdat, then moov holding a version-0 mvhd (a 1 kHz timescale by default)."""
    mvhd_body = bytes(4) + struct.pack(">IIII", 0, 0, timescale, milliseconds) + bytes(80)
    mvhd = struct.pack(">I4s", 8 + len(mvhd_body), b"mvhd") + mvhd_body
    return (struct.pack(">I4s", 16, b"ftyp") + b"isom" + bytes(4)
            + struct.pack(">I4s", 8 + len(filler), b"mdat") + filler
            + struct.pack(">I4s", 8 + len(mvhd), b"moov") + mvhd)


def _members(value) -> dict[str, bytes]:
    """Three cells' review videos, plus every member that must never be copied."""
    candidate = value["candidates"][0]
    cells = [cell["cell_id"] for cell in candidate["scenario_cells"]]
    members = {}
    for index, cell in enumerate(cells[:3]):
        episode = f"{RUN}--{cell}--{candidate['candidate_id']}"
        media = f"cell_runs/{index:02d}/episodes/media/{episode}"
        for camera_index, camera in enumerate(CAMERAS):
            members[f"{media}/{camera}.mp4"] = _mp4(2500 + index, f"{cell}:{camera}".encode() * 64)
            members[f"{media}--prestart-readiness/{camera}.mp4"] = _mp4(250, f"ready:{cell}:{camera}".encode())
            members[f"{media}/frames/{camera}/{camera_index:06d}.png"] = b"\x89PNG frame"
        members[f"cell_runs/{index:02d}/episodes/{episode}.depth.npy"] = b"\x93NUMPY"
        members[f"{media}/multicamera_frame_manifest.json"] = b"{}"
    members["cell_runs/00/scene/splat.ply"] = b"ply splat"
    members["cell_runs/00/scene/mesh.glb"] = b"glTF mesh"
    members[f"cell_runs/00/episodes/media/beta-{'b' * 32}--{cells[0]}--{candidate['candidate_id']}/external.mp4"] = b"x"
    members[f"cell_runs/00/episodes/media/{RUN}--{cells[0]}--{candidate['candidate_id']}/frames/wrist.mp4"] = b"y"
    members["native_task_arena_policy_canary_session_result.v1.json"] = b"{}"
    return members


def _zip(path: Path, members: dict[str, bytes]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path.read_bytes()


@pytest.fixture
def world(tmp_path):
    value = plan()
    candidate = value["candidates"][0]
    attempt = tmp_path / "work/attempts/screen-0123"
    lane = attempt / "allocator/attempts/attempt_001"
    capture = tmp_path / "storage/bucket/scenes/site-job-1/captures/walkthrough-job-1"
    (capture / "pipeline").mkdir(parents=True)
    members = _members(value)
    archive = _zip(lane / videos.LOCAL_ARCHIVE, members)
    return SimpleNamespace(plan=value, candidate=candidate, attempt=attempt, lane=lane, capture=capture,
                           members=members, archive=archive, storage=tmp_path / "storage",
                           directory=capture / "pipeline" / videos.DIRNAME / attempt.name)


def _retain(world, **options):
    options.setdefault("disk_usage", lambda _path: PLENTY)
    return videos.retain(plan=world.plan, candidate=world.candidate, prepared={"run_id": RUN},
                         adapter={"attempt_root": str(world.lane)}, attempt_dir=world.attempt,
                         capture_root=world.capture, **options)


def _rows(members):
    return [{"path": name, "kind": "file", "size": len(data), "compressed_size": len(data)}
            for name, data in members.items()]


def _select(world, members=None):
    return videos.select_review_videos(_rows(members or world.members), run_id=RUN,
                                       candidate_id=world.candidate["candidate_id"],
                                       cell_ids=[cell["cell_id"] for cell in world.candidate["scenario_cells"]])


def test_only_this_runs_mp4_review_videos_are_selected(world):
    rows, omitted = _select(world)
    selected = {row["member"]["path"] for row in rows}
    assert selected == {name for name in world.members if name.endswith(".mp4") and RUN in name
                        and "/frames/" not in name}
    assert len(selected) == 18
    assert not any(name.endswith((".png", ".npy", ".ply", ".glb", ".json")) for name in selected)
    # Another run's video and one nested below an episode's media directory are left out, and counted.
    assert omitted == {"unexpected_member_name": 2}
    # Run videos come before set-up checks, in plan order.
    assert [row["phase"] for row in rows] == ["run"] * 9 + ["setup_check"] * 9
    assert rows[0]["cell_id"] == world.candidate["scenario_cells"][0]["cell_id"]
    assert {row["camera"] for row in rows} == set(CAMERAS)
    assert rows[0]["relative_path"] == "videos/00-run-external.mp4"


def test_selection_is_bounded_in_count_bytes_and_per_video(world, monkeypatch):
    monkeypatch.setattr(videos, "MAX_VIDEOS", 4)
    rows, omitted = _select(world)
    assert len(rows) == 4 and {row["phase"] for row in rows} == {"run"}
    assert omitted["over_video_count_cap"] == 14
    monkeypatch.setattr(videos, "MAX_VIDEOS", 80)
    sizes = sorted(len(data) for name, data in world.members.items() if name.endswith(".mp4") and RUN in name)
    monkeypatch.setattr(videos, "MAX_TOTAL_BYTES", sum(sizes[-9:]))
    rows, omitted = _select(world)
    assert sum(row["member"]["size"] for row in rows) <= videos.MAX_TOTAL_BYTES
    assert omitted["over_total_byte_cap"] == 18 - len(rows) and len(rows) >= 9
    monkeypatch.setattr(videos, "MAX_TOTAL_BYTES", 1024**3)
    monkeypatch.setattr(videos, "MAX_VIDEO_BYTES", 1000)
    rows, omitted = _select(world)
    assert omitted["over_video_byte_cap"] == 9 and all(row["phase"] == "setup_check" for row in rows)


def test_a_download_run_keeps_verified_videos_beside_its_capture_with_a_manifest(world):
    reference = _retain(world)
    assert reference["status"] == "retained" and reference["blockers"] == []
    assert reference["video_count"] == 18
    manifest_path = world.directory / videos.MANIFEST_NAME
    manifest = videos.read_manifest(manifest_path)
    assert manifest is not None and reference["manifest_digest"] == manifest["manifest_digest"]
    assert reference["manifest_path"] == str(manifest_path)
    assert manifest["source"] == {"kind": "local_archive"} and manifest["attempt_id"] == "screen-0123"
    assert (manifest["site_id"], manifest["capture_id"]) == ("site-job-1", "walkthrough-job-1")
    for video in manifest["videos"]:
        data = world.members[video["source_member"]]
        copied = world.directory / video["relative_path"]
        assert copied.read_bytes() == data and video["size_bytes"] == len(data)
        assert video["sha256"] == "sha256:" + hashlib.sha256(data).hexdigest()
        assert stat.S_IMODE(copied.stat().st_mode) == 0o440
        assert video["episode_id"] == f"{RUN}--{video['cell_id']}--candidate-000"
        expected = 0.25 if video["phase"] == "setup_check" else 2.5 + int(video["cell_id"][-3:]) / 1000
        assert video["duration_seconds"] == pytest.approx(expected)
    on_disk = sorted(path.name for path in world.directory.rglob("*") if path.is_file())
    assert len(on_disk) == 19 and all(name.endswith(".mp4") for name in on_disk if name != videos.MANIFEST_NAME)
    assert reference["total_bytes"] == sum(video["size_bytes"] for video in manifest["videos"])
    # A second pass reuses the sealed manifest and copies nothing.
    before = manifest_path.stat().st_mtime_ns
    assert _retain(world) == reference and manifest_path.stat().st_mtime_ns == before


def _promoted_index(world, tmp_path):
    from blueprint_pipeline.provider_output_member_index import (
        LocalArchiveRangeSource, build_member_index, seal_durable_reference,
    )
    with LocalArchiveRangeSource(world.lane / videos.LOCAL_ARCHIVE, block_bytes=128 * 1024) as source:
        index = build_member_index(source, maximum_expanded_bytes=10**9)
    digest = index["archive"]["sha256"]
    reference = {"schema_version": "task_evaluation_scene_artifact_reference.v1", "status": "remote_verified",
                 "artifact_kind": "policy-canary-provider-output", "digest": digest,
                 "uri": f"s3://bucket/x/policy-canary-provider-output/sha256/{digest[7:]}/output.zip",
                 "size_bytes": index["archive"]["size"], "content_addressed_key": True,
                 "remote_identity_verified": True, "full_byte_service_account_readback_passed": True}
    return seal_durable_reference(index, reference)


def test_a_streamed_run_reads_each_video_once_by_range_and_checks_the_sealed_index(world, tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_configured_scene_object_store as object_store

    index = _promoted_index(world, tmp_path)
    (world.lane / videos.LOCAL_ARCHIVE).unlink()  # a streamed run has no local archive
    monkeypatch.setattr(videos, "_source", lambda _lane: ("promoted_archive", index))
    monkeypatch.setattr(object_store, "presign_configured_scene_artifact", lambda **_kw: URL)
    store = RangeStore(world.archive)
    reference = _retain(world, opener=store.opener)
    assert reference["status"] == "retained" and reference["video_count"] == 18
    by_path = {row["path"]: row for row in index["members"]}
    manifest = videos.read_manifest(world.directory / videos.MANIFEST_NAME)
    assert manifest["source"] == {"kind": "promoted_archive", "archive_sha256": index["archive"]["sha256"],
                                  "member_index_digest": index["index_digest"]}
    for video in manifest["videos"]:
        assert video["sha256"] == by_path[video["source_member"]]["sha256"]
    # One probe, then one range per video and never the whole archive.
    assert store.whole_object_gets() == 0 and len(store.ranges()) == 19
    assert manifest["transferred_bytes"] == 1 + sum(by_path[v["source_member"]]["compressed_size"]
                                                    for v in manifest["videos"])
    assert "X-Amz-Signature" not in json.dumps(manifest) and "https:" not in json.dumps(manifest)


def test_a_video_that_disagrees_with_the_index_is_left_out_and_the_rest_are_kept(world, tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_configured_scene_object_store as object_store

    index = _promoted_index(world, tmp_path)
    cell = world.candidate["scenario_cells"][0]["cell_id"]
    row = next(row for row in index["members"] if row["path"].endswith(f"{RUN}--{cell}--candidate-000/wrist.mp4"))
    row["sha256"] = "sha256:" + "0" * 64
    monkeypatch.setattr(videos, "_source", lambda _lane: ("promoted_archive", index))
    monkeypatch.setattr(object_store, "presign_configured_scene_artifact", lambda **_kw: URL)
    reference = _retain(world, opener=RangeStore(world.archive).opener)
    assert reference["status"] == "partial" and reference["video_count"] == 17
    assert "provider_output_member_digest_mismatch" in reference["blockers"]
    assert not list(world.directory.rglob("*.partial"))


@pytest.mark.parametrize("fault", ["raises", "no_lane", "no_source", "no_space", "transport"])
def test_a_failure_to_keep_videos_is_recorded_and_never_raises(world, monkeypatch, fault):
    options = {}
    if fault == "raises":
        monkeypatch.setattr(videos, "_retain", lambda *_a: (_ for _ in ()).throw(RuntimeError("secret /path")))
    elif fault == "no_lane":
        world.lane = world.lane.parent / "missing"
    elif fault == "no_source":
        (world.lane / videos.LOCAL_ARCHIVE).unlink()
    elif fault == "no_space":
        options["disk_usage"] = lambda _path: SimpleNamespace(free=1)
    else:
        from blueprint_pipeline import task_evaluation_configured_scene_object_store as object_store
        index = _promoted_index(world, world.attempt)
        monkeypatch.setattr(videos, "_source", lambda _lane: ("promoted_archive", index))
        monkeypatch.setattr(object_store, "presign_configured_scene_artifact", lambda **_kw: URL)
        store = RangeStore(world.archive)
        store.absent = True
        options["opener"] = store.opener
    reference = _retain(world, **options)
    assert reference["status"] == "not_retained" and reference["videos"] == [] and reference["blockers"]
    assert "secret" not in json.dumps(reference)
    assert not (world.directory / videos.MANIFEST_NAME).exists()
    assert not world.directory.exists() or not any(world.directory.rglob("*"))


def _completed_run(root, world, review_videos):
    coordinator.enqueue_screening_campaign(world.plan, queue_root=root)
    observation = {"status": "completed", "provider_allocation_performed": True, "provider_zero_verified": True,
                   "result": native_result(world.plan),
                   **({"review_videos": review_videos} if review_videos is not None else {})}
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=root, executor=Backend(observation))
    ledger = json.loads((root / coordinator.LEDGER_FILENAME).read_text())
    row = ledger["campaigns"][world.plan["campaign_id"]]["candidates"][world.candidate["candidate_id"]]
    return row, publication(root)


def test_a_failure_to_keep_videos_never_changes_the_verdict_settlement_or_cost(tmp_path, world):
    plain_row, plain = _completed_run(tmp_path / "plain", world, None)
    for name, failed in (("failed", videos.not_retained("beta_screening_review_videos_source_unavailable")),
                         ("garbage", {"garbage": True})):
        row, published = _completed_run(tmp_path / name, world, failed)
        assert row["status"] == plain_row["status"] == "completed"
        assert {key: value for key, value in row["summary"].items() if key != "review_videos"} == plain_row["summary"]
        assert row["attempt"]["reservation_status"] == plain_row["attempt"]["reservation_status"]
        assert published["candidates"] == plain["candidates"] and published["budget"] == plain["budget"]
        assert "review_videos" not in published
    assert row["summary"].get("review_videos") is None  # garbage is never recorded
    failed_row, _ = _completed_run(tmp_path / "recorded", world, videos.not_retained("x_failed"))
    assert failed_row["summary"]["review_videos"] == {"schema_version": videos.SCHEMA, "status": "not_retained",
                                                      "videos": [], "blockers": ["x_failed"]}


def test_the_manifest_is_referenced_from_the_summary_and_the_runs_are_published(tmp_path, world):
    from jsonschema import Draft202012Validator, FormatChecker

    reference = _retain(world)
    plain_row, plain = _completed_run(tmp_path / "plain", world, None)
    row, published = _completed_run(tmp_path / "videos", world, reference)
    summary = row["summary"]["review_videos"]
    assert summary["manifest_path"] == reference["manifest_path"]
    assert summary["manifest_digest"] == reference["manifest_digest"] and len(summary["videos"]) == 18
    # The counts, the estimate and the money are those of the same run without videos.
    assert published["candidates"] == plain["candidates"] and published["budget"] == plain["budget"]
    [entry] = published["review_videos"]
    assert entry["candidate_id"] == "candidate-000" and entry["manifest_digest"] == reference["manifest_digest"]
    outcomes = row["summary"]["episode_outcomes"]
    assert [run["cell_id"] for run in entry["practice_runs"]] == [outcome["cell_id"] for outcome in outcomes]
    assert [run["outcome"] for run in entry["practice_runs"]] == [outcome["outcome"] for outcome in outcomes]
    assert [len(run["videos"]) for run in entry["practice_runs"]] == [6, 6, 6] + [0] * 7
    text = json.dumps(published)
    assert str(world.capture) not in text and "manifest_path" not in text and "relative_path" not in text
    schema = json.loads((ROOT / "docs/schemas/beta_policy_screening_publication.v3.schema.json").read_text())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(published)
    assert published["publication_digest"] == cross_runtime_canonical_digest(published, digest_field="publication_digest")


def test_the_shared_wire_fixture_with_practice_run_videos_is_valid():
    from jsonschema import Draft202012Validator, FormatChecker

    value = json.loads((ROOT / "tests/fixtures/beta_policy_screening_review_videos_publication.v2.json").read_text())
    schema = json.loads((ROOT / "docs/schemas/beta_policy_screening_publication.v2.schema.json").read_text())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)
    assert value["publication_digest"] == cross_runtime_canonical_digest(value, digest_field="publication_digest")
    assert value["review_videos"][0]["practice_runs"][0]["videos"]


def _withdraw(capture: Path) -> None:
    marker = capture.parent.parent / "website_withdrawal/tombstone.json"
    marker.parent.mkdir(parents=True)
    value = {"schema_version": "website_capture_withdrawal_tombstone.v1", "scene_id": "site-job-1",
             "request_id": "job-1"}
    value["digest"] = canonical_digest(value, digest_field="digest")
    marker.write_text(json.dumps(value))


def test_only_a_sealed_manifest_video_of_a_current_capture_is_served(world):
    from blueprint_pipeline.website_atlas_preview_store import PreviewUnavailable

    reference = _retain(world)
    video = reference["videos"][0]
    stream, size = videos.open_video(request_id="job-1", capture_id="walkthrough-job-1", digest=video["sha256"],
                                     storage_root=world.storage)
    with stream:
        assert hashlib.sha256(stream.read()).hexdigest() == video["sha256"][7:] and size == video["size_bytes"]
    for request_id, capture_id, digest in (("job-2", "walkthrough-job-1", video["sha256"]),
                                           ("job-1", "walkthrough-job-1", "sha256:" + "1" * 64),
                                           ("job-1", "../walkthrough-job-1", video["sha256"])):
        with pytest.raises(PreviewUnavailable):
            videos.open_video(request_id=request_id, capture_id=capture_id, digest=digest, storage_root=world.storage)
    _withdraw(world.capture)
    with pytest.raises(PreviewUnavailable):
        videos.open_video(request_id="job-1", capture_id="walkthrough-job-1", digest=video["sha256"],
                          storage_root=world.storage)


def test_an_altered_manifest_serves_nothing(world):
    from blueprint_pipeline.website_atlas_preview_store import PreviewUnavailable

    reference = _retain(world)
    path = world.directory / videos.MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["videos"][0]["size_bytes"] += 1
    path.chmod(0o640)
    path.write_text(json.dumps(manifest))
    with pytest.raises(PreviewUnavailable):
        videos.open_video(request_id="job-1", capture_id="walkthrough-job-1", digest=reference["videos"][0]["sha256"],
                          storage_root=world.storage)


def test_the_signed_read_serves_inline_uncached_video_only_to_the_webapp(world, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from blueprint_pipeline import beta_screening_review_videos_http as http

    def admission(request: Request):
        request.state.intake_client_id = request.headers.get("x-test-client", "")
    app = FastAPI()
    http.register_beta_screening_review_video_routes(app, require_admission=admission)
    video = _retain(world)["videos"][0]
    monkeypatch.setenv(http.HANDOFF_STORAGE_ROOT_ENV, str(world.storage))
    client = TestClient(app)
    url = (f"/api/live-pipeline/beta-screening-review-videos/{video['sha256'][7:]}"
           "?request_id=job-1&capture_id=walkthrough-job-1")
    signed = {"x-blueprint-pipeline-signature": "sha256=x", "x-test-client": "blueprint-webapp"}
    response = client.get(url, headers=signed)
    assert response.status_code == 200 and len(response.content) == video["size_bytes"]
    assert hashlib.sha256(response.content).hexdigest() == video["sha256"][7:]
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["content-disposition"] == "inline"
    assert client.get(url, headers={**signed, "x-test-client": "robot-team"}).status_code == 403
    assert client.get(url, headers={"x-test-client": "blueprint-webapp"}).status_code == 403
    assert client.get(url.replace("job-1&", "job-2&"), headers=signed).status_code == 404
    assert client.get(url.replace(video["sha256"][7:], "f" * 64), headers=signed).status_code == 404


def test_mp4_duration_reads_both_header_versions_and_refuses_anything_else(tmp_path):
    path = tmp_path / "v.mp4"
    path.write_bytes(_mp4(1234, b"x"))
    assert videos.mp4_duration_seconds(path) == 1.234
    body = bytes([1, 0, 0, 0]) + struct.pack(">QQIQ", 0, 0, 600, 1800) + bytes(80)
    mvhd = struct.pack(">I4s", 8 + len(body), b"mvhd") + body
    path.write_bytes(struct.pack(">I4s", 8 + len(mvhd), b"moov") + mvhd)
    assert videos.mp4_duration_seconds(path) == 3.0
    for data in (b"", b"not an mp4 at all", struct.pack(">I4s", 4, b"moov")):
        path.write_bytes(data)
        assert videos.mp4_duration_seconds(path) is None
    # A header stating more than a day is not trusted: the website refuses such a duration.
    path.write_bytes(_mp4(100_000, b"x", timescale=1))
    assert videos.mp4_duration_seconds(path) is None
    path.write_bytes(_mp4(86_400, b"x", timescale=1))
    assert videos.mp4_duration_seconds(path) == 86_400


def test_an_out_of_range_duration_never_reaches_the_ledger_or_the_publication(world):
    reference = _retain(world)
    for bad in (100_000.0, -1.0, float("nan"), True):
        tampered = json.loads(json.dumps(reference))
        tampered["videos"][0]["duration_seconds"] = bad
        assert videos.valid_reference(tampered) is None
    summary = {"review_videos": json.loads(json.dumps(reference)), "episode_outcomes": [
        {"cell_id": cell["cell_id"], "outcome": "success"} for cell in world.candidate["scenario_cells"]]}
    summary["review_videos"]["videos"][0]["duration_seconds"] = 100_000.0
    row = {"candidate_id": "candidate-000", "status": "completed", "episodes_planned": 10, "episodes_succeeded": 10,
           "episodes_failed": 0, "episodes_void": 0}
    entry = videos.publication_entry(row, summary)
    assert entry is not None and entry["practice_runs"][0]["videos"][0]["duration_seconds"] is None
    assert videos.publication_entry_valid(entry, row)


@pytest.mark.parametrize("change", ["cell_id", "counts", "camera", "size", "outcome", "no_videos"])
def test_an_entry_the_website_would_refuse_is_left_out_and_the_result_still_published(world, change):
    reference = _retain(world)
    outcomes = [{"cell_id": cell["cell_id"], "outcome": "success"} for cell in world.candidate["scenario_cells"]]
    row = {"candidate_id": "candidate-000", "status": "completed", "episodes_planned": 10, "episodes_succeeded": 10,
           "episodes_failed": 0, "episodes_void": 0}
    summary = {"review_videos": json.loads(json.dumps(reference)), "episode_outcomes": outcomes}
    assert videos.publication_entry(row, summary) is not None
    if change == "cell_id":
        outcomes[0]["cell_id"] = "not an identifier"
    elif change == "counts":
        row["episodes_succeeded"], row["episodes_failed"] = 9, 1
    elif change == "camera":
        summary["review_videos"]["videos"][0]["camera"] = "Side View"
    elif change == "size":
        summary["review_videos"]["videos"][0]["size_bytes"] = videos.MAX_VIDEO_BYTES + 1
    elif change == "outcome":
        outcomes[0]["outcome"] = "maybe"
    else:
        summary["review_videos"]["videos"] = [dict(video, cell_id="elsewhere") for video in reference["videos"]]
    assert videos.publication_entry(row, summary) is None


def test_production_acceptance_reads_a_publication_with_practice_run_videos():
    from blueprint_pipeline.production_autonomy_acceptance import AcceptanceEvidenceError, _publication_contract

    fixture = json.loads((ROOT / "tests/fixtures/beta_policy_screening_review_videos_publication.v2.json").read_text())
    assert _publication_contract(fixture)["schema_version"] == "beta_policy_screening_publication.v2"

    def sealed(value):
        return {**value, "publication_digest": cross_runtime_canonical_digest(value, digest_field="publication_digest")}
    broken = json.loads(json.dumps(fixture))
    broken["review_videos"][0]["practice_runs"][0]["outcome"] = "failure"
    with pytest.raises(AcceptanceEvidenceError, match="publication_contract_invalid"):
        _publication_contract(sealed(broken))
    retained = {**json.loads(json.dumps(fixture)), "schema_version": "beta_policy_screening_publication.v1",
                "purpose": "screening", "claim_ceiling": "diagnostic_policy_execution", "recommendation_eligible": False}
    with pytest.raises(AcceptanceEvidenceError, match="publication_contract_invalid"):
        _publication_contract(sealed(retained))


def test_a_pass_waits_for_withdrawal_deletion_and_never_recreates_a_removed_capture(world, monkeypatch):
    import fcntl

    base = world.capture / "pipeline" / videos.DIRNAME
    base.mkdir()
    monkeypatch.setattr(videos, "LOCK_WAIT_SECONDS", 0)
    with (base / videos.LOCK_NAME).open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert _retain(world)["blockers"] == ["beta_screening_review_videos_capture_busy"]
    assert not world.directory.exists()
    import shutil
    shutil.rmtree(world.capture / "pipeline")
    reference = _retain(world)
    assert reference["blockers"] == [videos.CHANGED] and not (world.capture / "pipeline").exists()


def test_a_capture_withdrawn_mid_copy_loses_what_was_copied_and_gets_no_manifest(world, monkeypatch):
    copy = videos._copy

    def withdraw_after_first(*args, **kwargs):
        record = copy(*args, **kwargs)
        if not (world.capture.parent.parent / "website_withdrawal").exists():
            _withdraw(world.capture)
        return record
    monkeypatch.setattr(videos, "_copy", withdraw_after_first)
    assert _retain(world)["blockers"] == [videos.CHANGED]
    assert not world.directory.exists()


def test_a_capture_replaced_mid_copy_is_never_written_into(world, monkeypatch):
    copy, moved = videos._copy, world.capture.with_name("walkthrough-job-1.old")

    def replace_after_first(*args, **kwargs):
        record = copy(*args, **kwargs)
        if not moved.exists():
            world.capture.rename(moved)
            (world.capture / "pipeline").mkdir(parents=True)
        return record
    monkeypatch.setattr(videos, "_copy", replace_after_first)
    assert _retain(world)["blockers"] == [videos.CHANGED]
    assert list((world.capture / "pipeline").iterdir()) == []


def test_every_streamed_write_keeps_the_free_space_floor(world, tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_configured_scene_object_store as object_store

    index = _promoted_index(world, tmp_path)
    monkeypatch.setattr(videos, "_source", lambda _lane: ("promoted_archive", index))
    monkeypatch.setattr(object_store, "presign_configured_scene_artifact", lambda **_kw: URL)
    calls = []

    def filling(_path):
        calls.append(1)
        return PLENTY if len(calls) == 1 else SimpleNamespace(free=1)
    reference = _retain(world, opener=RangeStore(world.archive).opener, disk_usage=filling)
    assert reference["status"] == "not_retained" and videos.NO_SPACE in reference["blockers"]
    assert len(calls) > 1 and not world.directory.exists()


def test_a_video_is_found_however_many_attempts_the_capture_has(world):
    reference = _retain(world)
    for index in range(70):
        (world.capture / "pipeline" / videos.DIRNAME / f"a{index:03d}").mkdir()
    video = reference["videos"][0]
    stream, _size = videos.open_video(request_id="job-1", capture_id="walkthrough-job-1", digest=video["sha256"],
                                      storage_root=world.storage, manifest_digest=reference["manifest_digest"])
    stream.close()
    from blueprint_pipeline.website_atlas_preview_store import PreviewUnavailable
    with pytest.raises(PreviewUnavailable):
        videos.open_video(request_id="job-1", capture_id="walkthrough-job-1", digest=video["sha256"],
                          storage_root=world.storage, manifest_digest="sha256:" + "0" * 64)


def test_native_g1_runs_videos_are_selected_from_its_own_layout():
    """The native G1 runtime keeps cells/<cell_id>/episode/media/<candidate_id>/<camera>.mp4 (2026-11-08)."""
    candidate, cells = "humanoidarena_dp_g1_dex3_sonic_vision_navi", ["screening.site.000", "screening.site.001"]
    paths = [f"cells/{cell}/episode/media/{candidate}/{camera}.mp4" for cell in cells for camera in ("head", "overview")]
    paths += [f"cells/{cells[0]}/episode/media/{candidate}/frames/head_0000.png",
              f"cells/{cells[0]}/episode/media/other_candidate/head.mp4",
              f"cells/screening.site.009/episode/media/{candidate}/head.mp4"]
    members = [{"path": path, "kind": "file", "size": 1000, "compressed_size": 900} for path in paths]

    rows, omitted = videos.select_review_videos(members, run_id=RUN, candidate_id=candidate, cell_ids=cells)

    assert [row["relative_path"] for row in rows] == [
        "videos/00-run-head.mp4", "videos/00-run-overview.mp4", "videos/01-run-head.mp4", "videos/01-run-overview.mp4"]
    assert [row["cell_id"] for row in rows] == [cells[0], cells[0], cells[1], cells[1]]
    assert {row["phase"] for row in rows} == {"run"} and {row["camera"] for row in rows} == {"head", "overview"}
    assert all(row["episode_id"] == f"{RUN}--{row['cell_id']}--{candidate}" for row in rows)
    assert omitted == {"unexpected_member_name": 2}
