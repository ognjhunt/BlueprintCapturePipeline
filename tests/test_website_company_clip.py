"""The silent company clip: made from the normalized clip, archived, delivered and read by range.

Hermetic: temporary capture roots, real local ffmpeg/ffprobe on synthetic clips, and fakes for the
WebApp rights read, the private artifact store, the WebApp acceptance and the bucket. No provider,
bucket or host is reached, and nothing is paid for.
"""
# Covers: src/blueprint_pipeline/website_company_clip.py
# Covers: src/blueprint_pipeline/sharing_statement.py
from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi import Request

from blueprint_pipeline import website_company_clip as clip
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.sharing_statement import SHARING_STATEMENT_VERSIONS, covers_sharing

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                reason="needs local ffmpeg and ffprobe")
SHARED, OLDER = "2026-11-12.v4", "2026-11-11.v3"
NOW = 1_794_500_000


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def _video(path: Path, *, size: str = "1920x1080", seconds: int = 2, rate: int = 30) -> Path:
    """A synthetic phone-like clip with an audio track and a location tag."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", str(seconds),
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-metadata", "location=+48.8584+002.2945/", "-metadata", "title=Back office",
                    "-metadata", "comment=Shot by the manager", "-metadata", "make=Apple", "-metadata", "model=iPhone 17",
                    "-metadata", "creation_time=2026-11-12T09:00:00Z",
                    "-metadata:s:v:0", "handler_name=Core Media Video",
                    "-metadata:s:v:0", "com.apple.quicktime.location.ISO6709=+48.8584+002.2945/",
                    "-metadata:s:v:0", "creation_time=2026-11-12T09:00:00Z", str(path)],
                   check=True, capture_output=True)
    return path


def _audio_streams(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "csv=p=0", str(path)], check=True, capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines() if line.strip())


def _rights(version: str | None = SHARED, *, consent: str = "granted", revoked: bool = False) -> dict:
    return {"derived_scene_generation_allowed": consent == "granted", "data_licensing_allowed": False,
            "capture_contributor_payout_eligible": False, "consent_status": consent, "consent_revoked": revoked,
            "consent_scope": ["derived_scene_generation", "robot_evaluation"], "statement_version": version,
            "recorded_at_iso": "2026-11-12T10:00:00Z"}


def _capture(tmp_path: Path, *, version: str = SHARED, size: str = "1920x1080", capture_id: str = "walkthrough-req1",
             normalized: bool = True):
    """A website capture as the host holds it; ``normalized=False`` is one handed off before normalization."""
    from blueprint_pipeline.website_source_video import preparation_source_video
    root = tmp_path / f"storage/source-bucket/scenes/site-req1/captures/{capture_id}"
    _write(root / "raw/manifest.json", {"site_submission_id": "req1", "scene_id": "site-req1",
                                        "capture_id": capture_id, "capture_rights": _rights(version),
                                        "video_uri": f"scenes/site-req1/captures/{capture_id}/raw/walkthrough.mp4"})
    raw = _video(root / "raw/walkthrough.mp4", size=size)
    if normalized:
        preparation_source_video(capture_root=root, raw_video=raw)  # writes normalization.json
    else:
        (root / "pipeline/clean_plate").mkdir(parents=True)  # a job that started before normalization existed
    return root


@pytest.fixture
def capture(tmp_path):
    return _capture(tmp_path)


def _withdraw(capture: Path) -> None:
    value = {"schema_version": "website_capture_withdrawal_tombstone.v1", "scene_id": "site-req1",
             "request_id": "req1", "consent_revoked_at": "2026-11-12T12:00:00Z"}
    _write(capture.parents[1] / "website_withdrawal/tombstone.json",
           {**value, "digest": canonical_digest(value, digest_field="digest")})


class World:
    """The live rights read, private store and WebApp acceptance a pass talks to."""

    def __init__(self, *, rights=None, delivery_error: Exception | None = None, on_archive=None):
        self.rights = _rights() if rights is None else rights
        self.delivery_error, self.on_archive = delivery_error, on_archive
        self.archived, self.objects, self.delivered, self.retracted, self.rights_reads = [], {}, [], [], 0

    def check_rights(self, identity):
        self.rights_reads += 1
        if isinstance(self.rights, Exception):
            raise self.rights
        return dict(self.rights)

    def archive(self, path, *, identity):
        digest, name = _sha(path), clip.clip_filename(identity["capture_id"], identity["request_id"])
        self.archived.append(path)
        self.objects[f"{_prefix()}/website-company-clip/sha256/{digest[7:]}/{name}"] = path.read_bytes()
        if self.on_archive:
            self.on_archive()
        return {"uri": f"s3://bucket/x/{digest[7:]}/{name}", "digest": digest, "size_bytes": path.stat().st_size,
                "filename": name}

    def retract(self, uri):
        self.retracted.append(uri)
        self.objects.clear()

    def deliver(self, record, *, identity):
        if self.delivery_error:
            raise self.delivery_error
        self.delivered.append(record)

    def publish(self, capture, **options):
        options = {"now": lambda: NOW, **options}
        return clip.publish_company_clip(capture, check_rights=self.check_rights, archive=self.archive,
                                         retract=self.retract, deliver=self.deliver, **options)


def _prefix() -> str:
    from blueprint_pipeline.task_evaluation_configured_scene_object_keys import LARGE_ARTIFACT_KEY_PREFIX
    return LARGE_ARTIFACT_KEY_PREFIX


def _manifest(capture: Path) -> dict:
    return json.loads((capture.joinpath(*clip.DIRECTORY) / clip.MANIFEST_FILE).read_text())


def test_the_sharing_statement_is_matched_exactly():
    assert SHARING_STATEMENT_VERSIONS == ("2026-11-12.v4",)
    assert covers_sharing("2026-11-12.v4")
    assert not any(map(covers_sharing, (OLDER, "2026-11-12.v4 ", "2026-11-12.V4", None, ["2026-11-12.v4"])))


def test_the_clip_has_no_audio_no_tags_is_seekable_and_is_never_the_upload(tmp_path):
    # A 2560 px upload is normalized to 1920 px with its sound kept; the company clip drops it.
    capture = _capture(tmp_path, size="2560x1440")
    record = json.loads((capture / "pipeline/source_video/normalization.json").read_text())
    normalized = capture / record["preparation_video"]
    assert record["outcome"] == "normalized" and _audio_streams(normalized) == 1
    world = World()
    assert world.publish(capture)["status"] == "published"
    made = capture.joinpath(*clip.DIRECTORY) / clip.CLIP_FILE
    assert _audio_streams(made) == 0
    streams = clip.probe_streams(made)
    assert [row["codec_type"] for row in streams["streams"]] == ["video"]
    assert streams["streams"][0]["codec_name"] == "h264"
    assert (streams["streams"][0]["width"], streams["streams"][0]["height"]) == (1280, 720)
    assert clip.moov_before_mdat(made)
    assert clip.metadata_clean(streams)
    tags = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json",
                           str(made)], check=True, capture_output=True, text=True).stdout
    for planted in ("location", "48.8584", "Back office", "manager", "Apple", "iPhone", "creation_time",
                    "Core Media", "quicktime", "title", "comment", "make", "model"):
        assert planted not in tags, planted
    sent = world.delivered[0]
    assert sent["has_audio"] is False and sent["content_type"] == "video/mp4" and sent["width"] == 1280
    assert sent["digest"] not in {_sha(capture / "raw/walkthrough.mp4"), _sha(normalized)}
    assert sent["permission_basis"] == "site_sharing_statement" and sent["statement_version"] == SHARED
    assert sent["clip_digest"] == cross_runtime_canonical_digest(
        {key: value for key, value in sent.items() if key != "clip_digest"})
    assert world.archived == [made]


def test_a_clip_carrying_any_source_tag_is_refused_and_retried(capture, monkeypatch):
    raw = capture / "raw/walkthrough.mp4"
    planted = clip.probe_streams(raw)
    assert not clip.metadata_clean(planted)  # the upload itself carries location, make and model
    assert not clip.metadata_clean({"format_tags": {"encoder": "Lavf60.16.100", "location": "+1+2/"}, "streams": []})
    assert not clip.metadata_clean({"format_tags": {}, "streams": [{"tags": {"handler_name": "Core Media Video"}}]})
    real = clip._encode

    def leaky(source, target, **kwargs):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-map", "0:v:0", "-an",
                        "-vf", f"scale={kwargs['width']}:{kwargs['height']}", "-c:v", "libx264",
                        "-preset", "ultrafast", "-movflags", "+faststart", "-f", "mp4", str(target)],
                       check=True, capture_output=True)  # keeps the source's tags
    monkeypatch.setattr(clip, "_encode", leaky)
    world = World()
    assert world.publish(capture) == {"status": "failed", "reason": "website_company_clip_output_metadata_not_clean"}
    assert world.archived == [] and not (capture.joinpath(*clip.DIRECTORY) / clip.CLIP_FILE).exists()
    monkeypatch.setattr(clip, "_encode", real)
    assert world.publish(capture)["status"] == "published"


def test_an_upload_already_within_bounds_is_still_re_encoded_never_served_as_is(capture):
    raw = capture / "raw/walkthrough.mp4"
    record = json.loads((capture / "pipeline/source_video/normalization.json").read_text())
    assert record["outcome"] == "within_bounds" and record["preparation_video"] == "raw/walkthrough.mp4"
    world = World()
    assert world.publish(capture)["status"] == "published"
    assert world.delivered[0]["digest"] != _sha(raw)
    assert all(body != raw.read_bytes() for body in world.objects.values())
    assert _audio_streams(raw) == 1 and _audio_streams(capture.joinpath(*clip.DIRECTORY) / clip.CLIP_FILE) == 0


def test_published_once_then_current_without_rebuilding_or_resending(capture):
    world = World()
    first = world.publish(capture)
    assert first["status"] == "published"
    assert world.publish(capture) == {"status": "current", "clip_revision": first["clip_revision"]}
    assert len(world.archived) == 1 and len(world.delivered) == 1


def test_an_older_statement_gets_nothing_and_no_request(tmp_path):
    capture = _capture(tmp_path, version=OLDER)
    world = World(rights=_rights(OLDER))
    # The live statement is read once (it may have changed on the job page) and recorded.
    assert world.publish(capture) == {"status": "not_permitted"}
    assert world.rights_reads == 1 and world.archived == [] and world.delivered == []
    directory = capture.joinpath(*clip.DIRECTORY)
    assert not (directory / clip.CLIP_FILE).exists() and not (directory / clip.MANIFEST_FILE).exists()
    # Within the recheck window nothing is read or written again.
    assert world.publish(capture, now=lambda: NOW + 60) == {"status": "not_permitted"}
    assert world.rights_reads == 1
    assert world.publish(capture, now=lambda: NOW + clip.LIVE_STATEMENT_RECHECK_SECONDS) == {"status": "not_permitted"}
    assert world.rights_reads == 2 and world.archived == [] and world.delivered == []


def test_an_unanswered_live_read_for_an_older_statement_is_not_repeated_every_pass(tmp_path):
    capture = _capture(tmp_path, version=OLDER)
    world = World(rights=ValueError("website_control_task-context_http_503:unclassified"))
    assert world.publish(capture)["status"] == "refused"
    assert world.publish(capture, now=lambda: NOW + 60) == {"status": "not_permitted"}
    assert world.rights_reads == 1


def test_a_site_that_accepts_the_sharing_statement_later_gets_its_clip(tmp_path):
    # The capture's files keep the submission snapshot (older); the live read has the new consent.
    capture = _capture(tmp_path, version=OLDER)
    world = World(rights=_rights(SHARED))
    assert world.publish(capture)["status"] == "published"
    assert world.delivered[0]["statement_version"] == SHARED


def test_a_live_older_statement_takes_back_a_clip_made_under_the_sharing_one(capture):
    world = World()
    assert world.publish(capture)["status"] == "published"
    # The next rebuild re-reads the rights; the live statement no longer covers sharing.
    (capture.joinpath(*clip.DIRECTORY) / clip.CLIP_FILE).unlink()
    world.rights = _rights(OLDER)
    assert world.publish(capture) == {"status": "not_permitted"}
    assert len(world.retracted) == 1 and _manifest(capture)["archive"]["retracted"] is True


def _owned_list(tmp_path: Path, rows) -> Path:
    return _write(tmp_path / "owned.json", {"schema_version": "blueprint_owned_captures.v1", "purpose": "test",
                                            "captures": rows})


def test_blueprint_owned_captures_on_the_allow_list_get_a_clip(tmp_path, monkeypatch):
    capture = _capture(tmp_path, version=OLDER)
    digest = _sha(capture / "raw/walkthrough.mp4")
    monkeypatch.setattr(clip, "OWNED_CAPTURES_LIST", _owned_list(tmp_path, [
        {"capture_id": "walkthrough-req1", "approved_by": "owner", "approved_on": "2026-11-12", "evidence": "test",
         "video_sha256_prefix": digest[7:19]}]))
    world = World(rights=_rights(OLDER))
    assert world.publish(capture)["status"] == "published"
    assert world.delivered[0]["permission_basis"] == "blueprint_owned_capture"
    assert world.delivered[0]["statement_version"] == OLDER


def test_an_owned_row_whose_video_digest_does_not_match_is_not_owned(tmp_path, monkeypatch):
    capture = _capture(tmp_path, version=OLDER)
    monkeypatch.setattr(clip, "OWNED_CAPTURES_LIST", _owned_list(tmp_path, [
        {"capture_id": "walkthrough-req1", "video_sha256_prefix": "000000000000"}]))
    world = World(rights=_rights(OLDER))
    assert world.publish(capture) == {"status": "not_permitted"} and world.archived == []


def test_the_sharing_basis_reads_a_strict_allow_list(tmp_path):
    from blueprint_pipeline import sharing_statement as sharing
    digest = "sha256:" + "ab" * 32
    context = {"capture_id": "walkthrough-x", "capture_rights": {"statement_version": OLDER}}
    shared = {"capture_id": "walkthrough-x", "capture_rights": {"statement_version": SHARED}}
    good = _owned_list(tmp_path, [{"capture_id": "walkthrough-x", "approved_by": "owner", "approved_on": "2026-11-12",
                                   "evidence": "e", "video_sha256_prefix": "ab" * 6}])
    assert sharing.sharing_basis(shared, owned_list=good) == {"permission_basis": "site_sharing_statement",
                                                               "statement_version": SHARED}
    assert sharing.sharing_basis(context, video_sha256=digest, owned_list=good) == {
        "permission_basis": "blueprint_owned_capture", "capture_id": "walkthrough-x"}
    assert sharing.sharing_basis(context, video_sha256=None, owned_list=good) is None
    assert sharing.sharing_basis(context, video_sha256="sha256:" + "cd" * 32, owned_list=good) is None
    assert sharing.sharing_basis({**context, "capture_id": "walkthrough-y"}, video_sha256=digest, owned_list=good) is None
    no_prefix = _owned_list(tmp_path, [{"capture_id": "walkthrough-x"}])
    assert sharing.sharing_basis(context, owned_list=no_prefix)["permission_basis"] == "blueprint_owned_capture"
    for bad in ([{"capture_id": "walkthrough-x", "role": "admin"}], [{"capture_id": 7}], ["walkthrough-x"],
                [{"capture_id": "walkthrough-x"}, {"capture_id": "walkthrough-x"}],
                [{"capture_id": "walkthrough-x", "video_sha256_prefix": "XYZ"}], {"capture_id": "walkthrough-x"},
                # As strict as the packet tool was: a prefix of 12 to 64 hex digits, never null; text where
                # text is expected; a real approval date.
                [{"capture_id": "walkthrough-x", "video_sha256_prefix": "abab"}],
                [{"capture_id": "walkthrough-x", "video_sha256_prefix": None}],
                [{"capture_id": "walkthrough-x", "video_sha256_prefix": "ab" * 33}],
                [{"capture_id": "walkthrough-x", "evidence": " "}], [{"capture_id": "walkthrough-x", "approved_by": 7}],
                [{"capture_id": "walkthrough-x", "approved_on": "yesterday"}]):
        path = _owned_list(tmp_path, bad)
        assert sharing.owned_captures(path) == ({}, "blueprint_owned_capture_list_invalid")
        assert sharing.sharing_basis(context, video_sha256=digest, owned_list=path) is None
    (tmp_path / "broken.json").write_text("{not json")
    assert sharing.owned_captures(tmp_path / "broken.json")[1] == "blueprint_owned_capture_list_invalid"
    assert sharing.owned_captures(tmp_path / "missing.json")[1] == "blueprint_owned_capture_list_invalid"
    (tmp_path / "dupe.json").write_text('{"schema_version": "blueprint_owned_captures.v1", "captures": [], "captures": []}')
    assert sharing.owned_captures(tmp_path / "dupe.json")[1] == "blueprint_owned_capture_list_invalid"


def test_the_shipped_allow_list_reads_cleanly():
    from blueprint_pipeline import sharing_statement as sharing
    owned, flag = sharing.owned_captures()
    assert flag is None and "walkthrough-req1" not in owned


def test_revoked_consent_or_an_unreadable_rights_read_is_a_refusal(capture):
    for rights in (_rights(consent="revoked", revoked=True), ValueError("website_task_context_digest_mismatch")):
        world = World(rights=rights)
        result = world.publish(capture)
        assert result["status"] == "refused" and world.archived == [] and world.delivered == []


def test_a_withdrawn_capture_makes_nothing(capture):
    _withdraw(capture)
    world = World()
    assert world.publish(capture) == {"status": "withdrawn"}
    assert world.archived == [] and not capture.joinpath(*clip.DIRECTORY).exists()


def test_a_withdrawal_during_the_upload_takes_the_copy_back(capture):
    world = World(on_archive=lambda: _withdraw(capture))
    assert world.publish(capture) == {"status": "withdrawn"}
    assert len(world.retracted) == 1 and world.delivered == []


def test_an_open_takedown_refuses_and_takes_the_copy_back(capture):
    world = World()
    assert world.publish(capture)["status"] == "published"
    _write(capture / "pipeline/consent_takedown/takedown_manifest.json", {"takedown_open": True})
    assert world.publish(capture)["status"] == "refused"
    assert len(world.retracted) == 1 and _manifest(capture)["archive"]["retracted"] is True


def test_a_webapp_rights_refusal_takes_the_copy_back(capture):
    world = World(delivery_error=ValueError("website_control_company-clip_http_409:website_company_clip_not_permitted"))
    assert world.publish(capture)["status"] == "refused"
    assert len(world.retracted) == 1


def test_a_normalization_record_that_cannot_be_used_never_falls_back_to_the_raw_upload(capture):
    (capture / "pipeline/source_video/normalization.json").write_text("{not json")
    world = World()
    assert world.publish(capture) == {"status": "no_source"}
    assert world.rights_reads == 0 and not capture.joinpath(*clip.DIRECTORY).exists()


def _planted_tags(path: Path) -> str:
    return subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json",
                           str(path)], check=True, capture_output=True, text=True).stdout


def test_a_capture_handed_off_before_normalization_gets_a_silent_clean_clip_from_its_raw_upload(tmp_path, monkeypatch):
    capture = _capture(tmp_path, size="2560x1440", normalized=False)
    raw = capture / "raw/walkthrough.mp4"
    before = raw.read_bytes()
    assert not (capture / "pipeline/source_video").exists() and _audio_streams(raw) == 1
    assert "48.8584" in _planted_tags(raw) and "Back office" in _planted_tags(raw)
    world = World()
    assert world.publish(capture)["status"] == "published"
    made = capture.joinpath(*clip.DIRECTORY) / clip.CLIP_FILE
    streams = clip.probe_streams(made)
    assert _audio_streams(made) == 0 and [row["codec_type"] for row in streams["streams"]] == ["video"]
    assert (streams["streams"][0]["width"], streams["streams"][0]["height"]) == (1280, 720)
    assert clip.metadata_clean(streams) and clip.moov_before_mdat(made)
    tags = _planted_tags(made)
    for planted in ("location", "48.8584", "Back office", "manager", "Apple", "iPhone", "creation_time",
                    "Core Media", "quicktime", "title", "comment", "make", "model"):
        assert planted not in tags, planted
    manifest = _manifest(capture)
    assert manifest["source"] == "raw_upload"
    assert manifest["source_video"]["relative"] == "raw/walkthrough.mp4" and manifest["source_video"]["digest"] == _sha(raw)
    # The raw upload itself is never served, and it is left as it was.
    sent = world.delivered[0]
    assert sent["digest"] != _sha(raw) and sent["has_audio"] is False and "source" not in sent
    assert all(body != before for body in world.objects.values()) and raw.read_bytes() == before
    # A current clip does not read the upload again on every pass.
    monkeypatch.setattr("blueprint_pipeline.website_source_video.probe_video",
                        lambda path: pytest.fail("probed a raw upload already made into a clip"))
    monkeypatch.setattr(clip, "_sha256_file", lambda path: pytest.fail("hashed a raw upload already made into a clip"))
    assert world.publish(capture)["status"] == "current" and len(world.archived) == 1


@pytest.mark.parametrize("case", ["missing", "broken", "empty", "not_named", "outside", "symlink", "not_a_video_name",
                                  "never_handed_off"])
def test_a_missing_or_broken_raw_upload_is_no_source_and_raises_nothing(tmp_path, case):
    capture = _capture(tmp_path, normalized=False)
    raw, manifest = capture / "raw/walkthrough.mp4", capture / "raw/manifest.json"
    value = json.loads(manifest.read_text())
    if case == "missing":
        raw.unlink()
    elif case == "broken":
        raw.write_bytes(b"\x00\x00\x00\x18ftypmp42 not really a video" * 64)
    elif case == "empty":
        raw.write_bytes(b"")
    elif case == "not_named":
        _write(manifest, {key: item for key, item in value.items() if key != "video_uri"})
    elif case == "outside":
        _write(manifest, {**value, "video_uri": "scenes/site-req1/captures/walkthrough-req1/raw/../../x.mp4"})
    elif case == "symlink":
        outside = _video(tmp_path / "elsewhere/clip.mp4")
        raw.unlink()
        raw.symlink_to(outside)
    elif case == "never_handed_off":
        shutil.rmtree(capture / "pipeline")
    else:
        raw.rename(capture / "raw/walkthrough.json")
        _write(manifest, {**value, "video_uri": "scenes/site-req1/captures/walkthrough-req1/raw/walkthrough.json"})
    world = World()
    assert world.publish(capture) == {"status": "no_source"}
    assert world.rights_reads == 0 and world.archived == [] and world.delivered == []
    assert not (capture.joinpath(*clip.DIRECTORY) / clip.CLIP_FILE).exists()


def test_a_capture_with_a_normalization_record_still_uses_it(tmp_path):
    capture = _capture(tmp_path, size="2560x1440")
    record = json.loads((capture / "pipeline/source_video/normalization.json").read_text())
    world = World()
    assert world.publish(capture)["status"] == "published"
    manifest = _manifest(capture)
    assert manifest["source"] == "normalized" and manifest["source_video"]["kind"] == "normalized"
    assert manifest["source_video"]["relative"] == record["preparation_video"] != "raw/walkthrough.mp4"
    assert manifest["source_video"]["digest"] == record["normalized"]["sha256"]


def test_the_tick_covers_a_capture_handed_off_before_normalization(tmp_path):
    legacy = _capture(tmp_path, normalized=False)
    world = World()
    assert clip.capture_roots(tmp_path / "storage") == [legacy]
    counts = clip.tick(tmp_path / "storage", check_rights=world.check_rights, archive=world.archive,
                       retract=world.retract, deliver=world.deliver, now=lambda: NOW)
    assert counts == {"published": 1} and _manifest(legacy)["source"] == "raw_upload"


def test_a_failed_make_never_raises_and_is_retried_by_the_next_pass(capture, monkeypatch):
    real = clip._encode

    def broken(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["ffmpeg"])
    monkeypatch.setattr(clip, "_encode", broken)
    world = World()
    assert world.publish(capture) == {"status": "failed", "reason": "website_company_clip_encode_failed"}
    monkeypatch.setattr(clip, "_encode", real)
    assert world.publish(capture)["status"] == "published"


def test_a_clip_that_cannot_fit_the_cap_is_flagged_once_and_not_re_encoded(capture, monkeypatch):
    world = World()
    assert world.publish(capture, max_bytes=4096) == {"status": "over_size_cap"}
    assert _manifest(capture)["refused"] == clip.OVER_CAP and world.archived == []
    assert not (capture.joinpath(*clip.DIRECTORY) / clip.CLIP_FILE).exists()
    monkeypatch.setattr(clip, "build_clip", lambda *a, **k: pytest.fail("re-encoded a refused clip"))
    assert world.publish(capture, max_bytes=4096) == {"status": "over_size_cap"}


def test_the_bitrate_keeps_a_two_minute_clip_under_the_cap():
    assert clip.video_kbps(120) == clip.TARGET_VIDEO_KBPS
    assert clip.TARGET_VIDEO_KBPS * 1000 * 120 / 8 < clip.MAX_CLIP_BYTES * 0.8
    assert clip.video_kbps(1200) * 1000 * 1200 / 8 <= clip.MAX_CLIP_BYTES
    assert clip.video_kbps(4 * 3600) is None
    assert clip.target_dimensions(1080, 1920) == (720, 1280) and clip.target_dimensions(640, 480) == (640, 480)


def test_the_tick_covers_every_capture_and_one_failure_stops_none(tmp_path, monkeypatch):
    good = _capture(tmp_path)
    other = _capture(tmp_path, capture_id="supplement-" + "a" * 40)
    (other / "pipeline/source_video/normalization.json").write_text("{not json")
    world = World()
    counts = clip.tick(tmp_path / "storage", check_rights=world.check_rights, archive=world.archive,
                       retract=world.retract, deliver=world.deliver, now=lambda: NOW)
    assert counts == {"published": 1, "no_source": 1} and good.joinpath(*clip.DIRECTORY).is_dir()


def test_the_atlas_preview_timer_pass_runs_the_clip_pass_and_swallows_its_failure(monkeypatch, tmp_path):
    from blueprint_pipeline import website_atlas_preview
    monkeypatch.setattr(clip, "tick", lambda root: (_ for _ in ()).throw(RuntimeError("boom")))
    assert website_atlas_preview.company_clip_tick(tmp_path) == {"tick_failed": 1}


def test_delivery_uses_the_signed_company_clip_operation_and_checks_the_acceptance(monkeypatch):
    from blueprint_pipeline import website_task_context
    calls, sent = [], []
    record = {"clip_revision": "sha256:" + "1" * 64, "clip_digest": "sha256:" + "2" * 64}
    acceptance = {"schema_version": "website_company_clip_acceptance.v1", "accepted": True, **record}
    monkeypatch.setattr(website_task_context, "website_webapp_request", lambda **kw: calls.append(kw) or dict(acceptance))
    identity = {"request_id": "req1", "scene_id": "site-req1", "capture_id": "walkthrough-req1"}
    clip._deliver(record, identity=identity)
    assert calls == [{"capture_id": "walkthrough-req1", "operation": "company-clip",
                      "payload": {"request_id": "req1", "scene_id": "site-req1", "clip": record}}]
    acceptance["clip_digest"] = "sha256:" + "3" * 64
    with pytest.raises(ValueError, match="acceptance_invalid"):
        clip._deliver(record, identity=identity)
    monkeypatch.undo()

    class Response:
        body = b'{"accepted": true}'
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.invalid")
    monkeypatch.setattr(website_task_context, "load_pipeline_sync_token", lambda: "fixture-sync-token")
    monkeypatch.setattr(website_task_context, "safe_request", lambda url, **_kw: sent.append(url) or Response())
    website_task_context.website_webapp_request(capture_id="walkthrough-req1", operation="company-clip", payload={})
    assert sent == ["https://webapp.invalid/api/internal/pipeline/creator-captures/walkthrough-req1/company-clip"]


@pytest.mark.parametrize(("header", "expected"), [
    (None, None), ("", None), ("bytes=0-99", (0, 99)), ("bytes=100-", (100, 999)), ("bytes=-100", (900, 999)),
    ("bytes=900-5000", (900, 999)), ("bytes=0-1,5-9", None), ("items=0-9", None), ("bytes=9-3", None),
    ("bytes=-5000", (0, 999))])
def test_one_byte_range_is_parsed(header, expected):
    assert clip.parse_range(header, 1000) == expected


@pytest.mark.parametrize("header", ["bytes=1000-", "bytes=5000-6000", "bytes=-0"])
def test_a_range_past_the_end_is_not_satisfiable(header):
    with pytest.raises(clip.RangeNotSatisfiable) as caught:
        clip.parse_range(header, 1000)
    assert caught.value.size == 1000


class _Bucket:
    def __init__(self, objects):
        self.objects, self.ranges = objects, []

    def head_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise type("NoSuchKey", (Exception,), {"response": {"Error": {"Code": "404"}}})()
        body = self.objects[Key]
        return {"ContentLength": len(body), "Metadata": {"sha256": hashlib.sha256(body).hexdigest()}}

    def get_object(self, *, Bucket, Key, Range=None):
        body = self.objects[Key]
        self.ranges.append(Range)
        if Range:
            first, last = map(int, Range.removeprefix("bytes=").split("-"))
            body = body[first:last + 1]
        return {"Body": _Body(body)}


class _Body(io.BytesIO):
    def iter_chunks(self, size):
        while chunk := self.read(size):
            yield chunk


@pytest.fixture
def served(capture, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from blueprint_pipeline import website_atlas_preview_http as http
    world = World()
    assert world.publish(capture)["status"] == "published"
    bucket = _Bucket(world.objects)
    digest = _manifest(capture)["archive"]["digest"]

    def admission(request: Request):
        request.state.intake_client_id = request.headers.get("x-test-client", "")
    app = FastAPI()
    http.register_website_atlas_preview_routes(app, require_admission=admission)
    monkeypatch.setenv(http.HANDOFF_STORAGE_ROOT_ENV, str(capture.parents[4]))
    real = clip.open_company_clip
    monkeypatch.setattr(clip, "open_company_clip", lambda **kw: real(**kw, client=bucket, bucket="b"))
    url = f"/api/live-pipeline/website-company-clips/walkthrough-req1/clip.mp4?request_id=req1&digest={digest}"
    body = (capture.joinpath(*clip.DIRECTORY) / clip.CLIP_FILE).read_bytes()
    return {"client": TestClient(app), "url": url, "body": body, "digest": digest, "bucket": bucket,
            "capture": capture, "world": world,
            "signed": {"x-blueprint-pipeline-signature": "sha256=x", "x-test-client": "blueprint-webapp"}}


def test_the_webapp_reads_the_whole_clip_and_any_one_range(served):
    client, url, body, signed = served["client"], served["url"], served["body"], served["signed"]
    whole = client.get(url, headers=signed)
    assert whole.status_code == 200, whole.text
    assert whole.content == body
    assert whole.headers["content-type"] == "video/mp4" and whole.headers["accept-ranges"] == "bytes"
    assert whole.headers["cache-control"] == "private, no-store"
    assert whole.headers["x-blueprint-artifact-sha256"] == served["digest"]
    part = client.get(url, headers={**signed, "range": "bytes=100-1123"})
    assert part.status_code == 206 and part.content == body[100:1124]
    assert part.headers["content-range"] == f"bytes 100-1123/{len(body)}"
    assert part.headers["content-length"] == "1024"
    tail = client.get(url, headers={**signed, "range": "bytes=-10"})
    assert tail.status_code == 206 and tail.content == body[-10:]
    past = client.get(url, headers={**signed, "range": f"bytes={len(body)}-"})
    assert past.status_code == 416 and past.headers["content-range"] == f"bytes */{len(body)}"
    assert client.get(url, headers={**signed, "range": "bytes=0-1,4-5"}).content == body
    assert served["bucket"].ranges[:2] == [None, "bytes=100-1123"]


def test_only_the_webapp_reads_and_only_the_named_clip(served):
    client, url, signed = served["client"], served["url"], served["signed"]
    assert client.get(url, headers={**signed, "x-test-client": "robot-team"}).status_code == 403
    assert client.get(url, headers={"x-test-client": "blueprint-webapp"}).status_code == 403
    assert client.get(url.replace("req1&", "req2&"), headers=signed).status_code == 404
    assert client.get(url.replace(served["digest"][7:], "0" * 64), headers=signed).status_code == 404
    raw = _sha(served["capture"] / "raw/walkthrough.mp4")
    assert client.get(url.replace(served["digest"], raw), headers=signed).status_code == 404


def test_a_withdrawn_capture_has_no_clip_left_and_the_route_refuses(served, tmp_path):
    # Withdrawal: the route refuses at once; deletion then removes the host copy and every archived
    # version (website_capture_withdrawal_deletion, proven there), after which nothing is left to read.
    client, url, signed, capture = served["client"], served["url"], served["signed"], served["capture"]
    _withdraw(capture)
    assert client.get(url, headers=signed).status_code == 404
    assert client.get(url, headers={**signed, "range": "bytes=0-10"}).status_code == 404
    shutil.rmtree(capture.parents[1])  # withdrawal deletion removes the scene directory
    served["bucket"].objects.clear()   # and every version of the archived clip
    assert client.get(url, headers=signed).status_code == 404
    assert not capture.exists() and served["bucket"].objects == {}


def test_an_open_takedown_closes_the_route(served):
    _write(served["capture"] / "pipeline/consent_takedown/takedown_manifest.json", {"takedown_open": True})
    assert served["client"].get(served["url"], headers=served["signed"]).status_code == 404


def test_a_storage_fault_is_503_and_names_nothing(served, monkeypatch):
    def broken(**_kw):
        raise RuntimeError("https://secret.invalid/signed?token=x")
    monkeypatch.setattr(clip, "open_company_clip", broken)
    response = served["client"].get(served["url"], headers=served["signed"])
    assert response.status_code == 503 and "secret" not in response.text
