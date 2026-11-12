"""The company clip's privacy review fixes: captions, a basis that ends, a WebApp refusal, the pass's limits
and the route. Each reviewer probe is ported here as an asserting test.

Hermetic like ``test_website_company_clip``: temporary capture roots, real local ffmpeg/ffprobe on
synthetic clips, fakes for the WebApp, the store and the bucket. Nothing is paid for.
"""
# Covers: src/blueprint_pipeline/website_company_clip.py
# Covers: src/blueprint_pipeline/website_atlas_preview_http.py
# Covers: src/blueprint_pipeline/sharing_statement.py
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi import Request

from blueprint_pipeline import website_company_clip as clip
from tests.test_website_company_clip import (
    NOW, OLDER, SHARED, World, _Bucket, _capture, _manifest, _owned_list, _rights, _sha, _withdraw, _write,
)

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                reason="needs local ffmpeg and ffprobe")


@pytest.fixture(autouse=True)
def _route_clock(monkeypatch):
    monkeypatch.setattr(clip, "_wall_clock", lambda: NOW + 60)
HOUR, DAY = 3600, 86400
PLACE = b"GPS 48.8584N SECRETLOC"


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), check=True, capture_output=True, cwd=cwd)


def _parity(byte: int) -> int:
    return byte | 0x80 if bin(byte).count("1") % 2 == 0 else byte


def _captioned_source(work: Path) -> Path:
    """An H.264 clip whose every frame carries an A53 (CEA-608) caption with a planted place name, as a
    phone or a broadcast recorder writes it: T.35 SEI units inside the video bitstream itself."""
    work.mkdir(parents=True, exist_ok=True)
    raw = work / "cc.h264"
    _run("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=3",
         "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1", "-pix_fmt", "yuv420p", "-f", "h264", str(raw))
    pairs = [(0x94, 0x25)] * 2 + [(_parity(PLACE[i]), _parity(PLACE[i + 1] if i + 1 < len(PLACE) else 0x20))
                                  for i in range(0, len(PLACE), 2)] + [(0x94, 0xAD)] * 2 + [(0x80, 0x80)] * 200
    data, starts, position = raw.read_bytes(), [], 0
    while (found := data.find(b"\x00\x00\x01", position)) >= 0:
        starts.append(found)
        position = found + 3
    out, frame = bytearray(), 0
    for index, start in enumerate(starts):
        unit = data[start:(starts[index + 1] if index + 1 < len(starts) else len(data))]
        if unit[3] & 0x1F in (1, 5):
            first, second = pairs[frame]
            frame += 1
            payload = bytes([0xB5, 0x00, 0x31]) + b"GA94" + bytes([0x03, 0x41, 0xFF, 0xFC, first, second, 0xFF])
            out += b"\x00\x00\x01\x06" + bytes([4, len(payload)]) + payload + b"\x80"
        out += unit
    raw.write_bytes(bytes(out))
    source = work / "cc.src.mp4"
    _run("ffmpeg", "-v", "error", "-y", "-r", "30", "-i", str(raw), "-c", "copy", str(source))
    return source


def _decoded_captions(path: Path) -> str:
    """The caption text a decoder reads out of the file's video, or ""."""
    srt = path.with_suffix(".srt")
    srt.unlink(missing_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"movie={path.name}[out0+subcc]",
                    "-map", "0:s", "-c:s", "srt", srt.name], cwd=path.parent, capture_output=True)
    return srt.read_text() if srt.exists() else ""


def _side_data(path: Path) -> set[str]:
    out = _run("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_entries",
               "frame=side_data_list", "-of", "json", str(path)).stdout
    return {row.get("side_data_type") for frame in json.loads(out)["frames"] for row in frame.get("side_data_list") or []}


# --- MF1: captions inside the video ----------------------------------------------------------------------

def test_closed_captions_in_the_source_never_reach_the_clip(tmp_path):
    source = _captioned_source(tmp_path / "cc")
    assert "SECRETLOC" in _decoded_captions(source)  # the probe works: the source decodes the place name
    target = tmp_path / "cc" / "clip.mp4"
    built = clip.build_clip(source, target)
    assert built["width"] == 640 and built["has_audio"] is False
    assert "SECRETLOC" not in _decoded_captions(target) and "48.8584" not in _decoded_captions(target)
    body = target.read_bytes()
    assert b"SECRETLOC" not in body and b"48.8584" not in body
    assert _side_data(target) == set() and clip.frames_clean(target) and not clip.sei_units_present(target)
    assert clip.sei_units_present(source)  # the check sees the source's caption units


def test_both_encoder_settings_are_needed_and_the_after_write_proof_decodes(tmp_path, monkeypatch):
    # Which settings are needed, proven on the same source: libx264 re-embeds the source's captions unless
    # -a53cc 0; without the bitstream filter x264's own SEI still comes out as frame side data, so the
    # proof "no frame carries side data" holds only with both.
    source = _captioned_source(tmp_path / "cc")
    outputs = {}
    for name, extra in (("plain", []), ("a53cc_off", ["-a53cc", "0"]),
                        ("both", ["-a53cc", "0", "-bsf:v", "filter_units=remove_types=6"])):
        outputs[name] = tmp_path / "cc" / f"{name}.mp4"
        _run("ffmpeg", "-v", "error", "-y", "-i", str(source), "-map", "0:v:0", "-c:v", "libx264", "-preset",
             "ultrafast", "-pix_fmt", "yuv420p", *extra, str(outputs[name]))
    assert "SECRETLOC" in _decoded_captions(outputs["plain"]) and clip.sei_units_present(outputs["plain"])
    assert "SECRETLOC" not in _decoded_captions(outputs["a53cc_off"]) and _side_data(outputs["a53cc_off"])
    assert clip.sei_units_present(outputs["a53cc_off"])
    assert _side_data(outputs["both"]) == set() and not clip.sei_units_present(outputs["both"])

    # An encoder that lets the captions through is caught after writing, by decoding the frames.
    def leaky(source, target, **kwargs):
        _run("ffmpeg", "-v", "error", "-y", "-i", str(source), "-map", "0:v:0", "-map_metadata", "-1",
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-f", "mp4", str(target))
    monkeypatch.setattr(clip, "_encode", leaky)
    with pytest.raises(ValueError, match="website_company_clip_output_frame_side_data"):
        clip.build_clip(source, tmp_path / "cc" / "leaky.mp4")
    assert not (tmp_path / "cc" / "leaky.mp4").exists()


def _leaky(source, target, **kwargs):
    """An encoder that keeps the source's captions and x264's own SEI."""
    _run("ffmpeg", "-v", "error", "-y", "-i", str(source), "-map", "0:v:0", "-map_metadata", "-1", "-c:v", "libx264",
         "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(target))


def test_the_frame_side_data_proof_refuses_on_its_own(tmp_path, monkeypatch):
    # Pins the decode proof alone: with the SEI-unit proof switched off it still refuses.
    source = _captioned_source(tmp_path / "cc")
    monkeypatch.setattr(clip, "_encode", _leaky)
    monkeypatch.setattr(clip, "sei_units_present", lambda path, **kw: False)
    with pytest.raises(ValueError, match="website_company_clip_output_frame_side_data"):
        clip.build_clip(source, tmp_path / "cc" / "out.mp4")


def test_the_sei_unit_proof_refuses_on_its_own(tmp_path, monkeypatch):
    source = _captioned_source(tmp_path / "cc")
    monkeypatch.setattr(clip, "_encode", _leaky)
    monkeypatch.setattr(clip, "frames_clean", lambda path, **kw: True)
    with pytest.raises(ValueError, match="website_company_clip_output_sei_units_left"):
        clip.build_clip(source, tmp_path / "cc" / "out.mp4")


def test_a_clean_clip_whose_bytes_happen_to_hold_the_caption_marker_is_published(tmp_path, monkeypatch):
    # Clip bytes are close to random: "GA94" turns up in about one clip in 80. Outside an SEI unit it means
    # nothing, so it never refuses a clip.
    capture = _capture(tmp_path)
    real = clip._encode

    def with_marker(source, target, **kwargs):
        real(source, target, **kwargs)
        with open(target, "ab") as stream:  # a trailing MP4 "free" box holding the marker
            stream.write((12).to_bytes(4, "big") + b"free" + b"GA94")
    monkeypatch.setattr(clip, "_encode", with_marker)
    world = World()
    assert world.publish(capture)["status"] == "published"
    assert any(b"GA94" in body for body in world.objects.values())


def test_a_clip_an_after_encode_proof_refuses_is_recorded_and_not_re_encoded_every_pass(tmp_path, monkeypatch):
    capture = _capture(tmp_path)
    builds, real = [], clip.build_clip

    def counting(*args, **kwargs):
        builds.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(clip, "build_clip", counting)
    monkeypatch.setattr(clip, "frames_clean", lambda path, **kw: False)
    world = World()
    statuses = [world.publish(capture, now=lambda t=t: NOW + t) for t in (0, 60, 120)]
    assert statuses == [{"status": "failed", "reason": "website_company_clip_output_frame_side_data"}] * 3
    assert len(builds) == 1 and world.archived == []
    manifest = _manifest(capture)
    assert manifest["refused"] == "website_company_clip_output_frame_side_data" and manifest["refused_at_epoch"] == NOW
    # It does not starve the captures after it: later passes return at once, and a window later it is tried again.
    monkeypatch.setattr(clip, "frames_clean", lambda path, **kw: True)
    assert world.publish(capture, now=lambda: NOW + clip.LIVE_STATEMENT_RECHECK_SECONDS)["status"] == "published"
    assert len(builds) == 2


# --- MF2: a basis that ends takes the clip back ------------------------------------------------------------

def _open(capture: Path, world: World, digest: str, *, storage_root: Path | None = None, range_header=None):
    return clip.open_company_clip(request_id="req1", capture_id="walkthrough-req1", digest=digest,
                                  storage_root=storage_root or capture.parents[4], range_header=range_header,
                                  client=_Bucket(world.objects), bucket="b")


def _owned_row(capture: Path) -> dict:
    return {"capture_id": "walkthrough-req1", "approved_by": "owner", "approved_on": "2026-11-12", "evidence": "t",
            "video_sha256_prefix": _sha(capture / "raw/walkthrough.mp4")[7:19]}


def _published(tmp_path: Path, monkeypatch, case: str):
    """A published clip under the basis ``case`` ends, and what ends it."""
    capture = _capture(tmp_path, version=OLDER)
    if case == "live_not_sharing":
        world = World(rights=_rights(SHARED))  # the site accepted the sharing statement on its job page

        def end():
            world.rights = _rights(OLDER)
    else:
        monkeypatch.setattr(clip, "OWNED_CAPTURES_LIST", _owned_list(tmp_path, [_owned_row(capture)]))
        world = World(rights=_rights(OLDER))

        def end():
            if case == "owned_row_removed":
                monkeypatch.setattr(clip, "OWNED_CAPTURES_LIST", _owned_list(tmp_path, []))
            else:
                (tmp_path / "owned.json").write_text("{not json")
    assert world.publish(capture)["status"] == "published"
    return capture, world, end


@pytest.mark.parametrize("case", ["owned_row_removed", "owned_list_malformed", "live_not_sharing"])
@pytest.mark.parametrize("later", [60, 7 * HOUR, 30 * DAY])
def test_a_clip_whose_basis_ends_is_taken_back_and_the_route_refuses(tmp_path, monkeypatch, case, later):
    capture, world, end = _published(tmp_path, monkeypatch, case)
    manifest = _manifest(capture)
    digest = manifest["archive"]["digest"]
    expected = "blueprint_owned_capture" if case != "live_not_sharing" else "site_sharing_statement"
    assert manifest["permission"]["permission_basis"] == expected
    assert _open(capture, world, digest)[1] > 0
    end()
    result = world.publish(capture, now=lambda: NOW + later)
    if case == "live_not_sharing" and later < clip.LIVE_STATEMENT_RECHECK_SECONDS:
        # The live statement is re-read once a window: within it the clip stays, with no extra request.
        reads = world.rights_reads
        assert result["status"] == "current" and world.retracted == [] and world.rights_reads == reads
        return
    assert result == {"status": "not_permitted"}
    assert len(world.retracted) == 1 and world.objects == {} and _manifest(capture)["archive"]["retracted"] is True
    with pytest.raises(clip.CompanyClipUnavailable):
        _open(capture, world, digest)
    # Nothing comes back on later passes while the basis stays gone.
    assert world.publish(capture, now=lambda: NOW + later + 60)["status"] == "not_permitted"
    assert len(world.archived) == 1 and len(world.retracted) == 1


def test_a_published_clip_rereads_its_live_statement_once_a_window_and_records_success(tmp_path, monkeypatch):
    capture, world, _ = _published(tmp_path, monkeypatch, "live_not_sharing")
    check = json.loads((capture.joinpath(*clip.DIRECTORY) / clip.STATEMENT_CHECK_FILE).read_text())
    assert check["outcome"] == "covers_sharing" and check["granted_at_epoch"] == NOW
    reads = world.rights_reads
    statuses = [world.publish(capture, now=lambda t=t: NOW + t)["status"] for t in (60, HOUR, 6 * HOUR - 1)]
    assert statuses == ["current"] * 3 and world.rights_reads == reads
    assert world.publish(capture, now=lambda: NOW + 6 * HOUR)["status"] == "current"
    assert world.rights_reads == reads + 1
    check = json.loads((capture.joinpath(*clip.DIRECTORY) / clip.STATEMENT_CHECK_FILE).read_text())
    assert check["checked_at_epoch"] == NOW + 6 * HOUR and check["granted_at_epoch"] == NOW + 6 * HOUR


def test_the_route_refuses_a_grant_older_than_its_lifetime_with_no_answer_since(tmp_path, monkeypatch):
    # The pass is kept away (a hold, the budget): the route alone bounds an unanswered grant at 12 hours.
    capture, world, _ = _published(tmp_path, monkeypatch, "live_not_sharing")
    digest = _manifest(capture)["archive"]["digest"]
    check_path = capture.joinpath(*clip.DIRECTORY) / clip.STATEMENT_CHECK_FILE
    check = json.loads(check_path.read_text())
    check.update(outcome="unanswered", checked_at_epoch=NOW + 7 * HOUR)
    check_path.write_text(json.dumps(check))
    monkeypatch.setattr(clip, "_wall_clock", lambda: NOW + 11 * HOUR)
    assert _open(capture, world, digest)[1] > 0  # a grant 11 hours old is still served
    monkeypatch.setattr(clip, "_wall_clock", lambda: NOW + clip.LIVE_GRANT_LIFETIME_SECONDS)
    with pytest.raises(clip.CompanyClipUnavailable):
        _open(capture, world, digest)
    monkeypatch.setattr(clip, "_wall_clock", lambda: NOW + 31 * DAY)
    with pytest.raises(clip.CompanyClipUnavailable):
        _open(capture, world, digest)


def test_a_raw_upload_is_hashed_once_even_before_a_manifest_exists(tmp_path, monkeypatch):
    capture = _capture(tmp_path, normalized=False)
    world = World(rights=ValueError("website_control_task-context_http_503:unclassified"))
    assert world.publish(capture)["status"] == "refused"  # no answer: nothing is built, no manifest
    assert not (capture.joinpath(*clip.DIRECTORY) / clip.MANIFEST_FILE).exists()
    kept = json.loads((capture.joinpath(*clip.DIRECTORY) / clip.RAW_DIGEST_FILE).read_text())
    assert kept["digest"] == _sha(capture / "raw/walkthrough.mp4") and kept["relative"] == "raw/walkthrough.mp4"
    monkeypatch.setattr(clip, "_sha256_file", lambda path: pytest.fail("hashed the raw upload again"))
    monkeypatch.setattr("blueprint_pipeline.website_source_video.probe_video",
                        lambda path: pytest.fail("probed the raw upload again"))
    assert world.publish(capture, now=lambda: NOW + 60)["status"] == "refused"


def test_an_unanswered_reread_keeps_the_clip_for_one_more_window_then_takes_it_back(tmp_path, monkeypatch):
    capture, world, _ = _published(tmp_path, monkeypatch, "live_not_sharing")
    world.rights = ValueError("website_control_task-context_http_503:unclassified")
    assert world.publish(capture, now=lambda: NOW + 7 * HOUR)["status"] == "current"
    assert world.publish(capture, now=lambda: NOW + 13 * HOUR)["status"] == "not_permitted"
    assert len(world.retracted) == 1


def test_the_route_refuses_the_moment_blueprints_list_drops_the_capture(tmp_path, monkeypatch):
    capture, world, end = _published(tmp_path, monkeypatch, "owned_row_removed")
    digest = _manifest(capture)["archive"]["digest"]
    end()  # no pass has run since: the route itself checks the list
    with pytest.raises(clip.CompanyClipUnavailable):
        _open(capture, world, digest)


def test_a_live_revoked_answer_during_a_rebuild_takes_the_old_archive_back(tmp_path):
    capture = _capture(tmp_path)
    world = World()
    assert world.publish(capture)["status"] == "published"
    digest = _manifest(capture)["archive"]["digest"]
    (capture.joinpath(*clip.DIRECTORY) / clip.CLIP_FILE).unlink()  # forces a rebuild
    world.rights = _rights(SHARED, consent="revoked", revoked=True)
    assert world.publish(capture, now=lambda: NOW + 60) == {"status": "refused",
                                                            "reason": "website_company_clip_rights_refused"}
    assert len(world.retracted) == 1 and world.objects == {}
    with pytest.raises(clip.CompanyClipUnavailable):
        _open(capture, world, digest)


def test_a_definite_refusal_on_the_reread_of_a_current_clip_takes_it_back(tmp_path, monkeypatch):
    capture, world, _ = _published(tmp_path, monkeypatch, "live_not_sharing")
    world.rights = _rights(SHARED, consent="revoked", revoked=True)
    assert world.publish(capture, now=lambda: NOW + 7 * HOUR)["status"] == "refused"
    assert len(world.retracted) == 1


# --- MF3: a WebApp refusal is remembered -------------------------------------------------------------------

def test_a_webapp_refusal_is_remembered_and_not_uploaded_again_every_pass(tmp_path):
    capture = _capture(tmp_path)
    world = World(delivery_error=ValueError("website_control_company-clip_http_409:website_company_clip_not_permitted"))
    statuses = [world.publish(capture, now=lambda t=t: NOW + t)["status"] for t in (0, 60, 120, 180)]
    assert statuses == ["refused"] * 4
    assert len(world.archived) == 1 and len(world.retracted) == 1 and world.objects == {}
    assert _manifest(capture)["refused_by_webapp"]["reason"].endswith("website_company_clip_not_permitted")
    # A window later it is offered once more, and published when the WebApp now accepts it.
    world.delivery_error = None
    assert world.publish(capture, now=lambda: NOW + 6 * HOUR)["status"] == "published"
    assert len(world.archived) == 2 and "refused_by_webapp" not in _manifest(capture)


def test_a_webapp_refusal_is_offered_again_when_the_live_rights_change(tmp_path, monkeypatch):
    # Blueprint's own capture, which the WebApp has no record of: it refuses. Nothing is uploaded again
    # until the site's live rights change (here it accepts the sharing statement), then the clip is offered.
    capture = _capture(tmp_path, version=OLDER)
    monkeypatch.setattr(clip, "OWNED_CAPTURES_LIST", _owned_list(tmp_path, [_owned_row(capture)]))
    world = World(rights=_rights(OLDER),
                  delivery_error=ValueError("website_control_company-clip_http_409:website_company_clip_not_permitted"))
    assert [world.publish(capture, now=lambda t=t: NOW + t)["status"] for t in (0, 60)] == ["refused"] * 2
    assert len(world.archived) == 1
    world.rights, world.delivery_error = _rights(SHARED), None
    assert world.publish(capture, now=lambda: NOW + 120)["status"] == "published"
    assert len(world.archived) == 2 and world.delivered[-1]["permission_basis"] == "site_sharing_statement"


# --- SF1: the pass's limits ------------------------------------------------------------------------------

def test_the_pass_stops_for_a_hold_and_keeps_to_its_time_budget(tmp_path, monkeypatch):
    first = _capture(tmp_path)
    _capture(tmp_path, capture_id="supplement-" + "a" * 40)
    world = World()
    options = {"check_rights": world.check_rights, "archive": world.archive, "retract": world.retract,
               "deliver": world.deliver, "now": lambda: NOW}
    assert clip.tick(tmp_path / "storage", stop_reason=lambda: "operator_hold", **options) == {
        "deferred": 2, "stopped_operator_hold": 1}
    assert world.archived == [] and not first.joinpath(*clip.DIRECTORY).exists()
    ticks = iter([0.0, 0.0, clip.TICK_BUDGET_SECONDS])
    counts = clip.tick(tmp_path / "storage", clock=lambda: next(ticks), **options)
    assert counts == {"published": 1, "deferred": 1, "stopped_time_budget": 1}


def test_the_atlas_preview_pass_hands_its_stop_check_to_the_clip_pass(monkeypatch, tmp_path):
    from blueprint_pipeline import website_atlas_preview
    seen = {}
    monkeypatch.setattr(clip, "tick", lambda root, **kw: seen.update(kw) or {"current": 1})

    def stop():
        return None
    assert website_atlas_preview.company_clip_tick(tmp_path, stop) == {"current": 1}
    assert seen == {"stop_reason": stop}


def test_an_encode_that_times_out_is_recorded_and_not_retried_every_pass(tmp_path, monkeypatch):
    capture = _capture(tmp_path)
    real, real_build = clip._encode, clip.build_clip

    def slow(*args, **kwargs):
        raise subprocess.TimeoutExpired(["ffmpeg"], kwargs["timeout"])
    monkeypatch.setattr(clip, "_encode", slow)
    world = World()
    assert world.publish(capture) == {"status": "encode_timed_out"}
    assert _manifest(capture)["refused"] == clip.ENCODE_TIMED_OUT and world.archived == []
    monkeypatch.setattr(clip, "build_clip", lambda *a, **k: pytest.fail("re-encoded a timed-out source"))
    assert world.publish(capture, now=lambda: NOW + 60) == {"status": "encode_timed_out"}
    monkeypatch.setattr(clip, "build_clip", real_build)
    monkeypatch.setattr(clip, "_encode", real)
    assert world.publish(capture, now=lambda: NOW + 6 * HOUR)["status"] == "published"


def test_the_encode_bounds_its_output_size_and_its_time(tmp_path, monkeypatch):
    capture = _capture(tmp_path)
    commands = []
    real = subprocess.run

    def recording(command, *args, **kwargs):
        if command and command[0] == "ffmpeg":
            commands.append((list(command), kwargs.get("timeout")))
        return real(command, *args, **kwargs)
    monkeypatch.setattr(clip.subprocess, "run", recording)
    monkeypatch.setattr(clip, "video_kbps", lambda *a, **k: 50_000)  # a rate that overshoots any small cap
    with pytest.raises(ValueError, match=clip.OVER_CAP):
        clip.build_clip(capture / "raw/walkthrough.mp4", tmp_path / "out.mp4", max_bytes=40_000)
    command, timeout = commands[0]
    assert command[command.index("-fs") + 1] == "40001" and 0 < timeout <= clip.ENCODE_TIMEOUT_SECONDS
    assert command[command.index("-a53cc") + 1] == "0"
    assert command[command.index("-bsf:v") + 1] == "filter_units=remove_types=6"
    assert not (tmp_path / "out.mp4").exists() and not (tmp_path / "out.mp4.partial").exists()


# --- SF4: the frame rate ---------------------------------------------------------------------------------

def test_a_source_just_over_30_fps_comes_out_at_30(tmp_path):
    source = tmp_path / "fast.mp4"
    _run("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=6001/200:duration=2",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(source))
    built = clip.build_clip(source, tmp_path / "clip.mp4")
    assert built["fps"] <= clip.MAX_CLIP_FPS and built["fps"] == 30.0


# --- the route: storage root, ranges, issuer, retraction, schema --------------------------------------------

@pytest.fixture
def published(tmp_path):
    capture = _capture(tmp_path)
    world = World()
    assert world.publish(capture)["status"] == "published"
    return capture, world, _manifest(capture)["archive"]["digest"]


def _app(capture: Path, world: World, monkeypatch, *, storage_root: bool = True):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from blueprint_pipeline import website_atlas_preview_http as http

    def admission(request: Request):
        request.state.intake_client_id = request.headers.get("x-test-client", "")
    app = FastAPI()
    http.register_website_atlas_preview_routes(app, require_admission=admission)
    if storage_root:
        monkeypatch.setenv(http.HANDOFF_STORAGE_ROOT_ENV, str(capture.parents[4]))
    else:
        monkeypatch.delenv(http.HANDOFF_STORAGE_ROOT_ENV, raising=False)
    real = clip.open_company_clip
    bucket = _Bucket(world.objects)
    monkeypatch.setattr(clip, "open_company_clip", lambda **kw: real(**kw, client=bucket, bucket="b"))
    return TestClient(app)


def _url(digest: str) -> str:
    return f"/api/live-pipeline/website-company-clips/walkthrough-req1/clip.mp4?request_id=req1&digest={digest}"


SIGNED = {"x-blueprint-pipeline-signature": "sha256=x", "x-test-client": "blueprint-webapp"}


def test_with_no_storage_root_nothing_is_served(published, monkeypatch):
    capture, world, digest = published
    with pytest.raises(clip.CompanyClipUnavailable):
        clip.open_company_clip(request_id="req1", capture_id="walkthrough-req1", digest=digest, storage_root=None,
                               client=_Bucket(world.objects), bucket="b")
    client = _app(capture, world, monkeypatch, storage_root=False)
    _withdraw(capture)
    assert client.get(_url(digest), headers=SIGNED).status_code == 404


@pytest.mark.parametrize("header", ["bytes=" + "9" * 5000 + "-", "bytes=0-" + "9" * 5000, "bytes=-" + "9" * 5000],
                         ids=["first", "last", "suffix"])
def test_an_absurdly_long_range_number_is_the_whole_file_never_a_storage_fault(published, monkeypatch, header):
    capture, world, digest = published
    assert clip.parse_range(header, 1000) is None
    response = _app(capture, world, monkeypatch).get(_url(digest), headers={**SIGNED, "range": header})
    assert response.status_code == 200 and len(response.content) == _manifest(capture)["clip"]["size_bytes"]


def test_the_issuer_refusal_names_the_company_clip(published, monkeypatch):
    capture, world, digest = published
    response = _app(capture, world, monkeypatch).get(_url(digest), headers={"x-test-client": "blueprint-webapp"})
    assert response.status_code == 403 and response.json()["detail"] == "website company clip issuer not authorized"


@pytest.mark.parametrize("change", [
    lambda manifest: manifest["archive"].update(retracted=True),
    lambda manifest: manifest["archive"].update(retraction="pending"),
    lambda manifest: manifest.update(schema_version="website_company_clip_manifest.v0"),
    lambda manifest: manifest["permission"].update(permission_basis="someone_said_so"),
])
def test_the_route_refuses_a_taken_back_archive_another_schema_or_an_unknown_basis(published, change):
    capture, world, digest = published
    path = capture.joinpath(*clip.DIRECTORY) / clip.MANIFEST_FILE
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))
    assert world.objects  # the stored object is still there: only the manifest says no
    with pytest.raises(clip.CompanyClipUnavailable):
        _open(capture, world, digest)


# --- the not-permitted pass and the after-write proofs -----------------------------------------------------

def test_the_not_permitted_precheck_takes_an_archived_copy_back_without_a_live_read(tmp_path, monkeypatch):
    capture, world, end = _published(tmp_path, monkeypatch, "owned_row_removed")
    end()
    reads = world.rights_reads
    assert world.publish(capture, now=lambda: NOW + 60) == {"status": "not_permitted"}
    assert world.rights_reads == reads and len(world.retracted) == 1


@pytest.mark.parametrize("tags", [
    {"format_tags": {"encoder": "Lavf61.1.100 ENCSECRET"}, "streams": []},
    {"format_tags": {}, "streams": [{"tags": {"encoder": "Lavc61.3.100 libx264 at 48.8584"}}]},
    {"format_tags": {}, "streams": [{"tags": {"encoder": "x264"}}]},
])
def test_the_tag_allow_list_refuses_an_unexpected_encoder_value(tags):
    assert not clip.metadata_clean(tags)
    assert clip.metadata_clean({"format_tags": {"encoder": "Lavf61.1.100"},
                                "streams": [{"tags": {"encoder": "Lavc61.3.100 libx264"}}]})


def test_the_audio_count_after_encoding_is_enforced(tmp_path, monkeypatch):
    from blueprint_pipeline import capture_audio_minimization
    capture = _capture(tmp_path)
    monkeypatch.setattr(capture_audio_minimization, "probe_audio_stream_count", lambda path: 1)
    with pytest.raises(ValueError, match="website_company_clip_output_audio_not_proven_absent"):
        clip.build_clip(capture / "raw/walkthrough.mp4", tmp_path / "out.mp4")
    assert not (tmp_path / "out.mp4").exists()


# --- Blueprint's own list and the raw-upload fallback --------------------------------------------------------

def test_an_owned_capture_with_no_normalization_record_is_matched_by_its_raw_upload(tmp_path, monkeypatch):
    capture = _capture(tmp_path, version=OLDER, normalized=False)
    monkeypatch.setattr(clip, "OWNED_CAPTURES_LIST", _owned_list(tmp_path, [_owned_row(capture)]))
    world = World(rights=_rights(OLDER))
    assert world.publish(capture)["status"] == "published"
    manifest = _manifest(capture)
    assert manifest["source"] == "raw_upload" and manifest["upload_sha256"] == _sha(capture / "raw/walkthrough.mp4")
    other = tmp_path / "other"
    capture = _capture(other, version=OLDER, normalized=False)
    monkeypatch.setattr(clip, "OWNED_CAPTURES_LIST", _owned_list(other, [{**_owned_row(capture),
                                                                          "video_sha256_prefix": "0" * 12}]))
    assert World(rights=_rights(OLDER)).publish(capture)["status"] == "not_permitted"


def test_a_brand_new_capture_waits_for_normalization(tmp_path):
    capture = _capture(tmp_path, normalized=False)
    shutil.rmtree(capture / "pipeline/clean_plate")
    _write(capture / "pipeline/website_task_context.json", {"capture_rights": _rights()})
    world = World()
    assert world.publish(capture) == {"status": "no_source"}
    assert world.rights_reads == 0 and not capture.joinpath(*clip.DIRECTORY).exists()


def test_the_shipped_allow_list_names_one_video_per_row():
    from blueprint_pipeline.sharing_statement import owned_captures
    owned, flag = owned_captures()
    assert flag is None and owned and all(len(prefix) >= 12 for prefix in owned.values())
