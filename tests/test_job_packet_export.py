"""The job packet export: one packet, shared only on a recorded basis, never carrying who the site is or where.

Every input tree is synthetic and built here, in the host's real shapes; no real site data.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from blueprint_pipeline import job_packet_export as packet
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.job_packet_export import (
    EARLY_TRIAL_LABEL, TERMS_LINE, JobInputs, JobPacketRefused, OwnedCaptureApproval, build_job_packet,
    done_sentence,
)

CURRENT = "2026-11-12.v4"
OLDER = "2026-11-02.v2"
SENTENCE = "Move the blue tray from the counter to the back shelf"
NOW = datetime(2026, 11, 12, 18, 0, tzinfo=timezone.utc)
SITE = "site-req-1"
CAPTURE = "walkthrough-req-1"
CAPTURE_REL = f"scenes/{SITE}/captures/{CAPTURE}"
PREP = "pipeline/website_scene_preparation"
REVISION = "sha256:" + "d" * 64
HOST = "/var/lib/PLANTHOSTPATH/pubsub-handoffs/PLANTBUCKET"
SPLAT = f"{PREP}/atlas/level/abc/atlas_depth_splat_levelled.ply"
COLLISION = f"{PREP}/collider/def/collider_conditioned.glb"
FIT = f"{PREP}/native/robot_fit_table.json"
VIDEO = "raw/walkthrough.mp4"
#: Planted in every field of the capture's files that names the site or its people, says where it is, or holds a
#: token, a link or a storage path. None may reach the packet folder.
NAME, ORG, STREET, EMAIL, PHONE = "PLANTSITENAME", "PLANTORGANISATION", "PLANTSTREET 17", "plant@example.invalid", "+1 555 0142 99"
USER, TOKEN, LINK, STORAGE, REGION = "PLANTUSERID", "PLANTTOKEN", "https://plant.example.invalid/x", "PLANTSTORAGE", "PLANTREGION"
FORBIDDEN = (NAME, ORG, STREET, EMAIL, PHONE, USER, TOKEN, LINK, STORAGE, REGION, "PLANTHOSTPATH", "PLANTBUCKET",
             "gs://", "req-1", SITE, CAPTURE)


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    elif isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sealed(value: dict, field: str) -> dict:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _host(relative: str) -> str:
    return f"{HOST}/{CAPTURE_REL}/{relative}"


@pytest.fixture(scope="session")
def clip_with_sound(tmp_path_factory) -> bytes:
    """A one-second mp4 with a video and an audio stream."""
    path = tmp_path_factory.mktemp("clip") / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5:duration=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:v", "mpeg4", "-c:a", "aac", "-shortest",
                    str(path)], check=True)
    return path.read_bytes()


def _streams(path: Path, kind: str) -> list[str]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", kind, "-show_entries", "stream=codec_name",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def _contract() -> dict:
    return {"schema_version": "rigid_task_success_contract.v1", "scope": {"site_id": SITE, "task_id": "t"},
            "provenance": {"author_source": "task_owner", "confirmation_status": "proposal_only"},
            "criteria": {"surface_target": {"radius_m": 0.2, "surface_position_world_m": [3.14159, 2.71828, 0.9],
                                            "visible_label": "back shelf", "maximum_tilt_rad": 0.2617993878,
                                            "stable_seconds": 1.0},
                         "support": {"height_mode": "required", "height_interval_m": [0.88, 0.92]},
                         "gripper_state": {"mode": "released", "threshold_m": 0.07},
                         "motion": {"minimum_lift_m": 0.05}},
            "contract_digest": "sha256:" + "c" * 64}


def _rights(statement: str | None) -> dict:
    return {"derived_scene_generation_allowed": True, "data_licensing_allowed": False,
            "capture_contributor_payout_eligible": False, "consent_status": "granted", "consent_revoked": False,
            "consent_scope": ["derived_scene_generation", "robot_evaluation"], "statement_version": statement,
            "recorded_at_iso": "2026-11-12T10:00:00.000Z"}


def _context(statement: str | None = CURRENT) -> dict:
    return _sealed({
        "schema_version": "website_site_task_context.v1", "purpose": "scene_preparation", "request_id": "req-1",
        "scene_id": SITE, "capture_id": CAPTURE, "description": SENTENCE, "task_family": "rigid_pick_and_place",
        "confirmed": True, "confirmed_at": "2026-11-01T00:00:00Z", "operator_answers": {"where": "Back room"},
        "unresolved": [], "operator_task_details": {"item_weight": "about 2 kg", "item_make_model": "house tray"},
        "success_criteria": {"successDefinition": "Tray on the shelf", "successRate": 90, "cycleTimeSeconds": 30,
                             "unknown": False},
        "task_items": [{"item_id": "i1", "label": "tray", "location_note": "counter", "basis": "described",
                        "images": [{"storage_path": f"scenes/{SITE}/items/i1/{STORAGE}.jpg"}]}],
        "capture_rights": _rights(statement)}, "context_digest")


def _fit_rows() -> list[dict]:
    return [{"model_id": "universal_robots_ur5e", "name": "UR5e", "manufacturer": "Universal Robots",
             "body_type": "fixed_arm", "verdict": "fits_spec", "reasons": [],
             "checks": [{"check": "reach", "result": "pass", "reason": "within 3.14159 m"},
                        {"check": "payload", "result": "unknown", "reason": "mass unknown"},
                        {"check": "cycle_time", "result": "not_requested", "reason": "no target"}]},
            {"model_id": "universal_robots_ur10e", "name": "UR10e", "manufacturer": "Universal Robots",
             "body_type": "fixed_arm", "verdict": "doesnt_fit", "reasons": ["reach: 2.71828 m over"],
             "checks": [{"check": "reach", "result": "fail", "reason": "2.71828 m over"},
                        {"check": "payload", "result": "pass", "reason": "fine"},
                        {"check": "cycle_time", "result": "not_requested", "reason": "no target"}]}]


def _absent(prefix: str = "") -> list[str]:
    capture = f"{prefix}{CAPTURE_REL}"
    return [f"{prefix}scenes/{SITE}/website_withdrawal", f"{capture}/pipeline/consent_takedown",
            f"{capture}/raw/rights_consent.json", f"{capture}/rights_consent.json",
            f"{capture}/pipeline_job_terminal_receipt.json", f"{capture}/{PREP}/revisions"]


def _pull_record(root: Path, buckets: list[str], absent: list[str]) -> Path:
    return _write(root / "pull_record.json", {"schema_version": "job_packet_pull_record.v1", "buckets": buckets,
                                              "absent_on_host": absent})


def _seal_first_look(capture: Path, *, rows=None, table_change=None) -> None:
    """Write the fit table file and the handoff's pointer to it, sealed as the Pipeline seals them."""
    context = json.loads((capture / "pipeline" / "website_task_context.json").read_text())
    prep_path = capture / PREP / "preparation.json"
    preparation = json.loads(prep_path.read_text())
    preparation["binding"] = {"task_context_digest": context["context_digest"]}
    _write(prep_path, _sealed(preparation, "digest"))
    table = {"schema_version": "website_robot_fit_table.v1", "preparation_digest": preparation["digest"],
             "evidence_label": "spec_bounded_estimate", "claim_ceiling": "development_only",
             "physical_world_truth_claimed": False,
             "inputs": {"scale_band": [0.75, 1.33], "mass_basis": "estimated", "mass_kg_range": [0.4, 1.8],
                        "straight_line_m": [0.43, 0.76], "carry_path_m": [0.59, 0.92], "cycle_time_target_s": None},
             "rows": _fit_rows() if rows is None else rows, "counts": {"fits_spec": 1, "doesnt_fit": 1}}
    if table_change:
        table_change(table)
    _write(capture / FIT, _sealed(table, "table_digest"))
    _write(capture / PREP / "handoff.json", _sealed({
        "schema_version": "website_scene_handoff.v1", "status": "intake_ready", "capture_id": CAPTURE, "scene_id": SITE,
        "simulator_ready": False, "base_scene_path": _host(f"{PREP}/base_scene.json"),
        "preparation_path": _host(f"{PREP}/preparation.json"), "preparation_digest": preparation["digest"],
        "thumbnail": {"path": _host(f"{PREP}/{STORAGE}.png")}, "website_intake_outbox": {"id": TOKEN},
        "robot_fit_table": {"path": _host(FIT), "digest": table["table_digest"],
                            "counts": {"fits": 0, "fits_spec": 1, "doesnt_fit": 1, "unknown": 0}}}, "digest"))


def _identity_files(capture: Path) -> None:
    """The capture's records as the host holds them, with every name, contact, location, token, link and path planted."""
    _write(capture / "capture_descriptor.json", {
        "schema_version": "v1", "scene_id": SITE, "capture_id": CAPTURE, "site_submission_id": "req-1",
        "raw_prefix_uri": f"gs://{STORAGE}/{CAPTURE_REL}/raw", "frames_index_uri": f"gs://{STORAGE}/frames",
        "keyframe_uri": f"gs://{STORAGE}/key.jpg", "qa_report_uri": LINK, "raw_video_uri": f"gs://{STORAGE}/v.mp4",
        "region_id": REGION, "environment_type_hint": "unknown",
        "metadata": {"site_identity": {"name": NAME, "organisation": ORG, "address": STREET, "email": EMAIL,
                                       "phone": PHONE, "region": REGION},
                     "media_metadata": {"original_video_uri": f"gs://{STORAGE}/o.mp4", "original_video_object": STORAGE,
                                        "device_model": NAME},
                     "clean_plate": {"stage_manifest_uri": f"gs://{STORAGE}/m.json"},
                     "website_scene_execution_authority": {"owner": {"user_id": USER, "organization_id": ORG},
                                                           "consent": {"accepted_by": USER, "rights_reference": TOKEN}},
                     "task_statement": SENTENCE, "capture_rights": _rights(CURRENT)}})
    _write(capture / "raw" / "capture_upload_complete.json", {
        "scene_id": SITE, "capture_id": CAPTURE, "raw_prefix": f"{CAPTURE_REL}/raw", "video_uri": f"{CAPTURE_REL}/{VIDEO}",
        "producer_delivery": {"object_name": f"{STORAGE}/delivery.json", "delivery_key": TOKEN}})
    _write(capture / "pipeline_staging_manifest.json", {"bucket": "PLANTBUCKET", "prefix": f"{CAPTURE_REL}/raw",
                                                        "objects": [{"name": f"{STORAGE}/{VIDEO}", "sha256": "sha256:0"}]})
    _write(capture / "pipeline_handoff.json", {"bucket": "PLANTBUCKET", "raw_prefix_uri": f"gs://{STORAGE}/raw"})


def _job(tmp_path: Path, clip: bytes, *, trial: str = "zero", simulation: bool = True,
         statement: str | None = CURRENT) -> JobInputs:
    root = tmp_path / "job"
    capture = root / CAPTURE_REL
    _write(capture / "pipeline" / "website_task_context.json", _context(statement))
    _write(capture / "pipeline_job_ledger.json", {"schema_version": "pipeline_job_ledger.v1", "status": "completed",
                                                  "output_commit_path": f"{HOST}/{STORAGE}", "producer_delivery_key": TOKEN})
    _write(capture / VIDEO, clip)
    # As the WebApp's browser upload writes it (browserCaptureManifest.ts): the consent the delivery gate reads.
    _write(capture / "raw" / "manifest.json", {"schema_version": "v1", "scene_id": SITE, "capture_id": CAPTURE,
                                               "site_submission_id": "req-1", "request_id": "req-1",
                                               "video_uri": f"{CAPTURE_REL}/{VIDEO}", "device_model": "browser_self_capture",
                                               "capture_rights": _rights(CURRENT)})
    _identity_files(capture)
    _pull_record(root, [""], _absent())
    prep = capture / PREP
    _write(prep / "preparation.json", {
        "intake_request": {"task": {"strategy": "pick_and_place"}, "owner": {"user_id": USER, "organization_id": ORG}},
        "subject": {"description": "tray", "aabb_min_xyz": [1.41421, 0.0, 0.9], "aabb_max_xyz": [1.81421, 0.3, 0.95]},
        "support": {"basis": "observed"},
        "destination": {"relation": "on", "visible_label": "back shelf", "position_world_m": [3.14159, 2.71828, 0.9],
                        "support": {"description": "back shelf"}},
        "registration": {"task_region": {"radius_m": 1.0, "trimmed_rmse_m": 0.072}},
        "coordinate_frame": {"scale_authority": "registration_estimate", "physical_scale_measured": False,
                             "task_coordinates": "Z_up_estimated_meters", "declared_up_axis": "Z"},
        "authoring_inputs": {"metric_envelope": {"maximum_dimension_relative_error": 0.25},
                             "source_frames": [{"path": _host(f"{STORAGE}/frame.jpg")}]},
        "thumbnail": {"path": _host(f"{STORAGE}.png")},
        "physics": {"bounds": {"mass_kg": [0.4, 1.8], "static_friction": [0.2, 0.9], "dynamic_friction": [0.15, 0.8]}}})
    _write(prep / "base_scene.json", {"up_axis": "Y", "scene_estimates": [{"code": "scene_estimate:level"}],
                                      "splat_path": _host(SPLAT), "collision_mesh_path": _host(COLLISION)})
    _write(capture / SPLAT, b"ply splat")
    _write(capture / COLLISION, b"glb mesh")
    _seal_first_look(capture)
    inputs = {"capture_dir": capture, "pull_root": root}
    if simulation:
        team = tmp_path / "team-scene"
        _write(team / "team_scene_inputs.v1.json", {"task_success_contract": _contract(), "scene_revision_digest": REVISION,
                                                    "owner_success_assumption": {"status": "assumed"},
                                                    "source_path": f"{HOST}/{STORAGE}"})
        _write(team / "native-task-packet" / "native_task_arena_scene_plan.v1.json",
               {"scene_id": SITE, "task_spec": {"task_success_contract": _contract()}, "launch_root": f"{HOST}/x"})
        _write(team / "native-task-packet" / "collision.usdc", b"usd collision")
        _write(team / "native-task-packet" / "scene_appearance.usdc", b"usd appearance")
        configured = tmp_path / "configured"
        _write(configured / "configured_scene_revision.v1.json", {"revision_digest": REVISION, "storage": LINK})
        configured.mkdir(exist_ok=True)
        with zipfile.ZipFile(configured / "configured_scene_bundle.v1.zip", "w") as bundle:
            for member, data in (("appearance.usdc", b"look"), ("collision.usdc", b"walls"), ("replacement.usdz", b"obj"),
                                 ("configured_scene_bundle_candidate.v1.json", json.dumps(
                                     {"scene_id": SITE, "capture_id": CAPTURE, "source": f"gs://{STORAGE}"}))):
                bundle.writestr(member, data)
        _write(configured / "replacement_asset.usdz", b"usdz")
        inputs.update(team_scene_dir=team, configured_scene_dir=configured)
    if trial != "none":
        succeeded = 0 if trial == "zero" else 3
        inputs["screening_publication"] = _write(tmp_path / "publication.json", {
            "site_id": SITE, "status": "completed", "source": {"capture_id": CAPTURE, "storage": f"{HOST}/{STORAGE}"},
            "candidates": [{"candidate_id": "pi05_droid", "display_name": "π0.5 DROID", "status": "completed",
                            "episodes_scored": 10, "episodes_succeeded": succeeded}]})
        _write(tmp_path / "clips" / "manifest.json", {"site_id": SITE, "capture_id": CAPTURE, "root": f"{HOST}/{STORAGE}"})
        _write(tmp_path / "clips" / "cell_runs" / "01" / f"{CAPTURE}-camera.mp4", b"clip")
        inputs["review_videos_dir"] = tmp_path / "clips"
    return JobInputs(**inputs)


@pytest.fixture
def job(tmp_path, clip_with_sound):
    return lambda **kwargs: _job(kwargs.pop("root", tmp_path), clip_with_sound, **kwargs)


@pytest.fixture(autouse=True)
def _catalog_copy(tmp_path_factory, monkeypatch):
    """The screening catalog loader refuses a group-writable file; read a private copy, whatever the checkout's umask."""
    copy_path = tmp_path_factory.mktemp("catalog") / "beta_policy_candidates.v1.json"
    shutil.copyfile(packet.POLICY_CATALOG_PATH, copy_path)
    copy_path.chmod(0o644)
    monkeypatch.setattr(packet, "POLICY_CATALOG_PATH", copy_path)
    yield
    for stop in _UNBLOCK:
        stop.set()
    _UNBLOCK.clear()


_UNBLOCK: list[threading.Event] = []


def _unblock_readers(path: Path) -> None:
    """Open the pipe's write end as soon as anyone reads it, so a reader that should never have opened it
    gets end-of-file at once and its test fails instead of hanging."""
    stop = threading.Event()
    _UNBLOCK.append(stop)

    def writer():
        while not stop.wait(0.02):
            try:
                os.close(os.open(path, os.O_WRONLY | os.O_NONBLOCK))
            except OSError:
                continue
    threading.Thread(target=writer, daemon=True).start()


def _out(tmp_path: Path, name: str = "out") -> Path:
    path = tmp_path / name
    path.mkdir()
    return path


def _owned_list(tmp_path: Path, ids: list[str]) -> Path:
    return _write(tmp_path / "owned.json", {"schema_version": "blueprint_owned_captures.v1",
                                            "captures": [{"capture_id": item} for item in ids]})


def _owned() -> OwnedCaptureApproval:
    return OwnedCaptureApproval(capture_id=CAPTURE, approved_by="owner", approval_evidence="Yes, share the reference job.",
                                approved_on="2026-11-12")


def _owned_basis(tmp_path: Path) -> dict:
    return {"owned": _owned(), "owned_list": _owned_list(tmp_path, [CAPTURE]), "now": NOW}


def _refusal(inputs: JobInputs, tmp_path: Path, **kwargs) -> JobPacketRefused:
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    with pytest.raises(JobPacketRefused) as refused:
        build_job_packet(inputs, output_root=out, **kwargs)
    assert list(out.iterdir()) == []
    return refused.value


def _refused(inputs: JobInputs, tmp_path: Path, **kwargs) -> str:
    refusal = _refusal(inputs, tmp_path, **kwargs)
    # Every refusal says which place or file it is about.
    assert str(refusal).startswith(refusal.code + ": ") and len(str(refusal)) > len(refusal.code) + 2
    return refusal.code


def _all_bytes(folder: Path) -> bytes:
    return b"\n".join([folder.name.encode()] + [path.relative_to(folder).as_posix().encode() + b"\n" + path.read_bytes()
                                                for path in sorted(folder.rglob("*")) if path.is_file()])


def _edit(path: Path, change) -> None:
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def _context_path(inputs: JobInputs) -> Path:
    return inputs.capture_dir / "pipeline" / "website_task_context.json"


def _record(inputs: JobInputs, change) -> None:
    _edit(inputs.pull_root / "pull_record.json", change)


def _present(inputs: JobInputs, relative: str) -> None:
    """The place is now pulled: it leaves the record's absent list."""
    place = f"{inputs.capture_dir.relative_to(inputs.pull_root).as_posix()}/{relative}"
    _record(inputs, lambda value: value["absent_on_host"].remove(place) if place in value["absent_on_host"] else None)


def _gone(inputs: JobInputs, relative: str) -> None:
    """The place is not on the host: removed here and recorded absent."""
    path = inputs.capture_dir / relative
    shutil.rmtree(path) if path.is_dir() else path.unlink()
    _record(inputs, lambda value: value["absent_on_host"].append(f"{CAPTURE_REL}/{relative}"))


def _sheet(manifest: dict) -> str:
    return (Path(manifest["packet_dir"]) / "JOB_SHEET.md").read_text()


# --- the one packet --------------------------------------------------------------------------------------

def test_the_packet_holds_the_whole_job(tmp_path, job, clip_with_sound) -> None:
    out = _out(tmp_path)
    manifest = build_job_packet(job(), output_root=out, now=NOW, operator="op")
    folder = Path(manifest["packet_dir"])
    assert re.fullmatch(r"packet-[0-9a-f]{16}", folder.name) and folder.name == f"packet-{manifest['packet_number']}"
    sheet = _sheet(manifest)
    assert f"Packet number: {manifest['packet_number']}" in sheet and TERMS_LINE in sheet
    assert f"> {SENTENCE}" in sheet and "Kind of job: move one object from one surface to another." in sheet
    assert "- The site's own success definition: Tray on the shelf" in sheet and "- where: Back room" in sheet
    assert "| UR5e | Universal Robots | fixed arm | fits on published figures | payload (unknown) |" in sheet
    assert "| UR10e | Universal Robots | fixed arm | does not fit | reach (failed) |" in sheet
    assert EARLY_TRIAL_LABEL in sheet and "| π0.5 DROID | completed | 10 | 0 of 10 |" in sheet
    assert "within 20 cm of the point (3.142, 2.718, 0.900) m, on the back shelf" in sheet
    assert "Carry distance: 0.43–0.76 m in a straight line" in sheet and "0.4–1.8 kg" in sheet
    assert "Reply to the person who sent you this packet" in sheet and "interested; not a fit; would need something" in sheet
    assert ("Status: In this packet: the site's description of the job, the video, without sound, the first look on "
            "paper, the 3D reconstruction of the site, the simulation of the site, the success rule and the early "
            "trial's result.") in sheet
    assert "everything here comes from one video and a simulation of it." in sheet
    paths = {row["path"] for row in manifest["files"]}
    assert paths == {"JOB_SHEET.md", "README.md", "files/video/video.mp4", "files/object/replacement_asset.usdz",
                     "files/reconstruction/atlas_depth_splat_levelled.ply",
                     "files/reconstruction/collider_conditioned.glb",
                     "files/simulation/configured-scene/appearance.usdc", "files/simulation/configured-scene/collision.usdc",
                     "files/simulation/configured-scene/replacement.usdz",
                     "files/simulation/team-scene/native-task-packet/collision.usdc",
                     "files/simulation/team-scene/native-task-packet/scene_appearance.usdc",
                     "files/task/success_rule.json", "files/early-trial/clips/clip-01.mp4"}
    on_disk = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()}
    assert on_disk == paths | {"manifest.json"}
    for row in manifest["files"]:
        data = (folder / row["path"]).read_bytes()
        assert row["size"] == len(data) and row["sha256"] == hashlib.sha256(data).hexdigest()
    assert json.loads((folder / "files/task/success_rule.json").read_text()) == {"criteria": _contract()["criteria"]}
    on_disk_manifest = json.loads((folder / "manifest.json").read_text())
    assert on_disk_manifest["permission_basis"] == {"rule": "site_sharing_permission", "statement_version": CURRENT}
    assert on_disk_manifest["first_look_source"] == "pipeline_robot_fit_table"
    readme = (folder / "README.md").read_text()
    assert "Isaac Sim 6.0.1" in readme and "Z_up_estimated_meters" in readme and "sound removed" in readme
    # The operator record sits beside the folder, never inside it, and ties the packet to the job.
    record = json.loads(Path(manifest["operator_record"]).read_text())
    assert Path(manifest["operator_record"]).parent == out
    assert record["packet_folder"] == folder.name and record["packet_number"] == manifest["packet_number"]
    assert record["job"] == {"request_id": "req-1", "scene_id": SITE, "capture_id": CAPTURE}
    assert record["run_by"] == "op" and "pipeline/website_task_context.json" in record["input_sha256"]
    assert record["video_sound_removal"]["output_audio_stream_count"] == 0


def test_the_video_goes_in_without_sound_and_its_picture_unchanged(tmp_path, job) -> None:
    inputs = job()
    source = inputs.capture_dir / VIDEO
    assert _streams(source, "a") == ["aac"]
    copied = Path(build_job_packet(inputs, output_root=_out(tmp_path))["packet_dir"]) / "files/video/video.mp4"
    assert _streams(copied, "a") == [] and _streams(copied, "v") == _streams(source, "v") == ["mpeg4"]


def test_no_sound_tool_is_a_refusal_never_a_copy_with_sound(tmp_path, job, monkeypatch) -> None:
    inputs = job()
    real = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None if name in {"ffmpeg", "ffprobe"} else real(name, *a, **k))
    assert _refused(inputs, tmp_path) == "job_packet_video_sound_tool_missing"


def test_a_strip_that_leaves_sound_is_a_refusal_and_leaves_nothing(tmp_path, job, monkeypatch) -> None:
    import blueprint_pipeline.capture_audio_minimization as audio

    def keeps_sound(source, destination, **kwargs):
        shutil.copyfile(source, destination)
        return {"audio_removed": False, "evidence": "ffprobe_output_has_audio_stream", "output_audio_stream_count": 1,
                "reason": "audio_stream_present_after_encode"}
    monkeypatch.setattr(audio, "write_audio_free_video", keeps_sound)
    refusal = _refusal(job(), tmp_path)
    assert refusal.code == "job_packet_video_sound_not_removed" and "walkthrough.mp4" in str(refusal)


def test_nothing_that_says_who_or_where_the_site_is_reaches_the_packet(tmp_path, job) -> None:
    inputs = job()
    blob = _all_bytes(Path(build_job_packet(inputs, output_root=_out(tmp_path))["packet_dir"])).decode("latin-1")
    for planted in (*FORBIDDEN, str(tmp_path)):
        assert planted not in blob, planted
    assert SENTENCE in blob  # the site's own sentence is the site's text, included as written


def test_a_packet_is_never_held_for_the_simulation_the_success_rule_or_the_trial(tmp_path, job) -> None:
    manifest = build_job_packet(job(trial="none", simulation=False), output_root=_out(tmp_path))
    sheet, readme = _sheet(manifest), (Path(manifest["packet_dir"]) / "README.md").read_text()
    assert ("The simulation of the site with the job's object in it is not finished; it will follow with the success "
            "rule. No early trial of openly available robot software has been added.") in sheet
    assert "No success rule is in this packet yet" in sheet and "No early trial of openly available robot software is in this packet." in sheet
    assert "everything here comes from one video." in sheet and "a simulation of it" not in sheet
    assert "team-scene" not in readme and "files/reconstruction/" in readme


def test_the_sheet_says_only_what_was_checked(tmp_path, job) -> None:
    sheet = _sheet(build_job_packet(job(), output_root=_out(tmp_path)))
    assert ("Checked against published specifications and the scene's estimated geometry: reach, payload. "
            "It is not a test of any robot.") in sheet
    assert "Cycle time was not checked: the site gave no target." in sheet
    assert "Payload is unknown for 1 of the 2 robots listed: the object was not weighed" in sheet


def test_a_missing_video_is_said_plainly(tmp_path, job) -> None:
    inputs = job()
    _gone(inputs, VIDEO)
    sheet = _sheet(build_job_packet(inputs, output_root=_out(tmp_path)))
    assert "The video is not in this packet." in sheet and "Not in it: the video." in sheet


def test_a_nonzero_trial_and_a_zero_count_are_shown(tmp_path, job) -> None:
    assert "3 of 10" in _sheet(build_job_packet(job(trial="three"), output_root=_out(tmp_path)))


def test_kind_of_job_falls_back_to_fixed_words() -> None:
    assert packet.kind_of_job(None, {}) == "a pick-up-and-put-down job"
    assert packet.kind_of_job(None, {"task_family": SENTENCE}) == "a pick-up-and-put-down job"
    assert packet.kind_of_job({"intake_request": {"task": {"strategy": "pick_and_place"}},
                               "destination": {"interior_bounds_world_m": {}}}, {}) == "put one object into a container"


def test_check_names_match_the_fit_table_writer() -> None:
    source = (Path(packet.__file__).parent / "website_robot_fit.py").read_text()
    assert set(re.findall(r'_check\("([a-z_]+)"', source)) == set(packet.CHECK_WORDS)
    from blueprint_pipeline.beta_policy_screening_result import CANDIDATE_STATUSES
    from blueprint_pipeline.website_robot_fit import VERDICTS
    assert set(VERDICTS) == set(packet.VERDICT_WORDS) and CANDIDATE_STATUSES == set(packet.STATUS_WORDS)


# --- the first look: followed through the handoff's pointer and checked ------------------------------------

def test_the_first_look_is_read_from_the_file_the_handoff_points_to(tmp_path, job) -> None:
    inputs = job()
    handoff = json.loads((inputs.capture_dir / PREP / "handoff.json").read_text())
    assert "rows" not in handoff["robot_fit_table"]  # the host's shape: a pointer, the rows in their own file
    assert "| UR5e |" in _sheet(build_job_packet(inputs, output_root=_out(tmp_path)))


@pytest.mark.parametrize("case", ["table_edited", "pointer_digest", "outside_capture", "other_preparation",
                                  "context_changed", "not_an_estimate"])
def test_a_first_look_that_cannot_be_checked_is_not_shown_or_claimed(tmp_path, job, case) -> None:
    inputs = job()
    handoff_path, table_path = inputs.capture_dir / PREP / "handoff.json", inputs.capture_dir / FIT
    if case == "table_edited":
        _edit(table_path, lambda value: value["rows"].pop())
    elif case == "pointer_digest":
        _edit(handoff_path, lambda value: value["robot_fit_table"].update(digest="sha256:" + "0" * 64))
        _edit(handoff_path, lambda value: _sealed(value, "digest"))
    elif case == "outside_capture":
        _edit(handoff_path, lambda value: value["robot_fit_table"].update(path=f"{HOST}/scenes/{SITE}/elsewhere.json"))
        _edit(handoff_path, lambda value: _sealed(value, "digest"))
    elif case == "other_preparation":
        _seal_first_look(inputs.capture_dir, table_change=lambda value: value.update(preparation_digest="sha256:" + "1" * 64))
    elif case == "context_changed":
        _edit(_context_path(inputs), lambda value: _sealed(value.update(description="Another job") or value, "context_digest"))
    else:
        _seal_first_look(inputs.capture_dir, table_change=lambda value: value.update(evidence_label="measured"))
    manifest = build_job_packet(inputs, output_root=_out(tmp_path))
    sheet = _sheet(manifest)
    assert "| UR5e |" not in sheet and manifest["first_look_source"] == "none"
    expected = ("The first look is not shown: its table did not match the record that names it" if case != "outside_capture"
                else "The first look is not shown")
    assert expected in sheet and "Not in it: the first look on paper." in sheet


def test_a_first_look_recorded_absent_on_the_host_is_said_plainly(tmp_path, job) -> None:
    inputs = job()
    _gone(inputs, FIT)
    manifest = build_job_packet(inputs, output_root=_out(tmp_path))
    assert "No first look on paper is in this packet yet." in _sheet(manifest) and manifest["first_look_source"] == "none"


def test_fit_rows_take_names_from_the_catalog_and_leave_out_the_rest(tmp_path, job) -> None:
    rows = _fit_rows()
    rows += [{**rows[0], "model_id": "marigold_arm"}, {**rows[1], "checks": [{"check": "reach to the shelf",
                                                                             "result": "fail"}]}, rows[0]]
    inputs = job()
    _seal_first_look(inputs.capture_dir, rows=rows)
    sheet = _sheet(build_job_packet(inputs, output_root=_out(tmp_path)))
    assert sheet.count("| UR5e |") == 1 and "Some rows were left out (3)" in sheet
    # A verdict outside the writer's set makes the whole table unverified, as the Pipeline's own reader treats it.
    _seal_first_look(inputs.capture_dir, rows=[*rows, {**rows[0], "verdict": "maybe"}])
    assert "The first look is not shown" in _sheet(build_job_packet(inputs, output_root=_out(tmp_path, "again")))


def test_repeats_are_left_out_and_large_counts_are_capped(tmp_path, job) -> None:
    inputs = job()
    _edit(inputs.screening_publication, lambda value: value.update(
        candidates=value["candidates"] * 3 + [{"candidate_id": "junk"}] * 7))
    _seal_first_look(inputs.capture_dir, rows=_fit_rows() * 4 + [{"model_id": "x", "verdict": "unknown"}] * 42)
    sheet = _sheet(build_job_packet(inputs, output_root=_out(tmp_path)))
    assert sheet.count("| π0.5 DROID |") == 1 and sheet.count("| UR5e |") == 1 and sheet.count("| UR10e |") == 1
    assert "Some rows were left out (9)" in sheet and "Some rows were left out (more than 20)" in sheet
    assert "48" not in sheet


# --- who may get a packet -------------------------------------------------------------------------------

def test_the_current_statement_is_the_only_one_that_shares() -> None:
    assert packet.SHARING_PERMISSION_STATEMENT_VERSIONS == (CURRENT,)


@pytest.mark.parametrize("statement", ["2026-09-18.v1", OLDER, "2026-11-11.v3", None, "2026-11-12.V4", " 2026-11-12.v4"])
def test_a_job_under_any_older_statement_is_refused_until_the_site_accepts_the_current_one(tmp_path, job, statement) -> None:
    refusal = _refusal(job(statement=statement), tmp_path)
    assert refusal.code == "job_packet_sharing_statement_not_accepted"
    assert f"the site must accept the current statement ({CURRENT}) first" in str(refusal)
    assert "capture_rights.statement_version" in str(refusal)


def test_committed_owned_capture_list_is_empty() -> None:
    assert packet.owned_capture_ids() == set()


@pytest.mark.parametrize(("ids", "approval", "code"), [
    ([], _owned(), "job_packet_capture_not_blueprint_owned"),
    (["walkthrough-other"], _owned(), "job_packet_capture_not_blueprint_owned"),
    ([CAPTURE, "walkthrough-other"], OwnedCaptureApproval("walkthrough-other", "owner", "yes", "2026-11-12"),
     "job_packet_owned_capture_id_mismatch"),
    ([CAPTURE], OwnedCaptureApproval(CAPTURE, "owner", " ", "2026-11-12"), "job_packet_owned_capture_approval_missing"),
    ([CAPTURE], OwnedCaptureApproval(CAPTURE, "owner", "yes"), "job_packet_owned_capture_approval_missing"),
    ([CAPTURE], OwnedCaptureApproval(CAPTURE, "owner", "yes", "last week"), "job_packet_owned_capture_approval_missing"),
])
def test_blueprint_owned_capture_needs_the_list_and_the_owners_words(tmp_path, job, ids, approval, code) -> None:
    assert _refused(job(statement=OLDER), tmp_path, owned=approval, owned_list=_owned_list(tmp_path, ids), now=NOW) == code


@pytest.mark.parametrize("raw", ["{", json.dumps([CAPTURE]), json.dumps({"schema_version": "blueprint_owned_captures.v1",
                                                                         "captures": [[CAPTURE]]}),
                                 json.dumps({"schema_version": "blueprint_owned_captures.v1", "captures": CAPTURE}),
                                 json.dumps({"schema_version": "blueprint_owned_captures.v1", "captures": [{"capture_id": "*"}]})])
def test_a_malformed_allow_list_is_a_clean_refusal(tmp_path, job, raw) -> None:
    path = _write(tmp_path / "owned.json", raw)
    assert _refused(job(statement=OLDER), tmp_path, owned=_owned(), owned_list=path, now=NOW) == \
        "job_packet_owned_capture_list_invalid"


def test_an_allow_listed_capture_records_the_approval_beside_the_folder_only(tmp_path, job) -> None:
    manifest = build_job_packet(job(statement=OLDER), output_root=_out(tmp_path), **_owned_basis(tmp_path))
    assert manifest["permission_basis"] == {"rule": "blueprint_owned_capture"}
    basis = json.loads(Path(manifest["operator_record"]).read_text())["permission_basis"]
    assert basis["approval_evidence"] == "Yes, share the reference job." and basis["approved_on"] == "2026-11-12"
    assert basis["capture_id"] == CAPTURE


@pytest.mark.parametrize("approved_on", ["2026-11-13", "2026-08-13", "2020-01-01"])
def test_an_approval_dated_in_the_future_or_over_90_days_ago_is_refused(tmp_path, job, approved_on) -> None:
    approval = OwnedCaptureApproval(CAPTURE, "owner", "yes", approved_on)
    assert _refused(job(statement=OLDER), tmp_path, owned=approval, now=NOW,
                    owned_list=_owned_list(tmp_path, [CAPTURE])) == "job_packet_owned_capture_approval_out_of_date"
    assert build_job_packet(job(root=tmp_path / "ok", statement=OLDER), output_root=_out(tmp_path, "ok-out"), now=NOW,
                            owned=OwnedCaptureApproval(CAPTURE, "owner", "yes", "2026-08-14"),
                            owned_list=_owned_list(tmp_path, [CAPTURE]))["permission_basis"]["rule"] == "blueprint_owned_capture"


# --- refusals: withdrawn, unconsented, ended or closed ------------------------------------------------------

def _rights_edit(**fields):
    return lambda inputs: _edit(_context_path(inputs), lambda value: value["capture_rights"].update(**fields))


def _file(relative, value):
    def change(inputs):
        _write(inputs.capture_dir / relative, value)
        _present(inputs, relative)
    return change


def _drop_status(inputs):
    _edit(_context_path(inputs), lambda value: value["capture_rights"].pop("consent_status"))


def _drop_revoked(inputs):
    _edit(_context_path(inputs), lambda value: value["capture_rights"].pop("consent_revoked"))


def _ledger(**fields):
    return _file("pipeline_job_ledger.json", {"status": "failed_retryable", **fields})


def _tombstone(where):
    def change(inputs):
        site = inputs.capture_dir.parent.parent
        target = {"own": site, "bucket": site.parents[1] / "bucketB" / "scenes" / site.name,
                  "hidden": site.parents[1] / ".scene-workspace-inventory" / "scenes" / site.name}[where]
        _write(target / "website_withdrawal" / "tombstone.json", "{}")
    return change


def _fifo(relative):
    def change(inputs):
        path = (inputs.capture_dir.parent.parent / relative if relative.startswith("website_withdrawal")
                else inputs.capture_dir / relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        os.mkfifo(path)
        _unblock_readers(path)
    return change


def _record_edit(change):
    return lambda inputs: _record(inputs, change)


def _takedown_open(inputs):
    _write(inputs.capture_dir / "pipeline" / "consent_takedown" / "takedown_manifest.json", {"takedown_open": True})
    _present(inputs, "pipeline/consent_takedown")


def _both(*changes):
    return lambda inputs: [change(inputs) for change in changes]


REFUSALS = {
    "takedown_manifest_open": (_takedown_open, "job_packet_delivery_gate_blocked_open_consent_takedown"),
    "consent_unverifiable": (lambda inputs: _gone(inputs, "raw/manifest.json"),
                             "job_packet_delivery_gate_blocked_consent_state_unverifiable"),
    "consent_file_bom": (_file("raw/rights_consent.json", "﻿" + json.dumps({"consent_status": "revoked"})),
                         "job_packet_consent_state_unreadable"),
    "consent_file_truncated": (_file("raw/rights_consent.json", '{"consent_status":"revo'), "job_packet_consent_state_unreadable"),
    "consent_file_list": (_file("rights_consent.json", '[{"consent_status":"revoked"}]'), "job_packet_consent_state_unreadable"),
    "consent_file_duplicate_keys": (_file("rights_consent.json", '{"consent_status":"revoked","consent_status":"granted"}'),
                                    "job_packet_consent_state_unreadable"),
    "first_active_second_revoked": (_both(_file("raw/rights_consent.json", {"consent_status": "granted"}),
                                          _file("raw/manifest.json", {"consent_status": "revoked"})),
                                    "job_packet_capture_withdrawn"),
    "consent_expired": (_rights_edit(consent_expires_at="2020-01-01T00:00:00Z"), "job_packet_consent_expired"),
    "ledger_code_as_key": (_file("pipeline_job_ledger.json", {"status": "terminal_success",
                                                             "terminal": {"website_run_closed_by_owner": True}}),
                           "job_packet_job_closed_by_owner"),
    "ledger_code_in_sentence": (_ledger(last_error="WebApp said consent_expired for this job"),
                                "job_packet_capture_authority_ended"),
    "ledger_code_as_status": (_file("pipeline_job_ledger.json", {"status": "source_revoked"}),
                              "job_packet_capture_authority_ended"),
    "tombstone_dir_other_bucket": (lambda inputs: (inputs.pull_root / "bkt" / "scenes" / SITE / "website_withdrawal"
                                                   / "tombstone.json").mkdir(parents=True), "job_packet_capture_withdrawn"),
    # Not a bucket, but a tombstone anywhere still refuses: looking costs nothing and fails closed.
    "tombstone_hidden_folder": (_tombstone("hidden"), "job_packet_capture_withdrawn"),
    "no_pull_record": (lambda inputs: (inputs.pull_root / "pull_record.json").unlink(), "job_packet_withdrawal_state_not_pulled"),
    "withdrawal_not_pulled": (_record_edit(lambda value: value["absent_on_host"].remove(f"scenes/{SITE}/website_withdrawal")),
                              "job_packet_withdrawal_state_not_pulled"),
    "takedown_not_pulled": (_record_edit(lambda value: value["absent_on_host"].remove(
        f"{CAPTURE_REL}/pipeline/consent_takedown")), "job_packet_withdrawal_state_not_pulled"),
    "recorded_absent_but_pulled": (lambda inputs: (inputs.capture_dir.parent.parent / "website_withdrawal").mkdir(),
                                   "job_packet_withdrawal_state_not_pulled"),
    "bucket_not_in_record": (lambda inputs: (inputs.pull_root / "bkt" / "scenes").mkdir(parents=True),
                             "job_packet_withdrawal_state_not_pulled"),
    "status_withdrawn": (_rights_edit(consent_status="withdrawn"), "job_packet_capture_withdrawn"),
    "status_rescinded": (_rights_edit(consent_status="rescinded"), "job_packet_capture_withdrawn"),
    "status_pending": (_rights_edit(consent_status="pending"), "job_packet_consent_not_active"),
    "status_missing": (_drop_status, "job_packet_consent_not_active"),
    "camel_revoked": (_rights_edit(consentRevoked=True), "job_packet_capture_withdrawn"),
    "camel_status": (_rights_edit(consentStatus="revoked"), "job_packet_capture_withdrawn"),
    "revoked_at": (_rights_edit(consent_revoked_at="2026-11-10T00:00:00Z"), "job_packet_capture_withdrawn"),
    "rights_consent_file": (_file("raw/rights_consent.json", {"consent_status": "revoked"}), "job_packet_capture_withdrawn"),
    "raw_manifest": (_file("raw/manifest.json", {"consent_status": "withdrawn"}), "job_packet_capture_withdrawn"),
    "descriptor": (_file("capture_descriptor.json", {"consent_revoked": True}), "job_packet_capture_withdrawn"),
    "pending_consent_file": (_file("rights_consent.json", {"consent_status": "pending"}), "job_packet_consent_not_active"),
    "rights_not_granted": (_rights_edit(derived_scene_generation_allowed=False), "job_packet_capture_rights_not_granted"),
    "rights_granted_as_text": (_rights_edit(derived_scene_generation_allowed="true"), "job_packet_capture_rights_not_granted"),
    "revoked_flag_absent": (_drop_revoked, "job_packet_capture_rights_not_granted"),
    "revoked_flag_not_false": (_rights_edit(consent_revoked=None), "job_packet_capture_rights_not_granted"),
    "tombstone_own_site": (_tombstone("own"), "job_packet_capture_withdrawn"),
    "tombstone_other_bucket": (_tombstone("bucket"), "job_packet_capture_withdrawn"),
    "tombstone_pipe": (_fifo("website_withdrawal/tombstone.json"), "job_packet_capture_withdrawn"),
    "consent_file_pipe": (_fifo("raw/manifest.json"), "job_packet_consent_state_unreadable"),
    "terminal_authority": (_file("pipeline_job_ledger.json", {"status": "terminal_authority_ended",
                                                              "terminal_code": "consent_expired"}),
                           "job_packet_capture_authority_ended"),
    "receipt_authority": (_file("pipeline_job_terminal_receipt.json", {"status": "authority_ended", "code": "source_revoked"}),
                          "job_packet_capture_authority_ended"),
    "retryable_source_revoked": (_ledger(last_error="website_control_task-context_http_409:source_revoked"),
                                 "job_packet_capture_authority_ended"),
    "retryable_consent_expired": (_ledger(last_error="website_control_scene-sponsorship_http_409:consent_expired"),
                                  "job_packet_capture_authority_ended"),
    "retryable_no_grant": (_ledger(last_error="stage failed: website_control_task-context_http_409:"
                                              "website_scene_job_no_grant_ended"), "job_packet_capture_authority_ended"),
    "terminal_owner_closed": (_file("pipeline_job_ledger.json", {"status": "terminal_authority_ended",
                                                                 "terminal_code": "website_run_closed_by_owner"}),
                              "job_packet_job_closed_by_owner"),
    "retryable_owner_closed": (_ledger(last_error="website_control_task-context_http_409:website_run_closed_by_owner"),
                               "job_packet_job_closed_by_owner"),
    "ledger_missing": (lambda inputs: _gone(inputs, "pipeline_job_ledger.json"), "job_packet_job_ledger_missing"),
    "ledger_unreadable": (_file("pipeline_job_ledger.json", '{"status": "terminal_success"} xx'), "job_packet_job_state_unreadable"),
    "ledger_duplicate_keys": (_file("pipeline_job_ledger.json", '{"status":"terminal_authority_ended","status":"terminal_success"}'),
                              "job_packet_job_state_unreadable"),
    "ledger_bom": (_file("pipeline_job_ledger.json", '﻿{"status":"terminal_success"}'), "job_packet_job_state_unreadable"),
    "ledger_corrupt": (_file("pipeline_job_ledger.json", {"status": "corrupt"}), "job_packet_job_state_unreadable"),
    "ledger_pipe": (_fifo("pipeline_job_ledger.json"), "job_packet_job_state_unreadable"),
    "receipt_garbage": (_file("pipeline_job_terminal_receipt.json", "nope"), "job_packet_job_state_unreadable"),
}


@pytest.mark.parametrize("basis", ["site_statement", "blueprint_owned"])
@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_withdrawn_unconsented_ended_or_closed_jobs_are_refused(tmp_path, job, basis, case) -> None:
    change, code = REFUSALS[case]
    inputs = job(statement=CURRENT if basis == "site_statement" else OLDER)
    change(inputs)
    assert _refused(inputs, tmp_path, **(_owned_basis(tmp_path) if basis == "blueprint_owned" else {})) == code


def test_the_delivery_gate_is_asked_for_a_job_packet(tmp_path, job, monkeypatch) -> None:
    import blueprint_pipeline.consent_takedown as takedown
    asked = []
    real = takedown.evaluate_delivery_time_takedown_gate
    monkeypatch.setattr(takedown, "evaluate_delivery_time_takedown_gate",
                        lambda **kwargs: asked.append(kwargs) or real(**kwargs))
    inputs = job()
    build_job_packet(inputs, output_root=_out(tmp_path))
    assert asked == [{"capture_root": inputs.capture_dir, "surface": "job_packet"}]


def test_the_pipelines_withdrawal_readers_are_obeyed(tmp_path, job, monkeypatch) -> None:
    import blueprint_pipeline.screening_site_withdrawal as screening
    import blueprint_pipeline.website_atlas_preview_store as preview
    inputs = job()
    monkeypatch.setattr(screening, "site_withdrawal_state", lambda *args, **kwargs: "unknown")
    assert _refused(inputs, tmp_path / "a") == "job_packet_withdrawal_state_unknown"
    monkeypatch.undo()
    seen = []
    monkeypatch.setattr(preview, "capture_withdrawn", lambda root: seen.append(root) or True)
    assert _refused(inputs, tmp_path / "b") == "job_packet_capture_withdrawn" and seen == [inputs.capture_dir]


def test_a_capture_pulled_without_the_site_root_layout_is_refused(tmp_path, job) -> None:
    inputs = job()
    flat = tmp_path / "flat" / "x" / "y" / CAPTURE
    flat.parent.mkdir(parents=True)
    inputs.capture_dir.rename(flat)
    assert _refused(JobInputs(capture_dir=flat), tmp_path) == "job_packet_pull_root_required"
    assert _refused(JobInputs(capture_dir=flat, pull_root=tmp_path / "flat"), tmp_path) == "job_packet_withdrawal_state_unknown"


def test_a_capture_under_a_bucket_is_checked_across_every_bucket(tmp_path, job) -> None:
    inputs = job()
    root = tmp_path / "pulled"
    moved = root / "bucketA" / "scenes" / SITE
    moved.parent.mkdir(parents=True)
    inputs.capture_dir.parent.parent.rename(moved)
    capture = moved / "captures" / CAPTURE
    _pull_record(root, ["bucketA", "bucketB"], [*_absent("bucketA/"), f"bucketB/scenes/{SITE}/website_withdrawal"])
    build_job_packet(JobInputs(capture_dir=capture, pull_root=root), output_root=_out(tmp_path))
    _write(root / "bucketB" / "scenes" / SITE / "website_withdrawal" / "tombstone.json", "{}")
    assert _refused(JobInputs(capture_dir=capture, pull_root=root), tmp_path / "again") == "job_packet_capture_withdrawn"


def test_a_hidden_folder_is_never_a_bucket_to_pull(tmp_path, job) -> None:
    # The host's pull root holds hidden spools and the root-only `.scene-workspace-inventory` cache beside the
    # buckets. Listed or present, none of them has to be pulled or recorded absent.
    inputs = job()
    _record(inputs, lambda value: value["buckets"].extend([".scene-workspace-inventory", ".pubsub_retry_deferrals"]))
    (inputs.pull_root / ".scene-workspace-inventory" / "scenes").mkdir(parents=True)
    build_job_packet(inputs, output_root=_out(tmp_path))
    from blueprint_pipeline.site_roots import is_bucket
    assert not is_bucket(".scene-workspace-inventory") and not is_bucket("scenes") and is_bucket("_pipeline_failures")


# --- a pull cut short: every file the tool reads is pulled or recorded absent -------------------------------

@pytest.mark.parametrize(("relative", "code"), [
    ("raw/rights_consent.json", "job_packet_withdrawal_state_not_pulled"),
    ("rights_consent.json", "job_packet_withdrawal_state_not_pulled"),
    ("raw/manifest.json", "job_packet_withdrawal_state_not_pulled"),
    ("capture_descriptor.json", "job_packet_withdrawal_state_not_pulled"),
    ("pipeline_job_terminal_receipt.json", "job_packet_input_not_pulled"),
    ("pipeline_job_ledger.json", "job_packet_input_not_pulled"),
    (f"{PREP}/handoff.json", "job_packet_input_not_pulled"),
    (f"{PREP}/preparation.json", "job_packet_input_not_pulled"),
    (f"{PREP}/revisions", "job_packet_input_not_pulled"),
    (f"{PREP}/base_scene.json", "job_packet_input_not_pulled"),
    (FIT, "job_packet_input_not_pulled"),
    (VIDEO, "job_packet_input_not_pulled"),
    (SPLAT, "job_packet_input_not_pulled"),
    (COLLISION, "job_packet_input_not_pulled"),
])
def test_a_file_neither_pulled_nor_recorded_absent_is_refused_by_name(tmp_path, job, relative, code) -> None:
    inputs = job()
    path = inputs.capture_dir / relative
    if path.exists():
        path.unlink()
    _present(inputs, relative)
    refusal = _refusal(inputs, tmp_path)
    assert refusal.code == code and f"pull {CAPTURE_REL}/{relative}, or record it absent" in str(refusal)


# --- input binding and files ------------------------------------------------------------------------------

def _other_video(inputs: JobInputs) -> JobInputs:
    other = _write(inputs.capture_dir.parents[4] / "other" / "theirs.mp4", b"OTHER SITE VIDEO")
    return JobInputs(**{**inputs.__dict__, "video": other})


def _clips_of(**fields):
    def change(inputs):
        _write(inputs.review_videos_dir / "manifest.json", {"site_id": SITE, "capture_id": CAPTURE, **fields})
    return change


NOT_THIS_JOB = {
    "video_outside": _other_video,
    "clips_other_site": _clips_of(site_id="site-OTHER"),
    "clips_other_capture": _clips_of(capture_id="walkthrough-OTHER"),
    "clips_without_manifest": lambda inputs: (inputs.review_videos_dir / "manifest.json").unlink(),
    "team_other_site": lambda inputs: _edit(inputs.team_scene_dir / "native-task-packet" / "native_task_arena_scene_plan.v1.json",
                                            lambda value: value.update(scene_id="site-OTHER")),
    "contract_other_site": lambda inputs: _edit(inputs.team_scene_dir / "team_scene_inputs.v1.json",
                                                lambda value: value["task_success_contract"]["scope"].update(site_id="site-OTHER")),
    "configured_other_revision": lambda inputs: _edit(inputs.configured_scene_dir / "configured_scene_revision.v1.json",
                                                      lambda value: value.update(revision_digest="sha256:" + "e" * 64)),
    "publication_other_site": lambda inputs: _edit(inputs.screening_publication, lambda value: value.update(site_id="site-OTHER")),
}


@pytest.mark.parametrize("case", sorted(NOT_THIS_JOB))
def test_a_packet_takes_only_this_jobs_files(tmp_path, job, case) -> None:
    inputs = job()
    inputs = NOT_THIS_JOB[case](inputs) or inputs
    assert _refused(inputs, tmp_path) == "job_packet_input_not_this_job"


def test_packet_id_is_stable_for_the_same_inputs(tmp_path, job) -> None:
    inputs = job()
    first = build_job_packet(inputs, output_root=_out(tmp_path, "a"), now=NOW)
    second = build_job_packet(inputs, output_root=_out(tmp_path, "b"), now=NOW)
    assert first["packet_id"] == second["packet_id"] and first["packet_number"] != second["packet_number"]
    third = build_job_packet(job(root=tmp_path / "t", trial="three"), output_root=_out(tmp_path, "c"), now=NOW)
    assert third["packet_id"] != first["packet_id"]


@pytest.mark.parametrize("where", ["team", "raw"])
def test_a_symlink_out_of_the_inputs_is_refused(tmp_path, job, where) -> None:
    inputs = job()
    secret = _write(tmp_path / "elsewhere" / "secret.usdc", b"not this job")
    link = (inputs.team_scene_dir / "native-task-packet" / "extra.usdc" if where == "team"
            else inputs.capture_dir / VIDEO)
    if link.exists():
        link.unlink()
    link.symlink_to(secret)
    assert _refused(inputs, tmp_path) == "job_packet_input_symlink_outside_inputs"


def test_a_hard_link_or_a_pipe_where_a_file_is_expected_is_refused(tmp_path, job) -> None:
    inputs = job()
    os.link(_write(tmp_path / "outside.usdc", b"OTHER SITE"), inputs.team_scene_dir / "native-task-packet" / "x.usdc")
    assert _refused(inputs, tmp_path) == "job_packet_input_hard_linked"
    (inputs.team_scene_dir / "native-task-packet" / "x.usdc").unlink()
    (inputs.capture_dir / VIDEO).unlink()
    os.mkfifo(inputs.capture_dir / VIDEO)
    assert _refused(inputs, tmp_path) == "job_packet_input_not_a_regular_file"


def test_an_output_root_that_is_a_link_is_refused(tmp_path, job) -> None:
    real = _out(tmp_path, "real")
    (tmp_path / "linked").symlink_to(real)
    with pytest.raises(JobPacketRefused, match="job_packet_output_root_invalid"):
        build_job_packet(job(), output_root=tmp_path / "linked")
    assert list(real.iterdir()) == []


def test_two_files_for_one_packet_path_are_refused() -> None:
    rows = [(Path("a"), Path("a/x"), "files/object/x.usdz"), (Path("b"), Path("b/x"), "files/object/x.usdz")]
    with pytest.raises(JobPacketRefused, match="job_packet_file_name_collision"):
        packet.unique_destinations(rows)


def test_a_failure_while_writing_leaves_nothing_behind(tmp_path, job, monkeypatch) -> None:
    calls = []

    def failing_copy(source, target):
        calls.append(target)
        if len(calls) == 3:
            raise OSError("disk full")
        Path(target).write_bytes(Path(source).read_bytes())
    monkeypatch.setattr(packet.shutil, "copyfile", failing_copy)
    out = _out(tmp_path)
    with pytest.raises(OSError):
        build_job_packet(job(), output_root=out)
    assert len(calls) == 3 and list(out.iterdir()) == []


def test_the_command_writes_and_refuses(tmp_path, job, capsys) -> None:
    spec = importlib.util.spec_from_file_location(
        "export_job_packet", Path(__file__).resolve().parents[1] / "scripts" / "export_job_packet.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    inputs = job()
    common = ["--pull-root", str(inputs.pull_root), "--capture-dir", str(inputs.capture_dir), "--operator", "op"]
    assert script.main([*common, "--output-root", str(_out(tmp_path))]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "written" and Path(result["operator_record"]).is_file()
    _edit(_context_path(inputs), lambda value: value["capture_rights"].update(statement_version=OLDER))
    assert script.main([*common, "--output-root", str(_out(tmp_path, "o2"))]) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["code"] == "job_packet_sharing_statement_not_accepted" and CURRENT in refused["detail"]
    with pytest.raises(SystemExit):
        script.main([*common, "--output-root", str(tmp_path / "o2"), "--blueprint-owned-capture",
                     "--capture-id", CAPTURE, "--approved-by", "owner", "--approval-evidence", "yes"])


def test_the_operator_record_keeps_the_pull_record_whole_and_outside_the_packet(tmp_path, job) -> None:
    inputs = job()
    record_path = inputs.pull_root / "pull_record.json"
    supplied = json.loads(record_path.read_text())
    manifest = build_job_packet(inputs, output_root=_out(tmp_path))
    record = json.loads(Path(manifest["operator_record"]).read_text())
    assert record["pull_record"] == supplied
    assert record["pull_record_sha256"] == hashlib.sha256(record_path.read_bytes()).hexdigest()
    blob = _all_bytes(Path(manifest["packet_dir"])).decode("latin-1")
    assert "absent_on_host" not in blob and "website_withdrawal" not in blob and "consent_takedown" not in blob


def test_duplicate_keys_in_the_task_context_are_refused(tmp_path, job) -> None:
    inputs = job()
    path = _context_path(inputs)
    path.write_text(path.read_text()[:-1] + ', "capture_id": "walkthrough-other"}')
    assert _refused(inputs, tmp_path) == "job_packet_input_unreadable"


def test_a_group_writable_catalog_is_a_clean_refusal(tmp_path, job) -> None:
    packet.POLICY_CATALOG_PATH.chmod(0o664)
    assert _refused(job(), tmp_path) == "job_packet_policy_catalog_unreadable"


@pytest.mark.parametrize(("scored", "succeeded"), [(0, 0), (1001, 3), (5, 6), (True, 0), (10, -1)])
def test_trial_counts_out_of_bounds_are_left_out(tmp_path, job, scored, succeeded) -> None:
    inputs = job()
    _edit(inputs.screening_publication, lambda value: value["candidates"][0].update(
        episodes_scored=scored, episodes_succeeded=succeeded))
    sheet = _sheet(build_job_packet(inputs, output_root=_out(tmp_path)))
    assert "No row of the trial could be shown." in sheet and "Some rows were left out (1)" in sheet


# --- the success rule -----------------------------------------------------------------------------------

@pytest.mark.parametrize(("answer", "words"), [
    (None, "no answer from the site is recorded"),
    ("confirmed", "the site confirmed the same thresholds later"),
    ("this_is_wrong", "the site said they are wrong"),
])
def test_the_sites_answer_to_the_thresholds_is_read(answer, words) -> None:
    assumption = {"status": "assumed", **({"site_answer": answer} if answer else {})}
    sentence = done_sentence(_contract(), object_name="tray", place_label=None, assumption=assumption)
    assert words in sentence and ("has not" not in sentence)


def test_done_sentence_from_every_field() -> None:
    sentence = done_sentence(_contract(), object_name="tray", place_label=None)
    assert sentence.startswith("Done means: the tray's centre ends within 20 cm of the point (3.142, 2.718, 0.900) m, "
                               "on the back shelf; it is tilted no more than 15°; it stays still for 1 s; it rests "
                               "on a surface with its base 0.88–0.92 m high; the gripper has let go; it was lifted "
                               "at least 5 cm on the way.")
    assert "not recorded" not in sentence


@pytest.mark.parametrize(("path", "words"), [
    (("criteria", "surface_target", "radius_m"), "within a distance not recorded"),
    (("criteria", "surface_target", "surface_position_world_m"), "of the point not recorded"),
    (("criteria", "surface_target", "visible_label"), "a place whose name is not recorded"),
    (("criteria", "surface_target", "maximum_tilt_rad"), "tilt: not recorded"),
    (("criteria", "surface_target", "stable_seconds"), "how long it stays still: not recorded"),
    (("criteria", "support", "height_interval_m"), "the height it must rest at: not recorded"),
    (("criteria", "gripper_state", "mode"), "the gripper's state: not recorded"),
    (("criteria", "motion", "minimum_lift_m"), "how high it must be lifted: not recorded"),
    (("provenance",), "Who set these thresholds: not recorded"),
])
def test_done_sentence_says_not_recorded_for_each_missing_field(path, words) -> None:
    contract = copy.deepcopy(_contract())
    parent = contract
    for key in path[:-1]:
        parent = parent[key]
    del parent[path[-1]]
    assert words in done_sentence(contract, object_name="tray", place_label=None)
    assert "its name: not recorded" in done_sentence(_contract(), object_name=None, place_label=None)


@pytest.mark.parametrize(("field", "value", "words"), [
    ("radius_m", 0.004, "within 4 mm of the point"),
    ("radius_m", -0.2, "within a distance not recorded (invalid value)"),
    ("stable_seconds", -1.0, "how long it stays still: not recorded (invalid value)"),
    ("maximum_tilt_rad", -0.1, "tilt: not recorded (invalid value)"),
])
def test_done_sentence_small_and_invalid_values(field, value, words) -> None:
    contract = copy.deepcopy(_contract())
    contract["criteria"]["surface_target"][field] = value
    assert words in done_sentence(contract, object_name="tray", place_label=None)


def test_a_reversed_range_is_invalid() -> None:
    contract = copy.deepcopy(_contract())
    contract["criteria"]["support"]["height_interval_m"] = [0.92, 0.88]
    assert "the height it must rest at: not recorded (invalid value)" in done_sentence(
        contract, object_name="tray", place_label=None)


# --- scripted runs and every value the pipeline's builders write ------------------------------------------

SCRIPTED_ID = "blueprint_scripted_kinova_gen3"
SCRIPTED_KIND = "scripted_simulator_state_baseline"
LEARNED_ROW = {"candidate_id": "pi05_droid", "status": "completed", "episodes_scored": 10, "episodes_succeeded": 0}


def _scripted_row(**fields) -> dict:
    return {"candidate_id": SCRIPTED_ID, "status": "completed", "episodes_scored": 3, "episodes_succeeded": 3,
            "execution_evidence_kind": SCRIPTED_KIND, "candidate_policy_queried": False, **fields}


def _sheet_with_rows(tmp_path: Path, job, rows: list[dict]) -> str:
    inputs = job()
    _edit(inputs.screening_publication, lambda value: value.update(candidates=rows))
    return _sheet(build_job_packet(inputs, output_root=_out(tmp_path)))


def _sections(sheet: str) -> tuple[str, str]:
    trial = sheet.split("## Early trial", 1)[1].split("## ", 1)[0]
    check = sheet.split(f"## {packet.SCRIPTED_CHECK_TITLE}", 1)[1].split("## ", 1)[0] if packet.SCRIPTED_CHECK_TITLE in sheet else ""
    return trial, check


def test_a_scripted_run_is_a_check_of_the_simulation_never_robot_software(tmp_path, job) -> None:
    sheet = _sheet_with_rows(tmp_path, job, [LEARNED_ROW, _scripted_row()])
    trial, check = _sections(sheet)
    assert "| π0.5 DROID | completed | 10 | 0 of 10 |" in trial
    assert "Blueprint scripted" not in trial and "left out" not in trial
    assert packet.SCRIPTED_CHECK_SENTENCE in check and "| Blueprint scripted pick/place — kinova_gen3 | completed | 3 | 3 of 3 |" in check
    assert "13" not in sheet and "3 of 13" not in sheet  # the two are never added together


def test_a_catalog_scripted_row_without_its_evidence_kind_is_still_not_robot_software(tmp_path, job) -> None:
    row = _scripted_row()
    del row["execution_evidence_kind"], row["candidate_policy_queried"]
    trial, check = _sections(_sheet_with_rows(tmp_path, job, [row]))
    assert "Blueprint scripted" not in trial and "No row of the trial could be shown." in trial
    assert "kinova_gen3 | completed | 3 | 3 of 3 |" in check


@pytest.mark.parametrize("row", [
    {**LEARNED_ROW, "execution_evidence_kind": SCRIPTED_KIND},
    {**LEARNED_ROW, "candidate_policy_queried": False},
    _scripted_row(execution_evidence_kind="learned_policy"),
    _scripted_row(candidate_policy_queried=True),
])
def test_a_row_whose_kind_disagrees_with_the_catalog_is_left_out(tmp_path, job, row) -> None:
    trial, check = _sections(_sheet_with_rows(tmp_path, job, [row]))
    assert "No row of the trial could be shown." in trial and "Some rows were left out (1)" in trial
    assert "| " not in check


def test_every_fit_result_the_writer_uses_is_known() -> None:
    source = (Path(packet.__file__).parent / "website_robot_fit.py").read_text()
    written = set(re.findall(r'_check\("[a-z_]+", "([a-z_]+)"', source)) | {"pass", "fail", "unknown"}
    assert written == set(packet.RESULT_WORDS)


def _real_fit_table(tmp_path: Path, cycle):
    from blueprint_pipeline.robot_spec_catalog import load_robot_spec_catalog
    from blueprint_pipeline.website_robot_fit import robot_fit_table
    from tests.test_website_robot_fit import _collision, _verdict
    from tests.test_website_robot_fit import _job as fit_job
    fit = fit_job(cycle=cycle)
    collision = _collision(tmp_path)
    return robot_fit_table(preparation=fit, collision_path=collision, verdict=_verdict(fit, collision),
                           catalog=load_robot_spec_catalog())


@pytest.mark.parametrize("cycle", [None, 30, 1])
def test_the_real_fit_table_loses_no_row(tmp_path, cycle) -> None:
    table = _real_fit_table(tmp_path, cycle)
    lines = packet.fit_lines(table)
    shown = [line for line in lines if line.startswith("| ") and not line.startswith("| Robot")]
    assert len(shown) == len(table["rows"]) and not any("left out" in line for line in lines)
    assert table["table_digest"] == canonical_digest(table, digest_field="table_digest")
    if cycle is None:
        assert {check["result"] for row in table["rows"] for check in row["checks"]
                if check["check"] == "cycle_time"} == {"not_requested"}
        assert not any("no target given" in line or "cycle time" in line for line in shown)
        assert "Cycle time was not checked: the site gave no target." in lines


def _real_publication():
    from blueprint_pipeline.beta_policy_screening_result import screening_publication
    from tests.test_beta_scripted_publication import publication_inputs
    inputs = publication_inputs()
    scripted = inputs["plan"]["candidates"][0]
    scripted_summary = inputs["state"]["candidates"].pop(scripted["candidate_id"])
    learned = {**scripted, "candidate_id": "pi05_droid", "display_name": "x"}
    del learned["execution_evidence_kind"]
    learned_summary = copy.deepcopy(scripted_summary)
    for key in ("execution_evidence_kind", "action_source", "scripted_rollout_count", "learned_policy_rollout_count",
                "scripted_controller_invoked", "candidate_policy_queried"):
        del learned_summary["summary"][key]
    scripted = {**scripted, "candidate_id": SCRIPTED_ID}
    inputs["plan"]["candidates"] = [learned, scripted]
    inputs["state"]["candidates"] = {"pi05_droid": learned_summary, SCRIPTED_ID: scripted_summary}
    return screening_publication(**inputs)


def test_the_real_publication_loses_no_row_and_keeps_scripted_apart() -> None:
    text = "\n".join(packet.trial_lines(_real_publication()))
    assert "left out" not in text and "π0.5 DROID" in text.split(packet.SCRIPTED_CHECK_TITLE)[0]
    assert "kinova_gen3" in text.split(packet.SCRIPTED_CHECK_TITLE)[1]
