"""The job packet export: a summary cannot carry site text; a full packet needs a permission and this job's files.

Every input tree is synthetic and built here; no real site data.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import string
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from blueprint_pipeline import job_packet_export as packet
from blueprint_pipeline.job_packet_export import (
    EARLY_TRIAL_LABEL, TERMS_LINE, JobInputs, JobPacketRefused, OwnedCaptureApproval, build_job_packet,
    done_sentence,
)

SITE_NAME = "Rosa's Marigold Bakery"
ADDRESS = "17 Quillfeather Lane"
SENTENCE = f"Move the blue tray at {SITE_NAME}, {ADDRESS}, from the counter to the back shelf"
COORDINATES = ("3.14159", "2.71828", "1.41421")
NOW = datetime(2026, 11, 12, 18, 0, tzinfo=timezone.utc)
CAPTURE = "walkthrough-req-1"
REVISION = "sha256:" + "d" * 64


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    elif isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _contract() -> dict:
    return {"schema_version": "rigid_task_success_contract.v1", "scope": {"site_id": "site-req-1", "task_id": "t"},
            "provenance": {"author_source": "task_owner", "confirmation_status": "proposal_only"},
            "criteria": {"surface_target": {"radius_m": 0.2, "surface_position_world_m": [3.14159, 2.71828, 0.9],
                                            "visible_label": "back shelf", "maximum_tilt_rad": 0.2617993878,
                                            "stable_seconds": 1.0},
                         "support": {"height_mode": "required", "height_interval_m": [0.88, 0.92]},
                         "gripper_state": {"mode": "released", "threshold_m": 0.07},
                         "motion": {"minimum_lift_m": 0.05}},
            "contract_digest": "sha256:" + "c" * 64}


def _context() -> dict:
    return {"schema_version": "website_site_task_context.v1", "request_id": "req-1", "scene_id": "site-req-1",
            "capture_id": CAPTURE, "description": SENTENCE, "confirmed": True, "confirmed_at": "2026-11-01T00:00:00Z",
            "operator_answers": {"where": f"Back room at {ADDRESS}"},
            "operator_task_details": {"item_weight": "about 2 kg", "item_make_model": f"{SITE_NAME} house tray"},
            "success_criteria": {"successDefinition": f"Tray on the shelf at {SITE_NAME}", "successRate": 90,
                                 "cycleTimeSeconds": 30, "unknown": False},
            "capture_rights": {"derived_scene_generation_allowed": True, "consent_status": "granted",
                               "consent_revoked": False, "statement_version": "2026-11-02.v2"},
            "context_digest": "sha256:" + "a" * 64}


def _job(tmp_path: Path, *, trial: str = "zero", simulation: bool = True) -> JobInputs:
    capture = tmp_path / "job" / "scenes" / "site-req-1" / "captures" / CAPTURE
    _write(capture / "pipeline" / "website_task_context.json", _context())
    _write(capture / "pipeline_job_ledger.json", {"schema_version": "pipeline_job_ledger.v1", "status": "terminal_success"})
    _write(capture / "raw" / "video.mp4", b"synthetic video bytes")
    # As the WebApp's browser upload writes it (browserCaptureManifest.ts): the consent the delivery gate reads.
    _write(capture / "raw" / "manifest.json", {"schema_version": "v1", "scene_id": "site-req-1", "capture_id": CAPTURE,
                                               "site_submission_id": "req-1", "capture_rights": {
                                                   "derived_scene_generation_allowed": True, "consent_status": "granted",
                                                   "consent_revoked": False}})
    _pull_record(tmp_path / "job", [""], ["scenes/site-req-1/website_withdrawal",
                                          f"scenes/site-req-1/captures/{CAPTURE}/pipeline/consent_takedown"])
    prep = capture / "pipeline" / "website_scene_preparation"
    _write(prep / "preparation.json", {
        "digest": "sha256:" + "b" * 64,
        "intake_request": {"task": {"strategy": "pick_and_place"}},
        "subject": {"description": "tray", "aabb_min_xyz": [1.41421, 0.0, 0.9], "aabb_max_xyz": [1.81421, 0.3, 0.95]},
        "support": {"description": "counter"},
        "destination": {"relation": "on", "visible_label": "back shelf", "position_world_m": [3.14159, 2.71828, 0.9],
                        "support": {"description": "back shelf"}},
        "registration": {"task_region": {"radius_m": 1.0, "trimmed_rmse_m": 0.072}},
        "coordinate_frame": {"scale_authority": "registration_estimate", "physical_scale_measured": False,
                             "task_coordinates": "Z_up_estimated_meters", "declared_up_axis": "Z"},
        "authoring_inputs": {"metric_envelope": {"maximum_dimension_relative_error": 0.25}},
        "physics": {"bounds": {"mass_kg": [0.4, 1.8], "static_friction": [0.2, 0.9], "dynamic_friction": [0.15, 0.8]}}})
    _write(prep / "base_scene.json", {"up_axis": "Y", "scene_estimates": [{"code": "scene_estimate:level"}]})
    _write(prep / "handoff.json", {"preparation_digest": "sha256:" + "b" * 64, "robot_fit_table": {
        "inputs": {"grasp_point_world_m": [3.14159, 2.71828, 1.41421], "scale_band": [0.75, 1.33]},
        "counts": {"fits_spec": 1, "doesnt_fit": 1},
        "rows": [{"model_id": "universal_robots_ur5e", "name": "UR5e", "manufacturer": "Universal Robots",
                  "body_type": "fixed_arm", "verdict": "fits_spec", "reasons": [],
                  "checks": [{"check": "reach", "result": "pass", "reason": "within 3.14159 m"}]},
                 {"model_id": "universal_robots_ur10e", "name": "UR10e", "manufacturer": "Universal Robots",
                  "body_type": "fixed_arm", "verdict": "doesnt_fit", "reasons": ["payload: 2.71828 kg over"],
                  "checks": [{"check": "payload", "result": "fail", "reason": "2.71828 kg over"}]}]}})
    inputs = {"capture_dir": capture, "pull_root": tmp_path / "job"}
    if simulation:
        _write(capture / "pipeline" / "atlas" / "atlas_refined_splat.spz", b"splat")
        _write(capture / "pipeline" / "atlas" / "atlas_mesh_levelled.glb", b"glb")
        team = tmp_path / "team-scene"
        _write(team / "team_scene_inputs.v1.json", {"task_success_contract": _contract(), "scene_revision_digest": REVISION,
                                                    "owner_success_assumption": {"status": "assumed"}})
        _write(team / "native-task-packet" / "native_task_arena_scene_plan.v1.json",
               {"scene_id": "site-req-1", "task_spec": {"task_success_contract": _contract()}})
        _write(team / "native-task-packet" / "collision.usdc", b"usd collision")
        _write(team / "native-task-packet" / "scene_appearance.usdc", b"usd appearance")
        configured = tmp_path / "configured"
        _write(configured / "configured_scene_revision.v1.json", {"revision_digest": REVISION})
        _write(configured / "configured_scene_bundle.v1.zip", b"zip")
        _write(configured / "replacement_asset.usdz", b"usdz")
        inputs.update(team_scene_dir=team, configured_scene_dir=configured)
    if trial != "none":
        succeeded = 0 if trial == "zero" else 3
        inputs["screening_publication"] = _write(tmp_path / "publication.json", {
            "site_id": "site-req-1", "status": "completed", "source": {"capture_id": CAPTURE},
            "candidates": [{"candidate_id": "pi05_droid", "display_name": "π0.5 DROID", "status": "completed",
                            "episodes_scored": 10, "episodes_succeeded": succeeded}]})
        _write(tmp_path / "clips" / "manifest.json", {"site_id": "site-req-1", "capture_id": CAPTURE})
        _write(tmp_path / "clips" / "01-run-external.mp4", b"clip")
        inputs["review_videos_dir"] = tmp_path / "clips"
    return JobInputs(**inputs)


def _pull_record(root: Path, buckets: list[str], absent: list[str]) -> Path:
    return _write(root / "pull_record.json", {"schema_version": "job_packet_pull_record.v1", "buckets": buckets,
                                              "absent_on_host": absent})


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


def _full(tmp_path: Path) -> dict:
    return {"level": "full", "owned": _owned(), "owned_list": _owned_list(tmp_path, [CAPTURE])}


def _refused(inputs: JobInputs, tmp_path: Path, **kwargs) -> str:
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    with pytest.raises(JobPacketRefused) as refused:
        build_job_packet(inputs, output_root=out, **kwargs)
    assert list(out.iterdir()) == []
    return refused.value.code


def _all_bytes(folder: Path) -> bytes:
    return b"\n".join([folder.name.encode()] + [path.relative_to(folder).as_posix().encode() + b"\n" + path.read_bytes()
                                                for path in sorted(folder.rglob("*")) if path.is_file()])


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def _edit(path: Path, change) -> None:
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def _context_path(inputs: JobInputs) -> Path:
    return inputs.capture_dir / "pipeline" / "website_task_context.json"


# --- the summary ---------------------------------------------------------------------------------------

def test_summary_carries_the_three_things_and_nothing_that_identifies_the_site(tmp_path: Path) -> None:
    out = _out(tmp_path)
    manifest = build_job_packet(_job(tmp_path), output_root=out, now=NOW, operator="op")
    folder = Path(manifest["packet_dir"])
    assert re.fullmatch(r"packet-[0-9a-f]{16}", folder.name)
    sheet = (folder / "JOB_SHEET.md").read_text()
    assert "Kind of job: move one object from one surface to another." in sheet
    assert "| UR5e | Universal Robots | fixed arm | fits on published figures | none |" in sheet
    assert "| UR10e | Universal Robots | fixed arm | does not fit | payload (failed) |" in sheet
    assert EARLY_TRIAL_LABEL in sheet and "| π0.5 DROID | completed | 10 | 0 of 10 |" in sheet
    readme = (folder / "README.md").read_text()
    assert TERMS_LINE in readme and "operator" not in readme.casefold()
    assert sorted(row["path"] for row in manifest["files"]) == ["JOB_SHEET.md", "README.md"]
    on_disk = json.loads((folder / "manifest.json").read_text())
    assert set(on_disk) == {"schema_version", "packet_id", "level", "created_on", "code_revision", "permission_basis",
                            "first_look_source", "files", "recipients"}
    blob = _all_bytes(folder).decode()
    for planted in (SITE_NAME, ADDRESS, SENTENCE, "Back room", "house tray", *COORDINATES, "synthetic video",
                    "req-1", CAPTURE, "2026-11-02.v2", "tombstone", str(tmp_path)):
        assert planted not in blob
    # The operator record sits beside the folder, never inside it, and ties the packet to the job.
    record = json.loads(Path(manifest["operator_record"]).read_text())
    assert Path(manifest["operator_record"]).parent == out
    assert record["packet_folder"] == folder.name and record["packet_id"] == manifest["packet_id"]
    assert record["job"] == {"request_id": "req-1", "scene_id": "site-req-1", "capture_id": CAPTURE}
    assert record["run_by"] == "op" and "pipeline/website_task_context.json" in record["input_sha256"]


def test_summary_rows_take_names_from_the_catalog_and_leave_out_the_rest(tmp_path: Path) -> None:
    inputs = _job(tmp_path)
    handoff = inputs.capture_dir / "pipeline" / "website_scene_preparation" / "handoff.json"

    def plant(value):
        rows = value["robot_fit_table"]["rows"]
        rows[0].update(name="Arm One for Marigold", manufacturer="Jane Doe", body_type="arm at 17 Quillfeather Lane")
        rows[1]["checks"][0]["check"] = "reach to MarigoldOven shelf"
        rows.append({**rows[0], "model_id": "marigold_arm"})
        rows.append({**rows[0], "verdict": "doesnt_fit: Jane Doe says no"})
        value["robot_fit_table"]["counts"]["fits near 40.7128,-74.0060"] = 1
    _edit(handoff, plant)

    def plant_publication(value):
        value["candidates"][0]["display_name"] = "Open Policy A (blue tray run)"
        value["candidates"].append({**value["candidates"][0], "candidate_id": "pi05_droid", "status": "done by Jane"})
        value["candidates"].append({**value["candidates"][0], "episodes_scored": 5, "episodes_succeeded": 9})
        value["status"] = "completed - see jane@marigold.example 555-0142"
    _edit(inputs.screening_publication, plant_publication)
    manifest = build_job_packet(inputs, output_root=_out(tmp_path))
    sheet = (Path(manifest["packet_dir"]) / "JOB_SHEET.md").read_text()
    for secret in ("Marigold", "Quillfeather", "Jane", "40.7128", "blue tray", "555-0142"):
        assert secret not in sheet
    assert "| UR5e | Universal Robots | fixed arm | fits on published figures | none |" in sheet
    # The fit table: an unknown check, an unknown robot and an unknown verdict; the trial: a repeat that also has
    # an unknown status, and one with more successes than tries.
    assert sheet.count("Some rows were left out (3)") == 1 and sheet.count("Some rows were left out (2)") == 1
    assert "| π0.5 DROID | completed | 10 | 0 of 10 |" in sheet
    assert "Trial status: not recorded." in sheet


def _leaves(value, path=()):
    if isinstance(value, str):
        yield path
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaves(item, (*path, index))


def _set(value, path, new):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = new


#: The fields the hard stops read: a secret there refuses the packet, so they keep their values here
#: (the summary test asserts none of these values reaches the packet either).
GATE_FIELDS = {"website_task_context.json": {("schema_version",), ("request_id",), ("scene_id",), ("capture_id",),
                                             ("capture_rights", "consent_status")},
               "pipeline_job_ledger.json": {("status",)},
               "manifest.json": {("capture_rights", "consent_status")},
               "publication.json": {("site_id",), ("source", "capture_id")}}


@pytest.mark.parametrize("seed", range(12))
def test_property_no_planted_secret_reaches_any_byte_of_the_summary(tmp_path: Path, seed: int) -> None:
    rng = random.Random(seed)
    share = 1.0 if seed % 3 == 0 else 0.6
    inputs = _job(tmp_path, trial=rng.choice(["zero", "three"]))
    planted = []
    for path in sorted(tmp_path.rglob("*.json")):
        value = json.loads(path.read_text())
        for leaf in list(_leaves(value)):
            if path.name == "pull_record.json" or leaf in GATE_FIELDS.get(path.name, set()) or rng.random() >= share:
                continue
            token = "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(10))
            secret = rng.choice([token, f"{token} Lane", f"{token}’s Bakery", f"{token[:5]}\n{token[5:]}",
                                 f"{token.upper()}  {token}"])
            planted.append(secret)
            _set(value, leaf, rng.choice([secret, f"{_get(value, leaf)} {secret}", f"{secret}{_get(value, leaf)}"]))
        path.write_text(json.dumps(value))
    manifest = build_job_packet(inputs, output_root=_out(tmp_path))
    blob = _squash(_all_bytes(Path(manifest["packet_dir"])).decode())
    assert planted
    for secret in planted:
        assert _squash(secret) not in blob


def _get(value, path):
    for key in path:
        value = value[key]
    return value


def test_summary_takes_no_first_look_file_and_no_redaction_options(tmp_path: Path) -> None:
    inputs = _job(tmp_path)
    look = _write(tmp_path / "look" / "first-look.md", "A fixed arm.")
    with_look = JobInputs(capture_dir=inputs.capture_dir, first_look_file=look)
    assert _refused(with_look, tmp_path) == "job_packet_first_look_full_level_only"
    assert _refused(inputs, tmp_path, first_look_checked_by="op") == "job_packet_option_removed"
    assert _refused(inputs, tmp_path, redact_terms=[SITE_NAME]) == "job_packet_option_removed"


def test_a_nonzero_trial_and_a_zero_count_are_shown(tmp_path: Path) -> None:
    manifest = build_job_packet(_job(tmp_path, trial="three"), output_root=_out(tmp_path))
    assert "3 of 10" in (Path(manifest["packet_dir"]) / "JOB_SHEET.md").read_text()


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


# --- refusals at both levels ---------------------------------------------------------------------------

def _rights(**fields):
    return lambda inputs: _edit(_context_path(inputs), lambda value: value["capture_rights"].update(**fields))


def _file(relative, value):
    return lambda inputs: _write(inputs.capture_dir / relative, value)


def _drop_status(inputs):
    _edit(_context_path(inputs), lambda value: value["capture_rights"].pop("consent_status"))


def _drop_revoked(inputs):
    _edit(_context_path(inputs), lambda value: value["capture_rights"].pop("consent_revoked"))


def _ledger(**fields):
    return _file("pipeline_job_ledger.json", {"status": "failed_retryable", **fields})


def _tombstone(where):
    def change(inputs):
        site = inputs.capture_dir.parent.parent
        target = {"own": site, "bucket": site.parents[1] / "bucketB" / "scenes" / site.name}[where]
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
    def apply(inputs):
        _edit(inputs.pull_root / "pull_record.json", change)
    return apply


def _takedown_open(inputs):
    _write(inputs.capture_dir / "pipeline" / "consent_takedown" / "takedown_manifest.json", {"takedown_open": True})
    _record_edit(lambda value: value["absent_on_host"].remove(
        f"scenes/site-req-1/captures/{CAPTURE}/pipeline/consent_takedown"))(inputs)


def _both(*changes):
    return lambda inputs: [change(inputs) for change in changes]


REFUSALS = {
    "takedown_manifest_open": (_takedown_open, "job_packet_delivery_gate_blocked_open_consent_takedown"),
    "consent_unverifiable": (lambda inputs: (inputs.capture_dir / "raw" / "manifest.json").unlink(),
                             "job_packet_delivery_gate_blocked_consent_state_unverifiable"),
    "consent_file_bom": (_file("raw/rights_consent.json", "\ufeff" + json.dumps({"consent_status": "revoked"})),
                         "job_packet_consent_state_unreadable"),
    "consent_file_truncated": (_file("raw/rights_consent.json", '{"consent_status":"revo'), "job_packet_consent_state_unreadable"),
    "consent_file_list": (_file("rights_consent.json", '[{"consent_status":"revoked"}]'), "job_packet_consent_state_unreadable"),
    "consent_file_duplicate_keys": (_file("rights_consent.json", '{"consent_status":"revoked","consent_status":"granted"}'),
                                    "job_packet_consent_state_unreadable"),
    "first_active_second_revoked": (_both(_file("raw/rights_consent.json", {"consent_status": "granted"}),
                                          _file("raw/manifest.json", {"consent_status": "revoked"})),
                                    "job_packet_capture_withdrawn"),
    "consent_expired": (_rights(consent_expires_at="2020-01-01T00:00:00Z"), "job_packet_consent_expired"),
    "ledger_code_as_key": (_file("pipeline_job_ledger.json", {"status": "terminal_success",
                                                             "terminal": {"website_run_closed_by_owner": True}}),
                           "job_packet_job_closed_by_owner"),
    "ledger_code_in_sentence": (_ledger(last_error="WebApp said consent_expired for this job"),
                                "job_packet_capture_authority_ended"),
    "ledger_code_as_status": (_file("pipeline_job_ledger.json", {"status": "source_revoked"}),
                              "job_packet_capture_authority_ended"),
    "tombstone_dir_other_bucket": (lambda inputs: (inputs.pull_root / "bkt" / "scenes" / "site-req-1" / "website_withdrawal"
                                                   / "tombstone.json").mkdir(parents=True), "job_packet_capture_withdrawn"),
    "no_pull_record": (lambda inputs: (inputs.pull_root / "pull_record.json").unlink(), "job_packet_withdrawal_state_not_pulled"),
    "withdrawal_not_pulled": (_record_edit(lambda value: value["absent_on_host"].remove("scenes/site-req-1/website_withdrawal")),
                              "job_packet_withdrawal_state_not_pulled"),
    "takedown_not_pulled": (_record_edit(lambda value: value["absent_on_host"].remove(
        f"scenes/site-req-1/captures/{CAPTURE}/pipeline/consent_takedown")), "job_packet_withdrawal_state_not_pulled"),
    "recorded_absent_but_pulled": (lambda inputs: (inputs.capture_dir.parent.parent / "website_withdrawal").mkdir(),
                                   "job_packet_withdrawal_state_not_pulled"),
    "bucket_not_in_record": (lambda inputs: (inputs.pull_root / "bkt" / "scenes").mkdir(parents=True),
                             "job_packet_withdrawal_state_not_pulled"),
    "status_withdrawn": (_rights(consent_status="withdrawn"), "job_packet_capture_withdrawn"),
    "status_rescinded": (_rights(consent_status="rescinded"), "job_packet_capture_withdrawn"),
    "status_pending": (_rights(consent_status="pending"), "job_packet_consent_not_active"),
    "status_missing": (_drop_status, "job_packet_consent_not_active"),
    "camel_revoked": (_rights(consentRevoked=True), "job_packet_capture_withdrawn"),
    "camel_status": (_rights(consentStatus="revoked"), "job_packet_capture_withdrawn"),
    "revoked_at": (_rights(consent_revoked_at="2026-11-10T00:00:00Z"), "job_packet_capture_withdrawn"),
    "rights_consent_file": (_file("raw/rights_consent.json", {"consent_status": "revoked"}), "job_packet_capture_withdrawn"),
    "raw_manifest": (_file("raw/manifest.json", {"consent_status": "withdrawn"}), "job_packet_capture_withdrawn"),
    "descriptor": (_file("capture_descriptor.json", {"consent_revoked": True}), "job_packet_capture_withdrawn"),
    "pending_consent_file": (_file("rights_consent.json", {"consent_status": "pending"}), "job_packet_consent_not_active"),
    "rights_not_granted": (_rights(derived_scene_generation_allowed=False), "job_packet_capture_rights_not_granted"),
    "rights_granted_as_text": (_rights(derived_scene_generation_allowed="true"), "job_packet_capture_rights_not_granted"),
    "revoked_flag_absent": (_drop_revoked, "job_packet_capture_rights_not_granted"),
    "revoked_flag_not_false": (_rights(consent_revoked=None), "job_packet_capture_rights_not_granted"),
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
    "ledger_missing": (lambda inputs: (inputs.capture_dir / "pipeline_job_ledger.json").unlink(), "job_packet_job_ledger_missing"),
    "ledger_unreadable": (_file("pipeline_job_ledger.json", '{"status": "terminal_success"} xx'), "job_packet_job_state_unreadable"),
    "ledger_duplicate_keys": (_file("pipeline_job_ledger.json", '{"status":"terminal_authority_ended","status":"terminal_success"}'),
                              "job_packet_job_state_unreadable"),
    "ledger_bom": (_file("pipeline_job_ledger.json", '﻿{"status":"terminal_success"}'), "job_packet_job_state_unreadable"),
    "ledger_corrupt": (_file("pipeline_job_ledger.json", {"status": "corrupt"}), "job_packet_job_state_unreadable"),
    "ledger_pipe": (_fifo("pipeline_job_ledger.json"), "job_packet_job_state_unreadable"),
    "receipt_garbage": (_file("pipeline_job_terminal_receipt.json", "nope"), "job_packet_job_state_unreadable"),
}


@pytest.mark.parametrize("level", ["summary", "full"])
@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_withdrawn_unconsented_ended_or_closed_jobs_are_refused_at_both_levels(tmp_path, level, case) -> None:
    change, code = REFUSALS[case]
    inputs = _job(tmp_path)
    change(inputs)
    assert _refused(inputs, tmp_path, **(_full(tmp_path) if level == "full" else {})) == code


def test_a_capture_pulled_without_the_site_root_layout_is_refused(tmp_path: Path) -> None:
    inputs = _job(tmp_path)
    flat = tmp_path / "flat" / "x" / "y" / CAPTURE
    flat.parent.mkdir(parents=True)
    inputs.capture_dir.rename(flat)
    assert _refused(JobInputs(capture_dir=flat), tmp_path) == "job_packet_pull_root_required"
    assert _refused(JobInputs(capture_dir=flat, pull_root=tmp_path / "flat"), tmp_path) == "job_packet_withdrawal_state_unknown"


def test_a_capture_under_a_bucket_is_checked_across_every_bucket(tmp_path: Path) -> None:
    inputs = _job(tmp_path)
    root = tmp_path / "pulled"
    moved = root / "bucketA" / "scenes" / "site-req-1"
    moved.parent.mkdir(parents=True)
    inputs.capture_dir.parent.parent.rename(moved)
    capture = moved / "captures" / CAPTURE
    _pull_record(root, ["bucketA", "bucketB"], ["bucketA/scenes/site-req-1/website_withdrawal",
                                                "bucketB/scenes/site-req-1/website_withdrawal",
                                                f"bucketA/scenes/site-req-1/captures/{CAPTURE}/pipeline/consent_takedown"])
    manifest = build_job_packet(JobInputs(capture_dir=capture, pull_root=root), output_root=_out(tmp_path))
    assert manifest["level"] == "summary"
    _write(root / "bucketB" / "scenes" / "site-req-1" / "website_withdrawal" / "tombstone.json", "{}")
    assert _refused(JobInputs(capture_dir=capture, pull_root=root), tmp_path / "again") == "job_packet_capture_withdrawn"


# --- the full level ------------------------------------------------------------------------------------

def test_full_level_is_refused_without_a_permission(tmp_path: Path) -> None:
    assert packet.SHARING_PERMISSION_STATEMENT_VERSIONS == ()
    assert _refused(_job(tmp_path), tmp_path, level="full") == "job_packet_full_level_requires_sharing_permission"


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
def test_blueprint_owned_capture_needs_the_list_and_the_owners_words(tmp_path, ids, approval, code) -> None:
    assert _refused(_job(tmp_path), tmp_path, level="full", owned=approval, owned_list=_owned_list(tmp_path, ids)) == code


@pytest.mark.parametrize("raw", ["{", json.dumps([CAPTURE]), json.dumps({"schema_version": "blueprint_owned_captures.v1",
                                                                         "captures": [[CAPTURE]]}),
                                 json.dumps({"schema_version": "blueprint_owned_captures.v1", "captures": CAPTURE}),
                                 json.dumps({"schema_version": "blueprint_owned_captures.v1", "captures": [{"capture_id": "*"}]})])
def test_a_malformed_allow_list_is_a_clean_refusal(tmp_path, raw) -> None:
    path = _write(tmp_path / "owned.json", raw)
    assert _refused(_job(tmp_path), tmp_path, level="full", owned=_owned(), owned_list=path) == \
        "job_packet_owned_capture_list_invalid"


def test_full_level_for_an_allow_listed_capture_records_the_approval(tmp_path: Path) -> None:
    manifest = build_job_packet(_job(tmp_path), output_root=_out(tmp_path), now=NOW, **_full(tmp_path))
    folder = Path(manifest["packet_dir"])
    basis = manifest["permission_basis"]
    assert basis["rule"] == "blueprint_owned_capture" and basis["approval_evidence"] == "Yes, share the reference job."
    assert basis["approved_on"] == "2026-11-12" and manifest["job"]["capture_id"] == CAPTURE
    paths = {row["path"] for row in manifest["files"]}
    assert {"files/video/video.mp4", "files/object/replacement_asset.usdz",
            "files/simulation/configured-scene/configured_scene_bundle.v1.zip",
            "files/simulation/team-scene/native-task-packet/collision.usdc",
            "files/simulation/reconstruction/atlas_refined_splat.spz",
            "files/task/team_scene_inputs.v1.json", "files/task/native-task-packet/native_task_arena_scene_plan.v1.json",
            "files/early-trial/publication.json", "files/early-trial/clips/01-run-external.mp4"} <= paths
    on_disk = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()}
    assert on_disk == paths | {"manifest.json"}
    for row in manifest["files"]:
        data = (folder / row["path"]).read_bytes()
        assert row["size"] == len(data) and row["sha256"] == hashlib.sha256(data).hexdigest()
    sheet = (folder / "JOB_SHEET.md").read_text()
    assert SENTENCE in sheet and "## Measured" in sheet and "## Estimated" in sheet
    assert "7 cm" in sheet and "scene_estimate:level" in sheet and "0.4–1.8 kg" in sheet
    assert "within 20 cm of the point (3.142, 2.718, 0.900) m, on the back shelf" in sheet
    assert "no answer from the site is recorded" in sheet and "No safety assessment" in sheet
    readme = (folder / "README.md").read_text()
    assert "Isaac Sim 6.0.1" in readme and "Z_up_estimated_meters" in readme and "files/video/" in readme


def test_full_readme_lists_only_files_present(tmp_path: Path) -> None:
    manifest = build_job_packet(_job(tmp_path, trial="none", simulation=False), output_root=_out(tmp_path), **_full(tmp_path))
    folder = Path(manifest["packet_dir"])
    readme = (folder / "README.md").read_text()
    assert "carries no simulation files" in readme and "team-scene" not in readme and "early-trial" not in readme
    sheet = (folder / "JOB_SHEET.md").read_text()
    assert "No early trial result is included" in sheet and "The simulation: not given to the tool" in sheet
    assert "What counts as done: not recorded" in sheet


def test_a_listed_sharing_statement_version_allows_the_full_level(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(packet, "SHARING_PERMISSION_STATEMENT_VERSIONS", ("2026-11-02.v2",))
    manifest = build_job_packet(_job(tmp_path), output_root=_out(tmp_path), level="full")
    assert manifest["permission_basis"] == {"rule": "site_sharing_permission", "statement_version": "2026-11-02.v2"}


def _other_video(inputs: JobInputs) -> JobInputs:
    other = _write(inputs.capture_dir.parents[4] / "other" / "theirs.mp4", b"OTHER SITE VIDEO")
    return JobInputs(**{**inputs.__dict__, "video": other})


def _clips_of(**fields):
    def change(inputs):
        _write(inputs.review_videos_dir / "manifest.json", {"site_id": "site-req-1", "capture_id": CAPTURE, **fields})
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
def test_full_level_takes_only_this_jobs_files(tmp_path, case) -> None:
    inputs = _job(tmp_path)
    inputs = NOT_THIS_JOB[case](inputs) or inputs
    assert _refused(inputs, tmp_path, **_full(tmp_path)) == "job_packet_input_not_this_job"


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


# --- files ---------------------------------------------------------------------------------------------

def test_packet_id_is_stable_for_the_same_inputs(tmp_path: Path) -> None:
    inputs = _job(tmp_path)
    first = build_job_packet(inputs, output_root=_out(tmp_path, "a"), **_full(tmp_path))
    second = build_job_packet(inputs, output_root=_out(tmp_path, "b"), **_full(tmp_path))
    assert first["packet_id"] == second["packet_id"]
    assert Path(first["packet_dir"]).name != Path(second["packet_dir"]).name
    rows = sorted(f"{row['path']} {row['sha256']}" for row in first["files"])
    assert first["packet_id"] == "sha256:" + hashlib.sha256("\n".join(rows).encode()).hexdigest()


@pytest.mark.parametrize("where", ["team", "raw"])
def test_a_symlink_out_of_the_inputs_is_refused(tmp_path: Path, where: str) -> None:
    inputs = _job(tmp_path)
    secret = _write(tmp_path / "elsewhere" / "secret.usdc", b"not this job")
    link = (inputs.team_scene_dir / "native-task-packet" / "extra.usdc" if where == "team"
            else inputs.capture_dir / "raw" / "video.mp4")
    if link.exists():
        link.unlink()
    link.symlink_to(secret)
    assert _refused(inputs, tmp_path, **_full(tmp_path)) == "job_packet_input_symlink_outside_inputs"


def test_a_hard_link_or_a_pipe_where_a_file_is_expected_is_refused(tmp_path: Path) -> None:
    inputs = _job(tmp_path)
    os.link(_write(tmp_path / "outside.usdc", b"OTHER SITE"), inputs.team_scene_dir / "native-task-packet" / "x.usdc")
    assert _refused(inputs, tmp_path, **_full(tmp_path)) == "job_packet_input_hard_linked"
    (inputs.team_scene_dir / "native-task-packet" / "x.usdc").unlink()
    (inputs.capture_dir / "raw" / "video.mp4").unlink()
    os.mkfifo(inputs.capture_dir / "raw" / "video.mp4")
    assert _refused(inputs, tmp_path, **_full(tmp_path)) == "job_packet_input_not_a_regular_file"


def test_an_output_root_that_is_a_link_is_refused(tmp_path: Path) -> None:
    real = _out(tmp_path, "real")
    (tmp_path / "linked").symlink_to(real)
    with pytest.raises(JobPacketRefused, match="job_packet_output_root_invalid"):
        build_job_packet(_job(tmp_path), output_root=tmp_path / "linked")
    assert list(real.iterdir()) == []


def test_two_files_for_one_packet_path_are_refused() -> None:
    rows = [(Path("a"), Path("a/x"), "files/object/x.usdz"), (Path("b"), Path("b/x"), "files/object/x.usdz")]
    with pytest.raises(JobPacketRefused, match="job_packet_file_name_collision"):
        packet.unique_destinations(rows)


def test_a_failure_while_writing_leaves_nothing_behind(tmp_path, monkeypatch) -> None:
    calls = []

    def failing_copy(source, target):
        calls.append(target)
        if len(calls) == 3:
            raise OSError("disk full")
        Path(target).write_bytes(Path(source).read_bytes())
    monkeypatch.setattr(packet.shutil, "copyfile", failing_copy)
    out = _out(tmp_path)
    with pytest.raises(OSError):
        build_job_packet(_job(tmp_path), output_root=out, **_full(tmp_path))
    assert len(calls) == 3 and list(out.iterdir()) == []


def test_the_command_writes_and_refuses(tmp_path: Path, capsys) -> None:
    spec = importlib.util.spec_from_file_location(
        "export_job_packet", Path(__file__).resolve().parents[1] / "scripts" / "export_job_packet.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    inputs = _job(tmp_path)
    common = ["--pull-root", str(tmp_path / "job"), "--capture-dir", str(inputs.capture_dir), "--operator", "op"]
    assert script.main([*common, "--output-root", str(_out(tmp_path))]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "written" and Path(result["operator_record"]).is_file()
    assert script.main([*common, "--output-root", str(_out(tmp_path, "o2")), "--level", "full"]) == 2
    assert json.loads(capsys.readouterr().out)["code"] == "job_packet_full_level_requires_sharing_permission"
    with pytest.raises(SystemExit):
        script.main([*common, "--output-root", str(tmp_path / "o2"), "--level", "full", "--blueprint-owned-capture",
                     "--capture-id", CAPTURE, "--approved-by", "owner", "--approval-evidence", "yes"])


# --- round 2 ----------------------------------------------------------------------------------------------

def test_the_delivery_gate_allows_an_ordinary_website_job(tmp_path: Path) -> None:
    from blueprint_pipeline.consent_takedown import evaluate_delivery_time_takedown_gate
    inputs = _job(tmp_path)
    gate = evaluate_delivery_time_takedown_gate(capture_root=inputs.capture_dir, surface="job_packet")
    assert gate["status"] == "allowed" and gate["consent"]["source_path"].endswith("raw/manifest.json")
    assert packet.job_state_refusal(inputs.capture_dir, _context(), pull_root=inputs.pull_root) is None


def test_the_pipelines_withdrawal_readers_are_obeyed(tmp_path, monkeypatch) -> None:
    import blueprint_pipeline.screening_site_withdrawal as screening
    import blueprint_pipeline.website_atlas_preview_store as preview
    inputs = _job(tmp_path)
    monkeypatch.setattr(screening, "site_withdrawal_state", lambda *args, **kwargs: "unknown")
    assert _refused(inputs, tmp_path / "a") == "job_packet_withdrawal_state_unknown"
    monkeypatch.undo()
    seen = []
    monkeypatch.setattr(preview, "capture_withdrawn", lambda root: seen.append(root) or True)
    assert _refused(inputs, tmp_path / "b") == "job_packet_capture_withdrawn" and seen == [inputs.capture_dir]


def test_duplicate_keys_in_the_task_context_are_refused(tmp_path: Path) -> None:
    inputs = _job(tmp_path)
    path = _context_path(inputs)
    path.write_text(path.read_text()[:-1] + ', "capture_id": "walkthrough-other"}')
    assert _refused(inputs, tmp_path) == "job_packet_input_unreadable"


def test_a_group_writable_catalog_is_a_clean_refusal(tmp_path: Path) -> None:
    packet.POLICY_CATALOG_PATH.chmod(0o664)
    assert _refused(_job(tmp_path), tmp_path) == "job_packet_policy_catalog_unreadable"


@pytest.mark.parametrize(("scored", "succeeded"), [(0, 0), (1001, 3), (5, 6), (True, 0), (10, -1)])
def test_trial_counts_out_of_bounds_are_left_out(tmp_path, scored, succeeded) -> None:
    inputs = _job(tmp_path)
    _edit(inputs.screening_publication, lambda value: value["candidates"][0].update(
        episodes_scored=scored, episodes_succeeded=succeeded))
    sheet = (Path(build_job_packet(inputs, output_root=_out(tmp_path))["packet_dir"]) / "JOB_SHEET.md").read_text()
    assert "No row of the trial could be shown." in sheet and "Some rows were left out (1)" in sheet


def test_repeats_are_left_out_and_large_counts_are_capped(tmp_path: Path) -> None:
    inputs = _job(tmp_path)
    _edit(inputs.screening_publication, lambda value: value.update(
        candidates=value["candidates"] * 3 + [{"candidate_id": "junk"}] * 7))
    handoff = inputs.capture_dir / "pipeline" / "website_scene_preparation" / "handoff.json"
    _edit(handoff, lambda value: value["robot_fit_table"].update(
        rows=value["robot_fit_table"]["rows"] * 4 + [{"model_id": "x"}] * 42))
    sheet = (Path(build_job_packet(inputs, output_root=_out(tmp_path))["packet_dir"]) / "JOB_SHEET.md").read_text()
    assert sheet.count("| π0.5 DROID |") == 1 and sheet.count("| UR5e |") == 1 and sheet.count("| UR10e |") == 1
    assert "Some rows were left out (9)" in sheet and "Some rows were left out (more than 20)" in sheet
    assert "48" not in sheet


@pytest.mark.parametrize("approved_on", ["2026-11-13", "2026-08-13", "2020-01-01"])
def test_an_approval_dated_in_the_future_or_over_90_days_ago_is_refused(tmp_path, approved_on) -> None:
    approval = OwnedCaptureApproval(CAPTURE, "owner", "yes", approved_on)
    assert _refused(_job(tmp_path), tmp_path, level="full", owned=approval, now=NOW,
                    owned_list=_owned_list(tmp_path, [CAPTURE])) == "job_packet_owned_capture_approval_out_of_date"
    assert build_job_packet(_job(tmp_path / "ok"), output_root=_out(tmp_path, "ok-out"), level="full", now=NOW,
                            owned=OwnedCaptureApproval(CAPTURE, "owner", "yes", "2026-08-14"),
                            owned_list=_owned_list(tmp_path, [CAPTURE]))["permission_basis"]["approved_on"] == "2026-08-14"


def test_the_operator_record_keeps_the_pull_record_whole_and_outside_the_packet(tmp_path: Path) -> None:
    inputs = _job(tmp_path)
    record_path = inputs.pull_root / "pull_record.json"
    supplied = json.loads(record_path.read_text())
    manifest = build_job_packet(inputs, output_root=_out(tmp_path))
    record = json.loads(Path(manifest["operator_record"]).read_text())
    assert record["pull_record"] == supplied
    assert record["pull_record_sha256"] == hashlib.sha256(record_path.read_bytes()).hexdigest()
    blob = _all_bytes(Path(manifest["packet_dir"])).decode()
    assert "absent_on_host" not in blob and "website_withdrawal" not in blob and "consent_takedown" not in blob
