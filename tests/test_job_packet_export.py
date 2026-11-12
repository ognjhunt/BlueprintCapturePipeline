"""The job packet export: a summary carries nothing the site said or filmed; a full packet needs a permission.

Every input tree is synthetic and built here; no real site data.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
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


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
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


def _job(tmp_path: Path, *, trial: str = "zero", simulation: bool = True) -> JobInputs:
    capture = tmp_path / "job" / "scenes" / "site-req-1" / "captures" / CAPTURE
    _write(capture / "pipeline" / "website_task_context.json", {
        "schema_version": "website_site_task_context.v1", "request_id": "req-1", "scene_id": "site-req-1",
        "capture_id": CAPTURE, "description": SENTENCE, "confirmed": True, "confirmed_at": "2026-11-01T00:00:00Z",
        "operator_answers": {"where": f"Back room at {ADDRESS}"},
        "operator_task_details": {"item_weight": "about 2 kg", "item_make_model": f"{SITE_NAME} house tray"},
        "success_criteria": {"successDefinition": f"Tray on the shelf at {SITE_NAME}", "successRate": 90,
                             "cycleTimeSeconds": 30, "unknown": False},
        "capture_rights": {"derived_scene_generation_allowed": True, "consent_status": "granted",
                           "consent_revoked": False, "statement_version": "2026-11-02.v2"},
        "context_digest": "sha256:" + "a" * 64})
    _write(capture / "pipeline_job_ledger.json", {"schema_version": "pipeline_job_ledger.v1", "status": "terminal_success"})
    _write(capture / "raw" / "video.mp4", b"synthetic video bytes")
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
        "rows": [{"name": "Arm One", "manufacturer": "Maker A", "body_type": "fixed_arm", "verdict": "fits_spec",
                  "reasons": [], "checks": [{"check": "reach", "result": "pass", "reason": "within 3.14159 m"}]},
                 {"name": "Arm Two", "manufacturer": "Maker B", "body_type": "fixed_arm", "verdict": "doesnt_fit",
                  "reasons": ["payload: 2.71828 kg over"],
                  "checks": [{"check": "payload", "result": "fail", "reason": "2.71828 kg over"}]}]}})
    inputs = {"capture_dir": capture}
    if simulation:
        _write(capture / "pipeline" / "atlas" / "atlas_refined_splat.spz", b"splat")
        _write(capture / "pipeline" / "atlas" / "atlas_mesh_levelled.glb", b"glb")
        team = tmp_path / "team-scene"
        _write(team / "team_scene_inputs.v1.json", {"task_success_contract": _contract(),
                                                    "owner_success_assumption": {"status": "assumed"}})
        _write(team / "native-task-packet" / "native_task_arena_scene_plan.v1.json",
               {"scene_id": "site-req-1", "task_spec": {"task_success_contract": _contract()}})
        _write(team / "native-task-packet" / "collision.usdc", b"usd collision")
        _write(team / "native-task-packet" / "scene_appearance.usdc", b"usd appearance")
        configured = tmp_path / "configured"
        _write(configured / "configured_scene_bundle.v1.zip", b"zip")
        _write(configured / "replacement_asset.usdz", b"usdz")
        inputs.update(team_scene_dir=team, configured_scene_dir=configured)
    if trial != "none":
        succeeded = 0 if trial == "zero" else 3
        inputs["screening_publication"] = _write(tmp_path / "publication.json", {
            "site_id": "site-req-1", "status": "completed",
            "candidates": [{"candidate_id": "droid-a", "display_name": "Open Policy A", "status": "completed",
                            "episodes_scored": 10, "episodes_succeeded": succeeded}]})
        _write(tmp_path / "clips" / "manifest.json", {"videos": 1})
        _write(tmp_path / "clips" / "01-run-external.mp4", b"clip")
        inputs["review_videos_dir"] = tmp_path / "clips"
    return JobInputs(**inputs)


def _out(tmp_path: Path, name: str = "out") -> Path:
    path = tmp_path / name
    path.mkdir()
    return path


def _owned_list(tmp_path: Path, ids: list[str]) -> Path:
    return _write(tmp_path / "owned.json", {"schema_version": "blueprint_owned_captures.v1",
                                            "captures": [{"capture_id": item} for item in ids]})


def _owned() -> OwnedCaptureApproval:
    return OwnedCaptureApproval(capture_id=CAPTURE, approved_by="owner", approval_evidence="Yes, share the reference job.")


def _all_text(folder: Path) -> str:
    return "\n".join(path.read_bytes().decode("utf-8", "replace") for path in folder.rglob("*") if path.is_file())


def test_summary_carries_the_three_things_and_nothing_the_site_said_or_filmed(tmp_path: Path) -> None:
    manifest = build_job_packet(_job(tmp_path), output_root=_out(tmp_path), now=NOW)
    folder = Path(manifest["packet_dir"])
    assert folder.name == "packet-req1-20261112"
    sheet = (folder / "JOB_SHEET.md").read_text()
    assert "Kind of job: move one object from one surface to another." in sheet
    assert "Arm One" in sheet and "does not fit" in sheet and "robot fit table" in sheet
    assert EARLY_TRIAL_LABEL in sheet and "0 of 10" in sheet
    assert TERMS_LINE in (folder / "README.md").read_text()
    assert not (folder / "files").exists()
    assert sorted(row["path"] for row in manifest["files"]) == ["JOB_SHEET.md", "README.md"]
    text = _all_text(folder)
    for planted in (SITE_NAME, ADDRESS, SENTENCE, "Back room", "house tray", *COORDINATES, "synthetic video"):
        assert planted not in text
    assert manifest["level"] == "summary" and manifest["recipients"] == []
    assert manifest["takedown_keys"]["capture_id"] == CAPTURE


def test_full_level_is_refused_without_a_permission(tmp_path: Path) -> None:
    assert packet.SHARING_PERMISSION_STATEMENT_VERSIONS == ()
    with pytest.raises(JobPacketRefused) as refused:
        build_job_packet(_job(tmp_path), output_root=_out(tmp_path), level="full")
    assert refused.value.code == "job_packet_full_level_requires_sharing_permission"
    assert list((tmp_path / "out").iterdir()) == []


def test_committed_owned_capture_list_is_empty() -> None:
    assert packet.owned_capture_ids() == set()


@pytest.mark.parametrize(("ids", "approval", "code"), [
    ([], _owned(), "job_packet_capture_not_blueprint_owned"),
    (["walkthrough-other"], _owned(), "job_packet_capture_not_blueprint_owned"),
    # The capture id the operator names must be the one the task context records.
    ([CAPTURE, "walkthrough-other"], OwnedCaptureApproval("walkthrough-other", "owner", "yes"),
     "job_packet_owned_capture_id_mismatch"),
    ([CAPTURE], OwnedCaptureApproval(CAPTURE, "owner", " "), "job_packet_owned_capture_approval_missing"),
])
def test_blueprint_owned_capture_needs_the_list_and_the_owners_words(tmp_path, ids, approval, code) -> None:
    with pytest.raises(JobPacketRefused) as refused:
        build_job_packet(_job(tmp_path), output_root=_out(tmp_path), level="full", owned=approval,
                         owned_list=_owned_list(tmp_path, ids))
    assert refused.value.code == code


def test_full_level_for_an_allow_listed_capture_records_the_approval(tmp_path: Path) -> None:
    manifest = build_job_packet(_job(tmp_path), output_root=_out(tmp_path), level="full", owned=_owned(),
                                owned_list=_owned_list(tmp_path, [CAPTURE]), now=NOW)
    folder = Path(manifest["packet_dir"])
    basis = manifest["permission_basis"]
    assert basis["rule"] == "blueprint_owned_capture" and basis["approval_evidence"] == "Yes, share the reference job."
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
    assert "7.2 cm" in sheet and "scene_estimate:level" in sheet and "0.4–1.8 kg" in sheet
    assert "within 20 cm of the point (3.142, 2.718, 0.900) m, on the back shelf" in sheet
    assert "the site has not confirmed them" in sheet and "No safety assessment" in sheet
    readme = (folder / "README.md").read_text()
    assert "Isaac Sim 6.0.1" in readme and "Z_up_estimated_meters" in readme


def test_a_listed_sharing_statement_version_allows_the_full_level(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(packet, "SHARING_PERMISSION_STATEMENT_VERSIONS", ("2026-11-02.v2",))
    manifest = build_job_packet(_job(tmp_path), output_root=_out(tmp_path), level="full")
    assert manifest["permission_basis"] == {"rule": "site_sharing_permission", "statement_version": "2026-11-02.v2"}


def _withdraw(capture: Path) -> None:
    _write(capture.parent.parent / "website_withdrawal" / "tombstone.json", {"consent_revoked": True})


def _revoke(capture: Path) -> None:
    path = capture / "pipeline" / "website_task_context.json"
    value = json.loads(path.read_text())
    value["capture_rights"].update(consent_revoked=True, consent_status="revoked")
    path.write_text(json.dumps(value))


def _ledger(**fields):
    return lambda capture: _write(capture / "pipeline_job_ledger.json", {"status": "terminal_authority_ended", **fields})


@pytest.mark.parametrize("level", ["summary", "full"])
@pytest.mark.parametrize(("change", "code"), [
    (_withdraw, "job_packet_capture_withdrawn"),
    (_revoke, "job_packet_capture_withdrawn"),
    (_ledger(terminal_code="consent_expired"), "job_packet_capture_authority_ended"),
    (lambda capture: _write(capture / "pipeline_job_terminal_receipt.json", {"status": "authority_ended",
                                                                              "code": "source_revoked"}),
     "job_packet_capture_authority_ended"),
    (_ledger(terminal_code="website_run_closed_by_owner"), "job_packet_job_closed_by_owner"),
    (lambda capture: (capture / "pipeline_job_ledger.json").unlink(), "job_packet_job_ledger_missing"),
])
def test_withdrawn_ended_or_closed_jobs_are_refused_at_both_levels(tmp_path, level, change, code) -> None:
    inputs = _job(tmp_path)
    change(inputs.capture_dir)
    with pytest.raises(JobPacketRefused) as refused:
        build_job_packet(inputs, output_root=_out(tmp_path), level=level, owned=_owned(),
                         owned_list=_owned_list(tmp_path, [CAPTURE]))
    assert refused.value.code == code
    assert list((tmp_path / "out").iterdir()) == []


def test_a_capture_pulled_without_its_site_directory_is_refused(tmp_path: Path) -> None:
    inputs = _job(tmp_path)
    flat = tmp_path / "flat" / CAPTURE
    flat.parent.mkdir()
    inputs.capture_dir.rename(flat)
    with pytest.raises(JobPacketRefused) as refused:
        build_job_packet(JobInputs(capture_dir=flat), output_root=_out(tmp_path))
    assert refused.value.code == "job_packet_withdrawal_state_unknown"


def test_no_trial_and_no_simulation_are_said_not_invented(tmp_path: Path) -> None:
    manifest = build_job_packet(_job(tmp_path, trial="none", simulation=False), output_root=_out(tmp_path),
                                level="full", owned=_owned(), owned_list=_owned_list(tmp_path, [CAPTURE]))
    folder = Path(manifest["packet_dir"])
    sheet = (folder / "JOB_SHEET.md").read_text()
    assert "No early trial result is included" in sheet
    assert "The simulation: not given to the tool" in sheet and "The sealed task spec: not given" in sheet
    assert "What counts as done: not recorded" in sheet
    assert "carries no simulation files" in (folder / "README.md").read_text()
    assert not (folder / "files" / "simulation").exists()


def test_a_nonzero_trial_is_shown_with_its_label(tmp_path: Path) -> None:
    manifest = build_job_packet(_job(tmp_path, trial="three"), output_root=_out(tmp_path))
    sheet = (Path(manifest["packet_dir"]) / "JOB_SHEET.md").read_text()
    assert EARLY_TRIAL_LABEL in sheet and "3 of 10" in sheet


def test_packet_id_is_stable_for_the_same_inputs(tmp_path: Path) -> None:
    inputs = _job(tmp_path)
    kwargs = {"level": "full", "owned": _owned(), "owned_list": _owned_list(tmp_path, [CAPTURE])}
    first = build_job_packet(inputs, output_root=_out(tmp_path, "a"), **kwargs)
    second = build_job_packet(inputs, output_root=_out(tmp_path, "b"), **kwargs)
    assert first["packet_id"] == second["packet_id"]
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
    with pytest.raises(JobPacketRefused) as refused:
        build_job_packet(inputs, output_root=_out(tmp_path), level="full", owned=_owned(),
                         owned_list=_owned_list(tmp_path, [CAPTURE]))
    assert refused.value.code == "job_packet_input_symlink_outside_inputs"
    assert list((tmp_path / "out").iterdir()) == []


def test_done_sentence_from_every_field() -> None:
    sentence = done_sentence(_contract(), object_name="tray", place_label=None, assumed=False)
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
    assert words in done_sentence(contract, object_name="tray", place_label=None, assumed=False)
    assert "its name: not recorded" in done_sentence(_contract(), object_name=None, place_label=None, assumed=False)


def test_summary_first_look_from_the_website_must_be_checked_and_repeat_no_site_words(tmp_path: Path) -> None:
    inputs = _job(tmp_path)
    clean = _write(tmp_path / "look" / "first-look.md", b"Most likely fit: a fixed arm with a parallel gripper.")
    named = _write(tmp_path / "look" / "named.md", f"A fixed arm would suit {SITE_NAME}.".encode())
    with pytest.raises(JobPacketRefused, match="job_packet_first_look_unchecked"):
        build_job_packet(JobInputs(capture_dir=inputs.capture_dir, first_look_file=clean), output_root=_out(tmp_path))
    with pytest.raises(JobPacketRefused, match="job_packet_first_look_repeats_site_words"):
        build_job_packet(JobInputs(capture_dir=inputs.capture_dir, first_look_file=named), output_root=tmp_path / "out",
                         first_look_checked_by="operator", redact_terms=[SITE_NAME])
    manifest = build_job_packet(JobInputs(capture_dir=inputs.capture_dir, first_look_file=clean),
                                output_root=tmp_path / "out", first_look_checked_by="operator")
    sheet = (Path(manifest["packet_dir"]) / "JOB_SHEET.md").read_text()
    assert "the website's first look" in sheet and "parallel gripper" in sheet
    assert manifest["first_look_source"] == "website_export"
    assert "No early trial result is included" in sheet


def test_kind_of_job_falls_back_to_generic_words() -> None:
    assert packet.kind_of_job(None, {}) == "a pick-up-and-put-down job"
    assert packet.kind_of_job({"intake_request": {"task": {"strategy": "pick_and_place"}},
                               "destination": {"interior_bounds_world_m": {}}}, {}) == "put one object into a container"


def test_the_command_writes_and_refuses(tmp_path: Path, capsys) -> None:
    spec = importlib.util.spec_from_file_location(
        "export_job_packet", Path(__file__).resolve().parents[1] / "scripts" / "export_job_packet.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    inputs = _job(tmp_path)
    out = _out(tmp_path)
    assert script.main(["--capture-dir", str(inputs.capture_dir), "--output-root", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "written"
    assert script.main(["--capture-dir", str(inputs.capture_dir), "--output-root", str(_out(tmp_path, "o2")),
                        "--level", "full"]) == 2
    assert json.loads(capsys.readouterr().out)["code"] == "job_packet_full_level_requires_sharing_permission"
