"""A website navigation job's handoff writes the G1 lane's scene half, and its hold names only what is missing.

Beta journey step: L3 website preparation of a job whose robot walks to a goal in a scene kept as
filmed (docs/G1_NAVIGATION_FROM_COMPILED_SPEC.md, part C). The handoff compiles the task from the
collider, writes ``navigation_task_spec.json`` and ``navigation_scene_assets.json``, then the scene
half (``website_navigation_scene_inputs``: the G1 at the compiled start, its probe receipt and the
sealed record) and the team registry's derived row naming it. Consent withdrawn, revoked or expired
writes neither. No navigation job is ever sent to the website's scene intake.

Hermetic: a synthetic corridor collider, the navigation scene-inputs tests' synthetic room and a tiny
G1-shaped robot under a fixed test identity. The Atlas source and the A2 USD authoring are stubbed
(each is tested on its own); nothing reaches a provider, the WebApp, money or the host.
"""
# Covers: src/blueprint_pipeline/website_scene_handoff.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import native_g1_site_registration as registration
from blueprint_pipeline import website_capture_withdrawal as withdrawal
from blueprint_pipeline import website_scene_handoff as handoff
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture
from blueprint_pipeline.team_candidate_registry_reader import REGISTRY_ENV
from blueprint_pipeline.website_locomotion_hold import HOLD_CODE
from blueprint_pipeline.website_locomotion_task import BUILT_NAVIGATION_PIECES, SOURCE_IDENTITY_PIECE
from blueprint_pipeline.website_navigation_scene_inputs import (
    OUTPUT_DIRECTORY,
    SOURCE_UNBOUND,
    WebsiteNavigationSceneInputsError,
    read_navigation_scene_inputs,
    screening_source,
)
from blueprint_pipeline.website_preparation_revision import workspace
from blueprint_pipeline.website_task_context_contracts import holds_native_task_preparation
from tests.test_website_capture_withdrawal_team_scene import _registry
from tests.test_website_locomotion_task import _corridor
from tests.test_website_navigation_scene_inputs import _scene_assets, navigation_scene_inputs_arguments

trimesh = pytest.importorskip("trimesh")

NOW = 1_800_000_000.0
ATLAS = "blueprint_pipeline.website_atlas_scene_source.atlas_scene_source"
RIGHTS = {"consent_status": "granted", "consent_revoked": False, "consent_scope": ["robot_evaluation"],
          "derived_scene_generation_allowed": True}


def _context(**rights) -> dict:
    context = {"schema_version": "website_site_task_context.v1", "request_id": "req1", "scene_id": "site-req1",
               "capture_id": "walkthrough-req1", "task_family": "navigation", "confirmed": True,
               "confirmed_at": "2026-11-06T18:30:00Z",
               "description": "Walk down the lane to its far end without touching either side.",
               "capture_rights": {**RIGHTS, **rights}}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    return context


@pytest.fixture
def job(tmp_path, monkeypatch):
    """One navigation job's capture, its consent records, a ready Atlas scene, the G1 and the registry."""
    capture = tmp_path / "pubsub/source-bucket/scenes/site-req1/captures/walkthrough-req1"
    write_json(capture / "raw/manifest.json", {"site_submission_id": "req1", "scene_id": "site-req1",
                                               "capture_id": "walkthrough-req1", "consent_status": "granted"})
    removal = capture / "pipeline/removal.json"
    write_json(removal, {"entries": []})
    base, path = _corridor(tmp_path)
    base.update(splat_path=base["collision_mesh_path"], splat_digest=base["collision_mesh_digest"], up_axis="Y")
    monkeypatch.setattr(ATLAS, lambda **_: {
        "status": "ready", "base_scene": base, "alignment": "anchored", "camera_path_levelled_m": path,
        "source_geometry": {"digest": "atlas-geometry", "geometry_backend": "world_labs_atlas"}})
    # The A2 USDs for a scene kept as filmed (website_navigation_scene_assets), authored where it writes them.
    monkeypatch.setattr("blueprint_pipeline.website_navigation_scene_assets.navigation_scene_assets",
                        lambda *, base_scene, output_root: _scene_assets(Path(output_root)))
    # The G1 registration's pinned Arena USD, found as it finds it: the manipulation packet's scene plan.
    (tmp_path / "fixture").mkdir()
    robot = navigation_scene_inputs_arguments(tmp_path / "fixture", monkeypatch)["robot_record"]
    packet = tmp_path / "g1-shared/manipulation"
    write_json(packet / "native_task_arena_scene_plan.v1.json", {"robot": {"usd_path": robot["path"]}})
    manifest = {"robot_asset": {"manipulation_packet_dir": str(packet), "sha256": robot["sha256"],
                                "size_bytes": robot["size_bytes"]}}
    monkeypatch.setattr(registration, "load_shared_inputs", lambda *_args: manifest)
    registry, derived = tmp_path / "etc/team-evaluation-registry.json", tmp_path / "inputs/team-scene-profiles"
    _registry(registry, derived)
    monkeypatch.setenv(REGISTRY_ENV, str(registry))
    # A navigation job never reaches the website's scene intake, whatever its consent says.
    monkeypatch.setattr("blueprint_pipeline.website_task_context.enqueue_website_prepared_scene",
                        lambda **_: pytest.fail("a navigation job has no website scene intake"))
    authority = {"schema_version": "website_scene_sponsorship.v1", "expires_at_epoch": NOW + 3600}
    return {"capture": capture, "removal": removal, "authority": authority, "derived": derived,
            "manifest": manifest, "robot": robot, "root": workspace(capture)}


def _run(job, *, context=None, authority=None):
    context = context or _context()
    descriptor = {"capture_id": "walkthrough-req1", "scene_id": "site-req1", "metadata": {
        "site_task_context": context,
        "website_scene_execution_authority": job["authority"] if authority is None else authority}}
    clean_plate = {"privacy_verified": True, "status": "noop", "source_geometry": None, "task_masks": {},
                   "removal_manifest_path": str(job["removal"])}
    return handoff.prepare_website_scene_handoff(descriptor=descriptor, clean_plate=clean_plate,
                                                 capture_root=job["capture"], now=NOW), context


def _missing(result) -> set[str]:
    return {item for row in result["native_task_preparation_gap"]["runnable_options"]["policies"]
            for item in row["missing"]}


def _rows(job) -> list[Path]:
    return sorted(job["derived"].glob("*.json")) if job["derived"].is_dir() else []


def test_the_handoff_writes_the_scene_half_and_registers_it(job):
    result, context = _run(job)
    assert result["status"] == "needs_input" and result["blockers"] == [HOLD_CODE]
    written = result["navigation_scene_inputs"]
    assert written["status"] == "written" and written["registry"]["status"] == "written"
    # The record sits in the capture's own workspace, where the site's withdrawal deletes it.
    path = Path(written["path"])
    assert path.is_relative_to(job["root"] / OUTPUT_DIRECTORY) and path.is_relative_to(job["capture"])
    record = read_navigation_scene_inputs(path)
    assert record["scene_inputs_digest"] == written["digest"]
    assert record["navigation_task_spec_digest"] == result["navigation_task_spec_digest"]
    assert record["scene_assets_digest"] == result["navigation_scene_assets_digest"]
    assert record["robot"]["usd_sha256"] == job["robot"]["sha256"]
    # The registry reads the derived row the handoff wrote, naming this record.
    profile = profile_for_capture(job["capture"])
    assert profile["navigation_scene_inputs_path"] == str(path)
    assert profile["scene_revision_digest"] == record["scene_revision_digest"]
    assert profile["robot_free_probe_root"] == str(path.parent) and profile["scenario_id"] == "scene.nominal"
    # The hold no longer names what this job has, and the hold contract re-derives it.
    built = result["native_task_preparation_gap"]["runnable_options"]["navigation_pieces_built"]
    assert {"navigation_task_spec_from_compiled_spec", "navigation_scene_assets",
            "g1_navigation_preparation"} <= set(built)
    assert not {"pipeline:g1_navigation_preparation", "pipeline:navigation_scene_assets",
                "pipeline:navigation_task_spec_from_compiled_spec"} & _missing(result)
    retained = json.loads((job["capture"] / "pipeline/website_scene_preparation/handoff.json").read_text())
    assert retained == result and holds_native_task_preparation(retained, context)
    # No website intake exists for it, so the screening source still refuses by name.
    assert "website_intake_outbox" not in result
    with pytest.raises(WebsiteNavigationSceneInputsError, match=SOURCE_UNBOUND):
        screening_source(record)


def test_once_built_the_hold_names_only_what_is_still_lacking(job):
    """Of the pipeline, only the website intake a screening source needs; then each body's and catalog row's own."""
    from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog
    from blueprint_pipeline.native_g1_navigation_goal import NAVIGATION_TASK_KIND
    from blueprint_pipeline.task_evaluation_embodiment_registry import EMBODIMENT_REGISTRY
    from blueprint_pipeline.task_evaluation_g1_catalog import G1_EMBODIMENT_ID

    result, context = _run(job)
    assert result["status"] == "needs_input" and result["blockers"] == [HOLD_CODE]
    options = result["native_task_preparation_gap"]["runnable_options"]
    assert options["navigation_pieces_built"] == sorted(BUILT_NAVIGATION_PIECES)
    bodies = {row.embodiment_id: row for row in EMBODIMENT_REGISTRY}
    rows = {row["candidate_id"]: row for row in load_default_catalog()["candidates"]}
    assert options["policies"]
    for policy in options["policies"]:
        kinds = rows[policy["candidate_id"]]["supported_task_kinds"]
        body_missing, _ = bodies[policy["embodiment_id"]].development_run_missing(NAVIGATION_TASK_KIND)
        expected = [f"embodiment:{item}" for item in body_missing]
        expected.append(f"pipeline:{SOURCE_IDENTITY_PIECE}")
        expected += [f"catalog:{code}" for code in rows[policy["candidate_id"]]["eligibility"]["blockers"]]
        if NAVIGATION_TASK_KIND not in kinds:
            expected.append("catalog:task_kind_not_supported_by_candidate" if kinds
                            else "catalog:task_kind_support_unverified")
        assert policy["missing"] == expected
        # No body declares a development-run allowance today, so each body's own gaps are named too.
        assert {f"embodiment:{item}" for item in bodies[policy["embodiment_id"]].missing} <= set(policy["missing"])
        assert "claim_ceiling" not in policy and "development_run_allowance" not in policy
    assert {policy["embodiment_id"] for policy in options["policies"]} == {G1_EMBODIMENT_ID}
    assert holds_native_task_preparation(result, context)


def test_a_second_handoff_finds_the_same_record_and_row(job):
    first, _ = _run(job)
    row = _rows(job)
    before = row[0].read_bytes()
    second, _ = _run(job)
    assert second["navigation_scene_inputs"] == first["navigation_scene_inputs"]
    assert _rows(job) == row and row[0].read_bytes() == before


def test_a_changed_record_is_refused_by_its_readers_and_written_again_by_the_next_handoff(job):
    first, _ = _run(job)
    path = first["navigation_scene_inputs"]["path"]
    record = read_navigation_scene_inputs(path)
    plan = Path(record["packet_root"]) / "native_task_arena_scene_plan.v1.json"
    plan.chmod(0o644)
    plan.write_text(plan.read_text().replace("unitree_g1", "unitree_g2", 1))
    # Nothing reads the changed record as a scene half: its reader and the registry's scene facts refuse it.
    with pytest.raises(WebsiteNavigationSceneInputsError, match="website_navigation_scene_inputs_packet_changed"):
        read_navigation_scene_inputs(path)
    from blueprint_pipeline.team_candidate_quick10_registry import TeamCandidateRegistryError, scene_facts
    with pytest.raises(TeamCandidateRegistryError):
        scene_facts(profile_for_capture(job["capture"]))
    # The next handoff keeps the changed bytes aside, inside the capture, and writes the record again.
    second, _ = _run(job)
    assert second["navigation_scene_inputs"] == first["navigation_scene_inputs"]
    assert read_navigation_scene_inputs(path) == record
    kept = list(Path(record["packet_root"]).parent.glob(".native-task-packet.incomplete-*"))
    assert len(kept) == 1 and kept[0].is_relative_to(job["capture"])
    assert scene_facts(profile_for_capture(job["capture"]))["inputs"] == record
    assert not {"pipeline:g1_navigation_preparation"} & _missing(second)


def test_a_failed_writer_is_this_jobs_missing_piece_and_the_job_goes_on(job, tmp_path, monkeypatch):
    manifest = {"robot_asset": {**job["manifest"]["robot_asset"],
                                "manipulation_packet_dir": str(tmp_path / "not-restored")}}
    monkeypatch.setattr(registration, "load_shared_inputs", lambda *_args: manifest)
    result, context = _run(job)
    assert result["status"] == "needs_input" and result["blockers"] == [HOLD_CODE]
    assert result["navigation_scene_inputs"] == {"status": "failed", "error": "g1_shared_robot_asset_missing"}
    missing = _missing(result)
    assert "pipeline:g1_navigation_preparation" in missing and "pipeline:navigation_scene_assets" not in missing
    assert not (job["root"] / OUTPUT_DIRECTORY).exists() and _rows(job) == []
    assert holds_native_task_preparation(result, context)


def test_failed_scene_files_name_both_pieces_and_write_no_scene_half(job, monkeypatch):
    def unavailable(**_):
        raise ValueError("website_native_appearance_runtime_unavailable")

    monkeypatch.setattr("blueprint_pipeline.website_navigation_scene_assets.navigation_scene_assets", unavailable)
    result, _ = _run(job)
    assert "navigation_scene_inputs" not in result and _rows(job) == []
    assert {"pipeline:navigation_scene_assets", "pipeline:g1_navigation_preparation"} <= _missing(result)


@pytest.mark.parametrize("consent,reason", [
    ("withdrawn", "website_capture_withdrawn"),
    ("expired", "website_scene_consent_expired"),
    ("rights_revoked", "website_scene_processing_rights_required"),
    ("no_sponsorship", "website_scene_execution_authority_required"),
])
def test_without_current_consent_no_scene_half_no_registration_no_intake_no_source(job, consent, reason):
    context, authority = _context(), None
    if consent == "withdrawn":
        withdrawal.acknowledge_withdrawal(capture_root=job["capture"], command={
            "schema_version": "website_capture_withdrawal.v1", "request_id": "req1", "scene_id": "site-req1",
            "capture_id": "walkthrough-req1", "withdrawal_id": "withdrawal-req1",
            "requested_at_iso": "2026-11-06T00:00:00Z"})
    elif consent == "expired":
        authority = {**job["authority"], "expires_at_epoch": NOW}
    elif consent == "rights_revoked":
        context = _context(consent_status="revoked", consent_revoked=True, derived_scene_generation_allowed=False)
    else:
        authority = {}
    result, _ = _run(job, context=context, authority=authority)
    assert result["navigation_scene_inputs"] == {"status": "not_written", "reason": reason}
    # Nothing a screening source could come from: no record, no registry row, no intake.
    assert not (job["root"] / OUTPUT_DIRECTORY).exists() and _rows(job) == []
    assert profile_for_capture(job["capture"]) is None and "website_intake_outbox" not in result
    assert "pipeline:g1_navigation_preparation" in _missing(result)


def test_the_sites_withdrawal_deletes_everything_this_handoff_wrote(job, tmp_path):
    from blueprint_pipeline import website_capture_withdrawal_deletion as deletion
    from tests import test_website_capture_withdrawal_deletion as base

    result, _ = _run(job)
    record = read_navigation_scene_inputs(result["navigation_scene_inputs"]["path"])
    written = sorted(path for path in job["capture"].rglob("*") if path.is_file())
    for name in ("navigation_task_spec_path", "navigation_scene_assets_path"):
        assert Path(result[name]) in written
    assert Path(record["robot_free_probe"]["path"]) in written and Path(record["packet_root"]).is_dir()
    rows = _rows(job)
    assert len(rows) == 1
    withdrawal.acknowledge_withdrawal(capture_root=job["capture"], command={
        "schema_version": "website_capture_withdrawal.v1", "request_id": "req1", "scene_id": "site-req1",
        "capture_id": "walkthrough-req1", "withdrawal_id": "withdrawal-req1",
        "requested_at_iso": "2026-11-07T00:00:00Z"})
    resolution = {"run_roots": (), "intent_root": None, "binding_root": None, "uncovered_roots": (),
                  "object_store": deletion.B2ObjectVersions(base.FakeB2(), base.BUCKET),
                  "screening_queue_root": None, "screening_work_root": None, "native_g1_registration_root": None,
                  "team_scene_halves_root": None, "team_registry": tmp_path / "etc/team-evaluation-registry.json",
                  "team_scene_packet_queue_root": None, "rigid_probe_queue_root": None,
                  "rigid_probe_work_root": None}
    plan = deletion.plan_withdrawal_deletion(capture_root=job["capture"], **resolution)
    assert plan["unresolved"] == [] and [row[0] for row in plan["team_scene"]["derived_profile_rows"]] == [
        str(rows[0])]
    deletion.apply_withdrawal_deletion(capture_root=job["capture"], expected_plan_digest=plan["digest"],
                                       authorize_complete_deletion=True, **resolution)
    assert not any(path.exists() for path in (*written, *rows))
    assert [path for path in job["capture"].parents[1].rglob("*")
            if path.is_file() and "website_withdrawal" not in path.parts] == []
    assert profile_for_capture(job["capture"]) is None
