"""A website navigation job's handoff writes the G1 lane's scene half, and its hold names only what is missing.

Beta journey step: L3 website preparation of a job whose robot walks to a goal in a scene kept as
filmed (docs/G1_NAVIGATION_FROM_COMPILED_SPEC.md, part C). The handoff compiles the task from the
collider, writes ``navigation_task_spec.json`` and ``navigation_scene_assets.json``, then the scene
half (``website_navigation_scene_inputs``: the G1 at the compiled start, its probe receipt and the
sealed record) and the team registry's derived row naming it. Consent withdrawn, revoked or expired
writes neither. Once the scene half is written the job's website navigation intake is sent with the
job's own sponsorship records (``website_navigation_intake``); the website's refusal holds the job by
its name, and a staged intent's receipt binds the screening source.

Hermetic: a synthetic corridor collider, the navigation scene-inputs tests' synthetic room and a tiny
G1-shaped robot under a fixed test identity. The Atlas source and the A2 USD authoring are stubbed
(each is tested on its own). The website is a double of its signed ``prepared-scene`` answer (the
website's own route is tested in the WebApp); the intent is staged by the real intake. Nothing reaches
a provider, money or the host.
"""
# Covers: src/blueprint_pipeline/website_scene_handoff.py
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline import native_g1_site_registration as registration
from blueprint_pipeline import website_capture_withdrawal as withdrawal
from blueprint_pipeline import website_scene_handoff as handoff
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.task_evaluation_scene_intake import stage_scene_intent
from blueprint_pipeline.task_evaluation_scene_intent_contracts import validate_request
from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture
from blueprint_pipeline.team_candidate_registry_reader import REGISTRY_ENV
from blueprint_pipeline.website_locomotion_hold import HOLD_CODE
from blueprint_pipeline.website_locomotion_task import BUILT_NAVIGATION_PIECES, SOURCE_IDENTITY_PIECE
from blueprint_pipeline.website_navigation_intake import (
    OUTBOX_FILENAME,
    RECEIPT_FILENAME,
    read_navigation_intake_receipt,
)
from blueprint_pipeline.website_navigation_scene_inputs import (
    OUTPUT_DIRECTORY,
    SOURCE_UNBOUND,
    WebsiteNavigationSceneInputsError,
    read_navigation_scene_inputs,
    screening_source,
)
from blueprint_pipeline.website_preparation_revision import workspace
from blueprint_pipeline.website_task_context_contracts import (
    holds_native_task_preparation,
    retains_navigation_preparation,
)
from tests.test_website_capture_withdrawal_team_scene import _registry
from tests.test_website_locomotion_task import _corridor
from tests.test_website_navigation_scene_inputs import _scene_assets, navigation_scene_inputs_arguments

trimesh = pytest.importorskip("trimesh")

NOW = 1_800_000_000.0
ATLAS = "blueprint_pipeline.website_atlas_scene_source.atlas_scene_source"
RIGHTS = {"consent_status": "granted", "consent_revoked": False, "consent_scope": ["robot_evaluation"],
          "derived_scene_generation_allowed": True}


OWNER = {"user_id": "blueprint", "organization_id": "blueprint"}
UNCONFIRMED = "website_control_prepared-scene_http_409:website_task_context_not_confirmed"


def _context(*, confirmed: bool = True, **rights) -> dict:
    context = {"schema_version": "website_site_task_context.v1", "request_id": "req1", "scene_id": "site-req1",
               "capture_id": "walkthrough-req1", "task_family": "navigation", "confirmed": confirmed,
               "confirmed_at": "2026-11-06T18:30:00Z" if confirmed else None,
               "description": "Walk down the lane to its far end without touching either side.",
               "capture_rights": {**RIGHTS, **rights}, **({} if confirmed else {"purpose": "scene_preparation"})}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    return context


def _authority(context: dict) -> dict:
    """The job's scene sponsorship as the website issues it: ``task_confirmed`` is the brief's own reading."""
    return {"schema_version": "website_scene_sponsorship.v1", "sponsor": "blueprint", "owner": dict(OWNER),
            "request_id": "req1", "scene_id": "site-req1", "capture_id": "walkthrough-req1",
            "task_context_digest": context["context_digest"], "max_total_spend_usd": 20, "max_paid_attempts": 2,
            "expires_at_epoch": NOW + 3600, "authoring_provider": "openai",
            "consent": {"rights_reference": canonical_digest(context["capture_rights"]),
                        "provider_terms_reference": "sha256:" + "c" * 64, "accepted_by": OWNER["user_id"],
                        "accepted_at_epoch": NOW - 60, "private_processing_authorized": True,
                        "provider_training_authorized": False, "task_confirmed": context["confirmed"],
                        "spend_authorized": True}}


@pytest.fixture
def job(tmp_path, monkeypatch):
    """One navigation job's capture, its consent records, a ready Atlas scene, the G1 and the registry."""
    capture = tmp_path / "pubsub/source-bucket/scenes/site-req1/captures/walkthrough-req1"
    write_json(capture / "raw/manifest.json", {"site_submission_id": "req1", "scene_id": "site-req1",
                                               "capture_id": "walkthrough-req1", "consent_status": "granted"})
    removal = capture / "pipeline/removal.json"
    write_json(removal, {"entries": []})
    base, path = _corridor(tmp_path)
    base.update(splat_path=base["collision_mesh_path"], splat_digest=base["collision_mesh_digest"], up_axis="Y",
                splat_binding_id="website-splat-" + base["collision_mesh_digest"][7:39])
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
    # The website's signed prepared-scene answer: it holds the intake (state forward_pending) under the id
    # the Pipeline's intake stages it with, or refuses an unconfirmed brief by name, as its route does.
    sent = []

    def website(*, capture_id, operation, payload, **_):
        assert operation == "prepared-scene" and capture_id == "walkthrough-req1"
        sent.append(payload["request"])
        request = payload["request"]
        if request["consent"]["task_confirmed"] is not True:
            raise ValueError(UNCONFIRMED)
        identity = cross_runtime_canonical_digest({"owner": request["owner"], "submission_id": request["submission_id"]})
        return {"id": "scene-" + identity[7:], "request_digest": cross_runtime_canonical_digest(request),
                "state": "forward_pending"}

    monkeypatch.setattr("blueprint_pipeline.website_task_context.website_webapp_request", website)
    intents = tmp_path / "control-plane/task-evaluation-scene-intents"
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", str(intents))
    return {"capture": capture, "removal": removal, "authority": _authority(_context()), "derived": derived,
            "manifest": manifest, "robot": robot, "root": workspace(capture), "sent": sent, "intents": intents}


def _stage(job) -> dict:
    """The website forwards the held intake and the Pipeline's own intake stages its intent."""
    return stage_scene_intent(value=job["sent"][-1], queue_root=job["intents"], authenticated_client="blueprint-webapp",
                              trusted_clients={"blueprint-webapp"}, now=NOW)


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
    # Its website intake is held by the website; until the Pipeline stages its intent there is no receipt,
    # so the source identity is this job's missing piece and the screening source refuses by name.
    assert "website_intake_outbox" not in result
    assert result["navigation_intake"]["status"] == "forward_pending" and result["navigation_intake"]["receipt"] is None
    assert (path.parent / OUTBOX_FILENAME).is_file() and not (path.parent / RECEIPT_FILENAME).exists()
    assert f"pipeline:{SOURCE_IDENTITY_PIECE}" in _missing(result)
    with pytest.raises(WebsiteNavigationSceneInputsError, match=SOURCE_UNBOUND):
        screening_source(record)


def test_once_built_the_hold_names_only_what_is_still_lacking(job):
    """Of the pipeline, only the website intake a screening source needs; then each body's and catalog row's own.

    The G1's development-run allowance stands in for its two items on each navigation checkpoint (owner
    decisions 2026-11-07), so the hold names neither and records the allowance, development_only.
    """
    from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog
    from blueprint_pipeline.native_g1_navigation_goal import NAVIGATION_TASK_KIND
    from blueprint_pipeline.task_evaluation_embodiment_registry import EMBODIMENT_REGISTRY, STOOD_IN
    from blueprint_pipeline.task_evaluation_g1_catalog import G1_EMBODIMENT_ID

    result, context = _run(job)
    assert result["status"] == "needs_input" and result["blockers"] == [HOLD_CODE]
    gap = result["native_task_preparation_gap"]
    options = gap["runnable_options"]
    # Every piece but the source identity, which this job has only once its own intake receipt exists.
    assert options["navigation_pieces_built"] == sorted(BUILT_NAVIGATION_PIECES - {SOURCE_IDENTITY_PIECE})
    bodies = {row.embodiment_id: row for row in EMBODIMENT_REGISTRY}
    rows = {row["candidate_id"]: row for row in load_default_catalog()["candidates"]}
    assert options["policies"]
    for policy in options["policies"]:
        kinds = rows[policy["candidate_id"]]["supported_task_kinds"]
        body_missing, _ = bodies[policy["embodiment_id"]].development_run_missing(
            NAVIGATION_TASK_KIND, candidate_id=policy["candidate_id"])
        expected = [f"embodiment:{item}" for item in body_missing]
        expected.append(f"pipeline:{SOURCE_IDENTITY_PIECE}")
        expected += [f"catalog:{code}" for code in rows[policy["candidate_id"]]["eligibility"]["blockers"]]
        if NAVIGATION_TASK_KIND not in kinds:
            expected.append("catalog:task_kind_not_supported_by_candidate" if kinds
                            else "catalog:task_kind_support_unverified")
        assert policy["missing"] == expected
        # Stood in, not satisfied: neither item is named missing, and the allowance is recorded.
        assert not {"embodiment:verified_site_episode", "embodiment:reviewed_checkpoint_terms"} & set(policy["missing"])
        assert policy["claim_ceiling"] == "development_only"
        stood_in = policy["development_run_allowance"]["stood_in"]
        assert sorted(stood_in) == ["reviewed_checkpoint_terms", "verified_site_episode"]
        assert {entry["status"] for entry in stood_in.values()} == {STOOD_IN}
    assert {policy["embodiment_id"] for policy in options["policies"]} == {G1_EMBODIMENT_ID}
    # The hold itself names only what is still lacking, and says what the allowance stands in for.
    assert gap["missing"] == [f"pipeline:{SOURCE_IDENTITY_PIECE}"]
    assert STOOD_IN in gap["reason"]
    # The body stays listed as it is: unavailable, with both items.
    assert next(row for row in options["embodiments"] if row["embodiment_id"] == G1_EMBODIMENT_ID)["missing"] == [
        "verified_site_episode", "reviewed_checkpoint_terms"]
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
    assert "navigation_intake" not in result and job["sent"] == []
    assert "pipeline:g1_navigation_preparation" in _missing(result)


def test_a_confirmed_brief_sends_the_jobs_real_records_and_its_receipt_binds_the_source(job):
    """The intake is the record's digests with the job's own sponsorship; the staged intent's receipt names the source."""
    first, context = _run(job)
    path = Path(first["navigation_scene_inputs"]["path"])
    record = read_navigation_scene_inputs(path)
    assert len(job["sent"]) == 1
    request = job["sent"][0]
    authority = job["authority"]
    # Owner, consent and rights exactly as the job's sponsorship carries them: task_confirmed is the
    # website's own reading of the brief, never set here. Execution as the sponsorship states it.
    assert request["owner"] == authority["owner"] and request["consent"] == authority["consent"]
    assert request["consent"]["task_confirmed"] is True and request["submission_id"] == context["capture_id"]
    assert request["execution"] == {"purpose": "scene_preparation", "max_total_spend_usd": 20, "max_paid_attempts": 2,
                                    "max_retries": 0, "expires_at_epoch": NOW + 3600,
                                    "allowed_providers": ["vast", "openai"], "policy_candidates": [],
                                    "claim_scope": "development_only"}
    assert request["task"] == {"task_id": record["task_id"], "strategy": "goal_region_navigation",
                               "geometry_origin": "kept_as_filmed",
                               **{key: record[key] for key in (
                                   "navigation_task_spec_digest", "compiled_task_spec_digest",
                                   "task_success_contract_digest", "runtime_contract_digest",
                                   "scene_revision_digest")}}
    assert request["task"]["task_id"] == "website-" + context["context_digest"][7:27]
    assert validate_request(request, now=NOW) == request
    # The website forwards it; the Pipeline's own intake stages the intent. A later pass binds its receipt.
    staged = _stage(job)
    second, _ = _run(job)
    receipt = read_navigation_intake_receipt(path)
    source = {"scene_intent_id": staged["intent_id"], "scene_intent_digest": staged["intent_digest"],
              "capture_id": "walkthrough-req1", "task_id": record["task_id"],
              "task_context_digest": context["context_digest"], "scene_revision_digest": record["scene_revision_digest"]}
    assert receipt["source"] == source and second["navigation_intake"]["receipt"]["source"] == source
    assert receipt["reserved_usd"] == 0 and receipt["construction_allowance_granted"] is False
    assert screening_source(record) == source
    # The source identity was this job's only missing piece (the G1's development-run allowance stands in
    # for its body items). Built now, every runnable option is a G1 navigation checkpoint with its lane
    # inputs in place, so the job is handed off to screening: development_only, no result claimed.
    assert _missing(first) == {f"pipeline:{SOURCE_IDENTITY_PIECE}"}
    assert second["status"] == "intake_ready" and second["blockers"] == []
    handed = second["navigation_preparation"]
    assert handed["lane"] == "g1_navigation_screening" and handed["claim_ceiling"] == "development_only"
    assert handed["result_claimed"] is False and "No result yet" in handed["summary"]
    assert handed["source"] == source and handed["candidates"]
    assert all(candidate.endswith("_vision_navi") for candidate in handed["candidates"])
    assert "native_task_preparation_gap" not in second
    assert retains_navigation_preparation(second, context) and not holds_native_task_preparation(second, context)
    retained = json.loads((job["capture"] / "pipeline/website_scene_preparation/handoff.json").read_text())
    assert retained == second
    # Bound to this task context: another context's record is not this job's.
    assert not retains_navigation_preparation(second, _context(confirmed=False))
    # The website already holds this record's intake: a later pass sends nothing and changes nothing.
    assert job["sent"] == [request] and read_navigation_intake_receipt(path) == receipt


def test_a_receipt_binds_only_the_exact_staged_request_and_never_a_withdrawn_capture(job):
    from blueprint_pipeline.website_navigation_intake import (
        WebsiteNavigationIntakeError, bind_navigation_intake_receipt,
    )

    result, _ = _run(job)
    path = Path(result["navigation_scene_inputs"]["path"])
    # Nothing staged yet: no receipt, nothing written.
    assert bind_navigation_intake_receipt(record_path=path, capture_root=job["capture"]) is None
    # An intent staged for another request under the same identity binds nothing.
    other = copy.deepcopy(job["sent"][-1])
    other["execution"]["expires_at_epoch"] = NOW + 1800
    stage_scene_intent(value=other, queue_root=job["intents"], authenticated_client="blueprint-webapp",
                       trusted_clients={"blueprint-webapp"}, now=NOW)
    with pytest.raises(WebsiteNavigationIntakeError, match="^website_navigation_intake_intent_mismatch$"):
        bind_navigation_intake_receipt(record_path=path, capture_root=job["capture"])
    assert not (path.parent / RECEIPT_FILENAME).exists()


def test_a_withdrawn_capture_gets_no_receipt(job):
    from blueprint_pipeline.website_navigation_intake import (
        WebsiteNavigationIntakeError, bind_navigation_intake_receipt,
    )

    result, _ = _run(job)
    path = Path(result["navigation_scene_inputs"]["path"])
    _stage(job)
    withdrawal.acknowledge_withdrawal(capture_root=job["capture"], command={
        "schema_version": "website_capture_withdrawal.v1", "request_id": "req1", "scene_id": "site-req1",
        "capture_id": "walkthrough-req1", "withdrawal_id": "withdrawal-req1",
        "requested_at_iso": "2026-11-07T00:00:00Z"})
    with pytest.raises(WebsiteNavigationIntakeError, match="^website_capture_withdrawn$"):
        bind_navigation_intake_receipt(record_path=path, capture_root=job["capture"])
    assert not (path.parent / RECEIPT_FILENAME).exists()


def test_a_runnable_option_on_a_body_with_no_lane_keeps_the_typed_route_missing(job, monkeypatch):
    """Only the G1 navigation checkpoints have a lane here; any other runnable body is still a typed gap."""
    from blueprint_pipeline import website_locomotion_task as locomotion

    real = locomotion.runnable_options

    def another_body(requirements, **kwargs):
        options = real(requirements, **kwargs)
        return {**options, "policies": [{**row, "embodiment_id": "another_walking_body"} if row["runnable"] else row
                                        for row in options["policies"]]}

    monkeypatch.setattr(locomotion, "runnable_options", another_body)
    _run(job)
    _stage(job)
    result, context = _run(job)
    assert result["navigation_intake"]["receipt"] is not None
    assert result["status"] == "awaiting_inputs"
    assert result["blockers"] == ["website_locomotion_task_execution_route_missing"]
    assert "navigation_preparation" not in result and not retains_navigation_preparation(result, context)


def test_an_unconfirmed_brief_is_refused_by_the_website_and_the_job_holds_by_that_name(job):
    context = _context(confirmed=False)
    result, _ = _run(job, context=context, authority=_authority(context))
    # The Pipeline sent the website's own reading (unconfirmed); the website refused it by name.
    assert [request["consent"]["task_confirmed"] for request in job["sent"]] == [False]
    assert result["navigation_intake"] == {"status": "refused", "reason": "website_task_context_not_confirmed"}
    assert result["status"] == "needs_input" and result["blockers"] == ["website_task_context_not_confirmed"]
    gap = result["native_task_preparation_gap"]
    assert gap["code"] == "website_task_context_not_confirmed" and gap["job_disposition"] == "held_for_preparation"
    assert f"pipeline:{SOURCE_IDENTITY_PIECE}" in {item for row in gap["runnable_options"]["policies"]
                                                   for item in row["missing"]}
    assert holds_native_task_preparation(result, context)
    # The scene half is kept; nothing of an intake exists and the screening source refuses by name.
    path = Path(result["navigation_scene_inputs"]["path"])
    assert result["navigation_scene_inputs"]["status"] == "written"
    assert not (path.parent / OUTBOX_FILENAME).exists() and not (path.parent / RECEIPT_FILENAME).exists()
    assert not job["intents"].exists()
    with pytest.raises(WebsiteNavigationSceneInputsError, match=SOURCE_UNBOUND):
        screening_source(read_navigation_scene_inputs(path))


def test_a_website_that_cannot_answer_is_this_jobs_missing_piece(job, monkeypatch):
    def unavailable(**_):
        raise ValueError("website_control_prepared-scene_http_503:website_scene_sponsorship_unavailable")

    monkeypatch.setattr("blueprint_pipeline.website_task_context.website_webapp_request", unavailable)
    result, context = _run(job)
    assert result["navigation_intake"]["status"] == "failed"
    assert result["blockers"] == [HOLD_CODE] and f"pipeline:{SOURCE_IDENTITY_PIECE}" in _missing(result)
    assert holds_native_task_preparation(result, context)


def test_the_sites_withdrawal_deletes_everything_this_handoff_wrote(job, tmp_path):
    from blueprint_pipeline import website_capture_withdrawal_deletion as deletion
    from tests import test_website_capture_withdrawal_deletion as base

    _run(job)
    staged = _stage(job)
    result, _ = _run(job)
    record = read_navigation_scene_inputs(result["navigation_scene_inputs"]["path"])
    written = sorted(path for path in job["capture"].rglob("*") if path.is_file())
    for name in ("navigation_task_spec_path", "navigation_scene_assets_path"):
        assert Path(result[name]) in written
    assert Path(record["robot_free_probe"]["path"]) in written and Path(record["packet_root"]).is_dir()
    # The intake's outbox and receipt sit beside the record, inside the capture.
    intake = [Path(record["packet_root"]).parent / name for name in (OUTBOX_FILENAME, RECEIPT_FILENAME)]
    assert all(path in written for path in intake)
    rows = _rows(job)
    assert len(rows) == 1
    withdrawal.acknowledge_withdrawal(capture_root=job["capture"], command={
        "schema_version": "website_capture_withdrawal.v1", "request_id": "req1", "scene_id": "site-req1",
        "capture_id": "walkthrough-req1", "withdrawal_id": "withdrawal-req1",
        "requested_at_iso": "2026-11-07T00:00:00Z"})
    resolution = {"run_roots": (), "intent_root": job["intents"], "binding_root": None, "uncovered_roots": (),
                  "object_store": deletion.B2ObjectVersions(base.FakeB2(), base.BUCKET),
                  "screening_queue_root": None, "screening_work_root": None, "native_g1_registration_root": None,
                  "team_scene_halves_root": None, "team_registry": tmp_path / "etc/team-evaluation-registry.json",
                  "team_scene_packet_queue_root": None, "rigid_probe_queue_root": None,
                  "rigid_probe_work_root": None}
    plan = deletion.plan_withdrawal_deletion(capture_root=job["capture"], **resolution)
    assert plan["unresolved"] == [] and [row[0] for row in plan["team_scene"]["derived_profile_rows"]] == [
        str(rows[0])]
    # The intake's staged intent is the site's lineage, found from its outbox in the capture.
    assert plan["lineage"]["intents"] == [staged["intent_id"]]
    receipt = deletion.apply_withdrawal_deletion(capture_root=job["capture"], expected_plan_digest=plan["digest"],
                                                 authorize_complete_deletion=True, **resolution)
    assert not any(path.exists() for path in (*written, *rows, *intake))
    # The intent record (identifiers, no capture bytes) is named as retained audit, as every intake kind's is.
    retained = next(row for row in receipt["remaining"] if row["holder"] == "retained_audit_and_lineage_metadata")
    assert retained["intents"] == [staged["intent_id"]]
    assert [path for path in job["capture"].parents[1].rglob("*")
            if path.is_file() and "website_withdrawal" not in path.parts] == []
    assert profile_for_capture(job["capture"]) is None


def test_the_library_shape_a_brief_confirmed_after_preparation_reaches_the_intake_with_nothing_bought_again(
        job, monkeypatch):
    """Prepared and held while unconfirmed, then confirmed on the website: the replayed stage sends the intake
    with the confirmed consent and the successor's task id, reading every paid input under the sealed parent."""
    from types import SimpleNamespace

    from blueprint_pipeline import website_candidate_ranking, website_gemini_receipts
    from blueprint_pipeline.website_capture_preparation import run_website_capture_preparation
    from blueprint_pipeline.website_task_context_lineage import record_confirmation_lineage

    for module, name in ((website_gemini_receipts, "retained_gemini_call"), (website_candidate_ranking, "_gemini")):
        monkeypatch.setattr(module, name, lambda *_, **__: pytest.fail("nothing paid is bought again"))
    parent = _context(confirmed=False)
    first, _ = _run(job, context=parent, authority=_authority(parent))
    assert first["blockers"] == ["website_task_context_not_confirmed"]
    assert job["sent"][-1]["consent"]["task_confirmed"] is False
    # The site confirms its brief on the website (14:30:45Z): the website's own reading, only that changed.
    successor = {key: value for key, value in parent.items() if key != "context_digest"}
    successor.update(confirmed=True, confirmed_at="2026-11-07T14:30:45Z")
    successor["context_digest"] = canonical_digest(successor, digest_field="context_digest")
    pipeline = job["capture"] / "pipeline"
    write_json(pipeline / "website_task_context.json", parent)
    assert record_confirmation_lineage(pipeline, parent=parent, successor=successor) is not None  # the entry
    write_json(pipeline / "website_task_context.json", successor)
    authority = _authority(successor)
    media = []

    def prepare_media(**kwargs):
        media.append(kwargs["task_context"]["context_digest"])
        return {"privacy_verified": True, "status": "noop", "source_geometry": None, "task_masks": {},
                "removal_manifest_path": str(job["removal"]), "privacy_status": "website_capture_admitted"}

    gate = SimpleNamespace(passed=True)
    metadata = {"site_task_context": successor, "website_scene_execution_authority": authority}
    outcome = run_website_capture_preparation(
        descriptor=SimpleNamespace(metadata=metadata, requested_outputs=[], to_dict=lambda: {
            "capture_id": "walkthrough-req1", "scene_id": "site-req1", "metadata": dict(metadata)}),
        capture_root=job["capture"], pipeline_dir=pipeline, source_video=None,
        derived_scene_generation_allowed=True, gates=[], prepare_media=prepare_media,
        check_privacy=lambda **_: gate, make_gate=lambda *_: gate)
    # The paid analysis was read under the sealed parent, so its retained calls are reused.
    assert media == [parent["context_digest"]]
    sent = job["sent"][-1]
    assert sent["consent"] == authority["consent"] and sent["consent"]["task_confirmed"] is True
    assert sent["task"]["task_id"] == "website-" + successor["context_digest"][7:27]
    assert sent["task"]["task_id"] != "website-" + parent["context_digest"][7:27]
    result = outcome["website_scene_preparation"]
    assert result["navigation_intake"]["status"] == "forward_pending"
    assert result["blockers"] == [HOLD_CODE] and holds_native_task_preparation(result, successor)


@pytest.mark.parametrize("confirmation", ["confirmation_only", "accepted_proposal"])
def test_the_library_shape_held_before_the_intake_hook_is_resumed_once_and_sends_the_intake(job, monkeypatch,
                                                                                            confirmation):
    """Completed and held before the intake hook existed (its runnable policy missing only the source
    identity, no intake record, no replay record kept), its brief confirmed on the website since: the
    operator's resume replays the held stage once, through the held-replay claim, on the original payload.
    It sends the intake with the confirmed consent and the successor's task id, and buys nothing again.
    The library job's confirmation also accepted Blueprint's proposed answer, marked no gate unknown and
    gave success criteria that state nothing; the website names the sealed parent for it."""
    import hashlib
    from types import SimpleNamespace

    from blueprint_pipeline import clean_plate_removal_analysis_gemini, clean_plate_stage, meta_sam31
    from blueprint_pipeline import pubsub_handoff_listener as listener
    from blueprint_pipeline import website_atlas_preview, website_candidate_ranking, website_gemini_receipts
    from blueprint_pipeline import website_held_handoff_replay as held
    from blueprint_pipeline import website_task_context as context_reader
    from blueprint_pipeline import website_task_masks
    from blueprint_pipeline.website_capture_preparation import run_website_capture_preparation
    from blueprint_pipeline.website_task_context_lineage import record_confirmation_lineage

    for module, name in ((website_gemini_receipts, "retained_gemini_call"), (website_candidate_ranking, "_gemini"),
                         (clean_plate_stage, "analyze_removal_targets"),
                         (clean_plate_removal_analysis_gemini, "analyze_removal_targets"),
                         (meta_sam31, "run_meta_sam31"), (website_task_masks, "run_meta_sam31"),
                         (website_atlas_preview, "retry_failed_job")):
        monkeypatch.setattr(module, name, lambda *_, **__: pytest.fail("nothing paid is bought again"))
    pipeline = job["capture"] / "pipeline"
    parent = _context(confirmed=False)
    write_json(pipeline / "website_task_context.json", parent)
    # Before the hook: the scene half is written and no intake is sent.
    intake = handoff._navigation_intake
    monkeypatch.setattr(handoff, "_navigation_intake", lambda **_: (False, None))
    first, _ = _run(job, context=parent, authority=_authority(parent))
    monkeypatch.setattr(handoff, "_navigation_intake", intake)
    assert first["blockers"] == [HOLD_CODE] and "navigation_intake" not in first and job["sent"] == []
    assert _missing(first) == {f"pipeline:{SOURCE_IDENTITY_PIECE}"}
    raw = b'{"original":"selected envelope"}'
    key = "sha256:" + "d" * 64
    ledger = {"schema_version": "v1", "status": "completed", "scene_id": "site-req1", "capture_id": "walkthrough-req1",
              "revision": 4, "attempt_count": 1, "source_payload_sha256": hashlib.sha256(raw).hexdigest(),
              "producer_delivery_key": key, "attempt_history": [{"attempt_number": 1, "status": "completed"}],
              "lease_owner": None, "lease_token": None, "lease_expires_at": None}
    write_json(job["capture"] / listener.JOB_LEDGER_FILENAME, ledger)
    assert held.FIELD not in ledger and held.held_kind(pipeline) == "integration_gap"
    # The site confirms its brief on the website; a release with the intake hook runs.
    successor = {key_: value for key_, value in parent.items() if key_ != "context_digest"}
    successor.update(confirmed=True, confirmed_at="2026-11-07T14:30:45Z")
    if confirmation == "accepted_proposal":
        successor.update(operator_answers={"taskShape": "single"}, unresolved=[],
                         success_criteria={"successDefinition": None, "successRate": None, "cycleTimeSeconds": None,
                                           "unknown": True},
                         task_evidence_provenance={"item_details": "owner_stated_unverified",
                                                   "success_criteria": "owner_stated_target"},
                         confirmation_parent_context_digest=parent["context_digest"])
    successor["context_digest"] = canonical_digest(successor, digest_field="context_digest")
    monkeypatch.setattr(context_reader, "load_current_website_task_context", lambda **_: dict(successor))
    monkeypatch.setattr("blueprint_pipeline.team_scene_packet_worker.current_release_commit", lambda: "a" * 40)
    media, outcomes = [], []

    def prepare_media(**kwargs):
        media.append(kwargs["task_context"]["context_digest"])
        return {"privacy_verified": True, "status": "noop", "source_geometry": None, "task_masks": {},
                "removal_manifest_path": str(job["removal"]), "privacy_status": "website_capture_admitted"}

    def process(payload, **kwargs):
        status, claimed = listener._claim_job_lease(
            job["capture"], scene_id="site-req1", capture_id="walkthrough-req1", owner="reviewed-recovery",
            lease_seconds=60, payload_sha256=hashlib.sha256(payload).hexdigest(), producer_delivery_key=key,
            expected_held_replay=kwargs["expected_held_replay"])
        assert status == "claimed" and claimed["attempt_count"] == 2
        # The orchestrator's entry: the website's current reading, its lineage kept; then the held stage.
        sealed = json.loads((pipeline / "website_task_context.json").read_text())
        current = context_reader.load_current_website_task_context(request_id="req1")
        assert record_confirmation_lineage(pipeline, parent=sealed, successor=current) is not None
        write_json(pipeline / "website_task_context.json", current)
        gate = SimpleNamespace(passed=True)
        metadata = {"site_task_context": current, "website_scene_execution_authority": _authority(current)}
        outcomes.append(run_website_capture_preparation(
            descriptor=SimpleNamespace(metadata=metadata, requested_outputs=[], to_dict=lambda: {
                "capture_id": "walkthrough-req1", "scene_id": "site-req1", "metadata": dict(metadata)}),
            capture_root=job["capture"], pipeline_dir=pipeline, source_video=None,
            derived_scene_generation_allowed=True, gates=[], prepare_media=prepare_media,
            check_privacy=lambda **_: gate, make_gate=lambda *_: gate))
        listener._finish_job_lease(job["capture"], owner="reviewed-recovery", token=claimed["lease_token"],
                                   update={"status": "completed"})
        return {"status": "processed", "queue_disposition": "terminal_success"}

    def resume(process):
        current = json.loads((job["capture"] / listener.JOB_LEDGER_FILENAME).read_text())
        return held.resume_held_preparation(
            SimpleNamespace(process_handoff_payload=process), capture_root=job["capture"],
            storage_root=job["capture"].parents[4], handoff_payload=raw, ledger_digest=canonical_digest(current),
            ledger_revision=current["revision"], provider="openai")

    assert resume(process)["status"] == "processed" and len(outcomes) == 1
    sent = job["sent"][-1]
    assert sent["consent"] == _authority(successor)["consent"] and sent["consent"]["task_confirmed"] is True
    assert sent["task"]["task_id"] == "website-" + successor["context_digest"][7:27]
    result = outcomes[0]["website_scene_preparation"]
    assert result["navigation_intake"]["status"] == "forward_pending"
    assert result["blockers"] == [HOLD_CODE] and holds_native_task_preparation(result, successor)
    # The paid analysis was read under the sealed parent, so its retained calls are reused.
    assert media == [parent["context_digest"]]
    # Once per fact: the same release never replays it again; it now waits for its intake receipt.
    assert held.held_kind(pipeline) == "intake_receipt"
    with pytest.raises(ValueError, match="website_held_handoff_resume_already_replayed"):
        resume(lambda *_, **__: pytest.fail("the same fact replayed twice"))
    assert len(job["sent"]) == 1
    # The record (and so the intake receipt staged from it) binds the confirmed successor's own digest.
    record = read_navigation_scene_inputs(Path(result["navigation_scene_inputs"]["path"]))
    assert record["source_gap"]["known"]["task_context_digest"] == successor["context_digest"]


def _republishing(job, monkeypatch, tmp_path) -> tuple[dict, dict]:
    """A job whose Atlas result can be published again, and the website's one intake per submission.

    The website double answers as its ``prepared-scene`` route does: one intake per owner and submission,
    a changed request under the same submission refused by name (``idempotency_conflict``).
    """
    import hashlib

    accepted = {}

    def website(*, capture_id, operation, payload, **_):
        request = payload["request"]
        job["sent"].append(request)
        identity = cross_runtime_canonical_digest({"owner": request["owner"], "submission_id": request["submission_id"]})
        digest = cross_runtime_canonical_digest(request)
        if accepted.setdefault(identity, digest) != digest:
            raise ValueError("website_control_prepared-scene_http_409:idempotency_conflict")
        return {"id": "scene-" + identity[7:], "request_digest": digest, "state": "forward_pending"}

    monkeypatch.setattr("blueprint_pipeline.website_task_context.website_webapp_request", website)
    (tmp_path / "scene").mkdir()
    collider, walked = _corridor(tmp_path / "scene")
    rebuilt_splat = tmp_path / "scene/rebuilt_splat.ply"
    rebuilt_splat.write_bytes(b"the rebuilt refined splat")
    scenes = {}
    for name, splat in (("parent", Path(collider["collision_mesh_path"])), ("rebuilt", rebuilt_splat)):
        splat_digest = "sha256:" + hashlib.sha256(splat.read_bytes()).hexdigest()
        scenes[name] = {**collider, "up_axis": "Y", "splat_path": str(splat), "splat_digest": splat_digest,
                        "splat_binding_id": "website-splat-" + splat_digest[7:39],
                        "atlas_result_digest": canonical_digest({"atlas_result": name})}
    current = {"base": scenes["parent"]}
    monkeypatch.setattr(ATLAS, lambda **_: {
        "status": "ready", "base_scene": dict(current["base"]), "alignment": "anchored",
        "camera_path_levelled_m": walked,
        "source_geometry": {"digest": "atlas-geometry", "geometry_backend": "world_labs_atlas"}})

    def assets(*, base_scene, output_root):
        # Authored from the base scene's own files, in a directory of their own (as the real authoring is).
        value = _scene_assets(Path(output_root) / base_scene["splat_digest"][7:23])
        value["source"] = {key: base_scene[key] for key in ("splat_digest", "collision_mesh_digest")}
        value["digest"] = canonical_digest(value, digest_field="digest")
        return value

    monkeypatch.setattr("blueprint_pipeline.website_navigation_scene_assets.navigation_scene_assets", assets)
    _run(job)
    _stage(job)
    handed, _ = _run(job)
    assert handed["status"] == "intake_ready"
    return scenes, current


def test_a_republished_atlas_result_derives_a_new_scene_half_and_reissues_its_intake_in_the_same_run(
        job, monkeypatch, tmp_path):
    """The library-aisle job after a splat rebuild: its handoff runs again on the new Atlas result.

    The new base scene names the new result; the scene half is written again, under its own key, and the
    team registry's row names it. The handed-off record, its outbox and its receipt stay as they were, so
    the old screening evidence still verifies, and the records the old base scene named are kept. The old
    receipt is never read as the new record's. The capture already holds an intake for the earlier record,
    so the new one's is that intake re-issued under its own submission: the website takes it in the same
    run, the Pipeline intake stages it as its own intent, and its receipt names the capture.
    """
    from blueprint_pipeline.website_preparation_revision_identity import REISSUE_FIELD, REISSUE_PREFIX, reissue
    from blueprint_pipeline.website_scene_handoff import SUPERSEDED_DIRECTORY

    scenes, current = _republishing(job, monkeypatch, tmp_path)
    handed_bytes = (job["capture"] / "pipeline/website_scene_preparation/handoff.json").read_bytes()
    old_path = Path(json.loads(handed_bytes)["navigation_scene_inputs"]["path"])
    old_record = read_navigation_scene_inputs(old_path)
    old_receipt = read_navigation_intake_receipt(old_path)
    kept = {path: path.read_bytes() for path in old_path.parent.rglob("*") if path.is_file()}
    assert {OUTBOX_FILENAME, RECEIPT_FILENAME} <= {path.name for path in kept}
    # The Atlas job publishes the rebuilt result and the preparation runs once more on it.
    current["base"] = scenes["rebuilt"]
    rerun, context = _run(job)
    base = json.loads((job["root"] / "base_scene.json").read_text())
    assert base["atlas_result_digest"] == scenes["rebuilt"]["atlas_result_digest"]
    new_path = Path(rerun["navigation_scene_inputs"]["path"])
    assert new_path.parent != old_path.parent and new_path.parent.parent == old_path.parent.parent
    new_record = read_navigation_scene_inputs(new_path)
    assert new_record["scene_revision_digest"] != old_record["scene_revision_digest"]
    assert profile_for_capture(job["capture"])["navigation_scene_inputs_path"] == str(new_path)
    # The handed-off record and everything beside it are unchanged, and still read.
    assert all(path.read_bytes() == data for path, data in kept.items())
    assert read_navigation_scene_inputs(old_path) == old_record
    assert read_navigation_intake_receipt(old_path) == old_receipt
    # The records the old base scene named are kept as they were, under its result.
    archive = (job["capture"] / "pipeline/website_scene_preparation" / SUPERSEDED_DIRECTORY
               / scenes["parent"]["atlas_result_digest"][7:23])
    assert (archive / "handoff.json").read_bytes() == handed_bytes
    assert json.loads((archive / "base_scene.json").read_text())["atlas_result_digest"] == (
        scenes["parent"]["atlas_result_digest"])
    # The intake went out re-issued, in this run: same capture, consent, owner and execution; its own
    # submission, its parent the capture's own, and the Atlas result the new base scene was built from.
    first, sent = job["sent"][0], job["sent"][-1]
    assert sent["submission_id"].startswith(REISSUE_PREFIX) and first["submission_id"] == context["capture_id"]
    assert reissue(sent) == {"capture_id": context["capture_id"], "parent_submission_id": context["capture_id"],
                             "atlas_result_digest": scenes["rebuilt"]["atlas_result_digest"]}
    assert {key: value for key, value in sent["task"].items() if key != REISSUE_FIELD} == {
        "task_id": new_record["task_id"], "strategy": "goal_region_navigation", "geometry_origin": "kept_as_filmed",
        **{key: new_record[key] for key in ("navigation_task_spec_digest", "compiled_task_spec_digest",
                                            "task_success_contract_digest", "runtime_contract_digest",
                                            "scene_revision_digest")}}
    assert {key: sent[key] for key in ("owner", "consent", "execution")} == {
        key: first[key] for key in ("owner", "consent", "execution")}
    assert sent["consent"]["task_confirmed"] is True and sent["source"]["content_digest"] == scenes["rebuilt"]["splat_digest"]
    assert rerun["navigation_intake"]["status"] == "forward_pending" and rerun["blockers"] != ["idempotency_conflict"]
    assert (new_path.parent / OUTBOX_FILENAME).is_file() and not (new_path.parent / RECEIPT_FILENAME).exists()
    with pytest.raises(WebsiteNavigationSceneInputsError, match=SOURCE_UNBOUND):
        screening_source(new_record)
    # The website forwards it and the Pipeline's own intake stages it as its own intent; the next pass
    # binds the new record's receipt, naming the capture, and the job is handed off on the new scene.
    staged = _stage(job)
    assert staged["intent_id"] != old_receipt["intent"]["intent_id"]
    handed, _ = _run(job)
    receipt = read_navigation_intake_receipt(new_path)
    assert receipt["source"] == {"scene_intent_id": staged["intent_id"], "scene_intent_digest": staged["intent_digest"],
                                 "capture_id": context["capture_id"], "task_id": new_record["task_id"],
                                 "task_context_digest": context["context_digest"],
                                 "scene_revision_digest": new_record["scene_revision_digest"]}
    assert screening_source(new_record) == receipt["source"] and receipt["reserved_usd"] == 0
    assert handed["status"] == "intake_ready" and handed["navigation_preparation"]["source"] == receipt["source"]
    # Two intakes in all, the original and the re-issue: nothing is sent again once the website holds one.
    assert [request["submission_id"] for request in job["sent"]] == [first["submission_id"], sent["submission_id"]]


def test_a_job_held_on_the_websites_conflict_from_a_re_derive_is_an_integration_gap_until_it_re_issues(
        job, monkeypatch, tmp_path):
    """A re-derive by a release before the re-issued intake held on the website's ``idempotency_conflict``.

    That is a Blueprint integration gap (``pipeline:navigation_intake_reissue``): each release re-evaluates
    it once, and this release's run sends the intake re-issued.
    """
    from blueprint_pipeline import website_held_handoff_replay as held
    from blueprint_pipeline import website_navigation_intake as intake

    scenes, current = _republishing(job, monkeypatch, tmp_path)
    pipeline = job["capture"] / "pipeline"
    write_json(pipeline / "website_task_context.json", _context())
    # The earlier release: the re-derived record's intake under the capture's own submission.
    reissue_parent = intake.reissue_parent
    monkeypatch.setattr(intake, "reissue_parent", lambda *_, **__: None)
    current["base"] = scenes["rebuilt"]
    stuck, context = _run(job)
    assert stuck["blockers"] == ["idempotency_conflict"] and holds_native_task_preparation(stuck, context)
    assert held.held_kind(pipeline) == "integration_gap"
    monkeypatch.setattr("blueprint_pipeline.team_scene_packet_worker.current_release_commit", lambda: "a" * 40)
    fact = held.arrived_fact(pipeline)
    assert fact["kind"] == "release" and fact["missing_pipeline"] == [held.REISSUE_MISSING]
    # Not every refusal by that name: a first record's own conflict names no re-derive.
    first = Path(stuck["navigation_scene_inputs"]["path"])
    assert not held._reissue_refused({**stuck, "navigation_scene_inputs": {
        **stuck["navigation_scene_inputs"], "path": str(tmp_path / "elsewhere/key" / first.name)}})
    # This release: the replayed run re-issues the intake and the job waits only on its receipt.
    monkeypatch.setattr(intake, "reissue_parent", reissue_parent)
    replayed, _ = _run(job)
    assert replayed["navigation_intake"]["status"] == "forward_pending"
    assert job["sent"][-1]["submission_id"].startswith("website-reissue-")
    assert held.held_kind(pipeline) == "intake_receipt"


