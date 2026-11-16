"""The astra Claude path names Anthropic on the site's withdrawal list before its rights admission is written."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

from blueprint_pipeline.common import write_json

import pytest

from blueprint_pipeline import website_capture_withdrawal_provider_copies as copies
from blueprint_pipeline import website_scene_handoff as handoff


def _capture(tmp_path, request_id="req1"):
    capture = tmp_path / f"site-{request_id}" / "captures" / f"walkthrough-{request_id}"
    capture.mkdir(parents=True)
    return capture


def test_a_rights_admission_naming_anthropic_names_it_in_the_sites_ledger(tmp_path):
    capture = _capture(tmp_path)
    site = capture.parent.parent
    assert copies.plan_providers(site)["not_deleted_at_provider"] == [dict(r) for r in copies.NOT_DELETED_AT_PROVIDER]
    assert handoff.name_anthropic_for_site({"anthropic_provider_terms_reference": "sha256:" + "a" * 64}, capture)
    plan = copies.plan_providers(site)
    assert plan["not_deleted_at_provider"][-1] == copies.ANTHROPIC_NOT_DELETED
    assert "inline" in copies.ANTHROPIC_NOT_DELETED["reason"] and "no stored object" in copies.ANTHROPIC_NOT_DELETED["reason"]
    # Recorded once; a second admission changes nothing.
    before = (site / "website_withdrawal" / copies.LEDGER_FILENAME).read_bytes()
    handoff.name_anthropic_for_site({"anthropic_provider_terms_reference": "sha256:" + "a" * 64}, capture)
    assert (site / "website_withdrawal" / copies.LEDGER_FILENAME).read_bytes() == before


def test_an_openai_rights_admission_writes_nothing(tmp_path):
    capture = _capture(tmp_path)
    assert handoff.name_anthropic_for_site({"status": "admitted_for_internal_development"}, capture) is False
    assert not (capture.parent.parent / "website_withdrawal").exists()


def test_without_the_capture_root_an_anthropic_admission_is_refused():
    with pytest.raises(ValueError, match="website_anthropic_admission_site_unknown"):
        handoff.name_anthropic_for_site({"anthropic_provider_terms_reference": "sha256:" + "a" * 64}, None)


def test_only_anthropic_can_be_recorded(tmp_path):
    with pytest.raises(ValueError, match="website_withdrawal_provider_admission_unknown"):
        copies.record_provider_admission(tmp_path, "google")


def test_the_ledger_is_written_before_the_rights_admission_file():
    source = inspect.getsource(handoff.prepare_rigid_native_inputs)
    assert source.index("name_anthropic_for_site(rights, capture_root)") < source.index("write_json(rights_path, rights)")


def _admission(capture, value):
    path = capture / "pipeline" / "website_scene_preparation" / "native" / "rights_admission.json"
    path.parent.mkdir(parents=True)
    path.write_text(value if isinstance(value, str) else __import__("json").dumps(value))
    return path


def test_a_site_with_no_anthropic_record_is_planned_byte_for_byte_as_today(tmp_path):
    # No ledger record: no lock, no new folder, no file written, and no scan of rights admissions.
    capture = _capture(tmp_path)
    _admission(capture, {"anthropic_provider_terms_reference": "sha256:" + "a" * 64})
    site = capture.parent.parent

    def tree():
        return {str(path.relative_to(site)): path.read_bytes() if path.is_file() else None
                for path in sorted(site.rglob("*"))}
    before = tree()
    plan = copies.plan_providers(site)
    assert tree() == before and not (site / "website_withdrawal").exists()
    assert plan["not_deleted_at_provider"] == [dict(row) for row in copies.NOT_DELETED_AT_PROVIDER]
    assert plan["copies"] == plan["unknown"] == []


# -- the real handoff, renewal and submission paths (review F1-F4) --------------------------------------------

def _real_handoff(tmp_path, monkeypatch, *, record=None):
    """One real handoff to ``intake_ready`` whose rights admission lets frames go to Anthropic."""
    from blueprint_pipeline import website_native_background as background
    from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
    from blueprint_pipeline.gaussian_splat_decode import write_standard_3dgs_ply
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    from blueprint_pipeline.website_scene_handoff import prepare_website_scene_handoff
    from tests.test_particlefield_usd import _splat
    from tests.test_website_preparation_renewal import ATLAS, NOW, _grant, _seal
    from tests.test_website_task_preparation import _arguments

    capture = _capture(tmp_path)
    pipeline = capture / "pipeline"
    pipeline.mkdir()
    args = _arguments(pipeline)
    data, _ = _splat(32)
    data.scales[:] = -5
    splat = Path(args["base_scene"]["splat_path"])
    write_standard_3dgs_ply(data, splat)
    digest = _sha256_file(splat)
    base = {**args["base_scene"], "splat_digest": digest, "splat_binding_id": "website-splat-" + digest[7:39],
            "provider": "world_labs_atlas"}
    context = args["task_context"]
    context["capture_rights"] = {"derived_scene_generation_allowed": True, "consent_revoked": False}
    _seal(context, "context_digest")
    grant = _grant(context, accepted=NOW - 100, parent="sha256:" + "a" * 64)
    geometry = _seal({**args["source_geometry"], "geometry_backend": "world_labs_atlas"}, "digest")
    masks = _seal({**args["task_masks"], "source_geometry_digest": geometry["digest"]}, "digest")
    atlas = {"status": "ready", "base_scene": base, "source_geometry": geometry, "alignment": "anchored",
             "source_pose_digest": "sha256:" + "1" * 64, "background_pose_digest": "sha256:" + "2" * 64}
    removal = pipeline / "removal.json"
    write_json(removal, args["removal_manifest"])
    original = background.construction_rights_admission
    seen = {}

    def anthropic_rights(**kwargs):
        value = dict(original(**kwargs), anthropic_provider_terms_reference="sha256:" + "a" * 64)
        seen["ledger_before_rights"] = (capture.parent.parent / "website_withdrawal" / copies.LEDGER_FILENAME).exists()
        return value
    monkeypatch.setattr(background, "construction_rights_admission", anthropic_rights)
    monkeypatch.delenv("BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT", raising=False)
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", str(tmp_path / "intents"))
    monkeypatch.setattr(ATLAS, lambda **_: atlas)
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry", lambda **_: masks)
    monkeypatch.setattr("blueprint_pipeline.website_task_context.enqueue_website_prepared_scene", lambda **kw: {
        "id": "scene-outbox", "state": "forward_pending", "request_digest": cross_runtime_canonical_digest(kw["request"])})
    if record is not None:
        monkeypatch.setattr(copies, "record_provider_admission", record)
    result = prepare_website_scene_handoff(descriptor={"capture_id": context["capture_id"],
        "scene_id": context["scene_id"], "metadata": {"site_task_context": context,
                                                      "website_scene_execution_authority": grant}},
        clean_plate={"privacy_verified": True, "status": "objects_removed", "task_masks": args["task_masks"],
                     "source_geometry": None, "removal_manifest_path": str(removal)},
        capture_root=capture, now=NOW)
    return result, capture, seen


def test_f2_the_real_handoff_names_anthropic_before_the_rights_file_exists(tmp_path, monkeypatch):
    rights_seen = []
    original = copies.record_provider_admission

    def record(site, provider):
        rights_seen.append(list(site.rglob("rights_admission.json")))
        return original(site, provider)
    result, capture, seen = _real_handoff(tmp_path, monkeypatch, record=record)
    assert result["status"] == "intake_ready" and result["website_intake_outbox"]["id"] == "scene-outbox"
    assert rights_seen == [[]]  # the ledger was written while no rights file existed in the site
    plan = copies.plan_providers(capture.parent.parent)
    assert plan["not_deleted_at_provider"][-1] == copies.ANTHROPIC_NOT_DELETED


def test_f2_a_failed_ledger_write_holds_the_job(tmp_path, monkeypatch):
    def failing(site, provider):
        raise OSError("disk full")
    result, capture, _ = _real_handoff(tmp_path, monkeypatch, record=failing)
    assert "website_intake_outbox" not in result and "source_registration" not in result
    assert not list(capture.rglob("rights_admission.json"))


def test_f2_the_renewal_passes_the_capture_root_on_both_calls():
    import inspect

    from blueprint_pipeline import website_preparation_renewal as renewal

    # Source checks: a behaviour test of a renewal needs the full renewal fixture (the suite's ``template``).
    assert "_register_and_enqueue(capture_root=capture_root," in inspect.getsource(renewal.reissue_renewed_preparation)
    assert "capture_root=capture_root)" in inspect.getsource(renewal._register_and_enqueue)


def test_f1_a_concurrent_plan_cannot_drop_the_anthropic_record(tmp_path):
    import threading

    from blueprint_pipeline import website_capture_withdrawal as withdrawal

    capture = _capture(tmp_path)
    site = capture.parent.parent
    write_json(site / "website_withdrawal" / copies.LEDGER_FILENAME,
               {"schema_version": copies.LEDGER_SCHEMA, "copies": {}})  # the ledger exists, as during a withdrawal
    started, finished = threading.Event(), []

    def record():
        started.set()
        copies.record_provider_admission(site, "anthropic")
        finished.append(True)
    with withdrawal._lock(site):  # a planner holding the lock while it reads and writes the ledger
        worker = threading.Thread(target=record)
        worker.start()
        started.wait()
        worker.join(0.3)
        assert not finished  # the recorder waits for the planner's lock
        ledger = json.loads((site / "website_withdrawal" / copies.LEDGER_FILENAME).read_text())
        write_json(site / "website_withdrawal" / copies.LEDGER_FILENAME, ledger)  # the planner saves its copy
    worker.join(5)
    assert finished and copies.plan_providers(site)["not_deleted_at_provider"][-1] == copies.ANTHROPIC_NOT_DELETED


def test_f3_the_journal_is_private(tmp_path):
    import os
    import stat

    fresh = _capture(tmp_path / "fresh")
    copies.record_provider_admission(fresh.parent.parent, "anthropic")
    assert stat.S_IMODE(os.stat(fresh.parent.parent / "website_withdrawal").st_mode) == 0o700


def test_f4_submission_names_anthropic_and_a_missing_site_holds_the_launch(tmp_path, monkeypatch):
    from blueprint_pipeline import website_native_submission as submission

    capture = _capture(tmp_path)
    preparation = capture / "pipeline" / "website_scene_preparation" / "preparation.json"
    preparation.parent.mkdir(parents=True)
    preparation.write_text("{}")
    rights = {"anthropic_provider_terms_reference": "sha256:" + "a" * 64}
    monkeypatch.setattr("blueprint_pipeline.task_evaluation_scene_owner_authority.reopen_scene_intent",
                        lambda authority, now=None: {"request": {}, "accepted_at_epoch": 1.0})
    monkeypatch.setattr(submission, "checked_file", lambda path, record: Path(path))
    monkeypatch.setattr(submission, "read", lambda path, digest_field=None: {"intake_request": {}})
    monkeypatch.setattr(submission, "construction_rights_admission", lambda **_: rights)
    monkeypatch.setattr(submission, "prepare_construction_stages", lambda **_: {"references": []})
    task = {"scene_intent_authority": {}, **{key: {"path": str(preparation)} for key in
                                             ("preparation", "runtime_inputs", "task_context")}}
    submission.verified_submission_inputs.__wrapped__(task) if hasattr(
        submission.verified_submission_inputs, "__wrapped__") else submission.verified_submission_inputs(task)
    assert copies.plan_providers(capture.parent.parent)["not_deleted_at_provider"][-1] == copies.ANTHROPIC_NOT_DELETED
    elsewhere = tmp_path / "loose" / "preparation.json"
    elsewhere.parent.mkdir()
    elsewhere.write_text("{}")
    task = {"scene_intent_authority": {}, **{key: {"path": str(elsewhere)} for key in
                                             ("preparation", "runtime_inputs", "task_context")}}
    with pytest.raises(ValueError, match="website_anthropic_admission_site_unknown"):
        submission.verified_submission_inputs.__wrapped__(task) if hasattr(
            submission.verified_submission_inputs, "__wrapped__") else submission.verified_submission_inputs(task)


# -- round 3 (R1, R2) ----------------------------------------------------------------------------------------

def test_r1_an_anthropic_entry_recorded_during_a_provider_delete_survives(tmp_path):
    capture = _capture(tmp_path)
    site = capture.parent.parent
    write_json(site / "website_withdrawal" / copies.LEDGER_FILENAME, {
        "schema_version": copies.LEDGER_SCHEMA, "copies": {
            "gemini:files/x": {"provider": "gemini", "kind": "file", "id": "files/x", "status": "pending"}}})

    def gemini(name):  # a network call: no lock is held, and Anthropic is admitted meanwhile
        copies.record_provider_admission(site, "anthropic")
        return {"deleted": True}
    section = {"ledger": str(site / "website_withdrawal" / copies.LEDGER_FILENAME)}
    [row] = copies.apply_providers(section, {"gemini": gemini})
    assert row["status"] == "deleted_at_provider"
    ledger = json.loads((site / "website_withdrawal" / copies.LEDGER_FILENAME).read_text())
    assert ledger["providers_used"] == ["anthropic"] and ledger["copies"]["gemini:files/x"]["status"] == "deleted_at_provider"


_COPY = {"gemini:files/x": {"provider": "gemini", "kind": "file", "id": "files/x", "status": "pending"}}


def test_r1_a_planner_that_must_save_waits_for_the_withdrawal_lock(tmp_path, monkeypatch):
    import threading

    from blueprint_pipeline import website_capture_withdrawal as withdrawal

    capture = _capture(tmp_path)
    site = capture.parent.parent
    monkeypatch.setattr(copies, "_discover", lambda _site: ({key: dict(row) for key, row in _COPY.items()}, []))
    finished = []
    with withdrawal._lock(site):
        worker = threading.Thread(target=lambda: finished.append(copies.plan_providers(site)))
        worker.start()
        worker.join(0.3)
        assert not finished  # it has a new copy to save, and saves only under the lock
    worker.join(5)
    assert finished and [row["key"] for row in finished[0]["copies"]] == ["gemini:files/x"]


def test_r1_a_planner_that_read_before_a_first_admission_never_saves_over_it(tmp_path, monkeypatch):
    # Deterministic: the planner reads (no ledger, no lock yet), the site's first Anthropic admission is written,
    # then the planner finishes and saves its new copy. The admission's entry is still there.
    capture = _capture(tmp_path)
    site = capture.parent.parent
    calls = []

    def discover(_site):
        if not calls:
            copies.record_provider_admission(site, "anthropic")
        calls.append(1)
        return {key: dict(row) for key, row in _COPY.items()}, []
    monkeypatch.setattr(copies, "_discover", discover)
    plan = copies.plan_providers(site)
    ledger = json.loads((site / "website_withdrawal" / copies.LEDGER_FILENAME).read_text())
    assert ledger["providers_used"] == ["anthropic"] and list(ledger["copies"]) == ["gemini:files/x"]
    assert plan["not_deleted_at_provider"][-1] == copies.ANTHROPIC_NOT_DELETED


def test_r1_an_owner_resolution_is_saved_under_the_lock(tmp_path, monkeypatch):
    from blueprint_pipeline import website_capture_withdrawal as withdrawal

    capture = _capture(tmp_path)
    site = capture.parent.parent
    write_json(site / "website_withdrawal" / copies.LEDGER_FILENAME, {
        "schema_version": copies.LEDGER_SCHEMA, "copies": {"unknown:meta:x": {"provider": "meta", "id": None,
                                                                             "status": copies.ID_NOT_RECORDED}}})
    held = []
    original = withdrawal._lock

    def watched(path):
        held.append(path)
        return original(path)
    monkeypatch.setattr(withdrawal, "_lock", watched)
    copies.resolve_unknown_copy(site, "unknown:meta:x", answer="absent", by="owner")
    assert held == [site]


@pytest.mark.parametrize("layout", ["website_scene_preparation/development_test",
                                    "website_scene_preparation/revisions/" + "b" * 64 + "/development_test"])
def test_r2_a_development_test_preparation_resolves_to_its_site(tmp_path, layout):
    from blueprint_pipeline.website_native_submission import _launch_capture_root

    capture = _capture(tmp_path)
    assert _launch_capture_root(capture / "pipeline" / layout / "preparation.json") == capture
    assert _launch_capture_root(capture / "pipeline" / "website_scene_preparation" / "preparation.json") == capture
    assert _launch_capture_root(tmp_path / "loose" / "development_test" / "preparation.json") is None



@pytest.mark.parametrize("layout", ["website_scene_preparation/development_test",
                                    "website_scene_preparation/revisions/" + "c" * 64 + "/development_test"])
def test_t1_submission_names_anthropic_for_a_development_test_preparation(tmp_path, monkeypatch, layout):
    """Through the submission itself: bypassing ``_launch_capture_root`` fails this test."""
    from blueprint_pipeline import website_native_submission as submission

    capture = _capture(tmp_path)
    preparation = capture / "pipeline" / layout / "preparation.json"
    preparation.parent.mkdir(parents=True)
    preparation.write_text("{}")
    rights = {"anthropic_provider_terms_reference": "sha256:" + "a" * 64}
    monkeypatch.setattr("blueprint_pipeline.task_evaluation_scene_owner_authority.reopen_scene_intent",
                        lambda authority, now=None: {"request": {}, "accepted_at_epoch": 1.0})
    monkeypatch.setattr(submission, "checked_file", lambda path, record: Path(path))
    monkeypatch.setattr(submission, "read", lambda path, digest_field=None: {"intake_request": {}})
    monkeypatch.setattr(submission, "construction_rights_admission", lambda **_: rights)
    monkeypatch.setattr(submission, "prepare_construction_stages", lambda **_: {"references": []})

    def run(path):
        task = {"scene_intent_authority": {}, **{key: {"path": str(path)} for key in
                                                 ("preparation", "runtime_inputs", "task_context")}}
        verified = submission.verified_submission_inputs
        return getattr(verified, "__wrapped__", verified)(task)
    run(preparation)
    assert copies.plan_providers(capture.parent.parent)["not_deleted_at_provider"][-1] == copies.ANTHROPIC_NOT_DELETED
    unknown = tmp_path / "loose" / "development_test" / "preparation.json"
    unknown.parent.mkdir(parents=True)
    unknown.write_text("{}")
    with pytest.raises(ValueError, match="website_anthropic_admission_site_unknown"):
        run(unknown)
