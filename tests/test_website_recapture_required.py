# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_recapture_required.py
#   src/blueprint_pipeline/website_scene_handoff.py
#   src/blueprint_pipeline/website_capture_preparation.py
#   src/blueprint_pipeline/website_preparation_status.py
#   src/blueprint_pipeline/website_preparation_outcome.py
"""A preparation stop only a new video can clear ends at once and names itself to the site (2026-11-11).

The object's facing (``website_front_normal_undetermined``) and its registration into the scene (the
five ``website_registration_*`` fit codes) used to be recorded as ``awaiting_inputs``, failed the
stage and were re-run six times on the same video; the site read nothing. Now the handoff keeps the
stop, the stage completes as a hold that nothing retries, and status v7 names the code as the outcome.
"""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_preparation_status as status_reader
from blueprint_pipeline import website_recapture_required as recapture
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_website_preparation_status import prepared_source  # noqa: F401
from tests.test_website_task_preparation import NOW, SPEND, _arguments

CODES = sorted(recapture.CODES)


def _handoff(tmp_path, monkeypatch, code):
    """The real handoff over a ready Atlas scene, with the object's registration or facing refused by ``code``."""
    from blueprint_pipeline import website_task_preparation as preparation
    from blueprint_pipeline.website_scene_handoff import prepare_website_scene_handoff
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    args = _arguments(pipeline)
    base = {**args["base_scene"], "provider": "world_labs_atlas", "scale_authority": "atlas_reference_estimate"}
    geometry = {**args["source_geometry"], "geometry_backend": "world_labs_atlas"}
    geometry["digest"] = canonical_digest(geometry, digest_field="digest")
    masks = {**args["task_masks"], "source_geometry_digest": geometry["digest"]}
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    removal_path = pipeline / "removal.json"
    write_json(removal_path, args["removal_manifest"])
    monkeypatch.setattr("blueprint_pipeline.website_atlas_scene_source.atlas_scene_source", lambda **_: {
        "status": "ready", "base_scene": base, "source_geometry": geometry, "alignment": "anchored",
        "source_pose_digest": "sha256:" + "1" * 64, "background_pose_digest": "sha256:" + "2" * 64})
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry", lambda **_: masks)

    def refuse(**_):
        if code == recapture.FRONT_NORMAL_CODE:
            raise preparation._front_normal_undetermined([0.0, 0.0, 1.0])
        raise ValueError(code)
    monkeypatch.setattr(preparation, "register_source_to_runtime", refuse)
    # Nothing may be bought or reserved on the way to the stop.
    from blueprint_pipeline import website_task_context
    monkeypatch.setattr(website_task_context, "reserve_website_preparation_spend",
                        lambda **_: pytest.fail("a recapture stop reserved spend"))
    context = args["task_context"]
    result = prepare_website_scene_handoff(
        descriptor={"capture_id": context["capture_id"], "scene_id": context["scene_id"], "metadata": {
            "site_task_context": context, "website_scene_execution_authority": SPEND}},
        clean_plate={"status": "objects_removed", "privacy_verified": True, "source_geometry": None,
                     "task_masks": args["task_masks"], "removal_manifest_path": str(removal_path)},
        capture_root=tmp_path, now=NOW)
    return result, context


@pytest.mark.parametrize("code", CODES)
def test_each_stop_only_a_new_video_clears_is_kept_as_a_hold_on_the_handoff(tmp_path, monkeypatch, code):
    result, context = _handoff(tmp_path, monkeypatch, code)
    assert result["status"] == "needs_input" and result["blockers"] == [code]
    assert result[recapture.FIELD] == {"schema_version": recapture.SCHEMA, "code": code,
                                       "task_context_digest": context["context_digest"], "next_actor": "site",
                                       "provider_mutation_performed": False}
    assert recapture.retains_recapture_required(result, context)
    assert json.loads((tmp_path / "pipeline/website_scene_preparation/handoff.json").read_text()) == result


def test_any_other_preparation_refusal_still_waits_as_before(tmp_path, monkeypatch):
    result, context = _handoff(tmp_path, monkeypatch, "website_registration_anchor_frame_invalid")
    assert result["status"] == "awaiting_inputs" and recapture.FIELD not in result
    assert not recapture.retains_recapture_required(result, context)


def _complete_recapture(source, code):
    """A preparation that ended on ``code``, as the stage leaves it: the handoff and a completed output commit."""
    root, owner, context, selected, ledger, _ = source
    handoff = {"schema_version": "website_scene_handoff.v1", "status": "needs_input",
               **{key: owner[key] for key in ("scene_id", "capture_id")}, "blockers": [code],
               recapture.FIELD: recapture.record(code, task_context_digest=context["context_digest"]),
               "simulator_ready": False, "visual_reconstruction_ready": True, "provider_mutation_performed": False}
    handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    write_json(root / "pipeline/website_scene_preparation/handoff.json", handoff)
    listener._write_output_commit(root, scene_id=owner["scene_id"], capture_id=owner["capture_id"],
        attempt_count=ledger["attempt_count"], result={"status": "completed", "native_execution_complete": False,
                                                     "website_scene_preparation": handoff},
        owner="synthetic-worker", token=ledger["lease_token"], ledger_update={"status": "completed"})
    return handoff


@pytest.mark.parametrize("code", CODES)
def test_status_v7_names_the_stop_as_the_outcome_and_older_readers_keep_their_vocabulary(prepared_source, code):  # noqa: F811
    from blueprint_pipeline import website_preparation_retry as retry
    root, _, _, selected, _, _ = prepared_source
    handoff = _complete_recapture(prepared_source, code)
    v7 = status_reader.read_preparation_status(capture_root=root, selectors=selected,
                                              schema_version=status_reader.STATUS_V7)
    assert v7["state"] == "awaiting_inputs" and v7["terminal_reason"] is None
    assert v7["outcome"] == {"category": "insufficient_evidence", "reason": code, "basis": "missing_evidence",
                             "robot_id": None, "evidence_digest": handoff["digest"]}
    v6 = status_reader.read_preparation_status(capture_root=root, selectors=selected,
                                              schema_version=status_reader.STATUS_V6)
    assert v6["state"] == "awaiting_inputs" and v6["outcome"] is None
    # The stage completed: nothing redelivers it and the host's retry never arms for it.
    ledger = listener._read_job_ledger(root)
    assert ledger["status"] == "completed" and retry.armed(ledger) is None
    acceptance = {"schema_version": "website_preparation_status_acceptance.v7", **selected, "accepted": True,
                  "state": v7["state"], "status_digest": v7["status_digest"], "native_execution_complete": False,
                  "correlation_id": "bp-prep-" + v7["status_digest"][7:23]}
    assert status_reader.accepted_status_version(status_reader.validate_acceptance(acceptance, selected)) \
        == status_reader.STATUS_V7
    assert status_reader.requested_status_version("7") == status_reader.STATUS_V7
    assert status_reader.requested_status_version("9") == status_reader.STATUS_V7


def test_a_stop_bound_to_another_task_context_is_not_a_hold(prepared_source):  # noqa: F811
    root, _, context, selected, _, _ = prepared_source
    handoff = _complete_recapture(prepared_source, "website_registration_poor_fit")
    assert not recapture.retains_recapture_required(handoff, {**context, "context_digest": "sha256:" + "9" * 64})
    tampered = {**handoff, "blockers": ["website_registration_poor_fit", "other"]}
    tampered["digest"] = canonical_digest(tampered, digest_field="digest")
    assert not recapture.retains_recapture_required(tampered, context)


@pytest.mark.parametrize("held", [True, False])
def test_the_stage_completes_on_the_hold_so_pubsub_acknowledges_it_and_nothing_retries(tmp_path, monkeypatch, held):
    from types import SimpleNamespace

    from blueprint_pipeline import website_scene_handoff
    from blueprint_pipeline.common import StageError
    from blueprint_pipeline.website_capture_preparation import run_website_capture_preparation
    from tests.test_qualification_coverage_edges import _descriptor
    capture = tmp_path / "capture"
    (capture / "pipeline").mkdir(parents=True)
    descriptor = _descriptor(metadata={"site_task_context": {"purpose": "scene_preparation"}})
    context = {"purpose": "scene_preparation", "scene_id": descriptor.scene_id, "capture_id": descriptor.capture_id,
               "context_digest": "sha256:" + "4" * 64}
    descriptor.metadata["site_task_context"] = context
    code = "website_registration_ambiguous"
    handoff = {"schema_version": "website_scene_handoff.v1", "status": "needs_input",
               "scene_id": descriptor.scene_id, "capture_id": descriptor.capture_id, "blockers": [code],
               **({recapture.FIELD: recapture.record(code, task_context_digest=context["context_digest"])}
                  if held else {})}
    handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    monkeypatch.setattr(website_scene_handoff, "prepare_website_scene_handoff", lambda **_: handoff)

    def run():
        return run_website_capture_preparation(
            descriptor=descriptor, capture_root=capture, pipeline_dir=capture / "pipeline", source_video=None,
            derived_scene_generation_allowed=True, gates=[],
            prepare_media=lambda **_: {"status": "noop", "mode": "task_aware_prepared_images",
                                       "privacy_status": "website_capture_admitted", "privacy_verified": True},
            check_privacy=lambda **_: SimpleNamespace(passed=True),
            make_gate=lambda _name, passed, _detail: SimpleNamespace(passed=passed))
    if not held:
        # Before: the same stop failed the stage, which the host then re-ran six times on the same video.
        with pytest.raises(StageError, match=code):
            run()
        return
    result = run()
    assert result["status"] == "completed" and result["website_scene_preparation"] == handoff
    assert listener._handoff_result_disposition(result) == ("terminal_success", [])
