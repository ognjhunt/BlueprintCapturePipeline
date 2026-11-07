"""A held job's preparation runs again, once, when its brief is confirmed, its intake receipt is bound or,
on a Blueprint integration gap, a new release runs.

The actual listener, lease, ledger, capture preparation stage, status reader and drain run. ``run_e2e``
stands in for the orchestrator around the website preparation stage (its entry keeps the confirmation
lineage, as ``prepare_website_task_descriptor`` does), and the scene handoff is cut down to the
navigation records it writes: the website's refusal of an unconfirmed brief, the hold on the missing
source identity, and the handoff to the G1 screening lane. Synthetic records only; no provider call.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_held_handoff_replay.py
#   src/blueprint_pipeline/website_task_context_lineage.py
#   src/blueprint_pipeline/pubsub_handoff_scene_operations.py
#   src/blueprint_pipeline/pubsub_handoff_listener.py
#   src/blueprint_pipeline/run_e2e.py
#   src/blueprint_pipeline/website_capture_preparation.py
#   src/blueprint_pipeline/website_preparation_status.py

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_held_handoff_replay as replay
from blueprint_pipeline import website_task_context as context_reader
from blueprint_pipeline.common import PipelineError, StageError, write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_preparation_status import (
    DELIVERY_FILE, STATUS_V6, read_preparation_status, retain_preparation_wakeup,
)
from tests.test_website_preparation_status import prepared_source  # noqa: F401
from tests.test_website_robot_fit_hold import _delivery

CONFIRMED_AT = "2026-11-07T14:30:45Z"
G1_CANDIDATE = "humanoidarena_pi05_g1_dex3_sonic_vision_navi"


def _seal(value: dict) -> dict:
    value = {key: item for key, item in value.items() if key != "context_digest"}
    value["context_digest"] = canonical_digest(value, digest_field="context_digest")
    return value


@pytest.fixture
def site(prepared_source, monkeypatch, tmp_path):  # noqa: F811 - imported pytest fixture
    """A navigation job prepared while its brief was unconfirmed, the website refusing its intake."""
    from blueprint_pipeline import website_scene_handoff
    from blueprint_pipeline.task_evaluation_g1_catalog import G1_EMBODIMENT_ID
    from blueprint_pipeline.task_spec_compiler import requirements
    from blueprint_pipeline.website_capture_preparation import run_website_capture_preparation
    from blueprint_pipeline.website_locomotion_hold import (
        navigation_intake_refusal_hold, navigation_preparation, runnable_policy_gap_from_options,
    )
    from blueprint_pipeline.website_task_context_lineage import record_confirmation_lineage

    root, owner, context, _, _, _ = prepared_source
    monkeypatch.delenv("BLUEPRINT_TEAM_CANDIDATE_QUICK10_REGISTRY", raising=False)
    raw = _delivery(root, owner, monkeypatch)
    parent = _seal({**context, "task_family": "navigation", "confirmed": False, "confirmed_at": None,
                    "purpose": "scene_preparation"})
    website = {"current": parent}
    monkeypatch.setattr(context_reader, "load_current_website_task_context", lambda **_: deepcopy(website["current"]))
    write_json(root / "pipeline/website_task_context.json", parent)
    work = root / "pipeline" / "website_scene_preparation"
    need = requirements(locomote=True, manipulate=False)
    spec = "sha256:" + "e" * 64
    record_path = work / "navigation_scene_inputs" / "key" / "website_navigation_scene_inputs.v1.json"
    # ``gap``: this release cannot send the intake yet (the hook is missing), so the job holds on its
    # integration gap alone; ``release``: the running release's commit.
    state = {"refuse_again": False, "receipt": None, "media": [], "gap": False, "release": None}
    monkeypatch.setattr("blueprint_pipeline.team_scene_packet_worker.current_release_commit",
                        lambda: state["release"])

    def options(runnable: bool) -> dict:
        return {"requirements": need, "embodiments": [], "navigation_pieces_built": [],
                "policies": [{"candidate_id": G1_CANDIDATE, "embodiment_id": G1_EMBODIMENT_ID, "runnable": runnable,
                              "missing": [] if runnable else [replay.SOURCE_IDENTITY_MISSING]}]}

    def write_record(context_now: dict) -> dict:
        record = {"scene_id": context_now["scene_id"], "task_id": "website-" + context_now["context_digest"][7:27],
                  "scene_revision_digest": "sha256:" + "c" * 64}
        record["scene_inputs_digest"] = canonical_digest(record, digest_field="scene_inputs_digest")
        write_json(record_path, record)
        return record

    def handoff(*, descriptor, **_):
        current = descriptor["metadata"]["site_task_context"]
        result = {"schema_version": "website_scene_handoff.v1", "claim_ceiling": "development_only",
                  "simulator_ready": False, "visual_reconstruction_ready": True,
                  "capture_id": owner["capture_id"], "scene_id": owner["scene_id"], "compiled_task_spec_digest": spec}
        if state["gap"]:
            # The scene half is written but no intake is sent: the hold names the missing piece only.
            record = write_record(current)
            gap = runnable_policy_gap_from_options(task_context=current, requirements=need, spec_digest=spec,
                                                   options=options(False))
            result.update(status="needs_input", blockers=[gap["code"]], native_task_preparation_gap=gap,
                          navigation_scene_inputs={"status": "written", "path": str(record_path),
                                                   "digest": record["scene_inputs_digest"],
                                                   "scene_revision_digest": record["scene_revision_digest"]})
        elif current["confirmed"] is not True or state["refuse_again"]:
            gap = navigation_intake_refusal_hold(task_context=current, requirements=need, spec_digest=spec,
                refusal="website_task_context_not_confirmed", options=options(False))
            result.update(status="needs_input", blockers=[gap["code"]], native_task_preparation_gap=gap,
                          navigation_intake={"status": "refused", "reason": gap["code"]})
        else:
            record = write_record(current)
            inputs = {"status": "written", "path": str(record_path), "digest": record["scene_inputs_digest"],
                      "scene_revision_digest": record["scene_revision_digest"]}
            if state["receipt"] is None:
                gap = runnable_policy_gap_from_options(task_context=current, requirements=need, spec_digest=spec,
                                                       options=options(False))
                result.update(status="needs_input", blockers=[gap["code"]], native_task_preparation_gap=gap,
                              navigation_scene_inputs=inputs,
                              navigation_intake={"status": "forward_pending", "receipt": None})
            else:
                receipt = {key: state["receipt"][key] for key in ("receipt_digest", "source")}
                result.update(status="intake_ready", blockers=[], job_disposition="accepted",
                              navigation_scene_inputs=inputs,
                              navigation_intake={"status": "forward_pending", "receipt": receipt},
                              navigation_preparation=navigation_preparation(
                                  task_context=current, spec_digest=spec, options=options(True),
                                  scene_inputs=inputs, receipt=receipt))
        result["digest"] = canonical_digest(result, digest_field="digest")
        write_json(work / "handoff.json", result)
        return result

    monkeypatch.setattr(website_scene_handoff, "prepare_website_scene_handoff", handoff)
    runs = []

    def run_e2e(**kwargs):
        runs.append(kwargs)
        # The entry (prepare_website_task_descriptor): the website's current reading, its lineage kept.
        sealed = json.loads((root / "pipeline/website_task_context.json").read_text())
        current = deepcopy(website["current"])
        record_confirmation_lineage(root / "pipeline", parent=sealed, successor=current)
        write_json(root / "pipeline/website_task_context.json", current)
        gate = SimpleNamespace(passed=True)

        def media(**kwargs):
            state["media"].append(kwargs["task_context"]["context_digest"])
            return {"status": "noop", "privacy_verified": True}
        try:
            run_website_capture_preparation(
                descriptor=SimpleNamespace(metadata={"site_task_context": current}, requested_outputs=[],
                                           to_dict=lambda: {"capture_id": owner["capture_id"],
                                                            "scene_id": owner["scene_id"], "metadata": {}}),
                capture_root=root, pipeline_dir=root / "pipeline", source_video=None,
                derived_scene_generation_allowed=False, gates=[],
                prepare_media=media, check_privacy=lambda **_: gate, make_gate=lambda *_: gate)
        except StageError as exc:
            raise PipelineError(str(exc)) from exc
        return {"status": "completed"}

    def deliver():
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai", run_e2e=run_e2e)
        retain_preparation_wakeup(root)
        return json.loads((root / DELIVERY_FILE).read_text())["selectors"]

    def bind_receipt(current: dict) -> None:
        record = json.loads(record_path.read_text())
        source = {"scene_intent_id": "scene-" + "a" * 64, "scene_intent_digest": "sha256:" + "a" * 64,
                  "capture_id": owner["capture_id"], "task_id": record["task_id"],
                  "task_context_digest": current["context_digest"],
                  "scene_revision_digest": record["scene_revision_digest"]}
        receipt = {"schema_version": "blueprint.website_navigation_intake_receipt.v1",
                   "claim_ceiling": "development_only", "scene_id": record["scene_id"],
                   "scene_inputs_digest": record["scene_inputs_digest"],
                   "scene_revision_digest": record["scene_revision_digest"], "outbox_digest": "sha256:" + "4" * 64,
                   "request_digest": "sha256:" + "5" * 64,
                   "intent": {"intent_id": source["scene_intent_id"], "intent_digest": source["scene_intent_digest"]},
                   "source": source, "reserved_usd": 0, "construction_allowance_granted": False}
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        write_json(record_path.parent / "website_navigation_intake_receipt.v1.json", receipt)
        state["receipt"] = receipt

    def reconcile(limit=1):
        return replay.reconcile_held_handoffs(listener, storage_root=root.parents[4], limit=limit,
                                              process_args={"provider": "openai", "run_e2e": run_e2e})

    return SimpleNamespace(root=root, parent=parent, website=website, runs=runs, state=state, deliver=deliver,
                           reconcile=reconcile, bind_receipt=bind_receipt, work=work)


def _confirm(site, **edit) -> dict:
    """The site confirms its brief on the website (or, with ``edit``, changes the task)."""
    site.website["current"] = _seal({**site.parent, "confirmed": True, "confirmed_at": CONFIRMED_AT, **edit})
    return site.website["current"]


def _handoff(site) -> dict:
    return json.loads((site.work / "handoff.json").read_text())


def test_an_unconfirmed_brief_holds_by_name_and_keeps_its_bytes_but_nothing_reruns_until_it_is_confirmed(site):
    site.deliver()
    ledger = listener._read_job_ledger(site.root)
    assert ledger["status"] == "completed" and _handoff(site)["blockers"] == ["website_task_context_not_confirmed"]
    assert ledger[replay.FIELD]["schema_version"] == replay.SCHEMA
    assert replay.held_kind(site.root / "pipeline") == "consent"
    assert site.reconcile() == {"replayed": 0, "held": 1, "already_replayed": 0} and len(site.runs) == 1
    assert replay.preparation_binding(site.root / "pipeline", None) is None


def test_a_brief_confirmed_after_preparation_reruns_the_held_stage_once_on_its_retained_outputs(site):
    selectors = site.deliver()
    successor = _confirm(site)
    # The status still reads while the job waits for its replay (the retained records stay bound).
    assert read_preparation_status(capture_root=site.root, selectors=selectors,
                                   schema_version=STATUS_V6)["state"] == "awaiting_inputs"
    assert site.reconcile()["replayed"] == 1 and len(site.runs) == 2
    ledger = listener._read_job_ledger(site.root)
    assert ledger["status"] == "completed" and ledger["attempt_count"] == 2
    fact = json.loads((site.work / replay.STATE_FILE).read_text())["fact"]
    assert fact["kind"] == "confirmed_successor" and fact["context_digest"] == successor["context_digest"]
    assert fact["confirmation_parent_context_digest"] == site.parent["context_digest"]
    assert fact["confirmed_at"] == CONFIRMED_AT
    assert ledger["attempt_history"][-1][replay.MARK] == fact["fact_digest"]
    # The stage's paid inputs were read under the sealed parent both times: nothing was bought again.
    assert site.state["media"] == [site.parent["context_digest"]] * 2
    # It now waits only on its intake receipt, under the confirmed successor.
    handoff = _handoff(site)
    assert handoff["blockers"] == ["website_task_runnable_policy_missing"]
    assert handoff["native_task_preparation_gap"]["task_context_digest"] == successor["context_digest"]
    assert replay.held_kind(site.root / "pipeline") == "intake_receipt"
    # The same confirmation is never replayed again.
    assert site.reconcile()["replayed"] == 0 and len(site.runs) == 2


def test_the_same_confirmation_is_replayed_once_even_if_the_job_holds_again(site):
    site.deliver()
    _confirm(site)
    site.state["refuse_again"] = True  # the replayed run is refused once more
    assert site.reconcile()["replayed"] == 1 and len(site.runs) == 2
    assert _handoff(site)["blockers"] == ["website_task_context_not_confirmed"]
    # Its fact is spent: the record names it, and no further replay runs for the same confirmation.
    assert json.loads((site.work / replay.STATE_FILE).read_text())["fact"]["kind"] == "confirmed_successor"
    for _ in range(2):
        assert site.reconcile()["replayed"] == 0
    assert len(site.runs) == 2


def test_a_replay_that_stops_before_its_claim_keeps_the_fact_eligible(site, monkeypatch):
    site.deliver()
    _confirm(site)
    real = listener.process_handoff_payload
    # A transient owner read returns before the job is claimed: nothing ran and the ledger did not move.
    monkeypatch.setattr(listener, "process_handoff_payload",
                        lambda *a, **k: {"status": "capture_owner_observation_unavailable_retryable"})
    revision = listener._read_job_ledger(site.root)["revision"]
    site.reconcile()
    assert listener._read_job_ledger(site.root)["revision"] == revision and len(site.runs) == 1
    # The next drain replays the same confirmation, and only once.
    monkeypatch.setattr(listener, "process_handoff_payload", real)
    assert site.reconcile()["replayed"] == 1 and len(site.runs) == 2
    assert site.reconcile()["replayed"] == 0 and len(site.runs) == 2


def test_an_edited_task_is_not_a_successor_and_is_not_replayed(site):
    site.deliver()
    _confirm(site, description="Walk to the other end instead")
    assert site.reconcile() == {"replayed": 0, "held": 1, "already_replayed": 0} and len(site.runs) == 1


def test_a_receipt_bound_after_the_handoff_reruns_it_once_and_the_job_is_handed_off(site):
    site.deliver()
    successor = _confirm(site)
    site.reconcile()
    assert replay.held_kind(site.root / "pipeline") == "intake_receipt"
    assert site.reconcile() == {"replayed": 0, "held": 1, "already_replayed": 0}
    # The screening pass binds the receipt beside the record (website_navigation_intake).
    site.bind_receipt(successor)
    pending_before = replay.arrived_fact(site.root / "pipeline")
    assert pending_before["kind"] == "intake_receipt"
    assert site.reconcile()["replayed"] == 1 and len(site.runs) == 3
    handoff = _handoff(site)
    assert handoff["status"] == "intake_ready" and handoff["navigation_preparation"]["result_claimed"] is False
    retain_preparation_wakeup(site.root)
    selectors = json.loads((site.root / DELIVERY_FILE).read_text())["selectors"]
    assert read_preparation_status(capture_root=site.root, selectors=selectors,
                                   schema_version=STATUS_V6)["state"] == "handed_off"
    assert replay.held_kind(site.root / "pipeline") is None
    assert site.reconcile() == {"replayed": 0, "held": 0, "already_replayed": 0} and len(site.runs) == 3
    assert site.state["media"] == [site.parent["context_digest"]] * 3
    assert Path(site.root / "pipeline/website_task_context_lineage").is_dir()


def test_an_integration_gap_keeps_its_bytes_and_is_rerun_once_per_new_release_paying_nothing(site):
    """A hold naming a missing Blueprint piece (``pipeline:*``) is re-evaluated once for each release."""
    site.state["gap"] = True
    site.deliver()
    ledger = listener._read_job_ledger(site.root)
    handoff = _handoff(site)
    assert ledger["status"] == "completed" and handoff["blockers"] == ["website_task_runnable_policy_missing"]
    assert "navigation_intake" not in handoff
    assert ledger[replay.FIELD]["schema_version"] == replay.SCHEMA
    assert replay.held_kind(site.root / "pipeline") == "integration_gap"
    # No release identity, no fact: nothing runs.
    assert site.reconcile() == {"replayed": 0, "held": 1, "already_replayed": 0} and len(site.runs) == 1
    site.state["release"] = "a" * 40
    assert site.reconcile()["replayed"] == 1 and len(site.runs) == 2
    fact = json.loads((site.work / replay.STATE_FILE).read_text())["fact"]
    assert fact["kind"] == "release" and fact["release_commit"] == "a" * 40
    assert fact["missing_pipeline"] == [replay.SOURCE_IDENTITY_MISSING]
    ledger = listener._read_job_ledger(site.root)
    assert ledger["attempt_history"][-1][replay.MARK] == fact["fact_digest"]
    assert ledger[replay.FIELD]["schema_version"] == replay.SCHEMA  # still held, still kept
    # The same release never reruns it.
    assert site.reconcile() == {"replayed": 0, "held": 0, "already_replayed": 1} and len(site.runs) == 2
    # A new release, inside the drain tick's existing limit.
    site.state["release"] = "b" * 40
    assert site.reconcile(limit=0)["replayed"] == 0 and len(site.runs) == 2
    # This release sends the intake; the brief was confirmed meanwhile.
    site.state["gap"] = False
    successor = _confirm(site)
    assert site.reconcile()["replayed"] == 1 and len(site.runs) == 3
    handoff = _handoff(site)
    assert handoff["navigation_intake"]["status"] == "forward_pending"
    assert handoff["native_task_preparation_gap"]["task_context_digest"] == successor["context_digest"]
    assert replay.held_kind(site.root / "pipeline") == "intake_receipt"
    assert site.reconcile()["replayed"] == 0 and len(site.runs) == 3
    # Every run read the stage's paid inputs under the sealed parent: nothing was bought again.
    assert site.state["media"] == [site.parent["context_digest"]] * 3
