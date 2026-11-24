"""Internal test jobs: the ``internal_test`` basis, the door's ``test-job`` request and its runner. Fakes only."""
# Covers: src/blueprint_pipeline/sharing_statement.py
# Covers: src/blueprint_pipeline/company_share_permission.py
# Covers: src/blueprint_pipeline/website_company_clip.py
# Covers: src/blueprint_pipeline/website_company_simulation.py
# Covers: src/blueprint_pipeline/capture_original_owner_observer.py
# Covers: src/blueprint_pipeline/internal_test_job_door.py
# Covers: src/blueprint_pipeline/website_scene_workspace_retention.py
# Covers: deploy/operator-door/operator_door/requests.py
# Covers: deploy/operator-door/operator_door/spool_runner.py
# Covers: deploy/operator-door/door-test-job.sh
# Covers: scripts/operator_door.py
# Covers: scripts/test_jobs.py
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import internal_test_job_door as door_job
from blueprint_pipeline import sharing_statement as statement
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from tests.test_operator_door_scripts import SETPRIV_STUB, _run, _write_stub, env  # noqa: F401 - env is a fixture

TEST = "internal_test.v1"
MARKER = {"label": "library-aisle", "footage_id": "pexels-854533"}
ROOT = Path(__file__).resolve().parents[1]


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------- P1: the basis

@pytest.mark.parametrize(("version", "marker", "basis"), [
    (TEST, MARKER, {"permission_basis": "internal_test", "statement_version": TEST, "label": "library-aisle"}),
    (TEST, None, None),                                      # the version without the marker
    ("2026-11-11.v3", MARKER, None),                         # the marker without the version
    (TEST, {"label": "library-aisle"}, None),                # a marker that names no footage
    ("2026-11-12.v4", MARKER, {"permission_basis": "site_sharing_statement", "statement_version": "2026-11-12.v4"}),
    ("2026-11-23.v6", MARKER, {"permission_basis": "site_sharing_statement", "statement_version": "2026-11-23.v6"}),
])
def test_the_internal_test_basis_needs_both_the_version_and_the_marker(version, marker, basis):
    context = {"capture_rights": {"statement_version": version}, **({"internal_test": marker} if marker else {})}
    assert statement.sharing_basis(context) == basis


def test_internal_test_is_never_a_sharing_version():
    assert statement.SHARING_STATEMENT_VERSIONS == ("2026-11-12.v4", "2026-11-23.v6") and not statement.covers_sharing(TEST)


def _test_context(capture: Path, *, marker=MARKER, version=TEST) -> None:
    path = capture / "pipeline" / "website_task_context.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"capture_rights": {"statement_version": version},
                                **({"internal_test": marker} if marker else {})}))


@pytest.mark.parametrize(("kept_marker", "live_marker", "status"), [
    (MARKER, MARKER, "published"),
    (MARKER, None, "not_permitted"),   # the live re-read must agree
    (None, MARKER, "not_permitted"),   # and so must the task context kept at submission
])
def test_a_test_jobs_clip_is_published_only_when_both_reads_agree(tmp_path, kept_marker, live_marker, status):
    from tests import test_website_company_clip as clips

    capture = clips._capture(tmp_path, version=TEST)
    _test_context(capture, marker=kept_marker)
    world = clips.World(rights={**clips._rights(TEST), **({"internal_test": live_marker} if live_marker else {})})
    first = world.publish(capture)
    assert first["status"] == status
    if status == "published":
        assert world.delivered[0]["permission_basis"] == "internal_test"
        assert clips._manifest(capture)["permission"]["label"] == "library-aisle"
        assert world.publish(capture)["status"] == "current" and len(world.delivered) == 1


def test_a_test_jobs_zip_is_built_published_kept_and_withdrawn_like_any_job(tmp_path):
    from tests import test_website_company_simulation as sims

    job = sims.Job(tmp_path, version=TEST)
    _test_context(job.capture)
    world = sims.World(job, rights={**dict(sims.World(job).rights, statement_version=TEST), "internal_test": MARKER})
    assert world.publish()["status"] == "published"
    assert world.delivered[0]["permission_basis"] == "internal_test"
    assert sims._manifest(job)["permission"] == {"permission_basis": "internal_test", "statement_version": TEST,
                                                 "label": "library-aisle"}
    assert world.publish()["status"] == "current"
    sims._withdraw(job.capture)
    assert world.publish()["status"] == "withdrawn"


@pytest.mark.parametrize(("second_read", "age", "holds"), [
    ("unanswered", 60, True),                                        # no answer keeps the grant ...
    ("unanswered", 2 * 6 * 3600, False),                             # ... only for its lifetime
    (OSError("website_capture_withdrawn"), 60, False),               # a definite no ends it at once
])
def test_a_test_jobs_grant_is_kept_through_no_answer_for_its_lifetime_only(tmp_path, second_read, age, holds):
    from blueprint_pipeline import company_share_permission as permission
    from tests import test_website_company_clip as clips

    capture = clips._capture(tmp_path, version=TEST)
    _test_context(capture)
    identity = permission.capture_identity(capture)
    rights = {**clips._rights(TEST), "internal_test": MARKER}
    permission.record_live(capture, identity, lambda _identity: rights, 1_000)
    failure = OSError("timeout") if second_read == "unanswered" else second_read
    check = permission.record_live(capture, identity, lambda _identity: (_ for _ in ()).throw(failure), 1_000 + age)
    assert permission.internal_test_holds(check, 1_000 + age) is holds
    assert permission.basis_holds(capture, identity["capture_id"], "internal_test", 1_000 + age) is holds


# ---------------------------------------------------------------------- P2: the intake read

@pytest.mark.parametrize(("marker", "admitted"), [(MARKER, True), (None, False), ({"label": "x"}, False)])
def test_the_owner_read_admits_internal_test_only_with_its_marker(marker, admitted):
    from blueprint_pipeline.capture_original_owner_observer import _SOURCE_KEYS, CaptureOwnerObservationError
    from tests.test_capture_original_owner_observer import attested_under, validate_original

    value = attested_under(TEST)
    if marker is not None:
        value["capture_rights"]["internal_test"] = marker
    value["source_projection_digest"] = cross_runtime_canonical_digest({key: value[key] for key in _SOURCE_KEYS})
    value["observation_digest"] = cross_runtime_canonical_digest(value, digest_field="observation_digest")
    if admitted:
        assert validate_original(value) == value
    else:
        with pytest.raises(CaptureOwnerObservationError):
            validate_original(value)


# ------------------------------------------------------------------------ P3: the door request

sys.path.insert(0, str(ROOT / "deploy" / "operator-door"))
# Bind the door package now, as test_scene_retirement_operator_bridge does. A module collected earlier in the same
# worker can leave scripts/operator_door.py (the client, a plain module) under the same name, and the lazy
# `from operator_door.requests import ...` below then fails with "'operator_door' is not a package".
if not hasattr(sys.modules.get("operator_door"), "__path__"):
    sys.modules.pop("operator_door", None)
import operator_door  # noqa: E402,F401

REQUEST = {"kind": "test-job", "idempotency_key": "20261201:library", "label": "library-aisle",
           "footage_id": "pexels-854533", "description": "Move the blue book to the top shelf."}


@pytest.mark.parametrize(("change", "valid"), [
    ({}, True), ({"task_target_hint": "blue book"}, False),
    ({"label": "../x"}, False), ({"footage_id": "Pexels_1"}, False), ({"idempotency_key": "short"}, False),
    ({"description": "  "}, False), ({"description": "x" * 2001}, False),
    ({"spend_usd": 5}, False), ({"label": None}, False),
])
def test_the_door_validates_a_test_job_like_the_website(change, valid):
    from operator_door.requests import RequestRefused, required_scope, validate_request

    assert required_scope("test-job") == "operate"
    body = {**REQUEST, **change}
    if valid:
        assert validate_request(body) == {k: body[k] for k in sorted(body)}
    else:
        with pytest.raises(RequestRefused, match="^test_job_fields_invalid$"):
            validate_request(body)


def test_the_door_runs_it_with_the_network_and_only_the_sync_token_of_the_provider_secrets():
    from operator_door.config import DoorConfig
    from operator_door.spool_runner import _LAUNCHES, _scene_window_properties

    launch, config = _LAUNCHES["test-job"], DoorConfig()
    properties = launch.properties(config, {})
    assert launch.script == "door-test-job.sh" and "PrivateNetwork=yes" not in properties
    assert properties[:-2] == _scene_window_properties(config, {}) and properties[-2:] == (
        "TemporaryFileSystem=/etc/blueprint/provider-secrets:ro",
        "BindReadOnlyPaths=/etc/blueprint/provider-secrets/pipeline_sync_token")


class Website:
    """The website's signed route: one job per key, as ``internal-test-jobs.ts`` answers."""

    def __init__(self, fail: Exception | None = None, footage_status: int = 201):
        self.calls, self.fail, self.footage_status = [], fail, footage_status

    def __call__(self, body):
        self.calls.append(dict(body))
        if self.fail:
            raise self.fail
        job = "itest-" + "a" * 39 + str(len(self.calls))
        return {"ok": True, "created": True, "job_id": job, "capture_id": f"walkthrough-{job}",
                "footage": {"state": "copied", "code": None, "status": self.footage_status}}


NOW = 1_800_000_000


def _ask(tmp_path, website, *, month=lambda: None, now=NOW, **change):
    return door_job.run({**REQUEST, **change}, door_request_id="r", ledger_root=tmp_path / "ledger",
                        handoff_root=tmp_path / "handoffs", intent_root=tmp_path / "intents", transport=website,
                        month=month, now=lambda: now)


def _busy(tmp_path, now=NOW):
    return door_job.in_flight(tmp_path / "ledger", tmp_path / "handoffs", now, tmp_path / "intents")


def test_a_test_job_returns_only_its_names(tmp_path):
    assert set(_ask(tmp_path, Website())) == {"door_outcome", "job_id", "capture_id", "label"}


@pytest.mark.parametrize("change", [{"label": "other"}, {"footage_id": "other"}, {"description": "Another job."}])
def test_the_same_key_is_the_same_job_and_a_changed_ask_under_it_is_refused(tmp_path, change):
    website = Website()
    first = _ask(tmp_path, website)
    assert _ask(tmp_path, website) == first and len(website.calls) == 1
    with pytest.raises(door_job.Refused, match="test_job_idempotency_key_reused"):
        _ask(tmp_path, website, **change)


@pytest.mark.parametrize(("website", "error", "holds_slot"), [
    (Website(fail=ValueError("test_job_webapp_http_404:internal_test_footage_not_found")), door_job.Refused, False),
    (Website(footage_status=422), door_job.Refused, False),          # created, but the copy was refused
    (Website(fail=ValueError("test_job_webapp_http_503")), ValueError, True),   # unknown: keeps its slot
    (Website(fail=OSError("connection reset")), OSError, True),
])
def test_only_an_ask_whose_outcome_is_unknown_keeps_its_slot(tmp_path, website, error, holds_slot):
    with pytest.raises(error):
        _ask(tmp_path, website)
    assert _busy(tmp_path) == (["library-aisle"] if holds_slot else [])
    assert _ask(tmp_path, Website())["door_outcome"] == "submitted"  # the same key, asked again


def _finish(tmp_path, result, *, delivered=None, ledger=None):
    """The job's capture on the host, with its package manifest and preparation ledger as given."""
    capture = (tmp_path / "handoffs" / "bucket" / "scenes" / f"site-{result['job_id']}" / "captures"
               / result["capture_id"])
    (capture / "pipeline" / "company_simulation").mkdir(parents=True)
    (capture / "pipeline/company_simulation/company_simulation.json").write_text(json.dumps({"delivered": delivered}))
    if ledger:
        (capture / "pipeline_job_ledger.json").write_text(json.dumps(ledger))


def _scene(tmp_path, result, *, progression=None, terminal_reason=None, name="scene-1"):
    from blueprint_pipeline.task_evaluation_scene_outcome import _decision, record_outcome

    directory = tmp_path / "intents" / name
    directory.mkdir(parents=True)
    intent = {"intent_id": name, "intent_digest": "sha256:" + "1" * 64,
              "request": {"submission_id": result["capture_id"]}}
    (directory / "intent.json").write_text(json.dumps(intent))
    if progression:
        (directory / "progression.json").write_text(json.dumps({"status": progression}))
    if terminal_reason:
        record_outcome(directory, intent, _decision(terminal_reason), now=NOW)


@pytest.mark.parametrize(("finish", "third_allowed"), [
    ({}, False),
    ({"ledger": "terminal_success"}, False),             # preparation done; the scene chain still runs
    ({"scene": {"progression": "running"}}, False),
    ({"scene": {"terminal_reason": "pipeline_fault"}}, False),   # a fault a fix can still re-drive
    ({"delivered": True}, True),
    ({"ledger": "terminal_authority_ended"}, True),
    ({"scene": {"progression": "completed"}}, True),
    ({"scene": {"terminal_reason": "retry_cap_exhausted"}}, True),
    # every intent of the capture must have ended, whatever order the folders list in
    ({"scenes": [{"terminal_reason": "retry_cap_exhausted", "name": "scene-a"},
                 {"progression": "running", "name": "scene-b"}]}, False),
    ({"scenes": [{"progression": "running", "name": "scene-a"},
                 {"terminal_reason": "retry_cap_exhausted", "name": "scene-b"}]}, False),
])
def test_a_third_job_waits_while_two_are_in_flight(tmp_path, finish, third_allowed):
    website = Website()
    first = _ask(tmp_path, website, idempotency_key="20261201:one")
    _ask(tmp_path, website, idempotency_key="20261201:two")
    _finish(tmp_path, first, delivered={"at": 1} if finish.get("delivered") else None,
            ledger={"status": finish["ledger"]} if finish.get("ledger") else None)
    for scene in ([finish["scene"]] if finish.get("scene") else finish.get("scenes", [])):
        _scene(tmp_path, first, **scene)
    if third_allowed:
        assert _ask(tmp_path, website, idempotency_key="20261201:three")["door_outcome"] == "submitted"
    else:
        with pytest.raises(door_job.Refused, match="test_job_two_in_flight"):
            _ask(tmp_path, website, idempotency_key="20261201:three")


def _held(tmp_path, result, *, ledger, handoff_status="needs_input", **fields):
    """The job's capture as the preparation leaves it: its ledger, its retained context, and a handoff that
    ended on a stop only a new video clears (``website_recapture_required``), one still waiting on Atlas, or
    (``fields``) any other."""
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.website_recapture_required import FIELD, NOT_OBSERVED_CODE, record

    _finish(tmp_path, result, ledger={"schema_version": "pipeline_job_ledger.v1", "status": ledger})
    capture = next((tmp_path / "handoffs").glob(f"*/scenes/site-{result['job_id']}/captures/{result['capture_id']}"))
    context = {"scene_id": "site-" + result["job_id"], "capture_id": result["capture_id"],
               "context_digest": "sha256:" + "4" * 64}
    (capture / "pipeline" / "website_task_context.json").write_text(json.dumps(context))
    stop = handoff_status == "needs_input" and not fields
    handoff = {"schema_version": "website_scene_handoff.v1", "scene_id": context["scene_id"],
               "capture_id": context["capture_id"], "status": handoff_status, "simulator_ready": False,
               "blockers": [NOT_OBSERVED_CODE] if stop else [] if fields else ["website_atlas_job_processing"],
               **({FIELD: record(NOT_OBSERVED_CODE, task_context_digest=context["context_digest"])} if stop else {}),
               **fields}
    handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    (capture / "pipeline" / "website_scene_preparation").mkdir(parents=True)
    (capture / "pipeline" / "website_scene_preparation" / "handoff.json").write_text(json.dumps(handoff))


def _two_asked(tmp_path):
    website = Website()
    first = _ask(tmp_path, website, idempotency_key="20261201:one", label="one")
    _ask(tmp_path, website, idempotency_key="20261201:two", label="two")
    return first


@pytest.mark.parametrize(("ledger", "handoff_status", "busy"), [
    ("completed", "needs_input", ["two"]),                     # held for the site: nobody answers a test job
    ("failed_retryable", "awaiting_reconstruction", ["one", "two"]),   # waiting on Atlas: still in flight
    ("processing", "needs_input", ["one", "two"]),             # a release reopened it: it counts again
])
def test_a_preparation_held_for_the_site_frees_its_slot_and_a_reopened_one_counts_again(
        tmp_path, ledger, handoff_status, busy):
    _held(tmp_path, _two_asked(tmp_path), ledger=ledger, handoff_status=handoff_status)
    assert _busy(tmp_path) == busy
    # The 7-day window is unchanged: past it, nothing counts.
    assert _busy(tmp_path, NOW + door_job.IN_FLIGHT_FOR_SECONDS) == []


def test_a_completed_preparation_handed_on_keeps_its_slot(tmp_path):
    """Completed is not ended: a preparation handed to the scene chain still has its heavy work ahead."""
    _held(tmp_path, _two_asked(tmp_path), ledger="completed", handoff_status="ready", simulator_ready=True,
          website_intake_outbox={"state": "forward_pending"})
    assert _busy(tmp_path) == ["one", "two"]


@pytest.mark.parametrize(("reader", "handoff_status", "busy"), [
    ("retains_task_family_preparation", "needs_input", ["two"]),            # a family prepared, then held
    ("retains_task_family_preparation", "task_evaluation_ready", ["one", "two"]),  # a family handed on
    ("retains_room_package", "room_package_written", ["one", "two"]),     # the room package goes on
])
def test_only_a_family_waiting_for_input_is_a_hold_and_a_room_package_is_not(tmp_path, monkeypatch, reader,
                                                                              handoff_status, busy):
    from blueprint_pipeline import website_task_context_contracts as contracts

    monkeypatch.setattr(contracts, reader, lambda handoff, context: True)
    _held(tmp_path, _two_asked(tmp_path), ledger="completed", handoff_status=handoff_status, kind=reader)
    assert _busy(tmp_path) == busy


def _retire(tmp_path, result, *, status="completed", proof=True, forge=None):
    """Storage retention's retirement of the job's scene as ``apply_scene_workspace_retirement`` leaves it
    (nav-doorway-96 and move-the-lamp, 2026-11-21): the receipt beside ``scenes/``, the workspace gone. ``forge``
    changes the receipt after its digest, or rewrites the file, as no retirement writes it."""
    import shutil
    from blueprint_pipeline import website_scene_workspace_retention as retention
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    scene_id = f"site-{result['job_id']}"
    scene = tmp_path / "handoffs" / "bucket" / "scenes" / scene_id
    capture = scene / "captures" / result["capture_id"]
    capture.mkdir(parents=True, exist_ok=True)
    ledger = {"schema_version": "pipeline_job_ledger.v1", "scene_id": scene_id, "capture_id": result["capture_id"],
              "status": status, "terminal_payload_sha256": "e" * 64}
    (capture / "pipeline_job_ledger.json").write_text(json.dumps(ledger))
    if proof and status == "completed":
        (capture / "pipeline_job_output_commit.json").write_text(json.dumps({
            "schema_version": "pipeline_job_output_commit.v1", "status": "committed", "scene_id": scene_id,
            "capture_id": result["capture_id"], "result_sha256": "f" * 64}))
    elif proof:
        ended = {"schema_version": "pipeline_job_terminal_receipt.v1", "status": "authority_ended",
                 "scene_id": scene_id, "capture_id": result["capture_id"], "payload_sha256": "e" * 64}
        ended["receipt_digest"] = canonical_digest(ended, digest_field="receipt_digest")
        (capture / "pipeline_job_terminal_receipt.json").write_text(json.dumps(ended))
    records = retention._capture_records(scene, [{"capture_id": result["capture_id"], "ledger_status": status}])
    document = retention._receipt_document(
        bucket="bucket", scene_id=scene_id, scene=scene, observed_at=NOW, records=records, archive=None,
        plan={"plan_digest": "sha256:" + "a" * 64, "cloud_verified": [], "totals": {}}, retiring_token="0" * 16)
    if forge is not None:
        document = forge(document)
    shutil.rmtree(scene)
    path = retention.receipt_path(tmp_path / "handoffs", "bucket", scene_id)
    assert retention._publish_receipt(path, document if isinstance(document, bytes)
                                      else retention._receipt_bytes(document))
    return path


@pytest.mark.parametrize("status", ["completed", "terminal_authority_ended"])
def test_a_test_job_whose_workspace_retention_retired_frees_its_slot(tmp_path, status):
    """2026-11-24: nav-doorway-96 and move-the-lamp were retired on 2026-11-21 and still held both slots: their
    capture folders were gone, and a test job has no scene intent to end."""
    website = Website()
    first = _ask(tmp_path, website, idempotency_key="20261201:one", label="one")
    second = _ask(tmp_path, website, idempotency_key="20261201:two", label="two")
    assert sorted(_busy(tmp_path)) == ["one", "two"]
    _retire(tmp_path, first, status=status)
    _retire(tmp_path, second, status=status)
    assert _busy(tmp_path) == []
    assert _ask(tmp_path, website, idempotency_key="20261201:three", label="three")["door_outcome"] == "submitted"


def test_a_delivered_job_stays_finished_once_its_workspace_is_retired(tmp_path):
    first = _two_asked(tmp_path)
    _finish(tmp_path, first, delivered={"at": 1})
    assert _busy(tmp_path) == ["two"]
    _retire(tmp_path, first)
    assert _busy(tmp_path) == ["two"]


def _rehashed(change):
    """A receipt changed and given a digest of its own again: it verifies as a document, never as a retirement."""
    def forge(document):
        from blueprint_pipeline.decision_evidence_contracts import canonical_digest
        document = change(document)
        document["receipt_digest"] = canonical_digest(document, digest_field="receipt_digest")
        return document
    return forge


def _ledger(**change):
    return _rehashed(lambda document: {**document, "captures": [
        {**document["captures"][0], "ledger": {**document["captures"][0]["ledger"], **change}}]})


@pytest.mark.parametrize(("forge", "proof"), [
    (lambda document: b"{}", True),                                         # a file by the receipt's name alone
    (lambda document: b"not json", True),
    (lambda document: {**document, "retired_at_epoch": 0}, True),           # changed after its digest
    (_rehashed(lambda document: {**document, "schema_version": "website_scene_workspace_retired.v0"}), True),
    (_ledger(status="failed_retryable"), True),                             # its ledger not final
    (_ledger(capture_id="walkthrough-other"), True),                        # another capture's ledger
    (_rehashed(lambda document: {**document, "captures": []}), True),       # names no capture
    (None, False),                                                           # a final ledger nothing proved
])
def test_a_receipt_that_does_not_verify_frees_no_slot(tmp_path, forge, proof):
    _retire(tmp_path, _two_asked(tmp_path), proof=proof, forge=forge)
    assert sorted(_busy(tmp_path)) == ["one", "two"]


def test_a_capture_staged_again_after_its_retirement_is_in_flight_again(tmp_path):
    """A new delivery stages a retired capture anew (the listener's ``retired_capture_status``): its folder is live."""
    first = _two_asked(tmp_path)
    _retire(tmp_path, first)
    assert _busy(tmp_path) == ["two"]
    _finish(tmp_path, first, ledger={"status": "processing"})
    assert _busy(tmp_path) == ["one", "two"]


def test_a_retry_of_an_unknown_outcome_reads_the_month(tmp_path):
    with pytest.raises(ValueError):
        _ask(tmp_path, Website(fail=ValueError("test_job_webapp_http_503")))
    with pytest.raises(door_job.Refused, match="^test_job_month_stop_line$"):
        _ask(tmp_path, Website(), month=lambda: "test_job_month_stop_line")


@pytest.mark.parametrize(("first", "kept_time"), [
    (Website(fail=ValueError("test_job_webapp_http_404:internal_test_footage_not_found")), False),  # refused
    (Website(fail=ValueError("test_job_webapp_http_503")), False),  # unknown, but its window has passed
])
def test_a_key_asked_again_after_its_window_is_a_new_job_with_its_own_slot(tmp_path, first, kept_time):
    with pytest.raises(ValueError):
        _ask(tmp_path, first)
    later = NOW + door_job.IN_FLIGHT_FOR_SECONDS + 1
    _ask(tmp_path, Website(), now=later)
    assert _busy(tmp_path, later) == ["library-aisle"]


def test_an_ask_never_sent_holds_no_slot(tmp_path, monkeypatch):
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    with pytest.raises(door_job.Refused, match="^test_job_not_sent:test_job_webapp_url_missing$"):
        _ask(tmp_path, door_job.post_test_job)
    assert _busy(tmp_path) == []


def test_a_job_stops_counting_after_the_in_flight_window(tmp_path):
    _ask(tmp_path, Website())
    assert _busy(tmp_path, NOW + door_job.IN_FLIGHT_FOR_SECONDS - 1) == ["library-aisle"]
    assert _busy(tmp_path, NOW + door_job.IN_FLIGHT_FOR_SECONDS) == []


def test_two_door_runs_at_once_cannot_both_take_the_last_slot(tmp_path):
    import threading
    import time

    _ask(tmp_path, Website(), idempotency_key="20261201:one")
    gate, outcomes, count = threading.Barrier(2), [], door_job.in_flight

    def slow_count(*args):  # widen the window between the count and the row's write
        busy = count(*args)
        time.sleep(0.3)
        return busy

    def ask(key):
        gate.wait()
        try:
            outcomes.append(_ask(tmp_path, Website(), idempotency_key=key)["door_outcome"])
        except door_job.Refused as exc:
            outcomes.append(str(exc))
    threads = [threading.Thread(target=ask, args=(key,)) for key in ("20261201:two", "20261201:three")]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(door_job, "in_flight", slow_count)
        [thread.start() for thread in threads]
        [thread.join() for thread in threads]
    assert sorted(outcomes) == ["submitted", "test_job_two_in_flight"]


def test_a_new_job_is_refused_when_the_month_says_so_and_a_known_one_is_not(tmp_path):
    website = Website()
    first = _ask(tmp_path, website)
    with pytest.raises(door_job.Refused, match="^test_job_month_stop_line$"):
        _ask(tmp_path, website, idempotency_key="20261201:two", month=lambda: "test_job_month_stop_line")
    assert _ask(tmp_path, website, month=lambda: "test_job_month_stop_line") == first  # no new job: no month read


@pytest.mark.parametrize(("blockers", "counted", "line", "refusal"), [
    (["account_monthly_spend_limit_reached"], "100", "300", "test_job_month_closed:account_monthly_spend_limit_reached"),
    ([], "279.99", "300", None),
    ([], "280", "300", "test_job_month_stop_line"),     # counted + one job's $20 reaches the line
    ([], None, "300", "test_job_month_closed:account_monthly_spend_report_missing_or_invalid"),
    ([], "no policy", "300", "test_job_month_closed:account_monthly_spend_limit_not_configured"),  # fail closed
])
def test_the_door_reads_the_month_from_the_hosts_own_report(monkeypatch, tmp_path, blockers, counted, line, refusal):
    from blueprint_pipeline import account_monthly_spend as month
    from blueprint_pipeline import paid_spend_meter as meter

    policy = None if counted == "no policy" else {"report_path": "r"}
    monkeypatch.setenv(meter.METER_ROOT_ENV, str(tmp_path / "meter"))  # an empty journal: nothing admitted since
    monkeypatch.setattr(month, "admission_blockers", lambda environ=None: blockers)
    monkeypatch.setattr(month, "configured_policy", lambda environ=None: (policy, None, None))
    monkeypatch.setattr(month, "read_report", lambda path: {"counted_usd": counted, "generated_at": "2026-11-29T12:00:00Z",
                                                            "alert_lines_usd": {door_job.STOP_LINE: line}})
    assert door_job.month_refusal() == refusal


def test_the_door_counts_the_paid_admissions_made_since_the_report_as_the_chokepoint_does(monkeypatch, tmp_path):
    """At $270 of the $300 line a test job ($20 expected) fits; a $15 call granted after the report was written is
    in none of its figures, and the door counts it as the chokepoint does: $285 + $20 reaches the line."""
    from datetime import datetime, timezone

    from blueprint_pipeline import account_monthly_spend as month
    from blueprint_pipeline import paid_spend_meter as meter

    generated = datetime(2026, 11, 29, 12, 0, tzinfo=timezone.utc).timestamp()
    monkeypatch.setenv(meter.METER_ROOT_ENV, str(tmp_path / "meter"))
    monkeypatch.setattr(month, "_clock", lambda: generated + 120)
    monkeypatch.setattr(month, "admission_blockers", lambda environ=None: [])
    monkeypatch.setattr(month, "configured_policy", lambda environ=None: ({"report_path": "r"}, None, None))
    monkeypatch.setattr(month, "read_report", lambda path: {"counted_usd": "270", "generated_at": "2026-11-29T12:00:00Z",
                                                            "alert_lines_usd": {door_job.STOP_LINE: "300"}})
    assert door_job.month_refusal() is None
    meter.record_admission(job_id="req-9", meter_key="before-the-report", resource_class="evaluator_api",
                           root=tmp_path / "meter", now=generated - 60, maximum_cost_usd="15.00")
    assert door_job.month_refusal() is None  # granted before the report was written: not added on top of it
    meter.record_admission(job_id="req-9", meter_key="since-the-report", resource_class="evaluator_api",
                           root=tmp_path / "meter", now=generated + 60, maximum_cost_usd="15.00")
    assert door_job.month_refusal() == "test_job_month_stop_line"


def test_the_stop_line_is_an_existing_alert_line_read_by_door_and_runner():
    from blueprint_pipeline.account_monthly_spend import ALERT_PERCENTS

    assert door_job.STOP_LINE in {str(percent) for percent in ALERT_PERCENTS}
    assert _runner().STOP_LINE is door_job.STOP_LINE and _runner().EXPECTED_JOB_USD is door_job.EXPECTED_JOB_USD


@pytest.fixture()
def test_job_env(env, tmp_path):  # noqa: F811 - the scripts module's fixture
    repository = tmp_path / "repository"
    repository.mkdir()
    _write_stub(tmp_path / "stubs" / "setpriv", SETPRIV_STUB)
    _write_stub(tmp_path / "job-python", "#!/bin/bash\nprintf 'job %s\\n' \"$*\" >> \"$STUB_LOG\"\n"
                                         "printf '%s\\n' \"${FAKE_JOB:-}\"\nexit \"${FAKE_TOOL_RC:-0}\"\n")
    return {**env, "DOOR_VENV_PYTHON": str(tmp_path / "job-python"), "DOOR_CONTROL_PLANE_REPO": str(repository),
            "DOOR_SERVICE_USER": "blueprint", "DOOR_TEST_JOB_REQUEST": json.dumps(REQUEST)}


@pytest.mark.parametrize(("rc", "status", "code"), [(0, "submitted", None), (2, "refused", "test_job_refused"),
                                                    (1, "failed", "test_job_failed")])
def test_the_door_script_runs_the_release_module_as_the_service_user(test_job_env, rc, status, code):
    request_id = "20261201T120000Z-test-job-0000abcd"
    done, outcome, calls = _run("door-test-job.sh", test_job_env, DOOR_REQUEST_ID=request_id,
                                FAKE_JOB=json.dumps({"door_outcome": status}), FAKE_TOOL_RC=str(rc))
    assert (done, outcome["status"], outcome["code"]) == (rc, status, code)
    assert calls[0] == "setpriv --reuid=blueprint --regid=blueprint --init-groups --inh-caps=-all --"
    assert calls[2].startswith("job -m blueprint_pipeline.internal_test_job_door --request-json ")
    assert '"$source_dir"/door-test-job.sh' in (ROOT / "deploy/operator-door/install.sh").read_text()


def test_the_door_script_passes_every_name_the_call_reads_from_the_env_file():
    """The env file's path is fixed, so read the allow-list: the URL, and the token or its file (the loader's two)."""
    script = (ROOT / "deploy/operator-door/door-test-job.sh").read_text()
    allowed = re.search(r'case "\$key" in (.*?)\) ;;', script, re.S).group(1)
    names = {name.strip() for name in allowed.replace("\\", " ").split("|")}
    assert {"PIPELINE_SYNC_WEBAPP_URL", "PIPELINE_SYNC_TOKEN", "PIPELINE_SYNC_TOKEN_FILE"} <= names


def test_a_rerun_of_the_same_key_is_a_new_door_request_and_the_host_decides(tmp_path, monkeypatch):
    """No operation key is forced: the door runs a rerun again, and the host's ledger returns the same job for a
    ``submitted`` row and asks the website again for a ``submit_failed`` one. ``--operation-key`` still works."""
    client = _module(ROOT / "scripts" / "operator_door.py", "operator_door_client_t1")
    sent = []
    monkeypatch.setattr(client, "_submit", lambda body, args: sent.append(args.operation_key) or 0)
    argv = ["test-job", "--idempotency-key", REQUEST["idempotency_key"], "--label", REQUEST["label"],
            "--footage-id", REQUEST["footage_id"], "--description", REQUEST["description"]]
    for extra in ([], [], ["--operation-key", "chosen-operation-key-1"]):
        assert client.run(client.build_parser().parse_args(argv + extra)) == 0
    assert sent == [None, None, "chosen-operation-key-1"]
    flaky = Website(fail=ValueError("test_job_webapp_http_503"))
    with pytest.raises(ValueError):
        _ask(tmp_path, flaky)
    website = Website()
    first = _ask(tmp_path, website)  # the submit_failed row is posted again
    assert _ask(tmp_path, website) == first and len(website.calls) == 1  # the submitted row is not


# ------------------------------------------------------------------------------ P4: the runner

def _runner():
    return _module(ROOT / "scripts" / "test_jobs.py", "test_jobs_t1")


def _status(counted, *, line="300", allowed=True):
    return {"spend_guard": {"account_monthly_spend": {"counted_usd": counted, "alert_lines_usd": {"100": line},
                                                      "admission_allowed": allowed, "blockers": []}}}


@pytest.mark.parametrize(("status", "refused"), [
    (_status(279.0), False),
    (_status(280.0), True),                    # counted (spend plus held) + 20 reaches the stop line
    (_status(10.0, allowed=False), True),      # the limit already closed admission
    ({"spend_guard": {}}, True),               # unreadable: never submit on a guess
])
def test_the_runner_reads_the_doors_stop_line_before_each_submit(status, refused):
    assert (_runner().spend_refusal(status) is not None) == refused


@pytest.mark.parametrize("label", ["rigid pick 1080p", "a", "x" * 80, "A.b_c-d e"])
def test_every_allowed_label_gives_a_key_the_door_accepts(label):
    from operator_door.requests import _TEST_JOB_KEY, _TEST_JOB_LABEL

    assert _TEST_JOB_LABEL.fullmatch(label) and _TEST_JOB_KEY.fullmatch(_runner().door_key("20261201", label))


ROOT_PATH = "/var/lib/blueprint/pubsub-handoffs/b/scenes/site-itest-1/captures/walkthrough-itest-1"


def _job_report(monkeypatch, manifest, *, jobs=(), submitted_at=1, ledger=None, scenes=(), handoff=None):
    runner = _runner()
    files = {f"{ROOT_PATH}/pipeline_job_ledger.json": ledger or {"status": "processing"},
             f"{ROOT_PATH}/pipeline/website_scene_preparation/handoff.json": handoff,
             f"{ROOT_PATH}/pipeline/atlas/task_semantics/s.json": {"semantics": {"task_family": "rigid_pick_place"}},
             f"{ROOT_PATH}/pipeline/company_simulation/company_simulation.json": manifest}
    listing = {runner.HANDOFFS: [{"name": "b"}], ROOT_PATH: [{"name": "pipeline_job_ledger.json", "mtime": 4}],
               f"{ROOT_PATH}/pipeline/atlas/task_semantics": [{"name": "s.json"}],
               str(runner.INTENT_ROOT): [{"name": f"scene-{n}"} for n in range(len(scenes) + 1)]}
    files[f"{runner.INTENT_ROOT}/scene-{len(scenes)}/intent.json"] = {"request": {"submission_id": "walkthrough-other"}}
    for n, (progression, outcome) in enumerate(scenes):
        intent = f"{runner.INTENT_ROOT}/scene-{n}"
        files.update({f"{intent}/intent.json": {"request": {"submission_id": "walkthrough-itest-1"}},
                      f"{intent}/progression.json": progression, f"{intent}/outcome.json": outcome})
    monkeypatch.setattr(runner, "cat_json", lambda path: files.get(path))
    monkeypatch.setattr(runner, "ls", lambda path: listing.get(path, []))
    status = {"spend_guard": {"account_monthly_spend": {"jobs": list(jobs)}}}
    report = runner.job_report({"label": "library-aisle", "footage_id": "pexels-854533", "job_id": "itest-1",
                                "capture_id": "walkthrough-itest-1", "submitted_at": submitted_at}, status)
    return runner, report


def test_the_runner_reports_a_job_from_the_hosts_own_records(monkeypatch):
    manifest = {"simulation_revision": "sha256:" + "r" * 64, "built_at_epoch": 5, "delivered": {"at": 6},
                "package": {"digest": "sha256:" + "d" * 64},
                "permission": {"permission_basis": "internal_test", "label": "library-aisle"}}
    runner, report = _job_report(monkeypatch, manifest, jobs=[
        {"job_id": "itest-1", "by_provider_settled_usd": {"world_labs": 9.0}, "held_usd": 0.0, "settled_usd": 9.0},
        {"job_id": "capture-other", "by_provider_settled_usd": {"openai": 4.0}}])
    assert (report["kind"], report["stage"], report["first_failure"]) == ("rigid_pick_place", "package_delivered", None)
    assert report["zip_digest"] == "sha256:" + "d" * 64 and report["permission"]["label"] == "library-aisle"
    assert [row["job_id"] for row in report["spend"]] == ["itest-1"]
    assert report["times"] == {"submitted": 1, "preparation_updated": 4, "package_built": 5}
    assert runner.done(report)


@pytest.mark.parametrize(("manifest", "ledger", "submitted_at", "failure", "done"), [
    ({"built_at_epoch": 5}, None, NOW, None, False),                                   # still going
    ({"refused_by_webapp": {"reason": "website_company_simulation_not_permitted"}}, None, NOW,
     "website_company_simulation_not_permitted", True),
    ({}, {"status": "terminal_authority_ended"}, NOW, "terminal_authority_ended", True),
    ({}, {"status": "terminal_success"}, NOW, None, False),
    ({}, None, 1, "stalled: no package and no ending after the door's in-flight window", True),
])
def test_follow_ends_with_a_clear_state(monkeypatch, manifest, ledger, submitted_at, failure, done):
    import time as clock

    monkeypatch.setattr(clock, "time", lambda: NOW + 60)
    runner, report = _job_report(monkeypatch, manifest, ledger=ledger, submitted_at=submitted_at)
    assert report["first_failure"] == failure and runner.done(report) is done
    assert report["spend"] is None  # not in the status's top jobs: "spend not shown", never $0.00



@pytest.mark.parametrize(("handoff", "stage"), [
    ({"status": "awaiting_reconstruction", "blockers": ["website_atlas_job_processing"]},
     "preparation:awaiting_reconstruction"),    # the 3D reading's designed wait, which the ledger calls failed_retryable
    (None, "preparation:failed_retryable"),     # no record of its own yet: the ledger's word, as before
])
def test_a_preparation_that_waits_reads_its_own_record_and_names_no_failure(monkeypatch, handoff, stage):
    import time as clock

    monkeypatch.setattr(clock, "time", lambda: NOW + 60)
    _, report = _job_report(monkeypatch, {}, ledger={"status": "failed_retryable"}, submitted_at=NOW, handoff=handoff)
    assert (report["stage"], report["first_failure"]) == (stage, None)


@pytest.mark.parametrize(("ledger", "handoff", "failure", "done"), [
    ({"status": "completed"}, {"status": "needs_input", "blockers": ["website_task_object_not_observed"]},
     "held:website_task_object_not_observed", True),
    ({"status": "failed_retryable"}, {"status": "awaiting_reconstruction", "blockers": ["website_atlas_job_processing"]},
     None, False),
    ({"status": "processing"}, {"status": "needs_input", "blockers": ["website_task_object_not_observed"]},
     None, False),                              # reopened by a release: running again
])
def test_a_preparation_held_for_the_site_ends_the_job_by_its_code(monkeypatch, ledger, handoff, failure, done):
    import time as clock

    monkeypatch.setattr(clock, "time", lambda: NOW + 60)
    runner, report = _job_report(monkeypatch, {}, ledger=ledger, submitted_at=NOW, handoff=handoff)
    assert (report["first_failure"], runner.done(report)) == (failure, done)
    assert report["preparation"] == {"status": handoff["status"], "blockers": handoff["blockers"]}


def test_the_report_names_the_packages_files(monkeypatch):
    manifest = {"delivered": {"at": 6}, "package": {"digest": "sha256:" + "d" * 64,
                                                    "files": ["README.txt", "scene.usda", "success_rule.json"]}}
    _, report = _job_report(monkeypatch, manifest)
    assert report["package_files"] == ["README.txt", "scene.usda", "success_rule.json"]


def test_the_runner_says_it_needs_the_pipelines_python(monkeypatch, capsys):
    runner = _runner()
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda name: None)
    assert runner.main(["report", "--state", "missing.json"]) == 2
    assert "needs the Pipeline's Python" in capsys.readouterr().err


def test_the_runner_never_runs_more_than_two_at_once():
    with pytest.raises(SystemExit):
        _runner().main(["submit", "--state", "s.json", "--matrix", "m.json", "-k", "3"])


def test_the_runner_stops_at_a_refusal_and_follow_ends_on_it(tmp_path, monkeypatch):
    runner, posts = _runner(), []
    monkeypatch.setattr(runner, "DEPLOY_WINDOW_HOUR_UTC", -1)  # any hour of the day: never inside the window
    matrix = tmp_path / "jobs.json"
    matrix.write_text(json.dumps({"jobs": [{"label": f"job {n}", "footage_id": "f", "description": "d"}
                                           for n in range(2)]}))

    def door_json(method, route, query=None, body=None):
        if route == "/status":
            return _status(10.0)
        if method == "POST":
            posts.append(body)
            return {"id": "req-1"}
        return {"outcome": {"status": "refused", "result": "/r.json"}}
    monkeypatch.setattr(runner.door, "_json", door_json)
    monkeypatch.setattr(runner, "cat_json", lambda path: {"door_outcome": "refused", "code": "test_job_two_in_flight"})
    monkeypatch.setattr(runner, "ls", lambda path: [])
    state = tmp_path / "run.json"
    assert runner.main(["submit", "--matrix", str(matrix), "--state", str(state), "--poll", "0"]) == 0
    assert len(posts) == 1  # the second row is never sent after the first was refused
    job = json.loads(state.read_text())["jobs"][0]
    assert runner.done(runner.job_report(job, {}))  # follow ends on it at once


@pytest.mark.parametrize(("scenes", "stage", "failure", "done"), [
    ([({"status": "running"}, None)], "scene:running", None, False),
    ([({"status": "blocked"}, {"terminal": True, "reason_code": "retry_cap_exhausted"})], "preparation:terminal_success",
     "retry_cap_exhausted", True),
    ([({"status": "blocked"}, {"terminal": True, "reason_code": "retry_cap_exhausted"}), ({"status": "running"}, None)],
     "scene:running", None, False),                     # one intent still runs: not ended, as the door reads it
    # a completed progression supersedes a terminal outcome it kept from earlier
    ([({"status": "completed"}, {"terminal": True, "reason_code": "retry_cap_exhausted"})],
     "preparation:terminal_success", None, False),
    ([({"status": "completed"}, {"terminal": True, "reason_code": "retry_cap_exhausted"})], "package_delivered", None,
     True),
    ([({"status": "completed"}, None), ({"status": "blocked"}, {"terminal": True, "reason_code": "retry_cap_exhausted"})],
     "preparation:terminal_success", None, False),
])
def test_follow_reads_the_scene_chain_after_a_successful_preparation(monkeypatch, scenes, stage, failure, done):
    manifest = {"delivered": {"at": 1}} if stage == "package_delivered" else {}
    runner, report = _job_report(monkeypatch, manifest, ledger={"status": "terminal_success"}, submitted_at=NOW,
                                 scenes=[(p, o or {}) for p, o in scenes])
    assert (report["stage"], report["first_failure"], runner.done(report)) == (stage, failure, done)
