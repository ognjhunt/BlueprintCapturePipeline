"""Offline selected-delivery boundaries; no provider/storage/native host calls."""

import ast
import copy
import hashlib
import json
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import pytest
from blueprint_pipeline.selected_handoff_recovery import (
    recover_selected_handoff,
    validate_selection,
    _json,
)


def fixture(tmp_path):
    selection = dict(
        kind="selected-handoff",
        mode="inspect",
        bucket="fixture-bucket",
        scene_id="site-fixture",
        capture_id="walkthrough-fixture",
        marker_generation="1",
        handoff_generation="2",
        receipt_generation="3",
    )
    marker = "scenes/site-fixture/captures/walkthrough-fixture/raw/capture_upload_complete.json"
    key = hashlib.sha256(
        json.dumps([selection["bucket"], marker, "1"], separators=(",", ":")).encode()
    ).hexdigest()
    prefix = f"scenes/site-fixture/captures/walkthrough-fixture/deliveries/{key}/"
    finalize = dict(
        bucket=selection["bucket"],
        object_name=marker,
        generation="1",
        event_id="fixture-event",
        event_source="fixture-source",
    )
    handoff = dict(
        handoff_envelope_version=2,
        source_finalize=finalize,
        source_membership_selector=dict(object_name="fixture-member"),
        pipeline_handoff_uri=f"gs://fixture-bucket/{prefix}pipeline_handoff.json",
    )
    receipt = dict(
        status="published",
        message_id="fixture-message",
        handoff_envelope_version=2,
        scene_id=selection["scene_id"],
        capture_id=selection["capture_id"],
        source_finalize=finalize,
        pipeline_handoff_uri=handoff["pipeline_handoff_uri"],
    )
    objects = {}
    for role, value, name in [
        ("handoff", handoff, "pipeline_handoff.json"),
        ("receipt", receipt, "pipeline_handoff_pubsub_receipt.json"),
    ]:
        raw = json.dumps(value).encode()
        objects[prefix + name] = raw
        selection[role + "_sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
        selection[role + "_size_bytes"] = len(raw)
    pins = []
    calls = []

    class Blob:
        def __init__(self, name, generation):
            self.name = name
            self.generation = generation
            self.size = len(objects[name])

        def reload(self, **kw):
            pins.append(("reload", self.generation, kw["if_generation_match"]))

        def download_as_bytes(self, **kw):
            pins.append(("download", self.generation, kw["if_generation_match"]))
            return objects[self.name]

    client = SimpleNamespace(
        bucket=lambda bucket: SimpleNamespace(blob=lambda name, generation: Blob(name, generation))
    )
    parsed = SimpleNamespace(
        scene_id="site-fixture",
        capture_id="walkthrough-fixture",
        source_membership_selector={"fixture": True},
    )
    listener = SimpleNamespace(
        parse_handoff_payload=lambda raw: parsed,
        _handoff_capture_root=lambda *a, **k: tmp_path / "capture",
        process_handoff_payload=lambda raw, **kw: (
            calls.append((raw, kw))
            or {"status": "processed", "queue_disposition": "terminal_success"}
        ),
    )

    def owner(**kw):
        return dict(
            request_id="fixture", producer_delivery={"kind": "website_browser_capture_delivery"}
        )

    def retirement(**kw):
        return None

    def admission(*a):
        return None

    return selection, objects, client, listener, owner, retirement, admission, pins, calls


def run(tmp_path, edit=None, **override):
    s, objects, client, listener, owner, retirement, admission, pins, calls = fixture(tmp_path)
    if edit:
        edit(s, objects, listener)
    result = recover_selected_handoff(
        s,
        storage_root=tmp_path,
        client=client,
        listener=listener,
        owner_reader=override.get("owner", owner),
        retirement_reader=override.get("retirement", retirement),
        admission_reader=override.get("admission", admission),
        membership_reader=override.get("membership", lambda **kw: None),
    )
    return result, calls, pins, objects


def test_inspect_exact_objects_without_native_dispatch(tmp_path):
    result, calls, pins, _ = run(tmp_path)
    assert result["status"] == "admitted" and not calls
    assert len(pins) == 4 and all(actual == expected for _, actual, expected in pins)
    assert not result["pubsub_pull_performed"] and not result["pubsub_ack_performed"]
    assert "zero" in result["accounting"]  # explicitly refuses zero-cost inference
    assert "fixture" not in json.dumps(result)


def test_dispatch_preserves_exact_bytes_and_uses_native_initial_claim(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_PUBSUB_HANDOFF_PROVIDER", "openai")
    result, calls, _, objects = run(tmp_path, lambda s, o, listener: s.update(mode="dispatch"))
    assert result["status"] == "native_handler_returned" and len(calls) == 1
    raw, kwargs = calls[0]
    assert raw == next(iter(objects.values()))
    assert kwargs == dict(
        storage_root=tmp_path, provider="openai", require_unattempted_delivery=True,
        expected_preparation_purpose="scene_preparation",
    )
    assert result["provider_dispatch_performed"].startswith("unknown")


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", []),
        ("capture_id", "../escape"),
        ("scene_id", "site-other"),
        ("marker_generation", "0"),
        ("handoff_generation", True),
        ("handoff_sha256", "a" * 64),
        ("receipt_size_bytes", True),
        ("handoff_size_bytes", 65537),
        ("payload_digest", "forged"),
        ("expected_assessment_resume", {}),
    ],
)
def test_selector_refuses_expanded_authority(tmp_path, field, value):
    selection, *_ = fixture(tmp_path)
    selection[field] = value
    with pytest.raises(ValueError):
        validate_selection(selection)


@pytest.mark.parametrize(
    "fault",
    [
        "digest",
        "generation",
        "publication",
        "receipt_source",
        "retired",
        "prior_state",
        "owner_failure",
        "admission_failure",
        "provider_missing",
    ],
)
def test_faults_never_reach_native_dispatch(tmp_path, monkeypatch, fault):
    s, objects, client, listener, owner, retirement, admission, pins, calls = fixture(tmp_path)
    s["mode"] = "dispatch"
    monkeypatch.setenv("BLUEPRINT_PUBSUB_HANDOFF_PROVIDER", "openai")

    def fail(*a, **k):
        raise ValueError("private signed URL must never escape")

    if fault == "digest":
        s["handoff_sha256"] = "sha256:" + "0" * 64
    if fault == "generation":
        client.bucket = lambda b: SimpleNamespace(
            blob=lambda *a, **k: SimpleNamespace(generation=99, size=1, reload=lambda **kw: None)
        )
    if fault in {"publication", "receipt_source"}:
        name = list(objects)[1]
        row = json.loads(objects[name])
        row["status"] = "pending" if fault == "publication" else "published"
        if fault == "receipt_source":
            row["source_finalize"]["generation"] = "99"
        raw = json.dumps(row).encode()
        objects[name] = raw
        s["receipt_size_bytes"] = len(raw)
        s["receipt_sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    if fault == "retired":

        def retirement(**kw):
            return {"status": "retired"}

    if fault == "prior_state":
        (tmp_path / "capture").mkdir()
        (tmp_path / "capture" / "old-output").write_text("uncertain")
    if fault == "owner_failure":
        owner = fail
    if fault == "admission_failure":
        admission = fail
    if fault == "provider_missing":
        monkeypatch.delenv("BLUEPRINT_PUBSUB_HANDOFF_PROVIDER", raising=False)
    result = recover_selected_handoff(
        s,
        storage_root=tmp_path,
        client=client,
        listener=listener,
        owner_reader=owner,
        retirement_reader=retirement,
        admission_reader=admission,
        membership_reader=lambda **kw: None,
    )
    assert result["status"] == "blocked" and not calls
    assert "private signed URL" not in json.dumps(result)


def test_unknown_native_outcome_is_not_reported_as_no_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_PUBSUB_HANDOFF_PROVIDER", "openai")
    s, o, c, listener, owner, ret, admit, _, _ = fixture(tmp_path)
    s["mode"] = "dispatch"

    def lost(*a, **kw):
        raise RuntimeError("provider outcome unknown")

    listener.process_handoff_payload = lost
    result = recover_selected_handoff(
        s,
        storage_root=tmp_path,
        client=c,
        listener=listener,
        owner_reader=owner,
        retirement_reader=ret,
        admission_reader=admit,
        membership_reader=lambda **kw: None,
    )
    assert result["status"] == "blocked" and result["provider_dispatch_performed"].startswith(
        "unknown"
    )


def test_duplicate_json_keys_refused():
    with pytest.raises(ValueError):
        _json(b'{"mode":"inspect","mode":"dispatch"}')


def claim_function(ledger):
    # Actual source function with fake lock/commit: isolated durable boundary,
    # not real flock/DB/provider execution. Captures the precheck-to-claim race.
    file = Path(__file__).parents[1] / "src/blueprint_pipeline/pubsub_handoff_listener.py"
    tree = ast.parse(file.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_claim_job_lease"
    )

    @contextmanager
    def locked(*a, **kw):
        yield ledger

    env = dict(
        datetime=datetime,
        timezone=timezone,
        timedelta=timedelta,
        re=re,
        uuid=uuid,
        Path=Path,
        JOB_LEDGER_FILENAME="pipeline_job_ledger.json",
        _locked_job_ledger=locked,
        _string=lambda v: str(v or ""),
        _attempt_history=lambda listener: listener.get("attempt_history", []),
        _ended_payload_digests=lambda listener: set(),
        _ended_delivery_keys=lambda listener: set(),
        _parse_utc_timestamp=lambda v: None,
        _iso_at=lambda v: v.isoformat(),
        TERMINAL_AUTHORITY_STATUS="authority_ended",
        _commit_job_ledger=lambda path, row, **kw: row,
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__", names=[ast.alias(name="annotations")], level=0
                        ),
                        node,
                    ],
                    type_ignores=[],
                )
            ),
            str(file),
            "exec",
        ),
        env,
    )
    return env["_claim_job_lease"]


@pytest.mark.parametrize(
    "ledger",
    [
        {"status": "failed_retryable", "attempt_count": 1},
        {"status": "processing"},
        {"status": "completed"},
        {"status": "corrupt"},
        {"revision": 1},
    ],
)
def test_native_atomic_guard_refuses_any_prior_state_without_commit(tmp_path, ledger):
    before = copy.deepcopy(ledger)
    status, after = claim_function(ledger)(
        tmp_path,
        scene_id="site-fixture",
        capture_id="walkthrough-fixture",
        owner="fixture",
        lease_seconds=30,
        require_unattempted_delivery=True,
    )
    assert status == "prior_delivery_effects_unresolved" and ledger == before and after == before


def test_ordinary_listener_retains_existing_retry_contract(tmp_path):
    status, row = claim_function({"status": "failed_retryable", "attempt_count": 1})(
        tmp_path,
        scene_id="site-fixture",
        capture_id="walkthrough-fixture",
        owner="fixture",
        lease_seconds=30,
    )
    assert status == "claimed" and row["attempt_count"] == 2


def test_new_delivery_can_claim_once(tmp_path):
    status, row = claim_function({})(
        tmp_path,
        scene_id="site-fixture",
        capture_id="walkthrough-fixture",
        owner="fixture",
        lease_seconds=30,
        require_unattempted_delivery=True,
    )
    assert status == "claimed" and row["attempt_count"] == 1


def test_actual_native_body_propagates_initial_only_guard_before_processing(tmp_path, monkeypatch):
    import sys
    from blueprint_pipeline.pubsub_handoff_scene_operations import _process_handoff_payload_body

    from blueprint_pipeline.capture_original_owner_observer import (
        CaptureOwnerObservationError, OWNER_OBSERVATION_REASON_CODES,
    )

    # Consumer boundary with synthetic authority lookup, same selected membership,
    # and actual claim code; no disk locks/provider/database are represented.
    monkeypatch.setitem(
        sys.modules,
        "blueprint_pipeline.capture_original_owner_observer",
        SimpleNamespace(
            CaptureOwnerObservationError=CaptureOwnerObservationError,
            OWNER_OBSERVATION_REASON_CODES=OWNER_OBSERVATION_REASON_CODES,
            load_original_owner_observation=lambda **kw: {
                "observation_digest": "fixture",
                "producer_delivery": {"delivery_key": "sha256:" + "a" * 64},
            }
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "blueprint_pipeline.website_scene_workspace_retention",
        SimpleNamespace(retired_capture_status=lambda **kw: None),
    )
    tmp_path.mkdir(exist_ok=True)
    claims = []
    processing = []

    def claim(*a, **kw):
        claims.append(kw)
        return claim_function({"status": "failed_retryable", "attempt_count": 1})(*a, **kw)

    handoff = SimpleNamespace(
        bucket="fixture-bucket",
        scene_id="site-fixture",
        capture_id="walkthrough-fixture",
        source_finalize={"generation": "1"},
        source_membership_selector={"fixture": True},
    )
    listener = SimpleNamespace(
        parse_handoff_payload=lambda p: handoff,
        payload_sha256=lambda p: "b" * 64,
        _handoff_capture_root=lambda *a, **kw: tmp_path,
        _claim_job_lease=claim,
        _lease_owner=lambda: "fixture",
        stage_handoff_capture=lambda *a, **kw: tmp_path,
        logger=SimpleNamespace(info=lambda *a, **kw: None),
        HandoffCaptureRetired=RuntimeError,
        TERMINAL_AUTHORITY_STATUS="authority_ended",
    )
    result = _process_handoff_payload_body(
        listener,
        b"original fixture",
        storage_root=tmp_path,
        provider="openai",
        run_e2e=lambda **kw: processing.append(kw),
        storage_client=None,
        run_evaluation_prep=True,
        run_e2e_enabled=True,
        stage_control_plane=False,
        control_plane_manifest_path=None,
        control_plane_work_dir=None,
        control_plane_staged_inputs_path=None,
        overwrite_control_plane_input=False,
        lease_owner=None,
        lease_seconds=30,
        payload_digest=None,
        require_unattempted_delivery=True,
    )
    assert result["status"] == "prior_delivery_effects_unresolved" and not processing
    assert claims[0]["require_unattempted_delivery"] is True


@pytest.mark.parametrize("reason", ["capture_owner_purpose_mismatch", "private response token=secret", None])
def test_native_denial_is_visible_as_blocked_not_operator_success(tmp_path, monkeypatch, reason):
    monkeypatch.setenv("BLUEPRINT_PUBSUB_HANDOFF_PROVIDER", "openai")
    s, o, c, listener, owner, ret, admit, _, _ = fixture(tmp_path)
    s["mode"] = "dispatch"
    listener.process_handoff_payload = lambda *a, **kw: {
        "status": "prior_delivery_effects_unresolved",
        "queue_disposition": "retryable",
        "owner_observation_reason": reason,
    }
    result = recover_selected_handoff(
        s,
        storage_root=tmp_path,
        client=c,
        listener=listener,
        owner_reader=owner,
        retirement_reader=ret,
        admission_reader=admit,
        membership_reader=lambda **kw: None,
    )
    assert (
        result["status"] == "blocked"
        and result["native_status"] == "prior_delivery_effects_unresolved"
    )
    assert result["provider_dispatch_performed"].startswith("unknown")
    if reason == "capture_owner_purpose_mismatch":
        assert result["owner_observation_reason"] == reason
    else:
        assert "owner_observation_reason" not in result
    assert "secret" not in json.dumps(result)


def test_unavailable_membership_blocks_inspection_and_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_PUBSUB_HANDOFF_PROVIDER", "openai")

    def unavailable(**kw):
        raise RuntimeError("private missing member must not leak")

    for mode in ("inspect", "dispatch"):
        result, calls, _, _ = run(
            tmp_path, lambda s, o, listener: s.update(mode=mode), membership=unavailable
        )
        assert result["status"] == "blocked" and not calls
        assert "private missing" not in json.dumps(result)
        assert not result.get("source_membership_verified", False)


def test_membership_preflight_uses_same_current_owner_and_parsed_handoff(tmp_path):
    seen = []

    def member(**kw):
        seen.append(kw)

    result, calls, _, _ = run(tmp_path, membership=member)
    assert result["status"] == "admitted" and result["source_membership_verified"]
    assert len(seen) == 1 and seen[0]["handoff"].source_membership_selector
    assert seen[0]["observation"]["producer_delivery"]["kind"] == "website_browser_capture_delivery"
    assert not calls


@pytest.mark.parametrize("reason", [
    "scene_capture_birth_policy_unavailable", "private response token=secret", None, 17,
])
def test_native_staging_reason_is_bounded_and_keeps_unknown_charge_accounting(tmp_path, monkeypatch, reason):
    monkeypatch.setenv("BLUEPRINT_PUBSUB_HANDOFF_PROVIDER", "openai")
    s, _, c, listener, owner, ret, admit, _, _ = fixture(tmp_path)
    s["mode"] = "dispatch"
    listener.process_handoff_payload = lambda *a, **kw: {
        "status": "capture_source_membership_unavailable_retryable",
        "queue_disposition": "retryable", "staging_reason": reason,
    }
    result = recover_selected_handoff(
        s, storage_root=tmp_path, client=c, listener=listener,
        owner_reader=owner, retirement_reader=ret, admission_reader=admit,
        membership_reader=lambda **kw: None,
    )
    assert result["status"] == "blocked"
    assert result["native_status"] == "capture_source_membership_unavailable_retryable"
    assert result["provider_dispatch_performed"].startswith("unknown")
    assert "no assertion of zero prior provider charges" in result["accounting"]
    if reason == "scene_capture_birth_policy_unavailable":
        assert result["staging_reason"] == reason
    else:
        assert "staging_reason" not in result
    assert "secret" not in json.dumps(result)


def test_selected_staging_recovery_refuses_partial_local_state_before_native_call(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_PUBSUB_HANDOFF_PROVIDER", "openai")
    root = tmp_path / "capture"
    root.mkdir()
    retained = root / "partial-staging.pending"
    retained.write_bytes(b"retained partial staging")
    before = retained.read_bytes()
    result, calls, _, _ = run(tmp_path, lambda s, o, listener: s.update(mode="dispatch"))
    assert result["status"] == "blocked" and not calls
    assert result["blockers"] == ["selected_handoff_existing_state_requires_reconciliation"]
    assert retained.read_bytes() == before


@pytest.mark.parametrize("mode", ["atlas-status", "atlas-retry"])
def test_selected_atlas_reuses_original_validators_before_existing_job_surface(tmp_path, monkeypatch, mode):
    from blueprint_pipeline import selected_atlas_preview_recovery as atlas
    selection, objects, client, listener, owner, retirement, _, pins, native_calls = fixture(tmp_path)
    selection["mode"] = mode
    if mode == "atlas-retry":
        selection["atlas_status_digest"] = "sha256:" + "c" * 64
    root = tmp_path / "capture"
    root.mkdir()
    (root / "existing").write_text("Retained state must be allowed only for the existing Atlas job surface")
    membership = (b"original membership", {"raw": []}, [])
    seen = []
    monkeypatch.setattr(atlas, "recover_selected_atlas_preview", lambda s, **kw: (
        seen.append((s, kw)) or {"status": "atlas_status_returned", "atlas_retry_performed": False}))

    def admission(*args):
        raise AssertionError("Atlas must not create or inherit native scene sponsorship")

    result = recover_selected_handoff(selection, storage_root=tmp_path, client=client, listener=listener,
        owner_reader=owner, retirement_reader=retirement, admission_reader=admission,
        membership_reader=lambda **kw: membership)
    assert result["status"] == "atlas_status_returned" and result["source_membership_verified"]
    assert not native_calls and len(pins) == 4 and len(seen) == 1
    assert seen[0][1]["membership"] is membership
    assert seen[0][1]["capture_root"] == root
    # Corrupt the exact immutable delivery selector: the Atlas helper is never reached.
    selection["handoff_sha256"] = "sha256:" + "0" * 64
    result = recover_selected_handoff(selection, storage_root=tmp_path, client=client, listener=listener,
        owner_reader=owner, retirement_reader=retirement, admission_reader=admission,
        membership_reader=lambda **kw: membership)
    assert result["status"] == "blocked" and len(seen) == 1


@pytest.mark.parametrize("mode", ["atlas-status", "atlas-retry"])
def test_selected_atlas_inherits_current_owner_membership_and_retirement_refusals(tmp_path, monkeypatch, mode):
    from blueprint_pipeline import selected_atlas_preview_recovery as atlas
    selection, _, client, listener, owner, retirement, admission, _, _ = fixture(tmp_path)
    selection["mode"] = mode
    if mode == "atlas-retry":
        selection["atlas_status_digest"] = "sha256:" + "c" * 64
    monkeypatch.setattr(atlas, "recover_selected_atlas_preview", lambda *a, **kw: pytest.fail("Atlas reached"))
    for changed in ({"owner_reader": lambda **kw: {"producer_delivery": {"kind": "other"}}},
                    {"membership_reader": lambda **kw: (_ for _ in ()).throw(ValueError("private secret"))},
                    {"retirement_reader": lambda **kw: {"status": "retired"}}):
        arguments = dict(owner_reader=owner, retirement_reader=retirement, admission_reader=admission,
                         membership_reader=lambda **kw: None)
        result = recover_selected_handoff(selection, storage_root=tmp_path, client=client, listener=listener,
                                          **{**arguments, **changed})
        assert result["status"] == "blocked" and "private" not in json.dumps(result)


def discovery_fixture(tmp_path, *, fault=None):
    full, objects, _, listener, owner, retirement, _, _, calls = fixture(tmp_path)
    selection = {key: full[key] for key in (
        "kind", "bucket", "scene_id", "capture_id", "marker_generation")}
    selection["mode"] = "discover-selector"
    reads = []

    class Blob:
        def __init__(self, name, generation=None):
            self.name = name
            self.pinned = generation is not None
            self.generation = generation or (2 if name.endswith("/pipeline_handoff.json") else 3)
            self.size = len(objects[name])
            if fault == "oversized_metadata":
                self.size = 65537
            elif fault == "invalid_generation":
                self.generation = "0"
            elif fault == "changed_metadata" and self.pinned:
                self.size += 1

        def reload(self, **kw):
            if self.pinned:
                assert kw == {"if_generation_match": self.generation, "timeout": 20}
            else:
                assert kw == {"timeout": 20}
            reads.append(("metadata", self.name, self.pinned))

        def download_as_bytes(self, **kw):
            assert self.pinned and kw == {"if_generation_match": self.generation,
                "timeout": 20, "start": 0, "end": self.size, "raw_download": True}
            reads.append(("download", self.name, kw))
            raw = objects[self.name]
            if fault == "oversized_response":
                return raw + b"x"
            return raw

    client = SimpleNamespace(bucket=lambda bucket: SimpleNamespace(blob=lambda name, generation=None: Blob(name, generation)))
    return selection, full, objects, client, listener, owner, retirement, reads, calls


def discover(tmp_path, *, fault=None, edit=None, **overrides):
    s, full, objects, client, listener, owner, retirement, reads, calls = discovery_fixture(tmp_path, fault=fault)
    if edit:
        edit(s, objects, listener)
    seen = []

    def membership(**kw):
        seen.append(kw)
        return None

    result = recover_selected_handoff(s, storage_root=tmp_path, client=client, listener=listener,
        owner_reader=overrides.get("owner", owner), retirement_reader=overrides.get("retirement", retirement),
        membership_reader=overrides.get("membership", membership),
        admission_reader=lambda *a: pytest.fail("selector discovery must not request a grant or admission"))
    return result, full, reads, calls, seen


def test_discovery_returns_exact_reusable_selector_despite_existing_job(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "existing-job.json").write_text("preserve")
    before = sorted(tmp_path.rglob("*"))
    result, full, reads, calls, seen = discover(tmp_path)
    assert result["status"] == "selector_discovered"
    assert result["selector"] == {key: value for key, value in full.items() if key not in {"kind", "mode"}}
    assert validate_selection({**result["selector"], "kind": "selected-handoff", "mode": "atlas-status"})
    assert result["source_membership_verified"] and len(seen) == 1
    assert not result["provider_dispatch_performed"] and not result["new_authority_created"]
    assert not calls and sorted(tmp_path.rglob("*")) == before
    assert (capture / "existing-job.json").read_text() == "preserve"
    assert len(reads) == 6
    assert all(row[2]["end"] <= 65536 for row in reads if row[0] == "download")


@pytest.mark.parametrize("fault,reason", [
    ("oversized_metadata", "selected_handoff_object_metadata_invalid"),
    ("invalid_generation", "selected_handoff_object_metadata_invalid"),
    ("changed_metadata", "selected_handoff_object_changed"),
    ("oversized_response", "selected_handoff_object_changed"),
])
def test_discovery_rejects_changed_or_oversized_objects_without_selector(tmp_path, fault, reason):
    result, _, reads, calls, _ = discover(tmp_path, fault=fault)
    assert result["status"] == "blocked" and result["blockers"] == [reason]
    assert "selector" not in result and not calls
    if fault in {"oversized_metadata", "invalid_generation", "changed_metadata"}:
        assert not any(row[0] == "download" for row in reads)


@pytest.mark.parametrize("fault", ["stale_receipt", "wrong_source", "not_published", "wrong_membership"])
def test_discovery_rechecks_original_publication_and_full_source_membership(tmp_path, fault):
    def edit(s, objects, listener):
        for name, raw in objects.copy().items():
            value = json.loads(raw)
            if fault == "stale_receipt" and name.endswith("_receipt.json"):
                value["source_finalize"]["generation"] = "99"
            elif fault == "not_published" and name.endswith("_receipt.json"):
                value["status"] = "publishing"
            elif fault == "wrong_source" and name.endswith("/pipeline_handoff.json"):
                value["source_finalize"]["object_name"] = "scenes/other/raw/capture_upload_complete.json"
            objects[name] = json.dumps(value).encode()

    def bad_membership(**kw):
        raise ValueError("source membership changed private token=secret")

    result, _, _, calls, _ = discover(tmp_path, edit=edit,
        **({"membership": bad_membership} if fault == "wrong_membership" else {}))
    assert result["status"] == "blocked" and "selector" not in result and not calls
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("which", ["owner", "retirement"])
def test_discovery_respects_current_owner_rights_and_retirement_without_jobs(tmp_path, which):
    def owner(**kw):
        raise ValueError("capture_owner_rights_not_admitted")

    result, _, _, calls, _ = discover(tmp_path, **{
        which: owner if which == "owner" else lambda **kw: {"status": "retired"}})
    assert result["status"] == "blocked" and "selector" not in result and not calls


@pytest.mark.parametrize("extra", [
    {"handoff_generation": "2"}, {"job": "/tmp/other"}, {"command": "retry"},
    {"maximum_cost_usd": 100}, {"atlas_status_digest": "sha256:" + "c" * 64},
])
def test_discovery_input_never_mixes_with_execution_or_caller_chosen_objects(tmp_path, extra):
    selection, *_ = discovery_fixture(tmp_path)
    with pytest.raises(ValueError, match="selected_handoff_fields_invalid"):
        validate_selection({**selection, **extra})
