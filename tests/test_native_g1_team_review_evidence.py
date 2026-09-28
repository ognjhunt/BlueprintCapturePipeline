"""Selected delivery reopens actual native worker output and sealed input bytes."""

import json
from pathlib import Path
import shutil
import zipfile

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import native_g1_private_review_delivery as delivery
from blueprint_pipeline import native_g1_private_review_ingest as ingest
from blueprint_pipeline import native_g1_team_provider_bundle as bundle_module
from blueprint_pipeline.native_g1_team_paid_output import verify_g1_team_paid_output
from blueprint_pipeline.native_g1_team_review_evidence import verify_retained_g1_team_review
from tests.test_native_g1_team_paid_output import _evidence
from tests.test_native_g1_team_provider_bundle import _inputs, COMMIT
from tests.test_native_g1_team_policy_run_request import NOW
from tests.test_native_g1_private_review_ingest import _Response


def _completed(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    args, _ = _inputs(inputs, monkeypatch, endpoint=True)
    bundle = bundle_module.build_g1_team_provider_bundle(**args)
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        packet = json.loads(archive.read(bundle_module.PACKET_RELATIVE_PATH))
    plan = json.loads((args["scene_packet_root"] / bundle_module.PLAN_FILENAME).read_text())
    worker = tmp_path / "worker"
    worker.mkdir()
    evidence, _ = _evidence(
        worker, monkeypatch, execution_packet=packet, scene_plan=plan,
        scene_packet_receipt_digest=bundle["scene_packet_receipt_digest"],
    )
    verified = verify_g1_team_paid_output(**evidence)
    run = tmp_path / "run"
    attempt = run / "attempts/attempt_001"
    root = attempt / "immutable_execution"
    (root / "selected-worker").mkdir(parents=True)
    shutil.copytree(evidence["output_dir"], root / "selected-worker/worker")
    native = {
        "schema_version": "native_g1_team_provider_result.v1",
        "status": "completed_development_only", "execution_packet_digest": packet["packet_digest"],
        "worker_output_relative_path": "selected-worker/worker", "candidate_policy_queried": True,
        "public_redistribution_authorized": False, "verified_output": verified,
    }
    native["result_digest"] = canonical_digest(native, digest_field="result_digest")
    (root / bundle_module.RESULT_FILENAME).write_text(json.dumps(native))
    adapter = {"schema_version": "native_g1_team_paid_policy_result.v1", "status": "completed",
               "continuing_spend_from_this_run": False, "attempt_root": str(attempt),
               "g1_team_output_verification": verified}
    adapter_path = run / "adapter_paid_123.json"
    adapter_path.write_text(json.dumps(adapter))
    return args, bundle, adapter_path, root


def _registered(tmp_path, monkeypatch):
    args, bundle, adapter, source = _completed(tmp_path, monkeypatch)
    evidence = verify_retained_g1_team_review(
        adapter_result_path=adapter, bundle_receipt_path=Path(bundle["receipt_path"]),
    )
    assert evidence.review["episodes"][0]["score"]["outcome"] == "failure"
    assert evidence.review["episodes"][0]["policy_query_count"] == 2
    retained = adapter.parent / "native_g1_team_private_review.v1.json"
    retained.write_text(json.dumps(evidence.review))
    result_root = tmp_path / "results"
    result_root.mkdir()
    common = {"adapter_result_path": adapter, "bundle_receipt_path": Path(bundle["receipt_path"]),
              "retained_review_path": retained, "result_root": result_root,
              "run_id": evidence.review["run_id"]}
    record = delivery.materialize_g1_private_review_delivery(**common)
    assert record["artifact_count"] == 3
    receipt = adapter.parent / "private_delivery.json"
    receipt.write_text(json.dumps(record))
    return args, evidence, {**common, "delivery_receipt_path": receipt}, source


def test_real_selected_bundle_worker_registry_and_signed_ingest(tmp_path, monkeypatch):
    args, evidence, common, _ = _registered(tmp_path, monkeypatch)
    # Expiration still refuses another launch, but verification of this output remains possible.
    monkeypatch.setattr("blueprint_pipeline.native_g1_team_policy_run_request.time.time", lambda: NOW + 7200)
    current_authority = {key: value for key, value in args["authority_arguments"].items() if key != "now_epoch"}
    with pytest.raises(ValueError):
        bundle_module.load_verified_g1_team_provider_bundle(
            common["bundle_receipt_path"], expected_implementation_commit=COMMIT,
            authority_arguments=current_authority,
        )
    calls = []
    class Opener:
        def open(self, request, *, timeout):
            payload = json.loads(request.data)
            calls.append(payload)
            return _Response(request.full_url, {"status": "ingested", "run_id": payload["run_id"],
                                                "review_digest": payload["review"]["review_digest"]})
    monkeypatch.setattr(ingest.urllib.request, "build_opener", lambda *_args: Opener())
    result = ingest.ingest_g1_private_review(
        **common, owner_user_id=evidence.review["owner_user_id"],
        organization_id=evidence.review["organization_id"],
        webapp_url="https://tryblueprint.io", sync_token="private-test-token",
    )
    assert result["review_url"] == "https://tryblueprint.io/app/g1-reviews/" + common["run_id"]
    assert len(calls) == 1
    assert calls[0]["review"] == evidence.review
    assert "private-test-token" not in json.dumps(calls)
    assert result["public_redistribution_authorized"] is False


def test_selected_ingest_refuses_owner_or_registered_byte_changes_before_network(tmp_path, monkeypatch):
    _, evidence, common, _ = _registered(tmp_path, monkeypatch)
    monkeypatch.setattr(ingest, "_post_review", lambda **kw: pytest.fail("unverified network contact"))
    with pytest.raises(ValueError, match="owner"):
        ingest.ingest_g1_private_review(
            **common, owner_user_id="different-owner", organization_id=evidence.review["organization_id"],
            webapp_url="https://tryblueprint.io", sync_token="test",
        )
    manifest = evidence.review["episodes"][0]["frame_manifest"]
    target = common["result_root"] / (common["run_id"] + "-activation/evidence")
    (target / manifest["relative_path"]).write_bytes(b"changed registered artifact")
    with pytest.raises(ValueError):
        ingest.ingest_g1_private_review(
            **common, owner_user_id=evidence.review["owner_user_id"], organization_id=evidence.review["organization_id"],
            webapp_url="https://tryblueprint.io", sync_token="test",
        )


def test_selected_retained_evidence_refuses_cached_flags_or_missing_policy_frame(tmp_path, monkeypatch):
    _, bundle, adapter, root = _completed(tmp_path, monkeypatch)
    common = {"adapter_result_path": adapter, "bundle_receipt_path": Path(bundle["receipt_path"])}
    value = json.loads(adapter.read_text())
    original = adapter.read_text()
    value["continuing_spend_from_this_run"] = True
    adapter.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="controller"):
        verify_retained_g1_team_review(**common)
    adapter.write_text(original)
    evidence = verify_retained_g1_team_review(**common)
    manifest = json.loads((root / evidence.review["episodes"][0]["frame_manifest"]["relative_path"]).read_text())
    view = manifest["policy_input_observations"][0]["views"]["head"]["relative_path"]
    (root / "selected-worker/worker/episode/episode" / view).unlink()
    with pytest.raises(ValueError, match="multicamera_frame_manifest"):
        verify_retained_g1_team_review(**common)
