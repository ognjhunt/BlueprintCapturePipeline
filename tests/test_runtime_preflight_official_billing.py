"""Recorded B/D preflight failure shapes close financially without policy claims."""
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import vast_official_billing_extractor as billing

CASES = json.loads((Path(__file__).parent / "fixtures/runtime_preflight_billing/saved_b_d_receipt_shapes.json").read_text())


def write(path, value, field=None):
    if field:
        value[field] = canonical_digest(value, digest_field=field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return value


def file_record(path):
    body = path.read_bytes()
    return {"path": str(path), "sha256": "sha256:" + hashlib.sha256(body).hexdigest(), "size_bytes": len(body)}


def materialize(tmp_path, case):
    """Rebase sanitized saved metadata; only the fixture bundle bytes are synthetic."""
    root = tmp_path / Path(case["original_root"]).name
    records = json.loads(json.dumps(case["records"]).replace(case["original_root"], str(root)))
    terminal = records["allocator-result.json"]
    bundle = records["frozen-bundle/composition_provider_bundle_receipt.json"]
    bundle_path = Path(bundle["bundle_path"])
    bundle_path.parent.mkdir(parents=True)
    bundle_path.write_bytes(("recorded " + case["variant"] + " fixture bundle").encode())
    digest = file_record(bundle_path)["sha256"]
    bundle.update(bundle_sha256=digest, bundle_size_bytes=bundle_path.stat().st_size)
    terminal["bundle_sha256"] = digest
    records["allocator/attempts/attempt_001/vast_provider_run/vast_provider_adapter_result.json"]["provider_bundle_sha256"] = digest
    records["allocator/attempts/attempt_001/object_store_staging/wam_provider_object_store_staging_manifest.json"]["bundle_sha256"] = digest.removeprefix("sha256:")
    manifest = records["allocator/attempts/attempt_001/artifact_manifest.json"]
    manifest["binding"]["bundle_sha256"] = digest
    for relative, value in records.items():
        write(root / relative, value, "authorization_digest" if relative == "operator-diagnostic-authority.json" else None)
    staging = root / "allocator/attempts/attempt_001/object_store_staging/wam_provider_object_store_staging_manifest.json"
    cleanup = root / "allocator/attempts/attempt_001/object_store_staging/wam_provider_object_store_cleanup.json"
    value = json.loads(cleanup.read_text())
    value["staging_manifest_sha256"] = file_record(staging)["sha256"].removeprefix("sha256:")
    write(cleanup, value)
    for key, path in (("adapter_result", Path(terminal["adapter_result_path"])), ("teardown_manifest", Path(terminal["teardown_manifest_path"]))):
        terminal["provider_closeout"][key] = file_record(path)
    for row in manifest["files"]:
        record = file_record(root / "allocator/attempts/attempt_001" / row["relative_path"])
        row.update(sha256=record["sha256"], size_bytes=record["size_bytes"])
    manifest["file_count"] = len(manifest["files"])
    manifest["total_size_bytes"] = sum(row["size_bytes"] for row in manifest["files"])
    write(Path(terminal["artifact_manifest_path"]), manifest, "manifest_digest")
    write(root / "allocator-result.json", terminal, "result_digest")
    return root, terminal


@pytest.mark.parametrize("case", CASES, ids=lambda row: "saved-v28" + row["variant"])
def test_actual_failed_runtime_shapes_admit_financial_closure_only(tmp_path, case):
    root, terminal = materialize(tmp_path, case)
    observed = billing._terminal_evidence(instance_id=case["instance_id"], terminal_result_path=root / "allocator-result.json")
    assert observed["financial_closeout_kind"] == "native_task_arena_runtime_preflight.v1"
    assert observed["terminal_status"] == "blocked"
    assert observed["source_commit"] == CASES[CASES.index(case)]["records"]["operator-diagnostic-authority.json"]["source_commit"]
    assert observed["bundle_sha256"] == terminal["bundle_sha256"]
    assert observed["provider_absence_confirmed"] is True
    assert observed["policy_evaluation_qualified"] is False
    assert observed["scientific_success_inferred"] is False
    assert json.loads((root / "allocator-result.json").read_text())["schema_version"] == "native_task_arena_runtime_preflight.v1"


@pytest.mark.parametrize("fault", ["schema", "digest", "instance", "source", "bundle", "teardown", "zero", "cleanup", "policy_calls"])
def test_corrupt_or_incomplete_closeout_cannot_authorize_reconciliation(tmp_path, fault):
    case = CASES[1]
    root, terminal = materialize(tmp_path, case)
    if fault in ("schema", "digest"):
        terminal["schema_version" if fault == "schema" else "result_digest"] = "invalid"
        write(root / "allocator-result.json", terminal)
    elif fault == "bundle":
        value = json.loads((root / "frozen-bundle/composition_provider_bundle_receipt.json").read_text())
        Path(value["bundle_path"]).write_bytes(b"changed")
    elif fault in ("source", "policy_calls"):
        path = root / "operator-diagnostic-authority.json"
        value = json.loads(path.read_text())
        value["source_commit" if fault == "source" else "policy_model_calls_authorized"] = "b" * 40 if fault == "source" else 1
        write(path, value, "authorization_digest")
    elif fault in ("instance", "teardown"):
        key = "adapter_result" if fault == "instance" else "teardown_manifest"
        path = Path(terminal["adapter_result_path" if fault == "instance" else "teardown_manifest_path"])
        value = json.loads(path.read_text())
        value["vast_instance_ids"] = [999]
        write(path, value)
        terminal["provider_closeout"][key] = file_record(path)
        write(root / "allocator-result.json", terminal, "result_digest")
    elif fault == "zero":
        path = Path(terminal["watchdog_receipt_path"])
        value = json.loads(path.read_text())
        value["final_global_inventory"]["live_resource_count"] = 1
        write(path, value)
    else:
        path = Path(terminal["object_store_cleanup_path"])
        value = json.loads(path.read_text())
        value["all_objects_absent"] = False
        write(path, value)
    with pytest.raises(billing.VastOfficialBillingExtractionError):
        billing._terminal_evidence(instance_id=case["instance_id"], terminal_result_path=root / "allocator-result.json")
