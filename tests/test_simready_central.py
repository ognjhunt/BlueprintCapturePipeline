from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline.simready_central import (
    BOM, CATALOG, MANIFEST, ROOTS, SimReadyCentralError,
    intake_package, normalize_producer_results, resolve_selection, verify_intake,
)
from blueprint_pipeline.simready_central_qualification import qualify_catalog_asset
from blueprint_pipeline.paired_target_native_import_bundle import (
    build_paired_target_native_import_bundle, validate_paired_target_native_import_bundle,
)

CATALOG_SHA = "a" * 40
DATASET_SHA = "b" * 40
INDEX_URL = f"{CATALOG}/{CATALOG_SHA}/submissions/nvidia/simready-assets"
BASE = f"https://huggingface.co/datasets/nvidia/simready-assets/resolve/{DATASET_SHA}/can/"
TREE_URL = f"https://huggingface.co/api/datasets/nvidia/simready-assets/tree/{DATASET_SHA}/can?recursive=true&limit=1000"


def _json(value):
    return json.dumps(value, sort_keys=True).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def package_fixture():
    usd = b'''#usda 1.0
(defaultPrim = "Can"
metersPerUnit = 1
upAxis = "Z")
def Xform "Can" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
{
    float physics:mass = 0.03
    def Mesh "Shape" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"])
    {
        uniform token physics:approximation = "convexHull"
        point3f[] points = [(-0.03,-0.03,-0.03), (0.03,-0.03,-0.03), (0.03,0.03,-0.03), (-0.03,0.03,-0.03), (-0.03,-0.03,0.03), (0.03,-0.03,0.03), (0.03,0.03,0.03), (-0.03,0.03,0.03)]
        int[] faceVertexCounts = [4,4,4,4,4,4]
        int[] faceVertexIndices = [0,3,2,1,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7]
        asset texture = @texture.png@
    }
}
'''
    content = {"asset.usda": usd, "texture.png": b"retained-texture-fixture"}
    bom = {"format_version": "1.0", "items": [{"relative_path": p, "size": len(d),
            "hash": {"sha256": _sha(d)}} for p, d in content.items()]}
    files = {**content, BOM: _json(bom), ROOTS: _json({"entries": ["asset.usda"]}),
             ".metadata/LICENSE.txt": b"CC-BY-4.0 fixture attribution",
             ".metadata/tests.json": _json({"timestamp": "2026-10-09", "assets": [{
                 "asset": "asset.usda", "features": [{"FET_grasp": {"passed": False}}]}]})}
    files[MANIFEST] = _json({"format_version": "1.0", "package_id": "fixture.can.1",
                            "license": "CC-BY-4.0", "metadata": [{"name": Path(BOM).name,
                            "hash": {"sha256": _sha(files[BOM])}}]})
    tree = [{"type": "file", "path": "can/" + p, "size": len(d),
             "oid": hashlib.sha1(f"blob {len(d)}\0".encode() + d).hexdigest()}
            for p, d in files.items()]
    payloads = {INDEX_URL: (BASE + MANIFEST + "\n").encode(), TREE_URL: _json(tree),
                **{BASE + p: d for p, d in files.items()}}
    return payloads


def pull(tmp_path, payloads=None):
    payloads = payloads if payloads is not None else package_fixture()
    def fetch(url, limit):
        assert len(payloads[url]) <= limit
        return payloads[url]
    receipt = intake_package(catalog_revision=CATALOG_SHA, submission="nvidia/simready-assets",
                             asset="can", accepted_license="CC-BY-4.0", output_root=tmp_path / "intake",
                             fetch=fetch)
    return tmp_path / "intake" / "intake.json", receipt


def test_real_usd_dependency_closure_and_false_producer_result_are_preserved(tmp_path):
    path, result = pull(tmp_path)
    assert verify_intake(path) == result
    assert result["native_simulator_import_qualified"] is False
    assert result["producer_results"][0]["status"] == "failed"
    assert result["producer_results"][0]["raw_result"] is False
    assert result["dependency_audit"]["dependencies"][0]["retained"] is True
    assert result["selection"]["dataset_revision"] == DATASET_SHA
    assert (path.parent / "packages/can" / ".metadata/LICENSE.txt").is_file()


@pytest.mark.parametrize("changed", ["texture.png", ".metadata/LICENSE.txt", MANIFEST])
def test_dependency_and_license_tampering_is_rejected_before_reuse(tmp_path, changed):
    path, _ = pull(tmp_path)
    (path.parent / "packages/can" / changed).write_bytes(b"changed")
    with pytest.raises(SimReadyCentralError, match="package_file_changed"):
        verify_intake(path)


def test_new_dependency_and_symlink_are_rejected(tmp_path):
    path, _ = pull(tmp_path)
    (path.parent / "packages/can" / "extra").symlink_to("texture.png")
    with pytest.raises(SimReadyCentralError, match="package_inventory_changed"):
        verify_intake(path)


def test_bom_and_revision_hashes_are_checked(tmp_path):
    payloads = package_fixture()
    payloads[BASE + "texture.png"] = b"x" * len(payloads[BASE + "texture.png"])
    with pytest.raises(SimReadyCentralError, match="package_content_hash_mismatch"):
        pull(tmp_path, payloads)
    assert not (tmp_path / "intake/intake.json").exists()


def test_metadata_is_verified_against_pinned_tree(tmp_path):
    payloads = package_fixture()
    payloads[BASE + ".metadata/LICENSE.txt"] = b"x" * len(payloads[BASE + ".metadata/LICENSE.txt"])
    with pytest.raises(SimReadyCentralError, match="package_revision_hash_mismatch"):
        pull(tmp_path, payloads)


def test_unlisted_external_reference_blocks_intake(tmp_path):
    payloads = package_fixture()
    # Keep every publisher hash consistent but point a USD property outside the package.
    old = payloads[BASE + "asset.usda"]
    new = old.replace(b"@texture.png@", b"@https://example.com/private.png@")
    payloads[BASE + "asset.usda"] = new
    bom = json.loads(payloads[BASE + BOM])
    bom["items"][0].update(size=len(new), hash={"sha256": _sha(new)})
    payloads[BASE + BOM] = _json(bom)
    manifest = json.loads(payloads[BASE + MANIFEST])
    manifest["metadata"][0]["hash"]["sha256"] = _sha(payloads[BASE + BOM])
    payloads[BASE + MANIFEST] = _json(manifest)
    tree = json.loads(payloads[TREE_URL])
    for row in tree:
        d = payloads[BASE + row["path"].removeprefix("can/")]
        row.update(size=len(d), oid=hashlib.sha1(f"blob {len(d)}\0".encode() + d).hexdigest())
    payloads[TREE_URL] = _json(tree)
    _, receipt = pull(tmp_path, payloads)
    assert receipt["status"] == "blocked_dependency_closure"
    assert receipt["dependency_audit"]["blockers"]


@pytest.mark.parametrize("revision", ["main", "v1", "a" * 39])
def test_catalog_selection_requires_immutable_revision(revision):
    with pytest.raises(SimReadyCentralError, match="catalog_revision_must_be_commit_sha"):
        resolve_selection(catalog_revision=revision, submission="nvidia/simready-assets", asset="can")


def test_supplied_results_keep_unknown_false_and_missing_distinct():
    rows = normalize_producer_results({"a": {"passed": False}, "b": {"passed": None},
                                      "c": {}, "d": {"passed": "false"}}, source="fixture")
    assert [(r["pointer"], r["status"], r["raw_result"]) for r in rows] == [
        ("/a", "failed", False), ("/b", "unknown", None), ("/d", "unknown", "false")]


def test_static_and_existing_native_bundle_retain_every_dependency(tmp_path):
    intake, receipt = pull(tmp_path)
    static_path = tmp_path / "static.json"
    static = qualify_catalog_asset(intake_receipt_path=intake, output_path=static_path)
    assert static["status"] == "catalog_structure_statically_qualified"
    assert static["task_id"] is None
    assert static["native_simulator_import_qualified"] is False
    result = build_paired_target_native_import_bundle(
        catalog_static_qualification_path=static_path,
        runner_path=Path(__file__).parents[1] / "scripts/run_paired_target_native_import_probe.py",
        output_root=tmp_path / "bundle", implementation_commit="c" * 40)
    validate_paired_target_native_import_bundle(tmp_path / "bundle/paired_target_native_import_bundle_receipt.v1.json")
    assert result["paid_execution_authorized_by_bundle"] is False
    with zipfile.ZipFile(result["bundle_path"]) as archive:
        for row in receipt["files"]:
            assert archive.read("provider_runtime/assets/catalog_00/packages/can/" + row["relative_path"]) == (
                intake.parent / "packages/can" / row["relative_path"]).read_bytes()
        request = json.loads(archive.read("provider_runtime/paired_target_native_import_request.v1.json"))
    assert request["source_kind"] == "licensed_catalog_package"
    assert request["replacements"][0]["asset_frame_registration_digest"] is None
    assert request["native_isaac_executed"] is False


def test_catalog_external_validation_uses_source_identity_without_inventing_capture(tmp_path):
    import jsonschema
    from blueprint_pipeline.external_simready_validation import run_external_simready_validation
    from tests.test_external_simready_validation import _command, _fake_validator
    intake, _ = pull(tmp_path)
    static_path = tmp_path / "static.json"
    qualify_catalog_asset(intake_receipt_path=intake, output_path=static_path)
    validator = _fake_validator(tmp_path)
    result = run_external_simready_validation(catalog_intake_receipt_path=intake,
        local_validation_baseline_path=static_path, validator_command=_command(validator),
        validator_version="1.2.3", validator_source_revision="fixture", license_compatible=True)
    assert result["status"] == "passed_advisory"
    request = json.loads(Path(result["request_path"]).read_text())
    assert request["scene_id"] is None and request["capture_id"] is None
    assert request["input"]["privacy_safe_pipeline_derived"] is False
    schema = Path(__file__).parents[1] / "docs/schemas/external_simready_catalog_validation_request.schema.json"
    jsonschema.validate(request, json.loads(schema.read_text()))


def test_optional_failure_does_not_become_a_pass_or_blocking_failure():
    from blueprint_pipeline.external_simready_validation import normalize_findings
    rows = normalize_findings({"features_summary": {
        "optional": {"passed": False, "blocking_failure": False},
        "required": {"passed": False, "blocking_failure": True},
        "unclassified": {"passed": False}}})
    assert [row["status"] for row in rows] == ["failed"] * 3
    assert [row["severity"] for row in rows] == ["warning", "error", "error"]
    assert [row["blocking_failure"] for row in rows] == [False, True, None]


def test_catalog_native_authority_has_own_cap_and_cannot_borrow_scene_lineage(tmp_path, monkeypatch):
    import blueprint_pipeline.paired_target_native_import_vast as vast
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    monkeypatch.setenv("BLUEPRINT_SPEND_AUTHORITY_ROOT", str(tmp_path / "spend"))
    intake, _ = pull(tmp_path)
    static_path = tmp_path / "static.json"
    qualify_catalog_asset(intake_receipt_path=intake, output_path=static_path)
    build_paired_target_native_import_bundle(
        catalog_static_qualification_path=static_path,
        runner_path=Path(__file__).parents[1] / "scripts/run_paired_target_native_import_probe.py",
        output_root=tmp_path / "bundle", implementation_commit="c" * 40)
    receipt_path = tmp_path / "bundle/paired_target_native_import_bundle_receipt.v1.json"
    bundle = validate_paired_target_native_import_bundle(receipt_path)
    arguments = dict(bundle_receipt_path=receipt_path, authorization_reference="test-only explicit cap",
                     authorized_by="test-user", authorized_on="2026-10-23", blueprint_commit="c" * 40,
                     max_hourly_rate_usd=0.5, hard_cap_usd=1.0, hard_ttl_seconds=3600,
                     output_path=tmp_path / "authority.json")
    authority = vast.materialize_paired_target_native_import_paid_attempt_authority(**arguments)
    assert authority["prior_terminal_artifixer"] is None
    assert authority["aggregate_goal_spend_before_attempt_usd"] == 0.0
    assert authority["aggregate_goal_spend_cap_usd"] == 1.0
    assert authority["maximum_automatic_retries"] == 0
    authority["aggregate_goal_spend_cap_usd"] = 12.0
    authority["authorization_digest"] = canonical_digest(authority, digest_field="authorization_digest")
    with pytest.raises(ValueError, match="prior_terminal_spend_mismatch"):
        vast.validate_paired_target_native_import_paid_attempt_authority(
            authority, prepared_bundle=bundle, max_hourly_rate_usd=0.5,
            hard_cap_usd=1.0, hard_ttl_seconds=3600)
    with pytest.raises(ValueError, match="cannot_borrow_scene_spend_authority"):
        vast.materialize_paired_target_native_import_paid_attempt_authority(
            **arguments, prior_artifixer_authority_path=tmp_path / "unrelated.json")
