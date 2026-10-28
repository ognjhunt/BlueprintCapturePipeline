# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_family_authoring_evidence.py
#   src/blueprint_pipeline/task_family_button_authoring.py
#   src/blueprint_pipeline/website_task_family_packet.py
#   src/blueprint_pipeline/native_task_arena_bundle.py
"""Source-authoring closure at real packet boundaries; no raw media staging."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_family_authoring_evidence as evidence
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_bundle import verify_native_task_arena_packet
from blueprint_pipeline.task_object_authoring_contracts import file_record
from blueprint_pipeline.website_task_family_packet import materialize_website_task_family_packet
from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
from tests.test_task_family_button_authoring import _fixture


def _prepared(tmp_path, monkeypatch):
    args, row = _fixture(tmp_path, monkeypatch)
    prepared = compile_website_scene_preparation(**args)
    native_path = tmp_path / "authored-native.json"
    write_json(native_path, prepared["task_evaluation_inputs"]["native_runtime_contract"])
    return args, row, prepared, native_path


def _packet(tmp_path, args, prepared, native_path):
    reference = file_record(native_path)
    return materialize_website_task_family_packet(preparation=prepared, runtime_inputs_path=native_path,
        output_root=tmp_path / "packet", intent_reference=reference, attempt_reference=reference,
        release_reference=reference, task_context=args["task_context"])


def _dependency(args, row, prepared, kind):
    inputs = prepared["task_evaluation_inputs"]
    result = inputs["authoring"]["request"]["producer_result"]
    if kind == "registration":
        return Path(args["task_masks"]["registered_layout"]["registration_evidence"]["path"])
    if kind == "engineering":
        return Path(row["authoring_assets"][0]["engineering_evidence"]["path"])
    if kind in {"image", "depth"}:
        return Path(args["source_geometry"]["frames"][0]["image_path" if kind == "image" else "geometry_path"])
    if kind == "retained-result":
        return Path(result["reference"]["path"])
    if kind == "requirements":
        return Path(inputs["authoring"]["reference"]["path"])
    prefix = {"source-stage": "source-", "operation-stage": "operation-", "native-stage": "native-inputs"}[kind]
    return next(Path(ref["path"]) for ref in result["stages"] if Path(ref["path"]).name.startswith(prefix))


@pytest.mark.parametrize("phase", ["before-materialization", "after-packet"])
@pytest.mark.parametrize("kind", ["registration", "engineering", "image", "depth", "retained-result",
                                  "requirements", "source-stage", "operation-stage", "native-stage"])
def test_mutated_authoring_dependencies_refuse_at_packet_boundary(tmp_path, monkeypatch, kind, phase):
    args, row, prepared, native_path = _prepared(tmp_path, monkeypatch)
    packet = _packet(tmp_path, args, prepared, native_path) if phase == "after-packet" else None
    if packet:
        verify_native_task_arena_packet(packet["packet_root"])
    path = _dependency(args, row, prepared, kind)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="authoring_evidence"):
        if packet:
            verify_native_task_arena_packet(packet["packet_root"])
        else:
            _packet(tmp_path, args, prepared, native_path)
    if packet is None:
        assert not (tmp_path / "packet").exists()


def test_source_closure_binds_receipts_without_copying_raw_evidence(tmp_path, monkeypatch):
    args, _row, prepared, native_path = _prepared(tmp_path, monkeypatch)
    packet = _packet(tmp_path, args, prepared, native_path)
    root, receipt, _ = verify_native_task_arena_packet(packet["packet_root"])
    request = json.loads((root / "native_task_arena_packet_request.v1.json").read_text())
    assert request["source_authoring_requirements"] == prepared["task_evaluation_inputs"]["authoring"]["reference"]
    native = prepared["task_evaluation_inputs"]["native_runtime_contract"]
    assert native["task_state_binding"]["source_authoring_evidence"]["sha256"]
    assert not list(root.rglob("*.png")) and not list(root.rglob("*.npz"))
    assert not list(root.rglob("engineering-evidence.json"))
    assert receipt["native_application_claimed"] is False and receipt["policy_episode_claimed"] is False


def _reseal_packet_file(root, name, value):
    field = "request_digest" if "request" in name else "contract_digest"
    value[field] = canonical_digest(value, digest_field=field)
    write_json(root / name, value)
    receipt_path = root / "native_task_arena_packet_receipt.v1.json"
    receipt = json.loads(receipt_path.read_text())
    for row in receipt["artifacts"]:
        if row["relative_path"] == name:
            observed = file_record(root / name)
            row.update(sha256=observed["sha256"], size_bytes=observed["size_bytes"])
    receipt["request_digest" if field == "request_digest" else "runtime_contract_digest"] = value[field]
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(receipt_path, receipt)


@pytest.mark.parametrize("removed", ["requirements", "manifest", "both"])
def test_source_authored_packet_cannot_downgrade_to_manual_compatibility(tmp_path, monkeypatch, removed):
    args, _row, prepared, native_path = _prepared(tmp_path, monkeypatch)
    packet = _packet(tmp_path, args, prepared, native_path)
    root = Path(packet["packet_root"])
    if removed in {"requirements", "both"}:
        name = "native_task_arena_packet_request.v1.json"
        request = json.loads((root / name).read_text())
        request.pop("source_authoring_requirements")
        _reseal_packet_file(root, name, request)
    if removed in {"manifest", "both"}:
        name = "native_task_runtime_contract.v1.json"
        native = json.loads((root / name).read_text())
        native["task_state_binding"].pop("source_authoring_evidence")
        _reseal_packet_file(root, name, native)
    with pytest.raises(ValueError, match="authoring_evidence_closure_required"):
        verify_native_task_arena_packet(root)


def test_source_authored_preparation_cannot_omit_immutable_receipt(tmp_path, monkeypatch):
    args, _row, prepared, native_path = _prepared(tmp_path, monkeypatch)
    prepared["task_evaluation_inputs"].pop("authoring")
    with pytest.raises(ValueError, match="authoring_evidence_closure_required"):
        _packet(tmp_path, args, prepared, native_path)
    assert not (tmp_path / "packet").exists()


def _remove_optional_authoring_markers(native):
    native["task_state_binding"].pop("source_authoring_evidence")
    for asset in native["assets"]:
        asset.pop("source_geometry_basis", None)
        asset.pop("engineering_record_digest", None)
    assert any("source_target_id" in asset for asset in native["assets"])
    native["contract_digest"] = canonical_digest(native, digest_field="contract_digest")


def test_retained_native_source_target_requires_closure_before_packet_creation(tmp_path, monkeypatch):
    args, _row, prepared, native_path = _prepared(tmp_path, monkeypatch)
    inputs = prepared["task_evaluation_inputs"]
    inputs.pop("authoring")
    native = inputs["native_runtime_contract"]
    _remove_optional_authoring_markers(native)
    prepared["intake_request"]["task"]["native_runtime_contract_digest"] = native["contract_digest"]
    write_json(native_path, native)
    with pytest.raises(ValueError, match="authoring_evidence_closure_required"):
        _packet(tmp_path, args, prepared, native_path)
    assert not (tmp_path / "packet").exists()


@pytest.mark.parametrize("mutate_source", [False, True])
def test_retained_native_source_target_requires_closure_in_resealed_packet(tmp_path, monkeypatch, mutate_source):
    args, _row, prepared, native_path = _prepared(tmp_path, monkeypatch)
    packet = _packet(tmp_path, args, prepared, native_path)
    root = Path(packet["packet_root"])
    request_name = "native_task_arena_packet_request.v1.json"
    native_name = "native_task_runtime_contract.v1.json"
    plan_name = "native_task_arena_scene_plan.v1.json"
    request, native, plan = [json.loads((root / name).read_text())
                             for name in (request_name, native_name, plan_name)]
    assert request["producer"] == "website_task_family_packet.v1"
    request.pop("source_authoring_requirements")
    _remove_optional_authoring_markers(native)
    request["native_runtime_contract_digest"] = native["contract_digest"]
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    plan["native_runtime_contract"] = native
    plan["runtime_contract_digest"] = native["contract_digest"]
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    receipt_path = root / "native_task_arena_packet_receipt.v1.json"
    receipt = json.loads(receipt_path.read_text())
    for name, value in ((request_name, request), (native_name, native), (plan_name, plan)):
        write_json(root / name, value)
        observed = file_record(root / name)
        for row in receipt["artifacts"]:
            if row["relative_path"] == name:
                row.update(sha256=observed["sha256"], size_bytes=observed["size_bytes"])
    receipt.update(request_digest=request["request_digest"], runtime_contract_digest=native["contract_digest"],
                   arena_scene_plan_digest=plan["plan_digest"])
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(receipt_path, receipt)
    if mutate_source:
        path = Path(args["task_masks"]["registered_layout"]["registration_evidence"]["path"])
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="authoring_evidence_closure_required"):
        verify_native_task_arena_packet(root)


def test_missing_source_frame_refuses_without_copy_or_permission_fallback(tmp_path, monkeypatch):
    args, _row, prepared, native_path = _prepared(tmp_path, monkeypatch)
    packet = _packet(tmp_path, args, prepared, native_path)
    image = Path(args["source_geometry"]["frames"][0]["image_path"])
    image.rename(image.with_name("retained-away.png"))
    with pytest.raises(ValueError, match="authoring_evidence_unavailable:source_rgb"):
        verify_native_task_arena_packet(packet["packet_root"])


def test_engineering_parent_swap_during_descriptor_read_refuses(tmp_path, monkeypatch):
    args, row, prepared, native_path = _prepared(tmp_path, monkeypatch)
    packet = _packet(tmp_path, args, prepared, native_path)
    path = Path(row["authoring_assets"][0]["engineering_evidence"]["path"])
    identity = path.stat()
    read, raced = evidence.os.read, []

    def swap_parent(descriptor, count):
        block = read(descriptor, count)
        opened = evidence.os.fstat(descriptor)
        if block and not raced and (opened.st_dev, opened.st_ino) == (identity.st_dev, identity.st_ino):
            saved = path.parent.with_name("retained-library")
            path.parent.rename(saved)
            path.parent.symlink_to(saved, target_is_directory=True)
            raced.append(True)
        return block

    monkeypatch.setattr(evidence.os, "read", swap_parent)
    with pytest.raises(ValueError, match="authoring_evidence_ancestor_changed:engineering"):
        verify_native_task_arena_packet(packet["packet_root"])
    assert raced == [True]


def test_growing_evidence_read_is_bounded_by_pinned_size(tmp_path, monkeypatch):
    path = tmp_path / "changing-evidence"
    path.write_bytes(b"x")
    reference = file_record(path)
    read, counts = evidence.os.read, []

    def grow_before_read(descriptor, count):
        if not counts:
            with path.open("ab") as stream:
                stream.write(b"y" * 1024 * 1024)
        counts.append(count)
        return read(descriptor, count)

    monkeypatch.setattr(evidence.os, "read", grow_before_read)
    with pytest.raises(ValueError, match="authoring_evidence_bytes_changed:growth"):
        evidence._read(reference, "growth")
    assert counts == [reference["size_bytes"] + 1]


def test_authoring_specific_metadata_bound_preserves_standalone_packet_compatibility(tmp_path, monkeypatch):
    from tests.test_native_task_arena_bundle import _packet as legacy_packet

    packet = legacy_packet(tmp_path, scene_id="manual-native")
    monkeypatch.setattr(evidence, "MAX_METADATA_BYTES", 1)
    verify_native_task_arena_packet(packet)


def test_source_authored_packet_retains_its_metadata_bound(tmp_path, monkeypatch):
    args, _row, prepared, native_path = _prepared(tmp_path, monkeypatch)
    packet = _packet(tmp_path, args, prepared, native_path)
    monkeypatch.setattr(evidence, "MAX_METADATA_BYTES", 1)
    with pytest.raises(ValueError, match="authoring_evidence_metadata_limit"):
        verify_native_task_arena_packet(packet["packet_root"])
