"""Reopen bounded website authoring evidence without staging raw source media.

The native contract pins an immutable input manifest; the packet pins the
requirements/producer receipts. Missing original evidence fails closed. This
does not grant a provider access to source files or qualify their measurements.
Standalone native packets without a website-authoring claim retain their route.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path

from .decision_evidence_contracts import canonical_digest

INPUT_SCHEMA = "website_family_authoring_evidence_inputs.v1"
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_SOURCE_FRAMES = 4096


class AuthoringEvidenceError(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise AuthoringEvidenceError("website_family_authoring_evidence_" + reason)


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _read(reference, kind, *, as_json=False, as_bytes=False, enforce_metadata_limit=True):
    """Hash the same no-follow descriptor read; check every ancestor linkage."""
    _require(isinstance(reference, Mapping) and set(reference) == {"path", "sha256", "size_bytes"}
        and isinstance(reference["path"], str)
        and isinstance(reference["sha256"], str) and re.fullmatch(r"sha256:[0-9a-f]{64}", reference["sha256"])
        and type(reference["size_bytes"]) is int and reference["size_bytes"] >= 0, "reference_invalid:" + kind)
    path = Path(reference["path"])
    _require(path.is_absolute() and ".." not in path.parts, "path_invalid:" + kind)
    _require(not (as_json and as_bytes), "read_mode_invalid:" + kind)
    _require(not (as_json or as_bytes) or not enforce_metadata_limit or reference["size_bytes"] <= MAX_METADATA_BYTES,
             "metadata_limit:" + kind)
    descriptors, links, chunks = [], [], []
    try:
        directory_flags = getattr(os, "O_PATH", os.O_RDONLY) | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        directory = os.open(path.anchor, directory_flags)
        descriptors.append(directory)
        for part in path.parts[1:-1]:
            child = os.open(part, directory_flags, dir_fd=directory)
            descriptors.append(child)
            links.append((directory, part, child))
            directory = child
        stream = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
        descriptors.append(stream)
        before = os.fstat(stream)
        _require(stat.S_ISREG(before.st_mode) and before.st_size == reference["size_bytes"], "bytes_changed:" + kind)
        digest = hashlib.sha256()
        remaining = reference["size_bytes"]
        while block := os.read(stream, min(1024 * 1024, remaining + 1)):
            _require(len(block) <= remaining, "bytes_changed:" + kind)
            remaining -= len(block)
            digest.update(block)
            if as_json or as_bytes:
                chunks.append(block)
        _require(remaining == 0 and _identity(before) == _identity(os.fstat(stream))
            == _identity(os.stat(path.name, dir_fd=directory, follow_symlinks=False)), "identity_changed:" + kind)
        for parent, name, child in links:
            linked, opened = os.stat(name, dir_fd=parent, follow_symlinks=False), os.fstat(child)
            _require(stat.S_ISDIR(linked.st_mode) and linked.st_dev == opened.st_dev
                     and linked.st_ino == opened.st_ino, "ancestor_changed:" + kind)
        _require("sha256:" + digest.hexdigest() == reference["sha256"], "bytes_changed:" + kind)
        if as_json:
            value = json.loads(b"".join(chunks))
            _require(isinstance(value, dict), "json_invalid:" + kind)
            return value
        if as_bytes:
            return b"".join(chunks)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuthoringEvidenceError("website_family_authoring_evidence_unavailable:" + kind) from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    return None


def _stage(reference, kind=None):
    value = _read(reference, "stage", as_json=True)
    _require(value.get("schema_version") == "website_family_authoring_stage.v1"
        and value.get("stage_digest") == canonical_digest(value, digest_field="stage_digest")
        and (kind is None or value.get("stage") == kind)
        and value.get("claim_ceiling") == "development_only"
        and isinstance(value.get("inputs_and_result"), dict), "stage_invalid")
    return value


def _source_claim(native):
    assets = native.get("assets")
    # A retained native-to-capture target binding still claims source authoring
    # when optional engineering labels and closure pointers were removed.
    return isinstance(assets, list) and any("source_target_id" in asset
               or asset.get("source_geometry_basis") == "admitted_engineering_model"
               or asset.get("engineering_record_digest") is not None for asset in assets if isinstance(asset, Mapping))


def _verify_inputs(manifest, native, outputs):
    _require(manifest.get("schema_version") == INPUT_SCHEMA, "input_schema_invalid")
    binding = native["task_state_binding"]["source_scene_binding"]
    _require(manifest.get("binding") == {key: binding[key] for key in (
        "task_context_digest", "source_geometry_digest", "task_masks_digest", "splat_digest", "collision_mesh_digest")},
        "source_binding_changed")
    registration = _read(manifest["registration_evidence"], "registration", as_json=True)
    if registration.get("schema_version") == "website_metric_layout_registration.v1":
        from .task_family_layout_authoring import verify_metric_layout_registration
        verify_metric_layout_registration(registration)
    _require(registration.get("frame_registration") == binding["frame_registration"]
             and registration.get("source_video_digest") == manifest.get("source_video_digest"), "registration_changed")
    catalog = _read(manifest["runtime_catalog"], "runtime_catalog", as_json=True)
    _require(catalog.get("catalog_digest") == canonical_digest(catalog, digest_field="catalog_digest")
             == outputs["task_family_runtime_selection"]["runtime_catalog_digest"], "catalog_changed")
    frames = manifest.get("source_frames")
    _require(isinstance(frames, list) and 0 < len(frames) <= MAX_SOURCE_FRAMES
             and len({frame["frame_id"] for frame in frames}) == len(frames), "frames_invalid")
    for frame in frames:
        _read(frame["image"], "source_rgb")
        _read(frame["geometry"], "source_geometry")
    records = manifest.get("engineering_assets")
    _require(isinstance(records, list) and 0 < len(records) <= 8, "engineering_assets_invalid")
    for record in records:
        _read(record["engineering_evidence"], "engineering")
    asset_records = {canonical_digest(record): record for record in records}
    for asset in native["assets"]:
        if asset.get("source_geometry_basis") == "admitted_engineering_model":
            record = asset_records.get(asset["engineering_record_digest"])
            _require(record is not None and record["source"]["sha256"] == asset["content_digest"], "engineering_asset_changed")


def _verify(native, reference, *, expected_request=None):
    request = _read(reference, "requirements", as_json=True)
    _require(request.get("schema_version") == "website_task_family_authoring_requirements.v1"
        and request.get("request_digest") == canonical_digest(request, digest_field="request_digest")
        and request.get("native_inputs_authored") is True and request.get("precursor_only") is False
        and request.get("claim_ceiling") == "development_only"
        and request.get("automatic_paid_execution_authorized") is False
        and request.get("provider_mutation_performed") is False
        and (expected_request is None or request == expected_request), "requirements_changed")
    contract = native["task_spec"]["task_success_contract"]
    _require(request["binding"]["native_runtime_contract_digest"] == native["contract_digest"]
        and request["binding"]["task_success_contract_digest"] == contract["contract_digest"]
        and request["binding"]["task_context_digest"] == contract["scope"]["task_context_digest"]
        and [row["stage_id"] for row in request["operations"]] == [row["stage_id"] for row in contract["stages"]],
        "native_binding_changed")
    result = request["producer_result"]
    retained = _stage(result["reference"], "authoring-result")["inputs_and_result"]
    _require(retained == {key: value for key, value in result.items() if key != "reference"}
        and retained.get("schema_version") == "website_button_family_authoring.v1"
        and retained.get("result_digest") == canonical_digest(retained, digest_field="result_digest")
        and retained.get("claim_ceiling") == "development_only"
        and retained.get("automatic_paid_execution_authorized") is False
        and retained.get("provider_mutation_performed") is False
        and retained.get("status") == "authored" and retained.get("requirements") == [], "producer_changed")
    outputs = retained["outputs"]
    from .native_task_runtime_contract import materialize_native_task_runtime_contract
    expected = materialize_native_task_runtime_contract(scene_id=native["scene_id"], task_id=native["task_id"],
        task_spec={"task_kind": native["task_family"], "task_success_contract": outputs["task_success_contract"]},
        **outputs["native_runtime_inputs"])
    _require(expected == native, "producer_native_changed")
    evidence = native["task_state_binding"].get("source_authoring_evidence")
    _require(evidence is not None, "input_manifest_required")
    manifest = _stage(evidence, "evidence-inputs")["inputs_and_result"]
    _verify_inputs(manifest, native, outputs)
    stages = retained.get("stages")
    _require(isinstance(stages, list) and 0 < len(stages) <= 26 and evidence in stages, "stage_closure_invalid")
    native_stages = []
    for stage_reference in stages:
        stage = _stage(stage_reference)
        if stage["stage"] == "native-inputs":
            native_stages.append(stage["inputs_and_result"])
    _require(native_stages == [outputs], "native_stage_changed")
    return dict(reference)


def preparation_authoring_reference(*, inputs, native):
    """Validate every producer dependency before registration or packet copying."""
    authoring = inputs.get("authoring") or {}
    claimed = (authoring.get("request") or {}).get("native_inputs_authored") is True
    bound = native["task_state_binding"].get("source_authoring_evidence")
    if not claimed and bound is None and not _source_claim(native):
        return None  # Existing supplied/manual native contracts make no producer claim.
    _require(claimed and bound is not None and isinstance(authoring.get("reference"), Mapping), "closure_required")
    try:
        return _verify(native, authoring["reference"], expected_request=authoring.get("request"))
    except (KeyError, TypeError, IndexError) as error:
        raise AuthoringEvidenceError("website_family_authoring_evidence_closure_invalid") from error


def verify_packet_authoring(root, by_path):
    """Reopen source refs on the validating host; never copy or grant access."""
    def packet_json(name):
        row = by_path[name]
        return _read({"path": str(Path(root) / name), "sha256": row["sha256"], "size_bytes": row["size_bytes"]},
                     "packet", as_json=True, enforce_metadata_limit=False)

    try:
        request = packet_json("native_task_arena_packet_request.v1.json")
        native = packet_json("native_task_runtime_contract.v1.json")
        reference = request.get("source_authoring_requirements")
        binding = native.get("task_state_binding")
        bound = binding.get("source_authoring_evidence") if isinstance(binding, Mapping) else None
        if reference is None and bound is None and not _source_claim(native):
            return
        _require(all(by_path[name]["size_bytes"] <= MAX_METADATA_BYTES for name in (
            "native_task_arena_packet_request.v1.json", "native_task_runtime_contract.v1.json")), "metadata_limit:packet")
        _require(reference is not None and bound is not None, "closure_required")
        _verify(native, reference)
    except (KeyError, TypeError, IndexError) as error:
        raise AuthoringEvidenceError("website_family_authoring_evidence_closure_invalid") from error
