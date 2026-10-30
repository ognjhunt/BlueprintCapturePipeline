"""Sealed TRLC runtime delivery through the existing native policy allocator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from .decision_evidence_contracts import canonical_digest
from .trlc_pi05_assets import (INTERFACE_ID, MODEL_ID, MODEL_REVISION, SOURCE_REVISION,
    RELEASE_INVENTORY_PATH, RELEASE_INVENTORY_SHA256, TOKENIZER_INVENTORY_PATH,
    TOKENIZER_INVENTORY_SHA256, release_inventory, tokenizer_inventory, resolve_tokenizer,
    verify_tokenizer_directory)
from .trlc_pi05_source import SOURCE_ARCHIVE, SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256, source_manifest
from .native_g1_beta_dependency_profile import (DEPENDENCY_MANIFEST_FILENAME, DEPENDENCY_MANIFEST_DIGEST,
    TOTAL_DOWNLOAD_BYTES, MAXIMUM_EXPANDED_BYTES, artifact_manifest, stage_dependency_manifest)

RUNTIME_MODULES = ("trlc_pi05_policy.py", "trlc_pi05_policy_server.py", "trlc_pi05_runtime.py",
    "trlc_pi05_assets.py", "trlc_pi05_source.py", "trlc_pi05_bundle.py", "trlc_pi05_provisioning.py",
    "native_g1_beta_dependency_profile.py", "local_policy_provisioning.py")
NOTICES_INVENTORY = "_catalog_data/trlc_pi05_notices.v1.json"
NOTICES_SHA256 = "ac5c9f1c8093da1e2fcbc588f4149a9c814db6b1eef72a31bb5a6f886c2525e9"
IMAGE_DOWNLOAD_ALLOWANCE_BYTES = 64 * 1024**3
MAXIMUM_OUTPUT_BYTES = 8_000_000_000
PROVIDER_DISK_BYTES = 200_000_000_000  # Existing native Arena adapter requests200GB.


def notices_inventory():
    raw = (Path(__file__).parent / NOTICES_INVENTORY).read_bytes()
    if hashlib.sha256(raw).hexdigest() != NOTICES_SHA256:
        raise ValueError("trlc_pi05_notices_inventory_changed")
    return json.loads(raw)


def runtime_contract():
    value = {"schema_version": "trlc_pi05_sealed_runtime.v1", "policy_interface_id": INTERFACE_ID,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "source_revision": SOURCE_REVISION,
        "checkpoint_inventory_sha256": RELEASE_INVENTORY_SHA256,
        "checkpoint_download_bytes": sum(row["size_bytes"] for row in release_inventory()["files"]),
        "tokenizer_inventory_sha256": TOKENIZER_INVENTORY_SHA256,
        "tokenizer_delivery": "six_verified_retained_files_embedded_no_gated_fetch",
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256, "notices_inventory_sha256": NOTICES_SHA256,
        "dependency_manifest_digest": DEPENDENCY_MANIFEST_DIGEST, "dependency_download_bytes": TOTAL_DOWNLOAD_BYTES,
        "dependency_expanded_bound_bytes": MAXIMUM_EXPANDED_BYTES,
        "cold_image_download_allowance_bytes": IMAGE_DOWNLOAD_ALLOWANCE_BYTES,
        "maximum_output_bytes": MAXIMUM_OUTPUT_BYTES, "maximum_provider_disk_bytes": PROVIDER_DISK_BYTES,
        "peak_gpu_memory_measured": False, "training_task": "Fold the t-shirt.",
        "rigid_task_evaluation": "out_of_distribution_diagnostic",
        "required_entries": ["blueprint_pipeline/" + name for name in
            (RELEASE_INVENTORY_PATH, TOKENIZER_INVENTORY_PATH, NOTICES_INVENTORY)]
            + [DEPENDENCY_MANIFEST_FILENAME, SOURCE_ARCHIVE, SOURCE_MANIFEST,
               "blueprint_pipeline/_vendor/" + DEPENDENCY_MANIFEST_FILENAME]
            + ["trlc_tokenizer/" + row["path"] for row in tokenizer_inventory()["files"]]
            + ["trlc_notices/" + row["path"] for row in notices_inventory()["files"]]}
    value["runtime_digest"] = canonical_digest(value, digest_field="runtime_digest")
    return value


def validate_runtime_contract(value):
    if value != runtime_contract():
        raise ValueError("trlc_pi05_runtime_identity_changed")
    return dict(value)


def validate_native_inputs(*, runtime_source_receipt: Path, base_bundle_path: Path) -> dict:
    """Refuse a missing native wheel or oversized paid disk before allocation."""
    from .native_task_runtime_source_packet import verify_native_task_runtime_source_packet
    receipt = verify_native_task_runtime_source_packet(runtime_source_receipt)
    wheels = {row["archive_path"]: row for row in receipt.get("runtime_dependency_wheels", [])}
    for row in artifact_manifest()["wheels"]:
        if row["source"] == "runtime_source_packet":
            actual = wheels.get(row["archive_path"], {})
            if (actual.get("size_bytes") != row["size_bytes"]
                    or str(actual.get("sha256", "")).removeprefix("sha256:") != row["sha256"]):
                raise ValueError("trlc_pi05_native_dependency_identity_missing")
    with zipfile.ZipFile(receipt["verified_packet_path"] ) as archive:
        native_expanded = sum(row.file_size for row in archive.infolist())
    with zipfile.ZipFile(base_bundle_path) as archive:
        base_expanded = sum(row.file_size for row in archive.infolist())
    contract = runtime_contract()
    footprint = (receipt["packet_size_bytes"] + native_expanded + base_bundle_path.stat().st_size + base_expanded
        + 2 * source_manifest()["archive_size_bytes"] + source_manifest()["expanded_bytes"]
        + 2 * sum(row["size_bytes"] for row in tokenizer_inventory()["files"])
        + contract["checkpoint_download_bytes"] + TOTAL_DOWNLOAD_BYTES + MAXIMUM_EXPANDED_BYTES
        + IMAGE_DOWNLOAD_ALLOWANCE_BYTES + MAXIMUM_OUTPUT_BYTES + 8 * 1024**3)
    if footprint > PROVIDER_DISK_BYTES:
        raise ValueError("trlc_pi05_provider_disk_budget_exceeded")
    return {"provider_disk_gb": 200, "bounded_footprint_bytes": footprint,
            "native_packet_expanded_bytes": native_expanded, "base_bundle_expanded_bytes": base_expanded}


def stage_runtime(runtime: Path) -> dict:
    from .trlc_pi05_source import stage_source
    directory, _ = resolve_tokenizer()
    target = runtime / "trlc_tokenizer"
    target.mkdir()
    for row in tokenizer_inventory()["files"]:
        shutil.copyfile(directory / row["path"], target / row["path"])
    verify_tokenizer_directory(target)
    package = Path(__file__).parent
    for name in (RELEASE_INVENTORY_PATH, TOKENIZER_INVENTORY_PATH, NOTICES_INVENTORY):
        destination = runtime / "blueprint_pipeline" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / name, destination)
    target = runtime / "trlc_notices"
    target.mkdir()
    for row in notices_inventory()["files"]:
        raw = (package / "_vendor/trlc_pi05_notices" / row["path"]).read_bytes()
        if len(raw) != row["size_bytes"] or "sha256:" + hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("trlc_pi05_license_notice_changed")
        (target / row["path"]).write_bytes(raw)
    stage_source(runtime)
    dependency_path = stage_dependency_manifest(runtime)
    # The shared wheel verifier also reopens its immutable package-local copy.
    # Both locations must work after extraction, without the controller repo.
    vendor = runtime / "blueprint_pipeline/_vendor"
    vendor.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dependency_path, vendor / DEPENDENCY_MANIFEST_FILENAME)
    return runtime_contract()


def transfer_byte_budget(bundle):
    contract = validate_runtime_contract(bundle["trlc_pi05_runtime"])
    source = bundle.get("runtime_source_packet") or {}
    external = source.get("packet_size_bytes", 0) if source.get("transport") == "content_addressed_external_layer.v1" else 0
    size = bundle.get("bundle_size_bytes")
    if type(size) is not int or size <= 0 or type(external) is not int or external < 0:
        raise ValueError("trlc_pi05_transfer_binding_invalid")
    return size + external + contract["checkpoint_download_bytes"] + TOTAL_DOWNLOAD_BYTES + IMAGE_DOWNLOAD_ALLOWANCE_BYTES, MAXIMUM_OUTPUT_BYTES


def provisioning_script():
    return '''#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$RUNTIME_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN || true
/isaac-sim/python.sh -m blueprint_pipeline.trlc_pi05_provisioning \\
 --runtime "$RUNTIME_DIR" --output "$OUT_DIR" --spec "$RUNTIME_DIR/adp009d_policy_execution_spec.json"
'''
