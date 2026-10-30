"""Exact sealed input inventory and transfer accounting for local YAM inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .molmoact2_yam_assets import (
    INTERFACE_ID, MODEL_ID, MODEL_REVISION, SOURCE_REVISION,
    RELEASE_INVENTORY_PATH, RELEASE_INVENTORY_SHA256, release_inventory,
)
from .molmoact2_yam_dependencies import DEPENDENCY_MANIFEST_FILENAME, DEPENDENCY_MANIFEST_DIGEST, TOTAL_DOWNLOAD_BYTES

RUNTIME_MODULES = (
    "molmoact2_yam_policy.py", "molmoact2_yam_assets.py", "molmoact2_yam_policy_server.py",
    "molmoact2_yam_runtime.py", "molmoact2_yam_provisioning.py", "molmoact2_yam_dependencies.py",
    "molmoact2_yam_bundle.py", "local_policy_provisioning.py",
)
IMAGE_DOWNLOAD_ALLOWANCE_BYTES = 64 * 1024**3
MAXIMUM_OUTPUT_BYTES = 8_000_000_000


def runtime_contract() -> dict[str, Any]:
    value = {"schema_version": "molmoact2_yam_sealed_runtime.v1", "policy_interface_id": INTERFACE_ID,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "source_revision": SOURCE_REVISION,
        "checkpoint_inventory_sha256": RELEASE_INVENTORY_SHA256,
        "checkpoint_download_bytes": sum(row["bytes"] for row in release_inventory()["files"] if row["required_for_inference"]),
        "checkpoint_python_source": "exact_inventory_files_verified_before_trust_remote_code",
        "dependency_manifest_digest": DEPENDENCY_MANIFEST_DIGEST,
        "dependency_download_bytes": TOTAL_DOWNLOAD_BYTES,
        "cold_image_download_allowance_bytes": IMAGE_DOWNLOAD_ALLOWANCE_BYTES,
        "maximum_output_bytes": MAXIMUM_OUTPUT_BYTES, "peak_gpu_memory_measured": False,
        "required_entries": ["blueprint_pipeline/" + RELEASE_INVENTORY_PATH, DEPENDENCY_MANIFEST_FILENAME]}
    value["runtime_digest"] = canonical_digest(value, digest_field="runtime_digest")
    return value


def stage_runtime(runtime: Path) -> dict[str, Any]:
    from .molmoact2_yam_dependencies import stage_dependency_manifest

    release_inventory()
    target = runtime / "blueprint_pipeline" / RELEASE_INVENTORY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((Path(__file__).parent / RELEASE_INVENTORY_PATH).read_bytes())
    stage_dependency_manifest(runtime)
    return runtime_contract()


def validate_runtime_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value != runtime_contract():
        raise ValueError("molmoact2_sealed_runtime_identity_invalid")
    return dict(value)


def transfer_byte_budget(bundle: Mapping[str, Any]) -> tuple[int, int]:
    contract = validate_runtime_contract(bundle["molmoact2_runtime"])
    source = bundle.get("runtime_source_packet") or {}
    external = source.get("packet_size_bytes", 0) if source.get("transport") == "content_addressed_external_layer.v1" else 0
    size = bundle.get("bundle_size_bytes")
    if type(size) is not int or size <= 0 or type(external) is not int or external < 0:
        raise ValueError("molmoact2_transfer_binding_invalid")
    return (size + external + contract["checkpoint_download_bytes"] + contract["dependency_download_bytes"]
            + IMAGE_DOWNLOAD_ALLOWANCE_BYTES, MAXIMUM_OUTPUT_BYTES)


def provisioning_script() -> str:
    return '''#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$RUNTIME_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN || true
/isaac-sim/python.sh -m blueprint_pipeline.molmoact2_yam_provisioning \\
  --runtime "$RUNTIME_DIR" --output "$OUT_DIR" --spec "$RUNTIME_DIR/adp009d_policy_execution_spec.json"
'''
