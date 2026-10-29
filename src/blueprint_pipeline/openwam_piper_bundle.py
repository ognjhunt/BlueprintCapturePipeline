"""Exact sealed artifacts and transfer accounting for the local PiPER runtime."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .openwam_piper_assets import EXPECTED_FILES
from .openwam_piper_policy import INTERFACE_ID, MODEL_ID, MODEL_REVISION, SOURCE_REVISION
from .openwam_piper_source import (
    SOURCE_ARCHIVE, SOURCE_ARCHIVE_BYTES, SOURCE_ARCHIVE_SHA256, SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256,
)

RUNTIME_MODULES = (
    "openwam_piper_policy.py", "openwam_piper_assets.py", "openwam_piper_policy_server.py",
    "openwam_piper_source.py", "openwam_piper_runtime.py", "openwam_piper_provisioning.py",
    "openwam_piper_dependencies.py", "openwam_piper_bundle.py",
    "openwam_piper_client_dependencies.py",
)
IMAGE_DOWNLOAD_ALLOWANCE_BYTES = 64 * 1024**3
MAXIMUM_OUTPUT_BYTES = 8_000_000_000


def stage_runtime(runtime: Path) -> dict[str, Any]:
    from .openwam_piper_dependencies import stage_dependency_manifest, verified_dependency_manifest
    from .openwam_piper_source import stage_source

    source = stage_source(runtime)
    stage_dependency_manifest(runtime)
    dependencies = verified_dependency_manifest()
    contract = {"schema_version": "openwam_piper_sealed_runtime.v1", "policy_interface_id": INTERFACE_ID,
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "source_revision": SOURCE_REVISION,
                "source_archive": source, "checkpoint_download_bytes": sum(size for size, _ in EXPECTED_FILES.values()),
                "dependency_manifest_digest": dependencies["manifest_digest"],
                "dependency_download_bytes": dependencies["total_download_bytes"],
                "cold_image_download_allowance_bytes": IMAGE_DOWNLOAD_ALLOWANCE_BYTES,
                "maximum_output_bytes": MAXIMUM_OUTPUT_BYTES, "peak_gpu_memory_measured": False,
                "required_entries": [SOURCE_ARCHIVE, SOURCE_MANIFEST, "openwam-piper-dependencies.v1.json",
                    *sorted({row["bundled_path"] for row in [*dependencies["server_wheels"], *dependencies["thin_client_wheels"]]
                             if row.get("bundled_path")})]}
    contract["runtime_digest"] = canonical_digest(contract, digest_field="runtime_digest")
    return contract


def validate_runtime_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    from .openwam_piper_dependencies import DEPENDENCY_MANIFEST_DIGEST, TOTAL_DOWNLOAD_BYTES

    expected_source = {"source_revision": SOURCE_REVISION, "archive_sha256": SOURCE_ARCHIVE_SHA256,
                       "archive_size_bytes": SOURCE_ARCHIVE_BYTES, "manifest_sha256": "sha256:" + SOURCE_MANIFEST_SHA256}
    expected_entries = [SOURCE_ARCHIVE, SOURCE_MANIFEST, "openwam-piper-dependencies.v1.json",
                        "openwam_dependencies/antlr4_python3_runtime-4.9.3-py3-none-any.whl"]
    if (not isinstance(value, Mapping) or value.get("schema_version") != "openwam_piper_sealed_runtime.v1"
            or value.get("policy_interface_id") != INTERFACE_ID or value.get("model_id") != MODEL_ID
            or value.get("model_revision") != MODEL_REVISION or value.get("source_revision") != SOURCE_REVISION
            or value.get("source_archive") != expected_source or value.get("required_entries") != expected_entries
            or value.get("dependency_manifest_digest") != DEPENDENCY_MANIFEST_DIGEST
            or value.get("checkpoint_download_bytes") != sum(size for size, _ in EXPECTED_FILES.values())
            or value.get("dependency_download_bytes") != TOTAL_DOWNLOAD_BYTES
            or value.get("cold_image_download_allowance_bytes") != IMAGE_DOWNLOAD_ALLOWANCE_BYTES
            or value.get("maximum_output_bytes") != MAXIMUM_OUTPUT_BYTES
            or value.get("peak_gpu_memory_measured") is not False
            or value.get("runtime_digest") != canonical_digest(value, digest_field="runtime_digest")):
        raise ValueError("openwam_sealed_runtime_identity_invalid")
    return dict(value)


def transfer_byte_budget(bundle: Mapping[str, Any]) -> tuple[int, int]:
    contract = validate_runtime_contract(bundle["openwam_runtime"])
    source = bundle.get("runtime_source_packet") or {}
    external = source.get("packet_size_bytes", 0) if source.get("transport") == "content_addressed_external_layer.v1" else 0
    size = bundle.get("bundle_size_bytes")
    if type(size) is not int or size <= 0 or type(external) is not int or external < 0:
        raise ValueError("openwam_transfer_binding_invalid")
    return (size + external + contract["checkpoint_download_bytes"] + contract["dependency_download_bytes"]
            + IMAGE_DOWNLOAD_ALLOWANCE_BYTES, MAXIMUM_OUTPUT_BYTES)


def provisioning_script() -> str:
    return '''#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$RUNTIME_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN || true
/isaac-sim/python.sh -m blueprint_pipeline.openwam_piper_provisioning \\
  --runtime "$RUNTIME_DIR" --output "$OUT_DIR" --spec "$RUNTIME_DIR/adp009d_policy_execution_spec.json"
'''
