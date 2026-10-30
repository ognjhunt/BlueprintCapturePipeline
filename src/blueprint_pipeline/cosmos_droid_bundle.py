"""Sealed source, interpreter, model transfer and allocation bounds for Cosmos."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .cosmos_droid_dependencies import DEPENDENCY_MANIFEST_FILENAME, DEPENDENCY_MANIFEST_DIGEST, TOTAL_DOWNLOAD_BYTES
from .cosmos_droid_source import SOURCE_ARCHIVE, SOURCE_ARCHIVE_SHA256, SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256
from .decision_evidence_contracts import canonical_digest

CANDIDATES = ("cosmos3_edge_policy_droid", "cosmos3_nano_policy_droid")
RUNTIME_MODULES = (
    "cosmos_edge_beta_runtime.py", "cosmos_edge_droid_policy_runtime.py", "cosmos_edge_droid_policy_server.py",
    "cosmos_nano_beta_runtime.py", "cosmos_nano_droid_policy_runtime.py", "cosmos_droid_bundle.py",
    "cosmos_droid_client_dependencies.py", "cosmos_droid_dependencies.py", "cosmos_droid_source.py", "cosmos_droid_provisioning.py", "local_policy_provisioning.py",
)
IMAGE_DOWNLOAD_ALLOWANCE_BYTES = 64 * 1024**3
MAXIMUM_OUTPUT_BYTES = 8_000_000_000
CHECKPOINT_BYTES = {"cosmos3_edge_policy_droid": 9154401203, "cosmos3_nano_policy_droid": 32_936_198_308}


def profile(candidate: str):
    if candidate == CANDIDATES[0]:
        from . import cosmos_edge_beta_runtime
        return cosmos_edge_beta_runtime
    if candidate == CANDIDATES[1]:
        from . import cosmos_nano_beta_runtime
        return cosmos_nano_beta_runtime
    raise ValueError("cosmos_policy_candidate_identity_invalid")


def runtime_contract(candidate: str) -> dict:
    selected = profile(candidate)
    value = {"schema_version": "cosmos_droid_sealed_runtime.v1", "candidate_id": candidate,
        "model_id": selected.MODEL_ID, "model_revision": selected.MODEL_REVISION,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256, "source_manifest_sha256": "sha256:" + SOURCE_MANIFEST_SHA256,
        "checkpoint_inventory_sha256": "sha256:" + selected.SNAPSHOT_MANIFEST_SHA256,
        "checkpoint_download_bytes": CHECKPOINT_BYTES[candidate], "format_prompt_as_json": candidate == CANDIDATES[0],
        "dependency_manifest_digest": DEPENDENCY_MANIFEST_DIGEST, "dependency_download_bytes": TOTAL_DOWNLOAD_BYTES,
        "server_python_version": "3.13.9", "minimum_gpu_memory_gb": selected.MINIMUM_GPU_MEMORY_GB,
        "cold_image_download_allowance_bytes": IMAGE_DOWNLOAD_ALLOWANCE_BYTES,
        "maximum_output_bytes": MAXIMUM_OUTPUT_BYTES, "peak_gpu_memory_measured": False,
        "required_entries": [selected.SNAPSHOT_STAGED_NAME, SOURCE_ARCHIVE, SOURCE_MANIFEST,
            DEPENDENCY_MANIFEST_FILENAME, "cosmos_dependencies/antlr4_python3_runtime-4.9.3-py3-none-any.whl"]}
    value["runtime_digest"] = canonical_digest(value, digest_field="runtime_digest")
    return value


def stage_runtime(runtime: Path, candidate: str) -> dict:
    from .cosmos_droid_dependencies import stage_dependency_manifest
    from .cosmos_droid_source import stage_source
    selected = profile(candidate)
    snapshot = selected.snapshot_manifest_path()
    manifest = selected.verified_snapshot_manifest(snapshot)
    if sum(row["size_bytes"] for row in manifest["required_files"]) != CHECKPOINT_BYTES[candidate]:
        raise ValueError("cosmos_checkpoint_transfer_identity_invalid")
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / selected.SNAPSHOT_STAGED_NAME).write_bytes(snapshot.read_bytes())
    stage_source(runtime)
    stage_dependency_manifest(runtime)
    return runtime_contract(candidate)


def validate_runtime_contract(value: Mapping) -> dict:
    if not isinstance(value, Mapping) or value != runtime_contract(value.get("candidate_id")):
        raise ValueError("cosmos_sealed_runtime_identity_invalid")
    return dict(value)


def transfer_byte_budget(bundle: Mapping) -> tuple[int, int]:
    contract = validate_runtime_contract(bundle["cosmos_runtime"])
    source = bundle.get("runtime_source_packet") or {}
    external = source.get("packet_size_bytes", 0) if source.get("transport") == "content_addressed_external_layer.v1" else 0
    size = bundle.get("bundle_size_bytes")
    if type(size) is not int or size <= 0 or type(external) is not int or external < 0:
        raise ValueError("cosmos_transfer_binding_invalid")
    return (size + external + contract["checkpoint_download_bytes"] + contract["dependency_download_bytes"]
            + IMAGE_DOWNLOAD_ALLOWANCE_BYTES, MAXIMUM_OUTPUT_BYTES)


def provisioning_script() -> str:
    return '''#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$RUNTIME_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN || true
/isaac-sim/python.sh -m blueprint_pipeline.cosmos_droid_provisioning \\
  --runtime "$RUNTIME_DIR" --output "$OUT_DIR" --spec "$RUNTIME_DIR/adp009d_policy_execution_spec.json"
'''
