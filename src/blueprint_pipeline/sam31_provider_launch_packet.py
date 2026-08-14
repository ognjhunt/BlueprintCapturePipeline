"""Materialize immutable SAM 3.1 provider and GPU-canary request packets.

This module is deliberately mutation-free.  It reopens exact local evidence,
bundle, and request bytes and produces inputs for the existing SAM 3.1 paid
admission seam; it never probes or invokes a provider.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence
import zipfile

from .common import ensure_dir, write_json
from .decision_evidence_contracts import canonical_digest
from .sam31_gpu_admission import (
    CHECKPOINT_DIGEST,
    CHECKPOINT_FAMILY,
    CHECKPOINT_REPOSITORY_REVISION,
    LICENSE_TERMS_DIGEST,
    MAX_CANARY_FRAMES,
    MAX_CANARY_INPUT_BUNDLE_BYTES,
    MAX_RETRY_CAP,
    MAX_TTL_SECONDS,
    OFFICIAL_CODE_REVISION,
    OPERATION,
    REQUEST_SCHEMA_VERSION,
    SOURCE_PROFILES,
)
from .sam31_source_track_canary_worker import BUNDLE_RECEIPT_SCHEMA_VERSION
from .scene_placement.sam31_source_track_provider import (
    FRAME_INPUT_MODE,
    RUN_REQUEST_SCHEMA_VERSION,
    RUNTIME_API,
)
from .scene_placement.semantic_gaussian_lifting import canonical_json_digest
from .scene_placement.semantic_source_track_import import MASK_ENCODING


WORKER_STACK_SCHEMA_VERSION = "semantic_sam31_worker_stack_manifest.v1"
RUNTIME_IMAGE_BUILD_RECEIPT_SCHEMA_VERSION = (
    "semantic_sam31_runtime_image_build_receipt.v1"
)
LICENSE_AUTHORIZATION_SCHEMA_VERSION = "semantic_sam31_license_use_authorization.v1"
PRIVACY_AUTHORIZATION_SCHEMA_VERSION = "semantic_sam31_privacy_use_authorization.v1"
TRADE_CONTROLS_SCHEMA_VERSION = "semantic_sam31_trade_controls_review.v1"
EXECUTION_AUTHORIZATION_SCHEMA_VERSION = "semantic_sam31_execution_authorization.v1"

_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class Sam31ProviderLaunchPacketError(ValueError):
    """Stable fail-closed error for SAM profile/request materialization."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file(path: str | Path, *, code: str) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise Sam31ProviderLaunchPacketError(code)
    resolved = unresolved.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise Sam31ProviderLaunchPacketError(code)
    return resolved


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Sam31ProviderLaunchPacketError(code) from exc
    if not isinstance(value, dict):
        raise Sam31ProviderLaunchPacketError(code)
    return value


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _reopen_record(value: Any, *, code: str) -> Path:
    if not isinstance(value, Mapping):
        raise Sam31ProviderLaunchPacketError(code)
    path = _file(str(value.get("path") or ""), code=code)
    if path.stat().st_size != value.get("size_bytes") or _sha256(path) != value.get("sha256"):
        raise Sam31ProviderLaunchPacketError(code)
    return path


def _self_digested(value: Mapping[str, Any], *, field: str = "receipt_digest") -> bool:
    return value.get(field) == canonical_digest(dict(value), digest_field=field)


def _output(path: str | Path, value: Mapping[str, Any], *, code: str) -> None:
    destination = Path(path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise Sam31ProviderLaunchPacketError(code)
    ensure_dir(destination.parent)
    write_json(destination, dict(value))


def _finite(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and (float(value) > 0 if positive else float(value) >= 0)
    )


def materialize_sam31_worker_stack_manifest(
    *,
    source_commit_sha: str,
    runtime_image_identity: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal the exact code, image, checkpoint, and license stack."""

    if (
        _COMMIT.fullmatch(source_commit_sha) is None
        or _IMAGE.fullmatch(runtime_image_identity) is None
    ):
        raise Sam31ProviderLaunchPacketError("sam31_worker_stack_configuration_invalid")
    manifest: dict[str, Any] = {
        "schema_version": WORKER_STACK_SCHEMA_VERSION,
        "source_commit_sha": source_commit_sha,
        "runtime_image_identity": runtime_image_identity,
        "runtime_digest": runtime_image_identity.rpartition("@")[2],
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "checkpoint_repository_revision": CHECKPOINT_REPOSITORY_REVISION,
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "license_terms_digest": LICENSE_TERMS_DIGEST,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    _output(
        output_path,
        manifest,
        code="sam31_worker_stack_manifest_output_exists",
    )
    return manifest


def materialize_sam31_execution_authorization(
    *,
    source_commit_sha: str,
    runtime_image_identity: str,
    authorized_by: str,
    authorized_on: str,
    authority_reference: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind retained human execution authority to one exact code/image pair."""

    if (
        _COMMIT.fullmatch(source_commit_sha) is None
        or _IMAGE.fullmatch(runtime_image_identity) is None
        or not authorized_by.strip()
        or not authorized_on.strip()
        or not authority_reference.strip()
    ):
        raise Sam31ProviderLaunchPacketError(
            "sam31_execution_authorization_configuration_invalid"
        )
    authorization: dict[str, Any] = {
        "schema_version": EXECUTION_AUTHORIZATION_SCHEMA_VERSION,
        "status": "authorized",
        "source_commit_sha": source_commit_sha,
        "runtime_image_identity": runtime_image_identity,
        "external_execution_authorized": True,
        "network_access_during_inference_forbidden": True,
        "model_self_grading_forbidden": True,
        "metric_claim_upgrade_forbidden": True,
        "physics_claim_upgrade_forbidden": True,
        "physical_claim_upgrade_forbidden": True,
        "authorized_by": authorized_by.strip(),
        "authorized_on": authorized_on.strip(),
        "authority_reference": authority_reference.strip(),
        "authority_issued_by_agent": False,
        "receipt_digest": "",
    }
    authorization["receipt_digest"] = canonical_digest(
        authorization, digest_field="receipt_digest"
    )
    _output(
        output_path,
        authorization,
        code="sam31_execution_authorization_output_exists",
    )
    return authorization


def _authorization_sources(
    *,
    license_path: Path,
    privacy_path: Path,
    trade_path: Path,
    execution_path: Path,
    source_commit_sha: str,
    runtime_image_identity: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    files = {
        "license_use": license_path,
        "privacy_use": privacy_path,
        "trade_controls": trade_path,
        "execution": execution_path,
    }
    values = {
        role: _read(path, code=f"sam31_{role}_authorization_invalid")
        for role, path in files.items()
    }
    license_value = values["license_use"]
    privacy_value = values["privacy_use"]
    trade_value = values["trade_controls"]
    execution_value = values["execution"]
    valid = (
        all(
            isinstance(value.get("authorized_by"), str)
            and bool(value["authorized_by"].strip())
            and isinstance(value.get("authorized_on"), str)
            and bool(value["authorized_on"].strip())
            and isinstance(value.get("authority_reference"), str)
            and bool(value["authority_reference"].strip())
            and value.get("authority_issued_by_agent") is False
            for value in values.values()
        )
        and
        license_value.get("schema_version") == LICENSE_AUTHORIZATION_SCHEMA_VERSION
        and license_value.get("status") == "accepted"
        and license_value.get("checkpoint_digest") == CHECKPOINT_DIGEST
        and license_value.get("license_terms_digest") == LICENSE_TERMS_DIGEST
        and license_value.get("checkpoint_access_authorized") is True
        and license_value.get("commercial_evidence_use_authorized") is True
        and license_value.get("customer_data_training_allowed") is False
        and license_value.get("allowed_evidence_uses") == ["semantic_analysis"]
        and privacy_value.get("schema_version") == PRIVACY_AUTHORIZATION_SCHEMA_VERSION
        and privacy_value.get("status") == "accepted"
        and privacy_value.get("rights_cleared_for_external_processing") is True
        and privacy_value.get("privacy_safe_for_external_processing") is True
        and privacy_value.get("customer_data_training_allowed") is False
        and trade_value.get("schema_version") == TRADE_CONTROLS_SCHEMA_VERSION
        and trade_value.get("status") == "reviewed"
        and trade_value.get("checkpoint_digest") == CHECKPOINT_DIGEST
        and trade_value.get("trade_controls_reviewed") is True
        and execution_value.get("schema_version")
        == EXECUTION_AUTHORIZATION_SCHEMA_VERSION
        and execution_value.get("status") == "authorized"
        and execution_value.get("source_commit_sha") == source_commit_sha
        and execution_value.get("runtime_image_identity") == runtime_image_identity
        and execution_value.get("external_execution_authorized") is True
        and execution_value.get("network_access_during_inference_forbidden") is True
        and execution_value.get("model_self_grading_forbidden") is True
        and execution_value.get("metric_claim_upgrade_forbidden") is True
        and execution_value.get("physics_claim_upgrade_forbidden") is True
        and execution_value.get("physical_claim_upgrade_forbidden") is True
        and all(_self_digested(value) for value in values.values())
    )
    if not valid:
        raise Sam31ProviderLaunchPacketError("sam31_authorization_source_invalid")
    records = {
        role: {**_record(files[role]), "receipt_digest": values[role]["receipt_digest"]}
        for role in files
    }
    return values, records


def materialize_sam31_provider_profile(
    *,
    worker_stack_manifest_path: str | Path,
    runtime_image_build_receipt_path: str | Path,
    license_use_authorization_path: str | Path,
    privacy_use_authorization_path: str | Path,
    trade_controls_review_path: str | Path,
    execution_authorization_path: str | Path,
    source_commit_sha: str,
    runtime_image_identity: str,
    method_version: str,
    output_probability_threshold: float,
    max_num_objects: int,
    multiplex_count: int,
    use_fa3: bool,
    compile_model: bool,
    warm_up: bool,
    async_loading_frames: bool,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind a provider profile to exact code, image, and authorization bytes."""

    paths = {
        "worker_stack": _file(
            worker_stack_manifest_path, code="sam31_worker_stack_manifest_missing"
        ),
        "runtime_image_build": _file(
            runtime_image_build_receipt_path,
            code="sam31_runtime_image_build_receipt_missing",
        ),
        "license": _file(
            license_use_authorization_path, code="sam31_license_authorization_missing"
        ),
        "privacy": _file(
            privacy_use_authorization_path, code="sam31_privacy_authorization_missing"
        ),
        "trade": _file(
            trade_controls_review_path, code="sam31_trade_controls_review_missing"
        ),
        "execution": _file(
            execution_authorization_path, code="sam31_execution_authorization_missing"
        ),
    }
    stack = _read(paths["worker_stack"], code="sam31_worker_stack_manifest_invalid")
    image_build = _read(
        paths["runtime_image_build"], code="sam31_runtime_image_build_receipt_invalid"
    )
    normalized_runtime_digest = runtime_image_identity.rpartition("@")[2]
    if (
        _COMMIT.fullmatch(source_commit_sha) is None
        or _IMAGE.fullmatch(runtime_image_identity) is None
        or stack.get("schema_version") != WORKER_STACK_SCHEMA_VERSION
        or stack.get("source_commit_sha") != source_commit_sha
        or stack.get("runtime_image_identity") != runtime_image_identity
        or stack.get("official_code_revision") != OFFICIAL_CODE_REVISION
        or stack.get("checkpoint_repository_revision") != CHECKPOINT_REPOSITORY_REVISION
        or stack.get("checkpoint_digest") != CHECKPOINT_DIGEST
        or stack.get("license_terms_digest") != LICENSE_TERMS_DIGEST
        or _DIGEST.fullmatch(str(stack.get("runtime_digest") or "")) is None
        or stack.get("runtime_digest") != normalized_runtime_digest
        or not _self_digested(stack, field="manifest_digest")
        or image_build.get("schema_version")
        != RUNTIME_IMAGE_BUILD_RECEIPT_SCHEMA_VERSION
        or image_build.get("status") != "published"
        or image_build.get("source_commit_sha") != source_commit_sha
        or image_build.get("runtime_image_identity") != runtime_image_identity
        or image_build.get("runtime_digest") != stack.get("runtime_digest")
        or image_build.get("official_code_revision") != OFFICIAL_CODE_REVISION
        or image_build.get("registry_api_digest_verified") is not True
        or _DIGEST.fullmatch(str(image_build.get("dockerfile_sha256") or "")) is None
        or _DIGEST.fullmatch(str(image_build.get("source_tree_digest") or "")) is None
        or _DIGEST.fullmatch(str(image_build.get("build_provenance_digest") or "")) is None
        or not _self_digested(image_build)
        or not method_version.strip()
        or not _finite(output_probability_threshold, positive=True)
        or float(output_probability_threshold) > 1
        or any(
            isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10_000
            for value in (max_num_objects, multiplex_count)
        )
        or any(
            not isinstance(value, bool)
            for value in (use_fa3, compile_model, warm_up, async_loading_frames)
        )
        or (warm_up and not compile_model)
    ):
        raise Sam31ProviderLaunchPacketError("sam31_provider_profile_configuration_invalid")
    _, authorization_records = _authorization_sources(
        license_path=paths["license"],
        privacy_path=paths["privacy"],
        trade_path=paths["trade"],
        execution_path=paths["execution"],
        source_commit_sha=source_commit_sha,
        runtime_image_identity=runtime_image_identity,
    )
    profile: dict[str, Any] = {
        "method_id": "meta.sam3.1.object_multiplex",
        "method_version": method_version.strip(),
        "runtime_api": RUNTIME_API,
        "checkpoint_family": CHECKPOINT_FAMILY,
        "frame_input_mode": FRAME_INPUT_MODE,
        "mask_encoding": MASK_ENCODING,
        "execution_mode": "local",
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "runtime_digest": stack["runtime_digest"],
        "runtime_image_identity": runtime_image_identity,
        "model_digest": CHECKPOINT_DIGEST,
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "checkpoint_repository_revision": CHECKPOINT_REPOSITORY_REVISION,
        "license_terms_digest": LICENSE_TERMS_DIGEST,
        "license_use_authorization_digest": authorization_records["license_use"]["sha256"],
        "privacy_use_authorization_digest": authorization_records["privacy_use"]["sha256"],
        "trade_controls_review_digest": authorization_records["trade_controls"]["sha256"],
        "execution_authorization_digest": authorization_records["execution"]["sha256"],
        "checkpoint_access_authorized": True,
        "commercial_evidence_use_authorized": True,
        "persistent_track_ids": True,
        "model_self_grading_forbidden": True,
        "source_frames_are_hash_verified": True,
        "network_access_during_inference_forbidden": True,
        "customer_data_training_allowed": False,
        "output_probability_threshold": float(output_probability_threshold),
        "max_num_objects": max_num_objects,
        "multiplex_count": multiplex_count,
        "use_fa3": use_fa3,
        "compile": compile_model,
        "warm_up": warm_up,
        "async_loading_frames": async_loading_frames,
        "worker_stack_manifest": {
            **_record(paths["worker_stack"]),
            "manifest_digest": stack["manifest_digest"],
        },
        "runtime_image_build_receipt": {
            **_record(paths["runtime_image_build"]),
            "receipt_digest": image_build["receipt_digest"],
        },
        "authorization_sources": authorization_records,
    }
    profile.update(
        {
            "source_commit_sha": source_commit_sha,
            "provider_mutations_performed": 0,
            "paid_execution_started": False,
        }
    )
    profile["profile_digest"] = canonical_json_digest(
        {key: value for key, value in profile.items() if key != "profile_digest"}
    )
    _output(output_path, profile, code="sam31_provider_profile_output_exists")
    return profile


def materialize_sam31_gpu_canary_request(
    *,
    provider_profile_path: str | Path,
    source_track_run_request_path: str | Path,
    input_bundle_path: str | Path,
    input_bundle_receipt_path: str | Path,
    source_profile: str,
    source_commit_sha: str,
    expected_camera_count: int,
    expected_frame_count: int,
    max_spend_usd: float,
    hard_ttl_seconds: int,
    retry_cap: int,
    authority_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build the exact request consumed by ``sam31_gpu_admission``."""

    profile_path = _file(
        provider_profile_path, code="sam31_provider_profile_missing"
    )
    run_request_path = _file(
        source_track_run_request_path, code="sam31_source_track_run_request_missing"
    )
    bundle_path = _file(input_bundle_path, code="sam31_input_bundle_missing")
    receipt_path = _file(input_bundle_receipt_path, code="sam31_input_bundle_receipt_missing")
    profile = _read(profile_path, code="sam31_provider_profile_invalid")
    run_request = _read(run_request_path, code="sam31_source_track_run_request_invalid")
    receipt = _read(receipt_path, code="sam31_input_bundle_receipt_invalid")
    frames = run_request.get("frame_registry")
    artifacts = run_request.get("frame_artifacts")
    bindings = run_request.get("bindings")
    if (
        profile.get("profile_digest")
        != canonical_json_digest(
            {key: value for key, value in profile.items() if key != "profile_digest"}
        )
        or profile.get("provider_mutations_performed") != 0
        or profile.get("paid_execution_started") is not False
        or run_request.get("schema_version") != RUN_REQUEST_SCHEMA_VERSION
        or run_request.get("provider_profile") != profile
        or run_request.get("allowed_evidence_uses") != ["semantic_analysis"]
        or not isinstance(frames, list)
        or not isinstance(artifacts, list)
        or not isinstance(bindings, Mapping)
        or len(frames) != len(artifacts)
        or len(frames) != expected_frame_count
        or isinstance(expected_camera_count, bool)
        or not isinstance(expected_camera_count, int)
        or expected_camera_count != expected_frame_count
        or isinstance(expected_frame_count, bool)
        or not 1 <= expected_frame_count <= MAX_CANARY_FRAMES
        or any(_DIGEST.fullmatch(str(bindings.get(field) or "")) is None for field in (
            "capture_digest",
            "retained_video_digest",
            "camera_solution_digest",
            "frame_registry_digest",
        ))
        or canonical_json_digest(frames) != bindings.get("frame_registry_digest")
        or receipt.get("schema_version") != BUNDLE_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "completed"
        or not _self_digested(receipt)
        or (receipt.get("bundle") or {}).get("sha256") != _sha256(bundle_path)
        or (receipt.get("bundle") or {}).get("size_bytes") != bundle_path.stat().st_size
        or receipt.get("frame_count") != expected_frame_count
        or not 1 <= bundle_path.stat().st_size <= MAX_CANARY_INPUT_BUNDLE_BYTES
        or source_profile not in SOURCE_PROFILES
        or profile.get("source_commit_sha") != source_commit_sha
        or _COMMIT.fullmatch(source_commit_sha) is None
        or not _finite(max_spend_usd, positive=True)
        or isinstance(hard_ttl_seconds, bool)
        or not isinstance(hard_ttl_seconds, int)
        or not 1 <= hard_ttl_seconds <= MAX_TTL_SECONDS
        or isinstance(retry_cap, bool)
        or retry_cap != MAX_RETRY_CAP
        or not isinstance(authority_id, str)
        or not authority_id.strip()
    ):
        raise Sam31ProviderLaunchPacketError("sam31_gpu_canary_request_configuration_invalid")
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            portable_request = json.loads(archive.read("request.json"))
    except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise Sam31ProviderLaunchPacketError("sam31_input_bundle_invalid") from exc
    if (
        manifest.get("source_track_run_request_digest")
        != receipt.get("source_track_run_request_digest")
        or manifest.get("frame_count") != expected_frame_count
        or canonical_json_digest(portable_request)
        != receipt.get("source_track_run_request_digest")
        or portable_request.get("provider_profile") != profile
    ):
        raise Sam31ProviderLaunchPacketError("sam31_input_bundle_binding_invalid")
    stack_record = profile.get("worker_stack_manifest")
    image_build_record = profile.get("runtime_image_build_receipt")
    authorization_records = profile.get("authorization_sources")
    if (
        not isinstance(stack_record, Mapping)
        or not isinstance(image_build_record, Mapping)
        or not isinstance(authorization_records, Mapping)
    ):
        raise Sam31ProviderLaunchPacketError("sam31_profile_source_bindings_missing")
    stack_path = _reopen_record(
        stack_record, code="sam31_worker_stack_manifest_bytes_changed"
    )
    image_build_path = _reopen_record(
        image_build_record, code="sam31_runtime_image_build_receipt_bytes_changed"
    )
    authorization_paths = {
        role: _reopen_record(
            authorization_records[role],
            code=f"sam31_{role}_authorization_bytes_changed",
        )
        for role in ("license_use", "privacy_use", "trade_controls", "execution")
    }
    stack = _read(stack_path, code="sam31_worker_stack_manifest_invalid")
    image_build = _read(
        image_build_path, code="sam31_runtime_image_build_receipt_invalid"
    )
    if (
        stack.get("schema_version") != WORKER_STACK_SCHEMA_VERSION
        or stack.get("source_commit_sha") != source_commit_sha
        or stack.get("runtime_image_identity") != profile.get("runtime_image_identity")
        or stack.get("runtime_digest") != profile.get("runtime_digest")
        or stack.get("runtime_digest")
        != str(profile.get("runtime_image_identity") or "").rpartition("@")[2]
        or stack.get("official_code_revision") != OFFICIAL_CODE_REVISION
        or stack.get("checkpoint_repository_revision") != CHECKPOINT_REPOSITORY_REVISION
        or stack.get("checkpoint_digest") != CHECKPOINT_DIGEST
        or stack.get("license_terms_digest") != LICENSE_TERMS_DIGEST
        or not _self_digested(stack, field="manifest_digest")
        or stack.get("manifest_digest") != stack_record.get("manifest_digest")
        or image_build.get("schema_version")
        != RUNTIME_IMAGE_BUILD_RECEIPT_SCHEMA_VERSION
        or image_build.get("status") != "published"
        or image_build.get("source_commit_sha") != source_commit_sha
        or image_build.get("runtime_image_identity") != profile.get("runtime_image_identity")
        or image_build.get("runtime_digest") != profile.get("runtime_digest")
        or image_build.get("official_code_revision") != OFFICIAL_CODE_REVISION
        or image_build.get("registry_api_digest_verified") is not True
        or _DIGEST.fullmatch(str(image_build.get("dockerfile_sha256") or "")) is None
        or _DIGEST.fullmatch(str(image_build.get("source_tree_digest") or "")) is None
        or _DIGEST.fullmatch(str(image_build.get("build_provenance_digest") or "")) is None
        or not _self_digested(image_build)
        or image_build.get("receipt_digest") != image_build_record.get("receipt_digest")
    ):
        raise Sam31ProviderLaunchPacketError("sam31_worker_stack_manifest_invalid")
    _, reopened_authorization_records = _authorization_sources(
        license_path=authorization_paths["license_use"],
        privacy_path=authorization_paths["privacy_use"],
        trade_path=authorization_paths["trade_controls"],
        execution_path=authorization_paths["execution"],
        source_commit_sha=source_commit_sha,
        runtime_image_identity=str(profile.get("runtime_image_identity") or ""),
    )
    if (
        reopened_authorization_records != dict(authorization_records)
        or profile.get("license_use_authorization_digest")
        != reopened_authorization_records["license_use"]["sha256"]
        or profile.get("privacy_use_authorization_digest")
        != reopened_authorization_records["privacy_use"]["sha256"]
        or profile.get("trade_controls_review_digest")
        != reopened_authorization_records["trade_controls"]["sha256"]
        or profile.get("execution_authorization_digest")
        != reopened_authorization_records["execution"]["sha256"]
    ):
        raise Sam31ProviderLaunchPacketError("sam31_authorization_source_binding_invalid")
    source_records = {
        "provider_profile": _record(profile_path),
        "source_track_run_request": _record(run_request_path),
        "input_bundle": _record(bundle_path),
        "input_bundle_receipt": _record(receipt_path),
        "worker_stack_manifest": dict(stack_record),
        "runtime_image_build_receipt": dict(image_build_record),
        "authorization_sources": dict(authorization_records),
    }
    request: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "operation": OPERATION,
        "source_profile": source_profile,
        "source_commit_sha": source_commit_sha,
        "worker_image_digest": profile["runtime_image_identity"],
        "worker_stack_manifest_digest": stack_record["manifest_digest"],
        "input_bundle_digest": _sha256(bundle_path),
        "input_bundle_size_bytes": bundle_path.stat().st_size,
        "source_track_run_request_digest": receipt["source_track_run_request_digest"],
        "capture_digest": bindings["capture_digest"],
        "retained_video_digest": bindings["retained_video_digest"],
        "camera_solution_digest": bindings["camera_solution_digest"],
        "frame_registry_digest": bindings["frame_registry_digest"],
        "camera_count": expected_camera_count,
        "frame_count": expected_frame_count,
        "checkpoint_family": CHECKPOINT_FAMILY,
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "checkpoint_repository_revision": CHECKPOINT_REPOSITORY_REVISION,
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "license_terms_digest": LICENSE_TERMS_DIGEST,
        "license_use_authorization_digest": profile["license_use_authorization_digest"],
        "privacy_use_authorization_digest": profile["privacy_use_authorization_digest"],
        "trade_controls_review_digest": profile["trade_controls_review_digest"],
        "execution_authorization_digest": profile["execution_authorization_digest"],
        "checkpoint_access_authorized": True,
        "commercial_evidence_use_authorized": True,
        "rights_cleared_for_external_processing": True,
        "privacy_safe_for_external_processing": True,
        "trade_controls_reviewed": True,
        "model_self_grading_forbidden": True,
        "metric_claim_upgrade_forbidden": True,
        "physics_claim_upgrade_forbidden": True,
        "physical_claim_upgrade_forbidden": True,
        "network_access_during_inference_forbidden": True,
        "customer_data_training_allowed": False,
        "allowed_evidence_uses": ["semantic_analysis"],
        "max_spend_usd": float(max_spend_usd),
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": retry_cap,
        "authority_id": authority_id.strip(),
        "source_records": source_records,
        "proof_effect": "none",
        "comparative_policy_ranking_verdict": "thesis_not_supported",
        "provider_mutations_performed": 0,
        "paid_execution_started": False,
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    _output(output_path, request, code="sam31_gpu_canary_request_output_exists")
    return request


def _bool_argument(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main(argv: Sequence[str] | None = None) -> int:
    """Expose both deterministic materializers without a Python session."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker_stack = commands.add_parser("worker-stack")
    worker_stack.add_argument("--source-commit", required=True)
    worker_stack.add_argument("--runtime-image-identity", required=True)
    worker_stack.add_argument("--output", required=True)
    execution = commands.add_parser("execution-authorization")
    execution.add_argument("--source-commit", required=True)
    execution.add_argument("--runtime-image-identity", required=True)
    execution.add_argument("--authorized-by", required=True)
    execution.add_argument("--authorized-on", required=True)
    execution.add_argument("--authority-reference", required=True)
    execution.add_argument("--output", required=True)
    profile = commands.add_parser("profile")
    profile.add_argument("--worker-stack-manifest", required=True)
    profile.add_argument("--runtime-image-build-receipt", required=True)
    profile.add_argument("--license-use-authorization", required=True)
    profile.add_argument("--privacy-use-authorization", required=True)
    profile.add_argument("--trade-controls-review", required=True)
    profile.add_argument("--execution-authorization", required=True)
    profile.add_argument("--source-commit", required=True)
    profile.add_argument("--runtime-image-identity", required=True)
    profile.add_argument("--method-version", required=True)
    profile.add_argument("--output-probability-threshold", type=float, required=True)
    profile.add_argument("--max-num-objects", type=int, required=True)
    profile.add_argument("--multiplex-count", type=int, required=True)
    profile.add_argument("--use-fa3", type=_bool_argument, required=True)
    profile.add_argument("--compile-model", type=_bool_argument, required=True)
    profile.add_argument("--warm-up", type=_bool_argument, required=True)
    profile.add_argument("--async-loading-frames", type=_bool_argument, required=True)
    profile.add_argument("--output", required=True)
    gpu = commands.add_parser("gpu-request")
    gpu.add_argument("--provider-profile", required=True)
    gpu.add_argument("--source-track-run-request", required=True)
    gpu.add_argument("--input-bundle", required=True)
    gpu.add_argument("--input-bundle-receipt", required=True)
    gpu.add_argument("--source-profile", required=True)
    gpu.add_argument("--source-commit", required=True)
    gpu.add_argument("--expected-camera-count", type=int, required=True)
    gpu.add_argument("--expected-frame-count", type=int, required=True)
    gpu.add_argument("--max-spend-usd", type=float, required=True)
    gpu.add_argument("--hard-ttl-seconds", type=int, required=True)
    gpu.add_argument("--retry-cap", type=int, required=True)
    gpu.add_argument("--authority-id", required=True)
    gpu.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "worker-stack":
        materialize_sam31_worker_stack_manifest(
            source_commit_sha=args.source_commit,
            runtime_image_identity=args.runtime_image_identity,
            output_path=args.output,
        )
    elif args.command == "execution-authorization":
        materialize_sam31_execution_authorization(
            source_commit_sha=args.source_commit,
            runtime_image_identity=args.runtime_image_identity,
            authorized_by=args.authorized_by,
            authorized_on=args.authorized_on,
            authority_reference=args.authority_reference,
            output_path=args.output,
        )
    elif args.command == "profile":
        materialize_sam31_provider_profile(
            worker_stack_manifest_path=args.worker_stack_manifest,
            runtime_image_build_receipt_path=args.runtime_image_build_receipt,
            license_use_authorization_path=args.license_use_authorization,
            privacy_use_authorization_path=args.privacy_use_authorization,
            trade_controls_review_path=args.trade_controls_review,
            execution_authorization_path=args.execution_authorization,
            source_commit_sha=args.source_commit,
            runtime_image_identity=args.runtime_image_identity,
            method_version=args.method_version,
            output_probability_threshold=args.output_probability_threshold,
            max_num_objects=args.max_num_objects,
            multiplex_count=args.multiplex_count,
            use_fa3=args.use_fa3,
            compile_model=args.compile_model,
            warm_up=args.warm_up,
            async_loading_frames=args.async_loading_frames,
            output_path=args.output,
        )
    else:
        materialize_sam31_gpu_canary_request(
            provider_profile_path=args.provider_profile,
            source_track_run_request_path=args.source_track_run_request,
            input_bundle_path=args.input_bundle,
            input_bundle_receipt_path=args.input_bundle_receipt,
            source_profile=args.source_profile,
            source_commit_sha=args.source_commit,
            expected_camera_count=args.expected_camera_count,
            expected_frame_count=args.expected_frame_count,
            max_spend_usd=args.max_spend_usd,
            hard_ttl_seconds=args.hard_ttl_seconds,
            retry_cap=args.retry_cap,
            authority_id=args.authority_id,
            output_path=args.output,
        )
    return 0


__all__ = [
    "EXECUTION_AUTHORIZATION_SCHEMA_VERSION",
    "LICENSE_AUTHORIZATION_SCHEMA_VERSION",
    "PRIVACY_AUTHORIZATION_SCHEMA_VERSION",
    "RUNTIME_IMAGE_BUILD_RECEIPT_SCHEMA_VERSION",
    "Sam31ProviderLaunchPacketError",
    "TRADE_CONTROLS_SCHEMA_VERSION",
    "WORKER_STACK_SCHEMA_VERSION",
    "materialize_sam31_gpu_canary_request",
    "materialize_sam31_execution_authorization",
    "materialize_sam31_provider_profile",
    "materialize_sam31_worker_stack_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
