"""Stable evidence and claim contracts for the retained G1 qualification fixture."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
from typing import Any, Mapping


RELEASE_BINDING_SCHEMA_VERSION = "single_g1_kitchen_qualification_release_binding.v1"
LAUNCH_REBIND_SCHEMA_VERSION = "single_g1_kitchen_qualification_launch_rebind.v1"
RUNTIME_ATTEMPT_OVERLAY_BASE_SCHEMA_VERSION = (
    "single_g1_kitchen_qualification_runtime_attempt_overlay_base.v1"
)
_RUNTIME_ATTEMPT_OVERLAY_TOP_LEVEL_FIELDS = (
    "prepared_launch_nonce",
    "allocation_launch_session_id",
    "launch_nonce",
    "qualification_attempt_bound",
    "qualification_attempt_sequence",
    "qualification_attempt_nonce",
    "qualification_attempt_nonce_sha256",
)
_LAUNCH_ONLY_REBIND_FIELDS = frozenset(
    {
        "derived_launch_plan_sha256",
        "derived_attempt_manifest_sha256",
        "runtime_attempt_manifest_rebound",
        "raw_source_bundle_modified",
    }
)
EMPTY_PATCH_SHA256 = hashlib.sha256(b"").hexdigest()
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_DIGEST_REF_RE = re.compile(r"[^\s@]+@sha256:([0-9a-f]{64})")


def valid_source_commit(value: object) -> bool:
    return bool(_COMMIT_RE.fullmatch(str(value or "")))


def valid_image_binding(image_ref: object, image_digest: object) -> bool:
    match = _DIGEST_REF_RE.fullmatch(str(image_ref or ""))
    return bool(match and image_digest == f"sha256:{match.group(1)}")


def release_binding_record_blockers(value: object) -> list[str]:
    binding = dict(value) if isinstance(value, Mapping) else {}
    blockers: list[str] = []
    if not binding:
        return ["missing"]
    if binding.get("schema_version") != RELEASE_BINDING_SCHEMA_VERSION:
        blockers.append("schema")
    if not valid_image_binding(binding.get("image_ref"), binding.get("image_digest")):
        blockers.append("image")
    if not valid_source_commit(binding.get("source_commit")):
        blockers.append("source_commit")
    if binding.get("source_patch_sha256") != EMPTY_PATCH_SHA256:
        blockers.append("source_patch")
    if binding.get("runnable_platform") != "linux/amd64":
        blockers.append("platform")
    if not str(binding.get("required_cuda_version") or ""):
        blockers.append("cuda_version")
    if not str(binding.get("required_cuda_version_source") or "").startswith(
        "image_config_env:"
    ):
        blockers.append("cuda_source")
    model_assets_mode = binding.get("model_assets_mode")
    if model_assets_mode is None and binding.get("models_externalized") is True:
        # Retained pre-mode sessions remain readable for status and teardown.
        model_assets_mode = "external"
    externalized = binding.get("models_externalized") is True
    embedded = binding.get("models_embedded_in_foundation") is True
    if model_assets_mode not in {"external", "embedded"}:
        blockers.append("model_assets_mode")
    if model_assets_mode == "external" and not externalized:
        blockers.append("models_externalized")
    if model_assets_mode == "embedded" and not embedded:
        blockers.append("models_embedded_in_foundation")
    if externalized == embedded:
        blockers.append("model_assets_mode_exclusive")
    if binding.get("release_evidence_status") != "completed":
        blockers.append("release_status")
    if binding.get("thin_release_contract_status") != "passed":
        blockers.append("thin_release_status")
    return sorted(set(blockers))


def embedded_launch_rebind(value: object) -> dict[str, Any]:
    binding = dict(value) if isinstance(value, Mapping) else {}
    return {
        key: item for key, item in binding.items() if key not in _LAUNCH_ONLY_REBIND_FIELDS
    }


def collected_attempt_release_blocker(
    attempt: Mapping[str, Any], session: Mapping[str, Any]
) -> str | None:
    release = dict(session.get("release_binding") or {})
    expected = {
        "source_commit": session.get("source_commit"),
        "source_dirty_patch_sha256": release.get("source_patch_sha256"),
        "image_digest": session.get("image_digest"),
        "qualification_release_rebind": embedded_launch_rebind(
            session.get("qualification_launch_rebind")
        ),
    }
    mismatches = sorted(
        key for key, value in expected.items() if not value or attempt.get(key) != value
    )
    return (
        "qualification_attempt_input_release_binding_mismatch:" + ",".join(mismatches)
        if mismatches
        else None
    )


def collected_attempt_derived_artifact_blockers(
    attempt_bytes: bytes,
    attempt: Mapping[str, Any],
    launch_rebind: Mapping[str, Any],
) -> list[str]:
    """Verify retained bytes and the immutable base beneath the runtime overlay."""

    blockers: list[str] = []
    canonical_attempt_bytes = (
        json.dumps(dict(attempt), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        parsed_attempt = json.loads(attempt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        parsed_attempt = None
    if parsed_attempt != dict(attempt) or attempt_bytes != canonical_attempt_bytes:
        blockers.append("qualification_collected_attempt_manifest_digest_mismatch")
    try:
        base_attempt = _restore_runtime_attempt_overlay_base(attempt)
    except ValueError:
        blockers.append("qualification_collected_attempt_runtime_overlay_base_invalid")
    else:
        base_attempt_bytes = (
            json.dumps(base_attempt, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        if hashlib.sha256(base_attempt_bytes).hexdigest() != launch_rebind.get(
            "derived_attempt_manifest_sha256"
        ):
            blockers.append("qualification_collected_attempt_manifest_digest_mismatch")
    plan_value = attempt.get("qualification_derived_launch_plan")
    plan = dict(plan_value) if isinstance(plan_value, Mapping) else {}
    plan_digest = _canonical_sha256(plan) if plan else ""
    if (
        not plan
        or attempt.get("qualification_derived_launch_plan_sha256") != plan_digest
        or launch_rebind.get("derived_launch_plan_sha256") != plan_digest
    ):
        blockers.append("qualification_collected_launch_plan_digest_mismatch")
    return sorted(set(blockers))


def capture_runtime_attempt_overlay_base(attempt: Mapping[str, Any]) -> dict[str, Any]:
    """Snapshot only fields the worker may replace after provider allocation."""

    artifacts_value = attempt.get("artifacts")
    artifacts = dict(artifacts_value) if isinstance(artifacts_value, Mapping) else {}
    return {
        "schema_version": RUNTIME_ATTEMPT_OVERLAY_BASE_SCHEMA_VERSION,
        "top_level_fields": {
            field: _field_snapshot(attempt, field)
            for field in _RUNTIME_ATTEMPT_OVERLAY_TOP_LEVEL_FIELDS
        },
        "task_success_contract_artifact": _field_snapshot(
            artifacts, "task_success_contract"
        ),
    }


def _field_snapshot(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    present = field in value
    return {
        "present": present,
        "value": copy.deepcopy(value.get(field)) if present else None,
    }


def _restore_snapshot_field(
    target: dict[str, Any], field: str, snapshot_value: object
) -> None:
    snapshot = dict(snapshot_value) if isinstance(snapshot_value, Mapping) else {}
    if set(snapshot) != {"present", "value"} or not isinstance(
        snapshot.get("present"), bool
    ):
        raise ValueError("qualification_runtime_attempt_overlay_snapshot_invalid")
    if snapshot["present"]:
        target[field] = copy.deepcopy(snapshot.get("value"))
    else:
        if snapshot.get("value") is not None:
            raise ValueError("qualification_runtime_attempt_overlay_snapshot_invalid")
        target.pop(field, None)


def _restore_runtime_attempt_overlay_base(
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    base = copy.deepcopy(dict(attempt))
    overlay_value = base.get("qualification_runtime_attempt_overlay_base")
    overlay = dict(overlay_value) if isinstance(overlay_value, Mapping) else {}
    if overlay.get("schema_version") != RUNTIME_ATTEMPT_OVERLAY_BASE_SCHEMA_VERSION:
        raise ValueError("qualification_runtime_attempt_overlay_base_schema_invalid")
    top_value = overlay.get("top_level_fields")
    top = dict(top_value) if isinstance(top_value, Mapping) else {}
    if set(top) != set(_RUNTIME_ATTEMPT_OVERLAY_TOP_LEVEL_FIELDS):
        raise ValueError("qualification_runtime_attempt_overlay_base_fields_invalid")
    for field in _RUNTIME_ATTEMPT_OVERLAY_TOP_LEVEL_FIELDS:
        _restore_snapshot_field(base, field, top[field])

    artifacts_value = base.get("artifacts")
    artifacts = dict(artifacts_value) if isinstance(artifacts_value, Mapping) else {}
    _restore_snapshot_field(
        artifacts,
        "task_success_contract",
        overlay.get("task_success_contract_artifact"),
    )
    if artifacts or isinstance(artifacts_value, Mapping):
        base["artifacts"] = artifacts
    else:
        base.pop("artifacts", None)

    launch_snapshot = dict(top["launch_nonce"])
    prepared_launch_nonce = attempt.get("prepared_launch_nonce")
    if (
        prepared_launch_nonce is not None
        and prepared_launch_nonce != launch_snapshot.get("value")
    ):
        raise ValueError("qualification_runtime_attempt_prepared_nonce_mismatch")
    source_contract_snapshot = dict(overlay["task_success_contract_artifact"])
    source_contract_value = source_contract_snapshot.get("value")
    source_contract = (
        dict(source_contract_value) if isinstance(source_contract_value, Mapping) else {}
    )
    current_artifacts_value = attempt.get("artifacts")
    current_artifacts = (
        dict(current_artifacts_value)
        if isinstance(current_artifacts_value, Mapping)
        else {}
    )
    current_contract_value = current_artifacts.get("task_success_contract")
    current_contract = (
        dict(current_contract_value)
        if isinstance(current_contract_value, Mapping)
        else {}
    )
    if current_contract.get("sha256") != attempt.get(
        "qualification_resolved_task_success_contract_sha256"
    ):
        raise ValueError("qualification_runtime_attempt_resolved_task_contract_mismatch")
    if current_contract != source_contract:
        source_contract_sha256 = attempt.get(
            "qualification_source_task_success_contract_sha256"
        ) or source_contract.get("sha256")
        if (
            not source_contract_sha256
            or current_contract.get("derived_from_sha256") != source_contract_sha256
        ):
            raise ValueError(
                "qualification_runtime_attempt_task_contract_lineage_mismatch"
            )
    return base


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_latest_attempt_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable dispatch binding used to validate one collection."""

    value = manifest.get("latest_attempt")
    latest = dict(value) if isinstance(value, Mapping) else {}
    try:
        sequence = int(latest.get("attempt_sequence"))
        overlay_revision = int(latest.get("overlay_revision"))
    except (TypeError, ValueError) as exc:
        raise ValueError("qualification_latest_attempt_binding_missing") from exc
    attempt_slug = f"attempt_{sequence:04d}"
    launch_session_id = str(manifest.get("launch_session_id") or "")
    attempt_nonce = f"{launch_session_id}:{attempt_slug}"
    expected = {
        "schema_version": "single_g1_kitchen_qualification_attempt_binding.v1",
        "attempt_sequence": sequence,
        "attempt_slug": attempt_slug,
        "attempt_nonce": attempt_nonce,
        "attempt_nonce_sha256": hashlib.sha256(attempt_nonce.encode("utf-8")).hexdigest(),
        "launch_session_id": launch_session_id,
        "episode_bootstrap_sha256": str(latest.get("episode_bootstrap_sha256") or ""),
        "bundle_sha256": str(manifest.get("bundle_sha256") or ""),
        "overlay_revision": overlay_revision,
    }
    attempt_rebind_value = latest.get("qualification_launch_rebind")
    attempt_rebind = (
        dict(attempt_rebind_value)
        if isinstance(attempt_rebind_value, Mapping)
        else {}
    )
    mismatches = [
        key
        for key, expected_value in expected.items()
        if key != "schema_version" and latest.get(key) != expected_value
    ]
    if sequence < 1:
        mismatches.append("attempt_sequence")
    if re.fullmatch(r"[0-9a-f]{64}", expected["episode_bootstrap_sha256"]) is None:
        mismatches.append("episode_bootstrap_sha256")
    release = dict(manifest.get("release_binding") or {})
    expected_rebind = {
        "schema_version": LAUNCH_REBIND_SCHEMA_VERSION,
        "status": "bound",
        "source_bundle_sha256": manifest.get("bundle_sha256"),
        "release_image_ref": manifest.get("image_ref"),
        "release_image_digest": manifest.get("image_digest"),
        "release_source_commit": manifest.get("source_commit"),
        "release_source_patch_sha256": release.get("source_patch_sha256"),
        "source_bundle_preserved": True,
        "derived_plan_and_attempt_required": True,
    }
    embedded_rebind = embedded_launch_rebind(attempt_rebind)
    mismatches.extend(
        f"qualification_launch_rebind.{key}"
        for key, expected_value in expected_rebind.items()
        if embedded_rebind.get(key) != expected_value
    )
    rebind_body = dict(embedded_rebind)
    binding_sha256 = rebind_body.pop("binding_sha256", None)
    if binding_sha256 != _canonical_sha256(rebind_body):
        mismatches.append("qualification_launch_rebind.binding_sha256")
    for key in ("derived_launch_plan_sha256", "derived_attempt_manifest_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(attempt_rebind.get(key) or "")) is None:
            mismatches.append(f"qualification_launch_rebind.{key}")
    if attempt_rebind.get("runtime_attempt_manifest_rebound") is not True:
        mismatches.append("qualification_launch_rebind.runtime_attempt_manifest_rebound")
    if attempt_rebind.get("raw_source_bundle_modified") is not False:
        mismatches.append("qualification_launch_rebind.raw_source_bundle_modified")
    if mismatches:
        raise ValueError(
            "qualification_latest_attempt_binding_invalid:"
            + ",".join(sorted(set(mismatches)))
        )
    return {**latest, **expected, "qualification_launch_rebind": attempt_rebind}


def bind_inputs_to_release(
    inputs: Mapping[str, Any], release_binding: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive an auditable qualification plan without rewriting bundle truth."""

    plan_value = inputs.get("plan")
    attempt_value = inputs.get("attempt")
    plan = copy.deepcopy(dict(plan_value)) if isinstance(plan_value, Mapping) else {}
    attempt = (
        copy.deepcopy(dict(attempt_value))
        if isinstance(attempt_value, Mapping)
        else {}
    )
    if plan.get("sealed_active") is not True or plan.get("blockers"):
        raise ValueError("qualification_source_launch_plan_not_sealed")
    original_image_ref = str(plan.get("image_ref") or "")
    match = _DIGEST_REF_RE.fullmatch(original_image_ref)
    if match is None:
        raise ValueError("qualification_source_launch_plan_image_not_digest_pinned")
    original_image_digest = f"sha256:{match.group(1)}"
    if attempt.get("image_digest") != original_image_digest:
        raise ValueError("qualification_source_attempt_image_mismatch")

    target_image_ref = str(release_binding.get("image_ref") or "")
    target_image_digest = str(release_binding.get("image_digest") or "")
    source_commit = str(release_binding.get("source_commit") or "")
    if not valid_image_binding(target_image_ref, target_image_digest):
        raise ValueError("qualification_target_release_image_invalid")
    if not valid_source_commit(source_commit):
        raise ValueError("qualification_target_release_source_commit_invalid")
    source_patch_sha256 = str(release_binding.get("source_patch_sha256") or "")
    if source_patch_sha256 != EMPTY_PATCH_SHA256:
        raise ValueError("qualification_target_release_source_patch_invalid")
    bundle_sha256 = str(inputs.get("bundle_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", bundle_sha256):
        raise ValueError("qualification_source_bundle_digest_invalid")

    binding_body = {
        "schema_version": LAUNCH_REBIND_SCHEMA_VERSION,
        "status": "bound",
        "source_bundle_sha256": bundle_sha256,
        "source_loaded_plan_sha256": _canonical_sha256(plan),
        "source_plan_origin": "validated_sealed_bundle_then_loader_derived_runtime_plan",
        "source_image_ref": original_image_ref,
        "source_image_digest": original_image_digest,
        "release_image_ref": target_image_ref,
        "release_image_digest": target_image_digest,
        "release_source_commit": source_commit,
        "release_source_patch_sha256": source_patch_sha256,
        "source_bundle_preserved": True,
        "derived_plan_and_attempt_required": True,
    }
    binding_sha256 = _canonical_sha256(binding_body)
    embedded_binding = {**binding_body, "binding_sha256": binding_sha256}

    plan["image_ref"] = target_image_ref
    plan["qualification_release_rebind"] = embedded_binding
    derived_plan_sha256 = _canonical_sha256(plan)
    task_contract_value = inputs.get("task_contract")
    if not isinstance(task_contract_value, Mapping):
        raise ValueError("qualification_resolved_task_contract_missing")
    resolved_task_contract_bytes = (
        json.dumps(dict(task_contract_value), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    resolved_task_contract_sha256 = hashlib.sha256(
        resolved_task_contract_bytes
    ).hexdigest()
    source_task_contract_sha256 = str(inputs.get("source_task_contract_sha256") or "")
    if re.fullmatch(r"[0-9a-f]{64}", source_task_contract_sha256) is None:
        raise ValueError("qualification_source_task_contract_digest_invalid")
    source_attempt_identity = {
        "source_commit": attempt.get("source_commit"),
        "source_dirty_patch_sha256": attempt.get("source_dirty_patch_sha256"),
        "image_digest": attempt.get("image_digest"),
        "preserved_from_source_bundle": True,
    }
    attempt["source_bundle_attempt_identity"] = source_attempt_identity
    attempt["source_commit"] = source_commit
    attempt["source_dirty_patch_sha256"] = source_patch_sha256
    attempt["image_digest"] = target_image_digest
    attempt["qualification_release_rebind"] = embedded_binding
    attempt["qualification_derived_launch_plan"] = copy.deepcopy(plan)
    attempt["qualification_derived_launch_plan_sha256"] = derived_plan_sha256
    attempt["qualification_resolved_task_success_contract_sha256"] = (
        resolved_task_contract_sha256
    )
    attempt["qualification_source_task_success_contract_sha256"] = (
        source_task_contract_sha256
    )
    attempt["qualification_runtime_attempt_overlay_base"] = (
        capture_runtime_attempt_overlay_base(attempt)
    )
    attempt_bytes = (json.dumps(attempt, indent=2, sort_keys=True) + "\n").encode("utf-8")

    bootstrap_script = str(inputs.get("bootstrap_script") or "")
    marker = (
        "cp /workspace/attempt_input_manifest_episode_001.json "
        "/workspace/attempt_input_manifest.json\n"
    )
    if bootstrap_script.count(marker) != 1:
        raise ValueError("qualification_attempt_manifest_rebind_marker_ambiguous")
    encoded_attempt = base64.b64encode(attempt_bytes).decode("ascii")
    replacement = (
        "python3 - <<'PY'\n"
        "import base64\n"
        "from pathlib import Path\n"
        f"payload = base64.b64decode({encoded_attempt!r}, validate=True)\n"
        "Path('/workspace/attempt_input_manifest.json').write_bytes(payload)\n"
        "PY\n"
    )

    updated = dict(inputs)
    updated["plan"] = plan
    updated["attempt"] = attempt
    updated["bootstrap_script"] = bootstrap_script.replace(marker, replacement, 1)
    launch_binding = {
        **embedded_binding,
        "derived_launch_plan_sha256": derived_plan_sha256,
        "derived_attempt_manifest_sha256": hashlib.sha256(attempt_bytes).hexdigest(),
        "runtime_attempt_manifest_rebound": True,
        "raw_source_bundle_modified": False,
    }
    updated["qualification_launch_rebind"] = launch_binding
    return updated, launch_binding


def build_release_binding(
    release: Mapping[str, Any], *, expected_source_commit: str
) -> tuple[dict[str, Any], list[str]]:
    """Derive an immutable worker binding from independent release evidence."""

    blockers: list[str] = []
    expected_source_commit = str(expected_source_commit or "")
    if not valid_source_commit(expected_source_commit):
        blockers.append("qualification_expected_source_commit_invalid")
    if release.get("schema_version") != "groot_oscar_thin_remote_build_result.v1":
        blockers.append("qualification_release_evidence_schema_invalid")
    if release.get("status") != "completed":
        blockers.append("qualification_release_evidence_not_completed")
    if release.get("blockers") != []:
        blockers.append("qualification_release_evidence_has_blockers")

    source_commit = str(release.get("source_commit") or "")
    if not valid_source_commit(source_commit):
        blockers.append("qualification_release_source_commit_invalid")
    elif source_commit != expected_source_commit:
        blockers.append("qualification_release_source_commit_mismatch")
    if release.get("source_patch_sha256") != EMPTY_PATCH_SHA256:
        blockers.append("qualification_release_source_patch_not_empty")

    image_ref = str(release.get("resolved_digest_ref") or "")
    digest_match = _DIGEST_REF_RE.fullmatch(image_ref)
    if digest_match is None:
        blockers.append("qualification_release_image_not_digest_pinned")
        image_digest = ""
    else:
        image_digest = f"sha256:{digest_match.group(1)}"
    if release.get("release_image_ref") != image_ref:
        blockers.append("qualification_release_image_ref_mismatch")
    if release.get("runnable_platform") != "linux/amd64":
        blockers.append("qualification_release_platform_not_linux_amd64")
    thin = release.get("thin_release_contract")
    thin = dict(thin) if isinstance(thin, Mapping) else {}
    if thin.get("schema_version") != "groot_oscar_thin_release_image_contract.v1":
        blockers.append("qualification_thin_release_contract_schema_invalid")
    if thin.get("status") != "passed" or release.get("thin_release_contract_status") != "passed":
        blockers.append("qualification_thin_release_contract_not_passed")
    if thin.get("blockers") != []:
        blockers.append("qualification_thin_release_contract_has_blockers")
    if thin.get("release_image_ref") != image_ref:
        blockers.append("qualification_thin_release_image_mismatch")
    foundation_model_assets = str(
        release.get("foundation_model_assets")
        or thin.get("foundation_model_assets")
        or (
            "external"
            if release.get("models_embedded") is False
            and thin.get("models_externalized") is True
            else ""
        )
    )
    externalized = (
        foundation_model_assets == "external"
        and release.get("models_embedded") is False
        and thin.get("models_externalized") is True
        and thin.get("models_embedded_in_foundation") is not True
    )
    embedded = (
        foundation_model_assets == "embedded"
        and release.get("models_embedded") is True
        and thin.get("models_embedded_in_foundation") is True
        and thin.get("models_externalized") is not True
    )
    if not externalized and not embedded:
        blockers.append("qualification_release_model_assets_mode_invalid")
    foundation_image_ref = str(release.get("foundation_image_ref") or "")
    if embedded and _DIGEST_REF_RE.fullmatch(foundation_image_ref) is None:
        blockers.append("qualification_embedded_foundation_not_digest_pinned")

    required_cuda_version = str(release.get("required_cuda_version") or "")
    required_cuda_source = str(release.get("required_cuda_version_source") or "")
    if not required_cuda_version:
        blockers.append("qualification_release_cuda_version_missing")
    if not required_cuda_source.startswith("image_config_env:"):
        blockers.append("qualification_release_cuda_source_unverified")

    binding = {
        "schema_version": RELEASE_BINDING_SCHEMA_VERSION,
        "image_ref": image_ref or None,
        "image_digest": image_digest or None,
        "source_commit": source_commit or None,
        "source_patch_sha256": release.get("source_patch_sha256"),
        "runnable_platform": release.get("runnable_platform"),
        "required_cuda_version": required_cuda_version or None,
        "required_cuda_version_source": required_cuda_source or None,
        "model_assets_mode": foundation_model_assets or None,
        "models_externalized": externalized,
        "models_embedded_in_foundation": embedded,
        "foundation_image_ref": foundation_image_ref or None,
        "release_evidence_status": release.get("status"),
        "thin_release_contract_status": thin.get("status"),
    }
    return binding, sorted(set(blockers))


def qualification_gate_matrix() -> list[dict[str, Any]]:
    """Forward proof ledger, separating historical attempts from future gates."""

    proven_016 = "attempt016_proven"
    proven_017 = "attempt017_proven"
    pending = "pending"
    rows = (
        (
            "image_bundle_assets",
            "Exact sealed image, episode bundle, kitchen assets, and runtime overlays",
            proven_016,
            "Exact image/bundle binding, startup asset gate, and image healthcheck passed.",
            proven_017,
            "Attempt 017 repeated the exact image, bundle, asset, and runtime-overlay proof.",
        ),
        (
            "groot_checkpoint_server",
            "GR00T N1.7 SONIC checkpoint preflight and live policy server readiness",
            proven_016,
            "Checkpoint preflight passed and the worker advanced beyond GR00T server readiness.",
            proven_017,
            "Attempt 017 recorded the live GR00T server ready and accepting the first real query.",
        ),
        (
            "isaac_scene_baseline",
            "Isaac RTX, kitchen stage, live G1, standing pose, and microwave baseline",
            proven_016,
            "Attempt 016 recorded RTX startup, live articulation, standing initialization, and the exact microwave door baseline.",
            proven_017,
            "Attempt 017 repeated RTX, kitchen, live G1 standing, and microwave-door baseline proof.",
        ),
        (
            "controller_init_done",
            "Official GEAR-SONIC controller emits Init Done",
            pending,
            "Attempt 016 stopped at official controller readiness.",
            proven_017,
            "Attempt 017 recorded the official controller Init Done marker.",
        ),
        (
            "native_dds_freshness",
            "Native DDS bridge identity, advancing publication count, and fresh Isaac samples",
            proven_016,
            "Attempt 016 recorded the compiled ELF-audited bridge and fresh live Isaac snapshots; this did not prove controller readiness.",
            proven_017,
            "Attempt 017 recorded the native DDS bridge ready with fresh live Isaac state.",
        ),
        (
            "first_groot_query",
            "First real GR00T policy query",
            pending,
            None,
            proven_017,
            "Attempt 017 produced a fresh 40-frame by 78-dimension receding-horizon response from the real initial observation.",
        ),
        (
            "protocol_v4_token_receipt",
            "Matching protocol-v4 action token received by the official GEAR-SONIC controller",
            pending,
            None,
            proven_017,
            "Attempt 017 recorded the matching 64D token and hands payload at frame/step 1.",
        ),
        (
            "first_official_action",
            "First matching official GEAR-SONIC FK/action output produced from the learned policy response",
            pending,
            None,
            "partial_protocol_v4_token_receipt_only",
            "The matching protocol-v4 token arrived in Attempt 017, but no matching g1_debug/FK output was produced.",
        ),
        (
            "isaac_apply_readback",
            "First action applied to live Isaac and read back from the same articulation",
            pending,
            None,
            pending,
            "Attempt 017 failed before an official FK output or Isaac apply/readback.",
        ),
        (
            "first_learned_oscar_transition",
            "First learned OSCAR/WAM transition",
            pending,
            None,
            pending,
            "Attempt 017 failed before the first learned OSCAR/WAM transition.",
        ),
        (
            "semantic_microwave_transition",
            "Microwave door semantic state transition satisfies the task contract",
            pending,
            None,
            pending,
            "Attempt 017 failed at step 1 before any semantic microwave-door transition.",
        ),
        (
            "ordered_review_render",
            "Ordered first-person and third-person review render",
            pending,
            None,
            pending,
            "The startup canary frame is not an ordered episode review render.",
        ),
        (
            "dynamic_termination",
            "Dynamic task-completion termination or the explicit 48-step safety cap",
            pending,
            None,
            pending,
            "Attempt 017 terminated on a step-1 component failure, not task completion or the safety cap.",
        ),
        (
            "validated_final_review_upload",
            "Validated final_review video, ordering evidence, digest, decode, and upload",
            pending,
            None,
            pending,
            "Attempt 017 final-review validation was blocked and produced no episode MP4.",
        ),
    )
    return [
        {
            "gate_id": gate_id,
            "requirement": requirement,
            "attempt_016_status": status_016,
            "attempt_016_evidence_note": note_016,
            "attempt_017_status": status_017,
            "attempt_017_evidence_note": note_017,
            "current_session_status": "pending",
            "evidence_note": note_017 or note_016,
            "required_for_full_episode_video": True,
        }
        for gate_id, requirement, status_016, note_016, status_017, note_017 in rows
    ]


def session_claim_boundary() -> dict[str, Any]:
    return {
        "allocation_is_not_runtime_readiness": True,
        "runtime_readiness_is_not_episode_success": True,
        "video_arrival_is_not_validated_review_video": True,
        "validated_review_video_is_not_semantic_task_success": True,
        "current_target": (
            "One dynamic G1 kitchen episode using real GR00T N1.7 SONIC, the official "
            "GEAR-SONIC controller, learned OSCAR/WAM, live Isaac action apply/readback, "
            "semantic microwave-door success, and a validated ordered final review video."
        ),
        "prior_persistent_result": {
            "policy_calls": 2,
            "learned_wam_transitions": 1,
            "isaac_kitchen_semantic_success_proven": False,
            "full_episode_video_proven": False,
            "must_not_be_promoted_to_current_goal_completion": True,
        },
        "attempt_016_result": {
            "exact_image_assets": True,
            "groot_server_ready": True,
            "isaac_rtx_kitchen_g1_baseline": True,
            "native_dds_freshness": True,
            "controller_init_done": False,
            "real_groot_query_proven": False,
            "learned_oscar_transition_proven": False,
            "semantic_success_proven": False,
            "full_episode_video_proven": False,
            "failure_boundary": "official_controller_readiness",
        },
        "attempt_017_result": {
            "exact_image_assets": True,
            "groot_server_ready": True,
            "fresh_action_horizon": {
                "frame_count": 40,
                "frame_dimension": 78,
                "selection_mode": "fresh_receding_horizon_first_frame",
                "real_initial_observation": True,
            },
            "isaac_rtx_kitchen_g1_baseline": True,
            "controller_init_done": True,
            "native_dds_ready": True,
            "protocol_v4_token_receipt_step": 1,
            "matching_g1_debug_fk_output_proven": False,
            "isaac_action_apply_readback_proven": False,
            "learned_oscar_transition_proven": False,
            "semantic_success_proven": False,
            "full_episode_video_proven": False,
            "failure_boundary": (
                "controller_fk_skeleton_command_nonzero_before_matching_g1_debug_output; "
                "plain_git_rev_parse_rejected_root_owned_opt_wbc_as_dubious_ownership"
            ),
            "must_not_be_promoted_to_current_goal_completion": True,
        },
    }
