"""Automatic policy-canary hand-off after the configured controls run completes.

Once the configured-controls progression has launched and completed the Franka
controls run for a Website-configured scene, nothing used to chain into the
Quick-10 policy canary: an operator authored the presubmission parameters,
attached the setup to a profile, published the profile and catalog, and clicked
the offering page.  This module composes exactly those production producers so
the progression timer performs the hand-off itself:

1. derive every presubmission input from the completed chain (configured run,
   base progression, compiled construction packet, terminal controls run,
   owner activation intent, repository manifests),
2. emit the presubmission setup and the profile materialization input,
3. materialize and publish the canary profile plus the WebApp catalog,
4. submit the canonical Quick-10 selection through the WebApp service channel.

Every stage is sealed in an immutable progression record so a retried tick
replays the recorded stage instead of minting a second setup or a second run.
The module never allocates provider resources: the canary is executed later by
the existing activation and dispatch workers under their own paid authority.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import subprocess  # nosec B404 - fixed interpreter and repository scripts only
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .task_evaluation_policy_canary_handoff_state import (
    PolicyCanaryHandoffError as PolicyCanaryHandoffError,
    write_immutable as _write_immutable,
    seal_state as _seal_state,
    serialized_handoff,
    submit_or_adopt,
    verify_completed_ack,
)
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .task_evaluation_launch_dispatcher import LAUNCH_RECEIPT_DIGEST_CANONICALIZATION
from .task_evaluation_launch_reconciler import validated_succeeded_webapp_sync_row
from .task_evaluation_policy_canary_model_rights import materialize_policy_canary_model_rights
from .task_evaluation_policy_canary_setup import policy_canary_setup_digest
from .task_evaluation_policy_canary_scene_setup import (
    _quick_cells,
    materialize_policy_canary_presubmission_setup,
)
from .task_evaluation_scene_configuration_activation_automation import (
    load_scene_configuration_activation_intent,
)
from . import task_evaluation_scene_policy_binding as scene_policy

__all__ = [
    "PolicyCanaryHandoffError",
    "STATE_FILENAME",
    "advance_policy_canary_handoff",
    "advance_policy_canary_handoff_for_plan",
    "rebind_policy_controller_configuration_to_scene",
]

STATE_FILENAME = "policy_canary_handoff_progression.json"
PROGRESSION_SCHEMA_VERSION = "task_evaluation_policy_canary_handoff_progression.v1"
CLIENT_ID = "blueprint-production-runner"
CANARY_HARD_CAP_USD = 4.0
CANARY_HARD_TTL_SECONDS = 9_000
CANARY_HOURLY_RATE_USD = 0.8
RELEASE_WINDOW_VALID_SECONDS = 3_600
NOTIFY_ON = ("completed", "blocked", "cancelled")
EPISODE_INTERPRETATION = dict(scene_policy.INTERPRETATION_DEFAULT)
MANIFESTS = {
    "policy_controller_template": (
        "docs/arm_decision_proof_v1/manifests/"
        "scene839873_policy_canary_controller_configuration.v1.json"
    ),
    "native_controller_configuration": (
        "docs/arm_decision_proof_v1/manifests/"
        "scene839873_policy_canary_native_controller_configuration.v1.json"
    ),
    "model_rights_template": (
        "docs/arm_decision_proof_v1/manifests/scene839873_policy_canary_model_rights.v1.json"
    ),
    "historical_policy_readiness": (
        "docs/arm_decision_proof_v1/manifests/adp009d_scene_840920_policy_readiness.v1.json"
    ),
    "pi05_checkpoint_inventory": (
        "docs/experiments/policy_ranking_thesis_20260726/openpi_polaris_checkpoint_inventory.json"
    ),
}
_STAGES = ("canary_presubmitted", "canary_profile_published", "canary_launch_submitted")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")
_TIMESTAMP_HEADER = "X-Blueprint-Launch-Timestamp"
_CLIENT_ID_HEADER = "X-Blueprint-Launch-Client-Id"
_NONCE_HEADER = "X-Blueprint-Launch-Nonce"
_SIGNATURE_HEADER = "X-Blueprint-Launch-Signature"
_IDEMPOTENCY_HEADER = "Idempotency-Key"
_MAX_RESPONSE_BYTES = 1_000_000

Publisher = Callable[..., Mapping[str, Any]]
Poster = Callable[..., tuple[int, bytes]]
ProfilePublisher = Callable[..., Mapping[str, Any]]





# ---------------------------------------------------------------- helpers


def _load(path: Path, *, blocker: str) -> dict[str, Any]:
    try:
        if path.is_symlink():
            raise PolicyCanaryHandoffError(blocker)
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PolicyCanaryHandoffError(blocker) from exc
    if not isinstance(value, Mapping):
        raise PolicyCanaryHandoffError(blocker)
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _payload(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_or_reuse(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Retain time-stamped documents exactly once; a retry reuses the sealed bytes."""

    if path.exists() or path.is_symlink():
        return _load(path, blocker=f"policy_canary_handoff_retained_invalid:{path.name}")
    _write_immutable(path, value)
    return dict(value)


def _is_sealed(value: Mapping[str, Any], *, field: str) -> bool:
    digest = value.get(field)
    return isinstance(digest, str) and digest == canonical_digest(value, digest_field=field)


def _reference(value: Any, *, blocker: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or not str(value.get("uri") or "").startswith(("s3://", "gs://", "https://"))
        or _DIGEST.fullmatch(str(value.get("digest") or "")) is None
        or isinstance(value.get("size_bytes"), bool)
        or not isinstance(value.get("size_bytes"), int)
        or value["size_bytes"] <= 0
    ):
        raise PolicyCanaryHandoffError(blocker)
    return {
        "uri": str(value["uri"]),
        "digest": str(value["digest"]),
        "size_bytes": int(value["size_bytes"]),
    }


def _publish(*, path: Path, object_name: str, publisher: Publisher) -> dict[str, Any]:
    published = dict(publisher(path=path, object_name=object_name))
    if (
        published.get("full_byte_service_account_readback_passed") is not True
        or published.get("digest") != _sha256(path)
    ):
        raise PolicyCanaryHandoffError("policy_canary_handoff_publication_readback_failed")
    return _reference(published, blocker="policy_canary_handoff_publication_invalid")


def _sealed_progression(path: Path, *, statuses: set[str]) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    value = _load(path, blocker="policy_canary_handoff_progression_invalid")
    if value.get("status") not in statuses or not _is_sealed(value, field="progression_digest"):
        raise PolicyCanaryHandoffError("policy_canary_handoff_progression_invalid")
    return value


# ---------------------------------------------------------------- terminal predecessor


def _terminal_evidence(
    run_root: Path, *, launch_id: str, result_schema: str, qualified_field: str
) -> dict[str, Path] | None:
    """Validate one completed native-arena run exactly as the progression worker does.

    Returns the predecessor artifact paths, or ``None`` while the run has not
    reached its terminal receipt, WebApp sync and post-teardown provider zero.
    """

    receipt_path = run_root / "launch_receipt.json"
    sync_path = run_root / "webapp_sync_succeeded.json"
    zero_path = run_root / "post_teardown_provider_zero_receipt.json"
    if not (receipt_path.is_file() and sync_path.is_file() and zero_path.is_file()):
        return None
    receipt = _load(receipt_path, blocker="policy_canary_handoff_predecessor_receipt_invalid")
    expected_receipt_digest = (
        cross_runtime_canonical_digest(receipt, digest_field="receipt_digest")
        if receipt.get("receipt_digest_canonicalization") == LAUNCH_RECEIPT_DIGEST_CANONICALIZATION
        else canonical_digest(receipt, digest_field="receipt_digest")
    )
    terminal = receipt.get("terminal_evidence")
    artifact = terminal.get("result") if isinstance(terminal, Mapping) else None
    if (
        receipt.get("schema_version") != "task_evaluation_launch_receipt.v1"
        or receipt.get("status") != "completed"
        or receipt.get("launch_id") != launch_id
        or receipt.get("receipt_digest") != expected_receipt_digest
        or not isinstance(terminal, Mapping)
        or terminal.get("status") != "passed"
        or not isinstance(artifact, Mapping)
        or artifact.get("exists") is not True
    ):
        raise PolicyCanaryHandoffError("policy_canary_handoff_predecessor_receipt_invalid")
    result_path = Path(str(artifact.get("path") or "")).expanduser()
    if result_path.is_symlink() or not result_path.is_file() or _sha256(result_path) != artifact.get("digest"):
        raise PolicyCanaryHandoffError("policy_canary_handoff_predecessor_result_invalid")
    result = _load(result_path, blocker="policy_canary_handoff_predecessor_result_invalid")
    native_path = Path(str(result.get("native_control_result_path") or "")).expanduser()
    native = _load(native_path, blocker="policy_canary_handoff_predecessor_native_result_invalid")
    if (
        result.get("schema_version") != "native_task_arena_vast_run.v1"
        or result.get("status") != "completed"
        or result.get("blockers") not in ([], ())
        or result.get("native_control_result_digest") != native.get("result_digest")
        or native.get("schema_version") != result_schema
        or native.get("status") != "completed"
        or native.get(qualified_field) is not True
        or native.get("candidate_policy_queried") is not False
        or native.get("blockers") not in ([], ())
        or not _is_sealed(native, field="result_digest")
    ):
        raise PolicyCanaryHandoffError("policy_canary_handoff_predecessor_native_result_invalid")
    sync = _load(sync_path, blocker="policy_canary_handoff_predecessor_webapp_sync_invalid")
    try:
        validated_succeeded_webapp_sync_row(receipt=receipt, attempt=sync)
    except Exception as exc:
        raise PolicyCanaryHandoffError("policy_canary_handoff_predecessor_webapp_sync_invalid") from exc
    zero = _load(zero_path, blocker="policy_canary_handoff_predecessor_provider_zero_invalid")
    if (
        zero.get("schema_version") != "task_evaluation_post_teardown_provider_zero.v1"
        or zero.get("status") != "provider_zero_confirmed"
        or zero.get("provider_zero_verified") is not True
        or zero.get("continuing_spend_from_this_run") is not False
        or zero.get("allocator_invoked") is not False
        or zero.get("provider_mutation_performed") is not False
        or zero.get("automatic_retry_performed") is not False
        or zero.get("blockers") != []
        or not _is_sealed(zero, field="provider_zero_receipt_digest")
        or any(
            zero.get(field) != receipt.get(field)
            for field in ("launch_id", "run_id", "request_digest", "receipt_digest", "launch_profile_digest")
        )
    ):
        raise PolicyCanaryHandoffError("policy_canary_handoff_predecessor_provider_zero_invalid")
    profile = _load(run_root / "launch_profile.json", blocker="policy_canary_handoff_predecessor_profile_invalid")
    if (
        profile.get("schema_version") != "task_evaluation_launch_profile.v1"
        or profile.get("profile_digest") != receipt.get("launch_profile_digest")
        or not _is_sealed(profile, field="profile_digest")
        or not isinstance(profile.get("immutable_inputs"), list)
    ):
        raise PolicyCanaryHandoffError("policy_canary_handoff_predecessor_profile_invalid")
    artifact_paths: dict[str, Path] = {
        "prior_result": result_path,
        "prior_launch_receipt": receipt_path,
        "prior_webapp_sync": sync_path,
        "prior_provider_zero": zero_path,
        "native_result": native_path,
    }
    for role, expected_name in (
        ("prior_authority", "native_task_arena_attempt_authority"),
        ("prior_spend_reconciliation", "native_task_arena_attempt_authority_prior_spend_reconciliation"),
    ):
        matches = [
            row
            for row in profile["immutable_inputs"]
            if isinstance(row, Mapping) and row.get("name") == expected_name
        ]
        if len(matches) != 1:
            raise PolicyCanaryHandoffError("policy_canary_handoff_predecessor_profile_invalid")
        path = Path(str(matches[0].get("path") or "")).expanduser()
        if path.is_symlink() or not path.is_file() or _sha256(path) != matches[0].get("digest"):
            raise PolicyCanaryHandoffError("policy_canary_handoff_predecessor_profile_input_invalid")
        artifact_paths[role] = path
    return artifact_paths


def _predecessor_lineage(
    *,
    launch_state_root: Path,
    controls_launch_id: str,
    construction_launch_id: str,
    publisher: Publisher,
) -> tuple[dict[str, Any], dict[str, str]] | None:
    """Publish the terminal controls run as the canary's predecessor lineage."""

    controls = _terminal_evidence(
        launch_state_root / controls_launch_id,
        launch_id=controls_launch_id,
        result_schema="native_task_arena_control_result.v1",
        qualified_field="controls_qualified",
    )
    if controls is None:
        return None
    construction = _terminal_evidence(
        launch_state_root / construction_launch_id,
        launch_id=construction_launch_id,
        result_schema="native_task_arena_construction_result.v1",
        qualified_field="construction_gate_qualified",
    )
    if construction is None:
        raise PolicyCanaryHandoffError("policy_canary_handoff_construction_predecessor_missing")
    artifact_paths = {
        role: path for role, path in controls.items() if role != "native_result"
    }
    artifact_paths["construction_result"] = construction["native_result"]
    lineage: dict[str, Any] = {"kind": "predecessor"}
    published_paths: dict[str, str] = {}
    for role, path in sorted(artifact_paths.items()):
        lineage[role] = _publish(
            path=path,
            object_name=f"policy-canary-predecessor/{controls_launch_id}/{role}.json",
            publisher=publisher,
        )
        published_paths[role] = str(path)
    return lineage, published_paths


# ---------------------------------------------------------------- scene-bound manifests


def rebind_policy_controller_configuration_to_scene(
    *,
    template_path: str | Path,
    scene_id: str,
    task_id: str,
    scene_revision_digest: str,
) -> dict[str, Any]:
    """Rebind the paired-policy controller configuration to one scene and its Quick-10 matrix.

    The checked-in template is bound to its authoring scene (identity, cell ids
    and matrix digest).  Everything embodiment-level is kept verbatim; only the
    scene identity and the deterministic Quick-10 matrix are re-derived, then
    the document is resealed.
    """

    template = _load(Path(template_path).expanduser(), blocker="policy_canary_handoff_controller_template_invalid")
    if (
        not str(template.get("schema_version") or "").endswith("_policy_canary_controller_configuration.v1")
        or not isinstance(template.get("quick_10"), Mapping)
        or not _is_sealed(template, field="configuration_digest")
    ):
        raise PolicyCanaryHandoffError("policy_canary_handoff_controller_template_invalid")
    if not re.fullmatch(r"[0-9]{1,12}", scene_id) or _IDENTIFIER.fullmatch(task_id) is None:
        raise PolicyCanaryHandoffError("policy_canary_handoff_scene_identity_invalid")
    cells = _quick_cells(scene_revision_digest, scene_id=scene_id)
    quick = dict(template["quick_10"])
    quick.update({"cells": cells, "matrix_digest": canonical_digest({"cells": cells})})
    document = dict(template)
    document.update(
        {
            "schema_version": f"scene{scene_id}_policy_canary_controller_configuration.v1",
            "scene_id": scene_id,
            "task_id": task_id,
            "quick_10": quick,
            "configuration_digest": "",
        }
    )
    document["configuration_digest"] = canonical_digest(document, digest_field="configuration_digest")
    return document


def _release_window_template(
    *, team_namespace: str, expected_production_commit: str, released_by: str, source_launch_id: str
) -> dict[str, Any]:
    template = {
        "schema_version": "task_evaluation_configured_controls_release_window_template.v1",
        "status": "authorized_for_dynamic_release",
        "team_namespace": team_namespace,
        "expected_production_commit": expected_production_commit,
        "allowed_mutations": ["profile_publication", "catalog_synchronization", "standing_authorization"],
        "provider_allowlist": ["vast"],
        "maximum_hard_cap_usd": CANARY_HARD_CAP_USD,
        "valid_for_seconds": RELEASE_WINDOW_VALID_SECONDS,
        "released_by": released_by,
        "release_reference": f"automatic policy-canary activation for {source_launch_id}",
        "provider_resource_allocation_allowed": False,
        "paid_request_allowed": False,
        "template_digest": "",
    }
    template["template_digest"] = canonical_digest(template, digest_field="template_digest")
    return template


# ---------------------------------------------------------------- default effectors


def default_attach(*, repo_root: Path, base_profile_path: Path, wrapper_path: Path, output_path: Path) -> Path:
    """Materialize the canary profile through the repository script, never in-place."""

    if not output_path.exists():
        completed = subprocess.run(  # nosec B603 - fixed interpreter and repository-owned script
            [
                sys.executable,
                str(repo_root / "scripts" / "attach_internal_policy_canary_setup.py"),
                "--profile", str(base_profile_path),
                "--profile-materialization-input", str(wrapper_path),
                "--output", str(output_path),
            ],
            cwd=repo_root,
            env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0 or not output_path.is_file():
            raise PolicyCanaryHandoffError(
                "policy_canary_handoff_profile_materialization_failed:" + _blocker_excerpt(completed.stderr)
            )
    return output_path


def _blocker_excerpt(stderr: str) -> str:
    """Keep only the script's final blocker line; repository scripts print no secret values."""

    lines = [line.strip() for line in str(stderr or "").splitlines() if line.strip()]
    return re.sub(r"[^A-Za-z0-9_:,.\- ]", "", lines[-1])[:240] if lines else "no_stderr"


def default_profile_publisher(*, repo_root: Path) -> ProfilePublisher:
    def publish(*, profile_path: Path, profile_dir: Path, webapp_catalog_out: Path) -> Mapping[str, Any]:
        completed = subprocess.run(  # nosec B603 - fixed interpreter and repository-owned script
            [
                sys.executable,
                str(repo_root / "scripts" / "publish_task_evaluation_launch_profiles.py"),
                "--profile", str(profile_path),
                "--profile-dir", str(profile_dir),
                "--webapp-catalog-out", str(webapp_catalog_out),
            ],
            cwd=repo_root,
            env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0:
            raise PolicyCanaryHandoffError(
                "policy_canary_handoff_profile_publication_failed:" + _blocker_excerpt(completed.stderr)
            )
        try:
            receipt = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError) as exc:
            raise PolicyCanaryHandoffError("policy_canary_handoff_profile_publication_failed") from exc
        return receipt

    return publish


def default_poster(*, endpoint: str, headers: Mapping[str, str], body: bytes) -> tuple[int, bytes]:
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query:
        raise PolicyCanaryHandoffError("policy_canary_handoff_endpoint_not_https")
    request = urllib.request.Request(endpoint, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # nosec B310 - HTTPS enforced above
            return int(response.status), response.read(_MAX_RESPONSE_BYTES)
    except urllib.error.HTTPError as exc:
        return int(exc.code), b""
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise PolicyCanaryHandoffError("policy_canary_handoff_webapp_transport_error") from exc


def signed_headers(*, secret: bytes, body: bytes, run_id: str, now: datetime | None = None) -> dict[str, str]:
    """Sign the exact WebApp service-channel byte sequence for one canary selection."""

    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    nonce = secrets.token_hex(16)
    canonical = f"{timestamp}.{CLIENT_ID}.{nonce}.".encode("utf-8") + body
    signature = hmac.new(secret, canonical, "sha256").hexdigest()
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        _TIMESTAMP_HEADER: timestamp,
        _CLIENT_ID_HEADER: CLIENT_ID,
        _NONCE_HEADER: nonce,
        _SIGNATURE_HEADER: f"sha256={signature}",
        _IDEMPOTENCY_HEADER: run_id,
    }


# ---------------------------------------------------------------- hand-off


def _configured_run(launch_state_root: Path, source_launch_id: str) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    run_root = launch_state_root / source_launch_id
    request_path = run_root / "launch_request.json"
    profile_path = run_root / "launch_profile.json"
    request = _load(request_path, blocker="policy_canary_handoff_configured_request_invalid")
    profile = _load(profile_path, blocker="policy_canary_handoff_configured_profile_invalid")
    if (
        request.get("schema_version") != "task_evaluation_launch_request.v1"
        or _COMMIT.fullmatch(str(request.get("source_commit") or "")) is None
        or _DIGEST.fullmatch(str(request.get("request_digest") or "")) is None
        or profile.get("schema_version") != "task_evaluation_launch_profile.v1"
        or profile.get("source_commit") != request["source_commit"]
        or not _is_sealed(profile, field="profile_digest")
        or request.get("launch_profile_digest") != profile["profile_digest"]
    ):
        raise PolicyCanaryHandoffError("policy_canary_handoff_configured_run_invalid")
    return request_path, request, profile_path, profile


def _base_progression(state_root: Path) -> dict[str, Any]:
    for candidate in (state_root / "configured_controls_progression.v1.json", state_root / "episode" / "configured_controls_progression.v1.json"):
        if candidate.is_file():
            base = _load(candidate, blocker="policy_canary_handoff_base_progression_invalid")
            request = base.get("episode_preparation_request")
            if (
                base.get("status") not in {"episode_preparation_queued"}
                or not isinstance(request, Mapping)
                or _DIGEST.fullmatch(str(base.get("configured_scene_revision_digest") or "")) is None
                or _DIGEST.fullmatch(str(base.get("configured_scene_offering_digest") or "")) is None
                or not str(base.get("configuration_run_id") or "")
            ):
                raise PolicyCanaryHandoffError("policy_canary_handoff_base_progression_invalid")
            return base
    raise PolicyCanaryHandoffError("policy_canary_handoff_base_progression_missing")


def _compiled_construction(
    episode_compilation_queue_root: Path, *, preparation_id: str, expected_production_commit: str
) -> dict[str, Any] | None:
    results = episode_compilation_queue_root / "results"
    for path in sorted(results.glob(f"{preparation_id}-*.json")) if results.is_dir() else []:
        if path.is_symlink():
            continue
        result = _load(path, blocker="policy_canary_handoff_compilation_invalid")
        if result.get("status") != "compiled_for_production_launch":
            continue
        if (
            result.get("compilation_id") != preparation_id
            or result.get("source_commit") != expected_production_commit
            or not _is_sealed(result, field="result_digest")
        ):
            raise PolicyCanaryHandoffError("policy_canary_handoff_compilation_invalid")
        adapter_path = Path(str(result.get("adapter_result_path") or "")).expanduser()
        adapter = _load(adapter_path, blocker="policy_canary_handoff_adapter_result_invalid")
        packet_root = Path(str(adapter.get("packet_root") or "")).expanduser()
        runtime_receipt = Path(str(adapter.get("runtime_source_receipt") or "")).expanduser()
        if (
            adapter.get("status") != "native_arena_adapter_materialized"
            or adapter.get("preparation_id") != preparation_id
            or adapter.get("source_commit") != expected_production_commit
            or not _is_sealed(adapter, field="result_digest")
            or result.get("adapter_result_digest") != adapter["result_digest"]
            or not (packet_root / "native_task_arena_scene_plan.v1.json").is_file()
            or not (packet_root / "native_task_arena_packet_receipt.v1.json").is_file()
            or not runtime_receipt.is_file()
        ):
            raise PolicyCanaryHandoffError("policy_canary_handoff_adapter_result_invalid")
        return {
            "scene_plan_path": packet_root / "native_task_arena_scene_plan.v1.json",
            "packet_receipt_path": packet_root / "native_task_arena_packet_receipt.v1.json",
            "runtime_source_receipt_path": runtime_receipt,
            "compilation_result_digest": result["result_digest"],
        }
    return None


def _selection(
    *, setup: Mapping[str, Any], run_id: str, notification_email: str
) -> dict[str, Any]:
    presets = [row for row in setup.get("episode_presets") or [] if row.get("preset_id") == "quick_10"]
    robots = list(setup.get("robot_presets") or [])
    if len(presets) != 1 or presets[0].get("availability") != "enabled" or len(robots) != 1:
        raise PolicyCanaryHandoffError("policy_canary_handoff_setup_quick10_unavailable")
    quick = presets[0]
    candidates = [row["candidate_id"] for row in robots[0].get("policy_candidates") or []]
    if candidates != ["pi05_droid", "groot_n17_droid"]:
        raise PolicyCanaryHandoffError("policy_canary_handoff_setup_candidates_invalid")
    contract = setup.get("task_success_contract")
    if not isinstance(contract, Mapping) or setup.get("task_success_contract_digest") != contract.get("contract_digest"):
        raise PolicyCanaryHandoffError("policy_canary_handoff_setup_contract_invalid")
    return {
        "schema_version": "task_evaluation_policy_canary_selection.v1",
        "run_kind": setup["run_kind"],
        "claim_ceiling": setup["claim_ceiling"],
        "run_id": run_id,
        "offering_digest": setup["offering_digest"],
        "setup_digest": setup["setup_digest"],
        "scene_revision_digest": setup["scene_revision_digest"],
        "robot_preset_id": robots[0]["robot_preset_id"],
        "policy_candidate_ids": candidates,
        "episode_preset_id": "quick_10",
        "variation_matrix_digest": quick["matrix"]["matrix_digest"],
        "task_success_contract": json.loads(json.dumps(contract)),
        "notification": {"email": notification_email, "notify_on": list(NOTIFY_ON)},
        "authorization": {
            "maximum_cost_usd": quick["estimate"]["maximum_authorized_cost_usd"],
            "hard_ttl_seconds": quick["estimate"]["hard_ttl_seconds"],
            "maximum_provider_allocations": 1,
            "retry_cap": 0,
        },
        "episode_interpretation": dict(EPISODE_INTERPRETATION),
        "confirm_unqualified_execution": True,
    }


def _row(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in ("status", "run_id", "profile_id", "setup_digest", "controls_launch_id", "provider_mutation_performed")
        if key in record
    }


@serialized_handoff
def advance_policy_canary_handoff(
    *,
    state_root: str | Path,
    source_launch_id: str,
    expected_production_commit: str,
    launch_state_root: str | Path,
    episode_compilation_queue_root: str | Path,
    activation_intent_root: str | Path,
    repo_root: str | Path,
    profile_dir: str | Path,
    webapp_catalog_out: str | Path,
    webapp_secret: bytes,
    webapp_endpoint: str,
    notification_email: str,
    publisher_factory: Callable[[], Publisher],
    profile_publisher: ProfilePublisher | None = None,
    poster: Poster = default_poster,
    attach: Callable[..., Path] = default_attach,
    presubmission: Callable[..., Mapping[str, Any]] = materialize_policy_canary_presubmission_setup,
    model_rights_materializer: Callable[..., Mapping[str, Any]] = materialize_policy_canary_model_rights,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Advance one configured scene from completed controls to a submitted Quick-10 canary."""

    state = Path(state_root).expanduser()
    launches = Path(launch_state_root).expanduser()
    repo = Path(repo_root).expanduser()
    if _COMMIT.fullmatch(str(expected_production_commit or "")) is None or _IDENTIFIER.fullmatch(source_launch_id) is None:
        raise PolicyCanaryHandoffError("policy_canary_handoff_identity_invalid")
    if not str(notification_email or "").strip() or "@" not in notification_email:
        raise PolicyCanaryHandoffError("policy_canary_handoff_notification_email_invalid")
    state_path = state / STATE_FILENAME
    existing = _sealed_progression(state_path, statuses=set(_STAGES))
    if existing is not None and (existing.get("source_launch_id") != source_launch_id
            or existing.get("expected_production_commit") != expected_production_commit):
        raise PolicyCanaryHandoffError("policy_canary_handoff_progression_invalid")

    controls_launch = _sealed_progression(
        state / "controls_launch_progression.json", statuses={"controls_pair_launch_queued"}
    )
    if controls_launch is None:
        return {"status": "awaiting_controls_launch", "source_launch_id": source_launch_id}
    construction_launch = _sealed_progression(
        state / "construction_launch_progression.json", statuses={"construction_launch_queued"}
    )
    if construction_launch is None:
        raise PolicyCanaryHandoffError("policy_canary_handoff_construction_launch_missing")
    controls_launch_id = str(controls_launch["launch_id"])
    construction_launch_id = str(construction_launch["launch_id"])

    base = _base_progression(state)
    episode_request = dict(base["episode_preparation_request"])
    team_namespace = str(episode_request.get("team_namespace") or "")
    scene_id = str(((episode_request.get("scene") or {}).get("identity") or {}).get("id") or "")
    task_id = str(((episode_request.get("task") or {}).get("identity") or {}).get("id") or "")
    preparation_id = str(episode_request.get("preparation_id") or "")
    numeric_scene_id = scene_id.removeprefix("interiorgs-")
    if not (team_namespace and scene_id and task_id and preparation_id) or not re.fullmatch(r"[0-9]{1,12}", numeric_scene_id):
        raise PolicyCanaryHandoffError("policy_canary_handoff_base_progression_invalid")
    intent = load_scene_configuration_activation_intent(
        intent_root=activation_intent_root, team_namespace=team_namespace, scene_id=scene_id, task_id=task_id
    )
    if intent is None:
        return {
            "status": "awaiting_scene_configuration_activation_intent",
            "source_launch_id": source_launch_id,
            "team_namespace": team_namespace,
            "scene_id": scene_id,
            "task_id": task_id,
        }
    if (intent["expected_production_commit"] != expected_production_commit
            or intent["configuration_source_commit"] != expected_production_commit):
        raise PolicyCanaryHandoffError("policy_canary_handoff_activation_intent_commit_mismatch")
    if existing is not None and existing.get("status") == "canary_launch_submitted":
        verify_completed_ack(state, existing)
        return _row(existing)
    publisher = publisher_factory()
    predecessor = _predecessor_lineage(
        launch_state_root=launches,
        controls_launch_id=controls_launch_id,
        construction_launch_id=construction_launch_id,
        publisher=publisher,
    )
    if predecessor is None:
        return {
            "status": "awaiting_controls_terminal",
            "source_launch_id": source_launch_id,
            "controls_launch_id": controls_launch_id,
        }
    lineage, _published_paths = predecessor
    request_path, launch_request, base_profile_path, _base_profile = _configured_run(launches, source_launch_id)
    owner = scene_policy.owner_for_profile(_base_profile,
        now=now.timestamp() if now is not None else None)
    compiled = _compiled_construction(
        Path(episode_compilation_queue_root).expanduser(),
        preparation_id=preparation_id,
        expected_production_commit=expected_production_commit,
    )
    if compiled is None:
        return {"status": "awaiting_construction_compilation", "source_launch_id": source_launch_id}

    inputs = state / "policy-canary-inputs"
    profile_id = f"{source_launch_id}-internal-policy-canary-{expected_production_commit[:10]}"
    if existing is None:
        authorization = dict(intent["authorization_template"])
        if owner is not None:
            remaining = int(owner["request"]["execution"]["expires_at_epoch"] -
                (now or datetime.now(timezone.utc)).timestamp())
            if remaining < 300:
                raise PolicyCanaryHandoffError("scene_policy_owner_authority_window_too_short")
            authorization["valid_for_seconds"] = min(authorization["valid_for_seconds"], remaining)
        native_controller_path = repo / MANIFESTS["native_controller_configuration"]
        controller_document = rebind_policy_controller_configuration_to_scene(
            template_path=repo / MANIFESTS["policy_controller_template"],
            scene_id=numeric_scene_id,
            task_id=task_id,
            scene_revision_digest=str(base["configured_scene_revision_digest"]),
        )
        controller_path = inputs / f"scene{numeric_scene_id}_policy_canary_controller_configuration.v1.json"
        _write_or_reuse(controller_path, controller_document)
        rights_path = inputs / f"scene{numeric_scene_id}_policy_canary_model_rights.v1.json"
        if not rights_path.exists():
            model_rights_materializer(
                template_path=str(repo / MANIFESTS["model_rights_template"]),
                repo_root=str(repo),
                source_commit=expected_production_commit,
                scene_id=numeric_scene_id,
                task_id=task_id,
                output_path=str(rights_path),
            )
        window_path = inputs / "policy_canary_release_window_template.v1.json"
        _write_or_reuse(
            window_path,
            _release_window_template(
                team_namespace=team_namespace,
                expected_production_commit=expected_production_commit,
                released_by=str(authorization["authorized_by"]),
                source_launch_id=source_launch_id,
            ),
        )
        prefix = f"policy-canary-inputs/{source_launch_id}/{expected_production_commit[:12]}"
        # The activation automation binds the template by immutable reference; the
        # canary preparation materializes it back as
        # ``policy_canary_activation.release_window_template``.
        window = _publish(path=window_path, object_name=f"{prefix}/{window_path.name}", publisher=publisher)
        parameters = {
            "profile_id": profile_id,
            "source_commit": expected_production_commit,
            "configured_source_launch_id": source_launch_id,
            "configured_offering_configuration_run_id": str(base["configuration_run_id"]),
            "configured_source_commit": str(launch_request["source_commit"]),
            "offering_digest": str(base["configured_scene_offering_digest"]),
            "scene_revision_digest": str(base["configured_scene_revision_digest"]),
            "request_digest": str(launch_request["request_digest"]),
            "launch_request_path": str(request_path),
            "launch_profile_path": str(base_profile_path),
            "configured_progression_path": str(
                (state / "configured_controls_progression.v1.json")
                if (state / "configured_controls_progression.v1.json").is_file()
                else state / "episode" / "configured_controls_progression.v1.json"
            ),
            "scene_plan_path": str(compiled["scene_plan_path"]),
            "packet_receipt_path": str(compiled["packet_receipt_path"]),
            "runtime_source_receipt_path": str(compiled["runtime_source_receipt_path"]),
            "historical_policy_readiness_path": str(repo / MANIFESTS["historical_policy_readiness"]),
            "pi05_checkpoint_inventory_path": str(repo / MANIFESTS["pi05_checkpoint_inventory"]),
            "policy_controller_configuration": _publish(
                path=controller_path, object_name=f"{prefix}/{controller_path.name}", publisher=publisher
            ),
            "native_controller_configuration": _publish(
                path=native_controller_path, object_name=f"{prefix}/{native_controller_path.name}", publisher=publisher
            ),
            "runtime_source_bundle": _reference(
                (episode_request.get("execution_adapter") or {}).get("runtime_source_bundle"),
                blocker="policy_canary_handoff_runtime_source_bundle_invalid",
            ),
            "runtime_source_implementation_commit": expected_production_commit,
            "model_rights": _publish(path=rights_path, object_name=f"{prefix}/{rights_path.name}", publisher=publisher),
            "activation_release_window_template": window,
            "activation_lineage": lineage,
            "activation_authorization": {
                "reference": str(authorization["reference"]),
                "authorized_by": str(authorization["authorized_by"]),
                "profile_revision": str(authorization["profile_revision"]),
                "valid_for_seconds": int(authorization["valid_for_seconds"]),
            },
            "output_dir": str(state / "policy-canary-presubmission"),
            "maximum_hourly_rate_usd": CANARY_HOURLY_RATE_USD,
            "hard_cap_usd": CANARY_HARD_CAP_USD,
            "hard_ttl_seconds": CANARY_HARD_TTL_SECONDS,
            "scene_id": numeric_scene_id,
        }
        parameters = _write_or_reuse(inputs / "presubmission_parameters.json", parameters)
        if parameters.get("profile_id") != profile_id or parameters.get("source_commit") != expected_production_commit:
            raise PolicyCanaryHandoffError("policy_canary_handoff_retained_parameters_mismatch")
        try:
            emitted = dict(presubmission(**parameters))
        except Exception as exc:  # the presubmission raises its own typed blockers
            raise PolicyCanaryHandoffError(f"policy_canary_handoff_presubmission_failed:{exc}") from exc
        setup = dict(emitted["setup"])
        wrapper = dict(emitted["profile_materialization_input"])
        setup_path = Path(str(emitted.get("setup_path") or (state / "policy-canary-presubmission" / "task_evaluation_policy_canary_setup.v1.json")))
        wrapper_path = Path(
            str(
                emitted.get("profile_materialization_input_path")
                or (state / "policy-canary-presubmission" / "task_evaluation_policy_canary_profile_materialization_input.v1.json")
            )
        )
        if not setup_path.is_file() or not wrapper_path.is_file():
            raise PolicyCanaryHandoffError("policy_canary_handoff_presubmission_outputs_missing")
        existing = _seal_state(
            state_path,
            {
                "status": "canary_presubmitted",
                "source_launch_id": source_launch_id,
                "expected_production_commit": expected_production_commit,
                "controls_launch_id": controls_launch_id,
                "profile_id": profile_id,
                "setup_digest": str(setup["setup_digest"]),
                "setup_path": str(setup_path),
                "profile_materialization_input_path": str(wrapper_path),
                "materialization_digest": str(wrapper["materialization_digest"]),
                "compilation_result_digest": compiled["compilation_result_digest"],
            },
        )
    setup_path = Path(str(existing["setup_path"]))
    wrapper_path = Path(str(existing["profile_materialization_input_path"]))
    setup = _load(setup_path, blocker="policy_canary_handoff_setup_invalid")
    if setup.get("setup_digest") != existing.get("setup_digest") or setup.get("setup_digest") != policy_canary_setup_digest(setup):
        raise PolicyCanaryHandoffError("policy_canary_handoff_setup_invalid")
    wrapper = _load(wrapper_path, blocker="policy_canary_handoff_wrapper_invalid")
    execution_plan = wrapper.get("internal_policy_canary_execution_plan") or {}
    if owner is not None:
        binding = scene_policy.validate_owner_binding(_base_profile,
            execution_plan.get("scene_policy_binding") or {}, source_commit=expected_production_commit)
        scene_policy.validate_setup_pair(setup, binding)

    profile_path = state / "policy-canary-profile.json"
    if existing["status"] == "canary_presubmitted":
        attach(repo_root=repo, base_profile_path=base_profile_path, wrapper_path=wrapper_path, output_path=profile_path)
        profile = _load(profile_path, blocker="policy_canary_handoff_profile_invalid")
        if (
            profile.get("profile_id") != profile_id
            or (profile.get("internal_policy_canary_setup") or {}).get("setup_digest") != existing["setup_digest"]
            or not _is_sealed(profile, field="profile_digest")
        ):
            raise PolicyCanaryHandoffError("policy_canary_handoff_profile_invalid")
        publish_profile = profile_publisher or default_profile_publisher(repo_root=repo)
        receipt = dict(
            publish_profile(
                profile_path=profile_path,
                profile_dir=Path(profile_dir).expanduser(),
                webapp_catalog_out=Path(webapp_catalog_out).expanduser(),
            )
        )
        if receipt.get("status") != "published":
            raise PolicyCanaryHandoffError("policy_canary_handoff_profile_publication_failed")
        existing = _seal_state(
            state_path,
            {
                **{key: existing[key] for key in existing if key not in ("schema_version", "status", "provider_mutation_performed", "progression_digest")},
                "status": "canary_profile_published",
                "profile_digest": str(profile["profile_digest"]),
                "webapp_catalog_digest": str(receipt.get("webapp_catalog_digest") or ""),
            },
        )

    run_id = f"{source_launch_id}-policy-canary-{str(existing['setup_digest']).removeprefix('sha256:')[:12]}"
    if _IDENTIFIER.fullmatch(run_id) is None:
        raise PolicyCanaryHandoffError("policy_canary_handoff_run_id_invalid")
    selection = _selection(setup=setup, run_id=run_id, notification_email=notification_email)
    if owner is not None:
        profile = _load(profile_path, blocker="policy_canary_handoff_profile_invalid")
        pair_blockers = scene_policy.profile_binding_blockers(profile)
        if pair_blockers:
            raise PolicyCanaryHandoffError(",".join(pair_blockers))
        selection["episode_interpretation"] = scene_policy.interpretation_for_owner(
            profile=_base_profile, plan=execution_plan, default=EPISODE_INTERPRETATION)
    body = json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
    _write_immutable(state / "policy-canary-selection.json", selection)
    endpoint = webapp_endpoint.rstrip("/") + "/policy-canary-runs/" + urllib.parse.quote(source_launch_id, safe="")
    receipt, receipt_digest = submit_or_adopt(root=state, endpoint=endpoint, selection=selection,
        source_commit=expected_production_commit,
        headers=lambda: signed_headers(secret=webapp_secret, body=body, run_id=run_id, now=now),
        poster=poster)
    existing = _seal_state(
        state_path,
        {
            **{key: existing[key] for key in existing if key not in ("schema_version", "status", "provider_mutation_performed", "progression_digest")},
            "status": "canary_launch_submitted",
            "run_id": run_id,
            "webapp_receipt_digest": receipt_digest,
            "already_exists": bool(receipt.get("already_exists")),
        },
    )
    return _row(existing)


def _read_secret(path: Path) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise PolicyCanaryHandoffError("policy_canary_handoff_secret_invalid")
        secret = path.read_bytes().strip()
    except OSError as exc:
        raise PolicyCanaryHandoffError("policy_canary_handoff_secret_invalid") from exc
    if len(secret) < 16:
        raise PolicyCanaryHandoffError("policy_canary_handoff_secret_invalid")
    return secret


def advance_policy_canary_handoff_for_plan(
    *,
    plan: Mapping[str, Any],
    progression_root: str | Path,
    launch_state_root: str | Path,
    episode_compilation_queue_root: str | Path,
    activation_intent_root: str | Path | None,
    repo_root: str | Path | None,
    webapp_secret_file: str | Path | None,
    webapp_endpoint: str,
    webapp_catalog_out: str | Path | None,
    notification_email: str | None,
    publisher_factory: Callable[[], Publisher],
    **overrides: Any,
) -> dict[str, Any]:
    """Run the hand-off for one configured-controls plan as the progression timer sees it."""

    source_launch_id = str(plan["source_launch_id"])
    commit = str(plan["expected_production_commit"])
    if not (activation_intent_root and repo_root and webapp_secret_file and webapp_catalog_out and notification_email):
        return {"status": "policy_canary_handoff_not_configured", "source_launch_id": source_launch_id}
    state_root = Path(progression_root).expanduser() / source_launch_id / f"franka-controls-{commit[:12]}"
    return advance_policy_canary_handoff(
        state_root=state_root,
        source_launch_id=source_launch_id,
        expected_production_commit=commit,
        launch_state_root=launch_state_root,
        episode_compilation_queue_root=episode_compilation_queue_root,
        activation_intent_root=activation_intent_root,
        repo_root=repo_root,
        profile_dir=str(plan["profile_dir"]),
        webapp_catalog_out=webapp_catalog_out,
        webapp_secret=_read_secret(Path(webapp_secret_file).expanduser()),
        webapp_endpoint=webapp_endpoint,
        notification_email=notification_email,
        publisher_factory=publisher_factory,
        **overrides,
    )
