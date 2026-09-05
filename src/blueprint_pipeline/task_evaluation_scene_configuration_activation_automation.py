"""Activate and launch a Website-started scene configuration from one owner intent.

Between "preparation result landed" and "Website launch fired" the 839873
rehearsal needed five hand-written scripts per attempt: observe provider zero,
copy the project spend baseline, author the activation request and its release
window, publish the governance objects, then sign a WebApp launch.  Each was a
per-run operator decision that could drift.  This module owns those joins as
production code driven by one registered, digest-bound owner intent.

It performs no provider allocation and no paid execution.  It stages one
activation into the existing authority-gated activation queue and submits one
launch through the canonical WebApp channel; the activation worker still
publishes the profile and standing authorization, and the dispatcher still
requires both before any allocator call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess  # nosec B404 - fixed interpreter and repository script argv
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_launch_activation_contract import (
    TaskEvaluationLaunchActivationContractError,
    validate_launch_activation_request,
)
from .task_evaluation_launch_activation_queue import stage_launch_activation_request
from .task_evaluation_scene_configuration_paid_authority import (
    MAX_PROVIDER_ZERO_AGE_SECONDS,
)
from .task_evaluation_shared_mutation_window import (
    TaskEvaluationSharedMutationWindowError,
    materialize_shared_mutation_window,
    validate_shared_mutation_window_template,
)
from .task_evaluation_standing_launch_authorization import (
    StandingAuthorizationError,
    consumption_totals,
    load_standing_authorization,
    validate_standing_authorization,
)
from .vast_evidence_contracts import valid_vast_provider_zero_api_call


INTENT_SCHEMA_VERSION = "task_evaluation_scene_configuration_activation_intent.v1"
PROGRESSION_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_activation_progression.v1"
)
LANE = "task_evaluation_scene_configuration"
AWAITING_ACTIVATION_STATUS = "queued_for_production_scene_configuration"
AUTHORITY_MATERIALIZED_STATUS = "profile_authority_materialized_no_execution"
PROJECT_SPEND_SCHEMA_VERSION = "adp_project_spend_reconciliation.v1"
PROVIDER_ZERO_SCHEMA_VERSION = "adp_paid_provider_zero.v1"
DEFAULT_INTENT_ROOT = (
    "/etc/blueprint/task-evaluation-scene-configuration-activation-intents"
)
DEFAULT_WEBAPP_ENDPOINT = (
    "https://tryblueprint.io/api/internal/task-evaluation-launch-submissions"
)
LINEAGE_KEY_PREFIX = (
    "task-evaluation/production-inputs/scene-configuration-activation-lineage"
)
RELEASE_WINDOW_KEY_PREFIX = (
    "task-evaluation/production-inputs/coordinator-release-windows"
)
SUBMITTED_BY = "blueprint-scene-configuration-activation-automation"
REQUIRED_MUTATIONS = {
    "profile_publication": True,
    "catalog_synchronization": True,
    "standing_authorization": True,
}
PROVIDER_ZERO_MODE = {"mode": "observe_live_before_authorization"}
STATE_DIRECTORY = "scene-configuration-activations"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_PLACEHOLDER_WINDOW = {
    "uri": "https://tryblueprint.io/internal/release-window-placeholder",
    "digest": "sha256:" + "0" * 64,
    "size_bytes": 1,
}

Publisher = Callable[..., Mapping[str, Any]]
PublisherFactory = Callable[[], Publisher]
Submitter = Callable[[Mapping[str, Any]], Mapping[str, Any]]
ProviderZeroCollector = Callable[[], Mapping[str, Any]]


class SceneConfigurationActivationAutomationError(RuntimeError):
    """A join between preparation, authority, and launch could not be made."""


def _load(path: Path, *, blocker: str) -> dict[str, Any]:
    try:
        if path.is_symlink():
            raise SceneConfigurationActivationAutomationError(blocker)
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SceneConfigurationActivationAutomationError(blocker) from exc
    if not isinstance(value, Mapping):
        raise SceneConfigurationActivationAutomationError(blocker)
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_intent_artifact_missing"
        )
    return {
        "path": str(path.resolve()),
        "digest": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
        path.chmod(0o440)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != payload:
            raise SceneConfigurationActivationAutomationError(
                "scene_configuration_activation_immutable_conflict"
            ) from None


def _copy_immutable(source: Path, destination: Path, *, expected: Mapping[str, Any]) -> None:
    if source.is_symlink() or not source.is_file():
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_intent_artifact_missing"
        )
    payload = source.read_bytes()
    if (
        "sha256:" + hashlib.sha256(payload).hexdigest() != expected.get("digest")
        or len(payload) != expected.get("size_bytes")
    ):
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_intent_artifact_drifted"
        )
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
        destination.chmod(0o440)
    except FileExistsError:
        if destination.is_symlink() or destination.read_bytes() != payload:
            raise SceneConfigurationActivationAutomationError(
                "scene_configuration_activation_immutable_conflict"
            ) from None


def _sealed(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    sealed = dict(value)
    sealed[field] = ""
    sealed[field] = canonical_digest(sealed, digest_field=field)
    return sealed


def _is_sealed(value: Mapping[str, Any], *, field: str) -> bool:
    return value.get(field) == canonical_digest(value, digest_field=field)


def _parse_time(value: Any, *, blocker: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise SceneConfigurationActivationAutomationError(blocker) from exc
    if parsed.tzinfo is None:
        raise SceneConfigurationActivationAutomationError(blocker)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _identifier(value: Any) -> str:
    text = str(value or "")
    if _IDENTIFIER.fullmatch(text) is None:
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_identifier_invalid"
        )
    return text


# --------------------------------------------------------------------------- intent


def scene_configuration_activation_registry_name(
    *, team_namespace: str, scene_id: str, task_id: str
) -> str:
    """Name the registry file exactly as the controls registry names its intent."""

    identity = canonical_digest(
        {"team_namespace": team_namespace, "scene_id": scene_id, "task_id": task_id}
    ).removeprefix("sha256:")
    return f"{identity}.json"


def _validate_authorization_template(value: Any) -> dict[str, Any]:
    template = dict(value) if isinstance(value, Mapping) else {}
    seconds = template.get("valid_for_seconds")
    if (
        set(template) != {"reference", "authorized_by", "profile_revision", "valid_for_seconds"}
        or not str(template.get("reference") or "").strip()
        or len(str(template["reference"])) > 1000
        or _IDENTIFIER.fullmatch(str(template.get("authorized_by") or "")) is None
        or _IDENTIFIER.fullmatch(str(template.get("profile_revision") or "")) is None
        or isinstance(seconds, bool)
        or not isinstance(seconds, int)
        or not 300 <= seconds <= 86_400
    ):
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_intent_authorization_invalid"
        )
    return template


def _intent_body(
    *,
    expected_production_commit: str,
    team_namespace: str,
    scene_id: str,
    task_id: str,
    authorization_template: Mapping[str, Any],
    rights_scope: str,
    inventory: Mapping[str, Mapping[str, Any]],
    enabled: bool,
) -> dict[str, Any]:
    return {
        "schema_version": INTENT_SCHEMA_VERSION,
        "enabled": enabled,
        "expected_production_commit": expected_production_commit,
        "configuration_source_commit": expected_production_commit,
        "team_namespace": team_namespace,
        "scene_id": scene_id,
        "task_id": task_id,
        "lane": LANE,
        "authorization_template": dict(authorization_template),
        "requested_mutations": dict(REQUIRED_MUTATIONS),
        "lineage_kind": "initial_project",
        "provider_zero": dict(PROVIDER_ZERO_MODE),
        "rights_scope": rights_scope,
        "artifact_inventory": {name: dict(row) for name, row in inventory.items()},
        "provider_mutation_performed": False,
        "paid_execution_requested": True,
        "intent_digest": "",
    }


def materialize_scene_configuration_activation_intent(
    *,
    expected_production_commit: str,
    team_namespace: str,
    scene_id: str,
    task_id: str,
    authorization_template: Mapping[str, Any],
    release_window_template_path: str | Path,
    project_spend_reconciliation_path: str | Path,
    rights_scope: str,
    output_path: str | Path,
    enabled: bool = True,
) -> dict[str, Any]:
    """Seal the owner's activation decision for one team/scene/task at one commit."""

    if _COMMIT.fullmatch(str(expected_production_commit or "")) is None:
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_intent_commit_invalid"
        )
    for value in (team_namespace, scene_id, task_id):
        _identifier(value)
    template = _validate_authorization_template(authorization_template)
    if not str(rights_scope or "").strip() or len(str(rights_scope)) > 1000:
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_intent_rights_scope_invalid"
        )
    window_path = Path(release_window_template_path).expanduser()
    try:
        validate_shared_mutation_window_template(
            _load(
                window_path,
                blocker="scene_configuration_activation_intent_release_window_template_invalid",
            ),
            team_namespace=team_namespace,
            expected_production_commit=expected_production_commit,
        )
    except TaskEvaluationSharedMutationWindowError as exc:
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_intent_release_window_template_invalid"
        ) from exc
    spend_path = Path(project_spend_reconciliation_path).expanduser()
    spend = _load(
        spend_path, blocker="scene_configuration_activation_intent_project_spend_invalid"
    )
    if spend.get("schema_version") != PROJECT_SPEND_SCHEMA_VERSION:
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_intent_project_spend_invalid"
        )
    intent = _intent_body(
        expected_production_commit=expected_production_commit,
        team_namespace=team_namespace,
        scene_id=scene_id,
        task_id=task_id,
        authorization_template=template,
        rights_scope=str(rights_scope),
        inventory={
            "release_window_template": _artifact(window_path),
            "project_spend_reconciliation": _artifact(spend_path),
        },
        enabled=bool(enabled),
    )
    intent["intent_digest"] = canonical_digest(intent, digest_field="intent_digest")
    _write_immutable(Path(output_path).expanduser(), intent)
    return intent


def validate_scene_configuration_activation_intent(value: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only an intent whose every field and bound byte still agrees."""

    try:
        intent = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_intent_invalid"
        ) from exc
    inventory = intent.get("artifact_inventory")
    if (
        intent.get("schema_version") != INTENT_SCHEMA_VERSION
        or intent.get("enabled") is not True
        or _COMMIT.fullmatch(str(intent.get("expected_production_commit") or "")) is None
        or intent.get("configuration_source_commit") != intent.get("expected_production_commit")
        or any(
            _IDENTIFIER.fullmatch(str(intent.get(name) or "")) is None
            for name in ("team_namespace", "scene_id", "task_id")
        )
        or intent.get("lane") != LANE
        or intent.get("requested_mutations") != REQUIRED_MUTATIONS
        or intent.get("lineage_kind") != "initial_project"
        or intent.get("provider_zero") != PROVIDER_ZERO_MODE
        or not str(intent.get("rights_scope") or "").strip()
        or len(str(intent.get("rights_scope"))) > 1000
        or not isinstance(inventory, Mapping)
        or set(inventory) != {"release_window_template", "project_spend_reconciliation"}
        or intent.get("provider_mutation_performed") is not False
        or intent.get("paid_execution_requested") is not True
        or not _is_sealed(intent, field="intent_digest")
    ):
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_intent_invalid"
        )
    _validate_authorization_template(intent.get("authorization_template"))
    for row in inventory.values():
        if not isinstance(row, Mapping) or set(row) != {"path", "digest", "size_bytes"}:
            raise SceneConfigurationActivationAutomationError(
                "scene_configuration_activation_intent_invalid"
            )
        path = Path(str(row["path"]))
        if not path.is_absolute() or dict(row) != _artifact(path):
            raise SceneConfigurationActivationAutomationError(
                "scene_configuration_activation_intent_artifact_drifted"
            )
    return intent


def load_scene_configuration_activation_intent(
    *, intent_root: str | Path, team_namespace: str, scene_id: str, task_id: str
) -> dict[str, Any] | None:
    """Return the registered intent for one identity, or ``None`` when absent."""

    root = Path(intent_root).expanduser()
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_intent_root_invalid"
        )
    candidate = root / scene_configuration_activation_registry_name(
        team_namespace=team_namespace, scene_id=scene_id, task_id=task_id
    )
    if not candidate.is_file() or candidate.is_symlink():
        return None
    return validate_scene_configuration_activation_intent(
        _load(candidate, blocker="scene_configuration_activation_intent_invalid")
    )


# ---------------------------------------------------------------- provider zero


def _validated_provider_zero(value: Mapping[str, Any]) -> dict[str, Any]:
    zero = dict(value)
    if (
        zero.get("schema_version") != PROVIDER_ZERO_SCHEMA_VERSION
        or zero.get("provider") != "vast"
        or zero.get("api_confirmed") is not True
        or not valid_vast_provider_zero_api_call(zero.get("api_command"))
        or zero.get("raw_secret_values_recorded") is not False
        or not isinstance(zero.get("stderr_present"), bool)
        or not _is_sealed(zero, field="provider_zero_digest")
    ):
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_provider_zero_invalid"
        )
    if (
        zero.get("global_live_resource_count") != 0
        or zero.get("provider_zero") is not True
        or zero.get("inventory") != []
    ):
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_provider_not_zero"
        )
    return zero


def default_provider_zero_collector() -> dict[str, Any]:
    """Observe live authenticated Vast inventory through the production seam."""

    from .adp_task_evaluation_abstention import collect_vast_provider_zero_receipt

    return collect_vast_provider_zero_receipt()


# ---------------------------------------------------------------- publication


def _publish(*, path: Path, object_name: str, publisher: Publisher) -> dict[str, Any]:
    expected_digest = _sha256(path)
    expected_size = path.stat().st_size
    observed = dict(publisher(path=path, object_name=object_name))
    reference = {key: observed.get(key) for key in ("uri", "digest", "size_bytes")}
    if (
        not isinstance(reference["uri"], str)
        or not re.fullmatch(r"(gs|s3|https)://\S+", reference["uri"])
        or reference["digest"] != expected_digest
        or reference["size_bytes"] != expected_size
        or observed.get("full_byte_service_account_readback_passed") is not True
        or observed.get("readback_digest") != expected_digest
        or observed.get("readback_size_bytes") != expected_size
    ):
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_publication_readback_invalid"
        )
    return reference


def lineage_publisher() -> Publisher:
    """Publish lineage objects under the activation worker's admitted prefix."""

    from .task_evaluation_configured_scene_object_store import (
        configured_scene_object_store_publisher,
    )

    return configured_scene_object_store_publisher(key_prefix=LINEAGE_KEY_PREFIX)


def release_window_publisher() -> Publisher:
    """Publish coordinator windows under the activation worker's exact prefix."""

    from .task_evaluation_configured_scene_object_store import (
        configured_scene_object_store_publisher,
    )

    return configured_scene_object_store_publisher(key_prefix=RELEASE_WINDOW_KEY_PREFIX)


# ---------------------------------------------------------------- activation


def _activation_id(preparation_id: str) -> str:
    stem = preparation_id.removesuffix("-preparation")
    return _identifier(f"{stem}-activation-auto")


def _preparation_context(
    *, preparation_result_path: Path, preparation_queue_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = _load(
        preparation_result_path,
        blocker="scene_configuration_activation_preparation_result_invalid",
    )
    if (
        result.get("schema_version") != "task_evaluation_launch_preparation_result.v1"
        or not _is_sealed(result, field="result_digest")
    ):
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_preparation_result_invalid"
        )
    preparation_id = _identifier(result.get("preparation_id"))
    envelope = _load(
        preparation_queue_root / "materialized" / preparation_result_path.name,
        blocker="scene_configuration_activation_preparation_envelope_invalid",
    )
    request = envelope.get("request")
    if (
        not isinstance(request, Mapping)
        or not _is_sealed(envelope, field="envelope_digest")
        or _DIGEST.fullmatch(str(envelope.get("request_digest") or "")) is None
        or request.get("preparation_id") != preparation_id
        or request.get("run_mode") != "scene_configuration"
        or request.get("team_namespace") != result.get("team_namespace")
        or _COMMIT.fullmatch(str(request.get("expected_production_commit") or "")) is None
        or request.get("expected_production_commit") != result.get("source_commit")
        or not isinstance(request.get("spend"), Mapping)
        or not isinstance((request.get("scene") or {}).get("identity"), Mapping)
        or not isinstance((request.get("task") or {}).get("identity"), Mapping)
    ):
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_preparation_envelope_invalid"
        )
    return result, envelope, dict(request)


def _existing_state(path: Path, *, statuses: set[str], **expected: Any) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    existing = _load(path, blocker="scene_configuration_activation_progression_invalid")
    if (
        existing.get("schema_version") != PROGRESSION_SCHEMA_VERSION
        or existing.get("status") not in statuses
        or not _is_sealed(existing, field="progression_digest")
        or any(existing.get(field) != value for field, value in expected.items())
    ):
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_progression_invalid"
        )
    return existing


def advance_scene_configuration_activation(
    *,
    preparation_result_path: str | Path,
    preparation_queue_root: str | Path,
    activation_queue_root: str | Path,
    progression_root: str | Path,
    intent_root: str | Path,
    provider_zero_collector: ProviderZeroCollector = default_provider_zero_collector,
    lineage_publisher_factory: PublisherFactory = lineage_publisher,
    release_window_publisher_factory: PublisherFactory = release_window_publisher,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Stage exactly one authority-gated activation for a prepared configuration."""

    result_path = Path(preparation_result_path).expanduser()
    queue_root = Path(preparation_queue_root).expanduser()
    result, envelope, request = _preparation_context(
        preparation_result_path=result_path, preparation_queue_root=queue_root
    )
    preparation_id = str(result["preparation_id"])
    if result.get("run_mode") != "scene_configuration" or result.get("status") != AWAITING_ACTIVATION_STATUS:
        return {
            "status": "preparation_not_awaiting_scene_configuration",
            "preparation_id": preparation_id,
            "preparation_status": result.get("status"),
        }
    state_root = Path(progression_root).expanduser() / STATE_DIRECTORY / preparation_id
    state_path = state_root / "activation_progression.json"
    existing = _existing_state(
        state_path,
        statuses={"scene_configuration_activation_queued"},
        preparation_id=preparation_id,
        preparation_result_digest=result["result_digest"],
    )
    if existing is not None:
        return existing
    scene_id = str(request["scene"]["identity"].get("id") or "")
    task_id = str(request["task"]["identity"].get("id") or "")
    team_namespace = str(request["team_namespace"])
    intent = load_scene_configuration_activation_intent(
        intent_root=intent_root, team_namespace=team_namespace, scene_id=scene_id, task_id=task_id
    )
    if intent is None:
        return {
            "status": "awaiting_scene_configuration_activation_intent",
            "preparation_id": preparation_id,
            "team_namespace": team_namespace,
            "scene_id": scene_id,
            "task_id": task_id,
        }
    commit = str(request["expected_production_commit"])
    if intent["expected_production_commit"] != commit:
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_intent_commit_mismatch"
        )
    zero = _validated_provider_zero(provider_zero_collector())
    zero_time = _parse_time(
        zero.get("observed_at_utc"),
        blocker="scene_configuration_activation_provider_zero_invalid",
    )
    observed_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(microsecond=0)
    # The paid authority refuses a zero observed after the authorization it
    # precedes, and second-precision authorization stamps can read as older
    # than a zero minted moments earlier: anchor strictly after the zero.
    if observed_now <= zero_time:
        observed_now = (zero_time + timedelta(seconds=1)).replace(microsecond=0)
    if (observed_now - zero_time).total_seconds() > MAX_PROVIDER_ZERO_AGE_SECONDS:
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_provider_zero_stale"
        )
    activation_id = _activation_id(preparation_id)
    state_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    zero_path = state_root / "adp_paid_provider_zero.v1.json"
    _write_immutable(zero_path, zero)
    spend_row = intent["artifact_inventory"]["project_spend_reconciliation"]
    spend_path = state_root / "project_spend_reconciliation.json"
    _copy_immutable(Path(spend_row["path"]), spend_path, expected=spend_row)
    publisher = lineage_publisher_factory()
    zero_reference = _publish(
        path=zero_path,
        object_name=f"{activation_id}/adp_paid_provider_zero.v1.json",
        publisher=publisher,
    )
    spend_reference = _publish(
        path=spend_path,
        object_name=f"{activation_id}/project_spend_reconciliation.json",
        publisher=publisher,
    )
    template_row = intent["artifact_inventory"]["release_window_template"]
    authorization_template = intent["authorization_template"]
    valid_for = int(authorization_template["valid_for_seconds"])
    base_request = {
        "schema_version": "task_evaluation_launch_activation_request.v1",
        "expected_production_commit": commit,
        "activation_id": activation_id,
        "team_namespace": team_namespace,
        "lane": LANE,
        "preparation": {
            "preparation_id": preparation_id,
            "request_digest": envelope["request_digest"],
            "result_digest": result["result_digest"],
        },
        "release_window": dict(_PLACEHOLDER_WINDOW),
        "lineage": {
            "kind": "initial_project",
            "project_spend_reconciliation": spend_reference,
            "initial_provider_zero": zero_reference,
        },
        "authorization": {
            "reference": authorization_template["reference"],
            "authorized_by": authorization_template["authorized_by"],
            "authorized_on": _iso(observed_now),
            "standing_authorization_expires_at": _iso(
                observed_now + timedelta(seconds=valid_for)
            ),
            "profile_revision": authorization_template["profile_revision"],
        },
        "requested_mutations": dict(intent["requested_mutations"]),
    }
    try:
        base_request = validate_launch_activation_request(base_request)
        template = validate_shared_mutation_window_template(
            _load(
                Path(template_row["path"]),
                blocker="scene_configuration_activation_release_window_template_invalid",
            ),
            team_namespace=team_namespace,
            expected_production_commit=commit,
        )
        window = materialize_shared_mutation_window(
            template,
            activation_request=base_request,
            provider_allowlist=list(request["spend"]["provider_allowlist"]),
            hard_cap_usd=float(request["spend"]["hard_cap_usd"]),
            now=observed_now,
        )
    except TaskEvaluationLaunchActivationContractError as exc:
        raise SceneConfigurationActivationAutomationError(
            f"scene_configuration_activation_request_invalid:{exc}"
        ) from exc
    except TaskEvaluationSharedMutationWindowError as exc:
        raise SceneConfigurationActivationAutomationError(
            f"scene_configuration_activation_release_window_invalid:{exc}"
        ) from exc
    window_path = state_root / f"{window['window_id']}.json"
    _write_immutable(window_path, window)
    window_reference = _publish(
        path=window_path,
        object_name=(
            f"{activation_id}/{str(window['window_digest']).removeprefix('sha256:')}.json"
        ),
        publisher=release_window_publisher_factory(),
    )
    activation_request = validate_launch_activation_request(
        {**base_request, "release_window": window_reference}
    )
    intake = dict(
        stage_launch_activation_request(
            value=activation_request,
            queue_root=activation_queue_root,
            submitted_by=SUBMITTED_BY,
        )
    )
    if (
        intake.get("status") != "queued_for_authority_gated_activation"
        or intake.get("accepted") is not True
        or intake.get("activation_id") != activation_id
    ):
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_intake_invalid"
        )
    state = _sealed(
        {
            "schema_version": PROGRESSION_SCHEMA_VERSION,
            "status": "scene_configuration_activation_queued",
            "preparation_id": preparation_id,
            "run_id": str(request.get("run_id") or ""),
            "team_namespace": team_namespace,
            "scene_id": scene_id,
            "task_id": task_id,
            "expected_production_commit": commit,
            "activation_id": activation_id,
            "intent_digest": intent["intent_digest"],
            "rights_scope": intent["rights_scope"],
            "preparation_result_digest": result["result_digest"],
            "preparation_request_digest": envelope["request_digest"],
            "provider_zero_digest": zero["provider_zero_digest"],
            "project_spend_reconciliation_digest": spend_reference["digest"],
            "release_window_digest": window["window_digest"],
            "activation_request_digest": canonical_digest(activation_request),
            "activation_intake_receipt_digest": intake.get("receipt_digest"),
            "provider_mutation_performed": False,
            "paid_execution_requested": False,
        },
        field="progression_digest",
    )
    _write_immutable(state_path, state)
    return state


# ---------------------------------------------------------------- launch


def _bounded_launch_id(activation_id: str) -> str:
    readable = activation_id + "-launch"
    if _IDENTIFIER.fullmatch(readable) is not None:
        return readable
    prefix = activation_id[:150].rstrip("._-")
    token = hashlib.sha256(activation_id.encode("utf-8")).hexdigest()[:24]
    return _identifier(f"{prefix}-{token}-launch")


def _materialized_activation(
    *, activation_queue_root: Path, activation_id: str
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    results = activation_queue_root / "results"
    for path in sorted(results.glob("*.json")) if results.is_dir() else []:
        try:
            value = _load(path, blocker="scene_configuration_launch_activation_result_invalid")
        except SceneConfigurationActivationAutomationError:
            continue
        if (
            value.get("activation_id") == activation_id
            and value.get("status") == AUTHORITY_MATERIALIZED_STATUS
        ):
            matches.append(value)
    if not matches:
        return None
    if len(matches) != 1:
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_launch_activation_result_ambiguous"
        )
    return matches[0]


def webapp_submitter(
    *, repo_root: str | Path, secret_file: str | Path, endpoint: str, state_root: str | Path
) -> Submitter:
    """Submit through the canonical WebApp client; the client owns signing."""

    root = Path(repo_root).expanduser()
    state = Path(state_root).expanduser()

    def submit(request: Mapping[str, Any]) -> Mapping[str, Any]:
        launch_id = str(request["launch_id"])
        request_path = state / f"{launch_id}.webapp-request.json"
        receipt_path = state / f"{launch_id}.webapp-submission.json"
        _write_immutable(request_path, request)
        if not receipt_path.exists():
            completed = subprocess.run(  # nosec B603 - fixed interpreter and repository script
                [
                    sys.executable,
                    str(root / "scripts" / "submit_task_evaluation_launch_via_webapp.py"),
                    "--request",
                    str(request_path),
                    "--secret-file",
                    str(Path(secret_file).expanduser()),
                    "--receipt-out",
                    str(receipt_path),
                    "--endpoint",
                    endpoint,
                ],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if completed.returncode != 0:
                raise SceneConfigurationActivationAutomationError(
                    "scene_configuration_launch_webapp_submission_failed"
                )
        evidence = _load(
            receipt_path, blocker="scene_configuration_launch_webapp_receipt_invalid"
        )
        web = evidence.get("webapp_receipt")
        if (
            evidence.get("status") not in {"submitted", "replayed"}
            or evidence.get("launch_id") != launch_id
            or not isinstance(web, Mapping)
            or web.get("provider_mutation_performed_inside_web_request") is not False
        ):
            raise SceneConfigurationActivationAutomationError(
                "scene_configuration_launch_webapp_receipt_invalid"
            )
        return {
            "status": "submitted" if evidence["status"] == "submitted" else "accepted",
            "launch_id": launch_id,
            "provider_mutation_performed_inside_web_request": False,
        }

    return submit


def advance_scene_configuration_launch(
    *,
    progression: Mapping[str, Any],
    activation_queue_root: str | Path,
    profile_dir: str | Path,
    standing_authorization_dir: str | Path,
    progression_root: str | Path,
    intent: Mapping[str, Any],
    submitter: Submitter,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fire the published profile once through the WebApp under its standing authority."""

    state = dict(progression)
    if (
        state.get("schema_version") != PROGRESSION_SCHEMA_VERSION
        or state.get("status") != "scene_configuration_activation_queued"
        or not _is_sealed(state, field="progression_digest")
        or intent.get("intent_digest") != state.get("intent_digest")
    ):
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_launch_progression_invalid"
        )
    preparation_id = str(state["preparation_id"])
    activation_id = str(state["activation_id"])
    launch_state_path = (
        Path(progression_root).expanduser() / STATE_DIRECTORY / preparation_id / "launch_progression.json"
    )
    existing = _existing_state(
        launch_state_path,
        statuses={"scene_configuration_launch_queued"},
        preparation_id=preparation_id,
        activation_id=activation_id,
    )
    if existing is not None:
        return existing
    activation = _materialized_activation(
        activation_queue_root=Path(activation_queue_root).expanduser(),
        activation_id=activation_id,
    )
    if activation is None:
        return {
            "status": "awaiting_scene_configuration_authority",
            "preparation_id": preparation_id,
            "activation_id": activation_id,
        }
    commit = str(state["expected_production_commit"])
    profile_id = str(activation.get("profile_id") or "")
    if (
        activation.get("schema_version") != "task_evaluation_launch_activation_result.v1"
        or activation.get("lane") != LANE
        or activation.get("source_commit") != commit
        or activation.get("preparation_id") != preparation_id
        or _IDENTIFIER.fullmatch(profile_id) is None
        or _DIGEST.fullmatch(str(activation.get("profile_digest") or "")) is None
        or activation.get("provider_mutation_performed") is not False
        or activation.get("paid_execution_requested") is not False
        or activation.get("blockers") not in ([], ())
        or not _is_sealed(activation, field="result_digest")
    ):
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_launch_activation_result_invalid"
        )
    profile_path = Path(profile_dir).expanduser() / f"{profile_id}.json"
    profile = _load(profile_path, blocker="scene_configuration_launch_profile_invalid")
    evidence = profile.get("evaluation_run_spec")
    if (
        profile.get("schema_version") != "task_evaluation_launch_profile.v1"
        or profile.get("profile_id") != profile_id
        or profile.get("source_commit") != commit
        or profile.get("profile_digest") != activation["profile_digest"]
        or not _is_sealed(profile, field="profile_digest")
        or not isinstance(evidence, Mapping)
        or not str(evidence.get("uri") or "").strip()
        or _DIGEST.fullmatch(str(evidence.get("digest") or "")) is None
    ):
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_launch_profile_invalid"
        )
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    try:
        standing = load_standing_authorization(
            profile_id=profile_id, directory=standing_authorization_dir
        )
        if standing is None:
            raise SceneConfigurationActivationAutomationError(
                "scene_configuration_launch_authority_missing"
            )
        launches, spent = consumption_totals(
            directory=standing_authorization_dir, profile_id=profile_id
        )
        blockers = validate_standing_authorization(
            standing,
            profile=profile,
            launches_consumed=launches,
            spend_consumed_usd=spent,
            now=moment,
        )
    except StandingAuthorizationError as exc:
        raise SceneConfigurationActivationAutomationError(
            f"scene_configuration_launch_authority_mismatch:{exc}"
        ) from exc
    max_spend = standing.get("max_total_spend_usd")
    profile_cap = (profile.get("allocator") or {}).get("max_spend_usd")
    if (
        blockers
        or standing.get("profile_digest") != profile["profile_digest"]
        or isinstance(max_spend, bool)
        or not isinstance(max_spend, (int, float))
        or not math.isfinite(float(max_spend))
        or not 0 < float(max_spend) <= float(profile_cap or 0)
        or not str(standing.get("expires_at") or "").strip()
    ):
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_launch_authority_mismatch"
            + (":" + ",".join(blockers) if blockers else "")
        )
    launch_id = _bounded_launch_id(activation_id)
    launch_request = {
        "launch_id": launch_id,
        "run_id": launch_id,
        "profile_id": profile_id,
        "profile_digest": profile["profile_digest"],
        "rights": {
            "scope": str(intent["rights_scope"]),
            "evidence": {"uri": str(evidence["uri"]), "digest": str(evidence["digest"])},
        },
        "spend": {
            "max_spend_usd": float(max_spend),
            "expires_at": str(standing["expires_at"]),
        },
        "confirm_execution": True,
    }
    response = dict(submitter(launch_request))
    if (
        response.get("status") not in {"submitted", "accepted", "queued"}
        or response.get("launch_id") != launch_id
        or response.get("provider_mutation_performed_inside_web_request") is not False
    ):
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_launch_webapp_submission_invalid"
        )
    launch_state = _sealed(
        {
            "schema_version": PROGRESSION_SCHEMA_VERSION,
            "status": "scene_configuration_launch_queued",
            "preparation_id": preparation_id,
            "activation_id": activation_id,
            "expected_production_commit": commit,
            "launch_id": launch_id,
            "run_id": launch_id,
            "profile_id": profile_id,
            "profile_digest": profile["profile_digest"],
            "activation_result_digest": activation["result_digest"],
            "standing_authorization_digest": _sha256(
                Path(standing_authorization_dir).expanduser() / f"{profile_id}.json"
            ),
            "submitted_through_webapp": True,
            "provider_mutation_performed_inside_progression": False,
            "paid_execution_requested": True,
        },
        field="progression_digest",
    )
    _write_immutable(launch_state_path, launch_state)
    return launch_state


# ---------------------------------------------------------------- orchestration


def _awaiting_scene_configurations(preparation_queue_root: Path) -> list[Path]:
    results = preparation_queue_root / "results"
    selected: list[Path] = []
    for path in sorted(results.glob("*.json")) if results.is_dir() else []:
        if path.is_symlink():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if (
            isinstance(value, Mapping)
            and value.get("run_mode") == "scene_configuration"
            and value.get("status") == AWAITING_ACTIVATION_STATUS
        ):
            selected.append(path)
    return selected


def process_scene_configuration_activations(
    *,
    preparation_queue_root: str | Path,
    activation_queue_root: str | Path,
    progression_root: str | Path,
    intent_root: str | Path,
    profile_dir: str | Path,
    standing_authorization_dir: str | Path,
    provider_zero_collector: ProviderZeroCollector = default_provider_zero_collector,
    lineage_publisher_factory: PublisherFactory = lineage_publisher,
    release_window_publisher_factory: PublisherFactory = release_window_publisher,
    submitter: Submitter | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Advance every prepared configuration one safe step; one row per preparation."""

    rows: list[dict[str, Any]] = []
    queue_root = Path(preparation_queue_root).expanduser()
    for result_path in _awaiting_scene_configurations(queue_root):
        preparation_id = result_path.name.split("-sha256", 1)[0]
        try:
            activation = advance_scene_configuration_activation(
                preparation_result_path=result_path,
                preparation_queue_root=queue_root,
                activation_queue_root=activation_queue_root,
                progression_root=progression_root,
                intent_root=intent_root,
                provider_zero_collector=provider_zero_collector,
                lineage_publisher_factory=lineage_publisher_factory,
                release_window_publisher_factory=release_window_publisher_factory,
                now=now,
            )
            preparation_id = str(activation.get("preparation_id") or preparation_id)
            if activation.get("status") != "scene_configuration_activation_queued":
                rows.append({"preparation_id": preparation_id, **activation})
                continue
            if submitter is None:
                raise SceneConfigurationActivationAutomationError(
                    "scene_configuration_launch_submitter_missing"
                )
            intent = load_scene_configuration_activation_intent(
                intent_root=intent_root,
                team_namespace=str(activation["team_namespace"]),
                scene_id=str(activation["scene_id"]),
                task_id=str(activation["task_id"]),
            )
            if intent is None:
                raise SceneConfigurationActivationAutomationError(
                    "scene_configuration_activation_intent_withdrawn"
                )
            launch = advance_scene_configuration_launch(
                progression=activation,
                activation_queue_root=activation_queue_root,
                profile_dir=profile_dir,
                standing_authorization_dir=standing_authorization_dir,
                progression_root=progression_root,
                intent=intent,
                submitter=submitter,
                now=now,
            )
            rows.append(
                {
                    "preparation_id": preparation_id,
                    "status": launch["status"],
                    "activation_status": activation["status"],
                    "activation_id": activation["activation_id"],
                    "launch_id": launch.get("launch_id"),
                    "provider_mutation_performed": False,
                }
            )
        except SceneConfigurationActivationAutomationError as exc:
            rows.append(
                {"preparation_id": preparation_id, "status": "blocked", "blockers": [str(exc)]}
            )
    return rows


# ---------------------------------------------------------------- CLI


def _env_default(name: str, fallback: str | None = None) -> str | None:
    value = str(os.getenv(name) or "").strip()
    return value or fallback


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    provision = commands.add_parser(
        "provision-intent", help="Register one owner activation intent (run as the intent root owner)."
    )
    provision.add_argument("--expected-production-commit", required=True)
    provision.add_argument("--team-namespace", required=True)
    provision.add_argument("--scene-id", required=True)
    provision.add_argument("--task-id", required=True)
    provision.add_argument("--authorization-reference", required=True)
    provision.add_argument("--authorized-by", required=True)
    provision.add_argument("--profile-revision", required=True)
    provision.add_argument("--valid-for-seconds", type=int, required=True)
    provision.add_argument("--release-window-template", required=True)
    provision.add_argument("--project-spend-reconciliation", required=True)
    provision.add_argument("--rights-scope", default="internal_noncommercial_research_only")
    provision.add_argument("--intent-root", default=DEFAULT_INTENT_ROOT)
    process = commands.add_parser(
        "process", help="Advance every prepared scene configuration one safe step."
    )
    process.add_argument(
        "--preparation-queue-root",
        default=_env_default("BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_QUEUE_ROOT"),
    )
    process.add_argument(
        "--activation-queue-root",
        default=_env_default("BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_QUEUE_ROOT"),
    )
    process.add_argument(
        "--progression-root",
        default=_env_default("BLUEPRINT_TASK_EVALUATION_CONFIGURED_CONTROLS_STATE_ROOT"),
    )
    process.add_argument(
        "--intent-root",
        default=_env_default(
            "BLUEPRINT_TASK_EVALUATION_SCENE_CONFIGURATION_ACTIVATION_INTENT_ROOT",
            DEFAULT_INTENT_ROOT,
        ),
    )
    process.add_argument(
        "--profile-dir", default=_env_default("BLUEPRINT_TASK_EVALUATION_LAUNCH_PROFILE_DIR")
    )
    process.add_argument(
        "--standing-authorization-dir",
        default=_env_default("BLUEPRINT_TASK_EVALUATION_STANDING_AUTHORIZATION_DIR"),
    )
    process.add_argument("--repo-root", default=_env_default("BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO"))
    process.add_argument(
        "--webapp-secret-file",
        default=_env_default("BLUEPRINT_TASK_EVALUATION_WEBAPP_SUBMISSION_SECRET_FILE"),
    )
    process.add_argument(
        "--webapp-endpoint",
        default=_env_default("BLUEPRINT_TASK_EVALUATION_WEBAPP_SUBMISSION_URL", DEFAULT_WEBAPP_ENDPOINT),
    )
    args = parser.parse_args(argv)
    if args.command == "provision-intent":
        root = Path(args.intent_root).expanduser()
        intent = materialize_scene_configuration_activation_intent(
            expected_production_commit=args.expected_production_commit,
            team_namespace=args.team_namespace,
            scene_id=args.scene_id,
            task_id=args.task_id,
            authorization_template={
                "reference": args.authorization_reference,
                "authorized_by": args.authorized_by,
                "profile_revision": args.profile_revision,
                "valid_for_seconds": args.valid_for_seconds,
            },
            release_window_template_path=args.release_window_template,
            project_spend_reconciliation_path=args.project_spend_reconciliation,
            rights_scope=args.rights_scope,
            output_path=root
            / scene_configuration_activation_registry_name(
                team_namespace=args.team_namespace, scene_id=args.scene_id, task_id=args.task_id
            ),
        )
        print(json.dumps({"status": "registered", "intent_digest": intent["intent_digest"]}))
        return 0
    required = {
        "preparation_queue_root": args.preparation_queue_root,
        "activation_queue_root": args.activation_queue_root,
        "progression_root": args.progression_root,
        "profile_dir": args.profile_dir,
        "standing_authorization_dir": args.standing_authorization_dir,
        "repo_root": args.repo_root,
        "webapp_secret_file": args.webapp_secret_file,
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        parser.error("missing: " + ", ".join(missing))
    rows = process_scene_configuration_activations(
        preparation_queue_root=args.preparation_queue_root,
        activation_queue_root=args.activation_queue_root,
        progression_root=args.progression_root,
        intent_root=args.intent_root,
        profile_dir=args.profile_dir,
        standing_authorization_dir=args.standing_authorization_dir,
        submitter=webapp_submitter(
            repo_root=args.repo_root,
            secret_file=args.webapp_secret_file,
            endpoint=args.webapp_endpoint,
            state_root=Path(args.progression_root) / STATE_DIRECTORY / "webapp-submissions",
        ),
    )
    report = {
        "schema_version": "task_evaluation_scene_configuration_activation_automation_report.v1",
        "status": "blocked" if any(row.get("status") == "blocked" for row in rows) else "completed",
        "rows": rows,
        "provider_mutation_performed": False,
        "allocator_invoked": False,
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "completed" else 2


__all__ = [
    "AWAITING_ACTIVATION_STATUS",
    "DEFAULT_INTENT_ROOT",
    "INTENT_SCHEMA_VERSION",
    "LANE",
    "LINEAGE_KEY_PREFIX",
    "PROGRESSION_SCHEMA_VERSION",
    "RELEASE_WINDOW_KEY_PREFIX",
    "SceneConfigurationActivationAutomationError",
    "advance_scene_configuration_activation",
    "advance_scene_configuration_launch",
    "default_provider_zero_collector",
    "lineage_publisher",
    "load_scene_configuration_activation_intent",
    "main",
    "materialize_scene_configuration_activation_intent",
    "process_scene_configuration_activations",
    "release_window_publisher",
    "scene_configuration_activation_registry_name",
    "validate_scene_configuration_activation_intent",
    "webapp_submitter",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
