"""Authority-gated worker for Task Evaluation profile activation.

The worker consumes one verified preparation plus one exact coordinator release
window, builds the existing native-Arena paid-lane preparation context, and
invokes the canonical no-allocation preparation graph.  It may publish an
immutable profile, synchronize the catalog, and publish one standing
authorization.  It never submits a Task Evaluation request, consumes the
authorization, or allocates a provider resource.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pwd
import re
import subprocess  # nosec B404 - fixed executable and repository-owned script
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .core.common import sha256_file as _sha256_file_hex
from .decision_evidence_contracts import canonical_digest
from . import control_plane_disk_budget as disk_budget
from .task_evaluation_configured_scene_revision import (
    TaskEvaluationConfiguredSceneRevisionError,
    validate_configured_scene_revision,
)
from .task_evaluation_configured_controls_autostart import (
    configured_controls_autostart_registry_name,
    validate_configured_controls_autostart_intent,
)
from .task_evaluation_launch_activation_contract import (
    launch_activation_intent_digest,
    validate_launch_activation_request,
)
from .task_evaluation_launch_activation_queue import (
    ENVELOPE_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    ensure_launch_activation_queue_root,
)
from .task_evaluation_launch_preparation_contract import (
    validate_launch_preparation_request,
)
from .task_evaluation_launch_preparation_queue import (
    ENVELOPE_SCHEMA_VERSION as PREPARATION_ENVELOPE_SCHEMA_VERSION,
    write_launch_preparation_record_exclusive,
)
from .task_evaluation_launch_preparation_worker import (
    RESULT_SCHEMA_VERSION as PREPARATION_RESULT_SCHEMA_VERSION,
    TaskEvaluationLaunchPreparationWorkerError,
    default_reference_fetcher,
    running_worker_source_commit,
    validate_allowed_uri_prefixes,
)
from .task_evaluation_policy_run_contract import (
    TaskEvaluationPolicyRunContractError,
    build_policy_campaign_activation_manifest,
)
from .task_evaluation_native_arena_preparation_adapter import (
    RESULT_SCHEMA_VERSION as ADAPTER_RESULT_SCHEMA_VERSION,
    control_search_warm_retention_requested as _control_search_warm_retention_requested,
)
from .task_evaluation_shared_mutation_window import (
    TaskEvaluationSharedMutationWindowError,
    validate_shared_mutation_window,
)
from .task_evaluation_scene_configuration_builtin_producers import (
    TOOLCHAIN_ROOT_ENV as SCENE_CONFIGURATION_TOOLCHAIN_ROOT_ENV,
    validate_scene_configuration_toolchain,
)
from .task_evaluation_scene_construction_queue import (
    ENVELOPE_SCHEMA_VERSION as SCENE_CONSTRUCTION_ENVELOPE_SCHEMA_VERSION,
    QUEUE_STATES as SCENE_CONSTRUCTION_QUEUE_STATES,
)
from .task_evaluation_scene_configuration_disclosure import renders_on_provider


QUEUE_ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_QUEUE_ROOT"
PREPARATION_QUEUE_ROOT_ENV = (
    "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_QUEUE_ROOT"
)
PREPARATION_INPUT_ROOT_ENV = (
    "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_INPUT_ROOT"
)
EPISODE_COMPILATION_QUEUE_ROOT_ENV = (
    "BLUEPRINT_TASK_EVALUATION_EPISODE_COMPILATION_QUEUE_ROOT"
)
EPISODE_COMPILATION_OUTPUT_ROOT_ENV = (
    "BLUEPRINT_TASK_EVALUATION_EPISODE_COMPILATION_OUTPUT_ROOT"
)
SCENE_CONSTRUCTION_QUEUE_ROOT_ENV = (
    "BLUEPRINT_TASK_EVALUATION_SCENE_CONSTRUCTION_QUEUE_ROOT"
)
ACTIVATION_ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_ROOT"
ALLOWED_URI_PREFIXES_ENV = (
    "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_ALLOWED_URI_PREFIXES_JSON"
)
SERVICE_ACCOUNT_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_SERVICE_ACCOUNT"
SERVICE_GROUP_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_SERVICE_GROUP"
REPOSITORY_ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO"
DESTINATION_PREFIX_ENV = (
    "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_DESTINATION_PREFIX"
)
RELEASE_WINDOW_PREFIX_ENV = (
    "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_RELEASE_WINDOW_PREFIX"
)
PROFILE_DIR_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_PROFILE_DIR"
WEBAPP_CATALOG_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_PROFILE_CATALOG"
STANDING_AUTHORIZATION_DIR_ENV = (
    "BLUEPRINT_TASK_EVALUATION_STANDING_AUTHORIZATION_DIR"
)
CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT_ENV = (
    "BLUEPRINT_TASK_EVALUATION_CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT"
)
POLICY_CANARY_DISPATCH_QUEUE_ROOT_ENV = (
    "BLUEPRINT_TASK_EVALUATION_POLICY_CANARY_DISPATCH_QUEUE_ROOT"
)

ReferenceFetcher = Callable[[str, Path, int], None]
ActivationPreparer = Callable[..., dict[str, Any]]


class TaskEvaluationLaunchActivationWorkerError(RuntimeError):
    """One authority-gated activation could not be completed safely."""


def _acquire_processing_lease(path: Path) -> Any:
    """Hold one claim inode until its worker result reaches a terminal state."""

    lease = path.open("rb")
    try:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lease.close()
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_processing_lease_unavailable"
        ) from None
    return lease


def validate_release_window_uri(uri: str, *, prefix: str) -> str:
    """Require coordinator windows to come from an operator-owned prefix."""

    validated_prefix = validate_allowed_uri_prefixes([prefix])[0]
    if not uri.startswith(validated_prefix):
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_release_window_prefix_not_authorized"
        )
    return validated_prefix


def _sha256_file(path: Path) -> str:
    return "sha256:" + _sha256_file_hex(path)


def _load_sealed(
    path: Path, *, schema_version: str, digest_field: str
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_sealed_record_invalid"
        ) from exc
    if (
        path.is_symlink()
        or not path.is_file()
        or not isinstance(value, Mapping)
        or value.get("schema_version") != schema_version
        or value.get(digest_field)
        != canonical_digest(value, digest_field=digest_field)
    ):
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_sealed_record_invalid"
        )
    return dict(value)


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _collect_references(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, Mapping):
            if set(node) == {"uri", "digest", "size_bytes"}:
                rows.append(
                    {
                        "contract_path": ".".join(path),
                        "uri": str(node["uri"]),
                        "digest": str(node["digest"]),
                        "size_bytes": int(node["size_bytes"]),
                    }
                )
                return
            for key, child in node.items():
                visit(child, (*path, str(key)))
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for index, child in enumerate(node):
                visit(child, (*path, str(index)))

    visit(value, ())
    return rows


def _materialize_activation_references(
    *,
    request: Mapping[str, Any],
    root: Path,
    allowed_uri_prefixes: Sequence[str],
    service_account: str,
    fetcher: ReferenceFetcher,
) -> dict[str, Path]:
    try:
        account = pwd.getpwnam(service_account)
    except KeyError as exc:
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_service_account_missing"
        ) from exc
    if os.geteuid() != account.pw_uid:
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_service_account_identity_mismatch"
        )
    prefixes = validate_allowed_uri_prefixes(allowed_uri_prefixes)
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    root = root.resolve(strict=True)
    by_path: dict[str, Path] = {}
    by_identity: dict[tuple[str, int], Path] = {}
    for row in _collect_references(request):
        parsed = urlparse(row["uri"])
        if (
            parsed.scheme not in {"gs", "s3", "https"}
            or not parsed.netloc
            or "@" in parsed.netloc
            or parsed.query
            or parsed.fragment
            or not any(row["uri"].startswith(prefix) for prefix in prefixes)
        ):
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_reference_prefix_not_allowed"
            )
        identity = (row["digest"], row["size_bytes"])
        destination = by_identity.get(identity)
        if destination is None:
            destination = root / row["digest"].removeprefix("sha256:")
            by_identity[identity] = destination
            if destination.is_symlink():
                raise TaskEvaluationLaunchActivationWorkerError(
                    "launch_activation_reference_target_unsafe"
                )
            if not destination.exists():
                temporary = root / (
                    f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
                )
                try:
                    fetcher(row["uri"], temporary, row["size_bytes"])
                    if (
                        temporary.stat().st_size != row["size_bytes"]
                        or _sha256_file(temporary) != row["digest"]
                    ):
                        raise TaskEvaluationLaunchActivationWorkerError(
                            "launch_activation_reference_readback_mismatch"
                        )
                    temporary.chmod(0o440)
                    descriptor = os.open(temporary, os.O_RDONLY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    try:
                        os.link(temporary, destination, follow_symlinks=False)
                    except FileExistsError:
                        pass
                finally:
                    temporary.unlink(missing_ok=True)
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.stat().st_size != row["size_bytes"]
                or _sha256_file(destination) != row["digest"]
            ):
                raise TaskEvaluationLaunchActivationWorkerError(
                    "launch_activation_reference_identity_mismatch"
                )
        by_path[row["contract_path"]] = destination
    return by_path


def _load_verified_preparation(
    *,
    activation_request: Mapping[str, Any],
    preparation_queue_root: Path,
    preparation_input_root: Path,
    episode_compilation_queue_root: Path | None = None,
    episode_compilation_output_root: Path | None = None,
    scene_construction_queue_root: Path | None = None,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path]
]:
    binding = activation_request["preparation"]
    preparation_id = str(binding["preparation_id"])
    filename = (
        f"{preparation_id}-{str(binding['request_digest']).removeprefix('sha256:')}.json"
    )
    envelope = _load_sealed(
        preparation_queue_root / "materialized" / filename,
        schema_version=PREPARATION_ENVELOPE_SCHEMA_VERSION,
        digest_field="envelope_digest",
    )
    result = _load_sealed(
        preparation_queue_root / "results" / filename,
        schema_version=PREPARATION_RESULT_SCHEMA_VERSION,
        digest_field="result_digest",
    )
    request = validate_launch_preparation_request(envelope["request"])
    accepted_statuses = {
        "native_arena_inputs_verified_awaiting_profile_authority",
        "queued_for_production_episode_compilation",
        "queued_for_production_scene_configuration",
    }
    if (
        envelope.get("request_digest") != binding["request_digest"]
        or result.get("result_digest") != binding["result_digest"]
        or result.get("status") not in accepted_statuses
        or result.get("full_byte_service_account_readback_passed") is not True
        or request["preparation_id"] != preparation_id
        or request["team_namespace"] != activation_request["team_namespace"]
        or request["expected_production_commit"]
        != activation_request["expected_production_commit"]
    ):
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_preparation_binding_mismatch"
        )
    if activation_request.get("lane") == "native_task_arena_policy_evaluation":
        policy_configuration = request.get("policy_run_configuration")
        policy_plan = result.get("policy_run_plan")
        configuration_run_kind = (
            policy_configuration.get("run_kind", "qualified_evaluation")
            if isinstance(policy_configuration, Mapping)
            else None
        )
        if (
            not isinstance(policy_configuration, Mapping)
            or not isinstance(policy_plan, Mapping)
            or policy_plan.get("schema_version")
            != "task_evaluation_policy_run_plan.v1"
            or policy_plan.get("configuration_digest")
            != policy_configuration.get("configuration_digest")
            or policy_plan.get("plan_digest")
            != canonical_digest(policy_plan, digest_field="plan_digest")
            or policy_plan.get("execution_performed") is not False
            or policy_plan.get("provider_mutation_performed") is not False
            or activation_request.get("run_kind", "qualified_evaluation")
            != configuration_run_kind
        ):
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_policy_run_plan_invalid"
            )
    preparation_root = (preparation_input_root / preparation_id).resolve()
    expected_references: dict[str, tuple[str, int]] = {
        row["contract_path"]: (row["digest"], row["size_bytes"])
        for row in _collect_references(request)
    }
    materialized_rows: dict[str, Mapping[str, Any]] = {}
    materialized_references: dict[str, Path] = {}
    for row in result.get("references") or []:
        if not isinstance(row, Mapping):
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_preparation_reference_invalid"
            )
        contract_path = str(row.get("contract_path") or "")
        path = Path(str(row.get("materialized_path") or "")).resolve()
        if (
            not contract_path
            or row.get("full_byte_service_account_readback_passed") is not True
            or not _under(path, preparation_root)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256_file(path) != row.get("digest")
        ):
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_preparation_reference_invalid"
            )
        if contract_path in materialized_rows:
            existing = materialized_rows[contract_path]
            if (row.get("digest"), row.get("size_bytes")) != (
                existing.get("digest"),
                existing.get("size_bytes"),
            ):
                raise TaskEvaluationLaunchActivationWorkerError(
                    "launch_activation_preparation_reference_invalid"
                )
            continue
        materialized_rows[contract_path] = row
        materialized_references[contract_path] = path
    construction_envelope: dict[str, Any] | None = None
    construction_envelope_path: Path | None = None
    if result.get("status") == "queued_for_production_scene_configuration":
        if scene_construction_queue_root is None:
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_scene_construction_queue_missing"
            )
        orchestration_id = str(result.get("construction_orchestration_id") or "")
        recipe_digest = str(result.get("construction_recipe_digest") or "")
        filename = f"{orchestration_id}-{recipe_digest.removeprefix('sha256:')}.json"
        candidates = [
            scene_construction_queue_root / state / filename
            for state in SCENE_CONSTRUCTION_QUEUE_STATES
            if (scene_construction_queue_root / state / filename).exists()
        ]
        if len(candidates) != 1:
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_scene_construction_envelope_ambiguous"
            )
        queue_state = candidates[0].parent.name
        if queue_state not in {"pending", "processing"}:
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_scene_construction_queue_state_invalid:"
                f"{queue_state}"
            )
        construction_envelope_path = candidates[0].resolve()
        construction_envelope = _load_sealed(
            construction_envelope_path,
            schema_version=SCENE_CONSTRUCTION_ENVELOPE_SCHEMA_VERSION,
            digest_field="envelope_digest",
        )
        if (
            construction_envelope.get("orchestration_id") != orchestration_id
            or construction_envelope.get("preparation_id") != preparation_id
            or construction_envelope.get("run_id") != request["run_id"]
            or construction_envelope.get("team_namespace")
            != request["team_namespace"]
            or construction_envelope.get("expected_production_commit")
            != activation_request["expected_production_commit"]
            or construction_envelope.get("recipe_digest") != recipe_digest
            or construction_envelope.get("envelope_digest")
            != result.get("construction_queue_envelope_digest")
            or construction_envelope.get("request") != request
            or construction_envelope.get("paid_execution_requested") is not False
            or construction_envelope.get("provider_mutation_performed") is not False
        ):
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_scene_construction_binding_mismatch"
            )
        for row in construction_envelope.get("stage_configuration_references") or []:
            if not isinstance(row, Mapping):
                raise TaskEvaluationLaunchActivationWorkerError(
                    "launch_activation_scene_configuration_reference_invalid"
                )
            expected_references[str(row.get("contract_path") or "")] = (
                str(row.get("digest") or ""),
                int(row.get("size_bytes") or 0),
            )
    if request["run_mode"] == "episode_evaluation":
        revision_path = materialized_references.get("scene.configured_revision")
        try:
            revision = validate_configured_scene_revision(
                _read_json(
                    revision_path or Path("/nonexistent"),
                    blocker="launch_activation_configured_revision_invalid",
                )
            )
        except TaskEvaluationConfiguredSceneRevisionError as exc:
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_configured_revision_invalid"
            ) from exc
        transitive = {
            "scene.configured_revision.configured_scene_bundle": revision[
                "configured_scene_bundle"
            ],
            "scene.configured_revision.source.manifest": revision["source"][
                "manifest"
            ],
            "scene.configured_revision.source.rights_admission": revision[
                "source"
            ]["rights_admission"],
            "scene.configured_revision.registration.metric": revision[
                "registration"
            ]["metric"],
            "scene.configured_revision.registration.support_plane": revision[
                "registration"
            ]["support_plane"],
            "scene.configured_revision.registration.robot_mount_interface": revision[
                "registration"
            ]["robot_mount_interface"],
            "scene.configured_revision.registration.camera_calibration": revision[
                "registration"
            ]["camera_calibration"],
            "scene.configured_revision.registration.workspace_clearance": revision[
                "registration"
            ]["workspace_clearance"],
            "scene.configured_revision.task_template.definition": revision[
                "task_template"
            ]["definition"],
            "scene.configured_revision.task_template.success_criteria": revision[
                "task_template"
            ]["success_criteria"],
            "scene.configured_revision.task_template.execution": revision[
                "task_template"
            ]["execution"],
        }
        optional_replacement_references = {
            "scene.configured_revision.replacement.source_object": revision[
                "replacement"
            ]["source_object"],
            "scene.configured_revision.replacement.static_qualification": revision[
                "replacement"
            ]["static_qualification"],
            "scene.configured_revision.replacement.native_import_qualification": revision[
                "replacement"
            ]["native_import_qualification"],
        }
        transitive.update(
            {
                contract_path: reference
                for contract_path, reference in optional_replacement_references.items()
                if contract_path in materialized_references
            }
        )
        transitive.update(
            {
                "scene.configured_revision.source.rights_evidence."
                f"{index}.artifact": evidence["artifact"]
                for index, evidence in enumerate(
                    revision["source"]["rights_evidence"]
                )
            }
        )
        expected_references.update(
            {
                contract_path: (reference["digest"], reference["size_bytes"])
                for contract_path, reference in transitive.items()
            }
        )
    if set(materialized_references) != set(expected_references):
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_preparation_reference_set_invalid"
        )
    for contract_path, expected in expected_references.items():
        row = materialized_rows[contract_path]
        if (row.get("digest"), row.get("size_bytes")) != expected:
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_preparation_reference_invalid"
            )
    if construction_envelope is not None and construction_envelope_path is not None:
        return (
            request,
            result,
            {
                "kind": "task_evaluation_scene_configuration",
                "construction_envelope_path": str(construction_envelope_path),
                "construction_envelope_digest": construction_envelope[
                    "envelope_digest"
                ],
                "recipe_digest": construction_envelope["recipe_digest"],
                "source_commit": construction_envelope[
                    "expected_production_commit"
                ],
            },
            materialized_references,
        )
    adapter_root = preparation_root / "native-arena-adapter"
    expected_adapter_digest = result.get("adapter_result_digest")
    if result.get("status") == "queued_for_production_episode_compilation":
        if (
            episode_compilation_queue_root is None
            or episode_compilation_output_root is None
        ):
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_episode_compilation_roots_missing"
            )
        compilation_id = str(result.get("episode_compilation_id") or "")
        envelope_digest = str(
            result.get("episode_compilation_queue_envelope_digest") or ""
        )
        compilation_filename = (
            f"{compilation_id}-{envelope_digest.removeprefix('sha256:')}.json"
        )
        compilation = _load_sealed(
            episode_compilation_queue_root / "results" / compilation_filename,
            schema_version="task_evaluation_episode_compilation_result.v1",
            digest_field="result_digest",
        )
        adapter_path = Path(
            str(compilation.get("adapter_result_path") or "")
        ).resolve()
        compiled_root = episode_compilation_output_root.resolve(strict=True)
        if (
            compilation.get("status") != "compiled_for_production_launch"
            or compilation.get("source_commit")
            != activation_request["expected_production_commit"]
            or compilation.get("configured_scene_revision_digest")
            != revision["revision_digest"]
            or not _under(adapter_path, compiled_root)
        ):
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_episode_compilation_invalid"
            )
        adapter_root = adapter_path.parent
        expected_adapter_digest = compilation.get("adapter_result_digest")
    else:
        adapter_path = (
            adapter_root
            / "task_evaluation_native_arena_adapter_result.v1.json"
        )
    adapter = _load_sealed(
        adapter_path,
        schema_version=ADAPTER_RESULT_SCHEMA_VERSION,
        digest_field="result_digest",
    )
    packet_root = Path(str(adapter.get("packet_root") or "")).resolve()
    runtime_receipt = Path(
        str(adapter.get("runtime_source_receipt") or "")
    ).resolve()
    if (
        adapter.get("result_digest") != expected_adapter_digest
        or adapter.get("status") != "native_arena_adapter_materialized"
        or adapter.get("preparation_id") != preparation_id
        or adapter.get("source_commit")
        != activation_request["expected_production_commit"]
        or not _under(packet_root, adapter_root)
        or not _under(runtime_receipt, adapter_root)
        or not packet_root.is_dir()
        or not runtime_receipt.is_file()
    ):
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_adapter_binding_mismatch"
        )
    return request, result, adapter, materialized_references


def _read_json(path: Path, *, blocker: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationLaunchActivationWorkerError(blocker) from exc
    if path.is_symlink() or not path.is_file() or not isinstance(value, Mapping):
        raise TaskEvaluationLaunchActivationWorkerError(blocker)
    return dict(value)


def _build_native_context(
    *,
    activation_request: Mapping[str, Any],
    preparation_request: Mapping[str, Any],
    policy_run_plan: Mapping[str, Any] | None,
    adapter: Mapping[str, Any] | None = None,
    preparation_materialized: Mapping[str, Path],
    activation_materialized: Mapping[str, Path],
    activation_root: Path,
    repository_root: Path,
    destination_prefix: str,
    profile_dir: Path,
    webapp_catalog: Path,
    standing_authorization_dir: Path,
    service_account: str,
    service_group: str,
) -> dict[str, Any]:
    configured_revision: dict[str, Any] | None = None
    if preparation_request["run_mode"] == "episode_evaluation":
        configured_revision = validate_configured_scene_revision(
            _read_json(
                preparation_materialized["scene.configured_revision"],
                blocker="launch_activation_configured_revision_invalid",
            )
        )
        source_manifest_path = preparation_materialized["scene.configured_revision.source.manifest"]
        source_manifest_digest = configured_revision["source"]["manifest"]["digest"]
        rights_admission_path = preparation_materialized[
            "scene.configured_revision.source.rights_admission"
        ]
        rights_admission_digest = configured_revision["source"]["rights_admission"][
            "digest"
        ]
        rights_evidence_contracts = configured_revision["source"]["rights_evidence"]
        rights_evidence_prefix = "scene.configured_revision.source.rights_evidence"
    else:
        source_manifest_path = preparation_materialized["scene.source_manifest"]
        source_manifest_digest = preparation_request["scene"]["source_manifest"]["digest"]
        rights_admission_path = preparation_materialized["scene.rights.admission"]
        rights_admission_digest = preparation_request["scene"]["rights"]["admission"]["digest"]
        rights_evidence_contracts = preparation_request["scene"]["rights"]["evidence"]
        rights_evidence_prefix = "scene.rights.evidence"
    _read_json(
        source_manifest_path, blocker="launch_activation_source_manifest_invalid"
    )
    _read_json(
        rights_admission_path, blocker="launch_activation_rights_admission_invalid"
    )
    packet_root = Path(str(adapter["packet_root"])).resolve()
    packet_request_path = packet_root / "native_task_arena_packet_request.v1.json"
    packet_request = _read_json(
        packet_request_path, blocker="launch_activation_packet_request_invalid"
    ) if packet_request_path.exists() else {}
    runtime_contract = _read_json(
        packet_root / "native_task_runtime_contract.v1.json",
        blocker="launch_activation_runtime_contract_invalid",
    )
    robot_configuration = runtime_contract.get("robot")
    if not isinstance(robot_configuration, Mapping):
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_runtime_contract_invalid"
        )
    rights_evidence = []
    for index, evidence in enumerate(rights_evidence_contracts):
        path = preparation_materialized[
            f"{rights_evidence_prefix}.{index}.artifact"
        ]
        rights_evidence.append(
            {"role": evidence["role"], "path": str(path), "sha256": _sha256_file(path)}
        )
    lineage = activation_request["lineage"]
    operations: dict[str, Any] = {
        "set_root": str(activation_root / "launch-set"),
        "repository_root": str(repository_root),
        "source_commit": activation_request["expected_production_commit"],
        "destination_prefix": destination_prefix.rstrip("/")
        + "/"
        + preparation_request["publication"]["input_namespace"]
        + "/"
        + activation_request["activation_id"],
        "profile_dir": str(profile_dir),
        "webapp_catalog_out": str(webapp_catalog),
        "standing_authorization_dir": str(standing_authorization_dir),
        "standing_authorization_expires_at": activation_request["authorization"][
            "standing_authorization_expires_at"
        ],
        "pod_name": activation_request["activation_id"],
        "revision": activation_request["authorization"]["profile_revision"],
        "authorization_reference": activation_request["authorization"]["reference"],
        "authorized_by": activation_request["authorization"]["authorized_by"],
        "authorized_on": activation_request["authorization"]["authorized_on"],
        "maximum_hourly_rate_usd": preparation_request["spend"][
            "maximum_hourly_rate_usd"
        ],
        "hard_total_spend_cap_usd": preparation_request["spend"]["hard_cap_usd"],
        "hard_ttl_seconds": preparation_request["spend"]["hard_ttl_seconds"],
        "provider": preparation_request["spend"]["selected_provider"],
        "machine_avoidlist": "",
        "python": sys.executable,
        "service_account": service_account,
        "service_group": service_group,
    }
    if _control_search_warm_retention_requested(
        packet_request=packet_request, lane=str(activation_request["lane"])
    ):
        operations["retain_warm_control_search"] = True
    if lineage["kind"] == "initial_project":
        operations.update(
            {
                "project_spend_reconciliation": str(
                    activation_materialized["lineage.project_spend_reconciliation"]
                ),
                "initial_provider_zero": str(
                    activation_materialized["lineage.initial_provider_zero"]
                ),
            }
        )
    else:
        for name in (
            "prior_authority",
            "prior_result",
            "prior_launch_receipt",
            "prior_webapp_sync",
            "prior_provider_zero",
            "prior_spend_reconciliation",
            "construction_result",
        ):
            operations[name] = str(activation_materialized[f"lineage.{name}"])
        if "zero_action_result" in lineage:
            operations["zero_action_result"] = str(
                activation_materialized["lineage.zero_action_result"]
            )
        for name in ("controls_qualification_manifest",):
            if name in lineage:
                operations[name] = str(
                    activation_materialized[f"lineage.{name}"]
                )
    context = {
        "schema_version": "native_task_arena_launch_preparation_context.v2",
        "lane": activation_request["lane"],
        "team_namespace": activation_request["team_namespace"],
        "references": {
            "scene": {
                "scene_id": preparation_request["scene"]["identity"]["id"],
                "packet_dir": str(packet_root),
                "packet_receipt_digest": adapter["packet_receipt_digest"],
                "source_manifest": str(source_manifest_path),
                "source_manifest_digest": source_manifest_digest,
                "rights_admission": str(rights_admission_path),
                "rights_admission_digest": rights_admission_digest,
                "rights_evidence": rights_evidence,
                **(
                    {
                        "configured_scene_revision": str(
                            preparation_materialized[
                                "scene.configured_revision"
                            ]
                        ),
                        "configured_scene_revision_digest": (
                            configured_revision["revision_digest"]
                        ),
                    }
                    if configured_revision is not None
                    else {}
                ),
            },
            "task": {
                "task_id": preparation_request["task"]["identity"]["id"],
                "task_spec_digest": runtime_contract["task_spec_digest"],
            },
            "robot": {
                "robot_id": robot_configuration["robot_id"],
                "configuration_digest": canonical_digest(robot_configuration),
            },
            "runtime": {
                "source_packet": adapter["runtime_source_receipt"],
                "source_packet_receipt_digest": adapter[
                    "runtime_source_receipt_digest"
                ],
                "container_image": preparation_request["runtime"]["oci_image"],
            },
        },
        "operations": operations,
    }
    policy_configuration = preparation_request.get("policy_run_configuration")
    if policy_configuration is not None:
        if not isinstance(policy_run_plan, Mapping):
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_policy_run_plan_missing"
            )
        context["references"]["policy_run"] = {
            "configuration_digest": policy_configuration[
                "configuration_digest"
            ],
            "plan_digest": policy_run_plan["plan_digest"],
            "setup_digest": policy_configuration["setup_digest"],
            "source_launch_id": policy_configuration["source_launch_id"],
            "offering_digest": policy_configuration["offering_digest"],
            "candidate_ids": policy_configuration["candidate_ids"],
            "run_kind": policy_configuration.get("run_kind", "qualified_evaluation"),
            "claim_ceiling": policy_configuration.get("claim_ceiling"),
        }
    return context


def _build_scene_configuration_context(
    *,
    activation_request: Mapping[str, Any],
    preparation_request: Mapping[str, Any],
    construction_input: Mapping[str, Any],
    activation_materialized: Mapping[str, Path],
    activation_root: Path,
    repository_root: Path,
    toolchain_root: Path,
    destination_prefix: str,
    profile_dir: Path,
    webapp_catalog: Path,
    standing_authorization_dir: Path,
    service_account: str,
    service_group: str,
    configured_controls_autostart_intent_root: Path,
) -> dict[str, Any]:
    """Build server-owned inputs for one Website-started configuration profile."""

    if (
        activation_request.get("lane")
        != "task_evaluation_scene_configuration"
        or preparation_request.get("run_mode") != "scene_configuration"
        or activation_request.get("lineage", {}).get("kind")
        != "initial_project"
        or construction_input.get("kind")
        != "task_evaluation_scene_configuration"
    ):
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_scene_configuration_context_invalid"
        )
    source_commit = str(activation_request["expected_production_commit"])
    registry_name = configured_controls_autostart_registry_name(
        team_namespace=str(activation_request["team_namespace"]),
        scene_id=str(preparation_request["scene"]["identity"]["id"]),
        task_id=str(preparation_request["task"]["identity"]["id"]),
    )
    intent_source = configured_controls_autostart_intent_root / registry_name
    # An absent continuation intent does not block initial configuration; the
    # progression worker still refuses controls until a valid intent exists.
    continuation_provisioned = (
        intent_source.is_file() and not intent_source.is_symlink()
    )
    if continuation_provisioned:
        try:
            intent = validate_configured_controls_autostart_intent(
                _read_json(
                    intent_source,
                    blocker="launch_activation_configured_controls_autostart_intent_invalid",
                )
            )
        except (OSError, ValueError) as exc:
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_configured_controls_autostart_intent_invalid"
            ) from exc
        if (
            intent["expected_production_commit"] != source_commit
            or intent["configuration_source_commit"] != source_commit
            or intent["configuration_adoption"]
            != {"mode": "same_commit_automatic"}
            or intent["team_namespace"] != activation_request["team_namespace"]
            or intent["scene_id"]
            != preparation_request["scene"]["identity"]["id"]
            or intent["task_id"]
            != preparation_request["task"]["identity"]["id"]
        ):
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_configured_controls_autostart_intent_mismatch"
            )
    unresolved_construction_envelope = Path(
        str(construction_input.get("construction_envelope_path") or "")
    )
    if unresolved_construction_envelope.is_symlink():
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_scene_configuration_context_invalid"
        )
    construction_envelope = unresolved_construction_envelope.resolve()
    construction_value = _read_json(
        construction_envelope,
        blocker="launch_activation_scene_configuration_envelope_invalid",
    )
    disclosure_decision = (
        construction_value.get("render_inputs_result") or {}
    ).get("disclosure_decision")
    raw_source_authorized = renders_on_provider(disclosure_decision or {})
    if (
        not construction_envelope.is_file()
        or construction_input.get("source_commit") != source_commit
    ):
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_scene_configuration_context_invalid"
        )
    try:
        toolchain_manifest = validate_scene_configuration_toolchain(
            root=toolchain_root.resolve(strict=True),
            expected_source_commit=source_commit,
        )
    except (OSError, ValueError) as exc:
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_scene_configuration_toolchain_invalid"
        ) from exc
    operations = {
        "set_root": str(activation_root / "launch-set"),
        "repository_root": str(repository_root),
        "source_commit": source_commit,
        "construction_envelope": str(construction_envelope),
        "toolchain_root": str(toolchain_root.resolve()),
        "destination_prefix": destination_prefix.rstrip("/")
        + "/"
        + preparation_request["publication"]["input_namespace"]
        + "/"
        + activation_request["activation_id"],
        "profile_dir": str(profile_dir),
        "webapp_catalog_out": str(webapp_catalog),
        "standing_authorization_dir": str(standing_authorization_dir),
        "standing_authorization_expires_at": activation_request[
            "authorization"
        ]["standing_authorization_expires_at"],
        "pod_name": activation_request["activation_id"],
        "revision": activation_request["authorization"]["profile_revision"],
        "authorization_reference": activation_request["authorization"][
            "reference"
        ],
        "authorized_by": activation_request["authorization"]["authorized_by"],
        "authorized_on": activation_request["authorization"]["authorized_on"],
        "maximum_hourly_rate_usd": preparation_request["spend"][
            "maximum_hourly_rate_usd"
        ],
        "hard_total_spend_cap_usd": preparation_request["spend"][
            "hard_cap_usd"
        ],
        "provider_compute_spend_cap_usd": preparation_request["spend"][
            "provider_compute_spend_cap_usd"
        ],
        "openai_max_cost_usd": preparation_request["spend"][
            "external_service_caps"
        ]["openai"]["maximum_cost_usd"],
        "openai_max_requests": preparation_request["spend"][
            "external_service_caps"
        ]["openai"]["maximum_requests"],
        "openai_artifixer_semantic_teacher_max_cost_usd": preparation_request[
            "spend"
        ]["external_service_caps"]["openai"]["stage_max_cost_usd"][
            "artifixer_semantic_teacher"
        ],
        "openai_artifixer_visual_review_max_cost_usd": preparation_request[
            "spend"
        ]["external_service_caps"]["openai"]["stage_max_cost_usd"][
            "artifixer_visual_review"
        ],
        "openai_content_agents_max_cost_usd": preparation_request["spend"][
            "external_service_caps"
        ]["openai"]["stage_max_cost_usd"]["content_agents"],
        "hard_ttl_seconds": preparation_request["spend"]["hard_ttl_seconds"],
        "container_image": preparation_request["runtime"]["oci_image"],
        "scene_id": preparation_request["scene"]["identity"]["id"],
        "task_id": preparation_request["task"]["identity"]["id"],
        "project_spend_reconciliation": str(
            activation_materialized["lineage.project_spend_reconciliation"]
        ),
        "initial_provider_zero": str(
            activation_materialized["lineage.initial_provider_zero"]
        ),
        # Empty when no continuation is authorized yet: the lane skips the
        # intent step and the profile is built without the continuation input,
        # which is exactly what the progression worker refuses to act on.
        "configured_controls_autostart_intent_source": (
            str(intent_source.resolve()) if continuation_provisioned else ""
        ),
        "configured_controls_autostart_intent_artifacts": (
            str(
                Path(activation_root / "launch-set")
                / "task_evaluation_configured_controls_autostart_intent.v1.json"
            )
            if continuation_provisioned
            else ""
        ),
        "python": sys.executable,
        "service_account": service_account,
        "service_group": service_group,
    }
    return {
        "schema_version": (
            "task_evaluation_scene_configuration_launch_preparation_context.v1"
        ),
        "lane": "task_evaluation_scene_configuration",
        "team_namespace": activation_request["team_namespace"],
        "reference_bindings": {
            "source_commit": source_commit,
            "preparation_id": preparation_request["preparation_id"],
            "run_id": preparation_request["run_id"],
            "scene_identity": dict(preparation_request["scene"]["identity"]),
            "task_identity": dict(preparation_request["task"]["identity"]),
            "construction_envelope_sha256": _sha256_file(
                construction_envelope
            ),
            "construction_envelope_digest": construction_input[
                "construction_envelope_digest"
            ],
            "recipe_digest": construction_input["recipe_digest"],
            "toolchain_digest": toolchain_manifest["toolchain_digest"],
            "raw_interiorgs_bytes_authorized_for_provider": raw_source_authorized,
            "provider_disclosure_decision_digest": (
                disclosure_decision.get("decision_digest")
                if isinstance(disclosure_decision, Mapping)
                else None
            ),
            "evaluation_episode_authorized": False,
        },
        "operations": operations,
    }


def default_activation_preparer(
    *, lane: str, context_path: Path, receipt_path: Path, repository_root: Path
) -> dict[str, Any]:
    """Run the fixed repository-owned no-allocation preparation command."""

    command = [
        sys.executable,
        str(repository_root / "scripts" / "prepare_paid_lane_launch.py"),
        "--lane",
        lane,
        "--context-file",
        str(context_path),
        "--receipt-out",
        str(receipt_path),
    ]
    completed = subprocess.run(  # nosec B603 - fixed argv, no shell
        command, capture_output=True, text=True, check=False, timeout=3600
    )
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise TaskEvaluationLaunchActivationWorkerError(
            f"launch_activation_preparer_exit_{completed.returncode}_without_receipt"
        )
    receipt = _read_json(
        receipt_path, blocker="launch_activation_preparation_receipt_invalid"
    )
    return receipt


def _artifact_for_step(receipt: Mapping[str, Any], step_id: str) -> Path:
    matches = [
        row
        for row in receipt.get("completed_steps") or []
        if isinstance(row, Mapping) and row.get("step_id") == step_id
    ]
    if len(matches) != 1:
        raise TaskEvaluationLaunchActivationWorkerError(
            f"launch_activation_step_missing:{step_id}"
        )
    path = Path(str(matches[0].get("artifact_path") or "")).resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or _sha256_file(path) != matches[0].get("artifact_sha256")
    ):
        raise TaskEvaluationLaunchActivationWorkerError(
            f"launch_activation_step_artifact_invalid:{step_id}"
        )
    return path


def _activation_result(
    *,
    request: Mapping[str, Any],
    preparation_result: Mapping[str, Any],
    window: Mapping[str, Any],
    preparation_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    preparation_blockers = [
        str(value)
        for value in preparation_receipt.get("blockers") or []
        if isinstance(value, str)
        and re.fullmatch(r"[A-Za-z0-9_.:-]{1,240}", value)
    ][:3]
    if (
        preparation_receipt.get("status") != "prepared"
        or preparation_receipt.get("source_commit")
        != request["expected_production_commit"]
        or preparation_receipt.get("provider_allocation_performed") is not False
        or preparation_receipt.get("paid_inference_performed") is not False
    ):
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_preparation_graph_blocked"
            + (
                ":" + ",".join(preparation_blockers)
                if preparation_blockers
                else ""
            )
        )
    profile_path = _artifact_for_step(preparation_receipt, "live_profile")
    publication_path = _artifact_for_step(
        preparation_receipt, "profile_publication"
    )
    authorization_path = _artifact_for_step(
        preparation_receipt, "standing_authorization"
    )
    profile = _read_json(profile_path, blocker="launch_activation_profile_invalid")
    authorization = _read_json(
        authorization_path, blocker="launch_activation_standing_authorization_invalid"
    )
    if (
        not str(profile.get("profile_id") or "")
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(profile.get("profile_digest") or ""))
        or authorization.get("profile_id") != profile["profile_id"]
        or authorization.get("profile_digest") != profile["profile_digest"]
        or authorization.get("provider_mutation_performed") is not False
    ):
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_published_identity_mismatch"
        )
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "profile_authority_materialized_no_execution",
        "activation_id": request["activation_id"],
        "preparation_id": request["preparation"]["preparation_id"],
        "team_namespace": request["team_namespace"],
        "lane": request["lane"],
        "source_commit": request["expected_production_commit"],
        "preparation_result_digest": preparation_result["result_digest"],
        "release_window_digest": window["window_digest"],
        "profile_id": profile["profile_id"],
        "profile_digest": profile["profile_digest"],
        "profile_publication_receipt_digest": _sha256_file(publication_path),
        "standing_authorization_digest": _sha256_file(authorization_path),
        "full_byte_activation_reference_readback_passed": True,
        "profile_publication_performed": True,
        "catalog_mutation_performed": True,
        "standing_authorization_published": True,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "blockers": [],
        "observed_at_iso": datetime.now(timezone.utc).isoformat(),
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(
        result, digest_field="result_digest"
    )
    return result


def _policy_campaign_activation_result(
    *,
    request: Mapping[str, Any],
    preparation_request: Mapping[str, Any],
    preparation_result: Mapping[str, Any],
    window: Mapping[str, Any],
    activation_materialized: Mapping[str, Path],
    activation_root: Path,
    adapter: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal the N-cell paired-campaign queue without profile or paid mutation."""

    try:
        controls_path = activation_materialized.get(
            "lineage.controls_qualification_manifest"
        )
        qualification = (
            _read_json(
                controls_path,
                blocker="launch_activation_controls_qualification_manifest_invalid",
            )
            if controls_path is not None
            else None
        )
        manifest = build_policy_campaign_activation_manifest(
            configuration=preparation_request["policy_run_configuration"],
            plan=preparation_result["policy_run_plan"],
            controls_qualification=qualification,
        )
    except (KeyError, TaskEvaluationPolicyRunContractError) as exc:
        raise TaskEvaluationLaunchActivationWorkerError(
            f"launch_activation_policy_campaign_invalid:{exc}"
        ) from exc
    manifest_path = (
        activation_root / "task_evaluation_policy_campaign_activation.v1.json"
    )
    write_launch_preparation_record_exclusive(manifest_path, manifest)
    runtime_inputs_path: Path | None = None
    runtime_inputs: dict[str, Any] | None = None
    if manifest["run_kind"] == "internal_policy_canary":
        if adapter is None:
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_policy_canary_adapter_missing"
            )
        construction_path = activation_materialized.get("lineage.construction_result")
        if construction_path is None:
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_policy_canary_construction_result_missing"
            )
        construction = _read_json(
            construction_path,
            blocker="launch_activation_policy_canary_construction_result_invalid",
        )
        strict_construction = (
            construction.get("schema_version")
            == "native_task_arena_construction_result.v1"
            and construction.get("status") == "completed"
            and construction.get("construction_gate_qualified") is True
            and construction.get("candidate_policy_queried") is False
            and construction.get("blockers") in ([], ())
            and construction.get("result_digest")
            == canonical_digest(construction, digest_field="result_digest")
        )
        compiled_scene_diagnostic = (
            construction.get("schema_version")
            == "task_evaluation_episode_compilation_result.v1"
            and construction.get("status") == "compiled_for_production_launch"
            and construction.get("blockers") == []
            and construction.get("configured_scene_revision_digest")
            == preparation_request["policy_run_configuration"][
                "scene_revision_digest"
            ]
            and construction.get("provider_mutation_performed") is False
            and construction.get("paid_execution_requested") is False
            and construction.get("result_digest")
            == canonical_digest(construction, digest_field="result_digest")
        )
        if not strict_construction and not compiled_scene_diagnostic:
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_policy_canary_construction_result_invalid"
            )
        packet_root = Path(str(adapter.get("packet_root") or "")).resolve()
        packet_receipt = packet_root / "native_task_arena_packet_receipt.v1.json"
        runtime_receipt = Path(
            str(adapter.get("runtime_source_receipt") or "")
        ).resolve()
        if (
            not packet_root.is_dir()
            or packet_receipt.is_symlink()
            or not packet_receipt.is_file()
            or runtime_receipt.is_symlink()
            or not runtime_receipt.is_file()
        ):
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_policy_canary_base_runtime_inputs_invalid"
            )
        cells = preparation_request["policy_run_configuration"]["matrix"]["cells"]
        spend = preparation_request.get("spend")
        if (
            not isinstance(spend, Mapping)
            or not isinstance(spend.get("hard_cap_usd"), (int, float))
            or isinstance(spend.get("hard_cap_usd"), bool)
            or float(spend["hard_cap_usd"]) <= 0
            or not isinstance(spend.get("hard_ttl_seconds"), int)
            or isinstance(spend.get("hard_ttl_seconds"), bool)
            or int(spend["hard_ttl_seconds"]) <= 0
            or not isinstance(spend.get("maximum_hourly_rate_usd"), (int, float))
            or isinstance(spend.get("maximum_hourly_rate_usd"), bool)
            or float(spend["maximum_hourly_rate_usd"]) <= 0
        ):
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_policy_canary_resource_authority_invalid"
            )
        resource_name = (
            "blueprint-native-task-policy-canary-"
            + manifest["activation_digest"].removeprefix("sha256:")[:32]
        )
        runtime_inputs = {
            "schema_version": "task_evaluation_policy_canary_runtime_inputs.v1",
            "run_id": preparation_request["run_id"],
            "run_kind": "internal_policy_canary",
            "claim_ceiling": "diagnostic_policy_execution",
            "scene_revision_digest": preparation_request[
                "policy_run_configuration"
            ]["scene_revision_digest"],
            "matrix_digest": preparation_request["policy_run_configuration"][
                "matrix"
            ]["scenario_set_digest"],
            "configuration_digest": preparation_request[
                "policy_run_configuration"
            ]["configuration_digest"],
            "plan_digest": preparation_result["policy_run_plan"]["plan_digest"],
            "activation_digest": manifest["activation_digest"],
            "base_native_packet": {
                "path": str(packet_receipt),
                "size_bytes": packet_receipt.stat().st_size,
                "sha256": _sha256_file(packet_receipt),
            },
            "runtime_source": {
                "path": str(runtime_receipt),
                "size_bytes": runtime_receipt.stat().st_size,
                "sha256": _sha256_file(runtime_receipt),
            },
            "construction_result": {
                "path": str(construction_path),
                "size_bytes": construction_path.stat().st_size,
                "sha256": _sha256_file(construction_path),
            },
            "policy_readiness": preparation_request["policy_run_setup"][
                "preregistration"
            ],
            "candidate_ids": preparation_request["policy_run_configuration"][
                "candidate_ids"
            ],
            "cells": [
                {
                    "cell_id": cell["cell_id"],
                    "seed": cell["seed"],
                    "family": cell["family"],
                    "cell_spec_digest": cell["cell_spec_digest"],
                    "resolved_scenario": cell["resolved_scenario"],
                    "resolved_scenario_digest": canonical_digest(
                        cell["resolved_scenario"]
                    ),
                    "control_diagnostic": {
                        "mode": "nonblocking_diagnostic_pending",
                        "typed_gap": "controls_pending_at_submission",
                        "policy_execution_blocked": False,
                    },
                }
                for cell in cells
            ],
            "execution_authority": {
                "maximum_provider_allocations": 1,
                "retry_cap": 0,
                "single_warm_provider_session_required": True,
                "caller_surviving_watchdog_required": True,
                "billing_teardown_provider_zero_required": True,
            },
            "resource_authority": {
                "resource_name": resource_name,
                "maximum_hourly_rate_usd": float(
                    spend["maximum_hourly_rate_usd"]
                ),
                "hard_cap_usd": float(spend["hard_cap_usd"]),
                "hard_ttl_seconds": int(spend["hard_ttl_seconds"]),
                "user_confirmed": True,
            },
            "capture_contract": {
                "schema_version": "task_evaluation_policy_canary_capture_contract.v1",
                "immutable_identity_fields": [
                    "scene_revision_digest",
                    "checkpoint_digest",
                    "runtime_identity_digest",
                    "matrix_digest",
                    "cell_spec_digest",
                    "seed",
                ],
                "calibration_and_timebase_required": True,
                "synchronized_timestamped_streams": [
                    "observation",
                    "action",
                    "simulator_state",
                    "contact",
                    "force",
                    "task_object",
                    "deterministic_scoring",
                ],
                "per_episode_evidence_required": [
                    "lossless_frame_manifest",
                    "derived_review_video",
                    "policy_query_receipt",
                    "action_delivery_readback",
                ],
                "runtime_system_telemetry_required": True,
                "indexed_telemetry_artifact_or_typed_format_gap_required": True,
                "artifact_inventory_fields": [
                    "role",
                    "media_type",
                    "size_bytes",
                    "digest",
                ],
                "metrics_required": [
                    "aggregate",
                    "per_family",
                    "per_episode",
                ],
                "terminal_receipts_required": [
                    "billing",
                    "teardown",
                    "provider_zero",
                    "notification_delivery",
                ],
                "typed_evidence_gaps_preserved": True,
                "ui_derived_evidence_forbidden": True,
            },
            "runtime_inputs_digest": "",
        }
        runtime_inputs["runtime_inputs_digest"] = canonical_digest(
            runtime_inputs, digest_field="runtime_inputs_digest"
        )
        runtime_inputs_path = (
            activation_root / "task_evaluation_policy_canary_runtime_inputs.v1.json"
        )
        write_launch_preparation_record_exclusive(runtime_inputs_path, runtime_inputs)
        website_request_digest_value = preparation_request[
            "policy_run_configuration"
        ].get("website_request_digest")
        if not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(website_request_digest_value or ""),
        ):
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_policy_canary_website_request_digest_invalid"
            )
        website_request_digest = str(website_request_digest_value)
    else:
        website_request_digest = None
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "policy_campaign_queue_materialized_no_execution",
        "activation_id": request["activation_id"],
        "preparation_id": request["preparation"]["preparation_id"],
        "team_namespace": request["team_namespace"],
        "lane": request["lane"],
        "source_commit": request["expected_production_commit"],
        "preparation_result_digest": preparation_result["result_digest"],
        "release_window_digest": window["window_digest"],
        "policy_campaign_activation_digest": manifest["activation_digest"],
        "policy_campaign_activation_sha256": _sha256_file(manifest_path),
        "campaign_unit_count": manifest["campaign_unit_count"],
        "run_kind": manifest["run_kind"],
        "claim_ceiling": manifest.get("claim_ceiling"),
        **(
            {
                "policy_canary_runtime_inputs_digest": runtime_inputs[
                    "runtime_inputs_digest"
                ],
                "policy_canary_runtime_inputs_sha256": _sha256_file(
                    runtime_inputs_path
                ),
                "policy_canary_runtime_inputs_path": str(runtime_inputs_path),
                "capture_session_id": request["capture_session_id"],
                "intake_id": request["intake_id"],
                "request_digest": request["preparation"]["request_digest"],
                "website_request_digest": website_request_digest,
            }
            if runtime_inputs is not None and runtime_inputs_path is not None
            else {}
        ),
        "full_byte_activation_reference_readback_passed": True,
        "profile_publication_performed": False,
        "catalog_mutation_performed": False,
        "standing_authorization_published": False,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "blockers": [],
        "observed_at_iso": datetime.now(timezone.utc).isoformat(),
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(
        result, digest_field="result_digest"
    )
    return result


def process_launch_activation_queue(
    *,
    queue_root: str | Path,
    preparation_queue_root: str | Path,
    preparation_input_root: str | Path,
    episode_compilation_queue_root: str | Path | None = None,
    episode_compilation_output_root: str | Path | None = None,
    activation_root: str | Path,
    allowed_uri_prefixes: Sequence[str],
    service_account: str,
    service_group: str,
    repository_root: str | Path,
    destination_prefix: str,
    release_window_prefix: str,
    profile_dir: str | Path,
    webapp_catalog: str | Path,
    standing_authorization_dir: str | Path,
    scene_construction_queue_root: str | Path | None = None,
    scene_configuration_toolchain_root: str | Path | None = None,
    configured_controls_autostart_intent_root: str | Path | None = None,
    policy_canary_dispatch_queue_root: str | Path | None = None,
    source_commit: str | None = None,
    max_messages: int = 1,
    fetcher: ReferenceFetcher = default_reference_fetcher,
    preparer: ActivationPreparer = default_activation_preparer,
    disk_reservation_root: str | Path | None = None,
) -> dict[str, Any]:
    """Claim and activate bounded queue items without paid execution."""

    if not isinstance(max_messages, int) or isinstance(max_messages, bool) or not 1 <= max_messages <= 8:
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_max_messages_invalid"
        )
    observed_commit = source_commit or running_worker_source_commit()
    if not re.fullmatch(r"[0-9a-f]{40}", observed_commit):
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_worker_source_commit_unproven"
        )
    validated_window_prefix = validate_release_window_uri(
        release_window_prefix, prefix=release_window_prefix
    )
    root = ensure_launch_activation_queue_root(queue_root)
    prep_queue = Path(preparation_queue_root).resolve(strict=True)
    prep_inputs = Path(preparation_input_root).resolve(strict=True)
    activation_base = Path(activation_root)
    activation_base.mkdir(parents=True, exist_ok=True, mode=0o750)
    activation_base = activation_base.resolve(strict=True)
    repository = Path(repository_root).resolve(strict=True)
    results_root = root / "results"
    conflicts_root = results_root / "conflicts"
    conflicts_root.mkdir(mode=0o750, exist_ok=True)
    processed: list[dict[str, Any]] = []
    processing_leases: list[Any] = []
    for source in sorted((root / "pending").glob("*.json"))[:max_messages]:
        claimed = root / "processing" / source.name
        try:
            reservation = os.open(claimed, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        else:
            os.close(reservation)
        try:
            os.replace(source, claimed)
        except FileNotFoundError:
            claimed.unlink(missing_ok=True)
            continue
        processing_leases.append(_acquire_processing_lease(claimed))
        terminal_state = "prepared"
        request, disk_reservation = None, None
        try:
            envelope = _load_sealed(
                claimed,
                schema_version=ENVELOPE_SCHEMA_VERSION,
                digest_field="envelope_digest",
            )
            request = validate_launch_activation_request(envelope["request"])
            if disk_reservation_root is not None:
                disk_reservation = disk_budget.reserve_control_plane_disk(
                    "launch_activation",
                    target_root=activation_base,
                    reservation_root=disk_reservation_root,
                )
            if request["expected_production_commit"] != observed_commit:
                raise TaskEvaluationLaunchActivationWorkerError(
                    "launch_activation_worker_source_commit_mismatch"
                )
            validate_release_window_uri(
                str(request["release_window"]["uri"]),
                prefix=validated_window_prefix,
            )
            (
                preparation_request,
                preparation_result,
                adapter,
                preparation_materialized,
            ) = (
                _load_verified_preparation(
                    activation_request=request,
                    preparation_queue_root=prep_queue,
                    preparation_input_root=prep_inputs,
                    episode_compilation_queue_root=(
                        Path(episode_compilation_queue_root).resolve(strict=True)
                        if episode_compilation_queue_root is not None
                        else None
                    ),
                    episode_compilation_output_root=(
                        Path(episode_compilation_output_root).resolve(strict=True)
                        if episode_compilation_output_root is not None
                        else None
                    ),
                    scene_construction_queue_root=(
                        Path(scene_construction_queue_root).resolve(strict=True)
                        if scene_construction_queue_root is not None
                        else None
                    ),
                )
            )
            owned_root = activation_base / request["activation_id"]
            owned_root.mkdir(parents=True, exist_ok=True, mode=0o750)
            materialized = _materialize_activation_references(
                request=request,
                root=owned_root / "references",
                allowed_uri_prefixes=allowed_uri_prefixes,
                service_account=service_account,
                fetcher=fetcher,
            )
            window_value = _read_json(
                materialized["release_window"],
                blocker="launch_activation_release_window_invalid",
            )
            window = validate_shared_mutation_window(
                window_value,
                activation_id=request["activation_id"],
                activation_intent_digest=launch_activation_intent_digest(request),
                team_namespace=request["team_namespace"],
                expected_production_commit=observed_commit,
                provider_allowlist=preparation_request["spend"]["provider_allowlist"],
                hard_cap_usd=preparation_request["spend"]["hard_cap_usd"],
            )
            if request["lane"] == "task_evaluation_scene_configuration":
                if scene_configuration_toolchain_root is None:
                    raise TaskEvaluationLaunchActivationWorkerError(
                        "launch_activation_scene_configuration_toolchain_missing"
                    )
                if configured_controls_autostart_intent_root is None:
                    raise TaskEvaluationLaunchActivationWorkerError(
                        "launch_activation_configured_controls_autostart_intent_root_missing"
                    )
                raw_autostart_root = Path(
                    configured_controls_autostart_intent_root
                ).expanduser()
                if raw_autostart_root.is_symlink():
                    raise TaskEvaluationLaunchActivationWorkerError(
                        "launch_activation_configured_controls_autostart_intent_root_invalid"
                    )
                try:
                    resolved_autostart_root = raw_autostart_root.resolve(strict=True)
                except OSError as exc:
                    raise TaskEvaluationLaunchActivationWorkerError(
                        "launch_activation_configured_controls_autostart_intent_root_invalid"
                    ) from exc
                if not resolved_autostart_root.is_dir():
                    raise TaskEvaluationLaunchActivationWorkerError(
                        "launch_activation_configured_controls_autostart_intent_root_invalid"
                    )
                context = _build_scene_configuration_context(
                    activation_request=request,
                    preparation_request=preparation_request,
                    construction_input=adapter,
                    activation_materialized=materialized,
                    activation_root=owned_root,
                    repository_root=repository,
                    toolchain_root=Path(
                        scene_configuration_toolchain_root
                    ),
                    destination_prefix=destination_prefix,
                    profile_dir=Path(profile_dir).resolve(),
                    webapp_catalog=Path(webapp_catalog).resolve(),
                    standing_authorization_dir=Path(
                        standing_authorization_dir
                    ).resolve(),
                    service_account=service_account,
                    service_group=service_group,
                    configured_controls_autostart_intent_root=(
                        resolved_autostart_root
                    ),
                )
                context_path = (
                    owned_root
                    / "task_evaluation_scene_configuration_"
                    "launch_preparation_context.v1.json"
                )
            else:
                context = _build_native_context(
                    activation_request=request,
                    preparation_request=preparation_request,
                    policy_run_plan=preparation_result.get("policy_run_plan"),
                    adapter=adapter,
                    preparation_materialized=preparation_materialized,
                    activation_materialized=materialized,
                    activation_root=owned_root,
                    repository_root=repository,
                    destination_prefix=destination_prefix,
                    profile_dir=Path(profile_dir).resolve(),
                    webapp_catalog=Path(webapp_catalog).resolve(),
                    standing_authorization_dir=Path(
                        standing_authorization_dir
                    ).resolve(),
                    service_account=service_account,
                    service_group=service_group,
                )
                context_path = (
                    owned_root
                    / "native_task_arena_launch_preparation_context.v2.json"
                )
            write_launch_preparation_record_exclusive(context_path, context)
            if request["lane"] == "native_task_arena_policy_evaluation":
                result = _policy_campaign_activation_result(
                    request=request,
                    preparation_request=preparation_request,
                    preparation_result=preparation_result,
                    window=window,
                    activation_materialized=materialized,
                    activation_root=owned_root,
                    adapter=adapter,
                )
            else:
                receipt_path = owned_root / "paid_lane_launch_preparation.v1.json"
                preparation_receipt = preparer(
                    lane=request["lane"],
                    context_path=context_path,
                    receipt_path=receipt_path,
                    repository_root=repository,
                )
                result = _activation_result(
                    request=request,
                    preparation_result=preparation_result,
                    window=window,
                    preparation_receipt=preparation_receipt,
                )
        except Exception as exc:
            terminal_state = "blocked"
            result = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": "blocked",
                "activation_id": (
                    request["activation_id"]
                    if isinstance(request, Mapping)
                    else re.sub(r"-[0-9a-f]{64}\.json$", "", source.name)
                ),
                "blockers": [
                    str(exc)
                    if isinstance(
                        exc,
                        (
                            TaskEvaluationLaunchActivationWorkerError,
                            TaskEvaluationLaunchPreparationWorkerError,
                            TaskEvaluationSharedMutationWindowError,
                            disk_budget.ControlPlaneDiskBudgetError,
                        ),
                    )
                    else (
                        "launch_activation_worker_failed:KeyError:"
                        + str(exc.args[0])
                        if isinstance(exc, KeyError)
                        and exc.args
                        and re.fullmatch(r"[A-Za-z0-9_.-]+", str(exc.args[0]))
                        else f"launch_activation_worker_failed:{type(exc).__name__}"
                    )
                ],
                "catalog_mutation_state": "unknown_if_preparation_started",
                "provider_mutation_performed": False,
                "paid_execution_requested": False,
                "observed_at_iso": datetime.now(timezone.utc).isoformat(),
                "result_digest": "",
            }
            result["result_digest"] = canonical_digest(
                result, digest_field="result_digest"
            )
        if disk_reservation is not None:
            disk_reservation.release()
        result_path = results_root / source.name
        try:
            write_launch_preparation_record_exclusive(result_path, result)
        except FileExistsError:
            existing = _load_sealed(
                result_path,
                schema_version=RESULT_SCHEMA_VERSION,
                digest_field="result_digest",
            )
            if existing.get("result_digest") != result.get("result_digest"):
                terminal_state = "blocked"
                conflict = {
                    "schema_version": RESULT_SCHEMA_VERSION,
                    "status": "blocked",
                    "activation_id": result.get("activation_id"),
                    "blockers": ["launch_activation_immutable_result_conflict"],
                    "existing_result_digest": existing.get("result_digest"),
                    "candidate_result_digest": result.get("result_digest"),
                    "provider_mutation_performed": False,
                    "paid_execution_requested": False,
                    "observed_at_iso": datetime.now(timezone.utc).isoformat(),
                    "result_digest": "",
                }
                conflict["result_digest"] = canonical_digest(
                    conflict, digest_field="result_digest"
                )
                try:
                    write_launch_preparation_record_exclusive(
                        conflicts_root
                        / f"{source.stem}-{conflict['result_digest'].removeprefix('sha256:')}.json",
                        conflict,
                    )
                except FileExistsError:
                    pass
                result = conflict
            else:
                result = existing
        if (
            terminal_state == "prepared"
            and result.get("run_kind") == "internal_policy_canary"
            and result.get("status") == "policy_campaign_queue_materialized_no_execution"
        ):
            if policy_canary_dispatch_queue_root is None:
                raise TaskEvaluationLaunchActivationWorkerError(
                    "policy_canary_dispatch_queue_root_missing"
                )
            dispatch_root = Path(policy_canary_dispatch_queue_root).expanduser().resolve()
            for name in ("pending", "processing", "completed", "blocked"):
                (dispatch_root / name).mkdir(parents=True, exist_ok=True, mode=0o750)
            dispatch_envelope = {
                "schema_version": "task_evaluation_policy_canary_dispatch_envelope.v1",
                "activation_id": result["activation_id"],
                "run_kind": "internal_policy_canary",
                "claim_ceiling": "diagnostic_policy_execution",
                "source_commit": result["source_commit"],
                "activation_result": {
                    "path": str(result_path),
                    "size_bytes": result_path.stat().st_size,
                    "sha256": _sha256_file(result_path),
                },
                "capture_session_id": result["capture_session_id"],
                "intake_id": result["intake_id"],
                "request_digest": result["website_request_digest"],
                "maximum_provider_allocations": 1,
                "retry_cap": 0,
                "automatic_retry_authorized": False,
                "provider_mutation_performed": False,
                "paid_execution_requested": False,
                "envelope_digest": "",
            }
            dispatch_envelope["envelope_digest"] = canonical_digest(
                dispatch_envelope, digest_field="envelope_digest"
            )
            dispatch_path = (
                dispatch_root
                / "pending"
                / f"{result['activation_id']}-{dispatch_envelope['envelope_digest'][7:]}.json"
            )
            write_launch_preparation_record_exclusive(
                dispatch_path, dispatch_envelope
            )
        os.replace(claimed, root / terminal_state / source.name)
        processed.append(result)
    for lease in processing_leases:
        lease.close()
    return {
        "schema_version": "task_evaluation_launch_activation_queue_run.v1",
        "status": "processed" if processed else "idle",
        "processed_count": len(processed),
        "results": processed,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", default=os.getenv(QUEUE_ROOT_ENV, ""))
    parser.add_argument(
        "--preparation-queue-root", default=os.getenv(PREPARATION_QUEUE_ROOT_ENV, "")
    )
    parser.add_argument(
        "--preparation-input-root", default=os.getenv(PREPARATION_INPUT_ROOT_ENV, "")
    )
    parser.add_argument(
        "--episode-compilation-queue-root",
        default=os.getenv(EPISODE_COMPILATION_QUEUE_ROOT_ENV, ""),
    )
    parser.add_argument(
        "--episode-compilation-output-root",
        default=os.getenv(EPISODE_COMPILATION_OUTPUT_ROOT_ENV, ""),
    )
    parser.add_argument(
        "--scene-construction-queue-root",
        default=os.getenv(SCENE_CONSTRUCTION_QUEUE_ROOT_ENV, ""),
    )
    parser.add_argument(
        "--scene-configuration-toolchain-root",
        default=os.getenv(SCENE_CONFIGURATION_TOOLCHAIN_ROOT_ENV, ""),
    )
    parser.add_argument(
        "--configured-controls-autostart-intent-root",
        default=os.getenv(CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT_ENV, ""),
    )
    parser.add_argument(
        "--policy-canary-dispatch-queue-root",
        default=os.getenv(POLICY_CANARY_DISPATCH_QUEUE_ROOT_ENV, ""),
    )
    parser.add_argument("--activation-root", default=os.getenv(ACTIVATION_ROOT_ENV, ""))
    parser.add_argument(
        "--allowed-uri-prefixes-json", default=os.getenv(ALLOWED_URI_PREFIXES_ENV, "")
    )
    parser.add_argument("--service-account", default=os.getenv(SERVICE_ACCOUNT_ENV, "blueprint"))
    parser.add_argument("--service-group", default=os.getenv(SERVICE_GROUP_ENV, "blueprint"))
    parser.add_argument("--repository-root", default=os.getenv(REPOSITORY_ROOT_ENV, ""))
    parser.add_argument("--destination-prefix", default=os.getenv(DESTINATION_PREFIX_ENV, ""))
    parser.add_argument(
        "--release-window-prefix", default=os.getenv(RELEASE_WINDOW_PREFIX_ENV, "")
    )
    parser.add_argument("--profile-dir", default=os.getenv(PROFILE_DIR_ENV, ""))
    parser.add_argument("--webapp-catalog", default=os.getenv(WEBAPP_CATALOG_ENV, ""))
    parser.add_argument(
        "--standing-authorization-dir",
        default=os.getenv(STANDING_AUTHORIZATION_DIR_ENV, ""),
    )
    parser.add_argument("--max-messages", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        prefixes = json.loads(args.allowed_uri_prefixes_json)
    except json.JSONDecodeError:
        prefixes = None
    required = (
        args.queue_root,
        args.preparation_queue_root,
        args.preparation_input_root,
        args.episode_compilation_queue_root,
        args.episode_compilation_output_root,
        args.scene_construction_queue_root,
        args.scene_configuration_toolchain_root,
        args.configured_controls_autostart_intent_root,
        args.policy_canary_dispatch_queue_root,
        args.activation_root,
        args.repository_root,
        args.destination_prefix,
        args.release_window_prefix,
        args.profile_dir,
        args.webapp_catalog,
        args.standing_authorization_dir,
    )
    if (
        not all(required)
        or not isinstance(prefixes, list)
        or not all(isinstance(item, str) and item for item in prefixes)
    ):
        print(json.dumps({
            "schema_version": "task_evaluation_launch_activation_queue_run.v1",
            "status": "blocked",
            "blockers": ["launch_activation_worker_configuration_invalid"],
            "provider_mutation_performed": False,
            "paid_execution_requested": False,
        }, sort_keys=True))
        return 2
    try:
        result = process_launch_activation_queue(
            queue_root=args.queue_root,
            preparation_queue_root=args.preparation_queue_root,
            preparation_input_root=args.preparation_input_root,
            episode_compilation_queue_root=args.episode_compilation_queue_root,
            episode_compilation_output_root=args.episode_compilation_output_root,
            scene_construction_queue_root=args.scene_construction_queue_root,
            scene_configuration_toolchain_root=(
                args.scene_configuration_toolchain_root
            ),
            configured_controls_autostart_intent_root=(
                args.configured_controls_autostart_intent_root
            ),
            policy_canary_dispatch_queue_root=(
                args.policy_canary_dispatch_queue_root
            ),
            activation_root=args.activation_root,
            allowed_uri_prefixes=prefixes,
            service_account=args.service_account,
            service_group=args.service_group,
            repository_root=args.repository_root,
            destination_prefix=args.destination_prefix,
            release_window_prefix=args.release_window_prefix,
            profile_dir=args.profile_dir,
            webapp_catalog=args.webapp_catalog,
            standing_authorization_dir=args.standing_authorization_dir,
            source_commit=running_worker_source_commit(),
            max_messages=args.max_messages,
            disk_reservation_root=os.getenv("BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT"),
        )
    except (TaskEvaluationLaunchActivationWorkerError, OSError) as exc:
        print(json.dumps({
            "schema_version": "task_evaluation_launch_activation_queue_run.v1",
            "status": "blocked",
            "blockers": [str(exc)],
            "provider_mutation_performed": False,
            "paid_execution_requested": False,
        }, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {"processed", "idle"} else 2


__all__ = [
    "TaskEvaluationLaunchActivationWorkerError",
    "default_activation_preparer",
    "process_launch_activation_queue",
    "validate_release_window_uri",
]

if __name__ == "__main__":
    raise SystemExit(main())
