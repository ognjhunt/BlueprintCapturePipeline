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
import hashlib
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

from .decision_evidence_contracts import canonical_digest
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
    default_reference_fetcher,
    running_worker_source_commit,
    validate_allowed_uri_prefixes,
)
from .task_evaluation_native_arena_preparation_adapter import (
    RESULT_SCHEMA_VERSION as ADAPTER_RESULT_SCHEMA_VERSION,
)
from .task_evaluation_shared_mutation_window import (
    TaskEvaluationSharedMutationWindowError,
    validate_shared_mutation_window,
)


QUEUE_ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_QUEUE_ROOT"
PREPARATION_QUEUE_ROOT_ENV = (
    "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_QUEUE_ROOT"
)
PREPARATION_INPUT_ROOT_ENV = (
    "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_INPUT_ROOT"
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

ReferenceFetcher = Callable[[str, Path, int], None]
ActivationPreparer = Callable[..., dict[str, Any]]


class TaskEvaluationLaunchActivationWorkerError(RuntimeError):
    """One authority-gated activation could not be completed safely."""


def validate_release_window_uri(uri: str, *, prefix: str) -> str:
    """Require coordinator windows to come from an operator-owned prefix."""

    validated_prefix = validate_allowed_uri_prefixes([prefix])[0]
    if not uri.startswith(validated_prefix):
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_release_window_prefix_not_authorized"
        )
    return validated_prefix


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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
    if (
        envelope.get("request_digest") != binding["request_digest"]
        or result.get("result_digest") != binding["result_digest"]
        or result.get("status")
        != "native_arena_inputs_verified_awaiting_profile_authority"
        or result.get("full_byte_service_account_readback_passed") is not True
        or request["preparation_id"] != preparation_id
        or request["team_namespace"] != activation_request["team_namespace"]
        or request["expected_production_commit"]
        != activation_request["expected_production_commit"]
    ):
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_preparation_binding_mismatch"
        )
    preparation_root = (preparation_input_root / preparation_id).resolve()
    expected_references = {
        row["contract_path"]: (row["digest"], row["size_bytes"])
        for row in _collect_references(request)
    }
    materialized_references: dict[str, Path] = {}
    for row in result.get("references") or []:
        if not isinstance(row, Mapping):
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_preparation_reference_invalid"
            )
        contract_path = str(row.get("contract_path") or "")
        path = Path(str(row.get("materialized_path") or "")).resolve()
        expected = expected_references.get(contract_path)
        if (
            expected is None
            or row.get("full_byte_service_account_readback_passed") is not True
            or not _under(path, preparation_root)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != expected[1]
            or _sha256_file(path) != expected[0]
        ):
            raise TaskEvaluationLaunchActivationWorkerError(
                "launch_activation_preparation_reference_invalid"
            )
        materialized_references[contract_path] = path
    if set(materialized_references) != set(expected_references):
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_preparation_reference_set_invalid"
        )
    adapter_path = (
        preparation_root
        / "native-arena-adapter"
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
        adapter.get("result_digest") != result.get("adapter_result_digest")
        or adapter.get("status") != "native_arena_adapter_materialized"
        or adapter.get("preparation_id") != preparation_id
        or adapter.get("source_commit")
        != activation_request["expected_production_commit"]
        or not _under(packet_root, preparation_root)
        or not _under(runtime_receipt, preparation_root)
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
    adapter: Mapping[str, Any],
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
    source_manifest_path = preparation_materialized["scene.source_manifest"]
    rights_admission_path = preparation_materialized["scene.rights.admission"]
    source_manifest = _read_json(
        source_manifest_path, blocker="launch_activation_source_manifest_invalid"
    )
    rights_admission = _read_json(
        rights_admission_path, blocker="launch_activation_rights_admission_invalid"
    )
    packet_root = Path(str(adapter["packet_root"])).resolve()
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
    for index, evidence in enumerate(preparation_request["scene"]["rights"]["evidence"]):
        path = preparation_materialized[f"scene.rights.evidence.{index}.artifact"]
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
        "machine_avoidlist": "",
        "python": sys.executable,
        "service_account": service_account,
        "service_group": service_group,
    }
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
    return {
        "schema_version": "native_task_arena_launch_preparation_context.v2",
        "lane": activation_request["lane"],
        "team_namespace": activation_request["team_namespace"],
        "references": {
            "scene": {
                "scene_id": preparation_request["scene"]["identity"]["id"],
                "packet_dir": str(packet_root),
                "packet_receipt_digest": adapter["packet_receipt_digest"],
                "source_manifest": str(source_manifest_path),
                "source_manifest_digest": source_manifest["source_manifest_digest"],
                "rights_admission": str(rights_admission_path),
                "rights_admission_digest": rights_admission["rights_admission_digest"],
                "rights_evidence": rights_evidence,
            },
            "task": {
                "task_id": preparation_request["task"]["identity"]["id"],
                "task_spec_digest": runtime_contract["task_spec_digest"],
            },
            "robot": {
                "robot_id": preparation_request["robot"]["identity"]["id"],
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
    if (
        preparation_receipt.get("status") != "prepared"
        or preparation_receipt.get("source_commit")
        != request["expected_production_commit"]
        or preparation_receipt.get("provider_allocation_performed") is not False
        or preparation_receipt.get("paid_inference_performed") is not False
    ):
        raise TaskEvaluationLaunchActivationWorkerError(
            "launch_activation_preparation_graph_blocked"
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


def process_launch_activation_queue(
    *,
    queue_root: str | Path,
    preparation_queue_root: str | Path,
    preparation_input_root: str | Path,
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
    source_commit: str | None = None,
    max_messages: int = 1,
    fetcher: ReferenceFetcher = default_reference_fetcher,
    preparer: ActivationPreparer = default_activation_preparer,
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
        terminal_state = "prepared"
        try:
            envelope = _load_sealed(
                claimed,
                schema_version=ENVELOPE_SCHEMA_VERSION,
                digest_field="envelope_digest",
            )
            request = validate_launch_activation_request(envelope["request"])
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
            context = _build_native_context(
                activation_request=request,
                preparation_request=preparation_request,
                adapter=adapter,
                preparation_materialized=preparation_materialized,
                activation_materialized=materialized,
                activation_root=owned_root,
                repository_root=repository,
                destination_prefix=destination_prefix,
                profile_dir=Path(profile_dir).resolve(),
                webapp_catalog=Path(webapp_catalog).resolve(),
                standing_authorization_dir=Path(standing_authorization_dir).resolve(),
                service_account=service_account,
                service_group=service_group,
            )
            context_path = owned_root / "native_task_arena_launch_preparation_context.v2.json"
            write_launch_preparation_record_exclusive(context_path, context)
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
                "activation_id": re.sub(r"-[0-9a-f]{64}\.json$", "", source.name),
                "blockers": [
                    str(exc)
                    if isinstance(
                        exc,
                        (
                            TaskEvaluationLaunchActivationWorkerError,
                            TaskEvaluationSharedMutationWindowError,
                        ),
                    )
                    else f"launch_activation_worker_failed:{type(exc).__name__}"
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
        os.replace(claimed, root / terminal_state / source.name)
        processed.append(result)
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
