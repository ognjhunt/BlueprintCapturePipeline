"""Canonical retry-zero Vast execution for one scene-configuration bundle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .common import (
    ensure_dir,
    redacted_failure_detail,
    utc_now_iso,
    write_json,
)
from .core.common import redacted_failure_text
from .decision_evidence_contracts import canonical_digest
from .openai_api_geography import OPENAI_API_SUPPORTED_COUNTRY_CODES
from .paid_resource_admission import (
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .provider_output_disk_capacity import observe_provider_output_disk_capacity
from .spend_authority_consumption_root import (
    SpendAuthorityRootError,
    prepare_consumption_root,
)
from .task_evaluation_artifact_manifest import (
    PROVIDER_RUN_DIRNAME,
    TaskEvaluationArtifactManifestError,
    build_task_evaluation_artifact_manifest,
    seal_preprovider_unallocated_lane_terminal_artifacts,
    seal_unallocated_provider_teardown,
)
from .task_evaluation_scene_configuration_bundle import (
    PROVIDER_BUNDLE_KIND,
    RESULT_FILENAME,
    load_scene_configuration_provider_bundle_receipt,
)
from .task_evaluation_scene_configuration_diagnostic_checkpoint import (
    validate_scene_configuration_diagnostic_checkpoint,
)
from .task_evaluation_scene_configuration_diagnostic_output import (
    validated_advanced_checkpoint_reference,
)
from .task_evaluation_scene_configuration_execution_binding import (
    provider_execution_binding_blockers as _provider_execution_binding_blockers,
)
from .task_evaluation_configured_scene_object_store import (
    configured_scene_object_store_publisher,
)
from .task_evaluation_scene_configuration_paid_authority import (
    validate_scene_configuration_paid_authority,
)
from .task_evaluation_scene_configuration_publication import (
    RESULT_SCHEMA_VERSION as PUBLICATION_RESULT_SCHEMA_VERSION,
    publish_configured_scene_revision,
)
from .task_evaluation_scene_configuration_runtime_budget import (
    OUTPUT_AND_CLOSURE_RESERVE_SECONDS,
    OUTPUT_CLOSURE_RESERVE_SECONDS_ENV,
    PARENT_DEADLINE_EPOCH_ENV,
    ceil_live_minutes,
    diagnostic_parent_runtime_budget_blockers,
    parent_runtime_budget_blockers,
)
from .task_evaluation_scene_configuration_transfer_budget import (
    scene_configuration_provider_transfer_byte_budget,
)
from .task_evaluation_scene_configuration_provider_cleanup import cleanup_scene_staging
from .task_evaluation_scene_construction_queue import (
    TaskEvaluationSceneConstructionQueueError,
    finalize_scene_construction,
    preflight_scene_construction_finalization,
)
from .task_evaluation_supervisor.openai_cost_authority import (
    OpenAICostAuthorityError,
    OpenAIOrganizationCostsClient,
)
from .task_evaluation_scene_configuration_openai_gate import (
    read_stage_scope_attestation,
    resolve_stage_scope_attestation,
    stage_paid_resource_class,
)
from .vast_independent_watchdog_control import (
    arm_independent_vast_watchdog,
    close_independent_vast_watchdog,
    close_independent_vast_watchdog_without_allocation,
)
from .vast_provider_adapter import run_vast_provider_adapter
from .vast_provider_transfer_upload import EXPECTED_PROVIDER_UPLOAD_BYTES_ENV
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


RESULT_SCHEMA_VERSION = "task_evaluation_scene_configuration_vast_result.v1"
DIAGNOSTIC_RESULT_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_diagnostic_vast_result.v1"
)
#: Provider-namespace label the independent watchdog reaps by. It is not the
#: run's admission identity: the authority's ``resource_name`` names *which
#: attempt* is authorized, while this names *whose instances* the watchdog may
#: destroy. The watchdog refuses any prefix outside the ``blueprint-``
#: namespace so it can never reap another tenant's instance, and appends its
#: own run-unique suffix, so this stays a constant like every other lane's.
WATCHDOG_POD_NAME_PREFIX = "blueprint-task-evaluation-scene-config-"
_VAST_MUTATION_ENV = (
    "BLUEPRINT_ALLOW_VAST_API_CALLS",
    "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH",
)
_VAST_STALE_OFFER_RETRY_ENV = (
    "BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"
)
# One exclusive (key file, key id, operator attestation) triple per OpenAI
# stage. The official-cost gate binds the observed same-day baseline for each
# ``(project_id, api_key_id)`` and charges only this stage's later delta.
_OPENAI_RUNTIME_FILE_ENVS = (
    "OPENAI_ADMIN_API_KEY_FILE",
    "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_FILE",
    "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE",
    "OPENAI_CONTENT_AGENTS_API_KEY_FILE",
    "BLUEPRINT_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_COST_SCOPE_ATTESTATION_FILE",
    "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE",
    "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE",
)
_OPENAI_RUNTIME_VALUE_ENVS = (
    "OPENAI_PROJECT_ID",
    "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_ID",
    "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID",
    "OPENAI_CONTENT_AGENTS_API_KEY_ID",
)
_OPENAI_STAGE_SCOPE_DISTINCT_GROUPS = (
    (
        "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_FILE",
        "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE",
        "OPENAI_CONTENT_AGENTS_API_KEY_FILE",
    ),
    (
        "BLUEPRINT_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_COST_SCOPE_ATTESTATION_FILE",
        "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE",
        "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE",
    ),
    (
        "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_ID",
        "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID",
        "OPENAI_CONTENT_AGENTS_API_KEY_ID",
    ),
)
_OPENAI_STAGE_SCOPE_BINDINGS = (
    (
        "artifixer_semantic_teacher",
        "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_ID",
        "BLUEPRINT_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_COST_SCOPE_ATTESTATION_FILE",
    ),
    (
        "artifixer_visual_review",
        "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID",
        "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE",
    ),
    (
        "content_agents",
        "OPENAI_CONTENT_AGENTS_API_KEY_ID",
        "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE",
    ),
)


class TaskEvaluationSceneConfigurationVastError(RuntimeError):
    """The canonical provider path could not run or close one configuration."""


def _stage_owner_only_runtime_secrets(
    *, job_dir: Path, secret_paths: Mapping[str, str]
) -> tuple[dict[str, str], Path | None]:
    """Copy each validated runtime secret into an owner-only private file.

    Two validators sit on this exact call path and disagree. This lane
    requires each source file to be group readable and no wider
    (``mode & ~0o640`` clear, ``mode & 0o440`` set), because the host keeps
    provider secrets ``root:blueprint 0640`` -- root owns them and the
    service reads them through its group. The Vast adapter then requires
    ``st_mode & 0o077 == 0`` on every path it is handed, because those bytes
    are about to travel toward a rented host, so it refuses anything a group
    can read. A root-owned file cannot satisfy both: owner-only is readable
    by the service only if the service owns it.

    So hand the adapter a copy this service owns, at ``0600``, under the run
    root. Both rules keep their full strength and the staged copy is strictly
    narrower than the source it came from.
    """

    if not secret_paths:
        return {}, None
    root = job_dir / "runtime-secrets"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    staged: dict[str, str] = {}
    for name, unresolved in sorted(secret_paths.items()):
        source = Path(unresolved)
        payload = source.read_bytes()
        if not payload:
            raise TaskEvaluationSceneConfigurationVastError(
                "scene_configuration_openai_runtime_secret_configuration_invalid"
            )
        destination = root / name
        destination.unlink(missing_ok=True)
        descriptor = os.open(
            destination,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.read_bytes() != payload:
            raise TaskEvaluationSceneConfigurationVastError(
                "scene_configuration_openai_runtime_secret_configuration_invalid"
            )
        staged[name] = str(destination)
    return staged, root


def _discard_staged_runtime_secrets(root: Path | None) -> list[str]:
    """Remove private copies and report any byte that could remain retained."""

    if root is None:
        return []
    failed = False
    try:
        children = sorted(root.iterdir())
    except OSError:
        children = []
        failed = True
    for child in children:
        try:
            child.unlink()
        except OSError:
            failed = True
    try:
        root.rmdir()
    except OSError:
        failed = True
    if root.exists():
        failed = True
    return (
        ["scene_configuration_openai_runtime_secret_cleanup_failed"]
        if failed
        else []
    )


def _collect_openai_cost_snapshot(
    *,
    admin_api_key_file: str,
    project_id: str,
    api_key_id: str,
    start_time: int,
    end_time: int,
) -> Mapping[str, Any]:
    """Query one exact stage scope without recording credential material."""

    source = Path(admin_api_key_file)
    payload = source.read_bytes()
    if not payload:
        raise OpenAICostAuthorityError("openai_admin_key_invalid")
    with tempfile.TemporaryDirectory(
        prefix="blueprint-openai-cost-preflight-"
    ) as raw:
        private_key = Path(raw) / "admin-key"
        descriptor = os.open(
            private_key,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if (
            stat.S_IMODE(private_key.stat().st_mode) != 0o600
            or private_key.read_bytes() != payload
        ):
            raise OpenAICostAuthorityError("openai_admin_key_private_copy_invalid")
        return OpenAIOrganizationCostsClient(
            project_id=project_id,
            api_key_id=api_key_id,
            admin_api_key_file=private_key,
        ).snapshot(start_time=start_time, end_time=end_time)


def _provider_runtime_inputs(
    authority: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    openai = authority["external_service_spend_caps"]["openai"]
    if float(openai["maximum_cost_usd"]) <= 0:
        return {}, {}
    secret_paths = {
        name: str(os.environ.get(name) or "").strip()
        for name in _OPENAI_RUNTIME_FILE_ENVS
    }
    values = {
        name: str(os.environ.get(name) or "").strip()
        for name in _OPENAI_RUNTIME_VALUE_ENVS
    }
    if not all(secret_paths.values()) or not all(values.values()):
        raise TaskEvaluationSceneConfigurationVastError(
            "scene_configuration_openai_runtime_secret_configuration_missing"
        )
    for name, unresolved in secret_paths.items():
        path = Path(unresolved).expanduser().absolute()
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as exc:
            raise TaskEvaluationSceneConfigurationVastError(
                "scene_configuration_openai_runtime_secret_configuration_invalid"
            ) from exc
        try:
            metadata = os.fstat(descriptor)
            mode = stat.S_IMODE(metadata.st_mode)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or mode & ~0o640
                or not mode & 0o440
                or not 0 < metadata.st_size <= 65_536
            ):
                raise TaskEvaluationSceneConfigurationVastError(
                    "scene_configuration_openai_runtime_secret_configuration_invalid"
                )
        finally:
            os.close(descriptor)
        secret_paths[name] = str(path)
    for group in _OPENAI_STAGE_SCOPE_DISTINCT_GROUPS:
        observed = [
            secret_paths.get(name) or values.get(name) for name in group
        ]
        if len(set(observed)) != len(group):
            raise TaskEvaluationSceneConfigurationVastError(
                "scene_configuration_openai_stage_scopes_not_distinct"
            )
    # The distinctness check above already proved that each stage holds its own
    # provisioned key, which is the exclusivity a scope receipt asserts. So a
    # missing or pre-rename receipt is resolved here rather than refused: the
    # lane derives an equivalent one and records it as agent-derived. A receipt
    # that is present and valid is still honoured exactly as written.
    for stage, api_key_id_env, attestation_file_env in _OPENAI_STAGE_SCOPE_BINDINGS:
        try:
            resolve_stage_scope_attestation(
                attestation=read_stage_scope_attestation(
                    secret_paths[attestation_file_env]
                ),
                paid_resource_class=stage_paid_resource_class(stage),
                project_id=values["OPENAI_PROJECT_ID"],
                api_key_id=values[api_key_id_env],
            )
        except OpenAICostAuthorityError as exc:
            raise TaskEvaluationSceneConfigurationVastError(
                f"scene_configuration_openai_stage_scope_attestation_invalid:{stage}"
            ) from exc
    now = datetime.now(UTC)
    day_start = datetime(now.year, now.month, now.day, tzinfo=UTC)
    runtime_window_end = now + timedelta(hours=1)
    attribution_end_day = datetime(
        runtime_window_end.year,
        runtime_window_end.month,
        runtime_window_end.day,
        tzinfo=UTC,
    ) + timedelta(days=1)
    for stage, api_key_id_env, _attestation_file_env in _OPENAI_STAGE_SCOPE_BINDINGS:
        try:
            snapshot = _collect_openai_cost_snapshot(
                admin_api_key_file=secret_paths["OPENAI_ADMIN_API_KEY_FILE"],
                project_id=values["OPENAI_PROJECT_ID"],
                api_key_id=values[api_key_id_env],
                start_time=int(day_start.timestamp()),
                end_time=int(attribution_end_day.timestamp()),
            )
            if not isinstance(snapshot, Mapping):
                raise ValueError("openai_cost_snapshot_invalid")
            observed_cost = snapshot.get("total_cost_usd")
            if (
                isinstance(observed_cost, bool)
                or not math.isfinite(float(observed_cost))
                or float(observed_cost) < 0
            ):
                raise ValueError("openai_cost_snapshot_invalid")
        except (KeyError, TypeError, ValueError, OpenAICostAuthorityError) as exc:
            raise TaskEvaluationSceneConfigurationVastError(
                f"scene_configuration_openai_stage_cost_baseline_invalid:{stage}"
            ) from exc
    stage_caps = openai["stage_max_cost_usd"]
    runtime_environment = {
        **values,
        "BLUEPRINT_SCENE_CONFIGURATION_AUTHORITY_DIGEST": str(
            authority["authority_digest"]
        ),
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD": str(
            openai["maximum_cost_usd"]
        ),
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_REQUESTS": str(
            openai["maximum_requests"]
        ),
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_MAX_COST_USD": str(
            stage_caps["artifixer_semantic_teacher"]
        ),
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_ARTIFIXER_VISUAL_REVIEW_MAX_COST_USD": str(
            stage_caps["artifixer_visual_review"]
        ),
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_CONTENT_AGENTS_MAX_COST_USD": str(
            stage_caps["content_agents"]
        ),
    }
    return secret_paths, runtime_environment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationVastError(
            "scene_configuration_vast_json_invalid"
        ) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationVastError(
            "scene_configuration_vast_json_invalid"
        )
    return dict(value)


def _validated_advanced_checkpoint_reference(
    *, extraction_root: Path, result: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    return validated_advanced_checkpoint_reference(
        extraction_root=extraction_root,
        result=result,
        checkpoint_validator=validate_scene_configuration_diagnostic_checkpoint,
    )


def _consume_authority_once(
    authority: Mapping[str, Any], *, source_commit: str
) -> dict[str, Any]:
    digest = str(authority.get("authority_digest") or "")
    if not digest.startswith("sha256:") or len(digest) != 71:
        return {"status": "blocked", "blockers": ["authority_identity_invalid"]}
    identity = digest.removeprefix("sha256:")
    try:
        root = prepare_consumption_root()
    except SpendAuthorityRootError as exc:
        return {"status": "blocked", "blockers": [str(exc)]}
    record = {
        "schema_version": "task_evaluation_scene_configuration_authority_consumption.v1",
        "authority_digest": digest,
        "bundle_sha256": authority.get("bundle_sha256"),
        "source_commit": source_commit,
        "consumed_at": utc_now_iso(),
        "maximum_provider_allocations": 1,
        "retry_cap": 0,
    }
    payload = (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    temporary = root / f".{identity}.{os.getpid()}.tmp"
    destination = root / f"scene-configuration-{identity}.json"
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    except FileExistsError:
        return {"status": "blocked", "blockers": ["authority_already_consumed"]}
    except OSError:
        return {
            "status": "blocked",
            "blockers": ["authority_consumption_write_failed"],
        }
    return {
        "status": "consumed",
        "authorization_digest": digest,
        "consumption_record_sha256": "sha256:"
        + hashlib.sha256(payload).hexdigest(),
        "record_location_disclosed": False,
    }


def _completed_stage_chain_valid(
    chain: Any,
    *,
    provider_result: Mapping[str, Any],
    diagnostic_only: bool,
) -> bool:
    if not isinstance(chain, Mapping):
        return False
    rows = chain.get("stage_results")
    if not isinstance(rows, list) or len(rows) != 6:
        return False
    stage_ids: set[str] = set()
    row_digests: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        stage_id = str(row.get("stage_id") or "")
        digest = str(row.get("stage_result_digest") or "")
        if (
            not stage_id
            or stage_id in stage_ids
            or row.get("schema_version")
            != "task_evaluation_scene_configuration_stage_result.v1"
            or row.get("status") != "completed"
            or row.get("canonical_allocator") is not None
            or row.get("provider_mutations_performed") != 0
            or row.get("paid_execution_requested") is not False
            or row.get("executed_inside_parent_configuration_run") is not True
            or row.get("raw_secret_values_recorded") is not False
            or not isinstance(row.get("output_artifacts"), list)
            or digest != canonical_digest(row, digest_field="stage_result_digest")
            or (
                diagnostic_only
                and (
                    row.get("diagnostic_only") is not True
                    or row.get("qualification_eligible") is not False
                    or row.get("executed_inside_one_parent_provider_run") is not False
                    or row.get("configured_revision_publication_permitted") is not False
                    or row.get("offering_publication_permitted") is not False
                    or row.get("terminal_e2e_completion_permitted") is not False
                )
            )
            or (
                not diagnostic_only
                and (
                    row.get("diagnostic_only") is True
                    or row.get("qualification_eligible") is False
                    or row.get("executed_inside_one_parent_provider_run") is False
                    or row.get("configured_revision_publication_permitted") is False
                    or row.get("offering_publication_permitted") is False
                )
            )
        ):
            return False
        stage_ids.add(stage_id)
        row_digests.append(digest)
    expected_schema = (
        "task_evaluation_scene_configuration_diagnostic_stage_chain.v1"
        if diagnostic_only
        else "task_evaluation_scene_configuration_provider_stage_chain.v1"
    )
    expected_status = (
        "completed_diagnostic_only_not_qualification_eligible"
        if diagnostic_only
        else "completed"
    )
    return bool(
        chain.get("schema_version") == expected_schema
        and chain.get("status") == expected_status
        and str(chain.get("run_id") or "")
        and (
            diagnostic_only
            or chain.get("run_id") == provider_result.get("run_id")
        )
        and chain.get("stage_count") == 6
        and chain.get("stage_result_digests") == row_digests
        and chain.get("executed_inside_one_parent_provider_run")
        is (not diagnostic_only)
        and chain.get("nested_provider_mutations_performed") == 0
        and chain.get("nested_paid_execution_requested") is False
        and chain.get("evaluation_episode_executed") is False
        and chain.get("retry_cap") == 0
        and chain.get("result_digest")
        == canonical_digest(chain, digest_field="result_digest")
    )


def _extract_provider_output(
    archive_path: Path,
    destination: Path,
    *,
    maximum_archive_bytes: int,
    diagnostic_only: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if (
        isinstance(maximum_archive_bytes, bool)
        or not isinstance(maximum_archive_bytes, int)
        or maximum_archive_bytes <= 0
    ):
        return {}, ["scene_configuration_provider_output_limit_invalid"]
    if destination.exists():
        return {}, ["scene_configuration_provider_output_destination_exists"]
    destination.mkdir(parents=True, mode=0o750)
    root = destination.resolve()
    if archive_path.is_file() and archive_path.stat().st_size > maximum_archive_bytes:
        return {}, [
            "scene_configuration_provider_output_zip_exceeds_declared_transfer_ceiling"
        ]
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(members) > PROVIDER_OUTPUT_MAXIMUM_MEMBER_COUNT:
                blockers.append(
                    "scene_configuration_provider_output_archive_member_count_invalid"
                )
            if len(names) != len(set(names)):
                blockers.append(
                    "scene_configuration_provider_output_archive_duplicate_member"
                )
            if sum(member.file_size for member in members) > (
                maximum_archive_bytes * PROVIDER_OUTPUT_MAXIMUM_EXPANSION_RATIO
            ):
                blockers.append(
                    "scene_configuration_provider_output_archive_expansion_invalid"
                )
            for member in members:
                target = (root / member.filename).resolve()
                mode = member.external_attr >> 16
                if (
                    (target != root and root not in target.parents)
                    or stat.S_ISLNK(mode)
                ):
                    blockers.append(
                        "scene_configuration_provider_output_archive_unsafe"
                    )
            if not blockers:
                archive.extractall(root)
    except (
        EOFError,
        NotImplementedError,
        OSError,
        RuntimeError,
        ValueError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ):
        blockers.append("scene_configuration_provider_output_zip_invalid")
    if blockers:
        return {}, sorted(set(blockers))
    result_path = root / RESULT_FILENAME
    if not result_path.is_file():
        blockers.append("scene_configuration_provider_result_missing")
        return {}, sorted(set(blockers))
    try:
        result = _read(result_path)
    except TaskEvaluationSceneConfigurationVastError:
        blockers.append("scene_configuration_provider_result_contract_invalid")
        return {}, sorted(set(blockers))
    provider_blockers = result.get("blockers")
    provider_blockers_valid = bool(
        isinstance(provider_blockers, list)
        and len(provider_blockers) <= 32
        and all(
            isinstance(item, str) and bool(item.strip())
            for item in provider_blockers
        )
    )
    expected_provider_schema = (
        "task_evaluation_scene_configuration_diagnostic_provider_result.v1"
        if diagnostic_only
        else "task_evaluation_scene_configuration_provider_result.v1"
    )
    if (
        result.get("schema_version") != expected_provider_schema
        or result.get("result_digest")
        != canonical_digest(result, digest_field="result_digest")
        or result.get("provider_zero_required_after_return") is not True
        or not provider_blockers_valid
    ):
        blockers.append("scene_configuration_provider_result_contract_invalid")
    if diagnostic_only:
        if (
            result.get("diagnostic_only") is not True
            or result.get("qualification_eligible") is not False
            or result.get("executed_inside_one_parent_provider_run") is not False
            or result.get("configured_revision_publication_permitted") is not False
            or result.get("offering_publication_permitted") is not False
            or result.get("terminal_e2e_completion_permitted") is not False
            or result.get("raw_secret_values_recorded") is not False
        ):
            blockers.append("scene_configuration_diagnostic_claim_boundary_invalid")
    elif (
        result.get("evaluation_episode_executed") is not False
        or result.get("candidate_policy_queried") is not False
    ):
        blockers.append("scene_configuration_provider_result_contract_invalid")
    completed_status = (
        "completed_diagnostic_only_not_qualification_eligible"
        if diagnostic_only
        else "completed"
    )
    if result.get("status") == completed_status:
        if provider_blockers != []:
            blockers.append("scene_configuration_provider_result_contract_invalid")
        chain = result.get(
            "diagnostic_stage_chain" if diagnostic_only else "stage_chain"
        )
        if not _completed_stage_chain_valid(
            chain, provider_result=result, diagnostic_only=diagnostic_only
        ):
            blockers.append("scene_configuration_stage_chain_invalid")
        if diagnostic_only and not blockers:
            advanced_reference, advanced_blocker = (
                _validated_advanced_checkpoint_reference(
                    extraction_root=root, result=result
                )
            )
            if advanced_blocker is not None:
                blockers.append(advanced_blocker)
            elif advanced_reference is not None:
                result["_validated_advanced_checkpoint"] = advanced_reference
    elif result.get("status") == (
        "blocked_diagnostic_only" if diagnostic_only else "blocked"
    ):
        if not provider_blockers_valid or not provider_blockers:
            blockers.append("scene_configuration_provider_result_contract_invalid")
        else:
            for item in provider_blockers:
                detail = " ".join(redacted_failure_text(item).split())
                if len(detail) > 300:
                    detail = detail[:297] + "..."
                blockers.append(f"provider_result_blocker:{detail}")
        if diagnostic_only and result.get("advanced_checkpoint") is not None:
            advanced_reference, advanced_blocker = (
                _validated_advanced_checkpoint_reference(
                    extraction_root=root, result=result
                )
            )
            if advanced_blocker is not None:
                blockers.append(advanced_blocker)
            elif advanced_reference is not None:
                result["_validated_advanced_checkpoint"] = advanced_reference
    else:
        blockers.append("scene_configuration_provider_result_status_invalid")
    return result, sorted(set(blockers))


def _portable_construction_envelope(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = Path(str(receipt.get("bundle_path") or ""))
    try:
        with zipfile.ZipFile(bundle) as archive:
            value = json.loads(
                archive.read(
                    "provider_runtime/input/portable_construction_envelope.v1.json"
                ).decode("utf-8")
            )
    except (
        KeyError,
        OSError,
        UnicodeError,
        ValueError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
    ) as exc:
        raise TaskEvaluationSceneConfigurationVastError(
            "scene_configuration_publication_envelope_unavailable"
        ) from exc
    envelope = dict(value) if isinstance(value, Mapping) else {}
    if (
        envelope.get("schema_version")
        != "task_evaluation_scene_construction_envelope.v1"
        or envelope.get("envelope_digest")
        != canonical_digest(envelope, digest_field="envelope_digest")
        or envelope.get("envelope_digest")
        != receipt.get("portable_construction_envelope_digest")
        or envelope.get("expected_production_commit")
        != receipt.get("source_commit")
        or envelope.get("run_id") != receipt.get("run_id")
    ):
        raise TaskEvaluationSceneConfigurationVastError(
            "scene_configuration_publication_envelope_invalid"
        )
    return envelope


def _publication_stage_results(
    execution: Mapping[str, Any], *, extraction_root: Path
) -> list[dict[str, Any]]:
    root = extraction_root.resolve()
    chain = execution.get("stage_chain")
    stage_results = chain.get("stage_results") if isinstance(chain, Mapping) else None
    if not isinstance(stage_results, list) or len(stage_results) != 6:
        raise TaskEvaluationSceneConfigurationVastError(
            "scene_configuration_publication_stage_results_invalid"
        )
    hydrated = json.loads(json.dumps(stage_results))
    for result in hydrated:
        artifacts = result.get("output_artifacts") if isinstance(result, Mapping) else None
        if not isinstance(artifacts, list):
            raise TaskEvaluationSceneConfigurationVastError(
                "scene_configuration_provider_artifact_portability_invalid"
            )
        for artifact in artifacts:
            relative = str(
                artifact.get("provider_output_relative_path")
                if isinstance(artifact, Mapping)
                else ""
            )
            relative_path = Path(relative)
            if (
                not relative
                or relative_path.is_absolute()
                or ".." in relative_path.parts
            ):
                raise TaskEvaluationSceneConfigurationVastError(
                    "scene_configuration_provider_artifact_portability_invalid"
                )
            target = root / relative_path
            if target.is_symlink():
                raise TaskEvaluationSceneConfigurationVastError(
                    "scene_configuration_provider_artifact_portability_invalid"
                )
            target = target.resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise TaskEvaluationSceneConfigurationVastError(
                    "scene_configuration_provider_artifact_portability_invalid"
                ) from exc
            if (
                not target.is_file()
                or target.stat().st_size != artifact.get("size_bytes")
                or _sha256(target) != artifact.get("digest")
            ):
                raise TaskEvaluationSceneConfigurationVastError(
                    "scene_configuration_provider_artifact_portability_invalid"
                )
            artifact["path"] = str(target)
    return hydrated


def _publish_completed_configuration(
    *,
    receipt: Mapping[str, Any],
    execution: Mapping[str, Any],
    extraction_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(mode=0o750)
    publication = publish_configured_scene_revision(
        envelope=_portable_construction_envelope(receipt),
        stage_results=_publication_stage_results(
            execution, extraction_root=extraction_root
        ),
        output_root=output_root,
        publisher=configured_scene_object_store_publisher(),
    )
    write_json(
        output_root / f"{PUBLICATION_RESULT_SCHEMA_VERSION}.json",
        publication,
    )
    return publication


@contextmanager
def _authority_environment():
    names = (*_VAST_MUTATION_ENV, _VAST_STALE_OFFER_RETRY_ENV)
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in _VAST_MUTATION_ENV:
            os.environ[name] = "1"
        os.environ[_VAST_STALE_OFFER_RETRY_ENV] = "0"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


#: The bundle is not all this lane pulls. Before the first stage the onstart
#: apt-installs its build and render toolchain, and the ArtiFixer runtime then
#: fetches ``uv`` and builds a venv. Declaring only ``bundle_size_bytes``
#: left all of that outside the hard-cap projection.
#:
#: The production ``dual_target_artifixer3d_only`` path skips the VIBE editor,
#: but it does *not* skip Torch: ``run_public_scene_artifixer3d.sh`` installs
#: the CUDA 12.8 build before importing the 3DGRUT JIT graph.  The exact Torch
#: wheel plus only cuDNN, NCCL, cuSPARSELt, NVSHMEM, and torchvision already
#: total this many bytes in their publisher indexes.  CUDA Toolkit components,
#: Triton, the remaining Python closure, apt packages, ``uv``, and pinned source
#: fetches are additional.  The previous 2 GB allowance was therefore smaller
#: than a provable subset of what the caller downloads and underpriced every
#: offer before allocation.
ARTIFIXER_PINNED_WHEEL_DOWNLOAD_FLOOR_BYTES = 2_209_255_046

#: Conservative pre-allocation ceiling for all non-bundle downloads above.
#: Keep this fail-closed: a tighter value needs a complete publisher-size
#: inventory, not an assumption that one of the executed install branches is
#: skipped.
PROVISIONING_DOWNLOAD_OVERHEAD_BYTES = 10_000_000_000

#: The result ZIP is bounded before its signed PUT is used.  The floor leaves
#: room for generated stage evidence even when a synthetic/test bundle is
#: tiny; the receipt-relative term scales for production bundles containing
#: the scene, renderer, browser, and native components.
PROVIDER_OUTPUT_UPLOAD_MINIMUM_BYTES = 1_000_000_000
PROVIDER_OUTPUT_UPLOAD_BUNDLE_MULTIPLIER = 2
PROVIDER_OUTPUT_MAXIMUM_EXPANSION_RATIO = 4
PROVIDER_OUTPUT_MAXIMUM_MEMBER_COUNT = 10_000
#: Space retained beyond the declared archive and its maximum permitted
#: expansion for filesystem metadata, terminal manifests, and publication
#: temporaries. This is operational headroom, not a larger ZIP allowance.
PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES = 512 * 1024 * 1024


def _provider_output_disk_requirements(maximum_archive_bytes: int) -> dict[str, int]:
    """Return the one capacity formula used before transfer and extraction."""

    if (
        isinstance(maximum_archive_bytes, bool)
        or not isinstance(maximum_archive_bytes, int)
        or maximum_archive_bytes <= 0
        or PROVIDER_OUTPUT_MAXIMUM_EXPANSION_RATIO < 1
        or PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES <= 0
    ):
        raise TaskEvaluationSceneConfigurationVastError(
            "scene_configuration_provider_output_disk_requirement_invalid"
        )
    maximum_expanded_bytes = (
        maximum_archive_bytes * PROVIDER_OUTPUT_MAXIMUM_EXPANSION_RATIO
    )
    return {
        "maximum_archive_bytes": maximum_archive_bytes,
        "maximum_expanded_bytes": maximum_expanded_bytes,
        "operational_reserve_bytes": PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES,
        "required_free_bytes_before_download": (
            maximum_archive_bytes
            + maximum_expanded_bytes
            + PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES
        ),
        # At this checkpoint the ZIP is already resident, so free space no
        # longer includes those bytes. Counting it again would require 2x the
        # archive without protecting any additional write.
        "required_free_bytes_before_extraction": (
            maximum_expanded_bytes + PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES
        ),
    }


def _provider_output_disk_capacity(
    *,
    destination_directory: Path,
    required_free_bytes: int,
    phase: str,
    disk_usage_provider: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Measure the destination filesystem and fail closed on unknown capacity."""

    return observe_provider_output_disk_capacity(
        destination_directory=destination_directory,
        required_free_bytes=required_free_bytes,
        phase=phase,
        schema_version="scene_configuration_provider_output_disk_capacity.v1",
        blocker_prefix="scene_configuration_provider_output",
        disk_usage_provider=disk_usage_provider,
    )


def _extract_provider_output_with_capacity_guard(
    archive_path: Path,
    destination: Path,
    *,
    maximum_archive_bytes: int,
    diagnostic_only: bool = False,
    disk_usage_provider: Callable[[Path], Any] | None = None,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Re-stat free bytes immediately before invoking the unchanged extractor."""

    requirements = _provider_output_disk_requirements(maximum_archive_bytes)
    capacity = _provider_output_disk_capacity(
        destination_directory=destination.parent,
        required_free_bytes=requirements[
            "required_free_bytes_before_extraction"
        ],
        phase="before_extraction",
        disk_usage_provider=disk_usage_provider,
    )
    if capacity["status"] != "ready":
        return {}, list(capacity["blockers"]), capacity
    result, blockers = _extract_provider_output(
        archive_path,
        destination,
        maximum_archive_bytes=maximum_archive_bytes,
        diagnostic_only=diagnostic_only,
    )
    return result, blockers, capacity


def _seal_terminal_result(job: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Persist every terminal return, including refusals before allocation."""

    result = dict(value)
    result.setdefault("generated_at", utc_now_iso())
    result.setdefault("raw_secret_values_recorded", False)
    result["result_digest"] = ""
    result["result_digest"] = canonical_digest(
        result, digest_field="result_digest"
    )
    write_json(job / f"{RESULT_SCHEMA_VERSION}.json", result)
    return result


def _seal_live_terminal_result(
    job: Path,
    value: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    scene_construction_queue_root: str | Path | None,
) -> dict[str, Any]:
    """Finalize the originating queue item before exposing a live terminal result."""

    result = dict(value)
    blockers = [str(item) for item in result.get("blockers") or [] if str(item)]
    try:
        if scene_construction_queue_root is None or not str(
            scene_construction_queue_root
        ).strip():
            raise TaskEvaluationSceneConstructionQueueError(
                "scene_construction_queue_finalization_root_missing"
            )
        finalization = finalize_scene_construction(
            queue_root=scene_construction_queue_root,
            envelope=_portable_construction_envelope(receipt),
            terminal_result=result,
        )
    except (OSError, ValueError) as exc:
        blockers.append(
            "scene_construction_queue_finalization_failed:"
            + redacted_failure_detail(exc)
        )
        finalization = {
            "schema_version": "task_evaluation_scene_construction_finalization.v1",
            "status": "blocked",
            "finalization_performed": False,
            "blockers": [redacted_failure_detail(exc)],
        }
    result["scene_construction_queue_finalization"] = finalization
    expected_queue_state = (
        "completed"
        if result.get("status") == "completed" and not blockers
        else "blocked"
    )
    if (
        finalization.get("finalization_performed") is not True
        or finalization.get("queue_state") != expected_queue_state
    ):
        blockers.append("scene_construction_queue_finalization_not_completed")
        result["status"] = "blocked"
    result["blockers"] = sorted(set(blockers))
    result = seal_preprovider_unallocated_lane_terminal_artifacts(
        result,
        attempt_root=job,
        lane=PROVIDER_BUNDLE_KIND,
        reason="scene_configuration_provider_adapter_not_invoked",
        binding={
            "source_commit": receipt.get("source_commit"),
            "bundle_sha256": receipt.get("bundle_sha256"),
            "provider": "vast",
            "result_schema_version": RESULT_SCHEMA_VERSION,
        },
    )
    return _seal_terminal_result(job, result)


def _provider_transfer_byte_budget(
    receipt: Mapping[str, Any],
) -> tuple[int, int]:
    return scene_configuration_provider_transfer_byte_budget(
        receipt,
        provisioning_download_overhead_bytes=PROVISIONING_DOWNLOAD_OVERHEAD_BYTES,
        artifixer_pinned_wheel_download_floor_bytes=(
            ARTIFIXER_PINNED_WHEEL_DOWNLOAD_FLOOR_BYTES
        ),
        provider_output_upload_minimum_bytes=PROVIDER_OUTPUT_UPLOAD_MINIMUM_BYTES,
        provider_output_upload_bundle_multiplier=(
            PROVIDER_OUTPUT_UPLOAD_BUNDLE_MULTIPLIER
        ),
        error_factory=TaskEvaluationSceneConfigurationVastError,
    )


def _close_watchdog_after_adapter(
    *,
    job_dir: Path,
    handle: Any,
    adapter: Mapping[str, Any],
    teardown: Mapping[str, Any],
    instance_ids: list[int],
) -> dict[str, Any]:
    """Close only from exact instance teardown or independently proven zero.

    A rejected create request sets ``provider_create_attempted`` because the
    mutation boundary was reached, but the adapter can still prove that Vast
    returned no instance identity and that no side effect may have occurred.
    Treating every attempted create as ambiguous left the independent watchdog
    alive until its hard TTL even after provider-zero was true.  The no-
    allocation closer is the stronger path here: it refuses if a started-id
    file exists and double-reads both lane-scoped and global Vast inventory
    before publishing terminal evidence.
    """

    provider_teardown_completed = (
        teardown.get("continuing_spend_from_this_run") is False
    )
    rejected_create_proves_no_allocation = bool(
        not instance_ids
        and provider_teardown_completed
        and adapter.get("provider_create_attempted") is True
        and adapter.get("vast_side_effects_may_have_occurred") is False
    )
    if rejected_create_proves_no_allocation:
        return close_independent_vast_watchdog_without_allocation(
            job_dir=job_dir,
            handle=handle,
        )
    return close_independent_vast_watchdog(
        job_dir=job_dir,
        handle=handle,
        instance_ids=instance_ids,
        provider_teardown_completed=provider_teardown_completed,
        provider_allocation_impossible=(
            not instance_ids and adapter.get("provider_create_attempted") is not True
        ),
    )
def _recover_escaped_adapter_failure(
    *,
    provider_run: Path,
    started_instance_id_path: Path,
    failure_detail: str,
) -> tuple[dict[str, Any], bool]:
    """Preserve post-create identity when adapter finalization escapes.

    The adapter writes the independent watchdog handoff immediately after Vast
    returns an instance id.  Its own ``finally`` normally destroys that
    instance and seals the adapter and teardown receipts, but an error while
    sealing those receipts can escape to this caller.  In that case the
    handoff is the only durable proof that allocation occurred; calling the
    unallocated sealer would contradict it.
    """

    started_path_present = (
        started_instance_id_path.exists() or started_instance_id_path.is_symlink()
    )
    instance_ids: list[int] = []
    blockers = [f"vast_adapter_failed:{failure_detail}"]
    if started_path_present:
        try:
            if started_instance_id_path.is_symlink():
                raise ValueError("started instance id path is a symlink")
            candidate = started_instance_id_path.read_text(encoding="utf-8").strip()
            instance_id = int(candidate)
            if instance_id <= 0 or candidate != str(instance_id):
                raise ValueError("started instance id is not canonical")
            instance_ids.append(instance_id)
        except (OSError, ValueError):
            blockers.append("vast_started_instance_id_evidence_invalid")

    adapter = {
        "schema_version": "vast_provider_adapter_result.v1",
        "status": "blocked",
        "reason": "vast_adapter_finalization_failed",
        "blockers": blockers,
        "provider_create_attempted": started_path_present,
        "vast_side_effects_may_have_occurred": started_path_present,
        "vast_instance_ids": instance_ids,
        "retained_owned": False,
        "continuing_spend_from_this_run": started_path_present,
        "raw_secret_values_recorded": False,
    }
    if started_path_present:
        try:
            write_json(
                provider_run / "vast_provider_adapter_result.json",
                adapter,
            )
        except OSError:
            adapter["blockers"].append(
                "vast_adapter_failure_receipt_write_failed"
            )
    return adapter, started_path_present


def run_scene_configuration_vast(
    *,
    job_dir: str | Path,
    bundle_receipt_path: str | Path,
    paid_attempt_authority_path: str | Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    diagnostic_only: bool = False,
    retain_warm_session: bool = False,
    warm_session_authority_path: str | Path | None = None,
    warm_session_output_root: str | Path | None = None,
    scene_construction_queue_root: str | Path | None = None,
    disk_usage_provider: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Run exactly one configuration allocation and close every owned resource."""

    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    receipt = load_scene_configuration_provider_bundle_receipt(
        bundle_receipt_path, diagnostic_only=diagnostic_only
    )
    result_schema_version = (
        DIAGNOSTIC_RESULT_SCHEMA_VERSION
        if diagnostic_only
        else RESULT_SCHEMA_VERSION
    )
    blocked_status = "blocked_diagnostic_only" if diagnostic_only else "blocked"
    diagnostic_claim_boundary = (
        {
            "diagnostic_only": True,
            "qualification_eligible": False,
            "executed_inside_one_parent_provider_run": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
        }
        if diagnostic_only
        else {}
    )
    authority_path = Path(paid_attempt_authority_path).expanduser().resolve()
    authority = validate_scene_configuration_paid_authority(
        _read(authority_path), bundle_receipt=receipt
    )
    warm_session_authority: dict[str, Any] | None = None
    if retain_warm_session:
        from .task_evaluation_scene_configuration_warm_bootstrap import (  # noqa: PLC0415
            validate_warm_bootstrap_request,
        )

        warm_session_authority = validate_warm_bootstrap_request(
            requested=True, diagnostic_only=diagnostic_only,
            authority_path=warm_session_authority_path,
            output_root=warm_session_output_root, bundle_receipt=receipt,
            paid_authority=authority,
            error_factory=TaskEvaluationSceneConfigurationVastError,
        )
    compute_cap = float(authority["provider_compute_spend_cap_usd"])
    rate = float(authority["maximum_hourly_rate_usd"])
    ttl = int(authority["maximum_single_resource_ttl_seconds"])
    runtime_budget_blockers = (
        diagnostic_parent_runtime_budget_blockers(
            completed_stage_prefix_count=int(
                receipt.get("carried_completed_stage_count") or 0
            ),
            ttl_seconds=ttl,
            maximum_hourly_rate_usd=rate,
            provider_compute_spend_cap_usd=compute_cap,
        )
        if diagnostic_only
        else parent_runtime_budget_blockers(
            ttl_seconds=ttl,
            maximum_hourly_rate_usd=rate,
            provider_compute_spend_cap_usd=compute_cap,
        )
    )
    if runtime_budget_blockers:
        blocked = {
            "schema_version": result_schema_version,
            "status": blocked_status,
            "run_id": receipt["run_id"],
            "source_commit": receipt["source_commit"],
            "bundle_sha256": receipt["bundle_sha256"],
            "authority_digest": authority["authority_digest"],
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            **diagnostic_claim_boundary,
            "continuing_spend_from_this_run": False,
            "blockers": runtime_budget_blockers,
        }
        if execute:
            return _seal_live_terminal_result(
                job,
                blocked,
                receipt=receipt,
                scene_construction_queue_root=scene_construction_queue_root,
            )
        return _seal_terminal_result(job, blocked)
    if execute and not diagnostic_only:
        try:
            if scene_construction_queue_root is None or not str(
                scene_construction_queue_root
            ).strip():
                raise TaskEvaluationSceneConstructionQueueError(
                    "scene_construction_queue_finalization_root_missing"
                )
            preflight_scene_construction_finalization(
                queue_root=scene_construction_queue_root,
                envelope=_portable_construction_envelope(receipt),
            )
        except (OSError, ValueError) as exc:
            blocked = {
                "schema_version": result_schema_version,
                "status": blocked_status,
                "run_id": receipt["run_id"],
                "source_commit": receipt["source_commit"],
                "bundle_sha256": receipt["bundle_sha256"],
                "authority_digest": authority["authority_digest"],
                "provider_mutations_performed": 0,
                "retry_cap": 0,
                **diagnostic_claim_boundary,
                "continuing_spend_from_this_run": False,
                "blockers": [
                    "scene_construction_queue_finalization_preflight_failed:"
                    + redacted_failure_detail(exc)
                ],
            }
            return _seal_live_terminal_result(
                job,
                blocked,
                receipt=receipt,
                scene_construction_queue_root=scene_construction_queue_root,
            )
    runtime_secret_paths, runtime_environment = _provider_runtime_inputs(authority)
    if not execute:
        return _seal_terminal_result(
            job,
            {
                "schema_version": result_schema_version,
                "generated_at": utc_now_iso(),
                "status": (
                    "dry_run_ready_diagnostic_only"
                    if diagnostic_only
                    else "dry_run_ready"
                ),
                "run_id": receipt["run_id"],
                "source_commit": receipt["source_commit"],
                "bundle_sha256": receipt["bundle_sha256"],
                "authority_digest": authority["authority_digest"],
                "provider_mutations_performed": 0,
                "retry_cap": 0,
                **diagnostic_claim_boundary,
                "blockers": [],
            },
        )
    if paid_resource_admission_grant is None:
        raise TaskEvaluationSceneConfigurationVastError(
            "scene_configuration_paid_admission_missing"
        )
    require_paid_resource_admission_grant(
        paid_resource_admission_grant,
        resource_class="vast_provider_adapter",
        require_allocation_binding=True,
    )
    external_cap = float(
        authority["external_service_spend_caps"]["openai"]["maximum_cost_usd"]
    )
    provider_all_in_cap = float(authority["hard_attempt_spend_cap_usd"]) - external_cap
    live_minutes = ceil_live_minutes(ttl)
    bundle_path = Path(str(receipt["bundle_path"])).resolve()
    expected_download_bytes, expected_upload_bytes = _provider_transfer_byte_budget(
        receipt
    )
    provider_run = job / PROVIDER_RUN_DIRNAME
    ensure_dir(provider_run)
    output_disk_requirements = _provider_output_disk_requirements(
        expected_upload_bytes
    )
    preallocation_disk_capacity = _provider_output_disk_capacity(
        destination_directory=provider_run,
        required_free_bytes=output_disk_requirements[
            "required_free_bytes_before_download"
        ],
        phase="before_allocation_and_staging",
        disk_usage_provider=disk_usage_provider,
    )
    if preallocation_disk_capacity["status"] != "ready":
        return _seal_live_terminal_result(
            job,
            {
                "schema_version": result_schema_version,
                "status": blocked_status,
                "run_id": receipt["run_id"],
                "source_commit": receipt["source_commit"],
                "bundle_sha256": receipt["bundle_sha256"],
                "authority_digest": authority["authority_digest"],
                "provider_mutations_performed": 0,
                "retry_cap": 0,
                **diagnostic_claim_boundary,
                "continuing_spend_from_this_run": False,
                "expected_provider_download_bytes": expected_download_bytes,
                "expected_provider_upload_bytes": expected_upload_bytes,
                "provider_output_disk_requirements": output_disk_requirements,
                "provider_output_disk_capacity": preallocation_disk_capacity,
                "blockers": list(preallocation_disk_capacity["blockers"]),
            },
            receipt=receipt,
            scene_construction_queue_root=scene_construction_queue_root,
        )
    runtime_environment = dict(runtime_environment)
    runtime_environment[EXPECTED_PROVIDER_UPLOAD_BYTES_ENV] = str(
        expected_upload_bytes
    )
    staging_dir = job / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir,
        bundle_path=bundle_path,
        key_prefix="blueprint/arm-decision-proof-v1/scene-configuration",
        expiration_seconds=ttl + 1_800,
    )
    if staging.get("status") != "completed":
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
        blockers = list(staging.get("blockers") or ["object_store_staging_blocked"])
        if cleanup.get("all_objects_absent") is not True:
            blockers.append("object_store_provider_zero_not_proven")
        blockers = sorted(set(blockers))
        return _seal_live_terminal_result(
            job,
            {
                "schema_version": result_schema_version,
                "status": blocked_status,
                "run_id": receipt["run_id"],
                "source_commit": receipt["source_commit"],
                "bundle_sha256": receipt["bundle_sha256"],
                "authority_digest": authority["authority_digest"],
                "provider_mutations_performed": 0,
                "retry_cap": 0,
                **diagnostic_claim_boundary,
                "object_store_cleanup": cleanup,
                "continuing_spend_from_this_run": False,
                "blockers": blockers,
            },
            receipt=receipt,
            scene_construction_queue_root=scene_construction_queue_root,
        )
    watchdog_handoff, watchdog = arm_independent_vast_watchdog(
        job_dir=job,
        max_live_minutes=live_minutes,
        generated_at=utc_now_iso(),
        allowed_active_instance_ids=(),
        pod_name_prefix=WATCHDOG_POD_NAME_PREFIX,
    )
    if watchdog is None:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
        blockers = ["independent_watchdog_not_armed"]
        if cleanup.get("all_objects_absent") is not True:
            blockers.append("object_store_provider_zero_not_proven")
        blockers = sorted(set(blockers))
        return _seal_live_terminal_result(
            job,
            {
                "schema_version": result_schema_version,
                "status": blocked_status,
                "run_id": receipt["run_id"],
                "source_commit": receipt["source_commit"],
                "bundle_sha256": receipt["bundle_sha256"],
                "authority_digest": authority["authority_digest"],
                "provider_mutations_performed": 0,
                "retry_cap": 0,
                **diagnostic_claim_boundary,
                "all_staged_objects_absent": cleanup.get("all_objects_absent"),
                "object_store_cleanup": cleanup,
                "independent_watchdog": watchdog_handoff,
                "continuing_spend_from_this_run": False,
                "blockers": blockers,
            },
            receipt=receipt,
            scene_construction_queue_root=scene_construction_queue_root,
        )
    runtime_environment = dict(runtime_environment)
    runtime_environment[PARENT_DEADLINE_EPOCH_ENV] = str(
        watchdog.deadline_epoch
    )
    runtime_environment[OUTPUT_CLOSURE_RESERVE_SECONDS_ENV] = str(
        OUTPUT_AND_CLOSURE_RESERVE_SECONDS
    )
    consumption = _consume_authority_once(
        authority, source_commit=str(receipt["source_commit"])
    )
    if consumption.get("status") != "consumed":
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
        watchdog_close = close_independent_vast_watchdog(
            job_dir=job,
            handle=watchdog,
            instance_ids=[],
            provider_teardown_completed=False,
            provider_allocation_impossible=True,
        )
        blockers = list(consumption.get("blockers") or [])
        if cleanup.get("all_objects_absent") is not True:
            blockers.append("object_store_provider_zero_not_proven")
        if watchdog_close.get("status") not in {
            "provider_terminal",
            "cancelled_no_allocation",
        }:
            blockers.append("independent_watchdog_not_closed")
        blockers = sorted(set(blockers))
        return _seal_live_terminal_result(
            job,
            {
                "schema_version": result_schema_version,
                "status": blocked_status,
                "run_id": receipt["run_id"],
                "source_commit": receipt["source_commit"],
                "bundle_sha256": receipt["bundle_sha256"],
                "authority_digest": authority["authority_digest"],
                "provider_mutations_performed": 0,
                "retry_cap": 0,
                **diagnostic_claim_boundary,
                "authorization_consumption": consumption,
                "all_staged_objects_absent": cleanup.get("all_objects_absent"),
                "object_store_cleanup": cleanup,
                "independent_watchdog": watchdog_close,
                "continuing_spend_from_this_run": False,
                "blockers": blockers,
            },
            receipt=receipt,
            scene_construction_queue_root=scene_construction_queue_root,
        )

    output_zip = provider_run / "vast_provider_runtime_output.zip"
    adapter: dict[str, Any] = {}
    staged_secret_root: Path | None = None
    runtime_secret_cleanup_blockers: list[str] = []
    try:
        runtime_secret_paths, staged_secret_root = (
            _stage_owner_only_runtime_secrets(
                job_dir=job, secret_paths=runtime_secret_paths
            )
        )
    except (OSError, TaskEvaluationSceneConfigurationVastError) as exc:
        cleanup_blockers = _discard_staged_runtime_secrets(staged_secret_root)
        raise TaskEvaluationSceneConfigurationVastError(
            cleanup_blockers[0]
            if cleanup_blockers
            else "scene_configuration_openai_runtime_secret_configuration_invalid"
        ) from exc
    try:
        with _authority_environment():
            adapter = run_vast_provider_adapter(
                job_dir=provider_run,
                mode="live-startup-probe",
                allow_vast_api_call=True,
                allow_instance_launch=True,
                max_hourly_rate=rate,
                target_spend_usd=provider_all_in_cap,
                hard_cap_usd=provider_all_in_cap,
                max_live_minutes=live_minutes,
                session_max_live_minutes=live_minutes,
                public_image=str(authority["container_image"]),
                isaac_image=str(authority["container_image"]),
                ngc_image_login_mode="always",
                provider_bundle=bundle_path,
                provider_bundle_url=(
                    staging_dir / "provider_bundle_url.txt"
                ).read_text(encoding="utf-8").strip(),
                provider_output_put_url=(
                    staging_dir / "provider_output_put_url.txt"
                ).read_text(encoding="utf-8").strip(),
                provider_output_get_url=(
                    staging_dir / "provider_output_get_url.txt"
                ).read_text(encoding="utf-8").strip(),
                provider_runtime_output_zip=output_zip,
                expected_provider_download_bytes=expected_download_bytes,
                expected_provider_upload_bytes=expected_upload_bytes,
                expected_provider_bundle_sha256=str(receipt["bundle_sha256"]),
                provider_output_minimum_free_bytes=(
                    output_disk_requirements[
                        "required_free_bytes_before_download"
                    ]
                ),
                enable_isaac_smoke=True,
                enable_blueprint_bundle=True,
                provider_bundle_kind=PROVIDER_BUNDLE_KIND,
                vast_launch_mode="ssh_direct",
                allow_cold_isaac_image_pull=True,
                min_cold_isaac_pull_live_minutes=18,
                disk_gb=100,
                min_gpu_ram_mb=24_000,
                poll_interval_seconds=10,
                startup_timeout_seconds=ttl,
                heartbeat_no_progress_seconds=min(1_200, max(300, ttl // 2)),
                session_budget_ledger_path=job
                / "scene_configuration_vast_session_budget.json",
                verify_staging_urls=True,
                require_known_supported_isaac_driver=True,
                preferred_gpu_keywords=("RTX 4090", "L40S", "RTX A6000"),
                prefer_isaac_rt=True,
                allowed_active_instance_ids=(),
                vast_launch_lock_file=job.parent
                / "scene_configuration_paid_launch.lock",
                instance_label_prefix=watchdog.pod_name_prefix,
                started_instance_id_path=watchdog.started_instance_id_path,
                forward_hf_token=False,
                paid_resource_admission_grant=paid_resource_admission_grant,
                runtime_secret_file_paths=runtime_secret_paths,
                provider_runtime_environment=runtime_environment,
                allowed_geolocation_country_codes=(
                    OPENAI_API_SUPPORTED_COUNTRY_CODES
                    if runtime_secret_paths
                    else ()
                ),
                retain_scene_configuration_warm_session=retain_warm_session,
                retention_watchdog_handoff=(
                    watchdog_handoff if retain_warm_session else None
                ),
            )
    except (OSError, RuntimeError, ValueError) as exc:
        adapter, provider_allocation_may_have_occurred = (
            _recover_escaped_adapter_failure(
                provider_run=provider_run,
                started_instance_id_path=watchdog.started_instance_id_path,
                failure_detail=redacted_failure_detail(exc),
            )
        )
        if not provider_allocation_may_have_occurred:
            seal_unallocated_provider_teardown(
                provider_run, reason="vast_adapter_failed"
            )
    finally:
        runtime_secret_cleanup_blockers = _discard_staged_runtime_secrets(
            staged_secret_root
        )
        cleanup = cleanup_scene_staging(
            adapter=adapter, staging_dir=staging_dir,
            cleanup=cleanup_staged_wam_provider_objects,
        )

    if retain_warm_session and adapter.get("retained_owned") is True:
        from .task_evaluation_scene_configuration_warm_bootstrap import (  # noqa: PLC0415
            handle_retained_scene_configuration_bootstrap,
        )

        return handle_retained_scene_configuration_bootstrap(
            adapter=adapter, cleanup=cleanup,
            runtime_secret_cleanup_blockers=list(runtime_secret_cleanup_blockers),
            output_zip=output_zip, job=job, provider_run=provider_run,
            expected_upload_bytes=expected_upload_bytes,
            warm_session_authority_path=warm_session_authority_path,
            warm_session_output_root=warm_session_output_root,
            paid_resource_admission_grant=paid_resource_admission_grant,
            watchdog=watchdog, watchdog_handoff=watchdog_handoff,
            receipt=receipt, authority=authority,
            warm_session_authority=warm_session_authority,
            diagnostic_claim_boundary=diagnostic_claim_boundary,
            extract_provider_output=_extract_provider_output,
            seal_terminal_result=_seal_terminal_result,
            close_independent_vast_watchdog=close_independent_vast_watchdog,
        )
    teardown_path = provider_run / "vast_teardown_manifest.json"
    teardown = _read(teardown_path) if teardown_path.is_file() else {}
    instance_ids = [
        int(value)
        for value in (
            teardown.get("vast_instance_ids")
            or adapter.get("vast_instance_ids")
            or []
        )
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    ]
    watchdog_close = _close_watchdog_after_adapter(
        job_dir=job,
        handle=watchdog,
        adapter=adapter,
        teardown=teardown,
        instance_ids=instance_ids,
    )
    if (
        cleanup.get("all_objects_absent") is not True
        and watchdog_close.get("status") == "provider_terminal"
    ):
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    execution, blockers, extraction_disk_capacity = (
        _extract_provider_output_with_capacity_guard(
            output_zip,
            job / "immutable_execution",
            maximum_archive_bytes=expected_upload_bytes,
            diagnostic_only=diagnostic_only,
            disk_usage_provider=disk_usage_provider,
        )
    )
    # The adapter's own refusal is the only record of *why* nothing was
    # allocated. Without it the result carries only the downstream
    # consequences -- provider result missing, output zip invalid -- which
    # describe an empty provider run identically no matter what caused it.
    blockers.extend(
        str(item) for item in adapter.get("blockers") or [] if str(item)
    )
    blockers.extend(runtime_secret_cleanup_blockers)
    expected_execution_status = (
        "completed_diagnostic_only_not_qualification_eligible"
        if diagnostic_only
        else "completed"
    )
    if execution.get("status") != expected_execution_status:
        blockers.append("scene_configuration_provider_not_completed")
    blockers.extend(
        _provider_execution_binding_blockers(
            execution, receipt, diagnostic_only=diagnostic_only
        )
    )
    if teardown.get("continuing_spend_from_this_run") is not False:
        blockers.append("provider_zero_not_proven")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("object_store_provider_zero_not_proven")
    if watchdog_close.get("status") not in {
        "provider_terminal",
        "cancelled_no_allocation",
    }:
        blockers.append("independent_watchdog_not_closed")

    advanced_checkpoint_reference: dict[str, Any] = {}
    advanced_checkpoint_reference_path = job / "advanced_checkpoint_reference.v1.json"
    validated_advanced = execution.pop("_validated_advanced_checkpoint", None)
    if diagnostic_only and isinstance(validated_advanced, Mapping):
        advanced_checkpoint_reference = {
            "schema_version": (
                "task_evaluation_scene_configuration_advanced_checkpoint_reference.v1"
            ),
            "status": "validated_diagnostic_checkpoint_ready_for_next_retry",
            **dict(validated_advanced),
            "source_provider_result_digest": execution.get("result_digest"),
            "diagnostic_only": True,
            "qualification_eligible": False,
            "reference_digest": "",
        }
        advanced_checkpoint_reference["reference_digest"] = canonical_digest(
            advanced_checkpoint_reference, digest_field="reference_digest"
        )
        write_json(advanced_checkpoint_reference_path, advanced_checkpoint_reference)
    elif diagnostic_only and not blockers:
        blockers.append("scene_configuration_diagnostic_advanced_checkpoint_missing")

    publication: dict[str, Any] = {}
    publication_root = job / "configured_scene_publication"
    if not blockers and not diagnostic_only:
        try:
            publication = _publish_completed_configuration(
                receipt=receipt,
                execution=execution,
                extraction_root=job / "immutable_execution",
                output_root=publication_root,
            )
        except Exception as exc:  # noqa: BLE001 - preserve terminal evidence
            blockers.append(
                "scene_configuration_configured_revision_publication_failed:"
                + redacted_failure_detail(exc)
            )
    if (
        not diagnostic_only
        and publication.get("status") != "configured_scene_published"
    ):
        blockers.append("scene_configuration_configured_revision_not_published")

    artifact_manifest_path = job / "artifact_manifest.json"
    artifact_roots = {
        "provider_runtime_evidence": job / "immutable_execution",
        "allocator_adapter_result": provider_run
        / "vast_provider_adapter_result.json",
        "teardown_manifest": teardown_path,
        "provider_run_diagnostics": provider_run,
    }
    required_roles = [
        "provider_runtime_evidence",
        "allocator_adapter_result",
        "teardown_manifest",
    ]
    if publication.get("status") == "configured_scene_published":
        artifact_roots["configured_scene_publication"] = publication_root
        required_roles.append("configured_scene_publication")
    try:
        artifact_manifest = build_task_evaluation_artifact_manifest(
            attempt_root=job,
            artifact_roots=artifact_roots,
            required_roles=tuple(required_roles),
            binding={
                "allocator_lane": PROVIDER_BUNDLE_KIND,
                "source_commit": receipt["source_commit"],
                "bundle_sha256": receipt["bundle_sha256"],
                "provider": "vast",
                "result_schema_version": (
                    DIAGNOSTIC_RESULT_SCHEMA_VERSION
                    if diagnostic_only
                    else RESULT_SCHEMA_VERSION
                ),
                "retry_cap": 0,
            },
            output_path=artifact_manifest_path,
        )
    except TaskEvaluationArtifactManifestError as exc:
        artifact_manifest = {"status": "blocked", "blockers": [str(exc)]}
        blockers.append("scene_configuration_artifact_manifest_invalid")
    if artifact_manifest.get("status") != "completed":
        blockers.extend(str(item) for item in artifact_manifest.get("blockers") or [])
    result: dict[str, Any] = {
        "schema_version": (
            DIAGNOSTIC_RESULT_SCHEMA_VERSION
            if diagnostic_only
            else RESULT_SCHEMA_VERSION
        ),
        "generated_at": utc_now_iso(),
        "status": (
            "completed_diagnostic_only"
            if diagnostic_only and not blockers
            else "completed"
            if not blockers
            else "blocked_diagnostic_only"
            if diagnostic_only
            else "blocked"
        ),
        "run_id": receipt["run_id"],
        "source_commit": receipt["source_commit"],
        "bundle_sha256": receipt["bundle_sha256"],
        "authority_digest": authority["authority_digest"],
        "authorization_consumption": consumption,
        "provider_adapter_result_path": str(
            provider_run / "vast_provider_adapter_result.json"
        ),
        "execution_result_path": str(
            job / "immutable_execution" / RESULT_FILENAME
        ),
        "advanced_checkpoint_reference_path": (
            str(advanced_checkpoint_reference_path)
            if advanced_checkpoint_reference_path.is_file()
            else None
        ),
        "advanced_checkpoint_reference_digest": advanced_checkpoint_reference.get(
            "reference_digest"
        ),
        "artifact_manifest_path": (
            str(artifact_manifest_path) if artifact_manifest_path.is_file() else None
        ),
        "teardown_manifest_path": (
            str(teardown_path) if teardown_path.is_file() else None
        ),
        "provider_runtime_output_zip_path": (
            str(output_zip) if output_zip.is_file() else None
        ),
        "provider_runtime_output_zip_sha256": (
            _sha256(output_zip) if output_zip.is_file() else None
        ),
        "runtime_secret_cleanup_completed": not runtime_secret_cleanup_blockers,
        "expected_provider_download_bytes": expected_download_bytes,
        "expected_provider_upload_bytes": expected_upload_bytes,
        "provider_output_disk_requirements": output_disk_requirements,
        "provider_output_disk_capacity": {
            "before_allocation_and_staging": preallocation_disk_capacity,
            "before_extraction": extraction_disk_capacity,
        },
        "stage_chain_result_digest": (
            execution.get(
                "diagnostic_stage_chain" if diagnostic_only else "stage_chain"
            )
            or {}
        ).get("result_digest"),
        "configuration_completed": (
            False if diagnostic_only else execution.get("status") == "completed"
        ),
        "diagnostic_execution_completed": (
            diagnostic_only and execution.get("status") == expected_execution_status
        ),
        "configured_scene_published": (
            not diagnostic_only
            and publication.get("status") == "configured_scene_published"
        ),
        "configured_scene_revision_path": (
            (publication.get("configured_scene_revision") or {}).get("path")
        ),
        "configured_scene_revision_reference": publication.get(
            "configured_scene_revision_reference"
        ),
        "configured_scene_revision_digest": publication.get(
            "configured_scene_revision_digest"
        ),
        "configured_scene_bundle_reference": publication.get(
            "configured_scene_bundle_reference"
        ),
        "task_thumbnail_reference": publication.get(
            "task_thumbnail_reference"
        ),
        "task_thumbnail_selection": publication.get(
            "task_thumbnail_selection"
        ),
        "task_thumbnail_selection_receipt_reference": publication.get(
            "task_thumbnail_selection_receipt_reference"
        ),
        "configured_scene_offering": publication.get(
            "configured_scene_offering"
        ),
        "publication_result_path": (
            str(
                publication_root
                / f"{PUBLICATION_RESULT_SCHEMA_VERSION}.json"
            )
            if publication
            else None
        ),
        "publication_result_digest": publication.get("result_digest"),
        "full_byte_service_account_readback_passed": publication.get(
            "full_byte_service_account_readback_passed"
        )
        is True,
        "evaluation_episode_executed": False,
        "candidate_policy_queried": False,
        "provider_mutations_performed": len(instance_ids),
        "retry_cap": 0,
        "independent_watchdog": watchdog_close,
        "object_store_cleanup": cleanup,
        "continuing_spend_from_this_run": teardown.get(
            "continuing_spend_from_this_run"
        ),
        "blockers": sorted(set(blockers)),
        "result_digest": "",
    }
    if diagnostic_only:
        result.update(
            {
                "diagnostic_only": True,
                "qualification_eligible": False,
                "executed_inside_one_parent_provider_run": False,
                "configured_revision_publication_permitted": False,
                "offering_publication_permitted": False,
                "terminal_e2e_completion_permitted": False,
            }
        )
    return _seal_live_terminal_result(
        job,
        result,
        receipt=receipt,
        scene_construction_queue_root=scene_construction_queue_root,
    )


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationVastError",
    "run_scene_configuration_vast",
]
