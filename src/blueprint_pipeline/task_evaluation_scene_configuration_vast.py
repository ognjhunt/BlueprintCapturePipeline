"""Canonical retry-zero Vast execution for one scene-configuration bundle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import zipfile
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .common import ensure_dir, redacted_failure_detail, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .openai_api_geography import OPENAI_API_SUPPORTED_COUNTRY_CODES
from .paid_resource_admission import (
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .spend_authority_consumption_root import (
    SpendAuthorityRootError,
    prepare_consumption_root,
)
from .task_evaluation_artifact_manifest import (
    PROVIDER_RUN_DIRNAME,
    TaskEvaluationArtifactManifestError,
    build_task_evaluation_artifact_manifest,
    seal_unallocated_provider_teardown,
)
from .task_evaluation_scene_configuration_bundle import (
    PROVIDER_BUNDLE_KIND,
    RESULT_FILENAME,
    load_scene_configuration_provider_bundle_receipt,
)
from .task_evaluation_scene_configuration_paid_authority import (
    validate_scene_configuration_paid_authority,
)
from .task_evaluation_supervisor.openai_cost_authority import (
    OpenAICostAuthorityError,
    OpenAIOrganizationCostsClient,
    validate_openai_cost_scope_attestation,
)
from .vast_independent_watchdog_control import (
    arm_independent_vast_watchdog,
    close_independent_vast_watchdog,
)
from .vast_provider_adapter import run_vast_provider_adapter
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


RESULT_SCHEMA_VERSION = "task_evaluation_scene_configuration_vast_result.v1"
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
# stage: the official-cost gate refuses a shared scope, both by attestation
# ``paid_resource_class`` and by the same-day zero-cost baseline it demands
# for each ``(project_id, api_key_id)`` before any stage may spend.
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


def _discard_staged_runtime_secrets(root: Path | None) -> None:
    """Remove the private copies; never leave secret bytes behind a run."""

    if root is None:
        return
    for child in sorted(root.glob("*")):
        try:
            child.unlink()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


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
    for stage, api_key_id_env, attestation_file_env in _OPENAI_STAGE_SCOPE_BINDINGS:
        try:
            attestation = json.loads(
                Path(secret_paths[attestation_file_env]).read_text(encoding="utf-8")
            )
            if not isinstance(attestation, Mapping):
                raise OpenAICostAuthorityError(
                    "openai_cost_scope_attestation_invalid"
                )
            validate_openai_cost_scope_attestation(
                attestation,
                provider_id="openai",
                paid_resource_class=f"task_evaluation_scene_configuration_{stage}",
                project_id=values["OPENAI_PROJECT_ID"],
                api_key_id=values[api_key_id_env],
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            OpenAICostAuthorityError,
        ) as exc:
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
        if float(observed_cost) != 0.0:
            raise TaskEvaluationSceneConfigurationVastError(
                f"scene_configuration_openai_stage_cost_baseline_not_zero:{stage}"
            )
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


def _extract_provider_output(
    archive_path: Path, destination: Path
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if destination.exists():
        return {}, ["scene_configuration_provider_output_destination_exists"]
    destination.mkdir(parents=True, mode=0o750)
    root = destination.resolve()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
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
    except (OSError, ValueError, zipfile.BadZipFile):
        blockers.append("scene_configuration_provider_output_zip_invalid")
    result_path = root / RESULT_FILENAME
    result = _read(result_path) if result_path.is_file() else {}
    if not result:
        blockers.append("scene_configuration_provider_result_missing")
        return result, sorted(set(blockers))
    if (
        result.get("schema_version")
        != "task_evaluation_scene_configuration_provider_result.v1"
        or result.get("result_digest")
        != canonical_digest(result, digest_field="result_digest")
        or result.get("evaluation_episode_executed") is not False
        or result.get("candidate_policy_queried") is not False
        or result.get("provider_zero_required_after_return") is not True
    ):
        blockers.append("scene_configuration_provider_result_contract_invalid")
    if result.get("status") == "completed":
        chain = result.get("stage_chain")
        if (
            not isinstance(chain, Mapping)
            or chain.get("status") != "completed"
            or chain.get("stage_count") != 6
            or len(chain.get("stage_results") or []) != 6
            or chain.get("executed_inside_one_parent_provider_run") is not True
            or chain.get("nested_provider_mutations_performed") != 0
            or chain.get("nested_paid_execution_requested") is not False
            or chain.get("evaluation_episode_executed") is not False
            or chain.get("retry_cap") != 0
            or chain.get("result_digest")
            != canonical_digest(chain, digest_field="result_digest")
        ):
            blockers.append("scene_configuration_stage_chain_invalid")
    elif result.get("status") != "blocked":
        blockers.append("scene_configuration_provider_result_status_invalid")
    return result, sorted(set(blockers))


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
#: Sized for the mode this lane actually runs. The configuration selects
#: ``dual_target_artifixer3d_only``, which the runtime admits only with
#: ``direct_editor_backend == "none"``, so the ``vibe_image_edit`` branch and
#: its multi-gigabyte ``cu124`` torch wheels never execute. A reserve big
#: enough for those wheels would price nearly the whole compute cap into
#: bandwidth and start excluding otherwise admissible offers, which is a real
#: cost for a download this mode does not perform. If a torch-backed direct
#: editor is ever selected here, this number has to be revisited with it.
PROVISIONING_DOWNLOAD_OVERHEAD_BYTES = 2_000_000_000


def _provider_transfer_byte_budget(
    receipt: Mapping[str, Any],
) -> tuple[int, int]:
    """Declare the transfer ceilings the hard-cap projection must price.

    Vast prices inbound and outbound bytes per GB *outside* the hourly rate.
    Leaving these at zero did not merely lose a number: it switched off
    ``_offer_fits_total_cost_bound`` entirely, so an offer whose bandwidth
    price alone could exceed the attempt's compute cap still passed
    admission, and the selection receipt recorded no projected total. The
    download ceiling is the exact byte count the receipt already seals and
    the staged object store already serves, so it is measured, not guessed.
    """

    bundle = receipt.get("bundle_size_bytes")
    if not isinstance(bundle, int) or isinstance(bundle, bool) or bundle <= 0:
        raise TaskEvaluationSceneConfigurationVastError(
            "scene_configuration_provider_transfer_budget_inputs_invalid"
        )
    download = bundle + PROVISIONING_DOWNLOAD_OVERHEAD_BYTES
    # The upload side has no contract to price. On the provider-render path
    # the frames are produced on the rented GPU, so the bundle manifest's
    # ``derived_rendered_view_count`` is 0, and neither the manifest nor the
    # signed output PUT declares a byte ceiling for what comes back. A
    # fabricated estimate would be indistinguishable from a measured one in
    # the selection receipt, so this stays 0 until an output contract
    # actually declares a bound.
    return download, 0


def run_scene_configuration_vast(
    *,
    job_dir: str | Path,
    bundle_receipt_path: str | Path,
    paid_attempt_authority_path: str | Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
) -> dict[str, Any]:
    """Run exactly one configuration allocation and close every owned resource."""

    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    receipt = load_scene_configuration_provider_bundle_receipt(bundle_receipt_path)
    authority_path = Path(paid_attempt_authority_path).expanduser().resolve()
    authority = validate_scene_configuration_paid_authority(
        _read(authority_path), bundle_receipt=receipt
    )
    runtime_secret_paths, runtime_environment = _provider_runtime_inputs(authority)
    if not execute:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "dry_run_ready",
            "run_id": receipt["run_id"],
            "source_commit": receipt["source_commit"],
            "bundle_sha256": receipt["bundle_sha256"],
            "authority_digest": authority["authority_digest"],
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "blockers": [],
            "result_digest": "",
        }
        result["result_digest"] = canonical_digest(
            result, digest_field="result_digest"
        )
        write_json(job / f"{RESULT_SCHEMA_VERSION}.json", result)
        return result
    if paid_resource_admission_grant is None:
        raise TaskEvaluationSceneConfigurationVastError(
            "scene_configuration_paid_admission_missing"
        )
    require_paid_resource_admission_grant(
        paid_resource_admission_grant,
        resource_class="vast_provider_adapter",
        require_allocation_binding=True,
    )
    hard_cap = float(authority["provider_compute_spend_cap_usd"])
    rate = float(authority["maximum_hourly_rate_usd"])
    ttl = int(authority["maximum_single_resource_ttl_seconds"])
    bundle_path = Path(str(receipt["bundle_path"])).resolve()
    expected_download_bytes, expected_upload_bytes = _provider_transfer_byte_budget(
        receipt
    )
    staging_dir = job / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir,
        bundle_path=bundle_path,
        key_prefix="blueprint/arm-decision-proof-v1/scene-configuration",
        expiration_seconds=ttl + 1_800,
    )
    if staging.get("status") != "completed":
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "run_id": receipt["run_id"],
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "blockers": list(
                staging.get("blockers") or ["object_store_staging_blocked"]
            ),
        }
    watchdog_handoff, watchdog = arm_independent_vast_watchdog(
        job_dir=job,
        max_live_minutes=max(1, ttl // 60),
        generated_at=utc_now_iso(),
        allowed_active_instance_ids=(),
        pod_name_prefix=WATCHDOG_POD_NAME_PREFIX,
    )
    if watchdog is None:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "run_id": receipt["run_id"],
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "all_staged_objects_absent": cleanup.get("all_objects_absent"),
            "independent_watchdog": watchdog_handoff,
            "blockers": ["independent_watchdog_not_armed"],
        }
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
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "run_id": receipt["run_id"],
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "authorization_consumption": consumption,
            "all_staged_objects_absent": cleanup.get("all_objects_absent"),
            "independent_watchdog": watchdog_close,
            "blockers": list(consumption.get("blockers") or []),
        }

    provider_run = job / PROVIDER_RUN_DIRNAME
    output_zip = provider_run / "vast_provider_runtime_output.zip"
    adapter: dict[str, Any] = {}
    staged_secret_root: Path | None = None
    try:
        runtime_secret_paths, staged_secret_root = (
            _stage_owner_only_runtime_secrets(
                job_dir=job, secret_paths=runtime_secret_paths
            )
        )
    except (OSError, TaskEvaluationSceneConfigurationVastError) as exc:
        _discard_staged_runtime_secrets(staged_secret_root)
        raise TaskEvaluationSceneConfigurationVastError(
            "scene_configuration_openai_runtime_secret_configuration_invalid"
        ) from exc
    try:
        with _authority_environment():
            adapter = run_vast_provider_adapter(
                job_dir=provider_run,
                mode="live-startup-probe",
                allow_vast_api_call=True,
                allow_instance_launch=True,
                max_hourly_rate=rate,
                target_spend_usd=hard_cap,
                hard_cap_usd=hard_cap,
                max_live_minutes=max(1, ttl // 60),
                session_max_live_minutes=max(1, ttl // 60),
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
            )
    except (OSError, RuntimeError, ValueError) as exc:
        adapter = {
            "status": "blocked",
            "blockers": [
                f"vast_adapter_failed:{redacted_failure_detail(exc)}"
            ],
        }
        seal_unallocated_provider_teardown(provider_run, reason="vast_adapter_failed")
    finally:
        _discard_staged_runtime_secrets(staged_secret_root)
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)

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
    watchdog_close = close_independent_vast_watchdog(
        job_dir=job,
        handle=watchdog,
        instance_ids=instance_ids,
        provider_teardown_completed=(
            teardown.get("continuing_spend_from_this_run") is False
        ),
        provider_allocation_impossible=(
            not instance_ids and adapter.get("provider_create_attempted") is not True
        ),
    )
    execution, blockers = _extract_provider_output(
        output_zip, job / "immutable_execution"
    )
    # The adapter's own refusal is the only record of *why* nothing was
    # allocated. Without it the result carries only the downstream
    # consequences -- provider result missing, output zip invalid -- which
    # describe an empty provider run identically no matter what caused it.
    blockers.extend(
        str(item) for item in adapter.get("blockers") or [] if str(item)
    )
    if execution.get("status") != "completed":
        blockers.append("scene_configuration_provider_not_completed")
    if execution.get("source_commit") != receipt.get("source_commit"):
        blockers.append("scene_configuration_provider_source_commit_mismatch")
    if execution.get("construction_envelope_digest") != receipt.get(
        "portable_construction_envelope_digest"
    ):
        blockers.append("scene_configuration_provider_envelope_mismatch")
    if teardown.get("continuing_spend_from_this_run") is not False:
        blockers.append("provider_zero_not_proven")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("object_store_provider_zero_not_proven")
    if watchdog_close.get("status") not in {
        "provider_terminal",
        "cancelled_no_allocation",
    }:
        blockers.append("independent_watchdog_not_closed")

    artifact_manifest_path = job / "artifact_manifest.json"
    try:
        artifact_manifest = build_task_evaluation_artifact_manifest(
            attempt_root=job,
            artifact_roots={
                "provider_runtime_evidence": job / "immutable_execution",
                "allocator_adapter_result": provider_run
                / "vast_provider_adapter_result.json",
                "teardown_manifest": teardown_path,
                "provider_run_diagnostics": provider_run,
            },
            required_roles=(
                "provider_runtime_evidence",
                "allocator_adapter_result",
                "teardown_manifest",
            ),
            binding={
                "allocator_lane": PROVIDER_BUNDLE_KIND,
                "source_commit": receipt["source_commit"],
                "bundle_sha256": receipt["bundle_sha256"],
                "provider": "vast",
                "result_schema_version": RESULT_SCHEMA_VERSION,
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
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed" if not blockers else "blocked",
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
        "expected_provider_download_bytes": expected_download_bytes,
        "expected_provider_upload_bytes": expected_upload_bytes,
        "stage_chain_result_digest": (execution.get("stage_chain") or {}).get(
            "result_digest"
        ),
        "configuration_completed": execution.get("status") == "completed",
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
    result["result_digest"] = canonical_digest(
        result, digest_field="result_digest"
    )
    write_json(job / f"{RESULT_SCHEMA_VERSION}.json", result)
    return result


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationVastError",
    "run_scene_configuration_vast",
]
