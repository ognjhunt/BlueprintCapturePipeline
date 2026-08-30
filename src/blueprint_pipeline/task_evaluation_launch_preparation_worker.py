"""No-spend worker for immutable Task Evaluation preparation references.

The worker claims the authenticated preparation queue, resolves only configured
object-store prefixes, performs full-byte digest readback as the service
account, and seals a result.  It does not construct a profile, mutate the
catalog, issue paid authority, call the allocator, or allocate a provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import re
import stat
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_launch_preparation_contract import (
    validate_launch_preparation_request,
)
from .task_evaluation_policy_run_contract import (
    TaskEvaluationPolicyRunContractError,
    build_policy_run_plan,
    validate_policy_run_setup,
)
from .task_evaluation_configured_scene_revision import (
    TaskEvaluationConfiguredSceneRevisionError,
    validate_configured_scene_revision,
)
from .task_evaluation_episode_compilation_queue import (
    TaskEvaluationEpisodeCompilationQueueError,
    stage_episode_compilation,
)
from .task_evaluation_native_arena_preparation_adapter import (
    materialize_native_arena_adapter,
)
from .task_evaluation_scene_configuration_disclosure import (
    RENDER_INPUT_STATUSES,
    render_inputs_disclosure_is_coherent,
)
from .task_evaluation_scene_construction_recipe import (
    TaskEvaluationSceneConstructionRecipeError,
    validate_scene_construction_recipe,
)
from .task_evaluation_scene_construction_queue import (
    TaskEvaluationSceneConstructionQueueError,
    stage_scene_construction,
)
from .task_evaluation_scene_configuration_render_inputs import (
    materialize_scene_configuration_render_inputs,
)
from .task_evaluation_launch_preparation_queue import (
    ENVELOPE_SCHEMA_VERSION,
    TaskEvaluationLaunchPreparationQueueError,
    ensure_launch_preparation_queue_root,
    write_launch_preparation_record_exclusive,
)


RESULT_SCHEMA_VERSION = "task_evaluation_launch_preparation_result.v1"
QUEUE_ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_QUEUE_ROOT"
INPUT_ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_INPUT_ROOT"
ALLOWED_URI_PREFIXES_ENV = (
    "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_ALLOWED_URI_PREFIXES_JSON"
)
SERVICE_ACCOUNT_ENV = (
    "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_SERVICE_ACCOUNT"
)
CONSTRUCTION_QUEUE_ROOT_ENV = (
    "BLUEPRINT_TASK_EVALUATION_SCENE_CONSTRUCTION_QUEUE_ROOT"
)
EPISODE_COMPILATION_QUEUE_ROOT_ENV = (
    "BLUEPRINT_TASK_EVALUATION_EPISODE_COMPILATION_QUEUE_ROOT"
)
ReferenceFetcher = Callable[[str, Path, int], None]
AdapterMaterializer = Callable[..., dict[str, Any]]
SceneRenderInputMaterializer = Callable[..., dict[str, Any]]
ALLOWED_REFERENCE_SCHEMES = frozenset({"gs", "https", "s3"})


class TaskEvaluationLaunchPreparationWorkerError(RuntimeError):
    """A claimed no-spend preparation could not be completed safely."""


def running_worker_source_commit(module_path: str | Path | None = None) -> str:
    """Read the exact detached/worktree commit that owns the running worker."""

    start = Path(module_path or __file__).resolve()
    for candidate in (start, *start.parents):
        marker = candidate / ".git"
        if not marker.exists():
            continue
        head_path = marker / "HEAD"
        if marker.is_file():
            try:
                pointer = marker.read_text(encoding="utf-8").strip()
            except OSError:
                return ""
            if not pointer.startswith("gitdir:"):
                return ""
            git_root = Path(pointer.split(":", 1)[1].strip())
            if not git_root.is_absolute():
                git_root = (candidate / git_root).resolve()
            head_path = git_root / "HEAD"
        try:
            head = head_path.read_text(encoding="utf-8").strip().lower()
        except OSError:
            return ""
        return head if re.fullmatch(r"[0-9a-f]{40}", head) else ""
    return ""


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def collect_preparation_references(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect every typed immutable reference with its JSON contract path."""

    references: list[dict[str, Any]] = []

    def visit(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, Mapping):
            if set(node) == {"uri", "digest", "size_bytes"}:
                references.append(
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
    identities: dict[str, tuple[str, int]] = {}
    for reference in references:
        identity = (reference["digest"], reference["size_bytes"])
        prior = identities.setdefault(reference["uri"], identity)
        if prior != identity:
            raise TaskEvaluationLaunchPreparationWorkerError(
                "launch_preparation_reference_uri_identity_conflict"
            )
    return references


def validate_allowed_uri_prefixes(prefixes: Sequence[str]) -> tuple[str, ...]:
    """Validate operator-owned object prefixes before any customer URI lookup."""

    if not prefixes:
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_allowed_uri_prefixes_missing"
        )
    validated: list[str] = []
    for prefix in prefixes:
        parsed = urlparse(prefix)
        if not (
            isinstance(prefix, str)
            and prefix.endswith("/")
            and parsed.scheme in ALLOWED_REFERENCE_SCHEMES
            and parsed.netloc
            and "@" not in parsed.netloc
            and not parsed.query
            and not parsed.fragment
        ):
            raise TaskEvaluationLaunchPreparationWorkerError(
                "launch_preparation_allowed_uri_prefix_invalid"
            )
        validated.append(prefix)
    return tuple(validated)


def _prefix_allowed(uri: str, allowed_uri_prefixes: Sequence[str]) -> bool:
    parsed = urlparse(uri)
    return (
        parsed.scheme in ALLOWED_REFERENCE_SCHEMES
        and bool(parsed.netloc)
        and "@" not in parsed.netloc
        and not parsed.query
        and not parsed.fragment
        and any(uri.startswith(prefix) for prefix in allowed_uri_prefixes)
    )


def _private_file_value(environment_name: str) -> str | None:
    """Read one canonical secret-file value without copying it into receipts."""

    raw_path = os.getenv(environment_name)
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    try:
        if path.is_symlink():
            raise TaskEvaluationLaunchPreparationWorkerError(
                "launch_preparation_object_store_secret_file_unsafe"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except TaskEvaluationLaunchPreparationWorkerError:
        raise
    except OSError as exc:
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_object_store_secret_file_unavailable"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        mode = stat.S_IMODE(metadata.st_mode)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or mode & ~0o640
            or not mode & 0o440
        ):
            raise TaskEvaluationLaunchPreparationWorkerError(
                "launch_preparation_object_store_secret_file_unsafe"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(4097)
    except TaskEvaluationLaunchPreparationWorkerError:
        raise
    except OSError as exc:
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_object_store_secret_file_unavailable"
        ) from exc
    finally:
        os.close(descriptor)
    if len(payload) > 4096:
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_object_store_secret_file_unsafe"
        )
    try:
        value = payload.decode("utf-8").strip()
    except UnicodeError as exc:
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_object_store_secret_file_unavailable"
        ) from exc
    if not value:
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_object_store_secret_file_unsafe"
        )
    return value


_LEGACY_OBJECT_STORE_FILE_ENV = {
    "access_key": "BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID_FILE",
    "secret_key": "BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY_FILE",
    "bucket": "BLUEPRINT_WAM_OBJECT_STORE_BUCKET_FILE",
    "endpoint": "BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL_FILE",
    "region": "BLUEPRINT_WAM_OBJECT_STORE_REGION_FILE",
}
_ARTIFACT_STORE_FILE_ENV = {
    "access_key": "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ACCESS_KEY_ID_FILE",
    "secret_key": "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_SECRET_ACCESS_KEY_FILE",
    "bucket": "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_BUCKET_FILE",
    "endpoint": "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ENDPOINT_URL_FILE",
    "region": "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_REGION_FILE",
}
_EXPECTED_ARTIFACT_BUCKET_ENV = (
    "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_EXPECTED_BUCKET"
)


def _s3_client(bucket: str) -> Any:
    """Use only the credential set explicitly bound to the requested bucket."""

    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError as exc:
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_s3_client_unavailable"
        ) from exc

    expected_artifact_bucket = str(
        os.getenv(_EXPECTED_ARTIFACT_BUCKET_ENV) or ""
    ).strip()
    if bucket == expected_artifact_bucket:
        artifact_bucket = _private_file_value(_ARTIFACT_STORE_FILE_ENV["bucket"])
        if bucket != artifact_bucket:
            raise TaskEvaluationLaunchPreparationWorkerError(
                "launch_preparation_artifact_store_bucket_identity_mismatch"
            )
        names = _ARTIFACT_STORE_FILE_ENV
        require_endpoint_and_region = True
    else:
        legacy_bucket = _private_file_value(_LEGACY_OBJECT_STORE_FILE_ENV["bucket"])
        if bucket != legacy_bucket:
            raise TaskEvaluationLaunchPreparationWorkerError(
                "launch_preparation_s3_bucket_not_configured"
            )
        names = _LEGACY_OBJECT_STORE_FILE_ENV
        require_endpoint_and_region = False

    access_key = _private_file_value(names["access_key"])
    secret_key = _private_file_value(names["secret_key"])
    if not access_key or not secret_key:
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_object_store_secret_pair_incomplete"
        )
    endpoint = _private_file_value(names["endpoint"])
    region = _private_file_value(names["region"])
    if require_endpoint_and_region and (not endpoint or not region):
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_artifact_store_endpoint_or_region_missing"
        )
    kwargs: dict[str, Any] = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
    }
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    if region:
        kwargs["region_name"] = region
    return boto3.client("s3", **kwargs)


def default_reference_fetcher(uri: str, destination: Path, maximum_bytes: int) -> None:
    """Fetch one admitted object with ambient service-account credentials."""

    parsed = urlparse(uri)
    if parsed.scheme == "https":
        request = urllib.request.Request(uri, method="GET")
        with urllib.request.urlopen(request, timeout=300) as response:  # nosec B310
            if response.geturl() != uri:
                raise TaskEvaluationLaunchPreparationWorkerError(
                    "launch_preparation_https_redirect_refused"
                )
            with destination.open("wb") as output:
                remaining = maximum_bytes + 1
                while remaining > 0:
                    chunk = response.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    output.write(chunk)
                    remaining -= len(chunk)
        return
    if parsed.scheme == "gs":
        try:
            from google.cloud import storage as gcs_storage  # type: ignore[import-untyped]
        except ImportError as exc:
            raise TaskEvaluationLaunchPreparationWorkerError(
                "launch_preparation_gcs_client_unavailable"
            ) from exc
        blob = gcs_storage.Client().bucket(parsed.netloc).blob(
            parsed.path.lstrip("/")
        )
        blob.reload()
        if blob.size != maximum_bytes:
            raise TaskEvaluationLaunchPreparationWorkerError(
                "launch_preparation_reference_declared_size_mismatch"
            )
        blob.download_to_filename(str(destination))
        return
    if parsed.scheme == "s3":
        response = _s3_client(parsed.netloc).get_object(
            Bucket=parsed.netloc, Key=parsed.path.lstrip("/")
        )
        if response.get("ContentLength") != maximum_bytes:
            raise TaskEvaluationLaunchPreparationWorkerError(
                "launch_preparation_reference_declared_size_mismatch"
            )
        with destination.open("wb") as output:
            remaining = maximum_bytes + 1
            while remaining > 0:
                chunk = response["Body"].read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                output.write(chunk)
                remaining -= len(chunk)
        return
    raise TaskEvaluationLaunchPreparationWorkerError(
        "launch_preparation_reference_scheme_unsupported"
    )


def materialize_preparation_references(
    *,
    request: Mapping[str, Any],
    input_root: str | Path,
    content_store_root: str | Path | None = None,
    allowed_uri_prefixes: Sequence[str],
    service_account: str,
    source_commit: str,
    fetcher: ReferenceFetcher = default_reference_fetcher,
) -> dict[str, Any]:
    """Materialize and read back every immutable input, content-addressed."""

    validated = validate_launch_preparation_request(request)
    if (
        not re.fullmatch(r"[0-9a-f]{40}", source_commit)
        or validated["expected_production_commit"] != source_commit
    ):
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_worker_source_commit_mismatch"
        )
    try:
        account = pwd.getpwnam(service_account)
    except KeyError as exc:
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_service_account_missing"
        ) from exc
    if os.geteuid() != account.pw_uid:
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_service_account_identity_mismatch"
        )
    validated_prefixes = validate_allowed_uri_prefixes(allowed_uri_prefixes)
    root = Path(input_root).expanduser()
    if root.is_symlink():
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_input_root_unsafe"
        )
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    root = root.resolve(strict=True)
    content_root = Path(content_store_root or root).expanduser()
    if content_root.is_symlink():
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_content_store_root_unsafe"
        )
    content_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    content_root = content_root.resolve(strict=True)
    rows, unique_object_count = _materialize_reference_records(
        references=collect_preparation_references(validated),
        input_root=root,
        content_store_root=content_root,
        allowed_uri_prefixes=validated_prefixes,
        fetcher=fetcher,
    )
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "inputs_materialized_awaiting_construction_adapter",
        "preparation_id": validated["preparation_id"],
        "run_id": validated["run_id"],
        "team_namespace": validated["team_namespace"],
        "source_commit": source_commit,
        "reference_count": len(rows),
        "unique_object_count": unique_object_count,
        "content_addressed_reuse_count": sum(
            row["content_addressed_reuse"] for row in rows
        ),
        "references": rows,
        "full_byte_service_account_readback_passed": all(
            row["full_byte_service_account_readback_passed"] for row in rows
        ),
        "service_account": service_account,
        "service_account_uid": account.pw_uid,
        "provider_mutation_performed": False,
        "catalog_mutation_performed": False,
        "paid_execution_requested": False,
        "observed_at_iso": datetime.now(timezone.utc).isoformat(),
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(
        result, digest_field="result_digest"
    )
    return result


def _materialize_reference_records(
    *,
    references: Sequence[Mapping[str, Any]],
    input_root: str | Path,
    content_store_root: str | Path | None = None,
    allowed_uri_prefixes: Sequence[str],
    fetcher: ReferenceFetcher,
) -> tuple[list[dict[str, Any]], int]:
    """Fetch and hash typed references without assuming their parent contract."""

    root = Path(input_root).resolve(strict=True)
    content_root = Path(content_store_root or root).resolve(strict=True)
    rows: list[dict[str, Any]] = []
    by_identity: dict[tuple[str, int], Path] = {}
    for reference in references:
        uri = reference["uri"]
        digest = reference["digest"]
        size = reference["size_bytes"]
        if not _prefix_allowed(uri, allowed_uri_prefixes):
            raise TaskEvaluationLaunchPreparationWorkerError(
                "launch_preparation_reference_prefix_not_allowed"
            )
        identity = (digest, size)
        destination = by_identity.get(identity)
        reused = destination is not None
        if destination is None:
            filename = digest.removeprefix("sha256:")
            cached = content_root / filename
            if cached.is_symlink():
                raise TaskEvaluationLaunchPreparationWorkerError(
                    "launch_preparation_materialized_target_unsafe"
                )
            reused = cached.exists()
            if not cached.exists():
                temporary = content_root / (
                    f".{cached.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
                )
                try:
                    fetcher(uri, temporary, size)
                    observed_digest, observed_size = _sha256_and_size(temporary)
                    if observed_digest != digest or observed_size != size:
                        raise TaskEvaluationLaunchPreparationWorkerError(
                            "launch_preparation_reference_readback_mismatch"
                        )
                    temporary.chmod(0o440)
                    descriptor = os.open(temporary, os.O_RDONLY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    try:
                        os.link(temporary, cached, follow_symlinks=False)
                    except FileExistsError:
                        pass
                    directory = os.open(
                        content_root,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                    )
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                finally:
                    temporary.unlink(missing_ok=True)
            if cached.is_symlink() or not stat.S_ISREG(
                cached.lstat().st_mode
            ):
                raise TaskEvaluationLaunchPreparationWorkerError(
                    "launch_preparation_materialized_target_unsafe"
                )
            observed_digest, observed_size = _sha256_and_size(cached)
            if observed_digest != digest or observed_size != size:
                raise TaskEvaluationLaunchPreparationWorkerError(
                    "launch_preparation_materialized_target_identity_mismatch"
                )
            destination = root / filename
            if destination != cached:
                if destination.is_symlink():
                    raise TaskEvaluationLaunchPreparationWorkerError(
                        "launch_preparation_materialized_target_unsafe"
                    )
                projection_created = False
                try:
                    os.link(cached, destination, follow_symlinks=False)
                    projection_created = True
                except FileExistsError:
                    pass
                if projection_created:
                    directory = os.open(
                        root,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                    )
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                if (
                    destination.is_symlink()
                    or not stat.S_ISREG(destination.lstat().st_mode)
                    or destination.stat().st_ino != cached.stat().st_ino
                    or destination.stat().st_dev != cached.stat().st_dev
                ):
                    raise TaskEvaluationLaunchPreparationWorkerError(
                        "launch_preparation_materialized_projection_invalid"
                    )
                observed_digest, observed_size = _sha256_and_size(destination)
                if observed_digest != digest or observed_size != size:
                    raise TaskEvaluationLaunchPreparationWorkerError(
                        "launch_preparation_materialized_target_identity_mismatch"
                    )
            by_identity[identity] = destination
        rows.append(
            {
                **reference,
                "materialized_path": str(destination),
                "content_addressed_reuse": reused,
                "full_byte_service_account_readback_passed": True,
            }
        )
    return rows, len(by_identity)


def materialize_recipe_configuration_references(
    *,
    recipe: Mapping[str, Any],
    input_root: str | Path,
    content_store_root: str | Path | None = None,
    allowed_uri_prefixes: Sequence[str],
    fetcher: ReferenceFetcher = default_reference_fetcher,
) -> list[dict[str, Any]]:
    """Read back every immutable stage configuration embedded in a recipe."""

    validated_recipe = validate_scene_construction_recipe(recipe)
    validated_prefixes = validate_allowed_uri_prefixes(allowed_uri_prefixes)
    root = Path(input_root).expanduser()
    if root.is_symlink():
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_input_root_unsafe"
        )
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    content_root = Path(content_store_root or root).expanduser()
    if content_root.is_symlink():
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_content_store_root_unsafe"
        )
    content_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    content_root = content_root.resolve(strict=True)
    references = [
        {
            "contract_path": f"construction.recipe.stage_sequence.{index}.configuration",
            **stage["configuration"],
        }
        for index, stage in enumerate(validated_recipe["stage_sequence"])
    ]
    rows, _ = _materialize_reference_records(
        references=references,
        input_root=root,
        content_store_root=content_root,
        allowed_uri_prefixes=validated_prefixes,
        fetcher=fetcher,
    )
    return rows


def _load_envelope(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_queue_envelope_invalid"
        ) from exc
    if (
        path.is_symlink()
        or not isinstance(value, Mapping)
        or value.get("schema_version") != ENVELOPE_SCHEMA_VERSION
        or value.get("envelope_digest")
        != canonical_digest(value, digest_field="envelope_digest")
    ):
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_queue_envelope_invalid"
        )
    validate_launch_preparation_request(value["request"])
    return dict(value)


def _validated_production_recipe(
    *, request: Mapping[str, Any], materialized_path: str | Path
) -> dict[str, Any]:
    path = Path(materialized_path).resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        recipe = validate_scene_construction_recipe(value)
    except (OSError, json.JSONDecodeError, TaskEvaluationSceneConstructionRecipeError) as exc:
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_construction_recipe_invalid"
        ) from exc
    # ``revision.source_commit`` identifies the historical configuration
    # release and is already protected by the revision digest.  The current
    # evaluator release is independently bound by the preparation request and
    # worker source-commit check; requiring those two identities to be equal
    # would make every immutable configured scene expire when main advances.
    expected = {
        "team_namespace": request["team_namespace"],
        "scene_identity": request["scene"]["identity"],
        "task_identity": request["task"]["identity"],
        "subject_identity": request["task"]["subject"]["identity"],
        "source_manifest_digest": request["scene"]["source_manifest"]["digest"],
        "rights_admission_digest": request["scene"]["rights"]["admission"]["digest"],
        "output_identity": request["construction"]["output_identity"],
    }
    if any(recipe.get(key) != expected_value for key, expected_value in expected.items()):
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_construction_recipe_binding_mismatch"
        )
    return recipe


def _validated_configured_scene_revision(
    *, request: Mapping[str, Any], materialized_path: str | Path
) -> dict[str, Any]:
    path = Path(materialized_path).resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        revision = validate_configured_scene_revision(value)
    except (
        OSError,
        json.JSONDecodeError,
        TaskEvaluationConfiguredSceneRevisionError,
    ) as exc:
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_configured_scene_revision_invalid"
        ) from exc
    expected = {
        "team_namespace": request["team_namespace"],
        "scene_identity": request["scene"]["identity"],
        "revision_digest": request["task"][
            "configured_scene_revision_digest"
        ],
    }
    if any(
        revision.get(key) != expected_value
        for key, expected_value in expected.items()
    ):
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_configured_scene_revision_binding_mismatch"
        )
    if (
        revision["task_template"]["identity"] != request["task"]["identity"]
        or revision["replacement"]["identity"]
        != request["task"]["subject"]["identity"]
    ):
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_configured_scene_revision_binding_mismatch"
        )
    return revision


def process_launch_preparation_queue(
    *,
    queue_root: str | Path,
    input_root: str | Path,
    allowed_uri_prefixes: Sequence[str],
    service_account: str,
    source_commit: str | None = None,
    max_messages: int = 1,
    fetcher: ReferenceFetcher = default_reference_fetcher,
    adapter_materializer: AdapterMaterializer = materialize_native_arena_adapter,
    scene_render_input_materializer: SceneRenderInputMaterializer = (
        materialize_scene_configuration_render_inputs
    ),
    construction_queue_root: str | Path | None = None,
    episode_compilation_queue_root: str | Path | None = None,
) -> dict[str, Any]:
    """Claim and materialize bounded queue items without any paid mutation."""

    if not isinstance(max_messages, int) or isinstance(max_messages, bool) or not 1 <= max_messages <= 32:
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_max_messages_invalid"
        )
    observed_source_commit = source_commit or running_worker_source_commit()
    if not re.fullmatch(r"[0-9a-f]{40}", observed_source_commit):
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_worker_source_commit_unproven"
        )
    root = ensure_launch_preparation_queue_root(queue_root)
    results_root = root / "results"
    results_root.mkdir(mode=0o750, exist_ok=True)
    conflicts_root = results_root / "conflicts"
    conflicts_root.mkdir(mode=0o750, exist_ok=True)
    content_store_root = Path(input_root) / "content-addressed" / "sha256"
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
        terminal_state = "materialized"
        try:
            envelope = _load_envelope(claimed)
            result = materialize_preparation_references(
                request=envelope["request"],
                input_root=Path(input_root) / str(envelope["request"]["preparation_id"]),
                content_store_root=content_store_root,
                allowed_uri_prefixes=allowed_uri_prefixes,
                service_account=service_account,
                source_commit=observed_source_commit,
                fetcher=fetcher,
            )
            references_by_path = {
                row["contract_path"]: row for row in result["references"]
            }
            policy_run_configuration = envelope["request"].get(
                "policy_run_configuration"
            )
            if policy_run_configuration is not None:
                try:
                    setup = validate_policy_run_setup(
                        envelope["request"]["policy_run_setup"]
                    )
                    policy_run_plan = build_policy_run_plan(
                        policy_run_configuration, setup=setup
                    )
                except (
                    TaskEvaluationPolicyRunContractError,
                ) as exc:
                    raise TaskEvaluationLaunchPreparationWorkerError(
                        f"launch_preparation_policy_run_setup_invalid:{exc}"
                    ) from exc
                result["policy_run_plan"] = policy_run_plan
            construction_mode = envelope["request"]["construction"]["mode"]
            runtime_source = references_by_path.get(
                "execution_adapter.runtime_source_bundle"
            )
            if runtime_source is None:
                raise TaskEvaluationLaunchPreparationWorkerError(
                    "launch_preparation_execution_adapter_inputs_missing"
                )
            if construction_mode == "reuse_configured_scene":
                configured_revision_record = references_by_path.get(
                    "scene.configured_revision"
                )
                if configured_revision_record is None:
                    raise TaskEvaluationLaunchPreparationWorkerError(
                        "launch_preparation_execution_adapter_inputs_missing"
                    )
                configured_revision = _validated_configured_scene_revision(
                    request=envelope["request"],
                    materialized_path=configured_revision_record[
                        "materialized_path"
                    ],
                )
                revision_inputs_root = (
                    Path(input_root)
                    / str(envelope["request"]["preparation_id"])
                    / "configured-scene-revision-inputs"
                )
                revision_inputs_root.mkdir(
                    parents=True, exist_ok=True, mode=0o750
                )
                transitive_references = [
                    {
                        "contract_path": (
                            "scene.configured_revision.configured_scene_bundle"
                        ),
                        **configured_revision["configured_scene_bundle"],
                    },
                    {
                        "contract_path": (
                            "scene.configured_revision.source.manifest"
                        ),
                        **configured_revision["source"]["manifest"],
                    },
                    {
                        "contract_path": (
                            "scene.configured_revision.source.rights_admission"
                        ),
                        **configured_revision["source"]["rights_admission"],
                    },
                    {
                        "contract_path": (
                            "scene.configured_revision.registration.metric"
                        ),
                        **configured_revision["registration"]["metric"],
                    },
                    {
                        "contract_path": (
                            "scene.configured_revision.replacement."
                            "static_qualification"
                        ),
                        **configured_revision["replacement"][
                            "static_qualification"
                        ],
                    },
                    {
                        "contract_path": (
                            "scene.configured_revision.replacement."
                            "native_import_qualification"
                        ),
                        **configured_revision["replacement"][
                            "native_import_qualification"
                        ],
                    },
                    {
                        "contract_path": (
                            "scene.configured_revision.registration.support_plane"
                        ),
                        **configured_revision["registration"]["support_plane"],
                    },
                    {
                        "contract_path": (
                            "scene.configured_revision.registration.robot_mount_interface"
                        ),
                        **configured_revision["registration"][
                            "robot_mount_interface"
                        ],
                    },
                    {
                        "contract_path": (
                            "scene.configured_revision.registration.camera_calibration"
                        ),
                        **configured_revision["registration"][
                            "camera_calibration"
                        ],
                    },
                    {
                        "contract_path": (
                            "scene.configured_revision.registration.workspace_clearance"
                        ),
                        **configured_revision["registration"][
                            "workspace_clearance"
                        ],
                    },
                    {
                        "contract_path": (
                            "scene.configured_revision.replacement.source_object"
                        ),
                        **configured_revision["replacement"]["source_object"],
                    },
                    {
                        "contract_path": (
                            "scene.configured_revision.replacement.static_qualification"
                        ),
                        **configured_revision["replacement"][
                            "static_qualification"
                        ],
                    },
                    {
                        "contract_path": (
                            "scene.configured_revision.replacement.native_import_qualification"
                        ),
                        **configured_revision["replacement"][
                            "native_import_qualification"
                        ],
                    },
                    {
                        "contract_path": (
                            "scene.configured_revision.task_template.definition"
                        ),
                        **configured_revision["task_template"]["definition"],
                    },
                    {
                        "contract_path": (
                            "scene.configured_revision.task_template.success_criteria"
                        ),
                        **configured_revision["task_template"]["success_criteria"],
                    },
                    {
                        "contract_path": (
                            "scene.configured_revision.task_template.execution"
                        ),
                        **configured_revision["task_template"]["execution"],
                    },
                ]
                transitive_references.extend(
                    {
                        "contract_path": (
                            "scene.configured_revision.source.rights_evidence."
                            f"{index}.artifact"
                        ),
                        **evidence["artifact"],
                    }
                    for index, evidence in enumerate(
                        configured_revision["source"]["rights_evidence"]
                    )
                )
                revision_input_rows, _ = _materialize_reference_records(
                    references=transitive_references,
                    input_root=revision_inputs_root,
                    content_store_root=content_store_root,
                    allowed_uri_prefixes=validate_allowed_uri_prefixes(
                        allowed_uri_prefixes
                    ),
                    fetcher=fetcher,
                )
                scene_bundle_record = next(
                    row
                    for row in revision_input_rows
                    if row["contract_path"]
                    == "scene.configured_revision.configured_scene_bundle"
                )
                result["references"].extend(revision_input_rows)
                result["reference_count"] = len(result["references"])
                result["unique_object_count"] = len(
                    {
                        (row["digest"], row["size_bytes"])
                        for row in result["references"]
                    }
                )
                result["content_addressed_reuse_count"] = sum(
                    row["content_addressed_reuse"]
                    for row in result["references"]
                )
                result["result_digest"] = canonical_digest(
                    result, digest_field="result_digest"
                )
                if episode_compilation_queue_root is None:
                    raise TaskEvaluationLaunchPreparationWorkerError(
                        "launch_preparation_episode_compilation_queue_missing"
                    )
                compilation = stage_episode_compilation(
                    request=envelope["request"],
                    preparation_result=result,
                    configured_revision=configured_revision,
                    configured_scene_bundle_reference=scene_bundle_record,
                    queue_root=episode_compilation_queue_root,
                )
                result.update(
                    {
                        "status": "queued_for_production_episode_compilation",
                        "run_mode": "episode_evaluation",
                        "configured_scene_revision_digest": configured_revision[
                            "revision_digest"
                        ],
                        "configured_scene_bundle_digest": scene_bundle_record[
                            "digest"
                        ],
                        "episode_compilation_id": compilation["compilation_id"],
                        "episode_compilation_queue_envelope_digest": compilation[
                            "envelope_digest"
                        ],
                        "episode_compilation_queue_receipt_digest": compilation[
                            "receipt_digest"
                        ],
                        "customer_supplied_prebuilt_episode_packet": False,
                        "construction_packet_materialized": False,
                        "automatic_progression_required": True,
                        "result_digest": "",
                    }
                )
            else:
                recipe_record = references_by_path.get("construction.recipe")
                if recipe_record is None:
                    raise TaskEvaluationLaunchPreparationWorkerError(
                        "launch_preparation_construction_recipe_missing"
                    )
                recipe = _validated_production_recipe(
                    request=envelope["request"],
                    materialized_path=recipe_record["materialized_path"],
                )
                recipe_configuration_references = (
                    materialize_recipe_configuration_references(
                        recipe=recipe,
                        input_root=(
                            Path(input_root)
                            / str(envelope["request"]["preparation_id"])
                            / "construction-stage-configurations"
                        ),
                        content_store_root=content_store_root,
                        allowed_uri_prefixes=allowed_uri_prefixes,
                        fetcher=fetcher,
                    )
                )
                result["references"].extend(recipe_configuration_references)
                result["reference_count"] = len(result["references"])
                result["unique_object_count"] = len(
                    {
                        (row["digest"], row["size_bytes"])
                        for row in result["references"]
                    }
                )
                result["content_addressed_reuse_count"] = sum(
                    row["content_addressed_reuse"]
                    for row in result["references"]
                )
                stage_one_path = Path(
                    str(recipe_configuration_references[0]["materialized_path"])
                ).resolve()
                try:
                    stage_one_configuration = json.loads(
                        stage_one_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    raise TaskEvaluationLaunchPreparationWorkerError(
                        "launch_preparation_scene_render_configuration_invalid"
                    ) from exc
                render_inputs = scene_render_input_materializer(
                    envelope={
                        **envelope,
                        "recipe": recipe,
                        "materialized_references": result["references"],
                    },
                    stage_one_configuration=stage_one_configuration,
                    output_root=(
                        Path(input_root)
                        / str(envelope["request"]["preparation_id"])
                        / "configuration-render-inputs"
                    ),
                )
                if (
                    render_inputs.get("schema_version")
                    != "task_evaluation_scene_configuration_render_inputs.v1"
                    or render_inputs.get("status")
                    not in RENDER_INPUT_STATUSES
                    or render_inputs.get("run_id") != envelope["request"]["run_id"]
                    or not render_inputs_disclosure_is_coherent(render_inputs)
                    or render_inputs.get("provider_mutation_performed") is not False
                    or render_inputs.get("paid_execution_requested") is not False
                    or render_inputs.get("result_digest")
                    != canonical_digest(
                        render_inputs, digest_field="result_digest"
                    )
                ):
                    raise TaskEvaluationLaunchPreparationWorkerError(
                        "launch_preparation_scene_render_result_invalid"
                    )
                if construction_queue_root is None:
                    raise TaskEvaluationLaunchPreparationWorkerError(
                        "launch_preparation_scene_construction_queue_missing"
                    )
                construction_intake = stage_scene_construction(
                    request=envelope["request"],
                    preparation_result=result,
                    recipe=recipe,
                    recipe_configuration_references=(
                        recipe_configuration_references
                    ),
                    render_inputs_result=render_inputs,
                    queue_root=construction_queue_root,
                )
                result.update(
                    {
                        "status": (
                            "queued_for_production_scene_configuration"
                        ),
                        "run_mode": "scene_configuration",
                        "construction_recipe_digest": recipe["recipe_digest"],
                        "construction_output_identity": recipe["output_identity"],
                        "construction_stage_configuration_count": len(
                            recipe_configuration_references
                        ),
                        "construction_stage_configurations_readback_passed": all(
                            row["full_byte_service_account_readback_passed"]
                            for row in recipe_configuration_references
                        ),
                        "construction_packet_materialized": False,
                        "configuration_render_inputs_result_digest": (
                            render_inputs["result_digest"]
                        ),
                        "configuration_render_input_count": render_inputs[
                            "derived_frame_count"
                        ],
                        "raw_interiorgs_bytes_in_provider_packet": (
                            render_inputs["raw_interiorgs_bytes_in_provider_packet"]
                        ),
                        "construction_orchestration_id": construction_intake[
                            "orchestration_id"
                        ],
                        "construction_queue_envelope_digest": construction_intake[
                            "envelope_digest"
                        ],
                        "construction_queue_receipt_digest": construction_intake[
                            "receipt_digest"
                        ],
                        "automatic_progression_required": True,
                        "runtime_source_bundle_readback_passed": True,
                        "result_digest": "",
                    }
                )
            result["result_digest"] = canonical_digest(
                result, digest_field="result_digest"
            )
        except Exception as exc:
            terminal_state = "blocked"
            result = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": "blocked",
                "preparation_id": re.sub(
                    r"-[0-9a-f]{64}\.json$", "", source.name
                ),
                "blockers": [
                    str(exc)
                    if isinstance(
                        exc,
                        (
                            TaskEvaluationLaunchPreparationWorkerError,
                            TaskEvaluationLaunchPreparationQueueError,
                            TaskEvaluationSceneConstructionQueueError,
                            TaskEvaluationEpisodeCompilationQueueError,
                        ),
                    )
                    else f"launch_preparation_worker_failed:{type(exc).__name__}"
                ],
                "provider_mutation_performed": False,
                "catalog_mutation_performed": False,
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
            try:
                existing = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            if existing.get("result_digest") == result.get("result_digest"):
                result = existing
            else:
                terminal_state = "blocked"
                conflict: dict[str, Any] = {
                    "schema_version": RESULT_SCHEMA_VERSION,
                    "status": "blocked",
                    "preparation_id": result.get("preparation_id"),
                    "blockers": [
                        "launch_preparation_immutable_result_conflict"
                    ],
                    "existing_result_digest": existing.get("result_digest"),
                    "candidate_result_digest": result.get("result_digest"),
                    "provider_mutation_performed": False,
                    "catalog_mutation_performed": False,
                    "paid_execution_requested": False,
                    "observed_at_iso": datetime.now(timezone.utc).isoformat(),
                    "result_digest": "",
                }
                conflict["result_digest"] = canonical_digest(
                    conflict, digest_field="result_digest"
                )
                conflict_path = conflicts_root / (
                    f"{source.stem}-{conflict['result_digest'].removeprefix('sha256:')}.json"
                )
                try:
                    write_launch_preparation_record_exclusive(
                        conflict_path, conflict
                    )
                except FileExistsError:
                    pass
                result = conflict
        os.replace(claimed, root / terminal_state / source.name)
        processed.append(result)
    return {
        "schema_version": "task_evaluation_launch_preparation_queue_run.v1",
        "status": "processed" if processed else "idle",
        "processed_count": len(processed),
        "results": processed,
        "provider_mutation_performed": False,
        "catalog_mutation_performed": False,
        "paid_execution_requested": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", default=os.getenv(QUEUE_ROOT_ENV, ""))
    parser.add_argument("--input-root", default=os.getenv(INPUT_ROOT_ENV, ""))
    parser.add_argument(
        "--construction-queue-root",
        default=os.getenv(CONSTRUCTION_QUEUE_ROOT_ENV, ""),
    )
    parser.add_argument(
        "--episode-compilation-queue-root",
        default=os.getenv(EPISODE_COMPILATION_QUEUE_ROOT_ENV, ""),
    )
    parser.add_argument(
        "--allowed-uri-prefixes-json",
        default=os.getenv(ALLOWED_URI_PREFIXES_ENV, ""),
    )
    parser.add_argument(
        "--service-account", default=os.getenv(SERVICE_ACCOUNT_ENV, "blueprint")
    )
    parser.add_argument("--max-messages", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        prefixes = json.loads(args.allowed_uri_prefixes_json)
    except json.JSONDecodeError:
        prefixes = None
    if (
        not args.queue_root
        or not args.input_root
        or not args.construction_queue_root
        or not args.episode_compilation_queue_root
        or not isinstance(prefixes, list)
        or not all(isinstance(item, str) and item for item in prefixes)
    ):
        print(
            json.dumps(
                {
                    "schema_version": (
                        "task_evaluation_launch_preparation_queue_run.v1"
                    ),
                    "status": "blocked",
                    "blockers": ["launch_preparation_worker_configuration_invalid"],
                    "provider_mutation_performed": False,
                    "catalog_mutation_performed": False,
                    "paid_execution_requested": False,
                },
                sort_keys=True,
            )
        )
        return 2
    try:
        result = process_launch_preparation_queue(
            queue_root=args.queue_root,
            input_root=args.input_root,
            allowed_uri_prefixes=prefixes,
            service_account=args.service_account,
            source_commit=running_worker_source_commit(),
            max_messages=args.max_messages,
            construction_queue_root=args.construction_queue_root,
            episode_compilation_queue_root=args.episode_compilation_queue_root,
        )
    except (
        TaskEvaluationLaunchPreparationWorkerError,
        TaskEvaluationSceneConstructionQueueError,
        TaskEvaluationEpisodeCompilationQueueError,
        OSError,
    ) as exc:
        print(
            json.dumps(
                {
                    "schema_version": (
                        "task_evaluation_launch_preparation_queue_run.v1"
                    ),
                    "status": "blocked",
                    "blockers": [
                        str(exc)
                        if isinstance(exc, TaskEvaluationLaunchPreparationWorkerError)
                        else f"launch_preparation_worker_failed:{type(exc).__name__}"
                    ],
                    "provider_mutation_performed": False,
                    "catalog_mutation_performed": False,
                    "paid_execution_requested": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "TaskEvaluationLaunchPreparationWorkerError",
    "collect_preparation_references",
    "default_reference_fetcher",
    "materialize_preparation_references",
    "process_launch_preparation_queue",
    "running_worker_source_commit",
    "validate_allowed_uri_prefixes",
]


if __name__ == "__main__":
    raise SystemExit(main())
