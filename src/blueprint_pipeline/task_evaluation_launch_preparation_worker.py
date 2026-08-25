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
ReferenceFetcher = Callable[[str, Path, int], None]
ALLOWED_REFERENCE_SCHEMES = frozenset({"gs", "https", "s3"})


class TaskEvaluationLaunchPreparationWorkerError(RuntimeError):
    """A claimed no-spend preparation could not be completed safely."""


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
    if path.is_symlink():
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_object_store_secret_file_unsafe"
        )
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
        mode = stat.S_IMODE(metadata.st_mode)
        value = resolved.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_object_store_secret_file_unavailable"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode) or mode & 0o077 or not value:
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_object_store_secret_file_unsafe"
        )
    return value


def _s3_client() -> Any:
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError as exc:
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_s3_client_unavailable"
        ) from exc
    access_key = _private_file_value(
        "BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID_FILE"
    )
    secret_key = _private_file_value(
        "BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY_FILE"
    )
    if bool(access_key) != bool(secret_key):
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_object_store_secret_pair_incomplete"
        )
    kwargs: dict[str, Any] = {}
    if access_key and secret_key:
        kwargs.update(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
    endpoint = _private_file_value(
        "BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL_FILE"
    )
    region = _private_file_value("BLUEPRINT_WAM_OBJECT_STORE_REGION_FILE")
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
        response = _s3_client().get_object(
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
    allowed_uri_prefixes: Sequence[str],
    service_account: str,
    fetcher: ReferenceFetcher = default_reference_fetcher,
) -> dict[str, Any]:
    """Materialize and read back every immutable input, content-addressed."""

    validated = validate_launch_preparation_request(request)
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
    rows: list[dict[str, Any]] = []
    by_identity: dict[tuple[str, int], Path] = {}
    for reference in collect_preparation_references(validated):
        uri = reference["uri"]
        digest = reference["digest"]
        size = reference["size_bytes"]
        if not _prefix_allowed(uri, validated_prefixes):
            raise TaskEvaluationLaunchPreparationWorkerError(
                "launch_preparation_reference_prefix_not_allowed"
            )
        identity = (digest, size)
        destination = by_identity.get(identity)
        reused = destination is not None
        if destination is None:
            destination = root / digest.removeprefix("sha256:")
            by_identity[identity] = destination
            if destination.is_symlink():
                raise TaskEvaluationLaunchPreparationWorkerError(
                    "launch_preparation_materialized_target_unsafe"
                )
            if not destination.exists():
                temporary = root / (
                    f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
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
                        os.link(temporary, destination, follow_symlinks=False)
                    except FileExistsError:
                        pass
                    directory = os.open(
                        root,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                    )
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                finally:
                    temporary.unlink(missing_ok=True)
            if destination.is_symlink() or not stat.S_ISREG(
                destination.lstat().st_mode
            ):
                raise TaskEvaluationLaunchPreparationWorkerError(
                    "launch_preparation_materialized_target_unsafe"
                )
            observed_digest, observed_size = _sha256_and_size(destination)
            if observed_digest != digest or observed_size != size:
                raise TaskEvaluationLaunchPreparationWorkerError(
                    "launch_preparation_materialized_target_identity_mismatch"
                )
        rows.append(
            {
                **reference,
                "materialized_path": str(destination),
                "content_addressed_reuse": reused,
                "full_byte_service_account_readback_passed": True,
            }
        )
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "inputs_materialized_awaiting_construction_adapter",
        "preparation_id": validated["preparation_id"],
        "run_id": validated["run_id"],
        "team_namespace": validated["team_namespace"],
        "reference_count": len(rows),
        "unique_object_count": len(by_identity),
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


def process_launch_preparation_queue(
    *,
    queue_root: str | Path,
    input_root: str | Path,
    allowed_uri_prefixes: Sequence[str],
    service_account: str,
    max_messages: int = 1,
    fetcher: ReferenceFetcher = default_reference_fetcher,
) -> dict[str, Any]:
    """Claim and materialize bounded queue items without any paid mutation."""

    if not isinstance(max_messages, int) or isinstance(max_messages, bool) or not 1 <= max_messages <= 32:
        raise TaskEvaluationLaunchPreparationWorkerError(
            "launch_preparation_max_messages_invalid"
        )
    root = ensure_launch_preparation_queue_root(queue_root)
    results_root = root / "results"
    results_root.mkdir(mode=0o750, exist_ok=True)
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
        terminal_state = "materialized"
        try:
            envelope = _load_envelope(claimed)
            result = materialize_preparation_references(
                request=envelope["request"],
                input_root=Path(input_root) / str(envelope["request"]["preparation_id"]),
                allowed_uri_prefixes=allowed_uri_prefixes,
                service_account=service_account,
                fetcher=fetcher,
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
            max_messages=args.max_messages,
        )
    except (TaskEvaluationLaunchPreparationWorkerError, OSError) as exc:
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
    "validate_allowed_uri_prefixes",
]


if __name__ == "__main__":
    raise SystemExit(main())
