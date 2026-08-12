"""Build a frozen Vast packet for released-code Gaussian ownership evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import zipfile
from contextlib import contextmanager
from email.parser import Parser
from pathlib import Path
from typing import Any, Mapping, Sequence

from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .paid_resource_admission import PaidResourceAdmissionGrant
from .paid_attempt_authority import (
    active_instance_allowlist_metadata_error,
    flatten_active_instance_allowlist,
    normalize_active_instance_allowlist,
)
from .provider_runtime_bundle_contract import provider_runtime_contract_blockers
from .provider_bundle_rehearsal import (
    provider_bundle_rehearsal_blockers,
    rehearse_provider_bundle_entrypoint,
)
from .public_scene_gaussian_excision_audit import FREEZE_SCHEMA
from .vast_independent_watchdog_control import (
    arm_independent_vast_watchdog,
    close_independent_vast_watchdog,
)
from .vast_provider_adapter import run_vast_provider_adapter
from .vast_session_budget_contract import attempt_estimated_cost, attempt_runtime_seconds
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


from .spend_authority_consumption_root import consumption_root

PROBE_KIND = "adp-gaussian-excision"
PROVIDER_BUNDLE_KIND = "adp_gaussian_excision"
SCHEMA_VERSION = "adp009b_gaussian_excision_vast_bundle.v1"
RESULT_SCHEMA_VERSION = "adp009b_gaussian_excision_vast_run.v1"
AUTHORITY_SCHEMA = "public_scene_gaussian_excision_execution_authority.v1"
PAID_ATTEMPT_AUTHORITY_SCHEMA = (
    "public_scene_gaussian_excision_paid_attempt_authority.v1"
)
SOURCE_REPOSITORY = "https://github.com/florinshen/FlashSplat"
SOURCE_COMMIT = "3e3b14786333bf0163ba1b8541e86a3765112d7d"
SOURCE_TREE = "a5b5d91656a17df12e9c12db240cea15062e5f43"
RASTERIZER_PATH = "submodules/flashsplat-rasterization"
RASTERIZER_REPOSITORY = "https://github.com/florinshen/flashsplat-rasterization"
RASTERIZER_COMMIT = "189c483ffa33dd6d5661343ce496df0c6eb80a0c"
DIFF_RASTERIZER_PATH = "submodules/diff-gaussian-rasterization"
DIFF_RASTERIZER_COMMIT = "8829d14f814fccdaf840b7b0f3021a616583c0a1"
GLM_PATH = "submodules/diff-gaussian-rasterization/third_party/glm"
GLM_COMMIT = "5c46b9c07008ae65cb81ab79cd677ecc1934b903"
SIMPLE_KNN_PATH = "submodules/simple-knn"
SIMPLE_KNN_REPOSITORY = "https://gitlab.inria.fr/bkerbl/simple-knn.git"
SIMPLE_KNN_COMMIT = "86710c2d4b46680c02301765dd79e465819c8f19"
DEFAULT_IMAGE = (
    "docker.io/pytorch/pytorch@"
    "sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014"
)
DEPENDENCY_WHEELHOUSE_SCHEMA = "adp_gaussian_excision_dependency_wheelhouse.v1"
DEPENDENCY_PYTHON_VERSION = "3.11"
DEPENDENCY_PLATFORM_TAGS = ("manylinux2014_x86_64", "manylinux_2_17_x86_64")
DEPENDENCY_REQUIREMENTS = {
    "ninja": "1.13.0",
    "numpy": "1.26.4",
    "opencv-python-headless": "4.11.0.86",
    "packaging": "25.0",
    "pillow": "10.2.0",
    "plyfile": "1.1.3",
    "setuptools": "80.9.0",
    "wheel": "0.45.1",
}
EXPECTED_SUBMODULES = {
    RASTERIZER_PATH: RASTERIZER_COMMIT,
    DIFF_RASTERIZER_PATH: DIFF_RASTERIZER_COMMIT,
    GLM_PATH: GLM_COMMIT,
    SIMPLE_KNN_PATH: SIMPLE_KNN_COMMIT,
}
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/gaussian-excision"
_VAST_MUTATION_ENV = (
    "BLUEPRINT_ALLOW_VAST_API_CALLS",
    "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH",
)
_VAST_SINGLE_ATTEMPT_ENV = "BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"


def gaussian_excision_lane_identity(freeze: Mapping[str, Any]) -> dict[str, str]:
    """Derive collision-free provider identities from frozen task evidence."""

    scene = freeze.get("scene")
    freeze_digest = str(freeze.get("freeze_digest") or "")
    if not isinstance(scene, Mapping):
        raise ValueError("gaussian_excision_lane_identity_invalid")
    scene_id = str(scene.get("publisher_scene_id") or "").strip()
    target_id = str(scene.get("target_instance_id") or "").strip()
    if (
        not scene_id
        or not target_id
        or not freeze_digest.startswith("sha256:")
        or len(freeze_digest) != 71
        or any(not value.replace("_", "-").isalnum() for value in (scene_id, target_id))
    ):
        raise ValueError("gaussian_excision_lane_identity_invalid")
    suffix = freeze_digest.removeprefix("sha256:")[:12]
    lane_id = f"{scene_id}-{target_id}-{suffix}"
    return {
        "lane_id": lane_id,
        "object_store_key_prefix": f"{DEFAULT_KEY_PREFIX}/{lane_id}",
        "instance_label_prefix": f"blueprint-adp-gaussian-excision-{lane_id}-",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_canonical(path: Path, *, field: str, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if (
        not isinstance(value, dict)
        or value.get(field) != canonical_digest(value, digest_field=field)
    ):
        raise ValueError(code)
    return value


def validate_gaussian_excision_paid_attempt_authority(
    authority: Mapping[str, Any],
    *,
    prepared_bundle: Mapping[str, Any],
    previous_attempt_receipt: Mapping[str, Any] | None,
    allowed_active_instance_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Bind one explicit paid-attempt grant to its bundle and prior receipt.

    Bundle construction and dry-run admission intentionally do not require this
    grant.  The grant is an execution-only, single-use capability layered over
    the broader scene/upload authority embedded in the immutable bundle.
    """

    value = dict(authority)
    ordinal = value.get("paid_attempt_ordinal")
    prior_digest = value.get("previous_attempt_receipt_digest")
    reference = value.get("authority_reference")
    authorized_by = value.get("authorized_by")
    authorized_on = value.get("authorized_on")
    structured_allowlist = "active_instance_allowlist" in value
    authority_allowlist_value = value.get(
        "active_instance_allowlist", value.get("external_instance_allowlist")
    )
    authority_allowlist = normalize_active_instance_allowlist(
        authority_allowlist_value
    )
    expected_allowlist = normalize_active_instance_allowlist(allowed_active_instance_ids)
    errors: list[str] = []
    if value.get("schema_version") != PAID_ATTEMPT_AUTHORITY_SCHEMA:
        errors.append("schema_invalid")
    if value.get("authority_kind") != "explicit_user_direction_in_current_goal":
        errors.append("authority_kind_invalid")
    if not isinstance(reference, str) or not reference.strip():
        errors.append("authority_reference_invalid")
    if not isinstance(authorized_by, str) or not authorized_by.strip():
        errors.append("authorized_by_invalid")
    if not isinstance(authorized_on, str) or not authorized_on.strip():
        errors.append("authorized_on_invalid")
    if value.get("purpose") != "released_code_gaussian_ownership_audit":
        errors.append("purpose_invalid")
    if value.get("provider") != "vast":
        errors.append("provider_invalid")
    if value.get("paid_compute_authorized") is not True:
        errors.append("paid_compute_not_authorized")
    if value.get("parent_execution_authority_digest") != prepared_bundle.get(
        "execution_authority_digest"
    ):
        errors.append("parent_execution_authority_digest_mismatch")
    if value.get("freeze_digest") != prepared_bundle.get("freeze_digest"):
        errors.append("freeze_digest_mismatch")
    if value.get("bundle_sha256") != prepared_bundle.get("bundle_sha256"):
        errors.append("bundle_sha256_mismatch")
    if value.get("corrective_blueprint_commit") != prepared_bundle.get(
        "blueprint_commit"
    ):
        errors.append("corrective_blueprint_commit_mismatch")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        errors.append("paid_attempt_ordinal_invalid")
    if value.get("maximum_paid_attempts") != 1:
        errors.append("maximum_paid_attempts_invalid")
    if value.get("maximum_automatic_retries") != 0:
        errors.append("maximum_automatic_retries_invalid")
    if value.get("automatic_paid_retry_authorized") is not False:
        errors.append("automatic_paid_retry_authorized_invalid")
    if value.get("hard_attempt_spend_cap_usd") != prepared_bundle.get(
        "hard_cap_usd"
    ):
        errors.append("hard_attempt_spend_cap_mismatch")
    if value.get("maximum_single_resource_ttl_seconds") != prepared_bundle.get(
        "hard_ttl_seconds"
    ):
        errors.append("maximum_single_resource_ttl_mismatch")
    if authority_allowlist is None:
        errors.append(
            "active_instance_allowlist_invalid"
            if structured_allowlist
            else "external_instance_allowlist_invalid"
        )
    elif expected_allowlist is None:
        errors.append("allowed_active_instance_ids_invalid")
    elif flatten_active_instance_allowlist(
        authority_allowlist
    ) != flatten_active_instance_allowlist(expected_allowlist):
        errors.append(
            "active_instance_allowlist_mismatch"
            if structured_allowlist
            else "external_instance_allowlist_mismatch"
        )
    elif (metadata_error := active_instance_allowlist_metadata_error(
        value, allowlist=authority_allowlist
    )) is not None:
        errors.append(metadata_error)
    if value.get("authorization_digest") != canonical_digest(
        value, digest_field="authorization_digest"
    ):
        errors.append("authorization_digest_invalid")

    if ordinal == 1:
        if prior_digest is not None or previous_attempt_receipt is not None:
            errors.append("unexpected_previous_attempt_receipt")
    elif isinstance(ordinal, int) and ordinal > 1:
        prior = (
            dict(previous_attempt_receipt)
            if isinstance(previous_attempt_receipt, Mapping)
            else None
        )
        if prior is None:
            errors.append("previous_attempt_receipt_missing")
        else:
            if prior.get("receipt_digest") != canonical_digest(
                prior, digest_field="receipt_digest"
            ):
                errors.append("previous_attempt_receipt_digest_invalid")
            if prior.get("schema_version") != "adp_gaussian_excision_attempt_receipt.v1":
                errors.append("previous_attempt_receipt_schema_invalid")
            if prior_digest != prior.get("receipt_digest"):
                errors.append("previous_attempt_receipt_digest_mismatch")
            if prior.get("freeze_digest") != prepared_bundle.get("freeze_digest"):
                errors.append("previous_attempt_freeze_digest_mismatch")
            if prior.get("status") not in {
                "sealed_blocked_attempt",
                "sealed_completed_attempt",
            }:
                errors.append("previous_attempt_status_invalid")
            if (
                prior.get("retry_cap") != 0
                or prior.get("continuing_spend") is not False
                or prior.get("provider_absence_confirmed") is not True
            ):
                errors.append("previous_attempt_closeout_invalid")
            prior_ordinal = prior.get("paid_attempt_ordinal")
            if prior_ordinal is None:
                if ordinal != 2:
                    errors.append("legacy_previous_attempt_ordinal_ambiguous")
            elif prior_ordinal != ordinal - 1:
                errors.append("previous_attempt_ordinal_mismatch")
    if errors:
        raise ValueError(
            "gaussian_excision_paid_attempt_authority_invalid:"
            + ",".join(sorted(set(errors)))
        )
    return value


def consume_gaussian_excision_paid_attempt_authority_once(
    authority: Mapping[str, Any], *, blueprint_commit: str
) -> dict[str, Any]:
    """Atomically consume a validated paid-attempt grant before provider mutation."""

    authorization_digest = str(authority.get("authorization_digest") or "")
    if not authorization_digest.startswith("sha256:") or len(authorization_digest) != 71:
        return {
            "status": "blocked",
            "blockers": ["gaussian_excision_paid_attempt_authority_identity_invalid"],
        }
    root = consumption_root()
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root_stat = root.stat()
        if (
            root.is_symlink()
            or root_stat.st_uid != os.getuid()
            or root_stat.st_mode & 0o077
        ):
            return {
                "status": "blocked",
                "blockers": ["gaussian_excision_authority_consumption_root_insecure"],
            }
        identity = authorization_digest.removeprefix("sha256:")
        destination = root / f"gaussian-excision-{identity}.json"
        record = {
            "schema_version": "gaussian_excision_paid_attempt_consumption.v1",
            "authorization_digest": authorization_digest,
            "paid_attempt_ordinal": authority.get("paid_attempt_ordinal"),
            "bundle_sha256": authority.get("bundle_sha256"),
            "freeze_digest": authority.get("freeze_digest"),
            "blueprint_commit": blueprint_commit,
            "consumed_at": utc_now_iso(),
            "maximum_provider_allocations": 1,
        }
        raw = (
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        temporary = root / f".{identity}.{os.getpid()}.tmp"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, destination)
            directory_descriptor = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)
    except FileExistsError:
        return {
            "status": "blocked",
            "blockers": ["gaussian_excision_paid_attempt_authority_consumed"],
        }
    except OSError:
        return {
            "status": "blocked",
            "blockers": ["gaussian_excision_authority_consumption_write_failed"],
        }
    return {
        "status": "consumed",
        "authorization_digest": authorization_digest,
        "consumption_record_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "record_location_disclosed": False,
    }


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_files(root: Path) -> list[Path]:
    output = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    rows = [Path(value.decode()) for value in output.split(b"\0") if value]
    return [row for row in rows if (root / row).is_file()]


def _write_source_archive(source: Path, destination: Path) -> None:
    entries: dict[str, Path] = {
        row.as_posix(): source / row for row in _tracked_files(source)
    }
    for submodule in sorted(EXPECTED_SUBMODULES):
        subroot = source / submodule
        entries.update(
            {
                (Path(submodule) / row).as_posix(): subroot / row
                for row in _tracked_files(subroot)
            }
        )
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def _deterministic_zip(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(row for row in source.rglob("*") if row.is_file()):
            info = zipfile.ZipInfo(
                path.relative_to(source).as_posix(),
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100755 << 16 if path.stat().st_mode & stat.S_IXUSR else 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def _source_identity(source: Path) -> dict[str, Any]:
    if (
        _git(source, "rev-parse", "HEAD") != SOURCE_COMMIT
        or _git(source, "rev-parse", "HEAD^{tree}") != SOURCE_TREE
        or _git(source, "status", "--short")
    ):
        raise ValueError("gaussian_excision_flashsplat_source_identity_invalid")
    submodules: dict[str, str] = {}
    for path, expected in EXPECTED_SUBMODULES.items():
        root = source / path
        if (
            not root.is_dir()
            or _git(root, "rev-parse", "HEAD") != expected
            or _git(root, "status", "--short")
        ):
            raise ValueError("gaussian_excision_flashsplat_submodule_identity_invalid")
        submodules[path] = expected
    return {
        "repository": SOURCE_REPOSITORY,
        "commit": SOURCE_COMMIT,
        "tree": SOURCE_TREE,
        "submodules": submodules,
        "source_modified": False,
    }


def materialize_gaussian_excision_dependency_wheelhouse(
    *, wheelhouse_path: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    """Seal the complete offline Python closure used by the provider runtime."""

    wheelhouse = Path(wheelhouse_path).expanduser().resolve()
    output = Path(manifest_path).expanduser().resolve()
    if not wheelhouse.is_dir() or wheelhouse.is_symlink() or output.exists():
        raise ValueError("gaussian_excision_dependency_wheelhouse_invalid")
    supported_tags = frozenset(
        [
            *cpython_tags(
                python_version=(3, 11), platforms=list(DEPENDENCY_PLATFORM_TAGS)
            ),
            *compatible_tags(
                python_version=(3, 11),
                interpreter="cp311",
                platforms=list(DEPENDENCY_PLATFORM_TAGS),
            ),
        ]
    )
    rows: list[dict[str, Any]] = []
    observed: dict[str, str] = {}
    for wheel_path in sorted(wheelhouse.glob("*.whl")):
        try:
            name, version, _build, tags = parse_wheel_filename(wheel_path.name)
        except ValueError as exc:
            raise ValueError("gaussian_excision_dependency_wheel_invalid") from exc
        normalized = canonicalize_name(name)
        expected = DEPENDENCY_REQUIREMENTS.get(normalized)
        if expected is None or str(version) != expected or not tags & supported_tags:
            raise ValueError("gaussian_excision_dependency_wheel_invalid")
        if normalized in observed:
            raise ValueError("gaussian_excision_dependency_wheel_duplicate")
        observed[normalized] = str(version)
        try:
            with zipfile.ZipFile(wheel_path) as archive:
                metadata_names = [
                    value
                    for value in archive.namelist()
                    if value.endswith(".dist-info/METADATA")
                    and value.count("/") == 1
                ]
                if len(metadata_names) != 1:
                    raise ValueError("gaussian_excision_dependency_wheel_invalid")
                metadata = archive.read(metadata_names[0]).decode("utf-8")
        except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
            raise ValueError("gaussian_excision_dependency_wheel_invalid") from exc
        metadata_fields = Parser().parsestr(metadata)
        if (
            canonicalize_name(metadata_fields.get("Name", "")) != normalized
            or metadata_fields.get("Version") != expected
        ):
            raise ValueError("gaussian_excision_dependency_wheel_invalid")
        rows.append(
            {
                "distribution": normalized,
                "version": expected,
                "filename": wheel_path.name,
                "size_bytes": wheel_path.stat().st_size,
                "sha256": _sha256(wheel_path),
            }
        )
    if observed != DEPENDENCY_REQUIREMENTS:
        raise ValueError("gaussian_excision_dependency_wheelhouse_incomplete")
    manifest = {
        "schema_version": DEPENDENCY_WHEELHOUSE_SCHEMA,
        "status": "ready",
        "container_image": DEFAULT_IMAGE,
        "python_version": DEPENDENCY_PYTHON_VERSION,
        "platform_tags": list(DEPENDENCY_PLATFORM_TAGS),
        "base_image_packages": {"torch": "2.5.1", "torchvision": "0.20.1"},
        "requirements": [
            {"distribution": name, "version": version}
            for name, version in sorted(DEPENDENCY_REQUIREMENTS.items())
        ],
        "wheels": rows,
        "provider_network_install_required": False,
        "sdists_allowed": False,
        "raw_secret_values_recorded": False,
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    write_json(output, manifest)
    return manifest


def _verified_dependency_wheelhouse(
    *, wheelhouse_path: Path, manifest_path: Path
) -> dict[str, Any]:
    manifest = _read_canonical(
        manifest_path,
        field="manifest_digest",
        code="gaussian_excision_dependency_wheelhouse_invalid",
    )
    expected_requirements = [
        {"distribution": name, "version": version}
        for name, version in sorted(DEPENDENCY_REQUIREMENTS.items())
    ]
    rows = manifest.get("wheels")
    if (
        manifest.get("schema_version") != DEPENDENCY_WHEELHOUSE_SCHEMA
        or manifest.get("status") != "ready"
        or manifest.get("container_image") != DEFAULT_IMAGE
        or manifest.get("python_version") != DEPENDENCY_PYTHON_VERSION
        or manifest.get("requirements") != expected_requirements
        or manifest.get("provider_network_install_required") is not False
        or manifest.get("sdists_allowed") is not False
        or not isinstance(rows, list)
        or len(rows) != len(DEPENDENCY_REQUIREMENTS)
    ):
        raise ValueError("gaussian_excision_dependency_wheelhouse_invalid")
    for row in rows:
        path = wheelhouse_path / str(row.get("filename") or "")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.parent != wheelhouse_path
            or row.get("size_bytes") != path.stat().st_size
            or row.get("sha256") != _sha256(path)
        ):
            raise ValueError("gaussian_excision_dependency_wheelhouse_invalid")
    return manifest


def _evidence_record(path: Path, *, evidence_root: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("gaussian_excision_evidence_file_invalid")
    try:
        relative = resolved.relative_to(evidence_root)
    except ValueError as exc:
        raise ValueError("gaussian_excision_evidence_outside_root") from exc
    return {
        "relative_path": relative.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def materialize_gaussian_excision_attempt_receipt(
    *,
    evidence_root: str | Path,
    bundle_receipt_path: str | Path,
    run_result_path: str | Path,
    execution_result_path: str | Path,
    teardown_manifest_path: str | Path,
    watchdog_evidence_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal one terminal blocked/completed attempt without upgrading its claims."""

    root = Path(evidence_root).expanduser().resolve()
    paths = {
        "bundle_receipt": Path(bundle_receipt_path),
        "run_result": Path(run_result_path),
        "execution_result": Path(execution_result_path),
        "teardown_manifest": Path(teardown_manifest_path),
        "watchdog_evidence": Path(watchdog_evidence_path),
    }
    payloads = {
        name: json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    bundle = payloads["bundle_receipt"]
    run = payloads["run_result"]
    execution = payloads["execution_result"]
    teardown = payloads["teardown_manifest"]
    watchdog = payloads["watchdog_evidence"]
    instance_ids = teardown.get("vast_instance_ids")
    watchdog_instance_ids = watchdog.get("instance_ids")
    if not isinstance(watchdog_instance_ids, list) or not watchdog_instance_ids:
        # ``vast_independent_watchdog_handoff.v1`` originally embedded its
        # terminal instance below this compatibility record.  Current handoffs
        # bind the provider IDs directly at top level.  Accept either exact
        # representation, but never a missing or plural/mismatched set.
        legacy_watchdog_instance = (
            watchdog.get("recorded_vast_instance_teardown") or {}
        ).get("instance_id")
        watchdog_instance_ids = (
            [legacy_watchdog_instance]
            if legacy_watchdog_instance is not None
            else None
        )
    attempt_authority_digest = run.get("attempt_authority_digest")
    paid_attempt_ordinal = run.get("paid_attempt_ordinal")
    authorization_consumption = run.get("authorization_consumption")
    attempt_binding_present = attempt_authority_digest is not None
    if (
        bundle.get("provider_bundle_kind") != PROVIDER_BUNDLE_KIND
        or bundle.get("bundle_sha256") != run.get("bundle_sha256")
        or run.get("status") not in {"blocked", "completed"}
        or execution.get("status") not in {"blocked", "completed"}
        or teardown.get("status") != "completed"
        or teardown.get("continuing_spend_from_this_run") is not False
        or not isinstance(instance_ids, list)
        or len(instance_ids) != 1
        or not isinstance(watchdog_instance_ids, list)
        or len(watchdog_instance_ids) != 1
        or str(instance_ids[0]) != str(watchdog_instance_ids[0])
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("provider_absence_confirmed") is not True
        or (
            "recorded_vast_instance_teardown" in watchdog
            and (watchdog.get("recorded_vast_instance_teardown") or {}).get(
                "provider_absence_confirmed"
            )
            is not True
        )
        or run.get("retry_cap") != 0
        or (
            attempt_binding_present
            and (
                not isinstance(paid_attempt_ordinal, int)
                or isinstance(paid_attempt_ordinal, bool)
                or paid_attempt_ordinal < 1
                or not str(attempt_authority_digest).startswith("sha256:")
                or not isinstance(authorization_consumption, Mapping)
                or authorization_consumption.get("status") != "consumed"
                or authorization_consumption.get("authorization_digest")
                != attempt_authority_digest
            )
        )
    ):
        raise ValueError("gaussian_excision_attempt_join_invalid")
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError("gaussian_excision_attempt_receipt_exists")
    receipt = {
        "schema_version": "adp_gaussian_excision_attempt_receipt.v1",
        "status": (
            "sealed_completed_attempt"
            if run.get("status") == "completed"
            else "sealed_blocked_attempt"
        ),
        "blueprint_commit": bundle.get("blueprint_commit"),
        "bundle_sha256": bundle.get("bundle_sha256"),
        "freeze_digest": bundle.get("freeze_digest"),
        "execution_status": execution.get("status"),
        "execution_blockers": list(execution.get("blockers") or []),
        "released_code_executed": execution.get("released_code_executed") is True,
        "heldout_cameras_accessed_for_classification": execution.get(
            "heldout_cameras_accessed_for_classification"
        )
        is True,
        "instance_id": instance_ids[0],
        "estimated_cost_usd": run.get("estimated_cost_usd"),
        "retry_cap": 0,
        "continuing_spend": False,
        "provider_absence_confirmed": True,
        "records": {
            name: _evidence_record(path, evidence_root=root)
            for name, path in paths.items()
        },
        "proof_boundaries": {
            "gaussian_contribution_evidence_completed": execution.get("status")
            == "completed",
            "gaussian_ownership_qualified": False,
            "source_removal_qualified": False,
            "inpainting_decision_qualified": False,
            "policy_outcome_available": False,
        },
        "raw_secret_values_recorded": False,
    }
    if attempt_binding_present:
        receipt.update(
            {
                "paid_attempt_ordinal": paid_attempt_ordinal,
                "attempt_authority_digest": attempt_authority_digest,
                "previous_attempt_receipt_digest": run.get(
                    "previous_attempt_receipt_digest"
                ),
                "authorization_consumption": authorization_consumption,
            }
        )
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    ensure_dir(output.parent)
    write_json(output, receipt)
    return receipt


def materialize_gaussian_excision_recovery_readiness(
    *,
    evidence_root: str | Path,
    dependency_manifest_path: str | Path,
    bundle_receipt_path: str | Path,
    admission_path: str | Path,
    dry_run_result_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal a mutation-free repaired bundle as ready for new launch authority."""

    root = Path(evidence_root).expanduser().resolve()
    paths = {
        "dependency_manifest": Path(dependency_manifest_path),
        "bundle_receipt": Path(bundle_receipt_path),
        "admission": Path(admission_path),
        "dry_run_result": Path(dry_run_result_path),
    }
    payloads = {
        name: json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    dependency = payloads["dependency_manifest"]
    bundle = payloads["bundle_receipt"]
    admission = payloads["admission"]
    dry_run = payloads["dry_run_result"]
    binding = admission.get("allocation_binding") or {}
    if (
        dependency.get("schema_version") != DEPENDENCY_WHEELHOUSE_SCHEMA
        or dependency.get("status") != "ready"
        or dependency.get("provider_network_install_required") is not False
        or bundle.get("status") != "ready"
        or bundle.get("container_image") != DEFAULT_IMAGE
        or bundle.get("dependency_wheelhouse_manifest_digest")
        != dependency.get("manifest_digest")
        or bundle.get("provider_network_dependency_install_required") is not False
        or (bundle.get("exact_bundle_entrypoint_rehearsal") or {}).get("status")
        != "passed"
        or admission.get("status") != "admitted"
        or binding.get("bundle_sha256") != bundle.get("bundle_sha256")
        or binding.get("orchestrator_source_commit") != bundle.get("blueprint_commit")
        or dry_run.get("status") != "dry_run_ready"
        or dry_run.get("provider_mutations_performed") != 0
        or dry_run.get("retry_cap") != 0
    ):
        raise ValueError("gaussian_excision_recovery_readiness_invalid")
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError("gaussian_excision_recovery_readiness_exists")
    receipt = {
        "schema_version": "adp_gaussian_excision_recovery_readiness.v1",
        "status": "ready_for_new_authority_not_executed",
        "blueprint_commit": bundle.get("blueprint_commit"),
        "container_image": bundle.get("container_image"),
        "bundle_sha256": bundle.get("bundle_sha256"),
        "freeze_digest": bundle.get("freeze_digest"),
        "dependency_wheelhouse_manifest_digest": dependency.get("manifest_digest"),
        "provider_network_dependency_install_required": False,
        "exact_bundle_rehearsal_passed": True,
        "canonical_paid_admission_dry_run_passed": True,
        "provider_mutations_performed": 0,
        "automatic_retry_authorized": False,
        "records": {
            name: _evidence_record(path, evidence_root=root)
            for name, path in paths.items()
        },
        "proof_boundaries": {
            "gpu_runtime_executed": False,
            "gaussian_ownership_qualified": False,
            "new_paid_authority_required": True,
        },
        "raw_secret_values_recorded": False,
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    ensure_dir(output.parent)
    write_json(output, receipt)
    return receipt


def materialize_gaussian_excision_task_abstention(
    *,
    scene_freeze_path: str | Path,
    task_freeze_path: str | Path,
    excision_freeze_path: str | Path,
    attempt_receipt_path: str | Path,
    recovery_readiness_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Convert a terminal contribution blocker into the standard task abstention."""

    scene_freeze = _read_canonical(
        Path(scene_freeze_path).expanduser().resolve(),
        field="scene_freeze_digest",
        code="gaussian_excision_scene_freeze_invalid",
    )
    task_freeze = _read_canonical(
        Path(task_freeze_path).expanduser().resolve(),
        field="task_freeze_digest",
        code="gaussian_excision_task_freeze_invalid",
    )
    excision_freeze = _read_canonical(
        Path(excision_freeze_path).expanduser().resolve(),
        field="freeze_digest",
        code="gaussian_excision_freeze_invalid",
    )
    attempt = _read_canonical(
        Path(attempt_receipt_path).expanduser().resolve(),
        field="receipt_digest",
        code="gaussian_excision_attempt_receipt_invalid",
    )
    readiness = _read_canonical(
        Path(recovery_readiness_path).expanduser().resolve(),
        field="receipt_digest",
        code="gaussian_excision_recovery_readiness_invalid",
    )
    excision_scene = excision_freeze.get("scene") or {}
    blockers = attempt.get("execution_blockers")
    if (
        attempt.get("status") != "sealed_blocked_attempt"
        or attempt.get("freeze_digest") != readiness.get("freeze_digest")
        or attempt.get("freeze_digest") != excision_freeze.get("freeze_digest")
        or task_freeze.get("scene_freeze_digest")
        != scene_freeze.get("scene_freeze_digest")
        or str(excision_scene.get("task_id") or "")
        != str(task_freeze.get("task_id") or "")
        or str(excision_scene.get("publisher_scene_id") or "")
        != str(scene_freeze.get("selected_scene_id") or "")
        or not isinstance(blockers, list)
        or not blockers
        or readiness.get("status") != "ready_for_new_authority_not_executed"
        or readiness.get("proof_boundaries", {}).get("new_paid_authority_required")
        is not True
        or not str(task_freeze.get("task_id") or "")
    ):
        raise ValueError("gaussian_excision_task_abstention_join_invalid")
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError("gaussian_excision_task_abstention_exists")
    receipt = {
        "schema_version": "adp_task_evaluation_run_abstention.v1",
        "status": "typed_evidence_backed_abstention",
        "scene_id": str(scene_freeze["selected_scene_id"]),
        "task_id": str(task_freeze["task_id"]),
        "task_freeze_digest": task_freeze.get("task_freeze_digest"),
        "gaussian_excision_freeze_digest": excision_freeze.get("freeze_digest"),
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "smallest_missing_capability": str(blockers[0]),
        "all_terminal_blockers": list(blockers),
        "gaussian_excision_attempt_receipt_digest": attempt["receipt_digest"],
        "recovery_readiness_receipt_digest": readiness["receipt_digest"],
        "controls_executed": False,
        "learned_candidate_episodes_executed": False,
        "episode_media_exists": False,
        "comparison_exists": False,
        "automatic_paid_retry_executed": False,
        "next_action": (
            "obtain new explicit one-attempt zero-retry authority for the sealed "
            "repaired contribution bundle"
        ),
        "claim_ceiling": (
            "public_dataset_simulator_construction_rehearsal_only; no physical, "
            "deployment, customer_value, or learned_policy claim"
        ),
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    ensure_dir(output.parent)
    write_json(output, receipt)
    return receipt


def materialize_gaussian_excision_provider_closeout(
    *,
    evidence_root: str | Path,
    attempt_receipt_paths: Sequence[str | Path],
    provider_inventory_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Prove lane-owned provider-zero without claiming unrelated fleet zero."""

    root = Path(evidence_root).expanduser().resolve()
    attempts = [
        _read_canonical(
            Path(path).expanduser().resolve(),
            field="receipt_digest",
            code="gaussian_excision_attempt_receipt_invalid",
        )
        for path in attempt_receipt_paths
    ]
    inventory_path = Path(provider_inventory_path).expanduser().resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    owned_ids = [str(row.get("instance_id")) for row in attempts]
    live_rows = [
        row
        for row in inventory.get("instances", [])
        if isinstance(row, Mapping) and row.get("live") is True
    ]
    if (
        len(attempts) != len(set(owned_ids))
        or not attempts
        or any(row.get("provider_absence_confirmed") is not True for row in attempts)
        or any(str(row.get("id")) in owned_ids for row in live_rows)
    ):
        raise ValueError("gaussian_excision_provider_closeout_invalid")
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError("gaussian_excision_provider_closeout_exists")
    receipt = {
        "schema_version": "adp_gaussian_excision_provider_closeout.v1",
        "status": "lane_owned_provider_zero",
        "owned_instance_ids": [int(value) for value in owned_ids],
        "attempt_receipt_digests": [row["receipt_digest"] for row in attempts],
        "combined_estimated_cost_usd": round(
            sum(float(row.get("estimated_cost_usd") or 0.0) for row in attempts), 6
        ),
        "continuing_lane_owned_spend": False,
        "external_live_instances": [
            {
                "instance_id": row.get("id"),
                "name": row.get("name"),
                "state": row.get("state"),
                "cost_per_hr_usd": row.get("cost_per_hr_usd"),
                "charged_to_this_lane": False,
                "provider_mutation_performed": False,
            }
            for row in live_rows
        ],
        "global_provider_zero_claimed": not live_rows,
        "provider_inventory": _evidence_record(
            inventory_path, evidence_root=root
        ),
        "raw_secret_values_recorded": False,
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    ensure_dir(output.parent)
    write_json(output, receipt)
    return receipt


def _validate_authority(
    authority: Mapping[str, Any], *, freeze: Mapping[str, Any]
) -> None:
    required_true = (
        "private_scene_derived_standard_splat_upload_authorized",
        "paid_compute_authorized",
        "provider_zero_required_before_and_after",
        "teardown_required",
    )
    required_false = (
        "raw_interiorgs_downloaded_bytes_upload_authorized",
        "public_disclosure_authorized",
        "model_training_authorized",
        "automatic_paid_retry_authorized",
    )
    if (
        authority.get("schema_version") != AUTHORITY_SCHEMA
        or authority.get("purpose") != "released_code_gaussian_ownership_audit"
        or authority.get("publisher_scene_id")
        != str((freeze.get("scene") or {}).get("publisher_scene_id"))
        or authority.get("target_instance_id")
        != str((freeze.get("scene") or {}).get("target_instance_id"))
        or authority.get("freeze_digest") != freeze.get("freeze_digest")
        or any(authority.get(key) is not True for key in required_true)
        or any(authority.get(key) is not False for key in required_false)
        or authority.get("retention_policy") != "bounded_to_goal_then_provider_zero"
        or authority.get("hard_attempt_spend_cap_usd") != 1.5
        or authority.get("maximum_single_resource_ttl_seconds") != 3600
        or authority.get("maximum_paid_attempts") != 1
        or authority.get("maximum_automatic_retries") != 0
    ):
        raise ValueError("gaussian_excision_execution_authority_invalid")


def build_gaussian_excision_vast_bundle(
    *,
    repo_root: str | Path,
    flashsplat_root: str | Path,
    freeze_path: str | Path,
    source_standard_splat_path: str | Path,
    camera_contract_path: str | Path,
    execution_authority_path: str | Path,
    dependency_wheelhouse_path: str | Path,
    dependency_manifest_path: str | Path,
    job_dir: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the immutable first-stage contribution packet without GPU mutation."""

    repo = Path(repo_root).expanduser().resolve()
    source = Path(flashsplat_root).expanduser().resolve()
    freeze_file = Path(freeze_path).expanduser().resolve()
    splat = Path(source_standard_splat_path).expanduser().resolve()
    cameras = Path(camera_contract_path).expanduser().resolve()
    authority_file = Path(execution_authority_path).expanduser().resolve()
    dependency_wheelhouse = Path(dependency_wheelhouse_path).expanduser().resolve()
    dependency_manifest_file = Path(dependency_manifest_path).expanduser().resolve()
    destination = Path(job_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("gaussian_excision_bundle_job_dir_not_empty")
    freeze = _read_canonical(
        freeze_file, field="freeze_digest", code="gaussian_excision_freeze_invalid"
    )
    authority = _read_canonical(
        authority_file,
        field="authorization_digest",
        code="gaussian_excision_execution_authority_invalid",
    )
    if freeze.get("schema_version") != FREEZE_SCHEMA:
        raise ValueError("gaussian_excision_freeze_invalid")
    _validate_authority(authority, freeze=freeze)
    if (
        not splat.is_file()
        or splat.is_symlink()
        or _sha256(splat) != (freeze.get("source_standard_splat") or {}).get("sha256")
        or not cameras.is_file()
        or cameras.is_symlink()
        or _sha256(cameras) != (freeze.get("camera_contract") or {}).get("sha256")
    ):
        raise ValueError("gaussian_excision_bound_input_invalid")
    blueprint_commit = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "status", "--short"):
        raise ValueError("gaussian_excision_blueprint_source_not_clean")
    released_source = _source_identity(source)
    lane_identity = gaussian_excision_lane_identity(freeze)
    dependency_manifest = _verified_dependency_wheelhouse(
        wheelhouse_path=dependency_wheelhouse,
        manifest_path=dependency_manifest_file,
    )

    runtime = destination / "provider_runtime"
    ensure_dir(runtime / "input")
    ensure_dir(runtime / "freeze")
    shutil.copy2(splat, runtime / "input" / "scene_standard.ply")
    shutil.copy2(cameras, runtime / "input" / "cameras.v1.json")
    shutil.copy2(freeze_file, runtime / "freeze" / freeze_file.name)
    shutil.copytree(freeze_file.parent / "masks", runtime / "freeze" / "masks")
    shutil.copy2(authority_file, runtime / "execution_authority.json")
    shutil.copytree(dependency_wheelhouse, runtime / "dependency_wheelhouse")
    shutil.copy2(
        dependency_manifest_file, runtime / "dependency_wheelhouse_manifest.json"
    )
    scripts = repo / "scripts"
    for name in (
        "run_adp_gaussian_excision_provider_runtime.sh",
        "adp_gaussian_excision_provider_runner.py",
    ):
        shutil.copy2(scripts / name, runtime / name)
    shutil.copy2(
        repo / "src/blueprint_pipeline/provider_archive.py",
        runtime / "provider_archive.py",
    )
    entrypoint = runtime / "run_adp_gaussian_excision_provider_runtime.sh"
    entrypoint.chmod(entrypoint.stat().st_mode | stat.S_IXUSR)
    source_archive = runtime / "flashsplat_source.zip"
    _write_source_archive(source, source_archive)
    blockers = provider_runtime_contract_blockers(
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
        entrypoint_text=entrypoint.read_text(encoding="utf-8"),
        runner_text=(runtime / "adp_gaussian_excision_provider_runner.py").read_text(
            encoding="utf-8"
        ),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or utc_now_iso(),
        "status": "ready" if not blockers else "blocked",
        "provider_bundle_kind": PROVIDER_BUNDLE_KIND,
        "container_image": DEFAULT_IMAGE,
        "blueprint_commit": blueprint_commit,
        "released_code": released_source,
        "source_archive_sha256": _sha256(source_archive),
        "freeze_digest": freeze["freeze_digest"],
        "execution_authority_digest": authority["authorization_digest"],
        "hard_cap_usd": authority["hard_attempt_spend_cap_usd"],
        "hard_ttl_seconds": authority["maximum_single_resource_ttl_seconds"],
        "maximum_paid_attempts": authority["maximum_paid_attempts"],
        "standard_splat_sha256": _sha256(runtime / "input" / "scene_standard.ply"),
        "camera_contract_sha256": _sha256(runtime / "input" / "cameras.v1.json"),
        "dependency_wheelhouse_manifest_digest": dependency_manifest[
            "manifest_digest"
        ],
        "provider_network_dependency_install_required": False,
        "calibration_camera_ids": freeze["camera_split"]["calibration_camera_ids"],
        "heldout_camera_ids": freeze["camera_split"]["heldout_camera_ids"],
        "deterministic_repetitions": freeze["policy"]["deterministic_repetitions"],
        "raw_interiorgs_downloaded_bytes_included": False,
        "private_scene_derived_standard_splat_included": True,
        "automatic_paid_retry_allowed": False,
        "provider_zero_required_after_return": True,
        "expected_output_filename": "adp009b_gaussian_excision_result.json",
        "blockers": blockers,
        "raw_secret_values_recorded": False,
        **lane_identity,
    }
    write_json(runtime / "adp_gaussian_excision_provider_manifest.json", manifest)
    bundle = destination / "adp_gaussian_excision_provider_runtime_bundle.zip"
    _deterministic_zip(runtime, bundle)
    rehearsal = rehearse_provider_bundle_entrypoint(
        bundle_path=bundle,
        entrypoint_relative_path="run_adp_gaussian_excision_provider_runtime.sh",
        evidence_path=destination / "adp_gaussian_excision_exact_bundle_rehearsal.json",
    )
    receipt = {
        **manifest,
        "bundle_path": str(bundle),
        "bundle_sha256": _sha256(bundle),
        "bundle_size_bytes": bundle.stat().st_size,
        "exact_bundle_entrypoint_rehearsal": rehearsal,
    }
    write_json(destination / "adp_gaussian_excision_bundle_receipt.json", receipt)
    return receipt


def _remaining_minutes(
    *, job: Path, hard_cap_usd: float, hard_ttl_seconds: int, max_hourly_rate_usd: float
) -> int:
    ledger_path = job / "gaussian_excision_vast_session_budget.json"
    if ledger_path.is_file():
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("gaussian_excision_budget_ledger_invalid") from exc
    else:
        ledger = {}
    attempts = [
        row for row in ledger.get("attempts", []) if isinstance(row, Mapping)
    ]
    prior_seconds = sum(attempt_runtime_seconds(row) for row in attempts)
    prior_cost = sum(attempt_estimated_cost(row) for row in attempts)
    return max(
        0,
        min(
            math.floor(max(0.0, hard_ttl_seconds - prior_seconds) / 60.0),
            math.floor(
                max(0.0, hard_cap_usd - prior_cost)
                * 60.0
                / max_hourly_rate_usd
            ),
        ),
    )


def _extract_provider_output(path: Path, destination: Path) -> dict[str, Any]:
    result_name = "adp009b_gaussian_excision_result.json"
    result_path = destination / result_name
    blockers: list[str] = []
    if not path.is_file():
        return {
            "status": "blocked",
            "execution": {},
            "result_path": str(result_path),
            "blockers": ["gaussian_excision_provider_output_zip_missing"],
        }
    ensure_dir(destination)
    root = destination.resolve()
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if target != root and root not in target.parents:
                    blockers.append("gaussian_excision_provider_output_zip_path_traversal")
            if not blockers:
                archive.extractall(destination)
    except (OSError, zipfile.BadZipFile):
        blockers.append("gaussian_excision_provider_output_zip_invalid")
    try:
        execution = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.is_file()
            else {}
        )
    except (OSError, json.JSONDecodeError):
        execution = {}
    if not isinstance(execution, dict) or not execution:
        execution = {}
        blockers.append("gaussian_excision_provider_result_missing")
    return {
        "status": "completed" if not blockers else "blocked",
        "execution": execution,
        "result_path": str(result_path),
        "blockers": sorted(set(blockers)),
    }


@contextmanager
def _authority_environment():
    names = (*_VAST_MUTATION_ENV, _VAST_SINGLE_ATTEMPT_ENV)
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in _VAST_MUTATION_ENV:
            os.environ[name] = "1"
        os.environ[_VAST_SINGLE_ATTEMPT_ENV] = "0"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_gaussian_excision_vast(
    *,
    job_dir: str | Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    prepared_bundle: Mapping[str, Any],
    paid_attempt_authority: Mapping[str, Any] | None = None,
    previous_attempt_receipt: Mapping[str, Any] | None = None,
    max_hourly_rate_usd: float = 0.60,
    hard_cap_usd: float = 1.50,
    hard_ttl_seconds: int = 3600,
    public_image: str = DEFAULT_IMAGE,
    allowed_active_instance_ids: Sequence[int] = (),
    machine_avoidlist_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute exactly one contribution attempt with watchdog and provider zero."""

    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    bundle = dict(prepared_bundle)
    bundle_path = Path(str(bundle.get("bundle_path") or "")).resolve()
    if public_image != DEFAULT_IMAGE:
        raise ValueError("gaussian_excision_container_image_not_frozen")
    if (
        bundle.get("status") != "ready"
        or bundle.get("provider_bundle_kind") != PROVIDER_BUNDLE_KIND
        or bundle.get("provider_network_dependency_install_required") is not False
        or not str(bundle.get("dependency_wheelhouse_manifest_digest") or "").startswith(
            "sha256:"
        )
        or not bundle_path.is_file()
        or _sha256(bundle_path) != bundle.get("bundle_sha256")
        or provider_bundle_rehearsal_blockers(
            bundle.get("exact_bundle_entrypoint_rehearsal"),
            bundle_sha256=str(bundle.get("bundle_sha256") or ""),
            entrypoint_relative_path="run_adp_gaussian_excision_provider_runtime.sh",
        )
    ):
        raise ValueError("gaussian_excision_prepared_bundle_binding_invalid")
    if not execute:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "dry_run_ready",
            "bundle": bundle,
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "blockers": [],
        }
        write_json(job / "gaussian_excision_vast_result.json", result)
        return result
    if paid_resource_admission_grant is None:
        raise ValueError("gaussian_excision_paid_resource_admission_grant_missing")
    if paid_attempt_authority is None:
        raise ValueError("gaussian_excision_paid_attempt_authority_missing")
    validated_attempt_authority = validate_gaussian_excision_paid_attempt_authority(
        paid_attempt_authority,
        prepared_bundle=bundle,
        previous_attempt_receipt=previous_attempt_receipt,
        allowed_active_instance_ids=allowed_active_instance_ids,
    )
    remaining_minutes = _remaining_minutes(
        job=job,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        max_hourly_rate_usd=max_hourly_rate_usd,
    )
    if remaining_minutes < 30:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "blockers": ["gaussian_excision_budget_below_minimum_live_window"],
        }
    authorization_consumption = consume_gaussian_excision_paid_attempt_authority_once(
        validated_attempt_authority,
        blueprint_commit=str(bundle.get("blueprint_commit") or ""),
    )
    if authorization_consumption.get("status") != "consumed":
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "authorization_consumption": authorization_consumption,
            "blockers": list(authorization_consumption.get("blockers") or []),
        }
    paid_attempt_binding = {
        "paid_attempt_ordinal": validated_attempt_authority["paid_attempt_ordinal"],
        "attempt_authority_digest": validated_attempt_authority[
            "authorization_digest"
        ],
        "previous_attempt_receipt_digest": validated_attempt_authority.get(
            "previous_attempt_receipt_digest"
        ),
        "authorization_consumption": authorization_consumption,
    }
    staging_dir = job / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir,
        bundle_path=str(bundle_path),
        key_prefix=str(bundle.get("object_store_key_prefix") or DEFAULT_KEY_PREFIX),
        expiration_seconds=max(hard_ttl_seconds + 1800, 7200),
    )
    if staging.get("status") != "completed":
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            **paid_attempt_binding,
            "blockers": staging.get("blockers")
            or ["gaussian_excision_object_store_staging_blocked"],
        }
    provider_run = job / "vast_provider_run"
    output_zip = provider_run / "vast_provider_runtime_output.zip"
    watchdog_handoff, watchdog_handle = arm_independent_vast_watchdog(
        job_dir=job,
        max_live_minutes=remaining_minutes,
        generated_at=utc_now_iso(),
        allowed_active_instance_ids=allowed_active_instance_ids,
    )
    if watchdog_handle is None:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            **paid_attempt_binding,
            "all_staged_objects_absent": cleanup.get("all_objects_absent"),
            "independent_watchdog": watchdog_handoff,
            "blockers": ["gaussian_excision_independent_watchdog_not_armed"],
        }
    adapter: dict[str, Any] = {}
    try:
        with _authority_environment():
            adapter = run_vast_provider_adapter(
                job_dir=provider_run,
                mode="live-startup-probe",
                allow_vast_api_call=True,
                allow_instance_launch=True,
                max_hourly_rate=max_hourly_rate_usd,
                target_spend_usd=hard_cap_usd,
                hard_cap_usd=hard_cap_usd,
                max_live_minutes=remaining_minutes,
                session_max_live_minutes=hard_ttl_seconds // 60,
                public_image=public_image,
                isaac_image=public_image,
                ngc_image_login_mode="never",
                provider_bundle=bundle_path,
                provider_bundle_url=(staging_dir / "provider_bundle_url.txt")
                .read_text(encoding="utf-8")
                .strip(),
                provider_output_put_url=(
                    staging_dir / "provider_output_put_url.txt"
                )
                .read_text(encoding="utf-8")
                .strip(),
                provider_output_get_url=(
                    staging_dir / "provider_output_get_url.txt"
                )
                .read_text(encoding="utf-8")
                .strip(),
                provider_runtime_output_zip=output_zip,
                enable_isaac_smoke=False,
                enable_blueprint_bundle=True,
                provider_bundle_kind=PROVIDER_BUNDLE_KIND,
                vast_launch_mode="ssh_direct",
                allow_cold_isaac_image_pull=False,
                disk_gb=64,
                min_gpu_ram_mb=16_000,
                poll_interval_seconds=10,
                startup_timeout_seconds=min(3600, remaining_minutes * 60),
                heartbeat_no_progress_seconds=1200,
                session_budget_ledger_path=job
                / "gaussian_excision_vast_session_budget.json",
                verify_staging_urls=True,
                require_known_supported_isaac_driver=False,
                preferred_gpu_keywords=("RTX 4090", "L40S", "RTX A6000", "A100"),
                prefer_isaac_rt=False,
                allowed_active_instance_ids=allowed_active_instance_ids,
                machine_avoidlist_path=machine_avoidlist_path,
                vast_launch_lock_file=job.parent
                / "gaussian_excision_paid_launch.lock",
                instance_label_prefix=str(
                    bundle.get("instance_label_prefix")
                    or "blueprint-adp-gaussian-excision-"
                ),
                started_instance_id_path=watchdog_handle.started_instance_id_path,
                forward_hf_token=False,
                paid_resource_admission_grant=paid_resource_admission_grant,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        adapter = {
            "status": "blocked",
            "blockers": [f"gaussian_excision_vast_adapter_failed:{type(exc).__name__}"],
            "raw_secret_values_recorded": False,
        }
    finally:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    extracted_root = job / "immutable_execution"
    extracted = _extract_provider_output(output_zip, extracted_root)
    execution = dict(extracted.get("execution") or {})
    teardown_path = provider_run / "vast_teardown_manifest.json"
    try:
        teardown = (
            json.loads(teardown_path.read_text(encoding="utf-8"))
            if teardown_path.is_file()
            else {}
        )
    except (OSError, json.JSONDecodeError):
        teardown = {}
    instance_ids = [
        int(value)
        for value in (
            teardown.get("vast_instance_ids")
            or adapter.get("vast_instance_ids")
            or []
        )
        if isinstance(value, int) and value > 0
    ]
    watchdog_close = close_independent_vast_watchdog(
        job_dir=job,
        handle=watchdog_handle,
        instance_ids=instance_ids,
        provider_teardown_completed=teardown.get("continuing_spend_from_this_run")
        is False,
        provider_allocation_impossible=(
            not instance_ids and adapter.get("provider_create_attempted") is not True
        ),
    )
    blockers = list(adapter.get("blockers") or []) + list(
        extracted.get("blockers") or []
    )
    if execution.get("status") != "completed":
        blockers.extend(
            execution.get("blockers") or ["gaussian_excision_execution_not_completed"]
        )
    elif (
        execution.get("released_code_executed") is not True
        or execution.get("heldout_cameras_accessed_for_classification") is not False
        or execution.get("provider_zero_required_after_return") is not True
        or execution.get("depth_anything_3_used") is not False
        or not isinstance(execution.get("contribution_manifest"), Mapping)
    ):
        blockers.append("gaussian_excision_execution_contract_invalid")
    if teardown.get("continuing_spend_from_this_run") is not False:
        blockers.append("gaussian_excision_vast_provider_zero_not_proven")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("gaussian_excision_object_store_provider_zero_not_proven")
    if watchdog_close.get("status") not in {
        "provider_terminal",
        "cancelled_no_allocation",
    }:
        blockers.append("gaussian_excision_independent_watchdog_not_closed")
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed" if not blockers else "blocked",
        "bundle_sha256": bundle["bundle_sha256"],
        "execution_result_path": extracted.get("result_path"),
        "adapter_result_path": str(
            provider_run / "vast_provider_adapter_result.json"
        ),
        "teardown_manifest_path": str(teardown_path),
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "hard_cap_usd": hard_cap_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": 0,
        **paid_attempt_binding,
        "continuing_spend_from_this_run": teardown.get(
            "continuing_spend_from_this_run"
        ),
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
        "independent_watchdog": watchdog_close,
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "raw_secret_values_recorded": False,
    }
    write_json(job / "gaussian_excision_vast_result.json", result)
    return result


__all__: Sequence[str] = (
    "AUTHORITY_SCHEMA",
    "DEFAULT_IMAGE",
    "PROBE_KIND",
    "PROVIDER_BUNDLE_KIND",
    "PAID_ATTEMPT_AUTHORITY_SCHEMA",
    "SOURCE_COMMIT",
    "SOURCE_TREE",
    "build_gaussian_excision_vast_bundle",
    "consume_gaussian_excision_paid_attempt_authority_once",
    "gaussian_excision_lane_identity",
    "run_gaussian_excision_vast",
    "validate_gaussian_excision_paid_attempt_authority",
)
