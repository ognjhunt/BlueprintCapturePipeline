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


PROBE_KIND = "adp-gaussian-excision"
PROVIDER_BUNDLE_KIND = "adp_gaussian_excision"
SCHEMA_VERSION = "adp009b_gaussian_excision_vast_bundle.v1"
RESULT_SCHEMA_VERSION = "adp009b_gaussian_excision_vast_run.v1"
AUTHORITY_SCHEMA = "public_scene_gaussian_excision_execution_authority.v1"
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
    "SOURCE_COMMIT",
    "SOURCE_TREE",
    "build_gaussian_excision_vast_bundle",
    "gaussian_excision_lane_identity",
    "run_gaussian_excision_vast",
)
