"""Build a frozen Vast packet for released-code Gaussian ownership evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .provider_runtime_bundle_contract import provider_runtime_contract_blockers
from .public_scene_gaussian_excision_audit import FREEZE_SCHEMA


PROBE_KIND = "adp-gaussian-excision"
PROVIDER_BUNDLE_KIND = "adp_gaussian_excision"
SCHEMA_VERSION = "adp009b_gaussian_excision_vast_bundle.v1"
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
DEFAULT_IMAGE = (
    "docker.io/nvidia/cuda@"
    "sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749"
)
EXPECTED_SUBMODULES = {
    RASTERIZER_PATH: RASTERIZER_COMMIT,
    DIFF_RASTERIZER_PATH: DIFF_RASTERIZER_COMMIT,
    GLM_PATH: GLM_COMMIT,
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

    runtime = destination / "provider_runtime"
    ensure_dir(runtime / "input")
    ensure_dir(runtime / "freeze")
    shutil.copy2(splat, runtime / "input" / "scene_standard.ply")
    shutil.copy2(cameras, runtime / "input" / "cameras.v1.json")
    shutil.copy2(freeze_file, runtime / "freeze" / freeze_file.name)
    shutil.copytree(freeze_file.parent / "masks", runtime / "freeze" / "masks")
    shutil.copy2(authority_file, runtime / "execution_authority.json")
    scripts = repo / "scripts"
    for name in (
        "run_adp_gaussian_excision_provider_runtime.sh",
        "adp_gaussian_excision_provider_runner.py",
    ):
        shutil.copy2(scripts / name, runtime / name)
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
        "standard_splat_sha256": _sha256(runtime / "input" / "scene_standard.ply"),
        "camera_contract_sha256": _sha256(runtime / "input" / "cameras.v1.json"),
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
    }
    write_json(runtime / "adp_gaussian_excision_provider_manifest.json", manifest)
    bundle = destination / "adp_gaussian_excision_provider_runtime_bundle.zip"
    _deterministic_zip(runtime, bundle)
    receipt = {
        **manifest,
        "bundle_path": str(bundle),
        "bundle_sha256": _sha256(bundle),
        "bundle_size_bytes": bundle.stat().st_size,
    }
    write_json(destination / "adp_gaussian_excision_bundle_receipt.json", receipt)
    return receipt


__all__: Sequence[str] = (
    "AUTHORITY_SCHEMA",
    "DEFAULT_IMAGE",
    "PROBE_KIND",
    "PROVIDER_BUNDLE_KIND",
    "SOURCE_COMMIT",
    "SOURCE_TREE",
    "build_gaussian_excision_vast_bundle",
)
