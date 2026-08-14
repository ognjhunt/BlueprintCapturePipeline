#!/usr/bin/env python3
"""Publish the exact SAM 3.1 source-track worker image from a clean checkout.

This command does not allocate a builder and does not read registry credentials.
It expects Docker Buildx and registry authentication to be configured on the
already-provisioned host.  A mutable tag is accepted only as the publication
target; the retained result is the registry-confirmed immutable digest ref.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "semantic_sam31_runtime_image_build_receipt.v1"
PLATFORM = "linux/amd64"
OFFICIAL_CODE_REVISION = "96914d2425f90a64f45ca977c2b5165418099543"
DOCKERFILE = Path("deploy/docker/sam31_source_tracks/Dockerfile")
ROOT_INPUTS = (Path("pyproject.toml"), Path("README.md"), Path("LICENSE"))
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IMAGE_TAG = re.compile(
    r"^(?:ghcr\.io|docker\.io)/[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*"
    r":[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$"
)
Runner = Callable[..., subprocess.CompletedProcess[str]]


class Sam31ImagePublicationError(ValueError):
    """Stable fail-closed publication error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any], *, field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _run(
    runner: Runner,
    argv: Sequence[str],
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return runner(
        list(argv),
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )


def _require_success(
    result: subprocess.CompletedProcess[str], *, code: str
) -> subprocess.CompletedProcess[str]:
    if result.returncode != 0:
        raise Sam31ImagePublicationError(code)
    return result


def _validate_image_tag(image_ref: str) -> None:
    if not _IMAGE_TAG.fullmatch(image_ref):
        raise Sam31ImagePublicationError("sam31_image_ref_not_versioned_registry_tag")
    tag = image_ref.rsplit(":", 1)[1].lower()
    if tag in {"latest", "local", "dev", "test"}:
        raise Sam31ImagePublicationError("sam31_image_ref_unstable_tag_forbidden")


def _tracked_context_paths(repo_root: Path, runner: Runner) -> list[Path]:
    result = _require_success(
        _run(
            runner,
            [
                "git",
                "ls-files",
                "-z",
                "--",
                *(str(path) for path in (*ROOT_INPUTS, DOCKERFILE)),
                "src",
            ],
            cwd=repo_root,
        ),
        code="sam31_context_git_inventory_failed",
    )
    paths = [Path(value) for value in result.stdout.split("\0") if value]
    required = {*ROOT_INPUTS, DOCKERFILE}
    if not required.issubset(paths) or not any(path.parts[:1] == ("src",) for path in paths):
        raise Sam31ImagePublicationError("sam31_context_required_inputs_missing")
    allowed = required | {path for path in paths if path.parts[:1] == ("src",)}
    if set(paths) != allowed or len(paths) != len(set(paths)):
        raise Sam31ImagePublicationError("sam31_context_inventory_invalid")
    return sorted(paths, key=lambda path: path.as_posix())


def _materialize_context(
    *, repo_root: Path, context_root: Path, paths: Sequence[Path]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for relative in paths:
        source = repo_root / relative
        if source.is_symlink() or not source.is_file():
            raise Sam31ImagePublicationError("sam31_context_input_missing_or_unsafe")
        destination = context_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        rows.append(
            {
                "path": relative.as_posix(),
                "size_bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": "sam31_source_tracks_build_context.v1",
        "files": rows,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = _canonical_digest(manifest, field="manifest_digest")
    return manifest


def _metadata_digest(payload: Mapping[str, Any]) -> str:
    descriptor = payload.get("containerimage.descriptor")
    descriptor = descriptor if isinstance(descriptor, Mapping) else {}
    value = payload.get("containerimage.digest") or descriptor.get("digest")
    return str(value or "")


def _registry_digest(stdout: str) -> str:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise Sam31ImagePublicationError("sam31_registry_inspection_invalid") from exc
    if not isinstance(payload, Mapping):
        raise Sam31ImagePublicationError("sam31_registry_inspection_invalid")
    manifest = payload.get("manifest") or payload.get("Manifest")
    manifest = manifest if isinstance(manifest, Mapping) else {}
    return str(manifest.get("digest") or manifest.get("Digest") or "")


def publish_sam31_source_tracks_image(
    *,
    repo_root: str | Path,
    source_commit: str,
    image_ref: str,
    output_dir: str | Path,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Build, push, and retain the immutable registry identity."""

    root = Path(repo_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not root.is_dir() or not _COMMIT.fullmatch(source_commit):
        raise Sam31ImagePublicationError("sam31_source_identity_invalid")
    _validate_image_tag(image_ref)
    if output.exists() or output.is_symlink():
        raise Sam31ImagePublicationError("sam31_publication_output_exists")

    head = _require_success(
        _run(runner, ["git", "rev-parse", "HEAD"], cwd=root),
        code="sam31_source_head_unavailable",
    ).stdout.strip()
    if head != source_commit:
        raise Sam31ImagePublicationError("sam31_source_commit_mismatch")
    status = _require_success(
        _run(
            runner,
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
        ),
        code="sam31_source_status_unavailable",
    )
    if status.stdout.strip():
        raise Sam31ImagePublicationError("sam31_source_checkout_not_clean")

    context_paths = _tracked_context_paths(root, runner)
    output.mkdir(parents=True, mode=0o700)
    metadata_path = output / "buildx_metadata.json"
    log_path = output / "buildx.log"
    context_manifest_path = output / "context_manifest.json"
    receipt_path = output / "publication_receipt.json"

    with tempfile.TemporaryDirectory(prefix="blueprint-sam31-image-context-") as temporary:
        context_root = Path(temporary)
        context_manifest = _materialize_context(
            repo_root=root, context_root=context_root, paths=context_paths
        )
        context_manifest_path.write_text(
            json.dumps(context_manifest, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        preflight_log: list[str] = []
        for command, code in (
            (["docker", "info"], "sam31_docker_daemon_unavailable"),
            (["docker", "buildx", "version"], "sam31_docker_buildx_unavailable"),
        ):
            observed = _require_success(_run(runner, command, cwd=root), code=code)
            preflight_log.extend((observed.stdout, observed.stderr))
        build = _run(
            runner,
            [
                "docker",
                "buildx",
                "build",
                "--platform",
                PLATFORM,
                "--progress",
                "plain",
                "--metadata-file",
                str(metadata_path),
                "--attest",
                "type=sbom",
                "--attest",
                "type=provenance,mode=max",
                "--label",
                f"org.opencontainers.image.revision={source_commit}",
                "-f",
                str(context_root / DOCKERFILE),
                "-t",
                image_ref,
                "--push",
                str(context_root),
            ],
            cwd=root,
        )
        log_path.write_text(
            "".join([*preflight_log, build.stdout, build.stderr]), encoding="utf-8"
        )
        _require_success(build, code="sam31_buildx_push_failed")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Sam31ImagePublicationError("sam31_buildx_metadata_invalid") from exc
    if not isinstance(metadata, Mapping):
        raise Sam31ImagePublicationError("sam31_buildx_metadata_invalid")
    built_digest = _metadata_digest(metadata)
    if not _DIGEST.fullmatch(built_digest):
        raise Sam31ImagePublicationError("sam31_buildx_digest_missing")
    inspected = _require_success(
        _run(
            runner,
            ["docker", "buildx", "imagetools", "inspect", "--format", "{{json .}}", image_ref],
            cwd=root,
        ),
        code="sam31_registry_inspection_failed",
    )
    registry_digest = _registry_digest(inspected.stdout)
    if registry_digest != built_digest:
        raise Sam31ImagePublicationError("sam31_registry_digest_mismatch")
    repository = image_ref.rsplit(":", 1)[0]
    immutable_ref = f"{repository}@{registry_digest}"
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "published",
        "source_commit_sha": source_commit,
        "source_checkout_clean": True,
        "image_tag": image_ref,
        "resolved_digest_ref": immutable_ref,
        "runtime_image_identity": immutable_ref,
        "runtime_digest": registry_digest,
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "registry_api_digest_verified": True,
        "platform": PLATFORM,
        "dockerfile": next(
            row for row in context_manifest["files"] if row["path"] == DOCKERFILE.as_posix()
        ),
        "context_manifest": {
            "path": context_manifest_path.name,
            "size_bytes": context_manifest_path.stat().st_size,
            "sha256": _sha256(context_manifest_path),
            "manifest_digest": context_manifest["manifest_digest"],
        },
        "dockerfile_sha256": next(
            row["sha256"]
            for row in context_manifest["files"]
            if row["path"] == DOCKERFILE.as_posix()
        ),
        "source_tree_digest": context_manifest["manifest_digest"],
        "buildx_metadata": {
            "path": metadata_path.name,
            "size_bytes": metadata_path.stat().st_size,
            "sha256": _sha256(metadata_path),
        },
        "build_provenance_digest": _sha256(metadata_path),
        "build_log": {
            "path": log_path.name,
            "size_bytes": log_path.stat().st_size,
            "sha256": _sha256(log_path),
        },
        "sbom_attestation_requested": True,
        "provenance_attestation_requested": True,
        "registry_credentials_read_by_publisher": False,
        "raw_secret_values_recorded": False,
        "provider_allocation_performed": False,
        "publication_is_not_worker_runtime_qualification": True,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = _canonical_digest(receipt, field="receipt_digest")
    receipt_path.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = publish_sam31_source_tracks_image(
            repo_root=args.repo_root,
            source_commit=args.source_commit,
            image_ref=args.image_ref,
            output_dir=args.output_dir,
        )
    except (OSError, Sam31ImagePublicationError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "resolved_digest_ref": receipt["resolved_digest_ref"],
                "receipt_digest": receipt["receipt_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
