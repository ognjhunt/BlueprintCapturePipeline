"""Prepare the canonical CPU-build packet for the Isaac evaluation worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, write_json
from .g1_kitchen_bundle_compatibility import (
    CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
)


SCHEMA_VERSION = "isaac_worker_remote_build_packet.v1"
PACKET_KIND = "isaac_worker_image"
PACKET_DIRNAME = "isaac_worker_remote_build"
BUILD_SCRIPT_NAME = "remote_build_isaac_worker_image.sh"
RESULT_NAME = "isaac_worker_image_manifest_diagnostic.json"
DEFAULT_IMAGE_REF = "docker.io/nijelhunt/blueprint-isaac-eval-worker:20260729"
DEFAULT_BASE_IMAGE_REF = (
    "nvcr.io/nvidia/isaac-sim:6.0.0@sha256:"
    "68735a60b6c15c85e0dd0098570c6d2cc79e928f2d068ce2790aa43284ac165d"
)
DOCKERFILE_RELATIVE_PATH = Path("deploy/docker/robot_eval_worker/isaac/Dockerfile")
REQUIRED_CONTEXT_PATHS = (
    DOCKERFILE_RELATIVE_PATH,
    Path("pyproject.toml"),
    Path("README.md"),
    Path("LICENSE"),
    Path("src/blueprint_pipeline/isaac_worker_image_manifest.py"),
)
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST_REF = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versioned_image_ref(image_ref: str) -> bool:
    leaf = image_ref.rsplit("/", 1)[-1]
    name, separator, tag = leaf.rpartition(":")
    return bool(
        name
        and separator
        and _TAG.fullmatch(tag)
        and tag not in {"latest", "dev", "test", "local"}
        and "@" not in image_ref
        and not any(char.isspace() for char in image_ref)
    )


def _context_sources(root: Path) -> tuple[list[Path], bool]:
    paths = [root / relative for relative in REQUIRED_CONTEXT_PATHS]
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "-z", "--", "src/blueprint_pipeline"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        tracked = b""
    paths.extend(root / value.decode("utf-8") for value in tracked.split(b"\0") if value)
    return sorted(set(paths)), bool(tracked)


def render_remote_build_script(
    *,
    image_ref: str,
    base_image_ref: str,
    source_commit: str,
    dockerfile_sha256: str,
    context_manifest_sha256: str,
) -> str:
    """Return the exact remote script bound into the packet manifest."""

    return f"""#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
context_dir="$script_dir/context"
image_ref={json.dumps(image_ref)}
base_image_ref={json.dumps(base_image_ref)}
source_commit={json.dumps(source_commit)}
clean_patch_sha256={json.dumps(CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256)}
dockerfile_sha256={json.dumps(dockerfile_sha256)}
context_manifest_sha256={json.dumps(context_manifest_sha256)}
username_file="${{BLUEPRINT_DOCKER_USERNAME_FILE:-$HOME/.blueprint-secrets/docker_username}}"
password_file="${{BLUEPRINT_DOCKER_PASSWORD_FILE:-$HOME/.blueprint-secrets/docker_pat}}"
metadata="$script_dir/isaac_worker_build_metadata.json"
result="$script_dir/{RESULT_NAME}"

cleanup() {{
  docker logout >/dev/null 2>&1 || true
  rm -f "$username_file" "$password_file"
}}
trap cleanup EXIT

test "$(sha256sum "$context_dir/{DOCKERFILE_RELATIVE_PATH.as_posix()}" | awk '{{print $1}}')" = "$dockerfile_sha256"
test -f "$username_file"
test -f "$password_file"
docker login -u "$(cat "$username_file")" --password-stdin < "$password_file"
docker buildx build \
  --platform linux/amd64 \
  --progress plain \
  --metadata-file "$metadata" \
  --build-arg "ISAAC_SIM_BASE_IMAGE=$base_image_ref" \
  --build-arg "BLUEPRINT_SOURCE_COMMIT=$source_commit" \
  --build-arg "BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256=$clean_patch_sha256" \
  -f "$context_dir/{DOCKERFILE_RELATIVE_PATH.as_posix()}" \
  -t "$image_ref" \
  --push \
  "$context_dir"
digest="$(python3 - "$metadata" <<'PY'
import json,re,sys
payload=json.load(open(sys.argv[1],encoding="utf-8"))
value=str(payload.get("containerimage.digest") or "")
print(value)
raise SystemExit(0 if re.fullmatch(r"sha256:[0-9a-f]{{64}}",value) else 2)
PY
)"
resolved="$(python3 - "$image_ref" "$digest" <<'PY'
import sys
ref,digest=sys.argv[1:]
repository=ref.rsplit("/",1)[0]+"/"+ref.rsplit("/",1)[-1].rsplit(":",1)[0]
print(repository+"@"+digest)
PY
)"
docker buildx imagetools inspect "$resolved" >/dev/null
python3 "$context_dir/src/blueprint_pipeline/isaac_worker_image_manifest.py" \
  --image "$resolved" \
  --output "$result"
python3 - "$result" "$resolved" "$source_commit" "$clean_patch_sha256" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1],encoding="utf-8"))
identity=payload.get("worker_build_identity") or {{}}
checks=(
  payload.get("status")=="completed",
  payload.get("resolved_digest_ref")==sys.argv[2],
  payload.get("runnable_platform")=="linux/amd64",
  identity.get("status")=="verified",
  identity.get("source_commit")==sys.argv[3],
  identity.get("source_dirty_patch_sha256")==sys.argv[4],
  identity.get("worker_image_family")=="isaac-eval-worker",
  identity.get("isaac_sim_major_version")==6,
  payload.get("raw_secret_values_recorded") is False,
)
raise SystemExit(0 if all(checks) else 2)
PY
"""


def validate_isaac_worker_archive(packet: Mapping[str, Any]) -> list[str]:
    """Verify archive inventory, member hashes, and executable script binding."""

    blockers: list[str] = []
    declared_names = packet.get("archive_members")
    names = declared_names if isinstance(declared_names, list) else []
    declared_digests = packet.get("archive_member_sha256")
    digests = declared_digests if isinstance(declared_digests, Mapping) else {}
    required_names = {
        f"{PACKET_DIRNAME}/README.md",
        f"{PACKET_DIRNAME}/{BUILD_SCRIPT_NAME}",
        *(f"{PACKET_DIRNAME}/context/{path.as_posix()}" for path in REQUIRED_CONTEXT_PATHS),
    }
    if names != sorted(names) or len(names) != len(set(names)) or not required_names <= set(names):
        blockers.append("builder_isaac_archive_member_contract_invalid")
    if sorted(digests) != names or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in digests.values()
    ):
        blockers.append("builder_isaac_archive_digest_contract_invalid")
    manifest_digest = hashlib.sha256(
        json.dumps(dict(digests), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if packet.get("archive_member_manifest_sha256") != manifest_digest:
        blockers.append("builder_isaac_archive_member_manifest_mismatch")

    path = Path(str(packet.get("tarball_path") or "")).expanduser().resolve()
    payloads: dict[str, bytes] = {}
    if not path.is_file():
        blockers.append("builder_isaac_archive_missing")
    elif _sha256(path) != packet.get("tarball_sha256"):
        blockers.append("builder_isaac_archive_tarball_mismatch")
    else:
        try:
            with tarfile.open(path, "r:gz") as archive:
                members = archive.getmembers()
                observed_names = [member.name for member in members]
                if observed_names != names or any(
                    not member.isfile()
                    or Path(member.name).is_absolute()
                    or ".." in Path(member.name).parts
                    for member in members
                ):
                    blockers.append("builder_isaac_archive_inventory_mismatch")
                script = next(
                    (member for member in members if member.name == f"{PACKET_DIRNAME}/{BUILD_SCRIPT_NAME}"),
                    None,
                )
                if script is None or script.mode & 0o111 == 0:
                    blockers.append("builder_isaac_archive_script_not_executable")
                for member in members:
                    stream = archive.extractfile(member)
                    if stream is not None:
                        payloads[member.name] = stream.read()
        except (OSError, tarfile.TarError):
            blockers.append("builder_isaac_archive_unreadable")
    if any(
        hashlib.sha256(payload).hexdigest() != digests.get(name)
        for name, payload in payloads.items()
    ):
        blockers.append("builder_isaac_archive_member_digest_mismatch")
    dockerfile_name = f"{PACKET_DIRNAME}/context/{DOCKERFILE_RELATIVE_PATH.as_posix()}"
    if dockerfile_name in payloads and hashlib.sha256(
        payloads[dockerfile_name]
    ).hexdigest() != packet.get("dockerfile_sha256"):
        blockers.append("builder_isaac_archive_dockerfile_binding_mismatch")
    expected_script = render_remote_build_script(
        image_ref=str(packet.get("image_ref") or ""),
        base_image_ref=str(packet.get("base_image_ref") or ""),
        source_commit=str(packet.get("source_commit") or ""),
        dockerfile_sha256=str(packet.get("dockerfile_sha256") or ""),
        context_manifest_sha256=str(packet.get("context_manifest_sha256") or ""),
    ).encode()
    if payloads.get(f"{PACKET_DIRNAME}/{BUILD_SCRIPT_NAME}") != expected_script:
        blockers.append("builder_isaac_archive_script_binding_mismatch")
    return sorted(set(blockers))


def prepare_remote_build_packet(
    *,
    output_dir: str | Path,
    repo_root: str | Path,
    image_ref: str,
    base_image_ref: str,
    source_commit: str,
    source_worktree_dirty: bool,
    generated_at: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    packet = output / PACKET_DIRNAME
    context = packet / "context"
    blockers: list[str] = []
    if not _versioned_image_ref(image_ref):
        blockers.append("isaac_worker_image_ref_not_versioned")
    if not _DIGEST_REF.fullmatch(base_image_ref):
        blockers.append("isaac_worker_base_image_not_digest_pinned")
    if not _COMMIT.fullmatch(source_commit):
        blockers.append("isaac_worker_source_commit_invalid")
    try:
        actual_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        actual_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        actual_commit, actual_dirty = "", True
        blockers.append("isaac_worker_source_git_identity_unavailable")
    if actual_commit != source_commit:
        blockers.append("isaac_worker_source_commit_not_exact_head")
    if actual_dirty != source_worktree_dirty:
        blockers.append("isaac_worker_source_dirty_claim_mismatch")
    if source_worktree_dirty or actual_dirty:
        blockers.append("isaac_worker_packet_requires_clean_source_worktree")

    ensure_dir(context)
    sources, tracked_source_available = _context_sources(root)
    if not tracked_source_available:
        blockers.append("isaac_worker_tracked_source_inventory_unavailable")
    missing = [relative.as_posix() for relative in REQUIRED_CONTEXT_PATHS if not (root / relative).is_file()]
    blockers.extend(f"isaac_worker_context_file_missing:{name}" for name in missing)
    copied: list[Path] = []
    for source in sources:
        if not source.is_file() or source.is_symlink():
            continue
        destination = context / source.relative_to(root)
        ensure_dir(destination.parent)
        shutil.copy2(source, destination)
        copied.append(destination)
    dockerfile = context / DOCKERFILE_RELATIVE_PATH
    dockerfile_sha256 = _sha256(dockerfile) if dockerfile.is_file() else ""
    context_member_sha256 = {
        destination.relative_to(context).as_posix(): _sha256(destination) for destination in copied
    }
    context_manifest_sha256 = hashlib.sha256(
        json.dumps(context_member_sha256, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    script = packet / BUILD_SCRIPT_NAME
    script.write_text(
        render_remote_build_script(
            image_ref=image_ref,
            base_image_ref=base_image_ref,
            source_commit=source_commit,
            dockerfile_sha256=dockerfile_sha256,
            context_manifest_sha256=context_manifest_sha256,
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    readme = packet / "README.md"
    readme.write_text(
        "# Isaac worker remote build\n\nThis packet builds an exact-source Isaac 6 worker image. It proves no GPU startup or camera result.\n",
        encoding="utf-8",
    )
    archive_paths = sorted(
        [readme, script, *copied], key=lambda path: path.relative_to(output).as_posix()
    )
    archive_member_sha256 = {
        path.relative_to(output).as_posix(): _sha256(path) for path in archive_paths
    }
    archive_members = sorted(archive_member_sha256)
    archive_member_manifest_sha256 = hashlib.sha256(
        json.dumps(archive_member_sha256, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    tarball = output / "isaac_worker_remote_build_packet.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        for path in archive_paths:
            archive.add(path, arcname=path.relative_to(output).as_posix(), recursive=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "packet_kind": PACKET_KIND,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "status": "blocked" if blockers else "ready",
        "blockers": sorted(set(blockers)),
        "packet_dir": str(packet),
        "tarball_path": str(tarball),
        "tarball_sha256": _sha256(tarball),
        "archive_members": archive_members,
        "archive_member_sha256": archive_member_sha256,
        "archive_member_manifest_sha256": archive_member_manifest_sha256,
        "context_manifest_sha256": context_manifest_sha256,
        "dockerfile_sha256": dockerfile_sha256,
        "image_ref": image_ref,
        "base_image_ref": base_image_ref,
        "source_commit": source_commit,
        "source_dirty_patch_sha256": CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
        "source_worktree_dirty": source_worktree_dirty,
        "provider_launch_performed_by_packet": False,
        "raw_secret_values_recorded": False,
        "claim_boundary": {
            "packet_is_not_image_build": True,
            "packet_is_not_registry_proof": True,
            "packet_is_not_provider_startup": True,
            "packet_is_not_camera_validation": True,
            "packet_is_not_task_success": True,
        },
    }
    manifest_path = output / "isaac_worker_remote_build_packet_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--image-ref", default=DEFAULT_IMAGE_REF)
    parser.add_argument("--base-image-ref", default=DEFAULT_BASE_IMAGE_REF)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-worktree-dirty", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_remote_build_packet(
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        image_ref=args.image_ref,
        base_image_ref=args.base_image_ref,
        source_commit=args.source_commit,
        source_worktree_dirty=args.source_worktree_dirty,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
