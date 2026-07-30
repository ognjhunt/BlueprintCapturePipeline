"""Prepare the canonical CPU-build packet for the OpenPI ranking image."""

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
from .openpi_policy_ranking_gpu_admission import MENAGERIE_REVISION, OPENPI_REVISION


SCHEMA_VERSION = "openpi_policy_ranking_remote_build_packet.v1"
PACKET_KIND = "openpi_policy_ranking_image"
PACKET_DIRNAME = "openpi_policy_ranking_remote_build"
BUILD_SCRIPT_NAME = "remote_build_openpi_policy_ranking_image.sh"
RESULT_NAME = "openpi_policy_ranking_gpu_release.json"
DEFAULT_IMAGE_REF = "docker.io/nijelhunt/blueprint-openpi-policy-ranking:20260726"
DEFAULT_CTRL_WORLD_IMAGE_REF = (
    "docker.io/nijelhunt/blueprint-openpi-ctrl-world-diagnostic:20260730-v1"
)
DEFAULT_CTRL_WORLD_OSCAR_IMAGE_REF = (
    "docker.io/nijelhunt/blueprint-openpi-ctrl-world-oscar-diagnostic:20260730-v1"
)
DOCKERFILE_RELATIVE_PATH = Path("deploy/docker/policy_ranking_openpi/Dockerfile")
CTRL_WORLD_DOCKERFILE_RELATIVE_PATH = Path(
    "deploy/docker/policy_ranking_openpi_ctrl_world/Dockerfile"
)
CTRL_WORLD_OSCAR_DOCKERFILE_RELATIVE_PATH = Path(
    "deploy/docker/policy_ranking_openpi_ctrl_world_oscar/Dockerfile"
)
REQUIRED_CONTEXT_PATHS = (
    DOCKERFILE_RELATIVE_PATH,
    Path("pyproject.toml"),
    Path("README.md"),
    Path("LICENSE"),
    Path(
        "docs/experiments/policy_ranking_thesis_20260726/warehouse_policy_cohort_v2_joint_position.json"
    ),
    Path(
        "docs/experiments/policy_ranking_thesis_20260726/openpi_polaris_checkpoint_inventory.json"
    ),
    Path(
        "docs/experiments/policy_ranking_thesis_20260726/captured_site_ranking_aggregator_v1.json"
    ),
)
CTRL_WORLD_REQUIRED_CONTEXT_PATHS = (
    CTRL_WORLD_DOCKERFILE_RELATIVE_PATH,
    Path("deploy/docker/policy_ranking_openpi_ctrl_world/requirements.lock"),
    Path("deploy/docker/policy_ranking_openpi_ctrl_world/ctrl_world_source_manifest.json"),
    Path("pyproject.toml"),
    Path("README.md"),
    Path("LICENSE"),
)
CTRL_WORLD_OSCAR_REQUIRED_CONTEXT_PATHS = (
    CTRL_WORLD_OSCAR_DOCKERFILE_RELATIVE_PATH,
    Path("deploy/docker/policy_ranking_openpi_ctrl_world/requirements.lock"),
    Path("deploy/docker/policy_ranking_openpi_ctrl_world/ctrl_world_source_manifest.json"),
    Path(
        "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/"
        "requirements_oscar_foundation.lock"
    ),
    Path("pyproject.toml"),
    Path("README.md"),
    Path("LICENSE"),
)
DEFAULT_IMAGE_VARIANT = "openpi"
CTRL_WORLD_IMAGE_VARIANT = "openpi_ctrl_world"
CTRL_WORLD_OSCAR_IMAGE_VARIANT = "openpi_ctrl_world_oscar"
IMAGE_VARIANTS = {
    DEFAULT_IMAGE_VARIANT: (DOCKERFILE_RELATIVE_PATH, REQUIRED_CONTEXT_PATHS),
    CTRL_WORLD_IMAGE_VARIANT: (
        CTRL_WORLD_DOCKERFILE_RELATIVE_PATH,
        CTRL_WORLD_REQUIRED_CONTEXT_PATHS,
    ),
    CTRL_WORLD_OSCAR_IMAGE_VARIANT: (
        CTRL_WORLD_OSCAR_DOCKERFILE_RELATIVE_PATH,
        CTRL_WORLD_OSCAR_REQUIRED_CONTEXT_PATHS,
    ),
}
_COMMIT = re.compile(r"[0-9a-f]{40}")
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


def _context_sources(
    root: Path, *, required_context_paths: Sequence[Path]
) -> tuple[list[Path], bool]:
    paths = [root / relative for relative in required_context_paths]
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
    source_commit: str,
    dockerfile_sha256: str,
    context_manifest_sha256: str,
    dockerfile_relative_path: Path = DOCKERFILE_RELATIVE_PATH,
) -> str:
    """Return the exact remote script bound into the packet manifest."""

    return f"""#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
context_dir="$script_dir/context"
image_ref={json.dumps(image_ref)}
source_commit={json.dumps(source_commit)}
dockerfile_sha256={json.dumps(dockerfile_sha256)}
context_manifest_sha256={json.dumps(context_manifest_sha256)}
username_file="${{BLUEPRINT_DOCKER_USERNAME_FILE:-$HOME/.blueprint-secrets/docker_username}}"
password_file="${{BLUEPRINT_DOCKER_PASSWORD_FILE:-$HOME/.blueprint-secrets/docker_pat}}"
metadata="$script_dir/openpi_policy_ranking_build_metadata.json"
result="$script_dir/{RESULT_NAME}"

cleanup() {{
  docker logout >/dev/null 2>&1 || true
  rm -f "$username_file" "$password_file"
}}
trap cleanup EXIT

test "$(sha256sum "$context_dir/{dockerfile_relative_path.as_posix()}" | awk '{{print $1}}')" = "$dockerfile_sha256"
test -f "$username_file"
test -f "$password_file"
docker login -u "$(cat "$username_file")" --password-stdin < "$password_file"
docker buildx build \\
  --platform linux/amd64 \\
  --progress plain \\
  --metadata-file "$metadata" \\
  --build-arg "BLUEPRINT_SOURCE_COMMIT=$source_commit" \\
  -f "$context_dir/{dockerfile_relative_path.as_posix()}" \\
  -t "$image_ref" \\
  --push \\
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
python3 - "$result" "$resolved" "$source_commit" "$dockerfile_sha256" "$context_manifest_sha256" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
payload={{
  "schema_version":"openpi_policy_ranking_gpu_release.v1",
  "generated_at":datetime.now(timezone.utc).isoformat(),
  "status":"passed",
  "blockers":[],
  "resolved_digest_ref":sys.argv[2],
  "source_commit":sys.argv[3],
  "runnable_platform":"linux/amd64",
  "dockerfile_sha256":sys.argv[4],
  "context_manifest_sha256":sys.argv[5],
  "openpi_revision":"{OPENPI_REVISION}",
  "menagerie_revision":"{MENAGERIE_REVISION}",
  "checkpoint_bytes_embedded":0,
  "interiorgs_assets_embedded":False,
  "raw_secret_values_recorded":False,
  "claim_boundary":{{
    "image_build_is_not_provider_startup":True,
    "image_build_is_not_policy_inference":True,
    "image_build_is_not_policy_ranking":True,
    "image_build_is_not_task_success":True,
  }},
}}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\\n",encoding="utf-8")
PY
"""


def validate_openpi_policy_ranking_archive(packet: Mapping[str, Any]) -> list[str]:
    """Verify archive inventory, member hashes, and executable script binding."""

    blockers: list[str] = []
    declared_names = packet.get("archive_members")
    names = declared_names if isinstance(declared_names, list) else []
    declared_digests = packet.get("archive_member_sha256")
    digests = declared_digests if isinstance(declared_digests, Mapping) else {}
    image_variant = str(packet.get("image_variant") or DEFAULT_IMAGE_VARIANT)
    variant = IMAGE_VARIANTS.get(image_variant)
    if variant is None:
        blockers.append("builder_openpi_image_variant_invalid")
        variant = IMAGE_VARIANTS[DEFAULT_IMAGE_VARIANT]
    dockerfile_relative_path, required_context_paths = variant
    required_names = {
        f"{PACKET_DIRNAME}/README.md",
        f"{PACKET_DIRNAME}/{BUILD_SCRIPT_NAME}",
        *(f"{PACKET_DIRNAME}/context/{path.as_posix()}" for path in required_context_paths),
    }
    if names != sorted(names) or len(names) != len(set(names)) or not required_names <= set(names):
        blockers.append("builder_openpi_archive_member_contract_invalid")
    if sorted(digests) != names or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
        for value in digests.values()
    ):
        blockers.append("builder_openpi_archive_digest_contract_invalid")
    manifest_digest = hashlib.sha256(
        json.dumps(dict(digests), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if packet.get("archive_member_manifest_sha256") != manifest_digest:
        blockers.append("builder_openpi_archive_member_manifest_mismatch")

    path = Path(str(packet.get("tarball_path") or "")).expanduser().resolve()
    payloads: dict[str, bytes] = {}
    if not path.is_file():
        blockers.append("builder_openpi_archive_missing")
    elif _sha256(path) != packet.get("tarball_sha256"):
        blockers.append("builder_openpi_archive_tarball_mismatch")
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
                    blockers.append("builder_openpi_archive_inventory_mismatch")
                script = next(
                    (m for m in members if m.name == f"{PACKET_DIRNAME}/{BUILD_SCRIPT_NAME}"), None
                )
                if script is None or script.mode & 0o111 == 0:
                    blockers.append("builder_openpi_archive_script_not_executable")
                for member in members:
                    stream = archive.extractfile(member)
                    if stream is not None:
                        payloads[member.name] = stream.read()
        except (OSError, tarfile.TarError):
            blockers.append("builder_openpi_archive_unreadable")
    if any(
        hashlib.sha256(payload).hexdigest() != digests.get(name)
        for name, payload in payloads.items()
    ):
        blockers.append("builder_openpi_archive_member_digest_mismatch")
    dockerfile_name = f"{PACKET_DIRNAME}/context/{dockerfile_relative_path.as_posix()}"
    if dockerfile_name in payloads and hashlib.sha256(
        payloads[dockerfile_name]
    ).hexdigest() != packet.get("dockerfile_sha256"):
        blockers.append("builder_openpi_archive_dockerfile_binding_mismatch")
    expected_script = render_remote_build_script(
        image_ref=str(packet.get("image_ref") or ""),
        source_commit=str(packet.get("source_commit") or ""),
        dockerfile_sha256=str(packet.get("dockerfile_sha256") or ""),
        context_manifest_sha256=str(packet.get("context_manifest_sha256") or ""),
        dockerfile_relative_path=dockerfile_relative_path,
    ).encode()
    if payloads.get(f"{PACKET_DIRNAME}/{BUILD_SCRIPT_NAME}") != expected_script:
        blockers.append("builder_openpi_archive_script_binding_mismatch")
    return sorted(set(blockers))


def prepare_remote_build_packet(
    *,
    output_dir: str | Path,
    repo_root: str | Path,
    image_ref: str,
    source_commit: str,
    source_worktree_dirty: bool,
    generated_at: str | None = None,
    image_variant: str = DEFAULT_IMAGE_VARIANT,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    packet = output / PACKET_DIRNAME
    context = packet / "context"
    blockers: list[str] = []
    variant = IMAGE_VARIANTS.get(image_variant)
    if variant is None:
        blockers.append("openpi_image_variant_invalid")
        variant = IMAGE_VARIANTS[DEFAULT_IMAGE_VARIANT]
    dockerfile_relative_path, required_context_paths = variant
    if not _versioned_image_ref(image_ref):
        blockers.append("openpi_image_ref_not_versioned")
    if not _COMMIT.fullmatch(source_commit):
        blockers.append("openpi_source_commit_invalid")
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
        blockers.append("openpi_source_git_identity_unavailable")
    if actual_commit != source_commit:
        blockers.append("openpi_source_commit_not_exact_head")
    if actual_dirty != source_worktree_dirty:
        blockers.append("openpi_source_dirty_claim_mismatch")
    if source_worktree_dirty or actual_dirty:
        blockers.append("openpi_packet_requires_clean_source_worktree")

    ensure_dir(context)
    sources, tracked_source_available = _context_sources(
        root, required_context_paths=required_context_paths
    )
    if not tracked_source_available:
        blockers.append("openpi_tracked_source_inventory_unavailable")
    missing = [
        relative.as_posix()
        for relative in required_context_paths
        if not (root / relative).is_file()
    ]
    blockers.extend(f"openpi_context_file_missing:{name}" for name in missing)
    copied: list[Path] = []
    for source in sources:
        if not source.is_file() or source.is_symlink():
            continue
        destination = context / source.relative_to(root)
        ensure_dir(destination.parent)
        shutil.copy2(source, destination)
        copied.append(destination)
    dockerfile = context / dockerfile_relative_path
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
            source_commit=source_commit,
            dockerfile_sha256=dockerfile_sha256,
            context_manifest_sha256=context_manifest_sha256,
            dockerfile_relative_path=dockerfile_relative_path,
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    readme = packet / "README.md"
    readme.write_text(
        "# OpenPI policy-ranking remote build\n\nThis packet builds an exact-source runtime image. Checkpoints and site assets remain external.\n",
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
    tarball = output / "openpi_policy_ranking_remote_build_packet.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        for path in archive_paths:
            archive.add(path, arcname=path.relative_to(output).as_posix(), recursive=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "packet_kind": PACKET_KIND,
        "image_variant": image_variant,
        "dockerfile_relative_path": dockerfile_relative_path.as_posix(),
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
        "source_commit": source_commit,
        "source_worktree_dirty": source_worktree_dirty,
        "openpi_revision": OPENPI_REVISION,
        "menagerie_revision": MENAGERIE_REVISION,
        "provider_launch_performed_by_packet": False,
        "raw_secret_values_recorded": False,
        "claim_boundary": {
            "packet_is_not_image_build": True,
            "packet_is_not_registry_proof": True,
            "packet_is_not_policy_ranking": True,
        },
    }
    manifest_path = output / "openpi_policy_ranking_remote_build_packet_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--image-ref", default=DEFAULT_IMAGE_REF)
    parser.add_argument(
        "--image-variant", choices=tuple(IMAGE_VARIANTS), default=DEFAULT_IMAGE_VARIANT
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-worktree-dirty", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_remote_build_packet(
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        image_ref=args.image_ref,
        source_commit=args.source_commit,
        source_worktree_dirty=args.source_worktree_dirty,
        image_variant=args.image_variant,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
