"""Prepare the canonical CPU-build packet for the small GR00T/OSCAR carrier."""

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
from typing import Any, Sequence

from .common import ensure_dir, write_json


SCHEMA_VERSION = "groot_oscar_carrier_remote_build_packet.v1"
PACKET_DIRNAME = "groot_oscar_carrier_remote_build"
DEFAULT_BASE_IMAGE = (
    "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:"
    "b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca"
)
DEFAULT_IMAGE_REF = "docker.io/nijelhunt/blueprint-groot-oscar-carrier:20260716-compatible-v1"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST_REF = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_remote_build_script(
    *, image_ref: str, base_image_ref: str, source_commit: str, dockerfile_sha256: str
) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
image_ref={json.dumps(image_ref)}
base_image_ref={json.dumps(base_image_ref)}
source_commit={json.dumps(source_commit)}
dockerfile_sha256={json.dumps(dockerfile_sha256)}
result="$script_dir/groot_oscar_carrier_remote_build_result.json"
username_file="${{BLUEPRINT_DOCKER_USERNAME_FILE:-$HOME/.blueprint-secrets/docker_username}}"
password_file="${{BLUEPRINT_DOCKER_PASSWORD_FILE:-$HOME/.blueprint-secrets/docker_pat}}"

if [[ "${{BLUEPRINT_REMOTE_IMAGE_BUILD_DOCKER_LOGIN:-false}}" == "true" ]]; then
  test -f "$username_file" && test -f "$password_file"
  docker login -u "$(cat "$username_file")" --password-stdin < "$password_file"
fi
test "$(sha256sum "$script_dir/context/Dockerfile" | awk '{{print $1}}')" = "$dockerfile_sha256"
metadata="$script_dir/carrier_build_metadata.json"
docker buildx build --platform linux/amd64 --progress plain --metadata-file "$metadata" \
  --build-arg "PYTORCH_CARRIER_BASE=$base_image_ref" \
  -f "$script_dir/context/Dockerfile" -t "$image_ref" --push "$script_dir/context"
digest="$(python3 - "$metadata" <<'PY'
import json,re,sys
payload=json.load(open(sys.argv[1], encoding="utf-8"))
value=str(payload.get("containerimage.digest") or "")
print(value)
raise SystemExit(0 if re.fullmatch(r"sha256:[0-9a-f]{{64}}", value) else 2)
PY
)"
resolved="$(python3 - "$image_ref" "$digest" <<'PY'
import sys
ref,digest=sys.argv[1:]
name=ref.rsplit(":", 1)[0] if ":" in ref.rsplit("/", 1)[-1] else ref
print(name + "@" + digest)
PY
)"
docker buildx imagetools inspect "$resolved" >/dev/null
python3 - "$result" "$image_ref" "$resolved" "$base_image_ref" "$dockerfile_sha256" "$source_commit" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
payload={{
  "schema_version":"groot_oscar_carrier_remote_build_result.v1",
  "generated_at":datetime.now(timezone.utc).isoformat(),
  "status":"completed",
  "blockers":[],
  "image_ref":sys.argv[2],
  "resolved_digest_ref":sys.argv[3],
  "base_image_ref":sys.argv[4],
  "dockerfile_sha256":sys.argv[5],
  "source_commit":sys.argv[6],
  "platform":"linux/amd64",
  "raw_secret_values_recorded":False,
  "claim_boundary":{{
    "image_build_is_not_runtime_bundle_verification":True,
    "image_build_is_not_provider_startup":True,
    "image_build_is_not_task_success":True,
  }},
}}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\\n",encoding="utf-8")
PY
docker logout >/dev/null 2>&1 || true
rm -f "$username_file" "$password_file"
"""


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
    ensure_dir(context)
    blockers: list[str] = []
    if not _DIGEST_REF.fullmatch(base_image_ref):
        blockers.append("carrier_base_image_not_digest_pinned")
    image_leaf = image_ref.rsplit("/", 1)[-1]
    image_name, tag_separator, image_tag = image_leaf.rpartition(":")
    if (
        not image_name
        or not tag_separator
        or not _TAG.fullmatch(image_tag)
        or image_tag in {"latest", "dev", "test", "local"}
        or "@" in image_ref
        or any(char.isspace() for char in image_ref)
    ):
        blockers.append("carrier_image_ref_not_versioned")
    if not _COMMIT.fullmatch(source_commit):
        blockers.append("carrier_source_commit_invalid")
    actual_commit = ""
    actual_dirty = True
    try:
        actual_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
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
        blockers.append("carrier_source_git_identity_unavailable")
    if actual_commit != source_commit:
        blockers.append("carrier_source_commit_not_exact_head")
    if actual_dirty != source_worktree_dirty:
        blockers.append("carrier_source_dirty_claim_mismatch")
    if source_worktree_dirty or actual_dirty:
        blockers.append("carrier_packet_requires_clean_source_worktree")
    source = root / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Carrier.Dockerfile"
    if not source.is_file():
        blockers.append("carrier_dockerfile_missing")
        dockerfile_sha256 = ""
    else:
        destination = context / "Dockerfile"
        shutil.copy2(source, destination)
        dockerfile_sha256 = _sha256(destination)
    script = packet / "remote_build_groot_oscar_carrier.sh"
    script.write_text(
        render_remote_build_script(
            image_ref=image_ref,
            base_image_ref=base_image_ref,
            source_commit=source_commit,
            dockerfile_sha256=dockerfile_sha256,
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    (packet / "README.md").write_text(
        "# GR00T OSCAR compatible carrier\n\nThis builds only the small runtime carrier. "
        "Runtime source and model checkpoints remain external.\n",
        encoding="utf-8",
    )
    tarball = output / "groot_oscar_carrier_remote_build_packet.tar.gz"
    archive_paths = (
        packet / "README.md",
        context / "Dockerfile",
        script,
    )
    archive_member_sha256 = {
        path.relative_to(output).as_posix(): _sha256(path) for path in archive_paths
    }
    archive_members = sorted(archive_member_sha256)
    archive_member_manifest_sha256 = hashlib.sha256(
        json.dumps(archive_member_sha256, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with tarfile.open(tarball, "w:gz") as archive:
        for path in archive_paths:
            archive.add(
                path,
                arcname=path.relative_to(output).as_posix(),
                recursive=False,
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "packet_kind": "carrier_image",
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "status": "blocked" if blockers else "ready",
        "blockers": sorted(set(blockers)),
        "packet_dir": str(packet),
        "tarball_path": str(tarball),
        "tarball_sha256": _sha256(tarball),
        "archive_members": archive_members,
        "archive_member_sha256": archive_member_sha256,
        "archive_member_manifest_sha256": archive_member_manifest_sha256,
        "run_script_path": str(script),
        "carrier_image_ref": image_ref,
        "carrier_base_image_ref": base_image_ref,
        "carrier_dockerfile_sha256": dockerfile_sha256,
        "source_commit": source_commit,
        "source_worktree_dirty": source_worktree_dirty,
        "provider_launch_performed_by_packet": False,
        "raw_secret_values_recorded": False,
        "claim_boundary": {
            "packet_is_not_image_build": True,
            "packet_is_not_registry_proof": True,
            "packet_is_not_provider_startup": True,
        },
    }
    manifest_path = output / "groot_oscar_carrier_remote_build_packet_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--image-ref", default=DEFAULT_IMAGE_REF)
    parser.add_argument("--base-image-ref", default=DEFAULT_BASE_IMAGE)
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
