"""Prepare a no-spend packet for an external amd64 thin-image builder.

Packet creation is local and non-mutating outside its output directory.  The
generated script performs registry writes only when an operator runs it on a
separately authorized builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .common import ensure_dir, write_json

SCHEMA_VERSION = "groot_oscar_thin_remote_build_packet.v1"
PACKET_DIRNAME = "groot_oscar_thin_remote_build"
REQUIRED_ROOT_FILES = ("pyproject.toml", "README.md", "LICENSE")
REQUIRED_IMAGE_FILES = (
    "Foundation.Dockerfile",
    "Release.Dockerfile",
    "requirements_robot_runtime.txt",
    "thin_release_entrypoint.sh",
    "groot_oscar_closed_loop_image_healthcheck.py",
    "isaac_6_g1_assets.sha256",
    "fetch_pinned_isaac_assets.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versioned_ref_blockers(ref: str, label: str) -> list[str]:
    leaf = ref.rsplit("/", 1)[-1]
    if not ref:
        return [f"missing_{label}_image_ref"]
    if ":" not in leaf and "@sha256:" not in ref:
        return [f"{label}_image_ref_must_be_versioned"]
    if leaf.endswith((":latest", ":local", ":dev", ":test")):
        return [f"{label}_image_ref_refuses_unstable_tag"]
    return []


def _context_sources(repo_root: Path) -> Iterable[tuple[Path, Path]]:
    for relative in REQUIRED_ROOT_FILES:
        yield repo_root / relative, Path(relative)
    docker_root = Path("deploy/docker/robot_eval_worker/groot_oscar_closed_loop")
    for filename in REQUIRED_IMAGE_FILES:
        yield repo_root / docker_root / filename, docker_root / filename
    dockerignore = repo_root / ".dockerignore"
    if dockerignore.is_file():
        yield dockerignore, Path(".dockerignore")
    for source in sorted((repo_root / "src").rglob("*")):
        if source.is_file() and "__pycache__" not in source.parts:
            yield source, source.relative_to(repo_root)


def _remote_script(
    *,
    foundation_ref: str,
    release_ref: str,
    source_commit: str,
    source_patch_sha256: str,
    min_free_gib: int,
    max_release_bytes: int,
) -> str:
    return f'''#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
context_dir="$script_dir/context"
foundation_ref="${{BLUEPRINT_GROOT_OSCAR_FOUNDATION_IMAGE_REF:-{foundation_ref}}}"
release_ref="${{BLUEPRINT_GROOT_OSCAR_RELEASE_IMAGE_REF:-{release_ref}}}"
min_free_gib="${{BLUEPRINT_GROOT_OSCAR_REMOTE_MIN_FREE_GIB:-{min_free_gib}}}"
max_release_bytes="${{BLUEPRINT_GROOT_OSCAR_RELEASE_MAX_COMPRESSED_BYTES:-{max_release_bytes}}}"
result="$script_dir/groot_oscar_thin_remote_build_result.json"
registry_user_file="${{BLUEPRINT_DOCKER_USERNAME_FILE:-$HOME/.blueprint-secrets/docker_username}}"
registry_password_file="${{BLUEPRINT_DOCKER_PASSWORD_FILE:-$HOME/.blueprint-secrets/docker_pat}}"

python3 - "$context_dir" "$script_dir/context_manifest.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(); manifest=json.load(open(sys.argv[2]))
actual=[]
for row in manifest["files"]:
    path=(root/row["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file(): raise SystemExit("remote_context_file_missing:"+row["path"])
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row["sha256"] or path.stat().st_size != row["bytes"]: raise SystemExit("remote_context_digest_mismatch:"+row["path"])
    actual.append(row["path"])
observed=sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
if sorted(actual) != observed: raise SystemExit("remote_context_inventory_mismatch")
PY

free_kib="$(df -Pk "$script_dir" | awk 'NR==2 {{print $4}}')"
required_kib=$((min_free_gib * 1024 * 1024))
[[ "$free_kib" -ge "$required_kib" ]] || {{ echo "remote builder needs ${{min_free_gib}} GiB free" >&2; exit 2; }}
docker info >/dev/null
if [[ "${{BLUEPRINT_REMOTE_IMAGE_BUILD_DOCKER_LOGIN:-false}}" == true ]]; then
  [[ -f "$registry_user_file" && -f "$registry_password_file" ]] || {{ echo "registry credential files missing" >&2; exit 2; }}
  docker login -u "$(cat "$registry_user_file")" --password-stdin < "$registry_password_file"
fi

foundation_metadata="$script_dir/foundation_buildx_metadata.json"
release_metadata="$script_dir/release_buildx_metadata.json"
docker buildx build --platform linux/amd64 --progress plain --metadata-file "$foundation_metadata" \
  -f "$context_dir/deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Foundation.Dockerfile" \
  -t "$foundation_ref" --push "$context_dir"
foundation_digest="$(python3 -c 'import json,sys;p=json.load(open(sys.argv[1]));print(p.get("containerimage.digest") or p.get("containerimage.descriptor",{{}}).get("digest") or "")' "$foundation_metadata")"
[[ "$foundation_digest" =~ ^sha256:[0-9a-f]{{64}}$ ]] || {{ echo "foundation digest missing" >&2; exit 2; }}
foundation_exact="$(python3 -c 'import sys;ref=sys.argv[1].split("@",1)[0];leaf=ref.rsplit("/",1)[-1];print((ref.rsplit(":",1)[0] if ":" in leaf else ref)+"@"+sys.argv[2])' "$foundation_ref" "$foundation_digest")"

docker buildx build --platform linux/amd64 --progress plain --metadata-file "$release_metadata" \
  --build-arg "FOUNDATION_IMAGE=$foundation_exact" \
  --build-arg "BLUEPRINT_SOURCE_COMMIT={source_commit}" \
  --build-arg "BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256={source_patch_sha256}" \
  -f "$context_dir/deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Release.Dockerfile" \
  -t "$release_ref" --push "$context_dir"
release_digest="$(python3 -c 'import json,sys;p=json.load(open(sys.argv[1]));print(p.get("containerimage.digest") or p.get("containerimage.descriptor",{{}}).get("digest") or "")' "$release_metadata")"
[[ "$release_digest" =~ ^sha256:[0-9a-f]{{64}}$ ]] || {{ echo "release digest missing" >&2; exit 2; }}
release_exact="$(python3 -c 'import sys;ref=sys.argv[1].split("@",1)[0];leaf=ref.rsplit("/",1)[-1];print((ref.rsplit(":",1)[0] if ":" in leaf else ref)+"@"+sys.argv[2])' "$release_ref" "$release_digest")"

PYTHONPATH="$context_dir/src" python3 -m blueprint_pipeline.isaac_worker_image_manifest --image "$foundation_exact" --output "$script_dir/foundation_registry_diagnostic.json"
PYTHONPATH="$context_dir/src" python3 -m blueprint_pipeline.isaac_worker_image_manifest --image "$release_exact" --output "$script_dir/release_registry_diagnostic.json"
PYTHONPATH="$context_dir/src" python3 - "$script_dir" "$result" "$foundation_exact" "$release_exact" "$max_release_bytes" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
from blueprint_pipeline.thin_release_image_contract import build_thin_release_contract
root=Path(sys.argv[1]); out=Path(sys.argv[2])
foundation=json.load(open(root/"foundation_registry_diagnostic.json")); release=json.load(open(root/"release_registry_diagnostic.json"))
contract=build_thin_release_contract(release,foundation,max_release_bytes=int(sys.argv[5]))
cuda=release.get("required_cuda_version")
blockers=[]
if contract["status"] != "passed": blockers.append("thin_release_contract_not_passed")
if not cuda: blockers.append("release_registry_cuda_version_missing")
payload={{"schema_version":"groot_oscar_thin_remote_build_result.v1","generated_at":datetime.now(timezone.utc).isoformat(),"status":"completed" if not blockers else "blocked","blockers":blockers,"foundation_image_ref":sys.argv[3],"release_image_ref":sys.argv[4],"resolved_digest_ref":sys.argv[4],"runnable_platform":"linux/amd64","required_cuda_version":cuda,"required_cuda_version_source":release.get("required_cuda_version_source"),"source_commit":"{source_commit}","source_patch_sha256":"{source_patch_sha256}","thin_release_contract_status":contract["status"],"thin_release_contract":contract,"models_embedded":False,"raw_secret_values_recorded":False,"claim_boundary":{{"remote_build_is_not_model_cache_verification":True,"remote_build_is_not_provider_startup":True,"remote_build_is_not_task_success":True}}}}
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
raise SystemExit(0 if payload["status"]=="completed" else 2)
PY
'''


def prepare_remote_build_packet(
    *,
    output_dir: str | Path,
    repo_root: str | Path,
    foundation_ref: str,
    release_ref: str,
    source_commit: str,
    source_patch_sha256: str,
    source_worktree_dirty: bool,
    min_free_gib: int = 120,
    max_release_bytes: int = 2 * 1024**3,
    generated_at: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    packet = output / PACKET_DIRNAME
    context = packet / "context"
    ensure_dir(context)
    blockers = [
        *_versioned_ref_blockers(foundation_ref, "foundation"),
        *_versioned_ref_blockers(release_ref, "release"),
    ]
    if source_worktree_dirty:
        blockers.append("remote_release_packet_requires_clean_source_worktree")
    if len(source_commit) != 40:
        blockers.append("source_commit_invalid")
    if len(source_patch_sha256) != 64:
        blockers.append("source_patch_sha256_invalid")

    rows: list[dict[str, Any]] = []
    for source, relative in _context_sources(root):
        if not source.is_file():
            blockers.append(f"remote_context_source_missing:{relative.as_posix()}")
            continue
        destination = context / relative
        ensure_dir(destination.parent)
        shutil.copy2(source, destination)
        rows.append({"path": relative.as_posix(), "sha256": _sha256(destination), "bytes": destination.stat().st_size})
    rows.sort(key=lambda row: row["path"])
    context_manifest = {"schema_version": "groot_oscar_thin_remote_context.v1", "files": rows}
    write_json(packet / "context_manifest.json", context_manifest)
    script = packet / "remote_build_groot_oscar_thin_images.sh"
    script.write_text(_remote_script(foundation_ref=foundation_ref, release_ref=release_ref, source_commit=source_commit, source_patch_sha256=source_patch_sha256, min_free_gib=min_free_gib, max_release_bytes=max_release_bytes), encoding="utf-8")
    script.chmod(0o755)
    (packet / "README.md").write_text(
        "# GR00T + OSCAR thin remote build\n\nRun `./remote_build_groot_oscar_thin_images.sh` on an authorized linux/amd64 Docker builder with registry access. The packet does not allocate infrastructure or prepare model volumes.\n",
        encoding="utf-8",
    )
    ensure_dir(output)
    tarball = output / "groot_oscar_thin_remote_build_packet.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        archive.add(packet, arcname=PACKET_DIRNAME)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "status": "blocked" if blockers else "ready",
        "blockers": sorted(set(blockers)),
        "packet_dir": str(packet),
        "tarball_path": str(tarball),
        "run_script_path": str(script),
        "foundation_image_ref": foundation_ref,
        "release_image_ref": release_ref,
        "source_commit": source_commit,
        "source_patch_sha256": source_patch_sha256,
        "source_worktree_dirty": source_worktree_dirty,
        "context_file_count": len(rows),
        "context_total_bytes": sum(row["bytes"] for row in rows),
        "min_free_gib": min_free_gib,
        "max_release_delta_bytes": max_release_bytes,
        "provider_launch_performed_by_packet": False,
        "supported_execution_planes": {
            "native_linux_amd64_docker_builder": True,
            "runpod_pod": False,
        },
        "raw_secret_values_recorded": False,
        "claim_boundary": {"packet_is_not_image_build": True, "packet_is_not_registry_proof": True, "packet_is_not_model_cache_verification": True},
    }
    manifest_path = output / "groot_oscar_thin_remote_build_packet_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output/groot_oscar_thin_remote_build_packet")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--foundation-ref", required=True)
    parser.add_argument("--release-ref", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-patch-sha256", required=True)
    parser.add_argument("--source-worktree-dirty", action="store_true")
    parser.add_argument("--min-free-gib", type=int, default=120)
    parser.add_argument("--max-release-bytes", type=int, default=2 * 1024**3)
    args = parser.parse_args(argv)
    result = prepare_remote_build_packet(output_dir=args.output_dir, repo_root=args.repo_root, foundation_ref=args.foundation_ref, release_ref=args.release_ref, source_commit=args.source_commit, source_patch_sha256=args.source_patch_sha256, source_worktree_dirty=args.source_worktree_dirty, min_free_gib=args.min_free_gib, max_release_bytes=args.max_release_bytes)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
