"""Prepare a no-spend packet for an external amd64 thin-image builder.

Packet creation is local and non-mutating outside its output directory.  The
generated script performs registry writes only when an operator runs it on a
separately authorized builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .common import ensure_dir, write_json

SCHEMA_VERSION = "groot_oscar_thin_remote_build_packet.v1"
PACKET_DIRNAME = "groot_oscar_thin_remote_build"
REQUIRED_ROOT_FILES = ("pyproject.toml", "README.md", "LICENSE")
REQUIRED_IMAGE_FILES = (
    "apt_transport_hardening.conf",
    "Foundation.Dockerfile",
    "oscar_cpu_import_probe.py",
    "Release.Dockerfile",
    "requirements_robot_runtime.txt",
    "requirements_oscar_foundation.lock",
    "requirements_runpod_serverless.in",
    "requirements_runpod_serverless.lock",
    "requirements_runpod_serverless_sdk.lock",
    "requirements_uv_bootstrap.txt",
    "thin_release_entrypoint.sh",
    "groot_oscar_closed_loop_image_healthcheck.py",
    "isaac_6_g1_assets.sha256",
    "fetch_pinned_isaac_assets.py",
)
_SAFE_VERSIONED_IMAGE_REF = re.compile(
    r"\A[a-z0-9]+(?:[._:-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*"
    r"(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}|@sha256:[0-9a-f]{64})\Z"
)
SYFT_VERSION = "1.44.0"
SYFT_ARCHIVE_URL = "https://github.com/anchore/syft/releases/download/v1.44.0/syft_1.44.0_linux_amd64.tar.gz"
SYFT_ARCHIVE_SHA256 = "0e91737aee2b5baf1d255b959630194a302335d848ff97bb07921eb6205b5f5a"


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
    if not _SAFE_VERSIONED_IMAGE_REF.fullmatch(ref):
        return [f"{label}_image_ref_invalid"]
    if "@sha256:" in ref:
        return [f"{label}_image_ref_must_use_tag"]
    if ":" not in leaf and "@sha256:" not in ref:
        return [f"{label}_image_ref_must_be_versioned"]
    if leaf.endswith((":latest", ":local", ":dev", ":test")):
        return [f"{label}_image_ref_refuses_unstable_tag"]
    return []


def _exact_digest_ref_blockers(ref: str, label: str) -> list[str]:
    if not ref:
        return [f"missing_{label}_image_ref"]
    if not _SAFE_VERSIONED_IMAGE_REF.fullmatch(ref):
        return [f"{label}_image_ref_invalid"]
    if not re.search(r"@sha256:[0-9a-f]{64}\Z", ref):
        return [f"{label}_image_ref_must_use_exact_digest"]
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
    reuse_foundation_exact: bool = False,
    foundation_model_assets: str = "external",
) -> str:
    foundation_default = shlex.quote(foundation_ref)
    release_default = shlex.quote(release_ref)
    foundation_build = (
        '''foundation_exact="$foundation_ref"
foundation_digest="${foundation_ref##*@}"
[[ "$foundation_digest" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo "foundation digest missing" >&2; exit 2; }
'''
        if reuse_foundation_exact
        else '''foundation_metadata="$script_dir/foundation_buildx_metadata.json"
docker buildx build --platform linux/amd64 --progress plain --metadata-file "$foundation_metadata" \\
  --attest type=sbom --attest type=provenance,mode=max \\
  -f "$context_dir/deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Foundation.Dockerfile" \\
  -t "$foundation_candidate_ref" --push "$context_dir"
foundation_digest="$(python3 -c 'import json,sys;p=json.load(open(sys.argv[1]));print(p.get("containerimage.digest") or p.get("containerimage.descriptor",{}).get("digest") or "")' "$foundation_metadata")"
[[ "$foundation_digest" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo "foundation digest missing" >&2; exit 2; }
foundation_exact="$foundation_repository@$foundation_digest"
'''
    )
    foundation_promotion = (
        '''# The digest-pinned foundation is intentionally reused byte-for-byte.
[[ "$foundation_exact" == "$foundation_ref" ]] || { echo "reused foundation digest changed" >&2; exit 2; }
'''
        if reuse_foundation_exact
        else '''docker buildx imagetools create --tag "$foundation_ref" "$foundation_exact"
promoted_foundation_digest="$(docker buildx imagetools inspect --format '{{json .}}' "$foundation_ref" | python3 -c 'import json,re,sys;p=json.load(sys.stdin);m=(p.get("manifest") or p.get("Manifest")) if isinstance(p,dict) else None;d=str(m.get("digest") or "") if isinstance(m,dict) else "";print(d);raise SystemExit(0 if re.fullmatch(r"sha256:[0-9a-f]{64}",d) else 2)' 2>/dev/null)" || { echo "promoted foundation digest missing" >&2; exit 2; }
[[ "$promoted_foundation_digest" == "$foundation_digest" ]] || { echo "promoted foundation digest mismatch" >&2; exit 2; }
'''
    )
    return f'''#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
context_dir="$script_dir/context"
foundation_ref="${{BLUEPRINT_GROOT_OSCAR_FOUNDATION_IMAGE_REF:-}}"
release_ref="${{BLUEPRINT_GROOT_OSCAR_RELEASE_IMAGE_REF:-}}"
[[ -n "$foundation_ref" ]] || foundation_ref={foundation_default}
[[ -n "$release_ref" ]] || release_ref={release_default}
source_commit={shlex.quote(source_commit)}
foundation_repository="$(python3 -c 'import sys;ref=sys.argv[1];leaf=ref.rsplit("/",1)[-1];print(ref.rsplit(":",1)[0] if ":" in leaf else ref)' "$foundation_ref")"
release_repository="$(python3 -c 'import sys;ref=sys.argv[1];leaf=ref.rsplit("/",1)[-1];print(ref.rsplit(":",1)[0] if ":" in leaf else ref)' "$release_ref")"
foundation_candidate_ref="$foundation_repository:blueprint-foundation-candidate-${{source_commit:0:12}}"
release_candidate_ref="$release_repository:blueprint-release-candidate-${{source_commit:0:12}}"
min_free_gib="${{BLUEPRINT_GROOT_OSCAR_REMOTE_MIN_FREE_GIB:-{min_free_gib}}}"
max_release_bytes="${{BLUEPRINT_GROOT_OSCAR_RELEASE_MAX_COMPRESSED_BYTES:-{max_release_bytes}}}"
result="$script_dir/groot_oscar_thin_remote_build_result.json"
registry_user_file="${{BLUEPRINT_DOCKER_USERNAME_FILE:-$HOME/.blueprint-secrets/docker_username}}"
registry_password_file="${{BLUEPRINT_DOCKER_PASSWORD_FILE:-$HOME/.blueprint-secrets/docker_pat}}"
tools_dir="$script_dir/tools"
syft_archive="$tools_dir/syft_{SYFT_VERSION}_linux_amd64.tar.gz"
syft_bin="$tools_dir/syft"

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
install -d -m 700 "$tools_dir"
curl --fail --location --silent --show-error {shlex.quote(SYFT_ARCHIVE_URL)} -o "$syft_archive"
echo "{SYFT_ARCHIVE_SHA256}  $syft_archive" | sha256sum --check --strict
tar -xzf "$syft_archive" -C "$tools_dir" syft
chmod 700 "$syft_bin"
"$syft_bin" version >/dev/null

release_metadata="$script_dir/release_buildx_metadata.json"
validation_result="$script_dir/groot_oscar_thin_remote_build_validation.json"
{foundation_build}

docker buildx build --platform linux/amd64 --progress plain --metadata-file "$release_metadata" \
  --attest type=sbom --attest type=provenance,mode=max \
  --build-arg "FOUNDATION_IMAGE=$foundation_exact" \
  --build-arg "BLUEPRINT_SOURCE_COMMIT={source_commit}" \
  --build-arg "BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256={source_patch_sha256}" \
  --build-arg "FOUNDATION_MODEL_ASSETS={foundation_model_assets}" \
  -f "$context_dir/deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Release.Dockerfile" \
  -t "$release_candidate_ref" --push "$context_dir"
release_digest="$(python3 -c 'import json,sys;p=json.load(open(sys.argv[1]));print(p.get("containerimage.digest") or p.get("containerimage.descriptor",{{}}).get("digest") or "")' "$release_metadata")"
[[ "$release_digest" =~ ^sha256:[0-9a-f]{{64}}$ ]] || {{ echo "release digest missing" >&2; exit 2; }}
release_exact="$release_repository@$release_digest"

PYTHONPATH="$context_dir/src" python3 -m blueprint_pipeline.isaac_worker_image_manifest --image "$foundation_exact" --output "$script_dir/foundation_registry_diagnostic.json"
PYTHONPATH="$context_dir/src" python3 -m blueprint_pipeline.isaac_worker_image_manifest --image "$release_exact" --output "$script_dir/release_registry_diagnostic.json"
docker buildx imagetools inspect --format '{{{{json .SBOM}}}}' "$release_exact" > "$script_dir/release_buildkit_sbom_attestation.json"
docker buildx imagetools inspect --format '{{{{json .Provenance}}}}' "$release_exact" > "$script_dir/release_buildkit_provenance_attestation.json"
docker buildx imagetools inspect --raw "$release_exact" > "$script_dir/release_buildkit_attestation_index.json"
PYTHONPATH="$context_dir/src" python3 - "$script_dir/release_registry_diagnostic.json" "$script_dir/release_supply_chain_disk_admission.json" "$script_dir" <<'PY'
import json,os,sys
from pathlib import Path
from blueprint_pipeline.groot_oscar_release_hardening import DiskAdmission
registry=json.load(open(sys.argv[1])); root=Path(sys.argv[3]); stats=os.statvfs(root)
compressed=int(registry.get("total_compressed_size_bytes") or 0)
evidence=DiskAdmission(available_bytes=stats.f_bavail*stats.f_frsize,image_compressed_bytes=compressed,image_unpacked_bytes=compressed).evidence()
Path(sys.argv[2]).write_text(json.dumps(evidence,indent=2,sort_keys=True)+"\\n",encoding="utf-8")
raise SystemExit(0 if compressed > 0 and evidence["status"] == "passed" else 2)
PY
"$syft_bin" "registry:$release_exact" -o "spdx-json=$script_dir/release_sbom.spdx.json"
PYTHONPATH="$context_dir/src" python3 -m blueprint_pipeline.groot_oscar_release_hardening validate-thin-release \
  --digest-ref "$release_exact" \
  --registry-diagnostic "$script_dir/release_registry_diagnostic.json" \
  --buildx-metadata "$release_metadata" \
  --spdx "$script_dir/release_sbom.spdx.json" \
  --buildkit-sbom "$script_dir/release_buildkit_sbom_attestation.json" \
  --buildkit-provenance "$script_dir/release_buildkit_provenance_attestation.json" \
  --buildkit-index "$script_dir/release_buildkit_attestation_index.json" \
  --provenance-output "$script_dir/release_provenance.json" \
  --layer-report-output "$script_dir/release_layer_report.json" \
  --manifest-output "$script_dir/release_supply_chain_manifest.json"

PYTHONPATH="$context_dir/src" python3 - "$script_dir" "$validation_result" "$foundation_exact" "$release_exact" "$max_release_bytes" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
from blueprint_pipeline.thin_release_image_contract import build_thin_release_contract
root=Path(sys.argv[1]); out=Path(sys.argv[2])
foundation=json.load(open(root/"foundation_registry_diagnostic.json")); release=json.load(open(root/"release_registry_diagnostic.json"))
contract=build_thin_release_contract(release,foundation,max_release_bytes=int(sys.argv[5]),foundation_model_assets={foundation_model_assets!r})
cuda=release.get("required_cuda_version")
blockers=[]
if contract["status"] != "passed": blockers.append("thin_release_contract_not_passed")
if not cuda: blockers.append("release_registry_cuda_version_missing")
sbom=json.load(open(root/"release_sbom.spdx.json"))
runpod_versions=sorted({{str(row.get("versionInfo") or "") for row in sbom.get("packages",[]) if str(row.get("name") or "").lower()=="runpod"}})
worker_source=(root/"context/src/blueprint_pipeline/groot_oscar_runpod_serverless_worker.py").is_file()
dockerfile=(root/"context/deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Release.Dockerfile").read_text(encoding="utf-8")
worker_command="blueprint_pipeline.groot_oscar_runpod_serverless_worker" in dockerfile
sdk_pinned='RUNPOD_SERVERLESS_SDK_VERSION=1.10.1' in dockerfile and runpod_versions==["1.10.1"]
model_assets_bound=contract.get("models_externalized") is True or contract.get("models_embedded_in_foundation") is True
serverless_contract={{"schema_version":"groot_oscar_runpod_serverless_release_contract.v1","status":"passed" if worker_source and worker_command and sdk_pinned and model_assets_bound else "blocked","worker_source_packaged":worker_source,"worker_command_packaged":worker_command,"runpod_sdk_versions":runpod_versions,"runpod_sdk_exactly_pinned":sdk_pinned,"models_externalized":contract.get("models_externalized") is True,"models_embedded_in_foundation":contract.get("models_embedded_in_foundation") is True}}
if serverless_contract["status"] != "passed": blockers.append("runpod_serverless_worker_contract_not_passed")
payload={{"schema_version":"groot_oscar_thin_remote_build_result.v1","generated_at":datetime.now(timezone.utc).isoformat(),"status":"completed" if not blockers else "blocked","blockers":blockers,"foundation_image_ref":sys.argv[3],"release_image_ref":sys.argv[4],"resolved_digest_ref":sys.argv[4],"runnable_platform":"linux/amd64","required_cuda_version":cuda,"required_cuda_version_source":release.get("required_cuda_version_source"),"source_commit":"{source_commit}","source_patch_sha256":"{source_patch_sha256}","foundation_model_assets":"{foundation_model_assets}","thin_release_contract_status":contract["status"],"thin_release_contract":contract,"serverless_worker_contract":serverless_contract,"models_embedded":contract.get("models_embedded_in_foundation") is True,"raw_secret_values_recorded":False,"claim_boundary":{{"remote_build_is_not_model_cache_verification":True,"remote_build_is_not_provider_startup":True,"remote_build_is_not_task_success":True}}}}
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\\n",encoding="utf-8")
raise SystemExit(0 if payload["status"]=="completed" else 2)
PY

# Final tags become visible only after the immutable candidate digests pass
# registry, disk, SBOM, provenance, layer, and thin-release contract validation.
docker buildx imagetools create --tag "$release_ref" "$release_exact"
promoted_release_digest="$(docker buildx imagetools inspect --format '{{{{json .}}}}' "$release_ref" | python3 -c 'import json,re,sys;p=json.load(sys.stdin);m=(p.get("manifest") or p.get("Manifest")) if isinstance(p,dict) else None;d=str(m.get("digest") or "") if isinstance(m,dict) else "";print(d);raise SystemExit(0 if re.fullmatch(r"sha256:[0-9a-f]{{64}}",d) else 2)' 2>/dev/null)" || {{ echo "promoted release digest missing" >&2; exit 2; }}
[[ "$promoted_release_digest" == "$release_digest" ]] || {{ echo "promoted release digest mismatch" >&2; exit 2; }}
{foundation_promotion}
mv "$validation_result" "$result"
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
    reuse_foundation_exact: bool = False,
    foundation_model_assets: str = "external",
    generated_at: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    packet = output / PACKET_DIRNAME
    context = packet / "context"
    ensure_dir(context)
    blockers = [
        *(
            _exact_digest_ref_blockers(foundation_ref, "foundation")
            if reuse_foundation_exact
            else _versioned_ref_blockers(foundation_ref, "foundation")
        ),
        *_versioned_ref_blockers(release_ref, "release"),
    ]
    if foundation_model_assets not in {"external", "embedded"}:
        blockers.append("foundation_model_asset_mode_invalid")
    if foundation_model_assets == "embedded" and not reuse_foundation_exact:
        blockers.append("embedded_foundation_assets_require_exact_reuse")
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
        blockers.append("source_git_identity_unavailable")
    if source_commit != actual_commit:
        blockers.append("source_commit_not_exact_head")
    if source_worktree_dirty != actual_dirty:
        blockers.append("source_worktree_dirty_claim_mismatch")
    if source_worktree_dirty or actual_dirty:
        blockers.append("remote_release_packet_requires_clean_source_worktree")
    if len(source_commit) != 40:
        blockers.append("source_commit_invalid")
    clean_patch_sha256 = hashlib.sha256(b"").hexdigest()
    if len(source_patch_sha256) != 64:
        blockers.append("source_patch_sha256_invalid")
    elif not actual_dirty and source_patch_sha256 != clean_patch_sha256:
        blockers.append("source_patch_sha256_not_clean_tree_digest")

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
    script.write_text(_remote_script(foundation_ref=foundation_ref, release_ref=release_ref, source_commit=source_commit, source_patch_sha256=source_patch_sha256, min_free_gib=min_free_gib, max_release_bytes=max_release_bytes, reuse_foundation_exact=reuse_foundation_exact, foundation_model_assets=foundation_model_assets), encoding="utf-8")
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
        "tarball_sha256": _sha256(tarball),
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
        "reuse_foundation_exact": reuse_foundation_exact,
        "foundation_model_assets": foundation_model_assets,
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
    parser.add_argument("--reuse-foundation-exact", action="store_true")
    parser.add_argument(
        "--foundation-model-assets",
        choices=("external", "embedded"),
        default="external",
    )
    args = parser.parse_args(argv)
    result = prepare_remote_build_packet(output_dir=args.output_dir, repo_root=args.repo_root, foundation_ref=args.foundation_ref, release_ref=args.release_ref, source_commit=args.source_commit, source_patch_sha256=args.source_patch_sha256, source_worktree_dirty=args.source_worktree_dirty, min_free_gib=args.min_free_gib, max_release_bytes=args.max_release_bytes, reuse_foundation_exact=args.reuse_foundation_exact, foundation_model_assets=args.foundation_model_assets)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
