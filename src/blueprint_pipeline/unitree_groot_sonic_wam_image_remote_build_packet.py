"""Prepare a small remote-build packet for the sealed GR00T/SONIC WAM image."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import ensure_dir, write_json


SCHEMA_VERSION = "unitree_groot_sonic_wam_remote_build_packet.v1"
DEFAULT_BASE_IMAGE = (
    "docker.io/nijelhunt/blueprint-oscar-wam@"
    "sha256:b0f3f675023d4333767d798b565fc049ac5ba788cd7041db5cac7f9784fd49b3"
)
DEFAULT_GROOT_REF = "e5749287857afd97b78f1147166137de29746392"
DEFAULT_IMAGE_REF = (
    "docker.io/nijelhunt/blueprint-unitree-groot-sonic-wam:20260703-sealed-groot"
)
DEFAULT_SOURCE_DIR = (
    Path(__file__).resolve().parents[2]
    / "deploy"
    / "docker"
    / "robot_eval_worker"
    / "unitree_groot_sonic_wam"
)
PACKET_DIRNAME = "unitree_groot_sonic_wam_remote_build"
CONTEXT_FILENAMES = (
    "Dockerfile",
    "requirements_unitree_groot_sonic_system_python.txt",
    "unitree_groot_sonic_wam_image_healthcheck.py",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _versioned_image_ref_status(image_ref: str) -> dict[str, Any]:
    blockers: list[str] = []
    if not image_ref:
        blockers.append("missing_unitree_groot_sonic_wam_image_ref")
    elif ":" not in image_ref and "@sha256:" not in image_ref:
        blockers.append("unitree_groot_sonic_wam_image_ref_must_be_versioned")
    elif image_ref.endswith((":latest", ":local", ":dev", ":test")):
        blockers.append("unitree_groot_sonic_wam_image_ref_refuses_unstable_tag")
    return {"valid": not blockers, "blockers": blockers}


def _remote_build_script(
    *,
    image_ref: str,
    base_image: str,
    groot_ref: str,
    platform: str,
    min_free_gib: int,
    prefetch_checkpoint: bool,
) -> str:
    prefetch = "true" if prefetch_checkpoint else "false"
    return f"""#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
context_dir="$script_dir/context"
image_ref="${{BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_WAM_IMAGE_REF:-{image_ref}}}"
platform="${{BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_WAM_PLATFORM:-{platform}}}"
base_image="${{BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_WAM_BASE_IMAGE:-{base_image}}}"
groot_ref="${{BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_GROOT_SOURCE_REF:-{groot_ref}}}"
prefetch_checkpoint="${{BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_PREFETCH_CHECKPOINT:-{prefetch}}}"
min_free_gib="${{BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_REMOTE_MIN_FREE_GIB:-{min_free_gib}}}"
hf_token_file="${{BLUEPRINT_UNITREE_GROOT_N17_SONIC_HF_TOKEN_FILE:-${{HF_TOKEN_FILE:-$HOME/.blueprint-secrets/hf_token}}}}"
manifest_output="${{BLUEPRINT_REMOTE_BUILD_MANIFEST_OUTPUT:-$script_dir/remote_unitree_groot_sonic_wam_image_build_manifest.json}}"
docker_username_file="${{BLUEPRINT_DOCKER_USERNAME_FILE:-$HOME/.blueprint-secrets/docker_username}}"
docker_password_file="${{BLUEPRINT_DOCKER_PASSWORD_FILE:-$HOME/.blueprint-secrets/docker_pat}}"

write_manifest() {{
  local status="$1"
  local blockers_json="$2"
  local free_kib="${{3:-}}"
  local required_kib="${{4:-}}"
  python3 - "$manifest_output" "$status" "$blockers_json" "$image_ref" "$platform" "$base_image" "$groot_ref" "$prefetch_checkpoint" "$free_kib" "$required_kib" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

def _int_or_none(value):
    try:
        return int(value) if value else None
    except ValueError:
        return None

out = Path(sys.argv[1]).expanduser()
free_kib = _int_or_none(sys.argv[9])
required_kib = _int_or_none(sys.argv[10])
payload = {{
    "schema_version": "unitree_groot_sonic_wam_remote_image_build_result.v1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": sys.argv[2],
    "blockers": json.loads(sys.argv[3]),
    "image_ref": sys.argv[4],
    "platform": sys.argv[5],
    "base_image": sys.argv[6],
    "groot_source_ref": sys.argv[7],
    "prefetch_checkpoint": sys.argv[8].lower() == "true",
    "local_disk_check": {{
        "min_free_gib": {min_free_gib},
        "free_kib": free_kib,
        "required_kib": required_kib,
        "available_free_gib": round(free_kib / 1024 / 1024, 3) if free_kib is not None else None,
        "required_free_gib": round(required_kib / 1024 / 1024, 3) if required_kib is not None else None,
    }},
    "raw_secret_values_recorded": False,
    "claim_boundary": {{
        "remote_image_build_is_not_provider_startup": True,
        "remote_image_build_is_not_policy_inference": True,
        "remote_image_build_is_not_task_success": True,
    }},
}}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(out)
PY
}}

if [[ -z "$image_ref" ]]; then
  write_manifest "blocked" '["missing_unitree_groot_sonic_wam_image_ref"]' >/dev/null
  echo "missing image ref" >&2
  exit 2
fi

if [[ "$image_ref" != *:* && "$image_ref" != *@sha256:* ]]; then
  write_manifest "blocked" '["unitree_groot_sonic_wam_image_ref_must_be_versioned"]' >/dev/null
  echo "image ref must be versioned: $image_ref" >&2
  exit 2
fi

case "$image_ref" in
  *:latest|*:local|*:dev|*:test)
    write_manifest "blocked" '["unitree_groot_sonic_wam_image_ref_refuses_unstable_tag"]' >/dev/null
    echo "refuses unstable image tag: $image_ref" >&2
    exit 2
    ;;
esac

docker info >/dev/null

free_kib="$(df -Pk "$script_dir" | awk 'NR==2 {{print $4}}')"
required_kib=$((min_free_gib * 1024 * 1024))
if [[ "${{free_kib:-0}}" -lt "$required_kib" ]]; then
  write_manifest "blocked" '["insufficient_remote_disk_for_unitree_groot_sonic_wam_image_build"]' "$free_kib" "$required_kib" >/dev/null
  echo "insufficient remote disk for sealed GR00T/SONIC image build: need ${{min_free_gib}}GiB free" >&2
  exit 2
fi

if [[ "${{BLUEPRINT_REMOTE_IMAGE_BUILD_DOCKER_LOGIN:-false}}" == "true" ]]; then
  if [[ ! -f "$docker_username_file" || ! -f "$docker_password_file" ]]; then
    write_manifest "blocked" '["docker_registry_login_files_missing"]' "$free_kib" "$required_kib" >/dev/null
    echo "docker login requested but username/password files are missing" >&2
    exit 2
  fi
  docker login -u "$(cat "$docker_username_file")" --password-stdin < "$docker_password_file"
fi

secret_args=()
if [[ "$prefetch_checkpoint" == "true" && -f "$hf_token_file" ]]; then
  secret_args=(--secret "id=hf_token,src=$hf_token_file")
fi

docker buildx build \\
  --platform "$platform" \\
  --progress plain \\
  --build-arg "BASE_IMAGE=$base_image" \\
  --build-arg "GROOT_SOURCE_REF=$groot_ref" \\
  --build-arg "PREFETCH_GROOT_CHECKPOINT=$prefetch_checkpoint" \\
  -f "$context_dir/Dockerfile" \\
  -t "$image_ref" \\
  --push \\
  "${{secret_args[@]}}" \\
  "$context_dir"

if docker buildx imagetools inspect --raw "$image_ref" >/tmp/blueprint_unitree_groot_sonic_wam_imagetools.json; then
  write_manifest "completed" "[]" "$free_kib" "$required_kib" >/dev/null
else
  write_manifest "built_manifest_inspection_blocked" '["unitree_groot_sonic_wam_image_manifest_inspection_failed"]' "$free_kib" "$required_kib" >/dev/null
  exit 2
fi
"""


def prepare_remote_build_packet(
    *,
    output_dir: str | Path,
    image_ref: str = DEFAULT_IMAGE_REF,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    base_image: str = DEFAULT_BASE_IMAGE,
    groot_ref: str = DEFAULT_GROOT_REF,
    platform: str = "linux/amd64",
    min_free_gib: int = 80,
    prefetch_checkpoint: bool = True,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or _utc_now_iso()
    resolved_output = Path(output_dir).expanduser().resolve()
    packet_root = resolved_output / PACKET_DIRNAME
    context_dir = packet_root / "context"
    ensure_dir(context_dir)

    resolved_source = Path(source_dir).expanduser().resolve()
    missing = [name for name in CONTEXT_FILENAMES if not (resolved_source / name).is_file()]
    image_ref_status = _versioned_image_ref_status(image_ref)
    blockers = list(missing)
    if missing:
        blockers = [f"missing_remote_build_context_file:{name}" for name in missing]
    blockers.extend(image_ref_status["blockers"])

    files: list[dict[str, Any]] = []
    if not blockers:
        for name in CONTEXT_FILENAMES:
            src = resolved_source / name
            dst = context_dir / name
            shutil.copy2(src, dst)
            files.append(
                {
                    "path": f"context/{name}",
                    "sha256": _sha256(dst),
                    "bytes": dst.stat().st_size,
                }
            )

    run_script = packet_root / "remote_build_unitree_groot_sonic_wam_image.sh"
    run_script.write_text(
        _remote_build_script(
            image_ref=image_ref,
            base_image=base_image,
            groot_ref=groot_ref,
            platform=platform,
            min_free_gib=min_free_gib,
            prefetch_checkpoint=prefetch_checkpoint,
        ),
        encoding="utf-8",
    )
    run_script.chmod(0o755)

    readme = packet_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Unitree GR00T/SONIC WAM Remote Image Build",
                "",
                "Run this packet on a Docker host with enough disk and registry access:",
                "",
                "```bash",
                "tar -xzf unitree_groot_sonic_wam_remote_build_packet.tar.gz",
                "cd unitree_groot_sonic_wam_remote_build",
                "./remote_build_unitree_groot_sonic_wam_image.sh",
                "```",
                "",
                "Optional inputs:",
                "",
                "- `BLUEPRINT_REMOTE_IMAGE_BUILD_DOCKER_LOGIN=true` to login from local files.",
                "- `BLUEPRINT_DOCKER_USERNAME_FILE` and `BLUEPRINT_DOCKER_PASSWORD_FILE`.",
                "- `BLUEPRINT_UNITREE_GROOT_N17_SONIC_HF_TOKEN_FILE` for checkpoint prefetch.",
                "",
                "This packet builds and pushes an image only. It does not prove provider startup,",
                "policy inference, WAM rollout quality, semantic task success, or physical readiness.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    tarball = resolved_output / "unitree_groot_sonic_wam_remote_build_packet.tar.gz"
    with tarfile.open(tarball, "w:gz") as tf:
        tf.add(packet_root, arcname=PACKET_DIRNAME)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated,
        "status": "blocked" if blockers else "ready",
        "blockers": blockers,
        "packet_dir": str(packet_root),
        "tarball_path": str(tarball),
        "run_script_path": str(run_script),
        "readme_path": str(readme),
        "source_dir": str(resolved_source),
        "image_ref": image_ref,
        "platform": platform,
        "base_image": base_image,
        "groot_source_ref": groot_ref,
        "prefetch_checkpoint": bool(prefetch_checkpoint),
        "min_free_gib": int(min_free_gib),
        "context_files": files,
        "remote_requirements": {
            "docker_buildx_required": True,
            "registry_push_access_required": True,
            "hf_token_file_optional_for_public_or_cached_checkpoint": True,
            "remote_disk_free_gib_required": int(min_free_gib),
        },
        "provider_use": {
            "digitalocean": "copy packet to a DO GPU or build droplet, run script, then destroy droplet",
            "runpod": "copy packet to a RunPod pod or network volume, run script, then terminate pod",
            "provider_launch_performed_by_packet": False,
        },
        "raw_secret_values_recorded": False,
        "claim_boundary": {
            "remote_build_packet_is_not_image_build": True,
            "remote_build_packet_is_not_provider_startup": True,
            "remote_build_packet_is_not_policy_inference": True,
            "remote_build_packet_is_not_task_success": True,
        },
    }
    manifest_path = resolved_output / "unitree_groot_sonic_wam_remote_build_packet_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a remote-build packet for the sealed GR00T/SONIC WAM image."
    )
    parser.add_argument(
        "--output-dir",
        default="output/unitree_groot_sonic_wam_remote_build_packet",
    )
    parser.add_argument("--image-ref", default=DEFAULT_IMAGE_REF)
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--base-image", default=DEFAULT_BASE_IMAGE)
    parser.add_argument("--groot-ref", default=DEFAULT_GROOT_REF)
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--min-free-gib", type=int, default=80)
    parser.add_argument(
        "--no-prefetch-checkpoint",
        action="store_true",
        help="Disable checkpoint prefetch in the generated remote build script.",
    )
    args = parser.parse_args(argv)
    manifest = prepare_remote_build_packet(
        output_dir=args.output_dir,
        image_ref=args.image_ref,
        source_dir=args.source_dir,
        base_image=args.base_image,
        groot_ref=args.groot_ref,
        platform=args.platform,
        min_free_gib=args.min_free_gib,
        prefetch_checkpoint=not args.no_prefetch_checkpoint,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest.get("status") == "ready" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
