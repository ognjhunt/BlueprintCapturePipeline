"""Provider-neutral, fail-closed build packet for the reconstruction worker.

The packet is an admission artifact.  It never launches a builder; paid launch
must enter through ``blueprint_pipeline.paid_resource_allocator cpu-build``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, write_json
from .decision_evidence_contracts import canonical_digest
from .reconstruction_worker_contracts import (
    ReconstructionWorkerContractError,
    build_worker_stack_manifest,
)
from .reconstruction_worker_license_inventory import (
    validate_reconstruction_worker_license_inventory,
    validate_reconstruction_worker_license_review_receipt,
)


SCHEMA_VERSION = "reconstruction_worker_build_packet.v2"
REMOTE_PACKET_SCHEMA_VERSION = "reconstruction_worker_remote_build_packet.v2"
PACKET_KIND = "reconstruction_worker_image"
PACKET_DIRNAME = "reconstruction_worker_remote_build"
BUILD_SCRIPT_NAME = "remote_build_reconstruction_worker_image.sh"
RESULT_NAME = "reconstruction_worker_build_receipt.json"
LICENSE_INVENTORY_NAME = "reconstruction_worker_license_inventory.json"
LICENSE_RECEIPT_NAME = "reconstruction_worker_license_review_receipt.json"
PAID_ENVELOPE_NAME = "reconstruction_worker_paid_execution_envelope.json"
PAID_ENVELOPE_SCHEMA_VERSION = "reconstruction_worker_paid_execution_envelope.v1"
DEFAULT_IMAGE_REF = "docker.io/nijelhunt/blueprint-reconstruction-worker:20260730"
DOCKERFILE_RELATIVE_PATH = Path("deploy/docker/reconstruction_worker/Dockerfile")
REQUIRED_CONTEXT_PATHS = (
    DOCKERFILE_RELATIVE_PATH,
    Path("deploy/docker/reconstruction_worker/build-requirements.in"),
    Path("deploy/docker/reconstruction_worker/build-requirements.lock"),
    Path("deploy/docker/reconstruction_worker/requirements.in"),
    Path("deploy/docker/reconstruction_worker/requirements.lock"),
    Path("scripts/compile_reconstruction_worker_lock.py"),
    Path("pyproject.toml"),
    Path("README.md"),
    Path("LICENSE"),
)
ALLOCATOR_ENTRYPOINT = [
    "python",
    "-m",
    "blueprint_pipeline.paid_resource_allocator",
    "cpu-build",
]

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_VERSIONED_IMAGE = re.compile(r"^[^\s@]+:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
_MAX_PAID_BUILD_TTL_SECONDS = 7200


class ReconstructionWorkerBuildPacketError(ValueError):
    pass


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise ReconstructionWorkerBuildPacketError("packet_not_json_serializable") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_paid_execution_envelope(
    value: Mapping[str, Any],
    *,
    source_commit: str,
    worker_stack_manifest_digest: str,
    license_inventory_digest: str,
    license_review_receipt_digest: str,
) -> list[str]:
    blockers: list[str] = []
    if value.get("schema_version") != PAID_ENVELOPE_SCHEMA_VERSION:
        blockers.append("reconstruction_worker_paid_envelope_schema_invalid")
    if value.get("authorized_action") != "cpu-build":
        blockers.append("reconstruction_worker_paid_envelope_action_invalid")
    if value.get("paid_mutation_authorized") is not True:
        blockers.append("reconstruction_worker_paid_envelope_authority_missing")
    if value.get("authority_issued_by_agent") is not False:
        blockers.append("reconstruction_worker_paid_envelope_agent_authority_forbidden")
    if not str(value.get("authority_id") or "").strip():
        blockers.append("reconstruction_worker_paid_envelope_authority_id_missing")
    max_spend = value.get("max_spend_usd")
    if (
        isinstance(max_spend, bool)
        or not isinstance(max_spend, (int, float))
        or not math.isfinite(float(max_spend))
        or float(max_spend) <= 0
    ):
        blockers.append("reconstruction_worker_paid_envelope_budget_invalid")
    ttl = value.get("hard_ttl_seconds")
    if (
        isinstance(ttl, bool)
        or not isinstance(ttl, int)
        or not 0 < ttl <= _MAX_PAID_BUILD_TTL_SECONDS
    ):
        blockers.append("reconstruction_worker_paid_envelope_ttl_invalid")
    retry_cap = value.get("retry_cap")
    if (
        isinstance(retry_cap, bool)
        or not isinstance(retry_cap, int)
        or not 0 <= retry_cap <= 1
    ):
        blockers.append("reconstruction_worker_paid_envelope_retry_cap_invalid")
    for field, expected in (
        ("source_commit_sha", source_commit),
        ("worker_stack_manifest_digest", worker_stack_manifest_digest),
        ("license_inventory_digest", license_inventory_digest),
        ("license_review_receipt_digest", license_review_receipt_digest),
    ):
        if value.get(field) != expected:
            blockers.append(f"reconstruction_worker_paid_envelope_{field}_mismatch")
    if value.get("paid_execution_envelope_digest") != canonical_digest(
        value, digest_field="paid_execution_envelope_digest"
    ):
        blockers.append("reconstruction_worker_paid_envelope_digest_mismatch")
    return sorted(set(blockers))


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
    source_commit: str,
    dockerfile_sha256: str,
    requirements_lock_sha256: str,
    context_manifest_sha256: str,
    worker_stack_manifest_digest: str,
    license_inventory_digest: str,
    license_review_receipt_digest: str,
    paid_execution_envelope_digest: str,
) -> str:
    """Return the only build command accepted for this exact packet."""

    return f'''#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
context_dir="$script_dir/context"
image_ref={json.dumps(image_ref)}
source_commit={json.dumps(source_commit)}
dockerfile_sha256={json.dumps(dockerfile_sha256)}
requirements_lock_sha256={json.dumps(requirements_lock_sha256)}
context_manifest_sha256={json.dumps(context_manifest_sha256)}
worker_stack_manifest_digest={json.dumps(worker_stack_manifest_digest)}
license_inventory_digest={json.dumps(license_inventory_digest)}
license_review_receipt_digest={json.dumps(license_review_receipt_digest)}
paid_execution_envelope_digest={json.dumps(paid_execution_envelope_digest)}
license_inventory_file="$script_dir/{LICENSE_INVENTORY_NAME}"
license_receipt_file="$script_dir/{LICENSE_RECEIPT_NAME}"
paid_execution_envelope_file="$script_dir/{PAID_ENVELOPE_NAME}"
username_file="${{BLUEPRINT_DOCKER_USERNAME_FILE:-$HOME/.blueprint-secrets/docker_username}}"
password_file="${{BLUEPRINT_DOCKER_PASSWORD_FILE:-$HOME/.blueprint-secrets/docker_pat}}"
metadata="$script_dir/reconstruction_worker_build_metadata.json"
result="$script_dir/{RESULT_NAME}"
started_epoch="$(date +%s)"

cleanup() {{
  docker logout >/dev/null 2>&1 || true
}}
trap cleanup EXIT

test -f "$username_file"
test -f "$password_file"
python3 - "$license_inventory_file" "$license_inventory_digest" "license_inventory_digest" "$license_receipt_file" "$license_review_receipt_digest" "license_review_receipt_digest" "$paid_execution_envelope_file" "$paid_execution_envelope_digest" "paid_execution_envelope_digest" <<'PY'
import hashlib,json,sys
for offset in (1,4,7):
    path,expected,digest_field=sys.argv[offset:offset+3]
    payload=json.load(open(path,encoding="utf-8"))
    supplied=payload.pop(digest_field,None)
    encoded=json.dumps(payload,sort_keys=True,separators=(",", ":")).encode()
    observed="sha256:"+hashlib.sha256(encoded).hexdigest()
    if supplied!=expected or observed!=expected:
        raise SystemExit("reconstruction_license_binding_mismatch:"+digest_field)
PY
test "$(sha256sum "$context_dir/{DOCKERFILE_RELATIVE_PATH.as_posix()}" | awk '{{print $1}}')" = "$dockerfile_sha256"
test "$(sha256sum "$context_dir/deploy/docker/reconstruction_worker/requirements.lock" | awk '{{print $1}}')" = "$requirements_lock_sha256"
python3 - "$context_dir" "$context_manifest_sha256" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
manifest=json.loads((root/"reconstruction_worker_context_manifest.json").read_text(encoding="utf-8"))
encoded=json.dumps(manifest,sort_keys=True,separators=(",", ":")).encode()
if hashlib.sha256(encoded).hexdigest()!=sys.argv[2]:
    raise SystemExit("reconstruction_context_manifest_digest_mismatch")
for relative,expected in manifest.items():
    target=(root/relative).resolve()
    if root not in target.parents or not target.is_file() or target.is_symlink():
        raise SystemExit("reconstruction_context_path_invalid:"+relative)
    if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
        raise SystemExit("reconstruction_context_member_digest_mismatch:"+relative)
PY
docker login -u "$(cat "$username_file")" --password-stdin < "$password_file"
docker buildx build \
  --platform linux/amd64 \
  --progress plain \
  --provenance=true \
  --sbom=true \
  --metadata-file "$metadata" \
  --build-arg "BLUEPRINT_SOURCE_COMMIT=$source_commit" \
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
python3 - "$result" "$resolved" "$source_commit" "$context_manifest_sha256" "$worker_stack_manifest_digest" "$license_inventory_digest" "$license_review_receipt_digest" "$paid_execution_envelope_digest" "$started_epoch" <<'PY'
import hashlib,json,sys
import time
from datetime import datetime,timezone
from pathlib import Path
payload={{
  "schema_version":"reconstruction_worker_build_receipt.v2",
  "timestamp":datetime.now(timezone.utc).isoformat(),
  "worker_stack_manifest_digest":sys.argv[5],
  "license_inventory_digest":sys.argv[6],
  "license_review_receipt_digest":sys.argv[7],
  "paid_execution_envelope_digest":sys.argv[8],
  "status":"built",
  "resolved_image_digest":sys.argv[2],
  "source_commit_sha":sys.argv[3],
  "build_context_digest":"sha256:"+sys.argv[4],
  "duration_seconds":max(0.0,time.time()-float(sys.argv[9])),
  "cost_usd":0.0,
  "logs":["reconstruction_worker_build_metadata.json"],
  "blockers":[],
  "scientific_qualification_inferred":False,
  "build_healthcheck_embedded":True,
  "runtime_gpu_healthcheck_completed":False,
  "raw_secret_values_recorded":False,
  "warnings":["paid compute cost is reconciled by the outer canonical CPU builder receipt"],
  "proof_effect":"none",
  "claim_ceiling":"resolved_worker_image_build_only",
}}
encoded=json.dumps(payload,sort_keys=True,separators=(",", ":")).encode()
payload["build_receipt_digest"]="sha256:"+hashlib.sha256(encoded).hexdigest()
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
'''


def _deterministic_tar_gz(path: Path, *, files: Sequence[tuple[Path, str]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for source, name in sorted(files, key=lambda item: item[1]):
                    payload = source.read_bytes()
                    info = tarfile.TarInfo(name=name)
                    info.size = len(payload)
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mode = 0o755 if source.name == BUILD_SCRIPT_NAME else 0o644
                    archive.addfile(info, io.BytesIO(payload))


def validate_reconstruction_worker_archive(packet: Mapping[str, Any]) -> list[str]:
    """Reject packet tampering, traversal, symlinks, and script substitution."""

    blockers: list[str] = []
    names = packet.get("archive_members")
    names = names if isinstance(names, list) else []
    digests = packet.get("archive_member_sha256")
    digests = digests if isinstance(digests, Mapping) else {}
    required = {
        f"{PACKET_DIRNAME}/README.md",
        f"{PACKET_DIRNAME}/{BUILD_SCRIPT_NAME}",
        f"{PACKET_DIRNAME}/{LICENSE_INVENTORY_NAME}",
        f"{PACKET_DIRNAME}/{LICENSE_RECEIPT_NAME}",
        f"{PACKET_DIRNAME}/{PAID_ENVELOPE_NAME}",
        f"{PACKET_DIRNAME}/context/reconstruction_worker_context_manifest.json",
        *(f"{PACKET_DIRNAME}/context/{path.as_posix()}" for path in REQUIRED_CONTEXT_PATHS),
    }
    if names != sorted(names) or len(names) != len(set(names)) or not required <= set(names):
        blockers.append("builder_reconstruction_archive_member_contract_invalid")
    if sorted(digests) != names or any(
        not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in digests.values()
    ):
        blockers.append("builder_reconstruction_archive_digest_contract_invalid")
    manifest_digest = hashlib.sha256(
        json.dumps(dict(digests), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if packet.get("archive_member_manifest_sha256") != manifest_digest:
        blockers.append("builder_reconstruction_archive_member_manifest_mismatch")

    path = Path(str(packet.get("tarball_path") or "")).expanduser().resolve()
    payloads: dict[str, bytes] = {}
    if not path.is_file():
        blockers.append("builder_reconstruction_archive_missing")
    elif _sha256(path) != packet.get("tarball_sha256"):
        blockers.append("builder_reconstruction_archive_tarball_mismatch")
    else:
        try:
            with tarfile.open(path, "r:gz") as archive:
                members = archive.getmembers()
                observed = [member.name for member in members]
                if observed != names or any(
                    not member.isfile()
                    or Path(member.name).is_absolute()
                    or ".." in Path(member.name).parts
                    for member in members
                ):
                    blockers.append("builder_reconstruction_archive_inventory_mismatch")
                script = next(
                    (
                        member
                        for member in members
                        if member.name == f"{PACKET_DIRNAME}/{BUILD_SCRIPT_NAME}"
                    ),
                    None,
                )
                if script is None or script.mode & 0o111 == 0:
                    blockers.append("builder_reconstruction_archive_script_not_executable")
                for member in members:
                    stream = archive.extractfile(member)
                    if stream is not None:
                        payloads[member.name] = stream.read()
        except (OSError, tarfile.TarError):
            blockers.append("builder_reconstruction_archive_unreadable")
    if any(
        hashlib.sha256(payload).hexdigest() != digests.get(name)
        for name, payload in payloads.items()
    ):
        blockers.append("builder_reconstruction_archive_member_digest_mismatch")
    for filename, digest_field in (
        (LICENSE_INVENTORY_NAME, "license_inventory_digest"),
        (LICENSE_RECEIPT_NAME, "license_review_receipt_digest"),
        (PAID_ENVELOPE_NAME, "paid_execution_envelope_digest"),
    ):
        try:
            bound_payload = json.loads(
                payloads[f"{PACKET_DIRNAME}/{filename}"].decode("utf-8")
            )
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
            blockers.append("builder_reconstruction_archive_license_artifact_invalid")
            continue
        if (
            bound_payload.get(digest_field) != packet.get(digest_field)
            or canonical_digest(bound_payload, digest_field=digest_field)
            != packet.get(digest_field)
        ):
            blockers.append("builder_reconstruction_archive_license_binding_mismatch")
    expected_script = render_remote_build_script(
        image_ref=str(packet.get("image_ref") or ""),
        source_commit=str(packet.get("source_commit") or ""),
        dockerfile_sha256=str(packet.get("dockerfile_sha256") or ""),
        requirements_lock_sha256=str(packet.get("requirements_lock_sha256") or ""),
        context_manifest_sha256=str(packet.get("context_manifest_sha256") or ""),
        worker_stack_manifest_digest=str(packet.get("worker_stack_manifest_digest") or ""),
        license_inventory_digest=str(packet.get("license_inventory_digest") or ""),
        license_review_receipt_digest=str(
            packet.get("license_review_receipt_digest") or ""
        ),
        paid_execution_envelope_digest=str(
            packet.get("paid_execution_envelope_digest") or ""
        ),
    ).encode()
    if payloads.get(f"{PACKET_DIRNAME}/{BUILD_SCRIPT_NAME}") != expected_script:
        blockers.append("builder_reconstruction_archive_script_binding_mismatch")
    return sorted(set(blockers))


def prepare_reconstruction_worker_build_packet(
    *,
    worker_stack_manifest: Mapping[str, Any],
    image_ref: str,
    source_commit_sha: str,
    source_tree_digest: str,
    source_worktree_dirty: bool,
    build_recipe_digest: str | None,
    dependency_lock_digest: str | None,
    license_inventory_digest: str | None,
    license_review_receipt_digest: str | None,
    max_spend_usd: float | None,
    ttl_seconds: int | None,
    retry_cap: int | None,
    authority_id: str | None,
    timestamp: str,
) -> dict[str, Any]:
    """Compile immutable build admission without performing paid execution."""

    manifest = build_worker_stack_manifest(worker_stack_manifest)
    blockers: list[str] = []
    if _COMMIT.fullmatch(source_commit_sha) is None:
        blockers.append("worker_build_source_commit_invalid")
    if manifest["source_commit_sha"] != source_commit_sha:
        blockers.append("worker_build_stack_source_commit_mismatch")
    if _DIGEST.fullmatch(source_tree_digest) is None:
        blockers.append("worker_build_source_tree_digest_invalid")
    if source_worktree_dirty:
        blockers.append("worker_build_requires_clean_immutable_commit")
    if _VERSIONED_IMAGE.fullmatch(image_ref) is None or image_ref.rsplit(":", 1)[-1] in {
        "latest",
        "dev",
        "test",
        "local",
    }:
        blockers.append("worker_build_image_ref_not_versioned")
    for value, blocker in (
        (build_recipe_digest, "worker_build_recipe_digest_missing"),
        (dependency_lock_digest, "worker_dependency_lock_digest_missing"),
        (license_inventory_digest, "worker_license_inventory_digest_missing"),
        (license_review_receipt_digest, "worker_license_review_receipt_missing"),
    ):
        if _DIGEST.fullmatch(str(value or "")) is None:
            blockers.append(blocker)
    if (
        isinstance(max_spend_usd, bool)
        or not isinstance(max_spend_usd, (int, float))
        or not math.isfinite(float(max_spend_usd))
        or float(max_spend_usd) <= 0
    ):
        blockers.append("worker_build_explicit_budget_missing")
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
        blockers.append("worker_build_explicit_ttl_missing")
    if isinstance(retry_cap, bool) or not isinstance(retry_cap, int) or retry_cap < 0:
        blockers.append("worker_build_explicit_retry_cap_missing")
    if not isinstance(authority_id, str) or not authority_id.strip():
        blockers.append("worker_build_paid_authority_missing")
    packet = {
        "schema_version": SCHEMA_VERSION,
        "packet_kind": "reconstruction_worker_image",
        "status": "ready" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "worker_stack_manifest_digest": manifest["worker_stack_manifest_digest"],
        "image_ref": image_ref,
        "source_commit_sha": source_commit_sha,
        "source_tree_digest": source_tree_digest,
        "source_worktree_dirty": source_worktree_dirty,
        "build_recipe_digest": build_recipe_digest,
        "dependency_lock_digest": dependency_lock_digest,
        "license_inventory_digest": license_inventory_digest,
        "license_review_receipt_digest": license_review_receipt_digest,
        "allocator_entrypoint": ALLOCATOR_ENTRYPOINT,
        "canonical_paid_resource_seam_required": True,
        "direct_provider_launcher_allowed": False,
        "provider_identity": None,
        "max_spend_usd": max_spend_usd,
        "ttl_seconds": ttl_seconds,
        "retry_cap": retry_cap,
        "authority_id": authority_id,
        "required_outputs": [
            "reconstruction_worker_build_receipt.v2",
            "reconstruction_worker_smoke_test_receipt.v1",
            "provider_teardown_receipt.v1",
            "provider_zero_verification.v1",
        ],
        "paid_execution_started": False,
        "allocation_success_is_scientific_success": False,
        "build_success_is_scientific_success": False,
        "timestamp": timestamp,
    }
    packet["build_packet_digest"] = canonical_digest(
        packet, digest_field="build_packet_digest"
    )
    return _clone(packet)


def prepare_reconstruction_worker_remote_build_packet(
    *,
    output_dir: str | Path,
    repo_root: str | Path,
    image_ref: str,
    source_commit: str,
    source_worktree_dirty: bool,
    worker_stack_manifest: Mapping[str, Any] | None,
    license_inventory: Mapping[str, Any] | None,
    license_review_receipt: Mapping[str, Any] | None,
    paid_execution_envelope: Mapping[str, Any] | None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Materialize a deterministic exact-source Linux/amd64 build archive."""

    root = Path(repo_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    packet_dir = output / PACKET_DIRNAME
    context = packet_dir / "context"
    blockers: list[str] = []
    normalized_stack: dict[str, Any] = {}
    if isinstance(worker_stack_manifest, Mapping):
        try:
            normalized_stack = build_worker_stack_manifest(worker_stack_manifest)
        except ReconstructionWorkerContractError:
            blockers.append("reconstruction_worker_stack_manifest_invalid")
    else:
        blockers.append("reconstruction_worker_stack_manifest_missing")
    if normalized_stack.get("source_commit_sha") != source_commit:
        blockers.append("reconstruction_worker_stack_source_commit_mismatch")
    normalized_inventory = (
        json.loads(json.dumps(dict(license_inventory)))
        if isinstance(license_inventory, Mapping)
        else {}
    )
    inventory_digest = normalized_inventory.get("license_inventory_digest")
    if not normalized_inventory:
        blockers.append("reconstruction_worker_license_inventory_missing")
    else:
        blockers.extend(
            validate_reconstruction_worker_license_inventory(
                normalized_inventory,
                source_commit_sha=source_commit,
                worker_stack_manifest=normalized_stack,
            )
        )
    license_receipt = (
        json.loads(json.dumps(dict(license_review_receipt)))
        if isinstance(license_review_receipt, Mapping)
        else {}
    )
    license_digest = license_receipt.get("license_review_receipt_digest")
    if not license_receipt:
        blockers.append("reconstruction_worker_license_review_receipt_missing")
    else:
        blockers.extend(
            validate_reconstruction_worker_license_review_receipt(
                license_receipt,
                license_inventory=normalized_inventory,
            )
        )
    normalized_paid_envelope = (
        json.loads(json.dumps(dict(paid_execution_envelope)))
        if isinstance(paid_execution_envelope, Mapping)
        else {}
    )
    paid_envelope_digest = normalized_paid_envelope.get(
        "paid_execution_envelope_digest"
    )
    if not normalized_paid_envelope:
        blockers.append("reconstruction_worker_paid_execution_envelope_missing")
    else:
        blockers.extend(
            _validate_paid_execution_envelope(
                normalized_paid_envelope,
                source_commit=source_commit,
                worker_stack_manifest_digest=str(
                    normalized_stack.get("worker_stack_manifest_digest") or ""
                ),
                license_inventory_digest=str(inventory_digest or ""),
                license_review_receipt_digest=str(license_digest or ""),
            )
        )
    if _VERSIONED_IMAGE.fullmatch(image_ref) is None or image_ref.rsplit(":", 1)[-1] in {
        "latest",
        "dev",
        "test",
        "local",
    }:
        blockers.append("reconstruction_worker_image_ref_not_versioned")
    if _COMMIT.fullmatch(source_commit) is None:
        blockers.append("reconstruction_worker_source_commit_invalid")
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
        actual_commit, actual_dirty = "", True
        blockers.append("reconstruction_worker_source_git_identity_unavailable")
    if actual_commit != source_commit:
        blockers.append("reconstruction_worker_source_commit_not_exact_head")
    if actual_dirty != source_worktree_dirty:
        blockers.append("reconstruction_worker_source_dirty_claim_mismatch")
    if source_worktree_dirty or actual_dirty:
        blockers.append("reconstruction_worker_packet_requires_clean_source_worktree")

    ensure_dir(context)
    sources, tracked_source_available = _context_sources(root)
    if not tracked_source_available:
        blockers.append("reconstruction_worker_tracked_source_inventory_unavailable")
    missing = [
        relative.as_posix()
        for relative in REQUIRED_CONTEXT_PATHS
        if not (root / relative).is_file() or (root / relative).is_symlink()
    ]
    blockers.extend(f"reconstruction_worker_context_file_missing:{name}" for name in missing)
    copied: list[Path] = []
    for source in sources:
        if not source.is_file() or source.is_symlink():
            continue
        destination = context / source.relative_to(root)
        ensure_dir(destination.parent)
        shutil.copy2(source, destination)
        copied.append(destination)

    dockerfile = context / DOCKERFILE_RELATIVE_PATH
    requirements_lock = context / "deploy/docker/reconstruction_worker/requirements.lock"
    dockerfile_sha256 = _sha256(dockerfile) if dockerfile.is_file() else ""
    requirements_lock_sha256 = (
        _sha256(requirements_lock) if requirements_lock.is_file() else ""
    )
    context_member_sha256 = {
        destination.relative_to(context).as_posix(): _sha256(destination)
        for destination in copied
    }
    context_manifest_sha256 = hashlib.sha256(
        json.dumps(context_member_sha256, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    context_manifest_path = context / "reconstruction_worker_context_manifest.json"
    context_manifest_path.write_text(
        json.dumps(context_member_sha256, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    script = packet_dir / BUILD_SCRIPT_NAME
    script.write_text(
        render_remote_build_script(
            image_ref=image_ref,
            source_commit=source_commit,
            dockerfile_sha256=dockerfile_sha256,
            requirements_lock_sha256=requirements_lock_sha256,
            context_manifest_sha256=context_manifest_sha256,
            worker_stack_manifest_digest=str(
                normalized_stack.get("worker_stack_manifest_digest") or ""
            ),
            license_inventory_digest=str(inventory_digest or ""),
            license_review_receipt_digest=str(license_digest or ""),
            paid_execution_envelope_digest=str(paid_envelope_digest or ""),
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    license_inventory_path = packet_dir / LICENSE_INVENTORY_NAME
    write_json(license_inventory_path, normalized_inventory)
    license_receipt_path = packet_dir / LICENSE_RECEIPT_NAME
    write_json(license_receipt_path, license_receipt)
    paid_envelope_path = packet_dir / PAID_ENVELOPE_NAME
    write_json(paid_envelope_path, normalized_paid_envelope)
    readme = packet_dir / "README.md"
    readme.write_text(
        "# Reconstruction worker remote build\n\n"
        "This exact-source packet builds and pushes one Linux/amd64 worker image. "
        "It proves no GPU startup, reconstruction quality, collision fidelity, "
        "Isaac compatibility, physical success, or deployment readiness.\n",
        encoding="utf-8",
    )
    archive_paths = [
        readme,
        script,
        license_inventory_path,
        license_receipt_path,
        paid_envelope_path,
        context_manifest_path,
        *copied,
    ]
    archive_files = [
        (path, path.relative_to(output).as_posix()) for path in archive_paths
    ]
    archive_member_sha256 = {
        name: _sha256(path) for path, name in sorted(archive_files, key=lambda item: item[1])
    }
    archive_members = sorted(archive_member_sha256)
    archive_member_manifest_sha256 = hashlib.sha256(
        json.dumps(archive_member_sha256, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    tarball = output / "reconstruction_worker_remote_build_packet.tar.gz"
    _deterministic_tar_gz(tarball, files=archive_files)
    manifest = {
        "schema_version": REMOTE_PACKET_SCHEMA_VERSION,
        "packet_kind": PACKET_KIND,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "status": "blocked" if blockers else "ready",
        "blockers": sorted(set(blockers)),
        "packet_dir": str(packet_dir),
        "tarball_path": str(tarball),
        "tarball_sha256": _sha256(tarball),
        "archive_members": archive_members,
        "archive_member_sha256": archive_member_sha256,
        "archive_member_manifest_sha256": archive_member_manifest_sha256,
        "context_manifest_sha256": context_manifest_sha256,
        "dockerfile_sha256": dockerfile_sha256,
        "requirements_lock_sha256": requirements_lock_sha256,
        "worker_stack_manifest_digest": normalized_stack.get(
            "worker_stack_manifest_digest"
        ),
        "license_inventory_digest": inventory_digest,
        "license_review_receipt_digest": license_digest,
        "paid_execution_envelope_digest": paid_envelope_digest,
        "paid_execution_envelope": normalized_paid_envelope,
        "image_ref": image_ref,
        "source_commit": source_commit,
        "source_worktree_dirty": source_worktree_dirty,
        "provider_launch_performed_by_packet": False,
        "raw_secret_values_recorded": False,
        "canonical_allocator_entrypoint": ALLOCATOR_ENTRYPOINT,
        "claim_boundary": {
            "packet_is_not_image_build": True,
            "packet_is_not_registry_proof": True,
            "packet_is_not_gpu_startup": True,
            "packet_is_not_reconstruction_quality": True,
            "packet_is_not_isaac_qualification": True,
        },
    }
    manifest["remote_build_packet_digest"] = canonical_digest(
        manifest, digest_field="remote_build_packet_digest"
    )
    manifest_path = output / "reconstruction_worker_remote_build_packet_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--repo-root", default=str(Path(__file__).resolve().parents[2])
    )
    parser.add_argument("--image-ref", default=DEFAULT_IMAGE_REF)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-worktree-dirty", action="store_true")
    parser.add_argument("--worker-stack-manifest", required=True)
    parser.add_argument("--license-inventory", required=True)
    parser.add_argument("--license-review-receipt", required=True)
    parser.add_argument("--paid-execution-envelope", required=True)
    args = parser.parse_args(argv)
    result = prepare_reconstruction_worker_remote_build_packet(
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        image_ref=args.image_ref,
        source_commit=args.source_commit,
        source_worktree_dirty=args.source_worktree_dirty,
        worker_stack_manifest=json.loads(
            Path(args.worker_stack_manifest).read_text(encoding="utf-8")
        ),
        license_inventory=json.loads(
            Path(args.license_inventory).read_text(encoding="utf-8")
        ),
        license_review_receipt=json.loads(
            Path(args.license_review_receipt).read_text(encoding="utf-8")
        ),
        paid_execution_envelope=json.loads(
            Path(args.paid_execution_envelope).read_text(encoding="utf-8")
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ready" else 2


__all__ = [
    "ALLOCATOR_ENTRYPOINT",
    "BUILD_SCRIPT_NAME",
    "DEFAULT_IMAGE_REF",
    "LICENSE_INVENTORY_NAME",
    "LICENSE_RECEIPT_NAME",
    "PAID_ENVELOPE_NAME",
    "PAID_ENVELOPE_SCHEMA_VERSION",
    "PACKET_DIRNAME",
    "PACKET_KIND",
    "REMOTE_PACKET_SCHEMA_VERSION",
    "RESULT_NAME",
    "ReconstructionWorkerBuildPacketError",
    "SCHEMA_VERSION",
    "prepare_reconstruction_worker_build_packet",
    "prepare_reconstruction_worker_remote_build_packet",
    "render_remote_build_script",
    "validate_reconstruction_worker_archive",
]


if __name__ == "__main__":
    raise SystemExit(main())
