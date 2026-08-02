"""Inspect an Isaac worker registry manifest without pulling the image.

This is registry metadata evidence only.  It is intentionally separate from the provider startup
canary and from Isaac/render success.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "isaac_worker_image_manifest_diagnostic.v2"
#: Schema versions a paid launch is allowed to consume as startup-timeout evidence.
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})
RUNNABLE_PLATFORM = "linux/amd64"
LARGE_TOTAL_BYTES = 8_000_000_000
LARGE_LAYER_BYTES = 3_000_000_000
_DIGEST_RE = re.compile(r"^Digest:\s*(sha256:[0-9a-f]{64})\s*$", re.MULTILINE)
_DIGEST_VALUE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CUDA_VERSION_RE = re.compile(r"^(\d+)\.(\d+)(?:\.\d+)?(?:[-+._].*)?$")
_CUDA_ENV_KEYS = (
    "BLUEPRINT_GROOT_OSCAR_REQUIRED_CUDA_VERSION",
    "CUDA_VERSION",
)
_SOURCE_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_REF_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_SOURCE_OVERLAY_BUILD_METHOD = "crane_exact_source_overlay_experimental"


def _digest_ref(image_ref: str, digest: str) -> str:
    repository = str(image_ref).split("@", 1)[0]
    if "/" in repository and ":" in repository.rsplit("/", 1)[-1]:
        repository = repository.rsplit(":", 1)[0]
    return f"{repository}@{digest}"


def derive_manifest_size_policy(
    *,
    total_compressed_size_bytes: int | None,
    largest_layer_size_bytes: int | None,
    layers_over_1gb: int | None,
) -> dict[str, Any]:
    """Derive pull-risk and timeout fields from registry layer measurements."""
    total = (
        total_compressed_size_bytes
        if type(total_compressed_size_bytes) is int
        and total_compressed_size_bytes >= 0
        else None
    )
    largest = (
        largest_layer_size_bytes
        if type(largest_layer_size_bytes) is int and largest_layer_size_bytes >= 0
        else None
    )
    large_layers = (
        layers_over_1gb
        if type(layers_over_1gb) is int and layers_over_1gb >= 0
        else None
    )
    return {
        "large_image_pull_risk": bool(
            (total is not None and total >= LARGE_TOTAL_BYTES)
            or (largest is not None and largest >= LARGE_LAYER_BYTES)
        ),
        "split_layer_layout_suitable": bool(
            total is not None
            and largest is not None
            and large_layers is not None
            and largest < LARGE_LAYER_BYTES
            and large_layers >= 2
        ),
        "recommended_startup_no_runtime_timeout_seconds": (
            max(600, min(1800, int((total / 8_000_000) + 120)))
            if total is not None
            else None
        ),
    }


def _runnable_child_digest(raw_manifest: Mapping[str, Any]) -> str | None:
    """Return the sole runnable linux/amd64 child of an OCI index.

    Buildx adds an ``unknown/unknown`` attestation manifest beside the image
    manifest. Layer sizing must inspect the runnable child while retaining the
    top-level index digest as the immutable pull identity.
    """
    manifests = raw_manifest.get("manifests")
    if not isinstance(manifests, list):
        return None
    candidates = []
    for manifest in manifests:
        if not isinstance(manifest, Mapping):
            continue
        platform = manifest.get("platform")
        platform = platform if isinstance(platform, Mapping) else {}
        digest = manifest.get("digest")
        if (
            platform.get("os") == "linux"
            and platform.get("architecture") == "amd64"
            and isinstance(digest, str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        ):
            candidates.append(digest)
    return candidates[0] if len(candidates) == 1 else None


def _image_environment(image_config: Mapping[str, Any] | None) -> dict[str, str]:
    payload = image_config if isinstance(image_config, Mapping) else {}
    config = payload.get("config")
    if not isinstance(config, Mapping):
        config = payload.get("Config")
    if not isinstance(config, Mapping):
        config = payload
    raw_env = config.get("Env")
    if not isinstance(raw_env, list):
        raw_env = config.get("env")
    env: dict[str, str] = {}
    if isinstance(raw_env, list):
        for item in raw_env:
            if isinstance(item, str) and "=" in item:
                key, value = item.split("=", 1)
                env[key] = value
    return env


def derive_required_cuda_version(
    image_config: Mapping[str, Any] | None,
) -> tuple[str | None, str | None]:
    """Derive the CUDA major/minor contract from immutable image config metadata."""
    env = _image_environment(image_config)
    for key in _CUDA_ENV_KEYS:
        match = _CUDA_VERSION_RE.fullmatch(env.get(key, "").strip())
        if match:
            return f"{match.group(1)}.{match.group(2)}", f"image_config_env:{key}"
    return None, None


def derive_worker_build_identity(
    image_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Read source identity embedded by the immutable Isaac worker build."""

    env = _image_environment(image_config)
    source_commit = env.get("BLUEPRINT_SOURCE_COMMIT", "").strip().lower()
    dirty_patch = env.get("BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256", "").strip().lower()
    family = env.get("BLUEPRINT_WORKER_IMAGE_FAMILY", "").strip()
    isaac_major_raw = env.get("BLUEPRINT_ISAAC_SIM_MAJOR_VERSION", "").strip()
    build_method = env.get("BLUEPRINT_WORKER_IMAGE_BUILD_METHOD", "").strip()
    base_image_digest = env.get("BLUEPRINT_WORKER_BASE_IMAGE_DIGEST", "").strip()
    source_manifest_sha256 = env.get(
        "BLUEPRINT_WORKER_SOURCE_MANIFEST_SHA256", ""
    ).strip()
    source_layer_digest = env.get("BLUEPRINT_WORKER_SOURCE_LAYER_DIGEST", "").strip()
    try:
        isaac_major = int(isaac_major_raw)
    except ValueError:
        isaac_major = None
    blockers = []
    if not _SOURCE_COMMIT_RE.fullmatch(source_commit):
        blockers.append("worker_image_source_commit_missing_or_invalid")
    if not _SHA256_RE.fullmatch(dirty_patch):
        blockers.append("worker_image_source_dirty_patch_missing_or_invalid")
    if family != "isaac-eval-worker":
        blockers.append("worker_image_family_invalid")
    if isaac_major != 6:
        blockers.append("worker_image_isaac_major_invalid")
    if build_method and build_method != _SOURCE_OVERLAY_BUILD_METHOD:
        blockers.append("worker_image_build_method_invalid")
    if build_method == _SOURCE_OVERLAY_BUILD_METHOD:
        if _DIGEST_REF_RE.fullmatch(base_image_digest) is None:
            blockers.append("worker_image_overlay_base_digest_invalid")
        if _DIGEST_VALUE_RE.fullmatch(source_manifest_sha256) is None:
            blockers.append("worker_image_overlay_source_manifest_digest_invalid")
        if _DIGEST_VALUE_RE.fullmatch(source_layer_digest) is None:
            blockers.append("worker_image_overlay_source_layer_digest_invalid")
    return {
        "status": "verified" if not blockers else "blocked",
        "blockers": blockers,
        "source_commit": source_commit or None,
        "source_dirty_patch_sha256": dirty_patch or None,
        "worker_image_family": family or None,
        "isaac_sim_major_version": isaac_major,
        "build_method": build_method or "dockerfile_full_build",
        "base_image_digest": base_image_digest or None,
        "source_manifest_sha256": source_manifest_sha256 or None,
        "source_layer_digest": source_layer_digest or None,
        "source_layer_matches_last_registry_layer": None,
        "identity_source": "immutable_registry_image_config_environment",
    }


def summarize_manifest(
    *,
    image_ref: str,
    raw_manifest: Mapping[str, Any],
    resolved_digest: str | None,
    runnable_platform: str | None = None,
    runnable_child_digest: str | None = None,
    image_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    layers = raw_manifest.get("layers")
    layers = layers if isinstance(layers, list) else []
    normalized: list[dict[str, Any]] = []
    sizes: list[int] = []
    invalid_layer_metadata = False
    for layer in layers:
        if not isinstance(layer, Mapping):
            invalid_layer_metadata = True
            normalized.append({"mediaType": None, "digest": None, "size": None})
            continue
        raw_size = layer.get("size")
        digest = layer.get("digest")
        size = raw_size if type(raw_size) is int and raw_size >= 0 else None
        digest_valid = isinstance(digest, str) and _DIGEST_VALUE_RE.fullmatch(digest)
        if size is None or not digest_valid:
            invalid_layer_metadata = True
        if size is not None:
            sizes.append(size)
        normalized.append(
            {
                "mediaType": layer.get("mediaType"),
                "digest": digest,
                "size": size,
            }
        )
    complete_layers = bool(layers) and not invalid_layer_metadata and len(sizes) == len(layers)
    total = sum(sizes) if complete_layers else None
    largest = max(sizes) if complete_layers else None
    digest_ref = None
    if resolved_digest and _DIGEST_VALUE_RE.fullmatch(resolved_digest):
        digest_ref = _digest_ref(image_ref, resolved_digest)
    layers_over_1gb = len([size for size in sizes if size >= 1_000_000_000])
    size_policy = derive_manifest_size_policy(
        total_compressed_size_bytes=total,
        largest_layer_size_bytes=largest,
        layers_over_1gb=layers_over_1gb,
    )
    config = image_config if isinstance(image_config, Mapping) else {}
    required_cuda_version, required_cuda_version_source = derive_required_cuda_version(
        config
    )
    worker_build_identity = derive_worker_build_identity(config)
    if worker_build_identity.get("build_method") == _SOURCE_OVERLAY_BUILD_METHOD:
        expected_source_layer = worker_build_identity.get("source_layer_digest")
        observed_source_layer = normalized[-1].get("digest") if normalized else None
        source_layer_matches = bool(
            expected_source_layer and expected_source_layer == observed_source_layer
        )
        worker_build_identity["source_layer_matches_last_registry_layer"] = (
            source_layer_matches
        )
        if not source_layer_matches:
            worker_build_identity["blockers"] = sorted(
                {
                    *worker_build_identity.get("blockers", []),
                    "worker_image_overlay_source_layer_not_last_registry_layer",
                }
            )
            worker_build_identity["status"] = "blocked"
    raw_history = config.get("history")
    raw_history = raw_history if isinstance(raw_history, list) else []
    nonempty_history = [
        item
        for item in raw_history
        if isinstance(item, Mapping) and item.get("empty_layer") is not True
    ]
    layer_history: list[dict[str, Any]] = []
    for index, layer in enumerate(normalized):
        history = nonempty_history[index] if index < len(nonempty_history) else {}
        created_by = str(history.get("created_by") or "")
        command = created_by.lower()
        checkpoint_chown = "chown" in command and "/opt/blueprint/ckpts" in command
        checkpoint_production = "snapshot_download" in command
        layer_history.append(
            {
                **layer,
                "index": index,
                "created_by": created_by or None,
                "checkpoint_ownership_only_copyup": bool(
                    checkpoint_chown and not checkpoint_production
                ),
            }
        )
    ownership_copyups = [
        item for item in layer_history if item["checkpoint_ownership_only_copyup"]
    ]
    history_available = bool(nonempty_history)
    blockers: list[str] = []
    if not complete_layers or not digest_ref:
        blockers.append("registry_manifest_metadata_incomplete")
    if invalid_layer_metadata:
        blockers.append("registry_manifest_layer_metadata_invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if not blockers else "blocked",
        "blockers": blockers,
        "image_ref": image_ref,
        "resolved_digest": resolved_digest,
        "resolved_digest_ref": digest_ref,
        "runnable_platform": runnable_platform,
        "runnable_child_digest": runnable_child_digest,
        "required_cuda_version": required_cuda_version,
        "required_cuda_version_source": required_cuda_version_source,
        "worker_build_identity": worker_build_identity,
        "history_available": history_available,
        "nonempty_history_count": len(nonempty_history),
        "history_layer_count_matches": bool(
            history_available and len(nonempty_history) == len(normalized)
        ),
        "checkpoint_ownership_copyup_detected": (
            bool(ownership_copyups) if history_available else None
        ),
        "checkpoint_ownership_copyup_layers": ownership_copyups,
        "mediaType": raw_manifest.get("mediaType"),
        "layer_count": len(layers),
        "measured_layer_count": len(sizes),
        "total_compressed_size_bytes": total,
        "largest_layer_size_bytes": largest,
        "layers_over_1gb": layers_over_1gb,
        **size_policy,
        "layers": normalized,
        "layer_history": layer_history,
        "raw_secret_values_recorded": False,
        "claim_boundary": (
            "Registry digest and compressed-layer metadata only. This does not prove provider "
            "runtime startup, CUDA health, Isaac execution, rendering, or task success."
        ),
    }


def _inspect_local_image(image_ref: str) -> dict[str, Any] | None:
    inspected = subprocess.run(
        ["docker", "image", "inspect", image_ref],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if inspected.returncode != 0:
        return None
    decoded = json.loads(inspected.stdout)
    if not isinstance(decoded, list) or not decoded or not isinstance(decoded[0], Mapping):
        return None
    image = decoded[0]
    root_fs = image.get("RootFS") if isinstance(image.get("RootFS"), Mapping) else {}
    layers = root_fs.get("Layers") if isinstance(root_fs, Mapping) else []
    layers = layers if isinstance(layers, list) else []
    config = image.get("Config") if isinstance(image.get("Config"), Mapping) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_local",
        "blockers": [],
        "image_ref": image_ref,
        "local_image_id": image.get("Id"),
        "resolved_digest": None,
        "resolved_digest_ref": None,
        "layer_count": len(layers),
        "total_compressed_size_bytes": None,
        "largest_layer_size_bytes": None,
        "local_uncompressed_size_bytes": image.get("Size"),
        "architecture": image.get("Architecture"),
        "os": image.get("Os"),
        "runtime_config": {
            "user": config.get("User") if isinstance(config, Mapping) else None,
            "entrypoint": config.get("Entrypoint") if isinstance(config, Mapping) else None,
            "working_dir": config.get("WorkingDir") if isinstance(config, Mapping) else None,
        },
        "raw_secret_values_recorded": False,
        "claim_boundary": (
            "Local image configuration and uncompressed size only. No registry digest or "
            "compressed-layer metadata is claimed, and this does not prove provider startup."
        ),
    }


def inspect_image_manifest(
    *, image_ref: str, output_path: Path, allow_local: bool = False
) -> dict[str, Any]:
    # Resolve a mutable tag exactly once. Every metadata query after this point
    # uses the immutable digest ref so layers, config history, and timeout
    # policy cannot be spliced across two tag versions.
    plain = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", image_ref],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    match = _DIGEST_RE.search(plain.stdout) if plain.returncode == 0 else None
    exact_ref = _digest_ref(image_ref, match.group(1)) if match else image_ref
    raw = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", "--raw", exact_ref],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if raw.returncode != 0 or plain.returncode != 0:
        local_payload = _inspect_local_image(image_ref) if allow_local else None
        payload = local_payload or {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "blocked",
            "blockers": ["worker_image_manifest_inspection_failed"],
            "image_ref": image_ref,
            "raw_secret_values_recorded": False,
            "stderr_tail": (raw.stderr + plain.stderr)[-1000:],
            "claim_boundary": "Registry inspection failed; no startup claim is available.",
        }
    else:
        try:
            decoded = json.loads(raw.stdout)
        except json.JSONDecodeError:
            decoded = {}
        child_digest = _runnable_child_digest(decoded)
        repository = exact_ref.split("@", 1)[0]
        platform_inspect_ref = (
            f"{repository}@{child_digest}" if child_digest else exact_ref
        )
        formatted = subprocess.run(
            [
                "docker",
                "buildx",
                "imagetools",
                "inspect",
                "--format",
                "{{json .}}",
                platform_inspect_ref,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        formatted_payload: dict[str, Any] = {}
        if formatted.returncode == 0:
            try:
                candidate = json.loads(formatted.stdout)
                if isinstance(candidate, dict):
                    formatted_payload = candidate
            except json.JSONDecodeError:
                formatted_payload = {}
        plain_platforms = re.findall(r"Platform:\s*(\S+)", plain.stdout)
        runnable_plain_platforms = [p for p in plain_platforms if p == RUNNABLE_PLATFORM]
        runnable_platform = (
            RUNNABLE_PLATFORM
            if child_digest or len(runnable_plain_platforms) == 1
            else None
        )
        if not isinstance(decoded.get("layers"), list) and child_digest:
            child = subprocess.run(
                [
                    "docker",
                    "buildx",
                    "imagetools",
                    "inspect",
                    "--raw",
                    f"{repository}@{child_digest}",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if child.returncode == 0:
                try:
                    decoded = json.loads(child.stdout)
                except json.JSONDecodeError:
                    decoded = {}
        payload = summarize_manifest(
            image_ref=image_ref,
            raw_manifest=decoded,
            resolved_digest=match.group(1) if match else None,
            runnable_platform=runnable_platform,
            runnable_child_digest=child_digest,
            image_config=(
                formatted_payload.get("image")
                if isinstance(formatted_payload.get("image"), Mapping)
                else None
            ),
        )
        payload["registry_inspection_ref"] = exact_ref if match else None
        if match is None:
            payload["status"] = "blocked"
            payload["blockers"] = sorted(
                {*payload.get("blockers", []), "registry_manifest_digest_resolution_failed"}
            )
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-local", action="store_true")
    args = parser.parse_args(argv)
    result = inspect_image_manifest(
        image_ref=args.image,
        output_path=args.output,
        allow_local=args.allow_local,
    )
    print(json.dumps({k: result.get(k) for k in ("status", "resolved_digest_ref", "layer_count", "total_compressed_size_bytes", "largest_layer_size_bytes", "recommended_startup_no_runtime_timeout_seconds")}, sort_keys=True))
    return 0 if result.get("status") in {"completed", "completed_local"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
