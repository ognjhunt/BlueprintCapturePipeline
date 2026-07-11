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
LARGE_TOTAL_BYTES = 8_000_000_000
LARGE_LAYER_BYTES = 3_000_000_000
_DIGEST_RE = re.compile(r"^Digest:\s*(sha256:[0-9a-f]{64})\s*$", re.MULTILINE)


def summarize_manifest(
    *, image_ref: str, raw_manifest: Mapping[str, Any], resolved_digest: str | None
) -> dict[str, Any]:
    layers = raw_manifest.get("layers")
    layers = layers if isinstance(layers, list) else []
    normalized: list[dict[str, Any]] = []
    sizes: list[int] = []
    for layer in layers:
        if not isinstance(layer, Mapping):
            continue
        raw_size = layer.get("size")
        size = int(raw_size) if isinstance(raw_size, (int, float)) and raw_size >= 0 else None
        if size is not None:
            sizes.append(size)
        normalized.append(
            {
                "mediaType": layer.get("mediaType"),
                "digest": layer.get("digest"),
                "size": size,
            }
        )
    total = sum(sizes) if sizes else None
    largest = max(sizes) if sizes else None
    digest_ref = None
    if resolved_digest:
        repository = str(image_ref).split("@", 1)[0]
        if "/" in repository and ":" in repository.rsplit("/", 1)[-1]:
            repository = repository.rsplit(":", 1)[0]
        digest_ref = f"{repository}@{resolved_digest}"
    split_suitable = bool(
        total is not None
        and largest is not None
        and largest < LARGE_LAYER_BYTES
        and len([size for size in sizes if size >= 1_000_000_000]) >= 2
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if sizes and resolved_digest else "blocked",
        "blockers": [] if sizes and resolved_digest else ["registry_manifest_metadata_incomplete"],
        "image_ref": image_ref,
        "resolved_digest": resolved_digest,
        "resolved_digest_ref": digest_ref,
        "mediaType": raw_manifest.get("mediaType"),
        "layer_count": len(sizes),
        "total_compressed_size_bytes": total,
        "largest_layer_size_bytes": largest,
        "layers_over_1gb": len([size for size in sizes if size >= 1_000_000_000]),
        "split_layer_layout_suitable": split_suitable,
        "large_image_pull_risk": bool(
            (total is not None and total >= LARGE_TOTAL_BYTES)
            or (largest is not None and largest >= LARGE_LAYER_BYTES)
        ),
        "recommended_startup_no_runtime_timeout_seconds": (
            max(600, min(1800, int((total / 8_000_000) + 120))) if total is not None else None
        ),
        "layers": normalized,
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
    raw = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", "--raw", image_ref],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    plain = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", image_ref],
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
        match = _DIGEST_RE.search(plain.stdout)
        payload = summarize_manifest(
            image_ref=image_ref,
            raw_manifest=json.loads(raw.stdout),
            resolved_digest=match.group(1) if match else None,
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
