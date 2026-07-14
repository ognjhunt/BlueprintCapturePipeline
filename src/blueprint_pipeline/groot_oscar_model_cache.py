"""Immutable external model-cache contract for the GR00T + OSCAR worker.

The cache lives on a provider volume or host model cache.  It is deliberately
not an OCI image layer.  Preparation may use the network; verification never
does and binds readiness to the SHA-256 of every required model byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import write_json

SCHEMA_VERSION = "groot_oscar_external_model_cache.v1"
MANIFEST_NAME = "groot_oscar_model_cache_manifest.json"
MODEL_PINS: tuple[tuple[str, str, str], ...] = (
    ("sonic", "LucaFrat/groot-bs16", "86b17337379926a8d8f1ad5c4580c7c33deeb49f"),
    ("cosmos", "nvidia/Cosmos-Reason2-2B", "9ce19a195e423419c349abfc86fd07178b230561"),
    ("oscar", "zywu2115/OSCAR-2B", "c9781ffa7dd8556d862d7d9f338a2ea008a58ca6"),
    ("gear_sonic", "nvidia/GEAR-SONIC", "5e22ddc69abcea2a9aafc40536b14c232d3f9d7f"),
)
GEAR_FILES = (
    "model_encoder.onnx",
    "model_decoder.onnx",
    "observation_config.yaml",
    "planner_sonic.onnx",
)
_IMMUTABLE_HF_REVISION = re.compile(r"[0-9a-f]{40}")


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "manifest_digest"}
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _model_files(root: Path) -> Iterable[Path]:
    for model_name, _, _ in MODEL_PINS:
        model_root = root / model_name
        if model_root.is_dir():
            yield from sorted(path for path in model_root.rglob("*") if path.is_file())


def build_manifest(root: Path) -> dict[str, Any]:
    """Inventory and hash a fully prepared cache.  Missing models fail closed."""

    root = root.expanduser().resolve()
    missing = [
        name
        for name, _, _ in MODEL_PINS
        if not any(path.is_file() for path in (root / name).rglob("*"))
    ]
    if missing:
        raise ValueError("model_cache_required_roots_missing:" + ",".join(missing))
    files = []
    for path in _model_files(root):
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"model_cache_file_escapes_root:{path}")
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not files:
        raise ValueError("model_cache_has_no_files")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repositories": [
            {"name": name, "repo_id": repo_id, "revision": revision}
            for name, repo_id, revision in MODEL_PINS
        ],
        "files": files,
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "network_download_allowed_at_runtime": False,
        "models_embedded_in_release_image": False,
    }
    payload["manifest_digest"] = _canonical_digest(payload)
    return payload


def verify_model_cache(root: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    """Verify pins, inventory, byte hashes, and manifest self-digest offline."""

    root = root.expanduser().resolve()
    manifest_path = (manifest_path or root / MANIFEST_NAME).expanduser().resolve()
    blockers: list[str] = []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        payload = {}
        blockers.append("model_cache_manifest_unreadable")
    expected_repositories = [
        {"name": name, "repo_id": repo_id, "revision": revision}
        for name, repo_id, revision in MODEL_PINS
    ]
    if payload.get("schema_version") != SCHEMA_VERSION:
        blockers.append("model_cache_manifest_schema_invalid")
    if payload.get("repositories") != expected_repositories:
        blockers.append("model_cache_repository_pins_mismatch")
    if payload.get("network_download_allowed_at_runtime") is not False:
        blockers.append("model_cache_runtime_network_policy_invalid")
    observed_digest = str(payload.get("manifest_digest") or "")
    if observed_digest != _canonical_digest(payload):
        blockers.append("model_cache_manifest_digest_mismatch")

    entries = payload.get("files")
    entries = entries if isinstance(entries, list) else []
    declared_paths: set[str] = set()
    verified_bytes = 0
    for entry in entries:
        if not isinstance(entry, dict):
            blockers.append("model_cache_file_entry_invalid")
            continue
        relative = str(entry.get("path") or "")
        if not relative or relative in declared_paths:
            blockers.append("model_cache_file_path_invalid_or_duplicate")
            continue
        declared_paths.add(relative)
        path = root / relative
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root) or not path.is_file():
                raise OSError
            size = path.stat().st_size
        except OSError:
            blockers.append(f"model_cache_file_missing:{relative}")
            continue
        if size != entry.get("size_bytes"):
            blockers.append(f"model_cache_file_size_mismatch:{relative}")
            continue
        if _sha256(path) != entry.get("sha256"):
            blockers.append(f"model_cache_file_digest_mismatch:{relative}")
            continue
        verified_bytes += size

    actual_paths = {path.relative_to(root).as_posix() for path in _model_files(root)}
    if actual_paths != declared_paths:
        blockers.append("model_cache_inventory_mismatch")
    if payload.get("file_count") != len(entries):
        blockers.append("model_cache_file_count_mismatch")
    if payload.get("total_size_bytes") != sum(
        int(entry.get("size_bytes") or 0) for entry in entries if isinstance(entry, dict)
    ):
        blockers.append("model_cache_total_size_mismatch")
    return {
        "schema_version": "groot_oscar_external_model_cache_verification.v1",
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "model_manifest_digest": observed_digest or None,
        "verified_file_count": len(declared_paths) if not blockers else 0,
        "verified_size_bytes": verified_bytes if not blockers else 0,
        "checks": {"models_cached_offline": not blockers},
        "raw_secret_values_recorded": False,
    }


def activate_model_cache(root: Path) -> dict[str, Any]:
    """Verify the cache, then expose model-only WBC assets at runtime paths."""

    verification = verify_model_cache(root)
    if verification["status"] != "passed":
        return verification
    root = root.expanduser().resolve()
    links = {
        root / "gear_sonic/model_encoder.onnx": Path(
            "/opt/wbc/gear_sonic_deploy/policy/release/model_encoder.onnx"
        ),
        root / "gear_sonic/model_decoder.onnx": Path(
            "/opt/wbc/gear_sonic_deploy/policy/release/model_decoder.onnx"
        ),
        root / "gear_sonic/observation_config.yaml": Path(
            "/opt/wbc/gear_sonic_deploy/policy/release/observation_config.yaml"
        ),
        root / "gear_sonic/planner_sonic.onnx": Path(
            "/opt/wbc/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"
        ),
    }
    for source, destination in links.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source)
    return {**verification, "runtime_links_activated": True}


def prepare_model_cache(root: Path, *, token: str | None = None) -> dict[str, Any]:
    """Download pinned snapshots to a provider cache, then write byte evidence."""

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:  # pragma: no cover - operational dependency
        raise RuntimeError("huggingface_hub_required_to_prepare_model_cache") from exc

    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    pins = {name: (repo, revision) for name, repo, revision in MODEL_PINS}
    invalid_pins = sorted(
        name
        for name, (_, revision) in pins.items()
        if _IMMUTABLE_HF_REVISION.fullmatch(revision) is None
    )
    if invalid_pins:
        raise RuntimeError("model_cache_revisions_must_be_commit_shas:" + ",".join(invalid_pins))
    # Each revision is validated as an immutable commit SHA above.
    snapshot_download(  # nosec B615
        repo_id=pins["sonic"][0],
        revision=pins["sonic"][1],
        local_dir=str(root / "sonic"),
        ignore_patterns=["checkpoint-*", "checkpoint-*/*"],
        token=token,
    )
    cosmos_source = Path(
        snapshot_download(  # nosec B615
            repo_id=pins["cosmos"][0],
            revision=pins["cosmos"][1],
            token=token,
        )
    )
    cosmos = root / "cosmos"
    if cosmos.exists():
        shutil.rmtree(cosmos)
    shutil.copytree(cosmos_source, cosmos, symlinks=False)
    snapshot_download(  # nosec B615
        repo_id=pins["oscar"][0],
        revision=pins["oscar"][1],
        local_dir=str(root / "oscar"),
        token=token,
    )
    gear = root / "gear_sonic"
    gear.mkdir(parents=True, exist_ok=True)
    for filename in GEAR_FILES:
        source = Path(
            hf_hub_download(  # nosec B615
                repo_id=pins["gear_sonic"][0],
                revision=pins["gear_sonic"][1],
                filename=filename,
                token=token,
            )
        )
        shutil.copyfile(source, gear / filename)

    sonic_config_path = root / "sonic" / "config.json"
    sonic_config = json.loads(sonic_config_path.read_text(encoding="utf-8"))
    if sonic_config.get("model_name") != pins["cosmos"][0]:
        raise RuntimeError("unexpected_sonic_nested_model")
    sonic_config["blueprint_original_model_name"] = pins["cosmos"][0]
    sonic_config["blueprint_model_revision"] = pins["cosmos"][1]
    sonic_config["model_name"] = str(root / "cosmos")
    sonic_config_path.write_text(
        json.dumps(sonic_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = build_manifest(root)
    write_json(root / MANIFEST_NAME, manifest)
    return manifest


def _token_from_file(path: str) -> str | None:
    if not path:
        return None
    token_path = Path(path).expanduser()
    return token_path.read_text(encoding="utf-8").strip() if token_path.is_file() else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "manifest", "activate"))
    parser.add_argument(
        "--root",
        default=os.environ.get(
            "BLUEPRINT_GROOT_OSCAR_MODEL_CACHE", "/models/blueprint-groot-oscar-v1"
        ),
    )
    parser.add_argument("--manifest")
    parser.add_argument("--token-file", default=os.environ.get("HF_TOKEN_FILE", ""))
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    root = Path(args.root)
    if args.command == "prepare":
        result = prepare_model_cache(root, token=_token_from_file(args.token_file))
    elif args.command == "manifest":
        result = build_manifest(root)
        write_json(Path(args.manifest) if args.manifest else root / MANIFEST_NAME, result)
    elif args.command == "activate":
        result = activate_model_cache(root)
    else:
        result = verify_model_cache(root, Path(args.manifest) if args.manifest else None)
    if args.out:
        write_json(Path(args.out), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status", "passed") != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
