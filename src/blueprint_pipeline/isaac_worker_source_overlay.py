"""Prepare and verify a deterministic source overlay for an Isaac worker image.

The overlay is an experimental-canary build path for updating a previously
qualified full Blueprint Isaac worker without downloading or rebuilding its
large Isaac foundation layers.  It is deliberately narrower than a clean
Dockerfile release build: registry verification proves exact source and layer
lineage only; a fresh paid startup/render canary is still required.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import stat
import subprocess
import tarfile
from typing import Any, Mapping, Sequence

from .common import ensure_dir, write_json
from .decision_evidence_contracts import canonical_digest
from .g1_kitchen_bundle_compatibility import (
    CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
)
from .isaac_worker_image_manifest import summarize_manifest


SCHEMA_VERSION = "isaac_worker_source_overlay.v1"
REGISTRY_RESULT_SCHEMA_VERSION = "isaac_worker_source_overlay_registry_result.v1"
BUILD_METHOD = "crane_exact_source_overlay_experimental"
DEFAULT_BASE_IMAGE_DIGEST = (
    "docker.io/nijelhunt/blueprint-isaac-eval-worker@sha256:"
    "865633c0bb99058ce15dd1d977876bf931b73f6b04057c94f5235dc701ea0a91"
)
LAYER_NAME = "isaac_worker_source_overlay.tar.gz"
PLAN_NAME = "isaac_worker_source_overlay.v1.json"
REGISTRY_RESULT_NAME = "isaac_worker_source_overlay_registry_result.v1.json"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_DIGEST_REF = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
_CONTEXT_ROOTS = (
    "src/blueprint_pipeline",
    "pyproject.toml",
    "README.md",
    "LICENSE",
)


class IsaacWorkerSourceOverlayError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _git(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise IsaacWorkerSourceOverlayError(
            ["isaac_worker_source_overlay_git_query_failed"]
        ) from exc


def _versioned_tag(value: str) -> bool:
    leaf = value.rsplit("/", 1)[-1]
    name, separator, tag = leaf.rpartition(":")
    return bool(
        name
        and separator
        and _TAG.fullmatch(tag)
        and tag not in {"latest", "dev", "test", "local"}
        and "@" not in value
        and not any(character.isspace() for character in value)
    )


def _tracked_sources(root: Path) -> list[Path]:
    encoded = _git(root, "ls-files", "-z", "--", *_CONTEXT_ROOTS)
    paths = [Path(value.decode("utf-8")) for value in encoded.split(b"\0") if value]
    blockers: list[str] = []
    if not paths:
        blockers.append("isaac_worker_source_overlay_inventory_empty")
    for relative in paths:
        source = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or source.is_symlink()
            or not source.is_file()
        ):
            blockers.append("isaac_worker_source_overlay_inventory_unsafe")
    if blockers:
        raise IsaacWorkerSourceOverlayError(blockers)
    return sorted(set(paths), key=lambda value: value.as_posix())


def _tar_info(name: str, *, mode: int, size: int = 0, directory: bool = False) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = mode
    info.size = size
    if directory:
        info.type = tarfile.DIRTYPE
    return info


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes, *, mode: int) -> None:
    archive.addfile(_tar_info(name, mode=mode, size=len(payload)), io.BytesIO(payload))


def _layer_member_name(relative: Path) -> str:
    return (Path("app") / relative).as_posix()


def prepare_source_overlay(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    base_image_digest: str,
    target_image_ref: str,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Create a reproducible source layer and non-authorizing build plan."""

    root = Path(repo_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    ensure_dir(output)
    head = _git(root, "rev-parse", "HEAD").decode().strip().lower()
    requested_commit = str(source_commit or head).strip().lower()
    dirty = bool(_git(root, "status", "--porcelain=v1", "--untracked-files=all"))
    blockers: list[str] = []
    if not root.is_dir():
        blockers.append("isaac_worker_source_overlay_repo_missing")
    if _DIGEST_REF.fullmatch(str(base_image_digest or "")) is None:
        blockers.append("isaac_worker_source_overlay_base_not_digest_pinned")
    if not _versioned_tag(str(target_image_ref or "")):
        blockers.append("isaac_worker_source_overlay_target_not_versioned")
    if _COMMIT.fullmatch(requested_commit) is None or requested_commit != head:
        blockers.append("isaac_worker_source_overlay_source_commit_mismatch")
    if dirty:
        blockers.append("isaac_worker_source_overlay_requires_clean_worktree")
    if blockers:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blockers": sorted(set(blockers)),
            "source_commit": requested_commit or None,
            "source_dirty_patch_sha256": CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
            "base_image_digest": base_image_digest or None,
            "target_image_ref": target_image_ref or None,
            "provider_mutations_performed": 0,
            "proof_effect": "none",
            "claim_ceiling": "build_plan_only",
        }

    sources = _tracked_sources(root)
    inventory = []
    for relative in sources:
        source = root / relative
        payload = source.read_bytes()
        executable = bool(source.stat().st_mode & stat.S_IXUSR)
        inventory.append(
            {
                "source_path": relative.as_posix(),
                "layer_path": _layer_member_name(relative),
                "sha256": _sha256_bytes(payload),
                "bytes": len(payload),
                "mode": "0755" if executable else "0644",
            }
        )
    source_manifest_sha256 = canonical_digest({"files": inventory})
    embedded_receipt = {
        "schema_version": SCHEMA_VERSION,
        "build_method": BUILD_METHOD,
        "source_commit": requested_commit,
        "source_dirty_patch_sha256": CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
        "base_image_digest": base_image_digest,
        "target_image_ref": target_image_ref,
        "source_manifest_sha256": source_manifest_sha256,
        "source_file_count": len(inventory),
        "lower_source_tree_hidden_by_opaque_whiteout": True,
        "dependency_layers_inherited_from_pinned_full_worker": True,
        "runtime_canary_required": True,
        "provider_mutations_performed": 0,
        "proof_effect": "none",
        "claim_ceiling": "exact_source_layer_preparation_only",
    }
    receipt_payload = (
        json.dumps(embedded_receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    layer_path = output / LAYER_NAME
    with layer_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for directory in ("app", "app/src", "opt", "opt/blueprint"):
                    archive.addfile(_tar_info(directory, mode=0o755, directory=True))
                _add_bytes(archive, "app/src/.wh..wh..opq", b"", mode=0o000)
                for relative, row in zip(sources, inventory, strict=True):
                    payload = (root / relative).read_bytes()
                    _add_bytes(
                        archive,
                        str(row["layer_path"]),
                        payload,
                        mode=int(str(row["mode"]), 8),
                    )
                _add_bytes(
                    archive,
                    "opt/blueprint/isaac_worker_source_overlay.v1.json",
                    receipt_payload,
                    mode=0o644,
                )
    plan = {
        **embedded_receipt,
        "status": "ready",
        "blockers": [],
        "source_inventory": inventory,
        "layer_path": str(layer_path.resolve()),
        "layer_sha256": _sha256(layer_path),
        "layer_bytes": layer_path.stat().st_size,
        "layer_member_count": len(inventory) + 6,
        "expected_final_environment": {
            "BLUEPRINT_SOURCE_COMMIT": requested_commit,
            "BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256": CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
            "BLUEPRINT_WORKER_IMAGE_FAMILY": "isaac-eval-worker",
            "BLUEPRINT_ISAAC_SIM_MAJOR_VERSION": "6",
            "BLUEPRINT_WORKER_IMAGE_BUILD_METHOD": BUILD_METHOD,
            "BLUEPRINT_WORKER_BASE_IMAGE_DIGEST": base_image_digest,
            "BLUEPRINT_WORKER_SOURCE_MANIFEST_SHA256": source_manifest_sha256,
        },
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
        "proof_effect": "none",
        "claim_ceiling": "exact_source_layer_preparation_only",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    write_json(output / PLAN_NAME, plan)
    return plan


def _environment(config: Mapping[str, Any]) -> dict[str, str]:
    raw = config.get("config")
    raw = raw if isinstance(raw, Mapping) else {}
    values = raw.get("Env")
    result: dict[str, str] = {}
    if isinstance(values, list):
        for item in values:
            if isinstance(item, str) and "=" in item:
                key, value = item.split("=", 1)
                result[key] = value
    return result


def verify_registry_overlay(
    *,
    plan: Mapping[str, Any],
    base_manifest: Mapping[str, Any],
    final_manifest: Mapping[str, Any],
    final_config: Mapping[str, Any],
    resolved_image_digest: str,
) -> dict[str, Any]:
    """Verify final registry layers and config against the prepared plan."""

    prepared = json.loads(json.dumps(dict(plan)))
    base_layers = base_manifest.get("layers")
    final_layers = final_manifest.get("layers")
    base_layers = base_layers if isinstance(base_layers, list) else []
    final_layers = final_layers if isinstance(final_layers, list) else []
    environment = _environment(final_config)
    blockers: list[str] = []
    if prepared.get("status") != "ready" or prepared.get("plan_digest") != canonical_digest(
        prepared, digest_field="plan_digest"
    ):
        blockers.append("isaac_worker_source_overlay_plan_invalid")
    if not base_layers or len(final_layers) != len(base_layers) + 1:
        blockers.append("isaac_worker_source_overlay_layer_count_invalid")
    elif final_layers[:-1] != base_layers:
        blockers.append("isaac_worker_source_overlay_base_layer_prefix_mismatch")
    source_layer_digest = (
        str(final_layers[-1].get("digest") or "") if final_layers else ""
    )
    if _DIGEST.fullmatch(source_layer_digest) is None:
        blockers.append("isaac_worker_source_overlay_registry_layer_digest_invalid")
    expected_environment = prepared.get("expected_final_environment")
    expected_environment = (
        expected_environment if isinstance(expected_environment, Mapping) else {}
    )
    for key, expected in expected_environment.items():
        if environment.get(str(key)) != str(expected):
            blockers.append(f"isaac_worker_source_overlay_environment_mismatch:{key}")
    if environment.get("BLUEPRINT_WORKER_SOURCE_LAYER_DIGEST") != source_layer_digest:
        blockers.append("isaac_worker_source_overlay_layer_environment_mismatch")
    if _DIGEST.fullmatch(str(resolved_image_digest or "")) is None:
        blockers.append("isaac_worker_source_overlay_final_digest_invalid")
    config_value = final_config.get("config")
    config_value = config_value if isinstance(config_value, Mapping) else {}
    if (
        final_config.get("architecture") != "amd64"
        or final_config.get("os") != "linux"
        or config_value.get("User") not in {"blueprint", "blueprint:blueprint"}
        or config_value.get("WorkingDir") != "/workspace"
        or config_value.get("Entrypoint") != ["blueprint-run-robot-eval-worker"]
    ):
        blockers.append("isaac_worker_source_overlay_runtime_config_drift")
    result = {
        "schema_version": REGISTRY_RESULT_SCHEMA_VERSION,
        "status": "verified" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "build_method": BUILD_METHOD,
        "resolved_image_digest": resolved_image_digest or None,
        "source_commit": prepared.get("source_commit"),
        "source_dirty_patch_sha256": prepared.get("source_dirty_patch_sha256"),
        "base_image_digest": prepared.get("base_image_digest"),
        "source_manifest_sha256": prepared.get("source_manifest_sha256"),
        "source_layer_digest": source_layer_digest or None,
        "base_layer_count": len(base_layers),
        "final_layer_count": len(final_layers),
        "base_layers_preserved_exactly": bool(
            base_layers and final_layers[:-1] == base_layers
        ),
        "runtime_config_preserved": "runtime_config_drift" not in " ".join(blockers),
        "runtime_canary_completed": False,
        "provider_startup_proven": False,
        "isaac_compatibility_proven": False,
        "raw_secret_values_recorded": False,
        "proof_effect": "none",
        "claim_ceiling": "immutable_registry_source_lineage_only",
    }
    result["registry_result_digest"] = canonical_digest(
        result, digest_field="registry_result_digest"
    )
    return result


def build_image_manifest_from_registry_evidence(
    *,
    image_ref: str,
    resolved_image_digest: str,
    final_manifest: Mapping[str, Any],
    final_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical image diagnostic from crane-fetched registry JSON."""

    return summarize_manifest(
        image_ref=image_ref,
        raw_manifest=final_manifest,
        resolved_digest=resolved_image_digest,
        runnable_platform="linux/amd64",
        image_config=final_config,
    )


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise IsaacWorkerSourceOverlayError(
            ["isaac_worker_source_overlay_json_object_required"]
        )
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repo-root", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--base-image", default=DEFAULT_BASE_IMAGE_DIGEST)
    prepare.add_argument("--target-image", required=True)
    prepare.add_argument("--source-commit")
    verify = subparsers.add_parser("verify-registry")
    verify.add_argument("--plan", required=True)
    verify.add_argument("--base-manifest", required=True)
    verify.add_argument("--final-manifest", required=True)
    verify.add_argument("--final-config", required=True)
    verify.add_argument("--resolved-digest", required=True)
    verify.add_argument("--output", required=True)
    manifest = subparsers.add_parser("write-image-manifest")
    manifest.add_argument("--image-ref", required=True)
    manifest.add_argument("--final-manifest", required=True)
    manifest.add_argument("--final-config", required=True)
    manifest.add_argument("--resolved-digest", required=True)
    manifest.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_source_overlay(
            repo_root=args.repo_root,
            output_dir=args.output_dir,
            base_image_digest=args.base_image,
            target_image_ref=args.target_image,
            source_commit=args.source_commit,
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("status") == "ready" else 2
    if args.command == "verify-registry":
        result = verify_registry_overlay(
            plan=_load(args.plan),
            base_manifest=_load(args.base_manifest),
            final_manifest=_load(args.final_manifest),
            final_config=_load(args.final_config),
            resolved_image_digest=args.resolved_digest,
        )
        success = result.get("status") == "verified"
    else:
        result = build_image_manifest_from_registry_evidence(
            image_ref=args.image_ref,
            resolved_image_digest=args.resolved_digest,
            final_manifest=_load(args.final_manifest),
            final_config=_load(args.final_config),
        )
        success = (
            result.get("status") == "completed"
            and (result.get("worker_build_identity") or {}).get("status") == "verified"
        )
    write_json(Path(args.output), result)
    print(json.dumps(result, sort_keys=True))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
