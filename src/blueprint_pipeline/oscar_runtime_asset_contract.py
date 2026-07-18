"""Pinned runtime assets required by the public OSCAR inference path.

The public OSCAR checkpoint is not a self-contained runtime.  Model
construction also opens the Wan2.1 VAE and the Cosmos-Reason1-7B text encoder.
This module binds those two assets to immutable Hugging Face revisions and
provides a prepare-once cache plus an offline, byte-verifying preflight.

Preparation may use the network.  Preflight never does.  Neither operation is
OSCAR inference, generated-video proof, Isaac execution, or task-success proof.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


ASSET_CONTRACT_SCHEMA_VERSION = "oscar_runtime_asset_contract.v1"
ASSET_EVIDENCE_SCHEMA_VERSION = "oscar_runtime_asset_evidence.v1"
OFFLINE_PREFLIGHT_SCHEMA_VERSION = "oscar_runtime_asset_offline_preflight.v1"

DEFAULT_RUNTIME_MODEL_CACHE_ROOT = Path("/workspace/runtime_model_cache")
DEFAULT_OSCAR_CHECKPOINT_ROOT = Path("/opt/blueprint/ckpts/oscar")
RUNTIME_ASSET_EVIDENCE_NAME = "oscar_runtime_asset_evidence.json"
OFFLINE_PREFLIGHT_EVIDENCE_NAME = "oscar_runtime_asset_offline_preflight.json"

OSCAR_SOURCE_REPO_ID = "wuzy2115/oscar-public"
OSCAR_SOURCE_REVISION = "4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb"
OSCAR_CHECKPOINT_REPO_ID = "zywu2115/OSCAR-2B"
OSCAR_CHECKPOINT_REVISION = "c9781ffa7dd8556d862d7d9f338a2ea008a58ca6"

REASON1_REPO_ID = "nvidia/Cosmos-Reason1-7B"
REASON1_REVISION = "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"
WAN_VAE_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B"
WAN_VAE_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
WAN_VAE_FILENAME = "Wan2.1_VAE.pth"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IMMUTABLE_REVISION_RE = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class PinnedRuntimeFile:
    """Exact byte identity for one runtime file."""

    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        relative = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != self.relative_path
        ):
            raise ValueError(f"runtime_asset_relative_path_invalid:{self.relative_path}")
        if self.size_bytes <= 0:
            raise ValueError(f"runtime_asset_size_invalid:{self.relative_path}")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError(f"runtime_asset_sha256_invalid:{self.relative_path}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


# Complete immutable snapshot inventory at REASON1_REVISION.  OSCAR uses the
# processor files and every indexed model shard; the two repository metadata
# files are also bound so preparation cannot silently accept a partial snapshot.
REASON1_SNAPSHOT_FILES: tuple[PinnedRuntimeFile, ...] = (
    PinnedRuntimeFile(
        ".gitattributes",
        1_519,
        "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    ),
    PinnedRuntimeFile(
        "README.md",
        25_877,
        "46ff42e4649159a8950c8492a7e9050fb59a32e6f1da393cb4f5fda9b3c03231",
    ),
    PinnedRuntimeFile(
        "chat_template.json",
        1_050,
        "ad60d90252ed0b0705ba14e2d0ad0fec0beac1ea955642b54059b36052d8bc96",
    ),
    PinnedRuntimeFile(
        "config.json",
        1_374,
        "77d9ec7321cc572e3579e2c84799c9cadaded63c49ce93b101733349fc330c43",
    ),
    PinnedRuntimeFile(
        "generation_config.json",
        217,
        "6c2953cfed29c8e1c54c9a6f7712f628af92ae63698b016141e55aa72d808e6c",
    ),
    PinnedRuntimeFile(
        "model-00001-of-00004.safetensors",
        4_968_243_304,
        "c28404126221997ae8eb70a23b919c96174d42e35ae1d537e0c95093d50b359a",
    ),
    PinnedRuntimeFile(
        "model-00002-of-00004.safetensors",
        4_991_495_816,
        "f281081864c10992d3e03874c79d526c84407e049d713747f19eb9c79cd16db3",
    ),
    PinnedRuntimeFile(
        "model-00003-of-00004.safetensors",
        4_932_751_040,
        "3bb3a62a8d0e83c6283388ddea99395b221f908f9181b8edd0f7f91d02260ebe",
    ),
    PinnedRuntimeFile(
        "model-00004-of-00004.safetensors",
        1_691_924_384,
        "91bacbe1ad798e16daa05023b4e4bec70b53c8cd7d757db86c5bc76c4e0bbf15",
    ),
    PinnedRuntimeFile(
        "model.safetensors.index.json",
        57_618,
        "f442fe7207a37695f59a543282c9a3b67c2491d0d974b4da72cbac3a109aaa45",
    ),
    PinnedRuntimeFile(
        "preprocessor_config.json",
        350,
        "f2058c716eef96ccaed1cc1e2d0c08306b62586d535b28d9d08e691b2fab7ca0",
    ),
    PinnedRuntimeFile(
        "tokenizer.json",
        7_031_645,
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
    PinnedRuntimeFile(
        "tokenizer_config.json",
        5_702,
        "4abd3520120e266da84c0864fee064d1fb10806f02225911a47253dd38dc5f56",
    ),
)

WAN_VAE_FILE = PinnedRuntimeFile(
    WAN_VAE_FILENAME,
    507_609_880,
    "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
)

OSCAR_DCP_FILES: tuple[PinnedRuntimeFile, ...] = (
    PinnedRuntimeFile(
        "model/.metadata",
        186_118,
        "74ea9359634d4f757ebf1dcf1a6279f6c2980120762f9d5b214d6b95a3cfdac9",
    ),
    PinnedRuntimeFile(
        "model/__0_0.distcp",
        4_119_693_122,
        "51612eb9d1335b248f3112a206e72c2c3e84c774cee9cf2c459cb3ee62383cd5",
    ),
)

REASON1_SNAPSHOT_TOTAL_SIZE_BYTES = sum(item.size_bytes for item in REASON1_SNAPSHOT_FILES)
RUNTIME_AUXILIARY_ASSET_TOTAL_SIZE_BYTES = (
    REASON1_SNAPSHOT_TOTAL_SIZE_BYTES + WAN_VAE_FILE.size_bytes
)


@dataclass(frozen=True)
class RuntimeAssetPaths:
    cache_root: Path
    hf_home: Path
    reason1_snapshot: Path
    wan_vae: Path
    evidence: Path
    offline_preflight_evidence: Path


def _canonical_digest(payload: Mapping[str, Any], *, digest_key: str) -> str:
    unsigned = {key: value for key, value in payload.items() if key != digest_key}
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_absolute_root(root: Path | str) -> Path:
    candidate = Path(root).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"runtime_asset_cache_root_not_absolute:{candidate}")
    resolved = candidate.resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise ValueError("runtime_asset_cache_root_is_filesystem_root")
    return resolved


def _confined_path(candidate: Path, root: Path, *, strict: bool) -> Path:
    resolved_root = root.resolve(strict=strict)
    resolved = candidate.resolve(strict=strict)
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"runtime_asset_path_escapes_cache:{candidate}")
    return resolved


def _confined_lexical_file_path(candidate: Path, root: Path) -> Path:
    """Keep a snapshot filename while proving its parent and target are confined.

    Hugging Face snapshot files are normally symlinks into the repository's
    ``blobs`` directory. Returning ``Path.resolve()`` here loses the immutable
    snapshot filename that OSCAR expects. Validate the resolved locations, but
    retain the lexical snapshot path as the runtime contract value.
    """

    resolved_root = root.resolve(strict=False)
    resolved_parent = candidate.parent.resolve(strict=False)
    if not resolved_parent.is_relative_to(resolved_root):
        raise ValueError(f"runtime_asset_path_escapes_cache:{candidate}")
    if candidate.exists() or candidate.is_symlink():
        try:
            resolved_target = candidate.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"runtime_asset_path_unresolvable:{candidate}") from exc
        if not resolved_target.is_relative_to(resolved_root):
            raise ValueError(f"runtime_asset_path_escapes_cache:{candidate}")
    return candidate


def _hf_repo_cache_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def runtime_asset_paths(
    cache_root: Path | str = DEFAULT_RUNTIME_MODEL_CACHE_ROOT,
) -> RuntimeAssetPaths:
    root = _normalise_absolute_root(cache_root)
    hf_home = _confined_path(root / "hf_home", root, strict=False)
    hub = _confined_path(hf_home / "hub", root, strict=False)
    reason1 = _confined_path(
        hub / _hf_repo_cache_name(REASON1_REPO_ID) / "snapshots" / REASON1_REVISION,
        root,
        strict=False,
    )
    wan_vae = _confined_lexical_file_path(
        hub
        / _hf_repo_cache_name(WAN_VAE_REPO_ID)
        / "snapshots"
        / WAN_VAE_REVISION
        / WAN_VAE_FILENAME,
        root,
    )
    return RuntimeAssetPaths(
        cache_root=root,
        hf_home=hf_home,
        reason1_snapshot=reason1,
        wan_vae=wan_vae,
        evidence=_confined_path(root / RUNTIME_ASSET_EVIDENCE_NAME, root, strict=False),
        offline_preflight_evidence=_confined_path(
            root / OFFLINE_PREFLIGHT_EVIDENCE_NAME, root, strict=False
        ),
    )


def offline_runtime_environment(
    cache_root: Path | str = DEFAULT_RUNTIME_MODEL_CACHE_ROOT,
) -> dict[str, str]:
    paths = runtime_asset_paths(cache_root)
    hub_cache = str(paths.hf_home / "hub")
    return {
        "HF_HOME": str(paths.hf_home),
        "HF_HUB_CACHE": hub_cache,
        "HUGGINGFACE_HUB_CACHE": hub_cache,
        "COSMOS_REASON_PATH": str(paths.reason1_snapshot),
        "WAN_VAE_PATH": str(paths.wan_vae),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def asset_contract_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": ASSET_CONTRACT_SCHEMA_VERSION,
        "oscar_source": {
            "repo_id": OSCAR_SOURCE_REPO_ID,
            "revision": OSCAR_SOURCE_REVISION,
        },
        "reason1": {
            "repo_id": REASON1_REPO_ID,
            "revision": REASON1_REVISION,
            "files": [item.as_dict() for item in REASON1_SNAPSHOT_FILES],
            "file_count": len(REASON1_SNAPSHOT_FILES),
            "total_size_bytes": REASON1_SNAPSHOT_TOTAL_SIZE_BYTES,
        },
        "wan_vae": {
            "repo_id": WAN_VAE_REPO_ID,
            "revision": WAN_VAE_REVISION,
            "file": WAN_VAE_FILE.as_dict(),
        },
        "oscar_checkpoint_dcp": {
            "repo_id": OSCAR_CHECKPOINT_REPO_ID,
            "revision": OSCAR_CHECKPOINT_REVISION,
            "files": [item.as_dict() for item in OSCAR_DCP_FILES],
        },
        "network_download_allowed_during_preparation": True,
        "network_download_allowed_during_runtime": False,
    }
    payload["contract_digest"] = _canonical_digest(payload, digest_key="contract_digest")
    return payload


def verify_pinned_file_set(
    asset_root: Path,
    files: Sequence[PinnedRuntimeFile],
    *,
    confinement_root: Path,
    strict_inventory: bool,
    blocker_prefix: str,
) -> dict[str, Any]:
    """Verify size/hash/inventory while rejecting symlink escapes."""

    blockers: list[str] = []
    verified: list[dict[str, Any]] = []
    expected_paths = [item.relative_path for item in files]
    if len(set(expected_paths)) != len(expected_paths):
        blockers.append(f"{blocker_prefix}_contract_has_duplicate_paths")
    try:
        confined_root = confinement_root.resolve(strict=True)
        resolved_asset_root = asset_root.resolve(strict=True)
        if not resolved_asset_root.is_relative_to(confined_root):
            raise ValueError
        if not asset_root.is_dir():
            raise OSError
    except (OSError, ValueError):
        return {
            "status": "blocked",
            "blockers": [f"{blocker_prefix}_root_missing_or_unconfined"],
            "files": [],
            "verified_file_count": 0,
            "verified_size_bytes": 0,
        }

    if strict_inventory:
        actual_paths = sorted(
            path.relative_to(asset_root).as_posix()
            for path in asset_root.rglob("*")
            if path.is_file() or path.is_symlink()
        )
        if actual_paths != sorted(expected_paths):
            blockers.append(f"{blocker_prefix}_snapshot_inventory_mismatch")

    for item in files:
        path = asset_root / item.relative_path
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(confined_root) or not path.is_file():
                raise OSError
            size = path.stat().st_size
        except OSError:
            blockers.append(f"{blocker_prefix}_file_missing_or_unconfined:{item.relative_path}")
            continue
        if size != item.size_bytes:
            blockers.append(f"{blocker_prefix}_file_size_mismatch:{item.relative_path}")
            continue
        digest = _sha256(path)
        if digest != item.sha256:
            blockers.append(f"{blocker_prefix}_file_sha256_mismatch:{item.relative_path}")
            continue
        verified.append(
            {
                **item.as_dict(),
                "resolved_path": str(resolved),
            }
        )

    return {
        "status": "passed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "files": verified,
        "verified_file_count": len(verified) if not blockers else 0,
        "verified_size_bytes": (
            sum(item["size_bytes"] for item in verified) if not blockers else 0
        ),
    }


def _read_evidence_manifest(path: Path, contract_digest: str) -> list[str]:
    blockers: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ["oscar_runtime_asset_evidence_unreadable"]
    if payload.get("schema_version") != ASSET_EVIDENCE_SCHEMA_VERSION:
        blockers.append("oscar_runtime_asset_evidence_schema_invalid")
    if payload.get("status") != "passed":
        blockers.append("oscar_runtime_asset_evidence_not_passed")
    if payload.get("asset_contract_digest") != contract_digest:
        blockers.append("oscar_runtime_asset_evidence_contract_digest_mismatch")
    if payload.get("evidence_digest") != _canonical_digest(payload, digest_key="evidence_digest"):
        blockers.append("oscar_runtime_asset_evidence_digest_mismatch")
    return blockers


def verify_runtime_asset_cache(
    cache_root: Path | str = DEFAULT_RUNTIME_MODEL_CACHE_ROOT,
    *,
    require_evidence_manifest: bool = False,
) -> dict[str, Any]:
    contract = asset_contract_payload()
    blockers: list[str] = []
    try:
        paths = runtime_asset_paths(cache_root)
    except ValueError as exc:
        return {
            "schema_version": ASSET_EVIDENCE_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": [str(exc)],
            "asset_contract_digest": contract["contract_digest"],
            "raw_secret_values_recorded": False,
        }

    reason1 = verify_pinned_file_set(
        paths.reason1_snapshot,
        REASON1_SNAPSHOT_FILES,
        confinement_root=paths.cache_root,
        strict_inventory=True,
        blocker_prefix="oscar_reason1",
    )
    wan = verify_pinned_file_set(
        paths.wan_vae.parent,
        (WAN_VAE_FILE,),
        confinement_root=paths.cache_root,
        strict_inventory=False,
        blocker_prefix="oscar_wan_vae",
    )
    blockers.extend(reason1["blockers"])
    blockers.extend(wan["blockers"])
    if paths.reason1_snapshot.name != REASON1_REVISION:
        blockers.append("oscar_reason1_revision_path_mismatch")
    if paths.wan_vae.parent.name != WAN_VAE_REVISION:
        blockers.append("oscar_wan_vae_revision_path_mismatch")
    if require_evidence_manifest:
        blockers.extend(_read_evidence_manifest(paths.evidence, contract["contract_digest"]))

    passed = not blockers
    return {
        "schema_version": ASSET_EVIDENCE_SCHEMA_VERSION,
        "status": "passed" if passed else "blocked",
        "blockers": sorted(set(blockers)),
        "asset_contract_digest": contract["contract_digest"],
        "cache_root": str(paths.cache_root),
        "hf_home": str(paths.hf_home),
        "reason1": reason1,
        "wan_vae": wan,
        "verified_file_count": (
            reason1["verified_file_count"] + wan["verified_file_count"] if passed else 0
        ),
        "verified_size_bytes": (
            reason1["verified_size_bytes"] + wan["verified_size_bytes"] if passed else 0
        ),
        "runtime_environment": offline_runtime_environment(paths.cache_root),
        "checks": {
            "immutable_revisions_bound": all(
                _IMMUTABLE_REVISION_RE.fullmatch(revision)
                for revision in (REASON1_REVISION, WAN_VAE_REVISION)
            ),
            "reason1_snapshot_inventory_exact": reason1["status"] == "passed",
            "wan_vae_bytes_exact": wan["status"] == "passed",
            "runtime_network_disabled_by_export_contract": True,
        },
        "claim_boundary": {
            "runtime_auxiliary_assets_byte_verified": passed,
            "oscar_checkpoint_loaded": False,
            "oscar_inference_ran": False,
            "generated_video_proven": False,
            "isaac_episode_ran": False,
            "semantic_task_success_proven": False,
        },
        "raw_secret_values_recorded": False,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.chmod(0o644)
    temporary.replace(path)


def _finalize_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["evidence_digest"] = _canonical_digest(result, digest_key="evidence_digest")
    return result


def prepare_runtime_asset_cache(
    cache_root: Path | str = DEFAULT_RUNTIME_MODEL_CACHE_ROOT,
    *,
    token: str | None = None,
) -> dict[str, Any]:
    """Download/resume the immutable auxiliary assets and write byte evidence."""

    paths = runtime_asset_paths(cache_root)
    paths.cache_root.mkdir(parents=True, exist_ok=True)
    paths.hf_home.mkdir(parents=True, exist_ok=True)

    current = verify_runtime_asset_cache(paths.cache_root)
    if current["status"] == "passed":
        evidence = _finalize_evidence({**current, "prepare_action": "reused_verified_cache"})
        _write_json_atomic(paths.evidence, evidence)
        return evidence

    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        hub_cache = paths.hf_home / "hub"
        reason1_download = Path(
            snapshot_download(
                repo_id=REASON1_REPO_ID,
                revision=REASON1_REVISION,
                cache_dir=str(hub_cache),
                token=token,
            )
        ).resolve(strict=True)
        if reason1_download != paths.reason1_snapshot.resolve(strict=True):
            raise RuntimeError("reason1_snapshot_returned_unexpected_revision_path")
        wan_download = Path(
            hf_hub_download(
                repo_id=WAN_VAE_REPO_ID,
                revision=WAN_VAE_REVISION,
                filename=WAN_VAE_FILENAME,
                cache_dir=str(hub_cache),
                token=token,
            )
        ).resolve(strict=True)
        if wan_download != paths.wan_vae.resolve(strict=True):
            raise RuntimeError("wan_vae_returned_unexpected_revision_path")
    except Exception as exc:
        failed = _finalize_evidence(
            {
                "schema_version": ASSET_EVIDENCE_SCHEMA_VERSION,
                "status": "blocked",
                "blockers": [f"oscar_runtime_asset_prepare_failed:{type(exc).__name__}"],
                "asset_contract_digest": asset_contract_payload()["contract_digest"],
                "cache_root": str(paths.cache_root),
                "prepare_action": "download_or_resume_failed",
                "raw_secret_values_recorded": False,
            }
        )
        _write_json_atomic(paths.evidence, failed)
        return failed

    verified = verify_runtime_asset_cache(paths.cache_root)
    evidence = _finalize_evidence(
        {**verified, "prepare_action": "downloaded_or_resumed_then_verified"}
    )
    _write_json_atomic(paths.evidence, evidence)
    return evidence


def verify_oscar_dcp_checkpoint(
    checkpoint_root: Path | str = DEFAULT_OSCAR_CHECKPOINT_ROOT,
    *,
    load_metadata: bool = True,
) -> dict[str, Any]:
    """Verify exact OSCAR DCP bytes and optionally parse DCP metadata offline."""

    root = _normalise_absolute_root(checkpoint_root)
    byte_result = verify_pinned_file_set(
        root,
        OSCAR_DCP_FILES,
        confinement_root=root,
        strict_inventory=False,
        blocker_prefix="oscar_dcp",
    )
    blockers = list(byte_result["blockers"])
    metadata_entry_count: int | None = None
    metadata_readable = False
    if not blockers and load_metadata:
        try:
            from torch.distributed.checkpoint import FileSystemReader

            metadata = FileSystemReader(str(root / "model")).read_metadata()
            state_dict_metadata = getattr(metadata, "state_dict_metadata", None)
            metadata_entry_count = len(state_dict_metadata or {})
            metadata_readable = metadata_entry_count > 0
            if not metadata_readable:
                blockers.append("oscar_dcp_metadata_has_no_state_dict_entries")
        except Exception as exc:
            blockers.append(f"oscar_dcp_metadata_read_failed:{type(exc).__name__}")
    elif not blockers:
        metadata_readable = True

    return {
        "status": "passed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "checkpoint_root": str(root),
        "repo_id": OSCAR_CHECKPOINT_REPO_ID,
        "revision": OSCAR_CHECKPOINT_REVISION,
        "byte_verification": byte_result,
        "metadata_probe_requested": load_metadata,
        "metadata_readable": metadata_readable,
        "metadata_state_dict_entry_count": metadata_entry_count,
        "checkpoint_loaded_into_model": False,
        "raw_secret_values_recorded": False,
    }


def offline_preflight(
    cache_root: Path | str = DEFAULT_RUNTIME_MODEL_CACHE_ROOT,
    *,
    oscar_checkpoint_root: Path | str | None = DEFAULT_OSCAR_CHECKPOINT_ROOT,
    processor_probe: bool = True,
    dcp_metadata_probe: bool = True,
    evidence_output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Fail closed unless auxiliary assets and optional OSCAR DCP are offline-ready."""

    paths = runtime_asset_paths(cache_root)
    expected_env = offline_runtime_environment(paths.cache_root)
    blockers = [
        f"oscar_runtime_environment_mismatch:{name}"
        for name, expected in expected_env.items()
        if os.environ.get(name) != expected
    ]
    assets = verify_runtime_asset_cache(paths.cache_root, require_evidence_manifest=True)
    blockers.extend(assets["blockers"])

    processor_constructible_offline = False
    if not blockers and processor_probe:
        try:
            from transformers import AutoProcessor

            AutoProcessor.from_pretrained(
                str(paths.reason1_snapshot),
                local_files_only=True,
            )
            processor_constructible_offline = True
        except Exception as exc:
            blockers.append(f"oscar_reason1_processor_offline_probe_failed:{type(exc).__name__}")
    elif not processor_probe:
        processor_constructible_offline = True

    dcp_result: dict[str, Any] | None = None
    if not blockers and oscar_checkpoint_root is not None:
        dcp_result = verify_oscar_dcp_checkpoint(
            oscar_checkpoint_root,
            load_metadata=dcp_metadata_probe,
        )
        blockers.extend(dcp_result["blockers"])

    passed = not blockers
    payload: dict[str, Any] = {
        "schema_version": OFFLINE_PREFLIGHT_SCHEMA_VERSION,
        "status": "passed" if passed else "blocked",
        "blockers": sorted(set(blockers)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(paths.cache_root),
        "runtime_environment": expected_env,
        "asset_verification": assets,
        "oscar_dcp_verification": dcp_result,
        "checks": {
            "runtime_environment_exact": not any(
                blocker.startswith("oscar_runtime_environment_mismatch:") for blocker in blockers
            ),
            "auxiliary_assets_byte_verified": assets["status"] == "passed",
            "reason1_processor_constructible_offline": processor_constructible_offline,
            "oscar_dcp_probe_required": oscar_checkpoint_root is not None,
            "oscar_dcp_bytes_and_metadata_verified": (
                dcp_result is not None and dcp_result["status"] == "passed"
                if oscar_checkpoint_root is not None
                else None
            ),
        },
        "claim_boundary": {
            "runtime_asset_preflight_passed": passed,
            "checkpoint_loaded_into_model": False,
            "oscar_inference_ran": False,
            "generated_video_proven": False,
            "isaac_episode_ran": False,
            "semantic_task_success_proven": False,
        },
        "raw_secret_values_recorded": False,
    }
    payload["evidence_digest"] = _canonical_digest(payload, digest_key="evidence_digest")
    output = (
        Path(evidence_output_path).expanduser().resolve(strict=False)
        if evidence_output_path is not None
        else paths.offline_preflight_evidence
    )
    _confined_path(output, paths.cache_root, strict=False)
    _write_json_atomic(output, payload)
    return payload


def render_prepare_once_script(
    cache_root: Path | str = DEFAULT_RUNTIME_MODEL_CACHE_ROOT,
    *,
    python_executable: str = "/opt/oscar-venv/bin/python",
) -> str:
    paths = runtime_asset_paths(cache_root)
    python = shlex.quote(str(python_executable).strip())
    if not str(python_executable).strip():
        raise ValueError("runtime_asset_python_executable_missing")
    return f"""#!/usr/bin/env bash
set -euo pipefail
export HF_HOME={shlex.quote(str(paths.hf_home))}
export HF_HUB_DISABLE_TELEMETRY=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
{python} - <<'PY'
import json
import os
from pathlib import Path

from blueprint_pipeline.oscar_runtime_asset_contract import prepare_runtime_asset_cache

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
result = prepare_runtime_asset_cache(Path({json.dumps(str(paths.cache_root))}), token=token)
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if result.get("status") == "passed" else 1)
PY
"""


def render_offline_export_script(
    cache_root: Path | str = DEFAULT_RUNTIME_MODEL_CACHE_ROOT,
) -> str:
    environment = offline_runtime_environment(cache_root)
    lines = [
        "# Source this file before OSCAR model construction.",
        "# It performs no download and exports no credential.",
    ]
    lines.extend(f"export {name}={shlex.quote(value)}" for name, value in environment.items())
    return "\n".join(lines) + "\n"


def render_offline_preflight_script(
    cache_root: Path | str = DEFAULT_RUNTIME_MODEL_CACHE_ROOT,
    *,
    oscar_checkpoint_root: Path | str | None = DEFAULT_OSCAR_CHECKPOINT_ROOT,
    python_executable: str = "/opt/oscar-venv/bin/python",
) -> str:
    paths = runtime_asset_paths(cache_root)
    python = shlex.quote(str(python_executable).strip())
    if not str(python_executable).strip():
        raise ValueError("runtime_asset_python_executable_missing")
    checkpoint_literal = (
        "None"
        if oscar_checkpoint_root is None
        else f"Path({json.dumps(str(_normalise_absolute_root(oscar_checkpoint_root)))})"
    )
    exports = render_offline_export_script(paths.cache_root)
    return f"""#!/usr/bin/env bash
set -euo pipefail
{exports.rstrip()}
{python} - <<'PY'
import json
from pathlib import Path

from blueprint_pipeline.oscar_runtime_asset_contract import offline_preflight

result = offline_preflight(
    Path({json.dumps(str(paths.cache_root))}),
    oscar_checkpoint_root={checkpoint_literal},
    evidence_output_path=Path({json.dumps(str(paths.offline_preflight_evidence))}),
)
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if result.get("status") == "passed" else 1)
PY
"""


__all__ = [
    "ASSET_CONTRACT_SCHEMA_VERSION",
    "ASSET_EVIDENCE_SCHEMA_VERSION",
    "DEFAULT_OSCAR_CHECKPOINT_ROOT",
    "DEFAULT_RUNTIME_MODEL_CACHE_ROOT",
    "OFFLINE_PREFLIGHT_SCHEMA_VERSION",
    "OSCAR_CHECKPOINT_REPO_ID",
    "OSCAR_CHECKPOINT_REVISION",
    "OSCAR_DCP_FILES",
    "OSCAR_SOURCE_REPO_ID",
    "OSCAR_SOURCE_REVISION",
    "PinnedRuntimeFile",
    "REASON1_REPO_ID",
    "REASON1_REVISION",
    "REASON1_SNAPSHOT_FILES",
    "REASON1_SNAPSHOT_TOTAL_SIZE_BYTES",
    "RUNTIME_AUXILIARY_ASSET_TOTAL_SIZE_BYTES",
    "RuntimeAssetPaths",
    "WAN_VAE_FILE",
    "WAN_VAE_FILENAME",
    "WAN_VAE_REPO_ID",
    "WAN_VAE_REVISION",
    "asset_contract_payload",
    "offline_preflight",
    "offline_runtime_environment",
    "prepare_runtime_asset_cache",
    "render_offline_export_script",
    "render_offline_preflight_script",
    "render_prepare_once_script",
    "runtime_asset_paths",
    "verify_oscar_dcp_checkpoint",
    "verify_pinned_file_set",
    "verify_runtime_asset_cache",
]
