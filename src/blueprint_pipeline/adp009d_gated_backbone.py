"""Fail-closed access and worker identity for GR00T N1.7's gated backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

try:  # flat provider-bundle layout
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:  # repository package
    from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "adp009d_gated_backbone_identity.v1"
ACCESS_SCHEMA_VERSION = "adp009d_gated_backbone_access.v1"
RUNTIME_BINDING_SCHEMA_VERSION = "adp009d_groot_offline_runtime_binding.v1"
MODEL_ID = "nvidia/Cosmos-Reason2-2B"
REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
EXPECTED_TOTAL_BYTES = 4_889_656_452

# Object identities from the exact Hugging Face revision.  Ordinary Git blobs
# use the Git blob SHA-1; LFS payloads use the publisher's content SHA-256.
EXPECTED_FILES: tuple[tuple[str, int, str, str], ...] = (
    (".gitattributes", 1707, "git_blob_sha1", "140a13b3da6ed38dcd849c3e26a0583743f76fae"),
    ("CR2 Performance by Category.png", 37375, "git_blob_sha1", "95ca01dddab7a1717588c53404c23898de197353"),
    ("PAI_Bench_Leaderboard.png", 473585, "sha256", "f9ac0320a57d6ee76a4f582388564dcb379ab170f2d99550e891fcd26a0b31c9"),
    ("README.md", 35046, "git_blob_sha1", "c67ef7eb71b31dde6a102f9a51a0bb0cc01733b9"),
    ("chat_template.json", 5502, "git_blob_sha1", "e49e75bdda104c8de941eb20e0b3765184fb27bc"),
    ("config.json", 1505, "git_blob_sha1", "0cd8c646eb55794594b14fefcd1d3acea5cc000f"),
    ("cosmos_cr2_2b_paiBench.png", 138441, "sha256", "4b79d34191efa63420e64f4ded2a113b43cd3a2dadbb65e226e9d99a3442d795"),
    ("generation_config.json", 269, "git_blob_sha1", "7bb37933d2d86e4ceaabcc6540be989658f238e9"),
    ("merges.txt", 1671839, "git_blob_sha1", "20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0"),
    ("model.safetensors", 4877470304, "sha256", "fa5a6e6ef4fce40216b185cc48a3b24d31637ac3e2ba69c107ed1f389c1e6ede"),
    ("preprocessor_config.json", 390, "git_blob_sha1", "2ea84a437d448ff71b08df68fdd949d5cc4ebb64"),
    ("tokenizer.json", 7032403, "git_blob_sha1", "c6cc1014128b19d1fc46b1d30a23e3b1d35db421"),
    ("tokenizer_config.json", 10868, "git_blob_sha1", "d3d3763207692c78780f4bf42d4dadf49a5c8012"),
    ("video_preprocessor_config.json", 385, "git_blob_sha1", "3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe"),
    ("vocab.json", 2776833, "git_blob_sha1", "4783fe10ac3adce15ac8f358ef5462739852c569"),
)


def _git_blob_sha1(path: Path, size: int) -> str:
    digest = hashlib.sha1()  # noqa: S324 - publisher Git object identity, not security auth
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_listing_digest() -> str:
    return canonical_digest(
        {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "files": [
                {
                    "path": path,
                    "size_bytes": size,
                    "object_id_kind": kind,
                    "object_id": object_id,
                }
                for path, size, kind, object_id in EXPECTED_FILES
            ],
        }
    )


def probe_gated_backbone_access(
    *,
    token: str | None = None,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Prove the local credential can read the exact gated revision, without mutation."""

    selected = str(token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()
    blockers: list[str] = []
    data: dict[str, Any] = {}
    if not selected:
        blockers.append("adp009d_gated_backbone_token_missing")
    else:
        url = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true"
        try:
            response = opener(
                Request(
                    url,
                    headers={
                        "Authorization": f"Bearer {selected}",
                        "User-Agent": "blueprint-adp009d-access-probe/1",
                    },
                ),
                timeout=40,
            )
            with response:
                loaded = json.load(response)
            if isinstance(loaded, Mapping):
                data = dict(loaded)
            else:
                blockers.append("adp009d_gated_backbone_access_response_invalid")
        except Exception as exc:  # noqa: BLE001 - typed redacted preflight receipt
            blockers.append(f"adp009d_gated_backbone_access_failed:{type(exc).__name__}")

    if data:
        if data.get("id") != MODEL_ID:
            blockers.append("adp009d_gated_backbone_model_id_mismatch")
        if data.get("sha") != REVISION:
            blockers.append("adp009d_gated_backbone_revision_mismatch")
        observed: dict[str, tuple[int, str | None, str | None]] = {}
        for row in data.get("siblings") or []:
            if not isinstance(row, Mapping):
                continue
            lfs = row.get("lfs") if isinstance(row.get("lfs"), Mapping) else {}
            observed[str(row.get("rfilename") or "")] = (
                int(row.get("size") or 0),
                str(row.get("blobId") or "") or None,
                str(lfs.get("sha256") or "") or None,
            )
        for path, size, kind, object_id in EXPECTED_FILES:
            row = observed.get(path)
            actual_id = row[2] if kind == "sha256" and row else row[1] if row else None
            if row is None or row[0] != size or actual_id != object_id:
                blockers.append(f"adp009d_gated_backbone_listing_mismatch:{path}")
        if set(observed) != {row[0] for row in EXPECTED_FILES}:
            blockers.append("adp009d_gated_backbone_listing_file_set_mismatch")

    receipt: dict[str, Any] = {
        "schema_version": ACCESS_SCHEMA_VERSION,
        "status": "authorized" if not blockers else "blocked",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "gated": data.get("gated") if data else None,
        "expected_file_count": len(EXPECTED_FILES),
        "expected_total_bytes": EXPECTED_TOTAL_BYTES,
        "expected_listing_digest": _expected_listing_digest(),
        "raw_secret_recorded": False,
        "secret_hash_recorded": False,
        "blockers": sorted(set(blockers)),
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def verify_and_seal_offline_cache(*, cache_dir: Path, output_path: Path) -> dict[str, Any]:
    """Verify downloaded bytes and bind unversioned model lookup to the frozen revision."""

    hub = cache_dir.expanduser().resolve()
    repo_cache = hub / "models--nvidia--Cosmos-Reason2-2B"
    snapshot = repo_cache / "snapshots" / REVISION
    blockers: list[str] = []
    files: list[dict[str, Any]] = []
    expected_paths = {row[0] for row in EXPECTED_FILES}
    observed_paths = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file()
    } if snapshot.is_dir() else set()
    if observed_paths != expected_paths:
        blockers.append("adp009d_gated_backbone_cache_file_set_mismatch")
    for relative, size, kind, expected_id in EXPECTED_FILES:
        path = snapshot / relative
        actual_size = path.stat().st_size if path.is_file() else None
        actual_id = None
        if actual_size == size:
            actual_id = _sha256(path) if kind == "sha256" else _git_blob_sha1(path, size)
        if actual_size != size or actual_id != expected_id:
            blockers.append(f"adp009d_gated_backbone_cache_object_mismatch:{relative}")
        files.append(
            {
                "path": relative,
                "size_bytes": actual_size,
                "object_id_kind": kind,
                "object_id_verified": actual_id == expected_id,
            }
        )
    total_bytes = sum(int(row.get("size_bytes") or 0) for row in files)
    if total_bytes != EXPECTED_TOTAL_BYTES:
        blockers.append("adp009d_gated_backbone_cache_total_bytes_mismatch")

    offline_ref_written = False
    if not blockers:
        refs = repo_cache / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        main = refs / "main"
        if main.exists() and main.read_text(encoding="utf-8").strip() != REVISION:
            blockers.append("adp009d_gated_backbone_offline_main_ref_conflict")
        else:
            main.write_text(REVISION + "\n", encoding="utf-8")
            offline_ref_written = True

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified" if not blockers else "blocked",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "snapshot_path": str(snapshot),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "expected_listing_digest": _expected_listing_digest(),
        "files": files,
        "offline_main_ref_written": offline_ref_written,
        "credentials_retained_after_materialization": False,
        "blockers": sorted(set(blockers)),
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("adp009d_gated_backbone_receipt_overwrite_forbidden")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if blockers:
        raise RuntimeError(";".join(sorted(set(blockers))))
    return receipt


def prepare_offline_runtime_binding(
    *,
    checkpoint_root: Path,
    backbone_snapshot: Path,
    runtime_checkpoint_root: Path,
    alias_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Bind the frozen GR00T checkpoint to a verified local backbone.

    GR00T N1.7 selects its Qwen3 backbone by looking for the literal publisher
    token ``nvidia/Cosmos-Reason2`` in ``config.model_name``.  Replacing that
    value with an ordinary local path therefore selects the wrong class.  The
    ``nvidia/Cosmos-Reason2-2B/../..`` alias retains the upstream selector token
    while resolving to a flat local directory of links to the verified
    snapshot.  A symlink-only checkpoint overlay keeps the downloaded frozen
    checkpoint untouched; only its small runtime config is derived.
    """

    checkpoint = checkpoint_root.expanduser().resolve()
    snapshot = backbone_snapshot.expanduser().resolve()
    runtime = runtime_checkpoint_root.expanduser().resolve()
    aliases = alias_root.expanduser().resolve()
    blockers: list[str] = []

    config_path = checkpoint / "config.json"
    config_bytes = config_path.read_bytes() if config_path.is_file() else b""
    processor_config_path = checkpoint / "processor_config.json"
    processor_config_bytes = (
        processor_config_path.read_bytes()
        if processor_config_path.is_file()
        else b""
    )
    try:
        config = json.loads(config_bytes) if config_bytes else {}
    except json.JSONDecodeError:
        config = {}
        blockers.append("adp009d_groot_checkpoint_config_invalid")
    if not isinstance(config, dict):
        config = {}
        blockers.append("adp009d_groot_checkpoint_config_invalid")
    if config.get("model_name") != MODEL_ID:
        blockers.append("adp009d_groot_checkpoint_backbone_identity_mismatch")
    try:
        processor_config = (
            json.loads(processor_config_bytes) if processor_config_bytes else {}
        )
    except json.JSONDecodeError:
        processor_config = {}
        blockers.append("adp009d_groot_processor_config_invalid")
    if not isinstance(processor_config, dict):
        processor_config = {}
        blockers.append("adp009d_groot_processor_config_invalid")
    processor_kwargs = processor_config.get("processor_kwargs")
    if not isinstance(processor_kwargs, dict):
        processor_kwargs = {}
        blockers.append("adp009d_groot_processor_kwargs_missing")
    elif processor_kwargs.get("model_name") not in {None, MODEL_ID}:
        blockers.append("adp009d_groot_processor_backbone_identity_mismatch")
    if not checkpoint.is_dir():
        blockers.append("adp009d_groot_checkpoint_root_missing")
    if not snapshot.is_dir():
        blockers.append("adp009d_gated_backbone_snapshot_missing")
    if runtime.exists() or runtime.is_symlink():
        blockers.append("adp009d_groot_runtime_checkpoint_root_not_fresh")
    if aliases.exists() or aliases.is_symlink():
        blockers.append("adp009d_groot_backbone_alias_root_not_fresh")
    if snapshot.is_dir() and (snapshot / "nvidia").exists():
        blockers.append("adp009d_groot_backbone_alias_reserved_name_collision")

    receipt: dict[str, Any] = {
        "schema_version": RUNTIME_BINDING_SCHEMA_VERSION,
        "status": "blocked" if blockers else "verified",
        "checkpoint_root": str(checkpoint),
        "checkpoint_config_sha256": (
            "sha256:" + hashlib.sha256(config_bytes).hexdigest()
            if config_bytes
            else None
        ),
        "processor_config_sha256": (
            "sha256:" + hashlib.sha256(processor_config_bytes).hexdigest()
            if processor_config_bytes
            else None
        ),
        "backbone_model_id": MODEL_ID,
        "backbone_revision": REVISION,
        "backbone_snapshot": str(snapshot),
        "runtime_checkpoint_root": str(runtime),
        "alias_root": str(aliases),
        "sealed_checkpoint_modified": False,
        "runtime_overlay_uses_symlinks": True,
        "credentials_required_at_runtime": False,
        "blockers": sorted(set(blockers)),
    }
    if blockers:
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        if output_path.exists() or output_path.is_symlink():
            raise FileExistsError("adp009d_groot_runtime_binding_receipt_overwrite_forbidden")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise RuntimeError(";".join(receipt["blockers"]))

    aliases.mkdir(parents=True)
    for relative, _size, _kind, _object_id in EXPECTED_FILES:
        source = snapshot / relative
        destination = aliases / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source)
    selector_anchor = aliases / "nvidia" / "Cosmos-Reason2-2B"
    selector_anchor.mkdir(parents=True)
    alias_name = str(selector_anchor / "../..")
    if Path(alias_name).resolve(strict=True) != aliases:
        raise RuntimeError("adp009d_groot_local_backbone_alias_resolution_failed")

    runtime.mkdir(parents=True)
    for source in checkpoint.iterdir():
        if source.name in {"config.json", "processor_config.json"}:
            continue
        (runtime / source.name).symlink_to(
            source, target_is_directory=source.is_dir()
        )
    runtime_config = dict(config)
    runtime_config["blueprint_original_model_name"] = MODEL_ID
    runtime_config["blueprint_runtime_backbone_path"] = str(snapshot)
    runtime_config["model_name"] = alias_name
    runtime_config_path = runtime / "config.json"
    runtime_config_path.write_text(
        json.dumps(runtime_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime_processor_config = dict(processor_config)
    runtime_processor_kwargs = dict(processor_kwargs)
    runtime_processor_kwargs["model_name"] = alias_name
    runtime_processor_config["processor_kwargs"] = runtime_processor_kwargs
    runtime_processor_config_path = runtime / "processor_config.json"
    runtime_processor_config_path.write_text(
        json.dumps(runtime_processor_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    receipt.update(
        {
            "alias_model_name": alias_name,
            "runtime_checkpoint_config_sha256": (
                "sha256:" + _sha256(runtime_config_path)
            ),
            "runtime_processor_config_sha256": (
                "sha256:" + _sha256(runtime_processor_config_path)
            ),
            "runtime_overlay_entry_count": len(list(runtime.iterdir())),
            "original_backbone_identity_preserved": True,
            "local_backbone_alias_resolves_to_verified_snapshot": all(
                (aliases / relative).resolve(strict=True)
                == (snapshot / relative).resolve(strict=True)
                for relative, _size, _kind, _object_id in EXPECTED_FILES
            ),
        }
    )
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("adp009d_groot_runtime_binding_receipt_overwrite_forbidden")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--runtime-checkpoint-root", type=Path)
    parser.add_argument("--alias-root", type=Path)
    parser.add_argument("--runtime-binding-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = verify_and_seal_offline_cache(
        cache_dir=args.cache_dir,
        output_path=args.output,
    )
    binding_args = (
        args.checkpoint_root,
        args.runtime_checkpoint_root,
        args.alias_root,
        args.runtime_binding_output,
    )
    if any(binding_args) and not all(binding_args):
        raise RuntimeError("adp009d_groot_runtime_binding_arguments_incomplete")
    if all(binding_args):
        prepare_offline_runtime_binding(
            checkpoint_root=args.checkpoint_root,
            backbone_snapshot=Path(receipt["snapshot_path"]),
            runtime_checkpoint_root=args.runtime_checkpoint_root,
            alias_root=args.alias_root,
            output_path=args.runtime_binding_output,
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACCESS_SCHEMA_VERSION",
    "EXPECTED_FILES",
    "EXPECTED_TOTAL_BYTES",
    "MODEL_ID",
    "REVISION",
    "RUNTIME_BINDING_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "main",
    "prepare_offline_runtime_binding",
    "probe_gated_backbone_access",
    "verify_and_seal_offline_cache",
]
