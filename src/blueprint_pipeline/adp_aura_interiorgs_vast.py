"""Build the distinct immutable AuraFusion360 InteriorGS Vast packet."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adp_aura_author_smoke_vast import (
    DEFAULT_IMAGE,
    SAM2_LICENSE_SHA256,
    SAM2_SOURCE_COMMIT,
    SAM2_SOURCE_REPOSITORY,
    SAM2_SOURCE_TREE,
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    SOURCE_TREE,
    SUBMODULES,
    WONDERWORLD_MARIGOLD_LICENSE_SHA256,
    WONDERWORLD_MARIGOLD_RUNTIME_FILES,
    WONDERWORLD_SOURCE_COMMIT,
    WONDERWORLD_SOURCE_REPOSITORY,
    WONDERWORLD_SOURCE_TREE,
    _RUNTIME_MODELS,
    _SD2,
    _deterministic_zip_directory,
    _deterministic_zip_files,
    _git,
    _read_json,
    _sha256,
    _source_files,
    _source_manifest,
    _tracked_files,
    _validate_prerequisite,
    _write_executable,
)
from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .provider_runtime_bundle_contract import provider_runtime_contract_blockers
from .public_scene_aura_adapter import (
    BIG_LAMA_SHA256,
    BIG_LAMA_SIZE,
    LAMA_COMMIT,
    LAMA_TREE,
    SCHEMA_VERSION as ADAPTER_SCHEMA,
)

PROBE_KIND = "adp-aurafusion360-interiorgs"
PROVIDER_BUNDLE_KIND = "adp_aura_interiorgs"
SCENE_ID = "840313"
TARGET_INSTANCE_ID = "ins160"


def _validated_adapter(
    receipt: Mapping[str, Any], root: Path
) -> list[tuple[str, Path]]:
    if (
        receipt.get("schema_version") != ADAPTER_SCHEMA
        or receipt.get("status") != "prepared_unexecuted"
        or canonical_digest(receipt, digest_field="receipt_digest")
        != receipt.get("receipt_digest")
    ):
        raise ValueError("adp_aura_interiorgs_adapter_receipt_invalid")
    scene = receipt.get("scene") or {}
    source = receipt.get("source") or {}
    execution = receipt.get("execution") or {}
    if scene.get("publisher_scene_id") != SCENE_ID or scene.get(
        "target_instance_id"
    ) != TARGET_INSTANCE_ID:
        raise ValueError("adp_aura_interiorgs_scene_or_target_mismatch")
    if source.get("commit") != SOURCE_COMMIT or source.get("tree") != SOURCE_TREE:
        raise ValueError("adp_aura_interiorgs_source_identity_mismatch")
    if any(bool(value) for value in execution.values()):
        raise ValueError("adp_aura_interiorgs_caller_asserted_execution_forbidden")
    rows: list[tuple[str, Path]] = []
    for record in receipt.get("artifacts") or []:
        relative = str(record.get("relative_path") or "")
        path = (root / relative).resolve()
        if root != path and root not in path.parents:
            raise ValueError("adp_aura_interiorgs_adapter_artifact_outside_root")
        if (
            not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
            or _sha256(path) != record.get("sha256")
        ):
            raise ValueError("adp_aura_interiorgs_adapter_artifact_changed")
        rows.append((relative, path))
    required = {
        "aurafusion360_interiorgs_execution_spec.json",
        "configs/Other-360/840313_ins160/train.config",
        "configs/Other-360/840313_ins160/remove.config",
        "configs/Other-360/840313_ins160/inpaint.config",
        "configs/Other-360/840313_ins160/sdedit.config",
        "reference_lama_input/low_approach.png",
        "reference_lama_input/low_approach_mask.png",
        "data/Other-360/840313_ins160/sparse/0/points3D.ply",
    }
    if not required.issubset({relative for relative, _ in rows}):
        raise ValueError("adp_aura_interiorgs_adapter_required_artifact_missing")
    return rows


def build_aura_interiorgs_bundle(
    *,
    repo_root: str | Path,
    aura_root: str | Path,
    sam2_root: str | Path,
    wonderworld_root: str | Path,
    lama_root: str | Path,
    prerequisite_receipt_path: str | Path,
    adapter_root: str | Path,
    adapter_receipt_path: str | Path,
    big_lama_path: str | Path,
    job_dir: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    aura = Path(aura_root).expanduser().resolve()
    sam2 = Path(sam2_root).expanduser().resolve()
    wonderworld = Path(wonderworld_root).expanduser().resolve()
    lama = Path(lama_root).expanduser().resolve()
    packet = Path(adapter_root).expanduser().resolve()
    adapter_file = Path(adapter_receipt_path).expanduser().resolve()
    prerequisite_file = Path(prerequisite_receipt_path).expanduser().resolve()
    big_lama = Path(big_lama_path).expanduser().resolve()
    job = Path(job_dir).expanduser().resolve()
    if job.exists() and any(job.iterdir()):
        raise ValueError("adp_aura_interiorgs_job_dir_not_empty")
    runtime = job / "provider_runtime"
    ensure_dir(runtime)
    if (
        _git(repo, "status", "--porcelain", "--untracked-files=no")
        or _git(aura, "rev-parse", "HEAD") != SOURCE_COMMIT
        or _git(aura, "rev-parse", "HEAD^{tree}") != SOURCE_TREE
        or _git(aura, "status", "--porcelain")
    ):
        raise ValueError("adp_aura_interiorgs_source_or_blueprint_dirty")
    if {
        path: _git(aura / path, "rev-parse", "HEAD") for path in SUBMODULES
    } != SUBMODULES:
        raise ValueError("adp_aura_interiorgs_submodule_mismatch")
    if (
        _git(sam2, "rev-parse", "HEAD") != SAM2_SOURCE_COMMIT
        or _git(sam2, "rev-parse", "HEAD^{tree}") != SAM2_SOURCE_TREE
        or _git(sam2, "status", "--porcelain")
    ):
        raise ValueError("adp_aura_interiorgs_sam2_identity_mismatch")
    if not (sam2 / "LICENSE").is_file() or _sha256(
        sam2 / "LICENSE"
    ) != SAM2_LICENSE_SHA256:
        raise ValueError("adp_aura_interiorgs_sam2_license_mismatch")
    if (
        _git(wonderworld, "rev-parse", "HEAD") != WONDERWORLD_SOURCE_COMMIT
        or _git(wonderworld, "rev-parse", "HEAD^{tree}")
        != WONDERWORLD_SOURCE_TREE
        or _git(wonderworld, "status", "--porcelain")
    ):
        raise ValueError("adp_aura_interiorgs_wonderworld_identity_mismatch")
    if _sha256(
        wonderworld / "marigold_module/LICENSE.txt"
    ) != WONDERWORLD_MARIGOLD_LICENSE_SHA256:
        raise ValueError("adp_aura_interiorgs_wonderworld_license_mismatch")
    if (
        _git(lama, "rev-parse", "HEAD") != LAMA_COMMIT
        or _git(lama, "rev-parse", "HEAD^{tree}") != LAMA_TREE
        or _git(lama, "status", "--porcelain", "--untracked-files=no")
    ):
        raise ValueError("adp_aura_interiorgs_lama_source_identity_mismatch")
    if (
        not big_lama.is_file()
        or big_lama.stat().st_size != BIG_LAMA_SIZE
        or _sha256(big_lama) != BIG_LAMA_SHA256
    ):
        raise ValueError("adp_aura_interiorgs_lama_checkpoint_changed")

    prerequisite = _read_json(prerequisite_file)
    snapshots = _validate_prerequisite(prerequisite)
    adapter = _read_json(adapter_file)
    adapter_rows = _validated_adapter(adapter, packet)
    aura_rows = _source_files(aura)
    sam2_rows = _tracked_files(sam2)
    wonderworld_rows = [
        (archive_path, wonderworld / source_path)
        for archive_path, source_path in sorted(WONDERWORLD_MARIGOLD_RUNTIME_FILES.items())
    ]
    lama_rows = _tracked_files(lama)
    _deterministic_zip_files(aura_rows, runtime / "aurafusion360_source.zip")
    _deterministic_zip_files(sam2_rows, runtime / "sam2_source.zip")
    _deterministic_zip_files(
        wonderworld_rows, runtime / "wonderworld_marigold_runtime.zip"
    )
    _deterministic_zip_files(lama_rows, runtime / "lama_source.zip")
    _deterministic_zip_files(adapter_rows, runtime / "interiorgs_adapter.zip")
    shutil.copy2(big_lama, runtime / "big-lama.zip")
    source_manifest = _source_manifest(aura_rows)
    sam2_manifest = _source_manifest(sam2_rows)
    wonderworld_manifest = _source_manifest(wonderworld_rows)
    lama_manifest = _source_manifest(lama_rows)
    adapter_manifest = _source_manifest(adapter_rows)
    sd2 = snapshots["aurafusion360_sd2_inpainting_exact_checkpoint"]["publisher"][
        "single_file_identity"
    ]
    workflow_names = (
        "train", "render", "remove", "sam2_masks", "inpaint_init", "sdedit",
        "inpaint_finetune",
    )
    workflow = [
        {"stage": name, "command": command}
        for name, command in zip(
            workflow_names, adapter["commands"]["author_workflow"], strict=True
        )
    ]
    spec = {
        "schema_version": "adp_aura_interiorgs_spec.v1",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "source_files": source_manifest,
        "submodules": SUBMODULES,
        "sam2_source": {
            "repository": SAM2_SOURCE_REPOSITORY, "commit": SAM2_SOURCE_COMMIT,
            "tree": SAM2_SOURCE_TREE, "license_sha256": SAM2_LICENSE_SHA256,
            "source_files": sam2_manifest,
        },
        "wonderworld_marigold_runtime": {
            "repository": WONDERWORLD_SOURCE_REPOSITORY,
            "commit": WONDERWORLD_SOURCE_COMMIT, "tree": WONDERWORLD_SOURCE_TREE,
            "license": "Apache-2.0", "license_sha256": WONDERWORLD_MARIGOLD_LICENSE_SHA256,
            "archive": "wonderworld_marigold_runtime.zip",
            "archive_sha256": _sha256(runtime / "wonderworld_marigold_runtime.zip"),
            "source_files": wonderworld_manifest,
        },
        "lama": {
            "source_archive": "lama_source.zip", "source_files": lama_manifest,
            "checkpoint_archive": "big-lama.zip", "checkpoint_sha256": BIG_LAMA_SHA256,
        },
        "adapter": {
            "archive": "interiorgs_adapter.zip",
            "archive_sha256": _sha256(runtime / "interiorgs_adapter.zip"),
            "receipt_digest": adapter["receipt_digest"], "files": adapter_manifest,
        },
        "runtime_models": _RUNTIME_MODELS,
        "sd2_checkpoint": {
            **_SD2, "size_bytes": sd2["size_bytes"], "sha256": sd2["lfs_sha256"],
        },
        "workflow": workflow,
        "claim_boundary": {
            "hidden_background_truth_available": False,
            "publisher_splat_edited_in_place": False,
            "output_claim_ceiling": "visual_candidate_only",
        },
    }
    write_json(runtime / "execution_spec.json", spec)
    scripts = repo / "scripts"
    _write_executable(
        runtime / "run_adp_aura_interiorgs_provider_runtime.sh",
        scripts / "run_adp_aura_interiorgs_provider_runtime.sh",
    )
    for name in (
        "adp_aura_interiorgs_provider_runner.py",
        "adp_aura_author_smoke_provider_runner.py",
    ):
        shutil.copy2(scripts / name, runtime / name)
    blockers = provider_runtime_contract_blockers(
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
        entrypoint_text=(runtime / "run_adp_aura_interiorgs_provider_runtime.sh").read_text(),
        runner_text=(runtime / "adp_aura_interiorgs_provider_runner.py").read_text(),
    )
    manifest = {
        "schema_version": "adp_aura_interiorgs_provider_bundle.v1",
        "generated_at": generated_at or utc_now_iso(),
        "status": "ready" if not blockers else "blocked",
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "blueprint_commit": _git(repo, "rev-parse", "HEAD"),
        "blueprint_tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "adapter_receipt_digest": adapter["receipt_digest"],
        "prerequisite_receipt_digest": prerequisite["receipt_digest"],
        "container_image": DEFAULT_IMAGE,
        "expected_output_filename": "adp_aura_interiorgs_result.json",
        "retry_cap": 0,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }
    write_json(runtime / "adp_aura_interiorgs_provider_manifest.json", manifest)
    bundle = job / "adp_aura_interiorgs_provider_runtime_bundle.zip"
    _deterministic_zip_directory(runtime, bundle)
    receipt = {
        **manifest,
        "bundle_path": str(bundle),
        "bundle_sha256": _sha256(bundle),
        "bundle_size_bytes": bundle.stat().st_size,
    }
    write_json(job / "adp_aura_interiorgs_bundle_receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repo-root", "aura-root", "sam2-root", "wonderworld-root", "lama-root",
        "prerequisite-receipt", "adapter-root", "adapter-receipt", "big-lama",
        "job-dir",
    ):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    receipt = build_aura_interiorgs_bundle(
        repo_root=args.repo_root, aura_root=args.aura_root, sam2_root=args.sam2_root,
        wonderworld_root=args.wonderworld_root, lama_root=args.lama_root,
        prerequisite_receipt_path=args.prerequisite_receipt,
        adapter_root=args.adapter_root, adapter_receipt_path=args.adapter_receipt,
        big_lama_path=args.big_lama, job_dir=args.job_dir,
    )
    print(json.dumps({"status": receipt["status"], "bundle_sha256": receipt["bundle_sha256"]}))
    return 0 if receipt["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
