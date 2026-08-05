#!/usr/bin/env python3
"""Execute one frozen AuraFusion360 InteriorGS challenger packet."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Sequence

import adp_aura_author_smoke_provider_runner as shared

SCHEMA_VERSION = "adp_aura_interiorgs_result.v1"
COMMAND_TIMEOUT_SECONDS = 21_600
SCENE = "840313_ins160"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, dict) else {}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _extract_verified(
    *, archive: Path, destination: Path, records: list[dict[str, Any]], prefix: str = ""
) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError("aurafusion360_interiorgs_archive_path_traversal")
        source.extractall(destination)
    for record in records:
        path = destination / prefix / str(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError("aurafusion360_interiorgs_materialized_bytes_changed")


def _copy_tree_additions(source: Path, destination: Path) -> None:
    """Add an adapter subtree without deleting or replacing publisher files."""

    if not source.is_dir():
        raise ValueError("aurafusion360_interiorgs_adapter_subtree_missing")
    destination.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            if target.exists() and not target.is_dir():
                raise ValueError("aurafusion360_interiorgs_adapter_path_conflict")
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not item.is_file() or target.exists():
            raise ValueError("aurafusion360_interiorgs_adapter_path_conflict")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def _runtime_environments(
    *, dependencies: Path, lama_source: Path
) -> tuple[dict[str, str], dict[str, str]]:
    aura_env = dict(os.environ)
    inherited_pythonpath = aura_env.get("PYTHONPATH")
    aura_env["PYTHONPATH"] = str(dependencies) + (
        os.pathsep + inherited_pythonpath if inherited_pythonpath else ""
    )
    lama_env = dict(aura_env)
    lama_env["PYTHONPATH"] = str(lama_source) + os.pathsep + aura_env["PYTHONPATH"]
    return aura_env, lama_env


def _materialize_lama_checkpoint(
    *, runtime: Path, source: Path, spec: dict[str, Any], receipt_path: Path | None = None
) -> dict[str, Any]:
    checkpoint_spec = spec["lama"]
    archive_path = runtime / checkpoint_spec["checkpoint_archive"]
    blockers: list[str] = []
    if (
        not archive_path.is_file()
        or _sha256(archive_path) != checkpoint_spec["checkpoint_sha256"]
    ):
        blockers.append("aurafusion360_interiorgs_lama_checkpoint_archive_changed")
    lama_source = source / "LaMa"
    required = {
        "config": lama_source / "big-lama/config.yaml",
        "checkpoint": lama_source / "big-lama/models/best.ckpt",
    }
    if not blockers:
        root = lama_source.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            if not {"big-lama/config.yaml", "big-lama/models/best.ckpt"}.issubset(names):
                blockers.append("aurafusion360_interiorgs_lama_checkpoint_members_missing")
            for member in archive.infolist():
                target = (lama_source / member.filename).resolve()
                if target != root and root not in target.parents:
                    blockers.append("aurafusion360_interiorgs_lama_checkpoint_path_traversal")
                    break
            if not blockers:
                archive.extractall(lama_source)
    files: dict[str, dict[str, Any] | None] = {}
    for name, path in required.items():
        files[name] = (
            {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            if path.is_file() and path.stat().st_size > 0
            else None
        )
        if files[name] is None:
            blockers.append(f"aurafusion360_interiorgs_lama_{name}_missing")
    receipt = {
        "schema_version": "adp_aura_lama_checkpoint_materialization.v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "accepted" if not blockers else "blocked",
        "archive_sha256": _sha256(archive_path) if archive_path.is_file() else None,
        "archive_identity_source": "pinned_bundle_execution_spec",
        "files": files,
        "blockers": sorted(set(blockers)),
    }
    if receipt_path is not None:
        _write(receipt_path, receipt)
    return receipt


def _prepare(runtime: Path, source: Path, spec: dict[str, Any]) -> int:
    from huggingface_hub import hf_hub_download, snapshot_download

    adapter_archive = runtime / spec["adapter"]["archive"]
    if _sha256(adapter_archive) != spec["adapter"]["archive_sha256"]:
        raise ValueError("aurafusion360_interiorgs_adapter_archive_changed")
    _extract_verified(
        archive=adapter_archive,
        destination=runtime / "adapter_extract",
        records=spec["adapter"]["files"],
    )
    adapter = runtime / "adapter_extract"
    for name in ("data", "configs", "reference_lama_input"):
        src = adapter / name
        dst = source / name if name != "reference_lama_input" else runtime / name
        _copy_tree_additions(src, dst)

    lama_source = source / "LaMa"
    _extract_verified(
        archive=runtime / spec["lama"]["source_archive"],
        destination=lama_source,
        records=spec["lama"]["source_files"],
    )
    checkpoint_materialization = _materialize_lama_checkpoint(
        runtime=runtime, source=source, spec=spec
    )
    if checkpoint_materialization["status"] != "accepted":
        raise ValueError("aurafusion360_interiorgs_lama_checkpoint_extract_failed")

    shared._extract_runtime_dependency(runtime, spec)
    cache = Path(os.environ["HF_HUB_CACHE"])
    resolved: dict[str, Path] = {}
    for model in spec["runtime_models"]:
        files = model.get("materialized_files")
        alias = str(model.get("cache_alias_of") or "")
        if alias:
            snapshot = shared._materialize_cache_alias(
                cache=cache, source_snapshot=resolved[alias], model=model
            )
        else:
            snapshot = Path(
                snapshot_download(
                    repo_id=model["repository"], revision=model["revision"],
                    allow_patterns=([str(item["path"]) for item in files] if files else None),
                    max_workers=1,
                )
            )
            shared._pin_hf_ref(cache, model["repository"], model["revision"])
        shared._verify_runtime_model_snapshot(snapshot, model)
        resolved[str(model["repository"])] = snapshot
    sd2 = spec["sd2_checkpoint"]
    downloaded = Path(
        hf_hub_download(
            repo_id=sd2["repository"], revision=sd2["revision"], filename=sd2["path"]
        )
    )
    destination = source / "utils/LeftRefill/pretrained_models/512-inpainting-ema.ckpt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded, destination)
    if destination.stat().st_size != sd2["size_bytes"] or _sha256(destination) != sd2["sha256"]:
        raise ValueError("aurafusion360_interiorgs_sd2_checkpoint_changed")
    _write(runtime / "prepare_receipt.json", {"status": "prepared"})
    return 0


def _ply_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        for raw in stream:
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[-1])
            if line == "end_header":
                break
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    runtime = Path(__file__).resolve().parent
    source = runtime / "AuraFusion360_official"
    output = Path(os.environ.get("BLUEPRINT_ADP_AURA_INTERIORGS_OUTPUT_DIR", runtime.parent / "runtime_output")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = _read(runtime / "execution_spec.json")
    if args.prepare_only:
        return _prepare(runtime, source, spec)

    checkpoint_materialization = _materialize_lama_checkpoint(
        runtime=runtime,
        source=source,
        spec=spec,
        receipt_path=output / "lama-checkpoint-materialization.json",
    )
    if checkpoint_materialization["status"] != "accepted":
        raise ValueError("aurafusion360_interiorgs_lama_checkpoint_materialization_failed")

    dependencies = runtime / "runtime_dependencies"
    if not dependencies.is_dir():
        raise ValueError("aurafusion360_interiorgs_runtime_dependencies_missing")
    aura_env, lama_env = _runtime_environments(
        dependencies=dependencies,
        lama_source=source / "LaMa",
    )
    source_before = shared._source_identity(source, spec)
    aura_python = str(source / ".venv/bin/python")
    lama_python = str(source / "LaMa/.venv/bin/python")
    workflow: list[dict[str, Any]] = []
    commands = [
        (
            "reference_lama",
            [
                lama_python,
                "bin/predict.py",
                "model.path=./big-lama",
                f"indir={runtime / 'reference_lama_input'}",
                f"outdir={runtime / 'reference_lama_output'}",
            ],
            source / "LaMa",
            lama_env,
        ),
        *[
            (row["stage"], [aura_python, *row["command"][1:]], source, aura_env)
            for row in spec["workflow"]
        ],
    ]
    for stage, command, cwd, env in commands:
        observed = shared._run(
            command,
            cwd=cwd,
            log_path=output / f"aura-{stage}.log",
            env=env,
        )
        observed["stage"] = stage
        workflow.append(observed)
        if observed["returncode"] != 0:
            break
        if stage == "reference_lama":
            produced = sorted((runtime / "reference_lama_output").glob("*.png"))
            if len(produced) != 1:
                raise ValueError("aurafusion360_interiorgs_reference_output_not_unique")
            target = source / f"data/Other-360/{SCENE}/reference/low_approach.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(produced[0], target)

    completed = {row["stage"] for row in workflow if row["returncode"] == 0}
    required = ["reference_lama", "train", "render", "remove", "sam2_masks", "inpaint_init", "sdedit", "inpaint_finetune"]
    final_ply = source / f"output/Other-360/{SCENE}/point_cloud/iteration_10000_object_inpaint/point_cloud.ply"
    blockers: list[str] = []
    for stage in required:
        if stage not in completed:
            blockers.append(f"aurafusion360_interiorgs_{stage}_failed_or_not_executed")
            break
    source_after = shared._source_identity(source, spec)
    if not source_before["matches"] or not source_after["matches"]:
        blockers.append("aurafusion360_interiorgs_source_modified")
    if not final_ply.is_file() or final_ply.stat().st_size == 0:
        blockers.append("aurafusion360_interiorgs_final_point_cloud_missing")
    retained = output / "artifacts/aurafusion360_840313_ins160_final.ply"
    if final_ply.is_file():
        retained.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_ply, retained)
    render_dir = source / f"output/Other-360/{SCENE}/train/ours_10000_object_inpaint/renders"
    retained_frames: list[dict[str, Any]] = []
    for frame in sorted(render_dir.glob("*.png"))[:8]:
        destination = output / "artifacts/final_frames" / frame.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(frame, destination)
        retained_frames.append({"path": destination.relative_to(output).as_posix(), "size_bytes": destination.stat().st_size, "sha256": _sha256(destination)})
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "completed" if not blockers else "blocked",
        "scene_id": "840313",
        "target_instance_id": "ins160",
        "source_commit": spec["source_commit"],
        "source_tree": spec["source_tree"],
        "source_identity_before": source_before,
        "source_identity_after": source_after,
        "source_modified": not source_after["matches"],
        "lama_checkpoint_materialization": checkpoint_materialization,
        "workflow": workflow,
        "reference_generation_executed": "reference_lama" in completed,
        "training_executed": "train" in completed,
        "removal_executed": "remove" in completed,
        "inpaint_init_executed": "inpaint_init" in completed,
        "inpaint_finetune_executed": "inpaint_finetune" in completed,
        "final_point_cloud": ({"relative_path": retained.relative_to(output).as_posix(), "size_bytes": retained.stat().st_size, "sha256": _sha256(retained), "vertex_count": _ply_count(retained)} if retained.is_file() else None),
        "final_frames": retained_frames,
        "claim_ceiling": "visual_candidate_only",
        "hidden_background_truth_available": False,
        "depth_anything3_used": False,
        "retry_cap": 0,
        "blockers": sorted(set(blockers)),
        "raw_secret_values_recorded": False,
    }
    _write(output / "adp_aura_interiorgs_result.json", result)
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
