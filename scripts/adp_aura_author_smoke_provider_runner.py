#!/usr/bin/env python3
"""Execute AuraFusion360's unchanged sunflower inpaint-init author command once."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "adp_aura_author_smoke_result.v1"
COMMAND_TIMEOUT_SECONDS = 5400


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(command: Sequence[str], *, cwd: Path, log_path: Path) -> dict[str, Any]:
    started = dt.datetime.now(dt.timezone.utc)
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
        returncode = completed.returncode
        output = completed.stdout + completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        output = stdout + stderr
        timed_out = True
    log_path.write_text(output, encoding="utf-8")
    finished = dt.datetime.now(dt.timezone.utc)
    return {
        "command": [str(item) for item in command],
        "cwd": str(cwd),
        "returncode": returncode,
        "timed_out": timed_out,
        "runtime_seconds": (finished - started).total_seconds(),
        "stdout_stderr_sha256": _sha256(log_path),
        "log": log_path.name,
    }


def _ply_vertex_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        for raw in stream:
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                try:
                    return int(line.rsplit(" ", 1)[-1])
                except ValueError:
                    return None
            if line == "end_header":
                return None
    return None


def _source_identity(source: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    changed: list[str] = []
    for record in manifest.get("source_files") or []:
        path = source / str(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _sha256(path) != record["sha256"]
        ):
            changed.append(str(record["path"]))
    return {"matches": not changed, "changed_files": changed[:100]}


def _pin_hf_ref(cache: Path, repository: str, revision: str) -> None:
    repo_dir = cache / ("models--" + repository.replace("/", "--"))
    refs = repo_dir / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(revision, encoding="utf-8")


def _extract_author_data(runtime: Path, source: Path, spec: dict[str, Any]) -> None:
    data = spec["author_data"]
    archive = runtime / data["archive"]
    if not archive.is_file() or _sha256(archive) != data["archive_sha256"]:
        raise ValueError("aurafusion360_author_data_archive_changed")
    destination = source / "data"
    selected = destination / data["path_prefix"]
    if selected.is_dir():
        shutil.rmtree(selected)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source_archive:
        for member in source_archive.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError("aurafusion360_author_data_archive_path_traversal")
        source_archive.extractall(destination)
    expected_paths: set[str] = set()
    for item in data["files"]:
        relative = str(item["path"])
        path = destination / relative
        if (
            not path.is_file()
            or path.stat().st_size != item["size_bytes"]
            or _sha256(path) != item["sha256"]
        ):
            raise ValueError("aurafusion360_author_data_materialized_bytes_changed")
        expected_paths.add(relative)
    actual_paths = {
        path.relative_to(destination).as_posix()
        for path in selected.rglob("*")
        if path.is_file()
    }
    expected_selected = {
        path for path in expected_paths if path.startswith(data["path_prefix"])
    }
    if actual_paths != expected_selected:
        raise ValueError("aurafusion360_author_data_materialized_file_set_changed")


def _prepare(runtime: Path, source: Path, spec: dict[str, Any]) -> int:
    from huggingface_hub import hf_hub_download, snapshot_download

    _extract_author_data(runtime, source, spec)
    expected = spec["expected_output"]
    expected_ply = Path(
        hf_hub_download(
            repo_id=expected["repository"],
            repo_type="dataset",
            revision=expected["revision"],
            filename=expected["expected_ply_path"],
        )
    )
    reference = runtime / "published_expected_point_cloud.ply"
    shutil.copy2(expected_ply, reference)
    if _sha256(reference) != expected["expected_ply_sha256"]:
        raise ValueError("aurafusion360_published_expected_ply_changed")

    cache = Path(os.environ["HF_HUB_CACHE"])
    for model in spec["runtime_models"]:
        snapshot_download(repo_id=model["repository"], revision=model["revision"])
        _pin_hf_ref(cache, model["repository"], model["revision"])
    sd2 = spec["sd2_checkpoint"]
    checkpoint = hf_hub_download(
        repo_id=sd2["repository"],
        revision=sd2["revision"],
        filename=sd2["path"],
    )
    destination = source / "utils/LeftRefill/pretrained_models/512-inpainting-ema.ckpt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, destination)
    if (
        destination.stat().st_size != sd2["size_bytes"]
        or _sha256(destination) != sd2["sha256"]
    ):
        raise ValueError("aurafusion360_sd2_checkpoint_changed")

    working_output = source / "output/360-USID/sunflower"
    for relative in (
        "point_cloud/iteration_object_inpaint_init",
        "train/ours_object_inpaint_init",
        "test/ours_object_inpaint_init",
        "traj/ours_object_inpaint_init",
    ):
        path = working_output / relative
        if path.is_dir():
            shutil.rmtree(path)
    _write_json(runtime / "prepare_receipt.json", {"status": "prepared"})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    runtime = Path(__file__).resolve().parent
    source = runtime / "AuraFusion360_official"
    output = Path(
        os.environ.get("BLUEPRINT_ADP_AURA_OUTPUT_DIR", runtime.parent / "runtime_output")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = _read_json(runtime / "smoke_spec.json")
    if args.prepare_only:
        return _prepare(runtime, source, spec)

    source_before = _source_identity(source, spec)
    hardware = _run(
        ["nvidia-smi", "-q"],
        cwd=source,
        log_path=output / "nvidia-smi.log",
    )
    command = [
        str(source / ".venv/bin/python"),
        "inpaint.py",
        "--config",
        "configs/360-USID/sunflower/inpaint.config",
    ]
    execution = _run(command, cwd=source, log_path=output / "aura-inpaint-init.log")
    produced = source / "output/360-USID/sunflower/point_cloud/iteration_object_inpaint_init/point_cloud.ply"
    expected = runtime / "published_expected_point_cloud.ply"
    source_after = _source_identity(source, spec)
    blockers: list[str] = []
    if not source_before["matches"] or not source_after["matches"]:
        blockers.append("aurafusion360_author_source_modified")
    if execution["returncode"] != 0:
        blockers.append("aurafusion360_inpaint_init_command_failed")
    if hardware["returncode"] != 0:
        blockers.append("aurafusion360_nvidia_hardware_probe_failed")
    if not produced.is_file() or produced.stat().st_size == 0:
        blockers.append("aurafusion360_inpaint_init_point_cloud_missing")
    retained_produced = output / "artifacts/aurafusion360_inpaint_init_point_cloud.ply"
    if produced.is_file():
        retained_produced.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, retained_produced)
    freeze = output / "pip-freeze.txt"
    if not freeze.is_file() or freeze.stat().st_size == 0:
        blockers.append("aurafusion360_python_environment_receipt_missing")
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "completed" if not blockers else "blocked",
        "source_commit": spec["source_commit"],
        "source_tree": spec["source_tree"],
        "source_identity_before": source_before,
        "source_identity_after": source_after,
        "author_source_modified": not source_after["matches"],
        "author_command": execution,
        "hardware_probe": hardware,
        "inpaint_init_executed": execution["returncode"] == 0 and produced.is_file(),
        "published_expected_output_bound": expected.is_file()
        and _sha256(expected) == spec["expected_output"]["expected_ply_sha256"],
        "produced_point_cloud": (
            {
                "size_bytes": produced.stat().st_size,
                "sha256": _sha256(produced),
                "vertex_count": _ply_vertex_count(produced),
                "retained_relative_path": retained_produced.relative_to(output).as_posix(),
            }
            if produced.is_file()
            else None
        ),
        "published_expected_point_cloud": {
            "size_bytes": expected.stat().st_size,
            "sha256": _sha256(expected),
            "vertex_count": _ply_vertex_count(expected),
        },
        "python_environment": (
            {"path": freeze.name, "sha256": _sha256(freeze)} if freeze.is_file() else None
        ),
        "depth_model": "prs-eth/marigold-depth-v1-0",
        "depth_anything3_used": False,
        "retry_cap": 0,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }
    _write_json(output / "adp_aura_author_smoke_result.json", result)
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
