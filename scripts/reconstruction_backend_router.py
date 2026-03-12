#!/usr/bin/env python3
"""Reconstruction backend router for Stage 1 reconstruction.

Routes reconstruction jobs to one of the supported backends.

Current supported backends:
- nurec_3dgrut (default; existing local shim)
- ttt_lrm (experimental integration for the tttLRM code path)
- loger (command-template integration)
- neoverse (remote Stage 1 service)
- gen3c (remote Stage 1 service)

The output directory contract remains NuRec-like:
- export_last.usdz
- nvblox_mesh.ply
- visual_mesh.glb
- mesh_manifest.json
- occupancy.bin
- object_point_cloud_index.json
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
NUREC_SHIM = REPO_ROOT / "scripts" / "nurec_shim.py"
NEOVERSE_RUNNER = REPO_ROOT / "scripts" / "run_neoverse_service.py"
GEN3C_RUNNER = REPO_ROOT / "scripts" / "run_gen3c_service.py"

BACKEND_NUREC_3DGRUT = "nurec_3dgrut"
BACKEND_TTT_LRM = "ttt_lrm"
BACKEND_LOGER = "loger"
BACKEND_NEOVERSE = "neoverse"
BACKEND_GEN3C = "gen3c"
REQUIRED_RECONTRACT_ARTIFACTS = (
    "export_last.usdz",
    "nvblox_mesh.ply",
    "visual_mesh.glb",
    "mesh_manifest.json",
    "occupancy.bin",
    "object_point_cloud_index.json",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(message: str) -> None:
    print(f"[reconstruction-router] {message}", flush=True)


def _normalize_backend_name(raw: str) -> str:
    value = (raw or "").strip().lower().replace("-", "_")
    if not value:
        return ""

    if value in {"nurec", "nurec3d", "nurec_3d", "nurec_3dgrut", "3dgrut", "nurec3dgrut"}:
        return BACKEND_NUREC_3DGRUT

    if value in {"ttt_lrm", "tttlrm", "ttt-lrm", "ttt", "ttt_lrm"}:
        return BACKEND_TTT_LRM

    if value in {"loger", "lo_ger", "lo-ger"}:
        return BACKEND_LOGER

    if value in {"neoverse", "neoverse"}:
        return BACKEND_NEOVERSE

    if value in {"gen3c", "gen_3c"}:
        return BACKEND_GEN3C

    raise ValueError(f"unsupported reconstruction backend '{raw}'")


def _parse_backend_csv(raw: str) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for token in (raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        normalized = _normalize_backend_name(token)
        if normalized not in seen:
            seen.add(normalized)
            items.append(normalized)
    return items


def _run_command(
    command: str | list[str],
    *,
    log_path: Path,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[int, str, str]:
    shell_mode = isinstance(command, str)
    final_env = dict(os.environ)
    if env:
        final_env.update(env)
    proc = subprocess.run(
        command,
        shell=shell_mode,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=final_env,
        capture_output=True,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    log_path.write_text(f"{stdout}\n{stderr}", encoding="utf-8")
    return proc.returncode, stdout, stderr


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _collect_file_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    info = {"path": str(path), "exists": True}
    try:
        info["size_bytes"] = int(path.stat().st_size)
    except OSError:
        pass
    return info


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if _is_nonempty_file(candidate):
            return candidate
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_ply_face_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as stream:
            first = stream.readline().strip().lower()
            if first != "ply":
                return 0
            while True:
                line = stream.readline()
                if not line:
                    return 0
                value = line.strip().lower()
                if value.startswith("element face "):
                    try:
                        return int(value.split()[-1])
                    except ValueError:
                        return 0
                if value == "end_header":
                    return 0
    except OSError:
        return 0
    return 0


def _copy_or_alias(
    source: Path | None,
    target: Path,
    *,
    notes: list[str],
    missing_note: str,
) -> Path | None:
    if source is None:
        notes.append(missing_note)
        return None
    if not source.exists():
        notes.append(missing_note)
        return None
    if source == target:
        return source
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        notes.append(f"aliased_{source.name}_to_{target.name}")
    except OSError as exc:
        notes.append(f"alias_failed_{source.name}_to_{target.name}: {exc}")
    return target


def _normalize_ttt_output(output_dir: Path) -> list[str]:
    notes: list[str] = []
    notes.extend(
        _normalize_output_contract(
            output_dir,
            source=BACKEND_TTT_LRM,
            preferred_visual_asset="export_last.usdz",
            mesh_method_default=os.getenv("TTT_LRM_MESH_METHOD", "poisson_open3d"),
        )
    )
    return notes


def _normalize_loger_output(output_dir: Path) -> list[str]:
    notes: list[str] = []
    notes.extend(
        _normalize_output_contract(
            output_dir,
            source=BACKEND_LOGER,
            preferred_visual_asset="visual_mesh.glb",
            mesh_method_default="loger_poisson",
        )
    )
    return notes


def _normalize_output_contract(
    output_dir: Path,
    *,
    source: str,
    preferred_visual_asset: str,
    mesh_method_default: str,
) -> list[str]:
    notes: list[str] = []
    if not output_dir.is_dir():
        return notes

    # USD / visual
    usdz_candidate = _first_existing(
        [
            output_dir / "export_last.usdz",
            output_dir / "scene.usdz",
            output_dir / "visual.usdz",
            output_dir / "export.usdz",
        ]
    )
    _copy_or_alias(usdz_candidate, output_dir / "export_last.usdz", notes=notes, missing_note="missing_export_last_usdz")

    # Collision mesh.
    collision_candidate = _first_existing(
        [
            output_dir / "nvblox_mesh.ply",
            output_dir / "collision_mesh.ply",
            output_dir / "mesh.ply",
            output_dir / "model.ply",
        ]
    )
    _copy_or_alias(collision_candidate, output_dir / "nvblox_mesh.ply", notes=notes, missing_note="missing_collision_mesh")

    # Visual mesh.
    visual_candidate = _first_existing(
        [
            output_dir / "visual_mesh.glb",
            output_dir / "mesh.glb",
            output_dir / "reconstructed_mesh.glb",
            output_dir / "inpainted_visual_mesh.glb",
        ]
    )
    if visual_candidate is None and _is_nonempty_file(output_dir / "export_last.usdz"):
        notes.append("visual_mesh_missing_from_ttt_output")
    _copy_or_alias(visual_candidate, output_dir / "visual_mesh.glb", notes=notes, missing_note="missing_visual_mesh_glb")

    # Occupancy.
    occupancy_candidate = _first_existing(
        [
            output_dir / "occupancy.bin",
            output_dir / "occupancy_0.bin",
            output_dir / "occupancy_1.bin",
        ]
    )
    if occupancy_candidate and occupancy_candidate.name != "occupancy.bin":
        try:
            shutil.copy2(occupancy_candidate, output_dir / "occupancy.bin")
            notes.append(f"aliased_{occupancy_candidate.name}_to_occupancy.bin")
        except OSError:
            notes.append("occupancy_alias_failed")
    elif occupancy_candidate is None:
        notes.append("missing_occupancy_bin")

    # Object index.
    object_index = _first_existing(
        [
            output_dir / "object_point_cloud_index.json",
            output_dir / "arkit_objects_index.json",
        ]
    )
    if object_index is not None and object_index.name != "object_point_cloud_index.json":
        _copy_or_alias(object_index, output_dir / "object_point_cloud_index.json", notes=notes, missing_note="")
    elif object_index is None:
        notes.append("missing_object_point_cloud_index")

    # Mesh manifest.
    manifest = _first_existing(
        [
            output_dir / "mesh_manifest.json",
            output_dir / "mesh_info.json",
        ]
    )
    if manifest is None or not manifest.is_file():
        manifest_payload = {
            "schema_version": "v1",
            "primary_visual_asset": (
                preferred_visual_asset
                if _is_nonempty_file(output_dir / preferred_visual_asset)
                else ("export_last.usdz" if _is_nonempty_file(output_dir / "export_last.usdz") else "")
            ),
            "generated_at": _utc_now_iso(),
            "source": source,
            "notes": "created_by_router",
        }
        _write_json(output_dir / "mesh_manifest.json", manifest_payload)
        notes.append("created_mesh_manifest_from_router")
    elif manifest != output_dir / "mesh_manifest.json":
        _copy_or_alias(manifest, output_dir / "mesh_manifest.json", notes=notes, missing_note="")

    # Backend marker.
    mesh_method = output_dir / "mesh_method.txt"
    if not mesh_method.exists():
        mesh_method.write_text(f"{mesh_method_default}\n", encoding="utf-8")

    return notes


def _run_nurec_3dgrut(
    *,
    job_spec_path: Path,
    input_video: Path,
    output_dir: Path,
    backend_args: list[str],
    log_path: Path,
) -> tuple[int, str, str]:
    if not NUREC_SHIM.is_file():
        return 1, "", f"missing nurec_shim.py at {NUREC_SHIM}"
    cmd = [
        sys.executable,
        str(NUREC_SHIM),
        "--job-spec",
        str(job_spec_path),
        "--output-dir",
        str(output_dir),
        "--raw-prefix",
        str(input_video),
    ]
    cmd.extend(backend_args)
    return _run_command(cmd, log_path=log_path)


def _build_ttt_command(
    *,
    input_video: Path,
    output_dir: Path,
    scene_id: str,
    capture_id: str,
    job_spec_path: Path,
) -> tuple[str | list[str] | None, bool]:
    template = os.getenv("TTT_LRM_CMD_TEMPLATE", "").strip()
    if template:
        try:
            command = template.format(
                INPUT_VIDEO=str(input_video),
                OUTPUT_DIR=str(output_dir),
                SCENE_ID=scene_id,
                CAPTURE_ID=capture_id,
                JOB_SPEC_PATH=str(job_spec_path),
            )
        except KeyError as exc:
            raise RuntimeError(
                f"invalid TTT_LRM_CMD_TEMPLATE: missing placeholder {exc}. "
                f"Expected INPUT_VIDEO, OUTPUT_DIR, SCENE_ID, CAPTURE_ID, JOB_SPEC_PATH"
            ) from exc
        return command, True

    executable = os.getenv("TTT_LRM_EXECUTABLE", "").strip()
    if not executable:
        return None, False
    try:
        parts = shlex.split(executable)
    except ValueError as exc:
        raise RuntimeError(
            f"invalid TTT_LRM_EXECUTABLE shell syntax: {executable}"
        ) from exc
    if not parts:
        return None, False
    parts.extend(["--input-video", str(input_video), "--output-dir", str(output_dir)])
    return parts, False


def _run_ttt_lrm(
    *,
    job_spec_path: Path,
    input_video: Path,
    output_dir: Path,
    scene_id: str,
    capture_id: str,
    log_path: Path,
) -> tuple[int, str, str]:
    cmd, use_shell = _build_ttt_command(
        input_video=input_video,
        output_dir=output_dir,
        scene_id=scene_id,
        capture_id=capture_id,
        job_spec_path=job_spec_path,
    )
    if cmd is None:
        return 1, "", "ttt_lrm backend not configured: set TTT_LRM_CMD_TEMPLATE or TTT_LRM_EXECUTABLE"
    return _run_command(cmd, log_path=log_path, env=os.environ)


def _build_loger_command(
    *,
    input_video: Path,
    output_dir: Path,
    scene_id: str,
    capture_id: str,
    job_spec_path: Path,
) -> tuple[str | list[str] | None, bool]:
    template = os.getenv("LOGER_CMD_TEMPLATE", "").strip()
    if template:
        try:
            command = template.format(
                INPUT_VIDEO=str(input_video),
                OUTPUT_DIR=str(output_dir),
                SCENE_ID=scene_id,
                CAPTURE_ID=capture_id,
                JOB_SPEC_PATH=str(job_spec_path),
            )
        except KeyError as exc:
            raise RuntimeError(
                f"invalid LOGER_CMD_TEMPLATE: missing placeholder {exc}. "
                f"Expected INPUT_VIDEO, OUTPUT_DIR, SCENE_ID, CAPTURE_ID, JOB_SPEC_PATH"
            ) from exc
        return command, True

    executable = os.getenv("LOGER_EXECUTABLE", "").strip()
    if not executable:
        return None, False
    try:
        parts = shlex.split(executable)
    except ValueError as exc:
        raise RuntimeError(f"invalid LOGER_EXECUTABLE shell syntax: {executable}") from exc
    if not parts:
        return None, False
    parts.extend(
        [
            "--input-video",
            str(input_video),
            "--output-dir",
            str(output_dir),
            "--scene-id",
            scene_id,
            "--capture-id",
            capture_id,
            "--job-spec",
            str(job_spec_path),
        ]
    )
    return parts, False


def _run_loger(
    *,
    job_spec_path: Path,
    input_video: Path,
    output_dir: Path,
    scene_id: str,
    capture_id: str,
    log_path: Path,
) -> tuple[int, str, str]:
    cmd, _use_shell = _build_loger_command(
        input_video=input_video,
        output_dir=output_dir,
        scene_id=scene_id,
        capture_id=capture_id,
        job_spec_path=job_spec_path,
    )
    if cmd is None:
        return 1, "", "loger backend not configured: set LOGER_CMD_TEMPLATE or LOGER_EXECUTABLE"
    return _run_command(cmd, log_path=log_path, env=os.environ)


def _run_neoverse(
    *,
    job_spec_path: Path,
    output_dir: Path,
    scene_id: str,
    capture_id: str,
    log_path: Path,
) -> tuple[int, str, str]:
    if not NEOVERSE_RUNNER.is_file():
        return 1, "", f"missing run_neoverse_service.py at {NEOVERSE_RUNNER}"
    cmd = [
        sys.executable,
        str(NEOVERSE_RUNNER),
        "--job-spec",
        str(job_spec_path),
        "--output-dir",
        str(output_dir),
        "--scene-id",
        scene_id,
        "--capture-id",
        capture_id,
    ]
    return _run_command(cmd, log_path=log_path, env=os.environ)


def _run_gen3c(
    *,
    job_spec_path: Path,
    output_dir: Path,
    scene_id: str,
    capture_id: str,
    log_path: Path,
) -> tuple[int, str, str]:
    if not GEN3C_RUNNER.is_file():
        return 1, "", f"missing run_gen3c_service.py at {GEN3C_RUNNER}"
    cmd = [
        sys.executable,
        str(GEN3C_RUNNER),
        "--job-spec",
        str(job_spec_path),
        "--output-dir",
        str(output_dir),
        "--scene-id",
        scene_id,
        "--capture-id",
        capture_id,
    ]
    return _run_command(cmd, log_path=log_path, env=os.environ)


def _normalize_neoverse_output(output_dir: Path) -> list[str]:
    notes: list[str] = []
    report_path = output_dir / "neoverse_backend_report.json"
    if report_path.is_file():
        notes.append("neoverse_backend_report_present")
    return notes


def _normalize_gen3c_output(output_dir: Path) -> list[str]:
    notes: list[str] = []
    report_path = output_dir / "gen3c_backend_report.json"
    if report_path.is_file():
        notes.append("gen3c_backend_report_present")
    return notes


def _artifact_contract_checks(output_dir: Path) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    files: dict[str, dict[str, Any]] = {}

    for artifact in REQUIRED_RECONTRACT_ARTIFACTS:
        artifact_path = output_dir / artifact
        exists = _is_nonempty_file(artifact_path)
        checks[artifact] = exists
        files[artifact] = _collect_file_summary(artifact_path)

    capture_quality = _read_json(output_dir / "capture_quality_report.json")
    manifest = _read_json(output_dir / "mesh_manifest.json")
    object_index = _read_json(output_dir / "object_point_cloud_index.json")

    if isinstance(object_index, list):
        object_index_shape_ok = True
    elif isinstance(object_index, dict):
        if "objects" in object_index:
            objects_payload = object_index.get("objects")
            object_index_shape_ok = isinstance(objects_payload, list)
        else:
            object_index_shape_ok = True
    else:
        object_index_shape_ok = False

    downstream_ok = (
        checks["export_last.usdz"]
        and checks["nvblox_mesh.ply"]
        and checks["visual_mesh.glb"]
        and checks["mesh_manifest.json"]
        and checks["occupancy.bin"]
        and checks["object_point_cloud_index.json"]
        and bool(capture_quality)
        and bool(manifest)
        and bool(manifest.get("primary_visual_asset"))
        and object_index_shape_ok
    )

    return {
        "required": checks,
        "required_all_pass": all(checks.values()),
        "missing_required": [artifact for artifact, present in checks.items() if not present],
        "downstream_compatibility_ok": bool(downstream_ok),
        "files": files,
        "capture_quality_exists": bool(capture_quality),
        "manifest_exists": bool(manifest),
        "object_index_exists": bool(object_index),
    }


def _collect_quality_metrics(output_dir: Path) -> dict[str, Any]:
    capture_quality = _read_json(output_dir / "capture_quality_report.json")
    refinement_gate = _read_json(output_dir / "refinement_quality_gate.json")
    gate_status = str(refinement_gate.get("status") or "").strip().lower()
    frame_count = int(capture_quality.get("frame_count", 0) or 0)
    return {
        "capture_quality": {
            "frame_count": frame_count,
            "passed": frame_count > 0,
            "payload": capture_quality,
        },
        "refinement_gate_status": gate_status or "missing",
        "refinement_gate_payload": refinement_gate,
    }


def _compute_hallucination_risk(
    *,
    contract: dict[str, Any],
    quality: dict[str, Any],
    placeholders: list[str],
) -> dict[str, Any]:
    risk = 0.0
    signals: list[str] = []

    if not contract.get("required_all_pass"):
        missing = contract.get("missing_required", [])
        risk += 0.65
        signals.append(f"missing_required_artifacts:{','.join(missing)}")

    if not contract.get("downstream_compatibility_ok"):
        risk += 0.15
        signals.append("downstream compatibility checks failed")

    gate_status = str(quality.get("refinement_gate_status") or "missing").lower()
    if gate_status == "passed":
        risk += 0.02
        signals.append("refinement gate passed")
    elif gate_status == "failed":
        risk += 0.45
        signals.append("refinement gate failed")
    else:
        risk += 0.25
        signals.append("refinement gate missing")

    if not quality.get("capture_quality", {}).get("passed"):
        risk += 0.12
        signals.append("capture quality report missing or empty")

    for note in placeholders:
        if note.startswith("missing_") or note.endswith("failed"):
            risk += 0.05
            signals.append(note)

    risk = min(1.0, round(risk, 6))
    if risk >= 0.7:
        tier = "high"
    elif risk >= 0.35:
        tier = "medium"
    else:
        tier = "low"

    return {"score": risk, "tier": tier, "signals": signals}


def _compute_geometry_score(
    *,
    output_dir: Path,
    contract: dict[str, Any],
    quality: dict[str, Any],
    runtime_sec: float,
) -> float:
    req_ok = 1.0 if contract.get("required_all_pass") else 0.0
    compatibility_bonus = 1.0 if contract.get("downstream_compatibility_ok") else 0.0

    mesh_faces = _read_ply_face_count(output_dir / "nvblox_mesh.ply")
    frame_count = int(quality.get("capture_quality", {}).get("frame_count", 0) or 0)

    face_score = min(1.0, mesh_faces / 500_000.0)
    frame_score = min(1.0, frame_count / 180.0)
    gate_status = str(quality.get("refinement_gate_status") or "missing").lower()
    gate_score = 1.0 if gate_status == "passed" else 0.0

    runtime_penalty = min(1.0, runtime_sec / 7200.0)
    return round(
        28.0 * req_ok
        + 20.0 * compatibility_bonus
        + 20.0 * face_score
        + 20.0 * frame_score
        + 12.0 * gate_score
        - 20.0 * runtime_penalty,
        3,
    )


def _evaluate_run(
    *,
    backend: str,
    output_dir: Path,
    runtime_sec: float,
    compatibility: dict[str, Any] | None = None,
    placeholders: list[str] | None = None,
) -> dict[str, Any]:
    compatibility = compatibility or _artifact_contract_checks(output_dir)
    quality = _collect_quality_metrics(output_dir)
    placeholders = placeholders or []

    risk = _compute_hallucination_risk(
        contract=compatibility,
        quality=quality,
        placeholders=placeholders,
    )
    geometry_score = _compute_geometry_score(
        output_dir=output_dir,
        contract=compatibility,
        quality=quality,
        runtime_sec=runtime_sec,
    )

    return {
        "backend": backend,
        "runtime_sec": runtime_sec,
        "contract": compatibility,
        "quality": quality,
        "hallucination_risk": risk,
        "geometry_score": geometry_score,
        "artifact_compatibility_ok": bool(compatibility.get("required_all_pass")),
        "downstream_compatibility_ok": bool(compatibility.get("downstream_compatibility_ok")),
        "required_backend_contract_files": compatibility.get("files", {}),
        "placeholders": list(placeholders),
    }


def _choose_winner(
    *,
    runs: dict[str, dict[str, Any]],
    winner_choice: str,
) -> str | None:
    if winner_choice != "auto":
        selected = _normalize_backend_name(winner_choice)
        if selected in runs and runs[selected].get("status") == "passed":
            return selected
        return None

    scored: list[tuple[float, str]] = []
    for backend, payload in runs.items():
        if payload.get("status") != "passed":
            continue
        score = float(payload.get("geometry_score", 0.0)) - float(payload.get("hallucination_risk", {}).get("score", 0.0) * 45.0)
        scored.append((score, backend))

    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][1]


def _sync_outputs(*, source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    if not source.exists():
        raise RuntimeError(f"winner output path missing: {source}")
    shutil.copytree(source, target)


def run_reconstruction(
    *,
    primary_backend: str,
    compare_backends: list[str],
    compare_winner: str,
    job_spec_path: Path,
    input_video: Path,
    output_dir: Path,
    scene_id: str,
    capture_id: str,
    backend_args: list[str],
    compare_report: Path,
) -> tuple[dict[str, Any], str]:
    candidates = [primary_backend]
    for backend in compare_backends:
        if backend not in candidates:
            candidates.append(backend)

    candidate_root = output_dir.parent / f"{output_dir.name}.candidate_backends"
    if candidate_root.exists():
        shutil.rmtree(candidate_root)
    candidate_root.mkdir(parents=True, exist_ok=True)

    run_results: dict[str, Any] = {}

    for backend in candidates:
        candidate_output = candidate_root / backend
        if candidate_output.exists():
            shutil.rmtree(candidate_output)
        candidate_output.mkdir(parents=True, exist_ok=True)
        log_path = candidate_output / "reconstruction.log"
        _log(f"Running backend={backend} -> {candidate_output}")

        start_ts = datetime.now(timezone.utc).timestamp()
        notes: list[str] = []
        if backend == BACKEND_NUREC_3DGRUT:
            return_code, stdout, stderr = _run_nurec_3dgrut(
                job_spec_path=job_spec_path,
                input_video=input_video,
                output_dir=candidate_output,
                backend_args=backend_args,
                log_path=log_path,
            )
        elif backend == BACKEND_TTT_LRM:
            return_code, stdout, stderr = _run_ttt_lrm(
                job_spec_path=job_spec_path,
                input_video=input_video,
                output_dir=candidate_output,
                scene_id=scene_id,
                capture_id=capture_id,
                log_path=log_path,
            )
            notes = _normalize_ttt_output(candidate_output)
        elif backend == BACKEND_LOGER:
            return_code, stdout, stderr = _run_loger(
                job_spec_path=job_spec_path,
                input_video=input_video,
                output_dir=candidate_output,
                scene_id=scene_id,
                capture_id=capture_id,
                log_path=log_path,
            )
            notes = _normalize_loger_output(candidate_output)
        elif backend == BACKEND_NEOVERSE:
            return_code, stdout, stderr = _run_neoverse(
                job_spec_path=job_spec_path,
                output_dir=candidate_output,
                scene_id=scene_id,
                capture_id=capture_id,
                log_path=log_path,
            )
            notes = _normalize_neoverse_output(candidate_output)
        elif backend == BACKEND_GEN3C:
            return_code, stdout, stderr = _run_gen3c(
                job_spec_path=job_spec_path,
                output_dir=candidate_output,
                scene_id=scene_id,
                capture_id=capture_id,
                log_path=log_path,
            )
            notes = _normalize_gen3c_output(candidate_output)
        else:
            return_code, stdout, stderr = 1, "", f"unsupported backend: {backend}"
            notes = [f"unsupported_backend:{backend}"]

        elapsed = max(0.0, datetime.now(timezone.utc).timestamp() - start_ts)
        if return_code == 0 and backend == BACKEND_TTT_LRM:
            # Ensure output contract alignment when ttt path is using different names.
            notes = _normalize_ttt_output(candidate_output)
        if return_code == 0 and backend == BACKEND_LOGER:
            notes = _normalize_loger_output(candidate_output)
        if return_code == 0 and backend == BACKEND_NEOVERSE:
            notes = _normalize_neoverse_output(candidate_output)
        if return_code == 0 and backend == BACKEND_GEN3C:
            notes = _normalize_gen3c_output(candidate_output)

        compatibility = _artifact_contract_checks(candidate_output) if return_code == 0 else {}
        metrics = _evaluate_run(
            backend=backend,
            output_dir=candidate_output,
            runtime_sec=elapsed,
            compatibility=compatibility,
            placeholders=notes,
        )
        metrics["status"] = "passed" if return_code == 0 else "failed"
        metrics["command"] = {
            "backend": backend,
            "return_code": int(return_code),
            "log_path": str(log_path),
            "stdout_tail": (stdout or "")[-1024:],
            "stderr_tail": (stderr or "")[-1024:],
            "notes": notes,
        }
        if return_code != 0:
            metrics["error"] = (stderr or "")[-2000:]
        run_results[backend] = metrics

    selected = _choose_winner(runs=run_results, winner_choice=compare_winner)
    if selected is None:
        raise RuntimeError("no successful backend candidate available for comparison")

    selected_output = candidate_root / selected
    _sync_outputs(source=selected_output, target=output_dir)

    report = {
        "schema_version": "v1",
        "generated_at": _utc_now_iso(),
        "input_video": str(input_video),
        "scene_id": scene_id,
        "capture_id": capture_id,
        "primary_backend": primary_backend,
        "compare_backends": compare_backends,
        "requested_winner": compare_winner,
        "selected_winner": selected,
        "runs": run_results,
        "output_dir": str(output_dir),
        "candidate_output_root": str(candidate_root),
    }
    _write_json(compare_report, report)
    return report, selected


def _build_report_only_message(report_path: Path) -> None:
    report = _read_json(report_path)
    winner = str(report.get("selected_winner") or "unknown")
    candidates = list((report.get("runs") or {}).keys())
    _log(f"Comparison report written: {report_path}")
    _log(f"Reconstruction winner: {winner}")
    _log(f"Candidate backends: {', '.join(candidates)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reconstruction backend router")
    parser.add_argument("--backend", default=BACKEND_NUREC_3DGRUT)
    parser.add_argument("--compare-backends", default="")
    parser.add_argument("--compare-winner", default="auto")
    parser.add_argument("--compare-report", default="")
    parser.add_argument("--job-spec", required=True)
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-id", default="")
    parser.add_argument("--capture-id", default="")
    parser.add_argument("backend_args", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)
    if args.backend_args and args.backend_args[0] == "--":
        args.backend_args = args.backend_args[1:]

    primary_backend = _normalize_backend_name(args.backend)
    compare_winner = args.compare_winner.strip().lower()
    if compare_winner and compare_winner != "auto":
        compare_winner = _normalize_backend_name(compare_winner)

    compare_backends = _parse_backend_csv(args.compare_backends)
    if primary_backend in compare_backends:
        compare_backends = [backend for backend in compare_backends if backend != primary_backend]

    output_dir = Path(args.output_dir)
    compare_report = Path(
        args.compare_report or str(output_dir / "reconstruction_compare_report.json")
    )
    job_spec_path = Path(args.job_spec)
    input_video = Path(args.input_video)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if compare_report.parent and not compare_report.parent.exists():
        compare_report.parent.mkdir(parents=True, exist_ok=True)

    report, selected = run_reconstruction(
        primary_backend=primary_backend,
        compare_backends=compare_backends,
        compare_winner=compare_winner,
        job_spec_path=job_spec_path,
        input_video=input_video,
        output_dir=output_dir,
        scene_id=args.scene_id,
        capture_id=args.capture_id,
        backend_args=args.backend_args,
        compare_report=compare_report,
    )

    _build_report_only_message(compare_report)
    _log(f"Selected winner output copied to: {output_dir}")
    _log(f"Selected winner backend: {selected}")
    _write_json(
        output_dir / "reconstruction_backend_meta.json",
        {
            "selected_winner": selected,
            "report": str(compare_report),
            "candidates": [primary_backend] + compare_backends,
            "runs": {
                backend: {
                    "status": run.get("status"),
                    "runtime_sec": run.get("runtime_sec"),
                    "geometry_score": run.get("geometry_score"),
                    "hallucination_risk": run.get("hallucination_risk", {}),
                    "artifact_compatibility_ok": run.get("artifact_compatibility_ok"),
                    "downstream_compatibility_ok": run.get("downstream_compatibility_ok"),
                }
                for backend, run in (report.get("runs") or {}).items()
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
