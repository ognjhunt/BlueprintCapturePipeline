#!/usr/bin/env python3
"""Execute Joint Agent inference, deterministic review, and owned-core resume."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import yaml
from pxr import Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.joint_agent_articulation_review import (
    JointAgentArticulationReviewError,
    review_joint_agent_articulation,
)


SCHEMA_VERSION = "adp_joint_agent_result.v1"
ROOT = Path(__file__).resolve().parent
OUTPUT = Path(os.environ.get("BLUEPRINT_ADP_JOINT_AGENT_OUTPUT_DIR", ROOT / "../runtime_output")).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_mapping_required:{path.name}")
    return value


def retain_joint_agent_artifacts(
    *,
    output_root: Path,
    artifacts: Mapping[str, Path],
) -> list[dict[str, Any]]:
    """Copy every downstream construction input into the provider return tree.

    Joint Agent's configured working directory is intentionally separate from
    ``BLUEPRINT_ADP_JOINT_AGENT_OUTPUT_DIR``.  Merely recording an absolute
    in-container path therefore loses the rigged USD when the provider output
    ZIP is assembled.  This stable role-based seam makes retention explicit
    and verifies every copy before the live instance is torn down.
    """

    retained_root = output_root / "retained"
    retained_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for role, source in sorted(artifacts.items()):
        source = source.resolve()
        if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
            raise ValueError(f"joint_agent_retained_artifact_invalid:{role}")
        suffix = "".join(source.suffixes)
        destination = retained_root / f"{role}{suffix}"
        shutil.copy2(source, destination)
        if _sha256(destination) != _sha256(source):
            raise ValueError(f"joint_agent_retained_artifact_digest_mismatch:{role}")
        rows.append(
            {
                "role": role,
                "relative_path": destination.relative_to(output_root).as_posix(),
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
    return rows


def retain_available_joint_agent_artifacts(
    *,
    output_root: Path,
    artifacts: Mapping[str, Path],
) -> list[dict[str, Any]]:
    """Retain whichever construction artifacts exist on a failure path.

    A blocked review or publication must still return the model topology
    evidence that already exists, because only ``runtime_output`` survives the
    provider teardown ZIP. Missing files are skipped rather than raised so a
    partial failure cannot also destroy the retention of its own evidence.
    """

    present = {
        role: source.resolve()
        for role, source in artifacts.items()
        if not source.resolve().is_symlink()
        and source.resolve().is_file()
        and source.resolve().stat().st_size > 0
    }
    if not present:
        return []
    return retain_joint_agent_artifacts(output_root=output_root, artifacts=present)


def _load_optimize_usd_local():
    from world_understanding.functions.graphics.scene_optimizer_local import (
        optimize_usd_local,
    )

    return optimize_usd_local


def scene_optimizer_probe(*, output_root: Path) -> list[str]:
    """Run one tiny splitMeshes optimization and retain its real diagnostics.

    The released OptimizeUSDTask quarantines every exception into the bare
    string "USD optimization failed" (v10 retained exactly that and nothing
    else). Calling the underlying scene_optimizer_local entrypoint directly
    surfaces the subprocess's stdout/stderr, which this probe retains in
    scene_optimizer_probe.log before any paid model work begins.
    """

    import tempfile

    log = output_root / "scene_optimizer_probe.log"
    try:
        optimize_usd_local = _load_optimize_usd_local()
        from pxr import Sdf as _Sdf, Usd as _Usd, UsdGeom as _UsdGeom

        with tempfile.TemporaryDirectory(prefix="so_probe_") as tmp:
            probe_source = Path(tmp) / "probe.usda"
            stage = _Usd.Stage.CreateNew(str(probe_source))
            _UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            _UsdGeom.SetStageUpAxis(stage, _UsdGeom.Tokens.z)
            root = _UsdGeom.Xform.Define(stage, "/Probe")
            stage.SetDefaultPrim(root.GetPrim())
            mesh = _UsdGeom.Mesh.Define(stage, "/Probe/mesh")
            mesh.CreatePointsAttr(
                [(0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0), (3, 0, 0), (2, 1, 0)]
            )
            mesh.CreateFaceVertexCountsAttr([3, 3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4, 5])
            for index in (0, 1):
                subset = _UsdGeom.Subset.Define(
                    stage, f"/Probe/mesh/component_{index}"
                )
                subset.CreateElementTypeAttr().Set(_UsdGeom.Tokens.face)
                subset.CreateIndicesAttr().Set([index])
                subset.CreateFamilyNameAttr().Set("blueprint_connected_components")
                subset.GetPrim().SetMetadata(
                    "customData", {"probe": True}
                )
            del _Sdf
            stage.GetRootLayer().Save()
            result = optimize_usd_local(
                probe_source,
                Path(tmp) / "probe_optimized.usdc",
                {
                    "scene_optimizer_settings": {
                        "enable_deinstance": False,
                        "enable_split_meshes": True,
                        "enable_deduplicate": False,
                    }
                },
            )
        log.write_text(
            "scene_optimizer_probe_ok\n"
            + json.dumps(result, indent=2, sort_keys=True, default=str)
            + "\n",
            encoding="utf-8",
        )
        return []
    except Exception as exc:
        log.write_text(
            f"scene_optimizer_probe_failed\n{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        return ["joint_agent_scene_optimizer_probe_failed"]


def _run(command: list[str], log_name: str) -> dict[str, Any]:
    started = dt.datetime.now(dt.timezone.utc)
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=7200)
    ended = dt.datetime.now(dt.timezone.utc)
    log = OUTPUT / log_name
    log.write_text(process.stdout + "\n--- STDERR ---\n" + process.stderr, encoding="utf-8")
    return {
        "returncode": process.returncode,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "duration_seconds": (ended - started).total_seconds(),
        "log_path": str(log),
        "log_sha256": _sha256(log),
    }


def _candidate_bounds(stage_path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise ValueError("joint_agent_optimized_stage_open_failed")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    result: dict[str, Any] = {}
    for candidate in document.get("candidates") or []:
        candidate_id = str(candidate.get("candidate_id") or "")
        paths = [str(item) for item in candidate.get("moving_part_prims") or []]
        ranges = []
        for prim_path in paths:
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                ranges.append(cache.ComputeWorldBound(prim).ComputeAlignedRange())
        if not ranges:
            result[candidate_id] = {"status": "unmeasured", "moving_part_prims": paths}
            continue
        minimum = [min(float(item.GetMin()[axis]) for item in ranges) for axis in range(3)]
        maximum = [max(float(item.GetMax()[axis]) for item in ranges) for axis in range(3)]
        result[candidate_id] = {
            "status": "measured_from_optimized_usd",
            "moving_part_prims": paths,
            "aabb_min": minimum,
            "aabb_max": maximum,
        }
    return result


def _result(blockers: list[str], **values: Any) -> int:
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "completed" if not blockers else "blocked",
        "joint_agent_inference_executed": bool(values.get("joint_agent_inference_executed")),
        "owned_core_publication_executed": bool(values.get("owned_core_publication_executed")),
        "retry_cap": 0,
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "model_output_is_not_ground_truth": True,
            "owned_core_topology_is_not_simready_qualification": True,
            "physical_equivalence_proven": False,
        },
        "raw_secret_values_recorded": False,
        **values,
    }
    (OUTPUT / "adp_joint_agent_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if not blockers else 2


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    try:
        manifest = _load(ROOT / "adp_joint_agent_provider_manifest.json")
        contract = _load(ROOT / "joint_review_contract.json")
        if manifest.get("status") != "ready":
            raise ValueError("joint_agent_provider_manifest_not_ready")
        if canonical_digest(contract, digest_field="contract_digest") != contract.get("contract_digest"):
            raise ValueError("joint_agent_review_contract_digest_invalid")
        if not os.environ.get("NVIDIA_API_KEY"):
            raise ValueError("joint_agent_nvidia_api_key_missing")
        probe_blockers = scene_optimizer_probe(output_root=OUTPUT)
        if probe_blockers:
            return _result(probe_blockers)
        config_path = ROOT / "joint_agent.yaml"
        python = str(ROOT / "content_agents_source/.venv/bin/python")
        cli = [python, "-m", "joint_agent.cli", "run", str(config_path)]
        dry = _run([*cli, "--dry-run"], "joint_agent_dry_run.log")
        if dry["returncode"] != 0:
            return _result(["joint_agent_released_cli_dry_run_failed"], dry_run=dry)
        inference = _run(cli, "joint_agent_inference.log")
        candidates_path = ROOT / "runtime_output/joint_agent_work/articulation_candidates/articulation_candidates.json"
        optimized_path = ROOT / "runtime_output/joint_agent_work/optimize_usd/articulated_source_optimized.usdc"
        if inference["returncode"] != 0 or not candidates_path.is_file():
            return _result(
                ["joint_agent_topology_inference_failed"],
                dry_run=dry,
                inference=inference,
                joint_agent_inference_executed=False,
            )
        candidates = _load(candidates_path)
        partial_artifacts: dict[str, Path] = {
            "articulation_candidates": candidates_path,
            "optimized_source": optimized_path,
        }
        partial_rows = retain_available_joint_agent_artifacts(
            output_root=OUTPUT, artifacts=partial_artifacts
        )
        try:
            bounds = _candidate_bounds(optimized_path, candidates)
        except Exception as exc:
            return _result(
                [f"joint_agent_candidate_bounds_measurement_failed:{type(exc).__name__}"],
                dry_run=dry,
                inference=inference,
                joint_agent_inference_executed=True,
                candidates_sha256=_sha256(candidates_path),
                retained_artifacts=partial_rows,
            )
        (OUTPUT / "joint_candidate_bounds.json").write_text(
            json.dumps(bounds, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            review = review_joint_agent_articulation(
                candidates_document=candidates,
                candidate_bounds=bounds,
                review_contract=contract,
            )
        except JointAgentArticulationReviewError as exc:
            return _result(
                [f"joint_agent_deterministic_review_failed:{item}" for item in exc.errors],
                dry_run=dry,
                inference=inference,
                joint_agent_inference_executed=True,
                candidates_sha256=_sha256(candidates_path),
                candidate_bounds_sha256=_sha256(OUTPUT / "joint_candidate_bounds.json"),
                retained_artifacts=partial_rows,
            )
        review_path = OUTPUT / "joint_agent_articulation_review.json"
        review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["steps"]["apply_joint_rigger"]["enabled"] = True
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        publication = _run([*cli, "--resume"], "joint_agent_owned_core_resume.log")
        rigged = ROOT / "runtime_output/joint_agent_work/joint_rigger/rigged.usdz"
        if publication["returncode"] != 0 or not rigged.is_file():
            blockers.append("joint_agent_owned_core_publication_failed")
            partial_rows = retain_available_joint_agent_artifacts(
                output_root=OUTPUT,
                artifacts={
                    **partial_artifacts,
                    "owned_core_rigged_asset": rigged,
                    "owned_core_diagnostics": ROOT
                    / "runtime_output/joint_agent_work/joint_rigger/joint_rigger_diagnostics.json",
                    "owned_core_validation": ROOT
                    / "runtime_output/joint_agent_work/joint_rigger/joint_rigger_validation.json",
                },
            )
            return _result(
                blockers,
                dry_run=dry,
                inference=inference,
                publication=publication,
                joint_agent_inference_executed=True,
                owned_core_publication_executed=False,
                review_receipt_sha256=_sha256(review_path),
                retained_artifacts=partial_rows,
            )
        stage = Usd.Stage.Open(str(rigged))
        joint_paths = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.IsA(UsdPhysics.Joint)
        ] if stage is not None else []
        if len(joint_paths) != review["assembly_joint_count"]:
            blockers.append("joint_agent_owned_core_joint_count_readback_mismatch")
        diagnostics = ROOT / "runtime_output/joint_agent_work/joint_rigger/joint_rigger_diagnostics.json"
        validation = ROOT / "runtime_output/joint_agent_work/joint_rigger/joint_rigger_validation.json"
        retained_artifacts = retain_joint_agent_artifacts(
            output_root=OUTPUT,
            artifacts={
                "articulation_candidates": candidates_path,
                "candidate_bounds": OUTPUT / "joint_candidate_bounds.json",
                "deterministic_articulation_review": review_path,
                "optimized_source": optimized_path,
                "owned_core_rigged_asset": rigged,
                "owned_core_diagnostics": diagnostics,
                "owned_core_validation": validation,
            },
        )
        return _result(
            blockers,
            dry_run=dry,
            inference=inference,
            publication=publication,
            joint_agent_inference_executed=True,
            owned_core_publication_executed=not blockers,
            candidates_sha256=_sha256(candidates_path),
            candidate_bounds_sha256=_sha256(OUTPUT / "joint_candidate_bounds.json"),
            review_receipt_sha256=_sha256(review_path),
            rigged_usdz_path=str(rigged),
            rigged_usdz_sha256=_sha256(rigged),
            authored_joint_paths=joint_paths,
            retained_artifacts=retained_artifacts,
        )
    except Exception as exc:
        return _result([f"joint_agent_runtime_exception:{type(exc).__name__}"])


if __name__ == "__main__":
    raise SystemExit(main())
