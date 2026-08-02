"""Headless PlayCanvas SplatTransform collision-candidate adapter."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import subprocess
from typing import Sequence

from .decision_evidence_contracts import canonical_digest
from .gaussian_splat_decode import find_splat_transform_cli


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _blocked(*blockers: str) -> dict:
    result = {
        "schema_version": "splat_transform_collision_candidate.v1",
        "status": "blocked",
        "blockers": sorted(set(blockers)),
        "collision_validated": False,
        "metric_scale_validated": False,
        "evaluation_authorized": False,
        "claim_ceiling": "none",
    }
    result["candidate_digest"] = canonical_digest(result, digest_field="candidate_digest")
    return result


def generate_splat_transform_collision_candidate(
    src: str | Path,
    output_voxel_json: str | Path,
    *,
    repo_root: str | Path | None = None,
    node: str = "node",
    timeout_seconds: int = 3_600,
    robust_bounds: Sequence[float] | None = None,
    voxel_size: float = 0.05,
    voxel_opacity: float = 0.1,
    collision_mesh: str = "faces",
    source_up_axis: str = "Y",
    source_handedness: str = "right",
) -> dict:
    """Generate upstream voxel and GLB artifacts as unqualified candidates.

    SplatTransform owns decoding, voxelization, and surface extraction. Blueprint
    owns immutable lineage and requires separate coordinate, metric, collider,
    and simulator-contact qualification before the GLB can support evaluation.
    """

    source = Path(src).expanduser().resolve()
    output = Path(output_voxel_json).expanduser().resolve()
    cli = find_splat_transform_cli(repo_root)
    blockers: list[str] = []
    if cli is None:
        blockers.append("splat_transform_cli_unavailable")
    if not source.is_file():
        blockers.append("splat_source_missing")
    if source == output:
        blockers.append("immutable_source_overwrite_forbidden")
    if not output.name.endswith(".voxel.json"):
        blockers.append("splat_collision_output_must_end_voxel_json")
    if collision_mesh not in {"faces", "smooth"}:
        blockers.append("splat_collision_mesh_mode_invalid")
    if source_up_axis not in {"Y", "Z"} or source_handedness != "right":
        blockers.append("splat_collision_source_coordinate_frame_invalid")
    if not math.isfinite(float(voxel_size)) or float(voxel_size) <= 0:
        blockers.append("splat_collision_voxel_size_invalid")
    if not math.isfinite(float(voxel_opacity)) or not 0 < float(voxel_opacity) < 1:
        blockers.append("splat_collision_voxel_opacity_invalid")
    if robust_bounds is not None and (
        len(robust_bounds) != 6
        or any(not math.isfinite(float(value)) for value in robust_bounds)
        or any(float(robust_bounds[index]) > float(robust_bounds[index + 3]) for index in range(3))
    ):
        blockers.append("splat_collision_robust_bounds_invalid")
    if blockers:
        return _blocked(*blockers)

    assert cli is not None
    try:
        version = subprocess.run(
            [node, str(cli), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return _blocked("node_runtime_unavailable")
    if version.returncode != 0:
        return _blocked("splat_transform_version_probe_failed")

    command = [
        node,
        str(cli),
        "--no-tty",
        "-w",
        str(source),
        "--stats",
        "--filter-nan",
        "--filter-value=opacity,lt,0.999999",
    ]
    normalized_bounds = None
    if robust_bounds is not None:
        normalized_bounds = [float(value) for value in robust_bounds]
        command.append("--filter-box=" + ",".join(f"{value:.12g}" for value in normalized_bounds))
    command.extend(
        [
            "--stats",
            str(output),
            f"--voxel-size={float(voxel_size):.12g}",
            f"--voxel-opacity={float(voxel_opacity):.12g}",
            f"--collision-mesh={collision_mesh}",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return _blocked("splat_transform_collision_timeout")
    prefix = output.name[: -len(".voxel.json")]
    binary = output.with_name(f"{prefix}.voxel.bin")
    collision = output.with_name(f"{prefix}.collision.glb")
    counts = [
        int(match.group(1))
        for match in re.finditer(r"(?m)^gaussians:\s*(\d+)\s*$", process.stdout or "")
    ]
    if (
        process.returncode != 0
        or len(counts) < 2
        or not all(path.is_file() for path in (output, binary, collision))
    ):
        result = _blocked("splat_transform_collision_generation_failed")
        result.update(
            returncode=process.returncode,
            stderr_tail=(process.stderr or "")[-2000:],
            stdout_tail=(process.stdout or "")[-4000:],
        )
        result["candidate_digest"] = canonical_digest(result, digest_field="candidate_digest")
        return result

    result = {
        "schema_version": "splat_transform_collision_candidate.v1",
        "status": "candidate_generated",
        "blockers": [
            "coordinate_frame_qualification_pending",
            "independent_metric_scale_pending",
            "isaac_collider_and_contact_qualification_pending",
        ],
        "source_asset_digest": _sha256_path(source),
        "upstream": {
            "tool": "@playcanvas/splat-transform",
            "version_output": (version.stdout or version.stderr or "").strip(),
            "implementation_digest": _sha256_path(cli),
        },
        "actions": {
            "filter_nonfinite": True,
            "opacity_upper_bound_exclusive": 0.999999,
            "robust_bounds": normalized_bounds,
            "global_decimation_applied": False,
            "voxel_size": float(voxel_size),
            "voxel_opacity": float(voxel_opacity),
            "collision_mesh": collision_mesh,
            "coordinate_transform_applied": False,
        },
        "source_coordinate_frame": {
            "up_axis": source_up_axis,
            "handedness": source_handedness,
        },
        "output_coordinate_frame": {
            "up_axis": source_up_axis,
            "handedness": source_handedness,
            "basis": "source_preserved",
        },
        "source_splat_count": counts[0],
        "retained_splat_count": counts[-1],
        "retained_splat_fraction": round(counts[-1] / counts[0], 12),
        "removed_splat_count": counts[0] - counts[-1],
        "artifacts": {
            "voxel_json": {"digest": _sha256_path(output), "bytes": output.stat().st_size},
            "voxel_binary": {"digest": _sha256_path(binary), "bytes": binary.stat().st_size},
            "collision_glb": {
                "digest": _sha256_path(collision),
                "bytes": collision.stat().st_size,
            },
        },
        "collision_validated": False,
        "metric_scale_validated": False,
        "evaluation_authorized": False,
        "claim_ceiling": "splat_derived_collision_candidate",
    }
    result["candidate_digest"] = canonical_digest(result, digest_field="candidate_digest")
    return result


__all__ = ["generate_splat_transform_collision_candidate"]
