"""Higher-order contact robustness and performance gates for swap scenes."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .common import StageError, has_nonempty_file, parse_bool, utc_now_iso


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class AdvancedQualityGateConfig:
    enabled: bool = parse_bool(os.getenv("ADVANCED_QUALITY_GATES_ENABLED"), default=True)
    drop_probe_grid: int = _safe_int(os.getenv("QUALITY_DROP_PROBE_GRID"), 3)
    drop_steps: int = _safe_int(os.getenv("QUALITY_DROP_STEPS"), 240)
    drop_probe_height_m: float = _safe_float(os.getenv("QUALITY_DROP_PROBE_HEIGHT_M"), 1.2)
    drop_min_pass_rate: float = _safe_float(os.getenv("QUALITY_DROP_MIN_PASS_RATE"), 0.9)
    floor_penetration_tolerance_m: float = _safe_float(
        os.getenv("QUALITY_FLOOR_PENETRATION_TOLERANCE_M"), 0.05
    )

    jitter_settle_steps: int = _safe_int(os.getenv("QUALITY_JITTER_SETTLE_STEPS"), 180)
    jitter_measure_steps: int = _safe_int(os.getenv("QUALITY_JITTER_MEASURE_STEPS"), 180)
    jitter_extra_settle_steps: int = _safe_int(os.getenv("QUALITY_JITTER_EXTRA_SETTLE_STEPS"), 480)
    jitter_rest_speed_mps: float = _safe_float(os.getenv("QUALITY_JITTER_REST_SPEED_MPS"), 0.03)
    jitter_rest_frames: int = _safe_int(os.getenv("QUALITY_JITTER_REST_FRAMES"), 24)
    jitter_max_drift_m: float = _safe_float(os.getenv("QUALITY_JITTER_MAX_DRIFT_M"), 0.04)
    jitter_max_vertical_span_m: float = _safe_float(
        os.getenv("QUALITY_JITTER_MAX_VERTICAL_SPAN_M"), 0.02
    )

    tunneling_steps: int = _safe_int(os.getenv("QUALITY_TUNNELING_STEPS"), 200)
    tunneling_initial_speed_mps: float = _safe_float(
        os.getenv("QUALITY_TUNNELING_INITIAL_SPEED_MPS"), 20.0
    )
    tunneling_max_penetration_m: float = _safe_float(
        os.getenv("QUALITY_TUNNELING_MAX_PENETRATION_M"), 0.06
    )

    perf_steps: int = _safe_int(os.getenv("QUALITY_PERF_STEPS"), 240)
    perf_max_step_ms: float = _safe_float(os.getenv("QUALITY_PERF_MAX_STEP_MS"), 25.0)
    max_collision_faces: int = _safe_int(os.getenv("QUALITY_MAX_COLLISION_FACES"), 500000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "drop_probe_grid": self.drop_probe_grid,
            "drop_steps": self.drop_steps,
            "drop_probe_height_m": self.drop_probe_height_m,
            "drop_min_pass_rate": self.drop_min_pass_rate,
            "floor_penetration_tolerance_m": self.floor_penetration_tolerance_m,
            "jitter_settle_steps": self.jitter_settle_steps,
            "jitter_measure_steps": self.jitter_measure_steps,
            "jitter_extra_settle_steps": self.jitter_extra_settle_steps,
            "jitter_rest_speed_mps": self.jitter_rest_speed_mps,
            "jitter_rest_frames": self.jitter_rest_frames,
            "jitter_max_drift_m": self.jitter_max_drift_m,
            "jitter_max_vertical_span_m": self.jitter_max_vertical_span_m,
            "tunneling_steps": self.tunneling_steps,
            "tunneling_initial_speed_mps": self.tunneling_initial_speed_mps,
            "tunneling_max_penetration_m": self.tunneling_max_penetration_m,
            "perf_steps": self.perf_steps,
            "perf_max_step_ms": self.perf_max_step_ms,
            "max_collision_faces": self.max_collision_faces,
        }


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "metrics": dict(self.metrics),
        }


def _import_bullet_and_trimesh():
    try:
        import pybullet as p  # type: ignore
    except Exception as exc:
        raise StageError("quality_gates", f"pybullet import failed: {exc}") from exc
    try:
        import trimesh  # type: ignore
    except Exception as exc:
        raise StageError("quality_gates", f"trimesh import failed: {exc}") from exc
    return p, trimesh


def _bounds_from_stats_or_mesh(
    nurec_outputs: Mapping[str, Any],
    mesh_bounds: Sequence[Sequence[float]],
) -> tuple[List[float], List[float]]:
    if len(mesh_bounds) >= 2:
        mesh_min = [float(mesh_bounds[0][idx]) if idx < len(mesh_bounds[0]) else 0.0 for idx in range(3)]
        mesh_max = [float(mesh_bounds[1][idx]) if idx < len(mesh_bounds[1]) else 0.0 for idx in range(3)]
        mesh_valid = all(math.isfinite(value) for value in mesh_min + mesh_max) and any(
            mesh_max[idx] > mesh_min[idx] for idx in range(3)
        )
        if mesh_valid:
            return mesh_min, mesh_max

    mesh_stats = nurec_outputs.get("mesh_stats") if isinstance(nurec_outputs.get("mesh_stats"), Mapping) else {}
    stats_bounds = mesh_stats.get("bounds") if isinstance(mesh_stats.get("bounds"), Mapping) else {}
    mins = stats_bounds.get("min") if isinstance(stats_bounds.get("min"), list) else [-3.0, 0.0, -3.0]
    maxs = stats_bounds.get("max") if isinstance(stats_bounds.get("max"), list) else [3.0, 3.0, 3.0]
    min_vec = [float(mins[idx]) if idx < len(mins) else 0.0 for idx in range(3)]
    max_vec = [float(maxs[idx]) if idx < len(maxs) else 0.0 for idx in range(3)]
    return min_vec, max_vec


def _face_count(nurec_outputs: Mapping[str, Any], mesh: Any) -> int:
    if hasattr(mesh, "faces"):
        try:
            mesh_face_count = int(len(mesh.faces))
            if mesh_face_count > 0:
                return mesh_face_count
        except Exception:
            pass

    mesh_stats = nurec_outputs.get("mesh_stats") if isinstance(nurec_outputs.get("mesh_stats"), Mapping) else {}
    raw = mesh_stats.get("face_count")
    if raw is not None:
        return max(0, _safe_int(raw, 0))
    return 0


def _probe_positions(min_vec: Sequence[float], max_vec: Sequence[float], grid: int) -> List[Tuple[float, float]]:
    steps = max(2, grid)
    x_min = float(min_vec[0])
    x_max = float(max_vec[0])
    z_min = float(min_vec[2])
    z_max = float(max_vec[2])
    dx = x_max - x_min
    dz = z_max - z_min
    margin_x = max(0.05, dx * 0.12)
    margin_z = max(0.05, dz * 0.12)
    use_x_min = x_min + margin_x
    use_x_max = x_max - margin_x
    use_z_min = z_min + margin_z
    use_z_max = z_max - margin_z
    if use_x_min >= use_x_max:
        use_x_min, use_x_max = x_min, x_max
    if use_z_min >= use_z_max:
        use_z_min, use_z_max = z_min, z_max

    out: List[Tuple[float, float]] = []
    for ix in range(steps):
        for iz in range(steps):
            tx = 0.0 if steps == 1 else ix / float(steps - 1)
            tz = 0.0 if steps == 1 else iz / float(steps - 1)
            x = use_x_min + (use_x_max - use_x_min) * tx
            z = use_z_min + (use_z_max - use_z_min) * tz
            out.append((x, z))
    return out


def _drop_test(
    p: Any,
    *,
    min_vec: Sequence[float],
    max_vec: Sequence[float],
    cfg: AdvancedQualityGateConfig,
) -> GateResult:
    floor_y = float(min_vec[1])
    probe_positions = _probe_positions(min_vec, max_vec, cfg.drop_probe_grid)
    sphere_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.06)

    passed = 0
    for x, z in probe_positions:
        start_y = max(float(max_vec[1]) + cfg.drop_probe_height_m, floor_y + 0.6)
        body = p.createMultiBody(
            baseMass=0.3,
            baseCollisionShapeIndex=sphere_shape,
            basePosition=[x, start_y, z],
        )
        for _ in range(max(1, cfg.drop_steps)):
            p.stepSimulation()
        position, _ = p.getBasePositionAndOrientation(body)
        valid = (
            math.isfinite(position[1])
            and position[1] >= floor_y - cfg.floor_penetration_tolerance_m
        )
        if valid:
            passed += 1
        p.removeBody(body)

    total = len(probe_positions)
    pass_rate = float(passed) / float(total) if total else 0.0
    ok = pass_rate >= cfg.drop_min_pass_rate
    return GateResult(
        name="drop_test",
        passed=ok,
        detail=(
            f"drop probe pass rate {pass_rate:.3f} >= {cfg.drop_min_pass_rate:.3f}"
            if ok
            else f"drop probe pass rate {pass_rate:.3f} below {cfg.drop_min_pass_rate:.3f}"
        ),
        metrics={
            "total_probes": total,
            "passed_probes": passed,
            "pass_rate": round(pass_rate, 6),
            "floor_y": floor_y,
        },
    )


def _jitter_test(
    p: Any,
    *,
    min_vec: Sequence[float],
    max_vec: Sequence[float],
    cfg: AdvancedQualityGateConfig,
) -> GateResult:
    floor_y = float(min_vec[1])
    center_x = (float(min_vec[0]) + float(max_vec[0])) * 0.5
    center_z = (float(min_vec[2]) + float(max_vec[2])) * 0.5

    box_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.18, 0.12, 0.18])
    body = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=box_shape,
        basePosition=[center_x, max(float(max_vec[1]) + 0.8, floor_y + 0.5), center_z],
    )
    p.changeDynamics(
        body,
        -1,
        lateralFriction=1.1,
        spinningFriction=0.01,
        rollingFriction=0.01,
        restitution=0.0,
        linearDamping=0.08,
        angularDamping=0.08,
    )

    for _ in range(max(1, cfg.jitter_settle_steps)):
        p.stepSimulation()

    extra_settle_steps = 0
    rest_frames = 0
    for _ in range(max(0, cfg.jitter_extra_settle_steps)):
        p.stepSimulation()
        extra_settle_steps += 1
        linear_velocity, _ = p.getBaseVelocity(body)
        speed = math.sqrt(
            float(linear_velocity[0]) ** 2
            + float(linear_velocity[1]) ** 2
            + float(linear_velocity[2]) ** 2
        )
        if speed <= cfg.jitter_rest_speed_mps:
            rest_frames += 1
            if rest_frames >= max(1, cfg.jitter_rest_frames):
                break
        else:
            rest_frames = 0

    samples: List[Tuple[float, float, float]] = []
    for _ in range(max(1, cfg.jitter_measure_steps)):
        p.stepSimulation()
        pos, _ = p.getBasePositionAndOrientation(body)
        samples.append((float(pos[0]), float(pos[1]), float(pos[2])))
    p.removeBody(body)

    if not samples:
        return GateResult(
            name="jitter_test",
            passed=False,
            detail="no jitter samples collected",
            metrics={},
        )

    window = samples[max(0, len(samples) // 2):] or samples
    mean_x = sum(sample[0] for sample in window) / float(len(window))
    mean_z = sum(sample[2] for sample in window) / float(len(window))
    lateral_drift = max(
        math.sqrt((sample[0] - mean_x) ** 2 + (sample[2] - mean_z) ** 2)
        for sample in window
    )
    ys = [sample[1] for sample in window]
    vertical_span = max(ys) - min(ys) if ys else 0.0

    ok = lateral_drift <= cfg.jitter_max_drift_m and vertical_span <= cfg.jitter_max_vertical_span_m
    return GateResult(
        name="jitter_test",
        passed=ok,
        detail=(
            "resting-body drift/span within thresholds"
            if ok
            else "resting-body jitter exceeds thresholds"
        ),
        metrics={
            "lateral_drift_m": round(lateral_drift, 6),
            "vertical_span_m": round(vertical_span, 6),
            "threshold_drift_m": cfg.jitter_max_drift_m,
            "threshold_vertical_span_m": cfg.jitter_max_vertical_span_m,
            "extra_settle_steps": extra_settle_steps,
            "rest_frames_observed": rest_frames,
            "rest_speed_threshold_mps": cfg.jitter_rest_speed_mps,
        },
    )


def _tunneling_test(
    p: Any,
    *,
    min_vec: Sequence[float],
    max_vec: Sequence[float],
    cfg: AdvancedQualityGateConfig,
) -> GateResult:
    floor_y = float(min_vec[1])
    center_x = (float(min_vec[0]) + float(max_vec[0])) * 0.5
    center_z = (float(min_vec[2]) + float(max_vec[2])) * 0.5
    start_y = max(float(max_vec[1]) + 2.0, floor_y + 1.5)

    sphere_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
    body = p.createMultiBody(
        baseMass=0.4,
        baseCollisionShapeIndex=sphere_shape,
        basePosition=[center_x, start_y, center_z],
    )
    p.resetBaseVelocity(body, linearVelocity=[0.0, -abs(cfg.tunneling_initial_speed_mps), 0.0])

    min_seen_y = start_y
    for _ in range(max(1, cfg.tunneling_steps)):
        p.stepSimulation()
        pos, _ = p.getBasePositionAndOrientation(body)
        min_seen_y = min(min_seen_y, float(pos[1]))

    p.removeBody(body)
    penetration = max(0.0, floor_y - min_seen_y)
    ok = penetration <= cfg.tunneling_max_penetration_m
    return GateResult(
        name="tunneling_test",
        passed=ok,
        detail=(
            "high-speed probe stayed within penetration budget"
            if ok
            else "high-speed probe exceeded penetration budget"
        ),
        metrics={
            "min_observed_y": round(min_seen_y, 6),
            "floor_y": round(floor_y, 6),
            "penetration_m": round(penetration, 6),
            "max_penetration_m": cfg.tunneling_max_penetration_m,
        },
    )


def _perf_budget_test(p: Any, *, cfg: AdvancedQualityGateConfig) -> GateResult:
    sphere_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.04)
    probe_ids: List[int] = []
    for idx in range(10):
        probe_ids.append(
            p.createMultiBody(
                baseMass=0.2,
                baseCollisionShapeIndex=sphere_shape,
                basePosition=[(idx % 5) * 0.08, 0.8 + (idx // 5) * 0.05, (idx % 3) * 0.06],
            )
        )

    start = time.perf_counter()
    for _ in range(max(1, cfg.perf_steps)):
        p.stepSimulation()
    elapsed = time.perf_counter() - start

    for probe_id in probe_ids:
        p.removeBody(probe_id)

    avg_ms = (elapsed * 1000.0) / float(max(1, cfg.perf_steps))
    ok = avg_ms <= cfg.perf_max_step_ms
    return GateResult(
        name="perf_budget",
        passed=ok,
        detail=(
            f"avg step {avg_ms:.3f} ms <= budget {cfg.perf_max_step_ms:.3f} ms"
            if ok
            else f"avg step {avg_ms:.3f} ms exceeds budget {cfg.perf_max_step_ms:.3f} ms"
        ),
        metrics={
            "avg_step_ms": round(avg_ms, 6),
            "budget_step_ms": cfg.perf_max_step_ms,
            "steps": max(1, cfg.perf_steps),
        },
    )


def run_advanced_quality_gates(
    *,
    storage_root: Path,
    assets_prefix: str,
    nurec_outputs: Mapping[str, Any],
    config: Optional[AdvancedQualityGateConfig] = None,
) -> Dict[str, Any]:
    cfg = config or AdvancedQualityGateConfig()
    report: Dict[str, Any] = {
        "schema_version": "v1",
        "status": "skipped",
        "generated_at": utc_now_iso(),
        "config": cfg.to_dict(),
        "gates": [],
    }

    if not cfg.enabled:
        report["detail"] = "advanced quality gates disabled by configuration"
        return report

    mesh_glb = storage_root / assets_prefix / "obj_scene_shell" / "mesh.glb"
    if not has_nonempty_file(mesh_glb):
        raise StageError("quality_gates", f"missing scene-shell collider mesh: {mesh_glb}")

    p, trimesh = _import_bullet_and_trimesh()

    mesh = trimesh.load_mesh(str(mesh_glb), process=False)
    if mesh is None:
        raise StageError("quality_gates", f"failed to load scene-shell mesh at {mesh_glb}")
    if hasattr(trimesh, "Scene") and isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if not geometries:
            raise StageError("quality_gates", "scene-shell mesh has no geometry")
        mesh = trimesh.util.concatenate(geometries)

    min_vec, max_vec = _bounds_from_stats_or_mesh(nurec_outputs, mesh.bounds)
    face_count = _face_count(nurec_outputs, mesh)

    results: List[GateResult] = []
    complexity_ok = face_count <= max(1, cfg.max_collision_faces)
    results.append(
        GateResult(
            name="collision_complexity_budget",
            passed=complexity_ok,
            detail=(
                f"face count {face_count} <= budget {cfg.max_collision_faces}"
                if complexity_ok
                else f"face count {face_count} exceeds budget {cfg.max_collision_faces}"
            ),
            metrics={
                "face_count": face_count,
                "budget_face_count": cfg.max_collision_faces,
            },
        )
    )

    bullet_client = -1
    with TemporaryDirectory(prefix="swap_quality_") as tmp_dir:
        tmp_obj = Path(tmp_dir) / "scene_shell.obj"
        mesh.export(tmp_obj)
        try:
            bullet_client = p.connect(p.DIRECT)
            p.resetSimulation()
            p.setGravity(0.0, -9.81, 0.0)
            p.setPhysicsEngineParameter(numSolverIterations=100)
            flags = getattr(p, "GEOM_FORCE_CONCAVE_TRIMESH", 0)
            collision_shape = p.createCollisionShape(
                p.GEOM_MESH,
                fileName=str(tmp_obj),
                meshScale=[1.0, 1.0, 1.0],
                flags=flags,
            )
            if collision_shape < 0:
                raise StageError("quality_gates", "failed to create collision shape from scene-shell mesh")
            p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=collision_shape)

            results.append(_drop_test(p, min_vec=min_vec, max_vec=max_vec, cfg=cfg))
            results.append(_jitter_test(p, min_vec=min_vec, max_vec=max_vec, cfg=cfg))
            results.append(_tunneling_test(p, min_vec=min_vec, max_vec=max_vec, cfg=cfg))
            results.append(_perf_budget_test(p, cfg=cfg))

        finally:
            if bullet_client >= 0:
                p.disconnect()

    report["gates"] = [item.to_dict() for item in results]
    report["status"] = "passed" if all(item.passed for item in results) else "failed"
    report["mesh"] = {
        "scene_shell_glb": str(mesh_glb),
        "bounds": {"min": min_vec, "max": max_vec},
        "face_count": face_count,
    }
    return report
