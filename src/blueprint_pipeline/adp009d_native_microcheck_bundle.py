"""Immutable input-bundle compiler for the ADP-009D native Isaac micro-check."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest


PROBE_KIND = "adp009d-franka-native-microcheck"
SCHEMA_VERSION = "adp009d_native_microcheck_bundle.v1"
DEFAULT_IMAGE = (
    "nvcr.io/nvidia/isaac-sim:6.0.0-dev2@"
    "sha256:c3e7bef5b2bfdb9972807c34195206078372bf8c6cff79716be130a3fe3e9ce9"
)
ARENA_REVISION = "8b4a3a47fc53de23e8205089d71109a2e2348acd"
ARENA_TREE = "03f31f3dd56c56d00f24dbfb09711ec0ab345de8"
ISAAC_LAB_REVISION = "e57379c634b42db5a0fe9f754341be6e2a7c7c43"
ISAAC_LAB_TREE = "454115265327a80acabd07cbd36e10071fc0c065"
ASSET_BINDINGS = {
    "approved_can.usda": "sha256:61c2a03bef425803d82cc5ef24ced5b2ccb4160923c53bb10c6ad0e3f52532ec",
    "sage_collision.usd": "sha256:b265706c24f6a8ace3ee6743fd138583c4e21d83f61b99a06fd435e6ac2d6b41",
}
APPROVED_CAN_ADAPTER_FILENAME = "approved_can_physx_sdf_adapter.usda"
APPROVED_CAN_DEFAULT_PRIM = "canned_beverage"
APPROVED_CAN_COLLIDER_PATH = "colliders/body_collider"
TARGET_COLLIDER_PRIM = "/Root/ZHQYGJJVAJYEYPTUKY888888"
SUPPORT_COLLIDER_PRIM = "/Root/_LTFTHJVAZ3VMPTUJU888888"
SEALED_SAGE_PROFILE = {
    "prim_count": 166,
    "mesh_count": 165,
    "point_count": 509_268,
    "face_count": 993_678,
    "collision_mesh_count": 165,
    "rigid_body_count": 0,
    "convex_decomposition_count": 164,
    "triangle_mesh_count": 1,
}
TASK_COLLISION_ROI_MIN_M = (2.4681748, -4.3100837, -0.1)
TASK_COLLISION_ROI_MAX_M = (4.4681748, -1.9100837, 1.8)
TASK_COLLISION_MAX_EDGE_M = 0.5
TASK_COLLISION_DERIVATIVE_FILENAME = "sage_task_collision.usda"
TASK_COLLISION_MANIFEST_FILENAME = "sage_task_collision_manifest.json"
SEALED_TASK_COLLISION_PROFILE = {
    "candidate_source_prim_count": 16,
    "active_source_prim_count": 15,
    "source_face_count": 47_359,
    "clipped_source_face_count": 24_248,
    "derived_face_count": 26_828,
    "derived_point_count": 80_484,
}
def _candidate_ids(value: str | None) -> list[str]:
    """Candidates bound to a run, in order, from a comma-separated value."""

    return [part.strip() for part in str(value or "").split(",") if part.strip()]


ENTRYPOINT = """#!/usr/bin/env bash
set +e
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${BLUEPRINT_ADP_ARENA_OUTPUT_DIR:-$RUNTIME_DIR/../runtime_output}"
export BLUEPRINT_ADP009D_OUTPUT_DIR="$OUT_DIR"
export BLUEPRINT_ADP009D_POLICY_CANDIDATE="@@POLICY_CANDIDATE@@"
export BLUEPRINT_ADP009D_CAMERA_WARMUP_FRAMES="@@CAMERA_WARMUP_FRAMES@@"
export BLUEPRINT_ADP009D_STOP_AFTER_FRAMES="@@STOP_AFTER_FRAMES@@"
export BLUEPRINT_ADP009D_CAMERA_RESOLUTION="@@CAMERA_RESOLUTION@@"
export BLUEPRINT_ADP009D_EVIDENCE_PROFILE="@@EVIDENCE_PROFILE@@"
mkdir -p "$OUT_DIR"

# Environment facts the policy-server design could not verify from off-worker:
# what Isaac's own interpreter is, the torch it ships, and how much of the GPU
# is already spoken for before a policy is co-resident.  Captured before
# anything else runs, so a later failure cannot erase them.
/isaac-sim/python.sh "$RUNTIME_DIR/adp009d_worker_environment_facts.py" "$OUT_DIR" || true

# Policy provisioning, when a candidate is bound.  Non-fatal by design: the
# micro-check's own evidence must survive a provisioning failure, and the
# exit code is retained so the failure is visible rather than inferred.
# Tested with -f and invoked through bash, never relying on the execute bit:
# Python's zipfile.extractall does not preserve Unix permissions, so a 755
# script in the archive arrives non-executable and an -x test skips it in
# silence.  That is exactly what happened on the first provisioning run.
provisioned_any=0
provisioning_worst_rc=0
provisioning_candidates="$BLUEPRINT_ADP009D_POLICY_CANDIDATE"
frames_only_provisioning_skip=0
case "$BLUEPRINT_ADP009D_STOP_AFTER_FRAMES" in
  ""|0|false) ;;
  *)
    # A frames-only diagnostic exits immediately after retaining the two camera
    # observations.  No policy query can occur, so fetching a 6-12 GB checkpoint
    # and starting a policy server would add cost without changing its evidence.
    provisioning_candidates=""
    frames_only_provisioning_skip=1
    echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:policy_provisioning_skipped_frames_only:completed"
    ;;
esac
for candidate in $(printf '%s' "$provisioning_candidates" | tr ',' ' '); do
  script="$RUNTIME_DIR/adp009d_policy_provisioning.$candidate.sh"
  [ -f "$script" ] || continue
  provisioned_any=1
  # Each candidate's provisioning and server start is independent, and a
  # failure in one must not deny the other its episodes: a comparison with
  # one arm missing is still evidence, while a run that aborts on the first
  # failure produces none.
  # Markers on the container's own stdout, because the script's output is
  # redirected to a file and the no-progress watchdog reads only the container
  # log.  Emitting them inside the script put them where nothing was looking,
  # and two runs were killed at thirty minutes while provisioning correctly.
  echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:provision_${candidate}:started"
  # Give each candidate its own process group.  Killing only the shell leaves
  # its policy worker and GPU server orphaned; a measured GR00T failure did
  # exactly that, then started Isaac beside an unowned server after the outer
  # timeout.  ``setsid`` makes the background PID the group leader so timeout
  # teardown covers every descendant without touching the provider entrypoint.
  RUNTIME_DIR="$RUNTIME_DIR" OUT_DIR="$OUT_DIR" \
    setsid bash "$script" >"$OUT_DIR/adp009d_policy_provisioning.$candidate.log" 2>&1 &
  provisioning_pid=$!
  # A checkpoint fetch runs for minutes with nothing to say.  Tick while it
  # works so the watchdog can tell a long download from a stalled one, and
  # carry the step the script last reached so a stall is attributable.
  waited=0
  while kill -0 "$provisioning_pid" 2>/dev/null; do
    sleep 60
    waited=$((waited + 60))
    step=$(grep -aoE 'BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:provision_[a-z0-9_]+:(started|completed)' \
      "$OUT_DIR/adp009d_policy_provisioning.$candidate.log" 2>/dev/null | tail -1 | sed 's/.*adp009d://')
    echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:provision_${candidate}_working:${step:-starting}"
    # Bounded, because provisioning blocks the runtime.  A candidate whose
    # server never answers would otherwise consume the whole TTL and the run
    # would end with no episodes at all -- not even from the candidate that
    # provisioned fine, which defeats the point of tolerating one bad arm.
    # Observed: a server start ran forty-six minutes against its own
    # fifteen-minute readiness timeout.
    if [ $waited -ge ${BLUEPRINT_ADP009D_PROVISION_TIMEOUT_SECONDS:-1500} ]; then
      echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:provision_${candidate}:abandoned"
      kill -TERM -- "-$provisioning_pid" 2>/dev/null || true
      sleep 10
      kill -KILL -- "-$provisioning_pid" 2>/dev/null || true
      break
    fi
  done
  wait "$provisioning_pid" 2>/dev/null
  rc=$?
  echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:provision_${candidate}:completed:rc=$rc"
  [ $rc -ne 0 ] && provisioning_worst_rc=$rc
  printf '{"candidate_id": "%s", "provisioning_exit_code": %d}\n' \
    "$candidate" "$rc" >"$OUT_DIR/adp009d_policy_provisioning_status.$candidate.json"
done
if [ $frames_only_provisioning_skip -eq 1 ]; then
  printf '{"provisioning_exit_code": null, "provisioning_ran": false, "skip_reason": "frames_only_diagnostic"}\n' \
    >"$OUT_DIR/adp009d_policy_provisioning_status.json"
elif [ $provisioned_any -eq 1 ]; then
  printf '{"provisioning_exit_code": %d, "provisioning_ran": true}\n' \
    "$provisioning_worst_rc" >"$OUT_DIR/adp009d_policy_provisioning_status.json"
else
  # A bound candidate whose script is absent must say so rather than vanish.
  printf '{"provisioning_exit_code": null, "provisioning_ran": false}\n' \
    >"$OUT_DIR/adp009d_policy_provisioning_status.json"
fi

# A single-candidate diagnostic has no policy evidence to gain from cloning and
# installing Arena after its only policy failed provisioning.  Multi-candidate
# runs still continue so an independently healthy candidate retains episodes.
candidate_count=$(printf '%s\n' "$provisioning_candidates" | tr ',' '\n' | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')
if [ "$candidate_count" = "1" ] && [ "$provisioning_worst_rc" -ne 0 ]; then
  /isaac-sim/python.sh - "$OUT_DIR" "$provisioning_worst_rc" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
code = int(sys.argv[2])
(out / "adp009d_native_microcheck.json").write_text(json.dumps({
    "schema_version": "adp009d_native_microcheck.v1",
    "status": "blocked",
    "blockers": ["adp009d_single_candidate_policy_provisioning_failed"],
    "provisioning_exit_code": code,
    "candidate_policy_queried": False,
    "candidate_outcomes_accessed": False,
    "arena_setup_skipped": True,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:single_candidate_provisioning_failed:completed"
  exit "$provisioning_worst_rc"
fi

/isaac-sim/python.sh "$RUNTIME_DIR/adp_arena_provider_runner.py"
runner_rc=$?
if [ $runner_rc -ne 0 ] && [ ! -f "$OUT_DIR/adp009d_native_microcheck.json" ]; then
/isaac-sim/python.sh - "$OUT_DIR" "$runner_rc" <<'PY'
import json
import signal
import sys
from pathlib import Path
out = Path(sys.argv[1])
code = int(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
# A native abort and an ordinary Python error both reach here, and they need
# different repairs: a CUDA out-of-memory kill from a co-resident policy server
# arrives as SIGABRT or SIGKILL, which no Python except clause can catch.
# Recording the shell's exit status separates them.
signal_number = code - 128 if code > 128 else None
signal_name = None
if signal_number is not None:
    try:
        signal_name = signal.Signals(signal_number).name
    except ValueError:
        signal_name = None
blockers = ["adp009d_worker_failed_without_runtime_result"]
if signal_name:
    blockers.append(f"adp009d_worker_terminated_by_signal:{signal_name}")
(out / "adp009d_native_microcheck.json").write_text(json.dumps({
    "schema_version": "adp009d_native_microcheck.v1",
    "status": "blocked",
    "blockers": sorted(blockers),
    "worker_exit_code": code,
    "worker_terminating_signal": signal_name,
    "candidate_policy_queried": False,
    "candidate_outcomes_accessed": False
}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
fi
exit $runner_rc
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _inspect_sage_collision_source(
    source_path: Path,
    *,
    enforce_sealed_profile: bool,
) -> dict[str, Any]:
    """Inspect the materialized SAGE bytes before authoring a runtime override."""

    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover - guarded by bundle preflight
        raise ValueError("adp009d_sage_usd_runtime_missing") from exc

    stage = Usd.Stage.Open(str(source_path), load=Usd.Stage.LoadNone)
    if stage is None:
        raise ValueError("adp009d_sage_collision_unreadable")
    if str(stage.GetDefaultPrim().GetPath()) != "/Root":
        raise ValueError("adp009d_sage_default_prim_invalid")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError("adp009d_sage_units_invalid")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError("adp009d_sage_up_axis_invalid")

    mesh_paths: list[str] = []
    approximation_counts: dict[str, int] = {}
    prim_count = 0
    point_count = 0
    face_count = 0
    collision_mesh_count = 0
    rigid_body_paths: list[str] = []
    for prim in stage.Traverse():
        prim_count += 1
        schemas = {str(value) for value in prim.GetAppliedSchemas()}
        if "PhysicsRigidBodyAPI" in schemas:
            rigid_body_paths.append(str(prim.GetPath()))
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        mesh_paths.append(path)
        mesh = UsdGeom.Mesh(prim)
        point_count += len(mesh.GetPointsAttr().Get() or [])
        face_count += len(mesh.GetFaceVertexCountsAttr().Get() or [])
        if "PhysicsCollisionAPI" not in schemas or "PhysicsMeshCollisionAPI" not in schemas:
            raise ValueError(f"adp009d_sage_mesh_collision_schema_missing:{path}")
        collision_mesh_count += 1
        approximation = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
        approximation_counts[approximation] = approximation_counts.get(approximation, 0) + 1

    if rigid_body_paths:
        raise ValueError("adp009d_sage_static_collision_has_rigid_body:" + ",".join(rigid_body_paths))
    for required_path, blocker in (
        (TARGET_COLLIDER_PRIM, "adp009d_sage_target_collider_missing"),
        (SUPPORT_COLLIDER_PRIM, "adp009d_sage_support_collider_missing"),
    ):
        prim = stage.GetPrimAtPath(required_path)
        if not prim.IsValid() or not prim.IsActive() or required_path not in mesh_paths:
            raise ValueError(blocker)

    observed = {
        "prim_count": prim_count,
        "mesh_count": len(mesh_paths),
        "point_count": point_count,
        "face_count": face_count,
        "collision_mesh_count": collision_mesh_count,
        "rigid_body_count": len(rigid_body_paths),
        "convex_decomposition_count": approximation_counts.get("convexDecomposition", 0),
        "triangle_mesh_count": approximation_counts.get("none", 0),
    }
    if enforce_sealed_profile and observed != SEALED_SAGE_PROFILE:
        raise ValueError("adp009d_sealed_sage_collision_profile_mismatch")
    unsupported = sorted(set(approximation_counts) - {"convexDecomposition", "none"})
    if unsupported:
        raise ValueError("adp009d_sage_collision_approximation_unsupported:" + ",".join(unsupported))
    return {
        **observed,
        "source_approximation_counts": approximation_counts,
        "mesh_prim_paths": sorted(mesh_paths),
        "target_collider_prim": TARGET_COLLIDER_PRIM,
        "support_collider_prim": SUPPORT_COLLIDER_PRIM,
        "runtime_approximation": "none",
        "runtime_approximation_semantics": "static_triangle_mesh",
        "sealed_source_mutated": False,
    }


def _overlay_text(sage_profile: Mapping[str, Any]) -> str:
    mesh_paths = [str(value) for value in sage_profile.get("mesh_prim_paths", [])]
    if not mesh_paths or TARGET_COLLIDER_PRIM not in mesh_paths:
        raise ValueError("adp009d_sage_overlay_mesh_inventory_invalid")
    overrides = []
    for prim_path in mesh_paths:
        if not prim_path.startswith("/Root/") or prim_path.count("/") != 2:
            raise ValueError(f"adp009d_sage_overlay_prim_path_invalid:{prim_path}")
        prim_name = prim_path.rsplit("/", 1)[-1]
        if prim_path == TARGET_COLLIDER_PRIM:
            overrides.append(f'''    over "{prim_name}" (
        active = false
    )
    {{
    }}''')
        else:
            overrides.append(f'''    over "{prim_name}"
    {{
        uniform token physics:approximation = "none"
    }}''')
    override_text = "\n".join(overrides)
    return f'''#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Root" (
    prepend references = @sage_collision.usd@</Root>
)
{{
{override_text}
}}
'''


def _triangle_area(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> float:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return 0.5 * math.sqrt(sum(value * value for value in cross))


def _edge_length_squared(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum((a[index] - b[index]) ** 2 for index in range(3))


def _refine_triangle_to_edge_limit(
    triangle: tuple[tuple[float, float, float], ...],
    *,
    max_edge_m: float,
) -> list[tuple[tuple[float, float, float], ...]]:
    """Split the longest edge until every coplanar child edge is bounded."""

    if max_edge_m <= 0.0:
        raise ValueError("adp009d_task_collision_edge_limit_invalid")
    limit_squared = max_edge_m * max_edge_m
    output: list[tuple[tuple[float, float, float], ...]] = []
    stack = [triangle]
    while stack:
        a, b, c = stack.pop()
        lengths = (
            _edge_length_squared(a, b),
            _edge_length_squared(b, c),
            _edge_length_squared(c, a),
        )
        longest = max(range(3), key=lengths.__getitem__)
        if lengths[longest] <= limit_squared * (1.0 + 1.0e-12):
            output.append((a, b, c))
            continue
        pairs = ((a, b, c), (b, c, a), (c, a, b))
        first, second, opposite = pairs[longest]
        midpoint = tuple((first[index] + second[index]) * 0.5 for index in range(3))
        stack.append((first, midpoint, opposite))
        stack.append((midpoint, second, opposite))
    return output


def _ranges_intersect(
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    *,
    roi_min: tuple[float, float, float],
    roi_max: tuple[float, float, float],
) -> bool:
    return all(upper[index] >= roi_min[index] and lower[index] <= roi_max[index] for index in range(3))


def _clip_polygon_to_axis_plane(
    polygon: list[tuple[float, float, float]],
    *,
    axis: int,
    boundary: float,
    keep_greater: bool,
) -> list[tuple[float, float, float]]:
    """Clip a coplanar polygon against one axis-aligned half-space."""

    if not polygon:
        return []

    def inside(point: tuple[float, float, float]) -> bool:
        return point[axis] >= boundary if keep_greater else point[axis] <= boundary

    output: list[tuple[float, float, float]] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            if abs(denominator) <= 1.0e-15:
                raise ValueError("adp009d_task_collision_clip_intersection_invalid")
            fraction = (boundary - previous[axis]) / denominator
            intersection = tuple(
                previous[index] + fraction * (current[index] - previous[index])
                for index in range(3)
            )
            output.append(intersection)
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return output


def _clip_triangle_to_aabb(
    triangle: tuple[tuple[float, float, float], ...],
    *,
    roi_min: tuple[float, float, float],
    roi_max: tuple[float, float, float],
) -> list[tuple[tuple[float, float, float], ...]]:
    """Return the exact coplanar portion of a triangle inside an AABB."""

    polygon = list(triangle)
    for axis in range(3):
        polygon = _clip_polygon_to_axis_plane(
            polygon,
            axis=axis,
            boundary=roi_min[axis],
            keep_greater=True,
        )
        polygon = _clip_polygon_to_axis_plane(
            polygon,
            axis=axis,
            boundary=roi_max[axis],
            keep_greater=False,
        )
    if len(polygon) < 3:
        return []
    triangles = [
        (polygon[0], polygon[index], polygon[index + 1])
        for index in range(1, len(polygon) - 1)
    ]
    return [value for value in triangles if _triangle_area(*value) > 1.0e-12]


def _build_sage_task_collision_derivative(
    source_path: Path,
    destination: Path,
    *,
    source_sha256: str,
    roi_min_m: tuple[float, float, float] = TASK_COLLISION_ROI_MIN_M,
    roi_max_m: tuple[float, float, float] = TASK_COLLISION_ROI_MAX_M,
    max_edge_m: float = TASK_COLLISION_MAX_EDGE_M,
) -> dict[str, Any]:
    """Author an exact-surface collision derivative for the frozen task envelope."""

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    source = Usd.Stage.Open(str(source_path), load=Usd.Stage.LoadNone)
    if source is None:
        raise ValueError("adp009d_sage_collision_unreadable")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    selected: list[Any] = []
    candidate_paths: list[str] = []
    for prim in source.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        world_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lower = tuple(float(value) for value in world_range.GetMin())
        upper = tuple(float(value) for value in world_range.GetMax())
        path = str(prim.GetPath())
        if path in {TARGET_COLLIDER_PRIM, SUPPORT_COLLIDER_PRIM} or _ranges_intersect(
            lower, upper, roi_min=roi_min_m, roi_max=roi_max_m
        ):
            candidate_paths.append(path)
            if path != TARGET_COLLIDER_PRIM:
                selected.append(prim)
    if TARGET_COLLIDER_PRIM not in candidate_paths:
        raise ValueError("adp009d_task_collision_target_not_in_roi")
    if SUPPORT_COLLIDER_PRIM not in {str(prim.GetPath()) for prim in selected}:
        raise ValueError("adp009d_task_collision_support_not_selected")

    derived = Usd.Stage.CreateNew(str(destination))
    if derived is None:
        raise ValueError("adp009d_task_collision_derivative_create_failed")
    UsdGeom.SetStageMetersPerUnit(derived, 1.0)
    UsdGeom.SetStageUpAxis(derived, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(derived, "/Root").GetPrim()
    derived.SetDefaultPrim(root)
    target = derived.DefinePrim(TARGET_COLLIDER_PRIM, "Mesh")
    target.SetActive(False)

    rows: list[dict[str, Any]] = []
    total_source_faces = 0
    total_clipped_faces = 0
    total_derived_faces = 0
    total_derived_points = 0
    selected_source_area_m2 = 0.0
    clipped_source_area_m2 = 0.0
    derived_area_m2 = 0.0
    observed_max_edge_m = 0.0
    for source_prim in sorted(selected, key=lambda value: str(value.GetPath())):
        source_mesh = UsdGeom.Mesh(source_prim)
        counts = list(source_mesh.GetFaceVertexCountsAttr().Get() or [])
        indices = list(source_mesh.GetFaceVertexIndicesAttr().Get() or [])
        if not counts or any(int(value) != 3 for value in counts):
            raise ValueError(f"adp009d_task_collision_non_triangle_source:{source_prim.GetPath()}")
        transform = xform_cache.GetLocalToWorldTransform(source_prim)
        world_points = [
            tuple(float(value) for value in transform.Transform(Gf.Vec3d(point)))
            for point in source_mesh.GetPointsAttr().Get() or []
        ]
        total_source_faces += len(counts)
        clipped: list[tuple[tuple[float, float, float], ...]] = []
        for offset in range(0, len(indices), 3):
            triangle = tuple(world_points[int(index)] for index in indices[offset : offset + 3])
            selected_source_area_m2 += _triangle_area(*triangle)
            clipped.extend(
                _clip_triangle_to_aabb(
                    triangle,
                    roi_min=roi_min_m,
                    roi_max=roi_max_m,
                )
            )
        if not clipped:
            continue
        leaves: list[tuple[tuple[float, float, float], ...]] = []
        for triangle in clipped:
            clipped_source_area_m2 += _triangle_area(*triangle)
            leaves.extend(_refine_triangle_to_edge_limit(triangle, max_edge_m=max_edge_m))

        points: list[Any] = []
        derived_indices: list[int] = []
        mesh_area_m2 = 0.0
        mesh_max_edge_m = 0.0
        for triangle in leaves:
            quantized_triangle = tuple(
                tuple(float(value) for value in Gf.Vec3f(*point))
                for point in triangle
            )
            base = len(points)
            points.extend(Gf.Vec3f(*point) for point in quantized_triangle)
            derived_indices.extend((base, base + 1, base + 2))
            mesh_area_m2 += _triangle_area(*quantized_triangle)
            mesh_max_edge_m = max(
                mesh_max_edge_m,
                *(
                    math.sqrt(
                        _edge_length_squared(
                            quantized_triangle[index],
                            quantized_triangle[(index + 1) % 3],
                        )
                    )
                    for index in range(3)
                ),
            )
        path = str(source_prim.GetPath())
        output_mesh = UsdGeom.Mesh.Define(derived, path)
        output_mesh.CreatePointsAttr(points)
        output_mesh.CreateFaceVertexCountsAttr([3] * len(leaves))
        output_mesh.CreateFaceVertexIndicesAttr(derived_indices)
        output_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        output_mesh.CreateDoubleSidedAttr(source_mesh.GetDoubleSidedAttr().Get() or False)
        output_mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        collision = UsdPhysics.CollisionAPI.Apply(output_mesh.GetPrim())
        collision.CreateCollisionEnabledAttr(True)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(output_mesh.GetPrim())
        mesh_collision.CreateApproximationAttr(UsdPhysics.Tokens.none)
        output_mesh.GetPrim().SetCustomDataByKey("blueprint:sourcePrim", path)

        total_clipped_faces += len(clipped)
        total_derived_faces += len(leaves)
        total_derived_points += len(points)
        derived_area_m2 += mesh_area_m2
        observed_max_edge_m = max(observed_max_edge_m, mesh_max_edge_m)
        rows.append(
            {
                "source_prim": path,
                "source_face_count": len(counts),
                "clipped_face_count": len(clipped),
                "derived_face_count": len(leaves),
                "derived_point_count": len(points),
                "derived_surface_area_m2": round(mesh_area_m2, 9),
                "maximum_edge_m": round(mesh_max_edge_m, 9),
            }
        )

    relative_area_error = abs(derived_area_m2 - clipped_source_area_m2) / max(
        clipped_source_area_m2, 1.0e-12
    )
    if relative_area_error > 1.0e-6:
        raise ValueError("adp009d_task_collision_surface_area_changed")
    if observed_max_edge_m > max_edge_m * (1.0 + 1.0e-6):
        raise ValueError("adp009d_task_collision_edge_limit_not_met")
    root.SetCustomDataByKey("blueprint:sealedSourceSha256", source_sha256)
    root.SetCustomDataByKey("blueprint:claimCeiling", "preregistered_franka_task_envelope_only")
    root.SetCustomDataByKey("blueprint:maxEdgeM", max_edge_m)
    derived.GetRootLayer().Save()
    result = {
        "schema_version": "adp009d_sage_task_collision_derivative.v1",
        "status": "ready",
        "sealed_source_sha256": source_sha256,
        "sealed_source_mutated": False,
        "derivative_filename": destination.name,
        "derivative_sha256": _sha256(destination),
        "roi_min_m": list(roi_min_m),
        "roi_max_m": list(roi_max_m),
        "maximum_edge_limit_m": max_edge_m,
        "observed_maximum_edge_m": round(observed_max_edge_m, 9),
        "candidate_source_prim_count": len(candidate_paths),
        "active_source_prim_count": len(rows),
        "source_target_prim_excluded": TARGET_COLLIDER_PRIM,
        "support_prim_included": SUPPORT_COLLIDER_PRIM,
        "source_face_count": total_source_faces,
        "clipped_source_face_count": total_clipped_faces,
        "derived_face_count": total_derived_faces,
        "derived_point_count": total_derived_points,
        "selected_source_surface_area_m2": round(selected_source_area_m2, 9),
        "clipped_source_surface_area_m2": round(clipped_source_area_m2, 9),
        "derived_surface_area_m2": round(derived_area_m2, 9),
        "relative_surface_area_error": relative_area_error,
        "surface_operation": "aabb_clip_then_coplanar_longest_edge_midpoint_retriangulation",
        "source_prim_rows": rows,
        "claim_ceiling": "preregistered_franka_task_envelope_only",
    }
    if source_sha256 == ASSET_BINDINGS["sage_collision.usd"]:
        observed_profile = {
            key: result[key] for key in SEALED_TASK_COLLISION_PROFILE
        }
        if observed_profile != SEALED_TASK_COLLISION_PROFILE:
            raise ValueError("adp009d_sealed_sage_task_collision_profile_mismatch")
    return result


def _approved_can_physx_sdf_adapter_text() -> str:
    """Compose the sealed can with the PhysX schema required to consume its SDF token."""

    collider_parts = APPROVED_CAN_COLLIDER_PATH.split("/")
    if len(collider_parts) != 2:
        raise ValueError("adp009d_approved_can_collider_path_invalid")
    scope_name, collider_name = collider_parts
    return f'''#usda 1.0
(
    defaultPrim = "{APPROVED_CAN_DEFAULT_PRIM}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{APPROVED_CAN_DEFAULT_PRIM}" (
    prepend references = @approved_can.usda@</{APPROVED_CAN_DEFAULT_PRIM}>
)
{{
    over "{scope_name}"
    {{
        over "{collider_name}" (
            prepend apiSchemas = ["PhysxSDFMeshCollisionAPI"]
        )
        {{
            uniform token physics:approximation = "sdf"
            float physxSDFMeshCollision:sdfMargin = 0.01
            float physxSDFMeshCollision:sdfNarrowBandThickness = 0.01
            int physxSDFMeshCollision:sdfResolution = 256
            int physxSDFMeshCollision:sdfSubgridResolution = 6
        }}
    }}
}}
'''


def _copy_bound_asset(source: Path, destination: Path, expected_digest: str) -> dict[str, Any]:
    if not source.is_file():
        raise ValueError(f"adp009d_bound_asset_missing:{destination.name}")
    observed = _sha256(source)
    if observed != expected_digest:
        raise ValueError(f"adp009d_bound_asset_digest_mismatch:{destination.name}")
    shutil.copy2(source, destination)
    return {
        "filename": destination.name,
        "sha256": observed,
        "size_bytes": destination.stat().st_size,
    }


def build_native_microcheck_bundle(
    *,
    job_dir: str | Path,
    approved_can_path: str | Path,
    sage_collision_path: str | Path,
    harness_manifest_path: str | Path,
    implementation_commit: str,
    policy_candidate_id: str | None = None,
    aura_particlefield_path: str | Path | None = None,
    generated_at: str | None = None,
    expected_asset_bindings: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compile a deterministic bundle from materialized, digest-verified bytes."""

    if len(implementation_commit) != 40 or any(ch not in "0123456789abcdef" for ch in implementation_commit):
        raise ValueError("adp009d_implementation_commit_invalid")
    job = Path(job_dir).expanduser().resolve()
    if job.exists():
        shutil.rmtree(job)
    runtime = job / "provider_runtime"
    assets = runtime / "assets"
    ensure_dir(assets)
    bindings = dict(expected_asset_bindings or ASSET_BINDINGS)
    sources = {
        "approved_can.usda": Path(approved_can_path).expanduser().resolve(),
        "sage_collision.usd": Path(sage_collision_path).expanduser().resolve(),
    }
    asset_rows = [
        _copy_bound_asset(sources[name], assets / name, bindings[name]) for name in sorted(bindings)
    ]
    task_collision_path = assets / TASK_COLLISION_DERIVATIVE_FILENAME
    task_collision = _build_sage_task_collision_derivative(
        sources["sage_collision.usd"],
        task_collision_path,
        source_sha256=bindings["sage_collision.usd"],
    )
    # Build the derivative before the independent whole-source audit. Opening the
    # million-face source twice in the opposite order leaves enough USD allocator
    # state resident to exhaust memory while the USDA layer is serialized on the
    # canonical macOS preflight host.
    sage_profile = _inspect_sage_collision_source(
        sources["sage_collision.usd"],
        enforce_sealed_profile=(bindings["sage_collision.usd"] == ASSET_BINDINGS["sage_collision.usd"]),
    )
    overlay_path = assets / "sage_collision_overlay.usda"
    overlay_path.write_text(_overlay_text(sage_profile), encoding="utf-8")
    asset_rows.append(
        {
            "filename": overlay_path.name,
            "sha256": _sha256(overlay_path),
            "size_bytes": overlay_path.stat().st_size,
            "composition_only": True,
            "sealed_source_mutated": False,
            "deactivated_source_prim": TARGET_COLLIDER_PRIM,
            "preserved_support_prim": SUPPORT_COLLIDER_PRIM,
            "static_triangle_mesh_override_count": sage_profile["mesh_count"] - 1,
            "source_collision_profile": {
                key: value for key, value in sage_profile.items() if key != "mesh_prim_paths"
            },
        }
    )
    task_collision_manifest_path = assets / TASK_COLLISION_MANIFEST_FILENAME
    write_json(task_collision_manifest_path, task_collision)
    asset_rows.extend(
        [
            {
                "filename": task_collision_path.name,
                "sha256": task_collision["derivative_sha256"],
                "size_bytes": task_collision_path.stat().st_size,
                "derived_from": "sage_collision.usd",
                "sealed_source_mutated": False,
                "claim_ceiling": task_collision["claim_ceiling"],
            },
            {
                "filename": task_collision_manifest_path.name,
                "sha256": _sha256(task_collision_manifest_path),
                "size_bytes": task_collision_manifest_path.stat().st_size,
                "binds_derivative": task_collision_path.name,
            },
        ]
    )
    can_adapter_path = assets / APPROVED_CAN_ADAPTER_FILENAME
    can_adapter_path.write_text(_approved_can_physx_sdf_adapter_text(), encoding="utf-8")
    asset_rows.append(
        {
            "filename": can_adapter_path.name,
            "sha256": _sha256(can_adapter_path),
            "size_bytes": can_adapter_path.stat().st_size,
            "composition_only": True,
            "sealed_source_mutated": False,
            "source_asset": "approved_can.usda",
            "collider_prim": (
                f"/{APPROVED_CAN_DEFAULT_PRIM}/{APPROVED_CAN_COLLIDER_PATH}"
            ),
            "required_applied_schema": "PhysxSDFMeshCollisionAPI",
            "required_approximation": "sdf",
        }
    )

    source_dir = Path(__file__).resolve().parent
    shutil.copy2(source_dir / "adp009d_native_microcheck_worker.py", runtime / "adp_arena_provider_runner.py")
    shutil.copy2(source_dir / "adp009d_isaac_runtime.py", runtime / "adp009d_isaac_runtime.py")
    shutil.copy2(
        source_dir / "adp009d_approach_capture.py",
        runtime / "adp009d_approach_capture.py",
    )
    if aura_particlefield_path is not None:
        # The sealed appearance, rendered by Isaac in the same pass as the
        # robot.  Shipped as a scene asset rather than rendered separately and
        # composited: a 15 Hz closed loop cannot call out to a second renderer
        # between steps, and the goal prompt rules a policy result invalid
        # unless both cameras see the Aura background together with the moving
        # arm and can in one time-synchronised frame.
        aura_source = Path(aura_particlefield_path).expanduser().resolve()
        if not aura_source.is_file():
            raise ValueError("adp009d_aura_particlefield_missing")
        # Staged under its own extension.  A fixed .usd name would rename a
        # NuRec .usdz into something Isaac opens as a flat layer, and the
        # appearance format is the whole question this lane is deciding: a
        # ParticleField that Omniverse has never rendered correctly, against a
        # NuRec volume it demonstrably has.
        if aura_source.suffix not in {".usd", ".usda", ".usdz"}:
            raise ValueError(f"adp009d_aura_appearance_extension_unsupported:{aura_source.suffix}")
        shutil.copy2(
            aura_source, assets / f"aura_ghost_removed_appearance{aura_source.suffix}"
        )
    harness_source = Path(harness_manifest_path).expanduser().resolve()
    shutil.copy2(harness_source, runtime / "adp009d_franka_eval_harness_manifest.v1.json")
    shutil.copy2(
        source_dir / "adp009d_worker_environment_facts.py",
        runtime / "adp009d_worker_environment_facts.py",
    )
    shutil.copy2(
        source_dir / "adp009d_checkpoint_fetch_worker.py",
        runtime / "adp009d_checkpoint_fetch_worker.py",
    )
    shutil.copy2(
        source_dir / "adp009d_provisioning_preflight.py",
        runtime / "adp009d_provisioning_preflight.py",
    )
    for module_name in (
        "adp009d_droid_observation.py",
        "adp009d_droid_action_execution.py",
        "droid_policy_bridge.py",
        "adp009d_policy_episode.py",
        # Wired into the runtime but never shipped, so a live run reached the
        # episode and died on ModuleNotFoundError after provisioning had
        # already succeeded.  An import the runtime makes must be a file the
        # bundle carries.
        "adp009d_episode_batch.py",
        "adp009d_isaac_episode_adapter.py",
        "adp009d_task_scoring.py",
        "episode_visual_evidence.py",
        # The episode imports the step trace unconditionally, and the runtime
        # imports the dataset capture recorder when the evidence profile asks
        # for control-rate streams.  Both must ship or a live run dies on
        # ModuleNotFoundError after provisioning has already succeeded.
        "adp009d_episode_step_trace.py",
        "adp009d_dataset_capture.py",
        "adp009d_policy_server_worker.py",
        "adp009d_groot_worker_identity.py",
        "adp009d_gated_backbone.py",
        "groot_n17_droid_policy_runtime.py",
        # Imported by the episode for its digest helper.  It has no intra-package
        # imports of its own, so it ships flat without dragging anything with it.
        # Omitting it made both arms of the dual-layout import fail: the flat one
        # with ModuleNotFoundError, the fallback with "attempted relative import
        # with no known parent package" -- a different exception than the one the
        # except clause names, so the fallback could never have caught it.
        "decision_evidence_contracts.py",
    ):
        shutil.copy2(source_dir / module_name, runtime / module_name)
    if policy_candidate_id:
        # The destination is frozen before any outcome exists; ship the receipt
        # itself rather than recomputing it on the worker, so the episode is
        # scored against exactly the value that was sealed.
        from .adp009d_task_destination import DESTINATION_SCHEMA_VERSION

        destination_receipt = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "docs/arm_decision_proof_v1/adp009d_task_destination.v1.json"
            ).read_text(encoding="utf-8")
        )
        if destination_receipt.get("schema_version") != DESTINATION_SCHEMA_VERSION:
            raise ValueError("adp009d_task_destination_schema_unexpected")
        (runtime / "adp009d_task_destination.v1.json").write_text(
            json.dumps(destination_receipt, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if policy_candidate_id:
        from .adp009d_policy_provisioning import build_provisioning_script

        # One script per candidate.  Ranking two policies needs both in the
        # same scene on the same host, and a single script could only ever
        # provision one -- which would have made the comparison two runs on
        # two machines, paying the twenty-minute boot twice and comparing
        # across hardware whose render speed already differs by 3x.
        for _candidate in _candidate_ids(policy_candidate_id):
            _write_executable(
                runtime / f"adp009d_policy_provisioning.{_candidate}.sh",
                build_provisioning_script(_candidate),
            )
    # Baked in at build time.  A passthrough of an unset variable reads as
    # empty, and the runtime then skips the episode in silence -- which is
    # exactly what a live run did: no episode, no error, nothing to read.
    camera_resolution = str(
        os.environ.get("BLUEPRINT_ADP009D_CAMERA_RESOLUTION", "")
    ).strip()
    if policy_candidate_id and not camera_resolution:
        # Policy runs must render at the resolution against which visibility
        # and observation conversion are specified.  A host export is not
        # inherited by the remote worker; v74 silently rendered 1280x720 and
        # weakened a 200-pixel gate by 16x.  Bind the default into the bundle.
        camera_resolution = "policy"
    _write_executable(
        runtime / "run_adp_arena_provider_runtime.sh",
        ENTRYPOINT.replace("@@POLICY_CANDIDATE@@", policy_candidate_id or "")
        .replace(
            "@@CAMERA_WARMUP_FRAMES@@",
            str(os.environ.get("BLUEPRINT_ADP009D_CAMERA_WARMUP_FRAMES", "")),
        )
        .replace(
            "@@STOP_AFTER_FRAMES@@",
            str(os.environ.get("BLUEPRINT_ADP009D_STOP_AFTER_FRAMES", "")),
        )
        .replace("@@CAMERA_RESOLUTION@@", camera_resolution)
        .replace(
            "@@EVIDENCE_PROFILE@@",
            str(os.environ.get("BLUEPRINT_ADP009D_EVIDENCE_PROFILE", "")),
        ),
    )
    generated = generated_at or utc_now_iso()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated,
        "status": "ready",
        "program_id": "arm-decision-proof-v1",
        "probe_kind": PROBE_KIND,
        "implementation_commit": implementation_commit,
        "container_image": DEFAULT_IMAGE,
        "official_sources": {
            "isaac_lab_arena": {
                "repository": "https://github.com/isaac-sim/IsaacLab-Arena",
                "revision": ARENA_REVISION,
                "tree": ARENA_TREE,
                "version": "release/0.2.1",
            },
            "isaac_lab": {
                "repository": "https://github.com/isaac-sim/IsaacLab",
                "revision": ISAAC_LAB_REVISION,
                "tree": ISAAC_LAB_TREE,
                "version": "3.0.0 nested by Arena",
            },
        },
        "asset_bindings": asset_rows,
        "harness_manifest_sha256": _sha256(harness_source),
        "runtime_entrypoint": "provider_runtime/run_adp_arena_provider_runtime.sh",
        "policy_candidate_id": policy_candidate_id,
        "camera_resolution_binding": camera_resolution or None,
        "expected_output_filename": "adp009d_native_microcheck.json",
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "private_data_uploaded": False,
        "retry_cap": 0,
        "provider_zero_required_after_return": True,
        "blockers": [],
    }
    manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
    write_json(runtime / "adp_arena_provider_manifest.json", manifest)
    bundle_path = job / "adp009d_native_microcheck_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", allowZip64=True) as archive:
        for path in sorted(runtime.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.relative_to(job).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_STORED)
    receipt = {
        **manifest,
        "bundle_path": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
    }
    write_json(job / "adp009d_native_microcheck_bundle_receipt.json", receipt)
    return receipt


def build_native_microcheck_bundle_isolated(
    *,
    job_dir: str | Path,
    approved_can_path: str | Path,
    sage_collision_path: str | Path,
    harness_manifest_path: str | Path,
    implementation_commit: str,
    policy_candidate_id: str | None = None,
    aura_particlefield_path: str | Path | None = None,
    generated_at: str | None = None,
    expected_asset_bindings: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the large USD bundle in a fresh, bounded process."""

    job = Path(job_dir).expanduser().resolve()
    command = [
        sys.executable,
        "-m",
        "blueprint_pipeline.adp009d_native_microcheck_bundle",
        "--isolated-child",
        "--job-dir",
        str(job),
        "--approved-can-path",
        str(Path(approved_can_path).expanduser().resolve()),
        "--sage-collision-path",
        str(Path(sage_collision_path).expanduser().resolve()),
        "--harness-manifest-path",
        str(Path(harness_manifest_path).expanduser().resolve()),
        "--implementation-commit",
        implementation_commit,
    ]
    if policy_candidate_id is not None:
        command.extend(("--policy-candidate-id", policy_candidate_id))
    if aura_particlefield_path is not None:
        command.extend(
            ("--aura-particlefield-path", str(Path(aura_particlefield_path).resolve()))
        )
    if generated_at is not None:
        command.extend(("--generated-at", generated_at))
    if expected_asset_bindings is not None:
        command.extend(
            (
                "--expected-asset-bindings-json",
                json.dumps(dict(expected_asset_bindings), sort_keys=True, separators=(",", ":")),
            )
        )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("adp009d_bundle_subprocess_timeout") from exc
    if completed.returncode != 0:
        raise ValueError(f"adp009d_bundle_subprocess_failed:{completed.returncode}")

    receipt_path = job / "adp009d_native_microcheck_bundle_receipt.json"
    if not receipt_path.is_file():
        raise ValueError("adp009d_bundle_subprocess_receipt_missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    bundle_path = job / "adp009d_native_microcheck_bundle.zip"
    if (
        receipt.get("status") != "ready"
        or receipt.get("implementation_commit") != implementation_commit
        or Path(receipt.get("bundle_path", "")).resolve() != bundle_path
        or not bundle_path.is_file()
        or receipt.get("bundle_sha256") != _sha256(bundle_path)
    ):
        raise ValueError("adp009d_bundle_subprocess_receipt_invalid")
    return receipt


def _isolated_child_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--isolated-child", action="store_true", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--approved-can-path", required=True)
    parser.add_argument("--sage-collision-path", required=True)
    parser.add_argument("--harness-manifest-path", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--policy-candidate-id", default=None)
    parser.add_argument("--aura-particlefield-path", default=None)
    parser.add_argument("--generated-at")
    parser.add_argument("--expected-asset-bindings-json")
    args = parser.parse_args(argv)
    expected_bindings = (
        json.loads(args.expected_asset_bindings_json)
        if args.expected_asset_bindings_json is not None
        else None
    )
    build_native_microcheck_bundle(
        job_dir=args.job_dir,
        approved_can_path=args.approved_can_path,
        sage_collision_path=args.sage_collision_path,
        harness_manifest_path=args.harness_manifest_path,
        implementation_commit=args.implementation_commit,
        policy_candidate_id=args.policy_candidate_id,
        aura_particlefield_path=args.aura_particlefield_path,
        generated_at=args.generated_at,
        expected_asset_bindings=expected_bindings,
    )
    return 0


__all__ = [
    "APPROVED_CAN_ADAPTER_FILENAME",
    "DEFAULT_IMAGE",
    "PROBE_KIND",
    "build_native_microcheck_bundle",
    "build_native_microcheck_bundle_isolated",
]


if __name__ == "__main__":
    raise SystemExit(_isolated_child_main())
