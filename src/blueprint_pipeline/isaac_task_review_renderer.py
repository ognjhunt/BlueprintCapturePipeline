"""Attempt-bound overview and robot-POV renderer for persistent Isaac tasks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence


def _norm(value: Sequence[float]) -> tuple[float, float, float]:
    magnitude = math.sqrt(sum(float(item) ** 2 for item in value)) or 1.0
    return tuple(float(item) / magnitude for item in value)  # type: ignore[return-value]


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def look_at_quaternion(
    eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0, 0, 1)
) -> tuple[float, float, float, float]:
    forward = _norm(tuple(target[i] - eye[i] for i in range(3)))
    z_axis = tuple(-item for item in forward)
    x_axis = _norm(_cross(up, z_axis))
    y_axis = _cross(z_axis, x_axis)
    m00, m01, m02 = x_axis[0], y_axis[0], z_axis[0]
    m10, m11, m12 = x_axis[1], y_axis[1], z_axis[1]
    m20, m21, m22 = x_axis[2], y_axis[2], z_axis[2]
    trace = m00 + m11 + m22
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        return (0.25 * scale, (m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale)
    if m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2
        return ((m21 - m12) / scale, 0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale)
    if m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2
        return ((m02 - m20) / scale, (m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale)
    scale = math.sqrt(1.0 + m22 - m00 - m11) * 2
    return ((m10 - m01) / scale, (m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale)


class IsaacTaskReviewRenderer:
    def __init__(self, *, stage: Any, app: Any, robot_prim_path: str, output_dir: Path):
        import omni.replicator.core as rep  # type: ignore
        from pxr import UsdGeom  # type: ignore

        self.stage = stage
        self.app = app
        self.robot_prim_path = robot_prim_path
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.overview_path = "/World/BlueprintReview/OverviewCamera"
        self.robot_pov_path = "/World/BlueprintReview/RobotPOVCamera"
        UsdGeom.Camera.Define(stage, self.overview_path)
        UsdGeom.Camera.Define(stage, self.robot_pov_path)
        self.rep = rep
        self.annotators = {}
        for role, camera in (
            ("overview", self.overview_path),
            ("robot_pov", self.robot_pov_path),
        ):
            product = rep.create.render_product(camera, (640, 480))
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([product])
            self.annotators[role] = annotator

    def _center(self, prim_path: str) -> tuple[float, float, float]:
        from pxr import Usd, UsdGeom  # type: ignore

        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"review_renderer_prim_missing:{prim_path}")
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        candidate = prim
        while candidate and candidate.IsValid():
            aligned = cache.ComputeWorldBound(candidate).ComputeAlignedRange()
            minimum, maximum = aligned.GetMin(), aligned.GetMax()
            values = [float(minimum[i] + maximum[i]) / 2.0 for i in range(3)]
            if all(math.isfinite(value) for value in values):
                return tuple(values)  # type: ignore[return-value]
            candidate = candidate.GetParent()
        raise RuntimeError(f"review_renderer_bound_missing:{prim_path}")

    def _head_center(self) -> tuple[float, float, float]:
        from pxr import Usd, UsdGeom  # type: ignore

        root = self.stage.GetPrimAtPath(self.robot_prim_path)
        for prim in Usd.PrimRange(root):
            if "head" in prim.GetName().lower():
                matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
                point = matrix.ExtractTranslation()
                return (float(point[0]), float(point[1]), float(point[2]))
        robot = self._center(self.robot_prim_path)
        return (robot[0], robot[1], robot[2] + 0.8)

    def _place(self, camera_path: str, eye: Sequence[float], target: Sequence[float]) -> None:
        from pxr import Gf, UsdGeom  # type: ignore

        quaternion = look_at_quaternion(eye, target)
        xform = UsdGeom.Xformable(self.stage.GetPrimAtPath(camera_path))
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*[float(item) for item in eye]))
        xform.AddOrientOp().Set(Gf.Quatf(*[float(item) for item in quaternion]))

    def render(self, *, step_index: int, target_prim_path: str) -> list[dict[str, Any]]:
        import hashlib
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        target = self._center(target_prim_path)
        robot = self._center(self.robot_prim_path)
        focus = tuple((target[i] + robot[i]) / 2.0 for i in range(3))
        span = max(2.5, math.dist(target, robot) * 2.0)
        overview_eye = (focus[0] - span, focus[1] - span, focus[2] + span * 0.65)
        head = self._head_center()
        robot_pov_eye = (head[0], head[1], head[2] + 0.05)
        self._place(self.overview_path, overview_eye, focus)
        self._place(self.robot_pov_path, robot_pov_eye, target)
        for _ in range(3):
            self.app.update()
        artifacts = []
        for role, annotator in self.annotators.items():
            data = np.asarray(annotator.get_data())
            if data.ndim != 3 or data.shape[2] not in (3, 4):
                raise RuntimeError(f"review_renderer_{role}_rgb_invalid")
            rgb = data[:, :, :3].astype("uint8")
            path = self.frames_dir / f"{role}_{step_index:04d}.png"
            Image.fromarray(rgb).save(path)
            artifacts.append(
                {
                    "camera_role": role,
                    "frame_index": step_index,
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "width": int(rgb.shape[1]),
                    "height": int(rgb.shape[0]),
                }
            )
        return artifacts
