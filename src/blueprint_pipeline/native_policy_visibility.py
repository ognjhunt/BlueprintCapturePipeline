"""Read visibility from the same native render products as policy RGB.

Bounding-box annotations stay outside Isaac's dense Camera buffers, which do
not support structured bbox data. No extra camera or render is created here.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
import hashlib
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .policy_observation_runtime_contract import NativeObservationProtocol


def policy_camera_visibility_contract(
    snapshot: Mapping[str, Any],
    *,
    preserve_official_droid_calibration: bool,
) -> dict[str, Any]:
    """Qualify task visibility according to each camera's policy role.

    A fixed external camera must frame the task at reset. The official DROID
    wrist camera is attached to the gripper, so its reset observation may put
    the task near an image edge before the policy approaches it. For that
    camera, visible native-semantic task pixels are the reset requirement;
    centering remains recorded as a typed notice and is measured during the
    rollout.
    """

    rows = {
        str(row.get("role")): row
        for row in snapshot.get("cameras") or []
        if isinstance(row, Mapping) and str(row.get("role") or "")
    }
    expected_roles = {"external", "wrist", "overview"}
    raw_visibility = {
        role: bool((row.get("observability") or {}).get("passed")) for role, row in rows.items()
    }
    qualifications: dict[str, Any] = {}
    blockers: list[str] = []
    notices: list[str] = []
    if set(rows) != expected_roles:
        blockers.append("policy_canary_camera_role_inventory_invalid")
    for role in sorted(expected_roles):
        row = rows.get(role)
        observability = (
            row.get("observability")
            if isinstance(row, Mapping) and isinstance(row.get("observability"), Mapping)
            else {}
        )
        raw_passed = bool(observability.get("passed"))
        pixel_count = int(observability.get("pixel_count") or 0)
        thresholds = observability.get("thresholds")
        minimum_pixels = (
            int(thresholds.get("effective_minimum_pixels") or 0)
            if isinstance(thresholds, Mapping)
            else 0
        )
        render_passed = observability.get("render_passed") is True
        centroid_within_margin = observability.get("centroid_within_margin") is True
        if preserve_official_droid_calibration and role == "wrist":
            passed = (
                render_passed
                and minimum_pixels > 0
                and pixel_count >= minimum_pixels
                and bool(observability.get("target_semantic_ids"))
            )
            status = (
                "centered"
                if passed and centroid_within_margin
                else "initial_edge_visible"
                if passed
                else "insufficient_task_pixels"
            )
            if passed and not centroid_within_margin:
                notices.append("droid_wrist_task_initially_near_frame_edge")
            if not passed:
                blockers.append("droid_wrist_task_pixels_below_threshold")
        else:
            passed = raw_passed
            status = "centered" if passed else "not_qualified"
            if not passed:
                blockers.append(f"policy_canary_{role}_task_visibility_failed")
        qualifications[role] = {
            "status": status,
            "passed": passed,
            "raw_observability_passed": raw_passed,
            "pixel_count": pixel_count,
            "minimum_pixels": minimum_pixels,
            "render_passed": render_passed,
            "centroid_within_margin": centroid_within_margin,
        }
    return {
        "passed": not blockers and set(rows) == expected_roles,
        "camera_visibility": {
            role: bool(qualifications.get(role, {}).get("passed"))
            for role in sorted(expected_roles)
        },
        "raw_camera_visibility": raw_visibility,
        "role_qualifications": qualifications,
        "blockers": sorted(set(blockers)),
        "notices": sorted(set(notices)),
    }


def array(value: Any) -> Any:
    import numpy as np

    value = getattr(value, "torch", value)
    if callable(getattr(value, "detach", None)):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def rgb_digest(value: Any) -> str:
    import numpy as np

    return "sha256:" + hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _frame_index(camera: Any) -> int:
    import math
    import numpy as np

    values = array(camera.frame).reshape(-1)
    if (
        values.size != 1
        or values.dtype == np.bool_
        or not math.isfinite(float(values[0]))
        or float(values[0]) < 0
        or float(values[0]) != int(values[0])
    ):
        raise ValueError("acquisition_native_single_camera_frame_required")
    return int(values[0])


def _reference_time(value: Mapping[str, Any]) -> Fraction:
    numerator, denominator = (
        value.get("referenceTimeNumerator"),
        value.get("referenceTimeDenominator"),
    )
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or numerator < 0
        or denominator <= 0
    ):
        raise ValueError("acquisition_native_reference_time_invalid")
    return Fraction(numerator, denominator)


def _box_rows(value: Any, *, target_label: str) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping) or not isinstance(value.get("info"), Mapping):
        raise ValueError("acquisition_native_bbox_output_missing")
    labels = value["info"].get("idToLabels")
    if not isinstance(labels, Mapping):
        raise ValueError("acquisition_native_bbox_labels_missing")
    data = value.get("data")
    if data is None:
        raise ValueError("acquisition_native_bbox_data_missing")
    rows = []
    for entry in data:
        try:
            identifier = int(entry["semanticId"])
            label = labels.get(identifier, labels.get(str(identifier)))
            if not isinstance(label, Mapping) or label.get("class") != target_label:
                continue
            ratio = float(entry["occlusionRatio"])
            if not 0 <= ratio <= 1:
                raise ValueError("invalid_ratio")
            rows.append(
                {
                    "semantic_id": identifier,
                    "occlusion_ratio": ratio,
                    "bbox_xyxy": [int(entry[k]) for k in ("x_min", "y_min", "x_max", "y_max")],
                }
            )
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            raise ValueError("acquisition_native_bbox_row_invalid") from exc
    return rows


class NativePolicyVisibilityReader:
    """Single-env native AOV capture with frame/time and RGB identity checks."""

    def __init__(self, *, built: Any, binding: Mapping[str, Any], annotator_factory: Any = None):
        self.built = built
        self.binding = NativeObservationProtocol.model_validate(binding)
        self._annotators: dict[str, Any] = {}
        self._attached: list[tuple[Any, str]] = []
        if annotator_factory is None:
            import omni.replicator.core as rep

            annotator_factory = rep.AnnotatorRegistry.get_annotator
        try:
            for role in self.binding.camera_setups:
                camera = built.env.unwrapped.scene[built.camera_scene_names[role]]
                render_data = getattr(camera, "_render_data", None)
                render_cameras = tuple(
                    getattr(getattr(render_data, "spec", None), "camera_prim_paths", ())
                )
                viewed_cameras = tuple(getattr(getattr(camera, "_view", None), "prim_paths", ()))
                if len(render_cameras) != 1 or render_cameras != viewed_cameras:
                    raise ValueError("acquisition_native_render_camera_identity_mismatch")
                render_product = getattr(render_data, "render_product", None)
                path = getattr(render_product, "path", None)
                if path is None:
                    # The earlier Isaac Lab 3.0 render-data contract used a list.
                    paths = getattr(render_data, "render_product_paths", None)
                    if not isinstance(paths, (list, tuple)) or len(paths) != 1:
                        raise ValueError("acquisition_native_render_product_unavailable")
                    path = paths[0]
                if not isinstance(path, str) or not path.startswith("/"):
                    raise ValueError("acquisition_native_render_product_invalid")
                channels = {
                    "bbox": annotator_factory(
                        "bounding_box_2d_loose_fast",
                        init_params={"semanticTypes": ["class"]},
                        device="cpu",
                    ),
                    "time": annotator_factory("ReferenceTime", device="cpu"),
                    "rgb": annotator_factory("rgb", device="cpu"),
                }
                self._annotators[role] = (path, channels)
                for annotator in channels.values():
                    annotator.attach([path])
                    self._attached.append((annotator, path))
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        errors = []
        attached, self._attached = self._attached, []
        for annotator, path in reversed(attached):
            try:
                annotator.detach([path])
            except Exception as exc:
                errors.append(str(exc))
        self._annotators.clear()
        if errors:
            raise RuntimeError("acquisition_native_annotation_detach_failed:" + ";".join(errors))

    def read_geometry(self, *, require_reset_pose: bool = True) -> dict[str, Any]:
        import numpy as np

        from .adp009d_isaac_episode_adapter import rotation_row_major_from_quaternion_xyzw
        from .native_task_arena_runtime import camera_runtime_parameters

        rows = {}
        for role, setup in self.binding.camera_setups.items():
            camera = self.built.env.unwrapped.scene[self.built.camera_scene_names[role]]
            planned = next(c for c in self.built.plan["cameras"] if c["role"] == role)
            parameters = camera_runtime_parameters(planned)
            offset = camera.cfg.offset
            if not np.allclose(
                array(offset.pos),
                parameters["offset_position_m"],
                rtol=0,
                atol=setup.pose_tolerance_m,
            ):
                raise ValueError("acquisition_native_camera_mount_position_mismatch")
            rotation = np.asarray(rotation_row_major_from_quaternion_xyzw(offset.rot)).reshape(3, 3)
            expected_rotation = np.asarray(setup.frame_from_camera_matrix).reshape(4, 4)[:3, :3]
            if offset.convention != "ros" or not np.allclose(
                rotation, expected_rotation, rtol=0, atol=setup.rotation_tolerance
            ):
                raise ValueError("acquisition_native_camera_mount_rotation_mismatch")
            intrinsic = array(camera.data.intrinsic_matrices)[0]
            intrinsics = planned["intrinsics"]
            expected_intrinsic = np.array(
                [
                    [intrinsics["fx"], 0, intrinsics["width"] * 0.5],
                    [0, intrinsics["fy"], intrinsics["height"] * 0.5],
                    [0, 0, 1],
                ]
            )
            if not np.allclose(intrinsic, expected_intrinsic, rtol=0, atol=1e-4):
                raise ValueError("acquisition_native_camera_intrinsics_mismatch")
            world = np.eye(4)
            world[:3, 3] = array(camera.data.pos_w)[0]
            world[:3, :3] = np.asarray(
                rotation_row_major_from_quaternion_xyzw(array(camera.data.quat_w_opengl)[0])
            ).reshape(3, 3) @ np.diag([1, -1, -1])
            expected_world = np.asarray(setup.reset_world_from_camera_opencv_matrix).reshape(4, 4)
            if require_reset_pose and (
                not np.allclose(
                    world[:3, 3], expected_world[:3, 3], rtol=0, atol=setup.pose_tolerance_m
                )
                or not np.allclose(
                    world[:3, :3], expected_world[:3, :3], rtol=0, atol=setup.rotation_tolerance
                )
            ):
                raise ValueError("acquisition_native_camera_reset_world_pose_mismatch")
            rows[role] = {
                "observed_world_from_camera_opencv": world.reshape(-1).tolist(),
                "intrinsic_matrix": intrinsic.tolist(),
                "mount_position_m": list(offset.pos),
                "calibration_digest": self.binding.acquisition.camera_calibration_digests[role],
                "intrinsic_readback_convention": "pinned_isaac_aperture_center_w_over_2_h_over_2",
            }
        subject = self.built.env.unwrapped.scene[self.built.scene_asset_names["task_object"]]
        actual_position = array(subject.data.root_pose_w)[0, :3]
        if require_reset_pose and not np.allclose(
            actual_position,
            self.binding.information.setup_object_coordinates.position_m,
            rtol=0,
            atol=1e-5,
        ):
            raise ValueError("acquisition_native_task_object_reset_position_mismatch")
        receipt = {
            "cameras": rows,
            "task_object_position_m": actual_position.tolist(),
            "binding_digest": self.binding.binding_digest,
            "geometry_passed": True,
            "reset_pose_checked": require_reset_pose,
        }
        receipt["geometry_digest"] = canonical_digest(receipt)
        return receipt

    def capture(self, raw_rgb: Mapping[str, Any]) -> dict[str, Any]:
        import math
        import numpy as np

        from .native_task_camera_observability import (
            measure_native_task_semantic_label_pixels,
            _semantic_identifier_candidates,
        )

        rows = {}
        times = set()
        geometry = self.read_geometry(require_reset_pose=False)
        simulation_time = getattr(
            getattr(self.built.env.unwrapped, "sim", None), "current_time", None
        )
        physics_step_reader = getattr(self.built.env.unwrapped.sim, "get_physics_step_count", None)
        physics_step = (
            physics_step_reader()
            if callable(physics_step_reader)
            else getattr(self.built.env.unwrapped.sim, "current_time_step_index", None)
        )
        if type(physics_step) is not int or physics_step < 0:
            raise ValueError("acquisition_native_physics_step_unavailable")
        if (
            isinstance(simulation_time, bool)
            or not isinstance(simulation_time, (int, float))
            or not math.isfinite(simulation_time)
        ):
            raise ValueError("acquisition_native_simulation_time_unavailable")
        for role, (path, channels) in self._annotators.items():
            camera = self.built.env.unwrapped.scene[self.built.camera_scene_names[role]]
            before = _reference_time(channels["time"].get_data())
            frame_index = _frame_index(camera)
            semantic = np.squeeze(array(camera.data.output["semantic_segmentation"])).copy()
            labels = (camera.data.info.get("semantic_segmentation") or {}).get("idToLabels")
            if not isinstance(labels, Mapping):
                raise ValueError("acquisition_native_semantic_labels_missing")
            expected = np.asarray(raw_rgb[role])
            rgb = array(channels["rgb"].get_data())
            if rgb.size == expected.shape[0] * expected.shape[1] * 4:
                rgb = rgb.reshape(*expected.shape[:2], 4)[..., :3]
            if (
                rgb.shape != expected.shape
                or rgb.dtype != expected.dtype
                or not np.array_equal(rgb, expected)
            ):
                raise ValueError("acquisition_native_policy_rgb_frame_mismatch")
            if semantic.shape != expected.shape[:2]:
                raise ValueError("acquisition_native_semantic_frame_shape_mismatch")
            boxes = _box_rows(channels["bbox"].get_data(), target_label="task_object")
            after = _reference_time(channels["time"].get_data())
            if before != after or frame_index != _frame_index(camera):
                raise ValueError("acquisition_native_frame_changed_during_capture")
            if abs(float(before) - simulation_time) > 1e-6:
                raise ValueError("acquisition_native_render_time_stale")
            measurement = measure_native_task_semantic_label_pixels(
                semantic_ids=semantic, id_to_labels=labels, target_label="task_object"
            )
            target_ids = [
                identifier
                for key, label in labels.items()
                if isinstance(label, Mapping) and label.get("class") == "task_object"
                for identifier in _semantic_identifier_candidates(key)
            ]
            mask = np.isin(semantic.astype(np.int64), target_ids)
            if measurement["pixel_count"] > 0 and not boxes:
                raise ValueError("acquisition_native_semantic_bbox_disagreement")
            times.add(before)
            rows[role] = {
                "mask": mask,
                "semantic_ids": semantic,
                "id_to_labels": dict(labels),
                "target_boxes": boxes,
                "native_pixel_count": measurement["pixel_count"],
                "raw_rgb_digest": rgb_digest(expected),
                "renderer_frame": frame_index,
                "render_product_path": path,
                "camera_prim_path": self.built.env.unwrapped.scene[
                    self.built.camera_scene_names[role]
                ]._render_data.spec.camera_prim_paths[0],
                "reference_time_numerator": before.numerator,
                "reference_time_denominator": before.denominator,
            }
        if (
            len(times) != 1
            or set(rows) != {"external", "wrist"}
            or self.built.env.unwrapped.sim.current_time != simulation_time
        ):
            raise ValueError("acquisition_native_camera_times_not_synchronized")
        return {
            "native_time_s": float(next(iter(times))),
            "physics_step": physics_step,
            "geometry_readback": geometry,
            "cameras": rows,
        }
