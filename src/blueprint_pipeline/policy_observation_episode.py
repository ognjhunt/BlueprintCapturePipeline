"""Runtime observation/acquisition recording around the existing episode adapter."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from .adp009d_droid_observation import (
    CANDIDATE_VIEW_SHAPES,
    DROID_EXTERIOR_VIEW_1,
    DROID_WRIST_VIEW,
    resize_with_pad,
)
from .decision_evidence_contracts import canonical_digest
from .native_policy_visibility import array, rgb_digest
from .policy_object_acquisition_contract import AcquisitionSample, assess_object_acquisition
from .policy_observation_runtime_contract import NativeObservationProtocol

_VIEW_FOR_ROLE = {"external": DROID_EXTERIOR_VIEW_1, "wrist": DROID_WRIST_VIEW}


class EpisodeAcquisition:
    """Optional episode hooks, keeping orchestration in the existing runner."""

    def __init__(
        self, environment: Any, binding: Any, prestart_required: bool, complete_media_required: bool
    ):
        self.environment = environment
        self.enabled = binding is not None
        self.has_sample = False
        if self.enabled:
            protocol = NativeObservationProtocol.model_validate(binding)
            if (
                not prestart_required
                or not complete_media_required
                or getattr(getattr(environment, "binding", None), "binding_digest", None)
                != protocol.binding_digest
                or not all(
                    callable(getattr(environment, name, None))
                    for name in (
                        "begin_policy_acquisition",
                        "record_policy_acquisition",
                        "object_acquisition_receipt",
                    )
                )
            ):
                raise ValueError("policy_observation_protocol_runtime_binding_required")

    def begin(self, episode_id: str | None, candidate_id: str) -> None:
        if self.enabled:
            self.environment.begin_policy_acquisition(
                episode_id=episode_id, candidate_id=candidate_id
            )

    def record(
        self,
        observation: Mapping[str, Any],
        query_index: int,
        frames: list[Any],
        output_root: Path | None,
        progress: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        try:
            progress["object_acquisition"] = self.environment.record_policy_acquisition(
                observation=observation,
                query_index=query_index,
                exact_frame=frames[-1],
                output_root=output_root,
            )
            self.has_sample = True
        except Exception:
            progress["object_acquisition"] = self.environment.object_acquisition_receipt()
            raise

    def attach(self, receipt: dict[str, Any]) -> None:
        if self.enabled and self.has_sample:
            receipt["object_acquisition"] = self.environment.object_acquisition_receipt(
                observation_series_complete=True
            )


def protocol_mount_selection(
    reader: Any, binding: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if reader is None:
        raise RuntimeError("policy_canary_observation_protocol_reader_required")
    setup = native_initial_protocol_setup(reader=reader, binding=binding)
    selected = {
        "source": "preregistered_observation_protocol",
        "binding_digest": binding["binding_digest"],
        "admitted": True,
        "blockers": [],
    }
    selection = {
        "schema_version": "policy_canary_wrist_camera_mount_selection.v1",
        "status": "selected",
        "selected_candidate": selected,
        "contact_sheet": None,
        "blockers": [],
        "selection_digest": canonical_digest(selected),
    }
    return setup, selection


def wrap_protocol_environment(
    environment: Any,
    receipt: Mapping[str, Any],
    plan: Mapping[str, Any],
    session_state: Mapping[str, Any],
) -> tuple[Any, Mapping[str, Any]]:
    binding = plan.get("observation_protocol")
    if binding is None:
        return environment, receipt
    reader = session_state.get("observation_visibility_reader")
    if reader is None:
        raise RuntimeError("policy_canary_observation_protocol_reader_missing_after_gate")
    wrapped = ObservationProtocolEnvironment(
        environment=environment, binding=binding, reader=reader
    )
    return wrapped, {
        **dict(receipt),
        "observation_protocol_binding_digest": binding["binding_digest"],
    }


def create_visibility_reader(
    factory: Any, built: Any, plan: Mapping[str, Any], session_state: dict[str, Any]
) -> dict[str, Any]:
    binding = plan.get("observation_protocol")
    if binding is None:
        return {}
    if factory is None:
        raise RuntimeError("policy_canary_observation_protocol_reader_unavailable")
    reader = factory(built=built, binding=binding)
    session_state["observation_visibility_reader"] = reader
    return {"observation_visibility_reader": reader}


def _policy_mask(mask: Any, *, candidate_id: str) -> Any:
    import numpy as np
    from PIL import Image

    height, width = CANDIDATE_VIEW_SHAPES[candidate_id]
    source = np.asarray(mask, dtype=np.uint8)
    scale = min(width / source.shape[1], height / source.shape[0])
    output_width, output_height = (
        max(1, round(source.shape[1] * scale)),
        max(1, round(source.shape[0] * scale)),
    )
    resized = np.asarray(
        Image.fromarray(source).resize((output_width, output_height), Image.Resampling.NEAREST)
    )
    result = np.zeros((height, width), dtype=np.uint8)
    top, left = (height - output_height) // 2, (width - output_width) // 2
    result[top : top + output_height, left : left + output_width] = resized
    return result


def sample_from_native(
    binding: NativeObservationProtocol,
    capture: Mapping[str, Any],
    raw_rgb: Mapping[str, Any],
    *,
    observation: Mapping[str, Any],
    candidate_id: str,
    episode_id: str,
    step_index: int,
    elapsed_s: float,
    episode_origin_s: float,
) -> tuple[AcquisitionSample, dict[str, Any]]:
    import numpy as np

    cameras, masks = {}, {}
    for role, view in _VIEW_FOR_ROLE.items():
        native = capture["cameras"][role]
        expected = resize_with_pad(
            raw_rgb[role],
            height=CANDIDATE_VIEW_SHAPES[candidate_id][0],
            width=CANDIDATE_VIEW_SHAPES[candidate_id][1],
        )
        actual = np.asarray(observation[view])
        if (
            native["raw_rgb_digest"] != rgb_digest(raw_rgb[role])
            or actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or not np.array_equal(actual, expected)
        ):
            raise ValueError("acquisition_exact_policy_frame_mismatch")
        mask = _policy_mask(native["mask"], candidate_id=candidate_id)
        masks[role] = mask
        count = int(mask.sum())
        boxes = native["target_boxes"]
        if count == 0:
            visibility = "fully_occluded" if boxes else "initially_out_of_view"
        else:
            visibility = (
                "partially_occluded"
                if any(row["occlusion_ratio"] > 0.0 for row in boxes)
                else "visible"
            )
        cameras[role] = {
            "frame_digest": rgb_digest(actual),
            "calibration_digest": binding.acquisition.camera_calibration_digests[role],
            "renderer_frame": native["renderer_frame"],
            "rendered": True,
            "fresh": True,
            "target_pixels": count,
            "target_visibility": visibility,
        }
    return AcquisitionSample.model_validate(
        {
            "episode_id": episode_id,
            "target_object_id": binding.information.setup_object_coordinates.object_id,
            "episode_elapsed_s": elapsed_s,
            "episode_started_at_sim_time_s": episode_origin_s,
            "observation_sim_time_s": capture["native_time_s"],
            "physics_step": step_index,
            "reset_digest": binding.reset_digest,
            "visibility_source": "deterministic_simulator_segmentation_and_bbox_occlusion",
            "cameras": cameras,
        }
    ), masks


def native_initial_protocol_setup(*, reader: Any, binding: Mapping[str, Any]) -> dict[str, Any]:
    """Use one actual reset capture to verify both candidates' initial domain."""

    from .adp009d_isaac_episode_adapter import rgb_from_camera_output

    protocol = NativeObservationProtocol.model_validate(binding)
    geometry = reader.read_geometry()
    raw = {}
    for role in _VIEW_FOR_ROLE:
        camera = reader.built.env.unwrapped.scene[reader.built.camera_scene_names[role]]
        raw[role] = rgb_from_camera_output(array(camera.data.output["rgb"])[0])
    capture = reader.capture(raw)
    assessments = {}
    for candidate_id, (height, width) in CANDIDATE_VIEW_SHAPES.items():
        if candidate_id not in {"pi05_droid", "groot_n17_droid"}:
            continue
        observation = {
            view: resize_with_pad(raw[role], height=height, width=width)
            for role, view in _VIEW_FOR_ROLE.items()
        }
        episode_id = f"prepolicy.{protocol.cell_id}.{candidate_id}"
        sample, _ = sample_from_native(
            protocol,
            capture,
            raw,
            observation=observation,
            candidate_id=candidate_id,
            episode_id=episode_id,
            step_index=capture["physics_step"],
            elapsed_s=0.0,
            episode_origin_s=capture["native_time_s"],
        )
        assessments[candidate_id] = assess_object_acquisition(
            protocol.acquisition,
            [sample],
            reset_digest=protocol.reset_digest,
            episode_id=episode_id,
            target_object_id=sample.target_object_id,
            episode_started_at_sim_time_s=capture["native_time_s"],
        )
    receipt = {
        "schema_version": "native_observation_protocol_setup.v1",
        "binding_digest": protocol.binding_digest,
        "initial_visibility": protocol.acquisition.initial_visibility,
        "geometry": geometry,
        "geometry_passed": True,
        "sensor_freshness_passed": True,
        "initial_condition_passed": True,
        "candidate_initial_assessments": assessments,
        "native_camera_evidence": {
            role: {k: v for k, v in row.items() if k not in {"mask", "semantic_ids"}}
            for role, row in capture["cameras"].items()
        },
        "native_validation_scope": "observed_simulator_setup_not_physical_proof",
    }
    receipt["receipt_digest"] = canonical_digest(receipt)
    return receipt


class ObservationProtocolEnvironment:
    """Retain acquisition evidence while delegating all control to the adapter."""

    def __init__(self, *, environment: Any, binding: Mapping[str, Any], reader: Any):
        self._environment = environment
        self.binding = NativeObservationProtocol.model_validate(binding)
        self._reader = reader
        self._samples: list[AcquisitionSample] = []
        self._records: list[dict[str, Any]] = []
        self._pending = None
        self._origin = None
        self._episode_id = None
        self._geometry = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    def reset(self) -> Any:
        result = self._environment.reset()
        self._geometry = self._reader.read_geometry()
        self._pending, self._origin, self._episode_id = None, None, None
        self._samples, self._records = [], []
        return result

    def begin_policy_acquisition(self, *, episode_id: str, candidate_id: str) -> None:
        if self._geometry is None:
            raise ValueError("acquisition_native_reset_readback_required")
        self._episode_id, self._candidate_id = episode_id, candidate_id

    def read_policy_inputs(self) -> dict[str, Any]:
        from .native_task_camera_observability import validate_native_task_policy_input_frames

        inputs = dict(self._environment.read_policy_inputs())
        raw = {role: inputs[view] for role, view in _VIEW_FOR_ROLE.items()}
        validate_native_task_policy_input_frames(raw)
        metadata = self._environment.read_control_observation_metadata()
        capture = self._reader.capture(raw)
        freshness = inputs.get("sensor_freshness")
        if not isinstance(freshness, Mapping) or set(freshness) != set(_VIEW_FOR_ROLE):
            raise ValueError("acquisition_native_sensor_freshness_missing")
        steps = set()
        for role, native in capture["cameras"].items():
            row = freshness[role]
            step = row.get("control_step_index")
            if (
                row.get("status") != "observed"
                or type(step) is not int
                or row.get("frame_index") != native["renderer_frame"]
            ):
                raise ValueError("acquisition_native_sensor_freshness_mismatch")
            steps.add(step)
        if len(steps) != 1:
            raise ValueError("acquisition_native_policy_steps_not_synchronized")
        elapsed = float(metadata["simulation_time_s"])
        self._pending = (capture, raw, capture["physics_step"], elapsed)
        return inputs

    def record_policy_acquisition(
        self,
        *,
        observation: Mapping[str, Any],
        query_index: int,
        exact_frame: Mapping[str, Any],
        output_root: Path,
    ) -> dict[str, Any]:
        import numpy as np

        if self._pending is None or self._episode_id is None or query_index != len(self._samples):
            raise ValueError("acquisition_policy_observation_sequence_invalid")
        capture, raw, step, elapsed = self._pending
        self._pending = None
        if self._origin is None:
            self._origin = float(capture["native_time_s"]) - elapsed
        sample, masks = sample_from_native(
            self.binding,
            capture,
            raw,
            observation=observation,
            candidate_id=self._candidate_id,
            episode_id=self._episode_id,
            step_index=step,
            elapsed_s=elapsed,
            episode_origin_s=self._origin,
        )
        folder = (
            output_root
            / "object_acquisition"
            / hashlib.sha256(self._episode_id.encode()).hexdigest()[:24]
        )
        folder.mkdir(parents=True, exist_ok=True)
        files = []
        for role in _VIEW_FOR_ROLE:
            path = folder / f"{query_index:05d}.{role}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    semantic_ids=capture["cameras"][role]["semantic_ids"],
                    policy_target_mask=masks[role],
                )
            files.append(
                {
                    "camera_role": role,
                    "relative_path": str(path.relative_to(output_root)),
                    "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
        value = {
            "sample": sample.model_dump(mode="json"),
            "binding_digest": self.binding.binding_digest,
            "exact_policy_frame_manifest_digest": exact_frame["frame_manifest_digest"],
            "native_camera_evidence": {
                role: {k: v for k, v in row.items() if k not in {"mask", "semantic_ids"}}
                for role, row in capture["cameras"].items()
            },
            "semantic_transform": "nearest_neighbor_with_production_letterbox_geometry",
            "files": files,
        }
        value["sample_evidence_digest"] = canonical_digest(value)
        path = folder / f"{query_index:05d}.json"
        with path.open("x") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        self._records.append(
            {
                "relative_path": str(path.relative_to(output_root)),
                "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
                "sample_evidence_digest": value["sample_evidence_digest"],
            }
        )
        self._samples.append(sample)
        # Retain the source evidence before refusing an invalid initial condition.
        result = self.object_acquisition_receipt()
        if result.get("status") == "invalid_observation":
            raise ValueError(result["blocker"])
        return result

    def object_acquisition_receipt(
        self, *, observation_series_complete: bool = False
    ) -> dict[str, Any]:
        result = {
            "schema_version": "policy_object_acquisition_receipt.v1",
            "binding_digest": self.binding.binding_digest,
            "episode_id": self._episode_id,
            "candidate_id": getattr(self, "_candidate_id", None),
            "geometry_readback": self._geometry,
            "sample_artifacts": list(self._records),
            "samples": [s.model_dump(mode="json") for s in self._samples],
            "candidate_policy_grades_acquisition": False,
            "observation_series_complete": observation_series_complete,
        }
        if self._samples:
            try:
                result["assessment"] = assess_object_acquisition(
                    self.binding.acquisition,
                    self._samples,
                    reset_digest=self.binding.reset_digest,
                    episode_id=self._episode_id,
                    target_object_id=self.binding.information.setup_object_coordinates.object_id,
                    episode_started_at_sim_time_s=self._origin,
                    observation_series_complete=observation_series_complete,
                )
                result["status"] = "observed"
            except ValueError as exc:
                result["status"] = "invalid_observation"
                result["blocker"] = str(exc)
        else:
            result["media_gap"] = "no_policy_observation_acquisition_sample"
        result["receipt_digest"] = canonical_digest(result)
        return result


def require_episode_acquisition_evidence(
    *, episode: Mapping[str, Any], binding: Mapping[str, Any], output_root: Path
) -> None:
    """Re-read the acquisition artifacts before accepting a completed episode."""

    import numpy as np

    protocol = NativeObservationProtocol.model_validate(binding)
    receipt = episode.get("object_acquisition")
    frames = episode.get("candidate_exact_policy_input_frames") or []
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("status") != "observed"
        or receipt.get("binding_digest") != protocol.binding_digest
        or receipt.get("episode_id") != episode.get("episode_id")
        or receipt.get("candidate_id") != episode.get("candidate_id")
        or receipt.get("observation_series_complete") is not True
        or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
        or not frames
        or len(receipt.get("sample_artifacts") or []) != len(frames)
        or len(receipt.get("samples") or []) != len(frames)
    ):
        raise ValueError("policy_acquisition_receipt_binding_invalid")

    def read_artifact(record):
        path = output_root / str(record.get("relative_path") or "")
        if (
            path.is_symlink()
            or output_root.resolve() not in path.resolve().parents
            or not path.is_file()
        ):
            raise ValueError("policy_acquisition_artifact_path_invalid")
        data = path.read_bytes()
        if len(data) != record.get("size_bytes") or "sha256:" + hashlib.sha256(
            data
        ).hexdigest() != record.get("sha256"):
            raise ValueError("policy_acquisition_artifact_digest_mismatch")
        return path, data

    samples = [AcquisitionSample.model_validate(s) for s in receipt["samples"]]
    for record, sample, frame in zip(receipt["sample_artifacts"], samples, frames, strict=True):
        _, data = read_artifact(record)
        evidence = json.loads(data)
        if (
            evidence.get("sample") != sample.model_dump(mode="json")
            or evidence.get("sample_evidence_digest")
            != canonical_digest(evidence, digest_field="sample_evidence_digest")
            or evidence.get("sample_evidence_digest") != record.get("sample_evidence_digest")
            or evidence.get("exact_policy_frame_manifest_digest")
            != frame.get("frame_manifest_digest")
            or {r.get("camera_role") for r in evidence.get("files") or []} != set(_VIEW_FOR_ROLE)
        ):
            raise ValueError("policy_acquisition_exact_frame_binding_invalid")
        for artifact in evidence["files"]:
            role = artifact["camera_role"]
            raw_binding = frame["raw_policy_input_camera_bindings"][role]["raw_rgb_sha256"]
            if evidence["native_camera_evidence"][role]["raw_rgb_digest"] != raw_binding:
                raise ValueError("policy_acquisition_native_rgb_binding_invalid")
            path, _ = read_artifact(artifact)
            with np.load(path, allow_pickle=False) as arrays:
                mask = arrays["policy_target_mask"]
                if (
                    mask.shape != tuple(frame["view_shapes"][_VIEW_FOR_ROLE[role]][:2])
                    or mask.dtype != np.uint8
                    or not np.isin(mask, [0, 1]).all()
                    or int(mask.sum()) != sample.cameras[role].target_pixels
                ):
                    raise ValueError("policy_acquisition_mask_readback_invalid")
    expected = assess_object_acquisition(
        protocol.acquisition,
        samples,
        reset_digest=protocol.reset_digest,
        episode_id=episode["episode_id"],
        target_object_id=protocol.information.setup_object_coordinates.object_id,
        episode_started_at_sim_time_s=samples[0].episode_started_at_sim_time_s,
        observation_series_complete=True,
    )
    if receipt.get("assessment") != expected:
        raise ValueError("policy_acquisition_assessment_readback_mismatch")
