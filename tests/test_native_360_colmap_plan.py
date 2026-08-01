from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_360_colmap_plan import (
    Native360ColmapPlanError,
    compile_native_360_colmap_execution_plan,
)
from blueprint_pipeline.native_360_colmap_runner import (
    Native360ColmapRunnerError,
    build_native_360_colmap_pose_estimator_service,
    execute_native_360_colmap_plan,
)
from blueprint_pipeline.reconstruction_worker_contracts import (
    PINNED_MODEL_ASSETS,
    build_pose_estimation_request,
)


CAPTURE_DIGEST = "sha256:" + "1" * 64
SPLIT_DIGEST = "sha256:" + "2" * 64
SOURCE_DIGEST = "sha256:" + "3" * 64
FRONT_MASK_BYTES = b"front-valid-pixel-mask"
REAR_MASK_BYTES = b"rear-valid-pixel-mask"
FRONT_MASK = "sha256:" + hashlib.sha256(FRONT_MASK_BYTES).hexdigest()
REAR_MASK = "sha256:" + hashlib.sha256(REAR_MASK_BYTES).hexdigest()
RIG_VALIDATION_REQUEST_DIGEST = "sha256:" + "6" * 64
NORMALIZATION_DIGEST = "sha256:" + "7" * 64
BINDING_DIGEST = "sha256:" + "8" * 64
IMAGE = "registry.example/blueprint/reconstruction@sha256:" + "a" * 64
SOURCE_SHA = "b" * 40


def _calibration(lens: str) -> dict:
    return {
        "lens_id": lens,
        "intrinsics": {
            "fx": 1900.0,
            "fy": 1901.0,
            "cx": 1920.0,
            "cy": 1920.0,
            "width": 3840,
            "height": 3840,
        },
        "distortion": {
            "model": "opencv_fisheye",
            "coefficients": [0.01, -0.001, 0.0001, -0.00001],
        },
        "valid_pixel_mask_digest": FRONT_MASK if lens == "front" else REAR_MASK,
        "calibration_source": "official_sdk_sidecar",
        "calibration_source_digest": "sha256:" + "9" * 64,
    }


def _rig(*, inverse_semantics: bool = False) -> dict:
    value = {
        "schema_version": "camera_360_rig_declaration.v1",
        "capture_digest": CAPTURE_DIGEST,
        "camera_model": "Insta360 fixture",
        "capture_mode": "dual_fisheye_video",
        "firmware_version": "fixture-1",
        "lens_calibrations": [_calibration("front"), _calibration("rear")],
        "rig_extrinsics": {
            "T_front_rear": [
                [1.0, 0.0, 0.0, 0.06],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "transform_semantics": (
                "front_rig_from_rear_camera" if inverse_semantics else "rear_camera_from_front_rig"
            ),
            "translation_units": "meters",
            "calibration_source": "official_sdk_sidecar",
            "calibration_source_digest": "sha256:" + "9" * 64,
        },
        "rig_is_fixed": True,
        "calibration_status": "valid",
        "metric_scale_status": "not_established",
        "agent_may_alter_calibration": False,
        "blockers": [],
    }
    value["rig_declaration_digest"] = canonical_digest(value, digest_field="rig_declaration_digest")
    return value


def _rig_result(rig: dict) -> dict:
    value = {
        "schema_version": "camera_rig_validation_result.v1",
        "source_capture_digest": CAPTURE_DIGEST,
        "camera_rig_validation_request_digest": RIG_VALIDATION_REQUEST_DIGEST,
        "native_360_normalization_digest": NORMALIZATION_DIGEST,
        "rig_declaration_digest": rig["rig_declaration_digest"],
        "dual_fisheye_binding_digest": BINDING_DIGEST,
        "status": "validated",
        "blockers": [],
        "fixed_rig_extrinsics_valid": True,
        "lens_calibration_valid": True,
        "lens_streams_synchronized": True,
        "capture_timeline_valid": True,
        "original_distorted_pixels_preserved": True,
        "agent_altered_calibration": False,
        "metric_scale_proven": False,
        "camera_trajectory_proven": False,
        "proof_effect": "calibrated_camera_rig_only",
        "claim_ceiling": "calibrated_camera_rig",
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "parent_artifact_or_event": {"native_360_normalization_digest": NORMALIZATION_DIGEST},
        "timestamp": "2026-08-01T12:00:00Z",
    }
    value["camera_rig_validation_result_digest"] = canonical_digest(
        value, digest_field="camera_rig_validation_result_digest"
    )
    return value


def _candidate() -> dict:
    frames = []
    for group_index in range(3):
        for lens_index, lens in enumerate(("front", "rear")):
            frames.append(
                {
                    "frame_id": f"segment-0000-{lens}-{group_index:09d}",
                    "decoded_frame_index": group_index,
                    "t_video_sec": group_index * 0.033333 + lens_index * 0.000001,
                    "frame_digest": "sha256:"
                    + hashlib.sha256(f"captured-{lens}-{group_index}".encode()).hexdigest(),
                    "split": "validation" if group_index == 2 else "training",
                    "candidate_relative_path": (
                        "frozen_dataset/candidate_dataset/"
                        f"{lens}/segment-0000-{lens}-{group_index:09d}.png"
                    ),
                    "image_metadata": {"width": 3840, "height": 3840},
                    "quality_signals": {"blur_score": 0.1},
                    "source_camera_identity": lens,
                    "observation_group_id": f"segment-0000-pair-{group_index:09d}",
                }
            )
    value = {
        "schema_version": "candidate_reconstruction_dataset_manifest.v1",
        "capture_digest": CAPTURE_DIGEST,
        "split_digest": SPLIT_DIGEST,
        "training_and_validation_only": True,
        "heldout_pixels_included": False,
        "frames": frames,
    }
    value["candidate_dataset_digest"] = canonical_digest(
        value, digest_field="candidate_dataset_digest"
    )
    return value


def _dataset(candidate: dict, rig: dict, **overrides: object) -> dict:
    value = {
        "schema_version": "reconstruction_dataset_manifest.v1",
        "source_capture_identity": "native-capture-1",
        "source_capture_digest": CAPTURE_DIGEST,
        "original_file_references": [
            {"relative_path": "native/capture.insv", "digest": SOURCE_DIGEST}
        ],
        "capture_authority_profile": "camera_360_native",
        "train_heldout_split_digest": SPLIT_DIGEST,
        "output_digests": {
            "candidate_dataset_digest": candidate["candidate_dataset_digest"],
            "hidden_heldout_digest": "sha256:" + "c" * 64,
        },
        "candidate_dataset_contains_hidden_heldout_pixels": False,
        "candidate_can_modify_split": False,
        "raw_capture_bytes_remain_authoritative": True,
        "camera_calibration_binding": {
            "camera_360_rig_declaration_digest": rig["rig_declaration_digest"]
        },
        "coordinate_frame_declaration": {
            "units": "meters",
            "handedness": "right_handed",
            "camera_axes": "+x right, +y down, +z forward",
            "rig_frame": "front_lens_optical_center",
        },
        "blockers": [],
    }
    value.update(overrides)
    value["dataset_manifest_digest"] = canonical_digest(
        value, digest_field="dataset_manifest_digest"
    )
    return value


def _pose(
    dataset: dict,
    rig: dict,
    rig_result: dict,
    *,
    method: str = "colmap_sift_bruteforce_v1",
) -> dict:
    method_fields = {
        "colmap_sift_bruteforce_v1": ("SIFT", "SIFT_BRUTEFORCE", None, None),
        "colmap_sift_lightglue_v1": (
            "SIFT",
            "SIFT_LIGHTGLUE",
            None,
            PINNED_MODEL_ASSETS[2]["digest"],
        ),
        "colmap_aliked_bruteforce_v1": (
            "ALIKED_N16ROT",
            "ALIKED_BRUTEFORCE",
            PINNED_MODEL_ASSETS[0]["digest"],
            PINNED_MODEL_ASSETS[3]["digest"],
        ),
        "colmap_aliked_lightglue_v1": (
            "ALIKED_N16ROT",
            "ALIKED_LIGHTGLUE",
            PINNED_MODEL_ASSETS[0]["digest"],
            PINNED_MODEL_ASSETS[1]["digest"],
        ),
    }
    extractor, matcher, feature_digest, matcher_digest = method_fields[method]
    return build_pose_estimation_request(
        {
            "stable_run_identity": "native-pose-request-1",
            "source_capture_identity": "native-capture-1",
            "source_capture_digest": CAPTURE_DIGEST,
            "original_file_references": [
                {"artifact_id": "native/capture.insv", "digest": SOURCE_DIGEST}
            ],
            "producing_method": "blueprint.native_360_pose_request_compiler",
            "implementation_version": "1.0.0",
            "container_image_digest": IMAGE,
            "source_commit_sha": SOURCE_SHA,
            "deterministic_configuration_digest": "sha256:" + "d" * 64,
            "input_digests": [
                {
                    "artifact_id": "reconstruction_dataset",
                    "digest": dataset["dataset_manifest_digest"],
                }
            ],
            "output_digests": [],
            "train_heldout_split_digest": SPLIT_DIGEST,
            "camera_calibration_binding": dataset["camera_calibration_binding"],
            "coordinate_frame_declaration": dataset["coordinate_frame_declaration"],
            "units": "unknown",
            "metric_scale_status": "anchor_required",
            "provider_runtime_identity": {"provider": "local", "runtime": "fixture"},
            "cost_usd": 0.0,
            "duration_seconds": 0.0,
            "authority_used": {"authority_id": "local-fixture"},
            "warnings": ["metric_scale_anchor_required_after_pose_estimation"],
            "blockers": [],
            "proof_effect": "none",
            "claim_ceiling": "request_only",
            "parent_artifact_or_event": {},
            "timestamp": "2026-08-01T12:00:00Z",
            "method_profile_id": method,
            "feature_extractor": extractor,
            "feature_matcher": matcher,
            "camera_model": "OPENCV_FISHEYE",
            "reconstruction_dataset_digest": dataset["dataset_manifest_digest"],
            "camera_rig_digest": rig_result["camera_rig_validation_result_digest"],
            "calibration_digest": rig["rig_declaration_digest"],
            "model_asset_digest": feature_digest,
            "matcher_model_asset_digest": matcher_digest,
            "deterministic_matching": True,
            "random_seed": 17,
            "resource_request": {"gpu_count": 1, "minimum_vram_gb": 16},
            "timeout_seconds": 900,
            "spend_cap_usd": 0.0,
            "candidate_dataset_contains_hidden_heldout_pixels": False,
            "candidate_can_change_split": False,
            "candidate_may_read_hidden_heldout": False,
        }
    )


def _arguments(
    *, inverse_semantics: bool = False, method: str = "colmap_sift_bruteforce_v1"
) -> dict:
    rig = _rig(inverse_semantics=inverse_semantics)
    candidate = _candidate()
    dataset = _dataset(candidate, rig)
    rig_result = _rig_result(rig)
    return {
        "stable_run_identity": "native-colmap-plan-1",
        "reconstruction_dataset": dataset,
        "candidate_dataset_manifest": candidate,
        "camera_rig_declaration": rig,
        "camera_rig_validation_result": rig_result,
        "pose_estimation_request": _pose(dataset, rig, rig_result, method=method),
        "valid_pixel_mask_references": {
            "front": {"relative_path": "calibration/front-mask.png", "digest": FRONT_MASK},
            "rear": {"relative_path": "calibration/rear-mask.png", "digest": REAR_MASK},
        },
        "timestamp": "2026-08-01T12:00:00Z",
    }


def _compile(**overrides: object) -> dict:
    arguments = _arguments()
    arguments.update(overrides)
    return compile_native_360_colmap_execution_plan(**arguments)


def test_native_colmap_plan_is_replayable_candidate_only_and_schema_valid() -> None:
    first = _compile()
    second = _compile(timestamp="2026-08-01T12:00:00Z")

    assert first == second
    assert first["plan_executed"] is False
    assert first["hidden_heldout_access_allowed"] is False
    assert len(first["image_materialization"]) == 6
    assert len(first["mask_materialization"]) == 6
    assert all(
        "held_out" not in row["source_relative_path"] for row in first["image_materialization"]
    )
    by_group: dict[str, set[str]] = {}
    by_group_names: dict[str, set[str]] = {}
    for row in first["image_materialization"]:
        by_group.setdefault(row["observation_group_id"], set()).add(row["sensor_id"])
        by_group_names.setdefault(row["observation_group_id"], set()).add(
            Path(row["destination_relative_path"]).name
        )
    assert all(lenses == {"front", "rear"} for lenses in by_group.values())
    assert all(len(names) == 1 for names in by_group_names.values())
    commands = {row["step_id"]: row["argv"] for row in first["commands"]}
    assert commands["configure_fixed_rig"][1] == "rig_configurator"
    assert commands["export_registered_model_text"][1] == "model_converter"
    assert "--Mapper.ba_refine_sensor_from_rig" in commands["map_fixed_calibrated_rig"]
    assert (
        commands["map_fixed_calibrated_rig"][
            commands["map_fixed_calibrated_rig"].index("--Mapper.ba_refine_sensor_from_rig") + 1
        ]
        == "0"
    )
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/native_360_colmap_execution_plan.v1.schema.json"
        ).read_text()
    )
    jsonschema.validate(first, schema)


def test_native_colmap_plan_honors_declared_transform_direction() -> None:
    direct = compile_native_360_colmap_execution_plan(**_arguments())
    inverse = compile_native_360_colmap_execution_plan(**_arguments(inverse_semantics=True))

    direct_rear = direct["rig_config"][0]["cameras"][1]
    inverse_rear = inverse["rig_config"][0]["cameras"][1]
    assert direct_rear["cam_from_rig_rotation"] == [1.0, 0.0, 0.0, 0.0]
    assert direct_rear["cam_from_rig_translation"] == [0.06, 0.0, 0.0]
    assert inverse_rear["cam_from_rig_rotation"] == [1.0, 0.0, 0.0, 0.0]
    assert inverse_rear["cam_from_rig_translation"] == [-0.06, 0.0, 0.0]


@pytest.mark.parametrize(
    ("method", "extractor", "matcher"),
    [
        ("colmap_sift_bruteforce_v1", "SIFT", "SIFT_BRUTEFORCE"),
        ("colmap_sift_lightglue_v1", "SIFT", "SIFT_LIGHTGLUE"),
        ("colmap_aliked_bruteforce_v1", "ALIKED_N16ROT", "ALIKED_BRUTEFORCE"),
        ("colmap_aliked_lightglue_v1", "ALIKED_N16ROT", "ALIKED_LIGHTGLUE"),
    ],
)
def test_native_colmap_plan_compiles_all_prequalified_feature_pairings(
    method: str, extractor: str, matcher: str
) -> None:
    plan = compile_native_360_colmap_execution_plan(**_arguments(method=method))
    commands = {row["step_id"]: row["argv"] for row in plan["commands"]}

    extraction = commands["extract_features"]
    matching = commands["match_sequential_frames"]
    assert extraction[extraction.index("--FeatureExtraction.type") + 1] == extractor
    assert matching[matching.index("--FeatureMatching.type") + 1] == matcher
    if extractor == "ALIKED_N16ROT":
        assert "/opt/models/colmap/aliked-n16rot.onnx" in extraction
    if matcher != "SIFT_BRUTEFORCE":
        assert any(value.startswith("/opt/models/colmap/") for value in matching)


def test_native_colmap_plan_rejects_axes_masks_distortion_and_hidden_pixels() -> None:
    axes_arguments = _arguments()
    axes_dataset = copy.deepcopy(axes_arguments["reconstruction_dataset"])
    axes_dataset["coordinate_frame_declaration"]["camera_axes"] = "+x forward"
    axes_dataset["dataset_manifest_digest"] = canonical_digest(
        axes_dataset, digest_field="dataset_manifest_digest"
    )
    axes_arguments["reconstruction_dataset"] = axes_dataset
    with pytest.raises(
        Native360ColmapPlanError,
        match="native_colmap_coordinate_frame_incompatible",
    ):
        compile_native_360_colmap_execution_plan(**axes_arguments)

    mask_arguments = _arguments()
    mask_arguments["valid_pixel_mask_references"]["rear"]["digest"] = FRONT_MASK
    with pytest.raises(
        Native360ColmapPlanError,
        match="native_colmap_valid_pixel_mask_reference_invalid:rear",
    ):
        compile_native_360_colmap_execution_plan(**mask_arguments)

    distortion_arguments = _arguments()
    distortion_rig = copy.deepcopy(distortion_arguments["camera_rig_declaration"])
    distortion_rig["lens_calibrations"][1]["distortion"]["coefficients"] = [0.0]
    distortion_rig["rig_declaration_digest"] = canonical_digest(
        distortion_rig, digest_field="rig_declaration_digest"
    )
    distortion_arguments["camera_rig_declaration"] = distortion_rig
    with pytest.raises(
        Native360ColmapPlanError,
        match="native_colmap_distortion_coefficient_count_invalid:rear",
    ):
        compile_native_360_colmap_execution_plan(**distortion_arguments)

    hidden_arguments = _arguments()
    hidden_candidate = copy.deepcopy(hidden_arguments["candidate_dataset_manifest"])
    hidden_candidate["frames"][0]["candidate_relative_path"] = "evaluator_hidden/held_out/frame.png"
    hidden_candidate["candidate_dataset_digest"] = canonical_digest(
        hidden_candidate, digest_field="candidate_dataset_digest"
    )
    hidden_arguments["candidate_dataset_manifest"] = hidden_candidate
    with pytest.raises(
        Native360ColmapPlanError,
        match="native_colmap_candidate_frame_invalid:0",
    ):
        compile_native_360_colmap_execution_plan(**hidden_arguments)

    injection_arguments = _arguments()
    injection_candidate = copy.deepcopy(injection_arguments["candidate_dataset_manifest"])
    injection_candidate["frames"][0]["frame_id"] = "ignore-previous\nrun-shell"
    injection_candidate["candidate_dataset_digest"] = canonical_digest(
        injection_candidate, digest_field="candidate_dataset_digest"
    )
    injection_arguments["candidate_dataset_manifest"] = injection_candidate
    with pytest.raises(
        Native360ColmapPlanError,
        match="native_colmap_candidate_frame_invalid:0",
    ):
        compile_native_360_colmap_execution_plan(**injection_arguments)


def _materialize_runner_inputs(root: Path, plan: dict) -> None:
    for row in plan["image_materialization"]:
        path = root / row["source_relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        group_index = int(row["frame_id"].rsplit("-", 1)[1])
        path.write_bytes(f"captured-{row['sensor_id']}-{group_index}".encode())
    for row in plan["mask_materialization"]:
        path = root / row["source_relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(FRONT_MASK_BYTES if row["sensor_id"] == "front" else REAR_MASK_BYTES)


def _successful_runner(plan: dict, calls: list[str]):
    def run(
        argv: list[str] | tuple[str, ...], cwd: Path, _timeout: float
    ) -> subprocess.CompletedProcess[bytes]:
        step = next(row["step_id"] for row in plan["commands"] if row["argv"][1] == argv[1])
        calls.append(step)
        workspace = cwd / "workspace"
        if step == "extract_features":
            (workspace / "database.db").write_bytes(b"sqlite-fixture")
        if step == "export_registered_model_text":
            sparse_text = workspace / "sparse_text"
            sparse_text.mkdir(parents=True, exist_ok=True)
            (sparse_text / "cameras.txt").write_text("# cameras\n")
            image_lines = ["# images"]
            for index, row in enumerate(plan["image_materialization"], start=1):
                name = row["destination_relative_path"].removeprefix("workspace/images/")
                image_lines.extend([f"{index} 1 0 0 0 0 0 0 1 {name}", "0 0 -1"])
            (sparse_text / "images.txt").write_text("\n".join(image_lines) + "\n")
            (sparse_text / "points3D.txt").write_text("# points\n")
        return subprocess.CompletedProcess(list(argv), 0, b"ok\n", b"")

    return run


def test_native_colmap_runner_executes_once_and_replays_typed_result(
    tmp_path: Path,
) -> None:
    plan = _compile()
    input_root = tmp_path / "inputs"
    output_root = tmp_path / "outputs"
    _materialize_runner_inputs(input_root, plan)
    calls: list[str] = []

    result = execute_native_360_colmap_plan(
        plan=plan,
        input_root=input_root,
        artifact_root=output_root,
        timestamp="2026-08-01T12:30:00Z",
        runner=_successful_runner(plan, calls),
    )

    assert result["status"] == "succeeded"
    assert result["failure_code"] is None
    assert len(result["registered_observation_ids"]) == 6
    assert result["rejected_observation_ids"] == []
    assert len(calls) == 5
    assert result["heldout_labels_included"] is False
    assert result["candidate_self_graded"] is False

    def forbidden_runner(*_args: object) -> subprocess.CompletedProcess[bytes]:
        raise AssertionError("an accepted result must replay without re-execution")

    replay = execute_native_360_colmap_plan(
        plan=plan,
        input_root=input_root,
        artifact_root=output_root,
        timestamp="2099-01-01T00:00:00Z",
        runner=forbidden_runner,
    )
    assert replay == result


def test_native_colmap_service_accepts_only_the_registered_pose_request(
    tmp_path: Path,
) -> None:
    arguments = _arguments()
    plan = compile_native_360_colmap_execution_plan(**arguments)
    input_root = tmp_path / "inputs"
    _materialize_runner_inputs(input_root, plan)
    service = build_native_360_colmap_pose_estimator_service(
        plan=plan,
        input_root=input_root,
        timestamp="2026-08-01T12:30:00Z",
        runner=_successful_runner(plan, []),
    )

    result = service(
        request=arguments["pose_estimation_request"],
        output_root=tmp_path / "pose_estimation",
    )
    assert result["status"] == "succeeded"

    rebound = copy.deepcopy(arguments["pose_estimation_request"])
    rebound["random_seed"] = 18
    rebound["pose_estimation_request_digest"] = canonical_digest(
        rebound, digest_field="pose_estimation_request_digest"
    )
    with pytest.raises(
        Native360ColmapRunnerError,
        match="native_colmap_service_pose_request_binding_mismatch",
    ):
        service(request=rebound, output_root=tmp_path / "rebound")


def test_native_colmap_runner_preserves_failure_and_stops_unchanged_retry(
    tmp_path: Path,
) -> None:
    plan = _compile()
    input_root = tmp_path / "inputs"
    output_root = tmp_path / "outputs"
    _materialize_runner_inputs(input_root, plan)
    calls: list[str] = []

    def failing_runner(
        argv: list[str] | tuple[str, ...], cwd: Path, timeout: float
    ) -> subprocess.CompletedProcess[bytes]:
        successful = _successful_runner(plan, calls)(argv, cwd, timeout)
        if argv[1] == "sequential_matcher":
            return subprocess.CompletedProcess(list(argv), 9, b"", b"match failure")
        return successful

    result = execute_native_360_colmap_plan(
        plan=plan,
        input_root=input_root,
        artifact_root=output_root,
        timestamp="2026-08-01T12:30:00Z",
        runner=failing_runner,
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "pose_estimation_failure"
    assert result["registered_observation_ids"] == []
    assert len(result["rejected_observation_ids"]) == 6
    assert calls == [
        "extract_features",
        "configure_fixed_rig",
        "match_sequential_frames",
    ]
    replay = execute_native_360_colmap_plan(
        plan=plan,
        input_root=input_root,
        artifact_root=output_root,
        timestamp="2099-01-01T00:00:00Z",
        runner=lambda *_args: (_ for _ in ()).throw(AssertionError("no retry")),
    )
    assert replay == result


def test_native_colmap_runner_rejects_worker_returned_image_substitution(
    tmp_path: Path,
) -> None:
    plan = _compile()
    input_root = tmp_path / "inputs"
    _materialize_runner_inputs(input_root, plan)
    base_runner = _successful_runner(plan, [])

    def substituted_runner(
        argv: list[str] | tuple[str, ...], cwd: Path, timeout: float
    ) -> subprocess.CompletedProcess[bytes]:
        completed = base_runner(argv, cwd, timeout)
        if argv[1] == "model_converter":
            images = cwd / "workspace/sparse_text/images.txt"
            images.write_text("# images\n1 1 0 0 0 0 0 0 1 injected/unknown.png\n0 0 -1\n")
        return completed

    result = execute_native_360_colmap_plan(
        plan=plan,
        input_root=input_root,
        artifact_root=tmp_path / "outputs",
        timestamp="2026-08-01T12:30:00Z",
        runner=substituted_runner,
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "malformed_output"
    assert "native_colmap_runner_registered_image_not_in_candidate" in result["blockers"]


def test_native_colmap_runner_rejects_command_path_and_symlink_escape(
    tmp_path: Path,
) -> None:
    plan = _compile()
    malicious = copy.deepcopy(plan)
    malicious["commands"][0]["argv"].extend(["--database_path", "/etc/passwd"])
    malicious["native_360_colmap_execution_plan_digest"] = canonical_digest(
        malicious, digest_field="native_360_colmap_execution_plan_digest"
    )
    with pytest.raises(
        Native360ColmapRunnerError,
        match="native_colmap_runner_command_path_invalid:0",
    ):
        execute_native_360_colmap_plan(
            plan=malicious,
            input_root=tmp_path,
            artifact_root=tmp_path / "outputs",
            timestamp="2026-08-01T12:30:00Z",
        )

    input_root = tmp_path / "inputs"
    _materialize_runner_inputs(input_root, plan)
    first = plan["image_materialization"][0]
    source = input_root / first["source_relative_path"]
    source.unlink()
    source.symlink_to(tmp_path / "outside.png")
    with pytest.raises(
        Native360ColmapRunnerError,
        match="native_colmap_runner_input_symlink_forbidden",
    ):
        execute_native_360_colmap_plan(
            plan=plan,
            input_root=input_root,
            artifact_root=tmp_path / "outputs",
            timestamp="2026-08-01T12:30:00Z",
        )
