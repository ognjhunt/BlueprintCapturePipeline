from __future__ import annotations

import hashlib
import json
import subprocess
import struct
import zipfile
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import local_reconstruction_adapters as adapters
from blueprint_pipeline.canonical_3dgs_admission import (
    build_canonical_3dgs_worker_admission,
)
from blueprint_pipeline.canonical_3dgs_pipeline import (
    Canonical3DGSPipelineError,
    build_canonical_3dgs_execution_plan,
    build_canonical_3dgs_source_admission,
    canonical_3dgs_worker_package_digest,
    canonical_3dgs_worker_wheel_package_digest,
    execute_canonical_3dgs_plan,
    finalize_canonical_3dgs_receipts,
    prepare_canonical_v32_training_dataset,
)
from blueprint_pipeline.canonical_3dgs_execution_request import (
    build_canonical_3dgs_execution_request,
)
from blueprint_pipeline.canonical_3dgs_transport import (
    Canonical3DGSTransportError,
    compile_canonical_3dgs_transport_bundle,
    extract_canonical_3dgs_transport_bundle,
    validate_canonical_3dgs_transport_receipt,
)
from blueprint_pipeline.canonical_3dgs_evaluation import (
    evaluate_canonical_3dgs_campaign,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


CAPTURE_DIGEST = "sha256:" + "a" * 64
SPLIT_DIGEST = "sha256:" + "b" * 64
SOURCE_COMMIT = "c" * 40


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_standard_splat(path: Path) -> None:
    properties = [
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        + "".join(f"property float {name}\n" for name in properties)
        + "end_header\n"
    )
    path.write_bytes(
        header.encode("ascii")
        + struct.pack("<14f", 0, 0, 1, 0, 0, 0, 1, -3, -3, -3, 1, 0, 0, 0)
    )


def _dataset(root: Path) -> dict:
    (root / "images").mkdir(parents=True)
    for index in range(3):
        (root / "images" / f"frame-{index}.png").write_bytes(
            b"fixture-image-" + str(index).encode()
        )
    (root / "sparse/0").mkdir(parents=True)
    (root / "sparse/0/cameras.txt").write_text(
        "1 PINHOLE 64 48 50 50 32 24\n", encoding="utf-8"
    )
    (root / "sparse/0/images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 frame-0.png\n\n", encoding="utf-8"
    )
    (root / "sparse/0/points3D.txt").write_text(
        "1 0 0 -1 128 128 128 0\n", encoding="utf-8"
    )
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "artifact_type": (
                        "candidate_image" if path.parent.name == "images" else "colmap_sparse_text"
                    ),
                    "relative_path": path.relative_to(root).as_posix(),
                    "digest": _digest(path),
                }
            )
    request_digest = _sha("9")
    dataset_digest = canonical_digest(
        {
            "images": [
                {"artifact_id": Path(row["relative_path"]).name, "digest": row["digest"]}
                for row in rows
                if row["artifact_type"] == "candidate_image"
            ],
            "sparse": {
                Path(row["relative_path"]).name: row["digest"]
                for row in rows
                if row["artifact_type"] == "colmap_sparse_text"
            },
            "request_digest": request_digest,
        }
    )
    result = {
        "schema_version": "colmap_training_dataset_export_result.v1",
        "status": "exported_candidate_only_colmap_text_dataset",
        "source_capture_digest": CAPTURE_DIGEST,
        "frozen_split_digest": SPLIT_DIGEST,
        "colmap_training_dataset_digest": dataset_digest,
        "output_artifacts": rows,
        "image_count": 3,
        "initialization_point_count": 1,
        "hidden_heldout_pixels_included": False,
        "trainer_self_grading_permitted": False,
        "raw_input_poses_modified": False,
        "parent_artifact_or_event": {"request_digest": request_digest},
    }
    result["colmap_training_dataset_export_result_digest"] = canonical_digest(
        result, digest_field="colmap_training_dataset_export_result_digest"
    )
    return result


def _preparation(dataset: dict) -> dict:
    value = {
        "schema_version": "canonical_v32_3dgs_preparation.v1",
        "status": "training_dataset_ready",
        "source_profile": "blueprint_raw_v3_2",
        "canonical_3dgs_source_admission_digest": _sha("8"),
        "source_capture_digest": CAPTURE_DIGEST,
        "raw_contract_3_2_proven": True,
        "colmap_training_dataset_digest": dataset["colmap_training_dataset_digest"],
        "colmap_training_dataset_export_result_digest": dataset[
            "colmap_training_dataset_export_result_digest"
        ],
        "frozen_split_digest": SPLIT_DIGEST,
        "pose_binding": "raw_arkit_pose_baseline",
        "world_frame": "canonical_arkit_world",
        "coordinate_frame_declaration": {"frame": "canonical_arkit_world"},
        "metric_scale_status": "sensor_metric_unvalidated",
        "timestamp": "2026-08-03T05:00:00Z",
    }
    value["canonical_v32_3dgs_preparation_digest"] = canonical_digest(
        value, digest_field="canonical_v32_3dgs_preparation_digest"
    )
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _canonical_v32_fixture(root: Path, frame_count: int = 8) -> list[float]:
    times = [round(index * 0.1, 6) for index in range(frame_count)]
    (root / "arkit/depth").mkdir(parents=True)
    (root / "arkit/confidence").mkdir(parents=True)
    (root / "walkthrough.mov").write_bytes(b"retained-v32-video")
    frame_ids = [f"{index + 1:06d}" for index in range(frame_count)]
    for frame_id in frame_ids:
        Image.fromarray(np.full((6, 8), 1000, dtype=np.uint16)).save(
            root / f"arkit/depth/{frame_id}.png"
        )
        Image.fromarray(np.full((6, 8), 2, dtype=np.uint8)).save(
            root / f"arkit/confidence/{frame_id}.png"
        )
    _write_json(
        root / "manifest.json",
        {
            "capture_schema_version": "3.2.0",
            "capture_profile_id": "iphone_arkit_lidar",
            "coordinate_frame_session_id": "fixture-session",
            "capture_capabilities": {
                "camera_pose": True,
                "camera_intrinsics": True,
                "depth": True,
                "depth_confidence": True,
                "tracking_state": True,
                "tracking_state_rows": frame_count,
            },
        },
    )
    _write_json(
        root / "video_track.json",
        {
            "video_file": "walkthrough.mov",
            "frame_count": frame_count,
            "frame_count_source": "decoded_sample_presentation_timestamps",
            "decoded_pts_verified": True,
            "write_attempt_count": frame_count,
            "retained_frame_count": frame_count,
            "dropped_frame_count": 0,
        },
    )
    _write_jsonl(
        root / "video_frame_retention.jsonl",
        [
            {
                "write_attempt_index": index,
                "frame_id": frame_id,
                "retention_status": "retained",
                "drop_reason": None,
                "encoded_frame_index": index,
                "t_video_sec": times[index],
            }
            for index, frame_id in enumerate(frame_ids)
        ],
    )
    _write_jsonl(
        root / "sync_map.jsonl",
        [
            {
                "frame_id": frame_id,
                "t_video_sec": times[index],
                "t_capture_sec": times[index],
                "sync_status": "encoded_decoded_pts_match",
                "pose_frame_id": frame_id,
                "encoded_frame_index": index,
                "write_attempt_index": index,
            }
            for index, frame_id in enumerate(frame_ids)
        ],
    )
    poses = []
    for index, frame_id in enumerate(frame_ids):
        pose = [[1, 0, 0, index * 0.05], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        poses.append(
            {
                "frame_id": frame_id,
                "coordinate_frame_session_id": "fixture-session",
                "T_world_camera": pose,
            }
        )
    _write_jsonl(root / "arkit/poses.jsonl", poses)
    _write_jsonl(root / "arkit/frames.jsonl", [{"frame_id": value} for value in frame_ids])
    _write_json(
        root / "arkit/session_intrinsics.json",
        {
            "coordinate_frame_session_id": "fixture-session",
            "intrinsics": {"fx": 50, "fy": 50, "cx": 32, "cy": 24, "width": 64, "height": 48},
        },
    )
    _write_json(
        root / "recording_session.json",
        {
            "coordinate_frame_session_id": "fixture-session",
            "world_frame_definition": "arkit_world_origin_at_session_start",
            "units": "meters",
            "up_axis": "Y",
            "handedness": "right_handed",
            "gravity_aligned": True,
            "session_reset_count": 0,
        },
    )
    depth_rows = [
        {
            "frame_id": frame_id,
            "depth_path": f"arkit/depth/{frame_id}.png",
            "paired_confidence_path": f"arkit/confidence/{frame_id}.png",
        }
        for frame_id in frame_ids
    ]
    confidence_rows = [
        {
            "frame_id": frame_id,
            "confidence_path": f"arkit/confidence/{frame_id}.png",
            "paired_depth_path": f"arkit/depth/{frame_id}.png",
        }
        for frame_id in frame_ids
    ]
    _write_json(
        root / "arkit/depth_manifest.json",
        {
            "schema_version": "arkit_depth_manifest.v2",
            "depth_encoding": "uint16_png",
            "scale_to_meters": 0.001,
            "camera_ray_convention": "arkit_x_right_y_up_z_backward",
            "depth_registered_to_arkit_camera": True,
            "depth_intrinsics": {"fx": 12, "fy": 12, "cx": 3.5, "cy": 2.5, "width": 8, "height": 6},
            "frames": depth_rows,
        },
    )
    _write_json(
        root / "arkit/confidence_manifest.json",
        {
            "schema_version": "arkit_confidence_manifest.v2",
            "confidence_encoding": "uint8_png",
            "accepted_confidence_values": [2],
            "frames": confidence_rows,
        },
    )
    return times


def _stub_v32_media(monkeypatch: pytest.MonkeyPatch, times: list[float]) -> None:
    monkeypatch.setattr(adapters.shutil, "which", lambda name: f"/fake/{name}")

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[-1] == "-version":
            return subprocess.CompletedProcess(command, 0, "fixture version\n", "")
        if "-show_frames" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "streams": [{"index": 0, "width": 64, "height": 48, "codec_name": "h264"}],
                        "frames": [
                            {
                                "best_effort_timestamp": str(round(value * 1000)),
                                "best_effort_timestamp_time": str(value),
                            }
                            for value in times
                        ],
                    }
                ),
                "",
            )
        output = Path(command[-1])
        selected = next(value for value in command if value.startswith("select="))
        index = int(selected.split(",")[-1].rstrip(")"))
        Image.new("RGB", (64, 48), color=(index * 20, 30, 40)).save(output, format="PNG")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(adapters.subprocess, "run", run)


def _runner(arm: dict, dataset_root: Path, output_root: Path) -> dict:
    assert (dataset_root / "sparse/0/points3D.txt").is_file()
    _write_standard_splat(output_root / "candidate.ply")
    (output_root / "training.log").write_text("fixture training completed\n", encoding="utf-8")
    artifacts = [
        {"kind": "standard_3dgs_ply", "relative_path": "candidate.ply"},
        {"kind": "training_log", "relative_path": "training.log"},
    ]
    if arm["role"] == "primary":
        (output_root / "project.psht").write_bytes(b"fixture-postshot-project")
        artifacts.append({"kind": "postshot_project", "relative_path": "project.psht"})
    else:
        (output_root / "config.yml").write_text("method: splatfacto\n", encoding="utf-8")
        artifacts.append({"kind": "nerfstudio_config", "relative_path": "config.yml"})
    return {
        "exit_code": 0,
        "argv": arm["train_argv_template"],
        "runtime_identity": {"runtime": "hermetic-fixture", "gpu_execution": False},
        "artifacts": artifacts,
        "timestamp": "2026-08-03T05:00:00Z",
    }


def test_canonical_plan_executes_postshot_primary_and_splatfacto_comparison_on_same_bytes(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset = _dataset(dataset_root)
    plan = build_canonical_3dgs_execution_plan(
        preparation=_preparation(dataset),
        dataset=dataset,
        dataset_root=dataset_root,
        source_commit_sha=SOURCE_COMMIT,
        timestamp="2026-08-03T05:00:00Z",
    )
    result = execute_canonical_3dgs_plan(
        plan=plan,
        dataset_root=dataset_root,
        output_root=tmp_path / "results",
        runners={
            "postshot-primary": _runner,
            "splatfacto-comparison": _runner,
        },
    )

    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/canonical_3dgs_execution.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(plan, schema)
    for arm_result in result["arm_results"]:
        jsonschema.validate(arm_result, schema)
        assert arm_result["status"] == "succeeded"
        assert arm_result["colmap_training_dataset_digest"] == dataset[
            "colmap_training_dataset_digest"
        ]
        assert arm_result["hidden_heldout_pixels_included"] is False
        assert arm_result["quality_claimed"] is False
    campaign = result["campaign"]
    jsonschema.validate(campaign, schema)
    assert campaign["status"] == "candidates_ready_for_independent_evaluation"
    assert campaign["primary_method_id"] == "jawset_postshot_splat3_v1"
    assert campaign["all_arms_used_identical_candidate_dataset"] is True
    assert campaign["quality_winner"] is None
    assert campaign["raw_capture_authority_upgraded"] is False
    assert len(campaign["appearance_fidelity_candidate_bindings"]) == 2
    assert {
        row["candidate_method_id"]
        for row in campaign["appearance_fidelity_candidate_bindings"]
    } == {
        "jawset_postshot_splat3_v1",
        "nerfstudio_splatfacto_v1_1_5",
    }
    assert campaign["next_quality_gate"] == {
        "schema_version": "appearance_fidelity_qualification.v1",
        "required_metrics": ["ssim", "psnr_db", "lpips"],
        "native_3dgs_exact_camera_render_required": True,
        "site_task_specific_thresholds_required": True,
        "default_thresholds_assumed": False,
        "selection_allowed_before_measurement": False,
    }


def test_v32_bundle_prepares_depth_seeded_candidate_only_dataset_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root = tmp_path / "capture"
    times = _canonical_v32_fixture(capture_root)
    _stub_v32_media(monkeypatch, times)

    prepared = prepare_canonical_v32_training_dataset(
        capture_root=capture_root,
        output_root=tmp_path / "derived",
        intake_id="fixture-intake",
        capture_digest=CAPTURE_DIGEST,
        rights_and_retention={
            "local_processing_authorized": True,
            "provider_upload_authorized": False,
            "paid_compute_authorized": False,
        },
        maximum_frames=8,
    )

    assert prepared["preparation"]["status"] == "training_dataset_ready"
    assert prepared["dataset"]["image_count"] >= 3
    assert prepared["dataset"]["initialization_point_count"] > 0
    assert prepared["dataset"]["hidden_heldout_pixels_included"] is False
    evaluator_path = (
        tmp_path
        / "derived"
        / prepared["preparation"]["hidden_evaluator_input_relative_path"]
    )
    evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))
    assert evaluator["status"] == "ready_for_independent_exact_camera_evaluation"
    assert evaluator["camera_count"] > 0
    assert evaluator["candidate_access_allowed"] is False
    assert evaluator["trainer_transport_contains_hidden_pixels"] is False
    assert evaluator["cameras"][0]["T_world_camera_provider_frame"][1][1] == -1.0
    assert evaluator["cameras"][0]["T_world_camera_provider_frame"][2][2] == -1.0
    assert (prepared["dataset_root"] / "sparse/0/points3D.txt").stat().st_size > 0
    first_colmap_pose = (
        prepared["dataset_root"] / "sparse/0/images.txt"
    ).read_text(encoding="utf-8").splitlines()[1].split()
    # The fixture's first raw ARKit pose is identity. COLMAP must receive the
    # explicit +Y/+Z camera-axis flip (180 degrees around +X), while the world
    # origin and metric translations remain unchanged.
    assert [float(value) for value in first_colmap_pose[1:5]] == pytest.approx(
        [0.0, 1.0, 0.0, 0.0]
    )
    plan = build_canonical_3dgs_execution_plan(
        preparation=prepared["preparation"],
        dataset=prepared["dataset"],
        dataset_root=prepared["dataset_root"],
        source_commit_sha=SOURCE_COMMIT,
        timestamp="2026-08-03T05:00:00Z",
    )
    assert plan["primary_method_id"] == "jawset_postshot_splat3_v1"
    assert plan["comparison_method_ids"] == ["nerfstudio_splatfacto_v1_1_5"]

    results_root = tmp_path / "campaign-results"
    campaign_result = execute_canonical_3dgs_plan(
        plan=plan,
        dataset_root=prepared["dataset_root"],
        output_root=results_root,
        runners={
            "postshot-primary": _runner,
            "splatfacto-comparison": _runner,
        },
    )

    evaluator_root = evaluator_path.parent

    def fake_renderer(**arguments: object) -> dict:
        output = Path(arguments["output_dir"])
        frames = output / "frames"
        frames.mkdir(parents=True)
        rendered = []
        for camera in arguments["cameras"]:
            source = evaluator_root / camera["reference_relative_path"]
            target = frames / f"{camera['camera_id']}.png"
            with Image.open(source) as image:
                array = np.asarray(image.convert("RGB")).copy()
            array[0, 0, 0] = (int(array[0, 0, 0]) + 1) % 255
            Image.fromarray(array).save(target)
            rendered.append(
                {
                    "camera_id": camera["camera_id"],
                    "relative_path": f"frames/{camera['camera_id']}.png",
                    "digest": _digest(target),
                }
            )
        return {
            "rendered_by": "fixture-native-3dgs-renderer",
            "renderer_identity": {
                "harness_digest": "sha256:" + "1" * 64,
                "render_entry_digest": "sha256:" + "2" * 64,
                "runtime": "fixture",
            },
            "renders": rendered,
        }

    def fake_evaluator(*, source_artifact: dict, output_root: Path) -> dict:
        del output_root
        postshot = source_artifact["candidate_method_id"] == "jawset_postshot_splat3_v1"
        aggregate = {
            "view_count": len(source_artifact["pairs"]),
            "mean_psnr_db": 31.0 if postshot else 28.0,
            "mean_global_ssim": 0.96 if postshot else 0.91,
            "mean_windowed_ssim": 0.95 if postshot else 0.90,
            "mean_absolute_error": 0.02 if postshot else 0.04,
            "mean_lpips": 0.04 if postshot else 0.08,
            "thresholds_passed": True,
        }
        report = {
            "schema_version": "visual_heldout_evaluation_report.v2",
            "candidate_method_id": source_artifact["candidate_method_id"],
            "by_trajectory": {
                "author_heldout": aggregate,
                "independent_short": {"view_count": 0, "thresholds_passed": None},
            },
            "status": "passed_appearance_only",
        }
        report["visual_heldout_evaluation_report_digest"] = canonical_digest(
            report, digest_field="visual_heldout_evaluation_report_digest"
        )
        return report

    comparison = evaluate_canonical_3dgs_campaign(
        campaign_result=campaign_result["campaign"],
        results_root=results_root,
        evaluator_input=evaluator,
        evaluator_root=evaluator_root,
        thresholds={
            "minimum_mean_psnr_db": 25.0,
            "minimum_mean_global_ssim": 0.8,
            "minimum_mean_windowed_ssim": 0.8,
            "maximum_mean_absolute_error": 0.1,
            "maximum_mean_lpips": 0.2,
        },
        lpips_model={
            "model_id": "lpips_alex_v0.1",
            "checkpoint_digest": "sha256:" + "9" * 64,
        },
        output_root=tmp_path / "quality",
        renderer=fake_renderer,
        appearance_evaluator=fake_evaluator,
    )
    assert comparison["status"] == "quality_winner_selected"
    assert comparison["quality_winner"] == "postshot-primary"
    assert comparison["candidate_hidden_pixel_access"] is False
    assert all(
        row["appearance_fidelity_status"] == "qualified"
        for row in comparison["candidate_reports"]
    )


def test_canonical_plan_rejects_dataset_byte_drift_before_any_runner(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset = _dataset(dataset_root)
    plan = build_canonical_3dgs_execution_plan(
        preparation=_preparation(dataset),
        dataset=dataset,
        dataset_root=dataset_root,
        source_commit_sha=SOURCE_COMMIT,
    )
    (dataset_root / "images/frame-0.png").write_bytes(b"tampered")

    with pytest.raises(Canonical3DGSPipelineError, match="digest_mismatch"):
        execute_canonical_3dgs_plan(
            plan=plan,
            dataset_root=dataset_root,
            output_root=tmp_path / "results",
            runners={
                "postshot-primary": _runner,
                "splatfacto-comparison": _runner,
            },
        )


def test_canonical_plan_rejects_missing_runner_exit_code(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset = _dataset(dataset_root)
    plan = build_canonical_3dgs_execution_plan(
        preparation=_preparation(dataset),
        dataset=dataset,
        dataset_root=dataset_root,
        source_commit_sha=SOURCE_COMMIT,
    )

    def invalid_runner(arm: dict, dataset: Path, output: Path) -> dict:
        (output / "training.log").write_text("missing exit code\n", encoding="utf-8")
        return {
            "artifacts": [{"kind": "training_log", "relative_path": "training.log"}],
        }

    with pytest.raises(Canonical3DGSPipelineError, match="runner_exit_code_invalid"):
        execute_canonical_3dgs_plan(
            plan=plan,
            dataset_root=dataset_root,
            output_root=tmp_path / "results",
            runners={
                "postshot-primary": invalid_runner,
                "splatfacto-comparison": invalid_runner,
            },
        )


def test_canonical_transport_is_reproducible_and_extracts_exact_plan_and_dataset(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset = _dataset(dataset_root)
    plan = build_canonical_3dgs_execution_plan(
        preparation=_preparation(dataset),
        dataset=dataset,
        dataset_root=dataset_root,
        source_commit_sha=SOURCE_COMMIT,
        timestamp="2026-08-03T05:00:00Z",
    )
    first_bundle = tmp_path / "first/campaign.zip"
    first_receipt_path = tmp_path / "first/receipt.json"
    first = compile_canonical_3dgs_transport_bundle(
        plan=plan,
        dataset_root=dataset_root,
        bundle_path=first_bundle,
        receipt_path=first_receipt_path,
    )
    second_bundle = tmp_path / "second/campaign.zip"
    second = compile_canonical_3dgs_transport_bundle(
        plan=plan,
        dataset_root=dataset_root,
        bundle_path=second_bundle,
        receipt_path=tmp_path / "second/receipt.json",
    )

    assert first_bundle.read_bytes() == second_bundle.read_bytes()
    assert first["transport_bundle_digest"] == second["transport_bundle_digest"]
    assert first["hidden_heldout_pixels_included"] is False
    extraction = extract_canonical_3dgs_transport_bundle(
        bundle_path=first_bundle,
        receipt=first,
        output_root=tmp_path / "worker",
    )
    materialized = (
        tmp_path / "worker" / first["transport_bundle_digest"].removeprefix("sha256:")
    )
    extracted_plan = json.loads(
        (materialized / extraction["plan_relative_path"]).read_text(encoding="utf-8")
    )
    assert extracted_plan == plan
    assert (materialized / extraction["dataset_root_relative_path"] / "sparse/0/cameras.txt").is_file()


def test_worker_wheel_digest_rejects_stale_or_extra_python_sources(tmp_path: Path) -> None:
    package = tmp_path / "source" / "blueprint_pipeline"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "worker.py").write_text("VALUE = 2\n", encoding="utf-8")
    wheel = tmp_path / "worker.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.write(package / "__init__.py", "blueprint_pipeline/__init__.py")
        archive.write(package / "worker.py", "blueprint_pipeline/worker.py")

    assert canonical_3dgs_worker_wheel_package_digest(
        wheel
    ) == canonical_3dgs_worker_package_digest(package)

    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("blueprint_pipeline/stale.py", "STALE = True\n")
    assert canonical_3dgs_worker_wheel_package_digest(
        wheel
    ) != canonical_3dgs_worker_package_digest(package)


def test_canonical_transport_rejects_bundle_byte_drift(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset = _dataset(dataset_root)
    plan = build_canonical_3dgs_execution_plan(
        preparation=_preparation(dataset),
        dataset=dataset,
        dataset_root=dataset_root,
        source_commit_sha=SOURCE_COMMIT,
    )
    bundle = tmp_path / "campaign.zip"
    receipt = compile_canonical_3dgs_transport_bundle(
        plan=plan,
        dataset_root=dataset_root,
        bundle_path=bundle,
        receipt_path=tmp_path / "receipt.json",
    )
    with bundle.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(Canonical3DGSTransportError, match="bundle_digest_mismatch"):
        extract_canonical_3dgs_transport_bundle(
            bundle_path=bundle,
            receipt=receipt,
            output_root=tmp_path / "worker",
        )


def test_canonical_transport_receipt_cannot_grant_paid_authority(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset = _dataset(dataset_root)
    plan = build_canonical_3dgs_execution_plan(
        preparation=_preparation(dataset),
        dataset=dataset,
        dataset_root=dataset_root,
        source_commit_sha=SOURCE_COMMIT,
    )
    receipt = compile_canonical_3dgs_transport_bundle(
        plan=plan,
        dataset_root=dataset_root,
        bundle_path=tmp_path / "campaign.zip",
        receipt_path=tmp_path / "receipt.json",
    )
    tampered = dict(receipt)
    tampered["paid_execution_authorized_by_bundle"] = True
    tampered["receipt_digest"] = canonical_digest(tampered, digest_field="receipt_digest")

    with pytest.raises(Canonical3DGSTransportError, match="paid_execution_authorized"):
        validate_canonical_3dgs_transport_receipt(tampered)


def test_proxy_source_admission_cannot_claim_raw_contract() -> None:
    common = {
        "source_profile": "public_dataset_arkitscenes_proxy",
        "source_capture_identity": "ARKitScenes:40958756",
        "source_capture_digest": _sha("1"),
        "source_artifact_commit_sha": SOURCE_COMMIT,
        "frozen_split_digest": _sha("2"),
        "colmap_training_dataset_digest": _sha("3"),
        "hidden_evaluator_input_digest": _sha("4"),
        "world_frame": "arkitscenes_official_loader_world",
        "coordinate_frame_declaration": {"units": "meters"},
        "metric_scale_status": "sensor_metric_unvalidated",
        "authority_used": {"local_processing_authorized": True},
        "input_artifacts": [{"artifact_id": "proxy", "digest": _sha("5")}],
        "claim_limitations": ["public_dataset_proxy_not_blueprint_raw_capture"],
        "timestamp": "2026-08-03T05:00:00Z",
    }
    admission = build_canonical_3dgs_source_admission(
        **common, raw_contract_3_2_proven=False
    )
    assert admission["raw_contract_3_2_proven"] is False
    assert admission["provider_upload_authorized_by_source_admission"] is False
    with pytest.raises(Canonical3DGSPipelineError, match="raw_contract_claim_invalid"):
        build_canonical_3dgs_source_admission(
            **common, raw_contract_3_2_proven=True
        )


def test_execution_request_names_missing_authority_without_a_winner(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset = _dataset(dataset_root)
    plan = build_canonical_3dgs_execution_plan(
        preparation=_preparation(dataset),
        dataset=dataset,
        dataset_root=dataset_root,
        source_commit_sha=SOURCE_COMMIT,
        timestamp="2026-08-03T05:00:00Z",
    )
    transport = compile_canonical_3dgs_transport_bundle(
        plan=plan,
        dataset_root=dataset_root,
        bundle_path=tmp_path / "transport.zip",
        receipt_path=tmp_path / "transport.json",
    )
    request = build_canonical_3dgs_execution_request(
        plan=plan,
        transport_receipt=transport,
        worker_wheel_digest=_sha("9"),
        worker_wheel_filename="blueprint_capture_pipeline-2.0.0-py3-none-any.whl",
        timestamp="2026-08-03T05:00:00Z",
    )
    schema = json.loads(
        (Path(__file__).parents[1] / "docs/schemas/canonical_3dgs_execution_request.v1.schema.json").read_text()
    )
    jsonschema.validate(request, schema)
    assert request["paid_execution_authorized"] is False
    assert request["quality_winner"] is None
    assert request["candidate_generated"] is False
    assert all(arm["retry_cap"] == 0 and arm["blockers"] for arm in request["arms"])
    assert "paid_resource_allocator gpu-canary" in request["provider_launch_entrypoint"]


def test_platform_worker_receipts_finalize_into_one_bound_campaign(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset = _dataset(dataset_root)
    plan = build_canonical_3dgs_execution_plan(
        preparation=_preparation(dataset),
        dataset=dataset,
        dataset_root=dataset_root,
        source_commit_sha=SOURCE_COMMIT,
        timestamp="2026-08-03T05:00:00Z",
    )
    transport = compile_canonical_3dgs_transport_bundle(
        plan=plan,
        dataset_root=dataset_root,
        bundle_path=tmp_path / "transport.zip",
        receipt_path=tmp_path / "transport.json",
    )
    results_root = tmp_path / "results"
    allocator_digests: list[str] = []
    for arm in plan["arms"]:
        run_root = results_root / arm["arm_id"]
        run_root.mkdir(parents=True)
        receipt = _runner(arm, dataset_root, run_root)
        receipt["runtime_identity"].update(
            {
                "worker_python_package_digest": plan[
                    "worker_python_package_digest"
                ],
                "source_commit_sha_bound_by_plan": plan["source_commit_sha"],
            }
        )
        receipt["canonical_3dgs_execution_plan_digest"] = plan[
            "canonical_3dgs_execution_plan_digest"
        ]
        worker_image = "blueprint/canonical-3dgs-worker@sha256:" + (
            "1" if arm["arm_id"] == "postshot-primary" else "2"
        ) * 64
        allocator_admission = {
            "schema_version": "reconstruction_gpu_canary_admission.v1",
            "status": "execute_ready",
            "blockers": [],
            "operation": "trainer_canary",
            "operation_request_digest": plan["canonical_3dgs_execution_plan_digest"],
            "operation_input_bundle_digest": transport["transport_bundle_digest"],
            "reconstruction_dataset_digest": plan["colmap_training_dataset_digest"],
            "frozen_split_digest": plan["frozen_split_digest"],
            "source_commit_sha": plan["source_commit_sha"],
            "worker_image_digest": worker_image,
            "max_spend_usd": 10.0,
            "hard_ttl_seconds": 3600,
            "retry_cap": 0,
            "authority_id": "fixture-authority",
            "watchdog_armed": True,
            "provider_zero_verified": True,
            "provider_mutations_performed": 0,
            "paid_execution_started": False,
            "execution_adapter_qualified": True,
        }
        allocator_admission["admission_digest"] = canonical_digest(
            allocator_admission, digest_field="admission_digest"
        )
        allocator_digests.append(allocator_admission["admission_digest"])
        admission = build_canonical_3dgs_worker_admission(
            transport_receipt=transport,
            arm_id=arm["arm_id"],
            worker_platform=(
                "windows" if arm["arm_id"] == "postshot-primary" else "linux"
            ),
            paid_allocator_admission=allocator_admission,
            worker_image_digest=worker_image,
            trainer_runtime_digest="sha256:" + "9" * 64,
            trainer_runtime_version="fixture-trainer-1.0",
            authority_id="fixture-authority",
            max_spend_usd=10.0,
            hard_ttl_seconds=3600,
            provider_upload_authorized=True,
            paid_compute_authorized=True,
            watchdog_armed=True,
            provider_zero_before_allocation=True,
            timestamp="2026-08-03T05:00:00Z",
        )
        receipt["runtime_identity"].update(
            {
                "trainer_runtime_digest": admission["trainer_runtime_digest"],
                "trainer_runtime_version": admission["trainer_runtime_version"],
            }
        )
        (run_root / "canonical_3dgs_transport_receipt.json").write_text(
            json.dumps(transport), encoding="utf-8"
        )
        (run_root / "canonical_3dgs_worker_admission.json").write_text(
            json.dumps(admission), encoding="utf-8"
        )
        receipt.update(
            {
                "transport_bundle_digest": transport["transport_bundle_digest"],
                "transport_receipt_digest": transport["receipt_digest"],
                "canonical_3dgs_worker_admission_digest": admission[
                    "canonical_3dgs_worker_admission_digest"
                ],
                "allocation_binding_digest": admission["allocation_binding_digest"],
                "provider_zero_required_after_execution": True,
                "transport_receipt_relative_path": "canonical_3dgs_transport_receipt.json",
                "worker_admission_relative_path": "canonical_3dgs_worker_admission.json",
            }
        )
        receipt["canonical_3dgs_worker_receipt_digest"] = canonical_digest(
            receipt, digest_field="canonical_3dgs_worker_receipt_digest"
        )
        (run_root / "worker_receipt.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )

    result = finalize_canonical_3dgs_receipts(
        plan=plan,
        dataset_root=dataset_root,
        results_root=results_root,
    )

    assert result["campaign"]["status"] == "candidates_ready_for_independent_evaluation"
    assert len(result["arm_results"]) == 2
    assert result["campaign"]["execution_control_summary"] == {
        "all_external_workers_admitted": True,
        "control_modes": ["external_worker_admission_bound"],
        "allocation_binding_digests": sorted(allocator_digests),
        "provider_zero_required_after_execution": True,
        "provider_zero_verified_after_execution": False,
        "resource_closeout_is_quality_evidence": False,
    }
    assert all(
        row["canonical_3dgs_execution_plan_digest"]
        == plan["canonical_3dgs_execution_plan_digest"]
        for row in result["arm_results"]
    )


def test_platform_finalizer_rejects_unadmitted_worker_receipts(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset = _dataset(dataset_root)
    plan = build_canonical_3dgs_execution_plan(
        preparation=_preparation(dataset),
        dataset=dataset,
        dataset_root=dataset_root,
        source_commit_sha=SOURCE_COMMIT,
    )
    for arm in plan["arms"]:
        run_root = tmp_path / "results" / arm["arm_id"]
        run_root.mkdir(parents=True)
        receipt = _runner(arm, dataset_root, run_root)
        receipt["canonical_3dgs_execution_plan_digest"] = plan[
            "canonical_3dgs_execution_plan_digest"
        ]
        (run_root / "worker_receipt.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )

    with pytest.raises(Canonical3DGSPipelineError, match="worker_receipt_digest_mismatch"):
        finalize_canonical_3dgs_receipts(
            plan=plan,
            dataset_root=dataset_root,
            results_root=tmp_path / "results",
        )
