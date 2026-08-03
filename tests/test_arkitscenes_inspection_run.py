from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import jsonschema

import blueprint_pipeline.arkitscenes_inspection_run as run_module
from blueprint_pipeline.arkitscenes_inspection_run import (
    ArkitScenesInspectionRunError,
    compile_arkitscenes_inspection_run,
    verify_retained_sources,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    root = tmp_path / "40958756"
    source = root / "source"
    source.mkdir(parents=True)
    original_files = []
    for name in (
        "40958756.mov",
        "confidence.zip",
        "lowres_depth.zip",
        "lowres_wide.traj",
        "lowres_wide.zip",
        "lowres_wide_intrinsics.zip",
    ):
        path = source / name
        path.write_bytes((name + "\n").encode())
        original_files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "digest": _sha(path),
            }
        )
    source_capture_digest = canonical_digest({"files": original_files})
    compilation_root = root / "compiled" / "arkitscenes_proxy_fixture"
    image = compilation_root / "candidate" / "sink.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (640, 480), (120, 120, 120)).save(image)

    floor_axis = np.linspace(-3.0, 3.0, 13)
    vertex_rows: list[dict] = []
    vertex_index: dict[tuple[int, int], str] = {}
    for row_index, z in enumerate(floor_axis):
        for column_index, x in enumerate(floor_axis):
            vertex_id = f"floor-{row_index}-{column_index}"
            vertex_index[(row_index, column_index)] = vertex_id
            vertex_rows.append({"vertex_id": vertex_id, "position_m": [float(x), 0.0, float(z)]})
    target_start = len(vertex_rows)
    for y in np.linspace(0.55, 0.85, 10):
        for x in np.linspace(-0.2, 0.2, 10):
            vertex_rows.append(
                {
                    "vertex_id": f"target-{len(vertex_rows) - target_start}",
                    "position_m": [float(x), float(y), 2.0],
                }
            )
    face_rows: list[dict] = []
    for row_index in range(len(floor_axis) - 1):
        for column_index in range(len(floor_axis) - 1):
            a = vertex_index[(row_index, column_index)]
            b = vertex_index[(row_index, column_index + 1)]
            c = vertex_index[(row_index + 1, column_index + 1)]
            d = vertex_index[(row_index + 1, column_index)]
            face_rows.extend(
                [
                    {"vertex_ids": [a, b, c]},
                    {"vertex_ids": [a, c, d]},
                ]
            )
    surface = {
        "schema_version": "arkit_observed_surface.v1",
        "source_capture_digest": source_capture_digest,
        "coordinate_frame_declaration": {
            "frame": "arkitscenes_official_loader_world",
            "units": "meters",
            "up_axis": "not_independently_validated",
            "handedness": "not_explicitly_declared_by_dataset",
            "gravity_aligned": False,
        },
        "metric_scale_status": "sensor_metric_unvalidated",
        "generated_fill_used": False,
        "vertices": vertex_rows,
        "faces": face_rows,
    }
    surface_path = compilation_root / "observed_surface_proxy_v1" / "arkit_observed_surface.json"
    _write_json(surface_path, surface)
    _write_json(
        compilation_root / "observed_surface_proxy_v1" / "arkit_depth_surface_proxy_result.json",
        {
            "schema_version": "arkit_depth_surface_compilation_result.v1",
            "surface_asset": {
                "relative_path": surface_path.relative_to(root).as_posix(),
                "digest": _sha(surface_path),
            },
        },
    )
    observation = {
        "observation_id": "arkitscenes-40958756-fixture-sink",
        "image_relative_path": image.relative_to(compilation_root).as_posix(),
        "image_digest": _sha(image),
        "camera": {
            "T_world_camera": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "rgb_intrinsics": {
                "width": 640,
                "height": 480,
                "fx": 500.0,
                "fy": 500.0,
            },
        },
    }
    camera_manifest = {
        "schema_version": "arkitscenes_camera_observations_proxy.v1",
        "observations": [observation],
    }
    camera_manifest["camera_observation_digest"] = canonical_digest(
        camera_manifest, digest_field="camera_observation_digest"
    )
    _write_json(compilation_root / "camera_observations_proxy.json", camera_manifest)
    compilation = {
        "schema_version": "arkitscenes_raw_proxy_compilation.v1",
        "status": "partial",
        "source_capture_identity": "arkitscenes-40958756",
        "source_capture_digest": source_capture_digest,
        "source_commit_sha": "a" * 40,
        "original_file_references": original_files,
        "authority_used": {
            "arkitscenes_license_accepted": True,
            "local_processing_authorized": True,
            "provider_upload_authorized": False,
            "paid_compute_authorized": False,
        },
        "coordinate_frame_declaration": surface["coordinate_frame_declaration"],
    }
    compilation["arkitscenes_proxy_compilation_digest"] = canonical_digest(
        compilation, digest_field="arkitscenes_proxy_compilation_digest"
    )
    compilation_path = compilation_root / "arkitscenes_raw_proxy_compilation.json"
    _write_json(compilation_path, compilation)
    return root, compilation_path, source_capture_digest


def _analyzer_contract() -> dict:
    value = {
        "analyzer_id": "fixture-visible-object-detector",
        "implementation_version": "1",
        "candidate_may_self_authorize": False,
    }
    value["analyzer_contract_digest"] = canonical_digest(
        value, digest_field="analyzer_contract_digest"
    )
    return value


def _analyzer(request, _runtime):
    return {
        "status": "completed",
        "analyzer_request_digest": request["analyzer_request_digest"],
        "candidate_may_self_authorize": False,
        "proposals": [
            {
                "proposal_id": "fixture-visible-sink",
                "object_label": "sink",
                "task_family": "franka_sink_inspection",
                "affordances": ["inspect"],
                "visual_confidence": 0.95,
                "supporting_view_ids": ["arkitscenes-40958756-fixture-sink"],
                "binding_view_id": "arkitscenes-40958756-fixture-sink",
                "bbox_xyxy_pixels": [250.0, 230.0, 390.0, 350.0],
            }
        ],
        "blockers": [],
    }


def test_source_verification_rehashes_every_recorded_file(tmp_path) -> None:
    root, compilation_path, _ = _fixture(tmp_path)

    _, verification = verify_retained_sources(
        scene_root=root, selected_compilation_path=compilation_path
    )

    assert verification["status"] == "verified"
    assert verification["all_recorded_source_files_match"] is True
    assert verification["retained_compilation_receipt_count"] == 1

    (root / "source" / "40958756.mov").write_bytes(b"drift")
    with pytest.raises(ArkitScenesInspectionRunError, match="digest_or_size_mismatch"):
        verify_retained_sources(scene_root=root, selected_compilation_path=compilation_path)


def test_compiles_bounded_five_controller_packet_without_claim_upgrade(
    tmp_path, monkeypatch
) -> None:
    root, compilation_path, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        run_module,
        "_runtime_available",
        lambda: {
            "host_platform": "test",
            "host_machine": "test",
            "isaac_sim_binary": None,
            "nvidia_smi_binary": None,
            "compatible_local_nvidia_isaac_runtime_available": False,
        },
    )
    output = tmp_path / "run"

    report = compile_arkitscenes_inspection_run(
        scene_root=root,
        selected_compilation_path=compilation_path,
        output_root=output,
        implementation_source_commit_sha="b" * 40,
        view_ids=["arkitscenes-40958756-fixture-sink"],
        allowed_object_labels=["sink"],
        analyzer_backend=_analyzer,
        analyzer_contract=_analyzer_contract(),
    )

    assert report["status"] == "abstained"
    assert report["source_class"] == "public-dataset proxy"
    assert report["evidence_class"] == ("scripted-controller evidence, not learned-policy evidence")
    assert report["scenario_scope"] == "single scenario"
    assert report["execution_scope"] == "simulation-only"
    assert report["task_evaluation_run_completed"] is False
    assert set(report["no_claims"]) == {
        "physical success",
        "deployment",
        "safety",
        "transfer",
    }
    packet = json.loads((output / "isaac_run_packet.json").read_text())
    assert packet["controller_count"] == 5
    assert [row["controller_id"] for row in packet["controller_identities"]] == [
        "franka-inspection-center-hold-v1",
        "franka-inspection-left-narrow-v1",
        "franka-inspection-right-narrow-v1",
        "franka-inspection-left-wide-v1",
        "franka-inspection-right-wide-v1",
    ]
    assert packet["implementation_source_commit_sha"] == "b" * 40
    assert (
        json.loads((output / "paid_runtime_authorization_request.json").read_text())[
            "exact_source_commit_sha"
        ]
        == "b" * 40
    )
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/arkitscenes_scripted_inspection_terminal_report.v1.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(report)
    assert (
        packet["claim_boundary"]["scripted_controller_evidence_not_learned_policy_evidence"] is True
    )
    target = json.loads((output / "target_orchestration.json").read_text())
    assert target["schema_version"] == "rendered_scene_task_target_orchestration.v1"
    assert target["task_zone_asset_requirement"]["verified_simready_asset_required"] is False
    profile = json.loads((output / "public_dataset_source_profile.json").read_text())
    assert profile["status"] == "admitted_provider_derived_support"
    assert profile["claim_boundary"]["blueprint_raw_contract_v3_2_proven"] is False
    assert profile["claim_boundary"]["iphone_route_proven"] is False
