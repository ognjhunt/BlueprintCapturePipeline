"""Integration-style tests for NuRec-first swap orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

import pytest

import blueprint_pipeline.swap_orchestrator as swap_orchestrator_module
from blueprint_pipeline.blueprintpipeline_runner import CommandResult
from blueprint_pipeline.quality_gates import AdvancedQualityGateConfig
from blueprint_pipeline.swap_orchestrator import OrchestratorConfig, run_swap_pipeline
from functions.storage_trigger import parse_descriptor_path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _write_scene_descriptor(root: Path) -> str:
    scene_id = "scene_demo"
    capture_id = "capture_demo"

    descriptor_uri = f"gs://bucket/scenes/{scene_id}/captures/{capture_id}/capture_descriptor.json"
    descriptor_path = root / "bucket/scenes" / scene_id / "captures" / capture_id / "capture_descriptor.json"

    raw_prefix_uri = f"gs://bucket/scenes/{scene_id}/iphone/{capture_id}/raw"
    qa_uri = f"gs://bucket/scenes/{scene_id}/captures/{capture_id}/qa_report.json"

    descriptor = {
        "schema_version": "v1",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "capture_source": "iphone",
        "capture_tier": "tier1_iphone",
        "raw_prefix_uri": raw_prefix_uri,
        "frames_index_uri": f"gs://bucket/scenes/{scene_id}/captures/{capture_id}/frames/index.jsonl",
        "keyframe_uri": f"gs://bucket/scenes/{scene_id}/images/{capture_id}_keyframe.jpg",
        "nurec_mode": "mono_pose_assisted",
        "quality": {"pose_match_rate": 0.95},
        "swap_focus": ["kitchen"],
        "intended_space_type": "kitchen",
        "qa_report_uri": qa_uri,
    }
    _write_json(descriptor_path, descriptor)

    qa_report = {
        "schema_version": "v1",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "status": "passed",
    }
    _write_json(root / "bucket/scenes" / scene_id / "captures" / capture_id / "qa_report.json", qa_report)

    manifest = {
        "scene_id": scene_id,
        "capture_source": "iphone",
        "capture_tier_hint": "tier1_iphone",
        "object_point_cloud_index": "arkit/objects/index.json",
        "object_point_cloud_count": 2,
    }
    _write_json(root / "bucket/scenes" / scene_id / "iphone" / capture_id / "raw/manifest.json", manifest)

    object_index = [
        {
            "id": "drawer_1",
            "label": "kitchen drawer",
            "pointCloudFile": "drawer_1.ply",
            "boundingBox": {
                "center": [0.6, 0.5, 1.2],
                "extents": [0.8, 0.5, 0.6],
                "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "orientationQuaternion": [1, 0, 0, 0],
            },
        },
        {
            "id": "tote_1",
            "label": "warehouse tote",
            "pointCloudFile": "tote_1.ply",
            "boundingBox": {
                "center": [1.5, 0.3, 2.0],
                "extents": [0.6, 0.4, 0.5],
                "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "orientationQuaternion": [1, 0, 0, 0],
            },
        },
    ]
    _write_json(
        root / "bucket/scenes" / scene_id / "iphone" / capture_id / "raw/arkit/objects/index.json",
        {"objects": object_index},
    )

    return descriptor_uri


class _StubNurecClient:
    def __init__(self, root: Path, bucket: str, pipeline_prefix: str) -> None:
        self.root = root
        self.bucket = bucket
        self.pipeline_prefix = pipeline_prefix

    def run(self, *, descriptor, descriptor_uri: str, object_index_uri: str):  # noqa: ANN001
        pipeline_dir = self.root / self.pipeline_prefix
        nurec_dir = pipeline_dir / "nurec"
        nurec_dir.mkdir(parents=True, exist_ok=True)

        (nurec_dir / "export_last.usdz").write_bytes(b"usdz")
        (nurec_dir / "occupancy.bin").write_bytes(b"occ")
        (nurec_dir / "visual_mesh.glb").write_bytes(b"glb")
        (nurec_dir / "visual_pointcloud.ply").write_bytes(b"ply")
        (nurec_dir / "mesh_manifest.json").write_text("{}", encoding="utf-8")

        # Minimal placeholder PLY file (conversion is monkeypatched in tests).
        (nurec_dir / "nvblox_mesh.ply").write_text(
            """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
0 1 0
3 0 1 2
""",
            encoding="utf-8",
        )

        outputs = {
            "schema_version": "v1",
            "scene_prefix": self.pipeline_prefix,
            "status": "completed",
            "artifacts": {
                "visual_usdz": f"gs://{self.bucket}/{self.pipeline_prefix}/nurec/export_last.usdz",
                "visual_mesh_glb": f"gs://{self.bucket}/{self.pipeline_prefix}/nurec/visual_mesh.glb",
                "visual_pointcloud_ply": f"gs://{self.bucket}/{self.pipeline_prefix}/nurec/visual_pointcloud.ply",
                "mesh_manifest_json": f"gs://{self.bucket}/{self.pipeline_prefix}/nurec/mesh_manifest.json",
                "collision_mesh_ply": f"gs://{self.bucket}/{self.pipeline_prefix}/nurec/nvblox_mesh.ply",
                "occupancy": [f"gs://{self.bucket}/{self.pipeline_prefix}/nurec/occupancy.bin"],
            },
            "mesh_stats": {
                "bounds": {
                    "min": [0.0, 0.0, 0.0],
                    "max": [3.0, 2.5, 4.0],
                }
            },
        }
        _write_json(pipeline_dir / "nurec_outputs.json", outputs)
        return outputs


class _StubBlueprintRunner:
    def __init__(
        self,
        root: Path,
        *,
        fail_required_ids: Optional[Set[str]] = None,
        retrieval_resolves: bool = True,
        simready_code: int = 0,
        usd_code: int = 0,
    ) -> None:
        self.root = root
        self.fail_required_ids = fail_required_ids or set()
        self.retrieval_resolves = retrieval_resolves
        self.simready_code = simready_code
        self.usd_code = usd_code

        self.materialize_calls: List[Dict[str, Any]] = []
        self.interactive_calls: List[Dict[str, Any]] = []
        self.simready_calls: List[Dict[str, Any]] = []
        self.usd_calls: List[Dict[str, Any]] = []

    def ensure_expected_commit(self, expected_commit_hash: str) -> None:
        self.expected_commit_hash = expected_commit_hash

    def materialize_text_assets(
        self,
        *,
        scene_id: str,
        assets_prefix: str,
        objects: List[Dict[str, Any]],
        room_type: str,
        generation_enabled: bool,
        retrieval_enabled: bool,
        retrieval_mode: str,
        generation_provider_chain: str,
    ) -> Dict[str, Any]:
        self.materialize_calls.append(
            {
                "scene_id": scene_id,
                "assets_prefix": assets_prefix,
                "objects": objects,
                "room_type": room_type,
                "generation_enabled": generation_enabled,
                "retrieval_enabled": retrieval_enabled,
                "retrieval_mode": retrieval_mode,
                "generation_provider_chain": generation_provider_chain,
            }
        )

        provenance = []
        for obj in objects:
            asset_id = str(obj["id"])
            asset_dir = self.root / assets_prefix / asset_id
            asset_dir.mkdir(parents=True, exist_ok=True)
            (asset_dir / "model.usd").write_text("#usda 1.0", encoding="utf-8")
            (asset_dir / "metadata.json").write_text("{}", encoding="utf-8")
            (asset_dir / "generated.glb").write_bytes(b"glb")

            if retrieval_enabled:
                if self.retrieval_resolves:
                    (asset_dir / "joint.usda").write_text(
                        'def PhysicsRevoluteJoint "joint_0" {}',
                        encoding="utf-8",
                    )
                else:
                    (asset_dir / "joint.usda").write_text("# no joints", encoding="utf-8")

            provenance.append(
                {
                    "object_id": asset_id,
                    "path": f"{assets_prefix}/{asset_id}/model.usd",
                    "materialization": "generated_sam3d" if generation_enabled else "retrieved_asset_bundle",
                }
            )

        return {
            "provenance_assets": provenance,
            "retrieval_report": {"method_counts": {"generated": len(objects)}},
        }

    def run_interactive_job(
        self,
        *,
        scene_id: str,
        assets_prefix: str,
        regen3d_prefix: str,
        extra_env: Optional[Mapping[str, str]] = None,
    ) -> CommandResult:
        self.interactive_calls.append(
            {
                "scene_id": scene_id,
                "assets_prefix": assets_prefix,
                "regen3d_prefix": regen3d_prefix,
                "extra_env": dict(extra_env or {}),
            }
        )

        manifest_path = self.root / assets_prefix / "scene_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        interactive_roles = {
            "manipulable_object",
            "deformable_object",
            "articulated_furniture",
            "articulated_appliance",
        }

        objects = []
        for obj in manifest.get("objects", []):
            if obj.get("sim_role") not in interactive_roles:
                continue
            object_id = str(obj.get("id"))
            required = bool((obj.get("articulation") or {}).get("required", False))
            failed = required and object_id in self.fail_required_ids
            objects.append(
                {
                    "id": object_id,
                    "status": "error" if failed else "ok",
                    "backend": "stub_interactive",
                    "required_articulation": required,
                    "is_articulated": False if failed else required,
                    "joint_count": 0 if failed else (1 if required else 0),
                }
            )

        payload = {
            "scene_id": scene_id,
            "objects": objects,
            "total_objects": len(objects),
            "ok_count": sum(1 for item in objects if item["status"] == "ok"),
            "error_count": sum(1 for item in objects if item["status"] == "error"),
            "fallback_count": 0,
            "articulated_count": sum(1 for item in objects if item.get("is_articulated")),
        }

        results_path = self.root / assets_prefix / "interactive" / "interactive_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        has_required_failures = any(
            item["required_articulation"] and not item["is_articulated"] for item in objects
        )
        return CommandResult(
            return_code=1 if has_required_failures else 0,
            stdout="",
            stderr="",
            command=["python", "interactive-job/run_interactive_assets.py"],
        )

    def run_simready_job(
        self,
        *,
        scene_id: str,
        assets_prefix: str,
        layout_prefix: str,
        usd_prefix: str,
    ) -> CommandResult:
        self.simready_calls.append(
            {
                "scene_id": scene_id,
                "assets_prefix": assets_prefix,
                "layout_prefix": layout_prefix,
                "usd_prefix": usd_prefix,
            }
        )
        return CommandResult(
            return_code=self.simready_code,
            stdout="",
            stderr="",
            command=["python", "simready-job/prepare_simready_assets.py"],
        )

    def run_usd_assembly_job(
        self,
        *,
        scene_id: str,
        assets_prefix: str,
        layout_prefix: str,
        usd_prefix: str,
    ) -> CommandResult:
        self.usd_calls.append(
            {
                "scene_id": scene_id,
                "assets_prefix": assets_prefix,
                "layout_prefix": layout_prefix,
                "usd_prefix": usd_prefix,
            }
        )
        if self.usd_code == 0:
            usd_path = self.root / usd_prefix / "scene.usda"
            usd_path.parent.mkdir(parents=True, exist_ok=True)
            usd_path.write_text("#usda 1.0", encoding="utf-8")
        return CommandResult(
            return_code=self.usd_code,
            stdout="",
            stderr="",
            command=["python", "usd-assembly-job/assemble_scene.py"],
        )


@pytest.fixture(autouse=True)
def _patch_mesh_conversion(monkeypatch):
    from blueprint_pipeline import sam3d_assets

    def _fake_ply_to_glb(ply_path: Path, glb_path: Path) -> None:
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        glb_path.write_bytes(ply_path.read_bytes())

    monkeypatch.setattr(sam3d_assets, "_ply_to_glb", _fake_ply_to_glb)
    monkeypatch.setattr(
        sam3d_assets,
        "_prune_scene_shell_mesh",
        lambda glb_path, swap_candidates: {"enabled": True, "faces_removed": 0},
    )
    monkeypatch.setattr(
        sam3d_assets,
        "_simplify_scene_shell_mesh",
        lambda glb_path, max_faces: {
            "enabled": False,
            "reason": "test_stub",
            "before_faces": 0,
            "after_faces": 0,
            "budget_faces": max_faces,
        },
    )


@pytest.fixture(autouse=True)
def _force_adapter_stage_d_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAGE_D_MATERIALIZATION_MODE", "adapter")


def _run_pipeline(
    tmp_path: Path,
    *,
    fail_required_ids: Optional[Set[str]] = None,
    retrieval_resolves: bool = True,
) -> tuple[Dict[str, Any], _StubBlueprintRunner]:
    descriptor_uri = _write_scene_descriptor(tmp_path)

    parsed = parse_descriptor_path("scenes/scene_demo/captures/capture_demo/capture_descriptor.json")
    assert parsed is not None

    pipeline_prefix = "scenes/scene_demo/captures/capture_demo/pipeline"
    nurec_client = _StubNurecClient(tmp_path / "bucket", "bucket", pipeline_prefix)
    runner = _StubBlueprintRunner(
        tmp_path / "bucket",
        fail_required_ids=fail_required_ids,
        retrieval_resolves=retrieval_resolves,
    )

    result = run_swap_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        config=OrchestratorConfig(
            gcs_root=tmp_path,
            blueprintpipeline_root=Path("/unused"),
            expected_blueprintpipeline_commit="",
            fail_on_commit_mismatch=False,
            runtime_preflight_enabled=False,
            advanced_quality_config=AdvancedQualityGateConfig(enabled=False),
        ),
        nurec_client=nurec_client,
        blueprint_runner=runner,
    )
    return result, runner


def test_interactive_success_path_no_fallback(tmp_path: Path) -> None:
    result, runner = _run_pipeline(tmp_path, fail_required_ids=set(), retrieval_resolves=True)

    assert result["status"] == "completed"
    assert len(runner.materialize_calls) == 1
    assert runner.materialize_calls[0]["generation_enabled"] is True
    assert runner.materialize_calls[0]["retrieval_enabled"] is False

    complete_marker = (
        tmp_path
        / "bucket/scenes/scene_demo/captures/capture_demo/pipeline/.swap_pipeline_complete"
    )
    assert complete_marker.is_file()


def test_interactive_failure_then_retrieval_recovery(tmp_path: Path) -> None:
    result, runner = _run_pipeline(tmp_path, fail_required_ids={"drawer_1"}, retrieval_resolves=True)

    assert result["status"] == "completed"
    assert len(runner.materialize_calls) == 2
    assert runner.materialize_calls[1]["generation_enabled"] is False
    assert runner.materialize_calls[1]["retrieval_enabled"] is True
    assert runner.materialize_calls[1]["retrieval_mode"] == "ann_primary"

    interactive_results = json.loads(
        (
            tmp_path
            / "bucket/scenes/scene_demo/assets/interactive/interactive_results.json"
        ).read_text(encoding="utf-8")
    )
    drawer = next(item for item in interactive_results["objects"] if item["id"] == "drawer_1")
    assert drawer["is_articulated"] is True
    assert drawer["backend"] == "retrieval_fallback"


def test_interactive_failed_marker_without_results_triggers_fallback(tmp_path: Path) -> None:
    descriptor_uri = _write_scene_descriptor(tmp_path)
    nurec_client = _StubNurecClient(
        tmp_path / "bucket",
        "bucket",
        "scenes/scene_demo/captures/capture_demo/pipeline",
    )

    class _NoResultsRunner(_StubBlueprintRunner):
        def run_interactive_job(
            self,
            *,
            scene_id: str,
            assets_prefix: str,
            regen3d_prefix: str,
            extra_env: Optional[Mapping[str, str]] = None,
        ) -> CommandResult:
            self.interactive_calls.append(
                {
                    "scene_id": scene_id,
                    "assets_prefix": assets_prefix,
                    "regen3d_prefix": regen3d_prefix,
                    "extra_env": dict(extra_env or {}),
                }
            )

            failure_path = self.root / assets_prefix / ".interactive_failed"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(
                json.dumps(
                    {
                        "scene_id": scene_id,
                        "status": "failed",
                        "reason": "required_articulation_unmet",
                        "details": {"required_objects": ["drawer_1"]},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            return CommandResult(
                return_code=1,
                stdout="",
                stderr="",
                command=["python", "interactive-job/run_interactive_assets.py"],
            )

    runner = _NoResultsRunner(
        tmp_path / "bucket",
        fail_required_ids={"drawer_1"},
        retrieval_resolves=True,
    )

    result = run_swap_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        config=OrchestratorConfig(
            gcs_root=tmp_path,
            blueprintpipeline_root=Path("/unused"),
            expected_blueprintpipeline_commit="",
            fail_on_commit_mismatch=False,
            runtime_preflight_enabled=False,
            advanced_quality_config=AdvancedQualityGateConfig(enabled=False),
        ),
        nurec_client=nurec_client,
        blueprint_runner=runner,
    )

    assert result["status"] == "completed"
    assert len(runner.materialize_calls) == 2
    interactive_results = json.loads(
        (
            tmp_path
            / "bucket/scenes/scene_demo/assets/interactive/interactive_results.json"
        ).read_text(encoding="utf-8")
    )
    drawer = next(item for item in interactive_results["objects"] if item["id"] == "drawer_1")
    assert drawer["is_articulated"] is True
    assert drawer["backend"] == "retrieval_fallback"


def test_hard_fail_when_retrieval_unresolved(tmp_path: Path) -> None:
    descriptor_uri = _write_scene_descriptor(tmp_path)
    nurec_client = _StubNurecClient(
        tmp_path / "bucket",
        "bucket",
        "scenes/scene_demo/captures/capture_demo/pipeline",
    )
    runner = _StubBlueprintRunner(
        tmp_path / "bucket",
        fail_required_ids={"drawer_1"},
        retrieval_resolves=False,
    )

    with pytest.raises(Exception):
        run_swap_pipeline(
            descriptor_gcs_uri=descriptor_uri,
            config=OrchestratorConfig(
                gcs_root=tmp_path,
                blueprintpipeline_root=Path("/unused"),
                expected_blueprintpipeline_commit="",
                fail_on_commit_mismatch=False,
                runtime_preflight_enabled=False,
                advanced_quality_config=AdvancedQualityGateConfig(enabled=False),
            ),
            nurec_client=nurec_client,
            blueprint_runner=runner,
        )

    failed_marker = (
        tmp_path
        / "bucket/scenes/scene_demo/captures/capture_demo/pipeline/.swap_pipeline_failed.json"
    )
    assert failed_marker.is_file()


def test_missing_descriptor_writes_failure_artifacts(tmp_path: Path) -> None:
    descriptor_uri = "gs://bucket/scenes/scene_demo/captures/capture_demo/capture_descriptor.json"

    with pytest.raises(Exception):
        run_swap_pipeline(
            descriptor_gcs_uri=descriptor_uri,
            config=OrchestratorConfig(
                gcs_root=tmp_path,
                blueprintpipeline_root=Path("/unused"),
                expected_blueprintpipeline_commit="",
                fail_on_commit_mismatch=False,
                runtime_preflight_enabled=False,
                advanced_quality_config=AdvancedQualityGateConfig(enabled=False),
            ),
        )

    pipeline_dir_bucket = tmp_path / "bucket/scenes/scene_demo/captures/capture_demo/pipeline"
    pipeline_dir_flat = tmp_path / "scenes/scene_demo/captures/capture_demo/pipeline"
    pipeline_dir = pipeline_dir_bucket if pipeline_dir_bucket.is_dir() else pipeline_dir_flat
    failed_payload = json.loads((pipeline_dir / ".swap_pipeline_failed.json").read_text(encoding="utf-8"))
    quality_payload = json.loads((pipeline_dir / "swap_quality_report.json").read_text(encoding="utf-8"))
    assert failed_payload["status"] == "failed"
    assert quality_payload["status"] == "failed"
    assert quality_payload["failed_stage"] == "intake"


def test_swap_candidates_file_matches_post_mutation_payload(tmp_path: Path) -> None:
    descriptor_uri = _write_scene_descriptor(tmp_path)
    object_index_path = (
        tmp_path / "bucket/scenes/scene_demo/iphone/capture_demo/raw/arkit/objects/index.json"
    )
    object_index = json.loads(object_index_path.read_text(encoding="utf-8"))
    objects = object_index["objects"]
    for item in objects:
        if item.get("id") == "drawer_1":
            item["reference_crop"] = "/tmp/crop_a.png"
            item["all_crops"] = ["/tmp/crop_a.png", "/tmp/crop_b.png"]
    object_index_path.write_text(json.dumps(object_index, indent=2), encoding="utf-8")

    nurec_client = _StubNurecClient(
        tmp_path / "bucket",
        "bucket",
        "scenes/scene_demo/captures/capture_demo/pipeline",
    )
    runner = _StubBlueprintRunner(tmp_path / "bucket")
    result = run_swap_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        config=OrchestratorConfig(
            gcs_root=tmp_path,
            blueprintpipeline_root=Path("/unused"),
            expected_blueprintpipeline_commit="",
            fail_on_commit_mismatch=False,
            runtime_preflight_enabled=False,
            image_conditioned_generation=False,
            advanced_quality_config=AdvancedQualityGateConfig(enabled=False),
        ),
        nurec_client=nurec_client,
        blueprint_runner=runner,
    )
    assert result["status"] == "completed"

    swap_candidates_payload = json.loads(
        (
            tmp_path
            / "bucket/scenes/scene_demo/captures/capture_demo/pipeline/swap_candidates.json"
        ).read_text(encoding="utf-8")
    )
    drawer_candidate = next(
        item for item in swap_candidates_payload.get("candidates", []) if item.get("object_id") == "drawer_1"
    )
    assert "reference_crop" not in drawer_candidate
    assert "all_crops" not in drawer_candidate


def test_unmapped_required_ids_hard_fail_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        swap_orchestrator_module,
        "find_required_articulation_failures",
        lambda **_: ["ghost_required_id"],
    )

    with pytest.raises(Exception):
        _run_pipeline(tmp_path, fail_required_ids=set(), retrieval_resolves=True)

    fallback_report = json.loads(
        (
            tmp_path
            / "bucket/scenes/scene_demo/captures/capture_demo/pipeline/retrieval_fallback_report.json"
        ).read_text(encoding="utf-8")
    )
    assert fallback_report["unresolved_ids"] == ["ghost_required_id"]
    assert fallback_report["mapping_error"] == "failed required articulation IDs missing from swap_candidates"

    failed_marker = (
        tmp_path
        / "bucket/scenes/scene_demo/captures/capture_demo/pipeline/.swap_pipeline_failed.json"
    )
    assert failed_marker.is_file()


def test_downstream_job_invocations_receive_expected_prefixes(tmp_path: Path) -> None:
    _, runner = _run_pipeline(tmp_path, fail_required_ids=set(), retrieval_resolves=True)

    assert runner.simready_calls
    assert runner.usd_calls

    simready_call = runner.simready_calls[0]
    usd_call = runner.usd_calls[0]

    assert simready_call["assets_prefix"] == "scenes/scene_demo/assets"
    assert simready_call["layout_prefix"] == "scenes/scene_demo/layout"
    assert usd_call["usd_prefix"] == "scenes/scene_demo/usd"


def test_end_to_end_dry_outputs_written(tmp_path: Path) -> None:
    _run_pipeline(tmp_path, fail_required_ids=set(), retrieval_resolves=True)

    pipeline_dir = tmp_path / "bucket/scenes/scene_demo/captures/capture_demo/pipeline"
    assert (pipeline_dir / "nurec_outputs.json").is_file()
    assert (pipeline_dir / "swap_candidates.json").is_file()
    assert (pipeline_dir / "swap_execution_report.json").is_file()
    assert (pipeline_dir / "runtime_preflight_report.json").is_file()
    assert (pipeline_dir / "advanced_quality_report.json").is_file()
    assert (pipeline_dir / "swap_quality_report.json").is_file()
    assert (pipeline_dir / ".swap_pipeline_complete").is_file()


def test_storage_trigger_path_parser() -> None:
    parsed = parse_descriptor_path("scenes/abc/captures/def/capture_descriptor.json")
    assert parsed == {
        "scene_id": "abc",
        "capture_id": "def",
        "object_name": "scenes/abc/captures/def/capture_descriptor.json",
    }

    assert parse_descriptor_path("scenes/abc/raw/manifest.json") is None
