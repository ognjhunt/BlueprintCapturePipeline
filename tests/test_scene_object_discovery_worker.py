from __future__ import annotations

import copy
import json
from pathlib import Path

from blueprint_pipeline.scene_object_discovery_queue import (
    scene_object_discovery_status,
    stage_scene_object_discovery_request,
)
from blueprint_pipeline.scene_object_discovery_worker import (
    process_scene_object_discovery_queue,
)
from tests.test_scene_object_discovery_contract import request


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _materializer(tmp_path: Path):
    def materialize(**kwargs):
        value = kwargs["request"]
        root = tmp_path / value["discovery_id"]
        root.mkdir(parents=True, exist_ok=True)
        source = root / "source.ply"
        source.write_bytes(b"sealed source")
        scene_analysis = root / "scene-analysis.json"
        scene_analysis.write_text(
            json.dumps(
                {
                    "scene_geometry": {
                        "aabb_min": [-2.0, -1.0, 0.0],
                        "aabb_max": [2.0, 3.0, 2.5],
                        "up_axis": 2,
                        "up_sign": 1.0,
                        "unseen_regions": ["behind_partition"],
                    },
                    "publisher_objects": [
                        {
                            "publisher_instance_id": "tote-17",
                            "label": "red tote",
                            "confidence": 0.99,
                            "metric_validated": True,
                            "bounds_min": [0.1, 0.2, 0.3],
                            "bounds_max": [0.8, 0.9, 1.0],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        renderer = root / "renderer.json"
        renderer.write_text(
            json.dumps(
                {
                    "production_method_input_authorized": True,
                    "source_splat_digest": value["scene"]["source_splat"]["digest"],
                }
            ),
            encoding="utf-8",
        )
        dummy = root / "dummy.json"
        dummy.write_text("{}", encoding="utf-8")
        paths = {
            "scene.source_splat": source,
            "scene.scene_analysis": scene_analysis,
            "scene.metric_registration": dummy,
            "scene.renderer_qualification": renderer,
            "rights.admission": dummy,
            "rights.human_authority_record": dummy,
        }
        return {
            "references": [
                {
                    "contract_path": contract_path,
                    "materialized_path": str(path),
                    "full_byte_service_account_readback_passed": True,
                }
                for contract_path, path in paths.items()
            ],
            "unique_object_count": len(paths),
            "full_byte_service_account_readback_passed": True,
        }

    return materialize


def _renderer(**kwargs):
    plan = kwargs["camera_plan"]
    return {
        "render_binding": {
            "source_splat_digest": plan["source_splat_digest"],
            "camera_plan_digest": plan["camera_plan_digest"],
            "render_manifest_digest": _digest("9"),
        },
        "renderer_capabilities": {
            "rgb": True,
            "depth": False,
            "per_gaussian_contributions": False,
        },
    }


def _publisher(**kwargs):
    return {
        "uri": "s3://blueprint/task-evaluation/production-inputs/" + kwargs["relative_name"],
        "digest": _digest("8"),
        "size_bytes": 100,
    }


def test_worker_auto_selects_and_publishes_unique_publisher_metric_object(
    tmp_path: Path,
) -> None:
    value = request()
    value["analysis"]["analyzers"] = ["publisher_semantics"]
    receipt = stage_scene_object_discovery_request(
        value=value, queue_root=tmp_path / "queue", submitted_by="webapp"
    )

    run = process_scene_object_discovery_queue(
        queue_root=tmp_path / "queue",
        input_root=tmp_path / "inputs",
        output_root=tmp_path / "outputs",
        allowed_uri_prefixes=["https://objects.example/"],
        service_account="unused-by-fake",
        publication_prefix="s3://blueprint/task-evaluation/production-inputs/discovery/",
        source_commit="a" * 40,
        input_materializer=_materializer(tmp_path / "materialized"),
        render_materializer=_renderer,
        artifact_publisher=_publisher,
    )

    assert run["processed_count"] == 1
    assert run["provider_mutation_performed"] is False
    status = scene_object_discovery_status(
        discovery_id=value["discovery_id"], queue_root=tmp_path / "queue"
    )
    assert status["status"] == "ready_auto_selected"
    assert status["source_commit"] == "a" * 40
    assert status["source_object"]["label"] == "red tote"
    assert status["source_object"]["source_object_artifact"]["digest"] == _digest("8")
    assert status["unseen_regions"] == ["behind_partition"]
    assert receipt["paid_execution_requested"] is False


def test_worker_refuses_provider_execution_without_activation(tmp_path: Path) -> None:
    value = copy.deepcopy(request())
    value["rights"]["source_bytes_redistributable"] = True
    value["rights"]["provider_disclosure_scope"] = "source_and_derived"
    value["execution"] = {
        "mode": "provider_gpu_after_activation",
        "selected_provider": "vast",
    }
    stage_scene_object_discovery_request(
        value=value, queue_root=tmp_path / "queue", submitted_by="webapp"
    )

    run = process_scene_object_discovery_queue(
        queue_root=tmp_path / "queue",
        input_root=tmp_path / "inputs",
        output_root=tmp_path / "outputs",
        allowed_uri_prefixes=["https://objects.example/"],
        service_account="unused",
        publication_prefix="s3://blueprint/task-evaluation/production-inputs/discovery/",
        source_commit="a" * 40,
        input_materializer=_materializer(tmp_path / "materialized"),
        render_materializer=_renderer,
        artifact_publisher=_publisher,
    )

    assert run["results"][0]["blockers"] == ["scene_object_discovery_provider_activation_required"]
    assert run["paid_execution_requested"] is False
    status = scene_object_discovery_status(
        discovery_id=value["discovery_id"], queue_root=tmp_path / "queue"
    )
    assert status["status"] == "blocked"
    assert status["blockers"] == ["scene_object_discovery_provider_activation_required"]
