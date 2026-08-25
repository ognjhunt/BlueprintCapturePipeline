from __future__ import annotations

import hashlib
import json
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_render_inputs import (
    materialize_scene_configuration_render_inputs,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_renders_derived_views_without_disclosing_raw_splat(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    splat = inputs / "scene.ply"
    splat.write_bytes(b"raw-interiorgs-source")
    source = {
        "artifacts": [
            {
                "role": "interiorgs_source_splat",
                "sha256": _sha256(splat),
                "size_bytes": splat.stat().st_size,
                "splat_count": 1024,
                "provider_upload_allowed": False,
            }
        ]
    }
    source_path = inputs / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    plan = {
        "schema_version": "task_evaluation_renderer_qualification_plan.v1",
        "status": "execute_during_scene_configuration_run",
        "appearance_source": "InteriorGS",
        "browser_preview_qualifies": False,
        "debug_sage_render_qualifies_as_appearance": False,
    }
    plan_path = inputs / "renderer-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    def reference(contract_path: str, path: Path) -> dict:
        return {
            "contract_path": contract_path,
            "materialized_path": str(path),
            "digest": _sha256(path),
            "size_bytes": path.stat().st_size,
            "full_byte_service_account_readback_passed": True,
        }

    envelope = {
        "run_id": "configure-scene-839873-v1",
        "request": {
            "scene": {
                "registration": {
                    "metric_registration": {
                        "digest": "sha256:" + "a" * 64
                    }
                }
            }
        },
        "materialized_references": [
            reference("scene.appearance.representation", splat),
            reference("scene.source_manifest", source_path),
            reference("scene.appearance.renderer_qualification", plan_path),
        ],
    }
    observed_splat: Path | None = None

    def render(**kwargs):
        nonlocal observed_splat
        observed_splat = Path(kwargs["splat_path"])
        output = Path(kwargs["output_dir"])
        frames = output / "frames"
        frames.mkdir(parents=True)
        rows = []
        for camera in kwargs["cameras"]:
            frame = frames / f"{camera['camera_id']}.png"
            frame.write_bytes((camera["camera_id"] + "-derived").encode())
            rows.append(
                {
                    "camera_id": camera["camera_id"],
                    "relative_path": f"frames/{frame.name}",
                    "digest": _sha256(frame),
                    "width": 1024,
                    "height": 1024,
                }
            )
        result = {
            "schema_version": "sealed_camera_render_manifest.v1",
            "status": "rendered_exact_cameras",
            "authorization_class": "method_input",
            "splat_digest": kwargs["source_splat_digest"],
            "renders": rows,
            "render_count": len(rows),
            "sealed_camera_render_manifest_digest": "",
        }
        result["sealed_camera_render_manifest_digest"] = canonical_digest(
            result, digest_field="sealed_camera_render_manifest_digest"
        )
        return result

    result = materialize_scene_configuration_render_inputs(
        envelope=envelope,
        stage_one_configuration={
            "schema_version": "observed_appearance_object_removal_configuration.v1",
            "production_render_required": True,
            "source_object": {
                "publisher_instance_id": "104",
                "aabb_min_xyz_m": [2.91, -6.83, 0.754],
                "aabb_max_xyz_m": [3.04, -6.69, 0.884],
            },
            "required_views": {"minimum": 8, "lossless_inputs": True},
            "provider_disclosure": {
                "raw_interiorgs_bytes": False,
                "derived_rendered_views": True,
            },
        },
        output_root=tmp_path / "output",
        renderer=render,
        runtime_resolver=lambda **_kwargs: {
            "node": "/runtime/node",
            "browser_executable": "/runtime/chrome",
            "renderer_root": "/runtime/renderer",
            "identity": {
                "mode": "immutable_host_runtime",
                "runtime_digest": "sha256:" + "d" * 64,
                "source_commit": "e" * 40,
                "full_byte_service_account_readback_passed": True,
            },
        },
    )

    assert observed_splat == splat
    assert result["derived_frame_count"] == 8
    assert result["raw_interiorgs_bytes_in_provider_packet"] is False
    assert result["provider_disclosure_scope"] == "derived_rendered_views_only"
    assert result["renderer_runtime"]["mode"] == "immutable_host_runtime"
    assert all(Path(row["path"]).read_bytes() != splat.read_bytes() for row in result["derived_frames"])
