from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.public_scene_aura_adapter import SCHEMA_VERSION as ADAPTER_SCHEMA
from blueprint_pipeline.public_scene_aura_execution import SCHEMA_VERSION as EXECUTION_SCHEMA
from blueprint_pipeline.public_scene_aura_native_render import (
    AuraNativeRenderManifestError,
    materialize_aura_native_render_manifest,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _fixture(
    tmp_path: Path,
    *,
    scene_id: str = "840313",
    target_id: str = "ins160",
) -> tuple[Path, Path, Path, Path]:
    evidence = tmp_path / "evidence"
    artifact_root = evidence / "run/immutable_execution"
    frames = artifact_root / "artifacts/final_frames"
    frames.mkdir(parents=True)
    runtime_path = artifact_root / "runtime.json"
    runtime_path.write_text("{}\n", encoding="utf-8")
    frame_rows = []
    for index in range(2):
        path = frames / f"{index:05d}.png"
        pixels = np.full((3, 4, 3), 40 + index * 100, dtype=np.uint8)
        pixels[0, 0] = [10, 20, 30]
        Image.fromarray(pixels).save(path)
        frame_rows.append(
            {
                "relative_path": f"artifacts/final_frames/{index:05d}.png",
                "size_bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
        )

    adapter = {
        "schema_version": ADAPTER_SCHEMA,
        "status": "prepared_unexecuted",
        "scene": {
            "publisher_scene_id": scene_id,
            "target_instance_id": target_id,
            "camera_count": 2,
            "source_resolution": [4, 3],
            "input_receipt_digest": "sha256:" + "1" * 64,
        },
        "artifacts": [
            {"relative_path": f"data/Other-360/{scene_id}_{target_id}/images/b.png"},
            {"relative_path": f"data/Other-360/{scene_id}_{target_id}/images/a.png"},
        ],
    }
    adapter["receipt_digest"] = canonical_digest(adapter, digest_field="receipt_digest")
    adapter_path = tmp_path / "adapter.json"
    _write_json(adapter_path, adapter)
    execution = {
        "schema_version": EXECUTION_SCHEMA,
        "status": "executed_candidate",
        "prepared_adapter": {
            "receipt_digest": adapter["receipt_digest"],
            "sha256": _sha(adapter_path),
            "size_bytes": adapter_path.stat().st_size,
        },
        "scene": adapter["scene"],
        "source": {
            "repository": "https://example.invalid/Aura",
            "commit": "a" * 40,
            "tree": "b" * 40,
        },
        "execution": {
            "runtime_result": {
                "path": str(runtime_path),
                "size_bytes": runtime_path.stat().st_size,
                "sha256": _sha(runtime_path),
            },
            "final_point_cloud": {"sha256": "sha256:" + "2" * 64},
            "final_frames": frame_rows,
        },
    }
    execution["receipt_digest"] = canonical_digest(
        execution, digest_field="receipt_digest"
    )
    execution_path = tmp_path / "execution.json"
    _write_json(execution_path, execution)
    output = artifact_root / "aura_native_manifest.json"
    return adapter_path, execution_path, evidence, output


@pytest.mark.parametrize(
    ("scene_id", "target_id"), [("840313", "ins160"), ("840796", "ins123")]
)
def test_materializes_actual_native_frames_in_sorted_camera_order(
    tmp_path: Path, scene_id: str, target_id: str
) -> None:
    adapter, execution, evidence, output = _fixture(
        tmp_path, scene_id=scene_id, target_id=target_id
    )
    result = materialize_aura_native_render_manifest(
        adapter_receipt_path=adapter,
        execution_receipt_path=execution,
        evidence_root=evidence,
        output_path=output,
    )
    assert result["status"] == "rendered_exact_cameras"
    assert result["splat_representation"] == "2d_gaussian_surfels_scale_0_scale_1"
    assert [row["camera_id"] for row in result["renders"]] == ["a", "b"]
    assert result["renderer_identity"]["renderer_independent_of_method"] is False
    assert result["scene"] == {
        "publisher_scene_id": scene_id,
        "target_instance_id": target_id,
    }
    assert result["camera_set_label"] == f"adp009b_{scene_id}_{target_id}_frozen_2"
    assert json.loads(output.read_text()) == result


def test_rejects_changed_native_frame(tmp_path: Path) -> None:
    adapter, execution, evidence, output = _fixture(tmp_path)
    frame = evidence / "run/immutable_execution/artifacts/final_frames/00000.png"
    Image.new("RGB", (4, 3), "red").save(frame)
    with pytest.raises(AuraNativeRenderManifestError) as raised:
        materialize_aura_native_render_manifest(
            adapter_receipt_path=adapter,
            execution_receipt_path=execution,
            evidence_root=evidence,
            output_path=output,
        )
    assert "aura_native_frame_changed:a" in raised.value.codes


def test_rejects_caller_reordered_frame_records(tmp_path: Path) -> None:
    adapter, execution_path, evidence, output = _fixture(tmp_path)
    execution = json.loads(execution_path.read_text())
    execution["execution"]["final_frames"].reverse()
    execution["receipt_digest"] = canonical_digest(
        execution, digest_field="receipt_digest"
    )
    _write_json(execution_path, execution)
    with pytest.raises(AuraNativeRenderManifestError) as raised:
        materialize_aura_native_render_manifest(
            adapter_receipt_path=adapter,
            execution_receipt_path=execution_path,
            evidence_root=evidence,
            output_path=output,
        )
    assert "aura_native_frame_order_invalid" in raised.value.codes
