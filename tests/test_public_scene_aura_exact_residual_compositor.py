from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.public_scene_aura_exact_residual_compositor import (
    AuraExactResidualCompositeError,
    SCHEMA_VERSION,
    materialize_aura_exact_residual_composite,
)
from blueprint_pipeline.public_scene_aura_exact_residual_preflight import (
    materialize_aura_exact_residual_preflight,
)
from test_public_scene_aura_exact_residual_preflight import _packet


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _preflight(tmp_path: Path) -> Path:
    packet = _packet(tmp_path)
    output = tmp_path / "preflight.json"
    materialize_aura_exact_residual_preflight(
        input_packet_path=packet, output_path=output
    )
    return output


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _provider_closeout(root: Path) -> dict[str, object]:
    """Build four distinct file-backed provider-zero receipts for a fixture."""

    adapter = {
        "api_call_performed": True,
        "provider_create_attempted": True,
        "final_validation_status": "passed",
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "estimated_cost_usd": 0.25,
        "hard_ttl_seconds": 900,
    }
    teardown = {
        "schema_version": "vast_teardown_manifest.v1",
        "status": "completed",
        "continuing_spend_from_this_run": False,
        "runner_gpu_teardown_completed": True,
        "vast_instance_ids": [123],
        "teardown_actions_performed": [
            {
                "instance_id": 123,
                "action": "destroy_instance",
                "status": "completed",
                "http_status_code": 200,
            }
        ],
    }
    final = {
        "schema_version": "vast_final_validation.v1",
        "status": "passed",
        "all_vast_instances_destroyed_by_adapter": True,
        "continuing_spend_from_this_run": False,
        "estimated_cost_usd": 0.25,
    }
    watchdog = {
        "status": "provider_terminal",
        "independent_process": True,
        "provider_absence_confirmed": True,
        "final_inventory": {"api_confirmed": True, "live_resource_count": 0},
        "final_global_inventory": {"api_confirmed": True, "live_resource_count": 0},
        "recorded_vast_instance_teardown": {"provider_absence_confirmed": True},
    }
    records: dict[str, dict[str, object]] = {}
    for label, value in {
        "adapter_result": adapter,
        "teardown_manifest": teardown,
        "final_validation": final,
        "watchdog_receipt": watchdog,
    }.items():
        path = root / f"{label}.json"
        _write(path, value)
        records[label] = _record(path)
    return records


def _raw_result(preflight_path: Path, *, mutate_outside: bool = False) -> Path:
    preflight = __import__("json").loads(preflight_path.read_text())
    rows: list[dict[str, object]] = []
    root = preflight_path.parent / "raw"
    root.mkdir()
    for index, input_row in enumerate(preflight["camera_inputs"]):
        before = Path(input_row["retained_scene_before"]["path"])
        mask = Path(input_row["exact_residual_mask"]["path"])
        image = np.asarray(Image.open(before).convert("RGB"), dtype=np.uint8).copy()
        mask_pixels = np.asarray(Image.open(mask).convert("L"), dtype=np.uint8) > 0
        image[mask_pixels] = np.array([7 + index, 170, 33], dtype=np.uint8)
        if mutate_outside:
            image[0, 0] = np.array([255, 0, 0], dtype=np.uint8)
        path = root / f"{input_row['task_id']}__{input_row['camera_id']}.png"
        Image.fromarray(image, mode="RGB").save(path)
        rows.append(
            {
                "task_id": input_row["task_id"],
                "camera_id": input_row["camera_id"],
                "native_aura_frame": {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                },
            }
        )
    result: dict[str, object] = {
        "schema_version": "public_scene_aura_exact_residual_raw_result.v1",
        "status": "aura_native_residual_frames_rendered",
        "preflight_digest": preflight["preflight_digest"],
        "aura_inpainting_executed": True,
        "provider_mutations_performed": 1,
        "learned_policy_outcomes_accessed": False,
        "provider_closeout": _provider_closeout(root),
        "frames": rows,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    path = root / "raw_result.json"
    _write(path, result)
    return path


def test_composites_raw_aura_output_inside_only_the_exact_masks(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path)
    raw = _raw_result(preflight, mutate_outside=True)

    receipt = materialize_aura_exact_residual_composite(
        preflight_path=preflight, raw_result_path=raw, output_root=tmp_path / "composite"
    )

    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["outside_mask_changed_pixels_total"] == 0
    assert receipt["replacement_object_count"] == 2
    assert len(receipt["frames"]) == 2
    for row in receipt["frames"]:
        before = np.asarray(
            Image.open(tmp_path / "composite" / row["task_id"] / row["retained_scene_before"]["relative_path"])
            .convert("RGB"),
            dtype=np.uint8,
        )
        after = np.asarray(
            Image.open(tmp_path / "composite" / row["task_id"] / row["exact_mask_composited_frame"]["relative_path"])
            .convert("RGB"),
            dtype=np.uint8,
        )
        mask = np.asarray(
            Image.open(tmp_path / "composite" / row["task_id"] / row["exact_residual_mask"]["relative_path"])
            .convert("L"),
            dtype=np.uint8,
        ) > 0
        assert np.array_equal(before[~mask], after[~mask])
        assert np.any(before[mask] != after[mask])
    assert receipt["composite_digest"] == canonical_digest(
        receipt, digest_field="composite_digest"
    )


def test_rejects_raw_result_with_a_missing_camera(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path)
    raw = _raw_result(preflight)
    result = __import__("json").loads(raw.read_text())
    result["frames"].pop()
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    _write(raw, result)

    with pytest.raises(AuraExactResidualCompositeError, match="raw_camera_set_mismatch"):
        materialize_aura_exact_residual_composite(
            preflight_path=preflight, raw_result_path=raw, output_root=tmp_path / "composite"
        )


def test_rejects_caller_asserted_provider_zero_without_destroy_receipt(
    tmp_path: Path,
) -> None:
    preflight = _preflight(tmp_path)
    raw = _raw_result(preflight)
    result = __import__("json").loads(raw.read_text())
    teardown = Path(result["provider_closeout"]["teardown_manifest"]["path"])
    value = __import__("json").loads(teardown.read_text())
    value["teardown_actions_performed"] = []
    _write(teardown, value)
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    _write(raw, result)

    with pytest.raises(AuraExactResidualCompositeError, match="provider_zero_not_proven"):
        materialize_aura_exact_residual_composite(
            preflight_path=preflight,
            raw_result_path=raw,
            output_root=tmp_path / "composite",
        )


def test_rejects_non_binary_or_empty_exact_mask(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path)
    raw = _raw_result(preflight)
    value = __import__("json").loads(preflight.read_text())
    mask = Path(value["camera_inputs"][0]["exact_residual_mask"]["path"])
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L").save(mask)

    with pytest.raises(AuraExactResidualCompositeError, match="mask_invalid"):
        materialize_aura_exact_residual_composite(
            preflight_path=preflight, raw_result_path=raw, output_root=tmp_path / "composite"
        )
