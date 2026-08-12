from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.public_scene_replacement_depth_composition import (
    COMPOSITION_SCHEMA,
    REQUEST_SCHEMA,
    ReplacementDepthCompositionError,
    build_replacement_depth_composition_request,
    materialize_replacement_depth_composition,
    validate_replacement_depth_composition,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, root: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _sweep(
    root: Path,
    *,
    asset_id: str,
    task_freeze_digest: str,
    depth: np.ndarray,
    cells: list[dict[str, object]] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    array = root / "depth.npy"
    np.save(array, depth.astype(np.float32), allow_pickle=False)
    if cells is None:
        cells = [
            {"camera_id": "camera_a", "cell_id": "reset"},
            {"camera_id": "camera_a", "cell_id": "target"},
        ]
    value: dict[str, object] = {
        "schema_version": "replacement_usd_depth_sweep.v2",
        "status": "actual_usd_geometry_depth_rasterized",
        "asset_id": asset_id,
        "task_freeze_digest": task_freeze_digest,
        "camera_contract_digest": _digest("a"),
        "camera_rows_digest": _digest("b"),
        "scene_state_role": "task_subject" if asset_id == "asset_1" else "co_present_passive",
        "actual_usd_geometry_depth_rasterized": True,
        "caller_supplied_coverage_mask": False,
        "resolution_scale": 1.0,
        "cells": cells,
        "depth_dimensions": [int(depth.shape[2]), int(depth.shape[1])],
        "arrays": _record(array, root),
        "manifest_digest": "",
    }
    value["manifest_digest"] = canonical_digest(value, digest_field="manifest_digest")
    path = root / "sweep.json"
    _write_json(path, value)
    return path


def _request(tmp_path: Path, paths: list[Path]) -> Path:
    value: dict[str, object] = {
        "schema_version": REQUEST_SCHEMA,
        "task_id": "task_a",
        "task_freeze_digest": _digest("1"),
        "scored_task_asset_id": "asset_1",
        "frozen_before_removal_execution": True,
        "learned_policy_outcomes_accessed": False,
        "input_sweep_manifest_paths": [str(path) for path in paths],
        "request_digest": "",
    }
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    path = tmp_path / "request.json"
    _write_json(path, value)
    return path


def test_composes_nearest_depth_for_five_co_present_replacements(tmp_path: Path) -> None:
    paths = []
    for index in range(1, 6):
        depth = np.full((2, 2, 2), np.inf, dtype=np.float32)
        depth[:, index % 2, (index // 2) % 2] = float(10 - index)
        paths.append(
            _sweep(
                tmp_path / f"sweep_{index}",
                asset_id=f"asset_{index}",
                task_freeze_digest=_digest("1") if index == 1 else _digest(str(index + 1)),
                depth=depth,
            )
        )
    packet = materialize_replacement_depth_composition(
        request_path=_request(tmp_path, paths), output_root=tmp_path / "output"
    )

    assert packet["schema_version"] == COMPOSITION_SCHEMA
    assert packet["replacement_asset_ids"] == [f"asset_{index}" for index in range(1, 6)]
    assert packet["actual_composed_depth_rasterized"] is True
    assert all(
        "finite_depth_pixel_count_by_cell" in row and "visible_in_any_composed_camera" in row
        for row in packet["input_sweeps"]
    )
    assert [
        row["finite_depth_pixel_count_by_cell"] for row in packet["input_sweeps"]
    ] == [
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
    ]
    assert [
        row["visible_in_any_composed_camera"] for row in packet["input_sweeps"]
    ] == [True, True, True, True, True]
    composed = np.load(tmp_path / "output" / "replacement_depth_composition.npy")
    assert np.isfinite(composed).sum() == 8
    assert float(composed[0, 1, 0]) == 5.0


def test_blocks_input_sweeps_with_mismatched_scene_cells(tmp_path: Path) -> None:
    first = _sweep(
        tmp_path / "first",
        asset_id="asset_1",
        task_freeze_digest=_digest("1"),
        depth=np.ones((2, 2, 2), dtype=np.float32),
    )
    second = _sweep(
        tmp_path / "second",
        asset_id="asset_2",
        task_freeze_digest=_digest("2"),
        depth=np.ones((2, 2, 2), dtype=np.float32),
        cells=[
            {"camera_id": "camera_a", "cell_id": "reset"},
            {"camera_id": "camera_a", "cell_id": "wrong_state"},
        ],
    )

    with pytest.raises(
        ReplacementDepthCompositionError, match="input_cell_or_camera_mismatch"
    ):
        materialize_replacement_depth_composition(
            request_path=_request(tmp_path, [first, second]), output_root=tmp_path / "output"
        )


def test_blocks_duplicate_asset_id_even_with_different_sweeps(tmp_path: Path) -> None:
    first = _sweep(
        tmp_path / "first",
        asset_id="asset_1",
        task_freeze_digest=_digest("1"),
        depth=np.ones((2, 2, 2), dtype=np.float32),
    )
    second = _sweep(
        tmp_path / "second",
        asset_id="asset_1",
        task_freeze_digest=_digest("2"),
        depth=np.ones((2, 2, 2), dtype=np.float32),
    )

    with pytest.raises(ReplacementDepthCompositionError, match="asset_ids_invalid"):
        materialize_replacement_depth_composition(
            request_path=_request(tmp_path, [first, second]), output_root=tmp_path / "output"
        )


def test_validation_recomputes_output_instead_of_trusting_a_resealed_array(
    tmp_path: Path,
) -> None:
    first = _sweep(
        tmp_path / "first",
        asset_id="asset_1",
        task_freeze_digest=_digest("1"),
        depth=np.ones((2, 2, 2), dtype=np.float32),
    )
    second = _sweep(
        tmp_path / "second",
        asset_id="asset_2",
        task_freeze_digest=_digest("2"),
        depth=np.full((2, 2, 2), 2.0, dtype=np.float32),
    )
    output = tmp_path / "output"
    materialize_replacement_depth_composition(
        request_path=_request(tmp_path, [first, second]), output_root=output
    )
    receipt_path = output / "public_scene_replacement_depth_composition.v1.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    array_path = output / receipt["arrays"]["relative_path"]
    changed = np.load(array_path, allow_pickle=False)
    changed[0, 0, 0] = 9.0
    np.save(array_path, changed, allow_pickle=False)
    receipt["arrays"]["sha256"] = _sha256(array_path)
    receipt["arrays"]["size_bytes"] = array_path.stat().st_size
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write_json(receipt_path, receipt)

    with pytest.raises(
        ReplacementDepthCompositionError, match="output_array_invalid"
    ):
        validate_replacement_depth_composition(receipt, receipt_path=receipt_path)


def test_validation_rejects_resealed_composition_with_a_moving_passive_role(
    tmp_path: Path,
) -> None:
    first = _sweep(
        tmp_path / "first",
        asset_id="asset_1",
        task_freeze_digest=_digest("1"),
        depth=np.ones((2, 2, 2), dtype=np.float32),
    )
    second = _sweep(
        tmp_path / "second",
        asset_id="asset_2",
        task_freeze_digest=_digest("2"),
        depth=np.full((2, 2, 2), 2.0, dtype=np.float32),
    )
    output = tmp_path / "output"
    materialize_replacement_depth_composition(
        request_path=_request(tmp_path, [first, second]), output_root=output
    )
    receipt_path = output / "public_scene_replacement_depth_composition.v1.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    passive = json.loads(second.read_text(encoding="utf-8"))
    passive["scene_state_role"] = "task_subject"
    passive["manifest_digest"] = canonical_digest(passive, digest_field="manifest_digest")
    _write_json(second, passive)
    receipt["input_sweeps"][1].update(
        {
            "size_bytes": second.stat().st_size,
            "sha256": _sha256(second),
            "manifest_digest": passive["manifest_digest"],
        }
    )
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write_json(receipt_path, receipt)

    with pytest.raises(
        ReplacementDepthCompositionError, match="scene_state_roles_invalid"
    ):
        validate_replacement_depth_composition(receipt, receipt_path=receipt_path)


def test_request_rejects_more_than_five_inputs_or_policy_outcomes() -> None:
    request = {
        "schema_version": REQUEST_SCHEMA,
        "task_id": "task_a",
        "task_freeze_digest": _digest("1"),
        "scored_task_asset_id": "asset_1",
        "frozen_before_removal_execution": True,
        "learned_policy_outcomes_accessed": True,
        "input_sweep_manifest_paths": [f"/sweep_{index}.json" for index in range(6)],
    }

    with pytest.raises(ReplacementDepthCompositionError) as error:
        build_replacement_depth_composition_request(request)

    assert "input_count_invalid" in str(error.value)
    assert "policy_outcome_leakage" in str(error.value)
