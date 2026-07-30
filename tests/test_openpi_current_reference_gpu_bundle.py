from __future__ import annotations

from pathlib import Path

import numpy as np

from blueprint_pipeline.common import write_json
from blueprint_pipeline.openpi_current_reference_droid_policy_runtime import (
    CURRENT_REFERENCE_INVENTORY_FILES,
)
from blueprint_pipeline.openpi_current_reference_gpu_bundle import (
    build_current_reference_gpu_input_bundle,
    extract_current_reference_gpu_input_bundle,
)
from blueprint_pipeline.openpi_current_reference_policy_canary import (
    load_current_reference_initial_observation,
)
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256, file_sha256


def _initial_packet(root: Path) -> Path:
    files = root / "packet"
    files.mkdir()
    views = {}
    for index, view_id in enumerate(
        (
            "observation/exterior_image_2_left",
            "observation/exterior_image_1_left",
            "observation/wrist_image_left",
        )
    ):
        path = files / f"view_{index}.png"
        path.write_bytes(f"fixture-view-{index}".encode())
        views[view_id] = {
            "frame_path": str(path),
            "frame_sha256": file_sha256(path),
        }
    state = {}
    for name, value in (
        ("joint_position", np.arange(7, dtype=np.float64)),
        ("gripper_position", np.asarray([0.4], dtype=np.float64)),
        ("cartesian_pose_7d", np.arange(7, dtype=np.float64) / 10),
    ):
        path = files / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        state[name] = {"path": str(path), "sha256": file_sha256(path)}
    manifest = {
        "schema_version": "ctrl_world_public_initial_observation.v1",
        "task_prompt": "pick up the object",
        "engineering_canary_eligible": True,
        "confirmation_eligible": False,
        "views": views,
        "state": state,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    path = files / "initial.json"
    write_json(path, manifest)
    return path


def test_current_reference_bundle_round_trip_is_portable_and_hash_bound(
    tmp_path: Path,
) -> None:
    inventories = tmp_path / "inventories"
    inventories.mkdir()
    for name in CURRENT_REFERENCE_INVENTORY_FILES.values():
        (inventories / name).write_text("{}", encoding="utf-8")
    freeze = tmp_path / "source_freeze.json"
    freeze.write_text("{}", encoding="utf-8")
    commit = "a" * 40
    receipt = build_current_reference_gpu_input_bundle(
        source_freeze_path=freeze,
        checkpoint_inventory_dir=inventories,
        initial_observation_manifest_path=_initial_packet(tmp_path),
        runtime_source_commit=commit,
        runtime_source_archive_url=(
            "https://codeload.github.com/ognjhunt/BlueprintCapturePipeline/tar.gz/" + commit
        ),
        runtime_source_archive_sha256="b" * 64,
        image_source_commit="c" * 40,
        output_zip=tmp_path / "input.zip",
    )
    extracted = extract_current_reference_gpu_input_bundle(
        bundle_path=receipt["bundle_path"],
        expected_bundle_sha256=receipt["bundle_sha256"],
        output_dir=tmp_path / "extracted",
    )
    manifest_path = extracted["initial_observation_manifest_path"]
    observation = load_current_reference_initial_observation(
        manifest_path,
        image_preprocessor=lambda path: np.zeros((224, 224, 3), dtype=np.uint8),
    )
    assert observation["observation/joint_position"].shape == (7,)
    assert receipt["manifest"]["checkpoint_weights_included"] is False
    assert receipt["manifest"]["physical_outcome_included"] is False
    assert receipt["manifest"]["runtime_source"]["commit"] == commit


def test_current_reference_bundle_never_overwrites_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "input.zip"
    output.write_bytes(b"user-owned")
    try:
        build_current_reference_gpu_input_bundle(
            source_freeze_path=tmp_path / "missing",
            checkpoint_inventory_dir=tmp_path / "missing",
            initial_observation_manifest_path=tmp_path / "missing",
            runtime_source_commit="a" * 40,
            runtime_source_archive_url=(
                "https://codeload.github.com/ognjhunt/BlueprintCapturePipeline/tar.gz/" + "a" * 40
            ),
            runtime_source_archive_sha256="b" * 64,
            image_source_commit="c" * 40,
            output_zip=output,
        )
    except FileExistsError:
        pass
    else:  # pragma: no cover - explicit destructive-regression assertion
        raise AssertionError("existing evidence output was not rejected")
    assert output.read_bytes() == b"user-owned"
