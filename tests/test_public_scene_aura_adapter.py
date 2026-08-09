from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.public_scene_aura_adapter import (
    AuraAdapterError,
    materialize_aura_adapter,
)
from blueprint_pipeline.provider_runtime_bundle_contract import (
    provider_runtime_contract_blockers,
)
from blueprint_pipeline.adp_aura_interiorgs_vast import (
    AURA_INTERIORGS_GPU_SELECTION_POLICY,
    PROVIDER_EXECUTION_TIMEOUT_SECONDS,
    PROVIDER_HEARTBEAT_NO_PROGRESS_SECONDS,
    _validated_adapter,
    _remaining_minutes,
    run_aura_interiorgs_vast,
)
from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight


def _record(path: Path, root: Path) -> dict:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _git_repo(path: Path, filename: str) -> tuple[str, str]:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / filename).write_text("fixture", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, tree


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    method, lama = tmp_path / "method", tmp_path / "lama"
    method_commit, method_tree = _git_repo(method, "LICENSE")
    lama_commit, lama_tree = _git_repo(lama, "LICENSE")
    monkeypatch.setattr("blueprint_pipeline.public_scene_aura_adapter.AURA_COMMIT", method_commit)
    monkeypatch.setattr("blueprint_pipeline.public_scene_aura_adapter.AURA_TREE", method_tree)
    monkeypatch.setattr("blueprint_pipeline.public_scene_aura_adapter.AURA_SUBMODULES", {})
    monkeypatch.setattr("blueprint_pipeline.public_scene_aura_adapter.LAMA_COMMIT", lama_commit)
    monkeypatch.setattr("blueprint_pipeline.public_scene_aura_adapter.LAMA_TREE", lama_tree)

    archive = data / "big-lama.zip"
    archive.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_aura_adapter.BIG_LAMA_SIZE", archive.stat().st_size
    )
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_aura_adapter.BIG_LAMA_SHA256",
        "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest(),
    )

    input_root = data / "input"
    (input_root / "images").mkdir(parents=True)
    (input_root / "masks").mkdir()
    splat = write_standard_3dgs_ply(
        SplatData(
            count=20,
            xyz=np.linspace(0.0, 1.0, 60, dtype=np.float32).reshape(20, 3),
            opacity=np.full(20, 5.0, np.float32),
            f_dc=np.full((20, 3), 0.2, np.float32),
            scales=np.full((20, 3), -3.0, np.float32),
            quats=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], np.float32), (20, 1)),
            properties=(),
            sh_rest=np.zeros((20, 45), np.float32),
        ),
        input_root / "scene_standard.ply",
    )
    names = [
        "approach_close",
        "approach_wide",
        "cabinet_context",
        "left_translate",
        "low_approach",
        "raised_left",
        "raised_right",
        "right_translate",
    ]
    cameras, images, masks = [], [], []
    for index, camera_id in enumerate(names):
        image = input_root / "images" / f"{camera_id}.png"
        mask = input_root / "masks" / f"{camera_id}.png"
        Image.new("RGB", (2048, 1536), (20 + index, 30, 40)).save(image)
        pixels = np.zeros((1536, 2048), np.uint8)
        side = 80 if camera_id == "low_approach" else 20 + index
        pixels[400 : 400 + side, 500 : 500 + side] = 255
        Image.fromarray(pixels, mode="L").save(mask)
        pose = np.eye(4)
        pose[:3, 3] = [index * 0.1, 1.0, 0.5]
        cameras.append(
            {
                "camera_id": camera_id,
                "T_world_camera_opencv": pose.tolist(),
                "intrinsics": {
                    "model": "PINHOLE",
                    "fx": 1600.0,
                    "fy": 1600.0,
                    "cx": 1024.0,
                    "cy": 768.0,
                    "width": 2048,
                    "height": 1536,
                },
            }
        )
        images.append({"camera_id": camera_id, **_record(image, input_root)})
        masks.append(
            {
                "camera_id": camera_id,
                "masked_pixel_count": int(np.count_nonzero(pixels)),
                **_record(mask, input_root),
            }
        )
    cameras_path = input_root / "cameras.v1.json"
    cameras_path.write_text(json.dumps(cameras), encoding="utf-8")
    frozen = {
        "schema_version": "adp009b_interiorgs_edit_input_receipt.v1",
        "status": "render_derived_input_packet_materialized",
        "scene": {
            "publisher_scene_id": "840313",
            "target_instance_id": "160",
            "target_semantic_label": "canned_beverage",
        },
        "proof_boundaries": {"inpainting_result": False},
        "derived_artifacts": {
            "cameras": _record(cameras_path, input_root),
            "standard_splat": _record(splat, input_root),
            "images": images,
            "masks": masks,
        },
    }
    frozen["receipt_digest"] = canonical_digest(frozen, digest_field="receipt_digest")
    frozen_path = repo / "frozen.json"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")

    required_ids = [
        "aurafusion360_sam2_hiera_large",
        "aurafusion360_marigold_depth_v1_0",
        "aurafusion360_marigold_agdd_v1_0",
        "aurafusion360_sd2_inpainting_exact_checkpoint",
    ]
    prerequisite = data / "prerequisite.json"
    prerequisite.write_text(
        json.dumps(
            {
                "methods": {
                    "aurafusion360_quality_challenger": {
                        "checkpoint_rights_established": True,
                        "remote_snapshots": [
                            {"artifact_id": item, "rights_established": True}
                            for item in required_ids
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    smoke = data / "smoke.json"
    smoke.write_text(
        json.dumps(
            {
                "status": "completed",
                "source_commit": method_commit,
                "source_tree": method_tree,
                "author_source_modified": False,
                "training_executed": True,
                "render_executed": True,
                "removal_executed": True,
                "sam2_masks_executed": True,
                "inpaint_init_executed": True,
            }
        ),
        encoding="utf-8",
    )
    return {
        "repo": repo,
        "data": data,
        "method": method,
        "lama": lama,
        "archive": archive,
        "input": input_root,
        "frozen": frozen_path,
        "prerequisite": prerequisite,
        "smoke": smoke,
    }


def _materialize(paths: dict[str, Path]) -> dict:
    return materialize_aura_adapter(
        input_receipt_path=paths["frozen"],
        input_root=paths["input"],
        method_prerequisite_receipt_path=paths["prerequisite"],
        author_smoke_receipt_path=paths["smoke"],
        repo_root=paths["repo"],
        data_root=paths["data"],
        method_root=paths["method"],
        lama_root=paths["lama"],
        big_lama_archive=paths["archive"],
        output_root=paths["data"] / "adapter",
        receipt_output=paths["repo"] / "retained.json",
    )


def test_aura_adapter_derives_native_packet_and_remains_unexecuted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = _materialize(paths)
    output = paths["data"] / "adapter"

    assert receipt["status"] == "prepared_unexecuted"
    assert receipt["scene"]["reference_camera_id"] == "low_approach"
    assert receipt["scene"]["reference_camera_selection"] == (
        "maximum_frozen_target_mask_pixel_count"
    )
    assert receipt["scene"]["reference_camera_index"] == 4
    assert receipt["scene"]["scene_slug"] == "840313_ins160"
    assert receipt["scene"]["method_resolution_divisor"] == 1
    assert receipt["reference_generator"]["checkpoint_archive"]["sha256"].startswith("sha256:")
    assert receipt["source"]["source_modified_for_adapter"] is False
    assert receipt["execution"]["aurafusion360_interiorgs_executed"] is False
    assert receipt["claim_boundary"]["inpainting_result"] is False
    assert receipt["blockers"] == ["aurafusion360_interiorgs_gpu_execution_missing"]
    assert (output / "data/Other-360/840313_ins160/sparse/0/points3D.ply").is_file()
    assert (output / "reference_lama_input/low_approach_mask.png").is_file()
    inpaint_config = (output / "configs/Other-360/840313_ins160/inpaint.config").read_text()
    remove_config = (output / "configs/Other-360/840313_ins160/remove.config").read_text()
    assert "reference_index = 4" in inpaint_config
    assert "resolution = 1" in inpaint_config
    assert "render_path = false" in inpaint_config
    assert "render_path = false" in remove_config
    assert all(
        "--render_path" not in command
        for command in receipt["commands"]["author_workflow"]
    )
    assert receipt["adapter"]["trajectory_media_contract"] == {
        "generated": False,
        "retained": False,
        "reason": "unretained_240_frame_trajectory_is_not_evaluation_evidence",
    }
    assert canonical_digest(receipt, digest_field="receipt_digest") == receipt["receipt_digest"]


def test_aura_adapter_derives_new_scene_slug_and_reference_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    frozen = json.loads(paths["frozen"].read_text())
    cameras_path = paths["input"] / "cameras.v1.json"
    cameras = json.loads(cameras_path.read_text())
    cameras_path.write_text(json.dumps(list(reversed(cameras))), encoding="utf-8")
    frozen["derived_artifacts"]["cameras"] = _record(cameras_path, paths["input"])
    frozen["scene"].update(
        {
            "publisher_scene_id": "840796",
            "target_instance_id": "123",
            "target_semantic_label": "refrigerator",
        }
    )
    selected = paths["input"] / "masks/raised_left.png"
    pixels = np.zeros((1536, 2048), np.uint8)
    pixels[300:500, 500:700] = 255
    Image.fromarray(pixels, mode="L").save(selected)
    for row in frozen["derived_artifacts"]["masks"]:
        if row["camera_id"] == "raised_left":
            row.update(_record(selected, paths["input"]))
            row["masked_pixel_count"] = int(np.count_nonzero(pixels))
    frozen["receipt_digest"] = canonical_digest(frozen, digest_field="receipt_digest")
    paths["frozen"].write_text(json.dumps(frozen), encoding="utf-8")

    receipt = _materialize(paths)

    assert receipt["scene"]["publisher_scene_id"] == "840796"
    assert receipt["scene"]["target_instance_id"] == "ins123"
    assert receipt["scene"]["scene_slug"] == "840796_ins123"
    assert receipt["scene"]["reference_camera_id"] == "raised_left"
    assert receipt["scene"]["reference_camera_index"] == 5
    assert receipt["scene"]["runtime_camera_order"][5] == "raised_left"
    assert receipt["scene"]["runtime_camera_order_derivation"] == (
        "released_aura_colmap_reader_sorted_by_image_name"
    )
    assert receipt["adapter"]["trajectory_media_contract"]["generated"] is False
    assert (
        paths["data"]
        / "adapter/data/Other-360/840796_ins123/reference"
    ).is_dir()

    monkeypatch.setattr(
        "blueprint_pipeline.adp_aura_interiorgs_vast.SOURCE_COMMIT",
        receipt["source"]["commit"],
    )
    monkeypatch.setattr(
        "blueprint_pipeline.adp_aura_interiorgs_vast.SOURCE_TREE",
        receipt["source"]["tree"],
    )
    rows, binding = _validated_adapter(receipt, paths["data"] / "adapter")
    assert rows
    assert binding["publisher_scene_id"] == "840796"
    assert binding["target_instance_id"] == "ins123"
    assert binding["reference_camera_id"] == "raised_left"


def test_aura_bundle_rejects_caller_asserted_unretained_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = _materialize(paths)
    receipt["adapter"]["trajectory_media_contract"]["generated"] = True
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    monkeypatch.setattr(
        "blueprint_pipeline.adp_aura_interiorgs_vast.SOURCE_COMMIT",
        receipt["source"]["commit"],
    )
    monkeypatch.setattr(
        "blueprint_pipeline.adp_aura_interiorgs_vast.SOURCE_TREE",
        receipt["source"]["tree"],
    )

    with pytest.raises(
        ValueError, match="adp_aura_interiorgs_trajectory_media_contract_invalid"
    ):
        _validated_adapter(receipt, paths["data"] / "adapter")


def test_aura_adapter_rejects_changed_frozen_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    mask = paths["input"] / "masks/low_approach.png"
    mask.write_bytes(mask.read_bytes() + b"changed")
    with pytest.raises(AuraAdapterError, match="mask_bytes_changed:low_approach"):
        _materialize(paths)


def test_aura_adapter_rejects_unproven_author_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    smoke = json.loads(paths["smoke"].read_text())
    smoke["inpaint_init_executed"] = False
    paths["smoke"].write_text(json.dumps(smoke), encoding="utf-8")
    with pytest.raises(AuraAdapterError, match="unchanged_author_smoke_not_proven"):
        _materialize(paths)


def test_aura_interiorgs_provider_runtime_has_distinct_fail_closed_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    entrypoint = (root / "scripts/run_adp_aura_interiorgs_provider_runtime.sh").read_text()
    runner = (root / "scripts/adp_aura_interiorgs_provider_runner.py").read_text()
    assert (
        provider_runtime_contract_blockers(
            provider_bundle_kind="adp_aura_interiorgs",
            entrypoint_text=entrypoint,
            runner_text=runner,
        )
        == []
    )
    assert (
        provider_runtime_contract_blockers(
            provider_bundle_kind="adp_aura_smoke",
            entrypoint_text=entrypoint,
            runner_text=runner,
        )
        != []
    )


def test_aura_interiorgs_vast_dry_run_binds_bundle_without_provider_mutation(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"immutable-aura-interiorgs-bundle")
    prepared = {
        "status": "ready",
        "bundle_path": str(bundle),
        "bundle_sha256": "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest(),
    }
    result = run_aura_interiorgs_vast(
        job_dir=tmp_path / "job",
        paid_resource_admission_grant=None,
        execute=False,
        prepared_bundle=prepared,
    )
    assert result["status"] == "dry_run_ready"
    assert result["provider_mutations_performed"] == 0
    assert result["retry_cap"] == 0


def test_aura_interiorgs_vast_reuses_completed_author_control_gpu_class() -> None:
    assert AURA_INTERIORGS_GPU_SELECTION_POLICY == {
        "policy_id": "aura_interiorgs_l40s_observed_control",
        "allowed_gpu_keywords": ("L40S",),
        "denied_gpu_keywords": (),
        "reason": "reuse the L40S class that completed the unchanged Aura author control",
    }
    source = Path(run_aura_interiorgs_vast.__code__.co_filename).read_text(encoding="utf-8")
    assert 'vast_launch_lock_file=job.parent / "aura_interiorgs_paid_launch.lock"' in source
    assert "allowed_active_instance_ids=allowed_active_instance_ids" in source
    assert PROVIDER_EXECUTION_TIMEOUT_SECONDS == 14_400
    assert PROVIDER_HEARTBEAT_NO_PROGRESS_SECONDS == 1800


def test_aura_interiorgs_vast_budget_uses_attempt_ledger(tmp_path: Path) -> None:
    (tmp_path / "adp_aura_interiorgs_vast_session_budget.json").write_text(
        json.dumps(
            {
                "attempts": [
                    {
                        "runtime_seconds_observed_by_adapter": 3600,
                        "estimated_cost_usd": 0.75,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert (
        _remaining_minutes(
            job=tmp_path,
            hard_cap_usd=6.0,
            hard_ttl_seconds=14_400,
            max_hourly_rate_usd=1.5,
        )
        == 180
    )


def test_vast_preflight_accepts_distinct_aura_interiorgs_bundle(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = tmp_path / "bundle.zip"
    entries = {
        "provider_runtime/run_adp_aura_interiorgs_provider_runtime.sh": (
            root / "scripts/run_adp_aura_interiorgs_provider_runtime.sh"
        ).read_bytes(),
        "provider_runtime/adp_aura_interiorgs_provider_runner.py": (
            root / "scripts/adp_aura_interiorgs_provider_runner.py"
        ).read_bytes(),
        "provider_runtime/adp_aura_interiorgs_provider_manifest.json": b"{}",
        "provider_runtime/sam2_source.zip": b"x",
        "provider_runtime/aurafusion360_source.zip": b"x",
        "provider_runtime/lama_source.zip": b"x",
        "provider_runtime/big-lama.zip": b"x",
        "provider_runtime/interiorgs_adapter.zip": b"x",
        "provider_runtime/execution_spec.json": b"{}",
    }
    with zipfile.ZipFile(bundle, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    result = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="2026-08-05T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_aura_interiorgs",
        bundle_path=bundle,
        provider_bundle_url="https://example.invalid/bundle.zip",
        provider_output_put_url="https://example.invalid/out.zip",
    )
    assert result["status"] == "passed"
    assert result["blockers"] == []
