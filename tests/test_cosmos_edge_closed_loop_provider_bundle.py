from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from blueprint_pipeline.cosmos_edge_closed_loop_provider_bundle import (
    build_cosmos_edge_policy_canary_bundle,
)
from blueprint_pipeline.policy_ranking_successor_gpu_admission import (
    EDGE_CLOSED_LOOP_PROFILE,
    inspect_successor_bundle,
)
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256


def test_policy_canary_bundle_is_identity_bound_and_secret_free(tmp_path: Path) -> None:
    frames = {}
    keys = (
        "observation/wrist_image_left",
        "observation/exterior_image_1_left",
        "observation/exterior_image_2_left",
    )
    for index, key in enumerate(keys):
        path = tmp_path / f"view-{index}.png"
        Image.fromarray(np.full((24, 32, 3), 20 + index, dtype=np.uint8)).save(path)
        frames[key] = path
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text('{"manifest_sha256":"' + "a" * 64 + '"}\n')
    first = tmp_path / "first.png"
    Image.new("RGB", (32, 24)).save(first)
    skeleton = tmp_path / "skeleton.mp4"
    skeleton.write_bytes(b"skeleton")

    receipt = build_cosmos_edge_policy_canary_bundle(
        output_dir=tmp_path / "out",
        policy_snapshot_manifest_path=snapshot,
        view_first_frames=frames,
        joint_position=[0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0],
        gripper_position=[0.0],
        prompt="Pick up the bottle.",
        oscar_fixture_first_frame=first,
        oscar_fixture_skeleton=skeleton,
        generated_at="2026-07-29T00:00:00Z",
    )

    bundle = Path(receipt["bundle_path"])
    assert bundle.is_file()
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        assert {
            "provider_runtime/wam_provider_runtime_runner.py",
            "provider_runtime/run_wam_provider_runtime.sh",
            "provider_runtime/wam_provider_runtime_manifest.json",
            "provider_runtime/wam_rollout_input_manifest.json",
            "provider_runtime/oscar_input/first_frame.png",
            "provider_runtime/oscar_input/blueprint_proxy_skeleton_conditioning.mp4",
        }.issubset(names)
        payload = json.loads(archive.read("provider_runtime/policy_canary/input.json"))
        assert payload["physical_outcome_fields_included"] is False
        assert len(payload["views"]) == 3
        runner = archive.read("provider_runtime/wam_provider_runtime_runner.py").decode()
        assert "OSCAR-2B" in runner
        assert "action_conditioned_video_rollout_generated" in runner
        assert "HF_TOKEN" not in runner
        assert 'uv_bin = uv / "bin/uv"' in runner
        assert "policy_server_client_readiness_timeout" in runner
        assert "policy_server_load_seconds" in runner
        assert "gpu_memory_after_inference_mb" in runner
        assert "commanded_state_advance_proven" in runner
        manifest = json.loads(archive.read("provider_runtime/wam_provider_runtime_manifest.json"))
        assert manifest["experiment_id"] == ("policy_ranking_cosmos3_edge_closed_loop_20260729")
        assert "@sha256:" in manifest["public_image"]
    receipt_payload = dict(receipt)
    recorded = receipt_payload.pop("receipt_sha256")
    assert recorded == canonical_sha256(receipt_payload)

    profile = replace(
        EDGE_CLOSED_LOOP_PROFILE,
        expected_bundle_sha256=receipt["bundle_sha256"],
        expected_bundle_size_bytes=receipt["bundle_size_bytes"],
        expected_embedded_input_hashes={
            key: receipt[key]
            for key in (
                "runtime_manifest_sha256",
                "canary_input_sha256",
                "runner_sha256",
                "entrypoint_sha256",
            )
        },
    )
    inspection = inspect_successor_bundle(
        bundle,
        receipt=receipt,
        smoke_inventory={},
        profile=profile,
    )
    assert inspection["status"] == "passed"
    assert inspection["blockers"] == []
