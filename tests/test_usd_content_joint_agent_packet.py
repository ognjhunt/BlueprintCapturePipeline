from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from blueprint_pipeline.articulated_source_asset import materialize_articulated_source_asset
from blueprint_pipeline.usd_content_joint_agent_packet import (
    JointAgentPacketError,
    build_joint_agent_packet,
    inspect_joint_agent_checkout,
)


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _fake_release(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "joint-agent"
    (root / "apps/joint_agent/configs").mkdir(parents=True)
    (root / "VERSION.md").write_text("0.test\n", encoding="utf-8")
    (root / "LICENSE").write_text("Apache test\n", encoding="utf-8")
    (root / "apps/joint_agent/README.md").write_text("preview\n", encoding="utf-8")
    template = root / "apps/joint_agent/configs/byoa_joint_rigger.yaml"
    template.write_text("project: {}\n", encoding="utf-8")
    _git(["init", "-q"], root)
    _git(["config", "user.email", "test@example.invalid"], root)
    _git(["config", "user.name", "Test"], root)
    _git(["add", "."], root)
    _git(["commit", "-qm", "fixture"], root)
    files = {}
    import hashlib

    for relative in (
        "VERSION.md",
        "LICENSE",
        "apps/joint_agent/README.md",
        "apps/joint_agent/configs/byoa_joint_rigger.yaml",
    ):
        data = (root / relative).read_bytes()
        files[relative] = "sha256:" + hashlib.sha256(data).hexdigest()
    identity = {
        "repository": "https://example.invalid/joint-agent",
        "tag": "v0.test",
        "version": "0.test",
        "commit": _git(["rev-parse", "HEAD"], root),
        "license": "Apache-2.0",
        "files": files,
    }
    return root, identity


def _source(tmp_path: Path) -> tuple[Path, Path]:
    from pxr import Usd, UsdGeom, UsdPhysics

    labels = tmp_path / "labels.json"
    corners = [
        {"x": x, "y": y, "z": z}
        for z in (0.0, 2.0)
        for x, y in ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
    ]
    labels.write_text(
        json.dumps([{"ins_id": "121", "label": "refrigerator", "bounding_box": corners}]),
        encoding="utf-8",
    )
    collision = tmp_path / "collision.usda"
    stage = Usd.Stage.CreateNew(str(collision))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Target")
    mesh.CreatePointsAttr(
        [
            (0, 0, 0), (0.45, 0, 0), (0.45, 1, 0), (0, 1, 0),
            (0.55, 0, 0), (1, 0, 0), (1, 1, 0), (0.55, 1, 0),
            (0, 0, 2), (0.45, 0, 2), (0.45, 1, 2), (0, 1, 2),
            (0.55, 0, 2), (1, 0, 2), (1, 1, 2), (0.55, 1, 2),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4])
    mesh.CreateFaceVertexIndicesAttr(
        [0, 1, 2, 3, 8, 11, 10, 9, 4, 5, 6, 7, 12, 15, 14, 13]
    )
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()
    output = tmp_path / "source"
    materialize_articulated_source_asset(
        labels_path=labels,
        target_instance_id="121",
        sage_collision_usd_path=collision,
        output_dir=output,
    )
    return output / "articulated_source_mesh.usda", output / "articulated_source_asset_receipt.json"


def test_packet_binds_release_source_and_stops_before_remote_execution(tmp_path: Path) -> None:
    checkout, identity = _fake_release(tmp_path)
    source, receipt = _source(tmp_path)

    packet = build_joint_agent_packet(
        source_asset_path=source,
        source_receipt_path=receipt,
        joint_agent_checkout=checkout,
        output_dir=tmp_path / "packet",
        expected_identity=identity,
    )

    assert packet["status"] == "blocked_before_remote_execution"
    assert packet["execution_admission"]["blockers"] == [
        "external_scene_derived_byte_disclosure_authority_missing",
        "fresh_paid_joint_agent_execution_authority_missing",
    ]
    assert packet["execution_admission"]["remote_execution_performed"] is False
    assert packet["config"]["completion_retries"] == 0
    assert packet["config"]["predicted_dataset_prim_upper_bound"] == 4
    config = yaml.safe_load((tmp_path / "packet/joint_agent.yaml").read_text())
    assert config["steps"]["optimize_usd"]["optimization_config"][
        "scene_optimizer_settings"
    ]["enable_split_meshes"] is True
    assert config["steps"]["predict"]["completion_retries"] == 0
    assert config["steps"]["infer_articulation_candidates"]["adjudication"][
        "max_images"
    ] == 4
    assert config["steps"]["apply_joint_rigger"]["enabled"] is False


def test_checkout_identity_rejects_commit_and_file_drift(tmp_path: Path) -> None:
    checkout, identity = _fake_release(tmp_path)
    drifted = copy.deepcopy(identity)
    drifted["commit"] = "0" * 40
    drifted["files"]["VERSION.md"] = "sha256:" + "0" * 64

    with pytest.raises(JointAgentPacketError) as caught:
        inspect_joint_agent_checkout(checkout, expected_identity=drifted)

    assert "joint_agent_checkout_commit_mismatch" in caught.value.errors
    assert "joint_agent_release_file_digest_mismatch:VERSION.md" in caught.value.errors


def test_packet_rejects_mutated_source_asset(tmp_path: Path) -> None:
    checkout, identity = _fake_release(tmp_path)
    source, receipt = _source(tmp_path)
    source.write_text(source.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")

    with pytest.raises(JointAgentPacketError) as caught:
        build_joint_agent_packet(
            source_asset_path=source,
            source_receipt_path=receipt,
            joint_agent_checkout=checkout,
            output_dir=tmp_path / "packet",
            expected_identity=identity,
        )

    assert "articulated_source_asset_digest_mismatch" in caught.value.errors
