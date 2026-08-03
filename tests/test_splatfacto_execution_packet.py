"""Splatfacto bakeoff arm packet builder (C5): pinned, hermetic, write-once."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest


REPO_ROOT = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "build_splatfacto_execution_packet",
    REPO_ROOT / "scripts/build_splatfacto_execution_packet.py",
)
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)


def _postshot_packet(*, dataset_path: str = "initialization/colmap_dataset_9de1972eae8fe5ef") -> dict:
    packet = {
        "schema_version": "postshot_execution_packet.v1",
        "status": "prepared_awaiting_worker",
        "source_capture_digest": "sha256:" + "1" * 64,
        "frozen_split_digest": "sha256:" + "2" * 64,
        "pose_only_dataset_digest": "sha256:" + "3" * 64,
        "arms": [
            {
                "arm_id": "P1",
                "input_dataset": {
                    "kind": "point_seeded_colmap_text",
                    "relative_path": dataset_path,
                    "colmap_training_dataset_digest": "sha256:" + "4" * 64,
                    "image_count": 265,
                    "initialization_point_count": 91990,
                },
                "training_profile": "postshot_splat3",
            },
            {"arm_id": "P2", "input_dataset": {"relative_path": dataset_path}},
        ],
    }
    packet["postshot_execution_packet_digest"] = canonical_digest(
        packet, digest_field="postshot_execution_packet_digest"
    )
    return packet


def _proxy_root(tmp_path: Path, packet: dict | None = None) -> Path:
    proxy_root = tmp_path / "mushroom_proxy_fea6da5dfeca8e6a"
    postshot_dir = proxy_root / "provider_packets" / "postshot"
    postshot_dir.mkdir(parents=True)
    (postshot_dir / "postshot_execution_packet.v1.json").write_text(
        json.dumps(packet or _postshot_packet(), indent=2), encoding="utf-8"
    )
    return proxy_root


def test_packet_pins_arms_environment_and_input_dataset(tmp_path: Path) -> None:
    proxy_root = _proxy_root(tmp_path)
    packet = builder.build_splatfacto_execution_packet(proxy_root=proxy_root)

    output_path = (
        proxy_root / "provider_packets/splatfacto/splatfacto_execution_packet.v1.json"
    )
    assert output_path.is_file()
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted == packet

    assert packet["schema_version"] == "splatfacto_execution_packet.v1"
    assert packet["source_capture_digest"] == "sha256:" + "1" * 64
    assert packet["frozen_split_digest"] == "sha256:" + "2" * 64
    assert packet["hidden_images_included"] is False
    assert packet["provider_sees_hidden_views"] is False
    assert packet["required_external_inputs"] == []

    assert [arm["arm_id"] for arm in packet["arms"]] == ["G1", "G2"]
    for arm in packet["arms"]:
        assert arm["input_dataset"] == _postshot_packet()["arms"][0]["input_dataset"]
        assert arm["pose_estimation_by_provider"] is False
        assert arm["training_profile"]["seed"] == 42
        assert arm["training_profile"]["max_iterations"] == 30000

    g1, g2 = packet["arms"]
    assert g1["training_profile"]["method"] == "splatfacto"
    assert g1["training_profile"]["strategy"] == "default"
    assert g2["training_profile"]["strategy"] == "mcmc"
    assert g2["training_profile"]["mcmc_max_gs_num"] == 1_000_000

    env = packet["environment"]
    assert env["g1"]["nerfstudio"] == "nerfstudio==1.1.5"
    assert env["g1"]["gsplat"] == "gsplat==1.4.0"
    assert env["g2"]["nerfstudio"] == (
        "nerfstudio @ git+https://github.com/nerfstudio-project/nerfstudio.git"
        "@50e0e3c70c775e89333256213363badbf074f29d"
    )
    assert env["g2"]["gsplat"] == "gsplat==1.4.0"
    assert env["g1"]["requirements_file"] == "requirements/splatfacto-arm-g1.txt"
    assert env["g2"]["requirements_file"] == "requirements/splatfacto-arm-g2.txt"

    outputs = packet["shared_training_intent"]["required_outputs"]
    assert "exported_splat_ply_or_spz" in outputs
    assert "training_log" in outputs
    assert "execution_receipt_with_versions_and_durations" in outputs

    assert packet["splatfacto_execution_packet_digest"] == canonical_digest(
        packet, digest_field="splatfacto_execution_packet_digest"
    )


def test_rerun_is_idempotent_but_content_change_is_refused(tmp_path: Path) -> None:
    proxy_root = _proxy_root(tmp_path)
    first = builder.build_splatfacto_execution_packet(proxy_root=proxy_root)
    second = builder.build_splatfacto_execution_packet(proxy_root=proxy_root)
    assert (
        first["splatfacto_execution_packet_digest"]
        == second["splatfacto_execution_packet_digest"]
    )

    changed = _postshot_packet()
    changed["frozen_split_digest"] = "sha256:" + "9" * 64
    changed["postshot_execution_packet_digest"] = canonical_digest(
        changed, digest_field="postshot_execution_packet_digest"
    )
    (proxy_root / "provider_packets/postshot/postshot_execution_packet.v1.json").write_text(
        json.dumps(changed, indent=2), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        builder.build_splatfacto_execution_packet(proxy_root=proxy_root)


def test_tampered_postshot_packet_digest_is_refused(tmp_path: Path) -> None:
    tampered = _postshot_packet()
    tampered["frozen_split_digest"] = "sha256:" + "8" * 64  # digest now stale
    proxy_root = _proxy_root(tmp_path, packet=tampered)
    with pytest.raises(SystemExit, match="postshot_execution_packet_digest_mismatch"):
        builder.build_splatfacto_execution_packet(proxy_root=proxy_root)


def test_hidden_path_leak_is_refused(tmp_path: Path) -> None:
    leaking = _postshot_packet(dataset_path="evaluator_hidden/long")
    leaking["postshot_execution_packet_digest"] = canonical_digest(
        {k: v for k, v in leaking.items() if k != "postshot_execution_packet_digest"},
        digest_field="postshot_execution_packet_digest",
    )
    proxy_root = _proxy_root(tmp_path, packet=leaking)
    with pytest.raises(SystemExit, match="hidden_path_leak"):
        builder.build_splatfacto_execution_packet(proxy_root=proxy_root)


def test_missing_postshot_packet_fails_closed(tmp_path: Path) -> None:
    proxy_root = tmp_path / "mushroom_proxy_x"
    proxy_root.mkdir()
    with pytest.raises(SystemExit, match="postshot_execution_packet_missing"):
        builder.build_splatfacto_execution_packet(proxy_root=proxy_root)
