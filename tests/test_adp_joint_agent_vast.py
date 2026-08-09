from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from blueprint_pipeline.adp_joint_agent_vast import build_joint_agent_vast_bundle
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.usd_content_joint_agent_packet import JOINT_AGENT_IDENTITY
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _probe_env,
    _probe_shell_script,
    _resolve_launch_mode,
)


def _checkout(root: Path) -> Path:
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for relative, digest in JOINT_AGENT_IDENTITY["files"].items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "VERSION.md":
            path.write_text("0.5.2", encoding="utf-8")
        else:
            # The builder's checkout inspector is exercised separately against
            # the real locked checkout; this fixture injects its own identity by
            # constructing the exact release files below in the test monkeypatch.
            path.write_text(digest, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture"],
        cwd=root,
        check=True,
    )
    return root


def test_builder_binds_scene_neutral_joint_runtime(monkeypatch, tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    checkout = _checkout(tmp_path / "checkout")
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\ndef Xform \"asset\" {}\n", encoding="utf-8")
    source_digest = "sha256:" + __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    source_receipt_digest = "sha256:" + "1" * 64
    config = {
        "project": {"name": "fixture", "working_dir": str(tmp_path / "work")},
        "input": {"usd_path": str(source)},
        "steps": {
            "optimize_usd": {"enabled": True},
            "identify_asset": {"enabled": True, "renderer": {"backend": "remote"}},
            "analyze_structure": {"enabled": True},
            "build_dataset_usd": {"enabled": True, "renderer": {"backend": "remote"}},
            "build_dataset_prepare_dataset": {"enabled": True},
            "predict": {"enabled": True, "completion_retries": 0},
            "consistency_pass": {"enabled": True},
            "infer_articulation_candidates": {"enabled": True},
            "apply_joint_rigger": {"enabled": False},
            "author_physics_schemas": {"enabled": False},
        },
    }
    config_path = tmp_path / "joint.yaml"
    import yaml
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    packet = {
        "schema_version": "usd_content_joint_agent_packet.v1",
        "source_asset": {
            "path": str(source),
            "sha256": source_digest,
            "source_receipt_digest": source_receipt_digest,
        },
        "config": {"path": str(config_path)},
        "packet_digest": "",
    }
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    freeze = {
        "scene": {"publisher_scene_id": "840796"},
        "member_geometry_observation": {
            "joint_axis_world": [0.0, 0.0, 1.0],
            "upper_member_vertical_interval_m": [0.94, 1.632],
        },
        "task_spec": {"target_joint_id": "upper_hinge"},
        "freeze_digest": "",
    }
    freeze["freeze_digest"] = canonical_digest(freeze, digest_field="freeze_digest")
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    authority = {
        "authorization_digest": "",
        "joint_agent_source_asset_digest": source_digest,
        "joint_agent_source_receipt_digest": source_receipt_digest,
        "publisher_scene_id": "840796",
        "freeze_digest": freeze["freeze_digest"],
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    monkeypatch.setattr(
        "blueprint_pipeline.adp_joint_agent_vast.validate_public_scene_execution_authority",
        lambda value: value,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.adp_joint_agent_vast.inspect_joint_agent_checkout",
        lambda value: {"commit": JOINT_AGENT_IDENTITY["commit"], "version": "0.5.2"},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.adp_joint_agent_vast.SOURCE_TREE",
        subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=checkout, check=True, capture_output=True, text=True
        ).stdout.strip(),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.adp_joint_agent_vast._blueprint_identity",
        lambda value: {"commit": "a" * 40, "tree": "b" * 40, "dirty": False},
    )
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")

    receipt = build_joint_agent_vast_bundle(
        repo_root=repo,
        joint_agent_root=checkout,
        packet_path=packet_path,
        execution_authority_path=authority_path,
        freeze_path=freeze_path,
        job_dir=tmp_path / "bundle",
        generated_at="2026-08-08T00:00:00+00:00",
    )

    assert receipt["status"] == "ready"
    assert receipt["provider_bundle_kind"] == "adp_joint_agent"
    assert receipt["input_usd_sha256"] == source_digest
    assert receipt["renderer"]["scene_bytes_leave_vast_instance"] is False
    assert receipt["blueprint_source"]["commit"] == "a" * 40
    assert receipt["completion_retries"] == 0
    assert "840313" not in (tmp_path / "bundle/provider_runtime/joint_agent.yaml").read_text()
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_joint_agent",
        bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://example.com/bundle.zip?signature=redacted",
        provider_output_put_url="https://example.com/output.zip?signature=redacted",
    )
    assert preflight["status"] == "passed"


def test_joint_agent_provider_uses_gpu_graphics_and_distinct_runtime(tmp_path: Path) -> None:
    assert _resolve_launch_mode(
        requested="auto",
        enable_isaac_smoke=False,
        enable_blueprint_bundle=True,
        provider_bundle_kind="adp_joint_agent",
    ) == "ssh_direct"
    env = _probe_env(
        job_dir=tmp_path,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_joint_agent",
        forward_hf_token=False,
    )
    assert env["NVIDIA_DRIVER_CAPABILITIES"] == "all"
    assert "ACCEPT_EULA" not in env
    script = _probe_shell_script(
        "https://example.com",
        enable_isaac_smoke=False,
        enable_blueprint_bundle=True,
        provider_bundle_kind="adp_joint_agent",
    )
    assert "run_adp_joint_agent_provider_runtime.sh" in script
    assert "adp_joint_agent_provider_runtime_output.zip" in script
    assert "adp_content_agents_provider_runtime_output.zip" not in script


def test_builder_preflight_failure_leaves_no_partial_output(
    monkeypatch, tmp_path: Path
) -> None:
    packet = {"packet_digest": ""}
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    authority = {"authorization_digest": ""}
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    freeze = {"freeze_digest": ""}
    freeze["freeze_digest"] = canonical_digest(freeze, digest_field="freeze_digest")
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(
        "blueprint_pipeline.adp_joint_agent_vast.validate_public_scene_execution_authority",
        lambda value: value,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.adp_joint_agent_vast.inspect_joint_agent_checkout",
        lambda value: (_ for _ in ()).throw(ValueError("release mismatch")),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.adp_joint_agent_vast._blueprint_identity",
        lambda value: {"commit": "a" * 40, "tree": "b" * 40, "dirty": False},
    )
    destination = tmp_path / "bundle"

    with pytest.raises(ValueError, match="release mismatch"):
        build_joint_agent_vast_bundle(
            repo_root=Path(__file__).resolve().parents[1],
            joint_agent_root=checkout,
            packet_path=packet_path,
            execution_authority_path=authority_path,
            freeze_path=freeze_path,
            job_dir=destination,
        )

    assert not destination.exists()
