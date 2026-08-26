from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_discovery_path_activates_hardened_no_spend_worker() -> None:
    path = _text("deploy/systemd/blueprint-scene-object-discovery.path")
    service = _text("deploy/systemd/blueprint-scene-object-discovery.service")

    assert "scene-object-discoveries/pending/*.json" in path
    assert "Unit=blueprint-scene-object-discovery.service" in path
    assert "User=blueprint" in service
    assert "NoNewPrivileges=true" in service
    assert "ProtectSystem=strict" in service
    assert "scene_object_discovery_worker" in service
    assert "paid_resource_allocator" not in service
    assert "ReadWritePaths=" in service


def test_installer_and_environment_create_and_enable_discovery_queue() -> None:
    installer = _text("scripts/install_live_pipeline_control_plane.sh")
    environment = _text("deploy/systemd/pipeline-control-plane.env.example")

    for state in ("pending", "processing", "blocked", "results", "identities", "selections"):
        assert f"scene-object-discoveries/{state}" in installer
    assert "blueprint-scene-object-discovery.service" in installer
    assert "blueprint-scene-object-discovery.path" in installer
    assert "systemctl enable --now blueprint-scene-object-discovery.path" in installer
    assert "BLUEPRINT_SCENE_OBJECT_DISCOVERY_QUEUE_ROOT=" in environment
    assert "BLUEPRINT_SCENE_OBJECT_DISCOVERY_PUBLICATION_PREFIX=" in environment
    assert "Provider-GPU mode remains blocked" in environment
