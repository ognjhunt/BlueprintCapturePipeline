from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline import native_g1_publisher_source_stage as stage


def test_verify_only_does_not_create_a_checkout(tmp_path: Path) -> None:
    output = tmp_path / "source-stage"
    with pytest.raises(ValueError, match="g1_publisher_source_checkout_missing"):
        stage.stage_g1_publisher_source(output, verify_only=True)
    assert not output.exists()


def test_inventory_must_pin_the_exact_upstream_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = json.loads(stage.INVENTORY.read_text(encoding="utf-8"))
    inventory["source_revision"] = "0" * 40
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    monkeypatch.setattr(stage, "INVENTORY", path)
    with pytest.raises(ValueError, match="g1_publisher_source_inventory_identity_mismatch"):
        stage.stage_g1_publisher_source(tmp_path / "output", verify_only=True)
    assert not (tmp_path / "output").exists()


def test_existing_checkout_with_wrong_remote_is_not_reused(tmp_path: Path) -> None:
    output = tmp_path / "stage"
    source = output / "source"
    source.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "remote", "add", "origin", "https://example.invalid/other"],
        check=True,
    )
    with pytest.raises(ValueError, match="g1_publisher_source_identity_mismatch"):
        stage.stage_g1_publisher_source(output, verify_only=True)
