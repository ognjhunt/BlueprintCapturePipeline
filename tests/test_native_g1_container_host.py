from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import native_g1_container_host as host
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE


def _host(monkeypatch: pytest.MonkeyPatch, *, gpu: str = "0, RTX 6000 Ada, 580.95.05, 49140",
          image: str = "sha256:" + "a" * 64, free: int = 64 * 1024**3) -> None:
    monkeypatch.setattr(host.sys, "platform", "linux")
    monkeypatch.setattr(host, "_read_command", lambda argv: gpu if argv[0] == "nvidia-smi" else image)
    monkeypatch.setattr(host.shutil, "disk_usage", lambda _path: SimpleNamespace(free=free))


def test_host_receipt_binds_gpu_and_pinned_local_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _host(monkeypatch)
    receipt = host.record_g1_container_host(output_dir=tmp_path)
    assert receipt["status"] == "host_ready_for_container_attempt"
    assert receipt["image_reference"] == NATIVE_TASK_ARENA_IMAGE
    assert receipt["gpu_index"] == 0
    assert receipt["gpu_memory_mib"] == 49140
    assert receipt["container_started"] is False
    assert receipt["episode_executed"] is False
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")
    assert json.loads((tmp_path / (host.SCHEMA + ".json")).read_text()) == receipt


@pytest.mark.parametrize(("gpu", "image", "free", "blocker"), [
    ("1, RTX 6000 Ada, 580.95.05, 49140", "sha256:" + "a" * 64,
     64 * 1024**3, "g1_container_gpu_zero_missing"),
    ("0, RTX 6000 Ada, 570.00.00, 49140", "sha256:" + "a" * 64,
     64 * 1024**3, "g1_container_driver_below_tested_isaac_floor"),
    ("0, RTX 6000 Ada, 580.95.05, 8192", "sha256:" + "a" * 64,
     64 * 1024**3, "g1_container_gpu_memory_insufficient"),
    ("0, NVIDIA A100, 580.95.05, 81920", "sha256:" + "a" * 64,
     64 * 1024**3, "g1_container_gpu_without_rt_cores"),
    ("0, RTX 6000 Ada, 580.95.05, 49140", "missing",
     64 * 1024**3, "g1_container_local_image_identity_invalid"),
    ("0, RTX 6000 Ada, 580.95.05, 49140", "sha256:" + "a" * 64,
     8 * 1024**3, "g1_container_free_disk_insufficient"),
])
def test_host_refuses_doomed_container_attempt_before_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    gpu: str, image: str, free: int, blocker: str,
) -> None:
    _host(monkeypatch, gpu=gpu, image=image, free=free)
    with pytest.raises(ValueError, match=blocker):
        host.record_g1_container_host(output_dir=tmp_path)
    assert not (tmp_path / (host.SCHEMA + ".json")).exists()


def test_host_refuses_missing_local_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _host(monkeypatch)
    def probe(argv: list[str]) -> str:
        if argv[0] == "docker":
            raise ValueError("probe failed")
        return "0, RTX 6000 Ada, 580.95.05, 49140"
    monkeypatch.setattr(host, "_read_command", probe)
    with pytest.raises(ValueError, match="g1_container_pinned_image_unavailable"):
        host.record_g1_container_host(output_dir=tmp_path)
    assert not (tmp_path / (host.SCHEMA + ".json")).exists()
