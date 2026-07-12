from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FETCHER_PATH = (
    ROOT
    / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop"
    / "fetch_pinned_isaac_assets.py"
)
MANIFEST_PATH = FETCHER_PATH.with_name("isaac_6_g1_assets.sha256")
DOCKERFILE_PATH = FETCHER_PATH.with_name("Dockerfile")
SPEC = importlib.util.spec_from_file_location("fetch_pinned_isaac_assets", FETCHER_PATH)
assert SPEC and SPEC.loader
FETCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCHER)


def test_official_isaac_6_g1_manifest_is_complete_and_pinned() -> None:
    rows = FETCHER._rows(MANIFEST_PATH)

    assert len(rows) == 36
    assert sum(size for _digest, size, _relative in rows) == 179_549_205
    assert any(relative == "g1.usd" for _digest, _size, relative in rows)
    assert any(
        relative == "configuration/g1_29dof_with_hand_rev_1_0_base.usd"
        for _digest, _size, relative in rows
    )


def test_sealed_dockerfile_fetches_assets_into_bound_runtime_path() -> None:
    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "isaac_6_g1_assets.sha256" in dockerfile
    assert "fetch_pinned_isaac_assets.py" in dockerfile
    assert "/Assets/Isaac/6.0/Isaac/Robots/Unitree/G1/" in dockerfile
    assert "--output-dir /isaac-sim/Isaac/Robots/Unitree/G1" in dockerfile


def test_manifest_parser_rejects_parent_traversal(tmp_path: Path) -> None:
    manifest = tmp_path / "assets.sha256"
    manifest.write_text(f"{'a' * 64} 1 ../escape.usd\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid pinned asset row"):
        FETCHER._rows(manifest)


def test_fetcher_writes_only_verified_bytes(tmp_path: Path, monkeypatch) -> None:
    content = b"pinned-usd"
    digest = hashlib.sha256(content).hexdigest()
    manifest = tmp_path / "assets.sha256"
    manifest.write_text(f"{digest} {len(content)} configuration/g1.usd\n")

    class Response:
        def __init__(self) -> None:
            self.offset = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def geturl(self) -> str:
            return "https://assets.example/configuration/g1.usd"

        def read(self, _size: int) -> bytes:
            if self.offset:
                return b""
            self.offset = len(content)
            return content

    monkeypatch.setattr(FETCHER.urllib.request, "urlopen", lambda *_a, **_k: Response())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetch_pinned_isaac_assets.py",
            "--manifest",
            str(manifest),
            "--base-url",
            "https://assets.example/",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert FETCHER.main() == 0
    assert (tmp_path / "out/configuration/g1.usd").read_bytes() == content
