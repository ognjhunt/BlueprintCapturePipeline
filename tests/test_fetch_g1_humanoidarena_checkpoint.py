from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fetch_g1_humanoidarena_checkpoint.py"
spec = importlib.util.spec_from_file_location("fetch_g1_checkpoint", SCRIPT)
assert spec is not None and spec.loader is not None
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


class _Response(io.BytesIO):
    def __init__(self, content: bytes, url: str) -> None:
        super().__init__(content)
        self._url = url

    def geturl(self) -> str:
        return self._url


def _inventory(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps({
        "schema_version": "g1_humanoidarena_checkpoint_inventory.v1",
        "candidates": [{
            "candidate_id": "dp",
            "subdirectory": "small/HOI_pp_box/model",
            "files": [{
                "path": "config.json",
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }],
        }],
    }))
    return path


def test_fetch_then_verify_exact_candidate_bytes(tmp_path: Path, monkeypatch) -> None:
    content = b'{"type":"diffusion"}'
    inventory = _inventory(tmp_path, content)
    calls: list[str] = []

    def open_url(url: str, timeout: int) -> _Response:
        assert timeout == 180
        calls.append(url)
        return _Response(content, url)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", open_url)
    output = tmp_path / "checkpoints"
    first = fetch.materialize_candidate(
        inventory_path=inventory, candidate_id="dp", output_dir=output
    )
    second = fetch.materialize_candidate(
        inventory_path=inventory, candidate_id="dp", output_dir=output,
        verify_only=True,
    )
    assert first == second
    assert first["status"] == "checkpoint_bytes_verified"
    assert (output / "small/HOI_pp_box/model/config.json").read_bytes() == content
    assert len(calls) == 1


def test_bad_download_never_publishes_checkpoint(tmp_path: Path, monkeypatch) -> None:
    inventory = _inventory(tmp_path, b"expected")
    monkeypatch.setattr(
        fetch.urllib.request, "urlopen",
        lambda url, timeout: _Response(b"wrong", url),
    )
    output = tmp_path / "checkpoints"
    with pytest.raises(ValueError, match="download_identity_mismatch"):
        fetch.materialize_candidate(
            inventory_path=inventory, candidate_id="dp", output_dir=output
        )
    assert not (output / "small/HOI_pp_box/model/config.json").exists()
    assert not list(output.rglob(".g1-checkpoint-*"))


def test_existing_wrong_bytes_and_path_escape_fail_closed(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, b"expected")
    output = tmp_path / "checkpoints"
    existing = output / "small/HOI_pp_box/model/config.json"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="existing_file_identity_mismatch"):
        fetch.materialize_candidate(
            inventory_path=inventory, candidate_id="dp", output_dir=output,
            verify_only=True,
        )
    value = json.loads(inventory.read_text())
    value["candidates"][0]["files"][0]["path"] = "../outside"
    inventory.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="file_inventory_invalid"):
        fetch.materialize_candidate(
            inventory_path=inventory, candidate_id="dp", output_dir=output,
            verify_only=True,
        )
