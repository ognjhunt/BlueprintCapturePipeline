"""Range prefetch must verify every response and the full pinned model digest."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prefetch_g1_checkpoint_ranges.py"
spec = importlib.util.spec_from_file_location("g1_range_prefetch", SCRIPT)
assert spec and spec.loader
prefetch_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prefetch_module)


def _inventory(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps({
        "schema_version": "g1_humanoidarena_checkpoint_inventory.v1",
        "candidates": [{
            "candidate_id": "candidate",
            "subdirectory": "pi/test",
            "files": [{"path": "model.safetensors", "sha256": hashlib.sha256(content).hexdigest(),
                       "size_bytes": len(content)}],
        }],
    }))
    return path


class _Response:
    status = 206

    def __init__(self, content: bytes, start: int, end: int, *, bad_header: bool = False):
        self.stream = io.BytesIO(content[start : end + 1])
        self.headers = {"Content-Range": f"bytes {start}-{end}/{len(content)}" if not bad_header else "invalid"}

    def geturl(self) -> str:
        return "https://modelscope.cn/signed-source"

    def read(self, size: int) -> bytes:
        return self.stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.stream.close()


class _Opener:
    def __init__(self, content: bytes, *, bad_header: bool = False):
        self.content = content
        self.bad_header = bad_header

    def open(self, request, timeout: int):
        assert timeout == 180
        value = request.get_header("Range")
        assert value and value.startswith("bytes=")
        start, end = (int(part) for part in value[6:].split("-"))
        return _Response(self.content, start, end, bad_header=self.bad_header)


def test_prefetch_reassembles_and_verifies_ranges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = b"abc12345" * (2 * 1024 * 1024 + 1)
    inventory = _inventory(tmp_path, content)
    monkeypatch.setattr(prefetch_module.urllib.request, "build_opener", lambda *_args: _Opener(content))
    result = prefetch_module.prefetch(
        inventory_path=inventory, candidate_id="candidate", output_dir=tmp_path / "output",
        workers=3, chunk_mib=8,
    )
    assert result["status"] == "verified"
    assert Path(result["path"]).read_bytes() == content
    assert prefetch_module.prefetch(
        inventory_path=inventory, candidate_id="candidate", output_dir=tmp_path / "output",
        workers=3, chunk_mib=8,
    )["status"] == "verified"


def test_prefetch_rejects_wrong_range_and_leaves_no_published_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"abc12345" * (1024 * 1024 + 1)
    inventory = _inventory(tmp_path, content)
    monkeypatch.setattr(
        prefetch_module.urllib.request, "build_opener",
        lambda *_args: _Opener(content, bad_header=True),
    )
    with pytest.raises(ValueError, match="g1_prefetch_range_response_invalid"):
        prefetch_module.prefetch(
            inventory_path=inventory, candidate_id="candidate", output_dir=tmp_path / "output",
            workers=2, chunk_mib=8,
        )
    folder = tmp_path / "output/pi/test"
    assert not (folder / "model.safetensors").exists()
    assert not (folder / ".model.safetensors.parallel-partial").exists()
