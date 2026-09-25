from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tempfile
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
    inventory = {
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
    }
    files = inventory["candidates"][0]["files"]
    inventory["candidates"][0]["inventory_digest"] = "sha256:" + hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(inventory))
    return path


def test_fetch_then_verify_exact_candidate_bytes(tmp_path: Path, monkeypatch) -> None:
    content = b'{"type":"diffusion"}'
    inventory = _inventory(tmp_path, content)
    calls: list[str] = []

    def open_url(url: str) -> _Response:
        calls.append(url)
        return _Response(content, url)

    monkeypatch.setattr(fetch, "_open_https", open_url)
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
        fetch, "_open_https",
        lambda url: _Response(b"wrong", url),
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


def test_redirect_guard_rejects_http_before_following() -> None:
    request = fetch.urllib.request.Request("https://modelscope.cn/source")
    with pytest.raises(ValueError, match="insecure_redirect"):
        fetch._HTTPSRedirectsOnly().redirect_request(
            request, None, 302, "Found", {}, "http://example.org/asset"
        )
    redirected = fetch._HTTPSRedirectsOnly().redirect_request(
        request, None, 302, "Found", {}, "https://cdn.example.org/asset"
    )
    assert redirected.full_url == "https://cdn.example.org/asset"


def test_candidate_inventory_digest_rejects_changed_file_identity(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, b"expected")
    value = json.loads(inventory.read_text())
    value["candidates"][0]["files"][0]["sha256"] = "0" * 64
    inventory.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="candidate_inventory_digest_invalid"):
        fetch.materialize_candidate(
            inventory_path=inventory, candidate_id="dp", output_dir=tmp_path / "checkpoints",
            verify_only=True,
        )


def test_large_checkpoint_ranges_are_complete_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"checkpoint" * (2 * 1024 * 1024 + 1)
    calls: list[tuple[int, int]] = []

    def open_range(url: str, *, headers: dict[str, str]):
        assert url.startswith("https://modelscope.cn/")
        start, end = (int(part) for part in headers["Range"][6:].split("-"))
        calls.append((start, end))
        response = _Response(content[start : end + 1], url)
        response.status = 206
        response.headers = {"Content-Range": f"bytes {start}-{end}/{len(content)}"}
        return response

    monkeypatch.setattr(fetch, "_open_https", open_range)
    with tempfile.NamedTemporaryFile(dir=tmp_path) as stream:
        assert fetch._download_pinned_ranges(
            "https://modelscope.cn/pinned", stream.fileno(), len(content),
            chunk_size=8 * 1024 * 1024, workers=3,
        ) == len(content)
        assert Path(stream.name).read_bytes() == content
    assert len(calls) == 3


def test_candidate_materialization_owns_ranged_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"range-verified-checkpoint"
    inventory = _inventory(tmp_path, content)
    monkeypatch.setattr(fetch, "RANGED_DOWNLOAD_MIN_BYTES", 1)

    def open_range(url: str, *, headers: dict[str, str]):
        assert headers["Range"] == f"bytes=0-{len(content) - 1}"
        response = _Response(content, url)
        response.status = 206
        response.headers = {"Content-Range": f"bytes 0-{len(content) - 1}/{len(content)}"}
        return response

    monkeypatch.setattr(fetch, "_open_https", open_range)
    output = tmp_path / "checkpoints"
    result = fetch.materialize_candidate(
        inventory_path=inventory, candidate_id="dp", output_dir=output,
    )
    assert result["status"] == "checkpoint_bytes_verified"
    assert (output / "small/HOI_pp_box/model/config.json").read_bytes() == content
    assert not list(output.rglob(".g1-checkpoint-*"))


def test_large_checkpoint_rejects_wrong_range_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"checkpoint" * 1024

    def open_range(url: str, *, headers: dict[str, str]):
        response = _Response(content, url)
        response.status = 200
        response.headers = {}
        return response

    monkeypatch.setattr(fetch, "_open_https", open_range)
    with tempfile.NamedTemporaryFile(dir=tmp_path) as stream:
        with pytest.raises(ValueError, match="g1_checkpoint_range_response_invalid"):
            fetch._download_pinned_ranges(
                "https://modelscope.cn/pinned", stream.fileno(), len(content),
                chunk_size=8 * 1024, workers=2,
            )


def test_large_checkpoint_deadline_cleans_partial_without_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"checkpoint" * 1024
    inventory = _inventory(tmp_path, content)
    monkeypatch.setattr(fetch, "RANGED_DOWNLOAD_MIN_BYTES", 1)
    output = tmp_path / "checkpoints"

    # Exercise the materializer's cleanup with a deliberately expired fetch.
    original = fetch._download_pinned_ranges
    monkeypatch.setattr(fetch, "_download_pinned_ranges", lambda *args, **kwargs:
                        original(*args, **kwargs, deadline_seconds=1e-9))
    with pytest.raises(TimeoutError, match="g1_checkpoint_download_deadline_exceeded"):
        fetch.materialize_candidate(
            inventory_path=inventory, candidate_id="dp", output_dir=output,
        )
    assert not (output / "small/HOI_pp_box/model/config.json").exists()
    assert not list(output.rglob(".g1-checkpoint-*"))


def test_pinned_navigation_candidates_are_distinct_40_value_movement_policies() -> None:
    inventory = json.loads(fetch.DEFAULT_INVENTORY.read_text())
    candidates = {row["candidate_id"]: row for row in inventory["candidates"]}
    assert len(candidates) == 4
    for suffix in ("dp", "pi05"):
        movement_id = f"humanoidarena_{suffix}_g1_dex3_sonic_vision_navi"
        movement = fetch._candidate(inventory, movement_id)
        assert movement["policy_role"] == "movement_navigation"
        assert movement["task_checkpoint"] == "HSI_vision_navi"
        assert movement["input_image_shape_hwc"] == [480, 640, 3]
        assert movement["input_state_width"] == 64
        assert movement["output_action_width"] == 40
        assert movement["action_interface"] == "humanoidarena_semantic_v3"
        assert any(row["path"] == "model.safetensors" for row in movement["files"])
        assert movement["files"] != candidates[f"humanoidarena_{suffix}_g1_dex3_sonic"]["files"]
