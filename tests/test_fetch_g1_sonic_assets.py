from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fetch_g1_sonic_assets.py"
spec = importlib.util.spec_from_file_location("fetch_g1_sonic_assets", SCRIPT)
assert spec is not None and spec.loader is not None
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


class _Response(io.BytesIO):
    def __init__(self, content: bytes, url: str) -> None:
        super().__init__(content)
        self._url = url

    def geturl(self) -> str:
        return self._url


def _inventory(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    content = {"encoder": b"encoder fixture", "decoder": b"decoder fixture"}
    value = {
        "schema_version": "g1_sonic_asset_inventory.v1",
        "source_repository": "https://huggingface.co/nvidia/GEAR-SONIC",
        "source_revision": "a" * 40,
        "variant_id": "default",
        "model_license": "NVIDIA Open Model License",
        "rights_review_required": True,
        "files": [
            {
                "role": role,
                "path": filename,
                "sha256": hashlib.sha256(content[role]).hexdigest(),
                "size_bytes": len(content[role]),
            }
            for role, filename in fetch.EXPECTED_FILES.items()
        ],
    }
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(value))
    return path, content


def test_fetch_then_offline_verify_exact_sonic_pair(tmp_path: Path, monkeypatch) -> None:
    inventory, content = _inventory(tmp_path)
    calls = []

    def open_url(url: str) -> _Response:
        calls.append(url)
        filename = url.rsplit("/", 1)[-1]
        role = next(key for key, value in fetch.EXPECTED_FILES.items() if value == filename)
        return _Response(content[role], url)

    monkeypatch.setattr(fetch, "_open_https", open_url)
    output = tmp_path / "sonic"
    first = fetch.stage_sonic_assets(inventory_path=inventory, output_dir=output)
    second = fetch.stage_sonic_assets(
        inventory_path=inventory, output_dir=output, verify_only=True
    )
    assert first == second
    assert first["rights_review_required"] is True
    assert first["inference_executed"] is False
    assert len(calls) == 2
    assert all("/resolve/" + "a" * 40 + "/" in url for url in calls)


def test_bad_download_does_not_publish_asset(tmp_path: Path, monkeypatch) -> None:
    inventory, _ = _inventory(tmp_path)
    monkeypatch.setattr(
        fetch,
        "_open_https",
        lambda url: _Response(b"wrong", url),
    )
    output = tmp_path / "sonic"
    with pytest.raises(ValueError, match="download_identity_mismatch"):
        fetch.stage_sonic_assets(inventory_path=inventory, output_dir=output)
    assert not (output / "model_encoder.onnx").exists()
    assert not list(output.rglob(".g1-sonic-*"))


def test_verify_only_rejects_missing_or_changed_bytes(tmp_path: Path) -> None:
    inventory, _ = _inventory(tmp_path)
    output = tmp_path / "sonic"
    with pytest.raises(ValueError, match="file_missing:encoder"):
        fetch.stage_sonic_assets(
            inventory_path=inventory, output_dir=output, verify_only=True
        )
    assert not output.exists()
    output.mkdir()
    (output / "model_encoder.onnx").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="existing_file_identity_mismatch:encoder"):
        fetch.stage_sonic_assets(
            inventory_path=inventory, output_dir=output, verify_only=True
        )


def test_redirect_guard_rejects_http_before_following() -> None:
    request = fetch.urllib.request.Request("https://huggingface.co/source")
    with pytest.raises(ValueError, match="insecure_redirect"):
        fetch._HTTPSRedirectsOnly().redirect_request(
            request, None, 302, "Found", {}, "http://example.org/asset"
        )
    redirected = fetch._HTTPSRedirectsOnly().redirect_request(
        request, None, 302, "Found", {}, "https://cdn.example.org/asset"
    )
    assert redirected.full_url == "https://cdn.example.org/asset"


def test_inventory_rejects_variant_or_path_substitution(tmp_path: Path) -> None:
    inventory, _ = _inventory(tmp_path)
    value = json.loads(inventory.read_text())
    value["files"][0]["path"] = "sonic_v1_1/model_encoder.onnx"
    inventory.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="inventory_file_invalid"):
        fetch.stage_sonic_assets(
            inventory_path=inventory, output_dir=tmp_path / "sonic", verify_only=True
        )


def test_default_inventory_pins_release_pair() -> None:
    inventory = fetch._inventory(fetch.DEFAULT_INVENTORY)
    files = {row["role"]: row for row in inventory["files"]}
    assert inventory["source_revision"] == "6733128a3d8a523b1418b06bca3cdf61c8b0987f"
    assert files["encoder"]["sha256"] == "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3"
    assert files["decoder"]["sha256"] == "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed"
