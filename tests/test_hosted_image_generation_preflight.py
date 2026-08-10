from __future__ import annotations

import base64
import json
import struct
from pathlib import Path

from blueprint_pipeline.hosted_image_generation_preflight import (
    materialize_hosted_image_generation_preflight,
)


def _png(width: int = 1024, height: int = 1024) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)


def test_gpt_image_2_preflight_executes_bounded_generation(tmp_path: Path) -> None:
    observed: dict = {}

    def post(endpoint, headers, payload):
        observed.update(
            endpoint=endpoint,
            authorization=headers["Authorization"],
            payload=json.loads(payload),
        )
        return 200, json.dumps(
            {
                "data": [{"b64_json": base64.b64encode(_png()).decode("ascii")}],
                "usage": {"input_tokens": 12, "output_tokens": 196, "total_tokens": 208},
            }
        ).encode("utf-8")

    result = materialize_hosted_image_generation_preflight(
        output_path=tmp_path / "receipt.json",
        generated_at="fixed",
        secret_loader=lambda: ("unit-test-secret", "fixture"),
        http_post=post,
    )

    assert result["status"] == "qualified"
    assert result["model"] == "gpt-image-2"
    assert result["output"]["width"] == 1024
    assert result["output"]["bytes_retained"] is False
    assert result["estimated_output_cost_usd"] == 0.006
    assert observed["payload"] == {
        "model": "gpt-image-2",
        "prompt": (
            "A single solid red square centered on a plain white background, "
            "flat geometric icon, no text, no logos."
        ),
        "n": 1,
        "size": "1024x1024",
        "quality": "low",
    }
    assert observed["authorization"] == "Bearer unit-test-secret"
    assert "unit-test-secret" not in json.dumps(result)


def test_image_preflight_rejects_catalog_only_or_wrong_size(tmp_path: Path) -> None:
    result = materialize_hosted_image_generation_preflight(
        output_path=tmp_path / "receipt.json",
        secret_loader=lambda: ("secret", "fixture"),
        http_post=lambda *_args: (
            200,
            json.dumps(
                {"data": [{"b64_json": base64.b64encode(_png(512, 512)).decode("ascii")}]}
            ).encode("utf-8"),
        ),
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["hosted_image_generation_png_invalid"]


def test_image_preflight_blocks_without_key_without_calling_provider(tmp_path: Path) -> None:
    result = materialize_hosted_image_generation_preflight(
        output_path=tmp_path / "receipt.json",
        secret_loader=lambda: ("", "missing"),
        http_post=lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected call")),
    )

    assert result["status"] == "blocked"
    assert result["inference_probe_performed"] is False
    assert result["blockers"] == ["hosted_image_generation_api_key_missing"]
