from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
pytest.importorskip("PIL")
from PIL import Image

from blueprint_pipeline.synthesis import cosmos_worker


def _stdout_json_lines(text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_cosmos_worker_emit_and_generate_success_and_missing_input(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    cosmos_worker._emit({"type": "event", "ok": True})
    assert _stdout_json_lines(capsys.readouterr().out) == [{"type": "event", "ok": True}]

    input_path = tmp_path / "input.png"
    output_path = tmp_path / "world.npy"
    Image.new("RGB", (2, 3), color=(1, 2, 3)).save(input_path)
    calls: dict[str, object] = {}

    def fake_cosmos_image_to_world(**kwargs):  # type: ignore[no-untyped-def]
        calls.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"world")
        Path(kwargs["output_path"]).with_suffix(".mp4").write_bytes(b"video")

    monkeypatch.setattr(cosmos_worker, "_cosmos_image_to_world", fake_cosmos_image_to_world)

    result = cosmos_worker._generate(
        model={"fake": "model"},
        request={
            "request_id": "req-1",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "num_frames": 4,
            "width": 5,
            "height": 6,
            "guidance_scale": 1.5,
            "num_steps": 7,
        },
    )

    assert result["type"] == "result"
    assert result["request_id"] == "req-1"
    assert result["ok"] is True
    assert result["video_path"] == str(output_path.with_suffix(".mp4"))
    assert calls["num_frames"] == 4
    assert calls["width"] == 5
    assert calls["height"] == 6

    missing = tmp_path / "missing.png"
    try:
        cosmos_worker._generate(
            model={},
            request={"request_id": "req-2", "input_path": str(missing), "output_path": str(output_path)},
        )
    except RuntimeError as exc:
        assert "conditioning_image_missing" in str(exc)
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("missing input should raise")


def test_cosmos_worker_main_handles_startup_and_request_loop(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cosmos_worker, "load_cosmos_model", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cosmos_worker.main() == 1
    assert _stdout_json_lines(capsys.readouterr().out) == [
        {"type": "error", "stage": "startup", "error": "boom"}
    ]

    input_path = tmp_path / "input.png"
    output_path = tmp_path / "world.npy"
    Image.new("RGB", (2, 2), color=(4, 5, 6)).save(input_path)

    def fake_generate(_model, request):  # type: ignore[no-untyped-def]
        if request.get("request_id") == "bad-generate":
            raise RuntimeError("generate failed")
        return {
            "type": "result",
            "request_id": request.get("request_id"),
            "ok": True,
            "output_path": request.get("output_path"),
            "video_path": None,
            "generation_ms": 1,
        }

    monkeypatch.setattr(cosmos_worker, "load_cosmos_model", lambda **_kwargs: {"model": "loaded"})
    monkeypatch.setattr(cosmos_worker, "describe_cosmos_model", lambda _model: {"worker_backend": "fake"})
    monkeypatch.setattr(cosmos_worker, "_generate", fake_generate)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            "\n".join(
                [
                    "",
                    "{bad-json",
                    json.dumps(["not", "object"]),
                    json.dumps({"type": "ping"}),
                    json.dumps({"type": "unknown"}),
                    json.dumps(
                        {
                            "type": "generate",
                            "request_id": "ok-generate",
                            "input_path": str(input_path),
                            "output_path": str(output_path),
                        }
                    ),
                    json.dumps(
                        {
                            "type": "generate",
                            "request_id": "bad-generate",
                            "input_path": str(input_path),
                            "output_path": str(output_path),
                        }
                    ),
                ]
            )
        ),
    )

    assert cosmos_worker.main() == 0
    lines = _stdout_json_lines(capsys.readouterr().out)
    assert lines[0] == {
        "type": "ready",
        "backend": "fake",
        "model_id": cosmos_worker._DEFAULT_COSMOS_MODEL_ID,
    }
    assert lines[1]["stage"] == "decode"
    assert lines[2] == {"type": "error", "stage": "request", "error": "request_must_be_object"}
    assert lines[3] == {"type": "pong"}
    assert lines[4]["error"] == "unsupported_request:unknown"
    assert lines[5]["request_id"] == "ok-generate"
    assert lines[6] == {
        "type": "result",
        "request_id": "bad-generate",
        "ok": False,
        "error": "generate failed",
    }
