from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import urllib.error

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.semantic_teacher_image_edit_worker import (
    MAX_PROVIDER_RESPONSE_BYTES,
    RUNTIME_REQUEST_SCHEMA_VERSION,
    SemanticTeacherImageEditWorkerError,
    execute_semantic_teacher_image_edits,
    main,
)


def _png_bytes(*, size: tuple[int, int], color: tuple[int, ...], mode: str) -> bytes:
    output = BytesIO()
    Image.new(mode, size, color).save(output, format="PNG")
    return output.getvalue()


def _record(path: Path, *, root: Path) -> dict:
    import hashlib

    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _runtime_request(tmp_path: Path, *, frame_count: int = 2) -> tuple[Path, list[Path]]:
    root = tmp_path / "runtime"
    inputs = root / "input_packet"
    inputs.mkdir(parents=True)
    frames = []
    source_paths = []
    for index in range(frame_count):
        source = inputs / f"source-{index}.png"
        mask = inputs / f"mask-{index}.png"
        source.write_bytes(
            _png_bytes(size=(6, 4), color=(10 + index, 20, 30), mode="RGB")
        )
        mask.write_bytes(_png_bytes(size=(6, 4), color=(255, 255, 255, 0), mode="RGBA"))
        source_paths.append(source)
        frames.append(
            {
                "frame_index": index,
                "camera_id": f"camera_{index}",
                "input_rgb": _record(source, root=root),
                "edit_mask": _record(mask, root=root),
            }
        )
    registry_entry = {
        "backend_id": "future_hosted_editor",
        "capability": "semantic_teacher_image_edit",
    }
    value = {
        "schema_version": RUNTIME_REQUEST_SCHEMA_VERSION,
        "backend": {
            "registry_entry": registry_entry,
            "backend_entry_digest": canonical_digest(registry_entry),
            "execution": {
                "adapter_id": "openai_images_edits_v1",
                "transport_kind": "hosted_image_edit",
                "endpoint": "https://editor.example.invalid/v1/images/edits",
                "model_snapshot": "future-editor-immutable-snapshot",
                "masked_image_edit_supported": True,
                "input_fidelity_parameter_supported": True,
                "external_disclosure_required": True,
                "supported_output_sizes": ["6x4"],
                "pricing_binding": {
                    "usage_required": False,
                    "usd_per_million_tokens": {},
                },
                "default_options": {"quality": "high", "output_format": "png"},
            },
        },
        "prompt": "Remove the task object and reconstruct the empty room.",
        "tasks": [{"task_id": "task_a", "frames": frames}],
        "retry_count": 0,
        "request_digest": "",
    }
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    path = root / "runtime-request.json"
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path, source_paths


class _Response:
    def __init__(
        self,
        payload: bytes,
        url: str = "https://editor.example.invalid/v1/images/edits",
    ):
        self.payload = payload
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int = -1) -> bytes:
        return self.payload if size < 0 else self.payload[:size]

    def geturl(self) -> str:
        return self.url


def _inline_response(image_bytes: bytes, *, usage: dict | None = None) -> bytes:
    value = {"data": [{"b64_json": base64.b64encode(image_bytes).decode("ascii")}]}
    if usage is not None:
        value["usage"] = usage
    return json.dumps(value).encode("utf-8")


def test_executes_each_bound_frame_once_without_model_hardcoding(tmp_path: Path) -> None:
    request_path, source_paths = _runtime_request(tmp_path)
    calls = []
    generated = _png_bytes(size=(6, 4), color=(90, 100, 110), mode="RGB")

    def opener(request, *, timeout: int):
        calls.append((request, timeout))
        return _Response(_inline_response(generated))

    result = execute_semantic_teacher_image_edits(
        runtime_request_path=request_path,
        output_root=tmp_path / "output",
        token="fixture-secret-token",
        opener=opener,
    )

    assert result["status"] == "completed_unreviewed_semantic_teacher_candidates"
    assert result["model_snapshot"] == "future-editor-immutable-snapshot"
    assert result["request_count"] == 2
    assert result["retry_count"] == 0
    assert result["raw_secret_values_recorded"] is False
    assert len(calls) == 2
    for index, ((request, timeout), source) in enumerate(zip(calls, source_paths, strict=True)):
        assert timeout == 300
        assert request.full_url == "https://editor.example.invalid/v1/images/edits"
        assert request.get_header("Authorization") == "Bearer fixture-secret-token"
        assert b"future-editor-immutable-snapshot" in request.data
        assert b"Remove the task object and reconstruct the empty room." in request.data
        assert b'name="model"\r\n\r\nfuture-editor-immutable-snapshot\r\n' in request.data
        assert b'name="response_format"' not in request.data
        assert b'name="size"\r\n\r\n6x4\r\n' in request.data
        assert b'name="image"; filename=' in request.data
        assert b'name="mask"; filename=' in request.data
        assert b'filename="input.png"' in request.data
        assert b'filename="mask.png"' in request.data
        assert source.read_bytes() in request.data
        output = tmp_path / "output" / result["tasks"][0]["frames"][index][
            "semantic_teacher_frame"
        ]["relative_path"]
        assert output.read_bytes() == generated
    serialized = json.dumps(result)
    assert "fixture-secret-token" not in serialized


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        json.dumps({"data": [{"url": "https://unbound.invalid/output.png"}]}).encode(),
        _inline_response(_png_bytes(size=(2, 2), color=(1, 2, 3), mode="RGB")),
        _inline_response(b"not-a-png"),
        json.dumps({"data": []}).encode(),
    ],
)
def test_rejects_unbound_or_invalid_provider_output(
    tmp_path: Path, payload: bytes
) -> None:
    request_path, _sources = _runtime_request(tmp_path, frame_count=1)

    with pytest.raises(
        SemanticTeacherImageEditWorkerError,
        match="semantic_teacher_provider_response_invalid",
    ):
        execute_semantic_teacher_image_edits(
            runtime_request_path=request_path,
            output_root=tmp_path / "output",
            token="fixture-token",
            opener=lambda *_args, **_kwargs: _Response(payload),
        )


def test_rejects_changed_input_before_any_provider_request(tmp_path: Path) -> None:
    request_path, sources = _runtime_request(tmp_path, frame_count=1)
    sources[0].write_bytes(b"changed")
    calls = []

    with pytest.raises(
        SemanticTeacherImageEditWorkerError,
        match="semantic_teacher_runtime_input_invalid",
    ):
        execute_semantic_teacher_image_edits(
            runtime_request_path=request_path,
            output_root=tmp_path / "output",
            token="fixture-token",
            opener=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
    assert calls == []


def test_rejects_missing_token_before_output_or_network(tmp_path: Path) -> None:
    request_path, _sources = _runtime_request(tmp_path, frame_count=1)
    calls = []

    with pytest.raises(
        SemanticTeacherImageEditWorkerError,
        match="semantic_teacher_runtime_request_invalid",
    ):
        execute_semantic_teacher_image_edits(
            runtime_request_path=request_path,
            output_root=tmp_path / "output",
            token="",
            opener=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
    assert calls == []


def test_mask_bytes_are_bound_in_the_multipart_request(tmp_path: Path) -> None:
    request_path, _sources = _runtime_request(tmp_path, frame_count=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    mask_path = request_path.parent / request["tasks"][0]["frames"][0]["edit_mask"][
        "relative_path"
    ]
    calls = []
    generated = _png_bytes(size=(6, 4), color=(4, 5, 6), mode="RGB")

    def opener(http_request, **_kwargs):
        calls.append(http_request)
        return _Response(_inline_response(generated))

    execute_semantic_teacher_image_edits(
        runtime_request_path=request_path,
        output_root=tmp_path / "output",
        token="fixture-token",
        opener=opener,
    )
    assert len(calls) == 1
    assert mask_path.read_bytes() in calls[0].data
    assert np.asarray(Image.open(mask_path).convert("RGBA"))[..., 3].max() == 0


def test_provider_failure_is_not_retried(tmp_path: Path) -> None:
    request_path, _sources = _runtime_request(tmp_path, frame_count=1)
    calls = []

    def opener(*args, **kwargs):
        calls.append((args, kwargs))
        raise OSError("fixture provider failure")

    with pytest.raises(
        SemanticTeacherImageEditWorkerError,
        match="semantic_teacher_provider_request_failed",
    ):
        execute_semantic_teacher_image_edits(
            runtime_request_path=request_path,
            output_root=tmp_path / "output",
            token="fixture-token",
            opener=opener,
        )
    assert len(calls) == 1


def test_http_failure_retains_only_digest_bound_safe_discriminators(
    tmp_path: Path,
) -> None:
    request_path, _sources = _runtime_request(tmp_path, frame_count=1)
    token = "fixture-secret-token"
    body_secret = "body-secret-must-never-be-recorded"
    header_secret = "header-secret-must-never-be-recorded"
    response_body = json.dumps(
        {
            "error": {
                "message": f"unsupported input_fidelity {body_secret} {token}",
                "type": "invalid_request_error",
                "code": "unsupported_parameter",
                "param": "input_fidelity",
            },
            "authorization": token,
        }
    ).encode()
    calls = []

    def opener(*args, **kwargs):
        calls.append((args, kwargs))
        raise urllib.error.HTTPError(
            "https://editor.example.invalid/v1/images/edits?secret=not-retained",
            400,
            f"provider prose {body_secret}",
            {
                "x-request-id": "req_fixture123",
                "authorization": f"Bearer {header_secret}",
            },
            BytesIO(response_body),
        )

    output = tmp_path / "output"
    with pytest.raises(
        SemanticTeacherImageEditWorkerError,
        match="semantic_teacher_provider_http_error",
    ):
        execute_semantic_teacher_image_edits(
            runtime_request_path=request_path,
            output_root=output,
            token=token,
            opener=opener,
        )
    assert len(calls) == 1
    result = json.loads(
        (output / "semantic_teacher_image_edit_runtime_result.v1.json").read_text()
    )
    failure = result["terminal_provider_failure"]
    assert failure == result["tasks"][0]["frames"][0]["provider_failure"]
    assert failure["http_status"] == 400
    assert failure["provider_error_type"] == "invalid_request_error"
    assert failure["provider_error_code"] == "unsupported_parameter"
    assert failure["provider_request_id"] == "req_fixture123"
    assert failure["failure_digest"] == canonical_digest(
        failure, digest_field="failure_digest"
    )
    assert result["result_digest"] == canonical_digest(
        result, digest_field="result_digest"
    )
    serialized = json.dumps(result)
    for forbidden in (
        token,
        body_secret,
        header_secret,
        "provider prose",
        "?secret=not-retained",
        "authorization",
    ):
        assert forbidden not in serialized


def test_http_failure_discards_secret_shaped_discriminators(tmp_path: Path) -> None:
    request_path, _sources = _runtime_request(tmp_path, frame_count=1)
    token = "sk-fixture-token"
    response_body = json.dumps(
        {
            "error": {
                "type": "secret_internal_type",
                "code": token,
                "message": token,
            }
        }
    ).encode()

    def opener(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "https://editor.example.invalid/v1/images/edits",
            401,
            token,
            {"x-request-id": f"req_{token}"},
            BytesIO(response_body),
        )

    output = tmp_path / "output"
    with pytest.raises(SemanticTeacherImageEditWorkerError):
        execute_semantic_teacher_image_edits(
            runtime_request_path=request_path,
            output_root=output,
            token=token,
            opener=opener,
        )
    result = json.loads(
        (output / "semantic_teacher_image_edit_runtime_result.v1.json").read_text()
    )
    failure = result["terminal_provider_failure"]
    assert failure["http_status"] == 401
    assert failure["provider_error_type"] is None
    assert failure["provider_error_code"] is None
    assert failure["provider_request_id"] is None
    assert token not in json.dumps(result)


def test_parameter_unsupported_backend_cannot_send_input_fidelity(
    tmp_path: Path,
) -> None:
    request_path, _sources = _runtime_request(tmp_path, frame_count=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    execution = request["backend"]["execution"]
    execution["input_fidelity_parameter_supported"] = False
    execution["default_options"]["input_fidelity"] = "high"
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    request_path.write_text(json.dumps(request), encoding="utf-8")
    calls = []

    with pytest.raises(
        SemanticTeacherImageEditWorkerError,
        match="semantic_teacher_runtime_request_invalid",
    ):
        execute_semantic_teacher_image_edits(
            runtime_request_path=request_path,
            output_root=tmp_path / "output",
            token="fixture-token",
            opener=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
    assert calls == []


def test_rejects_changed_request_before_any_provider_request(tmp_path: Path) -> None:
    request_path, _sources = _runtime_request(tmp_path, frame_count=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["prompt"] = "Changed after sealing"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    calls = []

    with pytest.raises(
        SemanticTeacherImageEditWorkerError,
        match="semantic_teacher_runtime_request_invalid",
    ):
        execute_semantic_teacher_image_edits(
            runtime_request_path=request_path,
            output_root=tmp_path / "output",
            token="fixture-token",
            opener=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
    assert calls == []


def test_rejects_mask_dimension_mismatch_before_any_request(tmp_path: Path) -> None:
    request_path, _sources = _runtime_request(tmp_path, frame_count=2)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    mask_record = request["tasks"][0]["frames"][1]["edit_mask"]
    mask_path = request_path.parent / mask_record["relative_path"]
    mask_path.write_bytes(_png_bytes(size=(2, 2), color=(0, 0, 0, 0), mode="RGBA"))
    mask_record.update(_record(mask_path, root=request_path.parent))
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path.write_text(json.dumps(request), encoding="utf-8")
    calls = []

    with pytest.raises(
        SemanticTeacherImageEditWorkerError,
        match="semantic_teacher_runtime_frame_media_invalid",
    ):
        execute_semantic_teacher_image_edits(
            runtime_request_path=request_path,
            output_root=tmp_path / "output",
            token="fixture-token",
            opener=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
    assert calls == []


@pytest.mark.parametrize(
    "reserved", ["model", "prompt", "response_format", "size", "image", "mask"]
)
def test_default_options_cannot_override_fixed_multipart_fields(
    tmp_path: Path, reserved: str
) -> None:
    request_path, _sources = _runtime_request(tmp_path, frame_count=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["backend"]["execution"]["default_options"][reserved] = "attacker-value"
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path.write_text(json.dumps(request), encoding="utf-8")
    calls = []
    with pytest.raises(
        SemanticTeacherImageEditWorkerError,
        match="semantic_teacher_runtime_request_invalid",
    ):
        execute_semantic_teacher_image_edits(
            runtime_request_path=request_path,
            output_root=tmp_path / "output",
            token="fixture-token",
            opener=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
    assert calls == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_default_options_are_rejected_before_network(
    tmp_path: Path, value: float
) -> None:
    request_path, _sources = _runtime_request(tmp_path, frame_count=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["backend"]["execution"]["default_options"]["strength"] = value
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path.write_text(json.dumps(request), encoding="utf-8")
    calls = []
    with pytest.raises(
        SemanticTeacherImageEditWorkerError,
        match="semantic_teacher_runtime_request_invalid",
    ):
        execute_semantic_teacher_image_edits(
            runtime_request_path=request_path,
            output_root=tmp_path / "output",
            token="fixture-token",
            opener=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
    assert calls == []


def test_provider_response_read_is_bounded(tmp_path: Path) -> None:
    request_path, _sources = _runtime_request(tmp_path, frame_count=1)
    requested_sizes = []

    class OversizedResponse(_Response):
        def read(self, size: int = -1) -> bytes:
            requested_sizes.append(size)
            return b"x" * size

    with pytest.raises(
        SemanticTeacherImageEditWorkerError,
        match="semantic_teacher_provider_response_invalid",
    ):
        execute_semantic_teacher_image_edits(
            runtime_request_path=request_path,
            output_root=tmp_path / "output",
            token="fixture-token",
            opener=lambda *_args, **_kwargs: OversizedResponse(b""),
        )
    assert requested_sizes == [MAX_PROVIDER_RESPONSE_BYTES + 1]


def test_redirected_response_is_rejected_without_forwarding_retry(tmp_path: Path) -> None:
    request_path, _sources = _runtime_request(tmp_path, frame_count=1)
    calls = []

    def opener(http_request, **_kwargs):
        calls.append(http_request)
        return _Response(
            _inline_response(_png_bytes(size=(6, 4), color=(1, 2, 3), mode="RGB")),
            url="https://redirected.example.invalid/v1/images/edits",
        )

    with pytest.raises(
        SemanticTeacherImageEditWorkerError,
        match="semantic_teacher_provider_redirect_rejected",
    ):
        execute_semantic_teacher_image_edits(
            runtime_request_path=request_path,
            output_root=tmp_path / "output",
            token="fixture-token",
            opener=opener,
        )
    assert len(calls) == 1


def test_post_request_failure_retains_partial_png_and_per_frame_states(
    tmp_path: Path,
) -> None:
    request_path, _sources = _runtime_request(tmp_path, frame_count=2)
    generated = _png_bytes(size=(6, 4), color=(1, 2, 3), mode="RGB")
    payloads = [_inline_response(generated), b"malformed-after-possible-billing"]

    def opener(*_args, **_kwargs):
        return _Response(payloads.pop(0))

    output = tmp_path / "output"
    with pytest.raises(
        SemanticTeacherImageEditWorkerError,
        match="semantic_teacher_provider_response_invalid",
    ):
        execute_semantic_teacher_image_edits(
            runtime_request_path=request_path,
            output_root=output,
            token="fixture-token",
            opener=opener,
        )
    result = json.loads(
        (output / "semantic_teacher_image_edit_runtime_result.v1.json").read_text()
    )
    assert result["status"] == "failed_with_retained_partial_inventory"
    assert result["attempted_request_count"] == 2
    assert result["successful_request_count"] == 1
    assert len(result["partial_png_inventory"]) == 1
    assert [row["terminal_state"] for row in result["tasks"][0]["frames"]] == [
        "completed_unreviewed_candidate",
        "failed_after_request_attempt",
    ]


def test_usage_tokens_and_computed_editor_cost_are_retained(tmp_path: Path) -> None:
    request_path, _sources = _runtime_request(tmp_path, frame_count=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["backend"]["execution"]["pricing_binding"] = {
        "usage_required": True,
        "usd_per_million_tokens": {
            "input_image_tokens": 2.0,
            "output_image_tokens": 4.0,
        },
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path.write_text(json.dumps(request), encoding="utf-8")
    usage = {
        "input_tokens": 100,
        "output_tokens": 200,
        "total_tokens": 300,
        "input_tokens_details": {"text_tokens": 25, "image_tokens": 75},
        "output_tokens_details": {"text_tokens": 50, "image_tokens": 150},
    }
    result = execute_semantic_teacher_image_edits(
        runtime_request_path=request_path,
        output_root=tmp_path / "output",
        token="fixture-token",
        opener=lambda *_args, **_kwargs: _Response(
            _inline_response(
                _png_bytes(size=(6, 4), color=(1, 2, 3), mode="RGB"),
                usage=usage,
            )
        ),
    )
    assert result["billing_qualified"] is True
    assert result["provider_usage_totals"]["input_image_tokens"] == 75
    assert result["provider_usage_totals"]["output_image_tokens"] == 150
    assert result["computed_editor_cost_usd"] == pytest.approx(0.00075)


def test_missing_required_usage_preserves_png_but_blocks_billing(tmp_path: Path) -> None:
    request_path, _sources = _runtime_request(tmp_path, frame_count=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["backend"]["execution"]["pricing_binding"]["usage_required"] = True
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path.write_text(json.dumps(request), encoding="utf-8")
    result = execute_semantic_teacher_image_edits(
        runtime_request_path=request_path,
        output_root=tmp_path / "output",
        token="fixture-token",
        opener=lambda *_args, **_kwargs: _Response(
            _inline_response(_png_bytes(size=(6, 4), color=(1, 2, 3), mode="RGB"))
        ),
    )
    assert result["status"] == "completed_candidates_billing_unqualified"
    assert result["successful_request_count"] == 1
    assert result["billing_qualified"] is False
    assert result["blockers"] == ["provider_usage_missing"]


def test_input_mutation_after_preflight_cannot_change_uploaded_bytes(tmp_path: Path) -> None:
    request_path, sources = _runtime_request(tmp_path, frame_count=1)
    original = sources[0].read_bytes()
    generated = _png_bytes(size=(6, 4), color=(4, 5, 6), mode="RGB")
    calls = []

    def opener(http_request, **_kwargs):
        sources[0].write_bytes(b"changed-after-preflight")
        calls.append(http_request)
        return _Response(_inline_response(generated))

    execute_semantic_teacher_image_edits(
        runtime_request_path=request_path,
        output_root=tmp_path / "output",
        token="fixture-token",
        opener=opener,
    )
    assert len(calls) == 1
    assert original in calls[0].data
    assert b"changed-after-preflight" not in calls[0].data


def test_blocked_receipt_redacts_os_error_details(tmp_path: Path, monkeypatch) -> None:
    request_path, _sources = _runtime_request(tmp_path, frame_count=1)
    secret = "fixture-secret-must-not-appear"

    def fail(**_kwargs):
        raise OSError(f"provider rejected token {secret}")

    monkeypatch.setattr(
        "blueprint_pipeline.semantic_teacher_image_edit_worker.execute_semantic_teacher_image_edits",
        fail,
    )
    output = tmp_path / "output"
    assert (
        main(
            [
                "--runtime-request",
                str(request_path),
                "--output-root",
                str(output),
            ]
        )
        == 1
    )
    serialized = (output / "semantic_teacher_image_edit_runtime_result.v1.json").read_text()
    assert secret not in serialized
    assert "semantic_teacher_runtime_execution_failed" in serialized


@pytest.mark.parametrize("task_id", [".", "..", "../escape", "bad/name", "bad\\name", "bad\nname"])
def test_rejects_unsafe_task_ids_before_network(tmp_path: Path, task_id: str) -> None:
    request_path, _sources = _runtime_request(tmp_path, frame_count=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["tasks"][0]["task_id"] = task_id
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path.write_text(json.dumps(request), encoding="utf-8")
    calls = []
    with pytest.raises(
        SemanticTeacherImageEditWorkerError,
        match="semantic_teacher_runtime_frame_set_invalid",
    ):
        execute_semantic_teacher_image_edits(
            runtime_request_path=request_path,
            output_root=tmp_path / "output",
            token="fixture-token",
            opener=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
    assert calls == []
