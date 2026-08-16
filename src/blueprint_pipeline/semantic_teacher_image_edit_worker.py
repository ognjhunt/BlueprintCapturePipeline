"""Execute one registry-selected hosted semantic-teacher image-edit packet.

The worker implements a transport adapter, not a model allowlist. Model identity,
snapshot, endpoint, options, prompt, and mask encoding arrive in the immutable
runtime request. This first adapter speaks the OpenAI Images Edits multipart
protocol and accepts only inline base64 PNG output, so no unbound second fetch is
introduced. It performs exactly one request per calibrated frame and no retries.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image


RUNTIME_REQUEST_SCHEMA_VERSION = "semantic_teacher_image_edit_runtime_request.v1"
RUNTIME_RESULT_SCHEMA_VERSION = "semantic_teacher_image_edit_runtime_result.v1"
SUPPORTED_ADAPTER = "openai_images_edits_v1"
MAX_PROVIDER_RESPONSE_BYTES = 48 * 1024 * 1024
MAX_GENERATED_PNG_BYTES = 32 * 1024 * 1024
MAX_INPUT_PNG_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 64 * 1024 * 1024
MAX_PROVIDER_ERROR_BYTES = 64 * 1024
RESERVED_MULTIPART_FIELDS = frozenset(
    {"image", "mask", "model", "prompt", "response_format", "size"}
)
_MULTIPART_FIELD = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_PROVIDER_ERROR_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}")
_PROVIDER_REQUEST_ID = re.compile(r"req_[A-Za-z0-9_-]{1,128}")
USAGE_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "input_text_tokens",
    "input_image_tokens",
    "output_text_tokens",
    "output_image_tokens",
)


class SemanticTeacherImageEditWorkerError(ValueError):
    """One immutable runtime input or provider response was invalid."""


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        raise SemanticTeacherImageEditWorkerError(
            "semantic_teacher_provider_redirect_rejected"
        )


def _open_no_redirect(request: Request, *, timeout: int) -> Any:
    return build_opener(_RejectRedirects()).open(request, timeout=timeout)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_digest(value: Mapping[str, Any], *, field: str) -> str:
    normalized = dict(value)
    normalized.pop(field, None)
    return "sha256:" + hashlib.sha256(
        _canonical_json(normalized).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticTeacherImageEditWorkerError(code) from exc
    if path.is_symlink() or not isinstance(value, dict):
        raise SemanticTeacherImageEditWorkerError(code)
    return value


def _bound_relative(
    root: Path, value: Any, *, code: str
) -> Path:
    if not isinstance(value, Mapping):
        raise SemanticTeacherImageEditWorkerError(code)
    relative = PurePosixPath(str(value.get("relative_path") or ""))
    if relative.is_absolute() or relative.as_posix() in {"", ".", ".."} or ".." in relative.parts:
        raise SemanticTeacherImageEditWorkerError(code)
    path = root.joinpath(*relative.parts)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != value.get("size_bytes")
        or _sha256(path) != value.get("sha256")
    ):
        raise SemanticTeacherImageEditWorkerError(code)
    return path


def _safe_component(value: Any, *, code: str) -> str:
    component = str(value or "")
    if (
        not component
        or component != component.strip()
        or not component.isprintable()
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
        or PurePosixPath(component).name != component
    ):
        raise SemanticTeacherImageEditWorkerError(code)
    return component


def _valid_default_options(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    for key, option in value.items():
        if (
            not isinstance(key, str)
            or _MULTIPART_FIELD.fullmatch(key) is None
            or key in RESERVED_MULTIPART_FIELDS
            or not isinstance(option, (str, int, float, bool))
            or (
                isinstance(option, float)
                and not math.isfinite(option)
            )
            or (isinstance(option, str) and ("\r" in option or "\n" in option))
        ):
            return False
    return True


def _valid_https_endpoint(value: Any) -> bool:
    try:
        parsed = urlsplit(str(value or ""))
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and bool(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
        and port in {None, 443}
        and bool(parsed.path)
        and not parsed.query
        and not parsed.fragment
    )


def _safe_provider_identifier(value: Any, *, token: str) -> str | None:
    """Return one inert provider discriminator, never diagnostic prose or secrets."""

    if not isinstance(value, str) or _PROVIDER_ERROR_IDENTIFIER.fullmatch(value) is None:
        return None
    lowered = value.lower()
    if (
        (token and token in value)
        or value.startswith(("sk-", "sk_"))
        or any(marker in lowered for marker in ("secret", "bearer", "api_key", "apikey"))
    ):
        return None
    return value


def _safe_provider_request_id(value: Any, *, token: str) -> str | None:
    if (
        not isinstance(value, str)
        or _PROVIDER_REQUEST_ID.fullmatch(value) is None
        or (token and token in value)
    ):
        return None
    return value


def _sanitized_http_failure(error: HTTPError, *, token: str) -> dict[str, Any]:
    """Extract only bounded, non-prose OpenAI failure discriminators.

    The response body is parsed in memory solely to recover ``error.type`` and
    ``error.code``. It is never retained. Messages, parameters, URLs, response
    headers, and authorization material are deliberately excluded.
    """

    provider_error_type: str | None = None
    provider_error_code: str | None = None
    try:
        payload = error.read(MAX_PROVIDER_ERROR_BYTES + 1)
        if len(payload) <= MAX_PROVIDER_ERROR_BYTES:
            decoded = json.loads(payload.decode("utf-8"))
            detail = decoded.get("error") if isinstance(decoded, Mapping) else None
            if isinstance(detail, Mapping):
                provider_error_type = _safe_provider_identifier(
                    detail.get("type"), token=token
                )
                provider_error_code = _safe_provider_identifier(
                    detail.get("code"), token=token
                )
    except (
        AttributeError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        pass
    headers = error.headers
    request_id = _safe_provider_request_id(
        headers.get("x-request-id") if headers is not None else None,
        token=token,
    )
    status = error.code if isinstance(error.code, int) and 100 <= error.code <= 599 else None
    result: dict[str, Any] = {
        "schema_version": "semantic_teacher_provider_failure.v1",
        "transport_error_type": "http_error",
        "http_status": status,
        "provider_error_type": provider_error_type,
        "provider_error_code": provider_error_code,
        "provider_request_id": request_id,
        "raw_provider_body_recorded": False,
        "raw_provider_headers_recorded": False,
        "raw_secret_values_recorded": False,
        "failure_digest": "",
    }
    result["failure_digest"] = _canonical_digest(result, field="failure_digest")
    return result


def _multipart(
    *,
    fields: Mapping[str, Any],
    image_bytes: bytes,
    mask_bytes: bytes,
    boundary: str,
) -> bytes:
    chunks: list[bytes] = []
    for name, raw_value in sorted(fields.items()):
        value = str(raw_value).lower() if isinstance(raw_value, bool) else str(raw_value)
        chunks.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            )
        )
    for field_name, filename, payload in (
        ("image", "input.png", image_bytes),
        ("mask", "mask.png", mask_bytes),
    ):
        chunks.extend(
            (
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{field_name}"; '
                    f'filename="{filename}"\r\n'
                ).encode(),
                b"Content-Type: image/png\r\n\r\n",
                payload,
                b"\r\n",
            )
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks)


def _normalized_usage(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError
    input_details = value.get("input_tokens_details") or {}
    output_details = value.get("output_tokens_details") or {}
    if not isinstance(input_details, Mapping) or not isinstance(output_details, Mapping):
        raise ValueError
    normalized = {
        "input_tokens": value.get("input_tokens", 0),
        "output_tokens": value.get("output_tokens", 0),
        "total_tokens": value.get("total_tokens", 0),
        "input_text_tokens": input_details.get("text_tokens", 0),
        "input_image_tokens": input_details.get("image_tokens", 0),
        "output_text_tokens": output_details.get("text_tokens", 0),
        "output_image_tokens": output_details.get("image_tokens", 0),
    }
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0
        for count in normalized.values()
    ):
        raise ValueError
    return normalized


def _usage_cost(usage: Mapping[str, int], pricing: Mapping[str, Any]) -> float:
    rates = pricing.get("usd_per_million_tokens")
    if not isinstance(rates, Mapping):
        raise ValueError
    return round(
        sum(
            float(usage[field]) * float(rates.get(field, 0))
            for field in USAGE_TOKEN_FIELDS
        )
        / 1_000_000,
        9,
    )


def _decode_inline_png(
    payload: bytes, *, expected_size: tuple[int, int]
) -> tuple[bytes, dict[str, int] | None]:
    try:
        if not isinstance(payload, bytes) or len(payload) > MAX_PROVIDER_RESPONSE_BYTES:
            raise ValueError
        value = json.loads(payload.decode("utf-8"))
        usage = _normalized_usage(value.get("usage"))
        data = value["data"]
        encoded = data[0]["b64_json"]
        if (
            not isinstance(data, list)
            or len(data) != 1
            or not isinstance(encoded, str)
            or len(encoded) > (MAX_GENERATED_PNG_BYTES * 4 // 3) + 8
        ):
            raise TypeError
        decoded = base64.b64decode(encoded, validate=True)
        if len(decoded) > MAX_GENERATED_PNG_BYTES:
            raise ValueError
        with Image.open(BytesIO(decoded)) as image:
            image.load()
            if image.format != "PNG" or image.size != expected_size:
                raise ValueError
    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise SemanticTeacherImageEditWorkerError(
            "semantic_teacher_provider_response_invalid"
        ) from exc
    return decoded, usage


def execute_semantic_teacher_image_edits(
    *,
    runtime_request_path: str | Path,
    output_root: str | Path,
    token: str,
    opener: Callable[..., Any] = _open_no_redirect,
) -> dict[str, Any]:
    """Execute every bound frame exactly once through the selected adapter."""

    request_path = Path(runtime_request_path).expanduser().resolve()
    request = _read(request_path, "semantic_teacher_runtime_request_invalid")
    backend = request.get("backend")
    execution = backend.get("execution") if isinstance(backend, Mapping) else None
    registry_entry = (
        backend.get("registry_entry") if isinstance(backend, Mapping) else None
    )
    tasks = request.get("tasks")
    default_options = (
        execution.get("default_options") if isinstance(execution, Mapping) else None
    )
    options_valid = _valid_default_options(default_options)
    pricing = execution.get("pricing_binding") if isinstance(execution, Mapping) else None
    rates = pricing.get("usd_per_million_tokens") if isinstance(pricing, Mapping) else None
    if (
        request.get("schema_version") != RUNTIME_REQUEST_SCHEMA_VERSION
        or request.get("request_digest")
        != _canonical_digest(request, field="request_digest")
        or not isinstance(execution, Mapping)
        or not isinstance(registry_entry, Mapping)
        or registry_entry.get("capability") != "semantic_teacher_image_edit"
        or backend.get("backend_entry_digest") != _canonical_digest(registry_entry, field="")
        or execution.get("adapter_id") != SUPPORTED_ADAPTER
        or execution.get("transport_kind") != "hosted_image_edit"
        or not _valid_https_endpoint(execution.get("endpoint"))
        or execution.get("masked_image_edit_supported") is not True
        or not isinstance(execution.get("input_fidelity_parameter_supported"), bool)
        or (
            execution.get("input_fidelity_parameter_supported") is False
            and "input_fidelity" in (default_options or {})
        )
        or execution.get("external_disclosure_required") is not True
        or not str(execution.get("model_snapshot") or "").strip()
        or not options_valid
        or not isinstance(pricing, Mapping)
        or not isinstance(pricing.get("usage_required"), bool)
        or not isinstance(rates, Mapping)
        or any(
            isinstance(rates.get(field, 0), bool)
            or not isinstance(rates.get(field, 0), (int, float))
            or not math.isfinite(float(rates.get(field, 0)))
            or float(rates.get(field, 0)) < 0
            for field in USAGE_TOKEN_FIELDS
        )
        or not isinstance(execution.get("supported_output_sizes"), list)
        or not execution.get("supported_output_sizes")
        or any(
            not isinstance(size, str)
            or re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", size) is None
            for size in execution.get("supported_output_sizes") or []
        )
        or not str(request.get("prompt") or "").strip()
        or request.get("retry_count") != 0
        or not isinstance(tasks, list)
        or not 1 <= len(tasks) <= 5
        or not token
        or "\n" in token
        or "\r" in token
    ):
        raise SemanticTeacherImageEditWorkerError(
            "semantic_teacher_runtime_request_invalid"
        )
    root = request_path.parent
    prepared_tasks: list[
        tuple[
            str,
            list[tuple[Mapping[str, Any], bytes, bytes, tuple[int, int]]],
        ]
    ] = []
    task_ids: set[str] = set()
    for task in tasks:
        frames = task.get("frames") if isinstance(task, Mapping) else None
        task_id = _safe_component(
            task.get("task_id") if isinstance(task, Mapping) else None,
            code="semantic_teacher_runtime_frame_set_invalid",
        )
        if task_id in task_ids or not isinstance(frames, list) or not frames:
            raise SemanticTeacherImageEditWorkerError(
                "semantic_teacher_runtime_frame_set_invalid"
            )
        task_ids.add(task_id)
        prepared_frames: list[tuple[Mapping[str, Any], bytes, bytes, tuple[int, int]]] = []
        camera_ids: set[str] = set()
        for expected_index, frame in enumerate(frames):
            if not isinstance(frame, Mapping) or frame.get("frame_index") != expected_index:
                raise SemanticTeacherImageEditWorkerError(
                    "semantic_teacher_runtime_frame_set_invalid"
                )
            camera_id = _safe_component(
                frame.get("camera_id"),
                code="semantic_teacher_runtime_frame_set_invalid",
            )
            if camera_id in camera_ids:
                raise SemanticTeacherImageEditWorkerError(
                    "semantic_teacher_runtime_frame_set_invalid"
                )
            camera_ids.add(camera_id)
            image_path = _bound_relative(
                root,
                frame.get("input_rgb"),
                code="semantic_teacher_runtime_input_invalid",
            )
            mask_path = _bound_relative(
                root,
                frame.get("edit_mask"),
                code="semantic_teacher_runtime_mask_invalid",
            )
            try:
                image_bytes = image_path.read_bytes()
                mask_bytes = mask_path.read_bytes()
                if (
                    len(image_bytes) > MAX_INPUT_PNG_BYTES
                    or len(mask_bytes) > MAX_INPUT_PNG_BYTES
                    or "sha256:" + hashlib.sha256(image_bytes).hexdigest()
                    != frame["input_rgb"]["sha256"]
                    or "sha256:" + hashlib.sha256(mask_bytes).hexdigest()
                    != frame["edit_mask"]["sha256"]
                ):
                    raise ValueError
                with Image.open(BytesIO(image_bytes)) as image:
                    image.load()
                    expected_size = image.size
                    image_format = image.format
                with Image.open(BytesIO(mask_bytes)) as mask:
                    mask.load()
                    mask_size = mask.size
                    mask_format = mask.format
            except (OSError, ValueError) as exc:
                raise SemanticTeacherImageEditWorkerError(
                    "semantic_teacher_runtime_frame_media_invalid"
                ) from exc
            if (
                image_format != "PNG"
                or mask_format != "PNG"
                or mask_size != expected_size
                or expected_size[0] * expected_size[1] > MAX_IMAGE_PIXELS
                or f"{expected_size[0]}x{expected_size[1]}"
                not in execution["supported_output_sizes"]
            ):
                raise SemanticTeacherImageEditWorkerError(
                    "semantic_teacher_runtime_frame_media_invalid"
                )
            prepared_frames.append((frame, image_bytes, mask_bytes, expected_size))
        prepared_tasks.append((task_id, prepared_frames))
    output = Path(output_root).expanduser().resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise SemanticTeacherImageEditWorkerError(
            "semantic_teacher_runtime_output_not_empty"
        )
    output.mkdir(parents=True, exist_ok=True)
    task_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, int]] = []
    editor_costs: list[float] = []
    request_count = 0
    for task_index, (task_id, frames) in enumerate(prepared_tasks):
        task_output = output / "tasks" / task_id
        task_output.mkdir(parents=True)
        frame_rows: list[dict[str, Any]] = []
        for expected_index, (frame, image_bytes, mask_bytes, expected_size) in enumerate(frames):
            fields = {
                **dict(execution["default_options"]),
                "model": execution["model_snapshot"],
                "prompt": request["prompt"],
                "size": f"{expected_size[0]}x{expected_size[1]}",
            }
            boundary = "blueprint" + request["request_digest"].split(":", 1)[-1][:24]
            body = _multipart(
                fields=fields,
                image_bytes=image_bytes,
                mask_bytes=mask_bytes,
                boundary=boundary,
            )
            http_request = Request(
                str(execution["endpoint"]),
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Accept": "application/json",
                },
            )
            request_count += 1
            try:
                with opener(http_request, timeout=300) as response:
                    if (
                        not hasattr(response, "geturl")
                        or response.geturl() != str(execution["endpoint"])
                    ):
                        raise SemanticTeacherImageEditWorkerError(
                            "semantic_teacher_provider_redirect_rejected"
                        )
                    payload = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
                    generated, usage = _decode_inline_png(
                        payload, expected_size=expected_size
                    )
                destination = task_output / f"{expected_index:05d}.png"
                destination.write_bytes(generated)
            except (HTTPError, OSError, SemanticTeacherImageEditWorkerError) as exc:
                provider_failure = (
                    _sanitized_http_failure(exc, token=token)
                    if isinstance(exc, HTTPError)
                    else None
                )
                blocker = (
                    "semantic_teacher_provider_http_error"
                    if isinstance(exc, HTTPError)
                    else (
                        str(exc)
                        if isinstance(exc, SemanticTeacherImageEditWorkerError)
                        else "semantic_teacher_provider_request_failed"
                    )
                )
                frame_rows.append(
                    {
                        "frame_index": expected_index,
                        "camera_id": str(frame.get("camera_id") or ""),
                        "terminal_state": "failed_after_request_attempt",
                        "failure_code": blocker,
                        "source_rgb_sha256": frame["input_rgb"]["sha256"],
                        "edit_mask_sha256": frame["edit_mask"]["sha256"],
                        "semantic_teacher_frame": None,
                        "provider_usage": None,
                        "computed_editor_cost_usd": None,
                        "billing_qualified": False,
                        "provider_failure": provider_failure,
                    }
                )
                for pending_index in range(expected_index + 1, len(frames)):
                    pending = frames[pending_index][0]
                    frame_rows.append(
                        {
                            "frame_index": pending_index,
                            "camera_id": str(pending.get("camera_id") or ""),
                            "terminal_state": "not_attempted_after_terminal_failure",
                            "failure_code": None,
                            "source_rgb_sha256": pending["input_rgb"]["sha256"],
                            "edit_mask_sha256": pending["edit_mask"]["sha256"],
                            "semantic_teacher_frame": None,
                            "provider_usage": None,
                            "computed_editor_cost_usd": None,
                            "billing_qualified": False,
                            "provider_failure": None,
                        }
                    )
                task_rows.append(
                    {
                        "task_id": task_id,
                        "camera_count": len(frames),
                        "frames": frame_rows,
                    }
                )
                for pending_task_id, pending_frames in prepared_tasks[task_index + 1 :]:
                    task_rows.append(
                        {
                            "task_id": pending_task_id,
                            "camera_count": len(pending_frames),
                            "frames": [
                                {
                                    "frame_index": pending_index,
                                    "camera_id": str(pending[0].get("camera_id") or ""),
                                    "terminal_state": "not_attempted_after_terminal_failure",
                                    "failure_code": None,
                                    "source_rgb_sha256": pending[0]["input_rgb"]["sha256"],
                                    "edit_mask_sha256": pending[0]["edit_mask"]["sha256"],
                                    "semantic_teacher_frame": None,
                                    "provider_usage": None,
                                    "computed_editor_cost_usd": None,
                                    "billing_qualified": False,
                                    "provider_failure": None,
                                }
                                for pending_index, pending in enumerate(pending_frames)
                            ],
                        }
                    )
                partial_frames = [
                    _record
                    for row in task_rows
                    for terminal in row["frames"]
                    if isinstance((_record := terminal.get("semantic_teacher_frame")), Mapping)
                ]
                failed: dict[str, Any] = {
                    "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
                    "status": "failed_with_retained_partial_inventory",
                    "source_runtime_request_digest": request["request_digest"],
                    "backend_id": backend["registry_entry"]["backend_id"],
                    "backend_entry_digest": backend["backend_entry_digest"],
                    "adapter_id": execution["adapter_id"],
                    "model_snapshot": execution["model_snapshot"],
                    "task_count": len(prepared_tasks),
                    "request_count": request_count,
                    "attempted_request_count": request_count,
                    "successful_request_count": len(partial_frames),
                    "retry_count": 0,
                    "blockers": [blocker],
                    "terminal_provider_failure": provider_failure,
                    "tasks": task_rows,
                    "partial_png_inventory": partial_frames,
                    "provider_usage_totals": {
                        field: sum(row[field] for row in usage_rows)
                        for field in USAGE_TOKEN_FIELDS
                    },
                    "computed_editor_cost_usd": round(sum(editor_costs), 9),
                    "billing_usage_required": pricing["usage_required"],
                    "billing_qualified": False,
                    "raw_secret_values_recorded": False,
                    "canonical_source_altered": False,
                    "appearance_qualified": False,
                    "result_digest": "",
                }
                failed["result_digest"] = _canonical_digest(
                    failed, field="result_digest"
                )
                (output / f"{RUNTIME_RESULT_SCHEMA_VERSION}.json").write_text(
                    _canonical_json(failed) + "\n", encoding="utf-8"
                )
                raise SemanticTeacherImageEditWorkerError(blocker) from exc
            usage_cost = _usage_cost(usage, pricing) if usage is not None else None
            if usage is not None:
                usage_rows.append(usage)
                editor_costs.append(float(usage_cost))
            frame_rows.append(
                {
                    "frame_index": expected_index,
                    "camera_id": str(frame.get("camera_id") or ""),
                    "terminal_state": "completed_unreviewed_candidate",
                    "failure_code": None,
                    "source_rgb_sha256": frame["input_rgb"]["sha256"],
                    "edit_mask_sha256": frame["edit_mask"]["sha256"],
                    "semantic_teacher_frame": {
                        "relative_path": destination.relative_to(output).as_posix(),
                        "size_bytes": destination.stat().st_size,
                        "sha256": _sha256(destination),
                    },
                    "provider_usage": usage,
                    "computed_editor_cost_usd": usage_cost,
                    "billing_qualified": usage is not None,
                    "provider_failure": None,
                    "visual_reviewed": False,
                    "multiview_consistency_qualified": False,
                }
            )
        task_rows.append(
            {"task_id": task_id, "camera_count": len(frame_rows), "frames": frame_rows}
        )
    usage_required = pricing["usage_required"]
    billing_qualified = len(usage_rows) == request_count
    result: dict[str, Any] = {
        "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
        "status": (
            "completed_unreviewed_semantic_teacher_candidates"
            if billing_qualified or not usage_required
            else "completed_candidates_billing_unqualified"
        ),
        "source_runtime_request_digest": request["request_digest"],
        "backend_id": backend["registry_entry"]["backend_id"],
        "backend_entry_digest": backend["backend_entry_digest"],
        "adapter_id": execution["adapter_id"],
        "model_snapshot": execution["model_snapshot"],
        "task_count": len(task_rows),
        "request_count": request_count,
        "attempted_request_count": request_count,
        "successful_request_count": request_count,
        "provider_usage_totals": {
            field: sum(row[field] for row in usage_rows) for field in USAGE_TOKEN_FIELDS
        },
        "computed_editor_cost_usd": round(sum(editor_costs), 9),
        "billing_usage_required": usage_required,
        "billing_qualified": billing_qualified,
        "blockers": (
            [] if billing_qualified or not usage_required else ["provider_usage_missing"]
        ),
        "retry_count": 0,
        "tasks": task_rows,
        "raw_secret_values_recorded": False,
        "canonical_source_altered": False,
        "simulator_or_policy_output_is_physical_evidence": False,
        "appearance_qualified": False,
        "result_digest": "",
    }
    result["result_digest"] = _canonical_digest(result, field="result_digest")
    (output / f"{RUNTIME_RESULT_SCHEMA_VERSION}.json").write_text(
        _canonical_json(result) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-request", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--token-env", default="BLUEPRINT_IMAGE_EDITOR_TOKEN")
    args = parser.parse_args(argv)
    output = Path(args.output_root).expanduser().resolve()
    try:
        result = execute_semantic_teacher_image_edits(
            runtime_request_path=args.runtime_request,
            output_root=output,
            token=os.environ.get(args.token_env, ""),
        )
    except (OSError, SemanticTeacherImageEditWorkerError) as exc:
        retained_result = output / f"{RUNTIME_RESULT_SCHEMA_VERSION}.json"
        if retained_result.is_file():
            return 1
        output.mkdir(parents=True, exist_ok=True)
        blocker = (
            str(exc)
            if isinstance(exc, SemanticTeacherImageEditWorkerError)
            else "semantic_teacher_runtime_execution_failed"
        )
        blocked: dict[str, Any] = {
            "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": [blocker],
            "retry_count": 0,
            "raw_secret_values_recorded": False,
            "canonical_source_altered": False,
            "appearance_qualified": False,
            "result_digest": "",
        }
        blocked["result_digest"] = _canonical_digest(blocked, field="result_digest")
        (output / f"{RUNTIME_RESULT_SCHEMA_VERSION}.json").write_text(
            _canonical_json(blocked) + "\n", encoding="utf-8"
        )
        return 1
    return (
        0
        if result.get("status") == "completed_unreviewed_semantic_teacher_candidates"
        else 1
    )


__all__ = [
    "RUNTIME_REQUEST_SCHEMA_VERSION",
    "RUNTIME_RESULT_SCHEMA_VERSION",
    "SUPPORTED_ADAPTER",
    "SemanticTeacherImageEditWorkerError",
    "execute_semantic_teacher_image_edits",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
