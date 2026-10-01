"""Completed Postshot evidence adoption; new AWS execution is retired."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
from .capture_reconstruction_downstream import dispatch_postshot_to_evidence_spine
from .decision_evidence_contracts import canonical_digest, canonical_json

class CapturePostshotAllocatorError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise CapturePostshotAllocatorError(f"expected_json_object:{path.name}")
    return dict(value)


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise CapturePostshotAllocatorError(
                f"capture_postshot_immutable_output_conflict:{path.name}"
            )


def _downstream_dispatcher(
    *,
    root: Path,
    raw_root: Path,
    request: Mapping[str, Any],
    publication: Mapping[str, Any],
):
    def dispatch(*, status: Mapping[str, Any]) -> dict[str, Any]:
        payload = dispatch_postshot_to_evidence_spine(
            capture_id=str(request["capture_id"]),
            capture_digest=str(request["capture_digest"]),
            raw_root=raw_root,
            derived_root=root,
            publication=publication,
        )
        payload["terminal_status_digest"] = status["status_digest"]
        payload["dispatch_digest"] = canonical_digest(payload, digest_field="dispatch_digest")
        _write_immutable_json(root / "downstream_analysis_dispatch.json", payload)
        return payload

    return dispatch


def load_postshot_downstream_dispatch(
    request_path: str | Path,
):
    """Rebuild the downstream callback after a dispatcher process restart."""

    path = Path(request_path).expanduser().resolve(strict=True)
    payload = _read(path)
    digest = payload.get("downstream_request_digest")
    if digest != canonical_digest(payload, digest_field="downstream_request_digest"):
        raise CapturePostshotAllocatorError(
            "capture_postshot_downstream_request_digest_invalid"
        )
    publication = payload.get("publication")
    if not isinstance(publication, Mapping):
        raise CapturePostshotAllocatorError(
            "capture_postshot_downstream_publication_invalid"
        )
    return _downstream_dispatcher(
        root=Path(str(payload["derived_root"])),
        raw_root=Path(str(payload["raw_root"])),
        request={
            "capture_id": payload["capture_id"],
            "capture_digest": payload["capture_digest"],
        },
        publication=publication,
    )


def execute_postshot_capture(**kwargs: Any) -> dict[str, Any]:
    """Refuse before credential reads, staging, admission, or provider calls."""
    return {"status": "blocked", "blockers": ["aws_provider_integration_removed"],
            "provider_mutations_performed": 0, "paid_execution_available": False}


def production_allocator(**kwargs: Any) -> dict[str, Any]:
    return execute_postshot_capture(**kwargs)
