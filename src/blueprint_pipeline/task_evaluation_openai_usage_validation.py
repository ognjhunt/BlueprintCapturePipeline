"""Read-only prompt-cache and retained usage validators, independent of sync transport."""
from __future__ import annotations

import hashlib
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROMPT_CACHE_INTENT_FIELDS = (
    "expected_proposal_reuse_probability",
    "expected_visual_review_reuse_probability",
    "expected_proposal_reuse_count",
    "expected_visual_review_reuse_count",
)


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class OpenAIInferenceUsageError(ValueError):
    """The retained usage or cache-policy evidence is incomplete."""


def prompt_cache_intent_values(
    *,
    proposal_probability: float,
    visual_review_probability: float,
    proposal_count: int,
    visual_review_count: int,
) -> dict[str, int | float]:
    return {
        "expected_proposal_reuse_probability": float(proposal_probability),
        "expected_visual_review_reuse_probability": float(
            visual_review_probability
        ),
        "expected_proposal_reuse_count": int(proposal_count),
        "expected_visual_review_reuse_count": int(visual_review_count),
    }


def prompt_cache_placement_intent_valid(placement: Mapping[str, Any]) -> bool:
    try:
        return (
            0
            <= float(placement.get("expected_proposal_reuse_probability", -1.0))
            <= 1
            and 0
            <= float(
                placement.get("expected_visual_review_reuse_probability", -1.0)
            )
            <= 1
            and 0 <= int(placement.get("expected_proposal_reuse_count", -1)) <= 20
            and 0
            <= int(placement.get("expected_visual_review_reuse_count", -1))
            <= 20
        )
    except (TypeError, ValueError):
        return False


def placement_prompt_cache_settings(
    placement: Mapping[str, Any],
) -> dict[str, int | float]:
    return {
        "expected_proposal_reuse_probability": float(
            placement["expected_proposal_reuse_probability"]
        ),
        "expected_visual_review_reuse_probability": float(
            placement["expected_visual_review_reuse_probability"]
        ),
        "expected_proposal_reuse_count": int(
            placement["expected_proposal_reuse_count"]
        ),
        "expected_visual_review_reuse_count": int(
            placement["expected_visual_review_reuse_count"]
        ),
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser()
    if not resolved.is_absolute() or resolved.is_symlink() or not resolved.is_file():
        raise OpenAIInferenceUsageError("openai_inference_usage_artifact_invalid")
    payload = resolved.read_bytes()
    metadata = resolved.stat()
    return {
        "path": str(resolved),
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size_bytes": metadata.st_size,
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
    }


def artifact_record_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        return dict(value) == _artifact_record(Path(str(value.get("path") or "")))
    except (OSError, OpenAIInferenceUsageError):
        return False


def result_projection_valid(result: Mapping[str, Any]) -> bool:
    packet = result.get("openai_inference_usage_packet")
    sync = result.get("openai_inference_usage_webapp_sync")
    return bool(
        artifact_record_valid(packet)
        and isinstance(sync, Mapping)
        and isinstance(sync.get("required"), bool)
        and (
            sync.get("status") == "succeeded"
            if sync.get("required") is True
            else sync.get("status") in {"succeeded", "skipped"}
        )
        and artifact_record_valid(sync.get("artifact"))
        and _DIGEST.fullmatch(str(sync.get("packet_digest") or ""))
        and 1 <= int(sync.get("call_count") or 0) <= 8
    )
