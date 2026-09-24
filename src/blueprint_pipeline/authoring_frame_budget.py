"""ADP-009: reference frames sized to fit the authoring provider's request limits.

Coverage retains up to 12 full-resolution upright frames. Every provider path
sends each one inline as PNG under a fixed per-request ceiling: the Claude
authoring invoker reserves 4,784 tokens per image inside an 80,000-token
request and 30 MB of JSON; the OpenAI invoker charges 1.2 tokens per 32 px
patch of the unresized image inside the same 80,000 tokens. Frames therefore
go to the builder as deterministic derivatives: the provider's frame cap is
applied in coverage priority order without dropping the only frame of a
required part, part state or the body-depth view, and each kept frame is
re-encoded at the largest long side on a fixed ladder that meets its byte
share. Both the retained original and the transmitted derivative digest are
recorded. What cannot fit fails closed with a typed code.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .local_reconstruction_adapters import _sha256_file
from .task_evaluation_supervisor.sdk_image_tools import image_context_tokens

PROVIDERS = ("openai", "anthropic")
# Half of the 80,000-token per-request authoring ceiling; the rest carries the
# prompt, schema, construction constraints and review renders.
SOURCE_FRAME_TOKEN_BUDGET = 40_000
# Raw PNG bytes; base64 makes this 20 MB of the 30 MB Claude request bound.
TOTAL_SOURCE_FRAME_BYTES = 15_000_000
MAX_FRAME_BYTES = 4_000_000
LONG_SIDES_PX = (1568, 1344, 1152, 1024, 896, 768)
MAX_FRAMES = 12
DERIVATIVE = "pil_rgb_lanczos_long_side_png_compress9_v1"


class FrameBudgetError(ValueError):
    """Reference frames cannot fit the provider's request without losing required coverage."""


def _frame_tokens(provider: str, width: int, height: int) -> int:
    if provider == "anthropic":
        from .claude_opus_authoring_invoker import _MAX_IMAGE_TOKENS
        return _MAX_IMAGE_TOKENS
    if provider == "openai":
        return image_context_tokens(width, height)
    raise FrameBudgetError("authoring_frame_provider_unsupported:" + str(provider))


def frame_cap(provider: str) -> int:
    """Frames that fit the token budget even at the largest long side, square."""
    side = LONG_SIDES_PX[0]
    return min(MAX_FRAMES, SOURCE_FRAME_TOKEN_BUDGET // _frame_tokens(provider, side, side))


def _groups(rows: Sequence[Mapping[str, Any]], required_parts: Sequence[Mapping[str, Any]],
            depth_frame_ids: Sequence[str]) -> dict[str, set[str]]:
    ids = {row["frame_id"] for row in rows}
    groups = {"part:" + part["part_id"]: set(part["observed_frame_ids"]) & ids for part in required_parts}
    for row in rows:
        if row["part_state"] != "not_visible":
            groups.setdefault("state:" + row["part_state"], set()).add(row["frame_id"])
    groups["body_depth"] = set(depth_frame_ids) & ids
    empty = sorted(name for name, members in groups.items() if not members)
    if empty:
        raise FrameBudgetError("authoring_frames_coverage_missing:" + empty[0])
    return groups


def choose_frames(rows: Sequence[Mapping[str, Any]], *, required_parts: Sequence[Mapping[str, Any]],
                  depth_frame_ids: Sequence[str], cap: int) -> list[dict[str, Any]]:
    """Keep at most ``cap`` frames by ``selection_rank``, never the only view of a part, state or depth."""
    if any(not isinstance(row.get("selection_rank"), int) for row in rows):
        raise FrameBudgetError("authoring_frames_priority_missing")
    groups = _groups(rows, required_parts, depth_frame_ids)
    ranked = sorted(rows, key=lambda row: row["selection_rank"])
    chosen = {next(iter(members)) for members in groups.values() if len(members) == 1}
    for row in ranked:
        if any(row["frame_id"] in members and not members & chosen for members in groups.values()):
            chosen.add(row["frame_id"])
    if len(chosen) > cap:
        raise FrameBudgetError("authoring_frames_required_coverage_exceeds_provider_cap")
    for row in ranked:
        if len(chosen) >= cap:
            break
        chosen.add(row["frame_id"])
    return [dict(row) for row in rows if row["frame_id"] in chosen]


def _derivative(path: Path, *, byte_cap: int) -> tuple[bytes, dict[str, Any]]:
    with Image.open(path) as image:
        image.load()
        original = image.size
        rgb = image.convert("RGB")
    for side in LONG_SIDES_PX:
        scale = min(1.0, side / max(original))
        size = (max(1, round(original[0] * scale)), max(1, round(original[1] * scale)))
        picture = rgb if size == original else rgb.resize(size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG", compress_level=9)
        data = buffer.getvalue()
        if len(data) <= byte_cap:
            return data, {"source_width": original[0], "source_height": original[1], "width": size[0],
                          "height": size[1], "long_side_px": max(size), "bytes": len(data),
                          "format": "png", "derivative": DERIVATIVE}
    raise FrameBudgetError("authoring_frame_bytes_exceed_budget:" + path.name)


def fit_reference_frames(contract: Mapping[str, Any], *, provider: str, output_root: Path) -> dict[str, Any]:
    """Trim and re-encode ``reference_frames`` for ``provider``; ids elsewhere follow the kept frames."""
    rows = list(contract["reference_frames"])
    cap = frame_cap(provider)
    kept = choose_frames(rows, required_parts=contract["required_parts"],
                         depth_frame_ids=contract["body_depth"]["frame_ids"], cap=cap)
    byte_cap = min(MAX_FRAME_BYTES, TOTAL_SOURCE_FRAME_BYTES // len(kept))
    output_root.mkdir(parents=True, exist_ok=True)
    frames, tokens = [], 0
    for row in kept:
        source = Path(row["path"])
        if _sha256_file(source) != row["sha256"]:
            raise FrameBudgetError("authoring_frame_source_changed:" + row["frame_id"])
        data, record = _derivative(source, byte_cap=byte_cap)
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        target = output_root / f"{row['frame_id']}-{digest[7:23]}.png"
        if not target.is_file() or _sha256_file(target) != digest:
            target.write_bytes(data)
        tokens += _frame_tokens(provider, record["width"], record["height"])
        frames.append({**row, "path": str(target), "sha256": digest,
                       "transmission": {**record, "source_path": row["path"], "source_sha256": row["sha256"]}})
    if tokens > SOURCE_FRAME_TOKEN_BUDGET:
        raise FrameBudgetError("authoring_frames_token_budget_exceeded")
    ids = {row["frame_id"] for row in frames}
    return {**contract, "reference_frames": frames,
            "required_parts": [{**part, "observed_frame_ids": [v for v in part["observed_frame_ids"] if v in ids]}
                               for part in contract["required_parts"]],
            "body_depth": {**contract["body_depth"],
                           "frame_ids": [v for v in contract["body_depth"]["frame_ids"] if v in ids]},
            "reference_frame_budget": {
                "provider": provider, "frame_cap": cap, "token_budget": SOURCE_FRAME_TOKEN_BUDGET,
                "tokens_reserved": tokens, "total_bytes": sum(row["transmission"]["bytes"] for row in frames),
                "per_frame_byte_cap": byte_cap, "total_byte_budget": TOTAL_SOURCE_FRAME_BYTES,
                "dropped_frame_ids": [row["frame_id"] for row in rows if row["frame_id"] not in ids],
                "policy": "selection_rank_order_keeping_every_part_state_and_depth_view"}}


def check_transmitted_frames(paths: Sequence[Path], *, provider: str) -> dict[str, int]:
    """Refuse retained frames that would not fit one provider request; nothing is resized here."""
    if not paths or len(paths) > frame_cap(provider):
        raise FrameBudgetError("authoring_frames_count_exceeds_provider_cap")
    total, tokens = 0, 0
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
                if image.format != "PNG" or max(image.size) > LONG_SIDES_PX[0]:
                    raise FrameBudgetError("authoring_frame_not_a_bounded_png:" + Path(path).name)
                size = image.size
        except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
            raise FrameBudgetError("authoring_frame_not_a_bounded_png:" + Path(path).name) from exc
        count = Path(path).stat().st_size
        if count > MAX_FRAME_BYTES:
            raise FrameBudgetError("authoring_frame_bytes_exceed_budget:" + Path(path).name)
        total += count
        tokens += _frame_tokens(provider, *size)
    if total > TOTAL_SOURCE_FRAME_BYTES or tokens > SOURCE_FRAME_TOKEN_BUDGET:
        raise FrameBudgetError("authoring_frames_request_budget_exceeded")
    return {"frames": len(paths), "bytes": total, "tokens_reserved": tokens}
