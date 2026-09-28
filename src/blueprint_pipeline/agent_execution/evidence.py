"""Digest-bound image inspection tools for SAM, appearance and episode agents.

Tools address admitted evidence by id, never by model-supplied file paths.
Crops preserve source pixel identity and are labelled as derived views; they
cannot create new observations or waive a mandatory review view.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
from io import BytesIO
import math
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .contracts import AgentExecutionError, AgentTool, ToolContext, canonical_json, digest


@dataclass(frozen=True)
class ImageEvidence:
    image_id: str
    path: Path
    sha256: str
    camera_id: str
    role: str
    time_seconds: float | None = None
    observation_index: int | None = None


class ImageEvidenceCatalog:
    def __init__(
        self,
        *,
        root: str | Path,
        images: Sequence[ImageEvidence],
        admitted_digests: frozenset[str],
        maximum_image_bytes: int = 16_000_000,
        maximum_image_pixels: int = 32_000_000,
        defer_path_validation: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.images = MappingProxyType({image.image_id: image for image in images})
        self._admitted_digests = frozenset(admitted_digests)
        if not images or len(self.images) != len(images):
            raise ValueError("agent_image_inventory_empty_or_duplicate")
        if not 1 <= maximum_image_bytes <= 128_000_000 or not 1 <= maximum_image_pixels <= 64_000_000:
            raise ValueError("agent_image_limits_invalid")
        self.maximum_image_bytes = maximum_image_bytes
        self.maximum_image_pixels = maximum_image_pixels
        for image in images:
            if not image.image_id or not image.camera_id or not image.role:
                raise ValueError("agent_image_identity_missing")
            if image.sha256 not in admitted_digests:
                raise AgentExecutionError("agent_image_disclosure_not_admitted")
            if image.time_seconds is not None and (
                type(image.time_seconds) not in (int, float)
                or not math.isfinite(image.time_seconds) or image.time_seconds < 0
            ):
                raise ValueError("agent_image_timestamp_invalid")
            if not defer_path_validation:
                self._safe_path(image)

    def _safe_path(self, image: ImageEvidence) -> Path:
        path = image.path.expanduser()
        if not path.is_absolute():
            path = self.root / path
        if not path.resolve().is_relative_to(self.root) or not path.is_file():
            raise AgentExecutionError("agent_image_path_outside_evidence")
        if any(parent.is_symlink() for parent in (path, *path.parents) if parent != self.root
               and parent.is_relative_to(self.root)):
            raise AgentExecutionError("agent_image_symlink_not_admitted")
        return path

    @staticmethod
    def metadata(image: ImageEvidence) -> dict[str, Any]:
        return {
            "image_id": image.image_id, "source_sha256": image.sha256,
            "camera_id": image.camera_id, "role": image.role,
            "time_seconds": image.time_seconds, "observation_index": image.observation_index,
        }

    def inventory(self, *, offset: int = 0, limit: int = 100) -> dict[str, Any]:
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise AgentExecutionError("agent_image_inventory_window_invalid")
        images = list(self.images.values())
        selected = images[offset:offset + limit]
        return {
            "schema_version": "blueprint_agent_image_inventory.v1",
            "images": [self.metadata(image) for image in selected],
            "total_count": len(images),
            "next_offset": offset + len(selected) if offset + len(selected) < len(images) else None,
            "inventory_digest": digest([self.metadata(image) for image in images]),
        }

    def inspect(self, image_id: str, *, crop: list[int] | None = None) -> list[dict[str, Any]]:
        image = self.images.get(image_id)
        if image is None:
            raise AgentExecutionError("agent_image_not_in_inventory")
        if image.sha256 not in self._admitted_digests:
            raise AgentExecutionError("agent_image_disclosure_not_admitted")
        path = self._safe_path(image)
        with path.open("rb") as stream:
            source = stream.read(self.maximum_image_bytes + 1)
        if len(source) > self.maximum_image_bytes:
            raise AgentExecutionError("agent_image_byte_limit_exceeded")
        source_digest = "sha256:" + hashlib.sha256(source).hexdigest()
        if source_digest != image.sha256:
            raise AgentExecutionError("agent_image_changed_before_disclosure")
        from PIL import Image, PngImagePlugin

        try:
            with Image.open(BytesIO(source)) as decoded:
                width, height = decoded.size
                if decoded.format != "PNG" or width * height > self.maximum_image_pixels:
                    raise AgentExecutionError("agent_image_format_or_dimensions_invalid")
                if (getattr(decoded, "n_frames", 1) != 1 or decoded.mode not in {"L", "LA", "RGB", "RGBA"}
                        or decoded.getexif().get(274, 1) != 1):
                    raise AgentExecutionError("agent_image_pixel_interpretation_not_qualified")
                decoded.load()
                if crop is None:
                    view = source
                    rectangle = [0, 0, width, height]
                else:
                    if (len(crop) != 4 or any(type(value) is not int for value in crop)
                            or not 0 <= crop[0] < crop[2] <= width
                            or not 0 <= crop[1] < crop[3] <= height):
                        raise AgentExecutionError("agent_image_crop_invalid")
                    rendered = BytesIO()
                    color_info = PngImagePlugin.PngInfo()
                    if "gamma" in decoded.info:
                        color_info.add(b"gAMA", struct.pack(">I", round(decoded.info["gamma"] * 100000)))
                    if "srgb" in decoded.info:
                        color_info.add(b"sRGB", bytes([decoded.info["srgb"]]))
                    if "chromaticity" in decoded.info:
                        color_info.add(b"cHRM", struct.pack(">8I", *[
                            round(value * 100000) for value in decoded.info["chromaticity"]
                        ]))
                    decoded.crop(tuple(crop)).save(
                        rendered, format="PNG", pnginfo=color_info,
                        **{key: decoded.info[key] for key in ("icc_profile", "transparency") if key in decoded.info},
                    )
                    view = rendered.getvalue()
                    rectangle = list(crop)
                source_pixels = decoded.crop(tuple(rectangle))
                pixel_identity = canonical_json({
                    "mode": source_pixels.mode, "size": list(source_pixels.size),
                }).encode() + b"\x00" + source_pixels.tobytes()
                source_pixel_digest = "sha256:" + hashlib.sha256(pixel_identity).hexdigest()
                with Image.open(BytesIO(view)) as emitted:
                    emitted.load()
                    if any(decoded.info.get(key) != emitted.info.get(key)
                           for key in ("gamma", "srgb", "chromaticity", "icc_profile", "transparency")):
                        raise AgentExecutionError("agent_image_color_interpretation_changed")
                    emitted_identity = canonical_json({
                        "mode": emitted.mode, "size": list(emitted.size),
                    }).encode() + b"\x00" + emitted.tobytes()
                view_pixel_digest = "sha256:" + hashlib.sha256(emitted_identity).hexdigest()
                if source_pixel_digest != view_pixel_digest:
                    raise AgentExecutionError("agent_image_crop_pixels_changed")
        except (OSError, ValueError) as exc:
            if isinstance(exc, AgentExecutionError):
                raise
            raise AgentExecutionError("agent_image_decode_failed") from exc
        metadata = {
            "schema_version": "blueprint_agent_image_observation.v1",
            **self.metadata(image),
            "view_sha256": "sha256:" + hashlib.sha256(view).hexdigest(),
            "source_dimensions": [width, height], "pixel_rectangle": rectangle,
            "derived_crop": crop is not None,
            "new_source_observation": False,
            "evidence_reference_digest": source_digest,
            "pixel_identity_encoding": "canonical_json_mode_size_nul_raw_row_major_v1",
            "source_rectangle_pixels_sha256": source_pixel_digest,
            "view_pixels_sha256": view_pixel_digest,
        }
        return [
            {"type": "input_text", "text": canonical_json(metadata)},
            {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(view).decode()},
        ]

    def tools(self) -> tuple[AgentTool, ...]:
        inventory_digest = self.inventory(limit=1)["inventory_digest"]

        def check_disclosure(context: ToolContext) -> None:
            if not {image.sha256 for image in self.images.values()} <= set(context.allowed_input_digests):
                raise AgentExecutionError("agent_image_task_disclosure_mismatch")

        def inventory(arguments: Mapping[str, Any], context: ToolContext):
            check_disclosure(context)
            return self.inventory(offset=arguments["offset"], limit=arguments["limit"])

        def inspect(arguments: Mapping[str, Any], context: ToolContext):
            check_disclosure(context)
            return self.inspect(arguments["image_id"], crop=arguments["crop"])

        return (
            AgentTool(
                "list_evidence_images", "1",
                "List admitted images and their camera/time identities. Inventory: " + inventory_digest,
                {"type": "object", "properties": {
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                }, "required": ["offset", "limit"], "additionalProperties": False},
                "read_only", inventory,
            ),
            AgentTool(
                "inspect_evidence_image", "1",
                "Inspect an admitted image at its original resolution or request a pixel crop. "
                "Cite evidence_reference_digest; a crop is not a new observation. Inventory: " + inventory_digest,
                {"type": "object", "properties": {
                    "image_id": {"type": "string", "minLength": 1, "maxLength": 192},
                    "crop": {"anyOf": [
                        {"type": "null"},
                        {"type": "array", "items": {"type": "integer", "minimum": 0},
                         "minItems": 4, "maxItems": 4},
                    ]},
                }, "required": ["image_id", "crop"], "additionalProperties": False},
                "read_only", inspect,
            ),
        )
