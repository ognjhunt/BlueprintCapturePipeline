"""Match a replacement's colour to the scene it replaces, and prove it.

The 840796 texture pass was told, in prose, to make a "matte pale off-white
painted refrigerator door panel". It produced exactly that: the hue was right
to within a thousandth - warmth `0.085` against the observed `0.083` - and the
result was still visibly wrong, because "pale off-white" landed about nine
percent brighter than the appliance actually is. Nothing downstream noticed,
because nothing compared the rendered twin against the scan.

Prose cannot carry a colour. This module measures the target from the observed
pixels inside the replacement region, corrects a generated albedo map onto that
target by a per-channel gain that leaves its grain intact, and closes the loop
with a gate on the rendered composite at the frozen cameras. Reporting warmth
separately matters: a hue error and a brightness error need different fixes,
and the 840796 failure was purely the second.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest


COLOUR_FIDELITY_SCHEMA_VERSION = "replacement_colour_fidelity.v1"
DEFAULT_MAXIMUM_DELTA = 3.0
MASK_THRESHOLD = 127


class ReplacementColourFidelityError(ValueError):
    """Stable, sorted colour-fidelity failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def _load_mask(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("L")) > MASK_THRESHOLD


def _delta(first: Sequence[float], second: Sequence[float]) -> float:
    """Scalar colour distance on 0-100, comparable across channels."""

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    return float(np.linalg.norm(a - b) * 100.0 / math.sqrt(3.0))


def measure_observed_target_colour(
    *,
    observed_image_paths: Sequence[str | Path],
    region_mask_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Average the observed pixels inside the replacement region."""

    images = [Path(p).expanduser().resolve() for p in observed_image_paths]
    masks = [Path(p).expanduser().resolve() for p in region_mask_paths]
    if not images or len(images) != len(masks):
        raise ReplacementColourFidelityError(
            ["replacement_colour_input_counts_mismatch"]
        )
    total = np.zeros(3, dtype=np.float64)
    count = 0
    for image_path, mask_path in zip(images, masks):
        if not image_path.is_file() or not mask_path.is_file():
            raise ReplacementColourFidelityError(["replacement_colour_input_missing"])
        image = _load_rgb(image_path)
        mask = _load_mask(mask_path)
        if mask.shape != image.shape[:2]:
            raise ReplacementColourFidelityError(
                ["replacement_colour_mask_shape_mismatch"]
            )
        selected = image[mask]
        if selected.size:
            total += selected.sum(axis=0)
            count += selected.shape[0]
    if count == 0:
        raise ReplacementColourFidelityError(["replacement_colour_region_mask_empty"])
    target = total / count
    return {
        "target_srgb": [float(v) for v in target],
        "warmth_r_minus_b": float(target[0] - target[2]),
        "sampled_pixel_count": int(count),
        "source": "observed_scan_pixels_inside_region_mask",
        "is_measured_not_prompted": True,
    }


def correct_albedo_to_target(
    *,
    albedo_path: str | Path,
    target_srgb: Sequence[float],
    destination: str | Path,
    maximum_gain: float = 2.0,
) -> dict[str, Any]:
    """Scale a generated albedo map onto a measured target, keeping its grain."""

    from PIL import Image

    source = Path(albedo_path).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    if not source.is_file():
        raise ReplacementColourFidelityError(["replacement_colour_albedo_missing"])
    target = np.asarray(target_srgb, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ReplacementColourFidelityError(["replacement_colour_target_invalid"])

    image = _load_rgb(source)
    mean = image.reshape(-1, 3).mean(axis=0)
    if float(mean.min()) <= 1e-6:
        raise ReplacementColourFidelityError(
            ["replacement_colour_albedo_mean_degenerate"]
        )
    gain = np.clip(target / mean, 1.0 / float(maximum_gain), float(maximum_gain))
    corrected = np.clip(image * gain, 0.0, 1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((corrected * 255).astype(np.uint8)).save(output)

    before_std = float(image.std())
    after_std = float(corrected.std())
    return {
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "corrected_path": str(output),
        "corrected_sha256": _sha256(output),
        "source_mean_srgb": [float(v) for v in mean],
        "target_srgb": [float(v) for v in target],
        "corrected_mean_srgb": [
            float(v) for v in corrected.reshape(-1, 3).mean(axis=0)
        ],
        "gain": [float(v) for v in gain],
        "detail_preserved": after_std > 0.5 * before_std,
        "method": "per_channel_gain_to_measured_target",
        "generated_detail_is_not_observed_truth": True,
    }


def evaluate_colour_fidelity(
    *,
    observed_image_paths: Sequence[str | Path],
    rendered_image_paths: Sequence[str | Path],
    region_mask_paths: Sequence[str | Path],
    camera_ids: Sequence[str] = (),
    maximum_delta: float = DEFAULT_MAXIMUM_DELTA,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Compare rendered composite against the scan inside the replacement region."""

    observed = [Path(p).expanduser().resolve() for p in observed_image_paths]
    rendered = [Path(p).expanduser().resolve() for p in rendered_image_paths]
    masks = [Path(p).expanduser().resolve() for p in region_mask_paths]
    if not observed or len(observed) != len(rendered) or len(observed) != len(masks):
        raise ReplacementColourFidelityError(
            ["replacement_colour_input_counts_mismatch"]
        )
    ids = list(camera_ids) or [f"camera_{index}" for index in range(len(observed))]

    rows: list[dict[str, Any]] = []
    for index, (o_path, r_path, m_path) in enumerate(zip(observed, rendered, masks)):
        for path in (o_path, r_path, m_path):
            if not path.is_file():
                raise ReplacementColourFidelityError(
                    ["replacement_colour_input_missing"]
                )
        o_image, r_image, mask = _load_rgb(o_path), _load_rgb(r_path), _load_mask(m_path)
        if mask.shape != o_image.shape[:2] or mask.shape != r_image.shape[:2]:
            raise ReplacementColourFidelityError(
                ["replacement_colour_mask_shape_mismatch"]
            )
        if not mask.any():
            raise ReplacementColourFidelityError(
                ["replacement_colour_region_mask_empty"]
            )
        o_mean = o_image[mask].mean(axis=0)
        r_mean = r_image[mask].mean(axis=0)
        rows.append(
            {
                "camera_id": ids[index],
                "observed_srgb": [float(v) for v in o_mean],
                "rendered_srgb": [float(v) for v in r_mean],
                "delta": round(_delta(o_mean, r_mean), 4),
                "brightness_delta": round(
                    float(r_mean.mean() - o_mean.mean()) * 100.0, 4
                ),
                "warmth_delta": round(
                    float((r_mean[0] - r_mean[2]) - (o_mean[0] - o_mean[2])), 4
                ),
                "sampled_pixel_count": int(mask.sum()),
            }
        )

    worst = max(row["delta"] for row in rows)
    passed = worst <= float(maximum_delta)
    blockers = [] if passed else ["replacement_colour_delta_above_ceiling"]
    receipt: dict[str, Any] = {
        "schema_version": COLOUR_FIDELITY_SCHEMA_VERSION,
        "status": "colour_fidelity_passed" if passed else "colour_fidelity_failed",
        "passed": passed,
        "cameras": rows,
        "worst_delta": worst,
        "maximum_delta": float(maximum_delta),
        "blockers": blockers,
        "claim_boundary": {
            "colour_target_is_measured_from_observed_pixels": True,
            "generated_surface_detail_is_not_observed_truth": True,
            "matching_colour_is_not_material_measurement": True,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    if destination is not None:
        write_json(Path(destination).expanduser().resolve(), receipt)
    return json.loads(json.dumps(receipt))


__all__ = [
    "COLOUR_FIDELITY_SCHEMA_VERSION",
    "ReplacementColourFidelityError",
    "correct_albedo_to_target",
    "evaluate_colour_fidelity",
    "measure_observed_target_colour",
]
