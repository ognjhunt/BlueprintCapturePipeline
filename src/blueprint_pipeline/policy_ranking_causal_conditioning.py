"""Label-blind causal/placebo diagnostics for OSCAR policy conditioning.

This module does not treat the released WAM as its own answer key.  It asks the
narrower falsification question: is generated-pixel motion temporally aligned
with the independently recorded DROID action trace more strongly than frozen
zero, shuffle, reversal, circular-shift, and within-session swapped-action
controls?  The skeleton overlay and the residual generated scene are measured
separately so motion in the visible conditioning annotation cannot masquerade
as world-model action following.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

import cv2  # type: ignore[import-not-found]
import numpy as np

from .common import write_json
from .policy_ranking_thesis import canonical_sha256, file_sha256
from .policy_ranking_wam_validity import load_restricted_roboarena_npz


SCHEMA_VERSION = "policy_ranking_causal_conditioning_report.v1"
DEFAULT_MEANINGFUL_MARGIN = 0.05
DEFAULT_MIN_ORIGINAL_CORRELATION = 0.10
DEFAULT_VALIDITY_PASS_RATE = 0.80


def _overlay_mask(frame: np.ndarray) -> np.ndarray:
    """Conservatively mask the bright OSCAR skeleton-rendering palette."""

    blue, green, red = cv2.split(frame)
    yellow = (red > 175) & (green > 175) & (blue < 120)
    red_line = (red > 175) & (green < 155) & (blue < 155)
    green_line = (green > 170) & (red < 190) & (blue < 190)
    violet = (red > 110) & (blue > 110) & (green < 190)
    mask = (yellow | red_line | green_line | violet).astype(np.uint8) * 255
    return cv2.dilate(mask, np.ones((7, 7), dtype=np.uint8), iterations=1) > 0


def generated_motion_channels(
    video_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(video_path).resolve()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("video_open_failed")
    previous: np.ndarray | None = None
    previous_mask: np.ndarray | None = None
    total: list[float] = []
    overlay: list[float] = []
    residual: list[float] = []
    coverages: list[float] = []
    frame_count = width = height = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            height, full_width = frame.shape[:2]
            width = full_width // 2
            generated = cv2.resize(frame[:, :width], (160, 120), interpolation=cv2.INTER_AREA)
            current = generated.astype(np.float32) / 255.0
            mask = _overlay_mask(generated)
            coverages.append(float(np.mean(mask)))
            if previous is not None and previous_mask is not None:
                difference = np.mean(np.abs(current - previous), axis=2)
                union = mask | previous_mask
                total.append(float(np.mean(difference)))
                overlay.append(float(np.mean(difference[union])) if np.any(union) else 0.0)
                residual.append(float(np.mean(difference[~union])) if np.any(~union) else 0.0)
            previous = current
            previous_mask = mask
            frame_count += 1
    finally:
        capture.release()
    if frame_count < 4:
        raise ValueError("insufficient_video_frames")
    return {
        "full_generated": np.asarray(total, dtype=np.float64),
        "overlay_region": np.asarray(overlay, dtype=np.float64),
        "overlay_masked_residual": np.asarray(residual, dtype=np.float64),
    }, {
        "video_sha256": file_sha256(path),
        "frame_count": frame_count,
        "generated_crop_pixels": [0, 0, width, height],
        "overlay_mask_fraction_mean": float(np.mean(coverages)),
        "third_party_physical_pixels_decoded": False,
    }


def _resample(values: np.ndarray, count: int) -> np.ndarray:
    if not len(values) or count <= 0:
        return np.zeros(max(0, count), dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, count)
    return np.interp(x_new, x_old, values).astype(np.float64)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right) or len(left) < 3:
        return 0.0
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _controls(
    action: np.ndarray, *, seed_material: str, swapped: np.ndarray
) -> dict[str, np.ndarray]:
    seed = int(canonical_sha256({"seed_material": seed_material})[:16], 16)
    generator = np.random.default_rng(seed)
    shuffled = action.copy()
    generator.shuffle(shuffled)
    return {
        "zero_actions": np.zeros_like(action),
        "shuffled_action_order": shuffled,
        "temporally_reversed_actions": action[::-1],
        "circularly_shifted_actions": np.roll(action, max(1, len(action) // 2)),
        "within_session_swapped_policy_actions": swapped,
    }


def _wilson_lower(successes: int, count: int, confidence: float = 0.95) -> float:
    if count <= 0:
        return 0.0
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = successes / count
    denominator = 1 + z * z / count
    center = proportion + z * z / (2 * count)
    spread = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count))
    return (center - spread) / denominator


def _cluster_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    channel: str,
    margin: float,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    session_values: dict[str, list[float]] = defaultdict(list)
    passes = 0
    for row in rows:
        result = row["channels"][channel]
        session_values[str(row["session_id"])].append(
            float(result["excess_over_strongest_placebo"])
        )
        passes += bool(result["validity_pass"])
    clustered = np.asarray(
        [float(np.mean(values)) for _, values in sorted(session_values.items())], dtype=np.float64
    )
    generator = np.random.default_rng(20260727)
    bootstrap = np.asarray(
        [
            float(np.mean(generator.choice(clustered, size=len(clustered), replace=True)))
            for _ in range(bootstrap_replicates)
        ],
        dtype=np.float64,
    )
    observed = float(np.mean(clustered))
    centered = clustered - margin
    signs = generator.choice(np.asarray([-1.0, 1.0]), size=(bootstrap_replicates, len(centered)))
    null_means = np.mean(signs * centered, axis=1)
    permutation_p = float(
        (1 + np.sum(null_means >= observed - margin)) / (bootstrap_replicates + 1)
    )
    return {
        "session_cluster_count": len(clustered),
        "row_count": len(rows),
        "mean_excess_over_strongest_placebo": observed,
        "clustered_bootstrap_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "one_sided_sign_flip_p_against_margin": permutation_p,
        "validity_pass_rate": passes / len(rows) if rows else 0.0,
        "validity_pass_rate_wilson_lower95": _wilson_lower(passes, len(rows)),
    }


def build_causal_report(
    inventory: Mapping[str, Any],
    *,
    roboarena_root: str | Path,
    meaningful_margin: float = DEFAULT_MEANINGFUL_MARGIN,
    minimum_original_correlation: float = DEFAULT_MIN_ORIGINAL_CORRELATION,
    minimum_validity_pass_rate: float = DEFAULT_VALIDITY_PASS_RATE,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    root = Path(roboarena_root).resolve()
    identities: dict[tuple[str, str], Mapping[str, Any]] = {}
    for request in inventory.get("requests", []):
        if isinstance(request, Mapping):
            identity = (str(request.get("session_id")), str(request.get("policy_id")))
            identities.setdefault(identity, request)
    prepared: dict[tuple[str, str], dict[str, Any]] = {}
    blockers: list[str] = []
    for (session_id, policy_id), request in sorted(identities.items()):
        candidates = sorted(
            (root / "evaluation_sessions" / session_id).glob(f"*_{policy_id}/*_npz_file.npz")
        )
        if len(candidates) != 1:
            blockers.append(f"npz_resolution:{session_id}:{policy_id}:{len(candidates)}")
            continue
        try:
            arrays = load_restricted_roboarena_npz(candidates[0])
            motion, metadata = generated_motion_channels(str(request["video_path"]))
        except Exception as exc:  # noqa: BLE001 - fail-closed evidence report
            blockers.append(f"preparation_failed:{session_id}:{policy_id}:{type(exc).__name__}")
            continue
        prepared[(session_id, policy_id)] = {
            "action": np.linalg.norm(arrays["action"][:, :7], axis=1),
            "npz_sha256": file_sha256(candidates[0]),
            "motion": motion,
            "metadata": metadata,
        }
    rows: list[dict[str, Any]] = []
    by_session: dict[str, list[str]] = defaultdict(list)
    for session_id, policy_id in prepared:
        by_session[session_id].append(policy_id)
    for (session_id, policy_id), item in sorted(prepared.items()):
        policies = sorted(by_session[session_id])
        swapped_policy = policies[(policies.index(policy_id) + 1) % len(policies)]
        swapped_source = prepared[(session_id, swapped_policy)]["action"]
        channels: dict[str, Any] = {}
        for channel, motion in item["motion"].items():
            action = _resample(item["action"], len(motion))
            swapped = _resample(swapped_source, len(motion))
            placebo_signals = _controls(
                action,
                seed_material=f"{session_id}:{policy_id}:{channel}",
                swapped=swapped,
            )
            original = _correlation(action, motion)
            placebos = {
                name: _correlation(signal, motion) for name, signal in placebo_signals.items()
            }
            strongest = max(placebos.values())
            excess = original - strongest
            channels[channel] = {
                "original_action_correlation": original,
                "placebo_correlations": placebos,
                "strongest_placebo_correlation": strongest,
                "excess_over_strongest_placebo": excess,
                "validity_pass": original >= minimum_original_correlation
                and excess >= meaningful_margin,
            }
        rows.append(
            {
                "session_id": session_id,
                "policy_id": policy_id,
                "swapped_policy_id": swapped_policy,
                "video_sha256": item["metadata"]["video_sha256"],
                "npz_sha256": item["npz_sha256"],
                "frame_count": item["metadata"]["frame_count"],
                "overlay_mask_fraction_mean": item["metadata"]["overlay_mask_fraction_mean"],
                "channels": channels,
            }
        )
    summaries = {
        channel: _cluster_summary(
            rows,
            channel=channel,
            margin=meaningful_margin,
            bootstrap_replicates=bootstrap_replicates,
        )
        for channel in ("full_generated", "overlay_region", "overlay_masked_residual")
        if rows
    }
    residual = summaries.get("overlay_masked_residual", {})
    gates = {
        "residual_excess_lower95_above_margin": bool(
            residual and residual["clustered_bootstrap_ci95"][0] > meaningful_margin
        ),
        "residual_validity_pass_rate_lower95": bool(
            residual and residual["validity_pass_rate_wilson_lower95"] >= minimum_validity_pass_rate
        ),
        "conditioning_annotation_not_sole_signal": bool(
            residual
            and summaries["overlay_region"]["mean_excess_over_strongest_placebo"]
            <= residual["mean_excess_over_strongest_placebo"] + meaningful_margin
        ),
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if not blockers and len(rows) == len(identities) else "blocked",
        "inventory_sha256": inventory.get("inventory_sha256"),
        "thresholds": {
            "meaningful_correlation_margin": meaningful_margin,
            "minimum_original_correlation": minimum_original_correlation,
            "minimum_validity_pass_rate": minimum_validity_pass_rate,
            "bootstrap_replicates": bootstrap_replicates,
        },
        "row_count": len(rows),
        "rows": rows,
        "clustered_summaries": summaries,
        "gates": gates,
        "all_action_following_validity_gates_passed": bool(gates) and all(gates.values()),
        "blockers": sorted(set(blockers)),
        "benchmark_labels_seen": False,
        "task_success_scored": False,
        "third_party_physical_video_pixels_decoded": False,
        "physical_action_trace_used_as_independent_condition_reference": True,
        "claim_boundary": (
            "Label-free temporal-alignment falsification diagnostic; not counterfactual WAM "
            "regeneration, 3D action following, task success, ranking fidelity, or physical proof."
        ),
    }
    result["report_sha256"] = canonical_sha256(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--roboarena-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build_causal_report(
        json.loads(Path(args.inventory).read_text(encoding="utf-8")),
        roboarena_root=args.roboarena_root,
    )
    write_json(Path(args.output), report)
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
