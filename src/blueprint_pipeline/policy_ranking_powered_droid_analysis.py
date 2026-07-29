"""Analyze a frozen powered Cosmos3 DROID causal-control matrix.

The GPU runtime only generates and preserves attributable media.  This module
performs the separate, label-free scientific analysis after transport.  It
keeps the three DROID camera views separate, treats sessions as the independent
unit, and applies thresholds supplied by the prospectively frozen protocol.

Passing this analysis qualifies action-conditioned open-loop replay only.  It
does not grant policy-ranking, closed-loop, captured-site, or physical-success
credit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import cv2  # type: ignore[import-not-found]
import numpy as np

from .common import write_json
from .policy_ranking_droid_reference_analysis import _decode_concat_view
from .policy_ranking_thesis import canonical_sha256, file_sha256
from .wam_rollout_reliability import (
    TIMING_SCOPE_SESSION,
    ReliabilityThresholds,
    SessionReliabilityThresholds,
    action_energy_series,
    assess_rollout_reliability,
    assess_session_reliability,
)

CONDITIONS = ("recorded", "zero", "shuffled", "reversed", "shifted", "policy_swapped")
TEMPORAL_PLACEBOS = ("shuffled", "reversed", "shifted")
SEEDS = (0, 1)
VIEWS = ("wrist", "left", "right")
BOOTSTRAP_SEED = 20260729
BOOTSTRAP_REPLICATES = 10_000


def _request_id(
    packet_sha256: str,
    checkpoint_revision: str,
    row: dict[str, Any],
    condition: str,
    seed: int,
) -> str:
    material = {
        "packet_sha256": packet_sha256,
        "session_id": str(row["session_id_internal_only"]),
        "window_index": int(row["window_index"]),
        "condition": condition,
        "seed": seed,
        "initial_observation_sha256": row["initial_observation_sha256"],
        "action_sha256": row["controls"][condition]["action_sha256"],
        "checkpoint_revision": checkpoint_revision,
    }
    return canonical_sha256(material)


def _motion_series(frames: list[np.ndarray], downscale_width: int) -> np.ndarray:
    gray: list[np.ndarray] = []
    for frame in frames:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = image.shape
        if width > downscale_width:
            target_height = max(1, int(round(height * downscale_width / width)))
            image = cv2.resize(
                image, (downscale_width, target_height), interpolation=cv2.INTER_AREA
            )
        gray.append(image)
    values: list[float] = []
    for before, after in zip(gray[:-1], gray[1:], strict=True):
        flow = cv2.calcOpticalFlowFarneback(
            before,
            after,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        median = np.median(flow.reshape(-1, 2), axis=0)
        residual = flow - median[None, None, :]
        values.append(float(np.mean(np.linalg.norm(residual, axis=2))))
    return np.asarray(values, dtype=np.float64)


def _correlation(action_energy: np.ndarray, motion: np.ndarray) -> float | None:
    count = min(len(action_energy), len(motion))
    if count < 4:
        return None
    left = np.asarray(action_energy[:count], dtype=np.float64)
    right = np.asarray(motion[:count], dtype=np.float64)
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def _scene_distance(left: list[np.ndarray], right: list[np.ndarray]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("powered_analysis_scene_pair_frame_count_mismatch")
    values: list[float] = []
    for left_frame, right_frame in zip(left, right, strict=True):
        if left_frame.shape != right_frame.shape:
            raise ValueError("powered_analysis_scene_pair_geometry_mismatch")
        values.append(
            float(
                np.mean(np.abs(left_frame.astype(np.float32) - right_frame.astype(np.float32)))
                / 255.0
            )
        )
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _percentile_interval(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    return float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))


def _clustered_window_bootstrap(
    rows: list[dict[str, Any]], *, replicates: int = BOOTSTRAP_REPLICATES
) -> dict[str, float | int]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["session_id"]), []).append(float(bool(row["passed"])))
    sessions = sorted(grouped)
    if not sessions:
        raise ValueError("powered_analysis_no_sessions_for_bootstrap")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: list[float] = []
    for _ in range(replicates):
        chosen = rng.choice(sessions, size=len(sessions), replace=True)
        cluster_values = [value for session in chosen for value in grouped[str(session)]]
        samples.append(float(np.mean(cluster_values)))
    lower, upper = _percentile_interval(samples)
    return {
        "estimate": float(np.mean([float(bool(row["passed"])) for row in rows])),
        "lower95": lower,
        "upper95": upper,
        "independent_session_count": len(sessions),
        "window_count": len(rows),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _session_bootstrap(
    rows: list[dict[str, Any]], *, replicates: int = BOOTSTRAP_REPLICATES
) -> dict[str, float | int]:
    if not rows:
        raise ValueError("powered_analysis_no_sessions_for_reliability_bootstrap")
    values = np.asarray([float(bool(row["reliable"])) for row in rows], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = [
        float(np.mean(rng.choice(values, size=len(values), replace=True)))
        for _ in range(replicates)
    ]
    lower, upper = _percentile_interval(samples)
    return {
        "estimate": float(np.mean(values)),
        "lower95": lower,
        "upper95": upper,
        "independent_session_count": len(rows),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _load_protocol_thresholds(
    protocol: dict[str, Any],
) -> tuple[ReliabilityThresholds, SessionReliabilityThresholds, dict[str, Any]]:
    rollout = dict(protocol["frozen_rollout_reliability_thresholds"])
    rollout.pop("threshold_anchor", None)
    session = protocol["frozen_session_reliability_rule"]
    reliability = ReliabilityThresholds(**rollout)
    session_reliability = SessionReliabilityThresholds(
        timing_correlation_min=float(session["timing_correlation_median_min"]),
        minimum_eligible_timing_windows=int(session["minimum_eligible_timing_windows"]),
    )
    return reliability, session_reliability, dict(protocol["frozen_causal_gates"])


def _load_matrix(
    *, packet: dict[str, Any], output_dir: Path, checkpoint_revision: str
) -> tuple[dict[tuple[str, int, str, int], Path], list[dict[str, Any]]]:
    packet_sha256 = str(packet["manifest_sha256"])
    videos: dict[tuple[str, int, str, int], Path] = {}
    records: list[dict[str, Any]] = []
    for row in packet["rows"]:
        session_id = str(row["session_id_internal_only"])
        window_index = int(row["window_index"])
        for condition in CONDITIONS:
            for seed in SEEDS:
                request_id = _request_id(packet_sha256, checkpoint_revision, row, condition, seed)
                response_path = output_dir / "responses" / f"{request_id}.json"
                video_path = output_dir / "videos" / f"{request_id}.mp4"
                if not response_path.is_file() or not video_path.is_file():
                    raise ValueError(f"powered_analysis_matrix_artifact_missing:{request_id}")
                response = json.loads(response_path.read_text(encoding="utf-8"))
                if response.get("request_id") != request_id:
                    raise ValueError("powered_analysis_response_request_id_mismatch")
                if response.get("accepted_first_valid") is not True:
                    raise ValueError("powered_analysis_response_not_accepted")
                if file_sha256(video_path) != (response.get("response") or {}).get("output_sha256"):
                    raise ValueError("powered_analysis_video_digest_mismatch")
                expected = (session_id, window_index, condition, seed)
                actual = (
                    str(response.get("session_id")),
                    int(response.get("window_index", -1)),
                    str(response.get("condition")),
                    int(response.get("seed", -1)),
                )
                if actual != expected:
                    raise ValueError("powered_analysis_response_identity_mismatch")
                videos[expected] = video_path
                records.append(response)
    expected_count = len(packet["rows"]) * len(CONDITIONS) * len(SEEDS)
    if len(videos) != expected_count:
        raise ValueError("powered_analysis_matrix_not_complete")
    return videos, records


def analyze_powered_droid_matrix(
    *,
    packet_path: str | Path,
    provider_output_dir: str | Path,
    protocol_path: str | Path,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Apply the frozen causal and reliability gates to one complete matrix."""

    packet_file = Path(packet_path).expanduser().resolve()
    output_dir = Path(provider_output_dir).expanduser().resolve()
    protocol_file = Path(protocol_path).expanduser().resolve()
    packet = json.loads(packet_file.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    packet_sha256 = str(packet.get("manifest_sha256") or "")
    if packet_sha256 != canonical_sha256(
        {key: value for key, value in packet.items() if key != "manifest_sha256"}
    ):
        raise ValueError("powered_analysis_packet_digest_mismatch")
    if packet.get("schema_version") != "policy_ranking_powered_droid_provider_packet.v1":
        raise ValueError("powered_analysis_requires_provider_packet_v1")
    protocol_sha256 = str(protocol.get("manifest_sha256") or "")
    if protocol_sha256 != canonical_sha256(
        {key: value for key, value in protocol.items() if key != "manifest_sha256"}
    ):
        raise ValueError("powered_analysis_protocol_digest_mismatch")
    if len(packet.get("rows") or []) != 51:
        raise ValueError("powered_analysis_packet_window_count_invalid")
    reliability_thresholds, session_thresholds, causal = _load_protocol_thresholds(protocol)
    videos, response_records = _load_matrix(
        packet=packet,
        output_dir=output_dir,
        checkpoint_revision=str(protocol["source_lock"]["checkpoint_revision"]),
    )

    decoded: dict[tuple[str, int, str, int], dict[str, list[np.ndarray]]] = {}
    reports: dict[tuple[str, int, str, int], Any] = {}
    for row in packet["rows"]:
        session_id = str(row["session_id_internal_only"])
        window_index = int(row["window_index"])
        for condition in CONDITIONS:
            actions = np.asarray(row["controls"][condition]["actions"], dtype=np.float64)
            for seed in SEEDS:
                key = (session_id, window_index, condition, seed)
                decoded[key] = _decode_concat_view(videos[key])
                reports[key] = assess_rollout_reliability(
                    videos[key],
                    actions,
                    reliability_thresholds,
                    timing_flag_scope=TIMING_SCOPE_SESSION,
                )

    window_results: list[dict[str, Any]] = []
    for row in packet["rows"]:
        session_id = str(row["session_id_internal_only"])
        window_index = int(row["window_index"])
        energies = {
            condition: action_energy_series(
                np.asarray(row["controls"][condition]["actions"], dtype=np.float64)
            )
            for condition in CONDITIONS
        }
        seed_results: list[dict[str, Any]] = []
        for seed in SEEDS:
            view_results: dict[str, Any] = {}
            for view in VIEWS:
                recorded_frames = decoded[(session_id, window_index, "recorded", seed)][view]
                zero_frames = decoded[(session_id, window_index, "zero", seed)][view]
                recorded_motion = _motion_series(
                    recorded_frames, reliability_thresholds.downscale_width
                )
                zero_motion = _motion_series(zero_frames, reliability_thresholds.downscale_width)
                own_correlation = _correlation(energies["recorded"], recorded_motion)
                placebo_correlations = {
                    condition: _correlation(energies[condition], recorded_motion)
                    for condition in TEMPORAL_PLACEBOS
                }
                finite_placebos = [
                    value for value in placebo_correlations.values() if value is not None
                ]
                strongest_placebo = max(finite_placebos) if finite_placebos else None
                excess = (
                    own_correlation - strongest_placebo
                    if own_correlation is not None and strongest_placebo is not None
                    else None
                )
                scene_distance = _scene_distance(recorded_frames, zero_frames)
                view_results[view] = {
                    "recorded_motion_mean": float(np.mean(recorded_motion)),
                    "zero_motion_mean": float(np.mean(zero_motion)),
                    "recorded_exceeds_zero": float(np.mean(recorded_motion))
                    > float(np.mean(zero_motion)),
                    "own_action_motion_correlation": own_correlation,
                    "temporal_placebo_correlations": placebo_correlations,
                    "strongest_temporal_placebo_correlation": strongest_placebo,
                    "excess_over_strongest_temporal_placebo": excess,
                    "same_seed_recorded_vs_zero_scene_distance": scene_distance,
                }
            seed_results.append({"seed": seed, "views": view_results})

        cross_seed: dict[str, Any] = {}
        for view in VIEWS:
            recorded_seed_noise = _scene_distance(
                decoded[(session_id, window_index, "recorded", 0)][view],
                decoded[(session_id, window_index, "recorded", 1)][view],
            )
            zero_seed_noise = _scene_distance(
                decoded[(session_id, window_index, "zero", 0)][view],
                decoded[(session_id, window_index, "zero", 1)][view],
            )
            action_effect = float(
                np.median(
                    [
                        result["views"][view]["same_seed_recorded_vs_zero_scene_distance"]
                        for result in seed_results
                    ]
                )
            )
            noise = max(recorded_seed_noise, zero_seed_noise)
            ratio = action_effect / max(noise, float(np.finfo(np.float64).eps))
            cross_seed[view] = {
                "action_effect": action_effect,
                "recorded_cross_seed_noise": recorded_seed_noise,
                "zero_cross_seed_noise": zero_seed_noise,
                "maximum_cross_seed_noise": noise,
                "action_effect_to_cross_seed_noise_ratio": ratio,
            }

        gate_reasons: list[str] = []
        for seed_result in seed_results:
            seed = int(seed_result["seed"])
            for view, metrics in seed_result["views"].items():
                prefix = f"seed{seed}:{view}"
                correlation = metrics["own_action_motion_correlation"]
                if correlation is None or correlation < float(
                    causal["minimum_original_action_motion_correlation"]
                ):
                    gate_reasons.append(f"{prefix}:original_action_motion_correlation_failed")
                excess = metrics["excess_over_strongest_temporal_placebo"]
                if excess is None or excess < float(
                    causal["minimum_excess_over_strongest_temporal_placebo"]
                ):
                    gate_reasons.append(f"{prefix}:temporal_placebo_excess_failed")
                if not metrics["recorded_exceeds_zero"]:
                    gate_reasons.append(f"{prefix}:recorded_did_not_exceed_zero")
                if metrics["same_seed_recorded_vs_zero_scene_distance"] < float(
                    causal["same_seed_recorded_vs_zero_scene_difference_min"]
                ):
                    gate_reasons.append(f"{prefix}:recorded_zero_scene_difference_failed")
        for view, metrics in cross_seed.items():
            if metrics["action_effect_to_cross_seed_noise_ratio"] < float(
                causal["action_effect_to_cross_seed_noise_ratio_min"]
            ):
                gate_reasons.append(f"{view}:action_effect_below_cross_seed_noise")
        hard_flags = sorted(
            {
                flag
                for condition in CONDITIONS
                for seed in SEEDS
                for flag in reports[(session_id, window_index, condition, seed)].flags
            }
        )
        if hard_flags:
            gate_reasons.extend(f"rollout_reliability:{flag}" for flag in hard_flags)
        window_results.append(
            {
                "session_id": session_id,
                "window_index": window_index,
                "seed_results": seed_results,
                "cross_seed": cross_seed,
                "rollout_hard_flags": hard_flags,
                "passed": not gate_reasons,
                "failure_reasons": gate_reasons,
            }
        )

    session_results: list[dict[str, Any]] = []
    for session_id in sorted({row["session_id"] for row in window_results}):
        recorded_reports = [
            reports[(session_id, window_index, "recorded", seed)]
            for window_index in (0, 1, 2)
            for seed in SEEDS
        ]
        session_report = assess_session_reliability(
            session_id, recorded_reports, session_thresholds
        )
        all_condition_flags = sorted(
            {
                flag
                for key, report in reports.items()
                if key[0] == session_id
                for flag in report.flags
            }
        )
        reliable = session_report.reliable and not all_condition_flags
        session_results.append(
            {
                **session_report.as_dict(),
                "all_condition_hard_flags": all_condition_flags,
                "reliable": reliable,
                "abstain": not reliable,
            }
        )

    validity = _clustered_window_bootstrap(window_results, replicates=bootstrap_replicates)
    reliability = _session_bootstrap(session_results, replicates=bootstrap_replicates)
    causal_gates = {
        "validity_pass_rate": validity["estimate"]
        >= float(causal["minimum_session_clustered_validity_pass_rate"]),
        "validity_pass_rate_lower95": validity["lower95"]
        >= float(causal["minimum_session_clustered_validity_pass_rate_lower95"]),
    }
    session_rule = protocol["frozen_session_reliability_rule"]
    reliability_gates = {
        "reliable_session_rate": reliability["estimate"]
        >= float(session_rule["minimum_reliable_session_rate"]),
        "reliable_session_rate_lower95": reliability["lower95"]
        >= float(session_rule["session_clustered_bootstrap_lower95_min"]),
    }
    causal_passed = all(causal_gates.values())
    reliability_passed = all(reliability_gates.values())
    qualified = causal_passed and reliability_passed
    result: dict[str, Any] = {
        "schema_version": "policy_ranking_powered_droid_analysis.v1",
        "status": "completed",
        "packet_file_sha256": file_sha256(packet_file),
        "packet_manifest_sha256": packet_sha256,
        "protocol_file_sha256": file_sha256(protocol_file),
        "protocol_manifest_sha256": protocol_sha256,
        "response_count": len(response_records),
        "video_count": len(videos),
        "window_results": window_results,
        "session_results": session_results,
        "session_clustered_causal_validity": validity,
        "session_reliability": reliability,
        "causal_gates": causal_gates,
        "reliability_gates": reliability_gates,
        "causal_gates_passed": causal_passed,
        "reliability_gates_passed": reliability_passed,
        "cosmos_wam_qualification": "passed" if qualified else "failed",
        "blueprint_abstains": not qualified,
        "thresholds": {
            "rollout_reliability": asdict(reliability_thresholds),
            "session_reliability": asdict(session_thresholds),
            "causal": causal,
        },
        "claim_ceiling": protocol["claim_ceiling"],
        "claims": {
            "same_snapshot_disjoint_session_open_loop_causal_qualification": qualified,
            "independent_new_snapshot_confirmation": False,
            "live_policy_wam_policy_closed_loop": False,
            "policy_ranking_fidelity": False,
            "captured_site_transfer": False,
            "physical_performance": False,
        },
    }
    result["analysis_sha256"] = canonical_sha256(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--provider-output-dir", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze_powered_droid_matrix(
        packet_path=args.packet,
        provider_output_dir=args.provider_output_dir,
        protocol_path=args.protocol,
    )
    write_json(Path(args.output), result)
    return 0


if __name__ == "__main__":  # pragma: no cover - focused subprocess coverage is sufficient
    raise SystemExit(main())
