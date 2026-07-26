from __future__ import annotations

import json
from pathlib import Path

import yaml

from blueprint_pipeline.policy_ranking_thesis import (
    DEFAULT_POLICIES,
    JUDGE_RESULT_SCHEMA,
    _benchmark_outcomes,
    build_hybrid_scene_bundle,
    build_controlled_scene_bundle,
    build_preregistration,
    build_rollout_index,
    canonical_sha256,
    evaluate_frozen_calibration,
    parse_lfs_pointer,
)


def _sessions() -> list[str]:
    return [f"00000000-0000-0000-0000-{index:012d}" for index in range(63)]


def _pointer(path: Path, digest: str = "a" * 64, size: int = 123) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{digest}\n"
        f"size {size}\n",
        encoding="utf-8",
    )


def _metadata(path: Path, session_index: int) -> None:
    rows = []
    for policy_index, policy in enumerate(DEFAULT_POLICIES):
        rows.append(
            {
                "policy_name": policy,
                "binary_success": policy_index >= session_index % len(DEFAULT_POLICIES),
                "partial_success": 1.0 if policy_index >= session_index % len(DEFAULT_POLICIES) else 0.0,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "language_instruction": "put the can in the tray",
                "evaluator_name": "must not leak",
                "evaluator_email": "must-not-leak@example.com",
                "evaluation_location": "must not leak",
                "policies": rows,
            }
        ),
        encoding="utf-8",
    )


def test_preregistration_is_deterministic_and_disjoint() -> None:
    first = build_preregistration(_sessions())
    second = build_preregistration(list(reversed(_sessions())))
    assert first == second
    assert first["protocol_sha256"] == canonical_sha256(
        {key: value for key, value in first.items() if key != "protocol_sha256"}
    )
    partitions = first["partitions"]
    assert {key: len(value) for key, value in partitions.items()} == {
        "pilot": 7,
        "calibration": 7,
        "heldout": 49,
    }
    assert not (set(partitions["pilot"]) & set(partitions["heldout"]))


def test_lfs_index_is_pii_free_and_label_blind(tmp_path: Path) -> None:
    rollout = tmp_path / "rollouts"
    benchmark = tmp_path / "benchmark"
    for index, session_id in enumerate(_sessions()):
        _metadata(benchmark / "evaluation_sessions" / session_id / "metadata.yaml", index)
        for policy in DEFAULT_POLICIES:
            _pointer(rollout / session_id / policy / "left" / "compare_overlay_vs_gt.mp4")
    protocol = build_preregistration(_sessions())
    result = build_rollout_index(rollout, benchmark, protocol)
    assert result["status"] == "ready"
    assert result["row_count"] == 441
    serialized = json.dumps(result)
    assert "must not leak" not in serialized
    assert "must-not-leak@example.com" not in serialized
    assert "binary_success" not in serialized
    assert all(not row["benchmark_labels_included"] for row in result["rows"])
    pointer = parse_lfs_pointer(
        rollout / _sessions()[0] / DEFAULT_POLICIES[0] / "left" / "compare_overlay_vs_gt.mp4"
    )
    assert pointer == {"sha256": "a" * 64, "size_bytes": 123}


def _judgments(protocol: dict, benchmark: Path, method: str) -> list[dict]:
    rows = []
    for session_id in protocol["partitions"]["heldout"]:
        session_index = _sessions().index(session_id)
        for policy_index, policy in enumerate(DEFAULT_POLICIES):
            actual = float(policy_index >= session_index % len(DEFAULT_POLICIES))
            rows.append(
                {
                    "schema_version": JUDGE_RESULT_SCHEMA,
                    "session_id": session_id,
                    "policy_id": policy,
                    "method": method,
                    "success_probability": 0.9 if actual else 0.1,
                    "judge_confidence": 0.95,
                    "action_following_confidence": 0.95,
                    "temporal_coherence_confidence": 0.95,
                    "critical_contradiction": False,
                    "abstained": False,
                    "evaluator_digest": "b" * 64,
                    "benchmark_labels_seen": False,
                    "third_party_physical_pixels_seen": False,
                }
            )
    return rows


def test_calibration_computes_frozen_metrics_and_rejects_leakage(tmp_path: Path) -> None:
    protocol = build_preregistration(_sessions())
    benchmark = tmp_path / "benchmark"
    for index, session_id in enumerate(_sessions()):
        _metadata(benchmark / "evaluation_sessions" / session_id / "metadata.yaml", index)
    full_method = protocol["evaluator"]["full_temporal_method"]
    cheap_method = protocol["evaluator"]["cheap_baseline_method"]
    full = _judgments(protocol, benchmark, full_method)
    cheap = _judgments(protocol, benchmark, cheap_method)
    for row in cheap:
        row["success_probability"] = 0.5
    report = evaluate_frozen_calibration(
        [*full, *cheap], protocol=protocol, roboarena_root=benchmark
    )
    assert report["status"] == "completed"
    assert report["methods"][full_method]["episode_accuracy"] == 1.0
    assert report["methods"][full_method]["session_pairwise_accuracy"] == 1.0
    assert report["methods"][full_method]["selective_session_pairwise_accuracy"] == 1.0
    assert report["methods"][full_method]["selective_session_pairwise_coverage"] == 1.0
    assert report["methods"][cheap_method]["session_pairwise_accuracy"] == 0.5
    assert report["methods"][full_method]["top_policy_regret"] == 0.0
    assert report["gates"]["beats_cheap_baseline"] is True
    assert report["claim_boundary"]["blueprint_conducted_physical_experiments"] is False

    full[0]["benchmark_labels_seen"] = True
    blocked = evaluate_frozen_calibration(
        [*full, *cheap], protocol=protocol, roboarena_root=benchmark
    )
    assert blocked["status"] == "blocked"
    assert any("benchmark_label_leakage" in item for item in blocked["blockers"])


def test_calibration_honors_explicit_abstention_and_digest(tmp_path: Path) -> None:
    protocol = build_preregistration(_sessions())
    benchmark = tmp_path / "benchmark"
    for index, session_id in enumerate(_sessions()):
        _metadata(benchmark / "evaluation_sessions" / session_id / "metadata.yaml", index)
    full_method = protocol["evaluator"]["full_temporal_method"]
    cheap_method = protocol["evaluator"]["cheap_baseline_method"]
    full = _judgments(protocol, benchmark, full_method)
    cheap = _judgments(protocol, benchmark, cheap_method)
    full[0]["abstained"] = True
    report = evaluate_frozen_calibration(
        [*full, *cheap],
        protocol=protocol,
        roboarena_root=benchmark,
        expected_evaluator_digest="b" * 64,
    )
    assert report["status"] == "completed"
    assert report["evaluator_digest"] == "b" * 64
    assert report["methods"][full_method]["selective_coverage"] < 1.0
    assert report["methods"][full_method]["selective_session_pairwise_coverage"] < 1.0

    mismatched = evaluate_frozen_calibration(
        [*full, *cheap],
        protocol=protocol,
        roboarena_root=benchmark,
        expected_evaluator_digest="c" * 64,
    )
    assert mismatched["status"] == "blocked"
    assert "evaluator_digest_mismatch" in mismatched["blockers"]


def test_calibration_accepts_roboarena_letter_keyed_policy_mapping(tmp_path: Path) -> None:
    path = tmp_path / "metadata.yaml"
    _metadata(path, 3)
    payload = yaml.safe_load(path.read_text())
    payload["policies"] = {
        chr(ord("A") + policy_index): row
        for policy_index, row in enumerate(payload["policies"])
    }
    path.write_text(yaml.safe_dump(payload))
    outcomes = _benchmark_outcomes(path)
    assert set(outcomes) == set(DEFAULT_POLICIES)
    assert outcomes[DEFAULT_POLICIES[0]]["binary_success"] == 0.0
    assert outcomes[DEFAULT_POLICIES[-1]]["binary_success"] == 1.0


def test_hybrid_scene_keeps_visual_spatial_and_interaction_layers_separate() -> None:
    spec = {
        "scene_id": "new_site_01",
        "site_visual": {
            "representation": "3dgs_ply",
            "sha256": "1" * 64,
            "license": "CC-BY-4.0",
        },
        "site_spatial": {
            "is_complete_usd_rebuild": False,
            "work_surface_proxy_only": True,
        },
        "interactive_assets": [
            {"asset_id": "can", "sha256": "2" * 64, "transform_site_m": [0, 0, 0]},
            {"asset_id": "tray", "sha256": "3" * 64, "transform_site_m": [0.3, 0, 0]},
        ],
        "robot_runtime": {"profile_id": "franka_droid_fixed_base_v1"},
        "task_semantics": {
            "instruction": "put the can in the tray",
            "success_predicates": ["can_contained_in_tray", "can_stable_2s"],
            "abstention_rules": ["target_occluded", "wam_sim_disagreement"],
        },
        "optional_physics_sidecar": {"backend": "mujoco", "advisory_only": True},
    }
    result = build_hybrid_scene_bundle(spec)
    assert result["status"] == "ready"
    assert result["full_site_usd_rebuild_required"] is False
    assert result["site_specific_physical_accuracy_proven"] is False

    spec["site_spatial"]["is_complete_usd_rebuild"] = True
    blocked = build_hybrid_scene_bundle(spec)
    assert blocked["status"] == "blocked"
    assert "complete_usd_rebuild_forbidden" in blocked["blockers"]


def test_controlled_usd_scene_is_never_a_physical_answer_key() -> None:
    spec = {
        "scene_id": "warehouse_control",
        "environment": {"representation": "usd", "sha256": "1" * 64},
        "interactive_assets": [
            {"asset_id": "can", "sha256": "2" * 64},
            {"asset_id": "tray", "sha256": "3" * 64},
        ],
        "robot_runtime": {"profile_id": "franka_droid_fixed_base_v1"},
        "task_semantics": {
            "success_predicates": ["can_contained_in_tray"],
            "physical_answer_key": False,
        },
    }
    result = build_controlled_scene_bundle(spec)
    assert result["status"] == "ready"
    assert result["independent_physical_ranking_answer_key"] is False

    spec["task_semantics"]["physical_answer_key"] = True
    blocked = build_controlled_scene_bundle(spec)
    assert blocked["status"] == "blocked"
    assert "controlled_scene_cannot_be_physical_answer_key" in blocked["blockers"]
