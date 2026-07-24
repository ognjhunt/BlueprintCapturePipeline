from __future__ import annotations

import base64
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline import policy_ranking_ladder as ladder_mod
from blueprint_pipeline import noise_degraded_policy_command_adapter as noise_adapter
from blueprint_pipeline import wam_score_claim_gate as claim_gate


pytestmark = [pytest.mark.slow, pytest.mark.integration]
RUNTIME_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x05" * 32)
VALIDATION_AUTHORITY_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x0d" * 32)
REGISTERED_ACTION_BOUNDS = noise_adapter.canonical_delta_ee_action_bounds_contract()
REGISTERED_ACTION_BOUNDS_SHA256 = noise_adapter.registered_action_bounds_sha256(
    REGISTERED_ACTION_BOUNDS
)


@pytest.fixture(autouse=True)
def _trusted_ladder_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    public_key = RUNTIME_PRIVATE_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    monkeypatch.setenv(
        ladder_mod.SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        hashlib.sha256(public_key).hexdigest(),
    )


def _runtime_attestation(payload: dict, root: Path, stem: str) -> dict:
    public_key = RUNTIME_PRIVATE_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    message = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    public_key_sha256 = hashlib.sha256(public_key).hexdigest()
    payload_sha256 = hashlib.sha256(message).hexdigest()
    report = root / f"{stem}-runtime-signature.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "sc3_signature_verification_report.v1",
                "algorithm": "Ed25519",
                "verification_status": "verified",
                "public_key_sha256": public_key_sha256,
                "signed_payload_sha256": payload_sha256,
                "signer_key_id": "ladder-runtime",
                "verifier_id": "blueprint-test-verifier",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "algorithm": "Ed25519",
        "signature_verified": True,
        "signer_key_id": "ladder-runtime",
        "verifier_id": "blueprint-test-verifier",
        "public_key_base64": base64.b64encode(public_key).decode("ascii"),
        "public_key_sha256": public_key_sha256,
        "signature_base64": base64.b64encode(RUNTIME_PRIVATE_KEY.sign(message)).decode("ascii"),
        "signed_payload_sha256": payload_sha256,
        "verification_report_artifact": {
            "path": str(report),
            "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        },
    }


def _ladder(**overrides):
    kwargs = {
        "inner_policy_id": "unitree_groot_n17_sonic",
        "inner_command": "groot-runner --serve",
        "inner_checkpoint_sha256": "d" * 64,
        "registered_action_bounds": REGISTERED_ACTION_BOUNDS,
        "amplitudes": (0.1, 0.3, 0.6),
        "seed": 7,
        "generated_at": "2026-07-02T00:00:00Z",
    }
    kwargs.update(overrides)
    return ladder_mod.build_known_ordering_policy_ladder(**kwargs)


def _scorecard(
    scores: dict[str, float],
    *,
    status: str = "completed",
    blockers=(),
    replicate_seed_ids: list[int] | None = None,
    task_by_policy: dict[str, str] | None = None,
    registered_action_bounds_sha256: str = REGISTERED_ACTION_BOUNDS_SHA256,
) -> dict:
    evidence_root = Path(tempfile.mkdtemp(prefix="policy-ladder-ground-truth-"))
    replicate_seed_ids = replicate_seed_ids or ladder_mod.replicate_seed_ids(7)
    task_by_policy = task_by_policy or {}
    condition_descriptor = {
        "schema_version": "policy_ladder_registered_condition.v1",
        "task_id": "ladder-task",
        "condition_id": "ladder-condition",
        "criterion_id": "registered_task_success",
    }
    condition_manifest_sha256 = hashlib.sha256(
        json.dumps(
            condition_descriptor,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    amplitudes_by_policy: dict[str, float | None] = {}
    for policy_id in scores:
        if policy_id == ladder_mod.REFERENCE_FLOOR_POLICY_ID:
            amplitudes_by_policy[policy_id] = None
        elif "_noise_" not in policy_id:
            amplitudes_by_policy[policy_id] = 0.0
        else:
            amplitudes_by_policy[policy_id] = float(
                policy_id.rsplit("_noise_", 1)[1].replace("p", ".")
            )
    ordered_empirical_policies = sorted(
        (
            (policy_id, amplitude)
            for policy_id, amplitude in amplitudes_by_policy.items()
            if amplitude is not None
        ),
        key=lambda item: item[1],
    )
    # Success counts preserve the intended per-rung *rate* spacing at any
    # replicate count.  Subtracting a fixed number of successes per rung instead
    # would shrink the separation as seeds are added, which is the opposite of
    # what more replicates should buy.
    span = max(1, len(ordered_empirical_policies) - 1)
    empirical_success_count = {
        policy_id: round(len(replicate_seed_ids) * (span - rank_index) / span)
        for rank_index, (policy_id, _amplitude) in enumerate(ordered_empirical_policies)
    }
    rankings = [
        {
            "policy_id": policy_id,
            "score": score,
            "task_success_rate": score,
            "predicted_success_rate": score,
            "replicate_seed_count": len(replicate_seed_ids),
            "replicate_seed_ids": replicate_seed_ids,
            "empirical_ground_truth_accepted": True,
        }
        for policy_id, score in scores.items()
    ]
    for row in rankings:
        outcome_records = []
        amplitude = amplitudes_by_policy[row["policy_id"]]
        success_count = empirical_success_count.get(row["policy_id"], 0)
        for index, seed in enumerate(replicate_seed_ids):
            if amplitude is None:
                adapter_command = ""
            elif amplitude == 0.0:
                adapter_command = "groot-runner --serve"
            else:
                adapter_command = ladder_mod._noise_variant_command(
                    inner_command="groot-runner --serve",
                    amplitude=amplitude,
                    seed=seed,
                    policy_id=row["policy_id"],
                    python_executable=sys.executable,
                    registered_action_bounds=REGISTERED_ACTION_BOUNDS,
                    registered_action_bounds_sha256_value=(REGISTERED_ACTION_BOUNDS_SHA256),
                )
            adapter_command_sha256 = (
                hashlib.sha256(adapter_command.encode("utf-8")).hexdigest()
                if adapter_command
                else None
            )
            runtime_session_id = f"runtime-{row['policy_id']}-{seed}"
            action_sequence = [
                round((amplitude or 0.0) + seed * 1e-6 + dimension * 0.01, 6)
                for dimension in range(7)
            ]
            trace = evidence_root / f"{row['policy_id']}-{seed}-actions.json"
            trace.write_text(
                json.dumps(
                    {
                        "schema_version": "policy_ladder_action_trace.v1",
                        "runtime_session_id": runtime_session_id,
                        "policy_id": row["policy_id"],
                        "replicate_seed": seed,
                        "noise_amplitude": amplitude,
                        "task_id": task_by_policy.get(row["policy_id"], "ladder-task"),
                        "condition_id": "ladder-condition",
                        "criterion_id": "registered_task_success",
                        "registered_condition_manifest_sha256": (condition_manifest_sha256),
                        "adapter_command_sha256": adapter_command_sha256,
                        "policy_checkpoint_sha256": "d" * 64 if amplitude is not None else None,
                        "registered_action_bounds_sha256": (registered_action_bounds_sha256),
                        "action_bounds_enforced": True,
                        "action_sequence": action_sequence,
                        "action_sequence_sha256": hashlib.sha256(
                            json.dumps(
                                action_sequence,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest(),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            success = index < success_count
            outcome = evidence_root / f"{row['policy_id']}-{seed}-outcome.json"
            outcome_payload = {
                "schema_version": "policy_ladder_runtime_outcome.v1",
                "runtime_session_id": runtime_session_id,
                "runtime_executor_id": "test-ladder-runtime",
                "runtime_executor_code_sha256": "e" * 64,
                "policy_id": row["policy_id"],
                "replicate_seed": seed,
                "noise_amplitude": amplitude,
                "task_id": task_by_policy.get(row["policy_id"], "ladder-task"),
                "condition_id": "ladder-condition",
                "criterion_id": "registered_task_success",
                "registered_condition_manifest_sha256": (condition_manifest_sha256),
                "empirical_success": success,
                "accepted": True,
                "action_trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                "adapter_command_sha256": adapter_command_sha256,
                "policy_checkpoint_sha256": "d" * 64 if amplitude is not None else None,
                "registered_action_bounds_sha256": (registered_action_bounds_sha256),
                "action_bounds_enforced": True,
            }
            outcome_payload["runtime_attestation"] = _runtime_attestation(
                outcome_payload,
                evidence_root,
                f"{row['policy_id']}-{seed}",
            )
            outcome.write_text(
                json.dumps(outcome_payload, sort_keys=True),
                encoding="utf-8",
            )
            outcome_records.append(
                {
                    "replicate_seed": seed,
                    "empirical_success": success,
                    "accepted": True,
                    "outcome_artifact": {
                        "path": str(outcome),
                        "sha256": hashlib.sha256(outcome.read_bytes()).hexdigest(),
                    },
                    "action_trace_artifact": {
                        "path": str(trace),
                        "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                    },
                }
            )
        artifact = evidence_root / f"{row['policy_id']}.json"
        artifact.write_text(
            json.dumps(
                {
                    "schema_version": "policy_ladder_empirical_ground_truth.v1",
                    "policy_id": row["policy_id"],
                    "accepted": True,
                    "replicate_seed_ids": replicate_seed_ids,
                    "noise_amplitude": amplitude,
                    "registered_condition_manifest_sha256": (condition_manifest_sha256),
                    "registered_action_bounds_sha256": (registered_action_bounds_sha256),
                    "outcome_records": outcome_records,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        row["empirical_ground_truth_artifact"] = {
            "path": str(artifact),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
    rankings.sort(key=lambda row: (-row["score"], row["policy_id"]))
    for rank, row in enumerate(rankings, start=1):
        row["rank"] = rank
    return {
        "schema_version": "policy_ranking_scorecard.v1",
        "status": status,
        "comparison_blockers": list(blockers),
        "policy_rankings": rankings,
    }


def test_ladder_structure_and_expected_ranking() -> None:
    ladder = _ladder()

    assert ladder["schema_version"] == ladder_mod.LADDER_SCHEMA_VERSION
    assert ladder["expected_ranking"] == [
        "unitree_groot_n17_sonic",
        "unitree_groot_n17_sonic_noise_0p1",
        "unitree_groot_n17_sonic_noise_0p3",
        "unitree_groot_n17_sonic_noise_0p6",
    ]
    assert ladder["policy_comparison_mode"] is True
    assert len(ladder["required_replicate_seeds"]) == ladder_mod.DEFAULT_LADDER_SEED_COUNT
    assert ladder["replicate_seed_count"] == ladder_mod.DEFAULT_LADDER_SEED_COUNT
    # The default must actually resolve the separation the ladder targets;
    # otherwise every ordering it accepts is indistinguishable from chance.
    assert ladder["recommended_seed_count_for_default_separation"] is not None
    assert (
        ladder["replicate_seed_count"]
        >= ladder["recommended_seed_count_for_default_separation"]
    )
    candidates = ladder["policy_candidates"]
    assert [c["policy_id"] for c in candidates] == [
        *ladder["expected_ranking"],
        ladder_mod.REFERENCE_FLOOR_POLICY_ID,
    ]
    clean = candidates[0]
    assert clean["expected_rank"] == 1
    assert clean["adapter_command"] == "groot-runner --serve"
    floor = candidates[-1]
    assert floor["reference_only"] is True
    assert floor["expected_ordering_provable"] is False
    assert floor["expected_rank"] is None
    assert ladder["claim_boundary"]["rank_fidelity_result_proven"] is False
    assert ladder["claim_boundary"]["degraded_variants_are_synthetic_not_real_checkpoints"] is True


def test_ladder_noise_rung_commands_are_runnable_wrapper_invocations() -> None:
    ladder = _ladder()
    rung = ladder["policy_candidates"][2]

    assert rung["policy_id"] == "unitree_groot_n17_sonic_noise_0p3"
    parts = shlex.split(rung["adapter_command"])
    assert parts[1:3] == ["-m", "blueprint_pipeline.noise_degraded_policy_command_adapter"]
    assert parts[parts.index("--inner-command") + 1] == "groot-runner --serve"
    assert parts[parts.index("--noise-amplitude") + 1] == "0.3"
    assert parts[parts.index("--seed") + 1] == "7"
    assert parts[parts.index("--policy-id") + 1] == "unitree_groot_n17_sonic_noise_0p3"
    assert (
        parts[parts.index("--registered-action-bounds-sha256") + 1]
        == REGISTERED_ACTION_BOUNDS_SHA256
    )


def test_ladder_without_inner_command_has_no_adapter_commands() -> None:
    ladder = _ladder(inner_command=None)

    assert all(candidate["adapter_command"] is None for candidate in ladder["policy_candidates"])
    assert ladder["inner_command_configured"] is False


def test_ladder_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        _ladder(inner_policy_id="")
    with pytest.raises(ValueError):
        _ladder(amplitudes=())
    with pytest.raises(ValueError):
        _ladder(amplitudes=(0.0, 0.3))
    with pytest.raises(ValueError):
        _ladder(amplitudes=(-0.1,))
    with pytest.raises(ValueError):
        _ladder(amplitudes=(float("nan"),))
    with pytest.raises(ValueError):
        _ladder(amplitudes=(float("inf"),))


def test_underpowered_ladder_is_not_accepted_as_recovered() -> None:
    """A strict ordering of Bernoulli means at three seeds is not evidence.

    At the structural minimum replicate count the attainable per-rung rates are
    0, 1/3, 2/3 and 1, so adjacent rungs differ by exactly one success and the
    exact one-sided p-value for that difference is 0.5.  The ladder used to
    report ``recovered`` on that evidence.
    """

    ladder = _ladder(replicate_seed_count=ladder_mod.MIN_LADDER_SEED_COUNT)
    scorecard = _scorecard(
        {
            "unitree_groot_n17_sonic": 0.9,
            "unitree_groot_n17_sonic_noise_0p1": 0.7,
            "unitree_groot_n17_sonic_noise_0p3": 0.4,
            "unitree_groot_n17_sonic_noise_0p6": 0.1,
        },
        replicate_seed_ids=ladder_mod.replicate_seed_ids(
            7, ladder_mod.MIN_LADDER_SEED_COUNT
        ),
    )

    validation = ladder_mod.validate_policy_ranking_scorecard(
        scorecard, ladder, generated_at="2026-07-02T00:00:00Z"
    )

    assert validation["status"] == "inconclusive_underpowered_separation"
    assert validation["ranker_ordering_recovered"] is False
    assert (
        "ladder_empirical_separation_not_statistically_resolvable"
        in validation["blockers"]
    )
    analysis = validation["empirical_separation_analysis"]
    assert analysis["all_adjacent_pairs_resolvable"] is False
    assert analysis["resolvable_adjacent_pair_count"] == 0
    # The single-success gap between adjacent rungs is a coin flip.
    assert all(
        row["one_sided_p_value"] == 0.5 for row in analysis["adjacent_pairs"]
    )
    # And the report says how many seeds the observed separation would need.
    assert analysis["minimum_replicate_seed_count_for_statistical_separation"] > (
        ladder_mod.MIN_LADDER_SEED_COUNT
    )


def test_separation_analysis_is_computed_even_when_blocked() -> None:
    """The strength of the ordering evidence is reported before the verdict."""

    ladder = _ladder()
    scorecard = _scorecard(
        {
            "unitree_groot_n17_sonic": 0.9,
            "unitree_groot_n17_sonic_noise_0p1": 0.7,
            "unitree_groot_n17_sonic_noise_0p3": 0.4,
            "unitree_groot_n17_sonic_noise_0p6": 0.1,
        },
        status="blocked",
    )

    validation = ladder_mod.validate_policy_ranking_scorecard(
        scorecard, ladder, generated_at="2026-07-02T00:00:00Z"
    )

    assert validation["ranker_ordering_recovered"] is False
    analysis = validation["empirical_separation_analysis"]
    assert analysis["adjacent_pair_count"] == 3
    assert analysis["all_adjacent_pairs_resolvable"] is True


def test_validation_recovers_monotone_scorecard() -> None:
    ladder = _ladder()
    scorecard = _scorecard(
        {
            "unitree_groot_n17_sonic": 0.9,
            "unitree_groot_n17_sonic_noise_0p1": 0.7,
            "unitree_groot_n17_sonic_noise_0p3": 0.4,
            "unitree_groot_n17_sonic_noise_0p6": 0.1,
            ladder_mod.REFERENCE_FLOOR_POLICY_ID: 0.2,
        }
    )

    validation = ladder_mod.validate_policy_ranking_scorecard(
        scorecard, ladder, generated_at="2026-07-02T00:00:00Z"
    )

    assert validation["status"] == "recovered"
    assert validation["ranker_ordering_recovered"] is True
    assert validation["spearman_rank_correlation_vs_expected"] == 1.0
    assert validation["pairwise_violations"] == []
    assert validation["maximum_score_violation"] == 0.0
    assert (
        validation["claim_boundary"]["recovered_ordering_is_not_rank_fidelity_vs_real_world"]
        is True
    )


def test_validation_flags_inverted_pair() -> None:
    ladder = _ladder()
    scorecard = _scorecard(
        {
            "unitree_groot_n17_sonic": 0.9,
            "unitree_groot_n17_sonic_noise_0p1": 0.3,
            "unitree_groot_n17_sonic_noise_0p3": 0.6,
            "unitree_groot_n17_sonic_noise_0p6": 0.1,
        }
    )

    validation = ladder_mod.validate_policy_ranking_scorecard(scorecard, ladder)

    assert validation["status"] == "not_recovered"
    assert validation["ranker_ordering_recovered"] is False
    assert validation["pairwise_violation_count"] == 1
    violation = validation["pairwise_violations"][0]
    assert violation["expected_better_policy_id"] == "unitree_groot_n17_sonic_noise_0p1"
    assert violation["expected_worse_policy_id"] == "unitree_groot_n17_sonic_noise_0p3"
    assert violation["score_violation"] == pytest.approx(0.3)
    assert validation["spearman_rank_correlation_vs_expected"] < 1.0


def test_validation_reports_ties_separately() -> None:
    ladder = _ladder()
    scorecard = _scorecard(
        {
            "unitree_groot_n17_sonic": 0.9,
            "unitree_groot_n17_sonic_noise_0p1": 0.5,
            "unitree_groot_n17_sonic_noise_0p3": 0.5,
            "unitree_groot_n17_sonic_noise_0p6": 0.1,
        }
    )

    validation = ladder_mod.validate_policy_ranking_scorecard(scorecard, ladder)

    assert validation["status"] == "inconclusive_tied_scores"
    assert validation["ranker_ordering_recovered"] is False
    assert len(validation["tied_pairs"]) == 1


def test_validation_rejects_legacy_score_field_and_unreplicated_ground_truth() -> None:
    ladder = _ladder()
    scorecard = _scorecard(
        {policy_id: 1.0 - index * 0.1 for index, policy_id in enumerate(ladder["expected_ranking"])}
    )
    for row in scorecard["policy_rankings"]:
        row.pop("predicted_success_rate")
        row["replicate_seed_count"] = 1
        row["empirical_ground_truth_accepted"] = False

    validation = ladder_mod.validate_policy_ranking_scorecard(scorecard, ladder)

    assert validation["ranker_ordering_recovered"] is False
    assert "ladder_policies_missing_from_scorecard_rankings" in validation["blockers"]
    assert "ladder_requires_multiple_replicate_seeds_per_rung" in validation["blockers"]
    assert "ladder_empirical_ground_truth_not_accepted" in validation["blockers"]


def test_validation_rejects_self_asserted_seed_count_and_ground_truth() -> None:
    ladder = _ladder()
    scorecard = _scorecard(
        {policy_id: 0.9 - index * 0.2 for index, policy_id in enumerate(ladder["expected_ranking"])}
    )
    for row in scorecard["policy_rankings"]:
        row.pop("replicate_seed_ids")
        row.pop("empirical_ground_truth_artifact")
        row["replicate_seed_count"] = 3
        row["empirical_ground_truth_accepted"] = True

    validation = ladder_mod.validate_policy_ranking_scorecard(scorecard, ladder)

    assert validation["ranker_ordering_recovered"] is False
    assert any(
        blocker.startswith("ladder_replicate_seed_ids_invalid")
        for blocker in validation["blockers"]
    )
    assert any(
        blocker.startswith("ladder_empirical_ground_truth_artifact_invalid")
        for blocker in validation["blockers"]
    )


def test_validation_requires_exact_shared_seeds_and_registered_condition() -> None:
    ladder = _ladder()
    scores = {
        policy_id: 0.9 - index * 0.2 for index, policy_id in enumerate(ladder["expected_ranking"])
    }
    unmatched = _scorecard(
        scores,
        replicate_seed_ids=[7, 104736, 999999],
    )
    unmatched_validation = ladder_mod.validate_policy_ranking_scorecard(
        unmatched,
        ladder,
    )
    assert unmatched_validation["ranker_ordering_recovered"] is False
    assert any(
        blocker.startswith("ladder_replicate_seed_set_mismatch")
        for blocker in unmatched_validation["blockers"]
    )

    drift_policy = ladder["expected_ranking"][1]
    mismatched_task = _scorecard(
        scores,
        task_by_policy={drift_policy: "different-task"},
    )
    task_validation = ladder_mod.validate_policy_ranking_scorecard(
        mismatched_task,
        ladder,
    )
    assert task_validation["ranker_ordering_recovered"] is False
    assert any(
        blocker.startswith(f"ladder_registered_condition_mismatch:{drift_policy}:")
        for blocker in task_validation["blockers"]
    )


def test_validation_requires_immutable_checkpoint_and_inner_command() -> None:
    scores = {
        policy_id: 0.9 - index * 0.2
        for index, policy_id in enumerate(_ladder()["expected_ranking"])
    }
    scorecard = _scorecard(scores)

    missing_checkpoint = ladder_mod.validate_policy_ranking_scorecard(
        scorecard,
        _ladder(inner_checkpoint_sha256=None),
    )
    assert missing_checkpoint["ranker_ordering_recovered"] is False
    assert "ladder_inner_checkpoint_sha256_invalid" in missing_checkpoint["blockers"]

    missing_command = ladder_mod.validate_policy_ranking_scorecard(
        scorecard,
        _ladder(inner_command=None),
    )
    assert missing_command["ranker_ordering_recovered"] is False
    assert "ladder_inner_policy_command_missing_or_invalid" in missing_command["blockers"]

    drifted_command_ladder = json.loads(json.dumps(_ladder()))
    drifted_candidate = drifted_command_ladder["policy_candidates"][1]
    parts = shlex.split(drifted_candidate["adapter_commands_by_seed"][0])
    parts[parts.index("--inner-command") + 1] = "different-policy-runner --serve"
    drifted_command = shlex.join(parts)
    drifted_candidate["adapter_commands_by_seed"][0] = drifted_command
    drifted_candidate["adapter_command"] = drifted_command
    drifted_digest = hashlib.sha256(drifted_command.encode("utf-8")).hexdigest()
    drifted_candidate["adapter_command_sha256"] = drifted_digest
    drifted_candidate["adapter_command_sha256_by_seed"]["7"] = drifted_digest
    drifted_validation = ladder_mod.validate_policy_ranking_scorecard(
        scorecard,
        drifted_command_ladder,
    )
    assert drifted_validation["ranker_ordering_recovered"] is False
    assert (
        "ladder_seeded_adapter_command_contract_mismatch:"
        "unitree_groot_n17_sonic_noise_0p1:7" in drifted_validation["blockers"]
    )

    aliased_clean_digest_ladder = json.loads(json.dumps(_ladder()))
    aliased_clean_digest_ladder["policy_candidates"][0]["adapter_command_sha256_by_seed"]["7"] = (
        "e" * 64
    )
    aliased_clean_validation = ladder_mod.validate_policy_ranking_scorecard(
        scorecard,
        aliased_clean_digest_ladder,
    )
    assert aliased_clean_validation["ranker_ordering_recovered"] is False
    assert (
        "ladder_clean_seeded_command_binding_mismatch:unitree_groot_n17_sonic"
        in aliased_clean_validation["blockers"]
    )

    extra_flag_ladder = json.loads(json.dumps(_ladder()))
    extra_flag_candidate = extra_flag_ladder["policy_candidates"][1]
    extra_flag_command = extra_flag_candidate["adapter_commands_by_seed"][0]
    extra_flag_command += " --print-manifest"
    extra_flag_candidate["adapter_commands_by_seed"][0] = extra_flag_command
    extra_flag_candidate["adapter_command"] = extra_flag_command
    extra_flag_digest = hashlib.sha256(extra_flag_command.encode("utf-8")).hexdigest()
    extra_flag_candidate["adapter_command_sha256"] = extra_flag_digest
    extra_flag_candidate["adapter_command_sha256_by_seed"]["7"] = extra_flag_digest
    extra_flag_validation = ladder_mod.validate_policy_ranking_scorecard(
        scorecard,
        extra_flag_ladder,
    )
    assert extra_flag_validation["ranker_ordering_recovered"] is False
    assert (
        "ladder_seeded_adapter_command_contract_mismatch:"
        "unitree_groot_n17_sonic_noise_0p1:7" in extra_flag_validation["blockers"]
    )


def test_validation_requires_strict_registered_noise_ordering() -> None:
    ladder = _ladder()
    ladder["policy_candidates"][2]["noise_amplitude"] = 0.1
    scorecard = _scorecard(
        {policy_id: 0.9 - index * 0.2 for index, policy_id in enumerate(ladder["expected_ranking"])}
    )

    validation = ladder_mod.validate_policy_ranking_scorecard(scorecard, ladder)

    assert validation["ranker_ordering_recovered"] is False
    assert "ladder_registered_noise_ordering_invalid" in validation["blockers"]


def test_validation_rejects_action_bounds_drift_and_oversized_contract() -> None:
    ladder = _ladder()
    scores = {
        policy_id: 0.9 - index * 0.2 for index, policy_id in enumerate(ladder["expected_ranking"])
    }
    drifted_evidence = _scorecard(
        scores,
        registered_action_bounds_sha256="f" * 64,
    )
    drift_validation = ladder_mod.validate_policy_ranking_scorecard(
        drifted_evidence,
        ladder,
    )
    assert drift_validation["ranker_ordering_recovered"] is False
    assert any(
        blocker.startswith("ladder_empirical_ground_truth_artifact_invalid")
        for blocker in drift_validation["blockers"]
    )

    oversized_ladder = json.loads(json.dumps(ladder))
    oversized_ladder["registered_action_bounds_contract"]["fields"]["action_chunk"]["upper"][0] = (
        noise_adapter.MAX_REGISTERED_ACTION_ABS_BOUND + 1.0
    )
    oversized_digest = noise_adapter.registered_action_bounds_sha256(
        oversized_ladder["registered_action_bounds_contract"]
    )
    oversized_ladder["registered_action_bounds_sha256"] = oversized_digest
    for candidate in oversized_ladder["policy_candidates"]:
        if candidate.get("expected_ordering_provable") is True:
            candidate["registered_action_bounds_sha256"] = oversized_digest
    oversized_validation = ladder_mod.validate_policy_ranking_scorecard(
        scorecard=_scorecard(scores),
        ladder=oversized_ladder,
    )
    assert oversized_validation["ranker_ordering_recovered"] is False
    assert any(
        "registered_action_bounds_oversized" in blocker
        for blocker in oversized_validation["blockers"]
    )


def test_validation_inconclusive_on_blocked_scorecard() -> None:
    ladder = _ladder()
    scorecard = _scorecard(
        {policy_id: 0.5 for policy_id in ladder["expected_ranking"]},
        status="blocked_inconclusive_ranking",
        blockers=["policy_comparison_policy_coverage_not_symmetric"],
    )

    validation = ladder_mod.validate_policy_ranking_scorecard(scorecard, ladder)

    assert validation["status"] == "inconclusive_scorecard_blocked"
    assert validation["ranker_ordering_recovered"] is False
    assert "scorecard_blocked_or_has_comparison_blockers" in validation["blockers"]


def test_validation_inconclusive_on_missing_ladder_policy() -> None:
    ladder = _ladder()
    scorecard = _scorecard(
        {
            "unitree_groot_n17_sonic": 0.9,
            "unitree_groot_n17_sonic_noise_0p1": 0.7,
            "unitree_groot_n17_sonic_noise_0p3": 0.4,
        }
    )

    validation = ladder_mod.validate_policy_ranking_scorecard(scorecard, ladder)

    assert validation["status"] == "inconclusive_missing_ladder_policies"
    assert validation["missing_policy_ids"] == ["unitree_groot_n17_sonic_noise_0p6"]


@pytest.mark.parametrize(
    "invalid_score",
    [float("nan"), float("inf"), -0.1, 1.1, True, "0.9"],
)
def test_validation_rejects_nonfinite_or_out_of_range_scores(
    invalid_score: object,
) -> None:
    ladder = _ladder()
    scorecard = _scorecard(
        {policy_id: 0.9 - index * 0.2 for index, policy_id in enumerate(ladder["expected_ranking"])}
    )
    scorecard["policy_rankings"][0]["predicted_success_rate"] = invalid_score

    validation = ladder_mod.validate_policy_ranking_scorecard(scorecard, ladder)

    assert validation["ranker_ordering_recovered"] is False
    assert "ladder_policies_missing_from_scorecard_rankings" in validation["blockers"]


def test_validation_floor_probe_never_fails_ordering() -> None:
    ladder = _ladder()
    scorecard = _scorecard(
        {
            "unitree_groot_n17_sonic": 0.9,
            "unitree_groot_n17_sonic_noise_0p1": 0.7,
            "unitree_groot_n17_sonic_noise_0p3": 0.4,
            "unitree_groot_n17_sonic_noise_0p6": 0.1,
            ladder_mod.REFERENCE_FLOOR_POLICY_ID: 1.0,
        }
    )

    validation = ladder_mod.validate_policy_ranking_scorecard(scorecard, ladder)

    assert validation["status"] == "recovered"
    floor = validation["reference_floor_probes"][0]
    assert floor["policy_id"] == ladder_mod.REFERENCE_FLOOR_POLICY_ID
    assert floor["observed_score"] == 1.0


def test_ladder_cli_build_and_validate_round_trip(tmp_path: Path) -> None:
    ladder_path = tmp_path / "ladder.json"
    bounds_path = tmp_path / "registered-action-bounds.json"
    bounds_path.write_text(
        json.dumps(REGISTERED_ACTION_BOUNDS, sort_keys=True),
        encoding="utf-8",
    )
    exit_code = ladder_mod.main(
        [
            "build",
            "--inner-policy-id",
            "unitree_groot_n17_sonic",
            "--inner-command",
            "groot-runner --serve",
            "--inner-checkpoint-sha256",
            "d" * 64,
            "--registered-action-bounds-manifest",
            str(bounds_path),
            "--amplitude",
            "0.1",
            "--amplitude",
            "0.3",
            "--seed",
            "7",
            "--out",
            str(ladder_path),
        ]
    )
    assert exit_code == 0
    ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
    assert ladder["noise_amplitudes"] == [0.1, 0.3]

    scorecard_path = tmp_path / "scorecard.json"
    scorecard_path.write_text(
        json.dumps(
            _scorecard(
                {
                    "unitree_groot_n17_sonic": 0.8,
                    "unitree_groot_n17_sonic_noise_0p1": 0.5,
                    "unitree_groot_n17_sonic_noise_0p3": 0.2,
                }
            )
        ),
        encoding="utf-8",
    )
    validation_path = tmp_path / "validation.json"
    exit_code = ladder_mod.main(
        [
            "validate",
            "--scorecard",
            str(scorecard_path),
            "--ladder",
            str(ladder_path),
            "--out",
            str(validation_path),
        ]
    )
    assert exit_code == 0
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["status"] == "recovered"


def test_signed_validation_producer_recomputes_and_binds_real_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ladder = _ladder()
    scorecard = _scorecard(
        {policy_id: 0.9 - index * 0.2 for index, policy_id in enumerate(ladder["expected_ranking"])}
    )
    ladder_path = tmp_path / "policy_ranking_ladder.json"
    scorecard_path = tmp_path / "policy_ranking_scorecard.json"
    output_path = tmp_path / "policy_ranking_ladder_validation.json"
    report_path = tmp_path / "policy_ranking_ladder_validation.report.json"
    key_path = tmp_path / "validation-authority.key"
    ladder_path.write_text(json.dumps(ladder, sort_keys=True), encoding="utf-8")
    scorecard_path.write_text(json.dumps(scorecard, sort_keys=True), encoding="utf-8")
    key_path.write_bytes(
        VALIDATION_AUTHORITY_PRIVATE_KEY.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )

    produced = ladder_mod.produce_signed_policy_ranking_ladder_validation(
        ladder_path=ladder_path,
        scorecard_path=scorecard_path,
        output_path=output_path,
        verification_report_path=report_path,
        signing_private_key_file=key_path,
        generated_at="2026-07-09T00:00:00Z",
    )
    validation_public_key = VALIDATION_AUTHORITY_PRIVATE_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    monkeypatch.setenv(
        claim_gate.CALIBRATION_ANCHOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        hashlib.sha256(validation_public_key).hexdigest(),
    )
    consumed = claim_gate.evaluate_wam_calibration_anchors(
        produced,
        allowed_source_root=tmp_path,
    )

    assert produced["status"] == "recovered"
    assert produced["source_validation_recomputed"] is True
    assert produced["validation_attestation"]["signature_verified"] is True
    assert '"path"' not in json.dumps(produced)
    assert consumed["anchors_passed"] is True
    assert consumed["evidence_binding_status"] == (
        claim_gate.CALIBRATION_ANCHOR_EVIDENCE_BINDING_STATUS
    )
    assert str(tmp_path) not in json.dumps(consumed)


def test_ladder_noise_rung_command_executes_wrapper_end_to_end(tmp_path: Path) -> None:
    inner = tmp_path / "fake_inner_adapter.py"
    inner.write_text(
        "\n".join(
            [
                "import json, os, sys",
                "payload = json.loads(sys.stdin.read() or '{}')",
                "response = {",
                "    'status': 'completed',",
                "    'policy_id': 'unitree_groot_n17_sonic',",
                "    'model_ran': True,",
                "    'action_bounds': {'action_chunk': {'lower': [-1.0] * 3, 'upper': [1.0] * 3}},",
                "    'action': {'action_chunk': [0.0, 0.1, 0.2]},",
                "    'claim_boundary': {},",
                "}",
                "open(os.environ['BLUEPRINT_POLICY_ACTION_OUTPUT'], 'w').write(json.dumps(response))",
            ]
        ),
        encoding="utf-8",
    )
    ladder = ladder_mod.build_known_ordering_policy_ladder(
        inner_policy_id="unitree_groot_n17_sonic",
        inner_command=f"{sys.executable} {inner}",
        inner_checkpoint_sha256="d" * 64,
        registered_action_bounds={
            "schema_version": "policy_ladder_action_bounds.v1",
            "contract_id": "test-three-action-bounds.v1",
            "action_representation": "test_action_chunk",
            "fields": {
                "action_chunk": {
                    "lower": [-1.0] * 3,
                    "upper": [1.0] * 3,
                }
            },
        },
        amplitudes=(0.3,),
        seed=7,
    )
    rung = ladder["policy_candidates"][1]

    completed = subprocess.run(
        shlex.split(rung["adapter_command"]),
        input=json.dumps({"observation": {"task_id": "contact_or_push_light_object"}}),
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    response = json.loads(completed.stdout)
    assert response["status"] == "completed"
    assert response["policy_id"] == "unitree_groot_n17_sonic_noise_0p3"
    assert response["action"]["noise_injected"] is True
    assert response["action"]["action_chunk"] != [0.0, 0.1, 0.2]
