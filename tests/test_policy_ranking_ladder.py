from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import policy_ranking_ladder as ladder_mod


pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _ladder(**overrides):
    kwargs = {
        "inner_policy_id": "unitree_groot_n17_sonic",
        "inner_command": "groot-runner --serve",
        "amplitudes": (0.1, 0.3, 0.6),
        "seed": 7,
        "generated_at": "2026-07-02T00:00:00Z",
    }
    kwargs.update(overrides)
    return ladder_mod.build_known_ordering_policy_ladder(**kwargs)


def _scorecard(scores: dict[str, float], *, status: str = "completed", blockers=()) -> dict:
    rankings = [
        {"policy_id": policy_id, "score": score, "task_success_rate": score}
        for policy_id, score in scores.items()
    ]
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


def test_ladder_without_inner_command_has_no_adapter_commands() -> None:
    ladder = _ladder(inner_command=None)

    assert all(
        candidate["adapter_command"] is None for candidate in ladder["policy_candidates"]
    )
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
    assert validation["claim_boundary"]["recovered_ordering_is_not_rank_fidelity_vs_real_world"] is True


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

    assert validation["status"] == "recovered_with_ties"
    assert validation["ranker_ordering_recovered"] is True
    assert len(validation["tied_pairs"]) == 1


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
    exit_code = ladder_mod.main(
        [
            "build",
            "--inner-policy-id",
            "unitree_groot_n17_sonic",
            "--inner-command",
            "groot-runner --serve",
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
