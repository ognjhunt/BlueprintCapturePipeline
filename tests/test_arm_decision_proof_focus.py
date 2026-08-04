"""Contract tests for Blueprint's sole active Arm Decision Proof program."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
PROGRAM_ROOT = ROOT / "docs" / "arm_decision_proof_v1"
CONTRACT_PATH = PROGRAM_ROOT / "north_star_contract.json"
SCHEMA_PATH = ROOT / "docs" / "schemas" / "arm_decision_proof_north_star.v1.schema.json"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_north_star_contract_is_schema_valid() -> None:
    schema = _read_json(SCHEMA_PATH)
    contract = _read_json(CONTRACT_PATH)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(contract)


def test_north_star_contract_preserves_proof_and_compatibility_boundaries() -> None:
    contract = _read_json(CONTRACT_PATH)

    assert contract["status"] == "sole_active_program"
    assert contract["execution_strategy"] == {
        "mode": "public_reference_harness_first",
        "harness_engineering": "dominant_until_harness_complete",
        "capture_feature_development": "zero_unless_measured_blocker",
        "partner_recruitment_and_protocol": "parallel_human_lane",
    }
    assert contract["north_star_metric"] == {
        "name": "prospectively_physically_validated_new_site_task_decisions",
        "current": 0,
        "target": 1,
    }
    assert contract["development_substrates"]["claim_ceiling"] == "development_only"
    assert (
        contract["development_substrates"]["primary_public_reference_candidate"]
        == "SIMPLER"
    )
    assert "sim_to_real_decision_fidelity" in contract["development_substrates"][
        "cannot_qualify"
    ]
    assert contract["compatibility"] == {
        "preserve_readers": True,
        "preserve_historical_evidence": True,
        "allow_unrelated_new_work": False,
    }


def test_canonical_active_documents_point_to_the_same_program() -> None:
    required_references = {
        ROOT / "AGENTS.md": "arm-decision-proof-v1",
        ROOT / "README.md": "Arm Decision Proof v1",
        ROOT / "PLATFORM_CONTEXT.md": "Arm Decision Proof v1",
        ROOT / "WORLD_MODEL_STRATEGY_CONTEXT.md": "Arm Decision Proof v1",
        ROOT / "VISION.md": "Arm Decision Proof v1",
        ROOT / "CLAUDE.md": "Arm Decision Proof v1",
        ROOT / "docs" / "README.md": "Arm Decision Proof v1",
        ROOT / "docs" / "architecture" / "ai-onboarding-map.md": "Arm Decision Proof v1",
    }

    for path, marker in required_references.items():
        assert marker in path.read_text(encoding="utf-8"), f"{path} lost the focus marker"


def test_master_goal_carries_scope_and_authority_guards() -> None:
    prompt = (PROGRAM_ROOT / "MASTER_GOAL_PROMPT.md").read_text(encoding="utf-8")

    required_phrases = (
        "SOLE-FOCUS TEST",
        "USE EXISTING CAPTURES AND SCENES NOW",
        "ENGINEERING ALLOCATION",
        "Every reused asset is `development_only`",
        "No paid compute, provider job/upload",
        "Two candidates do not establish rank correlation",
        "Do not start or expand humanoids/G1",
    )
    for phrase in required_phrases:
        assert phrase in prompt
