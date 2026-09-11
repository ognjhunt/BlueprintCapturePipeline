"""Freeze, split and grade without disclosing expected answers to either model."""
import json

import pytest

from blueprint_pipeline.agent_execution.comparison import freeze_sources, grade_case, _read_corpus, _task
from blueprint_pipeline.agent_execution.contracts import AgentExecutionError, digest
from tests.test_agent_production_service import fixture


def corpus(tmp_path):
    rows = [{"case_source_id": str(i), "source_sha256": digest({"source": i}), "status": "failed" if i < 14 else "completed",
        "phase": f"phase_{i % 6}", "blocker_code": "sam31_phase_file_reference_invalid" if i < 14 else None,
        "nested_blocker_codes": [], "source_kind": "retained_production_child_result"} for i in range(33)]
    source = tmp_path / "sources.json"
    source.write_text(json.dumps({"schema_version": "blueprint_agent_retained_diagnostic_sources.v1", "records": rows}))
    output = tmp_path / "corpus.json"
    return freeze_sources(source, output), output


def test_freeze_preserves_real_vs_constructed_cases_and_disjoint_group_split(tmp_path):
    value, path = corpus(tmp_path)
    assert value["case_count"] == 30 and value["retained_case_count"] == 24 and value["controlled_variant_count"] == 6
    groups = {partition: {row["group"] for row in value["cases"] if row["partition"] == partition} for partition in ("tuning", "heldout")}
    assert groups["tuning"] and groups["heldout"] and not groups["tuning"].intersection(groups["heldout"])
    assert _read_corpus(path) == value
    value["cases"][0]["expected"]["cause"] = "completed_stage"
    path.write_text(json.dumps(value))
    with pytest.raises(AgentExecutionError, match="corpus_changed"):
        _read_corpus(path)


def test_model_input_and_tool_output_do_not_contain_expected_answer(tmp_path):
    value, _ = corpus(tmp_path)
    service, *_ = fixture(tmp_path / "service")
    case = value["cases"][0]
    task, tool = _task(service, case, value, "openai_agents_sdk")
    assert "expected" not in json.dumps(task.input)
    output = tool.invoke({}, None)
    assert set(output) == {"case_id", "evidence", "evidence_digest"}
    state = {"state": "completed", "result": {"output": {"cause": case["expected"]["cause"],
        "next_actions": case["expected"]["required_actions"], "evidence_references": [case["evidence_digest"]]}}}
    assert not grade_case(case, state, [])["schema_valid_and_evidence_bound"]
    operations = [{"request": {"tool_id": "read_case_evidence"}, "outcome": {"success": True}}]
    assert grade_case(case, state, operations)["cause_and_action_correct"]
