from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_content_agents_driver import (
    execute_content_agents_component,
)
from blueprint_pipeline.task_evaluation_scene_configuration_content_agents_failure_evidence import (
    SCHEMA_VERSION,
    ContentAgentsRuntimeFailureEvidenceError,
    read_content_agents_runtime_result,
    retain_content_agents_runtime_failure_evidence,
)


def test_nested_setup_blocker_is_retained_and_propagated_without_secrets(
    tmp_path: Path,
) -> None:
    secret = "content-agents-runtime-secret"
    work = tmp_path / "released_content_agents_runtime"
    runtime_output = work / "runtime_output"
    runtime_output.mkdir(parents=True)
    result_path = runtime_output / "adp_content_agents_vast_result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": "adp_content_agents_vast_result.v1",
                "status": "blocked",
                "blockers": [
                    "content_agents_python312_install_failed",
                    f"setup_detail:{secret}",
                ],
                "debug": "https://provider.invalid/x?token=signed-value",
                "raw_secret_values_recorded": False,
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.CompletedProcess(
        ["run_adp_content_agents_provider_runtime.sh"],
        23,
        stdout=(
            "x" * 25_000
            + "\nBLUEPRINT_ADP_CONTENT_AGENTS_PROGRESS:runtime_dependency_bootstrap_started\n"
            + f"setup output {secret}\n"
        ),
        stderr=(
            "BLUEPRINT_ADP_CONTENT_AGENTS_BLOCKED:23\n"
            "provider rejected https://provider.invalid/y?X-Amz-Signature=signed\n"
        ),
    )
    evidence_path = work / "content_agents_runtime_failure_evidence.v1.json"

    with pytest.raises(
        ContentAgentsRuntimeFailureEvidenceError,
        match=(
            "scene_configuration_content_agents_runtime_failed:"
            "content_agents_python312_install_failed;setup_detail:REDACTED_SECRET"
        ),
    ):
        read_content_agents_runtime_result(
            completed=completed,
            runtime_result_path=result_path,
            evidence_path=evidence_path,
            secret_values=(secret,),
        )

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    encoded = evidence_path.read_text(encoding="utf-8")
    assert evidence["schema_version"] == SCHEMA_VERSION
    assert evidence["runtime_returncode"] == 23
    assert evidence["typed_blockers"] == [
        "content_agents_python312_install_failed",
        "setup_detail:REDACTED_SECRET",
    ]
    assert evidence["typed_markers"] == [
        "BLUEPRINT_ADP_CONTENT_AGENTS_PROGRESS:runtime_dependency_bootstrap_started",
        "BLUEPRINT_ADP_CONTENT_AGENTS_BLOCKED:23",
    ]
    assert evidence["stdout_tail"]["earlier_character_count_dropped"] > 0
    assert evidence["runtime_result_source_path_recorded"] is False
    assert evidence["runtime_result_source_sha256"].startswith("sha256:")
    assert evidence["raw_secret_values_recorded"] is False
    assert evidence["evidence_digest"] == canonical_digest(
        evidence, digest_field="evidence_digest"
    )
    assert secret not in encoded
    assert "signed-value" not in encoded
    assert "X-Amz-Signature=signed" not in encoded
    assert "REDACTED_SECRET" in encoded
    assert "<redacted>" in encoded
    assert len(encoded) < 90_000
    assert {
        ".venv",
        ".ovrtx_venv",
        ".ovrtx_native_venv",
        ".ovphysx_venv",
        "content_agents_source",
    }.isdisjoint(evidence_path.relative_to(work).parts)


def test_accepted_nested_runtime_does_not_emit_failure_evidence(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "adp_content_agents_vast_result.json"
    runtime_result = {
        "schema_version": "adp_content_agents_vast_result.v1",
        "status": "completed",
        "blockers": [],
        "result_digest": "",
    }
    runtime_result["result_digest"] = canonical_digest(
        runtime_result, digest_field="result_digest"
    )
    result_path.write_text(json.dumps(runtime_result), encoding="utf-8")
    evidence_path = tmp_path / "content_agents_runtime_failure_evidence.v1.json"

    result = read_content_agents_runtime_result(
        completed=subprocess.CompletedProcess(["runtime"], 0, "", ""),
        runtime_result_path=result_path,
        evidence_path=evidence_path,
    )

    assert result["blockers"] == []
    assert not evidence_path.exists()


def test_completed_nested_runtime_requires_its_canonical_result_digest(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "adp_content_agents_vast_result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": "adp_content_agents_vast_result.v1",
                "status": "completed",
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "content_agents_runtime_failure_evidence.v1.json"

    with pytest.raises(
        ContentAgentsRuntimeFailureEvidenceError,
        match="scene_configuration_content_agents_runtime_result_invalid",
    ):
        read_content_agents_runtime_result(
            completed=subprocess.CompletedProcess(["runtime"], 0, "", ""),
            runtime_result_path=result_path,
            evidence_path=evidence_path,
        )

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["typed_blockers"] == [
        "scene_configuration_content_agents_runtime_result_invalid"
    ]


def test_failure_evidence_refuses_provider_excluded_destination(
    tmp_path: Path,
) -> None:
    excluded = tmp_path / "content_agents_source"
    excluded.mkdir()

    with pytest.raises(
        ValueError,
        match="scene_configuration_content_agents_failure_evidence_path_invalid",
    ):
        retain_content_agents_runtime_failure_evidence(
            destination=excluded / "evidence.json",
            completed=subprocess.CompletedProcess(["runtime"], 2, "", ""),
            runtime_result_path=excluded / "result.json",
            runtime_result={"status": "blocked", "blockers": ["setup_failed"]},
        )


def test_component_uses_typed_failure_reader_after_nested_execution() -> None:
    source = inspect.getsource(execute_content_agents_component)

    assert "read_content_agents_runtime_result(" in source
    assert "failure_evidence_secret_values(" in source
    assert "content_agents_runtime_failure_evidence.v1.json" in source
    assert source.index("completed = runner(") < source.index(
        "read_content_agents_runtime_result("
    )
    assert "scene_configuration_content_agents_runtime_failed\"" not in source
