from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_artifixer_driver import (
    TaskEvaluationSceneConfigurationArtifixerError,
    _read_artifixer_runtime_result,
)
from blueprint_pipeline.task_evaluation_scene_configuration_artifixer_failure_evidence import (
    SCHEMA_VERSION,
    retain_artifixer_runtime_failure_evidence,
)


def test_nested_setup_blocker_is_retained_and_propagated_without_secrets(
    tmp_path: Path,
) -> None:
    secret = "arbitrary-runtime-secret-value"
    work = tmp_path / "released_artifixer_runtime"
    output = work / "artifixer_output"
    output.mkdir(parents=True)
    runtime_result_path = output / "public_scene_artifixer3d_runtime_result.json"
    runtime_result_path.write_text(
        json.dumps(
            {
                "schema_version": "public_scene_artifixer3d_runtime_result.v1",
                "status": "blocked",
                "tasks": [],
                "blockers": [
                    "artifixer3d_3dgrut_requirements_failed",
                    f"setup_detail:{secret}",
                ],
                "debug": "https://provider.invalid/x?token=signed-value",
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.CompletedProcess(
        ["run_public_scene_artifixer3d.sh"],
        2,
        stdout=(
            "x" * 25_000
            + "\nBLUEPRINT_PUBLIC_SCENE_ARTIFIXER3D_STAGE_STARTED:setup\n"
            + f"setup output {secret}\n"
        ),
        stderr=(
            "BLUEPRINT_PUBLIC_SCENE_ARTIFIXER3D_STAGE_FINISHED:setup:returncode=2\n"
            "provider rejected https://provider.invalid/y?X-Amz-Signature=signed\n"
        ),
    )
    evidence_path = work / "artifixer_runtime_failure_evidence.v1.json"

    with pytest.raises(
        TaskEvaluationSceneConfigurationArtifixerError,
        match=(
            "scene_configuration_artifixer_runtime_failed:"
            "artifixer3d_3dgrut_requirements_failed;setup_detail:REDACTED_SECRET"
        ),
    ):
        _read_artifixer_runtime_result(
            completed=completed,
            runtime_result_path=runtime_result_path,
            evidence_path=evidence_path,
            secret_values=(secret,),
        )

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    encoded = evidence_path.read_text(encoding="utf-8")
    assert evidence["schema_version"] == SCHEMA_VERSION
    assert evidence["status"] == "blocked"
    assert evidence["runtime_returncode"] == 2
    assert evidence["typed_blockers"] == [
        "artifixer3d_3dgrut_requirements_failed",
        "setup_detail:REDACTED_SECRET",
    ]
    assert evidence["typed_markers"] == [
        "BLUEPRINT_PUBLIC_SCENE_ARTIFIXER3D_STAGE_STARTED:setup",
        "BLUEPRINT_PUBLIC_SCENE_ARTIFIXER3D_STAGE_FINISHED:setup:returncode=2",
    ]
    assert evidence["stdout_tail"]["earlier_character_count_dropped"] > 0
    assert evidence["runtime_result"]["blockers"] == evidence["typed_blockers"]
    assert evidence["runtime_result_source_path_recorded"] is False
    assert evidence["runtime_result_source_sha256"].startswith("sha256:")
    assert evidence["raw_secret_values_recorded"] is False
    assert evidence["evidence_digest"] == canonical_digest(evidence, digest_field="evidence_digest")
    assert secret not in encoded
    assert "signed-value" not in encoded
    assert "X-Amz-Signature=signed" not in encoded
    assert "REDACTED_SECRET" in encoded
    assert "<redacted>" in encoded
    assert len(encoded) < 90_000
    assert {"artifixer_output", "artifixer_execution"}.isdisjoint(
        evidence_path.relative_to(work).parts
    )


def test_accepted_nested_runtime_does_not_emit_failure_evidence(
    tmp_path: Path,
) -> None:
    runtime_result_path = tmp_path / "public_scene_artifixer3d_runtime_result.json"
    runtime_result_path.write_text(
        json.dumps(
            {
                "status": (
                    "raw_artifixer3d_candidate_completed_requires_visual_and_multiview_review"
                ),
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.CompletedProcess(["runtime"], 0, stdout="", stderr="")
    evidence_path = tmp_path / "artifixer_runtime_failure_evidence.v1.json"

    result = _read_artifixer_runtime_result(
        completed=completed,
        runtime_result_path=runtime_result_path,
        evidence_path=evidence_path,
    )

    assert result["blockers"] == []
    assert not evidence_path.exists()


def test_failure_evidence_refuses_an_excluded_destination(tmp_path: Path) -> None:
    excluded = tmp_path / "artifixer_output"
    excluded.mkdir()

    with pytest.raises(
        ValueError,
        match="scene_configuration_artifixer_failure_evidence_path_invalid",
    ):
        retain_artifixer_runtime_failure_evidence(
            destination=excluded / "evidence.json",
            completed=subprocess.CompletedProcess(["runtime"], 2, "", ""),
            runtime_result_path=excluded / "result.json",
            runtime_result={"status": "blocked", "blockers": ["setup_failed"]},
        )
