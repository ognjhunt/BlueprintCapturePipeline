from __future__ import annotations

from pathlib import Path

import pytest

import blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_runtime as runtime_module
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_runtime import (
    STATUS,
    TaskEvaluationSceneConfigurationDiagnosticRuntimeError,
    execute_scene_configuration_diagnostic_stage_chain,
)
from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import (
    OUTPUT_AND_CLOSURE_RESERVE_SECONDS,
    SERIAL_GPU_STAGE_TIMEOUT_SECONDS,
)
from tests.test_task_evaluation_scene_configuration_provider_runtime import (
    _inputs,
    _producers,
    _registry,
)


def _diagnostic_inputs(tmp_path: Path):
    envelope, configurations = _inputs(tmp_path)
    envelope["render_inputs_result"] = {"status": "pending"}
    return envelope, configurations


def _carried_results(count: int) -> list[dict]:
    results = []
    for index in range(1, count + 1):
        result = {
            "schema_version": "task_evaluation_scene_configuration_stage_result.v1",
            "status": "completed",
            "stage_id": f"stage-{index}",
            "diagnostic_only": True,
            "qualification_eligible": False,
            "executed_inside_one_parent_provider_run": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
            "stage_result_digest": "",
        }
        result["stage_result_digest"] = canonical_digest(
            result, digest_field="stage_result_digest"
        )
        results.append(result)
    return results


def test_diagnostic_chain_resumes_stage_one_after_semantic_and_never_qualifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope, configurations = _diagnostic_inputs(tmp_path)
    checkpoint = {
        "checkpoint_digest": "sha256:" + "c" * 64,
        "diagnostic_only": True,
        "qualification_eligible": False,
    }
    monkeypatch.setattr(
        runtime_module,
        "diagnostic_checkpoint_scientific_binding_digest",
        lambda **_kwargs: "sha256:" + "b" * 64,
    )
    monkeypatch.setattr(
        runtime_module,
        "validate_scene_configuration_diagnostic_checkpoint",
        lambda **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        runtime_module,
        "hydrate_scene_configuration_diagnostic_completed_stages",
        lambda **_kwargs: [],
    )
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    observed: list[str] = []
    resume_calls: list[str] = []

    def resume(*, stage, checkpoint, **_kwargs):
        resume_calls.append(stage["stage_id"])
        assert checkpoint["diagnostic_only"] is True
        return ()

    now = 1000.0
    result = execute_scene_configuration_diagnostic_stage_chain(
        diagnostic_bootstrap_mode="checkpoint_resume",
        checkpoint_root=tmp_path / "checkpoint",
        envelope=envelope,
        configurations=configurations,
        output_root=outputs,
        registry=_registry(observed),
        producer_registry=_producers(),
        stage_one_resume_producer=resume,
        parent_deadline_epoch=(
            now
            + SERIAL_GPU_STAGE_TIMEOUT_SECONDS
            + OUTPUT_AND_CLOSURE_RESERVE_SECONDS
            + 1
        ),
        clock=lambda: now,
    )

    assert resume_calls == ["stage-1"]
    assert observed == [f"stage-{index}" for index in range(1, 7)]
    assert result["status"] == STATUS
    assert result["resumed_after_stage_one_semantic_teacher"] is True
    assert result["diagnostic_only"] is True
    assert result["qualification_eligible"] is False
    assert result["executed_inside_one_parent_provider_run"] is False
    assert result["configured_revision_publication_permitted"] is False
    assert result["offering_publication_permitted"] is False
    assert result["terminal_e2e_completion_permitted"] is False
    assert all(row["diagnostic_only"] is True for row in result["stage_results"])


def test_fresh_diagnostic_bootstrap_materializes_checkpoint_before_retention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope, configurations = _diagnostic_inputs(tmp_path)
    checkpoint = {
        "checkpoint_digest": "sha256:" + "c" * 64,
        "diagnostic_only": True,
        "qualification_eligible": False,
    }
    monkeypatch.setattr(
        runtime_module,
        "diagnostic_checkpoint_scientific_binding_digest",
        lambda **_kwargs: "sha256:" + "b" * 64,
    )
    validated_roots: list[Path] = []

    def validate(*, checkpoint_root, **_kwargs):
        validated_roots.append(Path(checkpoint_root))
        return checkpoint

    monkeypatch.setattr(
        runtime_module,
        "validate_scene_configuration_diagnostic_checkpoint",
        validate,
    )
    producers = _producers()
    original_produce = producers.execute

    def produce(**kwargs):
        result = original_produce(**kwargs)
        if kwargs["stage"]["stage_id"] == "stage-1":
            (Path(kwargs["output_root"]) / "diagnostic_checkpoint").mkdir()
        return result

    producers.execute = produce
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    checkpoints: list[tuple[int, Path]] = []
    observed: list[str] = []
    now = 1_000.0

    result = execute_scene_configuration_diagnostic_stage_chain(
        diagnostic_bootstrap_mode="fresh",
        checkpoint_root=None,
        envelope=envelope,
        configurations=configurations,
        output_root=outputs,
        registry=_registry(observed),
        producer_registry=producers,
        stage_one_resume_producer=lambda **_kwargs: pytest.fail(
            "fresh bootstrap must not call the resume producer"
        ),
        stage_checkpoint_writer=lambda rows, root: checkpoints.append(
            (len(rows), root)
        ),
        parent_deadline_epoch=now + 100_000,
        clock=lambda: now,
    )

    expected_root = outputs / "stage-1/producer/diagnostic_checkpoint"
    assert validated_roots == [expected_root]
    assert checkpoints == [(index, expected_root) for index in range(1, 7)]
    assert result["diagnostic_bootstrap_mode"] == "fresh"
    assert result["carried_completed_stage_count"] == 0
    assert all(row["diagnostic_only"] is True for row in result["stage_results"])


def test_diagnostic_chain_refuses_before_resume_when_remaining_budget_is_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope, configurations = _diagnostic_inputs(tmp_path)
    monkeypatch.setattr(
        runtime_module,
        "diagnostic_checkpoint_scientific_binding_digest",
        lambda **_kwargs: "sha256:" + "b" * 64,
    )
    monkeypatch.setattr(
        runtime_module,
        "validate_scene_configuration_diagnostic_checkpoint",
        lambda **_kwargs: {"checkpoint_digest": "sha256:" + "c" * 64},
    )
    monkeypatch.setattr(
        runtime_module,
        "hydrate_scene_configuration_diagnostic_completed_stages",
        lambda **_kwargs: [],
    )
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    resume_calls: list[str] = []
    now = 1000.0
    required = SERIAL_GPU_STAGE_TIMEOUT_SECONDS + OUTPUT_AND_CLOSURE_RESERVE_SECONDS

    with pytest.raises(
        TaskEvaluationSceneConfigurationDiagnosticRuntimeError,
        match="scene_configuration_diagnostic_runtime_budget_insufficient:stage-1",
    ):
        execute_scene_configuration_diagnostic_stage_chain(
            diagnostic_bootstrap_mode="checkpoint_resume",
            checkpoint_root=tmp_path / "checkpoint",
            envelope=envelope,
            configurations=configurations,
            output_root=outputs,
            registry=_registry([]),
            producer_registry=_producers(),
            stage_one_resume_producer=lambda **_kwargs: resume_calls.append("called") or (),
            parent_deadline_epoch=now + required - 1,
            clock=lambda: now,
        )

    assert resume_calls == []


@pytest.mark.parametrize("carried_count", [1, 3])
def test_progressive_diagnostic_chain_starts_at_first_incomplete_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    carried_count: int,
) -> None:
    envelope, configurations = _diagnostic_inputs(tmp_path)
    checkpoint = {"checkpoint_digest": "sha256:" + "c" * 64}
    carried = _carried_results(carried_count)
    monkeypatch.setattr(
        runtime_module,
        "diagnostic_checkpoint_scientific_binding_digest",
        lambda **_kwargs: "sha256:" + "b" * 64,
    )
    monkeypatch.setattr(
        runtime_module,
        "validate_scene_configuration_diagnostic_checkpoint",
        lambda **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        runtime_module,
        "hydrate_scene_configuration_diagnostic_completed_stages",
        lambda **_kwargs: carried,
    )
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    observed = [f"stage-{index}" for index in range(1, carried_count + 1)]
    resume_calls: list[str] = []
    now = 1_000.0

    result = execute_scene_configuration_diagnostic_stage_chain(
        diagnostic_bootstrap_mode="checkpoint_resume",
        checkpoint_root=tmp_path / "checkpoint",
        envelope=envelope,
        configurations=configurations,
        output_root=outputs,
        registry=_registry(observed),
        producer_registry=_producers(),
        stage_one_resume_producer=lambda **_kwargs: resume_calls.append("called") or (),
        parent_deadline_epoch=now + 100_000,
        clock=lambda: now,
    )

    assert resume_calls == []
    assert observed == [f"stage-{index}" for index in range(1, 7)]
    assert result["carried_completed_stage_count"] == carried_count
    assert result["resumed_from_stage_index"] == carried_count
    assert result["resumed_after_stage_one_semantic_teacher"] is False
