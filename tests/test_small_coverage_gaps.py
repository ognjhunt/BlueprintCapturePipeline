from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import industrial_ontology, local_capture, model_access_env
from blueprint_pipeline import optional_dependencies, wam_vision_success_judge
from blueprint_pipeline.agent_runtime import artifacts, contracts
from blueprint_pipeline.common import PipelineError
from blueprint_pipeline.core import optional_dependencies as core_optional_dependencies
from blueprint_pipeline.synthesis import plucker_rays


def test_model_access_env_and_optional_dependency_messages(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in (
        "HUGGINGFACE_HUB_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN_FILE",
        "HUGGINGFACE_TOKEN_FILE",
        "HUGGING_FACE_HUB_TOKEN_FILE",
        "HF_TOKEN_FILE",
        "NGC_API_KEY",
        "NVIDIA_NGC_API_KEY",
        "NGC_API_KEY_FILE",
        "NVIDIA_NGC_API_KEY_FILE",
    ):
        monkeypatch.delenv(key, raising=False)
    model_access_env.normalize_model_access_env()
    assert "HF_TOKEN" not in model_access_env.os.environ

    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", " hf-value ")
    monkeypatch.setenv("NVIDIA_NGC_API_KEY", " ngc-value ")
    model_access_env.normalize_model_access_env()
    assert model_access_env.os.environ["HF_TOKEN"] == "hf-value"
    assert model_access_env.os.environ["HUGGINGFACE_HUB_TOKEN"] == "hf-value"
    assert model_access_env.os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf-value"
    assert model_access_env.os.environ["NGC_API_KEY"] == "ngc-value"
    assert model_access_env.os.environ["NVIDIA_NGC_API_KEY"] == "ngc-value"

    for key in (
        "HUGGINGFACE_HUB_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "NGC_API_KEY",
        "NVIDIA_NGC_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    hf_file = tmp_path / "hf-token"
    ngc_file = tmp_path / "ngc-token"
    hf_file.write_text("hf-file-value\n", encoding="utf-8")
    ngc_file.write_text("ngc-file-value\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN_FILE", str(hf_file))
    monkeypatch.setenv("NGC_API_KEY_FILE", str(ngc_file))
    model_access_env.normalize_model_access_env()
    assert model_access_env.os.environ["HF_TOKEN"] == "hf-file-value"
    assert model_access_env.os.environ["NGC_API_KEY"] == "ngc-file-value"
    secret_status = model_access_env.model_access_secret_status()
    assert secret_status["huggingface"]["auth_ready"] is True
    assert secret_status["ngc"]["auth_ready"] is True
    serialized_secret_status = str(secret_status)
    assert "hf-file-value" not in serialized_secret_status
    assert "ngc-file-value" not in serialized_secret_status

    original_read_text = Path.read_text

    def raising_read_text(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self == hf_file:
            raise OSError("permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", raising_read_text)
    assert model_access_env._read_first_secret_file(
        file_env_names=("HF_TOKEN_FILE",),
        default_paths=(),
    ) == (None, None, None)

    original_stat = Path.stat

    def raising_stat(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self == hf_file:
            raise OSError("stat denied")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", original_read_text)
    original_exists = Path.exists
    original_is_file = Path.is_file

    def fake_exists(self: Path) -> bool:
        if self == hf_file:
            return True
        return original_exists(self)

    def fake_is_file(self: Path) -> bool:
        if self == hf_file:
            return True
        return original_is_file(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(Path, "is_file", fake_is_file)
    monkeypatch.setattr(Path, "stat", raising_stat)
    stat_status = model_access_env._file_status(
        path=hf_file,
        env_name="HF_TOKEN_FILE",
        configured_by_env=True,
    )
    assert stat_status["mode"] is None
    assert stat_status["size_bytes"] is None

    logger = logging.getLogger("test.optional")
    with caplog.at_level(logging.WARNING, logger="test.optional"):
        message = optional_dependencies.log_missing_optional_dependency(
            logger,
            feature="Feature",
            package="package-a",
            extra="llm",
        )
    assert optional_dependencies.install_extra_hint("llm") in message
    assert optional_dependencies.install_extra_hint is core_optional_dependencies.install_extra_hint
    assert "Feature requires optional dependency `package-a`" in caplog.text


def test_local_capture_error_branches_and_root_layout(tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="not inside a scenes"):
        local_capture.resolve_local_capture_context(tmp_path / "plain")
    with pytest.raises(PipelineError, match="does not match"):
        local_capture.resolve_local_capture_context(tmp_path / "bucket" / "scenes" / "scene-1" / "bad" / "cap-1")

    context = local_capture.resolve_local_capture_context(Path("/scenes/scene-1/captures/capture-1"))
    assert context.storage_root == Path("/")
    assert context.bucket == "/"
    assert context.capture_prefix == "scenes/scene-1/captures/capture-1"
    assert context.descriptor_uri == "gs:////scenes/scene-1/captures/capture-1/capture_descriptor.json"


def test_agent_runtime_contract_and_artifact_helpers(tmp_path: Path) -> None:
    context = local_capture.LocalCaptureContext(
        capture_root=tmp_path / "capture",
        raw_root=tmp_path / "capture" / "raw",
        pipeline_root=tmp_path / "capture" / "pipeline",
        descriptor_path=tmp_path / "capture" / "capture_descriptor.json",
        raw_complete_path=tmp_path / "capture" / "raw" / "capture_upload_complete.json",
        storage_root=tmp_path,
        bucket="bucket",
        scene_id="scene",
        capture_id="capture",
    )
    bundle = artifacts.PipelineReviewArtifacts(
        context=context,
        descriptor=types.SimpleNamespace(),
        qa_report={},
        site_intake={},
        capture_package_manifest={},
        capture_qa_scorecard={},
        task_scope_record={},
        qualification_record={},
        qualification_brief={},
        scene_graph={},
        route_graph={},
        geometry_evidence={},
        supplemental_geometry=[],
        capability_checks={},
        blocker_register={},
        readiness_decision={},
        readiness_report="",
        opportunity_handoff={},
        human_actions_required={},
        task_hypothesis_report={},
        normalized_task_hypothesis={},
    )
    assert bundle.pipeline_dir == context.pipeline_root
    with pytest.raises(PipelineError, match="Missing required pipeline artifact"):
        artifacts._read_required_json(tmp_path / "missing.json", "missing")

    output = contracts.ReviewOutputFile(name="memo", path="memo.md")
    step = contracts.ReviewStepResult(
        skill_name="skill",
        output_path="out.json",
        source="fixture",
        provider_metadata={"model": "fake"},
    )
    review = contracts.AgentReviewBundle(
        scene_id="scene",
        capture_id="capture",
        provider="openai",
        readiness_state="review_required",
        final_memo_path="memo.md",
        final_bundle_path="bundle.json",
        human_actions_required_path="actions.json",
        outputs=[output],
        steps=[step],
        runtime={"provider": "fixture"},
    ).to_dict()
    assert review["outputs"] == [{"name": "memo", "path": "memo.md"}]
    assert review["steps"][0]["provider_metadata"] == {"model": "fake"}
    assert contracts.ensure_mapping({"a": 1}) == {"a": 1}
    assert contracts.ensure_mapping(None) == {}


def test_plucker_tensor_and_zero_moment_normalization(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class FakeTensor:
        def __init__(self, array: np.ndarray) -> None:
            self.array = array
            self.unsqueeze_dim: int | None = None

        def unsqueeze(self, dim: int):
            self.unsqueeze_dim = dim
            return self

    fake_torch = types.SimpleNamespace(from_numpy=lambda array: FakeTensor(array))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    plucker = np.zeros((6, 2, 2), dtype=np.float32)
    tensor = plucker_rays.plucker_to_tensor(plucker)
    assert tensor.array is plucker
    assert tensor.unsqueeze_dim == 0
    normalized = plucker_rays.normalise_plucker(plucker)
    assert normalized.shape == (6, 2, 2)
    assert np.all(normalized == 0.0)


def test_industrial_ontology_and_wam_vision_judge_edges() -> None:
    assert industrial_ontology._normalized_tokens("") == []
    entity = industrial_ontology.classify_industrial_entity("forklift crossing")
    assert entity.entity_type == "forklift_lane"
    assert entity.to_dict()["hazard_relevant"] is True
    assert "hazard_relevant" in industrial_ontology.industrial_tags_for_label("wet floor")
    multi = industrial_ontology.classify_industrial_entities(
        ["rack tote pallet_zone traffic_zone workcell"]
    )
    assert {entity.entity_type for entity in multi} >= {
        "rack",
        "tote",
        "pallet_zone",
        "traffic_zone",
        "workcell",
    }
    assert industrial_ontology.derive_capture_plan_tags(["", None, "rack", "rack", "charger"]) == [
        "rack",
        "charger_candidate",
    ]

    assert wam_vision_success_judge._mapping({"a": 1}) == {"a": 1}
    assert wam_vision_success_judge._mapping(["bad"]) == {}
    assert wam_vision_success_judge._string_list("one") == ["one"]
    assert wam_vision_success_judge._string_list(["one", "", None, "two"]) == ["one", "two"]
    assert wam_vision_success_judge._string_list(123) == []
    assert wam_vision_success_judge._float(True, 0.5) == 0.5
    assert wam_vision_success_judge._float("bad", 0.7) == 0.7
    payload = wam_vision_success_judge.build_fixture_vision_success_labels(
        rollout_results={
            "evaluation_substrate": "fixture_wam",
            "rollouts": [
                {
                    "rollout_id": "rollout-a",
                    "predicted_success": True,
                    "uncertainty_score": 0.1,
                    "ood_flags": ["lighting"],
                    "failure_mode_ids": [],
                },
                ["ignored"],
            ],
        },
        generated_at="2026-06-20T00:00:00Z",
    )
    assert payload["status"] == "completed"
    assert payload["success_rate"] == 1.0
    assert payload["ood_label_count"] == 1
    assert payload["failure_mode_ids"] == ["wam_ood_uncertain"]
    assert (
        wam_vision_success_judge.build_fixture_vision_success_labels(rollout_results={})[
            "status"
        ]
        == "blocked_missing_rollouts"
    )
