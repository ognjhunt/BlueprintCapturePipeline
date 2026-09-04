from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import blueprint_pipeline.task_evaluation_passive_destination_cad_agent as cad_agent
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _request(tmp_path: Path) -> Path:
    draft = tmp_path / "draft.json"
    draft.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "request.json"
    cad_agent.materialize_passive_destination_cad_request(
        run_id="scene841757-passive-destination-v1",
        expected_production_commit=_head(),
        destination_identity={"id": "document-tray", "version": "v1"},
        visible_label="blue document tray",
        dimensions_m={
            "outer_x": 0.33,
            "outer_y": 0.48,
            "base_thickness": 0.005,
            "wall_thickness": 0.005,
            "wall_height_above_base": 0.02,
            "minimum_interior_x": 0.32,
            "minimum_interior_y": 0.47,
        },
        output_path=output,
    )
    return output


def _proposal(source: str) -> cad_agent.PassiveDestinationCadOutput:
    return cad_agent.PassiveDestinationCadOutput(
        cad_brief_markdown="# Passive destination\n",
        generator_source=source,
        outer_x_mm=330.0,
        outer_y_mm=480.0,
        base_thickness_mm=5.0,
        wall_thickness_mm=5.0,
        wall_height_above_base_mm=20.0,
        assumptions=[],
        cited_web_sources=[],
        uncertainty="Independent visual, static, native, and placement qualification pending.",
    )


def test_request_binds_owner_metrics_pinned_skill_and_no_web(tmp_path: Path) -> None:
    request = json.loads(_request(tmp_path).read_text(encoding="utf-8"))

    assert request["cad_backend"]["skill"] == "cad"
    assert request["cad_backend"]["web_research_allowed"] is False
    assert request["automatic_retries"] == 0
    assert request["request_digest"] == canonical_digest(
        request, digest_field="request_digest"
    )


def test_production_agent_executes_only_pinned_skill_then_reopens_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    text_root = tmp_path / "text-to-cad"
    skill = text_root / "skills/cad/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# CAD\n", encoding="utf-8")
    (text_root / "skills/cad/scripts").mkdir(parents=True)
    (text_root / "skills/cad/scripts/step").write_text("fixture\n", encoding="utf-8")
    (text_root / "packages/cadpy/src").mkdir(parents=True)
    (text_root / "packages/cadpy_metadata/src").mkdir(parents=True)
    monkeypatch.setattr(
        cad_agent,
        "validate_production_cad_skill_sources",
        lambda _root: {
            "receipt_digest": "sha256:" + "a" * 64,
            "sources": [
                {
                    "id": "text-to-cad",
                    "path": str(text_root),
                    "commit": cad_agent.SOURCE_SPECS[0]["commit"],
                    "tree": cad_agent.SOURCE_SPECS[0]["tree"],
                }
            ],
        },
    )
    observed: dict[str, object] = {}

    class Invoker:
        def invoke(self, spec, input_value):
            observed["spec"] = spec
            observed["input"] = json.loads(input_value)
            return SimpleNamespace(
                output=_proposal(
                    "from build123d import Align, Box\n\n"
                    "def gen_step():\n"
                    "    return Box(330, 480, 25, align=(Align.CENTER, Align.CENTER, Align.MIN))\n"
                ),
                provider="openai",
                model=cad_agent.MODEL,
                sdk_version="fixture",
                usage={"input_tokens": 10, "output_tokens": 10},
                cost_usd=0.01,
                cost_status="estimated",
            )

    def runner(argv, **kwargs):
        observed["argv"] = argv
        observed["env"] = kwargs["env"]
        output_index = argv.index("--output") + 1
        Path(argv[output_index]).write_bytes(b"STEP fixture")
        Path(argv[argv.index("--stl") + 1]).write_bytes(b"STL fixture")
        Path(argv[argv.index("--glb") + 1]).write_bytes(b"GLB fixture")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    def inspect(*, step_path, output_path):
        value = {
            "measured_envelope_mm": [330.0, 480.0, 25.0],
            "step": {"path": str(step_path)},
        }
        Path(output_path).write_text(json.dumps(value) + "\n", encoding="utf-8")
        return value

    monkeypatch.setattr(cad_agent, "materialize_step_inspection_receipt", inspect)
    result = cad_agent.execute_passive_destination_cad_agent(
        request_path=request,
        output_root=tmp_path / "output",
        cad_source_root=tmp_path / "sources",
        invoker=Invoker(),
        runner=runner,
        python_executable="/sealed/python",
    )

    assert result["status"] == (
        "candidate_authored_pending_visual_static_native_qualification"
    )
    assert result["agent"]["web_research_performed"] is False
    assert result["review_render_required"] is True
    assert result["simready_qualified"] is False
    assert observed["argv"][0] == "/sealed/python"
    assert observed["argv"][1] == str(text_root / "skills/cad/scripts/step")
    assert "packages/cadpy/src" in observed["env"]["PYTHONPATH"]


def test_agent_generator_cannot_access_files_or_network() -> None:
    with pytest.raises(
        cad_agent.PassiveDestinationCadAgentError,
        match="passive_destination_cad_generator_import_forbidden",
    ):
        cad_agent._validate_generator_source(
            "import requests\n\ndef gen_step():\n    return requests.get('https://example.com')\n"
        )
