from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.simready_graph_cad_candidate import (
    SimReadyGraphCadCandidateError,
    bind_graph_cad_candidate_receipt,
    materialize_graph_cad_candidate,
    normalize_step_header_for_digest,
    seal_graph_cad_request,
    validate_graph_cad_request,
    validate_graph_cad_candidate_binding,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "docs/arm_decision_proof_v1/manifests"


def _spec(task: str) -> tuple[Path, dict]:
    path = (
        MANIFESTS
        / f"third_scene_840920_task_{task}_simready_graph_asset_spec.v1.json"
    )
    return path, json.loads(path.read_text(encoding="utf-8"))


def _fake_step_exporter(spec: dict, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('fixture'),'2;1');\n"
        f"FILE_NAME('{spec['asset_id']}','','',(''),(''),'','','');\n"
        "ENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="ascii",
    )
    return {
        "exporter": "hermetic_fixture_exporter",
        "exporter_version": "1",
        "step_schema": "AP214",
        "part_count": sum(len(link["geometry"]) for link in spec["links"]),
        "geometry_provenance_counts": {},
    }


@pytest.mark.parametrize(
    ("task", "prompt_fragment"),
    [
        ("a", "front-loading washer"),
        ("b", "thin clamshell notebook"),
    ],
)
def test_checked_in_graph_specs_materialize_independent_cad_candidates(
    tmp_path: Path, task: str, prompt_fragment: str
) -> None:
    spec_path, spec = _spec(task)
    request = seal_graph_cad_request(
        spec=spec,
        request_id=f"840920-task-{task}-cad-v1",
        prompt=f"Create a dimensionally bounded {prompt_fragment} candidate.",
    )

    receipt = materialize_graph_cad_candidate(
        request=request,
        spec_path=spec_path,
        destination_step=tmp_path / f"task_{task}.step",
        output_receipt_path=tmp_path / f"task_{task}.receipt.json",
        exporter=_fake_step_exporter,
    )

    assert validate_graph_cad_request(request) == request
    assert receipt["asset_id"] == spec["asset_id"]
    assert receipt["source_spec"]["spec_digest"] == spec["spec_digest"]
    assert receipt["claim_boundary"]["generated_cad_candidate_only"] is True
    assert receipt["claim_boundary"]["native_simulator_import_qualified"] is False
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_cad_request_cannot_turn_prompt_into_geometry_authority() -> None:
    _, spec = _spec("a")
    request = seal_graph_cad_request(
        spec=spec,
        request_id="task-a-cad",
        prompt="invent a hidden motor and plumbing",
    )
    request["geometry_authority"] = "prompt_and_model_invention"
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )

    with pytest.raises(SimReadyGraphCadCandidateError) as excinfo:
        validate_graph_cad_request(request)

    assert "graph_cad_request_geometry_authority_invalid" in excinfo.value.codes


def test_external_candidate_receipt_binds_portably_and_rejects_byte_drift(
    tmp_path: Path,
) -> None:
    spec_path, spec = _spec("b")
    request = seal_graph_cad_request(
        spec=spec,
        request_id="task-b-cad",
        prompt="notebook candidate",
    )
    receipt_path = tmp_path / "task_b/receipt.json"
    materialize_graph_cad_candidate(
        request=request,
        spec_path=spec_path,
        destination_step=tmp_path / "task_b/candidate.step",
        output_receipt_path=receipt_path,
        exporter=_fake_step_exporter,
    )

    binding = bind_graph_cad_candidate_receipt(
        receipt_path=receipt_path, evidence_root=tmp_path
    )

    assert binding["receipt"]["relative_path"] == "task_b/receipt.json"
    assert binding["files"]["cad_output"]["relative_path"] == (
        "task_b/candidate.step"
    )
    (tmp_path / "task_b/candidate.step").write_text("changed", encoding="utf-8")
    with pytest.raises(SimReadyGraphCadCandidateError) as excinfo:
        bind_graph_cad_candidate_receipt(
            receipt_path=receipt_path, evidence_root=tmp_path
        )
    assert "graph_cad_binding_cad_output_invalid" in excinfo.value.codes


def test_cad_materializer_rejects_swapped_task_spec(tmp_path: Path) -> None:
    _, task_a = _spec("a")
    task_b_path, _ = _spec("b")
    request = seal_graph_cad_request(
        spec=task_a,
        request_id="task-a-cad",
        prompt="washer candidate",
    )

    with pytest.raises(SimReadyGraphCadCandidateError) as excinfo:
        materialize_graph_cad_candidate(
            request=request,
            spec_path=task_b_path,
            destination_step=tmp_path / "swapped.step",
            exporter=_fake_step_exporter,
        )

    assert "graph_cad_request_spec_mismatch" in excinfo.value.codes


def test_step_header_normalization_makes_wall_clock_irrelevant_to_digest() -> None:
    first = (
        "ISO-10303-21;\nFILE_NAME('Open CASCADE Shape Model',"
        "'2026-08-10T21:01:22',('Author'));\n"
        "#7 = PRODUCT('Open CASCADE STEP translator 7.8 1.1',"
        "'Open CASCADE STEP translator 7.8 1.1','',(#8));\n"
        "#8 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('1','','',#5,#6,$);\n"
        "END-ISO-10303-21;\n"
    )
    second = (
        first.replace("21:01:22", "21:03:07")
        .replace("translator 7.8 1.1", "translator 7.8 2.1")
        .replace("OCCURRENCE('1'", "OCCURRENCE('11'")
    )

    assert normalize_step_header_for_digest(first) == normalize_step_header_for_digest(
        second
    )


@pytest.mark.parametrize("task", ["a", "b"])
def test_checked_in_cad_candidate_bindings_validate_without_external_bytes(
    task: str,
) -> None:
    value = json.loads(
        (
            MANIFESTS
            / f"third_scene_840920_task_{task}_graph_cad_candidate_binding.v1.json"
        ).read_text(encoding="utf-8")
    )

    assert validate_graph_cad_candidate_binding(value) == value
