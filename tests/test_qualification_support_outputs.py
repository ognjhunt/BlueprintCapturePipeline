from __future__ import annotations

import json

import pytest

from blueprint_pipeline.qualification_support_outputs import (
    qualification_support_artifact_uris,
    qualification_support_webapp_projection,
    write_qualification_support_outputs,
)


def _launch_bundle() -> dict[str, object]:
    return {
        "qualification_summary": {"status": "review"},
        "capture_quality_summary": {"status": "good"},
        "rights_and_compliance_summary": {"status": "cleared"},
        "buyer_trust_score": {"score": 0.8},
        "recapture_requirements": {
            "required": True,
            "missing_evidence": ["robot_pov"],
            "recommendations": ["capture robot POV"],
        },
        "provider_preview_status": {"status": "not_requested"},
        "preview_status": "not_requested",
    }


def test_support_outputs_default_edge_writes_and_projects_nothing(tmp_path) -> None:
    assert write_qualification_support_outputs(
        pipeline_dir=tmp_path,
        launch_bundle=_launch_bundle(),
        enabled=False,
    ) == {}
    assert qualification_support_artifact_uris(
        bucket="bucket",
        pipeline_prefix="scenes/site/captures/cap/pipeline",
        enabled=False,
    ) == {}
    assert qualification_support_webapp_projection(
        buyer_trust_score={"score": 0.8},
        launch_bundle=_launch_bundle(),
        enabled=False,
    ) == {}
    assert list(tmp_path.iterdir()) == []


def test_admitted_support_edge_writes_and_projects_declared_artifacts(tmp_path) -> None:
    paths = write_qualification_support_outputs(
        pipeline_dir=tmp_path,
        launch_bundle=_launch_bundle(),
        enabled=True,
    )
    uris = qualification_support_artifact_uris(
        bucket="bucket",
        pipeline_prefix="scenes/site/captures/cap/pipeline",
        enabled=True,
    )
    projection = qualification_support_webapp_projection(
        buyer_trust_score={"score": 0.8},
        launch_bundle=_launch_bundle(),
        enabled=True,
    )

    assert set(paths) == {
        "qualification_summary",
        "capture_quality_summary",
        "rights_and_compliance_summary",
        "buyer_trust_score",
        "recapture_requirements",
        "provider_preview_status",
    }
    assert json.loads((tmp_path / "buyer_trust_score.json").read_text())["score"] == 0.8
    assert uris["buyer_trust_score_uri"].endswith("/buyer_trust_score.json")
    assert projection["missing_evidence"] == ["robot_pov"]
    assert projection["recapture_required"] is True


def test_admitted_support_edge_rejects_missing_payload(tmp_path) -> None:
    with pytest.raises(
        ValueError,
        match="qualification_support_payload_invalid:capture_quality_summary",
    ):
        write_qualification_support_outputs(
            pipeline_dir=tmp_path,
            launch_bundle={"qualification_summary": {}},
            enabled=True,
        )
