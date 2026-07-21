from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from blueprint_pipeline.nvidia_asset_conditioning_review import build_asset_conditioning_review
from tests.test_simready_assets import _build_capture_root


def _write(path: Path, payload: object | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def test_cad_conditioning_is_immutable_staged_proposal_only(tmp_path: Path) -> None:
    root = _build_capture_root(tmp_path)
    source = root / "pipeline" / "buyer_assets" / "assembly.step"
    candidate = root / "pipeline" / "buyer_assets" / "assembly_candidate.usda"
    _write(source, "buyer CAD")
    _write(candidate, "#usda 1.0")
    stages = {}
    for name in (
        "import",
        "minimum_usd_validation",
        "material_proposal",
        "physics_proposal",
        "conformance",
        "report",
    ):
        path = root / "pipeline" / "buyer_assets" / f"{name}.json"
        _write(path, {"stage": name})
        stages[name] = path
    output = root / "pipeline" / "buyer_assets" / "review.json"
    result = build_asset_conditioning_review(
        capture_root=root,
        component="cad_to_simready_skill",
        buyer_need_id="buyer-need-1",
        original_asset_path=source,
        candidate_output_paths=[candidate],
        component_version="1.0",
        source_revision="revision-1",
        license_id="Apache-2.0",
        license_compatible=True,
        output_path=output,
        staged_evidence=stages,
        as_of_date="2026-07-21",
    )
    assert result["status"] == "accepted_advisory_proposal"
    assert result["original_asset"]["modified"] is False
    assert (
        result["claim_boundary"]["material_mass_friction_semantics_or_colliders_authoritative"]
        is False
    )
    schema = json.loads(
        (
            Path(__file__).parents[1] / "docs/schemas/nvidia_asset_conditioning_review.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(result)


def test_content_agent_requires_human_approval_and_pipeline_staged_input(tmp_path: Path) -> None:
    root = _build_capture_root(tmp_path)
    source = root / "pipeline" / "buyer_assets" / "input.usda"
    candidate = root / "pipeline" / "buyer_assets" / "candidate.usda"
    _write(source, "#usda 1.0")
    _write(candidate, "#usda 1.0")
    result = build_asset_conditioning_review(
        capture_root=root,
        component="content_agents",
        buyer_need_id="buyer-need-2",
        original_asset_path=source,
        candidate_output_paths=[candidate],
        component_version="preview-1",
        source_revision="revision-2",
        license_id="Apache-2.0",
        license_compatible=True,
        output_path=root / "pipeline" / "buyer_assets" / "review.json",
        as_of_date="2026-07-21",
    )
    assert result["status"] == "blocked"
    assert "asset_conditioning_human_approval_missing_or_invalid" in result["blockers"]
