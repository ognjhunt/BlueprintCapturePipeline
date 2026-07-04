from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.site_reference_database import (
    WEBAPP_PROJECTION_SCHEMA_VERSION,
    assert_summary_projection_safe,
)
from blueprint_pipeline.site_reference_fixture import build_site_reference_database_v1_fixture

import pytest

pytestmark = pytest.mark.slow


_DENSE_TOKENS = (
    "depth_uri",
    "confidence_uri",
    "embedding_uri",
    "frame_uri",
    "thumbnail_uri",
    "T_world_camera",
    "T_site_camera",
    "intrinsics",
    "visibility_cells",
    "geometry_fingerprint",
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for child in value.values():
            keys.update(_all_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_all_keys(child))
        return keys
    return set()


def test_site_reference_v1_fixture_projects_staged_capture_without_dense_leakage(
    tmp_path: Path,
) -> None:
    first = build_site_reference_database_v1_fixture(tmp_path / "first")
    second = build_site_reference_database_v1_fixture(tmp_path / "first")

    projection_path = Path(first["summary_projection_path"])
    site_index_path = Path(first["site_reference_index_path"])
    validation_path = Path(first["retrieval_validation_path"])

    projection = _load_json(projection_path)
    assert projection["schema_version"] == WEBAPP_PROJECTION_SCHEMA_VERSION
    assert projection["storage_class"] == "firestore_summary_safe"
    assert projection["counts"]["capture_count"] == 1
    assert projection["counts"]["total_reference_frames"] > 0
    assert projection["artifact_uris"]["site_reference_index_uri"].endswith(
        "/sites/site-reference-fixture-site/reference_memory/site_reference_index.jsonl"
    )

    assert_summary_projection_safe(projection)
    projection_keys = _all_keys(projection)
    for token in _DENSE_TOKENS:
        assert token not in projection_keys

    reference_rows = _load_jsonl(site_index_path)
    assert reference_rows
    assert all(row["site_id"] == "site-reference-fixture-site" for row in reference_rows)
    assert all(row["geometry_source"] == "arkit" for row in reference_rows)
    assert all(row["privacy_source"] == "privacy/final_walkthrough.mov" for row in reference_rows)
    assert all(row.get("depth_uri") for row in reference_rows)
    assert all(row.get("embedding_uri") for row in reference_rows)
    assert all(row.get("rights_lineage") for row in reference_rows)

    validation = _load_json(validation_path)
    assert validation["record_schema_valid"] is True
    assert validation["manifest_schema_valid"] is True
    assert validation["summary_projection_safe"] is True
    assert validation["runtime_adapter_consumption"]["local_contract_ready"] is True
    assert validation["readiness"]["non_arkit_geometry"]["state"] == "not_applicable"
    assert validation["readiness"]["swm_world_model"]["state"] == "ready"
    assert validation["readiness"]["operational_live_provider_hosted"]["state"] == "blocked"

    assert first["reference_ids"] == second["reference_ids"]
    assert all(str(reference_id).startswith("ref_") for reference_id in first["reference_ids"])
