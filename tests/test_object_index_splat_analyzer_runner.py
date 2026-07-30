from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "object_index_splat_analyzer_runner.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("object_index_splat_analyzer_runner", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _clear_splat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "SPLAT_ANALYZER_ASSET_PATH",
        "SPLAT_ANALYZER_PROMPT",
        "SPLAT_ANALYZER_MAX_PROMPTS",
        "SPLAT_ANALYZER_COMMAND",
        "SPLAT_ANALYZER_RUN_LOCAL",
        "SPLAT_ANALYZER_REPO",
        "SPLAT_ANALYZER_QUALITY",
        "SPLAT_ANALYZER_INTERACTIONS_JSON",
    ):
        monkeypatch.delenv(name, raising=False)


def _capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "capture"
    (capture_root / "raw").mkdir(parents=True)
    (capture_root / "pipeline" / "worldlabs_assets").mkdir(parents=True)
    return capture_root


def _write_materialized_manifest(capture_root: Path, *items: dict[str, object]) -> None:
    manifest_path = capture_root / "pipeline" / "worldlabs_assets" / "materialized_assets_manifest.json"
    manifest_path.write_text(json.dumps({"downloads": list(items)}), encoding="utf-8")


def test_discover_splat_assets_prefers_explicit_then_materialized_ply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_splat_env(monkeypatch)
    capture_root = _capture_root(tmp_path)
    materialized_ply = capture_root / "pipeline" / "worldlabs_assets" / "worldlabs_high.ply"
    materialized_spz = capture_root / "pipeline" / "worldlabs_assets" / "worldlabs_high.spz"
    explicit_spz = capture_root / "raw" / "operator_override.spz"
    materialized_ply.write_text("ply", encoding="utf-8")
    materialized_spz.write_text("spz", encoding="utf-8")
    explicit_spz.write_text("explicit", encoding="utf-8")
    _write_materialized_manifest(
        capture_root,
        {"kind": "splat_spz", "local_path": "worldlabs_high.spz", "quality": "high"},
        {"kind": "splat_ply", "local_path": "worldlabs_high.ply", "quality": "high"},
    )
    payload = {"capture_root": str(capture_root), "raw_root": str(capture_root / "raw")}

    assets = runner.discover_splat_assets(payload)
    assert assets[0]["path"] == str(materialized_ply.resolve())
    assert assets[0]["source"] == "worldlabs_materialized_assets_manifest"

    monkeypatch.setenv("SPLAT_ANALYZER_ASSET_PATH", str(explicit_spz))
    assets = runner.discover_splat_assets(payload)
    assert assets[0]["path"] == str(explicit_spz.resolve())
    assert assets[0]["source"] == "explicit_splat_analyzer_asset_path"


def test_fixture_interactions_normalize_to_blueprint_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_splat_env(monkeypatch)
    capture_root = _capture_root(tmp_path)
    splat = capture_root / "pipeline" / "worldlabs_assets" / "worldlabs_high.ply"
    splat.write_text("ply", encoding="utf-8")
    _write_materialized_manifest(
        capture_root,
        {"kind": "splat_ply", "local_path": "worldlabs_high.ply", "quality": "high"},
    )
    interactions = tmp_path / "interactions.json"
    interactions.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "label": "cabinet door",
                        "position": {"x": 1.0, "y": 2.0, "z": 0.5},
                        "scale": {"x": 0.8, "y": 0.1, "z": 2.0},
                        "frames": [{"frame_idx": 3, "box": [0, 0, 10, 20], "score": 0.72}],
                    },
                    {
                        "label": "shelf",
                        "position": [1.15, 2.05, 1.0],
                        "size": [1.0, 0.35, 1.6],
                        "confidence": 0.55,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SPLAT_ANALYZER_INTERACTIONS_JSON", str(interactions))
    payload = {
        "capture_root": str(capture_root),
        "raw_root": str(capture_root / "raw"),
        "prompt_bank": {
            "task_specific": ["cabinet door"],
            "broad": ["shelf"],
            "all": ["cabinet door", "shelf"],
        },
        "descriptor": {
            "metadata": {
                "task_statement": "Open the cabinet door and inspect the shelf",
                "workflow_context": "kitchen opening workflow",
            }
        },
    }

    result = runner.run_splat_analyzer_backend(
        payload,
        output_path=tmp_path / "object_index_artifacts" / "splat_analyzer_output.json",
    )

    assert result["backend_status"] == "ok"
    assert result["backend_mode"] == "splat_analyzer"
    assert result["prompt_count"] == 2
    assert len(result["objects"]) == 2
    cabinet = result["objects"][0]
    assert cabinet["id"] == "splat_door_0001"
    assert cabinet["boundingBox"]["center"] == [1.0, 2.0, 0.5]
    assert cabinet["confidence"] == 0.72
    assert cabinet["evidence_frames"] == [3]
    assert cabinet["task_relevance"]["score"] >= 0.85
    assert cabinet["articulation_hints"]["interactive"] is True
    assert cabinet["provenance"]["source"] == "splat_analyzer"
    assert cabinet["provenance"]["canonical_truth"] is False
    assert cabinet["provenance"]["presentation_only"] is True
    assert cabinet["metric_placement_ready"] is False
    assert cabinet["physics_ready"] is False
    assert cabinet["boundingBox"]["kind"] == "rough_axis_aligned_interaction_volume_candidate"
    assert cabinet["provenance"]["claim_boundary"]["robot_spawn_validated"] is False
    assert cabinet["provenance"]["claim_boundary"]["metric_oriented_box_validated"] is False
    assert result["scene_relationship_candidates"]
    assert result["claim_boundary"]["physical_robot_readiness_proven"] is False


def test_input_preflight_distinguishes_standard_and_supersplat_ply(tmp_path: Path) -> None:
    standard = tmp_path / "standard.ply"
    standard.write_bytes(
        (
            "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
            + "\n".join(
                f"property float {name}"
                for name in sorted(runner.STANDARD_3DGS_VERTEX_PROPERTIES)
            )
            + "\nend_header\n"
        ).encode("ascii")
    )
    standard_report = runner.inspect_splat_analyzer_input(standard)
    assert standard_report["input_profile"] == "standard_3dgs_ply"
    assert standard_report["direct_splat_analyzer_compatible"] is True
    assert standard_report["world_axis_convention_verified"] is False

    compressed = tmp_path / "interiorgs.ply"
    compressed.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement chunk 1\n"
        b"property float min_x\nelement vertex 1\n"
        b"property uint packed_position\nproperty uint packed_rotation\n"
        b"property uint packed_scale\nproperty uint packed_color\nend_header\n"
    )
    compressed_report = runner.inspect_splat_analyzer_input(compressed)
    assert compressed_report["input_profile"] == "supersplat_chunked_compressed_ply"
    assert compressed_report["direct_splat_analyzer_compatible"] is False
    assert compressed_report["conversion_required"] is True
    assert "hash_bound_standard_3dgs_ply_derivative" in compressed_report["required_next_step"]


def test_compressed_ply_fails_closed_before_external_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_splat_env(monkeypatch)
    capture_root = _capture_root(tmp_path)
    compressed = capture_root / "pipeline" / "worldlabs_assets" / "compressed.ply"
    compressed.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement chunk 1\n"
        b"property float min_x\nelement vertex 1\n"
        b"property uint packed_position\nproperty uint packed_rotation\n"
        b"property uint packed_scale\nproperty uint packed_color\nend_header\n"
    )
    _write_materialized_manifest(
        capture_root,
        {"kind": "splat_ply", "local_path": "compressed.ply", "quality": "high"},
    )
    monkeypatch.setenv("SPLAT_ANALYZER_COMMAND", "must-not-run {SPLAT_PATH}")

    result = runner.run_splat_analyzer_backend(
        {
            "capture_root": str(capture_root),
            "raw_root": str(capture_root / "raw"),
            "prompt_bank": {"all": ["cabinet"]},
        },
        output_path=tmp_path / "artifacts" / "splat_output.json",
    )

    assert result["backend_status"] == "skipped"
    assert result["reason"] == "splat_analyzer_input_conversion_required"
    assert result["execution"]["execution_mode"] == "input_preflight"
    assert result["splat_asset"]["input_preflight"]["input_profile"] == "supersplat_chunked_compressed_ply"


def test_missing_runtime_skips_without_failing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_splat_env(monkeypatch)
    capture_root = _capture_root(tmp_path)
    splat = capture_root / "pipeline" / "worldlabs_assets" / "worldlabs_high.ply"
    splat.write_text("ply", encoding="utf-8")
    _write_materialized_manifest(
        capture_root,
        {"kind": "splat_ply", "local_path": "worldlabs_high.ply", "quality": "high"},
    )

    result = runner.run_splat_analyzer_backend(
        {
            "capture_root": str(capture_root),
            "raw_root": str(capture_root / "raw"),
            "prompt_bank": {"task_specific": ["cabinet"], "broad": [], "all": ["cabinet"]},
        },
        output_path=tmp_path / "artifacts" / "splat_output.json",
    )

    assert result["backend_status"] == "skipped"
    assert result["reason"] == "splat_analyzer_command_not_configured"
    assert result["objects"] == []
    assert result["claim_boundary"]["public_claim_upgrade_allowed"] is False


def test_main_writes_backend_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_splat_env(monkeypatch)
    capture_root = _capture_root(tmp_path)
    splat = capture_root / "pipeline" / "worldlabs_assets" / "worldlabs_high.ply"
    splat.write_text("ply", encoding="utf-8")
    _write_materialized_manifest(
        capture_root,
        {"kind": "splat_ply", "local_path": "worldlabs_high.ply", "quality": "high"},
    )
    interactions = tmp_path / "interactions.json"
    interactions.write_text(
        json.dumps({"objects": [{"label": "drawer", "position": [0, 0, 0], "size": [0.5, 0.4, 0.2]}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("SPLAT_ANALYZER_INTERACTIONS_JSON", str(interactions))
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(
        json.dumps(
            {
                "capture_root": str(capture_root),
                "raw_root": str(capture_root / "raw"),
                "prompt_bank": {"task_specific": ["drawer"], "broad": [], "all": ["drawer"]},
            }
        ),
        encoding="utf-8",
    )

    assert runner.main([str(input_path), str(output_path)]) == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["backend_status"] == "ok"
    assert payload["objects"][0]["label"] == "drawer"
