from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import g1_controlled_proof_setup as g1_setup
from blueprint_pipeline import owner_gpu_default_smoke_artifacts as smoke
from blueprint_pipeline import scenario_variation_instantiator as variations


def test_g1_controlled_proof_setup_private_helper_edges(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    assert g1_setup._safe_id("", "fallback") == "fallback"
    assert g1_setup._find_job_request_path(capture_root, None) is None
    (capture_root / "pipeline" / "robot_eval_jobs").mkdir(parents=True)
    assert g1_setup._find_job_request_path(capture_root, None) is None

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-1"
    job_dir.mkdir(parents=True)
    job_request = job_dir / "job_request.json"
    job_request.write_text('{"job_id":"job-1","requested_tasks":[]}', encoding="utf-8")
    assert g1_setup._find_job_request_path(capture_root, "job-1") == job_request
    assert g1_setup._find_job_request_path(capture_root, None) == job_request
    assert g1_setup._webapp_route_proof_score(None) == (-1, -1, -1, "")

    proof_dir = capture_root / "pipeline" / "webapp_route_forwarding_proof"
    proof_dir.mkdir(parents=True)
    (proof_dir / "webapp_route_forwarding_proof.json").write_text(
        json.dumps({"status": "forwarded_to_pipeline_intake", "job_request": {"job_id": "other"}}),
        encoding="utf-8",
    )
    (proof_dir / "webapp_route_forwarding_proof-1.json").write_text(
        json.dumps({"status": "forwarded_to_pipeline_intake"}),
        encoding="utf-8",
    )
    (proof_dir / "webapp_route_forwarding_proof-2.json").write_text(
        json.dumps(
            {
                "status": "forwarded_to_pipeline_intake",
                "generated_at": "2026-06-20T01:00:00Z",
                "proof_boundary": {"production_live_webapp_forwarding_proven": True},
                "pipeline_forward": {"accepted": True},
                "pipeline_intake": {"accepted": True, "input_blockers": []},
                "job_request": {
                    "job_id": "job-route",
                    "source": {"selection_state": {"task_id": "inspect"}},
                    "buyer_request_id": "buyer-1",
                },
            }
        ),
        encoding="utf-8",
    )

    route_request, route_path = g1_setup._find_webapp_route_job_request(
        capture_root, "job-route"
    )
    assert route_request["job_id"] == "job-route"
    assert route_path.endswith("webapp_route_forwarding_proof-2.json")
    job_context = g1_setup._job_context(tmp_path / "route-only-capture", "job-route")
    assert job_context["job_request_found"] is False

    route_only = tmp_path / "route-only-capture"
    route_proof_dir = route_only / "pipeline" / "webapp_route_forwarding_proof"
    route_proof_dir.mkdir(parents=True)
    (route_proof_dir / "webapp_route_forwarding_proof.json").write_text(
        json.dumps(
            {
                "status": "forwarded_to_pipeline_intake",
                "proof_boundary": {"production_live_webapp_forwarding_proven": True},
                "pipeline_forward": {"accepted": True},
                "pipeline_intake": {"accepted": True, "input_blockers": []},
                "job_request": {"job_id": "job-route"},
            }
        ),
        encoding="utf-8",
    )
    assert g1_setup._job_context(route_only, "job-route")["job_request_source"] == (
        "webapp_route_forwarding_proof"
    )


def test_owner_gpu_default_smoke_artifact_helper_edges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(ValueError, match="policy_trace_path is required"):
        smoke._path_from("", label="policy_trace_path")
    assert smoke._read_existing_manifest(tmp_path / "missing.json") == {}
    existing_manifest = tmp_path / "existing.json"
    existing_manifest.write_text('{"artifacts":[]}', encoding="utf-8")
    assert smoke._read_existing_manifest(existing_manifest) == {"artifacts": []}
    assert smoke._merge_artifacts(
        [
            "ignored",
            {"kind": "policy_trace", "path": "trace.json"},
            {"kind": "policy_trace", "path": "trace.json"},
        ],
        [{"kind": "sim_robot_pov", "path": "pov.json"}],
    ) == [
        {"kind": "policy_trace", "path": "trace.json"},
        {"kind": "sim_robot_pov", "path": "pov.json"},
    ]

    monkeypatch.setenv("BLUEPRINT_POLICY_EXECUTION_TRACE", str(tmp_path / "trace.json"))
    assert smoke._env_default("BLUEPRINT_POLICY_EXECUTION_TRACE") == str(
        tmp_path / "trace.json"
    )
    assert smoke.main(
        [
            "--policy-trace",
            str(tmp_path / "policy.json"),
            "--sim-robot-pov-evidence",
            str(tmp_path / "pov.json"),
            "--artifact-manifest",
            str(tmp_path / "artifacts.json"),
            "--sim-pov-frame",
            "frame.png",
        ]
    ) == 0
    assert "artifact_manifest=" in capsys.readouterr().out


def test_scenario_variation_instantiator_helper_edges(tmp_path: Path) -> None:
    assert variations._normal_variations("bad") == []
    assert variations._normal_variations([1, {"variation_id": "unknown"}]) == []
    assert variations._cards_from_payload({"cards": [{"id": "scenario-1"}, "bad"]}) == [
        {"id": "scenario-1"}
    ]
    assert variations._concrete_mutation("custom_owner_variation", ordinal=0) == {
        "review_mutation": {
            "variation_name": "custom_owner_variation",
            "requires_owner_mapping": True,
        }
    }
    assert variations._operation_kind("custom_owner_variation") == "review_mutation"
    assert variations._engine_mutation_plan([])["mujoco"]["status"] == "blocked_missing_instances"

    pipeline_dir = tmp_path / "pipeline"
    robot_eval_dir = pipeline_dir / "robot_eval_dataset"
    robot_eval_dir.mkdir(parents=True)
    (robot_eval_dir / "scenario_family_library.json").write_text(
        json.dumps(
            {
                "families": [
                    "ignored",
                    {
                        "family_id": "family-1",
                        "scenario_id": "scenario-1",
                        "variations": [{"variation_id": "lighting"}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    family_rows = variations._scenario_family_rows(
        pipeline_dir=pipeline_dir,
        generated_at="2026-06-20T00:00:00Z",
    )
    assert len(family_rows) == 1
    assert {row["variation_name"] for row in family_rows[0]["variations"]} >= {
        "lighting_variation",
        "object_rotation",
    }

    (robot_eval_dir / "scenario_family_library.json").unlink()
    (robot_eval_dir / "scenario_cards.json").write_text(
        json.dumps({"cards": [{"scenario_id": "scenario-card-1", "task_id": "inspect"}]}),
        encoding="utf-8",
    )
    card_rows = variations._scenario_family_rows(
        pipeline_dir=pipeline_dir,
        generated_at="2026-06-20T00:00:00Z",
    )
    assert card_rows[0]["scenario_id"] == "scenario-card-1"

    manifest = variations.build_scenario_variation_instances(
        capture_root=tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1",
        output_dir=tmp_path / "out",
        generated_at="2026-06-20T00:00:00Z",
    )
    assert manifest["status"] == "blocked_missing_scenario_families"


def test_scenario_variation_build_skips_malformed_variation_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        variations,
        "_scenario_family_rows",
        lambda **_kwargs: [
            {
                "family_id": "family-1",
                "scenario_id": "scenario-1",
                "task_id": "task-1",
                "variations": [1, {"variation_name": "unknown"}],
            }
        ],
    )

    manifest = variations.build_scenario_variation_instances(
        capture_root=tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1",
        output_dir=tmp_path / "out-malformed",
        generated_at="2026-06-20T00:00:00Z",
    )

    assert manifest["status"] == "blocked_missing_scenario_families"
