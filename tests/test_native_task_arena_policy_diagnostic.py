from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import (
    native_task_arena_policy_diagnostic_bundle as diagnostic_bundle_module,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_bundle import (
    build_native_task_arena_policy_bundle,
)
from blueprint_pipeline.native_task_arena_policy_diagnostic_bundle import (
    DIAGNOSTIC_CLAIM_CEILING,
    DIAGNOSTIC_EXECUTION_AUTHORITY,
    RESULT_FILENAME,
    build_native_task_arena_policy_diagnostic_bundle,
    build_policy_diagnostic_execution_spec,
    load_verified_native_task_arena_policy_diagnostic_bundle,
)
from blueprint_pipeline.native_task_arena_policy_worker import (
    _admission_binding_mismatches,
)
from tests.test_adp009d_policy_episode import _run
from tests.test_native_task_arena_bundle import (
    _articulated_packet,
    _native_bundle_preflight,
    _qualified_construction,
    _runtime_source_packet,
)


def _diagnostic_controls(root: Path, scene: dict, construction: Path) -> Path:
    construction_result = json.loads(construction.read_text(encoding="utf-8"))
    pair = {
        "schema_version": "adp_task_control_pair.v1",
        "program_id": "arm-decision-proof-v1",
        "cell_id": scene["scenario"]["cell_id"],
        "task_kind": scene["task_kind"],
        "task_spec_digest": canonical_digest(scene["task_spec"]),
        "controls": [
            {
                "control_id": "zero_action_negative",
                "control_passed": True,
                "observed_outcome": "never_moved",
                "receipt_digest": "sha256:" + "1" * 64,
            },
            {
                "control_id": "deterministic_scripted_positive",
                "control_passed": False,
                "observed_outcome": "never_moved",
                "receipt_digest": "sha256:" + "2" * 64,
            },
        ],
        "execution_order": [
            "zero_action_negative",
            "deterministic_scripted_positive",
        ],
        "cell_admitted_for_policy_execution": False,
        "policy_execution_blockers": [
            "deterministic_scripted_positive_failed:never_moved"
        ],
        "candidate_policy_queried": False,
        "pair_digest": "",
    }
    pair["pair_digest"] = canonical_digest(pair, digest_field="pair_digest")
    result = {
        "schema_version": "native_task_arena_control_result.v1",
        "status": "blocked",
        "controls_qualified": False,
        "scene_plan_digest": scene["plan_digest"],
        "construction_result_digest": construction_result["result_digest"],
        "control_pair": pair,
        "candidate_policy_queried": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    path = root / "diagnostic-controls.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return path


def test_policy_diagnostic_bundle_cli_forwards_explicit_authority_paths(
    tmp_path: Path, monkeypatch
) -> None:
    spec = tmp_path / "policy-spec.json"
    spec.write_text("{}", encoding="utf-8")
    observed = {}
    monkeypatch.setattr(
        diagnostic_bundle_module,
        "build_native_task_arena_policy_diagnostic_bundle",
        lambda **kwargs: observed.update(kwargs)
        or {
            "bundle_sha256": "sha256:" + "1" * 64,
            "policy_candidate_id": "groot_n17_droid",
        },
    )

    exit_code = diagnostic_bundle_module.main(
        [
            "--job-dir",
            str(tmp_path / "job"),
            "--packet-dir",
            str(tmp_path / "packet"),
            "--construction-result",
            str(tmp_path / "construction.json"),
            "--control-result",
            str(tmp_path / "controls.json"),
            "--runtime-source-packet-receipt",
            str(tmp_path / "runtime.json"),
            "--implementation-commit",
            "a" * 40,
            "--policy-execution-spec",
            str(spec),
            "--scene-policy-readiness-path",
            str(tmp_path / "readiness.json"),
            "--scenario-suite-path",
            str(tmp_path / "suite.json"),
        ]
    )

    assert exit_code == 0
    assert observed["scene_policy_readiness_path"] == str(
        tmp_path / "readiness.json"
    )
    assert observed["scenario_suite_path"] == str(tmp_path / "suite.json")


@pytest.mark.parametrize("candidate_id", ["pi05_droid", "groot_n17_droid"])
def test_diagnostic_spec_is_canonical_reset_and_cannot_claim_scoring(
    tmp_path: Path, candidate_id: str
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _diagnostic_controls(tmp_path, scene, construction)
    output = tmp_path / f"{candidate_id}.diagnostic.json"

    spec = build_policy_diagnostic_execution_spec(
        candidate_id=candidate_id,
        scene_plan_path=packet / "native_task_arena_scene_plan.v1.json",
        construction_result_path=construction,
        control_result_path=controls,
        output_path=output,
    )

    assert spec["execution_authority"] == DIAGNOSTIC_EXECUTION_AUTHORITY
    assert spec["claim_ceiling"] == DIAGNOSTIC_CLAIM_CEILING
    assert spec["initial_state"] == "canonical_scene_reset"
    assert spec["zero_action_negative_bound_separately"] is True
    assert spec["scientific_scoring_permitted"] is False
    assert spec["ranking_permitted"] is False
    assert spec["qualification_permitted"] is False
    assert spec["controls_qualified"] is False


def test_diagnostic_refuses_reset_target_absent_external_even_when_later_passes(
    tmp_path: Path,
) -> None:
    """Do not lend a later scripted external view to a learned policy."""

    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    value = json.loads(construction.read_text(encoding="utf-8"))
    reset_external = next(
        camera
        for camera in value["camera_snapshots"][0]["cameras"]
        if camera["role"] == "external"
    )
    reset_external["observability"].update(
        {
            "passed": False,
            "semantic_passed": False,
            "pixel_count": 0,
            "pixel_fraction": 0.0,
            "bbox_xyxy": None,
            "centroid_within_margin": False,
            "claim": "camera_observes_task_object_without_site_appearance",
            "blockers": ["native_task_camera_semantic_framing_below_threshold"],
        }
    )
    later_external = json.loads(json.dumps(reset_external))
    later_external["snapshot_id"] = "contact_sweep_clearance_00"
    later_external["observability"].update(
        {
            "passed": True,
            "semantic_passed": True,
            "pixel_count": 1000,
            "pixel_fraction": 0.02,
            "bbox_xyxy": [100, 30, 180, 120],
            "centroid_within_margin": True,
            "claim": "camera_observes_task_object_in_rendered_site",
            "blockers": [],
        }
    )
    value["camera_snapshots"].append(
        {
            "snapshot_id": "contact_sweep_clearance_00",
            "cameras": [later_external],
        }
    )
    value["camera_gates"]["external"] = {
        "passed": True,
        "best_snapshot_id": "contact_sweep_clearance_00",
    }
    value["result_digest"] = canonical_digest(
        value, digest_field="result_digest"
    )
    construction.write_text(json.dumps(value), encoding="utf-8")
    controls = _diagnostic_controls(tmp_path, scene, construction)

    with pytest.raises(
        ValueError,
        match="native_task_policy_start_camera_role_not_observable:external",
    ):
        build_policy_diagnostic_execution_spec(
            candidate_id="groot_n17_droid",
            scene_plan_path=packet / "native_task_arena_scene_plan.v1.json",
            construction_result_path=construction,
            control_result_path=controls,
            output_path=tmp_path / "must-not-exist.json",
        )


def test_diagnostic_refuses_missing_zero_action_negative(tmp_path: Path) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _diagnostic_controls(tmp_path, scene, construction)
    value = json.loads(controls.read_text(encoding="utf-8"))
    value["control_pair"]["controls"][0]["control_passed"] = False
    value["control_pair"]["pair_digest"] = canonical_digest(
        value["control_pair"], digest_field="pair_digest"
    )
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    controls.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="diagnostic_controls_invalid"):
        build_policy_diagnostic_execution_spec(
            candidate_id="pi05_droid",
            scene_plan_path=packet / "native_task_arena_scene_plan.v1.json",
            construction_result_path=construction,
            control_result_path=controls,
            output_path=tmp_path / "must-not-exist.json",
        )


def test_diagnostic_bundle_is_not_an_official_policy_bundle(tmp_path: Path) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _diagnostic_controls(tmp_path, scene, construction)
    spec = build_policy_diagnostic_execution_spec(
        candidate_id="pi05_droid",
        scene_plan_path=packet / "native_task_arena_scene_plan.v1.json",
        construction_result_path=construction,
        control_result_path=controls,
        output_path=tmp_path / "diagnostic-spec.json",
    )

    receipt = build_native_task_arena_policy_diagnostic_bundle(
        job_dir=tmp_path / "diagnostic-bundle",
        packet_dir=packet,
        construction_result_path=construction,
        control_result_path=controls,
        policy_execution_spec=spec,
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="d" * 40,
        generated_at="fixed",
    )

    assert receipt["execution_mode"] == "policy_diagnostic"
    assert receipt["expected_output_filename"] == RESULT_FILENAME
    preflight = _native_bundle_preflight(
        tmp_path, receipt, name="policy-diagnostic"
    )
    assert preflight["status"] == "passed"
    assert preflight["blockers"] == []
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        worker = archive.read(
            "provider_runtime/adp_arena_provider_runner.py"
        ).decode()
        assert "scoring_authorized=not diagnostic" in worker
        assert "policy_diagnostic" in worker
        extracted = tmp_path / "diagnostic-extracted"
        archive.extractall(extracted)
    runtime = extracted / "provider_runtime"
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))  # noqa: E731
    assert _admission_binding_mismatches(
        manifest=read(runtime / "adp_arena_provider_manifest.json"),
        spec=read(
            runtime
            / "runtime_inputs/native_task_arena_policy_execution_spec.v1.json"
        ),
        construction=read(
            runtime
            / "runtime_inputs/native_task_arena_construction_result.v1.json"
        ),
        controls=read(
            runtime / "runtime_inputs/native_task_arena_control_result.v1.json"
        ),
        scene_plan=read(
            runtime / "native_task_packet/native_task_arena_scene_plan.v1.json"
        ),
        diagnostic=True,
    ) == []
    loaded = load_verified_native_task_arena_policy_diagnostic_bundle(
        tmp_path
        / "diagnostic-bundle/native_task_arena_provider_bundle_receipt.v1.json",
        expected_implementation_commit="d" * 40,
    )
    assert loaded["bundle_sha256"] == receipt["bundle_sha256"]

    with pytest.raises(ValueError, match="native_task_policy_execution_authority_invalid"):
        build_native_task_arena_policy_bundle(
            job_dir=tmp_path / "ordinary-policy-must-refuse",
            packet_dir=packet,
            construction_result_path=construction,
            control_result_path=controls,
            policy_execution_spec=spec,
            runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
            implementation_commit="d" * 40,
        )


def test_provider_preflight_rejects_diagnostic_result_name_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _diagnostic_controls(tmp_path, scene, construction)
    spec = build_policy_diagnostic_execution_spec(
        candidate_id="pi05_droid",
        scene_plan_path=packet / "native_task_arena_scene_plan.v1.json",
        construction_result_path=construction,
        control_result_path=controls,
        output_path=tmp_path / "diagnostic-spec.json",
    )
    monkeypatch.setattr(
        diagnostic_bundle_module,
        "RESULT_FILENAME",
        "native_task_arena_policy_result.v1.json",
    )
    receipt = build_native_task_arena_policy_diagnostic_bundle(
        job_dir=tmp_path / "mismatched-diagnostic-bundle",
        packet_dir=packet,
        construction_result_path=construction,
        control_result_path=controls,
        policy_execution_spec=spec,
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="e" * 40,
        generated_at="fixed",
    )

    preflight = _native_bundle_preflight(
        tmp_path, receipt, name="mismatched-policy-diagnostic"
    )
    assert preflight["status"] == "blocked"
    assert "native_task_arena_provider_manifest_invalid" in preflight["blockers"]


def test_diagnostic_episode_retains_actions_but_skips_task_scoring() -> None:
    episode = _run(scoring_authorized=False)

    assert episode["candidate_policy_queried"] is True
    assert episode["scoring_authorized"] is False
    assert episode["queries"]
    assert episode["score"] == {
        "status": "not_scored",
        "blockers": ["unqualified_controls_policy_diagnostic"],
        "claim_boundary": (
            "policy actions and simulator observations retained; task outcome "
            "not scored, ranked, or qualified"
        ),
    }


def test_allocator_routes_explicit_diagnostic_probe_without_official_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _diagnostic_controls(tmp_path, scene, construction)
    execution_path = tmp_path / "diagnostic-execution.json"
    build_policy_diagnostic_execution_spec(
        candidate_id="pi05_droid",
        scene_plan_path=packet / "native_task_arena_scene_plan.v1.json",
        construction_result_path=construction,
        control_result_path=controls,
        output_path=execution_path,
    )
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "d" * 40, "checkout_clean": True}),
    )
    monkeypatch.setattr(
        allocator,
        "run_native_task_arena_policy_diagnostic_vast",
        lambda **kwargs: observed.update(kwargs) or {"status": "dry_run_ready"},
    )
    args = [
        "gpu-canary",
        "--probe-kind",
        "native-task-arena-policy-diagnostic",
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "unused-request.json"),
        "--release-evidence",
        str(tmp_path / "unused-release.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused-model.json"),
        "--preflight-bundle",
        str(tmp_path / "unused-preflight.json"),
        "--admission-out",
        str(tmp_path / "diagnostic-admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused-bound.json"),
        "--adapter-output",
        str(tmp_path / "diagnostic-adapter.json"),
        "--pod-name",
        "native-task-policy-diagnostic",
        "--native-task-arena-packet",
        str(packet),
        "--native-task-arena-runtime-source-packet",
        str(_runtime_source_packet(tmp_path)),
        "--native-task-arena-construction-result",
        str(construction),
        "--native-task-arena-control-result",
        str(controls),
        "--native-task-arena-policy-execution-spec",
        str(execution_path),
        "--adp-job-dir",
        str(tmp_path / "diagnostic-job"),
        "--adp-max-hourly-rate-usd",
        "0.64",
        "--adp-max-spend-usd",
        "0.5",
        "--adp-hard-ttl-seconds",
        "2800",
    ]

    assert allocator.main(args) == 0
    assert observed["prepared_bundle"]["execution_mode"] == "policy_diagnostic"
    admission = json.loads((tmp_path / "diagnostic-admission.json").read_text())
    assert admission["candidate_policy_queried"] is True
    assert admission["allocation_binding"]["execution_mode"] == "policy_diagnostic"
