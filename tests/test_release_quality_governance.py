from __future__ import annotations

import base64
import hashlib
import io
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

import scripts.run_native_lerobot_export_probe as native_lerobot_probe
from scripts.archive_release_evidence import archive_bundle, build_retention_envelope
from scripts.build_container_production_evidence import (
    SOURCE_NAMES as CONTAINER_SOURCE_NAMES,
    build_container_production_evidence,
)
from scripts.build_cpu_full_lane_evidence import build_cpu_full_lane_evidence
from scripts.build_gpu_provider_canary_evidence import build_gpu_provider_canary_evidence
from scripts.build_release_evidence_bundle import build_bundle
from scripts.build_supply_chain_evidence import build_evidence
from scripts.run_pubsub_emulator_integration import _validated_emulator_host
from scripts.validate_release_signature_evidence import validate_signature_evidence
from scripts.verify_bandit_policy import finding_fingerprint, validate_policy
from scripts.verify_critical_capability_lanes import (
    EXPECTED_ARTIFACT_KEYS,
    GPU_RAW_ARTIFACT_KEYS,
    evaluate_scope,
    validate_policy as validate_lanes,
)
from scripts.verify_source_governance import validate_source_governance


SHA = "a" * 40
IMAGE = f"sha256:{'b' * 64}"
NOW = datetime(2026, 7, 9, 12, tzinfo=timezone.utc)
GPU_IMAGE_URI = f"registry.example/blueprint/unitree@sha256:{'b' * 64}"


def _bandit_finding(root: Path, *, severity: str = "MEDIUM") -> dict[str, object]:
    source = root / "src" / "example.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("print('x')\n", encoding="utf-8")
    return {
        "filename": str(source),
        "line_number": 1,
        "test_id": "B310",
        "issue_text": "Audit url open for permitted schemes",
        "issue_severity": severity,
        "issue_confidence": "HIGH",
        "code": "1 print('x')\n",
    }


def test_bandit_policy_accepts_only_exact_expiring_medium_triage(tmp_path: Path) -> None:
    finding = _bandit_finding(tmp_path)
    fingerprint = finding_fingerprint(finding, root=tmp_path)
    triage = {
        "schema_version": "blueprint.bandit_triage.v1",
        "scanner_version": "1.8.6",
        "entries": [
            {
                "fingerprint": fingerprint,
                "test_id": "B310",
                "filename": "src/example.py",
                "line_number": 1,
                "severity": "MEDIUM",
                "disposition": "accepted_risk",
                "owner": "@security-owner",
                "reason": "Exact legacy URL boundary accepted until the migration deadline.",
                "reviewed_on": "2026-07-09",
                "expires_on": "2026-08-08",
            }
        ],
    }

    passed = validate_policy(
        bandit_report={"results": [finding]},
        triage=triage,
        root=tmp_path,
        today=date(2026, 7, 9),
    )
    assert passed["status"] == "passed"

    high = dict(finding, issue_severity="HIGH")
    blocked = validate_policy(
        bandit_report={"results": [high]},
        triage=triage,
        root=tmp_path,
        today=date(2026, 7, 9),
    )
    assert blocked["status"] == "blocked"
    assert any(item.startswith("high_finding:") for item in blocked["blockers"])
    assert any(item.startswith("orphaned_triage_entry:") for item in blocked["blockers"])

    triage["scanner_version"] = "unexpected"
    wrong_scanner = validate_policy(
        bandit_report={"results": [finding]},
        triage=triage,
        root=tmp_path,
        today=date(2026, 7, 9),
    )
    assert "triage_scanner_version_invalid" in wrong_scanner["blockers"]

    triage["scanner_version"] = "1.8.6"
    triage["entries"][0]["line_number"] = 2  # type: ignore[index]
    metadata_mismatch = validate_policy(
        bandit_report={"results": [finding]},
        triage=triage,
        root=tmp_path,
        today=date(2026, 7, 9),
    )
    assert any(item.endswith(":line_number") for item in metadata_mismatch["blockers"])

    triage["entries"][0]["line_number"] = 1  # type: ignore[index]
    triage["entries"][0]["expires_on"] = "2027-08-08"  # type: ignore[index]
    overlong = validate_policy(
        bandit_report={"results": [finding]},
        triage=triage,
        root=tmp_path,
        today=date(2026, 7, 9),
    )
    assert any(item.endswith(":review_window_exceeds_90_days") for item in overlong["blockers"])


def test_source_governance_blocks_module_cli_and_claim_growth(tmp_path: Path) -> None:
    module = tmp_path / "src" / "blueprint_pipeline" / "small.py"
    module.parent.mkdir(parents=True)
    module.write_text("value = 'public_claim_upgrade_allowed'\n", encoding="utf-8")
    test_path = tmp_path / "tests" / "test_small.py"
    test_path.parent.mkdir()
    test_path.write_text("def test_small():\n    assert True\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="1"\n[project.scripts]\none="small:main"\n',
        encoding="utf-8",
    )
    policy = {
        "schema_version": "blueprint.source_governance_policy.v1",
        "policy_owner": "@source-owner",
        "baseline_date": "2026-07-09",
        "review_by": "2026-08-08",
        "default_max_python_module_lines": 2,
        "grandfathered_module_line_limits": {},
        "max_project_scripts": 1,
        "claim_literal_maximums": {"public_claim_upgrade_allowed": 1},
        "characterization_tests": {},
        "forbidden_personal_path_patterns": [
            r"(?<![A-Za-z0-9._~:/-])/(?:Users|home)/[A-Za-z0-9._-]+/",
        ],
    }
    assert (
        validate_source_governance(
            root=tmp_path,
            policy=policy,
            today=date(2026, 7, 9),
        )["status"]
        == "passed"
    )

    module.write_text(
        "value = 'public_claim_upgrade_allowed'\n"
        "again = 'public_claim_upgrade_allowed'\n"
        "third = True\n",
        encoding="utf-8",
    )
    blocked = validate_source_governance(
        root=tmp_path,
        policy=policy,
        today=date(2026, 7, 9),
    )
    assert "module_line_budget_exceeded:src/blueprint_pipeline/small.py:3>2" in blocked["blockers"]
    assert any(
        item.startswith("duplicated_claim_literal_budget_exceeded:") for item in blocked["blockers"]
    )

    policy["review_by"] = "2026-07-08"
    expired = validate_source_governance(
        root=tmp_path,
        policy=policy,
        today=date(2026, 7, 9),
    )
    assert "source_governance_policy_expired" in expired["blockers"]

    policy["review_by"] = "2027-07-09"
    overlong = validate_source_governance(
        root=tmp_path,
        policy=policy,
        today=date(2026, 7, 9),
    )
    assert "source_governance_review_window_exceeds_90_days" in overlong["blockers"]


def test_source_governance_claim_budget_ignores_longer_identifier_substrings(
    tmp_path: Path,
) -> None:
    module = tmp_path / "src" / "blueprint_pipeline" / "small.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "proves_public_claim_upgrade_allowed = True\n"
        "label = 'public_claim_upgrade_allowed_elsewhere'\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_small.py").write_text("def test_small(): pass\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="1"\n[project.scripts]\n',
        encoding="utf-8",
    )
    policy = {
        "schema_version": "blueprint.source_governance_policy.v1",
        "policy_owner": "@source-owner",
        "baseline_date": "2026-07-09",
        "review_by": "2026-08-08",
        "default_max_python_module_lines": 10,
        "grandfathered_module_line_limits": {},
        "max_project_scripts": 0,
        "claim_literal_maximums": {"public_claim_upgrade_allowed": 0},
        "characterization_tests": {},
        "forbidden_personal_path_patterns": [
            r"(?<![A-Za-z0-9._~:/-])/(?:Users|home)/[A-Za-z0-9._-]+/",
        ],
    }
    result = validate_source_governance(
        root=tmp_path,
        policy=policy,
        today=date(2026, 7, 9),
    )
    assert result["claim_literal_counts"]["public_claim_upgrade_allowed"] == 0
    assert result["status"] == "passed"


def test_source_governance_blocks_personal_absolute_paths(tmp_path: Path) -> None:
    module = tmp_path / "src" / "blueprint_pipeline" / "small.py"
    module.parent.mkdir(parents=True)
    personal_path = "/" + "Users/alice/workspace/private"
    module.write_text(f'path = "{personal_path}"\n', encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="1"\n[project.scripts]\n',
        encoding="utf-8",
    )
    policy = {
        "schema_version": "blueprint.source_governance_policy.v1",
        "policy_owner": "@source-owner",
        "baseline_date": "2026-07-09",
        "review_by": "2026-08-08",
        "default_max_python_module_lines": 5,
        "grandfathered_module_line_limits": {},
        "max_project_scripts": 0,
        "claim_literal_maximums": {},
        "characterization_tests": {},
        "forbidden_personal_path_patterns": [
            r"(?<![A-Za-z0-9._~:/-])/(?:Users|home)/[A-Za-z0-9._-]+/",
        ],
    }

    result = validate_source_governance(
        root=tmp_path,
        policy=policy,
        today=date(2026, 7, 9),
    )
    assert result["status"] == "blocked"
    assert (
        "personal_absolute_path_forbidden:src/blueprint_pipeline/small.py:1" in result["blockers"]
    )

    module.write_text('path = "$HOME/workspace/private"\n', encoding="utf-8")
    assert (
        validate_source_governance(
            root=tmp_path,
            policy=policy,
            today=date(2026, 7, 9),
        )["status"]
        == "passed"
    )


def _critical_policy() -> dict[str, object]:
    all_scopes = ["BASE", "SIM", "PTDP", "SC3", "PAID", "LIVE"]
    lane_scopes = {
        "cpu_full": all_scopes,
        "container_production": all_scopes,
        "gpu_provider_canary": ["SC3", "PAID", "LIVE"],
        "pubsub_emulator_integration": ["PAID", "LIVE"],
        "native_lerobot_export": ["PTDP", "SC3", "PAID", "LIVE"],
    }
    schemas = {
        "cpu_full": "blueprint.full_pytest_lane.v1",
        "container_production": "blueprint.container_production_contract.v1",
        "gpu_provider_canary": "blueprint.provider_canary_evidence.v1",
        "pubsub_emulator_integration": "blueprint.pubsub_emulator_integration.v1",
        "native_lerobot_export": "blueprint.native_lerobot_export.v1",
    }
    return {
        "schema_version": "blueprint.critical_capability_lanes.v1",
        "lanes": {
            lane_id: {
                "required_for_scopes": scopes,
                "required_check_name": lane_id,
                "evidence_schema_version": schemas[lane_id],
                "skip_is_blocker": True,
                "local_contract_command": "pytest",
            }
            for lane_id, scopes in lane_scopes.items()
        },
    }


def _critical_evidence(lane_id: str) -> dict[str, object]:
    common: dict[str, object] = {
        "schema_version": "blueprint.critical_capability_lane_evidence.v1",
        "lane_id": lane_id,
        "evidence_schema_version": _critical_policy()["lanes"][lane_id]["evidence_schema_version"],
        "generated_at": "2026-07-09T00:00:00+00:00",
        "repository_sha": SHA,
        "status": "passed",
        "executed": True,
        "skipped_count": 0,
        "artifact_digests": {
            name: f"sha256:{index:064x}"
            for index, name in enumerate(sorted(EXPECTED_ARTIFACT_KEYS[lane_id]), start=1)
        },
        "artifact_sizes": {name: 1 for name in EXPECTED_ARTIFACT_KEYS[lane_id]},
        "blockers": [],
        "claim_boundary": {"fixture_contract_only": True},
    }
    if lane_id == "cpu_full":
        common.update(
            {
                "test_count": 1,
                "passed_count": 1,
                "failure_count": 0,
                "error_count": 0,
                "planned_test_count": 1,
                "executed_test_count": 1,
                "nodeids_sha256": "a" * 64,
                "testcase_outcomes_sha256": "b" * 64,
                "skipped_testcases": [],
                "skipped_testcases_truncated": False,
            }
        )
    elif lane_id == "container_production":
        common.update(
            {
                "production_image_id": f"sha256:{'a' * 64}",
                "development_image_id": f"sha256:{'b' * 64}",
                "production_nonroot_user": True,
                "development_nonroot_user": True,
                "compose_config_valid": True,
                "systemd_security_passed": True,
                "production_runtime_smoke_passed": True,
                "development_runtime_smoke_passed": True,
            }
        )
    elif lane_id == "gpu_provider_canary":
        image_uri = GPU_IMAGE_URI
        common.update(
            {
                "image_uri": image_uri,
                "approved_image_uri": image_uri,
                "image_digest": f"sha256:{'b' * 64}",
                "canary_result_schema_version": ("unitree_groot_n17_sonic_vast_image_canary.v1"),
                "canary_status": "completed",
                "spend_limits": {
                    "max_hourly_rate_usd": 0.6,
                    "target_spend_usd": 0.2,
                    "hard_cap_usd": 0.5,
                    "max_live_minutes": 20,
                    "startup_timeout_seconds": 900,
                },
                "result_contract": {
                    "heartbeat_completed": True,
                    "gpu_sanity_completed": True,
                    "provider_bundle_downloaded_and_ran": True,
                    "provider_output_upload_ok": True,
                    "provider_runtime_output_zip_produced": True,
                    "canary_marker_observed": True,
                    "continuing_spend_from_this_run": False,
                },
                "estimated_cost_usd": 0.05,
                "selected_machine_id": 1,
                "selected_gpu_ram_mb": 49152,
                "selected_hourly_rate_usd": 0.3,
                "actual_live_runtime_seconds": 600.0,
                "raw_artifact_digests": {
                    name: f"sha256:{index + 100:064x}"
                    for index, name in enumerate(sorted(GPU_RAW_ARTIFACT_KEYS), start=1)
                },
                "source_manifest_digest": f"sha256:{'d' * 64}",
                "source_bundle_digest": f"sha256:{'e' * 64}",
            }
        )
    elif lane_id == "pubsub_emulator_integration":
        common.update(
            {
                "emulator_loopback_only": True,
                "published_message_id": "message-1",
                "round_trip_payload_received": True,
                "message_acknowledged": True,
                "cleanup_succeeded": True,
                "round_trip_payload_sha256": f"sha256:{'d' * 64}",
            }
        )
    elif lane_id == "native_lerobot_export":
        common.update(
            {
                "export_dir": "release-export",
                "export_file_count": 1,
                "export_total_bytes": 2,
                "export_tree_sha256": f"sha256:{'d' * 64}",
                "validation_report": {
                    "status": "passed",
                    "loader": "lerobot_native+hermetic",
                    "checks": {"lerobot_native_load": "passed"},
                },
            }
        )
    return common


def _write_cpu_sources(evidence_dir: Path) -> dict[str, object]:
    nodeid = "tests/test_release.py::test_release"
    digest = hashlib.sha256(nodeid.encode("utf-8")).hexdigest()
    for phase in ("planned", "executed"):
        (evidence_dir / f"full-test-lane-{phase}.json").write_text(
            json.dumps(
                {
                    "schema_version": "blueprint_full_lane_collection.v1",
                    "phase": phase,
                    "test_count": 1,
                    "nodeids_sha256": digest,
                    "nodeids": [nodeid],
                }
            ),
            encoding="utf-8",
        )
    (evidence_dir / "full-test-lane-junit.xml").write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="tests.test_release" name="test_release"><properties>'
        f'<property name="blueprint_nodeid" value="{nodeid}"/>'
        "</properties></testcase></testsuite></testsuites>",
        encoding="utf-8",
    )
    evidence = build_cpu_full_lane_evidence(
        planned=evidence_dir / "full-test-lane-planned.json",
        executed=evidence_dir / "full-test-lane-executed.json",
        junit=evidence_dir / "full-test-lane-junit.xml",
        repository_sha=SHA,
    )
    evidence["generated_at"] = "2026-07-09T00:00:00+00:00"
    return evidence


def _write_container_sources(evidence_dir: Path) -> dict[str, object]:
    source_dir = evidence_dir / "container-production-sources"
    source_dir.mkdir(exist_ok=True)
    values = {
        "production_image_id.txt": f"sha256:{'a' * 64}\n",
        "development_image_id.txt": f"sha256:{'b' * 64}\n",
        "production_user.txt": "blueprint:blueprint\n",
        "development_user.txt": "blueprint:blueprint\n",
        "compose-config.yml": "services:\n  pipeline:\n    image: test\n",
        "compose-config.ok": "compose-config-ok\n",
        "systemd-security.log": "blueprint-pipeline.service 3.2 OK\n",
        "systemd-security.ok": "systemd-security-ok\n",
        "production-runtime-smoke.log": "container-smoke-ok\n",
        "development-runtime-smoke.log": "container-smoke-ok\n",
    }
    assert set(values) == set(CONTAINER_SOURCE_NAMES.values())
    for name, value in values.items():
        (source_dir / name).write_text(value, encoding="utf-8")
    evidence = build_container_production_evidence(
        source_dir=source_dir,
        repository_sha=SHA,
    )
    evidence["generated_at"] = "2026-07-09T00:00:00+00:00"
    return evidence


def _build_gpu_evidence(evidence_dir: Path) -> dict[str, object]:
    from tests.test_gpu_provider_canary_evidence import _seed_result

    job_dir = evidence_dir / "gpu-canary-raw-run"
    evidence = build_gpu_provider_canary_evidence(
        result_path=_seed_result(job_dir),
        job_dir=job_dir,
        evidence_dir=evidence_dir,
        image_uri=GPU_IMAGE_URI,
        approved_image_uri=GPU_IMAGE_URI,
        repository_sha=SHA,
        max_hourly_rate=0.6,
        target_spend_usd=0.2,
        hard_cap_usd=0.5,
        max_live_minutes=20,
        startup_timeout_seconds=900,
    )
    evidence["generated_at"] = "2026-07-09T00:00:00+00:00"
    return evidence


def _write_pubsub_sources(evidence_dir: Path) -> dict[str, object]:
    evidence = _critical_evidence("pubsub_emulator_integration")
    payload = {"probe_id": "a" * 32, "kind": "pipeline_handoff"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    evidence["round_trip_payload_sha256"] = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    source = {
        "schema_version": "blueprint.pubsub_emulator_round_trip_source.v1",
        "generated_at": evidence["generated_at"],
        "repository_sha": SHA,
        "emulator_loopback_only": True,
        "expected_payload": payload,
        "received_payload": payload,
        "published_message_id": evidence["published_message_id"],
        "message_acknowledged": True,
        "cleanup_succeeded": True,
        "claim_boundary": {
            "local_emulator_transcript_only": True,
            "not_deployed_pubsub_proof": True,
        },
    }
    path = evidence_dir / "pubsub_emulator_round_trip_source.json"
    path.write_text(json.dumps(source, sort_keys=True), encoding="utf-8")
    content = path.read_bytes()
    evidence["artifact_digests"] = {path.name: f"sha256:{hashlib.sha256(content).hexdigest()}"}
    evidence["artifact_sizes"] = {path.name: len(content)}
    return evidence


def _write_native_sources(evidence_dir: Path) -> dict[str, object]:
    evidence = _critical_evidence("native_lerobot_export")
    content = b"{}"
    file_digest = hashlib.sha256(content).hexdigest()
    row = f"meta.json\t{len(content)}\t{file_digest}"
    evidence["export_total_bytes"] = len(content)
    evidence["export_tree_sha256"] = f"sha256:{hashlib.sha256(row.encode()).hexdigest()}"
    source = {
        "schema_version": "blueprint.native_lerobot_export_source_manifest.v1",
        "generated_at": evidence["generated_at"],
        "repository_sha": SHA,
        "export_dir": evidence["export_dir"],
        "export_file_count": 1,
        "export_total_bytes": len(content),
        "export_tree_sha256": evidence["export_tree_sha256"],
        "files": [
            {
                "path": "meta.json",
                "size": len(content),
                "sha256": f"sha256:{file_digest}",
            }
        ],
        "validation_report": evidence["validation_report"],
        "claim_boundary": {
            "relative_file_manifest_only": True,
            "source_manifest_is_not_dataset_quality_proof": True,
        },
    }
    path = evidence_dir / "native_lerobot_export_source_manifest.json"
    path.write_text(json.dumps(source, sort_keys=True), encoding="utf-8")
    source_content = path.read_bytes()
    evidence["artifact_digests"] = {
        path.name: f"sha256:{hashlib.sha256(source_content).hexdigest()}"
    }
    evidence["artifact_sizes"] = {path.name: len(source_content)}
    return evidence


def _write_critical_evidence(evidence_dir: Path, lane_id: str) -> dict[str, object]:
    if lane_id == "cpu_full":
        evidence = _write_cpu_sources(evidence_dir)
    elif lane_id == "container_production":
        evidence = _write_container_sources(evidence_dir)
    elif lane_id == "gpu_provider_canary":
        evidence = _build_gpu_evidence(evidence_dir)
    elif lane_id == "pubsub_emulator_integration":
        evidence = _write_pubsub_sources(evidence_dir)
    elif lane_id == "native_lerobot_export":
        evidence = _write_native_sources(evidence_dir)
    else:
        evidence = _critical_evidence(lane_id)
    (evidence_dir / f"{lane_id}.json").write_text(json.dumps(evidence), encoding="utf-8")
    return evidence


def test_critical_lane_policy_and_evidence_are_fail_closed(tmp_path: Path) -> None:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text(
        "jobs:\n"
        + "".join(
            f"  {lane_id}:\n    name: {lane_id}\n    runs-on: ubuntu-latest\n"
            for lane_id in _critical_policy()["lanes"]
        ),
        encoding="utf-8",
    )
    policy = _critical_policy()
    assert validate_lanes(root=tmp_path, policy=policy) == []

    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    missing = evaluate_scope(
        root=tmp_path,
        policy=policy,
        scope="BASE",
        evidence_dir=evidence_dir,
        repository_sha=SHA,
        now=NOW,
    )
    assert missing["status"] == "blocked"
    assert "critical_lane_evidence_missing:cpu_full" in missing["blockers"]

    for lane_id in ("cpu_full", "container_production"):
        _write_critical_evidence(evidence_dir, lane_id)
    assert (
        evaluate_scope(
            root=tmp_path,
            policy=policy,
            scope="BASE",
            evidence_dir=evidence_dir,
            repository_sha=SHA,
            now=NOW,
        )["status"]
        == "passed"
    )
    systemd_sentinel = evidence_dir / "container-production-sources" / "systemd-security.ok"
    systemd_sentinel.write_text("forged\n", encoding="utf-8")
    tampered_container = evaluate_scope(
        root=tmp_path,
        policy=policy,
        scope="BASE",
        evidence_dir=evidence_dir,
        repository_sha=SHA,
        now=NOW,
    )
    assert any(
        blocker.startswith("critical_lane_container_source_invalid:")
        for blocker in tampered_container["blockers"]
    )
    systemd_sentinel.write_text("systemd-security-ok\n", encoding="utf-8")

    for lane_id in ("native_lerobot_export", "gpu_provider_canary"):
        _write_critical_evidence(evidence_dir, lane_id)
    expected_image = GPU_IMAGE_URI
    sc3 = evaluate_scope(
        root=tmp_path,
        policy=policy,
        scope="SC3",
        evidence_dir=evidence_dir,
        repository_sha=SHA,
        expected_gpu_image_uri=expected_image,
        now=NOW,
    )
    assert sc3["status"] == "passed"
    _write_critical_evidence(evidence_dir, "pubsub_emulator_integration")
    paid = evaluate_scope(
        root=tmp_path,
        policy=policy,
        scope="PAID",
        evidence_dir=evidence_dir,
        repository_sha=SHA,
        expected_gpu_image_uri=expected_image,
        now=NOW,
    )
    assert paid["status"] == "passed"
    pubsub_source = evidence_dir / "pubsub_emulator_round_trip_source.json"
    pubsub_source.unlink()
    missing_pubsub_source = evaluate_scope(
        root=tmp_path,
        policy=policy,
        scope="PAID",
        evidence_dir=evidence_dir,
        repository_sha=SHA,
        expected_gpu_image_uri=expected_image,
        now=NOW,
    )
    assert "critical_lane_pubsub_source_missing_or_unsafe" in missing_pubsub_source["blockers"]
    _write_critical_evidence(evidence_dir, "pubsub_emulator_integration")
    pubsub_source = evidence_dir / "pubsub_emulator_round_trip_source.json"
    pubsub_source.write_text(pubsub_source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    tampered_pubsub = evaluate_scope(
        root=tmp_path,
        policy=policy,
        scope="PAID",
        evidence_dir=evidence_dir,
        repository_sha=SHA,
        expected_gpu_image_uri=expected_image,
        now=NOW,
    )
    assert "critical_lane_pubsub_source_digest_mismatch" in tampered_pubsub["blockers"]
    _write_critical_evidence(evidence_dir, "pubsub_emulator_integration")

    native_source = evidence_dir / "native_lerobot_export_source_manifest.json"
    native_source.unlink()
    missing_native_source = evaluate_scope(
        root=tmp_path,
        policy=policy,
        scope="SC3",
        evidence_dir=evidence_dir,
        repository_sha=SHA,
        expected_gpu_image_uri=expected_image,
        now=NOW,
    )
    assert "critical_lane_native_source_missing_or_unsafe" in missing_native_source["blockers"]
    _write_critical_evidence(evidence_dir, "native_lerobot_export")
    native_source = evidence_dir / "native_lerobot_export_source_manifest.json"
    native_source.write_text(native_source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    tampered_native = evaluate_scope(
        root=tmp_path,
        policy=policy,
        scope="SC3",
        evidence_dir=evidence_dir,
        repository_sha=SHA,
        expected_gpu_image_uri=expected_image,
        now=NOW,
    )
    assert "critical_lane_native_source_digest_mismatch" in tampered_native["blockers"]
    _write_critical_evidence(evidence_dir, "native_lerobot_export")

    wrong_image = evaluate_scope(
        root=tmp_path,
        policy=policy,
        scope="SC3",
        evidence_dir=evidence_dir,
        repository_sha=SHA,
        expected_gpu_image_uri=f"registry.example/wrong@sha256:{'d' * 64}",
        now=NOW,
    )
    assert "critical_lane_gpu_wrong_expected_image" in wrong_image["blockers"]
    retained_output = evidence_dir / "vast_provider_runtime_output.json"
    retained_output.write_text(
        retained_output.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    tampered_gpu_source = evaluate_scope(
        root=tmp_path,
        policy=policy,
        scope="SC3",
        evidence_dir=evidence_dir,
        repository_sha=SHA,
        expected_gpu_image_uri=expected_image,
        now=NOW,
    )
    assert any(
        blocker.startswith("critical_lane_gpu_sanitized_source_")
        for blocker in tampered_gpu_source["blockers"]
    )

    minimal = json.loads((evidence_dir / "cpu_full.json").read_text(encoding="utf-8"))
    minimal.pop("artifact_digests")
    minimal["operator_asserted"] = True
    (evidence_dir / "cpu_full.json").write_text(json.dumps(minimal), encoding="utf-8")
    malformed = evaluate_scope(
        root=tmp_path,
        policy=policy,
        scope="BASE",
        evidence_dir=evidence_dir,
        repository_sha=SHA,
        now=NOW,
    )
    assert "critical_lane_field_missing:cpu_full:artifact_digests" in malformed["blockers"]
    assert "critical_lane_field_unexpected:cpu_full:operator_asserted" in malformed["blockers"]
    _write_critical_evidence(evidence_dir, "cpu_full")

    invalid_scope = evaluate_scope(
        root=tmp_path,
        policy=policy,
        scope="UNKNOWN",
        evidence_dir=evidence_dir,
        repository_sha=SHA,
        now=NOW,
    )
    assert "critical_lane_scope_invalid:UNKNOWN" in invalid_scope["blockers"]

    (workflows / "ci.yml").write_text(
        "jobs:\n"
        "  unrelated:\n"
        "    name: unrelated\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - name: cpu_full\n"
        "        run: true\n",
        encoding="utf-8",
    )
    step_name_only = validate_lanes(root=tmp_path, policy=policy)
    assert "lane_required_check_missing:cpu_full:cpu_full" in step_name_only

    evidence = json.loads((evidence_dir / "cpu_full.json").read_text(encoding="utf-8"))
    evidence["skipped_count"] = False
    (evidence_dir / "cpu_full.json").write_text(json.dumps(evidence), encoding="utf-8")
    bool_skip = evaluate_scope(
        root=tmp_path,
        policy=policy,
        scope="BASE",
        evidence_dir=evidence_dir,
        repository_sha=SHA,
        now=NOW,
    )
    assert "critical_lane_skipped:cpu_full:False" in bool_skip["blockers"]

    evidence["skipped_count"] = 0
    evidence["generated_at"] = "2026-07-07T00:00:00+00:00"
    (evidence_dir / "cpu_full.json").write_text(json.dumps(evidence), encoding="utf-8")
    stale = evaluate_scope(
        root=tmp_path,
        policy=policy,
        scope="BASE",
        evidence_dir=evidence_dir,
        repository_sha=SHA,
        now=NOW,
    )
    assert "critical_lane_evidence_stale:cpu_full" in stale["blockers"]


def test_native_lerobot_probe_redacts_prepared_runner_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_dir = tmp_path / "private-runner-home" / "release-export"
    export_dir.mkdir(parents=True)
    (export_dir / "meta.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        native_lerobot_probe.importlib.util,
        "find_spec",
        lambda _name: object(),
    )
    monkeypatch.setattr(
        native_lerobot_probe,
        "validate_lerobot_export",
        lambda _path: {
            "status": "passed",
            "loader": "lerobot_native+hermetic",
            "checks": {"lerobot_native_load": "passed"},
            "blockers": [],
            "export_dir": str(export_dir),
        },
    )

    source_manifest = tmp_path / "native_lerobot_export_source_manifest.json"
    result = native_lerobot_probe.run_probe(
        export_dir=export_dir,
        root=tmp_path,
        source_manifest_path=source_manifest,
    )

    assert result["status"] == "passed"
    assert result["export_dir"] == "release-export"
    assert result["validation_report"]["export_dir"] == "release-export"
    assert result["export_file_count"] == 1
    assert result["artifact_digests"][source_manifest.name].startswith("sha256:")
    assert result["export_tree_sha256"].startswith("sha256:")
    assert source_manifest.is_file()
    assert str(tmp_path) not in json.dumps(result)


def test_supply_chain_evidence_requires_exact_reviewed_component_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    for name in ("uv.lock", "requirements.txt", "requirements-geometry.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo-project"\nversion = "1.0"\nlicense = "MIT"\n',
        encoding="utf-8",
    )
    artifact = tmp_path / "package.whl"
    artifact.write_bytes(b"wheel")
    sdist = tmp_path / "package.tar.gz"
    sdist.write_bytes(b"sdist")
    cdx = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "metadata": {},
        "components": [
            {
                "type": "library",
                "bom-ref": "demo@1",
                "name": "demo",
                "version": "1.0",
                "purl": "pkg:pypi/demo@1.0",
            }
        ],
    }
    policy = {
        "schema_version": "blueprint.runtime_dependency_license_policy.v1",
        "components": {
            "demo==1.0": {
                "license_expression": "MIT",
                "approved": True,
                "owner": "@legal",
                "reviewed_on": "2026-07-09",
                "expires_on": "2027-07-09",
                "source": "distribution license metadata",
            }
        },
    }
    cdx_out, spdx, provenance, report = build_evidence(
        root=tmp_path,
        cyclonedx=cdx,
        license_policy=policy,
        repository_sha=SHA,
        image_digest=IMAGE,
        artifact_paths=[artifact, sdist],
        today=date(2026, 7, 9),
    )
    assert report["status"] == "passed"
    assert cdx_out["metadata"]["properties"]
    assert spdx["spdxVersion"] == "SPDX-2.3"
    assert spdx["packages"][0]["SPDXID"] == "SPDXRef-RootPackage"
    assert spdx["relationships"][0]["relationshipType"] == "DESCRIBES"
    assert provenance["predicateType"] == "https://slsa.dev/provenance/v1"
    assert provenance["predicate"]["runDetails"]["builder"]["id"].endswith("/local-unsigned")
    assert report["signature_status"] == "external_signature_not_verified"

    cdx["components"][0]["version"] = "1.1"
    blocked = build_evidence(
        root=tmp_path,
        cyclonedx=cdx,
        license_policy=policy,
        repository_sha=SHA,
        image_digest=IMAGE,
        artifact_paths=[artifact, sdist],
        today=date(2026, 7, 9),
    )[3]
    assert "license_review_missing:demo==1.1" in blocked["blockers"]
    assert "orphaned_license_review:demo==1.0" in blocked["blockers"]


def _retention_policy(groups: list[str]) -> dict[str, object]:
    return {
        "schema_version": "blueprint.release_evidence_retention_policy.v1",
        "minimum_retention_days": 2555,
        "object_lock_mode": "COMPLIANCE",
        "require_bucket_object_lock": True,
        "require_version_id": True,
        "require_sha256_checksum_readback": True,
        "required_evidence_groups_by_scope": {"BASE": groups},
    }


def test_release_bundle_is_reproducible_and_missing_groups_block(tmp_path: Path) -> None:
    evidence = tmp_path / "ci.json"
    evidence.write_text('{"status":"success"}\n', encoding="utf-8")
    groups = {"ci": evidence}
    policy = _retention_policy(["ci"])
    first = build_bundle(
        scope="BASE",
        repository_sha=SHA,
        image_digest=IMAGE,
        evidence_groups=groups,
        policy=policy,
        output_path=tmp_path / "first.tar.gz",
        manifest_path=tmp_path / "first.json",
        generated_at="2026-07-09T00:00:00+00:00",
    )
    second = build_bundle(
        scope="BASE",
        repository_sha=SHA,
        image_digest=IMAGE,
        evidence_groups=groups,
        policy=policy,
        output_path=tmp_path / "second.tar.gz",
        manifest_path=tmp_path / "second.json",
        generated_at="2026-07-09T00:00:00+00:00",
    )
    assert first["status"] == second["status"] == "ready_to_archive"
    assert first["bundle_sha256"] == second["bundle_sha256"]

    blocked = build_bundle(
        scope="BASE",
        repository_sha=SHA,
        image_digest=IMAGE,
        evidence_groups={},
        policy=policy,
        output_path=tmp_path / "blocked.tar.gz",
        manifest_path=tmp_path / "blocked.json",
        generated_at="2026-07-09T00:00:00+00:00",
    )
    assert blocked["status"] == "blocked"
    assert "required_evidence_group_missing:ci" in blocked["blockers"]

    symlink = tmp_path / "ci-link.json"
    symlink.symlink_to(evidence)
    symlink_blocked = build_bundle(
        scope="BASE",
        repository_sha=SHA,
        image_digest=IMAGE,
        evidence_groups={"ci": symlink},
        policy=policy,
        output_path=tmp_path / "symlink.tar.gz",
        manifest_path=tmp_path / "symlink.json",
        generated_at="2026-07-09T00:00:00+00:00",
    )
    assert "evidence group may not be a symlink: ci" in symlink_blocked["blockers"]


class _FakeS3:
    def __init__(self, *, object_lock: bool = True) -> None:
        self.object_lock = object_lock
        self.body = b""
        self.put_kwargs: dict[str, object] = {}

    def get_object_lock_configuration(self, **_kwargs: object) -> dict[str, object]:
        return {
            "ObjectLockConfiguration": {
                "ObjectLockEnabled": "Enabled" if self.object_lock else "Disabled"
            }
        }

    def put_object(self, **kwargs: object) -> dict[str, object]:
        self.put_kwargs = kwargs
        self.body = kwargs["Body"].read()  # type: ignore[union-attr]
        return {"VersionId": "version-1", "ChecksumSHA256": kwargs["ChecksumSHA256"]}

    def head_object(self, **_kwargs: object) -> dict[str, object]:
        return {
            "ObjectLockMode": "COMPLIANCE",
            "ObjectLockRetainUntilDate": self.put_kwargs["ObjectLockRetainUntilDate"],
            "ContentLength": len(self.body),
            "ChecksumSHA256": self.put_kwargs["ChecksumSHA256"],
            "Metadata": self.put_kwargs["Metadata"],
        }

    def get_object(self, **_kwargs: object) -> dict[str, object]:
        return {
            "Body": io.BytesIO(self.body),
            "ChecksumSHA256": self.put_kwargs["ChecksumSHA256"],
        }


def test_archive_requires_object_lock_version_and_checksum_readback(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.tar.gz"
    evidence = tmp_path / "ci.json"
    evidence.write_text('{"status":"success"}\n', encoding="utf-8")
    manifest = build_bundle(
        scope="BASE",
        repository_sha=SHA,
        image_digest=IMAGE,
        evidence_groups={"ci": evidence},
        policy=_retention_policy(["ci"]),
        output_path=bundle,
        manifest_path=tmp_path / "bundle.json",
        generated_at="2026-07-09T00:00:00+00:00",
    )
    digest = str(manifest["bundle_sha256"])
    client = _FakeS3()
    result = archive_bundle(
        client=client,
        bundle_path=bundle,
        bundle_manifest=manifest,
        archive_uri="s3://release-evidence/pipeline",
        policy=_retention_policy(["ci"]),
        now=datetime(2026, 7, 9, tzinfo=timezone.utc),
    )
    assert result["status"] == "archived_immutable"
    assert result["version_id"] == "version-1"
    assert result["restore_readback_verified"] is True
    assert client.put_kwargs["ObjectLockMode"] == "COMPLIANCE"
    assert base64.b64decode(client.put_kwargs["ChecksumSHA256"]) == bytes.fromhex(digest)
    envelope = build_retention_envelope(
        receipt=result,
        source_artifact_digest=f"sha256:{'d' * 64}",
        now=datetime(2026, 7, 9, tzinfo=timezone.utc),
    )
    assert envelope["evidence_id"] == "immutable_retention"
    assert envelope["status"] == "archived_immutable"
    assert envelope["repository_sha"] == SHA

    blocked = archive_bundle(
        client=_FakeS3(object_lock=False),
        bundle_path=bundle,
        bundle_manifest=manifest,
        archive_uri="s3://release-evidence/pipeline",
        policy=_retention_policy(["ci"]),
        now=datetime(2026, 7, 9, tzinfo=timezone.utc),
    )
    assert blocked["status"] == "blocked"
    assert "archive_bucket_object_lock_not_enabled" in blocked["blockers"]

    weakened_policy = _retention_policy(["ci"])
    weakened_policy["minimum_retention_days"] = 365
    weakened = archive_bundle(
        client=_FakeS3(),
        bundle_path=bundle,
        bundle_manifest=manifest,
        archive_uri="s3://release-evidence/pipeline",
        policy=weakened_policy,
        now=datetime(2026, 7, 9, tzinfo=timezone.utc),
    )
    assert "minimum_retention_days_invalid" in weakened["blockers"]

    relabeled_manifest = dict(manifest)
    relabeled_manifest["repository_sha"] = "c" * 40
    relabeled = archive_bundle(
        client=_FakeS3(),
        bundle_path=bundle,
        bundle_manifest=relabeled_manifest,
        archive_uri="s3://release-evidence/pipeline",
        policy=_retention_policy(["ci"]),
        now=datetime(2026, 7, 9, tzinfo=timezone.utc),
    )
    assert "bundle_embedded_manifest_mismatch" in relabeled["blockers"]


def test_pubsub_emulator_host_must_be_loopback() -> None:
    assert _validated_emulator_host("127.0.0.1:8085") == "127.0.0.1:8085"
    try:
        _validated_emulator_host("pubsub.googleapis.com:443")
    except ValueError as exc:
        assert "local emulator only" in str(exc)
    else:  # pragma: no cover - regression guard
        raise AssertionError("live Pub/Sub endpoint was accepted")
    with pytest.raises(ValueError, match="may not include a path"):
        _validated_emulator_host("http://127.0.0.1:8085/not-an-emulator-target")


def test_signature_evidence_requires_fresh_bound_raw_proof(tmp_path: Path) -> None:
    proof = tmp_path / "verification.sigstore.json"
    proof.write_bytes(b"signed-proof")
    proof_digest = f"sha256:{hashlib.sha256(proof.read_bytes()).hexdigest()}"
    envelope = {
        "schema_version": "blueprint.release_evidence.v1",
        "evidence_id": "artifact_signature",
        "evidence_schema_version": "blueprint.release_signature_verification.v1",
        "status": "verified",
        "repository_sha": SHA,
        "image_digest": IMAGE,
        "generated_at": "2026-07-09T00:00:00+00:00",
        "expires_at": "2026-07-10T00:00:00+00:00",
        "source_artifact_digest": proof_digest,
        "evidence_uri": "https://github.com/example/attestations/1",
    }
    (tmp_path / "artifact_signature.json").write_text(json.dumps(envelope))

    passed = validate_signature_evidence(
        evidence_dir=tmp_path,
        repository_sha=SHA,
        image_digest=IMAGE,
        now=datetime(2026, 7, 9, 12, tzinfo=timezone.utc),
    )
    assert passed["status"] == "passed"

    proof.write_bytes(b"tampered")
    blocked = validate_signature_evidence(
        evidence_dir=tmp_path,
        repository_sha=SHA,
        image_digest=IMAGE,
        now=datetime(2026, 7, 9, 12, tzinfo=timezone.utc),
    )
    assert "signature_source_proof_missing_or_digest_mismatch" in blocked["blockers"]

    envelope["evidence_uri"] = f"file://{proof}"
    (tmp_path / "artifact_signature.json").write_text(json.dumps(envelope))
    local_uri = validate_signature_evidence(
        evidence_dir=tmp_path,
        repository_sha=SHA,
        image_digest=IMAGE,
        now=datetime(2026, 7, 9, 12, tzinfo=timezone.utc),
    )
    assert "signature_evidence_uri_not_durable" in local_uri["blockers"]


def test_repository_release_quality_policies_are_machine_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    critical = json.loads((root / "docs" / "critical_capability_lanes.json").read_text())
    assert validate_lanes(root=root, policy=critical) == []

    workflows = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((root / ".github" / "workflows").glob("*.yml"))
    )
    assert "github/codeql-action/init@99df26d4f13ea111d4ec1a7dddef6063f76b97e9" in workflows
    assert "actions/attest@a1948c3f048ba23858d222213b7c278aabede763" in workflows
    assert (root / ".github" / "dependabot.yml").is_file()
    retention = json.loads((root / "docs" / "release_evidence_retention_policy.json").read_text())
    assert {"source", "codeql", "signature"} <= set(
        retention["required_evidence_groups_by_scope"]["BASE"]
    )
    assert "sim_only" in retention["required_evidence_groups_by_scope"]["SC3"]
    requirements = json.loads((root / "docs" / "release_evidence_requirements.json").read_text())
    assert {"artifact_signature", "immutable_retention"} <= set(requirements["scopes"]["BASE"])

    full_lane = (root / ".github" / "workflows" / "full-test-lane.yml").read_text()
    assert "--extra groot-libero-cpu" in full_lane
    assert "--extra groot-libero\n" not in full_lane
    assert "scripts/build_cpu_full_lane_evidence.py" in full_lane
    assert "output/ci/cpu_full.json" in full_lane

    critical_workflow = (
        root / ".github" / "workflows" / "critical-capability-lanes.yml"
    ).read_text()
    assert "inputs.evidence_dir" not in critical_workflow
    assert "unitree_groot_n17_sonic_vast_image_canary" in critical_workflow
    assert "BLUEPRINT_GPU_CANARY_APPROVED_IMAGE_URI" in critical_workflow
    assert "--approved-image-uri" in critical_workflow
    assert "build_gpu_provider_canary_evidence.py convert" in critical_workflow
    assert "pubsub_emulator_round_trip_source.json" in critical_workflow
    assert "native_lerobot_export_source_manifest.json" in critical_workflow

    ci_workflow = (root / ".github" / "workflows" / "ci.yml").read_text()
    assert "build_container_production_evidence.py" in ci_workflow
    assert "production_user.txt" in ci_workflow
    assert "development_user.txt" in ci_workflow


def test_committed_bandit_triage_is_internally_consistent() -> None:
    """The committed triage file must never carry stale-path or malformed entries.

    Exact code fingerprints can only be checked by a real Bandit scan (the CI
    sast job), but path existence, schema shape, uniqueness, dispositions, and
    the owner-bound 90-day expiry window are all checkable hermetically.
    """
    from datetime import timedelta

    from scripts.verify_bandit_policy import (
        ALLOWED_DISPOSITIONS,
        EXPECTED_SCANNER_VERSION,
        SCHEMA_VERSION as TRIAGE_SCHEMA_VERSION,
    )

    root = Path(__file__).resolve().parents[1]
    triage = json.loads((root / "docs" / "bandit_triage.json").read_text())
    assert triage["schema_version"] == TRIAGE_SCHEMA_VERSION
    assert triage["scanner_version"] == EXPECTED_SCANNER_VERSION
    assert triage["policy"]["expired_or_orphaned_entries_block"] is True
    assert triage["policy"]["high_findings_may_not_be_baselined"] is True

    fingerprints: set[str] = set()
    for entry in triage["entries"]:
        fingerprint = entry["fingerprint"]
        assert len(fingerprint) == 64 and all(c in "0123456789abcdef" for c in fingerprint)
        assert fingerprint not in fingerprints, f"duplicate_triage_fingerprint:{fingerprint}"
        fingerprints.add(fingerprint)
        assert (root / entry["filename"]).is_file(), f"triaged_file_missing:{entry['filename']}"
        assert entry["severity"] == "MEDIUM"
        assert entry["disposition"] in ALLOWED_DISPOSITIONS
        assert len(str(entry["owner"]).strip()) >= 3
        assert len(str(entry["reason"]).strip()) >= 20
        reviewed_on = date.fromisoformat(entry["reviewed_on"])
        expires_on = date.fromisoformat(entry["expires_on"])
        assert reviewed_on <= expires_on <= reviewed_on + timedelta(days=90)


def test_outbound_http_call_sites_route_through_the_safe_boundary() -> None:
    """FABLE-007 regression pin: the hardened modules must not regrow ad hoc
    ``urllib.request.urlopen`` sites; the centralized boundary keeps exactly one."""
    root = Path(__file__).resolve().parents[1]
    rewired = (
        "src/blueprint_pipeline/gpu_render_providers.py",
        "src/blueprint_pipeline/isaac_persistent_task_completion_client.py",
        "src/blueprint_pipeline/wam_strict_action_consistency_scorer_client.py",
        "scripts/run_isaac_g1_kitchen_parity_eval.py",
    )
    for relative in rewired:
        text = (root / relative).read_text(encoding="utf-8")
        assert "urllib.request.urlopen(" not in text, f"ad_hoc_urlopen_reintroduced:{relative}"

    boundary = (root / "src" / "blueprint_pipeline" / "safe_outbound_http.py").read_text(
        encoding="utf-8"
    )
    assert boundary.count("urllib.request.build_opener(") == 1
    assert "# nosec" not in boundary
