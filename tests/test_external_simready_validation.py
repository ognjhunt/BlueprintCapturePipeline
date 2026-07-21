from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import jsonschema

from blueprint_pipeline.external_simready_validation import (
    CLAIM_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    main,
    normalize_findings,
    run_external_simready_validation,
)
from blueprint_pipeline.simready_assets import build_simready_assets
from tests.test_simready_assets import (
    _build_capture_root,
    _object_geometry_manifest,
    _site_world_spec,
    _task_anchor_manifest,
)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fake_validator(tmp_path: Path) -> Path:
    path = tmp_path / "isolated-simready-validator"
    path.write_text(
        """#!/bin/sh
output="$1"
input="$2"
validator_version="$3"
profile="$4"
profile_version="$5"
if grep -q 'defaultPrim' "$input"; then
  status="passed"
  finding_status="passed"
  exit_code=0
else
  status="failed"
  finding_status="failed"
  exit_code=1
fi
printf '{"status":"%s","validator_version":"%s","profile_name":"%s","profile_version":"%s","findings":[{"requirement_id":"SAMP.001","status":"%s","prim_path":"/","message":"default prim check","suggestion":"define defaultPrim"}]}\n' "$status" "$validator_version" "$profile" "$profile_version" "$finding_status" > "$output"
echo "fixture validator $status"
exit "$exit_code"
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _build_simready_capture(tmp_path: Path) -> Path:
    capture_root = _build_capture_root(tmp_path)
    build_simready_assets(
        capture_root=capture_root,
        object_geometry_manifest=_object_geometry_manifest(),
        task_anchor_manifest=_task_anchor_manifest(),
        site_world_spec=_site_world_spec(),
    )
    return capture_root


def _command(validator: Path) -> list[str]:
    return [
        str(validator),
        "{output}",
        "{input}",
        "1.2.3",
        "{profile}",
        "{profile_version}",
    ]


def test_external_simready_validation_passes_advisory_and_preserves_input(
    tmp_path: Path,
) -> None:
    capture_root = _build_simready_capture(tmp_path / "pass")
    validator = _fake_validator(tmp_path)
    usd_path = capture_root / "pipeline" / "simready" / "isaac_sim" / "site_scene.usda"
    before = usd_path.read_bytes()

    run = run_external_simready_validation(
        capture_root=capture_root,
        validator_command=_command(validator),
        validator_version="1.2.3",
        validator_source_revision="fixture-revision-abc123",
        license_compatible=True,
    )

    simready_dir = capture_root / "pipeline" / "simready"
    request = _read_json(simready_dir / "external_validation_request.json")
    result = _read_json(simready_dir / "external_validation_result.json")
    boundary = _read_json(simready_dir / "external_validation_claim_boundary.json")

    assert run["status"] == "passed_advisory"
    assert request["schema_version"] == REQUEST_SCHEMA_VERSION
    assert request["transformations"]["enabled"] is False  # type: ignore[index]
    assert request["execution_policy"]["network_policy"] == "disabled"  # type: ignore[index]
    assert result["schema_version"] == RESULT_SCHEMA_VERSION
    assert result["external_validator_ran"] is True
    assert result["version_identity_verified"] is True
    assert result["input_modified"] is False
    assert result["normalized_findings"][0]["rule_id"] == "SAMP.001"  # type: ignore[index]
    assert result["claim_boundary"]["simulator_execution_proven"] is False  # type: ignore[index]
    assert result["claim_boundary"]["rank_fidelity_result_proven"] is False  # type: ignore[index]
    assert boundary["schema_version"] == CLAIM_SCHEMA_VERSION
    assert boundary["advisory_only"] is True
    assert usd_path.read_bytes() == before
    assert Path(result["execution"]["stdout_path"]).is_file()  # type: ignore[index]
    assert result["execution"]["environment_policy"]["secret_values_in_artifact"] is False  # type: ignore[index]


def test_external_simready_validation_normalizes_failed_fixture_stably(tmp_path: Path) -> None:
    capture_root = _build_simready_capture(tmp_path / "fail")
    malformed = capture_root / "pipeline" / "simready" / "malformed.usda"
    malformed.write_text('#usda 1.0\ndef Xform "NotDefault" {}\n', encoding="utf-8")
    validator = _fake_validator(tmp_path)
    kwargs = {
        "capture_root": capture_root,
        "input_usd": malformed,
        "validator_command": _command(validator),
        "validator_version": "1.2.3",
        "validator_source_revision": "fixture-revision-abc123",
        "license_compatible": True,
    }

    first = run_external_simready_validation(**kwargs)
    first_result = _read_json(Path(first["result_path"]))
    second = run_external_simready_validation(**kwargs)
    second_result = _read_json(Path(second["result_path"]))

    assert first["status"] == "validation_failed"
    assert second["status"] == "validation_failed"
    assert first_result["normalized_findings"] == second_result["normalized_findings"]
    assert first_result["raw_report_sha256"] == second_result["raw_report_sha256"]
    assert first_result["failed_finding_count"] == 1
    assert first_result["normalized_findings"][0]["severity"] == "error"  # type: ignore[index]


def test_external_simready_validation_blocks_missing_command_transformations_and_core_venv(
    tmp_path: Path,
) -> None:
    capture_root = _build_simready_capture(tmp_path)
    missing = run_external_simready_validation(
        capture_root=capture_root,
        validator_version="1.2.3",
        validator_source_revision="fixture-revision",
    )
    missing_result = _read_json(Path(missing["result_path"]))
    assert missing["status"] == "blocked"
    assert "external_validator_command_not_configured" in missing_result["blockers"]
    assert missing_result["external_validator_ran"] is False

    transformed = run_external_simready_validation(
        capture_root=capture_root,
        validator_command=["/bin/false"],
        validator_version="1.2.3",
        validator_source_revision="fixture-revision",
        transformations=["stamp_asset_validation"],
    )
    transformed_result = _read_json(Path(transformed["result_path"]))
    assert (
        "transformations_requested_without_explicit_authorization" in transformed_result["blockers"]
    )
    assert transformed_result["external_validator_ran"] is False

    core = run_external_simready_validation(
        capture_root=capture_root,
        validator_command=[str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python")],
        validator_version="1.2.3",
        validator_source_revision="fixture-revision",
    )
    core_result = _read_json(Path(core["result_path"]))
    assert "validator_must_not_use_core_environment" in core_result["blockers"]
    assert core_result["external_validator_ran"] is False


def test_simready_builder_runs_explicit_external_validator_and_cli(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path / "builder")
    validator = _fake_validator(tmp_path)
    result = build_simready_assets(
        capture_root=capture_root,
        object_geometry_manifest=_object_geometry_manifest(),
        task_anchor_manifest=_task_anchor_manifest(),
        site_world_spec=_site_world_spec(),
        external_validator_command=_command(validator),
        external_validator_version="1.2.3",
        external_validator_source_revision="fixture-revision",
        external_validator_license_compatible=True,
    )
    manifest = _read_json(capture_root / "pipeline" / "simready" / "simready_scene_manifest.json")
    assert result["external_validation"]["status"] == "passed_advisory"  # type: ignore[index]
    assert manifest["external_validation"]["advisory_only"] is True  # type: ignore[index]

    assert (
        main(
            [
                "--capture-root",
                str(capture_root),
                "--validator-command",
                " ".join(_command(validator)),
                "--validator-version",
                "1.2.3",
                "--validator-source-revision",
                "fixture-revision",
                "--license-compatible",
            ]
        )
        == 0
    )


def test_external_simready_contract_schemas_validate_emitted_artifacts(tmp_path: Path) -> None:
    capture_root = _build_simready_capture(tmp_path)
    validator = _fake_validator(tmp_path)
    run_external_simready_validation(
        capture_root=capture_root,
        validator_command=_command(validator),
        validator_version="1.2.3",
        validator_source_revision="fixture-revision",
        license_compatible=True,
    )
    repo_root = Path(__file__).resolve().parents[1]
    simready_dir = capture_root / "pipeline" / "simready"
    pairs = (
        ("external_validation_request.json", "external_simready_validation_request.schema.json"),
        ("external_validation_result.json", "external_simready_validation_result.schema.json"),
        (
            "external_validation_claim_boundary.json",
            "external_simready_validation_claim_boundary.schema.json",
        ),
    )
    for artifact_name, schema_name in pairs:
        jsonschema.Draft202012Validator(
            _read_json(repo_root / "docs" / "schemas" / schema_name)
        ).validate(_read_json(simready_dir / artifact_name))


def test_finding_normalizer_handles_nested_unknown_and_duplicate_rows() -> None:
    finding = {
        "rule_id": "UN.001",
        "severity": "warning",
        "path": "/World",
        "message": "units review",
    }
    normalized = normalize_findings({"profiles": [{"findings": [finding, finding]}], "x": 1})
    assert normalized == [
        {
            "rule_id": "UN.001",
            "severity": "warning",
            "status": "warning",
            "object_path": "/World",
            "message": "units review",
            "suggested_action": None,
        }
    ]


def test_external_simready_cli_blocks_without_command(tmp_path: Path) -> None:
    capture_root = _build_simready_capture(tmp_path)
    assert (
        main(
            [
                "--capture-root",
                str(capture_root),
                "--validator-version",
                "1.2.3",
                "--validator-source-revision",
                "fixture-revision",
            ]
        )
        == 2
    )
    assert os.environ.get("SIMREADY_VALIDATOR_SECRET") is None
