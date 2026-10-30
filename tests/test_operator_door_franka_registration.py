"""Fixed single-FR3 public registration; development-only artifacts and no compute."""

from copy import deepcopy
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from tests.test_operator_door_public_policy_registration import (
    contract,
    installer,
    request,
    retained_fixture,
    write,
    PROFILE_ROOT,
    validate_request,
    RequestRefused,
    enqueue,
    process_spool,
    DoorConfig,
    CommandResult,
)
from tests.test_deployment_embodiment_fr3_composition import fr3_composed, composed  # noqa: F401
from tests.test_openwam_franka_runtime import development_model
from blueprint_pipeline.openwam_franka_runtime import build_policy_spec, configure_prepared_inputs


def test_fixed_identity_public_license_and_profile_parity(tmp_path):
    req = request(franka=True)
    assert (
        validate_request({"kind": "install-public-franka-policy", "request": req})["request"] == req
    )
    profile = contract.load_profile(req, PROFILE_ROOT)
    _, original = development_model(tmp_path)
    manifest = configure_prepared_inputs({"robot_embodiment": original})["robot_embodiment"]
    assert contract.expected_manifest(original, profile) == manifest
    assert contract.expected_policy(manifest, profile) == build_policy_spec(manifest)
    rights = contract.rights_record(req, profile)
    assert (
        rights["license_id"] == "Apache-2.0"
        and rights["private_vendor_authorization_claimed"] is False
    )
    assert "e75522eb38f9b4148d5ecb985ab80c490ddb3854/README.md" in rights["model_card_url"]
    assert (
        "publisher_SKU_mount_TCP_calibration_unknown"
        in profile["policy_spec_template"]["binding"]["frame_binding_reference"]
    )
    for kind in ("install-public-piper-policy", "install-public-yam-policy"):
        with pytest.raises(RequestRefused):
            validate_request({"kind": kind, "request": req})


def test_fixed_franka_cas_preserves_every_other_candidate_and_idempotent_receipt(tmp_path):
    retained = retained_fixture(tmp_path, franka=True)
    before = deepcopy(retained.registry)
    receipt = installer.apply(retained.request, **retained.options)
    assert receipt["candidate_id"] == "openwam_alpha_franka"
    assert receipt["schema_version"] == "openwam_franka_policy_install_receipt.v1"
    assert (
        receipt["native_policy_execution_validated"] is False
        and receipt["provider_allocations_performed"] == 0
    )
    assert installer.apply(retained.request, **retained.options) == receipt
    after = contract.read(retained.registry_path)
    for settings in [after["derived_profiles"]["defaults"], *after["profiles"]]:
        assert settings["screening_candidates"].pop(contract.FRANKA_CANDIDATE_ID) == {
            "robot_policy": retained.registration
        }
    after["registry_digest"] = before["registry_digest"]
    assert after == before


@pytest.mark.parametrize("fault", ["body", "tool_frame", "cadence", "endpoint", "license", "card"])
def test_franka_installer_reconstructs_exact_profile_instead_of_helper_selected_fields(
    tmp_path, fault
):
    retained = retained_fixture(tmp_path, franka=True)
    row = retained.proposal["registration"]
    if fault == "body":
        row["policy_spec"]["embodiment_id"] = "franka_research_3_bimanual"
    elif fault == "tool_frame":
        row["policy_spec"]["binding"]["native_tcp_from_policy_tcp"]["arm"]["position_m"][0] = 0.1
    elif fault == "cadence":
        row["policy_spec"]["action_hz"] = 30
    elif fault == "endpoint":
        row["endpoint_url"] = "https://untrusted.invalid"
    elif fault == "license":
        row["rights_binding"]["private_vendor_authorization_claimed"] = True
    else:
        evidence = tmp_path / "evidence"
        shutil.copytree(PROFILE_ROOT, evidence)
        (evidence / "openwam_franka_model_card.md").write_text("changed")
        retained.options["profile_root"] = evidence
    write(retained.destination / "proposal.json", retained.proposal)
    with pytest.raises(ValueError):
        installer.apply(retained.request, **retained.options)
    assert contract.read(retained.registry_path) == retained.registry


def test_franka_spool_never_launches_compute_or_passes_provider_credentials(tmp_path):
    config = DoorConfig(state_root=str(tmp_path / "door"))
    for sub in ("pending", "processing", "completed", "results"):
        (Path(config.spool_root) / sub).mkdir(parents=True)
    req = request(franka=True)
    enqueue(
        config,
        {"kind": "install-public-franka-policy", "request": req},
        requested_by="development-only",
        operation_key="development-only",
    )
    calls = []
    process_spool(
        config,
        runner=SimpleNamespace(
            run=lambda argv, timeout: calls.append(argv) or CommandResult(0, "", "")
        ),
    )
    (launch,) = calls
    assert launch[-1] == config.install_root + "/door-install-public-piper-policy.sh"
    assert "--property=PrivateNetwork=yes" in launch
    env = dict(
        v.removeprefix("--setenv=").split("=", 1) for v in launch if v.startswith("--setenv=")
    )
    assert (
        env["DOOR_SERVICE_USER"] == "blueprint"
        and env["DOOR_PUBLIC_POLICY_OPERATION"] == "install-public-franka-policy"
    )
    assert not any("TOKEN" in k or "KEY" in k for k in env)


def test_actual_source_import_binding_reaches_both_registration_stages(
    fr3_composed,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    from blueprint_pipeline import deployment_embodiment_runtime_configuration as runtime
    from blueprint_pipeline import deployment_embodiment_asset_import as importer
    from blueprint_pipeline import openwam_piper_registration as builder
    from tests.test_deployment_embodiment_runtime_configuration import _fake_sdk_from_urdf

    source, _, _, _ = fr3_composed
    models = tmp_path / "models"
    body = models / contract.FRANKA_EMBODIMENT_ID / ("d" * 64)
    body.mkdir(parents=True)
    config = runtime.prepare_runtime_configuration(source, body / "configuration")
    _fake_sdk_from_urdf(monkeypatch)
    importer.import_source_asset(
        body / "configuration",
        body / "prepared",
        fix_base=True,
        station_parameters=config["station_parameters"],
    )
    binding = runtime.bind_prepared_configuration(
        body / "configuration",
        body / "prepared",
        "https://assets.example.invalid/development-only.usda",
    )
    write(body / "binding-inputs.json", binding)
    publication = {
        "status": "completed",
        "claim_ceiling": "robot_asset_import_only",
        "candidate_policy_queried": False,
        "task_executed": False,
        "results": [
            {
                "embodiment_id": contract.FRANKA_EMBODIMENT_ID,
                "status": "retained",
                "directory": str(body / "prepared"),
                "binding_inputs_path": str(body / "binding-inputs.json"),
                "binding_inputs_sha256": contract.file_record(body / "binding-inputs.json")[
                    "sha256"
                ],
            }
        ],
    }
    publication["publication_digest"] = contract.digest(publication)
    write(models / "batches/development-only-assets.json", publication)
    registry = {
        "schema_version": "blueprint.team_candidate_quick10_registry.v1",
        "profiles": [],
        "derived_profiles": {"defaults": {}},
    }
    registry["registry_digest"] = contract.digest(registry)
    write(tmp_path / "etc/registry.json", registry)
    req = request(
        franka=True,
        preparation_publication_digest=publication["publication_digest"],
        expected_registry_digest=registry["registry_digest"],
    )
    state = tmp_path / "state"
    result = builder.prepare(
        req, state_root=state, model_root=models, release_commit=req["release_commit"]
    )
    assert result["status"] == "prepared" and result["provider_allocations_performed"] == 0
    options = {
        "state_root": state,
        "model_root": models,
        "registry_path": tmp_path / "etc/registry.json",
        "profile_root": PROFILE_ROOT,
        "audit_root": tmp_path / "audit",
    }
    receipt = installer.apply(req, **options)
    assert (
        receipt["status"] == "installed" and receipt["native_policy_execution_validated"] is False
    )
    assert installer.apply(req, **options) == receipt
    row = contract.read(tmp_path / "etc/registry.json")["derived_profiles"]["defaults"][
        "screening_candidates"
    ]["openwam_alpha_franka"]["robot_policy"]
    assert row["policy_spec"]["action_hz"] == 15
    assert row["policy_spec"]["binding"]["camera_names"] == ["overhead", "arm_wrist"]
