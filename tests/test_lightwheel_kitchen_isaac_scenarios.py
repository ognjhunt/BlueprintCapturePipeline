from __future__ import annotations

import json
import os
import py_compile
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest
from tests.runpy_entrypoint import run_module_as_main


pytestmark = [pytest.mark.slow, pytest.mark.integration]
pytest.importorskip("PIL")
from PIL import Image

import blueprint_pipeline.lightwheel_kitchen_isaac_scenarios as lks
from blueprint_pipeline.lightwheel_kitchen_isaac_scenarios import (
    CONTACT_COLLISION_MANIFEST_NAME,
    FINAL_READINESS_MANIFEST_NAME,
    ISAAC_EXECUTION_MANIFEST_NAME,
    MAIN_USD_RELATIVE,
    PER_SCENARIO_RESULTS_NAME,
    PROVIDER_BUNDLE_NAME,
    PROVIDER_PACKET_NAME,
    PROVIDER_PRIVATE_ENV_NAME,
    PROVIDER_REQUEST_NAME,
    RUNPOD_CONTAINER_REGISTRY_AUTH_ID_ENV,
    RUNPOD_DIRECT_LAUNCH_REQUEST_LOCAL_NAME,
    RUNPOD_DIRECT_LAUNCH_REQUEST_NAME,
    THUMBNAIL_RELATIVE,
    VIDEO_ARTIFACT_CHECKS_NAME,
    _asset_inventory_from_zip,
    _default_scenarios,
    _draw_previews,
    _provider_runtime_probe_shell_script,
    _redact_signed_url_text,
    _runtime_preflight,
    _unitree_g1_mjcf_summary,
    _usd_dependency_presence_audit,
    _write_blocked_execution_outputs,
    _write_provider_bundle,
    _write_provider_packet,
    _write_provider_request,
    _write_runpod_direct_launch_request,
    _write_isaac_runner_script,
)


def _make_provider_fixture(tmp_path: Path) -> tuple[Path, dict, dict, Path]:
    source_zip = tmp_path / "Lightwheel_Kitchen.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr(MAIN_USD_RELATIVE, "PXR-USDC")
    runner_script = tmp_path / "run_lightwheel_kitchen_isaac_scenarios.py"
    runner_script.write_text("print('runner')\n", encoding="utf-8")
    local_request = tmp_path / "isaac_execution_request.json"
    local_request.write_text("{}", encoding="utf-8")
    provider_request = tmp_path / PROVIDER_REQUEST_NAME
    scenario_manifest = tmp_path / "lightwheel_kitchen_scenarios.json"
    runtime_preflight = tmp_path / "lightwheel_kitchen_isaac_runtime_preflight.json"
    handoff = tmp_path / "lightwheel_kitchen_isaac_handoff_manifest.json"
    scenario_manifest.write_text("{}", encoding="utf-8")
    runtime_preflight.write_text("{}", encoding="utf-8")
    handoff.write_text("{}", encoding="utf-8")
    unitree_root = tmp_path / "unitree_g1"
    unitree_root.mkdir()
    (unitree_root / "g1.xml").write_text("<mujoco/>\n", encoding="utf-8")
    (unitree_root / "mesh.stl").write_text("solid mesh\nendsolid mesh\n", encoding="utf-8")
    (unitree_root / "LICENSE").write_text("license\n", encoding="utf-8")
    unitree_summary = _unitree_g1_mjcf_summary(unitree_root)
    _write_provider_request(
        path=provider_request,
        scenarios=_default_scenarios(),
        unitree_g1_mjcf=unitree_summary,
    )
    bundle = _write_provider_bundle(
        output_dir=tmp_path,
        source_zip=source_zip,
        unitree_g1_mjcf_root=unitree_root,
        runner_script=runner_script,
        local_execution_request_path=local_request,
        provider_execution_request_path=provider_request,
        scenario_manifest_path=scenario_manifest,
        runtime_preflight_path=runtime_preflight,
        handoff_path=handoff,
    )
    return source_zip, bundle, unitree_summary, provider_request


def test_lightwheel_inventory_routes_usd_only_scene_to_isaac(tmp_path: Path) -> None:
    source_zip = tmp_path / "Lightwheel_Kitchen.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr(MAIN_USD_RELATIVE, "PXR-USDC")
        archive.writestr("Collected_KitchenRoom/texture/Kitchen_Ground_BC.jpg", "not-real-jpg")

    inventory = _asset_inventory_from_zip(source_zip)

    assert inventory["main_usd_present"] is True
    assert inventory["extension_counts"][".usd"] == 1
    assert inventory["mujoco_native_asset_present"] is False


def test_lightwheel_scenarios_are_specs_not_navigation_proof() -> None:
    scenarios = _default_scenarios()

    assert len(scenarios) == 5
    assert all(scenario["spawn_position_xyz"] for scenario in scenarios)
    assert all(scenario["target_position_xyz"] for scenario in scenarios)
    assert all(scenario["execution_proven"] is False for scenario in scenarios)
    assert all("navigation_policy_boundary" in scenario for scenario in scenarios)


def test_generated_isaac_runner_compiles_and_prefers_official_g1_usd(tmp_path: Path) -> None:
    runner = tmp_path / "run_lightwheel_kitchen_isaac_scenarios.py"
    _write_isaac_runner_script(runner)

    py_compile.compile(str(runner), doraise=True)
    source = runner.read_text(encoding="utf-8")
    assert "Isaac/Robots/Unitree/G1/g1.usd" in source
    assert "official_isaac_unitree_g1_usd" in source


def test_generated_isaac_runner_resolves_g1_before_any_stage_update(tmp_path: Path) -> None:
    runner = tmp_path / "run_lightwheel_kitchen_isaac_scenarios.py"
    _write_isaac_runner_script(runner)
    source = runner.read_text(encoding="utf-8")

    required_phases = [
        "runner_referencing_official_g1",
        "runner_official_g1_resolved",
        "runner_official_g1_reference_added",
        "runner_robot_api_evidence_collection_starting",
        "runner_robot_api_evidence_collected",
        "runner_unitree_g1_binding_starting",
        "runner_lightwheel_stage_update_starting",
        "runner_scenario_attempts_starting",
    ]
    for phase in required_phases:
        assert phase in source
    assert source.index("runner_referencing_official_g1") < source.index(
        "runner_official_g1_resolved"
    )
    assert source.index("runner_official_g1_resolved") < source.index(
        "runner_official_g1_reference_added"
    )
    assert source.index("runner_official_g1_reference_added") < source.index(
        "runner_robot_api_evidence_collection_starting"
    )
    assert source.index("runner_robot_api_evidence_collection_starting") < source.index(
        "runner_robot_api_evidence_collected"
    )
    assert source.index("runner_unitree_g1_binding_starting") < source.index(
        "runner_robot_api_evidence_collected"
    )
    assert source.index("runner_robot_api_evidence_collected") < source.index(
        "runner_lightwheel_stage_update_starting"
    )
    assert source.index("runner_lightwheel_stage_update_starting") < source.index(
        "runner_scenario_attempts_starting"
    )
    assert 'BLUEPRINT_ISAAC_STAGE_UPDATE_TICKS", "0"' in source
    assert source.count("simulation_app.update()") == 1


def test_generated_isaac_runner_emits_per_scenario_verification_contract(
    tmp_path: Path,
) -> None:
    runner = tmp_path / "run_lightwheel_kitchen_isaac_scenarios.py"
    _write_isaac_runner_script(runner)
    source = runner.read_text(encoding="utf-8")

    assert "_run_scenario_attempts(" in source
    assert '"scenarios_attempted": len(scenario_results)' in source
    assert '"scenarios_verified": scenarios_verified' in source
    assert '"per_scenario_results": scenario_results' in source
    assert "all_five_lightwheel_scenarios_not_verified" in source
    assert "unitree_g1_control_command_not_applied_for_scenario" in source
    assert "runner_finished_without_terminal_result" in source
    assert "runner_robot_api_evidence_failed" in source
    assert "runner_unitree_g1_binding_failed" in source
    assert "runner_lightwheel_stage_update_failed" in source
    assert "runner_scenario_attempt_generation_failed" in source
    assert "runner_phase_before_terminalizer" in source
    assert "runner_terminal_phase" in source
    assert "Usd.PrimRange(prim)" in source
    assert "GetDescendants()" not in source
    assert "blocked_before_scenario_attempt" in source


def test_generated_isaac_runner_reaches_scenario_attempts_with_usd_prim_range(
    tmp_path: Path,
) -> None:
    runner = tmp_path / "run_lightwheel_kitchen_isaac_scenarios.py"
    _write_isaac_runner_script(runner)

    fake_root = tmp_path / "fake_isaac"
    (fake_root / "isaacsim" / "storage").mkdir(parents=True)
    (fake_root / "omni").mkdir()
    (fake_root / "pxr").mkdir()
    (fake_root / "isaacsim" / "__init__.py").write_text(
        """class SimulationApp:
    def __init__(self, config):
        self.config = config

    def update(self):
        pass

    def close(self):
        pass
""",
        encoding="utf-8",
    )
    (fake_root / "isaacsim" / "storage" / "__init__.py").write_text("", encoding="utf-8")
    (fake_root / "isaacsim" / "storage" / "native.py").write_text(
        """def get_assets_root_path():
    return "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
""",
        encoding="utf-8",
    )
    (fake_root / "omni" / "__init__.py").write_text("", encoding="utf-8")
    (fake_root / "omni" / "usd.py").write_text(
        """class Prim:
    def GetReferences(self):
        return self

    def AddReference(self, uri):
        self.uri = uri

    def IsValid(self):
        return True

    def HasAPI(self, api):
        return False


class ApiPrim:
    def HasAPI(self, api):
        return True


class Stage:
    def DefinePrim(self, path, type_name):
        return Prim()

    def GetPrimAtPath(self, path):
        return Prim()


class Context:
    def open_stage(self, path):
        self.path = path

    def get_stage(self):
        return Stage()


def get_context():
    return Context()
""",
        encoding="utf-8",
    )
    (fake_root / "pxr" / "__init__.py").write_text("", encoding="utf-8")
    (fake_root / "pxr" / "Usd.py").write_text(
        """from omni.usd import ApiPrim


def PrimRange(prim):
    return [prim, ApiPrim()]
""",
        encoding="utf-8",
    )
    (fake_root / "pxr" / "UsdPhysics.py").write_text(
        """class ArticulationRootAPI:
    pass


class CollisionAPI:
    pass


class RigidBodyAPI:
    pass
""",
        encoding="utf-8",
    )
    scene = tmp_path / "KitchenRoom.usd"
    scene.write_text("#usda 1.0\n", encoding="utf-8")
    request = tmp_path / "isaac_execution_request.json"
    request.write_text(
        json.dumps({"scene_usd_path": str(scene), "scenarios": _default_scenarios()}),
        encoding="utf-8",
    )
    output = tmp_path / "result.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_root)
    completed = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--request",
            str(request),
            "--output",
            str(output),
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2, completed.stdout + completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "blocked"
    assert result["provider_runtime_phase"] == "runner_scenario_attempts_completed"
    assert result["runner_terminal_phase"] == "runner_scenario_attempts_completed"
    assert "unitree_g1_control_command_not_applied_for_scenario" in result["blockers"]
    assert "isaac_runner_in_progress" not in result["blockers"]
    assert result["scenario_count"] == 5
    assert result["scenarios_attempted"] == 5
    assert result["scenarios_verified"] == 0
    assert result["unitree_g1_binding"]["spawned_or_resolved"] is True
    assert result["unitree_g1_binding"]["collision_enabled_verified"] is True
    assert result["unitree_g1_binding"]["controllable_articulation_detected"] is True
    assert all(
        item["status"] == "blocked_after_runtime_preflight"
        and item["execution_attempted"] is True
        for item in result["per_scenario_results"]
    )


def test_runtime_preflight_does_not_record_secret_values(monkeypatch) -> None:
    fake_key = "r" + "pa_should_not_be_recorded"
    monkeypatch.setenv("RUNPOD_API_KEY", fake_key)
    preflight = _runtime_preflight()

    provider = preflight["provider_credentials"]
    assert provider["runpod_api_key_env_present"] is True
    assert provider["raw_secret_values_recorded"] is False
    assert "ngc_api_key_file_env_present" in provider
    assert "nvidia_api_key_file_env_present" in provider
    assert fake_key not in str(preflight)


def test_draw_previews_marks_preview_as_not_simulation(tmp_path: Path) -> None:
    thumb = tmp_path / THUMBNAIL_RELATIVE
    thumb.parent.mkdir(parents=True)
    Image.new("RGBA", (256, 256), (240, 230, 210, 255)).save(thumb)

    previews = _draw_previews(
        thumbnail_path=thumb,
        scenarios=_default_scenarios()[:1],
        output_dir=tmp_path / "previews",
    )

    assert len(previews) == 1
    assert Path(previews[0]["path"]).is_file()
    assert previews[0]["simulator_rendered"] is False
    assert previews[0]["purpose"] == "route_storyboard_preview_not_navigation_proof"


def test_signed_url_redaction_is_case_insensitive() -> None:
    signed_url_signature_param = "X-Goog-" + "Signature="
    signed = (
        "https://storage.example/bundle.zip?"
        f"X-Goog-Algorithm=GOOG4-RSA-SHA256&{signed_url_signature_param}abc123"
    )

    redacted = _redact_signed_url_text(signed)

    assert ("x-goog-" + "signature=").lower() not in redacted.lower()
    assert "x-goog-redacted-signature-param=<redacted:signed-url-signature>" in redacted


def _here_doc_body(script: str, marker: str) -> str:
    start = script.index(f"<<'{marker}'")
    body_start = script.index("\n", start) + 1
    end = script.index(f"\n{marker}", body_start)
    return script[body_start:end]


def test_provider_runtime_probe_shell_script_has_compilable_python_heartbeats() -> None:
    script = _provider_runtime_probe_shell_script()

    assert "provider_runtime_probe_started" in script
    assert "provider_bundle_downloaded" in script
    assert "provider_isaac_command_starting" in script
    assert "BLUEPRINT_LIGHTWHEEL_ISAAC_TIMEOUT_SECONDS" in script
    assert "subprocess.TimeoutExpired" in script
    assert "provider_isaac_command_timeout" in script
    assert "provider_runtime_finished_without_terminal_result" in script
    assert "provider_isaac_command_exit_code" in script
    assert "runner_phase_before_provider_finalizer" in script
    assert '"runner/run_lightwheel_kitchen_isaac_scenarios.py"' in script
    assert "provider_runtime_final_upload" in script
    assert 'exit "$rc"' not in script
    assert "exit 20" not in script
    assert "exit 21" not in script
    assert "exit 22" not in script
    compile(_here_doc_body(script, "PYUTIL"), "<provider-probe-PYUTIL>", "exec")
    compile(_here_doc_body(script, "PYDL"), "<provider-probe-PYDL>", "exec")
    compile(_here_doc_body(script, "PYSUPERVISE"), "<provider-probe-PYSUPERVISE>", "exec")
    compile(_here_doc_body(script, "PYFINAL"), "<provider-probe-PYFINAL>", "exec")


def test_runpod_direct_launch_request_redacts_public_signed_urls(
    tmp_path: Path, monkeypatch
) -> None:
    signed_url_signature_param = "X-Goog-" + "Signature="
    private_file = tmp_path / "signed_urls.local"
    private_file.write_text(
        json.dumps(
            {
                "bundle_get": [
                    {
                        "signed_url": (
                            "https://storage.example/bundle.zip?"
                            f"{signed_url_signature_param}bundle-secret"
                        ),
                        "http_verb": "GET",
                        "resource": "gs://bucket/bundle.zip",
                    }
                ],
                "runtime_result_put": [
                    {
                        "signed_url": (
                            "https://storage.example/result.json?"
                            f"{signed_url_signature_param}put-secret"
                        ),
                        "http_verb": "PUT",
                        "resource": "gs://bucket/result.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv(RUNPOD_CONTAINER_REGISTRY_AUTH_ID_ENV, "registry-auth-123")
    summary = _write_runpod_direct_launch_request(
        output_dir=tmp_path,
        provider_packet={
            "launch_contract": {
                "selected_isaac_runtime_image_ref": "nvcr.io/nvidia/isaac-sim:5.1.0"
            }
        },
        provider_private_signed_url_file=private_file,
    )
    public_path = tmp_path / RUNPOD_DIRECT_LAUNCH_REQUEST_NAME
    private_path = tmp_path / RUNPOD_DIRECT_LAUNCH_REQUEST_LOCAL_NAME
    public_text = public_path.read_text(encoding="utf-8")
    private_request = json.loads(private_path.read_text(encoding="utf-8"))
    provider_shape = private_request["provider_request_shape"]
    command = provider_shape["command"]

    assert summary["status"] == "request_manifest_ready"
    assert summary["private_request_written"] is True
    assert public_path.is_file()
    assert private_path.is_file()
    assert ("x-goog-" + "signature=").lower() not in public_text.lower()
    assert "x-goog-redacted-signature-param=<redacted:signed-url-signature>" in public_text
    assert "registry-auth-123" not in public_text
    assert private_request["provider_request_shape"]["inputs"]["manifest_uri"].startswith(
        "https://storage.example/bundle.zip?"
    )
    assert provider_shape["image"]["configured_image_ref"] == "nvcr.io/nvidia/isaac-sim:5.1.0"
    assert provider_shape["image"]["container_registry_auth_id"] == "registry-auth-123"
    public_request = json.loads(public_text)
    assert public_request["provider_request_shape"]["image"]["container_registry_auth_id"] == (
        "<redacted:runpod-container-registry-auth-id>"
    )
    assert provider_shape["docker_entrypoint"] == ["bash"]
    assert provider_shape["docker_start_cmd"][0] == "-lc"
    assert "provider_runtime_probe_started" in provider_shape["docker_start_cmd"][1]
    assert "provider_runtime_probe_started" in command


def test_provider_packet_records_bundle_and_external_blockers(tmp_path: Path) -> None:
    source_zip = tmp_path / "Lightwheel_Kitchen.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr(MAIN_USD_RELATIVE, "PXR-USDC")
    runner_script = tmp_path / "run_lightwheel_kitchen_isaac_scenarios.py"
    runner_script.write_text("print('runner')\n", encoding="utf-8")
    local_request = tmp_path / "isaac_execution_request.json"
    local_request.write_text("{}", encoding="utf-8")
    provider_request = tmp_path / PROVIDER_REQUEST_NAME
    scenario_manifest = tmp_path / "lightwheel_kitchen_scenarios.json"
    runtime_preflight = tmp_path / "lightwheel_kitchen_isaac_runtime_preflight.json"
    handoff = tmp_path / "lightwheel_kitchen_isaac_handoff_manifest.json"
    scenario_manifest.write_text("{}", encoding="utf-8")
    runtime_preflight.write_text("{}", encoding="utf-8")
    handoff.write_text("{}", encoding="utf-8")
    unitree_root = tmp_path / "unitree_g1"
    unitree_root.mkdir()
    (unitree_root / "g1.xml").write_text("<mujoco/>\n", encoding="utf-8")
    (unitree_root / "LICENSE").write_text("license\n", encoding="utf-8")
    unitree_summary = _unitree_g1_mjcf_summary(unitree_root)
    _write_provider_request(
        path=provider_request,
        scenarios=_default_scenarios(),
        unitree_g1_mjcf=unitree_summary,
    )

    bundle = _write_provider_bundle(
        output_dir=tmp_path,
        source_zip=source_zip,
        unitree_g1_mjcf_root=unitree_root,
        runner_script=runner_script,
        local_execution_request_path=local_request,
        provider_execution_request_path=provider_request,
        scenario_manifest_path=scenario_manifest,
        runtime_preflight_path=runtime_preflight,
        handoff_path=handoff,
    )
    packet = _write_provider_packet(
        output_dir=tmp_path,
        source_zip=source_zip,
        bundle=bundle,
        unitree_g1_mjcf=unitree_summary,
        provider_execution_request_path=provider_request,
        runtime={
            "blockers": ["isaac_sim_runtime_unavailable"],
            "provider_credentials": {
                "ngc_api_key_file_env_present": False,
                "nvidia_api_key_file_env_present": False,
            },
        },
        provider_proof_path=None,
        provider_artifact_root_uri="gs://blueprint-artifacts/lightwheel",
        upload_provider_packet=False,
        provider_private_signed_url_file=None,
        repo_commit="abc123",
        selected_isaac_runtime_image_ref="nvcr.io/nvidia/isaac-sim:5.0.0",
    )
    persisted = (tmp_path / PROVIDER_PACKET_NAME).read_text(encoding="utf-8")

    assert Path(bundle["path"]).is_file()
    assert bundle["name"] == PROVIDER_BUNDLE_NAME
    assert bundle["contains_lightwheel_source_zip"] is True
    assert bundle["contains_provider_execution_request"] is True
    assert bundle["contains_unitree_g1_mjcf"] is True
    assert packet["status"] == "provider_packet_prepared_external_inputs_blocked"
    assert packet["asset_bundle"]["provider_bundle_uri"].endswith(PROVIDER_BUNDLE_NAME)
    assert packet["asset_bundle"]["provider_fetchability"]["provider_fetchable_by_runpod"] is False
    assert "versioned_provider_fetchable_BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF" not in packet[
        "required_missing_inputs"
    ]
    assert "official_isaac_unitree_g1_usd_runtime_probe_not_run" in packet[
        "execution_blockers"
    ]
    assert packet["unitree_g1_binding"]["mjcf_asset_bundled"] is True
    assert packet["unitree_g1_binding"]["official_isaac_usd_candidate"][
        "assets_root_relative_path"
    ] == "Isaac/Robots/Unitree/G1/g1.usd"
    assert packet["launch_contract"]["selected_isaac_runtime_image_ref"] == (
        "nvcr.io/nvidia/isaac-sim:5.0.0"
    )
    assert packet["launch_contract"]["selected_isaac_runtime_image_ref_source"] == "argument"
    assert ("r" + "pa_") not in persisted
    assert ("x-goog-" + "signature=") not in persisted.lower()


def test_provider_packet_uses_private_signed_url_file_without_persisting_raw_signature(
    tmp_path: Path,
) -> None:
    source_zip = tmp_path / "Lightwheel_Kitchen.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr(MAIN_USD_RELATIVE, "PXR-USDC")
    runner_script = tmp_path / "run_lightwheel_kitchen_isaac_scenarios.py"
    runner_script.write_text("print('runner')\n", encoding="utf-8")
    local_request = tmp_path / "isaac_execution_request.json"
    local_request.write_text("{}", encoding="utf-8")
    provider_request = tmp_path / PROVIDER_REQUEST_NAME
    scenario_manifest = tmp_path / "lightwheel_kitchen_scenarios.json"
    runtime_preflight = tmp_path / "lightwheel_kitchen_isaac_runtime_preflight.json"
    handoff = tmp_path / "lightwheel_kitchen_isaac_handoff_manifest.json"
    scenario_manifest.write_text("{}", encoding="utf-8")
    runtime_preflight.write_text("{}", encoding="utf-8")
    handoff.write_text("{}", encoding="utf-8")
    unitree_root = tmp_path / "unitree_g1"
    unitree_root.mkdir()
    (unitree_root / "g1.xml").write_text("<mujoco/>\n", encoding="utf-8")
    unitree_summary = _unitree_g1_mjcf_summary(unitree_root)
    _write_provider_request(
        path=provider_request,
        scenarios=_default_scenarios(),
        unitree_g1_mjcf=unitree_summary,
    )
    bundle = _write_provider_bundle(
        output_dir=tmp_path,
        source_zip=source_zip,
        unitree_g1_mjcf_root=unitree_root,
        runner_script=runner_script,
        local_execution_request_path=local_request,
        provider_execution_request_path=provider_request,
        scenario_manifest_path=scenario_manifest,
        runtime_preflight_path=runtime_preflight,
        handoff_path=handoff,
    )
    signed_url_signature_param = "X-Goog-" + "Signature="
    private_file = tmp_path / "signed_urls.local"
    private_file.write_text(
        json.dumps(
            {
                "bundle_get": [
                    {
                        "signed_url": (
                            "https://storage.example/bundle.zip?"
                            f"{signed_url_signature_param}bundle-secret"
                        ),
                        "http_verb": "GET",
                        "resource": "gs://bucket/bundle.zip",
                    }
                ],
                "runtime_result_put": [
                    {
                        "signed_url": (
                            "https://storage.example/result.json?"
                            f"{signed_url_signature_param}put-secret"
                        ),
                        "http_verb": "PUT",
                        "resource": "gs://bucket/result.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    packet = _write_provider_packet(
        output_dir=tmp_path,
        source_zip=source_zip,
        bundle=bundle,
        unitree_g1_mjcf=unitree_summary,
        provider_execution_request_path=provider_request,
        runtime={"blockers": [], "provider_credentials": {}},
        provider_proof_path=None,
        provider_artifact_root_uri="gs://blueprint-artifacts/lightwheel",
        upload_provider_packet=False,
        provider_private_signed_url_file=private_file,
        repo_commit="abc123",
    )
    persisted = (tmp_path / PROVIDER_PACKET_NAME).read_text(encoding="utf-8")

    assert packet["asset_bundle"]["provider_fetchability"]["provider_fetchable_by_runpod"] is True
    assert packet["launch_contract"]["runtime_manifest_signed_put_private_file_present"] is True
    assert "provider_fetchable_lightwheel_asset_bundle_signed_get_url_or_runtime_gcs_credentials" not in packet[
        "required_missing_inputs"
    ]
    assert "provider_writable_artifact_output_uri_or_signed_runtime_manifest_put_url" not in packet[
        "required_missing_inputs"
    ]
    assert (tmp_path / PROVIDER_PRIVATE_ENV_NAME).is_file()
    assert "x-goog-redacted-signature-param=<redacted:signed-url-signature>" in persisted
    assert ("x-goog-" + "signature=") not in persisted.lower()


def test_usd_dependency_presence_audit_records_missing_lightwheel_dependencies(
    tmp_path: Path,
) -> None:
    source_zip = tmp_path / "Lightwheel_Kitchen.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr(MAIN_USD_RELATIVE, "PXR-USDC")
    usdchecker = {
        "status": "failed",
        "stderr_tail": (
            "Could not load sublayer @./UnitsAdjust-deadbeef.metricsAssembler@ "
            "and failed to resolve @OmniPBR.mdl@ plus "
            "@./textures/3d66Model-19913701-files-024.JPG@"
        ),
    }

    audit = _usd_dependency_presence_audit(
        asset_root=tmp_path / "assets",
        source_zip=source_zip,
        usdchecker=usdchecker,
    )

    assert audit["status"] == "missing_required_dependencies"
    assert "UnitsAdjust-deadbeef.metricsAssembler" in audit["missing_dependency_names"]
    assert "OmniPBR.mdl" in audit["missing_dependency_names"]
    assert "3d66Model-19913701-files-024.JPG" in audit["missing_dependency_names"]
    assert "usd_missing_sublayer_metrics_assembler" in audit["blockers"]
    assert "usd_requires_omniverse_mdl_material_registry" in audit["blockers"]


def test_blocked_execution_outputs_are_fail_closed_and_scan_clean(tmp_path: Path) -> None:
    provider_proof = tmp_path / "runpod_active_pod_count_probe.live.json"
    provider_proof.write_text(
        """{
  "status": "runpod_live_proof_collected",
  "api_call_performed": true,
  "runpod_side_effects_may_have_occurred": false,
  "api_key_source": "RUNPOD_CONFIG_FILE",
  "active_pod_count_before": 0,
  "active_pod_count_after": 0,
  "secret_values_in_artifact": false,
  "raw_api_key_stored": false,
  "blockers": []
}
""",
        encoding="utf-8",
    )

    outputs = _write_blocked_execution_outputs(
        output_dir=tmp_path,
        scenarios=_default_scenarios(),
        scene_usd=tmp_path / "KitchenRoom.usd",
        runtime_preflight_path=tmp_path / "runtime_preflight.json",
        handoff_path=tmp_path / "handoff.json",
        execution_request_path=tmp_path / "isaac_execution_request.json",
        blockers=["isaac_sim_runtime_unavailable"],
        runtime={
            "isaac_local_runtime_ready": False,
            "blockers": ["isaac_sim_runtime_unavailable"],
            "provider_credentials": {
                "runpod_config_file_present": True,
                "ngc_api_key_file_env_present": False,
                "nvidia_api_key_file_env_present": False,
            },
        },
        usdchecker={"status": "failed", "blockers": ["usd_requires_omniverse_mdl_material_registry"]},
        dependency_audit={
            "status": "missing_required_dependencies",
            "blockers": ["usd_requires_omniverse_mdl_material_registry"],
            "missing_dependency_names": ["OmniPBR.mdl"],
        },
        stage_summary={"status": "opened"},
        provider_proof_path=provider_proof,
        provider_packet_path=None,
    )

    for name in [
        ISAAC_EXECUTION_MANIFEST_NAME,
        PER_SCENARIO_RESULTS_NAME,
        CONTACT_COLLISION_MANIFEST_NAME,
        VIDEO_ARTIFACT_CHECKS_NAME,
        FINAL_READINESS_MANIFEST_NAME,
    ]:
        assert (tmp_path / name).is_file()

    execution = json.loads((tmp_path / ISAAC_EXECUTION_MANIFEST_NAME).read_text(encoding="utf-8"))
    results = json.loads((tmp_path / PER_SCENARIO_RESULTS_NAME).read_text(encoding="utf-8"))
    videos = json.loads((tmp_path / VIDEO_ARTIFACT_CHECKS_NAME).read_text(encoding="utf-8"))
    readiness = json.loads((tmp_path / FINAL_READINESS_MANIFEST_NAME).read_text(encoding="utf-8"))
    persisted = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in outputs.values()
    )

    assert execution["status"] == "blocked"
    assert execution["provider_probe"]["active_pod_count_after"] == 0
    assert execution["provider_launch_contract"]["status"] == "blocked"
    assert execution["proof_boundary"]["unitree_g1_navigation_proven"] is False
    assert results["scenarios_blocked"] == 5
    assert all(item["status"] == "blocked_not_executed" for item in results["results"])
    assert videos["checks_performed"] is False
    assert readiness["readiness"]["ready_for_customer_delivery"] is False
    assert ("r" + "pa_") not in persisted
    assert ("x-goog-" + "signature=") not in persisted


def test_lightwheel_private_helpers_cover_fail_closed_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not-json", encoding="utf-8")
    assert lks._read_json_mapping(bad_json) == {}
    assert lks._read_json_mapping(tmp_path / "missing.json") == {}

    redacted = lks._redacted_jsonable(
        ("https://storage.example/object?X-Goog-Signature=secret", {"nested": ["ok"]})
    )
    assert redacted[0].endswith("<redacted:signed-url-signature>")

    assert lks._provider_uri_is_fetchable_without_extra_credentials("") is False
    assert lks._provider_uri_is_fetchable_without_extra_credentials(
        "https://storage.example/bundle.zip?X-Goog-Signature=abc"
    )
    assert lks._provider_uri_is_fetchable_without_extra_credentials(
        "https://s3.example/bundle.zip?X-Amz-Signature=abc"
    )
    assert lks._provider_uri_requires_storage_credentials("s3://bucket/object")
    assert not lks._provider_uri_requires_storage_credentials("file:///tmp/object")

    assert lks._classify_upload_error(
        scheme="gs",
        error=RuntimeError("The billing account for the owning project is disabled"),
    ) == "upload_failed:gs_billing_account_disabled"
    assert lks._classify_upload_error(
        scheme="s3",
        error=RuntimeError("AccessDenied"),
    ) == "upload_failed:s3_access_denied"
    assert lks._classify_upload_error(
        scheme="r2",
        error=RuntimeError("403 Forbidden"),
    ) == "upload_failed:r2_forbidden"
    assert lks._classify_upload_error(
        scheme="file",
        error=ValueError("nope"),
    ) == "upload_failed:ValueError"

    source = tmp_path / "source.bin"
    source.write_bytes(b"bundle")
    copied = lks._upload_provider_file(source, str(tmp_path / "copy.bin"))
    assert copied["status"] == "uploaded"
    assert Path(copied["destination_uri"]).read_bytes() == b"bundle"
    monkeypatch.chdir(tmp_path)
    relative_copy = lks._upload_provider_file(source, "relative-copy.bin")
    assert Path(relative_copy["destination_uri"]).read_bytes() == b"bundle"
    unsupported = lks._upload_provider_file(source, "https://storage.example/copy.bin")
    assert unsupported["status"] == "blocked"
    blocking_parent = tmp_path / "blocking-parent"
    blocking_parent.write_text("not a directory", encoding="utf-8")
    failed = lks._upload_provider_file(source, str(blocking_parent / "copy.bin"))
    assert failed["status"] == "blocked"

    uploads: list[tuple[str, str, str]] = []

    class FakeBlob:
        def __init__(self, bucket_name: str, key: str) -> None:
            self.bucket_name = bucket_name
            self.key = key

        def upload_from_filename(self, filename: str) -> None:
            uploads.append((self.bucket_name, self.key, filename))

    class FakeBucket:
        def __init__(self, name: str) -> None:
            self.name = name

        def blob(self, key: str) -> FakeBlob:
            return FakeBlob(self.name, key)

    class FakeClient:
        def bucket(self, name: str) -> FakeBucket:
            return FakeBucket(name)

    google_module = types.ModuleType("google")
    cloud_module = types.ModuleType("google.cloud")
    storage_module = types.ModuleType("google.cloud.storage")
    storage_module.Client = FakeClient
    cloud_module.storage = storage_module
    google_module.cloud = cloud_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_module)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_module)
    gs_result = lks._upload_provider_file(source, "gs://bucket/path/provider.zip")
    assert gs_result["status"] == "uploaded"
    assert uploads == [("bucket", "path/provider.zip", str(source))]

    assert lks._signed_url_from_private_inputs(
        {"bundle_get": {"signed_url": "https://example"}},
        "bundle_get",
    ) == "https://example"
    summary = lks._signed_url_entry_summary(
        private_inputs={"bundle_get": {"signed_url": "https://example?X-Goog-Signature=abc"}},
        key="bundle_get",
        private_input_path=tmp_path / "private.json",
    )
    assert summary["signed_url_signature_present"] is True


def test_asset_materialization_commands_and_unitree_resolution_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_zip = tmp_path / "Lightwheel_Kitchen.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("Collected_KitchenRoom/", "")
        archive.writestr(MAIN_USD_RELATIVE, "PXR-USDC")
        archive.writestr("Collected_KitchenRoom/robot.gltf", "{}")

    inventory = _asset_inventory_from_zip(source_zip)
    assert inventory["extension_counts"][".gltf"] == 1
    assert inventory["mujoco_native_asset_present"] is True

    asset_root = lks._materialize_assets(
        source_zip=source_zip,
        asset_output_dir=tmp_path / "assets",
    )
    assert (asset_root / MAIN_USD_RELATIVE).is_file()
    assert lks._materialize_assets(source_zip=source_zip, asset_output_dir=asset_root) == asset_root

    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as archive:
        archive.writestr("other.usd", "PXR-USDC")
    with pytest.raises(FileNotFoundError):
        lks._materialize_assets(source_zip=bad_zip, asset_output_dir=tmp_path / "bad-assets")

    assert lks._run_command([str(tmp_path / "definitely-missing-command")])["status"] == (
        "not_available"
    )
    timeout = lks._run_command(
        [sys.executable, "-c", "import time; time.sleep(1)"],
        timeout=0,
    )
    assert timeout["status"] == "failed"
    assert timeout["reason"] == "timeout"

    unitree_root = tmp_path / "unitree_g1"
    unitree_root.mkdir()
    (unitree_root / "g1.xml").write_text("<mujoco/>\n", encoding="utf-8")
    assert lks._resolve_unitree_g1_mjcf_root(explicit_root=unitree_root) == unitree_root
    monkeypatch.delenv("BLUEPRINT_MUJOCO_G1_MODEL_ROOT", raising=False)
    fake_repo = tmp_path / "fake_repo"
    relative_unitree_root = fake_repo / "relative_unitree"
    relative_unitree_root.mkdir(parents=True)
    (relative_unitree_root / "g1.xml").write_text("<mujoco/>\n", encoding="utf-8")
    monkeypatch.setattr(
        lks,
        "__file__",
        str(fake_repo / "src" / "blueprint_pipeline" / "lightwheel.py"),
    )
    assert lks._resolve_unitree_g1_mjcf_root(explicit_root="relative_unitree") == (
        relative_unitree_root
    )
    (tmp_path / "fake_cwd").mkdir()
    monkeypatch.chdir(tmp_path / "fake_cwd")
    assert lks._resolve_unitree_g1_mjcf_root(explicit_root=tmp_path / "missing") is None
    assert _unitree_g1_mjcf_summary(None)["status"] == "not_found"

    legacy_runner = tmp_path / "legacy_runner.py"
    lks._write_isaac_runner_script_legacy_unused(legacy_runner)
    assert "Isaac Sim runtime entrypoint" in legacy_runner.read_text(encoding="utf-8")


def test_usd_stage_and_checker_summaries_cover_runtime_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd_path = tmp_path / "KitchenRoom.usd"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    mesh_type = type("Mesh", (), {})
    xformable_type = type("Xformable", (), {})

    class FakePrim:
        def __init__(self, type_name: str, *apis: type) -> None:
            self._type_name = type_name
            self._apis = apis

        def GetTypeName(self) -> str:
            return self._type_name

        def IsA(self, api: type) -> bool:
            return api in self._apis

        def GetPath(self) -> str:
            return "/World"

    class FakeBox:
        def GetMin(self) -> tuple[float, float, float]:
            return (0.0, 1.0, 2.0)

        def GetMax(self) -> tuple[float, float, float]:
            return (3.0, 4.0, 5.0)

    class FakeBound:
        def ComputeAlignedBox(self) -> FakeBox:
            return FakeBox()

    class FakeBBoxCache:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def ComputeWorldBound(self, root) -> FakeBound:
            return FakeBound()

    class FakeTimeCode:
        @staticmethod
        def Default() -> str:
            return "default"

    class FakeStageObject:
        def Traverse(self) -> list[FakePrim]:
            return [
                FakePrim("Mesh", mesh_type, xformable_type),
                FakePrim("Xform", xformable_type),
            ]

        def GetPseudoRoot(self) -> object:
            return object()

        def GetDefaultPrim(self) -> FakePrim:
            return FakePrim("Xform")

    class FakeStage:
        open_result: object | None = FakeStageObject()
        should_raise = False

        @staticmethod
        def Open(path: str) -> object | None:
            if FakeStage.should_raise:
                raise RuntimeError(f"cannot open {path}")
            return FakeStage.open_result

    usd_module = types.ModuleType("pxr.Usd")
    usd_module.Stage = FakeStage
    usd_module.TimeCode = FakeTimeCode
    usd_geom_module = types.ModuleType("pxr.UsdGeom")
    usd_geom_module.BBoxCache = FakeBBoxCache
    usd_geom_module.Mesh = mesh_type
    usd_geom_module.Xformable = xformable_type
    pxr_module = types.ModuleType("pxr")
    pxr_module.Usd = usd_module
    pxr_module.UsdGeom = usd_geom_module
    monkeypatch.setitem(sys.modules, "pxr", pxr_module)
    monkeypatch.setitem(sys.modules, "pxr.Usd", usd_module)
    monkeypatch.setitem(sys.modules, "pxr.UsdGeom", usd_geom_module)

    opened = lks._usd_stage_summary(usd_path)
    assert opened["status"] == "opened"
    assert opened["prim_count"] == 2
    assert opened["mesh_count"] == 1

    FakeStage.open_result = None
    assert lks._usd_stage_summary(usd_path)["reason"] == "stage_open_returned_none"
    FakeStage.should_raise = True
    assert lks._usd_stage_summary(usd_path)["status"] == "failed"

    monkeypatch.setattr(lks, "_command_path", lambda name: None)
    assert lks._usdchecker_summary(usd_path)["reason"] == "usdchecker_unavailable"

    monkeypatch.setattr(lks, "_command_path", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        lks,
        "_run_command",
        lambda command, timeout=60: {
            "status": "failed",
            "stdout_head": "Could not load sublayer UnitsAdjust-deadbeef.metricsAssembler",
            "stderr_head": "OmniPBR.mdl UnresolvableDependency ShaderSdrCompliance",
        },
    )
    checked = lks._usdchecker_summary(usd_path)
    assert checked["status"] == "failed"
    assert "usd_missing_sublayer_metrics_assembler" in checked["blockers"]
    assert "usd_shader_registry_or_type_compliance_errors" in checked["blockers"]

    no_refs_zip = tmp_path / "no_refs.zip"
    with zipfile.ZipFile(no_refs_zip, "w") as archive:
        archive.writestr(MAIN_USD_RELATIVE, "PXR-USDC")
    no_refs = _usd_dependency_presence_audit(
        asset_root=tmp_path,
        source_zip=no_refs_zip,
        usdchecker={"status": "passed"},
    )
    assert no_refs["status"] == "not_applicable"


def test_runtime_preflight_docker_success_and_malformed_output_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lks, "_command_path", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(lks, "_module_available", lambda name: True)

    def fake_run_command(command, timeout=60):
        if command[0] == "docker":
            return {
                "status": "passed",
                "stdout_head": "x86_64\nlinux\n1048576\nfalse\n",
                "stderr_head": "",
            }
        return {"status": "passed", "stdout_head": "RTX 4090, 24576 MiB, 555.0", "stderr_head": ""}

    monkeypatch.setattr(lks, "_run_command", fake_run_command)
    preflight = _runtime_preflight()
    assert preflight["docker"]["architecture"] == "x86_64"
    assert preflight["docker"]["memory_total_bytes"] == 1048576
    assert "docker_nvidia_runtime_unavailable" in preflight["blockers"]

    def malformed_docker(command, timeout=60):
        if command[0] == "docker":
            return {"status": "passed", "stdout_head": "too-few-lines\n", "stderr_head": ""}
        return {"status": "passed", "stdout_head": "RTX 4090", "stderr_head": ""}

    monkeypatch.setattr(lks, "_run_command", malformed_docker)
    malformed = _runtime_preflight()
    assert malformed["docker"]["architecture"] is None
    assert malformed["docker"]["nvidia_runtime_present"] is False


def test_draw_previews_skips_missing_or_invalid_points(tmp_path: Path) -> None:
    assert _draw_previews(
        thumbnail_path=tmp_path / "missing.png",
        scenarios=_default_scenarios()[:1],
        output_dir=tmp_path / "previews",
    ) == []

    thumb = tmp_path / THUMBNAIL_RELATIVE
    thumb.parent.mkdir(parents=True)
    Image.new("RGBA", (64, 64), (240, 230, 210, 255)).save(thumb)
    previews = _draw_previews(
        thumbnail_path=thumb,
        scenarios=[
            {
                "scenario_id": "invalid_points",
                "spawn_position_xyz": None,
                "waypoints_xyz": [["bad"], [0.0, 0.0, 0.0]],
                "target_position_xyz": [1.0, 1.0, 0.0],
            }
        ],
        output_dir=tmp_path / "previews",
    )
    assert len(previews) == 1
    assert Path(previews[0]["path"]).is_file()


def test_provider_packet_uploads_local_bundle_and_packet_with_private_signed_inputs(
    tmp_path: Path,
) -> None:
    source_zip, bundle, unitree_summary, provider_request = _make_provider_fixture(tmp_path)
    remote_root = tmp_path / "remote"
    private_file = tmp_path / "signed_urls.local"
    private_file.write_text(
        json.dumps(
            {
                "bundle_get": {
                    "signed_url": "https://storage.example/bundle.zip?X-Goog-Signature=secret",
                    "resource": "gs://bucket/bundle.zip",
                    "http_verb": "GET",
                },
                "runtime_result_put": {
                    "signed_url": "https://storage.example/result.json?X-Goog-Signature=secret",
                    "resource": "gs://bucket/result.json",
                    "http_verb": "PUT",
                },
            }
        ),
        encoding="utf-8",
    )

    packet = _write_provider_packet(
        output_dir=tmp_path,
        source_zip=source_zip,
        bundle=bundle,
        unitree_g1_mjcf=unitree_summary,
        provider_execution_request_path=provider_request,
        runtime={"blockers": [], "provider_credentials": {}},
        provider_proof_path=None,
        provider_artifact_root_uri=f"file://{remote_root}",
        upload_provider_packet=True,
        provider_private_signed_url_file=private_file,
        repo_commit="abc123",
        selected_isaac_runtime_image_ref="nvcr.io/nvidia/isaac-sim:5.0.0",
    )

    assert packet["status"] == "provider_packet_uploaded_ready_for_runtime_probe"
    assert packet["asset_bundle"]["upload_status"] == "uploaded"
    assert packet["provider_packet_upload"]["status"] == "uploaded"
    assert (remote_root / PROVIDER_BUNDLE_NAME).is_file()
    assert (remote_root / PROVIDER_PACKET_NAME).is_file()


def test_provider_packet_upload_blockers_and_missing_image_ref_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_zip, bundle, unitree_summary, provider_request = _make_provider_fixture(tmp_path)
    monkeypatch.delenv("BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF", raising=False)
    monkeypatch.delenv("BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF", raising=False)
    monkeypatch.delenv("BLUEPRINT_LIGHTWHEEL_ISAAC_RUNTIME_IMAGE_REF", raising=False)
    monkeypatch.setattr(lks, "DEFAULT_ISAAC_RUNTIME_IMAGE_REF", "")

    missing_root = _write_provider_packet(
        output_dir=tmp_path,
        source_zip=source_zip,
        bundle=bundle,
        unitree_g1_mjcf=unitree_summary,
        provider_execution_request_path=provider_request,
        runtime={"blockers": [], "provider_credentials": {}},
        provider_proof_path=None,
        provider_artifact_root_uri=None,
        upload_provider_packet=True,
        provider_private_signed_url_file=None,
        repo_commit="abc123",
    )
    assert missing_root["status"] == "provider_packet_upload_blocked"
    assert "missing_provider_artifact_root_uri" in missing_root["blockers"]
    assert "versioned_provider_fetchable_BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF" in missing_root[
        "required_missing_inputs"
    ]

    unsupported = _write_provider_packet(
        output_dir=tmp_path,
        source_zip=source_zip,
        bundle=bundle,
        unitree_g1_mjcf=unitree_summary,
        provider_execution_request_path=provider_request,
        runtime={"blockers": [], "provider_credentials": {}},
        provider_proof_path=None,
        provider_artifact_root_uri="ftp://provider.example/lightwheel",
        upload_provider_packet=True,
        provider_private_signed_url_file=None,
        repo_commit="abc123",
        selected_isaac_runtime_image_ref="nvcr.io/nvidia/isaac-sim:5.0.0",
    )
    assert unsupported["status"] == "provider_packet_upload_blocked"
    assert any("unsupported_provider_packet_upload_scheme:ftp" in item for item in unsupported["blockers"])


def test_build_lightwheel_scenarios_writes_fail_closed_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    source_zip = repo_root / "Lightwheel_Kitchen.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr(MAIN_USD_RELATIVE, "PXR-USDC")

    real_inventory = lks._asset_inventory_from_zip

    def inventory_with_missing_main(path: Path) -> dict:
        inventory = real_inventory(path)
        inventory["main_usd_present"] = False
        return inventory

    monkeypatch.setattr(lks, "_asset_inventory_from_zip", inventory_with_missing_main)
    monkeypatch.setattr(lks, "_usd_stage_summary", lambda path: {"status": "opened"})
    monkeypatch.setattr(
        lks,
        "_usdchecker_summary",
        lambda path: {
            "status": "failed",
            "blockers": ["usd_requires_omniverse_mdl_material_registry"],
        },
    )
    monkeypatch.setattr(
        lks,
        "_usd_dependency_presence_audit",
        lambda **kwargs: {
            "status": "missing_required_dependencies",
            "blockers": ["usd_unresolvable_texture_dependencies"],
            "missing_dependency_names": ["3d66Model-missing.png"],
        },
    )
    monkeypatch.setattr(
        lks,
        "_runtime_preflight",
        lambda: {
            "status": "blocked",
            "blockers": ["isaac_sim_runtime_unavailable"],
            "isaac_local_runtime_ready": False,
            "provider_credentials": {},
        },
    )

    result = lks.build_lightwheel_kitchen_isaac_scenarios(
        source_repo_root=repo_root,
        output_dir=tmp_path / "out",
        repo_commit="abc123",
    )

    assert result["status"] == "blocked"
    assert "lightwheel_kitchen_main_usd_missing" in result["blockers"]
    assert "usd_requires_omniverse_mdl_material_registry" in result["blockers"]
    assert "usd_unresolvable_texture_dependencies" in result["blockers"]
    assert result["scenario_execution_status"] == "blocked_not_executed"
    assert Path(result["manifest_path"]).is_file()
    assert Path(result["provider_packet_manifest"]).is_file()
    assert result["final_output_manifests"]["execution_manifest"].endswith(
        ISAAC_EXECUTION_MANIFEST_NAME
    )


def test_build_requires_source_zip(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        lks.build_lightwheel_kitchen_isaac_scenarios(
            source_repo_root=tmp_path / "empty-repo",
            output_dir=tmp_path / "out",
        )


def test_main_passes_cli_arguments_and_reports_status(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    def fake_build(**kwargs) -> dict[str, str]:
        received.update(kwargs)
        return {"status": "ready_for_isaac_execution", "manifest_path": "/tmp/handoff.json"}

    monkeypatch.setattr(lks, "build_lightwheel_kitchen_isaac_scenarios", fake_build)
    code = lks.main(
        [
            "--capture-root",
            "/tmp/capture",
            "--source-zip",
            "/tmp/source.zip",
            "--output-dir",
            "/tmp/out",
            "--upload-provider-packet",
            "--repo-commit",
            "abc123",
        ]
    )

    assert code == 0
    assert received["capture_root"] == "/tmp/capture"
    assert received["source_zip"] == "/tmp/source.zip"
    assert received["upload_provider_packet"] is True
    assert json.loads(capsys.readouterr().out)["status"] == "ready_for_isaac_execution"


def test_module_entrypoint_uses_main_guard_without_external_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_zip = tmp_path / "Lightwheel_Kitchen.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr(MAIN_USD_RELATIVE, "PXR-USDC")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lightwheel_kitchen_isaac_scenarios.py",
            "--source-zip",
            str(source_zip),
            "--output-dir",
            str(tmp_path / "entrypoint-out"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_module_as_main("blueprint_pipeline.lightwheel_kitchen_isaac_scenarios")

    assert exc_info.value.code == 2
