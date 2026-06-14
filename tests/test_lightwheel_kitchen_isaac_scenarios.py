from __future__ import annotations

import json
import os
import py_compile
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image

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
