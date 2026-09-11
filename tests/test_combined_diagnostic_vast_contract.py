"""The actual native bundle must pass Vast's final static gate before ready."""

import json
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline.native_task_arena_bundle import build_native_task_arena_bundle
from blueprint_pipeline.native_task_arena_execution_contract import (
    COMBINED_DIAGNOSTIC_VARIANT,
    COMPOSITION_GATE_VARIANT,
    composition_gate_runner_valid,
    execution_contract,
    combined_diagnostic_runner_valid,
)
from blueprint_pipeline.native_task_composition_bundle import composition_runtime_sources
from blueprint_pipeline.native_task_arena_runtime_preflight_bundle import (
    load_verified_native_task_arena_runtime_preflight_bundle,
)
from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight
from tests.test_native_task_arena_bundle import _packet, _runtime_source_packet


def preflight(bundle, tmp):
    return _blueprint_bundle_preflight(
        job_dir=tmp,
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="native_task_arena",
        bundle_path=Path(bundle),
        provider_bundle_url="https://example.com/offline.zip?sig=offline",
        provider_output_put_url="https://example.com/offline-output.zip?sig=offline",
        verify_staging_urls=False,
        allow_staging_output_put_probe=False,
    )


@pytest.mark.parametrize("variant", [None, COMBINED_DIAGNOSTIC_VARIANT, COMPOSITION_GATE_VARIANT])
def test_real_builder_bundle_and_verified_loader_share_actual_vast_gate(
    tmp_path, monkeypatch, variant
):
    import urllib.request

    def forbidden_network(*args, **kwargs):
        raise AssertionError("static archive preflight cannot call a provider")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden_network)
    package = Path(__file__).resolve().parents[1] / "src/blueprint_pipeline"
    source = tmp_path / "input.json"
    source.write_text(json.dumps({'require_composition_gate': variant == COMPOSITION_GATE_VARIANT}))
    inputs = {
        name: source
        for name in (
            "composition_request.json",
            "composition_scene_plan.json",
            "original_policy_runtime_inputs.json",
            "replay_request.json",
            "replay_scene_plan.json",
            "retained_cell_result.json",
            "retained_adapter_reset.json",
        )
    }
    receipt = build_native_task_arena_bundle(
        job_dir=tmp_path / "bundle",
        packet_dir=_packet(tmp_path, scene_id="840920"),
        worker_source=package / ("native_task_composition_worker.py" if variant == COMPOSITION_GATE_VARIANT
                                 else "native_task_combined_diagnostic_worker.py"),
        runtime_module_sources=composition_runtime_sources(include_replay=True),
        implementation_commit="e" * 40,
        execution_mode="runtime_preflight",
        runtime_variant=variant,
        expected_output_filename="native_task_arena_runtime_preflight.v1.json",
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        bound_runtime_inputs=inputs,
        generated_at="fixed",
    )
    observed = preflight(receipt["bundle_path"], tmp_path / "adapter")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    if variant is None:
        assert observed["blockers"] == [
            "native_task_arena_provider_manifest_invalid",
            "provider_runner_missing_native_task_arena_runtime_contract",
        ]
        with pytest.raises(ValueError, match="runtime_preflight_adapter_blocked"):
            load_verified_native_task_arena_runtime_preflight_bundle(
                receipt_path, expected_implementation_commit="e" * 40
            )
    else:
        assert observed["status"] == "passed" and observed["blockers"] == []
        assert (
            load_verified_native_task_arena_runtime_preflight_bundle(
                receipt_path, expected_implementation_commit="e" * 40
            )["bundle_sha256"]
            == receipt["bundle_sha256"]
        )
        with zipfile.ZipFile(receipt["bundle_path"]) as archive:
            parent = archive.read("provider_runtime/adp_arena_provider_runner.py").decode()
        validator = composition_gate_runner_valid if variant == COMPOSITION_GATE_VARIANT else combined_diagnostic_runner_valid
        assert validator(parent)


def test_fake_comments_and_unregistered_variant_never_satisfy_contract():
    assert execution_contract("runtime_preflight", "unknown") is None
    assert execution_contract("policy", COMBINED_DIAGNOSTIC_VARIANT) is None
    assert not combined_diagnostic_runner_valid("# CHILDREN runner run_diagnostic_children\npass\n")
    assert not composition_gate_runner_valid("# launch_native_task_isaaclab build_native_task_arena_environment run_native_asset_composition_gate\npass\n")
    source = (
        Path(__file__).resolve().parents[1]
        / "src/blueprint_pipeline/native_task_combined_diagnostic_worker.py"
    )
    real = source.read_text()
    assert not combined_diagnostic_runner_valid(
        real.replace(
            "blueprint_pipeline.native_task_retained_command_worker",
            "blueprint_pipeline.unregistered_worker",
        )
    )
