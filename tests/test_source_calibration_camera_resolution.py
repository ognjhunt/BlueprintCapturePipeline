"""Real packet and Node return cross the independent Python camera validator."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

from PIL import Image
import pytest

from blueprint_pipeline import public_scene_inpainting_preparation as preparation
from blueprint_pipeline import source_calibration_render_packet as packet
from blueprint_pipeline import source_calibration_render_return as returned
from blueprint_pipeline.decision_evidence_contracts import canonical_json, cross_runtime_canonical_digest
from tests.test_source_calibration_render_packet import valid_source_bundle_inputs

pytestmark = pytest.mark.slow


def _real_worker_return(tmp_path, monkeypatch):
    from tests import test_public_scene_inpainting_preparation as cpu
    original = cpu._prepare

    def with_reserves(root):
        paths, base = original(root)
        request = deepcopy(base["context"]["request"])
        reserve = deepcopy(request["camera_policy"]["views"])
        for i, row in enumerate(reserve):
            row["camera_id"] = f"reserve-{i + 1:02d}"
            row["position_offset_m"][0] += .02
            row["position_offset_m"][2] += .05
        request["camera_policy"]["replacement_views"] = reserve
        request.pop("request_digest")
        request_path = paths["data"] / "with-reserves.json"
        request_path.write_text(canonical_json(cpu.module.build_public_scene_inpainting_input_request(request)))
        output = paths["data"] / "with-reserves"
        value = preparation.prepare_public_scene_inpainting_inputs(request_path=request_path,
            repo_root=paths["repo"], data_root=paths["data"], output_root=output)
        return {**paths, "output": output}, value

    monkeypatch.setattr(cpu, "_prepare", with_reserves)
    args, prepared = valid_source_bundle_inputs(tmp_path, monkeypatch)
    receipt = packet.build_source_calibration_gpu_render_bundle(**args)
    assert packet.validate_source_calibration_bundle(receipt) == receipt
    runtime = args["job_dir"] / "provider_runtime"
    renderer = runtime / "renderer"
    for name, visible in (("good", True), ("bad", False)):
        image = Image.new("RGB", (1280, 1280))
        if visible:
            image.paste((255, 255, 255), (512, 512, 768, 768))
        image.save(renderer / f"{name}.png")
    # Only the external renderer and nvidia-smi are hermetic; packet, worker,
    # PNG measurements, camera selection and host verification remain real.
    (renderer / "render_splat.mjs").unlink()  # Bundle staging may hardlink immutable source code.
    (renderer / "render_splat.mjs").write_text('''
import fs from 'node:fs'; import path from 'node:path';
const arg=k=>process.argv[process.argv.indexOf('--'+k)+1];
const cameras=JSON.parse(fs.readFileSync(arg('cameras'))), out=arg('out');
fs.mkdirSync(out,{recursive:true}); const role=path.basename(arg('splat'),'.ply');
for(const {id} of cameras) {
 const visible=role==='target_support' || (role==='images' && id!=='source-07');
 fs.copyFileSync(new URL(visible?'./good.png':'./bad.png',import.meta.url),path.join(out,id+'.png'));
}
process.stdout.write(JSON.stringify({status:'completed',graphics_diagnostics:{webgl_available:true,renderer:'hermetic NVIDIA fixture'}}));
''')
    binaries = tmp_path / "bin"
    binaries.mkdir()
    gpu = binaries / "nvidia-smi"
    gpu.write_text("#!/bin/sh\necho 'Hermetic GPU, fixture-driver'\n")
    gpu.chmod(0o755)
    output = tmp_path / "provider-return"
    command = ["node", str(runtime / "adp_retained_scene_render_provider_runner.mjs"),
               "--runtime", str(runtime), "--output", str(output)]
    process = subprocess.run(command, env={**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"]},
                             capture_output=True, text=True, timeout=60)
    result_path = output / (returned.RESULT_SCHEMA + ".json")
    result = json.loads(result_path.read_text())
    assert process.returncode == 0, (result, process.stderr)
    return prepared, result_path, result, output


def test_actual_packet_worker_and_host_agree_on_bounded_repair(tmp_path, monkeypatch):
    prepared, result_path, result, output = _real_worker_return(tmp_path, monkeypatch)
    path = output / "verified-return.json"
    value = returned.materialize_source_calibration_return(
        prepared_inputs=prepared, result_path=result_path, output_path=path)
    groups = returned.verify_source_calibration_return(prepared, path)
    assert set(groups) == set(returned.ROLES)
    assert len(result["camera_resolution"]["measurement_rows"]) == 32
    assert result["camera_resolution"]["selection"][7] == {
        "camera_id": "source-07", "candidate_camera_id": "reserve-01"}
    assert value["preparation_digest"] == prepared["preparation_digest"]
    assert all(row["manifest"]["render_count"] == 16 for row in groups.values())
    # Finalization must retain the resolved poses, not regenerate original07.
    closed = {**value, "execution_closure": {"fixture_only": True}}
    monkeypatch.setattr(returned, "require_source_calibration_closure", lambda *args: closed)
    final = preparation.finalize_public_scene_inpainting_inputs(
        preparation_path=prepared["preparation_path"], returned_group_path=path)
    cameras = json.loads((Path(prepared["context"]["paths"]["output"]) /
                          final["derived_artifacts"]["cameras"]["relative_path"]).read_text())
    assert cameras[7]["T_world_camera_provider_frame"] == prepared["camera_recovery"]["replacement_cameras"][0]["T_world_camera_provider_frame"]
    assert final["source_calibration_render"]["camera_resolution"]["rounds_used"] == 1
    assert "--verify-existing" in final["replay_command"]
    assert preparation.adopt_finalized_public_scene_inpainting_inputs(
        preparation_path=prepared["preparation_path"], returned_group_path=path) == final


@pytest.mark.parametrize("mutation", ["metric", "selection", "rounds"])
def test_resealed_worker_claims_do_not_override_pixels_or_frozen_candidates(tmp_path, monkeypatch, mutation):
    prepared, result_path, result, output = _real_worker_return(tmp_path, monkeypatch)
    resolution = result["camera_resolution"]
    if mutation == "metric":
        resolution["measurement_rows"][7]["passed"] = True
    elif mutation == "selection":
        resolution["selection"][7]["candidate_camera_id"] = "reserve-02"
    else:
        resolution["rounds_used"] = 2
    result["result_digest"] = cross_runtime_canonical_digest(result, digest_field="result_digest")
    result_path.write_text(canonical_json(result))
    with pytest.raises(ValueError, match="source_calibration_camera_"):
        returned.materialize_source_calibration_return(
            prepared_inputs=prepared, result_path=result_path, output_path=output / "rejected-return.json")
    assert not (output / "rejected-return.json").exists()
