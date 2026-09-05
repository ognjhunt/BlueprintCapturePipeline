"""Run the real Node worker; only its GPU/renderer boundaries are hermetic."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
ROLES = ["images", "target_support", "scene_without_target"]
pytestmark = pytest.mark.slow


def _sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(root, path):
    return {"relative_path": str(path.relative_to(root)), "sha256": _sha(path), "size_bytes": path.stat().st_size}


def _run(tmp_path, *, initial_good=8, reserve_count=16, reserve_good=16, recovery=True):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name in ["adp_retained_scene_render_provider_runner.mjs", "source_calibration_camera_recovery.mjs"]:
        if recovery or name != "source_calibration_camera_recovery.mjs":
            shutil.copy2(ROOT / "scripts" / name, runtime / name)
    renderer = runtime / "renderer"
    renderer.mkdir()
    for name, white in [("good", True), ("bad", False)]:
        img = Image.new("RGB", (1280, 1280))
        if white:
            img.paste((255, 255, 255), (0, 0, 256, 256))
        img.save(renderer / f"{name}.png")
    (renderer / "render_splat.mjs").write_text('''
import fs from 'node:fs'; import path from 'node:path';
const arg=k=>process.argv[process.argv.indexOf('--'+k)+1];
const cameras=JSON.parse(fs.readFileSync(arg('cameras'))), out=arg('out');
fs.mkdirSync(out,{recursive:true});
const role=path.basename(arg('splat'),'.ply');
for(const {id} of cameras) {
  const n=Number(id.split('-')[1]);
  const pass=id.startsWith('reserve-') ? n<=RESERVE_GOOD : n<=INITIAL_GOOD;
  const good=role==='target_support' || (role==='images' && pass);
  fs.copyFileSync(new URL(good ? './good.png' : './bad.png',import.meta.url),path.join(out,id+'.png'));
}
process.stdout.write(JSON.stringify({status:'completed',graphics_diagnostics:{webgl_available:true,renderer:'hermetic NVIDIA fixture'}}));
'''.replace("INITIAL_GOOD", str(initial_good)).replace("RESERVE_GOOD", str(reserve_good)))
    binaries = tmp_path / "bin"
    binaries.mkdir()
    gpu = binaries / "nvidia-smi"
    gpu.write_text("#!/bin/sh\necho 'Hermetic GPU, fixture-driver'\n")
    gpu.chmod(0o755)

    def cameras(prefix, count):
        return [{"camera_id": f"{prefix}-{n:02}", "T_world_camera_provider_frame":
                 [[1, 0, 0, n + (100 if prefix == 'reserve' else 0)], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                 "intrinsics": {"fx": 1000, "fy": 1000, "cx": 640, "cy": 640, "width": 1280, "height": 1280,
                                "near": 0.01, "far": 100000}} for n in range(1, count + 1)]

    camera_file = runtime / "cameras.json"
    camera_file.write_text(json.dumps(cameras("source", 16), indent=2) + "\n")
    layers = {}
    for role in ROLES:
        ply = runtime / f"{role}.ply"
        ply.write_text("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 0\n")
        layers[role] = {**_record(runtime, ply), "gaussian_count": 1, "camera_set_label": role, "purpose": "hermetic"}
    request = {"schema_version": "adp009d_source_calibration_gpu_renderer_runtime_request.v1",
               "render_scope": "source_calibration", "layers": layers, "camera_count": 16, "expected_png_count": 48,
               "candidate_policy_queried": False, "paid_inference_performed": False,
               "camera_contract": _record(runtime, camera_file), "dimensions": {"width": 1280, "height": 1280},
               "render_options": {"warmup_ms": 0, "settle_frames": 1, "settle_ms": 0}, "renderer_identity": {},
               "preparation_digest": "sha256:" + "a" * 64, "blueprint_commit": "b" * 40}
    if recovery:
        reserve = runtime / "reserve.json"
        reserve.write_text(json.dumps(cameras("reserve", reserve_count)))
        request["camera_recovery"] = {"schema_version": "source_calibration_camera_recovery.v1", "maximum_rounds": 1,
                                      "replacement_camera_contract": _record(runtime, reserve),
                                      "visibility_gate": {"support_threshold_8bit": 24,
                                                          "visual_contribution_threshold_8bit": 8,
                                                          "minimum_visible_target_fraction": 0.01}}
    # Request has only integral values and these exactly representable thresholds.
    request["runtime_request_digest"] = "sha256:" + hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (runtime / "render_request.json").write_text(json.dumps(request))
    output = tmp_path / "output"
    run = subprocess.run(["node", str(runtime / "adp_retained_scene_render_provider_runner.mjs"),
                          "--runtime", str(runtime), "--output", str(output)],
                         env={**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"]},
                         capture_output=True, text=True, timeout=60)
    result = json.loads((output / "adp009d_source_calibration_gpu_render_result.v1.json").read_text())
    return run, result, output, request, camera_file


def test_real_worker_replaces_only_failed_cameras_in_one_round(tmp_path):
    run, result, output, _, _ = _run(tmp_path)
    assert run.returncode == 0, result
    resolution = result["camera_resolution"]
    assert resolution["rounds_used"] == 1
    assert len(resolution["measurement_rows"]) == 32
    assert len(resolution["original_render_groups"]) == len(resolution["replacement_render_groups"]) == 3
    selection = resolution["selection"]
    assert selection[0] == {"camera_id": "source-01", "candidate_camera_id": "source-01"}
    assert selection[8] == {"camera_id": "source-09", "candidate_camera_id": "reserve-01"}
    resolved = json.loads((output / resolution["resolved_camera_file"]["relative_path"]).read_text())
    assert resolved[8]["T_world_camera_provider_frame"][0][3] == 101
    assert resolved[8]["camera_id"] == "source-09"
    assert len(list(output.rglob("*.png"))) == 144
    for group in result["render_groups"]:
        manifest_path = output / group["manifest"]["relative_path"]
        manifest = json.loads(manifest_path.read_text())
        assert manifest["calibrated_camera_file"]["digest"] == resolution["resolved_camera_file"]["sha256"]
        assert manifest["calibrated_cameras"][8]["id"] == "source-09"
        row = manifest["renders"][8]
        measurement = next(m for m in resolution["measurement_rows"] if m["candidate_camera_id"] == "reserve-01")
        assert _sha(manifest_path.parent / row["relative_path"]) == measurement["frame_digests"][group["role"]]


def test_real_worker_keeps_original_bytes_when_all_initial_cameras_pass(tmp_path):
    run, result, output, _, camera_file = _run(tmp_path, initial_good=16)
    assert run.returncode == 0, result
    resolution = result["camera_resolution"]
    assert resolution["rounds_used"] == 0 and resolution["replacement_render_groups"] == []
    assert result["render_groups"] == resolution["original_render_groups"]
    assert (output / resolution["resolved_camera_file"]["relative_path"]).read_bytes() == camera_file.read_bytes()
    assert len(list(output.rglob("*.png"))) == 48


@pytest.mark.parametrize("reserve_count,reserve_good", [(16, 0), (0, 0), (2, 2)])
def test_real_worker_exhaustion_retains_candidates_and_never_completes(tmp_path, reserve_count, reserve_good):
    run, result, output, _, _ = _run(tmp_path, reserve_count=reserve_count, reserve_good=reserve_good)
    assert run.returncode == 2 and result["status"] == "blocked", result
    assert result["blockers"] == ["source_calibration_camera_recovery_exhausted"]
    resolution = result["camera_resolution"]
    assert resolution["rounds_used"] == int(reserve_count > 0)
    assert len(resolution["measurement_rows"]) == 16 + reserve_count
    assert len(list(output.rglob("*.png"))) == 48 + 3 * reserve_count
    assert result["render_groups"] == []


def test_legacy_source_request_needs_no_recovery_helper(tmp_path):
    run, result, output, _, _ = _run(tmp_path, recovery=False)
    assert run.returncode == 0 and result["status"] == "completed", result
    assert "camera_resolution" not in result
    assert len(list(output.rglob("*.png"))) == 48


@pytest.mark.parametrize("channels", [3, 4])
def test_builtin_png_decoder_handles_all_filters_and_rejects_corruption(tmp_path, channels):
    import struct
    import zlib

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    rows = [bytes([10 + y, 30, 200, 255][:channels] + [220, 20 + y, 40, 255][:channels]) for y in range(5)]
    raw = bytearray()
    for y, row in enumerate(rows):
        raw.append(y)
        for x, value in enumerate(row):
            left = row[x - channels] if x >= channels else 0
            up = rows[y - 1][x] if y else 0
            diagonal = rows[y - 1][x - channels] if y and x >= channels else 0
            p = left + up - diagonal
            a, b, c = abs(p - left), abs(p - up), abs(p - diagonal)
            paeth = left if a <= b and a <= c else up if b <= c else diagonal
            predictor = [0, left, up, (left + up) // 2, paeth][y]
            raw.append((value - predictor) % 256)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 5, 8, 2 if channels == 3 else 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b"")
    source = tmp_path / "all-filters.png"
    source.write_bytes(png)
    expected = b"".join(row[offset:offset + 3] for row in rows for offset in range(0, len(row), channels))
    script = tmp_path / "decode.mjs"
    script.write_text(f'''
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {{decodePng}} from {json.dumps((ROOT / 'scripts/source_calibration_camera_recovery.mjs').as_uri())};
const png=fs.readFileSync({json.dumps(str(source))});
assert.equal(decodePng(png,{{width:2,height:5}}).toString('hex'),{json.dumps(expected.hex())});
const broken=Buffer.from(png); broken[broken.length-1]^=1;
assert.throws(()=>decodePng(broken,{{width:2,height:5}}),/png_crc_invalid/);
assert.throws(()=>decodePng(png,{{width:1280,height:1280}}),/png_format_invalid/);
''')
    completed = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
