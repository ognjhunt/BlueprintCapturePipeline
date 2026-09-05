from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline import sealed_camera_render as renderer
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_sealed_camera_render import _StalledRendererProcess


def test_watchdog_preserves_streams_and_last_startup_checkpoint(tmp_path, monkeypatch, capsys):
    class Process(_StalledRendererProcess):
        def communicate(self, timeout=None):
            if self.terminated:
                self.returncode = -15
                return '{"status":"blocked","error":"splat_load_timeout"}', 'fixture startup diagnostic'
            return super().communicate(timeout)
    frames = tmp_path/'frames'
    frames.mkdir()
    (frames/'renderer_progress.jsonl').write_text('{"event":"splat_load_started"}\n')
    clock = iter((0., 0., 1.1))
    monkeypatch.setattr(renderer.time, 'monotonic', lambda: next(clock))
    output = tmp_path/'timeout.json'
    with pytest.raises(renderer.SealedCameraRenderError, match='render_harness_initial_progress_timeout'):
        renderer._wait_for_renderer_with_progress_watchdog(
            process=Process(), expected_frame_paths=[frames/'first.png'],
            render_timeout_seconds=10, initial_progress_timeout_seconds=1,
            progress_timeout_seconds=1, diagnostics_path=output,
        )
    evidence = json.loads(output.read_text())
    assert 'splat_load_timeout' in evidence['stdout_tail']
    assert 'fixture startup diagnostic' in evidence['stderr_tail']
    assert 'splat_load_started' in evidence['renderer_progress_tail']
    assert evidence['completed_frame_count'] == 0
    assert evidence['initial_progress_timeout_seconds'] == 1
    assert evidence['render_qualification_claimed'] is False
    assert evidence['diagnostic_digest'] == canonical_digest(evidence, digest_field='diagnostic_digest')
    assert 'fixture startup diagnostic' in capsys.readouterr().err


@pytest.mark.slow
def test_real_driver_configures_camera_before_warmup_and_retains_error_when_close_stalls(tmp_path):
    node = shutil.which('node')
    assert node, 'Node runtime required for the renderer lifecycle rehearsal'
    driver_source = Path(__file__).resolve().parents[1]/'tools/splat_render/render_splat.mjs'
    fake = tmp_path/'fake-playwright.mjs'
    fake.write_text('''
let configured = false;
const page = {
  on() {}, async goto() {}, async waitForFunction() {}, async exposeFunction() {},
  async evaluate(fn, args) {
    const source = String(fn);
    if (source.includes("BlueprintSplat.load")) return {radius: 1, center: [0,0,0], size: [1,1,1]};
    if (source.includes("getContext")) return {webgl_available: true, renderer: "fixture"};
    if (source.includes("BlueprintSplat.setCamera")) {
      if (args.pose.T_world_camera_opencv[0][3] !== 3) throw new Error("wrong_initial_camera");
      configured = true; return;
    }
    if (source.includes("BlueprintSplat.warmup")) {
      if (!configured) throw new Error("uncalibrated_origin_warmup");
      throw new Error("fixture_warmup_failed_after_calibration");
    }
    throw new Error("unexpected_driver_operation");
  }
};
export const chromium = { async launch() {return {
  async newPage() {return page;}, version() {return "fixture-browser";},
  close() {return new Promise(() => {});}
};}};
''')
    driver = tmp_path/'driver.mjs'
    driver.write_text(driver_source.read_text().replace('from "playwright"', f'from "{fake.as_uri()}"'))
    frames = tmp_path/'frames'
    cameras = tmp_path/'cameras.json'
    cameras.write_text(json.dumps([{'id':'first', 'spec':{'pose':{'T_world_camera_opencv':
        [[1,0,0,3],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}, 'intrinsics':{'width':1280,'height':1280}}}]))
    process = subprocess.Popen([node,str(driver),'--splat',str(tmp_path/'fixture.ply'),
                                '--out',str(frames),'--cameras',str(cameras)],
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    output = tmp_path/'timeout.json'
    try:
        with pytest.raises(renderer.SealedCameraRenderError, match='render_harness_initial_progress_timeout'):
            renderer._wait_for_renderer_with_progress_watchdog(
                process=process, expected_frame_paths=[frames/'first.png'],
                render_timeout_seconds=10, initial_progress_timeout_seconds=1,
                progress_timeout_seconds=1, diagnostics_path=output,
            )
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    progress = [json.loads(row) for row in (frames/'renderer_progress.jsonl').read_text().splitlines()]
    events = [row['event'] for row in progress]
    assert events.index('first_camera_configured') < events.index('warmup_started')
    assert events[-1] == 'browser_close_started'
    evidence = json.loads(output.read_text())
    assert 'fixture_warmup_failed_after_calibration' in evidence['stdout_tail']
    assert 'uncalibrated_origin_warmup' not in evidence['stdout_tail']
    assert evidence['completed_frame_count'] == 0
    assert process.poll() is not None
