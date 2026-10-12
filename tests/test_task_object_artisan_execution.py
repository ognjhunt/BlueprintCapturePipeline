"""ADP-009 cumulative authority, interruption and OS isolation contracts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from blueprint_pipeline.task_object_artisan_metering import Meter
from blueprint_pipeline.task_object_artisan_process import run_controller


def test_resume_keeps_cost_request_count_and_original_deadline(tmp_path):
    path = tmp_path / "usage.json"
    kwargs = {"maximum_cost_usd": 40, "maximum_requests": 2, "deadline_epoch": time.time() + 30}
    first = Meter(path, **kwargs)
    first.after(first.before("design", "fixture-model"), 8, usage={"input_tokens": 10})
    resumed = Meter(path, **kwargs)
    assert resumed.remaining() == 32
    resumed.after(resumed.before("coding", "fixture-model"), 2)
    with pytest.raises(RuntimeError, match="authority_exhausted"):
        resumed.before("review", "fixture-model")
    with pytest.raises(ValueError, match="authority_changed"):
        Meter(path, **{**kwargs, "maximum_cost_usd": 41})
    with pytest.raises(ValueError, match="authority_changed"):
        Meter(path, **{**kwargs, "deadline_epoch": kwargs["deadline_epoch"] + 30})


def test_interrupted_or_unknown_charge_reserves_remaining_and_blocks_retry(tmp_path):
    path = tmp_path / "usage.json"
    kwargs = {"maximum_cost_usd": 15, "maximum_requests": 20, "deadline_epoch": time.time() + 30}
    meter = Meter(path, **kwargs)
    identity = meter.before("compaction", "fixture-model")
    assert meter.read()["calls"][0]["reserved_usd"] == 15
    with pytest.raises(RuntimeError, match="requires_reconciliation"):
        Meter(path, **kwargs).before("retry", "fixture-model")
    with pytest.raises(RuntimeError, match="requires_reconciliation"):
        meter.after(identity, None)
    assert meter.read()["calls"][0]["status"] == "reconciliation_required"


def test_absolute_deadline_refuses_without_new_reservation(tmp_path):
    meter = Meter(tmp_path / "usage.json", maximum_cost_usd=15, maximum_requests=10, deadline_epoch=time.time() - 1)
    with pytest.raises(RuntimeError, match="authority_exhausted"):
        meter.before("design", "fixture-model")
    assert meter.read()["calls"] == []


def test_real_openai_resource_hook_bounds_output_and_disables_hidden_transport_retries(tmp_path, monkeypatch):
    """A local HTTP cassette exercises the SDK API surface; no provider call."""
    import types
    import httpx
    from openai import OpenAI
    from openai.resources.responses.responses import Responses, AsyncResponses
    from blueprint_pipeline.task_object_artisan_metering import install_metering
    for name in ("pipeline", "pipeline.agents"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    cost = types.ModuleType("pipeline.cost")
    cost.pricing_for_provider_model = lambda *_: {"input_uncached": 10, "input_cached": 1, "input_cache_write": 12.5, "output": 50}
    cost.calculate_cost = lambda usage, _: types.SimpleNamespace(total_cost=(usage["prompt_tokens"] * 10 + usage["candidates_tokens"] * 50) / 1_000_000)
    monkeypatch.setitem(sys.modules, "pipeline.cost", cost)
    codec = types.ModuleType("pipeline.llm.openai_codec")
    codec.extract_usage = lambda response: {"prompt_tokens": response.usage.input_tokens, "candidates_tokens": response.usage.output_tokens}
    monkeypatch.setitem(sys.modules, "pipeline.llm.openai_codec", codec)
    design = types.ModuleType("pipeline.agents.design")
    design._generate_image = lambda *_args, **_kwargs: (True, {"accounting": {"budget_cost_usd": 0.1}})
    monkeypatch.setitem(sys.modules, "pipeline.agents.design", design)
    for resource in (Responses, AsyncResponses):
        for name in ("create", "compact"):
            monkeypatch.setattr(resource, name, getattr(resource, name))
    seen = []
    def transport(request):
        payload = json.loads(request.content)
        seen.append((request.url.path, payload))
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 100})
        return httpx.Response(200, json={"id": "resp_fixture", "object": "response", "created_at": 0,
            "status": "completed", "model": "fixture-model", "output": [], "usage": {
                "input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
                "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}})
    meter = Meter(tmp_path / "usage.json", maximum_cost_usd=0.003, maximum_requests=2, deadline_epoch=time.time() + 30)
    install_metering(meter)
    with OpenAI(api_key="synthetic-never-sent", http_client=httpx.Client(transport=httpx.MockTransport(transport))) as client:
        client.responses.create(model="fixture-model", input="small prompt")
        assert client.max_retries == 0
    assert seen[-1][1]["max_output_tokens"] == 35
    assert meter.read()["calls"][0]["cost_usd"] == pytest.approx(0.0015)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux provider process boundary")
def test_controller_deadline_terminates_child_processes(tmp_path):
    pid_file = tmp_path / "child.json"
    source = "import subprocess,sys,time,json; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); open(sys.argv[1],'w').write(json.dumps(p.pid)); time.sleep(30)"
    with pytest.raises(subprocess.TimeoutExpired):
        run_controller([sys.executable, "-c", source, str(pid_file)], cwd=tmp_path, env=dict(os.environ), timeout=0.3)
    pid = json.loads(pid_file.read_text())
    status = Path(f"/proc/{pid}/stat")
    assert not status.exists() or status.read_text().rpartition(")")[2].split()[0] == "Z"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux kernel sandbox")
def test_kernel_confines_generated_writes_and_hides_credentials(tmp_path):
    from blueprint_pipeline.asset_authoring_sandbox import SandboxedAssetRunner, AssetSandboxError
    root, assets = tmp_path / "workspace", tmp_path / "workspace/assets"
    assets.mkdir(parents=True)
    for name in ("tmp", "xdg", "cache"):
        (root / name).mkdir()
    observed = root / "observed.txt"
    observed.write_text("immutable observed bytes")
    credential = tmp_path / "credential.txt"
    credential.write_text("synthetic credential marker")
    runner = SandboxedAssetRunner(read_roots=[Path(sys.base_prefix)], write_root=root,
        write_paths=[assets, *(root / p for p in ("tmp", "xdg", "cache"))])
    try:
        runner.preflight()
    except AssetSandboxError:
        receipt = json.loads((tmp_path / "workspace.sandbox_preflight.json").read_text())
        assert receipt["status"] == "blocked" and receipt["unsandboxed_execution_allowed"] is False
        pytest.skip("Host kernel provides neither network namespaces nor Landlock; execution refused")
    program = '''import pathlib,sys,socket
for path in sys.argv[1:3]:
    try: pathlib.Path(path).write_text('forbidden')
    except PermissionError: pass
    else: raise RuntimeError('protected write permitted')
try: pathlib.Path(sys.argv[3]).read_text()
except PermissionError: pass
else: raise RuntimeError('credential readable')
try: socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except PermissionError: pass
else: raise RuntimeError('network permitted')
pathlib.Path(sys.argv[4]).write_text('allowed')
'''
    result = runner([sys.executable, "-S", "-c", program, str(observed), str(root / "model.py"),
                     str(credential), str(assets / "mesh.txt")], timeout=10)
    assert result.returncode == 0, result.stderr
    assert observed.read_text() == "immutable observed bytes"
    assert (assets / "mesh.txt").read_text() == "allowed"
