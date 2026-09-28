"""The host metadata/control boundary must not import the simulator stack."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from blueprint_pipeline import native_g1_team_vm_bootstrap as bootstrap

ROOT = Path(__file__).resolve().parents[1]


def test_host_hardware_preflight_does_not_import_allocator_for_driver_constant():
    # Fake hardware responses prove the import boundary, not actual VM/GPU health.
    code = bootstrap._PROBE + '''
from types import SimpleNamespace
from blueprint_pipeline import native_g1_team_vm_host as host
host.platform.system = lambda: 'Linux'
host.platform.machine = lambda: 'x86_64'
host.os.geteuid = lambda: 0
host.sys.version_info = (3, 12)
host.version = lambda name: {'numpy': '2.3.1', 'rfc8785': '0.1.4'}[name]
host._verified_local_image = lambda name: 'sha256:' + 'a' * 64
def response(command, **kwargs):
    if command[0] == 'docker':
        return SimpleNamespace(returncode=0, stdout='{"nvidia":{}}', stderr='')
    assert command[0] == 'nvidia-smi'
    return SimpleNamespace(returncode=0, stdout='0, 580.65.06', stderr='')
host.subprocess.run = response
result = host.preflight_g1_vm_host({'packet_digest': 'sha256:' + 'b' * 64,
    'request': {'policy_profile': {'delivery': {
    'mode': 'container', 'image_ref': 'example/policy@sha256:' + 'a' * 64}}}})
assert result['python_abi'] == 'cp312'
assert result['guest_gpu_inference_verified'] is False
assert 'blueprint_pipeline.vast_provider_adapter' not in sys.modules
'''
    child = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT / "src")],
                           capture_output=True, text=True, timeout=60, check=False)
    assert child.returncode == 0, child.stderr


def test_fresh_production_host_probe_denies_model_render_and_provider_imports():
    denied = '''
for name in ('torch', 'onnxruntime', 'pxr', 'isaaclab', 'isaacsim', 'PIL',
             'yaml', 'packaging', 'blueprint_pipeline.vast_provider_adapter'):
    assert name not in sys.modules
    try:
        importlib.import_module(name)
    except ImportError as exc:
        assert str(exc) == 'g1_host_import_forbidden:' + name
    else:
        raise AssertionError('host dependency denial was not enforced:' + name)
'''
    child = subprocess.run(
        [sys.executable, "-I", "-c", bootstrap._PROBE + denied, str(ROOT / "src")],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout)["sealed_module_origins_verified"] is True


@pytest.mark.parametrize("operation", ["metadata", "tensor"])
def test_fresh_host_metadata_does_not_hide_missing_tensor_runtime(operation):
    code = r'''
import importlib.abc, pathlib, sys
class DenyTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise ImportError('host_import_forbidden:torch')
sys.meta_path.insert(0, DenyTorch())
sys.path.insert(0, sys.argv[1])
from blueprint_pipeline import task_evaluation_g1_catalog as catalogue
from blueprint_pipeline.native_g1_official_sonic_target_bridge import WxyzRootDataView
assert catalogue.G1_EMBODIMENT_ID == 'unitree_g1_dex3_v1'
assert 'torch' not in sys.modules
if sys.argv[2] == 'tensor':
    try:
        WxyzRootDataView._tensor(object())
    except ImportError as exc:
        assert str(exc) == 'host_import_forbidden:torch'
    else:
        raise AssertionError('missing tensor runtime was hidden')
'''
    child = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT / "src"), operation],
                           capture_output=True, text=True, timeout=60, check=False)
    assert child.returncode == 0, child.stderr
