"""Catalog contract resources survive a source-only or installed package."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = (
    'task_evaluation_policy_canary_setup.v1.schema.json',
    'rigid_task_success_contract.v1.schema.json',
    'articulated_task_success_contract.v1.schema.json',
)


@pytest.mark.parametrize('name', SCHEMAS)
def test_packaged_catalog_schema_matches_public_contract(name):
    resource = ROOT / 'src/blueprint_pipeline/_catalog_schemas' / name
    assert resource.read_bytes() == (ROOT / 'docs/schemas' / name).read_bytes()


@pytest.mark.slow
def test_catalog_schema_loaders_work_without_repository_docs(tmp_path):
    package = tmp_path / 'src/blueprint_pipeline'
    shutil.copytree(ROOT / 'src/blueprint_pipeline', package,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    assert not (tmp_path / 'docs').exists()
    script = '''import json,sys
from pathlib import Path
root=Path(sys.argv[1])
sys.path.insert(0,str(root/'src'))
def no_external_calls(event,args):
    if event.startswith('socket.') or event in {'subprocess.Popen','os.system','os.posix_spawn'}:
        raise RuntimeError('schema loading must remain local')
sys.addaudithook(no_external_calls)
from blueprint_pipeline import task_evaluation_policy_canary_setup as canary
from blueprint_pipeline import rigid_task_success_contract_schema as rigid
from blueprint_pipeline import articulated_task_success_contract_schema as articulated
for module,loader in ((canary,canary.policy_canary_setup_schema),
                      (rigid,rigid.rigid_task_success_contract_schema),
                      (articulated,articulated.articulated_task_success_contract_schema)):
    assert Path(module.__file__).resolve().is_relative_to(root/'src')
    assert module.SCHEMA_PATH.is_relative_to(root/'src/blueprint_pipeline/_catalog_schemas')
    assert isinstance(loader(),dict)
print(json.dumps({'schema_loaders_passed':3,'repository_docs_present':False}))
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(tmp_path)],
                            capture_output=True, text=True, timeout=20, check=False,
                            env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'})
    assert result.returncode == 0, result.stderr
    assert '"schema_loaders_passed": 3' in result.stdout
