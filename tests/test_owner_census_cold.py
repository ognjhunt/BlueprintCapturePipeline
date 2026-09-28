"""Explicit cold ordinary/default paths stay detached from opt-in authority."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_owner_consents.py
#   src/blueprint_pipeline/control_plane_lane_scratch_decisions.py
#   src/blueprint_pipeline/control_plane_reference_budget.py
#   scripts/lane_scratch_census.py
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_owner_census_budget import payloads


@pytest.mark.slow
def test_cold_none_validator_never_loads_opt_in_budget_door_or_provider():
    census, annotations = payloads()
    code = f"""import sys
from blueprint_pipeline.control_plane_lane_scratch_decisions import validate_census_annotations
result=validate_census_annotations({census!r},{annotations!r},now=1000,allowed_roots=('/work','/inputs'))
assert result['decision_count']==1
assert 'blueprint_pipeline.control_plane_reference_budget' not in sys.modules
assert 'blueprint_pipeline.control_plane_lane_owner_consents' not in sys.modules
assert not any(n.startswith(('operator_door','boto','google.cloud','openai','anthropic','agents')) for n in sys.modules)
"""
    done = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.slow
def test_cold_leaf_import_is_pure_and_installed_bridge_is_not_loaded():
    code = """import sys
import blueprint_pipeline.control_plane_lane_owner_consents
assert 'blueprint_pipeline.control_plane_reference_budget' not in sys.modules
assert not any(n.startswith(('operator_door','boto','google.cloud','openai','anthropic','agents')) for n in sys.modules)
"""
    done = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.slow
def test_cold_ordinary_validation_cli_keeps_opt_in_unloaded(tmp_path):
    census, annotations = payloads()
    a = tmp_path / "census"
    b = tmp_path / "annotations"
    a.write_bytes(census)
    b.write_bytes(annotations)
    source = Path(__file__).parents[1] / "scripts/lane_scratch_census.py"
    code = f"""import importlib.util,sys
spec=importlib.util.spec_from_file_location('ordinary_census',{str(source)!r})
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.time.time=lambda:1000
assert module.main(['--validate-census',{str(a)!r},'--annotations',{str(b)!r},'--work-root','/work','--inputs-root','/inputs'])==0
assert 'blueprint_pipeline.control_plane_reference_budget' not in sys.modules
assert 'blueprint_pipeline.control_plane_lane_owner_consents' not in sys.modules
"""
    done = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert done.returncode == 0, done.stderr
