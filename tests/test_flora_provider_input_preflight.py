"""Exercise the worker's lazy corrected-reference imports without host fallback."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from blueprint_pipeline import flora_episode_benchmark as benchmark
from blueprint_pipeline import flora_provider_runtime as worker
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.episode_interpretation import materialize_episode_interpretation_rights
from blueprint_pipeline.flora_episode_interpretation import FloraEpisodeInterpreter
from blueprint_pipeline.native_task_arena_policy_canary_bundle import stage_policy_contract_dependencies
from tests.test_flora_episode_benchmark import _corrected_source


# These are image-supplied dependencies on the input-validation import path.
# The child may resolve only these names; its sys.path never gains site-packages.
_BASE_PACKAGES = (
    "PIL", "annotated_types", "dotenv", "numpy", "packaging", "pydantic",
    "pydantic_core", "typing_extensions", "typing_inspection", "yaml",
)
_ISOLATE_BASE_DEPENDENCIES = r'''
import importlib.abc, importlib.machinery, json, sys
allowed = json.loads(sys.argv.pop(2))
assert "rfc8785" not in allowed
assert not any("site-packages" in path for path in sys.path)
class ImageDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in allowed:
            return importlib.machinery.PathFinder.find_spec(fullname, [allowed[fullname]])
sys.meta_path.insert(0, ImageDependencies())
'''


@pytest.mark.slow
def test_worker_preflight_validates_corrected_inputs_with_only_bundled_canonicalizer(tmp_path):
    source_path, evidence, original, _, correction_path, profile = _corrected_source(
        tmp_path, use_ingest=True)
    original_bytes = source_path.read_bytes()
    profile.update(execution_site="external_provider", compute_provider_id="vast")
    profile["runtime_config"].update(model_path="/opt/blueprint/flora/model",
        source_root="/opt/blueprint/flora/source", tensor_parallel_size=2)
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    prepared = tmp_path / "prepared"
    dataset = benchmark.prepare_flora_benchmark(session_result_path=source_path,
        evidence_root=evidence, profile=profile, output_dir=prepared,
        corrected_reference_path=correction_path)
    assert dataset["selected_episode_count"] == 20
    assert dataset["reference_kind"] == "corrected_retained_deterministic_simulator_score"
    interpreter = FloraEpisodeInterpreter.from_profile(profile)
    _, requests = benchmark.load_flora_benchmark(prepared / "manifest.json", profile)
    for request in requests:
        materialize_episode_interpretation_rights(
            episode_id=request.episode_id, input_bundle_digest=request.input_receipt["input_bundle_digest"],
            identity=interpreter.identity, allowed_artifact_roles=interpreter.disclosed_artifact_roles(request),
            external_disclosure_authorized=True, accepted_by="hermetic-fixture-owner",
            accepted_on="2026-10-20T00:00:00Z", authority_reference="hermetic-input-validation-only",
            source_rights_admission_digest="sha256:" + "a" * 64,
            output_path=prepared / "rights" / (request.input_receipt["input_bundle_digest"][7:] + ".json"))

    # Match the bundle layout. Only candidate Python source and the canonical
    # four-file dependency package enter this temporary source tree.
    source = tmp_path / "provider_runtime/source/src"
    package = Path(worker.__file__).resolve().parent
    for path in package.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        target = source / "blueprint_pipeline" / path.relative_to(package)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    dependency = stage_policy_contract_dependencies(source)
    assert dependency["distribution"] == "rfc8785" and dependency["version"] == "0.1.4"
    allowed = {}
    for name in _BASE_PACKAGES:
        spec = importlib.util.find_spec(name)
        assert spec is not None and spec.origin is not None
        origin = Path(spec.origin).resolve()
        allowed[name] = str(origin.parent.parent if spec.submodule_search_locations else origin.parent)

    def preflight(output):
        return subprocess.run([sys.executable, "-I", "-S", "-B", "-c",
            _ISOLATE_BASE_DEPENDENCIES + worker._INPUT_PREFLIGHT,
            str(source), json.dumps(allowed), "preflight",
            "--manifest", str(prepared / "manifest.json"), "--profile", str(prepared / "profile.json"),
            "--rights-root", str(prepared / "rights"), "--output-dir", str(output)],
            cwd=prepared, capture_output=True, text=True, timeout=45, check=False)

    output = tmp_path / "verified"
    completed = preflight(output)
    assert completed.returncode == 0, completed.stderr
    records = list((output / "input_preflights").glob("*.json"))
    assert len(records) == 1
    result = json.loads(records[0].read_text())
    assert result["status"] == "inputs_verified" and result["blockers"] == []
    assert result["eligible_episode_count"] == result["selected_episode_count"] == 20
    assert result["inference_performed"] is result["runtime_execution_verified"] is False
    assert result["dataset_digest"] == dataset["dataset_digest"]
    assert result["preflight_digest"] == canonical_digest(result, digest_field="preflight_digest")
    assert not (output / "execution_attempt.json").exists()
    assert source_path.read_bytes() == original_bytes
    assert json.loads(source_path.read_text()) == original

    # Removing only the bundled dependency must reproduce the paid failure in
    # the actual corrected-reference path, even when the host has rfc8785.
    shutil.rmtree(source / "rfc8785")
    missing = preflight(tmp_path / "missing-canonicalizer")
    assert missing.returncode != 0
    assert "cross_runtime_canonical_json" in missing.stderr
    assert "No module named 'rfc8785'" in missing.stderr
