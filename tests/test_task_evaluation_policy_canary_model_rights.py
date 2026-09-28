"""Current model-rights bindings never rewrite historical runtime evidence."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_policy_canary_model_rights as binding
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


@pytest.fixture
def source(tmp_path):
    root = Path(__file__).resolve().parents[1]
    template = root / "docs/arm_decision_proof_v1/manifests/scene839873_policy_canary_model_rights.v1.json"
    original = template.read_bytes()
    repo = tmp_path / "repo"
    repo.mkdir()
    for row in json.loads(original)["blueprint_adapter_code"]["modules"]:
        target = repo / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / row["path"]).read_bytes())
    for args in (["init", "-q"], ["add", "src"], ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "Exact source fixture"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    return {"template_path": template, "repo_root": repo, "source_commit": commit,
            "scene_id": "841757", "task_id": "book-to-tray", "output_path": tmp_path / "rights.json"}, original


def test_current_binding_preserves_terms_and_hashes_exact_git_source(source):
    kwargs, original = source
    result = binding.materialize_policy_canary_model_rights(**kwargs)
    historical = json.loads(original)
    assert result["rights_digest"] == canonical_digest(result, digest_field="rights_digest")
    assert result["candidates"] == historical["candidates"]
    assert result["historical_runtime_smoke"] == historical["historical_runtime_smoke"]
    assert result["source_template"]["rights_digest"] == historical["rights_digest"]
    assert result["source_commit"] == kwargs["source_commit"]
    assert result["scene_id"] == "841757"
    assert result["rights_reauthorization_performed"] is False
    assert result["current_scene_runtime_proof"] is False
    for row in result["blueprint_adapter_code"]["modules"]:
        payload = (kwargs["repo_root"] / row["path"]).read_bytes()
        assert row["sha256"] == "sha256:" + hashlib.sha256(payload).hexdigest()
        assert row["size_bytes"] == len(payload)
    assert kwargs["template_path"].read_bytes() == original
    assert binding.materialize_policy_canary_model_rights(**kwargs) == result


@pytest.mark.parametrize("drift", ["commit", "source", "template"])
def test_new_binding_rejects_drift_before_emitting_a_current_receipt(source, tmp_path, drift):
    kwargs, original = source
    if drift == "commit":
        kwargs["source_commit"] = "f" * 40
    elif drift == "source":
        row = json.loads(original)["blueprint_adapter_code"]["modules"][0]
        with (kwargs["repo_root"] / row["path"]).open("ab") as stream:
            stream.write(b"\n# uncommitted drift\n")
    else:
        template = json.loads(original)
        template["scene_promotion_permitted"] = True
        path = tmp_path / "changed-template.json"
        path.write_text(json.dumps(template))
        kwargs["template_path"] = path
    with pytest.raises(binding.PolicyCanaryModelRightsError):
        binding.materialize_policy_canary_model_rights(**kwargs)
    assert not kwargs["output_path"].exists()
