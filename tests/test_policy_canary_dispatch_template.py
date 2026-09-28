import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.policy_canary_dispatch_template import resolve_execution_template
from blueprint_pipeline.task_evaluation_policy_canary_scene_setup import PolicyCanarySetupError


def _handoff(tmp_path, *, directory="team-eval-one", run="policy-one", commit="a" * 40):
    root = tmp_path / "task-evaluation-configured-controls" / "website-scene" / directory
    root.mkdir(parents=True)
    value = {
        "schema_version": "task_evaluation_policy_canary_handoff_progression.v1",
        "status": "canary_launch_submitted", "run_id": run,
        "source_launch_id": "website-scene", "expected_production_commit": commit,
        "evaluation_authority": {"scene_intent_digest": "sha256:" + "b" * 64},
        "profile_materialization_input_path": str(root / "policy-canary-presubmission" /
            "task_evaluation_policy_canary_profile_materialization_input.v1.json"),
    }
    value["progression_digest"] = canonical_digest(value, digest_field="progression_digest")
    path = root / "policy_canary_handoff_progression.json"
    path.write_text(json.dumps(value))
    return path


def _resolve(tmp_path, **overrides):
    envelope = {"capture_session_id": "website-scene", "activation_id": "policy-one-activation",
                "source_commit": "a" * 40, "scene_intent_digest": "sha256:" + "b" * 64}
    envelope.update(overrides)
    return resolve_execution_template(queue_root=tmp_path / "task-evaluation-policy-canary-dispatches",
        envelope=envelope, legacy_template=tmp_path / "old-scene-template.json")


def test_selects_this_evaluation_not_global_template_or_another_run(tmp_path):
    selected = _handoff(tmp_path)
    _handoff(tmp_path, directory="other-evaluation", run="policy-two")
    _handoff(tmp_path, directory="old-release", commit="c" * 40)
    assert _resolve(tmp_path) == selected.parent / "policy-canary-presubmission" / (
        "task_evaluation_policy_canary_execution_setup_template.v1.json")


@pytest.mark.parametrize("mutation", ["digest", "owner", "path"])
def test_rejects_tampered_or_foreign_handoff(tmp_path, mutation):
    path = _handoff(tmp_path)
    value = json.loads(path.read_text())
    if mutation == "digest":
        value["progression_digest"] = "sha256:" + "0" * 64
    elif mutation == "owner":
        value["evaluation_authority"]["scene_intent_digest"] = "sha256:" + "c" * 64
    else:
        value["profile_materialization_input_path"] = "/another-scene/wrapper.json"
    if mutation != "digest":
        value["progression_digest"] = canonical_digest(value, digest_field="progression_digest")
    path.write_text(json.dumps(value))
    with pytest.raises(PolicyCanarySetupError):
        _resolve(tmp_path)


def test_missing_or_duplicate_handoff_never_falls_back_to_global_scene(tmp_path):
    with pytest.raises(PolicyCanarySetupError, match="missing_or_ambiguous"):
        _resolve(tmp_path)
    _handoff(tmp_path)
    _handoff(tmp_path, directory="duplicate")
    with pytest.raises(PolicyCanarySetupError, match="missing_or_ambiguous"):
        _resolve(tmp_path)


def test_legacy_nonowner_dispatch_keeps_explicit_template(tmp_path):
    assert _resolve(tmp_path, scene_intent_digest=None) == tmp_path / "old-scene-template.json"


def test_refuses_source_path_traversal(tmp_path):
    with pytest.raises(PolicyCanarySetupError, match="source_invalid"):
        _resolve(tmp_path, capture_session_id="../elsewhere")
