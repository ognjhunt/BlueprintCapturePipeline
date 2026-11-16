"""Deployment receipts, not an operator's rewritten SHA, own scene release identity."""
import json
import pytest

from blueprint_pipeline import task_evaluation_scene_release_binding as binding
from tests.test_task_evaluation_public_scene_attempt_factory import context as context, write, ref


def test_release_automaterializes_and_reopens_exact_evidence(context):
    args, source = context
    original = json.loads(args["release_binding_path"].read_text())
    commit = original["source_commit"]
    root = args["release_binding_path"].parent
    deploy = json.loads(source["deploy_receipt"].read_text())
    deploy.update(status="deployed", release_path=original["repo_root"],
        scene_configuration_environment={**ref(source["release_environment"]), "credential_values_recorded": False})
    deploy["release_provenance"].update(ref(source["release_provenance"]))
    receipt = write(root / "deploys" / "current.json", deploy)
    config = dict(deployment_receipt_root=str(receipt.parent), release_binding_root=str(root / "bindings"),
        running_repo_root=original["repo_root"], runtime_publication_root=original["runtime_publication_root"])
    first = binding.resolve_release_binding(config, running_commit=commit)
    assert first["source_commit"] == commit
    assert binding.resolve_release_binding(config, running_commit=commit) == first
    # Rotating the host's mutable env after deployment cannot change this attempt.
    source["release_environment"].write_text("next release")
    assert binding.resolve_release_binding(config, running_commit=commit) == first
    # The retained deployment receipt itself remains mandatory, digest checked.
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="input_bytes_mismatch"):
        binding.resolve_release_binding(config, running_commit=commit)


def test_absent_current_deploy_does_not_reuse_previous_release(tmp_path):
    write(tmp_path / "old.json", {"status": "deployed", "source_commit": "a" * 40})
    config = dict(deployment_receipt_root=str(tmp_path), release_binding_root=str(tmp_path / "bindings"))
    with pytest.raises(ValueError, match="current_deployment_receipt_missing"):
        binding.resolve_release_binding(config, running_commit="b" * 40)


def test_unreadable_unrelated_history_does_not_block_current_release(context, monkeypatch):
    args, source = context
    original = json.loads(args["release_binding_path"].read_text())
    commit = original["source_commit"]
    root = args["release_binding_path"].parent
    deploy = json.loads(source["deploy_receipt"].read_text())
    deploy.update(status="deployed", release_path=original["repo_root"],
        scene_configuration_environment={**ref(source["release_environment"]), "credential_values_recorded": False})
    deploy["release_provenance"].update(ref(source["release_provenance"]))
    receipt = write(root / "deploys/current.json", deploy)
    historical = write(receipt.parent / "private-historical-provenance.json", {"source_commit": "b" * 40})
    actual_read = binding.read
    def read(path, **kwargs):
        if path == historical:
            raise ValueError("scene_configuration_submission_input_json_invalid") from PermissionError("private history")
        return actual_read(path, **kwargs)
    monkeypatch.setattr(binding, "read", read)
    config = dict(deployment_receipt_root=str(receipt.parent), release_binding_root=str(root / "bindings"),
        running_repo_root=original["repo_root"], runtime_publication_root=original["runtime_publication_root"])
    assert binding.resolve_release_binding(config, running_commit=commit)["source_commit"] == commit


def test_unreadable_current_receipt_never_falls_back_to_an_old_release(tmp_path, monkeypatch):
    private = write(tmp_path / "current.json", {"status": "deployed", "source_commit": "b" * 40})
    write(tmp_path / "old.json", {"status": "deployed", "source_commit": "a" * 40})
    actual_read = binding.read
    def read(path, **kwargs):
        if path == private:
            raise ValueError("unreadable") from PermissionError("current receipt")
        return actual_read(path, **kwargs)
    monkeypatch.setattr(binding, "read", read)
    with pytest.raises(ValueError, match="current_deployment_receipt_missing"):
        binding.resolve_release_binding({"deployment_receipt_root": str(tmp_path),
            "release_binding_root": str(tmp_path / "bindings")}, running_commit="b" * 40)


def _bound_release(context):
    args, source = context
    original = json.loads(args["release_binding_path"].read_text())
    commit = original["source_commit"]
    root = args["release_binding_path"].parent
    deploy = json.loads(source["deploy_receipt"].read_text())
    deploy.update(status="deployed", release_path=original["repo_root"],
        scene_configuration_environment={**ref(source["release_environment"]), "credential_values_recorded": False})
    deploy["release_provenance"].update(ref(source["release_provenance"]))
    receipt = write(root / "deploys" / "iteration_door.json", deploy)
    config = dict(deployment_receipt_root=str(receipt.parent), release_binding_root=str(root / "bindings"),
        running_repo_root=original["repo_root"], runtime_publication_root=original["runtime_publication_root"])
    first = binding.resolve_release_binding(config, running_commit=commit)
    return config, commit, receipt, deploy, source, first


def test_redeploying_the_same_commit_keeps_the_binding_and_progression_going(context):
    config, commit, receipt, deploy, _source, first = _bound_release(context)
    bound = (binding.safe_path(config["release_binding_root"]) / commit / "release.json").read_bytes()
    # The hand rollback of 2026-11-15: the same release deployed again rewrote
    # the receipt at the bound path with new timestamps.
    write(receipt, {**deploy, "stage_timings_seconds": {"release_activated": 1.5},
                    "redeployed_at": "2026-11-15T16:00:00Z"})
    assert binding.resolve_release_binding(config, running_commit=commit) == first
    assert (binding.safe_path(config["release_binding_root"]) / commit / "release.json").read_bytes() == bound


@pytest.mark.parametrize("change", ["other_commit", "not_deployed", "other_release_path", "provenance_bytes",
                                    "environment_bytes", "unreadable"])
def test_a_rewritten_receipt_that_is_not_the_same_release_is_refused(context, change):
    config, commit, receipt, deploy, source, _first = _bound_release(context)
    changed = json.loads(json.dumps(deploy))
    changed["redeployed_at"] = "2026-11-15T16:00:00Z"
    if change == "other_commit":
        changed["source_commit"] = "f" * 40
    elif change == "not_deployed":
        changed["status"] = "blocked"
    elif change == "other_release_path":
        changed["release_path"] = str(receipt.parent)
    elif change == "provenance_bytes":
        other = write(receipt.parent / "other-provenance.json", {"git_sha": commit, "status": "other"})
        changed["release_provenance"].update(ref(other))
    elif change == "environment_bytes":
        source["release_environment"].write_text("next release")
        changed["scene_configuration_environment"].update(ref(source["release_environment"]))
    if change == "unreadable":
        receipt.write_text("not json")
    else:
        write(receipt, changed)
    with pytest.raises(ValueError, match="input_bytes_mismatch"):
        binding.resolve_release_binding(config, running_commit=commit)
