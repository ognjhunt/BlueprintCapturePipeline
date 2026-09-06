"""Exercise real argparse and public materialization before scientific execution."""
import json
from pathlib import Path

import pytest

import blueprint_pipeline.task_evaluation_sam31_prefix_adoption as adoption
from tests.test_sam31_prefix_adoption import OLD, NEW, prefix as prefix, write


class ScientificBoundaryReached(Exception):
    pass


def _arguments(prefix, tmp_path, monkeypatch):
    value, old_plan, old_profile, _ = prefix
    request = json.loads(Path(value["original_parent_envelope"]["path"]).read_text())["request"]
    parent = Path(value["original_parent_envelope"]["path"])
    parent.rename(parent.with_name(request["preparation_id"] + "-" + value["original_parent_request_digest"][7:] + ".json"))
    execution = tmp_path / "executions" / value["original_parent_request_digest"][7:]
    execution.mkdir()
    for row in value["phase_records"]:
        source = Path(row["execution_receipt"]["path"]).parent
        source.rename(execution / source.name)
    zero = tmp_path / "zero.json"
    write(zero, {"provider": "vast", "status": "observed", "api_confirmed": True, "name_prefix": "",
                 "live_resource_count": 0, "resources": [], "http": 200, "observed_at_epoch": 1000.})
    monkeypatch.setattr(adoption.time, "time", lambda: 1001.)
    return dict(source_plan_path=value["source_plan"]["path"], source_profile_path=value["source_profile"]["path"],
        parent_request_digest=value["original_parent_request_digest"], through_phase="calibrated_views",
        current_host_inputs=old_plan["host_inputs"],
        current_provider_profile_path=old_profile["artifact_references"]["sam31_provider_profile"]["path"],
        current_repo_root=str(tmp_path), expected_source_commit=NEW, provider_zero_path=str(zero), output_path=None,
        approved_roots=(str(tmp_path),), queue_root=str(tmp_path / "queue"), parent_queue_root=str(tmp_path / "parents"),
        execution_root=str(tmp_path / "executions"))


def _argv(args):
    aliases = {"source_plan_path": "source-plan", "source_profile_path": "source-profile",
               "current_provider_profile_path": "current-sam31-provider-profile", "expected_source_commit": "source-commit",
               "provider_zero_path": "provider-zero"}
    argv = ["--check-only"]
    for key, value in args.items():
        if key == "output_path":
            continue
        if key == "current_host_inputs":
            for name, row in value.items():
                argv += ["--current-" + name.replace("_", "-"), row["path"]]
        elif key == "approved_roots":
            for root in value:
                argv += ["--approved-root", str(root)]
        else:
            argv += ["--" + aliases.get(key, key.replace("_", "-")), str(value)]
    return argv


@pytest.mark.parametrize("route", ["cli", "python_strings", "python_paths"])
def test_cli_and_public_api_normalize_real_paths_before_hashing(prefix, tmp_path, monkeypatch, route):
    args = _arguments(prefix, tmp_path, monkeypatch)
    calls = []
    def scientific_boundary(outcome, artifacts, old_plan, current_repo, through_phase):
        assert old_plan["source_commit"] == OLD
        assert isinstance(current_repo, Path) and through_phase == "calibrated_views"
        assert outcome["status"] == "completed" and artifacts["standard_splat"]
        calls.append("all exact prefix paths resolved")
        raise ScientificBoundaryReached
    monkeypatch.setattr(adoption, "validate_render", scientific_boundary)
    # Run the real public materializer through SHA computation, exact queue
    # identity lookup and original parent validation. Only the following GPU
    # evidence validation is stopped; no fabricated completion is returned.
    with pytest.raises(ScientificBoundaryReached):
        if route == "cli":
            adoption.main(_argv(args))
        else:
            if route == "python_paths":
                for name in ("source_plan_path", "source_profile_path", "current_provider_profile_path",
                             "current_repo_root", "provider_zero_path"):
                    args[name] = Path(args[name])
            adoption.materialize_completed_prefix_adoption(**args)
    assert calls == ["all exact prefix paths resolved"]
    assert not list(tmp_path.glob("*adoption*.json"))


def test_cli_output_report_serializes_path_arguments(prefix, tmp_path, monkeypatch, capsys):
    args = _arguments(prefix, tmp_path, monkeypatch)
    argv = _argv(args)
    argv.remove("--check-only")
    output = tmp_path / "adoption-output.json"
    argv += ["--output", str(output)]
    def completed(**kwargs):
        assert isinstance(kwargs["source_plan_path"], Path)
        assert kwargs["output_path"] == output
        return {"status": "verified_completed_prefix", "adoption_digest": "sha256:" + "d" * 64}
    monkeypatch.setattr(adoption, "materialize_completed_prefix_adoption", completed)
    assert adoption.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["output"] == str(output)
