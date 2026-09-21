"""ADP-009D/day-21: selected requests stay separate through robot placement."""
from pathlib import Path
import json

import pytest

from blueprint_pipeline import task_evaluation_configured_controls_autostart as autostart
from blueprint_pipeline import task_evaluation_configured_controls_continuation_provisioning as provisioning
from tests.test_task_evaluation_configured_controls_continuation_provisioning import _provision, COMMIT
from tests.test_task_evaluation_configured_controls_autostart import _intent


def test_two_team_requests_install_distinct_exact_intents(tmp_path):
    installed = []
    for run_id in ("one", "two"):
        result, _ = _provision(tmp_path / run_id, evaluation_run_id=run_id)
        intent = json.loads(Path(result["intent_path"]).read_text())
        assert intent["evaluation_run_id"] == run_id
        row = provisioning.install_intent_into_registry(
            intent_path=result["intent_path"], intent_root=tmp_path / "registry",
            expected_production_commit=COMMIT, service_group=None,
        )
        assert Path(row["registry_path"]).read_bytes() == Path(result["intent_path"]).read_bytes()
        installed.append(row)
    assert installed[0]["registry_path"] != installed[1]["registry_path"]
    assert len(list((tmp_path / "registry").glob("*.json"))) == 2


def test_selected_request_cannot_be_staged_as_site_preparation_or_another_run(tmp_path):
    path, intent = _intent(tmp_path, evaluation_run_id="selected-run")
    kwargs = dict(source_path=path, expected_production_commit=intent["expected_production_commit"],
        team_namespace=intent["team_namespace"], scene_id=intent["scene_id"], task_id=intent["task_id"],
        output_path=tmp_path / "staged" / "intent.json")
    for wrong in (None, "another-run"):
        with pytest.raises(autostart.TaskEvaluationConfiguredControlsAutostartError, match="identity_mismatch"):
            autostart.stage_configured_controls_autostart_intent(**kwargs, evaluation_run_id=wrong)
        assert not Path(kwargs["output_path"]).exists()
    autostart.stage_configured_controls_autostart_intent(**kwargs, evaluation_run_id="selected-run")
    assert Path(kwargs["output_path"]).read_bytes() == path.read_bytes()


@pytest.mark.parametrize("run_id", ["", "../other", "x" * 193, 3])
def test_invalid_evaluation_scope_refuses_before_provider_inspection(tmp_path, run_id):
    def forbidden():
        raise AssertionError("provider inspection must not happen")
    with pytest.raises(ValueError, match="evaluation_run_id_invalid"):
        _provision(tmp_path, evaluation_run_id=run_id, provider_zero_collector=forbidden)


def test_provisioning_preserves_evaluation_authority(tmp_path):
    authority = {"evaluation_run_id":"selected-run", "source_launch_id":"source-launch",
        "source_profile_digest":"sha256:"+"a"*64, "configured_scene_revision_digest":"sha256:"+"b"*64,
        "scene_intent_digest":"sha256:"+"c"*64}
    result, _ = _provision(tmp_path, evaluation_run_id="selected-run", evaluation_authority=authority)
    intent = autostart.validate_configured_controls_autostart_intent(json.loads(Path(result["intent_path"]).read_text()))
    assert intent["evaluation_authority"] == authority
    with pytest.raises(ValueError, match="team_evaluation_authority_invalid"):
        _provision(tmp_path/'different', evaluation_run_id="another-run", evaluation_authority=authority)
