"""A scene reaches its first configuration before controls can continue it.

The configured-controls continuation binds a trajectory plan, and that plan is
only authored against a published configured revision.  Requiring the
continuation intent *before* admitting the configuration is therefore an
ordering the first run of any scene can never satisfy.

Admitting a configuration without one is not fail-open: the progression worker
skips every profile that carries no continuation intent, so controls still
cannot start until one is authorized.  An intent that is present but malformed
or bound elsewhere stays a refusal.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> Any:
    """Import a repository script by path, the way its lane consumers do."""

    path = REPOSITORY_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lane_module() -> Any:
    return _load_script("prepare_paid_lane_launch")


def _scene_configuration_step(lane_module: Any, step_id: str) -> Any:
    for step in lane_module.LANES["task_evaluation_scene_configuration"]:
        if step.step_id == step_id:
            return step
    raise AssertionError(f"scene configuration lane has no {step_id} step")


def test_intent_step_declares_the_context_it_exists_to_consume(
    lane_module: Any,
) -> None:
    """The continuation step is conditional on a provisioned intent."""

    step = _scene_configuration_step(
        lane_module, "configured_controls_autostart_intent"
    )

    assert step.requires_context == (
        "configured_controls_autostart_intent_source",
    )


def test_step_is_skipped_when_no_continuation_is_provisioned(
    lane_module: Any,
) -> None:
    """An empty continuation context removes the step from the lane run."""

    step = _scene_configuration_step(
        lane_module, "configured_controls_autostart_intent"
    )

    assert lane_module.step_is_active(step, {}) is False
    assert (
        lane_module.step_is_active(
            step, {"configured_controls_autostart_intent_source": ""}
        )
        is False
    )
    assert (
        lane_module.step_is_active(
            step,
            {"configured_controls_autostart_intent_source": "/etc/blueprint/i.json"},
        )
        is True
    )


def test_unconditional_steps_stay_unconditional(lane_module: Any) -> None:
    """Only steps that declare a requirement may ever be skipped."""

    step = _scene_configuration_step(lane_module, "live_profile")

    assert step.requires_context == ()
    assert lane_module.step_is_active(step, {}) is True


def test_profile_binds_the_continuation_only_when_one_exists(
    lane_module: Any,
) -> None:
    """The profile omits the flag rather than naming an unwritten path.

    Passing ``--configured-controls-autostart-intent`` with a path the skipped
    step never produced would make the profile builder fail on a file that was
    correctly never created.
    """

    step = _scene_configuration_step(lane_module, "live_profile")

    assert "--configured-controls-autostart-intent" not in step.argv
    assert (
        "--configured-controls-autostart-intent",
        "configured_controls_autostart_intent_artifacts",
    ) in step.repeated_argv
    assert lane_module._repeated_values("") == ()
    assert lane_module._repeated_values(None) == ()
    assert lane_module._repeated_values("/etc/blueprint/i.json") == (
        "/etc/blueprint/i.json",
    )


def test_validator_accepts_a_context_without_a_continuation(
    lane_module: Any,
) -> None:
    """A skipped step's placeholders are not reported as a missing context.

    The validator, the plan, and the executor must agree about which steps are
    part of this run; if only the validator disagreed, a configuration that
    runs fine would be refused before its first command.
    """

    context = {
        name: "x"
        for step in lane_module.LANES["task_evaluation_scene_configuration"]
        for name in lane_module.step_placeholders(step)
    }
    for name, _key in (
        pair
        for step in lane_module.LANES["task_evaluation_scene_configuration"]
        for pair in step.exports
    ):
        context.pop(name, None)
    context["configured_controls_autostart_intent_source"] = ""

    lane_module.validate_lane_context(
        "task_evaluation_scene_configuration", context
    )


def test_validator_still_refuses_a_genuinely_incomplete_context(
    lane_module: Any,
) -> None:
    """Skipping conditional steps must not soften the completeness check."""

    with pytest.raises(lane_module.PaidLaneLaunchPreparationError) as excinfo:
        lane_module.validate_lane_context(
            "task_evaluation_scene_configuration",
            {"configured_controls_autostart_intent_source": ""},
        )

    assert "paid_lane_context_incomplete" in str(excinfo.value)


def test_activation_names_the_artifact_the_lane_actually_writes(
    lane_module: Any, tmp_path: Path
) -> None:
    """The context value and the step's output must stay the same file.

    The profile now learns the intent path from the activation context instead
    of the lane's own argv, so nothing inside the lane would notice if the two
    drifted apart -- the flag would simply point at a file no step wrote.
    """

    from blueprint_pipeline import task_evaluation_launch_activation_worker

    step = _scene_configuration_step(
        lane_module, "configured_controls_autostart_intent"
    )
    activation_root = tmp_path / "activation"
    rendered = step.produces.format(set_root=str(activation_root / "launch-set"))

    source = Path(
        task_evaluation_launch_activation_worker.__file__
    ).read_text(encoding="utf-8")
    assert (
        "task_evaluation_configured_controls_autostart_intent.v1.json" in source
    )
    assert rendered == str(
        activation_root
        / "launch-set"
        / "task_evaluation_configured_controls_autostart_intent.v1.json"
    )
