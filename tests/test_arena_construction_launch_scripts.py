from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LAUNCH = REPO / "scripts" / "arena_construction_launch_chain.sh"
FIRE = REPO / "scripts" / "arena_construction_fire.sh"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_launch_and_fire_scripts_are_committed_and_parse() -> None:
    """The arena launch sequence lived only on the control-plane host.

    Every step of it was learned from a failed paid run, so a host-only copy
    is a fix that stays local: rebuilding the host loses the ordering, and
    nothing reviews a change to it.
    """

    for script in (LAUNCH, FIRE):
        assert script.exists(), f"{script.name} is not in the repository"
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_profile_build_passes_a_revision() -> None:
    """Without --revision a retry at the same commit cannot publish.

    Published profiles are immutable and the profile id is keyed on the
    commit, but the profile body embeds per-attempt paths and a necessarily
    fresh one-shot attempt authority. So the same commit produces different
    bytes under the same id and the publisher refuses with
    `immutable_profile_conflict`, which is exactly what a retry after a
    transient provider failure needs to do. r16 hit this after r15's
    instance died provider-side.
    """

    text = _text(LAUNCH)
    assert "build_native_task_arena_live_profile.py" in text
    assert "--revision" in text, (
        "the profile builder must receive a revision or a same-commit retry "
        "cannot publish its profile"
    )


def test_execute_gate_is_armed_before_submission() -> None:
    """The dispatcher fires on queue-write and blocks at retry-0 without a
    matching EXECUTE_ID, so arming after submission loses the run."""

    text = _text(FIRE)
    arm = text.find("BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE_ID")
    submit = text.find("submit_task_evaluation_launch_via_webapp.py")
    assert arm != -1 and submit != -1
    assert arm < submit, "the execute gate must be armed before the submission"


def test_fire_script_never_embeds_a_secret_value() -> None:
    """The submit secret is read from Render into a mode-600 temp file."""

    text = _text(FIRE)
    assert "--secret-file" in text
    assert "mktemp" in text
    assert "rm -f" in text, "the temp secret file must be removed on exit"
