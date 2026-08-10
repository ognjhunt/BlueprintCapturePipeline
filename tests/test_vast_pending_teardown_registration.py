"""The teardown obligation must exist before the thing that needs tearing down."""

from __future__ import annotations

import ast
from pathlib import Path


from blueprint_pipeline.paid_lane_guard import (
    bind_pending_teardown_instance,
    open_pending_teardown,
)


ADAPTER = Path(__file__).resolve().parents[1] / "src/blueprint_pipeline/vast_provider_adapter.py"


def _line_of(pattern: str) -> int:
    for index, line in enumerate(ADAPTER.read_text(encoding="utf-8").splitlines(), 1):
        if pattern in line:
            return index
    raise AssertionError(f"{pattern!r} not found in the adapter")


def test_the_adapter_registers_a_pending_teardown():
    """It registered none at all, so the reaper's registry was always empty.

    rt21 created an instance, crashed on a full disk, and left it billing. The
    orphan reaper exists for exactly that, and found nothing - not because the
    record was stale but because this lane never wrote one.
    """

    source = ADAPTER.read_text(encoding="utf-8")

    assert "open_pending_teardown" in source
    assert "bind_pending_teardown_instance" in source


def test_the_record_is_opened_before_the_create_call():
    """After the create, a crash in between leaves an untracked instance.

    The window is small and rt21 landed in it. Opening first means the worst
    case is a record for an instance that was never created, which the reaper
    resolves harmlessly.
    """

    open_line = _line_of("pending_teardown_record = open_pending_teardown(")
    create_line = _line_of('path=f"/asks/{selected_offer[\'ask_contract_id\']}/",')

    assert open_line < create_line, (
        f"open_pending_teardown at line {open_line} must precede the create call "
        f"at line {create_line}"
    )


def test_the_instance_id_is_bound_once_it_is_known():
    bind_line = _line_of("bind_pending_teardown_instance(")
    id_line = _line_of("instance_id = _instance_id_from_create_response(create_response)")

    assert bind_line > id_line


def test_an_opened_record_is_visible_to_the_reaper_before_any_instance_exists(tmp_path):
    """A record with no instance id yet is still a record."""

    record = open_pending_teardown(
        provider="vast",
        lane="adp009d-franka-native-microcheck",
        run_id="rt-test",
        job_dir=tmp_path,
        registry_dir=tmp_path / "registry",
        max_age_seconds=3600,
    )

    assert Path(record["path"]).is_file()
    assert (tmp_path / "registry").is_dir()

    bound = bind_pending_teardown_instance(record["path"], "12345")

    assert str(bound.get("instance_id")) == "12345"


def test_registration_failure_never_blocks_a_launch(tmp_path):
    """A bookkeeping fault must not stop real work.

    The record is a safety net; refusing to fly because the net could not be
    hung would trade a small risk for a certain loss.
    """

    source = ADAPTER.read_text(encoding="utf-8")
    tree = ast.parse(source)

    guarded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        body = ast.unparse(node)
        if "open_pending_teardown" in body and "except" in body.lower():
            guarded = True
    assert guarded, "open_pending_teardown must be inside a try/except"
