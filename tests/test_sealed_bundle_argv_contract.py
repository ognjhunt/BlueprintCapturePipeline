"""Weld the digest-pinned episode bundle's argv to the eval CLI, hermetically.

Attempts 066 and 067 each burned a GPU session on a failure this file now
catches for free: the sealed bundle's ``closed_loop_command`` predates
``--start-frame-evidence``, the #178 validator required evidence the argv never
delivered, and step 1 silently substituted DeterministicWalkToTargetPolicy —
whose token action then crashed FK conditioning. The bundle is digest-pinned
(sha256 e3274f14…), so its argv is frozen; the committed fixture is that exact
argv, and these tests fail on any CLI or wiring drift that would break it.
"""

from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.initial_policy_observation_contract import (
    resolve_start_frame_evidence_path,
)
from blueprint_pipeline.oscar_isaac_closed_loop_eval import build_arg_parser

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "kitchen_task_min"
    / "sealed_bundle_closed_loop_argv.json"
)


def _bundle_argv() -> list[str]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    command = payload["closed_loop_command"]
    # Strip the interpreter prefix: <python> -m blueprint_pipeline.oscar_isaac_closed_loop_eval
    assert command[1] == "-m" and command[2].endswith("oscar_isaac_closed_loop_eval")
    return [str(token) for token in command[3:]]


def test_sealed_bundle_argv_parses_with_the_current_cli() -> None:
    """Any flag rename or removal that strands the pinned bundle goes red here."""

    args = build_arg_parser().parse_args(_bundle_argv())
    assert args.groot_sonic_policy_server_url == "tcp://127.0.0.1:5550"
    assert args.start_frame == "/workspace/initial_policy_frame.png"


def test_sealed_bundle_argv_omits_evidence_so_the_canonical_fallback_must_exist(
    tmp_path: Path,
) -> None:
    """The frozen argv has no --start-frame-evidence; the eval must fall back.

    Without the fallback, a configured endpoint plus absent evidence reproduces
    the 066/067 silent-walk failure on every bundle-driven run, forever.
    """

    args = build_arg_parser().parse_args(_bundle_argv())
    assert args.start_frame_evidence is None

    canonical = tmp_path / "controller_fk_camera_projection_context.json"
    assert (
        resolve_start_frame_evidence_path(None, canonical_path=str(canonical)) is None
    ), "missing canonical file must resolve to None so the loop fails closed"

    canonical.write_text("{}", encoding="utf-8")
    assert resolve_start_frame_evidence_path(None, canonical_path=str(canonical)) == str(
        canonical
    )
    assert (
        resolve_start_frame_evidence_path("/explicit.json", canonical_path=str(canonical))
        == "/explicit.json"
    ), "explicit argv must always win over the canonical fallback"
