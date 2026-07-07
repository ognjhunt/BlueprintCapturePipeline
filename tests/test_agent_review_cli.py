from __future__ import annotations

import pytest

from blueprint_pipeline import agent_review_cli as cli
from blueprint_pipeline.agent_runtime.openai_phase2 import OpenAIPhase2Config


def _run_main_capturing_args(monkeypatch: pytest.MonkeyPatch, argv: list[str]):
    """Drive the real cli.main parser, capturing the resolved phase-2 config.

    The provider-launch path (run_agent_review) is stubbed so the test stays
    offline: it makes no codex/openai network calls and touches no filesystem.
    """

    captured: dict[str, object] = {}

    def _fake_run_agent_review(*, openai_phase2_config, **kwargs):
        captured["config"] = openai_phase2_config
        captured["provider_name"] = kwargs["provider_name"]
        return {
            "provider": kwargs["provider_name"],
            "readiness_state": "not_ready_yet",
            "final_memo_path": "memo.md",
            "final_bundle_path": "bundle.json",
        }

    monkeypatch.setattr(cli, "run_agent_review", _fake_run_agent_review)
    exit_code = cli.main(argv)
    return exit_code, captured.get("config"), captured.get("provider_name")


@pytest.mark.parametrize("mode", ["disabled", "codex_cli", "sdk", "auto"])
def test_openai_phase2_mode_choices_accept_all_supported_modes(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    # Every mode honored by build_openai_skill_runner / OpenAIPhase2Config must
    # parse without an argparse error (SystemExit) through the real CLI parser.
    exit_code, _config, _provider_name = _run_main_capturing_args(
        monkeypatch,
        ["--capture-root", "/tmp/x", "--provider", "openai", "--openai-phase2-mode", mode],
    )
    assert exit_code == 0


def test_sdk_mode_reaches_openai_phase2_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _exit_code, config, _provider_name = _run_main_capturing_args(
        monkeypatch,
        ["--capture-root", "/tmp/x", "--provider", "openai", "--openai-phase2-mode", "sdk"],
    )
    assert isinstance(config, OpenAIPhase2Config)
    assert config.mode == "sdk"
    assert config.normalized_mode() == "sdk"


def test_auto_mode_reaches_openai_phase2_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _exit_code, config, _provider_name = _run_main_capturing_args(
        monkeypatch,
        ["--capture-root", "/tmp/x", "--provider", "openai", "--openai-phase2-mode", "auto"],
    )
    assert isinstance(config, OpenAIPhase2Config)
    assert config.mode == "auto"
    assert config.normalized_mode() == "auto"


def test_invalid_phase2_mode_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # An unsupported mode must still be rejected by argparse with SystemExit.
    monkeypatch.setattr(cli, "run_agent_review", lambda **_kwargs: {})
    with pytest.raises(SystemExit):
        cli.main(
            ["--capture-root", "/tmp/x", "--provider", "openai", "--openai-phase2-mode", "bogus"]
        )


def test_local_provider_cli_is_no_llm_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    exit_code, config, provider_name = _run_main_capturing_args(
        monkeypatch,
        ["--capture-root", "/tmp/x", "--provider", "local"],
    )
    assert exit_code == 0
    assert config is None
    assert provider_name == "local"
