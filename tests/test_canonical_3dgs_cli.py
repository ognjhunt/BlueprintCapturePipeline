from __future__ import annotations

from blueprint_pipeline import canonical_3dgs_cli


def test_unified_cli_dispatches_exact_subcommand_arguments(monkeypatch) -> None:
    observed: list[str] = []

    def fake(arguments):
        observed.extend(arguments)
        return 7

    monkeypatch.setitem(canonical_3dgs_cli._COMMANDS, "prepare", fake)
    assert canonical_3dgs_cli.main(["prepare", "--capture-root", "bundle"]) == 7
    assert observed == ["--capture-root", "bundle"]


def test_unified_cli_help_and_unknown_command_fail_closed(capsys) -> None:
    assert canonical_3dgs_cli.main(["--help"]) == 0
    assert "commands:" in capsys.readouterr().out
    assert canonical_3dgs_cli.main(["unknown"]) == 2
    assert "unknown canonical 3DGS command" in capsys.readouterr().err
