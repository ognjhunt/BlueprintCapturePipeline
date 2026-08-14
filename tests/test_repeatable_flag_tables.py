"""A repeatable flag has to survive being repeated.

Three scripts now share the house flag table, where one `Param` mapping builds
both the parser and the keyword arguments. Every one of them declared its
repeatable flags as `Param(..., accumulate=True, default=())`, because `()` is
what the materializer downstream needs -- a `Sequence[int]` parameter that gets
`None` iterates `None`.

`argparse`'s `action="append"` appends to whatever default it is handed, and a
tuple has no `append`. So the declared default was correct for the callee and
fatal for the parser: the flag worked when omitted and raised `AttributeError`
the first time anyone passed it.

Nothing caught it because the existing contracts check the *table* -- that the
parameter is accumulating, typed `int`, and defaults to `()` -- and the one
parser test only ever omitted the flag. Both are true of code that cannot parse
its own flag.

The flags this hid behind are not incidental:

    --allow-active-instance     which instances may already be running when a
                                paid attempt starts; anything unlisted fails the
                                attempt closed
    --object-absent-reference   the object-absent receipts that feed the
                                ArtiFixer3D candidate-inputs receipt

So this rediscovers the scripts from source rather than listing them, and
inspects the parser each one actually builds rather than the table it declares.
The fix keeps `default=()` in the table and hands `argparse` a `None` it can
append to, which means a test asserting the table would pass on both the broken
and the fixed code.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"

#: The marker of the shared flag table: a `Param` dataclass with an
#: `accumulate` field, plus the `build_parser` it feeds.
PATTERN_MARKERS = ("accumulate: bool", "def build_parser(")


def _flag_table_scripts() -> list[Path]:
    return [
        path
        for path in sorted(SCRIPTS.glob("*.py"))
        if all(marker in path.read_text(encoding="utf-8") for marker in PATTERN_MARKERS)
    ]


FLAG_TABLE_SCRIPTS = _flag_table_scripts()


def _load(path: Path):
    name = f"flag_table_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # `dataclass` resolves annotations through `sys.modules[cls.__module__]`,
    # so a file-loaded module has to be registered before it is executed.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _append_actions(parser: argparse.ArgumentParser) -> list[tuple[str, argparse.Action]]:
    """Every appending action in a parser and in each of its subparsers."""

    found: list[tuple[str, argparse.Action]] = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                found.extend((f"{name} {flag}", found_action) for flag, found_action in _append_actions(sub))
        elif isinstance(action, argparse._AppendAction):
            found.append((action.option_strings[0] if action.option_strings else action.dest, action))
    return found


def test_the_flag_table_scripts_are_discoverable() -> None:
    """A scan that matched nothing would make the checks below vacuous."""

    assert len(FLAG_TABLE_SCRIPTS) >= 3, (
        f"only {len(FLAG_TABLE_SCRIPTS)} scripts matched the shared flag table; "
        "the scan is broken"
    )


@pytest.mark.parametrize("path", FLAG_TABLE_SCRIPTS, ids=lambda p: p.stem)
def test_every_repeatable_flag_starts_from_something_appendable(path: Path) -> None:
    """`action="append"` mutates its default in place, so it cannot be a tuple."""

    parser = _load(path).build_parser()
    actions = _append_actions(parser)

    unusable = {
        flag: action.default
        for flag, action in actions
        if action.default is not None and not isinstance(action.default, list)
    }

    assert not unusable, (
        f"{path.name} gives argparse an unappendable default for {sorted(unusable)}. "
        "`action='append'` calls `.append` on it, so passing the flag once raises "
        "AttributeError. Hand argparse `None` and restore the declared default in "
        "`call_arguments`."
    )


@pytest.mark.parametrize("path", FLAG_TABLE_SCRIPTS, ids=lambda p: p.stem)
def test_every_repeatable_flag_can_actually_be_repeated(path: Path) -> None:
    """The end-to-end version: parse the flag twice and keep both values.

    Asserting on the default alone would still pass if a future `action=` swap
    broke accumulation, so this drives the action the way an operator does.
    """

    parser = _load(path).build_parser()
    actions = _append_actions(parser)
    assert actions, f"{path.name} declares no repeatable flag; the scan is broken"

    for flag, action in actions:
        namespace = argparse.Namespace()
        setattr(namespace, action.dest, action.default)
        values = [1, 2] if action.type is int else ["a", "b"]
        for value in values:
            action(parser, namespace, value, flag)

        assert list(getattr(namespace, action.dest)) == values, (
            f"{path.name} {flag} did not accumulate both values"
        )
