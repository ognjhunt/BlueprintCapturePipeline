"""A test that skips for missing private bytes has to say so up front.

The hermetic CPU full lane runs `-m "not external_runtime and not
external_data"`, and its evidence builder blocks on *any* skip -- deliberately,
because a silently skipped test proves nothing. Twenty-nine tests reached a
private capture asset without carrying the marker, so on every runner they
skipped at runtime and the lane came back
`cpu_full_junit_skipped:29, status=blocked`. Eleven thousand seven hundred
fifty-two tests passed and the lane was red anyway, which is the same as having
no signal at all.

The marker already existed and already meant exactly this: *"requires retained
source bytes that are not installed on hermetic CI runners"*. Nothing needed
loosening -- the tests needed declaring.

This catches the next one here, in a second, rather than after a twenty-five
minute lane.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent

#: A skip reason naming one of these is a private-asset skip: capture bytes
#: that are never in a source checkout.
#:
#: Deliberately narrow. "ffmpeg not installed", "lpips runtime not installed",
#: and "splat renderer dependencies not installed" are *runtime* skips, and the
#: hermetic runner has all three -- a lane that started skipping those is a
#: misprovisioned runner and must keep blocking. `external_data` is documented
#: as retained source *bytes*, and this stays on that side of the line.
ABSENT_INPUT_PHRASES = (
    "not present locally",
    "not present in this checkout",
)


def _skip_reason_strings(node: ast.AST) -> list[str]:
    """Every literal reason a skip in this subtree gives."""

    reasons: list[str] = []
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        target = call.func
        name = getattr(target, "attr", None) or getattr(target, "id", None)
        if name not in {"skip", "skipif"}:
            continue
        for argument in call.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                reasons.append(argument.value)
        for keyword in call.keywords:
            if keyword.arg == "reason" and isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, str):
                    reasons.append(keyword.value.value)
    return reasons


def _has_external_data_marker(node: ast.FunctionDef, module: ast.Module) -> bool:
    for decorator in node.decorator_list:
        if "external_data" in ast.dump(decorator):
            return True
    # A module-level `pytestmark` covers every test in the file.
    for statement in module.body:
        if isinstance(statement, ast.Assign) and any(
            getattr(target, "id", None) == "pytestmark" for target in statement.targets
        ):
            if "external_data" in ast.dump(statement.value):
                return True
    return False


def _offenders() -> list[str]:
    offenders: list[str] = []
    for path in sorted(TESTS.glob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        if not any(phrase in source for phrase in ABSENT_INPUT_PHRASES):
            continue
        module = ast.parse(source)
        # Reasons given anywhere in the module, keyed by the helper or test that
        # gives them; a test calling such a helper inherits the skip.
        helper_names = {
            node.name
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and not node.name.startswith("test_")
            and any(
                phrase in reason
                for reason in _skip_reason_strings(node)
                for phrase in ABSENT_INPUT_PHRASES
            )
        }
        for node in ast.walk(module):
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            segment = ast.get_source_segment(source, node) or ""
            own = any(
                phrase in reason
                for reason in _skip_reason_strings(node)
                for phrase in ABSENT_INPUT_PHRASES
            )
            inherited = any(helper in segment for helper in helper_names)
            if (own or inherited) and not _has_external_data_marker(node, module):
                offenders.append(f"{path.name}::{node.name}")
    return offenders


def test_the_scan_can_still_find_private_asset_skips() -> None:
    """Guards the guard: a scan that matches nothing would pass vacuously."""

    matched = [
        path.name
        for path in TESTS.glob("test_*.py")
        if any(phrase in path.read_text(encoding="utf-8") for phrase in ABSENT_INPUT_PHRASES)
    ]
    assert matched, "no test declares a private-asset skip; the phrases must have changed"


def test_every_private_asset_skip_is_declared_external_data() -> None:
    offenders = _offenders()
    if offenders:
        pytest.fail(
            "these skip for bytes that are never in a source checkout but do not "
            "carry @pytest.mark.external_data, so they skip at runtime on the "
            "hermetic lane and block it on cpu_full_junit_skipped: "
            + ", ".join(offenders)
        )
