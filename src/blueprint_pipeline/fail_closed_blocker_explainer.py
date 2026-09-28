"""Name the predicates behind an anonymous fail-closed blocker.

The package has over a thousand validators of the shape::

    if (a or b or ... or z):
        raise SomeError("something_invalid")

and the mirror image ``_require(a and b and c, "something_invalid")``.  Each is
correct, and each reports the symptom rather than which predicate was true, so
a production run that trips one costs a redeploy-and-resubmit cycle to learn one
fact.  On 2026-09-05 a SAM tracking child was refused with
``sam31_gpu_canary_request_configuration_invalid``, one of forty predicates, and
the cause (a provider profile bound to a commit from another scene's run three
weeks earlier) took an hour of host archaeology to name.

This module answers that question at the moment the exception is raised, from
the frame that raised it.  It finds the boolean statement at the raising line,
splits its top-level ``or`` (or ``and``) chain, evaluates every predicate against
the frame's own locals, and names the ones that decided the outcome.  Predicates
are the validator's own expressions, so evaluating them again reads what the
validator already read; anything that fails to evaluate is reported, never
raised.  The output carries predicate *source text*, never values, so it can be
recorded in a blocker string without leaking what the validator looked at.
"""

from __future__ import annotations

import ast
import linecache
import types
from collections.abc import Mapping
from typing import Any

MAX_PREDICATE_CHARS = 140
MAX_FIRED = 8
MAX_ANNOTATION_CHARS = 700
_STATEMENT_FRAMES = 4


def _flatten(expression: ast.expr) -> tuple[str, list[ast.expr]]:
    """Return the top-level boolean operator ("or"/"and"/"single") and its operands."""

    if isinstance(expression, ast.BoolOp):
        operator = "or" if isinstance(expression.op, ast.Or) else "and"
        return operator, list(expression.values)
    return "single", [expression]


def _module_tree(filename: str) -> ast.Module | None:
    lines = linecache.getlines(filename)
    if not lines:
        return None
    try:
        return ast.parse("".join(lines), filename=filename)
    except SyntaxError:
        return None


def _boolean_statement(tree: ast.Module, line: int) -> tuple[str, ast.expr] | None:
    """The boolean expression that decided the statement at ``line``.

    A ``raise`` inside an ``if`` body yields that ``if``'s test (innermost wins).
    A call whose first argument is a boolean expression (the ``_require(cond,
    code)`` idiom) yields that argument.
    """

    innermost: ast.If | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            for statement in node.body:
                start = statement.lineno
                end = getattr(statement, "end_lineno", start) or start
                if start <= line <= end and (innermost is None or node.lineno > innermost.lineno):
                    innermost = node
    if innermost is not None:
        return "if", innermost.test
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Expr, ast.Assign, ast.Return)):
            continue
        start = node.lineno
        end = getattr(node, "end_lineno", start) or start
        if not start <= line <= end:
            continue
        value = node.value
        if isinstance(value, ast.Call) and value.args:
            first = value.args[0]
            if isinstance(first, (ast.BoolOp, ast.Compare, ast.UnaryOp, ast.Call)):
                return "require", first
    return None


def _source(node: ast.expr) -> str:
    text = " ".join(ast.unparse(node).split())
    return text if len(text) <= MAX_PREDICATE_CHARS else text[: MAX_PREDICATE_CHARS - 1] + "…"


def _evaluate(node: ast.expr, frame: types.FrameType) -> tuple[bool | None, str | None]:
    try:
        code = compile(ast.Expression(body=node), frame.f_code.co_filename, "eval")
        # Locals are merged into the globals so a comprehension inside the
        # predicate (its own scope) still resolves the validator's names.
        scope = {**frame.f_globals, **frame.f_locals}
        return bool(eval(code, scope)), None  # noqa: S307 - the validator's own expression, its own frame
    except Exception as exc:  # noqa: BLE001 - an unevaluable predicate is a finding, not a failure
        return None, f"{type(exc).__name__}"


def explain_frame(frame: types.FrameType, line: int) -> dict[str, Any] | None:
    """Explain the boolean statement at ``line`` in ``frame``; None when there is none."""

    tree = _module_tree(frame.f_code.co_filename)
    if tree is None:
        return None
    found = _boolean_statement(tree, line)
    if found is None:
        return None
    kind, expression = found
    operator, operands = _flatten(expression)
    # In an ``if`` the predicates that were true fired (an ``and`` chain only
    # raises when all are true); in a requirement the false ones fired.
    fired_when = kind == "if"
    fired: list[str] = []
    errors: list[str] = []
    for operand in operands:
        value, error = _evaluate(operand, frame)
        if error is not None:
            errors.append(f"{_source(operand)} -> {error}")
        elif value is fired_when:
            fired.append(_source(operand))
    return {
        "file": frame.f_code.co_filename,
        "line": line,
        "function": frame.f_code.co_name,
        "kind": kind,
        "operator": operator,
        "predicates_total": len(operands),
        "fired": fired[:MAX_FIRED],
        "fired_total": len(fired),
        "evaluation_errors": errors[:MAX_FIRED],
    }


def explain_blocker(exc: BaseException) -> dict[str, Any]:
    """Walk the traceback from the raising frame outward and explain the first boolean statement found."""

    frames: list[tuple[types.FrameType, int]] = []
    traceback = exc.__traceback__
    while traceback is not None:
        frames.append((traceback.tb_frame, traceback.tb_lineno))
        traceback = traceback.tb_next
    explanations: list[dict[str, Any]] = []
    for frame, line in reversed(frames[-_STATEMENT_FRAMES:] if frames else []):
        explanation = explain_frame(frame, line)
        if explanation is not None:
            explanations.append(explanation)
    return {
        "schema_version": "fail_closed_blocker_explanation.v1",
        "blocker": str(exc),
        "exception_type": type(exc).__name__,
        "explanations": explanations,
    }


def fired_predicates(exc: BaseException) -> list[str]:
    """The most informative fired predicates: a chain over a lone ``not condition``.

    A ``_require(cond, code)`` helper raises from a single-predicate ``if``; the
    chain that decided the outcome sits one frame out, so multi-predicate
    statements win over single ones, innermost first within each group.
    """

    # A single-predicate statement (``_require(value, code)``, ``if not ok:``)
    # says nothing the blocker code does not already say, so only chains count.
    for row in explain_blocker(exc)["explanations"]:
        if row["fired"] and row["predicates_total"] > 1:
            return list(row["fired"])
    return []


def annotate_blocker(code: str, exc: BaseException) -> str:
    """Append the fired predicates to a blocker code: ``code:predicates=a | b``.

    The annotation is source text only and bounded, so it is safe in a result
    record.  When nothing can be explained the code is returned unchanged.
    """

    try:
        fired = fired_predicates(exc)
    except Exception:  # noqa: BLE001 - explaining must never mask the original failure
        return code
    if not fired:
        return code
    annotation = f"{code}:predicates={' | '.join(fired)}"
    return annotation if len(annotation) <= MAX_ANNOTATION_CHARS else annotation[: MAX_ANNOTATION_CHARS - 1] + "…"


def explain_call(function: Any, /, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run ``function``; on failure return its explanation instead of raising.

    A look-ahead can call a stage's validator on the real artifacts before a
    submission and read exactly which predicate would refuse it.
    """

    try:
        function(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the exception is the subject
        return {"status": "refused", **explain_blocker(exc)}
    return {"status": "accepted"}


def annotate_mapping(blocker: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    """Attach a full explanation to a result mapping without changing its blocker code."""

    return {**blocker, "blocker_explanation": explain_blocker(exc)}


__all__ = [
    "annotate_blocker",
    "annotate_mapping",
    "explain_blocker",
    "explain_call",
    "explain_frame",
    "fired_predicates",
]
