"""One bounded authoring session in which the agent repairs its own model.

Today's authoring runs the model one turn at a time and lets Python decide the
sequence: propose a Blender program, execute it, and if the appearance review
fails, call the model again with a feedback string. Two attempts, both
hardcoded. An ordinary modelling error -- a wrong axis, a missing boolean, a
material the renderer rejects -- costs a whole scripted round trip, and any
error nobody scripted a repair for ends the attempt.

This runs the same work as a single bounded session instead. The agent is
given the capture stills, the brief and the nominal dimensions once, plus one
tool: propose a candidate program, and receive back what our own executor and
validators actually reported. It reads the real error and revises, inside one
context, until a candidate passes or the session's bound is reached.

Three properties make that safe to hand a model:

* **The agent proposes; our validators dispose.** Acceptance is
  ``AttemptOutcome.accepted``, computed by the caller's executor from our
  renderer and reviewers. Nothing the model emits can declare success, and the
  session returns the accepted candidate rather than the model's own summary.
* **The bound is declared before the first call.** Turn count, per-turn output
  and tool-output bytes are fixed up front, so the invoker reserves the whole
  session's worst-case cost before it starts, exactly as it does for a single
  call. A session cannot outspend the cap by iterating.
* **Every turn is evidence.** Each proposal, the exact report returned to the
  model, and the deciding outcome are retained, so an accepted asset can be
  explained and a refused one can be debugged without re-running the model.

The session is opt-in and off unless its turn budget is configured, so the
scripted path stays the default until this is proven on a real object.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json

TURNS_ENV = "BLUEPRINT_ASSET_AUTHORING_SESSION_MAX_TURNS"
SESSION_SCHEMA = "task_object_authoring_session.v1"
TOOL_ID = "author_and_validate_candidate"
# Two turns is the scripted path's behaviour and proves nothing new; beyond
# eight, a session that has not converged is failing for a reason more turns
# will not fix, and each turn is reserved at the full per-turn ceiling.
MIN_TURNS = 3
MAX_TURNS = 8
# The report handed back to the model. Large enough for a validator's message
# and a measurement table, small enough that the growth bound stays provable.
MAX_REPORT_BYTES = 8_000
DEFAULT_TOOL_TIMEOUT_SECONDS = 900.0


class AuthoringSessionError(RuntimeError):
    """The session could not be bounded, executed or evidenced safely."""


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise AuthoringSessionError("authoring_session_" + code)


@dataclass(frozen=True)
class AttemptOutcome:
    """What our executor and validators found for one candidate program.

    ``accepted`` is the only thing that ends the session successfully, and the
    caller computes it. ``report`` is what the model is allowed to read back:
    it must carry the actual failure text, because a session that cannot see
    why a candidate failed cannot repair it.
    """

    accepted: bool
    report: Mapping[str, Any]
    artifacts: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionTurn:
    index: int
    program_digest: str
    accepted: bool
    report: Mapping[str, Any]


def authoring_session_turns(environment: Mapping[str, str]) -> int:
    """The configured turn budget, or 0 when the scripted path should run."""
    raw = str(environment.get(TURNS_ENV) or "").strip()
    if not raw:
        return 0
    try:
        turns = int(raw)
    except ValueError as exc:
        raise AuthoringSessionError("authoring_session_turn_budget_invalid") from exc
    _require(MIN_TURNS <= turns <= MAX_TURNS, "turn_budget_out_of_range")
    return turns


def _bounded_report(outcome: AttemptOutcome) -> dict[str, Any]:
    """Exactly what the model may read back, truncated to the declared bound."""
    _require(isinstance(outcome, AttemptOutcome), "outcome_invalid")
    _require(isinstance(outcome.report, Mapping), "outcome_report_invalid")
    report = {"accepted": bool(outcome.accepted), **{str(k): v for k, v in outcome.report.items()}}
    report.pop("artifacts", None)
    encoded = canonical_json(report)
    if len(encoded.encode()) > MAX_REPORT_BYTES:
        # Keep the failure text over the measurements: the text is what the
        # next turn needs, and a silently truncated document would be unparseable.
        text = str(report.get("failure") or report.get("message") or "")
        report = {"accepted": bool(outcome.accepted), "truncated": True,
                  "failure": text[: MAX_REPORT_BYTES // 2]}
    return report


def build_candidate_tool(
    *,
    execute_attempt: Callable[[str, int], AttemptOutcome],
    turns: Sequence[SessionTurn],
    max_turns: int,
    timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
) -> Any:
    """Bind the one tool the session gets: author a candidate, read the verdict.

    The tool runs the caller's executor -- the same renderer, validators and
    sandbox the scripted path uses -- and returns only a bounded structured
    report. It never returns artifact paths or source bytes, and it refuses
    once the session's turn budget is spent, so the bound holds even if the
    model keeps calling.
    """
    from .task_evaluation_supervisor.tools import RegisteredToolBinding

    def invoke(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        program = payload.get("program")
        _require(isinstance(program, str) and program.strip(), "candidate_program_invalid")
        _require(len(turns) < max_turns, "turn_budget_exhausted")
        index = len(turns) + 1
        outcome = execute_attempt(program, index)
        report = _bounded_report(outcome)
        turns.append(SessionTurn(index=index, accepted=bool(outcome.accepted),
                                 program_digest=canonical_digest({"program": program}),
                                 report=report))
        return report

    return RegisteredToolBinding(
        tool_id=TOOL_ID,
        description=(
            "Author one candidate asset from a complete Blender program and return "
            "what the renderer, geometry readback and appearance review reported. "
            "Revise and call again when a candidate is not accepted. Acceptance is "
            "decided by those checks, not by you."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["program", "rationale"],
            "properties": {
                "program": {"type": "string",
                            "description": "The complete Blender program for this candidate."},
                "rationale": {"type": "string",
                              "description": "What you changed since the last candidate and why."},
            },
        },
        timeout_seconds=float(timeout_seconds),
        invoke=invoke,
    )


def run_authoring_session(
    *,
    invoke_session: Callable[[Any, int], Any],
    execute_attempt: Callable[[str, int], AttemptOutcome],
    max_turns: int,
    output_root: str | Path,
    tool_timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run one bounded authoring session and seal its evidence.

    ``invoke_session`` receives the bound tool and the turn budget and performs
    the actual model call through the budgeted invoker, which reserves the
    whole session's worst-case cost before the first turn. The session's
    verdict is taken from the executor's outcomes, never from the model's
    reply, so a model that claims success without an accepted candidate is a
    refusal.
    """
    _require(MIN_TURNS <= int(max_turns) <= MAX_TURNS, "turn_budget_out_of_range")
    root = Path(output_root)
    _require(root.is_dir() and not root.is_symlink(), "output_root_invalid")
    turns: list[SessionTurn] = []
    binding = build_candidate_tool(execute_attempt=execute_attempt, turns=turns,
                                   max_turns=int(max_turns), timeout_seconds=tool_timeout_seconds)
    invocation_failure = None
    try:
        invoke_session(binding, int(max_turns))
    except Exception as exc:  # noqa: BLE001 - recorded as a refusal, never a success
        # A session that ran out of turns still produced candidates; keep them
        # and let the accepted-candidate test below decide the verdict.
        invocation_failure = type(exc).__name__ + ":" + str(exc)[:400]
    accepted = [turn for turn in turns if turn.accepted]
    _require(len(accepted) <= 1 or all(t.index >= accepted[0].index for t in accepted), "turn_order_invalid")
    record = {
        "schema_version": SESSION_SCHEMA,
        "status": "accepted" if accepted else "blocked",
        "max_turns": int(max_turns),
        "turns_used": len(turns),
        "accepted_turn_index": accepted[0].index if accepted else None,
        "accepted_program_digest": accepted[0].program_digest if accepted else None,
        "turns": [{"index": t.index, "program_digest": t.program_digest,
                   "accepted": t.accepted, "report": dict(t.report)} for t in turns],
        "invocation_failure": invocation_failure,
        "acceptance_decided_by": "blueprint_validators",
        "model_declared_success_honoured": False,
    }
    if not accepted:
        record["blockers"] = ["authoring_session_no_accepted_candidate"] + (
            [invocation_failure] if invocation_failure else [])
    record["session_digest"] = canonical_digest(record, digest_field="session_digest")
    (root / "authoring_session.v1.json").write_text(canonical_json(record) + "\n")
    return record


def session_initial_multimodal_tokens(*, frames: int, per_frame_tokens: int = 1_600,
                                      prompt_tokens: int = 4_000) -> int:
    """A conservative ceiling for the stills handed to the session.

    Image tokenization is provider dependent, so the invoker refuses a
    multi-turn multimodal call unless the caller declares what its payload
    costs. High-detail stills are well under this per-frame figure at the sizes
    authoring sends; over-declaring reserves more and can only refuse early.
    """
    _require(0 < int(frames) <= 16 and int(per_frame_tokens) > 0, "frame_count_invalid")
    return int(frames) * int(per_frame_tokens) + int(prompt_tokens)


__all__ = [
    "MAX_REPORT_BYTES", "MAX_TURNS", "MIN_TURNS", "TOOL_ID", "TURNS_ENV",
    "AttemptOutcome", "AuthoringSessionError", "SessionTurn", "authoring_session_turns",
    "build_candidate_tool", "run_authoring_session", "session_initial_multimodal_tokens",
]
