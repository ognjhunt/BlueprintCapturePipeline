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


def test_spend_reconciliation_walks_past_runs_that_allocated_nothing() -> None:
    """A non-allocating run is not a prior paid attempt.

    Reconciliation binds a real positive provider instance id -- it raises
    `prior_provider_instance_id_invalid` without one -- so a run that ended
    before allocating (`vast_instance_ids: []`, no billing row, $0.00) cannot
    be reconciled against and must not be chosen as the spend predecessor.
    Arena r16 ended exactly this way, and indexing [0] on its empty id list
    crashed the chain with IndexError before it could reach a GPU.

    Step 0 still seals the immediate predecessor, so skipping a run here
    proves it empty rather than ignoring it.
    """

    text = _text(LAUNCH)
    assert "vast_instance_ids" in text
    assert "allocated nothing" in text, "the walk must skip non-allocating runs"
    # the candidate list must span every tag, not just $PREV, or the walk
    # cannot reach past the immediate predecessor
    assert "_ALLRUNS" in text
    assert "JOBSPEND" in text and "ZEROSPEND" in text, (
        "terminal result, teardown, and provider zero must come from the same "
        "attempt"
    )
    assert "['vast_instance_ids'][0]" not in text, (
        "indexing the id list unguarded crashes on a run that allocated nothing"
    )


def test_authority_chains_off_the_same_attempt_the_reconciliation_describes() -> None:
    """The issuer matches the prior result against a reconciliation entry.

    It raises `prior_terminal_attempt_reconciliation_match_invalid` when they
    disagree, so the prior authority, terminal result, provider zero, and
    spend reconciliation must all describe ONE attempt. Once the spend walk
    skips a non-allocating run, chaining the authority off the immediate
    predecessor instead would describe a different attempt than the
    reconciliation and fail.

    Step 0 still seals the immediate predecessor, which is why it may differ
    from the spend predecessor without anything going unproven.
    """

    text = _text(LAUNCH)
    step0 = text.find("== 0. predecessor provider zero")
    step4 = text.find("== 4. attempt authority")
    assert step0 != -1 and step4 != -1

    seal = text[step0:text.find("== 1.")]
    assert "JOBPREV" in seal, "step 0 must seal the IMMEDIATE predecessor"

    authority = text[step4:text.find("== 5.")]
    for token in ("PSPEND", "JOBSPEND", "ZEROSPEND"):
        assert token in authority, (
            f"step 4 must chain off the spend predecessor ({token} missing)"
        )
    assert "JOBPREV" not in authority, (
        "chaining the authority off a predecessor that never allocated "
        "contradicts the reconciliation"
    )


def test_chain_refuses_a_packet_that_disagrees_with_deployed_constants() -> None:
    """The bundle rebuilds from deployed code; the packet is carried forward.

    That asymmetry is silent. Any fix to a value that lives in PACKET content
    never reaches the runtime, and the run looks entirely normal while
    executing the predecessor's plan.

    r19 paid for it: PR #786 raised the servo limits, deployed cleanly, and the
    run still executed 0.03/0.20 with joint travel identical to r17 to three
    decimals, because the packet was hardlinked from r18. A GPU run was spent
    discovering that a merged, deployed fix was inert.

    So the chain compares what the staged packet actually says against the
    deployed constants and refuses rather than running a stale plan.
    """

    text = _text(LAUNCH)
    guard = text.find("staged packet agrees with deployed control constants")
    stage = text.find("== 2. staged packet")
    bundle = text.find("== 3. construction bundle")
    assert guard != -1, "the chain must verify the packet against deployed code"
    assert stage < guard < bundle, "the check belongs between staging and build"

    for token in ("MAX_JOINT_DELTA_RAD", "MAX_JOINT_SETPOINT_LEAD_RAD"):
        assert token in text, f"the guard must compare {token}"
    assert "STALE PACKET" in text, "a stale packet must say so plainly"
    assert "SystemExit(1)" in text, "the guard must fail closed, not warn"
    # the guard imports from the deployed tree, so it needs the source on path
    assert "PYTHONPATH=$CP/src $PY - " in text, (
        "the guard must run with the deployed source importable"
    )


def test_chain_uses_the_dependency_complete_runtime_source_packet() -> None:
    """The original c3e8b79a packet was mutated to an incomplete 152 MB copy.

    The dependency-complete packet is the immutable 4.4 GB source bundle that
    already passed the production import/render preflight. Both the bundle and
    live profile must bind that same receipt or a fast deploy can fail before
    Isaac starts for reasons unrelated to the change under test.
    """

    text = _text(LAUNCH)
    declaration = "RUNTIME_SOURCE_PACKET_RECEIPT=${RUNTIME_SOURCE_PACKET_RECEIPT:-"
    assert declaration in text
    assert "native-task-runtime-source-c3e8b79a-dependency-complete-62ebe98e" in text
    assert text.count("$RUNTIME_SOURCE_PACKET_RECEIPT") == 2, (
        "bundle construction and live profile must consume one source receipt"
    )


def test_chain_carries_the_predecessor_machine_avoidlist_forward() -> None:
    """A failed machine must not silently re-enter the next offer search."""

    text = _text(LAUNCH)
    predecessor = "AVOIDLIST=$JOBPREV/adp_arena_vast_machine_avoidlist.json"
    fallback = "AVOIDLIST=$E/arena-launch-r5/machine_avoidlist.json"
    profile_flag = "--machine-avoidlist $AVOIDLIST"
    assert predecessor in text
    assert fallback in text
    assert profile_flag in text
    assert text.find(predecessor) < text.find(profile_flag)


def test_fire_waits_for_predecessor_reconciliation_before_submitting() -> None:
    """The site rejects a submit while a predecessor is unreconciled there.

    It answers HTTP 409, and the submit tool deliberately discards the error
    body -- an untrusted response cannot be launch evidence -- so the failure
    surfaces as a bare `webapp_http_error_409` with no cause attached. r16 and
    r19 both hit it, and both times the response was blind retrying on a timer
    rather than waiting for the condition that actually clears it.

    The reconciler timer runs about every two minutes and publishes whether
    anything is still in flight, so the fire step can wait on that instead of
    guessing.
    """

    text = _text(FIRE)
    wait = text.find("wait for predecessor reconciliation")
    submit = text.find("submit_task_evaluation_launch_via_webapp.py")
    assert wait != -1, "the fire step must wait for predecessor reconciliation"
    assert wait < submit, "the wait must happen before the submission"
    assert "processing_count" in text and "launches" in text, (
        "the wait must read the reconciler's in-flight fields"
    )
    # the wait must be bounded -- a hung reconciler cannot block forever
    assert "seq 1 20" in text, "the reconciliation wait must be bounded"


def test_execute_gate_is_still_armed_before_the_reconciliation_wait() -> None:
    """Adding the wait must not push arming after submission.

    The dispatcher fires on queue-write and blocks at retry-0 without a
    matching EXECUTE_ID, so the ordering that already cost one run must hold.
    """

    text = _text(FIRE)
    arm = text.find("BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE_ID")
    wait = text.find("wait for predecessor reconciliation")
    submit = text.find("submit_task_evaluation_launch_via_webapp.py")
    assert arm < wait < submit, "arm the execute gate, then wait, then submit"


def test_the_reconciliation_wait_does_not_claim_to_fix_the_409() -> None:
    """r20 ran the wait, reported reconciled, and was still refused.

    A plain retry about two minutes later succeeded, as it did for r19. So
    reconciler state is not what gates the 409; the evidence points at a
    server-side cooldown we cannot observe, because the submit tool discards
    the error body -- correctly, since an untrusted response cannot be launch
    evidence.

    The wait is worth keeping (submitting while a predecessor is genuinely in
    flight is wrong regardless, and it costs seconds) but a comment claiming
    it fixes the 409 would send the next person to the wrong place, and would
    justify deleting the retry that actually gets a run out.
    """

    text = _text(FIRE)
    assert "does not clear a 409 by itself" in text, (
        "the block must not imply it fixes the 409"
    )
    assert "HONEST STATUS" in text
    assert "cooldown" in text, "record where the evidence actually points"


def test_fire_retries_only_the_exact_provider_zero_409_once() -> None:
    """A server cooldown must not require a human or create a GPU retry loop."""

    text = _text(FIRE)
    assert text.count("submit_once") == 3  # definition, first call, one retry
    assert '"webapp_http_error_409"' in text
    assert '"provider_mutation_performed_by_this_tool": false' in text
    assert "seq 1 4" in text
    assert "webapp_submit_output.${CUR}.first_409.log" in text
    assert "--request $REQ" in text
    assert "Any other failure remains terminal" in text
