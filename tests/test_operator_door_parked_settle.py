"""What a parked deploy does, decided with real git against a real upstream."""

# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/parked_settle.py

from __future__ import annotations

import fcntl
import json
import os
import random as random_module
import subprocess
import sys
from pathlib import Path

import pytest

DOOR = Path(__file__).resolve().parents[1] / "deploy" / "operator-door"
sys.path.insert(0, str(DOOR))

from operator_door import parked_settle as settle  # noqa: E402

CARRIER = "20260923T120000Z-deploy-0000cccc"
OLDER = "20260923T115900Z-deploy-0000aaaa"
OLDEST = "20260923T115800Z-deploy-0000bbbb"


def _git(*args: str, cwd: Path) -> str:
    done = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                          cwd=cwd, capture_output=True, text=True, check=True)
    return done.stdout.strip()


@pytest.fixture()
def world(tmp_path: Path) -> dict[str, object]:
    """base - A - merge(F) - M3 - merge(G) - M4 on main.

    F branches from base and G from A, both merged. U is on an unmerged branch.
    """
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git("init", "--quiet", "--initial-branch=main", cwd=upstream)
    _git("commit", "--quiet", "--allow-empty", "-m", "base", cwd=upstream)
    base = _git("rev-parse", "HEAD", cwd=upstream)
    _git("switch", "--quiet", "-c", "feature", cwd=upstream)
    _git("commit", "--quiet", "--allow-empty", "-m", "F", cwd=upstream)
    feature = _git("rev-parse", "HEAD", cwd=upstream)
    _git("switch", "--quiet", "main", cwd=upstream)
    _git("commit", "--quiet", "--allow-empty", "-m", "A", cwd=upstream)
    a = _git("rev-parse", "HEAD", cwd=upstream)
    _git("merge", "--quiet", "--no-ff", "-m", "merge", "feature", cwd=upstream)
    merged_feature = _git("rev-parse", "HEAD", cwd=upstream)
    _git("commit", "--quiet", "--allow-empty", "-m", "M3", cwd=upstream)
    m3 = _git("rev-parse", "HEAD", cwd=upstream)
    _git("switch", "--quiet", "-c", "unmerged", cwd=upstream)
    _git("commit", "--quiet", "--allow-empty", "-m", "U", cwd=upstream)
    unmerged = _git("rev-parse", "HEAD", cwd=upstream)
    _git("switch", "--quiet", "-c", "second", a, cwd=upstream)
    _git("commit", "--quiet", "--allow-empty", "-m", "G", cwd=upstream)
    second = _git("rev-parse", "HEAD", cwd=upstream)
    _git("switch", "--quiet", "main", cwd=upstream)
    _git("merge", "--quiet", "--no-ff", "-m", "merge second", "second", cwd=upstream)
    m4 = _git("rev-parse", "HEAD", cwd=upstream)
    clone = tmp_path / "clone"
    _git("clone", "--quiet", "--no-checkout", str(upstream), str(clone), cwd=tmp_path)
    for name in ("parked", "results", "state/deploy-receipts", "releases"):
        (tmp_path / name).mkdir(parents=True)
    return {"root": tmp_path, "clone": clone, "upstream": upstream, "base": base, "F": feature, "A": a,
            "merged_F": merged_feature, "M3": m3, "G": second, "M4": m4, "U": unmerged}


def _active(world: dict[str, object], commit: str, *, proven: bool = False) -> None:
    root = Path(world["root"])  # type: ignore[arg-type]
    link = root / "active"
    link.unlink(missing_ok=True)
    link.symlink_to(root / "releases" / commit)
    if proven:
        receipt = root / "state/deploy-receipts" / f"iteration_{commit[:12]}_door.json"
        receipt.write_text(json.dumps({"status": "deployed", "source_commit": commit}))


def _park(world: dict[str, object], request_id: str, commit: str, *, active_at_park: str = "") -> None:
    root = Path(world["root"])  # type: ignore[arg-type]
    path = root / "parked" / f"{request_id}.json"
    path.write_text(json.dumps({"schema": settle.PARKED_SCHEMA, "id": request_id, "commit": commit,
                                "wait_for_idle": True, "parked_at_epoch": 1.0,
                                "active_release_at_park": active_at_park}))
    path.chmod(0o644)


def _settle(world: dict[str, object], *, carrier: str = CARRIER) -> tuple[int, str, str]:
    root = Path(world["root"])  # type: ignore[arg-type]
    return settle.settle(carrier=carrier, parked_dir=str(root / "parked"), results_dir=str(root / "results"),
                         clone=str(world["clone"]), active_link=str(root / "active"), state_root=str(root / "state"))


def _outcome(world: dict[str, object], request_id: str) -> dict | None:
    path = Path(world["root"]) / "results" / f"{request_id}.outcome.json"  # type: ignore[arg-type]
    return json.loads(path.read_text()) if path.exists() else None


def _parked(world: dict[str, object]) -> list[str]:
    return sorted(path.stem for path in (Path(world["root"]) / "parked").glob("*.json"))  # type: ignore[arg-type]


def _finished(world: dict[str, object], request_id: str, *, active: str | None = None) -> None:
    """What door-deploy.sh leaves when a deploy ends: its record gone, and a proven release if it worked."""
    (Path(world["root"]) / "parked" / f"{request_id}.json").unlink()  # type: ignore[arg-type]
    if active:
        _active(world, active, proven=True)


NOTHING = (settle.EXIT_NOTHING_TO_DEPLOY, "", "")
TRY_AGAIN = (settle.EXIT_TRY_AGAIN, "", "")


def test_a_newer_commit_requested_first_is_not_dropped_by_an_older_one_requested_later(world) -> None:
    """Two sessions ask in the order they happen to reach the door, not in commit order."""
    _active(world, world["base"], proven=True)
    _park(world, OLDER, world["M3"])
    _park(world, CARRIER, world["A"])

    assert _settle(world) == (settle.EXIT_DEPLOY, world["M3"], OLDER)


def test_a_request_a_deploy_covers_is_not_satisfied_until_a_receipt_proves_it(world) -> None:
    _active(world, world["base"], proven=True)
    _park(world, OLDEST, world["A"])
    _park(world, OLDER, world["F"])
    _park(world, CARRIER, world["M3"])

    assert _settle(world) == (settle.EXIT_DEPLOY, world["M3"], CARRIER)
    assert _outcome(world, OLDEST) is None and _outcome(world, OLDER) is None and _outcome(world, CARRIER) is None
    assert _parked(world) == sorted([OLDEST, OLDER, CARRIER])

    _finished(world, CARRIER, active=world["M3"])
    assert _settle(world, carrier=OLDER) == NOTHING
    for request_id in (OLDEST, OLDER):
        outcome = _outcome(world, request_id)
        assert outcome["status"] == "superseded" and outcome["exit_code"] == 0
        assert outcome["superseded_by_commit"] == world["M3"]
    assert _parked(world) == []


def test_a_failed_deploy_does_not_take_the_requests_it_would_have_covered_with_it(world) -> None:
    """The newest commit's CI went red. The older ones are fine, and each still gets its try."""
    _active(world, world["base"], proven=True)
    _park(world, OLDER, world["A"])
    _park(world, CARRIER, world["M3"])
    assert _settle(world) == (settle.EXIT_DEPLOY, world["M3"], CARRIER)

    _finished(world, CARRIER)  # failed: no receipt, the link did not move

    assert _outcome(world, OLDER) is None
    assert _settle(world, carrier=OLDER) == (settle.EXIT_DEPLOY, world["A"], OLDER)


def test_a_red_commit_someone_else_named_costs_only_that_request(world) -> None:
    """The order is reversed: the request that started has the good, older commit."""
    _active(world, world["base"], proven=True)
    _park(world, OLDER, world["M3"])
    _park(world, CARRIER, world["A"])
    assert _settle(world) == (settle.EXIT_DEPLOY, world["M3"], OLDER)

    _finished(world, OLDER)  # it failed for the request that named it

    assert _parked(world) == [CARRIER] and _outcome(world, CARRIER) is None
    assert _settle(world) == (settle.EXIT_DEPLOY, world["A"], CARRIER)


def test_the_request_whose_commit_goes_out_is_marked_so_a_kill_cannot_strand_it(world) -> None:
    root = Path(world["root"])
    _active(world, world["base"], proven=True)
    _park(world, OLDER, world["M3"])
    _park(world, CARRIER, world["A"])

    _settle(world)

    assert json.loads((root / "parked" / f"{OLDER}.json").read_text())["deploying_commit"] == world["M3"]
    assert "deploying_commit" not in json.loads((root / "parked" / f"{CARRIER}.json").read_text())
    assert not list((root / "parked").glob(".*"))


def test_main_can_hold_two_commits_neither_of_which_contains_the_other(world) -> None:
    """F and A are both on main. Deploying F after A would drop A, and the reverse too."""
    _active(world, world["base"], proven=True)
    _park(world, OLDER, world["F"])
    _park(world, CARRIER, world["A"])

    assert _settle(world) == (settle.EXIT_DEPLOY, world["A"], CARRIER)
    assert _outcome(world, OLDER) is None

    _finished(world, CARRIER, active=world["A"])
    assert _settle(world, carrier=OLDER) == NOTHING
    older = _outcome(world, OLDER)
    assert older["status"] == "failed" and older["code"] == "parked_commit_not_contained"
    assert older["active_release"] == world["A"]


def test_a_merged_branch_tip_behind_the_active_release_fails_no_one_who_can_deploy(world) -> None:
    """F does not contain A. M3 is ahead of A and contains F, whichever was asked for last."""
    _active(world, world["A"], proven=True)
    _park(world, OLDER, world["M3"])
    _park(world, CARRIER, world["F"])

    assert _settle(world) == (settle.EXIT_DEPLOY, world["M3"], OLDER)
    assert _outcome(world, OLDER) is None and _outcome(world, CARRIER) is None

    _finished(world, OLDER, active=world["M3"])
    assert _settle(world) == NOTHING
    assert _outcome(world, CARRIER)["status"] == "superseded"


def test_a_request_settled_now_leaves_the_others_for_the_deploy_that_can_still_help_them(world) -> None:
    """F cannot go out now, but M3 can and contains it. F must not fail because this run ended early."""
    _active(world, world["A"], proven=True)
    _park(world, OLDEST, world["F"])
    _park(world, OLDER, world["M3"])
    _park(world, CARRIER, world["A"], active_at_park=world["A"])

    assert _settle(world) == NOTHING
    assert _outcome(world, CARRIER)["status"] == "superseded"
    assert _outcome(world, OLDEST) is None and _outcome(world, OLDER) is None
    assert _settle(world, carrier=OLDER) == (settle.EXIT_DEPLOY, world["M3"], OLDER)


def test_a_request_that_cannot_go_out_lets_another_deploy_and_learns_the_result_after(world) -> None:
    """F is not inside A, and G is ahead of A without F: the deploy goes out, and F waits to hear."""
    _active(world, world["A"], proven=True)
    _park(world, OLDER, world["G"])
    _park(world, CARRIER, world["F"])

    assert _settle(world) == (settle.EXIT_DEPLOY, world["G"], OLDER)
    assert _outcome(world, CARRIER) is None

    _finished(world, OLDER, active=world["G"])
    assert _settle(world) == NOTHING
    assert _outcome(world, CARRIER)["code"] == "parked_commit_not_contained"


def test_the_deploy_that_goes_out_contains_the_starting_request_when_any_can(world) -> None:
    """G covers nothing else and M4 covers it all, so M4 goes out, for the request that named it."""
    _active(world, world["A"], proven=True)
    _park(world, OLDEST, world["M4"])
    _park(world, OLDER, world["M3"])
    _park(world, CARRIER, world["G"])

    assert _settle(world) == (settle.EXIT_DEPLOY, world["M4"], OLDEST)


def test_the_starting_requests_own_commit_beats_one_that_covers_more_but_not_it(world) -> None:
    """All three are ahead of A. M3 contains merged_F, so it covers two, and G covers only itself."""
    _active(world, world["A"], proven=True)
    _park(world, OLDEST, world["merged_F"])
    _park(world, OLDER, world["M3"])
    _park(world, CARRIER, world["G"])

    assert _settle(world) == (settle.EXIT_DEPLOY, world["G"], CARRIER)
    assert _outcome(world, OLDER) is None and OLDER in _parked(world)


def test_two_requests_for_one_commit_run_under_the_one_that_started(world) -> None:
    _active(world, world["A"], proven=True)
    _park(world, OLDER, world["M3"])
    _park(world, CARRIER, world["M3"])

    assert _settle(world) == (settle.EXIT_DEPLOY, world["M3"], CARRIER)
    assert _settle(world, carrier=OLDER) == (settle.EXIT_DEPLOY, world["M3"], OLDER)


def test_a_request_from_before_a_rollback_does_not_undo_it(world) -> None:
    """What runs is older than what ran when it was asked for, so someone rolled back."""
    _active(world, world["A"], proven=True)
    _park(world, OLDER, world["M3"], active_at_park=world["M3"])
    _park(world, CARRIER, world["M4"], active_at_park=world["M3"])

    assert _settle(world) == NOTHING
    for request_id in (OLDER, CARRIER):
        outcome = _outcome(world, request_id)
        assert outcome["status"] == "failed" and outcome["code"] == "parked_release_rolled_back"
        assert outcome["active_release"] == world["A"] and outcome["active_release_at_park"] == world["M3"]


def test_a_request_made_after_the_rollback_is_a_new_decision(world) -> None:
    _active(world, world["A"], proven=True)
    _park(world, CARRIER, world["M3"], active_at_park=world["A"])

    assert _settle(world) == (settle.EXIT_DEPLOY, world["M3"], CARRIER)


def test_an_older_parked_commit_not_on_main_fails_on_its_own(world) -> None:
    _git("fetch", "--quiet", "origin", "unmerged", cwd=world["clone"])
    _active(world, world["base"], proven=True)
    _park(world, OLDER, world["U"])
    _park(world, CARRIER, world["M3"])

    assert _settle(world) == (settle.EXIT_DEPLOY, world["M3"], CARRIER)
    assert _outcome(world, OLDER)["code"] == "commit_not_on_main"
    assert OLDER not in _parked(world)


def test_this_requests_own_commit_not_on_main_fails_alone_and_the_others_wait(world) -> None:
    _git("fetch", "--quiet", "origin", "unmerged", cwd=world["clone"])
    _active(world, world["base"], proven=True)
    _park(world, OLDER, world["M3"])
    _park(world, CARRIER, world["U"])

    assert _settle(world) == NOTHING
    assert _outcome(world, CARRIER)["code"] == "commit_not_on_main"
    assert _outcome(world, OLDER) is None
    assert _parked(world) == [OLDER]


def test_a_commit_merged_after_this_fetch_stays_parked_for_the_next_deploy(world) -> None:
    _active(world, world["base"], proven=True)
    _park(world, OLDER, "f" * 40)
    _park(world, CARRIER, world["A"])

    assert _settle(world) == (settle.EXIT_DEPLOY, world["A"], CARRIER)
    assert _outcome(world, OLDER) is None
    assert OLDER in _parked(world)


def test_a_failed_deploy_that_moved_the_link_is_retried_not_reported_as_success(world) -> None:
    """A deploy moves the release link before its last checks, and nothing moves it back."""
    _active(world, world["M3"], proven=False)
    _park(world, CARRIER, world["M3"])

    assert _settle(world) == (settle.EXIT_DEPLOY, world["M3"], CARRIER)
    assert _outcome(world, CARRIER) is None


def test_a_commit_a_proven_deploy_already_runs_does_not_deploy_again(world) -> None:
    _active(world, world["M3"], proven=True)
    _park(world, CARRIER, world["M3"])

    assert _settle(world) == NOTHING
    outcome = _outcome(world, CARRIER)
    assert outcome["status"] == "superseded" and outcome["exit_code"] == 0
    assert outcome["superseded_by_commit"] == world["M3"]
    assert _parked(world) == []


@pytest.mark.parametrize("fields", [{"status": "blocked"}, {"status": "deployed", "source_commit": "a" * 40}])
def test_a_blocked_receipt_or_one_for_another_commit_proves_nothing(world, fields: dict) -> None:
    root = Path(world["root"])
    _active(world, world["M3"])
    (root / "state/deploy-receipts" / f"iteration_{world['M3'][:12]}_door.json").write_text(
        json.dumps({"source_commit": world["M3"], **fields}))
    _park(world, CARRIER, world["M3"])

    assert _settle(world) == (settle.EXIT_DEPLOY, world["M3"], CARRIER)


def test_an_old_receipt_says_nothing_about_a_later_deploy_that_moved_the_link_and_died(world) -> None:
    root = Path(world["root"])
    _active(world, world["M3"], proven=True)
    receipt = root / "state/deploy-receipts" / f"iteration_{world['M3'][:12]}_door.json"
    os.utime(receipt, ns=(1_000_000_000, 1_000_000_000))
    _park(world, CARRIER, world["M3"])

    assert _settle(world) == (settle.EXIT_DEPLOY, world["M3"], CARRIER)


def test_a_production_receipt_proves_the_release_too(world) -> None:
    root = Path(world["root"])
    _active(world, world["M3"])
    (root / "state/deploy-receipts" / f"production_{world['M3'][:12]}_door.json").write_text(
        json.dumps({"status": "deployed", "source_commit": world["M3"]}))
    _park(world, CARRIER, world["M3"])

    assert _settle(world) == NOTHING
    assert _outcome(world, CARRIER)["status"] == "superseded"


@pytest.mark.parametrize("kind", ["fifo", "symlink", "oversize"])
def test_a_receipt_path_that_is_no_small_regular_file_proves_nothing_and_cannot_hang(world, kind: str) -> None:
    root = Path(world["root"])
    _active(world, world["M3"])
    path = root / "state/deploy-receipts" / f"iteration_{world['M3'][:12]}_door.json"
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "symlink":
        target = root / "elsewhere.json"
        target.write_text(json.dumps({"status": "deployed", "source_commit": world["M3"]}))
        path.symlink_to(target)
    else:
        path.write_text(json.dumps({"status": "deployed", "source_commit": world["M3"],
                                    "padding": "x" * settle.MAX_RECEIPT_BYTES}))
    _park(world, CARRIER, world["M3"])

    assert _settle(world) == (settle.EXIT_DEPLOY, world["M3"], CARRIER)


def test_a_stale_request_the_proven_release_moved_past_is_superseded(world) -> None:
    """It was newer than what ran when it was asked for, and a later deploy overtook it."""
    _active(world, world["M3"], proven=True)
    _park(world, CARRIER, world["A"], active_at_park=world["base"])

    assert _settle(world) == NOTHING
    assert _outcome(world, CARRIER)["status"] == "superseded"


def test_a_stale_request_is_not_success_when_no_receipt_proves_the_release(world) -> None:
    _active(world, world["M3"], proven=False)
    _park(world, CARRIER, world["A"], active_at_park=world["base"])

    assert _settle(world) == NOTHING
    outcome = _outcome(world, CARRIER)
    assert outcome["status"] == "failed" and outcome["code"] == "parked_active_release_not_proven"


def test_a_request_inside_an_unproven_release_waits_for_a_deploy_that_can_prove_it(world) -> None:
    _active(world, world["A"], proven=False)
    _park(world, OLDER, world["M3"], active_at_park=world["base"])
    _park(world, CARRIER, world["base"], active_at_park="")

    assert _settle(world) == (settle.EXIT_DEPLOY, world["M3"], OLDER)
    assert _outcome(world, CARRIER) is None


def test_a_rollback_asked_for_during_a_deploy_is_refused_not_reported_as_done(world) -> None:
    """A was inside what ran when it was asked for, so it asked to go backwards."""
    _active(world, world["M3"], proven=True)
    _park(world, CARRIER, world["A"], active_at_park=world["M3"])

    assert _settle(world) == NOTHING
    outcome = _outcome(world, CARRIER)
    assert outcome["status"] == "failed" and outcome["code"] == "parked_rollback_refused"
    assert outcome["exit_code"] != 0


def test_a_rollback_is_never_swallowed_by_a_newer_deploy(world) -> None:
    _active(world, world["A"], proven=True)
    _park(world, OLDER, world["base"], active_at_park=world["A"])
    _park(world, CARRIER, world["M3"], active_at_park=world["A"])

    assert _settle(world) == (settle.EXIT_DEPLOY, world["M3"], CARRIER)
    assert _outcome(world, OLDER)["code"] == "parked_rollback_refused"


def test_a_sibling_of_the_active_release_never_deploys_over_it(world) -> None:
    _active(world, world["A"], proven=True)
    _park(world, CARRIER, world["F"], active_at_park=world["base"])

    assert _settle(world) == NOTHING
    assert _outcome(world, CARRIER)["code"] == "parked_commit_not_contained"


def test_an_active_commit_missing_locally_is_fetched_before_comparing(world) -> None:
    """Not an unknown: a hand-run deploy can activate a commit this clone never fetched."""
    fresh = Path(world["root"]) / "main-only-clone"
    _git("clone", "--quiet", "--no-local", "--no-checkout", "--single-branch", "--branch", "main",
         str(world["upstream"]), str(fresh), cwd=Path(world["root"]))
    world = {**world, "clone": fresh}
    with pytest.raises(subprocess.CalledProcessError):
        _git("cat-file", "-e", f"{world['U']}^{{commit}}", cwd=fresh)
    _active(world, world["U"], proven=True)
    _park(world, CARRIER, world["M3"], active_at_park=world["base"])

    assert _settle(world) == NOTHING
    # Once fetched, U turns out to contain M3. "Unknown" would mean the fetch never happened.
    outcome = _outcome(world, CARRIER)
    assert outcome["status"] == "superseded" and outcome["superseded_by_commit"] == world["U"]


def test_an_active_commit_nobody_can_fetch_keeps_everything_parked(world) -> None:
    _active(world, "e" * 40)
    _park(world, CARRIER, world["M3"])

    assert _settle(world) == TRY_AGAIN
    assert _outcome(world, CARRIER) is None and _parked(world) == [CARRIER]


@pytest.mark.parametrize("target", ["not-a-commit", None])
def test_an_unreadable_release_link_keeps_everything_parked(world, target: str | None) -> None:
    root = Path(world["root"])
    if target:
        (root / "active").symlink_to(root / "releases" / target)
    _park(world, CARRIER, world["M3"])

    assert _settle(world) == TRY_AGAIN
    assert _outcome(world, CARRIER) is None and _parked(world) == [CARRIER]


def test_a_deploy_outside_the_doors_units_holds_everything_parked(world) -> None:
    """The iteration wrapper holds this lock without any unit the runner would notice."""
    root = Path(world["root"])
    _active(world, world["base"], proven=True)
    _park(world, CARRIER, world["M3"])
    lock = os.open(root / "state" / settle.DEPLOY_LOCK_NAME, os.O_RDWR | os.O_CREAT)
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        assert _settle(world) == TRY_AGAIN
    finally:
        os.close(lock)
    assert _parked(world) == [CARRIER]
    assert _outcome(world, CARRIER) is None


def test_a_request_with_no_record_does_not_deploy(world) -> None:
    _active(world, world["base"], proven=True)

    assert _settle(world) == NOTHING
    assert _outcome(world, CARRIER)["code"] == "deploy_parked_record_missing"


def test_one_failed_outcome_write_loses_nothing(world, monkeypatch) -> None:
    _active(world, world["M3"], proven=True)
    _park(world, OLDEST, world["A"])
    _park(world, OLDER, world["F"])
    _park(world, CARRIER, world["M4"])
    write = settle._write_outcome

    def full_disk(results, request_id, fields):
        if request_id == OLDEST:
            raise OSError(28, "No space left on device")
        write(results, request_id, fields)

    monkeypatch.setattr(settle, "_write_outcome", full_disk)

    assert _settle(world) == (settle.EXIT_DEPLOY, world["M4"], CARRIER)
    assert _outcome(world, OLDER)["status"] == "superseded"
    assert _outcome(world, OLDEST) is None
    assert OLDEST in _parked(world)  # the next deploy settles it


def test_failing_to_write_this_requests_own_outcome_is_not_silent(world, monkeypatch) -> None:
    _active(world, world["M3"], proven=True)
    _park(world, CARRIER, world["M3"])

    def full_disk(_results, _request_id, _fields):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(settle, "_write_outcome", full_disk)

    assert _settle(world) == (settle.EXIT_CANNOT_DECIDE, "", "")
    assert _parked(world) == [CARRIER]


@pytest.mark.parametrize("change", [
    {"schema": "other"},
    {"id": OLDER},
    {"commit": "not-a-commit"},
    {"commit": None},
    {"active_release_at_park": "not-a-commit"},
])
def test_a_malformed_record_is_ignored(world, change: dict) -> None:
    root = Path(world["root"])
    _active(world, world["base"], proven=True)
    _park(world, CARRIER, world["M3"])
    _park(world, OLDEST, world["A"])
    path = root / "parked" / f"{OLDEST}.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))

    assert _settle(world) == (settle.EXIT_DEPLOY, world["M3"], CARRIER)
    assert _outcome(world, OLDEST) is None


def test_a_record_anyone_else_could_rewrite_is_ignored(world) -> None:
    root = Path(world["root"])
    _active(world, world["base"], proven=True)
    _park(world, CARRIER, world["A"])
    _park(world, OLDEST, world["M3"])
    (root / "parked" / f"{OLDEST}.json").chmod(0o666)
    link = root / "parked" / f"{OLDER}.json"
    link.symlink_to(root / "parked" / f"{CARRIER}.json")

    assert _settle(world) == (settle.EXIT_DEPLOY, world["A"], CARRIER)


def test_a_record_someone_else_owns_authorizes_nothing(world, monkeypatch) -> None:
    _active(world, world["base"], proven=True)
    _park(world, CARRIER, world["M3"])
    monkeypatch.setattr(settle.os, "geteuid", lambda: os.getuid() + 1)

    assert _settle(world) == NOTHING
    assert _outcome(world, CARRIER)["code"] == "deploy_parked_record_missing"


def test_git_failing_to_answer_is_never_read_as_contained(world) -> None:
    assert settle._contains(str(world["clone"]), "e" * 40, world["M3"]) is None
    assert settle._contains(str(world["clone"]), world["A"], world["M3"]) is True
    assert settle._contains(str(world["clone"]), world["M3"], world["A"]) is False

    plan = settle.decide([settle.Parked(CARRIER, world["M3"], world["A"])], carrier=CARRIER, active=world["A"],
                         active_proven=True, contains=lambda _commit, _container: None, on_main=lambda _commit: True)

    assert plan.deploy is None
    assert plan.settled[CARRIER]["code"] == "parked_commit_not_contained"


def test_git_failing_to_answer_is_never_read_as_a_rollback(world) -> None:
    plan = settle.decide([settle.Parked(CARRIER, world["M3"], world["M4"])], carrier=CARRIER, active=world["A"],
                         active_proven=True, contains=lambda _commit, _container: None, on_main=lambda _commit: True)

    assert plan.settled[CARRIER]["code"] == "parked_commit_not_contained"


def _random_main(random: random_module.Random, size: int) -> dict[str, set[str]]:
    """Each commit and its ancestors: a main line with short branches merged back into it."""
    ancestors: dict[str, set[str]] = {}
    names: list[str] = []
    for index in range(size):
        name = f"{index:040x}"
        parents = random.sample(names[-3:], k=min(len(names[-3:]), random.choice((1, 1, 2)))) if names else []
        ancestors[name] = {name}.union(*(ancestors[parent] for parent in parents)) if parents else {name}
        names.append(name)
    return ancestors


@pytest.mark.parametrize("seed", range(300))
def test_no_plan_ever_goes_backward_strands_its_carrier_or_claims_what_is_not_proven(seed: int) -> None:
    random = random_module.Random(seed)
    ancestors = _random_main(random, 14)
    names = list(ancestors)
    active = random.choice(names)
    proven = random.random() < 0.7

    def contains(commit: str, container: str) -> bool:
        return commit in ancestors[container]

    records = [settle.Parked(f"20260923T1200{index:02d}Z-deploy-0000000{index}", random.choice(names),
                             random.choice([""] + names))
               for index in range(random.randint(1, 5))]
    carrier = random.choice(records).id
    plan = settle.decide(records, carrier=carrier, active=active, active_proven=proven, contains=contains,
                         on_main=lambda _commit: True)

    if plan.deploy is not None:
        assert contains(active, plan.deploy), "a parked deploy must not drop what runs"
        assert plan.deploy != active or not proven, "a proven release is not deployed again"
        named = {row.id: row.commit for row in records}
        assert named[plan.deploy_for] == plan.deploy, "a deploy runs for the request that named it"
        assert plan.deploy_for not in plan.settled and carrier not in plan.settled
    else:
        assert carrier in plan.settled, "every run ends the request that started it"
    for row in records:
        fields = plan.settled.get(row.id)
        if fields and fields["status"] == "superseded":
            assert proven and contains(row.commit, active) and fields["superseded_by_commit"] == active


def test_it_runs_isolated_with_the_standard_library_alone(world) -> None:
    root = Path(world["root"])
    _active(world, world["base"], proven=True)
    _park(world, CARRIER, world["M3"])

    done = subprocess.run(
        [sys.executable, "-I", "-S", str(DOOR / "operator_door" / "parked_settle.py")],
        capture_output=True, text=True, check=False,
        env={"PATH": os.environ["PATH"], "DOOR_REQUEST_ID": CARRIER, "DOOR_PARKED_DIR": str(root / "parked"),
             "DOOR_RESULTS_DIR": str(root / "results"), "DOOR_SOURCE_CLONE": str(world["clone"]),
             "DOOR_ACTIVE_RELEASE_LINK": str(root / "active"), "DOOR_STATE_ROOT": str(root / "state")},
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == f"{world['M3']} {CARRIER}"


def test_git_being_unavailable_keeps_every_request_parked(world) -> None:
    """A hung or missing git says nothing about any request."""
    root = Path(world["root"])
    _active(world, world["base"], proven=True)
    _park(world, CARRIER, world["M3"])

    done = subprocess.run(
        [sys.executable, "-I", "-S", str(DOOR / "operator_door" / "parked_settle.py")],
        capture_output=True, text=True, check=False,
        env={"PATH": str(root / "no-such-bin"), "DOOR_REQUEST_ID": CARRIER, "DOOR_PARKED_DIR": str(root / "parked"),
             "DOOR_RESULTS_DIR": str(root / "results"), "DOOR_SOURCE_CLONE": str(world["clone"]),
             "DOOR_ACTIVE_RELEASE_LINK": str(root / "active"), "DOOR_STATE_ROOT": str(root / "state")},
    )

    assert done.returncode == settle.EXIT_TRY_AGAIN, done.stderr
    assert done.stdout == "" and _parked(world) == [CARRIER] and _outcome(world, CARRIER) is None


def test_missing_environment_is_a_clean_failure_to_decide() -> None:
    done = subprocess.run([sys.executable, "-I", "-S", str(DOOR / "operator_door" / "parked_settle.py")],
                          capture_output=True, text=True, check=False, env={"PATH": os.environ["PATH"]})

    assert done.returncode == settle.EXIT_CANNOT_DECIDE
    assert done.stdout == ""
