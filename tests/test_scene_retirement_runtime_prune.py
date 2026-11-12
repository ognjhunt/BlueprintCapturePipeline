# Covers (for impacted-test selection):
#   src/blueprint_pipeline/scene_retirement_runtime_prune.py
#   deploy/systemd/blueprint-scene-retirement-runtime-prune-plan.service
#   deploy/systemd/blueprint-scene-retirement-runtime-prune.service
"""Old scene-retirement runtime snapshots are pruned by a plan an operator reads and an apply that removes only it.

A fixture runtime tree, boot root, release link and process table stand in for the host's; the
real plan, process index, installer lock and apply run against them.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline import scene_retirement_runtime_prune as prune

NOW = 2_000_000_000.0
DAY = 86400.0
COMMIT = "c" * 40


def _digest(index: int) -> str:
    return f"{index:064x}"


def _age(path: Path, epoch: float) -> None:
    for root, directories, files in os.walk(path):
        for name in (*directories, *files):
            os.utime(Path(root) / name, (epoch, epoch), follow_symlinks=False)
    os.utime(path, (epoch, epoch), follow_symlinks=False)


class Host:
    """The runtime roots, boot root, release link and a process table, all under ``tmp_path``."""

    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.runtime = tmp_path / "mnt" / "scene-retirement-runtime"
        self.inputs = prune.inputs_root_for(self.runtime)
        self.boot = tmp_path / "boot"
        self.proc = tmp_path / "proc"
        self.reports = tmp_path / "reports"
        self.approvals = tmp_path / "approvals"
        for path in (self.runtime / "dependencies", self.runtime / "src", self.boot, self.proc,
                     self.inputs / "git-objects", self.inputs / "sdk-tools"):
            path.mkdir(parents=True, exist_ok=True)
        (self.proc / "stat").write_text("cpu 0\nbtime 1000000000\n")
        lock = self.boot / "installation.lock"
        lock.touch()
        os.chmod(lock, 0o600)
        release = tmp_path / "releases" / COMMIT
        release.mkdir(parents=True)
        self.release_link = tmp_path / "task-evaluation-control-plane"
        self.release_link.symlink_to(release)

    def generation(self, kind: str, index: int, *, days_old: float, files: int = 2) -> Path:
        name = _digest(index)
        parent = "generations" if kind == "source" else "sdk-generations"
        path = self.runtime / parent / name
        (path / "src").mkdir(parents=True)
        for number in range(files):
            (path / "src" / f"m{number}.py").write_bytes(b"x" * 5000)
        manifest = self.runtime / "manifests" / f"{kind}-{name}.json"
        manifest.parent.mkdir(exist_ok=True)
        manifest.write_text("{}")
        _age(path, NOW - days_old * DAY)
        _age(manifest, NOW - days_old * DAY)
        return path

    def release_input(self, commit: str, *, days_old: float, wheels: tuple[str, ...] = ()) -> Path:
        path = self.inputs / "release-inputs" / commit
        path.mkdir(parents=True)
        (path / "uv.lock").write_text("version = 1\n" + "".join(f'hash = "sha256:{wheel}"\n' for wheel in wheels))
        claim = path.parent / f"{commit}.claim.json"
        claim.write_text("{}")
        _age(path, NOW - days_old * DAY)
        _age(claim, NOW - days_old * DAY)
        return path

    def wheel(self, digest: str, *, days_old: float) -> Path:
        path = self.inputs / "wheel-artifacts" / digest
        path.mkdir(parents=True)
        (path / "pkg.whl").write_bytes(b"w" * 3000)
        _age(path, NOW - days_old * DAY)
        return path

    def current(self, source: int, sdk: int | None, *, previous: bytes | None = None) -> bytes:
        sdk_root = (self.runtime / "sdk-generations" / _digest(sdk)) if sdk is not None else self.runtime / "dependencies"
        selector = None if previous is None else {"sha256": "sha256:" + hashlib.sha256(previous).hexdigest(),
                                                  "size_bytes": len(previous)}
        raw = json.dumps({
            "schema": "scene-retirement-runtime-cohort.v1", "runtime_root": str(self.runtime / "generations" / _digest(source)),
            "dependencies_root": str(sdk_root), "source_digest": _digest(source),
            "dependency_digest": _digest(sdk if sdk is not None else 999), "bootstrap": {},
            "previous": selector}).encode()
        (self.boot / "CURRENT.json").write_bytes(raw)
        return raw

    def bootstrap(self, pid: int, *, holds: tuple[int, ...] | None = None, sdk: int | None = None,
                  maps: str = "", started_days_ago: float = 1) -> None:
        """A process started through the protected bootstrap; ``holds`` the generations it holds open."""
        fds = () if holds is None else tuple(self.runtime / "generations" / _digest(i) for i in holds)
        if sdk is not None:
            fds += (self.runtime / "sdk-generations" / _digest(sdk),)
        self.process(pid, cmdline=f"/usr/bin/python3 -I -S {self.boot}/continuous_bootstrap.py --continuous-module m",
                     fds=fds, maps=maps, started_days_ago=started_days_ago)

    def process(self, pid: int, *, cmdline: str = "python", maps: str = "", fd_target: Path | None = None,
                fds: tuple[Path, ...] = (), started_days_ago: float | None = None) -> None:
        root = self.proc / str(pid)
        (root / "fd").mkdir(parents=True)
        (root / "cmdline").write_bytes(cmdline.encode().replace(b" ", b"\0"))
        (root / "environ").write_bytes(b"PATH=/usr/bin\0")
        (root / "maps").write_text(maps)
        (root / "cwd").symlink_to("/")
        (root / "exe").symlink_to("/usr/bin/python3")
        for number, target in enumerate(((fd_target,) if fd_target is not None else ()) + tuple(fds), start=3):
            (root / "fd" / str(number)).symlink_to(target)
        started = NOW - (started_days_ago if started_days_ago is not None else 1) * DAY
        ticks = os.sysconf("SC_CLK_TCK")
        fields = ["S"] + ["0"] * 18 + [str(int((started - 1_000_000_000) * ticks))]
        (root / "stat").write_text(f"{pid} (python3) " + " ".join(fields) + "\n")

    def plan(self, **kwargs):
        return prune.write_plan(self.reports, runtime_root=self.runtime, boot_root=self.boot,
                                release_link=self.release_link, process_root=self.proc,
                                minimum_age_seconds=kwargs.pop("minimum_age_seconds", int(7 * DAY)), now=NOW)

    def approve(self, plan_digest: str | None = None, *, approval_id: str = "owner-2026-11-12") -> Path:
        """An owner approval of the plan (by default the plan last written): a test stands in for the owner."""
        digest = plan_digest or json.loads((self.reports / "plan.json").read_text())["plan_digest"]
        path = self.approvals / f"{approval_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema_version": prune.APPROVAL_SCHEMA, "approval_id": approval_id,
                                    "approved_by": "test owner", "approval_evidence": "fixture approval",
                                    "approved_at": "2033-05-18T00:00:00Z", "expires_at": "2033-05-25T00:00:00Z",
                                    "plan_digest": digest}))
        return path

    def apply(self, *, at: float = NOW + 60, approve: bool = True, **kwargs):
        if approve and not any(self.approvals.glob("*.json")):
            self.approve()
        return prune.apply_plan(report_root=self.reports, ack=prune.APPLY_ACK, approval_dir=self.approvals,
                                process_root=self.proc, release_link=self.release_link, now=lambda: at, **kwargs)


def _rows(plan):
    return {(row["kind"], row["name"]): row for row in plan["rows"]}


@pytest.fixture
def host(tmp_path):
    host = Host(tmp_path)
    # Ten source generations, one a day, 20 to 11 days old; the current one is the oldest (a rollback).
    for index in range(10):
        host.generation("source", index, days_old=20 - index)
    host.generation("source", 50, days_old=1)  # Too new to prune.
    for index in (100, 101, 102, 103, 104, 105):
        host.generation("sdk", index, days_old=40 - (index - 100))
    host.current(source=0, sdk=100)
    old_wheel, kept_wheel = "a" * 64, "b" * 64
    host.wheel(old_wheel, days_old=30)
    host.wheel(kept_wheel, days_old=30)
    host.release_input(COMMIT, days_old=30, wheels=(kept_wheel,))
    for index in range(6):
        host.release_input(f"{index:040x}", days_old=25 - index)
    host.process(4242)  # An ordinary process holding nothing of the runtime.
    return host


def test_the_keep_rules(host):
    # A live process maps a compiled library from one old SDK generation, and one holds a file of an old source one.
    host.process(5000, maps=f"7f00-7f01 r-xp 0 08:01 99 {host.runtime}/sdk-generations/{_digest(101)}/lib/x.so\n")
    host.process(5001, fd_target=host.runtime / "generations" / _digest(2) / "src" / "m0.py")
    plan = host.plan()
    rows = _rows(plan)
    keep = {key for key, row in rows.items() if row["disposition"] == "keep"}

    def source(index):
        return ("source_generation", _digest(index))
    assert "current" in rows[source(0)]["reasons"]
    # The previous three: the three most recently created besides the current.
    assert all("previous_three" in rows[source(index)]["reasons"] for index in (50, 9, 8))
    assert "previous_three" not in rows[source(7)]["reasons"]
    assert "newer_than_minimum_age" in rows[source(50)]["reasons"]
    assert rows[source(2)]["reasons"] == ["in_use"]
    assert rows[("sdk_generation", _digest(101))]["reasons"] == ["in_use"]
    assert "current" in rows[("sdk_generation", _digest(100))]["reasons"]
    removed = {key for key, row in rows.items() if row["disposition"] == "remove"}
    assert {source(index) for index in (1, 3, 4, 5, 6, 7)} <= removed
    assert source(2) in keep
    # Release inputs: the active release's own and the three newest others; wheels its lock names.
    assert "current" in rows[("release_input", COMMIT)]["reasons"]
    assert rows[("wheel_artifact", "b" * 64)]["reasons"] == ["named_by_a_kept_release_lock"]
    assert rows[("wheel_artifact", "a" * 64)]["disposition"] == "remove"
    # Never listed: the first installation at the top of the runtime root, dependencies, git and SDK tools.
    assert all(Path(row["path"]).parent.name in {"generations", "sdk-generations", "release-inputs", "sdk-inputs",
                                                  "wheel-artifacts"} for row in plan["rows"])
    # Every listed item carries its measured size, and the totals add up.
    assert all(row["identity"]["apparent_bytes"] > 0 for row in plan["rows"])
    assert plan["remove_allocated_bytes"] == sum(row["identity"]["allocated_bytes"] for row in plan["rows"]
                                                 if row["disposition"] == "remove")


def test_a_bootstrap_keeps_the_generation_it_holds_even_after_a_rollback(host):
    """The review's probe: the current cohort is generation 9, but a process started on a rollback runs 2."""
    host.current(source=9, sdk=100)
    host.bootstrap(6000, holds=(2,), sdk=100, started_days_ago=14)
    plan = host.plan()
    rows = _rows(plan)
    assert rows[("source_generation", _digest(2))]["reasons"] == ["in_use"]
    assert rows[("source_generation", _digest(3))]["disposition"] == "remove"
    checks = plan["first_host_run_checks"]
    assert checks["process_inventory_readable"] is True and checks["running_bootstrap_processes"] == 1
    assert checks["bootstrap_processes"][0]["source_generation"] == [_digest(2)]
    assert checks["bootstrap_processes"][0]["running_generation_planned_for_removal"] == []
    host.apply()
    assert (host.runtime / "generations" / _digest(2)).is_dir()
    assert not (host.runtime / "generations" / _digest(3)).exists()


def test_a_bootstrap_without_a_selection_marker_stops_every_generation_removal(host):
    """A process started before this release names no generation: no source or SDK generation is removed."""
    host.current(source=9, sdk=100)
    host.bootstrap(6001, holds=None, started_days_ago=14)
    plan = host.plan()
    generations = [row for row in plan["rows"] if row["kind"] in prune._GENERATIONS]
    assert generations and all(row["disposition"] == "keep" and prune.UNKNOWN_SELECTION in row["reasons"]
                               for row in generations)
    assert plan["first_host_run_checks"]["bootstrap_processes_without_selection_marker"] == 1
    # One that starts after the plan, too: the apply re-checks before each generation and says why.
    (host.proc / "6001").rename(host.tmp / "gone-6001")
    plan = host.plan()
    assert any(row["kind"] == "source_generation" and row["disposition"] == "remove" for row in plan["rows"])
    (host.tmp / "gone-6001").rename(host.proc / "6001")
    result = host.apply()
    skipped = [row for row in result["outcomes"] if prune.UNKNOWN_SELECTION in row.get("reasons", ())]
    assert skipped and all(Path(row["path"]).is_dir() for row in skipped)
    assert all(Path(row["path"]).parent.name in {"generations", "sdk-generations"} for row in skipped)


def test_every_generation_the_cohort_history_names_is_kept(host):
    older = host.current(source=1, sdk=101)
    host.plan()  # The first run keeps a copy of each cohort record it reads.
    host.current(source=0, sdk=100, previous=older)
    plan = host.plan()
    rows = _rows(plan)
    assert "named_by_cohort_history" in rows[("source_generation", _digest(1))]["reasons"]
    assert "named_by_cohort_history" in rows[("sdk_generation", _digest(101))]["reasons"]
    assert [row["source_generation"] for row in plan["cohort_history"]["cohorts"]] == [_digest(0), _digest(1)]
    assert plan["cohort_history"]["ends_at"] == "chain_complete"
    # A previous record never seen ends the walk, and says so.
    host.current(source=0, sdk=100, previous=b"never retained")
    assert host.plan()["cohort_history"]["ends_at"].startswith("previous_record_not_retained:")


def test_an_unreadable_process_inventory_keeps_everything(host):
    host.process(7000)
    (host.proc / "7000" / "fd").rename(host.proc / "7000" / "fd-gone")
    (host.proc / "7000" / "fd").write_text("not a directory")  # Listing it fails with an error other than exit.
    plan = host.plan()
    assert plan["process_inventory_unreadable"] is True and plan["remove_count"] == 0


def test_a_minimum_age_under_two_days_is_refused(host):
    with pytest.raises(prune.RuntimePruneError, match="minimum_age_too_short"):
        host.plan(minimum_age_seconds=int(DAY))


def test_the_plan_needs_a_readable_current_cohort(host):
    (host.boot / "CURRENT.json").write_text("{}")
    with pytest.raises(prune.RuntimePruneError, match="current_cohort_unreadable"):
        host.plan()
    (host.boot / "CURRENT.json").unlink()
    with pytest.raises(prune.RuntimePruneError, match="current_cohort_missing"):
        host.plan()


def test_plan_and_apply_round_trip(host):
    plan = host.plan()
    planned = {row["path"] for row in plan["rows"] if row["disposition"] == "remove"}
    kept = {row["path"] for row in plan["rows"] if row["disposition"] == "keep"}
    assert planned and kept
    published = json.loads((host.reports / "plan.json").read_text())
    assert published["plan_digest"] == plan["plan_digest"]
    result = host.apply()
    assert result["outcome_counts"] == {"removed": len(planned)}
    assert result["removed_allocated_bytes"] == plan["remove_allocated_bytes"]
    for path in planned:
        assert not Path(path).exists()
    for path in kept:
        assert Path(path).is_dir()
    # Each generation's manifest went with it; kept ones keep theirs.
    for row in plan["rows"]:
        if row["companion"]:
            assert Path(row["companion"]["path"]).exists() == (row["disposition"] == "keep")
    # Nothing renamed aside is left behind, and every removal has its receipt.
    assert not [path for path in host.tmp.rglob(prune.ASIDE_PREFIX + "*")]
    receipts = sorted((host.reports / "receipts").glob("*.json"))
    assert len(receipts) == len(planned)
    assert all(json.loads(path.read_text())["status"] == "removed" for path in receipts)
    # The top of the runtime root is untouched.
    assert (host.runtime / "dependencies").is_dir() and (host.runtime / "src").is_dir()
    # Applying the same plan again removes nothing more.
    again = host.apply(at=NOW + 120)
    assert again["removed_count"] == 0 and again["outcome_counts"] == {"skipped": len(planned)}


def test_an_item_that_comes_into_use_after_the_plan_is_skipped(host):
    plan = host.plan()
    target = next(row for row in plan["rows"] if row["kind"] == "source_generation" and row["disposition"] == "remove")
    host.process(8000, fd_target=Path(target["path"]) / "src" / "m1.py")
    result = host.apply()
    outcome = next(row for row in result["outcomes"] if row["path"] == target["path"])
    assert outcome == {"path": target["path"], "outcome": "skipped", "reasons": ["now_kept", "in_use"]}
    assert Path(target["path"]).is_dir() and Path(target["companion"]["path"]).is_file()
    assert result["outcome_counts"]["removed"] == plan["remove_count"] - 1


def test_an_item_that_changed_after_the_plan_is_skipped(host):
    plan = host.plan()
    target = next(row for row in plan["rows"] if row["disposition"] == "remove")
    (Path(target["path"]) / "late.txt").write_text("written after the plan")
    result = host.apply()
    outcome = next(row for row in result["outcomes"] if row["path"] == target["path"])
    assert outcome["outcome"] == "skipped" and outcome["reasons"] == ["changed_since_plan"]
    assert Path(target["path"]).is_dir()


def test_the_apply_needs_its_ack_and_an_owner_approval_of_this_exact_plan(host):
    plan = host.plan()
    with pytest.raises(prune.RuntimePruneError, match="apply_not_acknowledged"):
        prune.apply_plan(report_root=host.reports, ack="", approval_dir=host.approvals, process_root=host.proc,
                         now=lambda: NOW)
    with pytest.raises(prune.RuntimePruneError, match="owner_approval_missing"):
        host.apply(approve=False)
    host.approve("sha256:" + "0" * 64)  # An approval of a plan that does not exist here.
    with pytest.raises(prune.RuntimePruneError, match="plan_invalid"):
        host.apply()
    host.approve(plan["plan_digest"])  # The approved plan, then altered on disk: its digest differs.
    path = host.reports / "plans" / (plan["plan_digest"][7:] + ".json")
    altered = json.loads(path.read_text())
    altered["rows"][0]["disposition"] = "remove"
    path.write_text(json.dumps(altered))
    with pytest.raises(prune.RuntimePruneError, match="plan_invalid"):
        host.apply()
    path.write_text(json.dumps(plan))
    with pytest.raises(prune.RuntimePruneError, match="owner_approval_missing"):
        host.apply(at=NOW + 30 * DAY)  # Outside the approval's window.
    assert all(Path(row["path"]).exists() for row in plan["rows"])


def test_a_deploy_holding_the_installer_lock_stops_the_run(host):
    plan = host.plan()
    with open(host.boot / "installation.lock") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        result = host.apply()
    assert result["stopped"] == "installer_busy" and result["removed_count"] == 0
    assert result["outcome_counts"] == {"not_attempted:installer_busy": plan["remove_count"]}
    assert all(Path(row["path"]).exists() for row in plan["rows"])


def test_an_old_bootstrap_with_only_a_mapped_sdk_library_is_unknown(host):
    """The review's state: started before this release, no descriptors, an SDK .so in its maps."""
    host.current(source=9, sdk=100)
    so = host.runtime / "sdk-generations" / _digest(100) / "numpy" / "_core.so"
    host.bootstrap(4243, maps=f"7f00-7f10 r-xp 00000000 08:01 1234 {so}\n", started_days_ago=14)
    plan = host.plan()
    checks = plan["first_host_run_checks"]
    assert checks["bootstrap_processes_with_unknown_source_generation"] == 1
    assert checks["bootstrap_processes_without_selection_marker"] == 1
    assert checks["bootstrap_processes"][0]["source_selection_known"] is False
    assert all(row["disposition"] == "keep" for row in plan["rows"] if row["kind"] in prune._GENERATIONS)
    host.apply()
    assert all((host.runtime / "generations" / _digest(i)).is_dir() for i in range(10))


def test_an_sdk_descriptor_says_nothing_about_the_source_generation(host):
    host.current(source=9, sdk=100)
    host.bootstrap(4244, holds=None, sdk=100)
    rows = _rows(host.plan())
    assert prune.UNKNOWN_SELECTION in rows[("source_generation", _digest(1))]["reasons"]
    assert prune.UNKNOWN_SELECTION not in rows[("sdk_generation", _digest(101))]["reasons"]
    assert rows[("sdk_generation", _digest(101))]["disposition"] == "remove"


def test_deploys_and_a_rollback_between_plan_and_apply_do_not_void_the_approval(host):
    """Plan on day 0; deploys on days 2 and 4; a rollback on day 6; apply on day 6."""
    first = host.current(source=0, sdk=100)
    host.bootstrap(5000, holds=(0,), sdk=100)
    plan = host.plan()
    host.approve(plan["plan_digest"])
    planned = {row["name"] for row in plan["rows"] if row["kind"] == "source_generation"
               and row["disposition"] == "remove"}
    assert {_digest(i) for i in (1, 2, 3, 4, 5, 6)} <= planned
    second = host.current(source=4, sdk=100, previous=first)       # day 2: a deploy that reuses generation 4
    third = host.current(source=5, sdk=100, previous=second)       # day 4
    host.bootstrap(5002, holds=(5,), sdk=100)                      # started on day 4, still running
    host.current(source=3, sdk=100, previous=third)                # day 6: a rollback to generation 3
    host.bootstrap(5001, holds=(3,), sdk=100)                      # restarted on it
    (host.proc / "5000").rename(host.tmp / "exited-5000")          # the day-0 process has exited
    host.generation("source", 60, days_old=-5)                     # day 2's new generation (made after the plan)
    result = host.apply(at=NOW + 6 * DAY)
    outcomes = {Path(row["path"]).name: row for row in result["outcomes"]}
    assert result["stopped"] is None
    # Current and in use (3), in use by a process still running it (5): skipped and reported as now kept.
    for index in (3, 5):
        assert outcomes[_digest(index)]["reasons"][0] == "now_kept", index
        assert (host.runtime / "generations" / _digest(index)).is_dir()
    assert "current" in outcomes[_digest(3)]["reasons"] and "in_use" in outcomes[_digest(3)]["reasons"]
    assert "in_use" in outcomes[_digest(5)]["reasons"]
    # Still old and unused (4 was current on day 2, but nothing runs it now): removed.
    for index in (1, 2, 4, 6, 7):
        assert outcomes[_digest(index)]["outcome"] == "removed", index
        assert not (host.runtime / "generations" / _digest(index)).exists()


def test_the_release_link_is_read_again_for_each_item(host):
    plan = host.plan()
    host.release_link.unlink()
    host.release_link.symlink_to(host.tmp / "not-a-commit")
    result = host.apply()
    assert result["stopped"] == "active_release_unreadable" and result["removed_count"] == 0
    assert result["outcome_counts"] == {"not_attempted:active_release_unreadable": plan["remove_count"]}


def test_a_stop_between_items_ends_the_run_cleanly(host):
    plan = host.plan()
    asked = iter([False, True])
    result = host.apply(should_stop=lambda: next(asked, True))
    assert result["stopped"] == "stop_requested" and result["removed_count"] == 1
    assert result["outcome_counts"]["not_attempted:stop_requested"] == plan["remove_count"] - 1
    assert not list(host.tmp.rglob(prune.ASIDE_PREFIX + "*"))


def test_a_stop_asked_mid_item_finishes_that_item_then_stops(host, monkeypatch):
    plan = host.plan()
    stop = {"asked": False}
    real = prune._remove_aside

    def removing(*args, **kwargs):
        stop["asked"] = True  # SIGTERM while this item's tree is being removed.
        return real(*args, **kwargs)
    monkeypatch.setattr(prune, "_remove_aside", removing)
    result = host.apply(should_stop=lambda: stop["asked"])
    assert result["removed_count"] == 1 and result["stopped"] == "stop_requested"
    assert result["outcome_counts"]["not_attempted:stop_requested"] == plan["remove_count"] - 1


def test_a_run_killed_after_rename_aside_is_finished_by_the_next_plan_and_apply(host, monkeypatch):
    plan = host.plan()

    def killed(*_args, **_kwargs):
        raise SystemExit("killed mid-item")
    monkeypatch.setattr(prune, "_remove_aside", killed)
    with pytest.raises(SystemExit):
        host.apply()
    monkeypatch.undo()
    stray = host.runtime / "generations" / (prune.ASIDE_PREFIX + "no-receipt")
    stray.mkdir()
    leftovers = host.plan()["first_host_run_checks"]["leftover_aside_trees"]
    assert sorted((str(row["receipt_status"]), row["receipt"] is None) for row in leftovers) == [
        ("None", True), ("removing", False)]
    for old in host.approvals.glob("*.json"):
        old.unlink()
    host.approve()
    result = host.apply(at=NOW + 120)
    assert result["outcome_counts"]["removed_after_interruption"] == 1
    assert result["outcome_counts"]["aside_left:no_receipt"] == 1
    assert stray.is_dir()  # One with no receipt is reported and left.
    finished = [path for path in host.tmp.rglob(prune.ASIDE_PREFIX + "*") if path != stray]
    assert finished == []
    receipts = [json.loads(path.read_text()) for path in (host.reports / "receipts").glob("*.json")]
    assert all(receipt["status"] == "removed" for receipt in receipts)
    assert len(receipts) == plan["remove_count"]


def test_a_plan_row_outside_its_own_parent_is_never_removed(host):
    plan = host.plan()
    stray = host.tmp / "elsewhere" / _digest(1)
    stray.mkdir(parents=True)
    row = next(row for row in plan["rows"] if row["disposition"] == "remove")
    row["path"] = str(stray)
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    (host.reports / "plans" / (plan["plan_digest"][7:] + ".json")).write_text(json.dumps(plan))
    host.approve(plan["plan_digest"])
    result = host.apply()
    assert {"path": str(stray), "outcome": "skipped", "reasons": ["plan_row_invalid"]} in result["outcomes"]
    assert stray.is_dir()


def test_no_timer_or_path_unit_starts_the_pruner():
    systemd = Path(__file__).resolve().parents[1] / "deploy" / "systemd"
    names = ("blueprint-scene-retirement-runtime-prune-plan.service", "blueprint-scene-retirement-runtime-prune.service")
    for name in names:
        assert (systemd / name).is_file()
    for unit in systemd.iterdir():
        if unit.suffix in {".timer", ".path"}:
            text = unit.read_text(encoding="utf-8")
            assert not any(name in text for name in names), unit.name
    apply_unit = (systemd / names[1]).read_text(encoding="utf-8")
    # The ack alone deletes nothing: the apply also needs the owner's approval of the exact plan.
    assert f"--ack {prune.APPLY_ACK}" in apply_unit and "[Install]" not in apply_unit
    assert "approvals" not in {path.name for path in (Path(__file__).resolve().parents[1] / "deploy").iterdir()}
