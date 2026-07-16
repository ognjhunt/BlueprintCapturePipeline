"""Hermetic tests for ``scripts/gpu_spend_guard.py``.

No network, no real secrets: RunPod/Vast API JSON is supplied as canned dicts and
the terminate path is monkeypatched so a paid call is never made. Covers the three
required reap scenarios — healthy booted pod (kept), stuck dud past the boot
threshold (reaped with ``--reap``), and an owned pod with a live owning process
(kept) — plus the file-based-secret and no-secret-in-output conventions.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import gpu_spend_guard as guard


UTC = timezone.utc


def _epoch(*args: int) -> float:
    return datetime(*args, tzinfo=UTC).timestamp()


# --------------------------- file-based secrets ---------------------------


def test_read_secret_reads_file_and_strips(tmp_path: Path) -> None:
    (tmp_path / "runpod_api_key").write_text("  rp-secret-value\n")
    assert guard._read_secret("runpod_api_key", secrets_dir=tmp_path) == "rp-secret-value"


def test_read_secret_missing_returns_none(tmp_path: Path) -> None:
    assert guard._read_secret("vast_api_key", secrets_dir=tmp_path) is None


# --------------------------- RunPod parsing ---------------------------


def test_parse_runpod_booted_pod_is_live_and_booted() -> None:
    now = _epoch(2026, 6, 27, 1, 0, 0)
    pod = {
        "id": "pod-healthy",
        "name": "blueprint-render-healthy",
        "desiredStatus": "RUNNING",
        "runtime": {"uptimeInSeconds": 3600},
        "costPerHr": 0.79,
        "lastStartedAt": "2026-06-27T00:00:00Z",
    }
    inst = guard._parse_runpod_pod(pod, now=now)
    assert inst.provider == "runpod"
    assert inst.id == "pod-healthy"
    assert inst.booted is True
    assert inst.live is True
    assert inst.cost_per_hr == pytest.approx(0.79)
    assert inst.age_seconds == pytest.approx(3600)


def test_parse_runpod_stuck_dud_is_live_but_not_booted() -> None:
    started = "2026-06-27T00:00:00Z"
    now = _epoch(2026, 6, 27, 0, 10, 0)  # 600s after start
    pod = {
        "id": "pod-dud",
        "name": "blueprint-render-dud",
        "desiredStatus": "RUNNING",
        "runtime": None,
        "costPerHr": 0.79,
        "lastStartedAt": started,
    }
    inst = guard._parse_runpod_pod(pod, now=now)
    assert inst.booted is False
    assert inst.live is True
    assert inst.age_seconds == pytest.approx(600)


def test_parse_runpod_exited_pod_is_not_live() -> None:
    now = _epoch(2026, 6, 27, 1, 0, 0)
    pod = {"id": "pod-old", "name": "x", "desiredStatus": "EXITED", "runtime": None}
    inst = guard._parse_runpod_pod(pod, now=now)
    assert inst.live is False


def test_parse_runpod_stopped_warm_pod_is_not_live_or_reapable() -> None:
    now = _epoch(2026, 6, 27, 1, 0, 0)
    pod = {
        "id": "pwbu7wxsvxpr0x",
        "name": "blueprint-warm-pool",
        "desiredStatus": "STOPPED",
        "runtime": None,
        "costPerHr": 0.79,
        "createdAt": "2026-01-01T00:00:00Z",
    }
    inst = guard._parse_runpod_pod(pod, now=now)
    assert inst.state == "stopped"
    assert inst.live is False
    assert guard.is_reapable(inst, max_boot_seconds=480, protected_ids=set()) is False


@pytest.mark.parametrize("provider", ["gcp", "aws"])
def test_parse_cloud_vm_is_visible_to_burn_and_orphan_logic(provider: str) -> None:
    now = _epoch(2026, 6, 27, 1, 0, 0)
    inst = guard._parse_cloud_vm(
        {
            "instance_id": "blueprint-vm" if provider == "gcp" else "i-0123456789abcdef0",
            "name": "blueprint-render",
            "status": "RUNNING" if provider == "gcp" else "running",
            "created_at": "2026-06-27T00:00:00Z",
            "cost_per_hour": 1.25,
        },
        provider=provider,
        now=now,
    )
    assert inst.live is True
    assert inst.booted is True
    assert inst.cost_per_hr == pytest.approx(1.25)
    assert inst.age_seconds == pytest.approx(3600)


def test_collect_instances_includes_gcp_and_aws() -> None:
    rows = [{"instance_id": "vm", "name": "blueprint", "status": "running"}]
    instances = guard.collect_instances(
        now=0,
        gcp_instances=rows,
        aws_instances=rows,
    )
    assert [item.provider for item in instances] == ["gcp", "aws"]


# --------------------------- Vast parsing ---------------------------


def test_parse_vast_running_instance_is_booted() -> None:
    now = _epoch(2026, 6, 27, 1, 0, 0)
    inst = guard._parse_vast_instance(
        {
            "id": 998877,
            "label": "blueprint-vast-run",
            "actual_status": "running",
            "cur_state": "running",
            "dph_total": 0.42,
            "start_date": _epoch(2026, 6, 27, 0, 0, 0),
        },
        now=now,
    )
    assert inst.provider == "vast"
    assert inst.id == "998877"
    assert inst.booted is True
    assert inst.live is True
    assert inst.cost_per_hr == pytest.approx(0.42)


def test_parse_vast_loading_instance_is_live_but_not_booted() -> None:
    now = _epoch(2026, 6, 27, 0, 10, 0)
    inst = guard._parse_vast_instance(
        {
            "id": 12345,
            "label": "stuck",
            "actual_status": "loading",
            "cur_state": "loading",
            "dph_total": 0.42,
            "start_date": _epoch(2026, 6, 27, 0, 0, 0),
        },
        now=now,
    )
    assert inst.booted is False
    assert inst.live is True
    assert inst.age_seconds == pytest.approx(600)


def test_parse_vast_stopped_instance_is_not_live() -> None:
    now = _epoch(2026, 6, 27, 1, 0, 0)
    inst = guard._parse_vast_instance(
        {"id": 1, "label": "s", "actual_status": "exited", "cur_state": "exited"},
        now=now,
    )
    assert inst.live is False


# --------------------------- reap rules ---------------------------


def _runpod_inst(**over: object) -> "guard.GpuInstance":
    base = dict(
        provider="runpod",
        id="pod-x",
        name="x",
        state="booting",
        booted=False,
        live=True,
        cost_per_hr=0.79,
        age_seconds=600.0,
    )
    base.update(over)
    return guard.GpuInstance(**base)  # type: ignore[arg-type]


def test_booted_pod_is_not_reapable() -> None:
    inst = _runpod_inst(booted=True, state="running")
    assert guard.is_reapable(inst, max_boot_seconds=480, protected_ids=set()) is False


def test_booted_orphan_past_hard_ttl_is_reapable() -> None:
    inst = _runpod_inst(booted=True, state="running", age_seconds=18_000.0)
    assert guard.is_reapable(inst, max_boot_seconds=480, protected_ids=set()) is True
    assert guard.reap_candidate_reason(
        inst,
        max_boot_seconds=480,
        protected_ids=set(),
    ) == "booted_orphan_past_hard_ttl"


def test_booted_owned_pod_past_hard_ttl_is_not_reapable() -> None:
    inst = _runpod_inst(id="pod-owned", booted=True, state="running", age_seconds=18_000.0)
    assert guard.is_reapable(
        inst,
        max_boot_seconds=480,
        protected_ids={"pod-owned"},
    ) is False


def test_unbooted_dud_past_threshold_is_reapable() -> None:
    inst = _runpod_inst(booted=False, age_seconds=600.0)
    assert guard.is_reapable(inst, max_boot_seconds=480, protected_ids=set()) is True


def test_unbooted_pod_within_threshold_is_not_reapable() -> None:
    inst = _runpod_inst(booted=False, age_seconds=120.0)
    assert guard.is_reapable(inst, max_boot_seconds=480, protected_ids=set()) is False


def test_unbooted_pod_with_unknown_age_is_not_reapable() -> None:
    inst = _runpod_inst(booted=False, age_seconds=None)
    assert guard.is_reapable(inst, max_boot_seconds=480, protected_ids=set()) is False


def test_dead_pod_is_not_reapable() -> None:
    inst = _runpod_inst(live=False, booted=False, age_seconds=99999.0)
    assert guard.is_reapable(inst, max_boot_seconds=480, protected_ids=set()) is False


def test_protected_dud_is_never_reapable() -> None:
    inst = _runpod_inst(id="pod-owned", booted=False, age_seconds=600.0)
    assert guard.is_reapable(inst, max_boot_seconds=480, protected_ids={"pod-owned"}) is False


def test_historical_warm_candidate_ids_do_not_bypass_dynamic_protection() -> None:
    for pod_id in guard.DEFAULT_WARM_CANDIDATE_IDS:
        inst = _runpod_inst(id=pod_id, booted=False, age_seconds=99999.0)
        assert guard.is_reapable(inst, max_boot_seconds=480, protected_ids=set()) is True
        assert guard.is_reapable(
            inst,
            max_boot_seconds=480,
            protected_ids={pod_id},
        ) is False


# ------------------- R056: booted-orphan reap rules (careful) -------------------


def _booted_inst(**over: object) -> "guard.GpuInstance":
    base = dict(
        provider="runpod",
        id="pod-booted",
        name="booted",
        state="running",
        booted=True,
        live=True,
        cost_per_hr=0.79,
        age_seconds=30_000.0,
    )
    base.update(over)
    return guard.GpuInstance(**base)  # type: ignore[arg-type]


def test_booted_orphan_reaping_uses_current_default_hard_ttl() -> None:
    # Current main is fail-closed: an unowned booted pod older than the default
    # hard TTL is eligible even when callers do not pass the compatibility alias.
    inst = _booted_inst()
    assert guard.is_reapable(inst, max_boot_seconds=480, protected_ids=set()) is True


def test_booted_orphan_old_and_unowned_is_reapable_when_enabled() -> None:
    inst = _booted_inst(age_seconds=30_000.0)
    assert (
        guard.is_reapable(
            inst,
            max_boot_seconds=480,
            protected_ids=set(),
            orphan_booted_max_age_seconds=21_600,
        )
        is True
    )


def test_booted_orphan_within_hard_age_is_not_reapable() -> None:
    inst = _booted_inst(age_seconds=600.0)
    assert (
        guard.is_reapable(
            inst,
            max_boot_seconds=480,
            protected_ids=set(),
            orphan_booted_max_age_seconds=21_600,
        )
        is False
    )


def test_booted_orphan_with_unknown_age_is_not_reapable() -> None:
    inst = _booted_inst(age_seconds=None)
    assert (
        guard.is_reapable(
            inst,
            max_boot_seconds=480,
            protected_ids=set(),
            orphan_booted_max_age_seconds=21_600,
        )
        is False
    )


def test_booted_warm_serve_protected_pod_is_never_reaped_even_when_old() -> None:
    # A live warm-serve pod (its id in protected_ids via the serving marker) must
    # survive booted-orphan reaping no matter how old it is.
    inst = _booted_inst(id="pod-serve", age_seconds=999_999.0)
    assert (
        guard.is_reapable(
            inst,
            max_boot_seconds=480,
            protected_ids={"pod-serve"},
            orphan_booted_max_age_seconds=21_600,
        )
        is False
    )


def test_stale_historical_warm_candidate_id_is_not_an_implicit_lease() -> None:
    for pod_id in guard.DEFAULT_WARM_CANDIDATE_IDS:
        inst = _booted_inst(id=pod_id, age_seconds=999_999.0)
        assert (
            guard.is_reapable(
                inst,
                max_boot_seconds=480,
                protected_ids=set(),
                orphan_booted_max_age_seconds=21_600,
            )
            is True
        )


def test_booted_fresh_owner_pod_is_never_reaped() -> None:
    inst = _booted_inst(id="pod-owned", age_seconds=999_999.0)
    assert (
        guard.is_reapable(
            inst,
            max_boot_seconds=480,
            protected_ids={"pod-owned"},
            orphan_booted_max_age_seconds=21_600,
        )
        is False
    )


# --------------------------- ownership / live owner ---------------------------


def _make_started_pod_id_file(root: Path, pod_id: str) -> Path:
    out_dir = root / "site-7" / "pipeline" / "render-job" / "isaac_particlefield_render"
    job_dir = out_dir / "object_store_real_run"
    job_dir.mkdir(parents=True)
    f = job_dir / "started_pod_id.txt"
    f.write_text(pod_id)
    return out_dir


def test_owned_pod_with_live_owner_is_protected(tmp_path: Path) -> None:
    out_dir = _make_started_pod_id_file(tmp_path, "pod-owned")
    cmdlines = [
        f"python -m blueprint_pipeline.isaac_particlefield_render_job "
        f"--out-dir {out_dir} --allow-paid --provider runpod",
    ]
    protected = guard.find_protected_pod_ids([tmp_path], process_cmdlines=cmdlines)
    assert protected == {"pod-owned"}


def test_owner_match_by_relative_out_dir_path(tmp_path: Path) -> None:
    # The render job may be launched with a *relative* --out-dir while the guard
    # discovers the run via an absolute path; the absolute path ends with the
    # relative one, so ownership must still be detected (fail-safe toward keep).
    out_dir = _make_started_pod_id_file(tmp_path, "pod-rel")
    rel = "site-7/pipeline/render-job/isaac_particlefield_render"
    assert str(out_dir).endswith(rel)  # guards the fixture's path shape
    protected = guard.find_protected_pod_ids(
        [tmp_path], process_cmdlines=[f"render --out-dir {rel} --allow-paid"]
    )
    assert protected == {"pod-rel"}


def test_started_pod_id_with_dead_owner_is_not_protected(tmp_path: Path) -> None:
    _make_started_pod_id_file(tmp_path, "pod-leaked")
    # No live process references this run -> the owning run died -> orphan.
    protected = guard.find_protected_pod_ids([tmp_path], process_cmdlines=["bash -lc echo hi"])
    assert protected == set()


def test_owner_match_by_pod_id_in_cmdline(tmp_path: Path) -> None:
    _make_started_pod_id_file(tmp_path, "pod-byid")
    protected = guard.find_protected_pod_ids(
        [tmp_path], process_cmdlines=["watcher --pod pod-byid"]
    )
    assert protected == {"pod-byid"}


def test_owned_vast_instance_with_live_owner_is_protected(tmp_path: Path) -> None:
    out_dir = tmp_path / "site-9" / "pipeline" / "render-job" / "isaac_particlefield_render"
    job_dir = out_dir / "object_store_real_run"
    job_dir.mkdir(parents=True)
    (job_dir / "started_vast_instance_id.txt").write_text("778899")
    cmdlines = [
        f"python -m blueprint_pipeline.isaac_particlefield_render_job "
        f"--out-dir {out_dir} --provider vast",
    ]
    protected = guard.find_protected_pod_ids([tmp_path], process_cmdlines=cmdlines)
    assert protected == {"778899"}


def test_started_pod_id_outside_pipeline_dir_is_ignored(tmp_path: Path) -> None:
    job_dir = tmp_path / "scratch" / "object_store_real_run"
    job_dir.mkdir(parents=True)
    (job_dir / "started_pod_id.txt").write_text("pod-nopipeline")
    protected = guard.find_protected_pod_ids(
        [tmp_path], process_cmdlines=[f"x --out-dir {job_dir.parent}"]
    )
    assert protected == set()


# --------------------------- collect + burn estimate ---------------------------


def test_collect_instances_and_total_burn() -> None:
    now = _epoch(2026, 6, 27, 1, 0, 0)
    runpod_pods = [
        {"id": "a", "name": "a", "desiredStatus": "RUNNING",
         "runtime": {"uptimeInSeconds": 10}, "costPerHr": 1.00,
         "lastStartedAt": "2026-06-27T00:00:00Z"},
        {"id": "b", "name": "b", "desiredStatus": "EXITED", "runtime": None,
         "costPerHr": 5.00},  # not live -> excluded from burn
    ]
    vast_instances = [
        {"id": 7, "label": "v", "actual_status": "running", "cur_state": "running",
         "dph_total": 0.50, "start_date": _epoch(2026, 6, 27, 0, 0, 0)},
    ]
    instances = guard.collect_instances(
        now=now, runpod_pods=runpod_pods, vast_instances=vast_instances
    )
    live = [i for i in instances if i.live]
    assert {i.id for i in live} == {"a", "7"}
    assert guard.total_burn_per_hour(instances) == pytest.approx(1.50)


def test_fleet_budget_guard_blocks_live_count_and_burn() -> None:
    instances = [
        guard.GpuInstance(
            provider="runpod",
            id="pod-a",
            name="a",
            state="running",
            booted=True,
            live=True,
            cost_per_hr=1.25,
            age_seconds=10.0,
        ),
        guard.GpuInstance(
            provider="vast",
            id="vast-b",
            name="b",
            state="running",
            booted=True,
            live=True,
            cost_per_hr=0.75,
            age_seconds=10.0,
        ),
    ]

    budget = guard.build_fleet_budget_guard(
        instances,
        max_live_instances=1,
        max_burn_usd_per_hour=1.0,
    )

    assert budget["status"] == "blocked"
    assert budget["live_instance_count"] == 2
    assert budget["total_burn_per_hour_usd"] == pytest.approx(2.0)
    assert budget["blockers"] == [
        "fleet_live_gpu_instance_limit_exceeded",
        "fleet_burn_rate_limit_exceeded",
    ]


def test_spend_ledger_accumulates_daily_total_and_blocks_budget(
    tmp_path: Path,
) -> None:
    now = _epoch(2026, 7, 9, 12, 0, 0)
    ledger_path = tmp_path / "gpu_spend_ledger.json"
    instances = [
        guard.GpuInstance(
            provider="runpod",
            id="pod-a",
            name="a",
            state="running",
            booted=True,
            live=True,
            cost_per_hr=2.0,
            age_seconds=3600.0,
        )
    ]

    first = guard.update_spend_ledger(
        instances,
        ledger_path=ledger_path,
        now=now,
    )
    assert first["schema_version"] == guard.SPEND_LEDGER_SCHEMA_VERSION
    assert first["daily_spend_usd"] == pytest.approx(2.0)
    assert first["total_spend_usd"] == pytest.approx(2.0)

    second = guard.update_spend_ledger(
        instances,
        ledger_path=ledger_path,
        now=now + 1800,
    )
    assert second["daily_spend_usd"] == pytest.approx(3.0)
    assert second["total_spend_usd"] == pytest.approx(3.0)

    budget = guard.build_fleet_budget_guard(
        instances,
        spend_ledger=second,
        max_daily_spend_usd=2.5,
        max_total_spend_usd=2.5,
    )
    assert budget["status"] == "blocked"
    assert budget["daily_spend_usd"] == pytest.approx(3.0)
    assert budget["total_spend_usd"] == pytest.approx(3.0)
    assert "fleet_daily_spend_limit_exceeded" in budget["blockers"]
    assert "fleet_total_spend_limit_exceeded" in budget["blockers"]


def test_fleet_budget_guard_requires_ledger_for_cumulative_limits() -> None:
    budget = guard.build_fleet_budget_guard(
        [],
        max_daily_spend_usd=1.0,
    )

    assert budget["status"] == "blocked"
    assert budget["blockers"] == ["fleet_cumulative_spend_ledger_missing"]


# --------------------------- no secret in output ---------------------------


def test_report_never_contains_secret_values() -> None:
    now = _epoch(2026, 6, 27, 1, 0, 0)
    instances = guard.collect_instances(
        now=now,
        runpod_pods=[{"id": "a", "name": "a", "desiredStatus": "RUNNING",
                      "runtime": {"uptimeInSeconds": 10}, "costPerHr": 1.0,
                      "lastStartedAt": "2026-06-27T00:00:00Z"}],
        vast_instances=[],
    )
    report = guard.build_report(instances, protected_ids=set(), max_boot_seconds=480)
    assert "rp-super-secret" not in report
    assert "va-super-secret" not in report


# --------------------------- main() integration: the 3 scenarios ---------------------------


@pytest.fixture()
def patched_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Wire main() to canned API JSON, a fixed clock, a fake owner process, and a
    recording terminate path so the three reap scenarios run without any network."""
    now = _epoch(2026, 6, 27, 0, 10, 0)  # 600s after the pods' start
    started = "2026-06-27T00:00:00Z"
    booted_orphan_started = "2026-06-26T19:00:00Z"

    out_dir = _make_started_pod_id_file(tmp_path, "pod-owned")

    runpod_pods = [
        # 1) healthy booted pod -> kept
        {"id": "pod-healthy", "name": "healthy", "desiredStatus": "RUNNING",
         "runtime": {"uptimeInSeconds": 600}, "costPerHr": 0.79, "lastStartedAt": started},
        # 2) stuck dud past threshold, unowned -> reaped with --reap
        {"id": "pod-dud", "name": "dud", "desiredStatus": "RUNNING",
         "runtime": None, "costPerHr": 0.79, "lastStartedAt": started},
        # 3) stuck dud past threshold BUT owned by a live process -> kept
        {"id": "pod-owned", "name": "owned", "desiredStatus": "RUNNING",
         "runtime": None, "costPerHr": 0.79, "lastStartedAt": started},
        # 4) booted orphan past hard TTL -> reaped with --reap
        {"id": "pod-booted-orphan", "name": "booted-orphan",
         "desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 18_000},
         "costPerHr": 0.79, "lastStartedAt": booted_orphan_started},
    ]

    monkeypatch.setattr(guard, "_now", lambda: now)
    monkeypatch.setattr(guard, "_read_secret",
                        lambda name, **_kw: {"runpod_api_key": "rp-super-secret",
                                             "vast_api_key": "va-super-secret"}.get(name))
    monkeypatch.setattr(guard, "fetch_runpod_pods", lambda key, **_kw: list(runpod_pods))
    monkeypatch.setattr(guard, "fetch_vast_instances", lambda key, **_kw: [])
    monkeypatch.setattr(
        guard, "list_process_cmdlines",
        lambda: [f"python -m blueprint_pipeline.isaac_particlefield_render_job "
                 f"--out-dir {out_dir} --allow-paid"],
    )

    terminated: list[str] = []

    def fake_http_request(method, url, *, key=None, body=None, timeout=30):
        if method == "DELETE":
            terminated.append(url)
        return 200, {}

    monkeypatch.setattr(guard, "_http_request", fake_http_request)
    return tmp_path, terminated


def test_main_with_no_credentials_is_a_noop(monkeypatch, capsys) -> None:
    monkeypatch.setattr(guard, "_read_secret", lambda name, **_kw: None)
    called: list[str] = []
    monkeypatch.setattr(guard, "fetch_runpod_pods", lambda *a, **k: called.append("rp") or [])
    monkeypatch.setattr(guard, "fetch_vast_instances", lambda *a, **k: called.append("v") or [])
    rc = guard.main([])
    captured = capsys.readouterr()
    assert rc == 2
    assert called == []  # never queries a provider without its key
    assert "No file-based GPU credentials" in captured.err


def test_main_dry_run_reports_but_never_terminates(patched_guard, capsys) -> None:
    tmp_path, terminated = patched_guard
    rc = guard.main(["--output-root", str(tmp_path), "--max-boot-seconds", "480"])
    out = capsys.readouterr().out
    assert rc == 0
    assert terminated == []  # dry-run by default
    assert "pod-healthy" in out and "pod-dud" in out
    assert "rp-super-secret" not in out and "va-super-secret" not in out


def test_main_reap_terminates_only_unowned_orphans(patched_guard, capsys) -> None:
    tmp_path, terminated = patched_guard
    rc = guard.main(["--reap", "--output-root", str(tmp_path), "--max-boot-seconds", "480"])
    out = capsys.readouterr().out
    assert rc == 0
    # The unbooted dud and the booted hard-TTL orphan are deleted.
    assert len(terminated) == 2
    assert any("pod-dud" in url for url in terminated)
    assert any("pod-booted-orphan" in url for url in terminated)
    # Healthy booted pod and owned dud are never deleted.
    assert all("pod-healthy" not in u for u in terminated)
    assert all("pod-owned" not in u for u in terminated)
    assert "rp-super-secret" not in out


def test_main_failed_reap_remains_red(
    patched_guard,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root, _terminated = patched_guard
    report_path = tmp_path / "failed-reap.json"
    monkeypatch.setattr(guard, "_http_request", lambda *_args, **_kwargs: (500, {}))
    assert guard.main(
        [
            "--reap",
            "--output-root",
            str(output_root),
            "--max-boot-seconds",
            "480",
            "--json-report",
            str(report_path),
        ]
    ) == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert all(row["status"] == "terminate_failed" for row in report["reap_results"])
    assert any(blocker.startswith("reap_failed:runpod") for blocker in report["blockers"])


def test_parse_digitalocean_droplet_counts_as_live_spend() -> None:
    """GPU droplets bill until DESTROYED (even powered off) — the guard must list
    them or a leaked droplet is invisible (the 2026-07-02 WAM-pod leak lesson,
    provider-generalized)."""
    from scripts.gpu_spend_guard import _parse_do_droplet, collect_instances

    droplet = {
        "id": 4242,
        "name": "blueprint-isaac-render",
        "status": "active",
        "created_at": "2026-07-02T17:00:00Z",
        "size": {"slug": "gpu-6000adax1-48gb", "price_hourly": 1.57},
    }
    inst = _parse_do_droplet(droplet, now=1_800_000_000.0)
    assert inst.provider == "digitalocean"
    assert inst.id == "4242"
    assert inst.live is True
    assert inst.cost_per_hr == 1.57

    off = dict(droplet, status="off")
    inst_off = _parse_do_droplet(off, now=1_800_000_000.0)
    assert inst_off.live is True  # off still bills!
    assert inst_off.state == "off"

    listed = collect_instances(now=1_800_000_000.0, do_droplets=[droplet])
    assert [i.provider for i in listed] == ["digitalocean"]


def test_main_json_report_persists_snapshot_on_dry_run(patched_guard, capsys) -> None:
    import json as _json

    tmp_path, terminated = patched_guard
    report_path = tmp_path / "ops" / "spend_snapshot.json"
    rc = guard.main([
        "--output-root", str(tmp_path),
        "--max-boot-seconds", "480",
        "--json-report", str(report_path),
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert terminated == []
    snapshot = _json.loads(report_path.read_text())
    assert snapshot["schema_version"] == guard.SCHEMA_VERSION
    assert snapshot["reap_mode"] is False
    assert snapshot["reap_results"] == []
    ids = {i["id"]: i for i in snapshot["instances"]}
    assert ids["pod-dud"]["reap_candidate"] is True
    assert ids["pod-dud"]["reap_candidate_reason"] == "unbooted_dud_past_boot_ttl"
    assert ids["pod-booted-orphan"]["reap_candidate"] is True
    assert ids["pod-booted-orphan"]["reap_candidate_reason"] == (
        "booted_orphan_past_hard_ttl"
    )
    assert ids["pod-healthy"]["reap_candidate"] is False
    assert ids["pod-owned"]["protected"] is True
    assert snapshot["total_burn_per_hour_usd"] > 0
    # File-based secrets never leak into the persisted snapshot.
    assert "rp-super-secret" not in report_path.read_text()
    assert "va-super-secret" not in out


def test_main_json_report_records_reap_results(patched_guard) -> None:
    import json as _json

    tmp_path, terminated = patched_guard
    report_path = tmp_path / "spend_snapshot.json"
    rc = guard.main([
        "--reap",
        "--output-root", str(tmp_path),
        "--max-boot-seconds", "480",
        "--json-report", str(report_path),
    ])
    assert rc == 0
    assert len(terminated) == 2
    snapshot = _json.loads(report_path.read_text())
    assert snapshot["reap_mode"] is True
    assert snapshot["reap_results"] == [
        {"provider": "runpod", "id": "pod-dud", "status": "terminated", "http": 200},
        {
            "provider": "runpod",
            "id": "pod-booted-orphan",
            "status": "terminated",
            "http": 200,
        },
    ]


def test_main_returns_exit_2_when_fleet_budget_is_blocked(patched_guard) -> None:
    import json as _json

    tmp_path, terminated = patched_guard
    report_path = tmp_path / "budget_snapshot.json"
    rc = guard.main([
        "--output-root", str(tmp_path),
        "--max-boot-seconds", "480",
        "--max-live-instances", "1",
        "--max-burn-usd-per-hour", "1.0",
        "--json-report", str(report_path),
    ])

    assert rc == 2
    assert terminated == []
    snapshot = _json.loads(report_path.read_text())
    assert snapshot["fleet_budget"]["status"] == "blocked"
    assert "fleet_live_gpu_instance_limit_exceeded" in snapshot["fleet_budget"]["blockers"]
    assert "fleet_burn_rate_limit_exceeded" in snapshot["fleet_budget"]["blockers"]


def test_inventory_http_failure_is_blocked_not_green_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "inventory-failed.json"
    monkeypatch.setattr(
        guard,
        "_read_secret",
        lambda name, **_kwargs: "runpod-secret" if name == "runpod_api_key" else None,
    )
    monkeypatch.setattr(
        guard,
        "fetch_runpod_pods",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            guard.ProviderInventoryError("runpod", 401)
        ),
    )

    assert guard.main(["--json-report", str(report_path)]) == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    runpod = next(
        row for row in report["inventory_results"] if row["provider"] == "runpod"
    )
    assert runpod["status"] == "failed"
    assert runpod["blockers"] == ["runpod_inventory_query_failed"]
    assert report["live_instance_count"] == 0


def test_digitalocean_reap_requires_verified_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = guard.GpuInstance(
        provider="digitalocean",
        id="4242",
        name="orphan",
        state="active",
        booted=True,
        live=True,
        age_seconds=20_000,
    )
    calls: list[tuple[str, str]] = []

    def verified_request(method: str, url: str, **_kwargs: object) -> tuple[int, dict]:
        calls.append((method, url))
        return (204, {}) if method == "DELETE" else (404, {})

    monkeypatch.setattr(guard, "_http_request", verified_request)
    verified = guard.terminate_instance(
        instance,
        runpod_key=None,
        vast_key=None,
        do_token="do-secret",
        verification_delay_seconds=0,
    )
    assert verified["status"] == "terminated"
    assert verified["absence_verified"] is True
    assert calls == [
        ("DELETE", f"{guard.DO_API}/droplets/4242"),
        ("GET", f"{guard.DO_API}/droplets/4242"),
    ]

    monkeypatch.setattr(
        guard,
        "_http_request",
        lambda method, _url, **_kwargs: (204, {}) if method == "DELETE" else (200, {}),
    )
    unverified = guard.terminate_instance(
        instance,
        runpod_key=None,
        vast_key=None,
        do_token="do-secret",
        verification_attempts=2,
        verification_delay_seconds=0,
    )
    assert unverified["status"] == "terminate_unverified"
    assert unverified["absence_verified"] is False


def test_main_reaps_owned_scope_digitalocean_orphan_and_verifies_absence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = _epoch(2026, 7, 9, 12, 0, 0)
    monkeypatch.setattr(guard, "_now", lambda: now)
    monkeypatch.setattr(
        guard,
        "_read_secret",
        lambda name, **_kwargs: "do-secret"
        if name == "digitalocean_api_token"
        else None,
    )
    monkeypatch.setattr(
        guard,
        "fetch_do_droplets",
        lambda *_args, **_kwargs: [
            {
                "id": 4242,
                "name": "blueprint-orphan",
                "status": "active",
                "created_at": datetime.fromtimestamp(
                    now - 20_000, timezone.utc
                ).isoformat(),
                "size": {"slug": "gpu-6000adax1-48gb", "price_hourly": 1.57},
            }
        ],
    )
    monkeypatch.setattr(guard, "list_process_cmdlines", lambda: [])
    calls: list[str] = []

    def request(method: str, url: str, **_kwargs: object) -> tuple[int, dict]:
        calls.append(f"{method}:{url}")
        return (204, {}) if method == "DELETE" else (404, {})

    monkeypatch.setattr(guard, "_http_request", request)
    report_path = tmp_path / "do-reap.json"
    assert guard.main(
        [
            "--reap",
            "--output-root",
            str(tmp_path),
            "--json-report",
            str(report_path),
        ]
    ) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["reap_results"] == [
        {
            "provider": "digitalocean",
            "id": "4242",
            "status": "terminated",
            "http": 204,
        }
    ]
    assert calls == [
        f"DELETE:{guard.DO_API}/droplets/4242",
        f"GET:{guard.DO_API}/droplets/4242",
    ]


def test_stale_warm_marker_does_not_protect_orphan(tmp_path: Path) -> None:
    now = _epoch(2026, 7, 9, 12, 0, 0)
    marker = tmp_path / "site" / "pipeline" / "warm_serve_pod.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "status": "serving",
                "pod_id": "pod-stale",
                "heartbeat_at": datetime.fromtimestamp(
                    now - 3600, timezone.utc
                ).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    assert guard.find_expected_serve_pod_ids([tmp_path], now=now) == set()

    marker.write_text(
        json.dumps(
            {
                "status": "serving",
                "pod_id": "pod-fresh",
                "lease_expires_at": datetime.fromtimestamp(
                    now + 60, timezone.utc
                ).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    assert guard.find_expected_serve_pod_ids([tmp_path], now=now) == {"pod-fresh"}


def test_spend_ledger_concurrent_updates_are_revisioned_and_atomic(tmp_path: Path) -> None:
    ledger_path = tmp_path / "spend.json"
    instance = guard.GpuInstance(
        provider="runpod",
        id="pod-one",
        name="one",
        state="running",
        booted=True,
        live=True,
        cost_per_hr=1.0,
        age_seconds=3600,
    )
    now = _epoch(2026, 7, 9, 12, 0, 0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: guard.update_spend_ledger(
                    [instance], ledger_path=ledger_path, now=now
                ),
                range(2),
            )
        )
    persisted = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert sorted(result["revision"] for result in results) == [1, 2]
    assert persisted["revision"] == 2
    assert persisted["status"] == "updated"


def test_corrupt_spend_ledger_is_preserved_and_blocks_update(tmp_path: Path) -> None:
    ledger_path = tmp_path / "spend.json"
    corrupt = b'{"status":"updated"'
    ledger_path.write_bytes(corrupt)
    result = guard.update_spend_ledger([], ledger_path=ledger_path, now=1_800_000_000)
    assert result["status"] == "blocked"
    assert result["blockers"] == ["spend_ledger_existing_state_invalid"]
    assert result["prior_state_preserved"] is True
    assert ledger_path.read_bytes() == corrupt


def test_required_billing_reconciliation_rejects_stale_or_incomplete_export(
    tmp_path: Path,
) -> None:
    now = _epoch(2026, 7, 9, 12, 0, 0)
    export = tmp_path / "billing.json"
    export.write_text(
        json.dumps(
            {
                "generated_at": datetime.fromtimestamp(
                    now - 2 * 86400, timezone.utc
                ).isoformat(),
                "provider_totals_usd": {},
            }
        ),
        encoding="utf-8",
    )
    instance = guard.GpuInstance(
        provider="runpod",
        id="pod-one",
        name="one",
        state="running",
        booted=True,
        live=True,
    )
    result = guard.reconcile_billing_export(
        billing_export_path=export,
        instances=[instance],
        now=now,
        required=True,
    )
    assert result["status"] == "blocked"
    assert "provider_billing_export_stale_or_invalid_time" in result["blockers"]
    assert "provider_billing_export_missing:runpod" in result["blockers"]


def test_provider_http_boundary_rejects_unpinned_or_non_https_origins() -> None:
    for url in (
        "http://rest.runpod.io/v1/pods",
        "https://rest.runpod.io.evil.example/v1/pods",
        "file:///etc/passwd",
        "https://user:secret@rest.runpod.io/v1/pods",
    ):
        status, payload = guard._http_request("GET", url, key="never-sent")
        assert status == 0
        assert "pinned HTTPS origins" in payload["error"]

    assert (
        guard._validated_provider_api_url("https://rest.runpod.io/v1/pods")
        == "https://rest.runpod.io/v1/pods"
    )


def test_required_billing_reconciliation_accepts_current_complete_usd_export(
    tmp_path: Path,
) -> None:
    now = _epoch(2026, 7, 9, 12, 0, 0)
    export = tmp_path / "billing.json"
    export.write_text(
        json.dumps(
            {
                "schema_version": guard.BILLING_EXPORT_SCHEMA_VERSION,
                "generated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                "currency": "USD",
                "scope": guard.BILLING_EXPORT_SCOPE,
                "provider_totals_usd": {
                    "runpod": 125.0,
                    "vast": 25.0,
                    "digitalocean": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    result = guard.reconcile_billing_export(
        billing_export_path=export,
        instances=[],
        now=now,
        required=True,
    )

    assert result["status"] == "reconciled"
    assert result["blockers"] == []
    assert result["billing_export_artifact_name"] == "billing.json"
    assert result["billing_export_sha256"].startswith("sha256:")
    assert str(tmp_path) not in json.dumps(result)


def test_required_billing_reconciliation_rejects_unsafe_bounded_inputs(
    tmp_path: Path,
) -> None:
    now = _epoch(2026, 7, 9, 12, 0, 0)
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"{" + b" " * guard.MAX_BILLING_EXPORT_BYTES + b"}")
    symlink = tmp_path / "billing-link.json"
    symlink.symlink_to(oversized)

    oversized_result = guard.reconcile_billing_export(
        billing_export_path=oversized,
        instances=[],
        now=now,
        required=True,
    )
    symlink_result = guard.reconcile_billing_export(
        billing_export_path=symlink,
        instances=[],
        now=now,
        required=True,
    )

    assert "provider_billing_export_too_large" in oversized_result["blockers"]
    assert oversized_result["billing_export_sha256"] is None
    assert "provider_billing_export_symlink" in symlink_result["blockers"]
    assert symlink_result["billing_export_sha256"] is None


def test_required_billing_reconciliation_rejects_unexpected_provider(
    tmp_path: Path,
) -> None:
    now = _epoch(2026, 7, 9, 12, 0, 0)
    export = tmp_path / "billing.json"
    export.write_text(
        json.dumps(
            {
                "schema_version": guard.BILLING_EXPORT_SCHEMA_VERSION,
                "generated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                "currency": "USD",
                "scope": guard.BILLING_EXPORT_SCOPE,
                "provider_totals_usd": {
                    "runpod": 0.0,
                    "vast": 0.0,
                    "digitalocean": 0.0,
                    "unapproved-provider": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    result = guard.reconcile_billing_export(
        billing_export_path=export,
        instances=[],
        now=now,
        required=True,
    )

    assert result["status"] == "blocked"
    assert (
        "provider_billing_export_unexpected:unapproved-provider"
        in result["blockers"]
    )


def test_main_exact_5000_threshold_locks_admission_and_emits_page_event(
    patched_guard,
    tmp_path: Path,
) -> None:
    output_root, terminated = patched_guard
    now = guard._now()
    billing = tmp_path / "provider-billing.json"
    billing.write_text(
        json.dumps(
            {
                "schema_version": guard.BILLING_EXPORT_SCHEMA_VERSION,
                "generated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                "currency": "USD",
                "scope": guard.BILLING_EXPORT_SCOPE,
                "provider_totals_usd": {
                    "runpod": 5000.0,
                    "vast": 0.0,
                    "digitalocean": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    admission_path = tmp_path / "paid-spend-admission.json"
    report_path = tmp_path / "spend-guard.json"

    rc = guard.main(
        [
            "--output-root",
            str(output_root),
            "--spend-ledger",
            str(tmp_path / "ledger.json"),
            "--max-total-spend-usd",
            "5000",
            "--billing-export",
            str(billing),
            "--admission-lock-report",
            str(admission_path),
            "--json-report",
            str(report_path),
        ]
    )

    assert rc == 2
    assert terminated == []
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    assert admission["status"] == "blocked"
    assert admission["admission_allowed"] is False
    assert admission["effective_spend_usd"] == 5000.0
    assert admission["page_event"]["required"] is True
    assert admission["controlled_drain"]["new_paid_work_stopped"] is True
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["spend_admission_lock"]["admission_allowed"] is False


def test_main_valid_override_only_waives_total_hard_stop(
    patched_guard,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root, terminated = patched_guard
    now = guard._now()
    monkeypatch.setattr(
        guard,
        "_read_secret",
        lambda name, **_kwargs: {
            "runpod_api_key": "rp-secret",
            "vast_api_key": "vast-secret",
            "digitalocean_api_token": "do-secret",
        }.get(name),
    )
    monkeypatch.setattr(guard, "fetch_do_droplets", lambda *_args, **_kwargs: [])
    billing = tmp_path / "provider-billing.json"
    billing.write_text(
        json.dumps(
            {
                "schema_version": guard.BILLING_EXPORT_SCHEMA_VERSION,
                "generated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                "currency": "USD",
                "scope": guard.BILLING_EXPORT_SCOPE,
                "provider_totals_usd": {
                    "runpod": 5001.0,
                    "vast": 0.0,
                    "digitalocean": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": guard.SPEND_LEDGER_SCHEMA_VERSION,
                "revision": 1,
                "daily_budget_day": "2026-06-27",
                "daily_spend_usd": 0.0,
                "total_spend_usd": 5001.0,
                "instances": {},
            }
        ),
        encoding="utf-8",
    )
    override = tmp_path / "override.json"
    issued_at = datetime.fromtimestamp(now, timezone.utc)
    override.write_text(
        json.dumps(
            {
                "schema_version": "blueprint.paid_spend_override.v1",
                "status": "approved",
                "scope": "paid_spend_hard_stop",
                "override_id": "override-20260627-001",
                "hard_stop_usd": 5000.0,
                "allow_new_paid_work": True,
                "requested_by": "oncall-operator",
                "approved_by": "finance-approver",
                "reason": "Time-bounded customer recovery approved after cost review.",
                "ticket_uri": "https://tickets.example.invalid/INC-4321",
                "issued_at": issued_at.isoformat(),
                "expires_at": (issued_at + timedelta(hours=2)).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    override.chmod(0o600)
    admission_path = tmp_path / "paid-spend-admission.json"

    rc = guard.main(
        [
            "--output-root",
            str(output_root),
            "--spend-ledger",
            str(ledger),
            "--max-total-spend-usd",
            "5000",
            "--billing-export",
            str(billing),
            "--admission-lock-report",
            str(admission_path),
            "--admission-override",
            str(override),
        ]
    )

    assert rc == 0
    assert terminated == []
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    assert admission["status"] == "override_open"
    assert admission["admission_allowed"] is True
    assert admission["override"]["override_id"] == "override-20260627-001"
