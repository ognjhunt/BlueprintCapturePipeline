"""Hermetic tests for ``scripts/gpu_spend_guard.py``.

No network, no real secrets: RunPod/Vast API JSON is supplied as canned dicts and
the terminate path is monkeypatched so a paid call is never made. Covers the three
required reap scenarios — healthy booted pod (kept), stuck dud past the boot
threshold (reaped with ``--reap``), and an owned pod with a live owning process
(kept) — plus the file-based-secret and no-secret-in-output conventions.
"""

from __future__ import annotations

from datetime import datetime, timezone
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


def test_warm_protected_ids_prefers_live_markers_and_falls_back_to_static() -> None:
    # R107: live warm-serve markers are the single source of truth for warm
    # protection; the static id set is used ONLY as a fail-safe when the live-marker
    # scan yields nothing, so a transient scan gap cannot make warm pods reapable.
    live = {"live-serve-1", "live-serve-2"}
    assert guard.warm_protected_ids(live) == live
    assert guard.warm_protected_ids(set()) == set(guard.DEFAULT_WARM_CANDIDATE_IDS)


def test_warm_candidate_ids_are_never_reapable_when_in_protected_set() -> None:
    # Warm protection now flows through protected_ids (main() unions it from
    # warm_protected_ids) rather than a hardcoded allowlist inside is_reapable.
    protected = set(guard.DEFAULT_WARM_CANDIDATE_IDS)
    for pod_id in guard.DEFAULT_WARM_CANDIDATE_IDS:
        inst = _runpod_inst(id=pod_id, booted=False, age_seconds=99999.0)
        assert guard.is_reapable(inst, max_boot_seconds=480, protected_ids=protected) is False


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


def test_booted_orphan_reaping_off_by_default_keeps_booted_pod() -> None:
    # Legacy behavior preserved: without the hard-age ceiling, no booted pod is reaped.
    inst = _booted_inst()
    assert guard.is_reapable(inst, max_boot_seconds=480, protected_ids=set()) is False


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


def test_booted_warm_candidate_id_is_never_reaped_even_when_old() -> None:
    # R107: warm protection flows through protected_ids (populated by main() from
    # live markers, or the static fallback when no live markers exist), so even a
    # very old booted warm pod is never reaped while it is in the protected set.
    protected = set(guard.DEFAULT_WARM_CANDIDATE_IDS)
    for pod_id in guard.DEFAULT_WARM_CANDIDATE_IDS:
        inst = _booted_inst(id=pod_id, age_seconds=999_999.0)
        assert (
            guard.is_reapable(
                inst,
                max_boot_seconds=480,
                protected_ids=protected,
                orphan_booted_max_age_seconds=21_600,
            )
            is False
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
    out = capsys.readouterr().out
    assert rc == 0
    assert called == []  # never queries a provider without its key
    assert "No file-based GPU credentials" in out


def test_main_dry_run_reports_but_never_terminates(patched_guard, capsys) -> None:
    tmp_path, terminated = patched_guard
    rc = guard.main(["--output-root", str(tmp_path), "--max-boot-seconds", "480"])
    out = capsys.readouterr().out
    assert rc == 0
    assert terminated == []  # dry-run by default
    assert "pod-healthy" in out and "pod-dud" in out
    assert "rp-super-secret" not in out and "va-super-secret" not in out


def test_main_reap_terminates_only_unowned_dud(patched_guard, capsys) -> None:
    tmp_path, terminated = patched_guard
    rc = guard.main(["--reap", "--output-root", str(tmp_path), "--max-boot-seconds", "480"])
    out = capsys.readouterr().out
    assert rc == 0
    # Exactly one DELETE, and it is the unowned dud.
    assert len(terminated) == 1
    assert "pod-dud" in terminated[0]
    # Healthy booted pod and owned dud are never deleted.
    assert all("pod-healthy" not in u for u in terminated)
    assert all("pod-owned" not in u for u in terminated)
    assert "rp-super-secret" not in out


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
    assert len(terminated) == 1
    snapshot = _json.loads(report_path.read_text())
    assert snapshot["reap_mode"] is True
    assert snapshot["reap_results"] == [
        {"provider": "runpod", "id": "pod-dud", "status": "terminated", "http": 200}
    ]
    # New R055/R056 evidence fields on the snapshot.
    assert snapshot["booted_orphan_reaping_enabled"] is False  # not enabled in this run
    assert snapshot["orphan_booted_max_age_seconds"] == 0
    assert snapshot["credentials_available"] is True


# ------------------- R055: durable snapshot without credentials -------------------


def test_main_no_credentials_still_writes_snapshot_for_scheduled_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    import json as _json

    monkeypatch.setattr(guard, "_read_secret", lambda name, **_kw: None)
    report_path = tmp_path / "snap.json"
    rc = guard.main(["--reap", "--json-report", str(report_path)])
    assert rc == 0
    snapshot = _json.loads(report_path.read_text())
    assert snapshot["schema_version"] == guard.SCHEMA_VERSION
    assert snapshot["reap_mode"] is True
    assert snapshot["credentials_available"] is False
    assert snapshot["live_instance_count"] == 0
    assert snapshot["instances"] == []


# ------------------- R056: booted-orphan reap end-to-end via main() -------------------


@pytest.fixture()
def patched_booted_orphan_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Four booted pods: a leaked orphan (reaped), a live warm-serve pod (kept via
    its serving marker), a live-owned pod (kept), and a fresh orphan (kept, too young)."""
    now = _epoch(2026, 6, 27, 12, 0, 0)

    owned_out_dir = _make_started_pod_id_file(tmp_path, "pod-owned-booted")

    serve_dir = tmp_path / "output" / "warm_worker_kitchen"
    serve_dir.mkdir(parents=True)
    (serve_dir / guard.WARM_SERVE_MARKER_FILENAME).write_text(
        __import__("json").dumps(
            {"status": "serving", "pod_id": "pod-warm-serve", "provider": "runpod"}
        )
    )

    runpod_pods = [
        # booted, old, unowned, no marker -> leaked orphan -> reaped
        {"id": "pod-leaked-booted", "name": "leaked", "desiredStatus": "RUNNING",
         "runtime": {"uptimeInSeconds": 30000}, "costPerHr": 0.79},
        # booted, old, live warm-serve marker -> kept
        {"id": "pod-warm-serve", "name": "serve", "desiredStatus": "RUNNING",
         "runtime": {"uptimeInSeconds": 30000}, "costPerHr": 0.69},
        # booted, old, owned by a live process -> kept
        {"id": "pod-owned-booted", "name": "owned", "desiredStatus": "RUNNING",
         "runtime": {"uptimeInSeconds": 30000}, "costPerHr": 0.79},
        # booted, recent -> under the hard age ceiling -> kept
        {"id": "pod-fresh-booted", "name": "fresh", "desiredStatus": "RUNNING",
         "runtime": {"uptimeInSeconds": 120}, "costPerHr": 0.79},
    ]

    monkeypatch.setattr(guard, "_now", lambda: now)
    monkeypatch.setattr(guard, "_read_secret",
                        lambda name, **_kw: {"runpod_api_key": "rp-secret"}.get(name))
    monkeypatch.setattr(guard, "fetch_runpod_pods", lambda key, **_kw: list(runpod_pods))
    monkeypatch.setattr(guard, "fetch_vast_instances", lambda key, **_kw: [])
    monkeypatch.setattr(
        guard, "list_process_cmdlines",
        lambda: [f"python -m blueprint_pipeline.isaac_particlefield_render_job "
                 f"--out-dir {owned_out_dir} --allow-paid"],
    )

    terminated: list[str] = []

    def fake_http_request(method, url, *, key=None, body=None, timeout=30):
        if method == "DELETE":
            terminated.append(url)
        return 200, {}

    monkeypatch.setattr(guard, "_http_request", fake_http_request)
    return tmp_path, terminated


def test_main_reaps_only_leaked_booted_orphan(patched_booted_orphan_guard) -> None:
    tmp_path, terminated = patched_booted_orphan_guard
    rc = guard.main([
        "--reap",
        "--output-root", str(tmp_path),
        "--orphan-booted-max-age-seconds", "21600",
    ])
    assert rc == 0
    assert len(terminated) == 1
    assert "pod-leaked-booted" in terminated[0]
    for kept in ("pod-warm-serve", "pod-owned-booted", "pod-fresh-booted"):
        assert all(kept not in url for url in terminated)


def test_main_booted_orphan_disabled_by_default_reaps_nothing(
    patched_booted_orphan_guard,
) -> None:
    tmp_path, terminated = patched_booted_orphan_guard
    # No --orphan-booted-max-age-seconds and no env -> feature off -> no booted reaping.
    rc = guard.main(["--reap", "--output-root", str(tmp_path)])
    assert rc == 0
    assert terminated == []


def test_main_booted_orphan_enabled_via_env(patched_booted_orphan_guard, monkeypatch) -> None:
    tmp_path, terminated = patched_booted_orphan_guard
    monkeypatch.setenv(guard.ORPHAN_BOOTED_MAX_AGE_ENV, "21600")
    rc = guard.main(["--reap", "--output-root", str(tmp_path)])
    assert rc == 0
    assert len(terminated) == 1
    assert "pod-leaked-booted" in terminated[0]
