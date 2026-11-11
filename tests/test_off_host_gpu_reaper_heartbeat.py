"""The host publishes its heartbeat and every live Vast watchdog's deadline, and pages once when it cannot."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from blueprint_pipeline import off_host_gpu_reaper as reaper
from blueprint_pipeline import off_host_gpu_reaper_heartbeat as heartbeat

NOW = 1_800_000_000.0
PREFIX = "blueprint-adp-joint-agent-20261111t120000z-"
EXACT = "blueprint-task-evaluation-scene-config-" + "b" * 32


class Store:
    def __init__(self):
        self.objects, self.fail = {}, False

    def read(self, name):
        return self.objects.get(name)

    def write(self, name, data):
        if self.fail:
            raise OSError("bucket unreachable")
        self.objects[name] = data


def _watchdog(out_dir, *, prefix, deadline, provider="vast", exact=None):
    argv = ["/opt/venv/bin/python", "-m", heartbeat.WATCHDOG_MODULE, "--out-dir", str(out_dir),
            "--pod-name-prefix", prefix, "--deadline-epoch", str(deadline), "--provider", provider]
    return argv + (["--resource-name-exact", exact] if exact else [])


def _proc(tmp_path: Path, argvs) -> Path:
    root = tmp_path / "proc"
    for pid, argv in enumerate(argvs, start=100):
        (root / str(pid)).mkdir(parents=True)
        (root / str(pid) / "cmdline").write_bytes(b"\0".join(part.encode() for part in argv) + b"\0")
    (root / "self").mkdir()
    (root / "200").mkdir()
    (root / "200" / "cmdline").write_bytes(b"")  # a kernel thread
    return root


def _publish(store, tmp_path, *, now=NOW, argvs=(), longest=27_000.0, poster=None, webhook="https://alerts.example"):
    return heartbeat.publish(store_factory=(lambda: store) if store is not None else None,
                             report_root=tmp_path / "state", now=now, argvs=lambda: ([list(a) for a in argvs], []),
                             longest=lambda: longest, webhook_url=webhook, poster=poster)


def test_live_vast_watchdogs_are_published_with_their_scope_id_and_deadline(tmp_path):
    exact_dir, prefix_dir = tmp_path / "exact", tmp_path / "prefix"
    exact_dir.mkdir()
    (exact_dir / heartbeat.STARTED_INSTANCE_ID_NAME).write_text("4242\n")
    argvs = [_watchdog(exact_dir, prefix=EXACT, exact=EXACT, deadline=NOW + 3600),
             _watchdog(prefix_dir, prefix=PREFIX, deadline=NOW + 7200),
             _watchdog(prefix_dir, prefix="blueprint-groot-oscar-canary-x-", deadline=NOW + 60, provider="runpod"),
             ["/usr/bin/python3", "-m", "http.server"]]
    found, scan_problems = heartbeat.process_argvs(_proc(tmp_path, argvs))
    entries, parse_problems = heartbeat.watchdog_entries(found)
    assert scan_problems == [] and parse_problems == []
    assert entries == [{"deadline_epoch": NOW + 3600, "instance_id": 4242, "label": EXACT},
                       {"deadline_epoch": NOW + 7200, "instance_id": None, "label_prefix": PREFIX}]
    store = Store()
    record = _publish(store, tmp_path, argvs=argvs)
    assert record["status"] == "published" and record["published_deadline_count"] == 2
    state = reaper.parse_host_state(store.objects[reaper.STATE_OBJECT])
    assert state.heartbeat_epoch == NOW and state.longest_deadline_seconds == 27_000.0
    assert state.deadlines_complete is True and record["deadlines_complete"] is True
    assert {entry.deadline_epoch for entry in state.entries} == {NOW + 3600, NOW + 7200}


def test_a_superseding_watchdogs_deadline_reaches_the_reaper_on_the_next_heartbeat(tmp_path):
    store = Store()
    machine = reaper.Instance(instance_id=9, label=PREFIX + "1", status="running", started_epoch=NOW - 600)
    _publish(store, tmp_path, argvs=[_watchdog(tmp_path, prefix=PREFIX, deadline=NOW + 60)])
    later = NOW + 30 * 60
    first = reaper.decide(machine, reaper.parse_host_state(store.objects[reaper.STATE_OBJECT]), now=later)
    assert first["action"] == "destroy"
    _publish(store, tmp_path, now=NOW + 120, argvs=[_watchdog(tmp_path, prefix=PREFIX, deadline=NOW + 7200)])
    second = reaper.decide(machine, reaper.parse_host_state(store.objects[reaper.STATE_OBJECT]), now=later)
    assert second["action"] == "leave" and second["deadline_epoch"] == NOW + 7200


def test_publish_failure_pages_once_after_ten_minutes_and_not_on_recovery(tmp_path):
    store, posts = Store(), []
    store.fail = True
    poster = lambda url, payload: posts.append((url, payload))  # noqa: E731
    assert _publish(store, tmp_path, poster=poster)["status"] == "failing"
    assert _publish(store, tmp_path, now=NOW + heartbeat.PUBLISH_FAILURE_PAGE_SECONDS, poster=poster)["paged_epoch"] is None
    assert posts == []
    paged = _publish(store, tmp_path, now=NOW + heartbeat.PUBLISH_FAILURE_PAGE_SECONDS + 120, poster=poster)
    assert paged["paged_epoch"] == NOW + heartbeat.PUBLISH_FAILURE_PAGE_SECONDS + 120 and len(posts) == 1
    alert = posts[0][1]["alerts"][0]
    assert alert["code"] == heartbeat.PAGE_CODE and alert["severity"] == "page" and alert["reason"] == "OSError"
    _publish(store, tmp_path, now=NOW + 3600, poster=poster)
    assert len(posts) == 1
    store.fail = False
    recovered = _publish(store, tmp_path, now=NOW + 3720, poster=poster)
    assert recovered["status"] == "published" and recovered["failing_since_epoch"] is None and len(posts) == 1
    store.fail = True
    _publish(store, tmp_path, now=NOW + 3840, poster=poster)
    _publish(store, tmp_path, now=NOW + 3840 + heartbeat.PUBLISH_FAILURE_PAGE_SECONDS + 1, poster=poster)
    assert len(posts) == 2


def test_a_page_that_cannot_be_posted_is_tried_again_next_tick(tmp_path):
    store = Store()
    store.fail = True

    def broken(_url, _payload):
        raise ConnectionError("webhook down")

    _publish(store, tmp_path, poster=broken)
    record = _publish(store, tmp_path, now=NOW + 700, poster=broken)
    assert record["paged_epoch"] is None and record["page_error"] == "ConnectionError"
    posts = []
    record = _publish(store, tmp_path, now=NOW + 820, poster=lambda url, payload: posts.append(payload))
    assert record["paged_epoch"] == NOW + 820 and len(posts) == 1


def test_an_unreadable_watchdog_publishes_an_incomplete_list_never_a_whole_one(tmp_path):
    store, posts = Store(), []
    bad = _watchdog(tmp_path, prefix=PREFIX, deadline=NOW + 60)
    bad[bad.index("--deadline-epoch") + 1] = "soon"
    good = _watchdog(tmp_path, prefix=EXACT, exact=EXACT, deadline=NOW + 600)
    record = _publish(store, tmp_path, argvs=[bad, good], poster=lambda url, payload: posts.append(payload))
    assert record["status"] == "published_incomplete" and record["deadlines_complete"] is False
    assert record["last_error"] == "deadlines_incomplete:watchdog_deadline_unparseable"
    state = reaper.parse_host_state(store.objects[reaper.STATE_OBJECT])
    assert state.deadlines_complete is False and [entry.label for entry in state.entries] == [EXACT]
    # Incomplete pages like a failed publish: once, after ten minutes.
    _publish(store, tmp_path, now=NOW + heartbeat.PUBLISH_FAILURE_PAGE_SECONDS + 1, argvs=[bad, good],
             poster=lambda url, payload: posts.append(payload))
    assert len(posts) == 1 and posts[0]["alerts"][0]["reason"].startswith("deadlines_incomplete:")


def test_an_older_publisher_without_the_flag_reads_as_not_complete():
    raw = json.dumps({"schema_version": reaper.STATE_SCHEMA_VERSION, "heartbeat_epoch": NOW,
                      "longest_deadline_seconds": 27_000.0, "entries": []}).encode()
    assert reaper.parse_host_state(raw).deadlines_complete is False


def test_an_unreadable_process_makes_the_scan_incomplete(tmp_path, monkeypatch):
    root = _proc(tmp_path, [_watchdog(tmp_path, prefix=PREFIX, deadline=NOW + 60)])
    original = Path.read_bytes

    def denied(self):
        if self.name == "cmdline" and self.parent.name == "100":
            raise PermissionError("hidden")
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", denied)
    argvs, problems = heartbeat.process_argvs(root)
    assert argvs == [] and problems == ["process_unreadable:PermissionError"]


def test_without_a_bucket_nothing_is_published_or_paged(tmp_path):
    posts = []
    for offset in (0, 3600):
        record = _publish(None, tmp_path, now=NOW + offset, poster=lambda url, payload: posts.append(payload))
        assert record["status"] == "not_configured"
    assert posts == []


def test_the_longest_deadline_is_the_largest_lane_ceiling_in_the_code():
    values = [float(getattr(importlib.import_module(f"blueprint_pipeline.{module}"), name))
              for module, name in heartbeat.DEADLINE_CEILINGS]
    assert all(value > 0 for value in values)
    assert heartbeat.longest_code_deadline_seconds() == max(values)
    from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import REQUIRED_PARENT_TTL_SECONDS

    assert heartbeat.longest_code_deadline_seconds() >= REQUIRED_PARENT_TTL_SECONDS


def test_main_writes_its_status_under_the_given_root(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(reaper.BUCKET_ENV, raising=False)
    assert heartbeat.main(["--report-root", str(tmp_path)]) == 0
    assert json.loads((tmp_path / heartbeat.STATUS_NAME).read_text())["status"] == "not_configured"
