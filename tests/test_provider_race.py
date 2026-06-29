"""Hermetic tests for the parallel provider race launcher + circuit breaker.

No GPU, no network, no real sleeping (a no-op ``sleep`` is injected, or a tiny real
poll interval where a timing bound is the thing under test). Every provider here is a
fake honoring ONLY the production surface the racer is allowed to touch:
``launch(job_dir, request, *, cold=False)`` and ``terminate(instance_id)``.

The motivating bug: today's RunPod->Vast failover is sequential, so a degraded pool on
the first provider stalls the whole job while it waits out a long boot timeout. The race
launches every provider at once and returns the FIRST to show its early boot marker,
terminating the losers — a degraded provider can no longer hold the job hostage.
"""
from __future__ import annotations

import io
import json
import threading
import time
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.provider_race import (
    ProviderCircuitBreaker,
    boot_marker_present,
    race_launch,
)


# ----------------------------- fakes -----------------------------

class FakeProvider:
    """Minimal stand-in honoring only ``.launch(job_dir, request, cold=)`` / ``.terminate(iid)``.

    ``boots`` + ``marker_after`` model how the early boot marker shows up: a fast booter
    flips its marker on the first poll; a dud never does. ``launches=False`` models a
    degraded pool (offer search / pod create returns blocked with no instance at all).
    """

    def __init__(self, name, *, boots=True, marker_after=1, launches=True,
                 launch_blockers=None):
        self.name = name
        self.boots = boots
        self.marker_after = marker_after
        self._launches = launches
        self._launch_blockers = list(launch_blockers or ["no_capacity"])
        # observability for assertions
        self.launch_calls = 0
        self.launch_cold = None
        self.last_request = None
        self.terminate_calls = []
        self.marker_calls = 0
        self._marker_seen = 0

    def launch(self, job_dir, request, *, cold=False):
        self.launch_calls += 1
        self.launch_cold = cold
        self.last_request = request
        assert isinstance(job_dir, Path)  # racer must hand each contender a real dir
        if not self._launches:
            return {"status": "blocked", "blockers": list(self._launch_blockers)}
        return {"status": "launched", "instance_id": f"{self.name}-iid",
                "mode": "fake_cold", "attempts": []}

    def terminate(self, instance_id):
        self.terminate_calls.append(instance_id)
        return {"status": "terminated", "http": 204, "instance_id": instance_id}

    # called by the shared marker_check below
    def has_marker(self, launch_result):
        self.marker_calls += 1
        if not self.boots:
            return False
        self._marker_seen += 1
        return self._marker_seen >= self.marker_after


def _marker_check(provider, launch_result):
    """The injectable seam: in prod this polls the signed GET url for bootstrap.json;
    here it just asks the fake. Proves the racer is agnostic to *how* booting is detected."""
    return provider.has_marker(launch_result)


_NO_SLEEP = lambda *_a, **_k: None  # noqa: E731 — deterministic, no wall-clock in unit tests


def _marker_zip(payload: dict | None = None, *, member: str = "bootstrap.json") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        if payload is not None:
            zf.writestr(member, json.dumps(payload))
    return buf.getvalue()


class _UrlOpenResponse:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


def _urlopen_for(data: bytes):
    def _open(_url, timeout=None):  # noqa: ANN001 - mirrors urllib.request.urlopen
        return _UrlOpenResponse(data)

    return _open


# ----------------------------- shared boot marker helper -----------------------------


def test_boot_marker_present_reads_bootstrap_from_signed_output(tmp_path: Path):
    (tmp_path / "provider_output_get_url.txt").write_text(
        "https://store.example/output.zip?sig=secret",
        encoding="utf-8",
    )

    assert boot_marker_present(
        tmp_path,
        expected_launch_session_id="fresh-session",
        urlopen=_urlopen_for(_marker_zip({"launch_session_id": "fresh-session"})),
    ) is True


def test_boot_marker_present_rejects_missing_marker(tmp_path: Path):
    (tmp_path / "provider_output_get_url.txt").write_text("https://store.example/output.zip?sig=secret")

    assert boot_marker_present(tmp_path, urlopen=_urlopen_for(_marker_zip(None))) is False


def test_boot_marker_present_rejects_stale_launch_session(tmp_path: Path):
    (tmp_path / "provider_output_get_url.txt").write_text("https://store.example/output.zip?sig=secret")

    assert boot_marker_present(
        tmp_path,
        expected_launch_session_id="fresh-session",
        urlopen=_urlopen_for(_marker_zip({"launch_session_id": "old-session"})),
    ) is False


def test_boot_marker_present_fails_closed_without_signed_url(tmp_path: Path):
    assert boot_marker_present(tmp_path, urlopen=_urlopen_for(_marker_zip({"phase": "container_bash_started"}))) is False


def test_boot_marker_present_fails_closed_on_url_errors(tmp_path: Path):
    (tmp_path / "provider_output_get_url.txt").write_text("https://store.example/output.zip?sig=secret")

    def _raise(_url, timeout=None):  # noqa: ANN001 - mirrors urllib.request.urlopen
        raise PermissionError("expired presigned url")

    assert boot_marker_present(tmp_path, urlopen=_raise) is False


# ----------------------------- ProviderCircuitBreaker -----------------------------

def test_breaker_trips_after_dud_rate_exceeds_threshold():
    cb = ProviderCircuitBreaker(window=10, dud_rate_threshold=0.5, min_samples=3)
    assert cb.is_tripped("vast") is False  # no samples => presumed healthy
    cb.record_dud("vast")
    cb.record_dud("vast")
    assert cb.is_tripped("vast") is False  # only 2 samples < min_samples
    cb.record_dud("vast")
    assert cb.is_tripped("vast") is True   # 3 duds, rate 1.0 >= 0.5
    # recovery: enough fresh successes drag the recent dud-rate back under threshold
    for _ in range(4):
        cb.record_success("vast")
    # recent window: d d d s s s s -> 3/7 ~= 0.43 < 0.5
    assert cb.is_tripped("vast") is False


def test_breaker_window_only_counts_recent_outcomes():
    cb = ProviderCircuitBreaker(window=5, dud_rate_threshold=0.5, min_samples=3)
    for _ in range(5):
        cb.record_success("runpod")     # window full of successes
    cb.record_dud("runpod")
    cb.record_dud("runpod")             # recent5: s s s d d -> 0.4
    assert cb.is_tripped("runpod") is False
    cb.record_dud("runpod")             # recent5: s s d d d -> 0.6 >= 0.5
    assert cb.is_tripped("runpod") is True


def test_breaker_partition_splits_runnable_from_tripped():
    cb = ProviderCircuitBreaker(min_samples=3, dud_rate_threshold=0.5)
    for _ in range(3):
        cb.record_dud("bad")
    good, bad = FakeProvider("good"), FakeProvider("bad")
    runnable, skipped = cb.partition([good, bad])
    assert [p.name for p in runnable] == ["good"]
    assert [p.name for p in skipped] == ["bad"]


def test_breaker_order_puts_healthiest_first():
    cb = ProviderCircuitBreaker(min_samples=1, dud_rate_threshold=1.5)  # never trips
    cb.record_success("runpod")
    cb.record_dud("runpod")             # 0.5
    cb.record_success("vast")
    cb.record_success("vast")           # 0.0
    rp, vs = FakeProvider("runpod"), FakeProvider("vast")
    assert [p.name for p in cb.order([rp, vs])] == ["vast", "runpod"]


def test_breaker_snapshot_reports_lifetime_counts_and_trip_state():
    cb = ProviderCircuitBreaker(min_samples=2, dud_rate_threshold=0.5)
    cb.record_success("runpod")
    cb.record_success("runpod")
    cb.record_dud("runpod")             # 1 dud / 3 == 0.33, does not EXCEED 0.5 -> healthy
    cb.record_dud("vast")
    cb.record_dud("vast")               # 2 duds / 2 == 1.0 -> tripped
    snap = cb.snapshot()
    assert snap["runpod"]["success"] == 2 and snap["runpod"]["dud"] == 1
    assert snap["runpod"]["tripped"] is False
    assert snap["vast"]["dud"] == 2 and snap["vast"]["tripped"] is True


# ----------------------------- race_launch core -----------------------------

def test_race_returns_fast_booter_and_terminates_the_rest(tmp_path: Path):
    """The headline behavior: fast booter wins, the launched loser is torn down, and a
    circuit-broken provider is never even launched."""
    fast = FakeProvider("fast", boots=True, marker_after=1)
    dud = FakeProvider("dud", boots=False)              # launches an instance, never boots
    tripped = FakeProvider("tripped", boots=True, marker_after=1)

    cb = ProviderCircuitBreaker(min_samples=3, dud_rate_threshold=0.5)
    for _ in range(3):
        cb.record_dud("tripped")                        # trip it so the racer skips it

    res = race_launch(
        [fast, dud, tripped], request={"spec": 1}, marker_check=_marker_check,
        marker_timeout=5, job_dir=tmp_path, poll_interval=0.01,
        circuit_breaker=cb, sleep=_NO_SLEEP,
    )

    assert res["status"] == "launched"
    assert res["provider"] == "fast"
    assert res["instance_id"] == "fast-iid"
    assert res["winner_provider"] is fast               # live object for watch+collect
    # winner is never terminated
    assert fast.terminate_calls == []
    # the dud launched a real instance and lost the race -> it is torn down (no leak)
    assert dud.terminate_calls == ["dud-iid"]
    # the tripped provider is skipped entirely: never launched, nothing to terminate
    assert tripped.launch_calls == 0
    assert tripped.terminate_calls == []
    assert "tripped" in res["skipped"]


def test_race_returns_first_marker_even_when_a_slower_provider_also_boots(tmp_path: Path):
    fast = FakeProvider("fast", boots=True, marker_after=1)
    slow = FakeProvider("slow", boots=True, marker_after=50)   # would boot, but much later
    res = race_launch(
        [fast, slow], request={}, marker_check=_marker_check,
        marker_timeout=5, job_dir=tmp_path, poll_interval=0.001,  # tiny real sleep
    )
    assert res["provider"] == "fast"
    assert slow.terminate_calls == ["slow-iid"]               # slower booter loses, torn down


def test_race_does_not_wait_out_a_degraded_providers_full_timeout(tmp_path: Path):
    """Regression guard for the stall this whole module exists to kill: a non-booting
    provider with a huge marker_timeout must NOT delay the winner. Real sleep is used so
    the bound is wall-clock; the racer must abort the loser's poll as soon as fast wins."""
    fast = FakeProvider("fast", boots=True, marker_after=1)
    slow = FakeProvider("slow", boots=False)                  # launches, never boots
    t0 = time.monotonic()
    res = race_launch(
        [fast, slow], request={}, marker_check=_marker_check,
        marker_timeout=30, job_dir=tmp_path, poll_interval=0.02,
    )
    elapsed = time.monotonic() - t0
    assert res["provider"] == "fast"
    assert slow.terminate_calls == ["slow-iid"]
    assert elapsed < 5.0          # nowhere near the 30s timeout -> the loser was cut short


def test_race_blocked_when_every_provider_duds_and_no_instances_leak(tmp_path: Path):
    no_capacity = FakeProvider("no_capacity", launches=False)   # never gets an instance
    never_boots = FakeProvider("never_boots", boots=False)      # gets one, never boots
    cb = ProviderCircuitBreaker()
    res = race_launch(
        [no_capacity, never_boots], request={}, marker_check=_marker_check,
        marker_timeout=0.05, job_dir=tmp_path, poll_interval=0.01,
        circuit_breaker=cb, sleep=_NO_SLEEP,
    )
    assert res["status"] == "blocked"
    assert res["provider"] is None and res["instance_id"] is None
    assert res["reason"] == "all_providers_dudded"
    # the launched-but-dead instance is still torn down; the one that never launched isn't
    assert never_boots.terminate_calls == ["never_boots-iid"]
    assert no_capacity.terminate_calls == []
    # both fed the breaker as duds (deterministic: there is no winner to abort them)
    snap = cb.snapshot()
    assert snap["no_capacity"]["dud"] >= 1
    assert snap["never_boots"]["dud"] >= 1


def test_race_feeds_circuit_breaker_winner_success_and_launch_dud(tmp_path: Path):
    """Deterministic breaker feedback: a launch-blocked provider is always a dud, the
    winner is always a success — independent of thread scheduling."""
    fast = FakeProvider("fast", boots=True, marker_after=1)
    no_cap = FakeProvider("no_cap", launches=False)
    cb = ProviderCircuitBreaker()
    res = race_launch(
        [fast, no_cap], request={}, marker_check=_marker_check,
        marker_timeout=1, job_dir=tmp_path, poll_interval=0.01,
        circuit_breaker=cb, sleep=_NO_SLEEP,
    )
    assert res["provider"] == "fast"
    snap = cb.snapshot()
    assert snap["fast"]["success"] == 1
    assert snap["no_cap"]["dud"] == 1
    assert no_cap.terminate_calls == []   # nothing was launched, nothing to terminate


# ----------------------------- race_launch plumbing/edges -----------------------------

def test_race_with_no_providers_is_blocked(tmp_path: Path):
    res = race_launch([], request={}, marker_check=_marker_check, marker_timeout=1,
                      job_dir=tmp_path)
    assert res["status"] == "blocked"
    assert res["reason"] == "no_providers"


def test_race_passes_cold_flag_and_per_provider_request(tmp_path: Path):
    fast = FakeProvider("fast", boots=True, marker_after=1)
    other = FakeProvider("other", boots=False)
    seen = {}

    def request_for(provider):
        seen[provider.name] = True
        return {"built_for": provider.name}

    res = race_launch(
        [fast, other], request=request_for, marker_check=_marker_check,
        marker_timeout=0.05, job_dir=tmp_path, poll_interval=0.01, cold=True,
        sleep=_NO_SLEEP,
    )
    assert res["provider"] == "fast"
    assert seen == {"fast": True, "other": True}            # request built per provider
    assert fast.last_request == {"built_for": "fast"}
    assert fast.launch_cold is True                          # cold forwarded to .launch


def test_race_forwards_launch_kwargs_to_capable_provider(tmp_path: Path):
    class KwProvider(FakeProvider):
        def __init__(self, name):
            super().__init__(name, boots=True, marker_after=1)
            self.launch_kwargs = None

        def launch(
            self,
            job_dir,
            request,
            *,
            cold=False,
            allow_cold_fallback=True,
            provider_hint=None,
        ):
            self.launch_calls += 1
            self.launch_cold = cold
            self.last_request = request
            self.launch_kwargs = {
                "allow_cold_fallback": allow_cold_fallback,
                "provider_hint": provider_hint,
            }
            assert isinstance(job_dir, Path)
            return {
                "status": "launched",
                "instance_id": f"{self.name}-iid",
                "mode": "warm_restart",
            }

    provider = KwProvider("runpod")

    res = race_launch(
        [provider],
        request={},
        marker_check=_marker_check,
        marker_timeout=1,
        job_dir=tmp_path,
        poll_interval=0.01,
        launch_kwargs=lambda p: {
            "allow_cold_fallback": False,
            "provider_hint": p.name,
        },
        sleep=_NO_SLEEP,
    )

    assert res["status"] == "launched"
    assert provider.launch_kwargs == {
        "allow_cold_fallback": False,
        "provider_hint": "runpod",
    }
    assert res["contenders"][0]["launch_kwargs"] == {
        "allow_cold_fallback": False,
        "provider_hint": "runpod",
    }


def test_race_launch_kwargs_are_compatible_with_legacy_cold_only_provider(tmp_path: Path):
    legacy = FakeProvider("legacy", boots=True, marker_after=1)

    res = race_launch(
        [legacy],
        request={},
        marker_check=_marker_check,
        marker_timeout=1,
        job_dir=tmp_path,
        poll_interval=0.01,
        launch_kwargs={"allow_cold_fallback": False},
        sleep=_NO_SLEEP,
    )

    assert res["status"] == "launched"
    assert legacy.launch_calls == 1
    assert res["contenders"][0]["launch_kwargs"] == {"allow_cold_fallback": False}
    assert "launch_kwargs_legacy_fallback" in res["contenders"][0]


def test_race_stops_warm_loser_instead_of_terminating(tmp_path: Path):
    class WarmLoser(FakeProvider):
        def __init__(self, name):
            super().__init__(name, boots=False)
            self.stop_calls = []

        def launch(self, job_dir, request, *, cold=False):
            self.launch_calls += 1
            self.launch_cold = cold
            self.last_request = request
            assert isinstance(job_dir, Path)
            return {"status": "launched", "instance_id": f"{self.name}-iid", "mode": "warm_restart"}

        def stop(self, instance_id):
            self.stop_calls.append(instance_id)
            return {"status": "stopped", "instance_id": instance_id}

    fast = FakeProvider("fast", boots=True, marker_after=1)
    warm_loser = WarmLoser("warm")

    res = race_launch(
        [fast, warm_loser],
        request={},
        marker_check=_marker_check,
        marker_timeout=0.05,
        job_dir=tmp_path,
        poll_interval=0.01,
        sleep=_NO_SLEEP,
    )

    assert res["provider"] == "fast"
    assert warm_loser.stop_calls == ["warm-iid"]
    assert warm_loser.terminate_calls == []
    warm_rec = next(c for c in res["contenders"] if c["provider"] == "warm")
    assert warm_rec["teardown_action"] == "stop"
    assert warm_rec["stopped"]["status"] == "stopped"


def test_race_records_warm_stop_failure_without_sinking_winner(tmp_path: Path):
    class StopFails(FakeProvider):
        def __init__(self, name):
            super().__init__(name, boots=False)

        def launch(self, job_dir, request, *, cold=False):
            self.launch_calls += 1
            return {"status": "launched", "instance_id": f"{self.name}-iid", "mode": "warm_restart"}

        def stop(self, _instance_id):
            raise RuntimeError("stop api failed")

    fast = FakeProvider("fast", boots=True, marker_after=1)
    loser = StopFails("warm")

    res = race_launch(
        [fast, loser],
        request={},
        marker_check=_marker_check,
        marker_timeout=0.05,
        job_dir=tmp_path,
        poll_interval=0.01,
        sleep=_NO_SLEEP,
    )

    assert res["status"] == "launched"
    warm_rec = next(c for c in res["contenders"] if c["provider"] == "warm")
    assert warm_rec["teardown_action"] == "stop"
    assert warm_rec["stopped"]["status"] == "stop_failed"


def test_race_terminate_losers_false_keeps_instances_but_reports_them(tmp_path: Path):
    fast = FakeProvider("fast", boots=True, marker_after=1)
    loser = FakeProvider("loser", boots=False)
    res = race_launch(
        [fast, loser], request={}, marker_check=_marker_check,
        marker_timeout=0.05, job_dir=tmp_path, poll_interval=0.01,
        terminate_losers=False, sleep=_NO_SLEEP,
    )
    assert res["provider"] == "fast"
    assert loser.terminate_calls == []                       # left running by request
    reported = {c["provider"]: c for c in res["contenders"]}
    assert reported["loser"]["instance_id"] == "loser-iid"   # but still surfaced


def test_race_treats_launch_exception_as_a_dud_not_a_crash(tmp_path: Path):
    class Boom(FakeProvider):
        def launch(self, job_dir, request, *, cold=False):
            raise RuntimeError("provider api exploded")

    boom = Boom("boom")
    fast = FakeProvider("fast", boots=True, marker_after=1)
    res = race_launch(
        [boom, fast], request={}, marker_check=_marker_check,
        marker_timeout=1, job_dir=tmp_path, poll_interval=0.01, sleep=_NO_SLEEP,
    )
    assert res["provider"] == "fast"                         # the crash didn't sink the race
    boom_rec = next(c for c in res["contenders"] if c["provider"] == "boom")
    assert boom_rec["instance_id"] is None
    assert "launch_raised" in (boom_rec.get("reason") or "")


def test_race_isolates_each_contender_with_its_own_job_dir(tmp_path: Path):
    """Real providers write started_*_id.txt into job_dir; the racer must give each its own
    subdir so concurrent launches can't clobber one another."""
    seen_dirs = []

    class DirCapturingProvider(FakeProvider):
        def launch(self, job_dir, request, *, cold=False):
            seen_dirs.append(job_dir)
            return super().launch(job_dir, request, cold=cold)

    a = DirCapturingProvider("a", boots=True, marker_after=1)
    b = DirCapturingProvider("b", boots=True, marker_after=1)
    race_launch([a, b], request={}, marker_check=_marker_check, marker_timeout=0.05,
                job_dir=tmp_path, poll_interval=0.01, sleep=_NO_SLEEP)
    assert len(seen_dirs) == 2
    assert len(set(seen_dirs)) == 2                          # distinct dirs
    for d in seen_dirs:
        assert d.is_dir() and tmp_path in d.parents


def test_race_actually_launches_providers_concurrently(tmp_path: Path):
    """If launches were sequential, the second provider couldn't enter launch() until the
    first returned. A barrier proves both are in-flight at once."""
    barrier = threading.Barrier(2, timeout=5)

    class BarrierProvider(FakeProvider):
        def launch(self, job_dir, request, *, cold=False):
            barrier.wait()   # blocks until BOTH providers reach here -> proves parallelism
            return super().launch(job_dir, request, cold=cold)

    p1 = BarrierProvider("p1", boots=True, marker_after=1)
    p2 = BarrierProvider("p2", boots=True, marker_after=1)
    res = race_launch([p1, p2], request={}, marker_check=_marker_check, marker_timeout=1,
                      job_dir=tmp_path, poll_interval=0.01, sleep=_NO_SLEEP)
    assert res["status"] == "launched"   # didn't deadlock -> they launched concurrently
