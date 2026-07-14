import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "scripts" / "probe_production_gpu_warm_bind.py"
SPEC = importlib.util.spec_from_file_location("warm_bind_probe", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_probe_binds_and_releases_without_provider_calls():
    calls = []

    def sender(_base, route, payload, _token):
        calls.append(route)
        if route.endswith("/bind"):
            return {
                "status": "bound_to_ready_worker",
                "worker_id": "worker-1",
                "lease_token": "opaque",
            }
        return {"state": "ready"}

    ticks = iter([0.0, 0.1, 1.0, 1.2, 2.0, 2.3])
    result = MODULE.run_probe(
        base_url="http://127.0.0.1:8790",
        token="t" * 32,
        host_image_id="host-1",
        worker_image_ref="registry/worker@sha256:" + "a" * 64,
        gpu_family="NVIDIA L40S",
        samples=3,
        sender=sender,
        clock=lambda: next(ticks),
    )
    assert result["status"] == "passed"
    assert result["provider_calls_performed"] == 0
    assert result["lease_tokens_recorded"] is False
    assert len(calls) == 6


def test_probe_fails_closed_when_pool_is_cold():
    result = MODULE.run_probe(
        base_url="http://127.0.0.1:8790",
        token="t" * 32,
        host_image_id="host-1",
        worker_image_ref="registry/worker@sha256:" + "a" * 64,
        gpu_family="NVIDIA L40S",
        samples=1,
        sender=lambda *_args: {"status": "queued_waiting_for_warm_worker"},
    )
    assert result["status"] == "failed"
    assert "warm_worker_not_immediately_available" in result["blockers"]
