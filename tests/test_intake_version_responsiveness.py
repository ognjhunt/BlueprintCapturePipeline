"""The release identity remains observable while synchronous intake work is busy."""
import asyncio
import threading

import anyio
import httpx

from blueprint_pipeline import live_pipeline_intake_service as service


def test_version_does_not_wait_for_the_shared_intake_thread_pool(monkeypatch):
    monkeypatch.setattr(service, "deployment_identity_payload", lambda: {
        "commit_proven": True, "source_commit": "a" * 40, "blockers": []})

    async def exercise():
        limiter = anyio.to_thread.current_default_thread_limiter()
        previous = limiter.total_tokens
        limiter.total_tokens = 1
        occupied, release = threading.Event(), threading.Event()

        def intake_work():
            occupied.set()
            release.wait(5)

        task = asyncio.create_task(anyio.to_thread.run_sync(intake_work))
        try:
            while not occupied.is_set():
                await asyncio.sleep(0.001)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.create_app()),
                                        base_url="http://test") as client:
                response = await asyncio.wait_for(client.get("/api/live-pipeline/version"), 0.5)
            assert response.status_code == 200
            assert response.json()["source_commit"] == "a" * 40
        finally:
            release.set()
            await task
            limiter.total_tokens = previous

    asyncio.run(exercise())


def test_slow_identity_probe_is_bounded_and_not_multiplied_by_polling(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def slow_probe():
        calls.append(True)
        entered.set()
        release.wait(5)
        return {"commit_proven": True, "source_commit": "b" * 40, "blockers": []}

    monkeypatch.setattr(service, "deployment_identity_payload", slow_probe)
    monkeypatch.setattr(service, "DEPLOYMENT_DIAGNOSTIC_TIMEOUT_SECONDS", 0.03, raising=False)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.create_app()),
                                    base_url="http://test") as client:
            try:
                responses = await asyncio.wait_for(asyncio.gather(*(
                    client.get("/api/live-pipeline/version") for _ in range(12))), 0.5)
                assert entered.is_set() and len(calls) == 1
                for response in responses:
                    assert response.status_code == 503
                    payload = response.json()
                    assert payload["commit_proven"] is False and payload["source_commit"] is None
                    assert payload["blockers"] == ["deployment_identity_probe_timeout"]
                    assert payload["disk_headroom"]["status"] == "unknown_fail_closed"
                    assert payload["disk_headroom"]["refused_roles"] == list(service.CHAIN_ROLES)
            finally:
                release.set()
            await asyncio.sleep(0.03)
            response = await client.get("/api/live-pipeline/version")
            assert response.status_code == 200
            assert response.json()["source_commit"] == "b" * 40

    asyncio.run(exercise())
