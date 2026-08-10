"""A bundle may only be reaped when it can be rebuilt."""

from __future__ import annotations

import json

import pytest

from blueprint_pipeline.provider_bundle_reaper import (
    ProviderBundleReaperError,
    reap_provider_bundle,
)


def _job(tmp_path, *, sources_present=True, with_result=True):
    job = tmp_path / "arena_run_rt99"
    bundle = job / "bundle" / "provider_runtime" / "assets"
    bundle.mkdir(parents=True)
    (bundle / "approved_can.usda").write_text("#usda 1.0\n" * 500, encoding="utf-8")
    if with_result:
        (job / "adp_arena_vast_result.json").write_text(
            json.dumps({"status": "blocked"}), encoding="utf-8"
        )
    sources = tmp_path / "scene_probe_v1"
    sources.mkdir()
    if sources_present:
        (sources / "twin.usda").write_text("#usda 1.0\n", encoding="utf-8")
    return job, [sources / "twin.usda"]


def test_reaps_the_bundle_when_every_source_is_still_present(tmp_path):
    job, sources = _job(tmp_path)

    receipt = reap_provider_bundle(job_dir=job, source_paths=sources)

    assert receipt["reaped"] is True
    assert not (job / "bundle").exists()
    assert receipt["reclaimed_bytes"] > 0


def test_keeps_the_bundle_when_a_source_has_gone(tmp_path):
    """Then the bundle is the only copy, and deleting it destroys evidence.

    Bundles rename and derive assets at build time, so a bundle is not a
    duplicate of anything. Across this tree two thirds of sampled bundle
    assets had no identical copy outside a bundle.
    """

    job, sources = _job(tmp_path, sources_present=False)

    receipt = reap_provider_bundle(job_dir=job, source_paths=sources)

    assert receipt["reaped"] is False
    assert (job / "bundle").exists()
    assert any("source_missing" in reason for reason in receipt["retained_because"])


def test_the_run_evidence_is_never_touched(tmp_path):
    job, sources = _job(tmp_path)

    reap_provider_bundle(job_dir=job, source_paths=sources)

    assert (job / "adp_arena_vast_result.json").is_file()


def test_a_run_with_no_result_yet_is_not_reaped(tmp_path):
    """An in-flight run still needs its bundle."""

    job, sources = _job(tmp_path, with_result=False)

    receipt = reap_provider_bundle(job_dir=job, source_paths=sources)

    assert receipt["reaped"] is False
    assert any("no_result" in reason for reason in receipt["retained_because"])


def test_a_missing_job_directory_fails_closed(tmp_path):
    with pytest.raises(ProviderBundleReaperError):
        reap_provider_bundle(job_dir=tmp_path / "nope", source_paths=[])


def test_reaping_is_idempotent(tmp_path):
    job, sources = _job(tmp_path)
    reap_provider_bundle(job_dir=job, source_paths=sources)

    receipt = reap_provider_bundle(job_dir=job, source_paths=sources)

    assert receipt["reaped"] is False
    assert any("already" in reason for reason in receipt["retained_because"])


def test_an_empty_source_list_refuses_rather_than_reaping(tmp_path):
    """"Nothing to check" must not read as "everything checks out"."""

    job, _sources = _job(tmp_path)

    receipt = reap_provider_bundle(job_dir=job, source_paths=[])

    assert receipt["reaped"] is False
    assert (job / "bundle").exists()


def test_the_allocator_reaps_on_completion(monkeypatch, tmp_path):
    """Wired in, not left as a habit - the whole point of the fix.

    A reaper nobody calls is a reaper that does not exist, and this one exists
    because the previous arrangement relied on someone remembering.
    """

    from types import SimpleNamespace

    from blueprint_pipeline import paid_resource_allocator as allocator

    job, sources = _job(tmp_path)
    args = SimpleNamespace(
        adp_job_dir=str(job),
        adp009d_approved_can=str(sources[0]),
        adp009d_sage_collision=None,
        adp009d_harness_manifest=None,
        adp009d_worker_source=None,
        adp009d_runtime_module_source=None,
        adp009d_extra_native=None,
    )

    receipt = allocator._reap_finished_bundle(args)

    assert receipt is not None and receipt["reaped"] is True
    assert not (job / "bundle").exists()


def test_the_allocator_reaper_never_fails_a_finished_run(tmp_path):
    """Reclaiming disk must not turn a completed run into a failed one."""

    from types import SimpleNamespace

    from blueprint_pipeline import paid_resource_allocator as allocator

    args = SimpleNamespace(
        adp_job_dir=str(tmp_path / "does-not-exist"),
        adp009d_approved_can="/nowhere/twin.usda",
        adp009d_sage_collision=None,
        adp009d_harness_manifest=None,
        adp009d_worker_source=None,
        adp009d_runtime_module_source=None,
        adp009d_extra_native=None,
    )

    receipt = allocator._reap_finished_bundle(args)

    assert receipt["reaped"] is False
    assert any("errored" in reason for reason in receipt["retained_because"])
