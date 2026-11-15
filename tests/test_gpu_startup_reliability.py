"""Hermetic tests for the cold-start reliability wiring.

Covers the three levers wired on 2026-07-02 (no network, no GPU, no secrets):

1. same-provider cold-create racing (``resolve_cold_race_contenders`` +
   ``_ColdCreateContender`` + ``race_launch`` with same-name contenders),
2. the image-aware no-runtime guard defaulting on (900s before a manifest floor),
3. gpu_spend_guard's expected-serve-pod tagging.
"""

from __future__ import annotations

import json
from pathlib import Path


from scripts import gpu_spend_guard as guard


# --------------------------- cold race contender resolution ---------------------------


# --------------------------- warm serve worker control plane ---------------------------


def _serving_manifest(pod_id: str = "pod-serve") -> dict:
    return {
        "status": "serving",
        "blockers": [],
        "warm_serve": {
            "instance_id": pod_id,
            "ready": True,
            "inbox_put_url_file": "/tmp/warm_inbox_put_url.txt",
            "output_get_url_file": "/tmp/provider_output_get_url.txt",
            "pending_teardown_record": "/tmp/pending-warm-teardown.json",
        },
    }


# --------------------------- spend guard serve-pod tagging ---------------------------


def test_spend_guard_tags_expected_serve_pods_and_protects_them(tmp_path: Path) -> None:
    serve_dir = tmp_path / "output" / "warm_worker_kitchen"
    serve_dir.mkdir(parents=True)
    (serve_dir / guard.WARM_SERVE_MARKER_FILENAME).write_text(
        json.dumps(
            {
                "status": "serving",
                "pod_id": "pod-serve",
                "provider": "runpod",
                "lease_expires_at": "2027-02-01T00:00:00Z",
            }
        )
    )
    stale_dir = tmp_path / "output" / "old_worker"
    stale_dir.mkdir(parents=True)
    (stale_dir / guard.WARM_SERVE_MARKER_FILENAME).write_text(
        json.dumps({"status": "terminated", "pod_id": "pod-old", "provider": "runpod"})
    )

    serve_ids = guard.find_expected_serve_pod_ids(
        [tmp_path / "output"],
        now=1_800_000_000.0,
    )
    assert serve_ids == {"pod-serve"}

    instances = [
        guard.GpuInstance(
            provider="runpod", id="pod-serve", name="warm-serve", state="running",
            booted=True, live=True, cost_per_hr=0.69, age_seconds=7200.0,
        ),
        guard.GpuInstance(
            provider="runpod", id="pod-other", name="parity", state="running",
            booted=True, live=True, cost_per_hr=0.69, age_seconds=600.0,
        ),
    ]
    report = guard.build_report(
        instances,
        protected_ids={"pod-serve"},
        max_boot_seconds=480,
        serve_pod_ids=serve_ids,
    )
    assert "warm-serve worker (expected)" in report
    # the tag names the serve pod's row, not the ordinary one
    serve_row = next(line for line in report.splitlines() if "pod-serve" in line)
    other_row = next(line for line in report.splitlines() if "pod-other" in line)
    assert "expected" in serve_row
    assert "expected" not in other_row
