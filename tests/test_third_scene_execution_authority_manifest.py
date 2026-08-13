"""The paid-compute allowlist must live in the repository, not a temp directory.

The retained-scene bundle digest-binds to this authority. During the first
website-triggered paid run a concurrent instance had to be admitted, and the
admission was made in a copy under /private/tmp. The bundle validated against
that copy, so a reboot clearing /private/tmp would have reintroduced
retained_scene_render_execution_authority_invalid with no trace of why.

An allowlist is a spend-relevant decision: it names the instances a prelaunch
inventory guard may see without refusing to allocate. It belongs under review.
"""

from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest

MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "docs/arm_decision_proof_v1/manifests/third_scene_dual_task_execution_authority.v1.json"
)


def _authority() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_authority_digest_binds_its_own_content() -> None:
    authority = _authority()
    assert authority["authority_digest"] == canonical_digest(
        authority, digest_field="authority_digest"
    ), "an editable authority is not an authority; the bundle binds this digest"


def test_allowlisted_instances_are_recorded_here_not_in_a_temp_copy() -> None:
    paid = _authority()["paid_compute"]
    allowlist = paid["external_instance_allowlist"]

    assert allowlist == sorted(set(allowlist)), "allowlist must be sorted and unique"
    assert all(isinstance(item, int) and item > 0 for item in allowlist)
    # Every admitted instance needs a recorded owner, or a future reader cannot
    # tell a deliberate concurrent lane from a forgotten orphan.
    assert str(paid.get("external_instance_owner") or "").strip()


def test_authority_references_no_temporary_or_developer_paths() -> None:
    """A tracked manifest that points into /private/tmp is not reproducible."""
    text = MANIFEST.read_text(encoding="utf-8")

    for fragment in ("/private/tmp", "/tmp/", "/Users/", "/home/"):
        assert fragment not in text, (
            f"{fragment} in a tracked authority makes the bundle unvalidatable "
            "once that directory is cleared"
        )


def test_paid_compute_keeps_its_fail_closed_contract() -> None:
    paid = _authority()["paid_compute"]

    assert paid["provider"] == "vast"
    assert paid["zero_retry"] is True
    assert paid["provider_zero_required_for_lane"] is True
