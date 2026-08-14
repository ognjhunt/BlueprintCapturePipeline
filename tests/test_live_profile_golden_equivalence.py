"""Factoring the builders must not change a single byte a lane launches with.

A launch profile is published once and read on every later launch, and its
digest binds the allocator argv the paid boundary opens. Two lanes had already
run for real against profiles the hand-written builders produced -- one of them
completed a $0.135 GPU run -- so a refactor that changed any field would be a
silent change to what those lanes do, not a refactor.

These goldens were captured from the pre-refactor builders at `origin/main`.
See `_normalized` for the two things that legitimately differ between two runs
of one builder and are therefore not compared.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDENS = REPO_ROOT / "tests" / "fixtures" / "live_profile_goldens"
COMMIT = "0" * 40
URI = f"https://raw.githubusercontent.com/example/repo/{COMMIT}/request.json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # `dataclasses` resolves annotations through `sys.modules[cls.__module__]`,
    # so a builder declaring a dataclass cannot be loaded without this.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _normalized(profile: dict, root: Path) -> dict:
    """Drop what legitimately varies between two runs of the same builder.

    Two things do. Where the inputs live, and the digest of any input whose own
    contents embed an absolute path -- a bundle receipt records `bundle_path`,
    so its bytes differ under a different temp directory even though the
    builder digested exactly the same file.

    Digests are aliased by order of first appearance rather than dropped, so
    *which fields share a digest* is still pinned. That aliasing is what would
    catch a refactor that digested the preflight where it used to digest the
    receipt.
    """

    text = json.dumps(profile, indent=1, sort_keys=True).replace(str(root), "{ROOT}")
    assert str(root) not in text, "a real path survived placeholdering"
    seen: dict[str, str] = {}

    def alias(match: re.Match[str]) -> str:
        value = match.group(0)
        if value not in seen:
            seen[value] = f"sha256:<{len(seen)}>"
        return seen[value]

    return json.loads(re.sub(r"sha256:[0-9a-f]{64}", alias, text))


def _golden(name: str) -> dict:
    return json.loads((GOLDENS / f"{name}.json").read_text(encoding="utf-8"))


def _compare(observed: dict, golden: dict) -> None:
    assert observed.pop("profile_digest", None), "a profile must carry its digest"
    golden = dict(golden)
    golden.pop("profile_digest", None)
    assert observed == golden


def test_retained_scene_profile_is_byte_identical_to_the_hand_written_builder(
    tmp_path: Path,
) -> None:
    builder = _load("build_retained_scene_render_live_profile")
    root = tmp_path.resolve()

    authority = root / "execution_authority.json"
    authority.write_text(
        json.dumps(
            {
                "schema_version": "third_scene_dual_task_execution_authority.v1",
                "paid_compute": {
                    "provider": "vast",
                    "external_instance_allowlist": [47373597, 47569249],
                },
            }
        ),
        encoding="utf-8",
    )
    bundle = root / "bundle.zip"
    bundle.write_bytes(b"retained-scene-bundle")
    request = root / "request.json"
    request.write_text(json.dumps({"schema_version": "request.v1"}), encoding="utf-8")
    attempt = root / "attempt_authority.json"
    attempt.write_text(json.dumps({"schema_version": "attempt.v1"}), encoding="utf-8")
    receipt = root / "bundle_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "ready",
                "probe_kind": "adp-retained-scene-gpu-render",
                "blueprint_commit": COMMIT,
                "hard_total_spend_cap_usd": 12.0,
                "bundle_path": str(bundle),
                "bundle_sha256": "sha256:"
                + hashlib.sha256(bundle.read_bytes()).hexdigest(),
                "execution_authority": {
                    "path": str(authority),
                    "sha256": "sha256:"
                    + hashlib.sha256(authority.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    profile = builder.build_retained_scene_render_live_profile(
        bundle_receipt_path=receipt,
        request_manifest_path=request,
        attempt_authority_path=attempt,
        source_commit=COMMIT,
        raw_manifest_uri=URI,
        max_hourly_rate_usd=2.0,
        hard_ttl_seconds=3600,
        revision="g1",
    )

    _compare(_normalized(profile, root), _golden("retained_scene_render"))


def test_content_agents_profile_is_byte_identical_to_the_hand_written_builder(
    tmp_path: Path,
) -> None:
    builder = _load("build_content_agents_live_profile")
    root = tmp_path.resolve()

    bundle = root / "bundle.zip"
    bundle.write_bytes(b"content-agents-bundle")
    digest = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()
    receipt = root / "adp_content_agents_bundle_receipt.json"
    receipt.write_text(
        json.dumps(
            {"status": "ready", "bundle_path": str(bundle), "bundle_sha256": digest}
        ),
        encoding="utf-8",
    )
    preflight = root / "preflight.json"
    preflight.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    attempt = root / "attempt_authority.json"
    attempt.write_text(
        json.dumps(
            {
                "hard_attempt_spend_cap_usd": 3.0,
                "maximum_hourly_rate_usd": 1.0,
                "maximum_single_resource_ttl_seconds": 7200,
                "bundle_sha256": digest,
            }
        ),
        encoding="utf-8",
    )

    profile = builder.build_content_agents_live_profile(
        bundle_receipt_path=receipt,
        config_preflight_path=preflight,
        attempt_authority_path=attempt,
        source_commit=COMMIT,
        candidate_id="a-golden",
        raw_manifest_uri=URI,
        revision="g1",
        max_hourly_rate_usd=1.0,
        max_spend_usd=3.0,
        hard_ttl_seconds=7200,
    )

    _compare(_normalized(profile, root), _golden("content_agents"))


@pytest.mark.parametrize(
    "name", ["retained_scene_render", "content_agents"], ids=["retained-scene", "content-agents"]
)
def test_every_golden_still_asks_for_the_controls_that_make_a_run_provable(
    name: str,
) -> None:
    """If a refactor quietly dropped one of these, the goldens would too."""

    profile = _golden(name)
    controls = profile["required_controls"]
    assert controls["teardown_required"] is True
    assert controls["provider_zero_required"] is True
    assert controls["watchdog_required"] is True
    assert controls["retry_cap"] == 0
    assert profile["allocator"]["retry_cap"] == 0
    assert sorted(profile["terminal_contract"]["required_path_fields"]) == [
        "artifact_manifest_path",
        "teardown_manifest_path",
    ]
