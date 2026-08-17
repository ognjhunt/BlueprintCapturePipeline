"""The two lanes #543 and #544 made website-reachable must seal what they write.

Both lanes ran paid attempts for months and named neither
``artifact_manifest_path`` nor ``teardown_manifest_path`` in the file their
allocator result lands in. That cost nothing while neither had a live profile
builder: nothing read the file. #543 and #544 gave them builders, and the
``terminal_contract.required_path_fields`` of those profiles are read straight
off that file -- so from that moment a run could rent a GPU, execute, tear down
with provider zero confirmed, and still report
``allocator_terminal_artifact_missing:`` for both fields.

These tests pin the file by the fields it carries rather than by any helper
name. A prior session grepped for ``seal_lane_terminal_artifacts``, concluded
five lanes were broken, and was wrong; the emitted field is the thing the launch
actually reads.

Nothing here touches a provider: every path exercised refuses before the paid
boundary, and the assertions include that it did.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import (
    nvidia_warehouse_native_camera_gpu_admission as native_camera,
)
from blueprint_pipeline import openpi_policy_ranking_runpod as openpi_runpod


LANES = pytest.mark.parametrize(
    "module",
    (native_camera, openpi_runpod),
    ids=lambda m: m.__name__.rsplit(".", 1)[-1],
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@LANES
def test_the_terminal_result_names_both_manifest_fields(module, tmp_path: Path) -> None:
    """The file the launch contract reads must name the fields it reads."""

    adapter_output = tmp_path / "allocator" / "result.json"
    adapter_output.parent.mkdir(parents=True)

    sealed = module._seal_terminal({"status": "blocked"}, adapter_output)

    assert "artifact_manifest_path" in sealed
    assert "teardown_manifest_path" in sealed


@LANES
def test_an_attempt_that_never_allocated_seals_explicit_absence(
    module, tmp_path: Path
) -> None:
    """A pre-provider refusal retains an explicit unallocated teardown.

    The direct-provider terminal adapter now materializes the same conventional
    artifact and teardown receipts used by the shared Vast path.  This is
    stronger than naming both fields as ``None``: the launch can distinguish a
    proven pre-allocation refusal from evidence that went missing, and the
    receipt explicitly says that there were no provider IDs or continuing
    spend.
    """

    adapter_output = tmp_path / "allocator" / "result.json"
    adapter_output.parent.mkdir(parents=True)

    sealed = module._seal_terminal(
        {"status": "blocked", "provider_mutations_performed": 0}, adapter_output
    )

    artifact_manifest = Path(sealed["artifact_manifest_path"])
    teardown_manifest = Path(sealed["teardown_manifest_path"])
    assert artifact_manifest.is_file()
    assert teardown_manifest.is_file()

    artifact = json.loads(artifact_manifest.read_text(encoding="utf-8"))
    teardown = json.loads(teardown_manifest.read_text(encoding="utf-8"))
    assert artifact["status"] == "completed"
    assert artifact["blockers"] == []
    assert set(artifact["required_roles"]) == {
        "allocator_adapter_result",
        "teardown_manifest",
    }
    assert teardown["status"] == "not_required_provider_adapter_never_invoked"
    assert teardown["vast_instance_ids"] == []
    assert teardown["teardown_actions_performed"] == []
    assert teardown["continuing_spend_from_this_run"] is False


@LANES
def test_the_seal_never_upgrades_a_status(module, tmp_path: Path) -> None:
    """Sealing records evidence; it must not turn a refusal into a success.

    The shared seal only adds blockers or downgrades, and a lane whose wrapper
    quietly re-stated the status would make every launch above it unreliable in
    the one direction that costs money.
    """

    adapter_output = tmp_path / "allocator" / "result.json"
    adapter_output.parent.mkdir(parents=True)

    sealed = module._seal_terminal(
        {"status": "blocked", "blockers": ["already_refused"]}, adapter_output
    )

    assert sealed["status"] == "blocked"
    assert "already_refused" in sealed["blockers"]


@LANES
def test_the_seal_reads_the_root_the_evidence_actually_lives_under(
    module, tmp_path: Path
) -> None:
    """Defect #501: a lane that seals a root its evidence is not under takes the
    sealer's dry-run path and reports success having sealed nothing.

    Both lanes lay their provider run beside the adapter result, so the attempt
    root is that file's own directory. Evidence written there must be found.
    """

    adapter_output = tmp_path / "allocator" / "result.json"
    provider_run = adapter_output.parent / "vast_provider_run"
    _write(
        provider_run / "vast_provider_adapter_result.json",
        {"status": "blocked", "provider_mutations_performed": 0},
    )
    _write(
        provider_run / "vast_teardown_manifest.json",
        {"continuing_spend_from_this_run": False},
    )

    sealed = module._seal_terminal(
        {"status": "blocked", "instance_id": "i-probe", "cost_usd": 0.0},
        adapter_output,
    )

    assert sealed["teardown_manifest_path"], (
        "the sealer did not find evidence written under the adapter result's own "
        "directory, so it is sealing a different root than the lane writes to"
    )


def test_the_native_camera_lane_seals_the_result_it_really_writes(
    tmp_path: Path,
) -> None:
    """End to end through the lane, on a path that refuses before any provider.

    The unit tests above prove the wrapper is correct; this proves the wrapper
    is actually on the path that writes the launch profile's `result.json`. The
    lane's first three refusals write `admission_out` instead, so the earliest
    terminal write is the not-admitted branch, which is what this reaches.
    """

    root = tmp_path / "allocator"
    root.mkdir()
    adapter_output = root / "result.json"

    result = native_camera.run_native_camera_gpu_lane(
        release_evidence=_write(tmp_path / "release.json", {"status": "blocked"}),
        input_bundle_receipt=_write(tmp_path / "bundle.json", {}),
        preflight_bundle=_write(
            tmp_path / "preflight.json",
            {"provider": "vast", "container_disk_bytes": 0},
        ),
        admission_out=root / "admission.json",
        bound_request_out=root / "bound_request.json",
        adapter_output=adapter_output,
        pod_name=f"{native_camera.CANARY_NAME_PREFIX}probe",
        expected_source_commit="0" * 40,
        execute=False,
        hard_ttl_seconds=600,
        max_spend_usd=1.0,
    )

    assert result["provider_mutations_performed"] == 0
    assert adapter_output.is_file(), "the lane wrote no terminal result at all"

    written = json.loads(adapter_output.read_text(encoding="utf-8"))
    assert "artifact_manifest_path" in written, (
        "the launch profile reads required_path_fields off this exact file; "
        "without this key the run reports allocator_terminal_artifact_missing"
    )
    assert "teardown_manifest_path" in written
