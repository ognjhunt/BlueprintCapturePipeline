from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paired_target_native_arena_request import (
    PairedTargetNativeArenaRequestError,
    _pose_from_matrix,
    materialize_paired_target_native_arena_requests,
)


def test_pose_roundtrip_preserves_registered_quarter_turn() -> None:
    import numpy as np

    matrix = np.asarray(
        [
            [0.0, -1.0, 0.0, 13.5],
            [1.0, 0.0, 0.0, 7.2],
            [0.0, 0.0, 1.0, 0.825],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    pose = _pose_from_matrix(matrix)

    assert pose["position_world_m"] == [13.5, 7.2, 0.825]
    assert pose["orientation_xyzw"] == pytest.approx(
        [0.0, 0.0, 2**-0.5, 2**-0.5]
    )


def test_invalid_task_count_fails_without_output(tmp_path: Path) -> None:
    output = tmp_path / "output"

    with pytest.raises(
        PairedTargetNativeArenaRequestError,
        match="paired_target_arena_request_inputs_invalid",
    ):
        materialize_paired_target_native_arena_requests(
            construction_bindings_path=tmp_path / "missing.json",
            task_inputs=[],
            evidence_root=tmp_path,
            output_root=output,
        )

    assert not output.exists()


def test_invalid_six_task_count_fails_before_source_read(tmp_path: Path) -> None:
    output = tmp_path / "output"

    with pytest.raises(
        PairedTargetNativeArenaRequestError,
        match="paired_target_arena_request_inputs_invalid",
    ):
        materialize_paired_target_native_arena_requests(
            construction_bindings_path=tmp_path / "missing.json",
            task_inputs=[{} for _ in range(6)],
            evidence_root=tmp_path,
            output_root=output,
        )

    assert not output.exists()


def test_real_two_task_requests_bind_registered_root_and_compile_packets() -> None:
    """Regression over the committed retained public-scene packet."""

    root = (
        Path(__file__).resolve().parent
        / "fixtures/paired_target_native_arena_requests_v2_8f181229"
    )
    receipt = json.loads(
        (root / "paired_target_native_arena_requests.v1.json").read_text()
    )
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    assert receipt["replacement_object_count"] == 2
    requests = {
        task["task_id"]: json.loads(
            (
                root
                / task["task_id"]
                / "native_task_arena_packet_request.v1.json"
            ).read_text()
        )
        for task in receipt["tasks"]
    }
    for task_id, request in requests.items():
        assert request["request_digest"] == canonical_digest(
            request, digest_field="request_digest"
        )
        assert len(
            [asset for asset in request["assets"] if asset["semantic_role"] == "replacement"]
        ) == 2
        assert request["task_spec"]["subject_asset_id"] in {
            asset["asset_id"]
            for asset in request["assets"]
            if asset["semantic_role"] == "replacement"
        }
    notebook = requests["task_b_notebook_relocation"]
    notebook_asset = next(
        asset
        for asset in notebook["assets"]
        if asset.get("asset_id") == "840920_simready_notebook_candidate"
    )
    assert notebook_asset["pose_world"]["orientation_xyzw"] == pytest.approx(
        [0.0, 0.0, 2**-0.5, 2**-0.5]
    )
    assert notebook["task_spec"]["start_pose_world"][3:] == [0, 0, 0, 1]
    washer = requests["task_a_washer_door_open"]
    path = washer["task_spec"]["interaction_affordance"]["joint_contact_path"]
    assert len(path) == 29
    assert max(
        abs(right["joint_positions"][joint] - left["joint_positions"][joint])
        for left, right in zip(path, path[1:])
        for joint in left["joint_positions"]
    ) <= 0.03


def test_missing_receipt_path_is_a_named_refusal_not_a_typeerror() -> None:
    """A task input omitting a required receipt path names the field's code.

    Both articulated (kinematic path) and rigid (support) inputs reach the
    reader through ``.get(...)``, so an absent key used to surface as
    ``TypeError: argument should be a str ... not 'NoneType'`` -- useless to
    an operator and not a fail-closed refusal.  Pin the named code instead.
    """

    from blueprint_pipeline.paired_target_native_arena_request import _bound_json

    for code in (
        "paired_target_arena_request_kinematic_invalid",
        "paired_target_arena_request_support_invalid",
    ):
        with pytest.raises(PairedTargetNativeArenaRequestError, match=code):
            _bound_json(
                None,
                schema="paired_target_articulated_kinematic_path.v1",
                digest_field="receipt_digest",
                code=code,
            )


def test_a_receipt_supplied_as_a_usd_asset_is_refused_before_any_spend(
    tmp_path: Path,
) -> None:
    """JSON staged under a ``.usdz`` name must not reach a rented GPU.

    On 2026-08-18 a task input named the appearance receipt instead of the
    exported asset. The request compiler checked only that the path was a
    readable file, the packet digested exactly what it copied, and the
    terminal-contract rehearsal passed, so 3.8 kB of JSON went to a provider
    as ``scene_appearance.usdz`` and the run died in ``UsdStage::Open``.
    """

    from blueprint_pipeline.native_task_arena_packet import (
        usd_payload_format_matches,
    )

    receipt = tmp_path / "native_appearance_export.v1.json"
    receipt.write_text('{\n  "schema_version": "x"\n}\n', encoding="utf-8")
    assert usd_payload_format_matches(receipt, "scene_appearance.usdz") is False

    # the real exported asset is a zip, and each sibling format is checked by
    # its own magic rather than by its extension alone
    usdz = tmp_path / "repaired_scene.usdz"
    usdz.write_bytes(b"PK\x03\x04rest-of-archive")
    assert usd_payload_format_matches(usdz, "scene_appearance.usdz") is True

    usda = tmp_path / "replacement.usda"
    usda.write_text("#usda 1.0\n", encoding="utf-8")
    assert usd_payload_format_matches(usda, "replacement__x.usda") is True
    assert usd_payload_format_matches(usda, "scene_collision.usdc") is False

    usdc = tmp_path / "scene.usdc"
    usdc.write_bytes(b"PXR-USDC\x00\x00")
    assert usd_payload_format_matches(usdc, "scene_collision.usdc") is True

    # an unreadable path is refused, never treated as acceptable
    assert usd_payload_format_matches(tmp_path / "absent.usdz", "a.usdz") is False

    # a format this packet does not stage is left alone rather than guessed at
    other = tmp_path / "notes.txt"
    other.write_text("free text", encoding="utf-8")
    assert usd_payload_format_matches(other, "notes.txt") is True
