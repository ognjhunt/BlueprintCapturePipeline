from __future__ import annotations

import json
import zipfile
from pathlib import Path

from blueprint_pipeline.provider_runtime_bundle_contract import (
    wam_registered_alternative_inputs_present,
)


def _write_powered_bundle(path: Path, *, omit_image: bool = False) -> list[str]:
    rows = [
        {"initial_observation_relative_path": f"images/session-{index:02d}/window.png"}
        for index in range(51)
    ]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "provider_runtime/cosmos3_powered_droid/packet.json",
            json.dumps(
                {
                    "schema_version": "policy_ranking_powered_droid_provider_packet.v1",
                    "rows": rows,
                }
            ),
        )
        for index, row in enumerate(rows):
            if omit_image and index == 50:
                continue
            archive.writestr(
                "provider_runtime/cosmos3_powered_droid/"
                + row["initial_observation_relative_path"],
                b"png",
            )
        for name in ("canary_manifest.json", "initial_observation.png", "action_streams.json"):
            archive.writestr(
                "provider_runtime/cosmos3_powered_droid/official_canary/" + name,
                b"{}" if name.endswith(".json") else b"png",
            )
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()


def test_wam_registered_alternative_accepts_standard_and_reference_layouts(tmp_path: Path) -> None:
    unused = tmp_path / "unused.zip"
    standard = [
        "provider_runtime/cosmos3_input/initial_observation.png",
        "provider_runtime/cosmos3_input/smoke_request_inventory.json",
        "provider_runtime/cosmos3_input/action_streams.json",
    ]
    reference = [
        "provider_runtime/cosmos3_droid_reference/canary_manifest.json",
        "provider_runtime/cosmos3_droid_reference/initial_observation.png",
        "provider_runtime/cosmos3_droid_reference/action_streams.json",
    ]
    ctrl_world = [
        "provider_runtime/ctrl_world_replay/canary_manifest.json",
        "provider_runtime/ctrl_world_replay/annotation.json",
        "provider_runtime/ctrl_world_replay/view_0.mp4",
        "provider_runtime/ctrl_world_replay/view_1.mp4",
        "provider_runtime/ctrl_world_replay/view_2.mp4",
    ]
    assert wam_registered_alternative_inputs_present(bundle_path=unused, zip_entries=standard)
    assert wam_registered_alternative_inputs_present(bundle_path=unused, zip_entries=reference)
    assert wam_registered_alternative_inputs_present(bundle_path=unused, zip_entries=ctrl_world)


def test_wam_registered_alternative_validates_powered_layout(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.zip"
    valid_entries = _write_powered_bundle(valid_path)
    assert wam_registered_alternative_inputs_present(
        bundle_path=valid_path, zip_entries=valid_entries
    )

    invalid_path = tmp_path / "missing-image.zip"
    invalid_entries = _write_powered_bundle(invalid_path, omit_image=True)
    assert not wam_registered_alternative_inputs_present(
        bundle_path=invalid_path, zip_entries=invalid_entries
    )
