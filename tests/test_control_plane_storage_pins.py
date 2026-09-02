"""Pins keep derived directories alive until their consumer is terminal or the TTL passes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.control_plane_storage_pins import (
    ControlPlaneStoragePinError,
    live_pinned_paths,
    load_storage_pins,
    pin_path,
    pins_root_from_environment,
    release_storage_pin,
    write_storage_pin,
)


def test_first_writer_wins_and_pins_are_typed(tmp_path: Path) -> None:
    pins = tmp_path / "pins"
    first = write_storage_pin(
        pins_root=pins,
        kind="preparation",
        owner_id="prep-1",
        paths=[tmp_path / "inputs" / "prep-1"],
        now=lambda: 1_000.0,
        ttl_seconds=100,
    )
    again = write_storage_pin(
        pins_root=pins,
        kind="preparation",
        owner_id="prep-1",
        paths=[tmp_path / "other"],
        now=lambda: 2_000.0,
    )

    assert again == first
    assert first["paths"] == [str(tmp_path / "inputs" / "prep-1")]
    assert first["expires_at_epoch"] == 1_100.0
    path = pin_path(pins, "preparation", "prep-1")
    assert path.stat().st_mode & 0o777 == 0o640
    assert json.loads(path.read_text(encoding="utf-8")) == first
    with pytest.raises(ControlPlaneStoragePinError, match="pin_kind_invalid"):
        write_storage_pin(pins_root=pins, kind="bogus", owner_id="x", paths=[tmp_path])
    with pytest.raises(ControlPlaneStoragePinError, match="pin_owner_invalid"):
        write_storage_pin(pins_root=pins, kind="preparation", owner_id="../x", paths=[tmp_path])
    with pytest.raises(ControlPlaneStoragePinError, match="path_not_absolute"):
        write_storage_pin(pins_root=pins, kind="preparation", owner_id="p", paths=["relative"])


def test_live_pins_expire_and_release_cascades_only_when_nothing_else_depends(
    tmp_path: Path,
) -> None:
    pins = tmp_path / "pins"
    prep = tmp_path / "inputs" / "prep-1"
    compiled = tmp_path / "compiled" / "prep-1"
    launch_a = tmp_path / "activations" / "act-a"
    launch_b = tmp_path / "activations" / "act-b"
    write_storage_pin(pins_root=pins, kind="preparation", owner_id="prep-1", paths=[prep], now=lambda: 0.0)
    write_storage_pin(
        pins_root=pins,
        kind="compilation",
        owner_id="prep-1",
        paths=[compiled],
        depends_on=[{"kind": "preparation", "owner_id": "prep-1"}],
        now=lambda: 0.0,
    )
    for owner, path in (("act-a", launch_a), ("act-b", launch_b)):
        write_storage_pin(
            pins_root=pins,
            kind="activation",
            owner_id=owner,
            paths=[path],
            depends_on=[
                {"kind": "preparation", "owner_id": "prep-1"},
                {"kind": "compilation", "owner_id": "prep-1"},
            ],
            now=lambda: 0.0,
        )

    assert live_pinned_paths(pins, now=lambda: 10.0) == {
        str(prep),
        str(compiled),
        str(launch_a),
        str(launch_b),
    }

    # Releasing one activation keeps the shared preparation and compilation
    # pinned for the other activation.
    first = release_storage_pin(pins_root=pins, kind="activation", owner_id="act-a", now=lambda: 20.0)
    assert first["released"] == [{"kind": "activation", "owner_id": "act-a"}]
    assert live_pinned_paths(pins, now=lambda: 30.0) == {str(prep), str(compiled), str(launch_b)}

    # The last dependent releases the chain.
    second = release_storage_pin(pins_root=pins, kind="activation", owner_id="act-b", now=lambda: 40.0)
    assert sorted(row["owner_id"] + ":" + row["kind"] for row in second["released"]) == [
        "act-b:activation",
        "prep-1:compilation",
        "prep-1:preparation",
    ]
    assert live_pinned_paths(pins, now=lambda: 50.0) == set()
    statuses = {(pin["kind"], pin["owner_id"]): pin["status"] for pin in load_storage_pins(pins, now=lambda: 50.0)}
    assert set(statuses.values()) == {"released"}

    # A pin whose release never arrives still expires.
    write_storage_pin(pins_root=pins, kind="preparation", owner_id="prep-2", paths=[tmp_path / "p2"], ttl_seconds=10, now=lambda: 100.0)
    assert str(tmp_path / "p2") in live_pinned_paths(pins, now=lambda: 105.0)
    assert str(tmp_path / "p2") not in live_pinned_paths(pins, now=lambda: 111.0)

    # Releasing an unknown pin is a no-op, never an error.
    assert release_storage_pin(pins_root=pins, kind="activation", owner_id="absent", now=lambda: 1.0)["released"] == []


def test_pins_root_comes_from_the_unit_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT", raising=False)
    assert pins_root_from_environment() is None
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT", "/var/lib/blueprint/pins")
    assert pins_root_from_environment() == Path("/var/lib/blueprint/pins")
