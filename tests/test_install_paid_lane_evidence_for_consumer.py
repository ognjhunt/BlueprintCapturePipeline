"""Bytes produced on the control plane still have to be openable by the service.

The config preflight is the case that forced this: it binds the deployed
commit, so it has to run *there*, and it drives Docker, so it runs as an
account the units are not. What it leaves behind looks perfect and the
pipeline cannot read a byte of it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


installer = _load("install_paid_lane_evidence_for_consumer")
StagingError = installer.StagingError


class _Transport:
    """A host where `owner` may or may not be able to open what it was handed."""

    def __init__(self, *, unreadable_for: tuple[str, ...] = (), corrupt: str | None = None) -> None:
        self.unreadable_for = unreadable_for
        self.corrupt = corrupt
        self.finalized: tuple[str, str] | None = None

    def finalize(self, local_dir: str, owner: str) -> None:
        self.finalized = (local_dir, owner)

    def digest(self, remote: str, *, as_user: str | None = None) -> str:
        name = Path(remote).name
        if as_user and name in self.unreadable_for:
            return ""
        if self.corrupt and name == self.corrupt:
            return "sha256:" + hashlib.sha256(b"different").hexdigest()
        return "sha256:" + hashlib.sha256(Path(remote).read_bytes()).hexdigest()


@pytest.fixture()
def evidence(tmp_path, monkeypatch) -> Path:
    """An evidence directory sitting under an admitted control-plane root."""

    root = tmp_path / "var" / "lib" / "blueprint"
    directory = root / "task-evaluation-inputs" / "lane" / "paid_config_preflight_v1"
    directory.mkdir(parents=True)
    (directory / "adp_content_agents_bundle_config_preflight.json").write_text(
        json.dumps({"status": "passed"}), encoding="utf-8"
    )
    (directory / "material-agent.log").write_text("ok\n", encoding="utf-8")
    monkeypatch.setattr(installer, "PRODUCTION_LAUNCH_INPUT_ROOTS", (str(root),))
    return directory


def test_install_proves_the_consumer_can_read_every_file(evidence) -> None:
    transport = _Transport()

    receipt = installer.install_paid_lane_evidence_for_consumer(
        evidence_dir=evidence, transport=transport
    )

    assert receipt["status"] == "installed"
    assert receipt["verified_readable_as"] == "blueprint"
    assert receipt["provider_mutation_performed"] is False
    assert transport.finalized == (str(evidence), "blueprint")
    names = sorted(row["relative_path"] for row in receipt["installed_files"])
    assert names == ["adp_content_agents_bundle_config_preflight.json", "material-agent.log"]
    for row in receipt["installed_files"]:
        assert row["sha256"].startswith("sha256:")
        assert row["size_bytes"] > 0


def test_a_file_the_consumer_cannot_open_is_not_an_install(evidence) -> None:
    """The exact shape of the failure: right bytes, wrong account."""

    transport = _Transport(unreadable_for=("material-agent.log",))

    with pytest.raises(StagingError) as excinfo:
        installer.install_paid_lane_evidence_for_consumer(
            evidence_dir=evidence, transport=transport
        )

    assert str(excinfo.value) == "evidence_consumer_cannot_read:material-agent.log"


def test_unreadable_is_reported_apart_from_corrupt(evidence) -> None:
    transport = _Transport(corrupt="adp_content_agents_bundle_config_preflight.json")

    with pytest.raises(StagingError) as excinfo:
        installer.install_paid_lane_evidence_for_consumer(
            evidence_dir=evidence, transport=transport
        )

    assert str(excinfo.value) == (
        "evidence_install_digest_mismatch:adp_content_agents_bundle_config_preflight.json"
    )


def test_installing_outside_a_control_plane_root_is_refused(tmp_path, monkeypatch) -> None:
    """Otherwise the residency gate refuses it later, at the paid boundary."""

    directory = tmp_path / "somewhere" / "evidence"
    directory.mkdir(parents=True)
    (directory / "receipt.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(installer, "PRODUCTION_LAUNCH_INPUT_ROOTS", ("/var/lib/blueprint",))

    with pytest.raises(StagingError) as excinfo:
        installer.install_paid_lane_evidence_for_consumer(
            evidence_dir=directory, transport=_Transport()
        )

    assert str(excinfo.value).startswith("evidence_dir_outside_control_plane:")


def test_an_empty_directory_is_not_an_install(tmp_path, monkeypatch) -> None:
    root = tmp_path / "var" / "lib" / "blueprint"
    directory = root / "evidence"
    directory.mkdir(parents=True)
    monkeypatch.setattr(installer, "PRODUCTION_LAUNCH_INPUT_ROOTS", (str(root),))

    with pytest.raises(StagingError) as excinfo:
        installer.install_paid_lane_evidence_for_consumer(
            evidence_dir=directory, transport=_Transport()
        )

    assert str(excinfo.value) == "evidence_dir_empty"


def test_the_local_transport_reads_back_as_the_consumer(tmp_path) -> None:
    """A digest read as the invoking account proves nothing about the service."""

    from stage_paid_lane_bundle import LocalTransport  # noqa: PLC0415

    target = tmp_path / "file.json"
    target.write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = hashlib.sha256(b"{}").hexdigest() + "  " + str(target)

    def _run(argv, **_kwargs):
        calls.append(list(argv))
        return _Result()

    import subprocess  # noqa: PLC0415

    original = subprocess.run
    subprocess.run = _run  # type: ignore[assignment]
    try:
        observed = LocalTransport().digest(str(target), as_user="blueprint")
    finally:
        subprocess.run = original  # type: ignore[assignment]

    assert observed == "sha256:" + hashlib.sha256(b"{}").hexdigest()
    # Not a bare sha256sum: the whole point is whose eyes are reading.
    assert calls == [["sudo", "-n", "-u", "blueprint", "sha256sum", str(target)]]
