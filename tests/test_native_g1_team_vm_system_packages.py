"""Observed guest and offline system bytes never imply GPU admission."""

import hashlib
import json

import pytest

from blueprint_pipeline import native_g1_team_vm_system_packages as system


def _digest(value):
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                {k: v for k, v in value.items() if k != "receipt_digest"},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
    )


def observation():
    guest = {
        "schema_version": "g1_vm_guest_system_cpu_observation.v1",
        "scope": "local_tcg_cpu_only",
        "uid": 0,
        "platform": "x86_64",
        "gpu_runtime_qualified": False,
        "policy_inference_performed": False,
        "provider_mutation_performed": False,
        "claim_ceiling": "development_only",
        "probes": {
            "kernel": {"exit_code": 0, "stdout": system.KERNEL + "\n"},
            "packages": {
                "exit_code": 0,
                "stdout": "\n".join(
                    name + " " + version for name, version in system.BASE_PACKAGES.items()
                ),
            },
        },
    }
    guest["receipt_digest"] = _digest(guest)
    result = {
        "schema_version": "g1_vm_system_cpu_preflight.v1",
        "status": "guest_cpu_observed",
        "implementation_commit": "a" * 40,
        "exit_code": 0,
        "child_terminal": True,
        "scope": "local_tcg_cpu_only",
        "gpu_runtime_qualified": False,
        "policy_inference_performed": False,
        "provider_mutation_performed": False,
        "claim_ceiling": "development_only",
        "image_ref": system.IMAGE_REF,
        "guest_disk_sha256": system.GUEST_DISK_SHA,
        "guest_observation": guest,
    }
    result["receipt_digest"] = _digest(result)
    return result


@pytest.mark.parametrize(
    "fault", [None, "digest", "base", "image", "live", "exit", "gpu", "inner", "kernel", "headers"]
)
def test_preparation_reopens_exact_terminal_guest_identity(tmp_path, fault):
    value = observation()
    if fault == "digest":
        value["receipt_digest"] = "sha256:" + "0" * 64
    elif fault == "base":
        value["guest_disk_sha256"] = "sha256:" + "0" * 64
    elif fault == "image":
        value["image_ref"] = "foreign:latest"
    elif fault == "live":
        value["child_terminal"] = False
    elif fault == "exit":
        value["exit_code"] = True
    elif fault == "gpu":
        value["gpu_runtime_qualified"] = True
    elif fault == "inner":
        value["guest_observation"]["receipt_digest"] = "sha256:" + "0" * 64
    elif fault in ("kernel", "headers"):
        field = "kernel" if fault == "kernel" else "packages"
        value["guest_observation"]["probes"][field]["stdout"] = "foreign"
        value["guest_observation"]["receipt_digest"] = _digest(value["guest_observation"])
    if fault != "digest":
        value["receipt_digest"] = _digest(value)
    path = tmp_path / "observation.json"
    path.write_text(json.dumps(value))
    if fault:
        with pytest.raises(ValueError, match="g1_vm_system"):
            system.verify_guest_observation(path)
    else:
        assert system.verify_guest_observation(path)["receipt_digest"] == value["receipt_digest"]


@pytest.mark.parametrize("fault", [None, "missing", "foreign", "hash", "symlink"])
def test_all_package_bytes_verified_before_manifest_or_command(tmp_path, monkeypatch, fault):
    content = b"fixture Debian package bytes"
    assets = {
        "fixture": {
            "package": "fixture",
            "version": "1",
            "filename": "fixture_1_amd64.deb",
            "size_bytes": len(content),
            "sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
            "url": "https://official.invalid/fixture_1_amd64.deb",
        }
    }
    monkeypatch.setattr(system, "SYSTEM_PACKAGES", assets)
    package = tmp_path / assets["fixture"]["filename"]
    package.write_bytes(content)
    paths = {"fixture": package}
    if fault == "missing":
        paths = {}
    elif fault == "foreign":
        paths["foreign"] = package
    elif fault == "hash":
        package.write_bytes(b"changed package")
    elif fault == "symlink":
        link = tmp_path / "link.deb"
        link.symlink_to(package)
        paths["fixture"] = link
    observed = tmp_path / "observation.json"
    observed.write_text(json.dumps(observation()))
    target = tmp_path / "prepared"
    if fault:
        with pytest.raises(ValueError, match="g1_vm_system"):
            system.prepare_system_packages(
                observation_path=observed,
                asset_paths=paths,
                output_root=target,
                implementation_commit="b" * 40,
            )
        assert not target.exists()
    else:
        value = system.prepare_system_packages(
            observation_path=observed,
            asset_paths=paths,
            output_root=target,
            implementation_commit="b" * 40,
        )
        assert value["gpu_runtime_qualified"] is False
        assert value["runtime_installation_performed"] is False
        assert value["owner_review_required"] is True
        assert package.read_bytes() == content
        assert (
            system.verify_system_packages(target, expected_implementation_commit="b" * 40) == value
        )
        with pytest.raises(ValueError, match="manifest"):
            system.verify_system_packages(target, expected_implementation_commit="c" * 40)
        manifest_path = target / system.MANIFEST
        declared = json.loads(manifest_path.read_text())
        declared["owner_review_required"] = False
        manifest_path.chmod(0o600)
        manifest_path.write_text(json.dumps(declared))
        with pytest.raises(ValueError, match="manifest"):
            system.verify_system_packages(target, expected_implementation_commit="b" * 40)
        manifest_path.write_text(json.dumps(value))
        argv = system.offline_apt_simulation_command(
            target, expected_implementation_commit="b" * 40
        )
        assert "--simulate" in argv and "--no-download" in argv
        assert "Dir::Etc::sourcelist=-" in argv and "Dir::Etc::sourceparts=-" in argv
        assert "Dir::State::lists=" + str(target / system.EMPTY_LISTS) in argv
        assert argv[-1] == str(target / assets["fixture"]["filename"])
        assert not any(
            x in argv for x in ("update", "upgrade", "autoremove", "--allow-unauthenticated")
        )
        (target / assets["fixture"]["filename"]).chmod(0o600)
        (target / assets["fixture"]["filename"]).write_bytes(b"changed retained byte")
        with pytest.raises(ValueError, match="g1_vm_system"):
            system.offline_apt_simulation_command(target, expected_implementation_commit="b" * 40)


def test_pinned_closure_covers_observed_driver_dkms_toolkit_and_sandbox_gaps():
    values = list(system.SYSTEM_PACKAGES.values())
    names = {row["package"] for row in values}
    assert {
        "nvidia-driver-580",
        "nvidia-dkms-580",
        "nvidia-firmware-580",
        "nvidia-kernel-source-580",
        "libnvidia-gl-580",
        "libnvidia-compute-580",
        "dkms",
        "bubblewrap",
        "libnvidia-container1",
        "libnvidia-container-tools",
        "nvidia-container-toolkit-base",
        "nvidia-container-toolkit",
    } <= names
    assert {
        r["version"]
        for r in values
        if r["package"].startswith(("libnvidia-container", "nvidia-container-toolkit"))
    } == {"1.19.0-1"}
    assert all(r["filename"].endswith(("_amd64.deb", "_all.deb")) for r in values)
    assert not any(
        r["package"].startswith(("linux-image", "linux-headers", "docker")) for r in values
    )


@pytest.mark.parametrize("mode", [0o444, 0o400, 0o600])
def test_explicit_link_reuses_only_verified_readonly_public_bytes(tmp_path, monkeypatch, mode):
    content = b"immutable cached Debian bytes"
    row = {
        "package": "fixture",
        "version": "1",
        "filename": "fixture.deb",
        "size_bytes": len(content),
        "sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
        "url": "https://official.invalid/fixture.deb",
    }
    monkeypatch.setattr(system, "SYSTEM_PACKAGES", {"fixture": row})
    asset = tmp_path / "cached.deb"
    asset.write_bytes(content)
    asset.chmod(mode)
    observed = tmp_path / "observed.json"
    observed.write_text(json.dumps(observation()))
    target = tmp_path / "prepared"
    if mode & 0o222:
        with pytest.raises(ValueError, match="g1_vm_system"):
            system.prepare_system_packages(
                observation_path=observed,
                asset_paths={"fixture": asset},
                output_root=target,
                implementation_commit="b" * 40,
                link_immutable_assets=True,
            )
        assert not target.exists()
    else:
        system.prepare_system_packages(
            observation_path=observed,
            asset_paths={"fixture": asset},
            output_root=target,
            implementation_commit="b" * 40,
            link_immutable_assets=True,
        )
        assert asset.stat().st_ino == (target / "fixture.deb").stat().st_ino
        assert asset.stat().st_mode & 0o777 == mode
        assert asset.read_bytes() == content


def test_cli_preparation_reaches_the_same_verifier_with_immutable_source(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        system.subprocess,
        "check_output",
        lambda argv, **kwargs: "b" * 40 + "\n" if argv[1] == "rev-parse" else "",
    )
    monkeypatch.setattr(
        system,
        "prepare_system_packages",
        lambda **kwargs: calls.append(kwargs) or {"manifest_digest": "sha256:" + "c" * 64},
    )
    monkeypatch.setattr(
        system, "fetch_fixed_package_bytes", lambda root: pytest.fail("not download mode")
    )
    system.main(
        [
            "--prepare-root",
            str(tmp_path / "prepared"),
            "--asset-root",
            str(tmp_path / "assets"),
            "--guest-observation",
            str(tmp_path / "observed.json"),
            "--implementation-commit",
            "b" * 40,
            "--link-immutable-assets",
        ]
    )
    assert len(calls) == 1 and calls[0]["link_immutable_assets"] is True
    assert set(calls[0]["asset_paths"]) == set(system.SYSTEM_PACKAGES)
