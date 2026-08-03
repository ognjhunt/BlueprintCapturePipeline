from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from scripts import bootstrap_measurement_chrono_development as chrono_bootstrap


def _environment(tmp_path: Path) -> tuple[Path, Path]:
    environment = tmp_path / "chrono"
    python = environment / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    record = environment / "conda-meta/pychrono-10.0.0-test_0.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "name": "pychrono",
                "version": "10.0.0",
                "build": "py312_test_0",
                "channel": chrono_bootstrap.PYCHRONO_CHANNEL,
                "subdir": "osx-arm64",
            }
        ),
        encoding="utf-8",
    )
    conda = tmp_path / "conda"
    conda.write_text("", encoding="utf-8")
    return environment, conda


def test_inspection_binds_conda_record_and_import_without_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment, conda = _environment(tmp_path)
    openmp = environment / "lib/libiomp5.dylib"
    openmp.parent.mkdir(parents=True)
    openmp.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        chrono_bootstrap,
        "_verify_import",
        lambda _python: {
            "python_version": "3.12.11",
            "openmp_library": str(openmp),
            "openmp_preload_used": True,
            "import_verified": True,
        },
    )
    receipt = chrono_bootstrap.inspect_environment(environment=environment, conda=conda)
    assert receipt["pychrono_version"] == "10.0.0"
    assert receipt["package_build"] == "py312_test_0"
    assert receipt["package_metadata_source"] == "conda-meta"
    assert receipt["openmp_preload_used"] is True
    assert receipt["granular_benchmark_established_by_environment"] is False
    assert receipt["production_route_eligible"] is False
    assert receipt["r7_admission"] is False
    normalized = dict(receipt)
    digest = normalized.pop("bootstrap_receipt_digest")
    assert digest == "sha256:" + hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/measurement_chrono_development_environment.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(receipt, schema)


def test_wrong_channel_and_existing_bootstrap_target_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment, conda = _environment(tmp_path)
    record = next((environment / "conda-meta").glob("pychrono-*.json"))
    value = json.loads(record.read_text(encoding="utf-8"))
    value["channel"] = "https://conda.anaconda.org/conda-forge"
    record.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(chrono_bootstrap.ChronoBootstrapError, match="channel_mismatch"):
        chrono_bootstrap.inspect_environment(environment=environment, conda=conda)
    monkeypatch.setattr(chrono_bootstrap, "_run", lambda _argv: None)
    with pytest.raises(chrono_bootstrap.ChronoBootstrapError, match="must_be_new"):
        chrono_bootstrap.bootstrap(environment=environment, conda=conda)


def test_conda_record_accepts_channel_url_with_subdir_suffix(tmp_path: Path) -> None:
    environment, _conda = _environment(tmp_path)
    record = next((environment / "conda-meta").glob("pychrono-*.json"))
    value = json.loads(record.read_text(encoding="utf-8"))
    value["channel"] = f"{chrono_bootstrap.PYCHRONO_CHANNEL}/osx-arm64"
    record.write_text(json.dumps(value), encoding="utf-8")
    normalized = chrono_bootstrap._conda_record(environment)
    assert normalized["package_channel"] == chrono_bootstrap.PYCHRONO_CHANNEL


def test_bootstrap_uses_exact_release_channels_and_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conda = tmp_path / "conda"
    conda.write_text("", encoding="utf-8")
    target = tmp_path / "new-chrono"
    observed: list[list[str]] = []

    def fake_run(argv: list[str]) -> None:
        observed.append(list(argv))

    monkeypatch.setattr(chrono_bootstrap, "_run", fake_run)
    monkeypatch.setattr(
        chrono_bootstrap,
        "inspect_environment",
        lambda **_kwargs: {"schema_version": chrono_bootstrap.SCHEMA_VERSION},
    )
    chrono_bootstrap.bootstrap(environment=target, conda=conda)
    assert observed == [
        [
            str(conda),
            "create",
            "--yes",
            "--prefix",
            str(target),
            "--override-channels",
            "--strict-channel-priority",
            "--channel",
            chrono_bootstrap.PYCHRONO_CHANNEL,
            "--channel",
            chrono_bootstrap.CONDA_FORGE_CHANNEL,
            "python=3.12",
            "pychrono=10.0.0",
        ]
    ]
