from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_groot_wire_wheels import (
    GROOT_WIRE_WHEEL_ARTIFACTS,
    SOURCE_ORIGIN,
    materialize_wire_wheels,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


class _Response(io.BytesIO):
    def __init__(self, content: bytes, url: str):
        super().__init__(content)
        self._url = url

    def geturl(self) -> str:
        return self._url


def _artifact(content: bytes) -> dict[str, object]:
    return {
        "distribution": "msgpack",
        "version": "1.1.0",
        "filename": "msgpack-1.1.0-py3-none-any.whl",
        "url": f"{SOURCE_ORIGIN}/packages/frozen/msgpack-1.1.0-py3-none-any.whl",
        "size_bytes": len(content),
        "sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
    }


def test_exact_wire_wheel_bytes_install_offline_and_are_receipted(tmp_path) -> None:
    content = b"exact synthetic wheel bytes"
    artifact = _artifact(content)
    commands: list[tuple[list[str], dict[str, str]]] = []

    def opener(request, *, timeout):
        assert request.full_url == artifact["url"]
        assert timeout == 120
        return _Response(content, str(artifact["url"]))

    def runner(command, *, check, env):
        assert check is True
        commands.append((list(command), dict(env)))
        target = Path(command[command.index("--target") + 1])
        metadata_root = target / "msgpack-1.1.0.dist-info"
        metadata_root.mkdir(parents=True)
        (metadata_root / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: msgpack\nVersion: 1.1.0\n",
            encoding="utf-8",
        )

    receipt_path = tmp_path / "out" / "wire-wheels.json"
    receipt = materialize_wire_wheels(
        uv_executable=Path("/tool/uv"),
        python_executable=Path("/isaac/python"),
        runtime_dir=tmp_path / "runtime",
        output_path=receipt_path,
        opener=opener,
        runner=runner,
        artifacts=(artifact,),
    )

    command, environment = commands[0]
    assert command[:6] == [
        "/tool/uv",
        "pip",
        "install",
        "--python",
        "/isaac/python",
        "--no-deps",
    ]
    assert "--only-binary=:all:" in command
    assert "--offline" in command
    assert environment["UV_NO_INDEX"] == "1"
    assert not any("numpy" in Path(argument).name for argument in command)
    assert receipt["installed_distributions"] == {"msgpack": "1.1.0"}
    assert receipt["numpy_distribution_staged"] is False
    assert receipt["installer_network_access"] is False
    assert receipt["installer_index_access"] is False
    assert receipt["dependency_resolution_allowed"] is False
    assert receipt["observed_artifacts"][0]["identity_verified"] is True
    assert receipt["observed_artifacts_digest"] == receipt["expected_artifacts_digest"]
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


def test_wire_wheel_hash_mismatch_fails_before_install_or_receipt(tmp_path) -> None:
    content = b"wrong bytes"
    artifact = {**_artifact(content), "sha256": "sha256:" + "0" * 64}
    runner_called = False

    def opener(request, *, timeout):
        return _Response(content, request.full_url)

    def runner(command, *, check, env):
        nonlocal runner_called
        runner_called = True

    receipt_path = tmp_path / "wire-wheels.json"
    with pytest.raises(ValueError, match="groot_wire_wheel_identity_mismatch"):
        materialize_wire_wheels(
            uv_executable=Path("/tool/uv"),
            python_executable=Path("/isaac/python"),
            runtime_dir=tmp_path / "runtime",
            output_path=receipt_path,
            opener=opener,
            runner=runner,
            artifacts=(artifact,),
        )

    assert runner_called is False
    assert not receipt_path.exists()


def test_production_wire_lock_names_only_three_exact_publisher_wheels() -> None:
    distributions = {str(artifact["distribution"]) for artifact in GROOT_WIRE_WHEEL_ARTIFACTS}

    assert distributions == {"pyzmq", "msgpack", "msgpack-numpy"}
    assert "numpy" not in distributions
    for artifact in GROOT_WIRE_WHEEL_ARTIFACTS:
        assert str(artifact["filename"]).endswith(".whl")
        assert str(artifact["url"]).startswith(SOURCE_ORIGIN + "/packages/")
        assert str(artifact["url"]).endswith(str(artifact["filename"]))
        assert len(str(artifact["sha256"]).removeprefix("sha256:")) == 64
        assert int(artifact["size_bytes"]) > 0
