from __future__ import annotations

import hashlib
import io
import tomllib
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.groot_oscar_model_cache_wheelhouse import (
    _wheel_compatible,
    _select_locked_wheel,
    build_model_cache_wheelhouse,
)


def _wheel(distribution: str, version: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{distribution}/__init__.py", "")
        archive.writestr(
            f"{distribution}-{version}.dist-info/METADATA",
            f"Name: {distribution}\nVersion: {version}\n",
        )
    return buffer.getvalue()


def test_wheelhouse_is_derived_from_locked_linux_closure(tmp_path: Path) -> None:
    bodies = {
        "boto3": _wheel("boto3", "1.0"),
        "huggingface_hub": _wheel("huggingface_hub", "2.0"),
        "shared": _wheel("shared", "3.0"),
    }

    def package(name: str, version: str, dependencies: str = "") -> str:
        filename = f"{name}-{version}-py3-none-any.whl"
        body = bodies[name]
        return f'''[[package]]
name = "{name.replace('_', '-')}"
version = "{version}"
dependencies = [{dependencies}]
wheels = [{{ url = "https://files.pythonhosted.org/{filename}", hash = "sha256:{hashlib.sha256(body).hexdigest()}", size = {len(body)} }}]
'''

    lock = tmp_path / "uv.lock"
    lock.write_text(
        "version = 1\n"
        + package("boto3", "1.0", '{ name = "shared" }')
        + package("huggingface_hub", "2.0", '{ name = "shared" }')
        + package("shared", "3.0"),
        encoding="utf-8",
    )

    def download(url: str, *, maximum_bytes: int) -> bytes:
        name = Path(url).name.split("-")[0]
        body = bodies[name]
        assert len(body) < maximum_bytes
        return body

    result = build_model_cache_wheelhouse(
        lockfile_path=lock,
        output_dir=tmp_path / "output",
        downloader=download,
    )
    assert result["status"] == "ready"
    assert result["python_version"] == "3.12"
    assert {row["name"] for row in result["requirements"]} == {
        "boto3",
        "huggingface-hub",
        "shared",
    }
    assert len(result["wheels"]) == 3
    assert result["sdists_allowed"] is False
    assert result["network_resolution_performed"] is False
    with pytest.raises(ValueError, match="output_already_exists"):
        build_model_cache_wheelhouse(
            lockfile_path=lock,
            output_dir=tmp_path / "output",
            downloader=download,
        )


def test_wheel_compatibility_respects_python_abi_direction() -> None:
    assert _wheel_compatible(
        "native-1.0-cp311-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
    )
    assert not _wheel_compatible(
        "native-1.0-cp313-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
    )


def test_locked_wheel_selection_prefers_exact_cp312_platform_over_universal() -> None:
    exact = {
        "url": "https://files.pythonhosted.org/charset_normalizer-3.4.9-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
    }
    universal = {
        "url": "https://files.pythonhosted.org/charset_normalizer-3.4.9-py3-none-any.whl"
    }
    filename, selected = _select_locked_wheel(
        [universal, exact], distribution="charset-normalizer"
    )
    assert filename.endswith("manylinux_2_17_x86_64.whl")
    assert selected is exact


def test_repository_lock_charset_normalizer_has_one_ranked_cp312_winner() -> None:
    root = Path(__file__).resolve().parents[1]
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    package = next(
        row
        for row in lock["package"]
        if row.get("name") == "charset-normalizer" and row.get("version") == "3.4.9"
    )
    filename, _selected = _select_locked_wheel(
        package["wheels"], distribution="charset-normalizer"
    )
    assert "cp312-cp312" in filename
