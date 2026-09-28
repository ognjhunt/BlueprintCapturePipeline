from __future__ import annotations

from pathlib import Path

import pytest

import blueprint_pipeline.artifixer_cuda_package_paths as cuda_paths
from blueprint_pipeline.artifixer_cuda_package_paths import (
    ArtifixerCudaPackagePathsError,
    REQUIRED_PACKAGE_HEADERS,
    ensure_artifixer_nvrtc_link_name,
    resolve_artifixer_cuda_package_paths,
)


def _cuda_wheel_tree(root: Path) -> None:
    for package, header in REQUIRED_PACKAGE_HEADERS.items():
        include = root / "nvidia" / package / "include"
        library = root / "nvidia" / package / "lib"
        include.mkdir(parents=True)
        library.mkdir(parents=True)
        (include / header).write_text("// fixture\n", encoding="utf-8")
        library_name = "libnvrtc.so.12" if package == "cuda_nvrtc" else f"lib{package}.so.12"
        (library / library_name).write_bytes(b"fixture")


def test_resolves_the_pinned_wheels_native_cuda_closure(tmp_path: Path) -> None:
    site_packages = tmp_path / "site-packages"
    _cuda_wheel_tree(site_packages)

    includes, libraries = resolve_artifixer_cuda_package_paths(
        site_package_roots=(site_packages,)
    )

    assert includes == tuple(
        site_packages / "nvidia" / package / "include"
        for package in REQUIRED_PACKAGE_HEADERS
    )
    assert libraries == tuple(
        site_packages / "nvidia" / package / "lib"
        for package in REQUIRED_PACKAGE_HEADERS
    )
    assert all(
        (include / header).is_file()
        for include, header in zip(
            includes, REQUIRED_PACKAGE_HEADERS.values(), strict=True
        )
    )


def test_refuses_when_the_cusparse_developer_header_is_missing(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    _cuda_wheel_tree(site_packages)
    (site_packages / "nvidia/cusparse/include/cusparse.h").unlink()

    with pytest.raises(
        ArtifixerCudaPackagePathsError,
        match="artifixer3d_cuda_package_invalid:cusparse",
    ):
        resolve_artifixer_cuda_package_paths(site_package_roots=(site_packages,))


def test_installs_private_nvrtc_linker_name_from_exact_pinned_library(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    _cuda_wheel_tree(site_packages)
    library = site_packages / "nvidia/cuda_nvrtc/lib"

    link = ensure_artifixer_nvrtc_link_name(library_path=library)

    assert link.is_symlink()
    assert link.readlink() == Path("libnvrtc.so.12")
    assert link.resolve() == (library / "libnvrtc.so.12").resolve()


def test_provider_cli_installs_nvrtc_linker_name_before_printing_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_packages = tmp_path / "site-packages"
    _cuda_wheel_tree(site_packages)
    monkeypatch.setattr(
        cuda_paths.sysconfig, "get_paths", lambda: {"purelib": str(site_packages)}
    )

    assert cuda_paths.main() == 0
    assert (site_packages / "nvidia/cuda_nvrtc/lib/libnvrtc.so").is_symlink()
