"""Resolve CUDA headers and libraries shipped by pinned PyTorch wheels.

The Isaac image supplies ``nvcc`` but its ``/usr/local/cuda`` tree is not a
complete toolkit.  PyTorch's CUDA wheels carry the missing developer headers
and shared libraries under ``site-packages/nvidia``.  Native ArtiFixer builds
must bind those exact installed paths rather than assuming the image owns them.
"""

from __future__ import annotations

from pathlib import Path
import sysconfig
from typing import Mapping, Sequence


REQUIRED_PACKAGE_HEADERS: Mapping[str, str] = {
    "cublas": "cublas_v2.h",
    "cuda_nvrtc": "nvrtc.h",
    "cusolver": "cusolverDn.h",
    "cusparse": "cusparse.h",
}
_NVRTC_PACKAGE = "cuda_nvrtc"
_NVRTC_LINK_NAME = "libnvrtc.so"


class ArtifixerCudaPackagePathsError(RuntimeError):
    """The pinned CUDA wheel closure is incomplete for native compilation."""


def resolve_artifixer_cuda_package_paths(
    *, site_package_roots: Sequence[str | Path]
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    include_paths: list[Path] = []
    library_paths: list[Path] = []
    for package, required_header in REQUIRED_PACKAGE_HEADERS.items():
        matches: list[tuple[Path, Path]] = []
        for raw_root in site_package_roots:
            root = Path(raw_root).expanduser().resolve()
            include = root / "nvidia" / package / "include"
            library = root / "nvidia" / package / "lib"
            if (
                not root.is_symlink()
                and include.is_dir()
                and not include.is_symlink()
                and (include / required_header).is_file()
                and library.is_dir()
                and not library.is_symlink()
                and any(
                    item.is_file() and not item.is_symlink()
                    for item in library.glob("lib*.so*")
                )
            ):
                matches.append((include.resolve(), library.resolve()))
        if len(matches) != 1:
            raise ArtifixerCudaPackagePathsError(
                f"artifixer3d_cuda_package_invalid:{package}"
            )
        include_paths.append(matches[0][0])
        library_paths.append(matches[0][1])
    return tuple(include_paths), tuple(library_paths)


def ensure_artifixer_nvrtc_link_name(*, library_path: str | Path) -> Path:
    """Install the unversioned linker name for the pinned NVRTC wheel.

    NVIDIA's wheel carries ``libnvrtc.so.12`` but no ``libnvrtc.so``. Native
    extensions link with ``-lnvrtc``, so a private link name is required inside
    the already self-created virtual environment. Refuse ambiguous versions or
    any pre-existing link that does not resolve to the same pinned directory.
    """

    library = Path(library_path).expanduser().resolve()
    if not library.is_dir() or library.is_symlink():
        raise ArtifixerCudaPackagePathsError(
            "artifixer3d_cuda_nvrtc_library_invalid"
        )
    link = library / _NVRTC_LINK_NAME
    candidates = tuple(
        path
        for path in sorted(library.glob(f"{_NVRTC_LINK_NAME}.*"))
        if path.is_file()
        and not path.is_symlink()
        and path.name[len(_NVRTC_LINK_NAME) + 1 :].isdigit()
    )
    if len(candidates) != 1:
        raise ArtifixerCudaPackagePathsError(
            "artifixer3d_cuda_nvrtc_library_invalid"
        )
    target = candidates[0].resolve()
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != target:
            raise ArtifixerCudaPackagePathsError(
                "artifixer3d_cuda_nvrtc_link_invalid"
            )
    else:
        link.symlink_to(target.name)
    if not link.is_symlink() or link.resolve() != target:
        raise ArtifixerCudaPackagePathsError(
            "artifixer3d_cuda_nvrtc_link_invalid"
        )
    return link


def main() -> int:
    purelib = sysconfig.get_paths().get("purelib")
    if not purelib:
        raise ArtifixerCudaPackagePathsError(
            "artifixer3d_cuda_site_packages_invalid"
        )
    includes, libraries = resolve_artifixer_cuda_package_paths(
        site_package_roots=(purelib,)
    )
    package_names = tuple(REQUIRED_PACKAGE_HEADERS)
    ensure_artifixer_nvrtc_link_name(
        library_path=libraries[package_names.index(_NVRTC_PACKAGE)]
    )
    print(":".join(str(path) for path in includes))
    print(":".join(str(path) for path in libraries))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by provider shell
    raise SystemExit(main())


__all__ = [
    "ArtifixerCudaPackagePathsError",
    "REQUIRED_PACKAGE_HEADERS",
    "ensure_artifixer_nvrtc_link_name",
    "resolve_artifixer_cuda_package_paths",
]
