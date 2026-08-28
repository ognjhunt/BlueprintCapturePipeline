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
    "cusolver": "cusolverDn.h",
    "cusparse": "cusparse.h",
}


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


def main() -> int:
    purelib = sysconfig.get_paths().get("purelib")
    if not purelib:
        raise ArtifixerCudaPackagePathsError(
            "artifixer3d_cuda_site_packages_invalid"
        )
    includes, libraries = resolve_artifixer_cuda_package_paths(
        site_package_roots=(purelib,)
    )
    print(":".join(str(path) for path in includes))
    print(":".join(str(path) for path in libraries))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by provider shell
    raise SystemExit(main())


__all__ = [
    "ArtifixerCudaPackagePathsError",
    "REQUIRED_PACKAGE_HEADERS",
    "resolve_artifixer_cuda_package_paths",
]
