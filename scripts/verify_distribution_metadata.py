#!/usr/bin/env python3
"""Verify built distributions carry the approved license and project metadata."""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path
from zipfile import ZipFile


REQUIRED_PROJECT_URLS = ("Homepage", "Repository", "Issues", "Support", "Security")


def verify(dist_dir: Path) -> list[str]:
    errors: list[str] = []
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1:
        errors.append(f"expected_one_wheel:{len(wheels)}")
    if len(sdists) != 1:
        errors.append(f"expected_one_sdist:{len(sdists)}")

    if len(wheels) == 1:
        with ZipFile(wheels[0]) as archive:
            names = archive.namelist()
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
            license_names = [name for name in names if name.endswith(".dist-info/licenses/LICENSE")]
            if len(metadata_names) != 1:
                errors.append(f"wheel_metadata_file_count:{len(metadata_names)}")
            else:
                metadata = archive.read(metadata_names[0]).decode("utf-8")
                if "Metadata-Version: 2.4" not in metadata:
                    errors.append("wheel_metadata_version_is_not_2_4")
                if "License-Expression: MIT" not in metadata:
                    errors.append("wheel_license_expression_is_not_mit")
                if "Classifier: License ::" in metadata:
                    errors.append("wheel_contains_deprecated_license_classifier")
                for label in REQUIRED_PROJECT_URLS:
                    if f"Project-URL: {label}, " not in metadata:
                        errors.append(f"wheel_project_url_missing:{label}")
            if len(license_names) != 1:
                errors.append(f"wheel_license_file_count:{len(license_names)}")
            elif not archive.read(license_names[0]).decode("utf-8").startswith("MIT License"):
                errors.append("wheel_license_file_is_not_mit")

    if len(sdists) == 1:
        with tarfile.open(sdists[0]) as archive:
            names = archive.getnames()
        for required in (
            "LICENSE",
            "SECURITY.md",
            "uv.lock",
            "requirements.txt",
            "requirements-geometry.txt",
        ):
            if not any(name.endswith(f"/{required}") for name in names):
                errors.append(f"sdist_file_missing:{required}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    errors = verify(args.dist_dir.resolve())
    if errors:
        for error in errors:
            print(f"[distribution-metadata] ERROR {error}", file=sys.stderr)
        return 1
    print("[distribution-metadata] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
