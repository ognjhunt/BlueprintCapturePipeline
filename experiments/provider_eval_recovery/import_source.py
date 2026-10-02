"""Stage already materialized source bytes; never downloads or guesses a URL."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import stat
import zipfile

from .harness import write_once

EXPECTED_SHA256 = "294f778b32670ee6ae07412a53c1627923f1004afeea501f32d3202e8b9cf736"
EXPECTED_BYTES = 26019
REQUIRED = {"public/inputs.json", "reviewer/oracle.json", "reviewer/spec.md"}


def stage_zip(source, destination, expected_sha256=EXPECTED_SHA256, expected_bytes=EXPECTED_BYTES):
    source, destination = Path(source), Path(destination)
    blob = source.read_bytes()
    if len(blob) != expected_bytes or hashlib.sha256(blob).hexdigest() != expected_sha256:
        raise ValueError("source bundle byte count or SHA256 mismatch")
    if destination.exists():
        raise ValueError("refuse to overwrite an existing source partition")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or not REQUIRED.issubset(names):
            raise ValueError("duplicate or missing required ZIP members")
        if len(names) > 50 or sum(i.file_size for i in archive.infolist()) > 2_000_000:
            raise ValueError("source ZIP expansion cap")
        for info in archive.infolist():
            path = Path(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                raise ValueError("unsafe ZIP member")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("ZIP symlink rejected")
        payloads = {name: archive.read(name) for name in REQUIRED}
        # Validate parseability but retain exact bytes; do not infer their schema.
        json.loads(payloads["public/inputs.json"])
        json.loads(payloads["reviewer/oracle.json"])
        payloads["reviewer/spec.md"].decode("utf-8")
    destination.mkdir(parents=True, mode=0o700)
    for name, data in payloads.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open("xb") as handle:
            path.chmod(0o600)
            handle.write(data)
    retained = {"library_file_id": "libfile_01ae219642688191a6063e9d3fba9c0b",
                "version": 0, "bundle_sha256": expected_sha256,
                "size_bytes": expected_bytes, "schema_integration": "pending_spec_review",
                "member_sha256": {name: hashlib.sha256(data).hexdigest()
                                  for name, data in payloads.items()}}
    write_once(destination / "source_identity.json", retained)
    return retained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(stage_zip(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
