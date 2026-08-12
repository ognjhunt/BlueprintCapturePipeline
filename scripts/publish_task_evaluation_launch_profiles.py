#!/usr/bin/env python3
"""Validate and immutably publish Pipeline launch profiles plus WebApp catalog.

Profiles contain no secret values. They bind canonical secret profile IDs,
allocator arguments, source/spec digests, spend/TTL ceilings, terminal evidence,
reconciliation providers, and WebApp sync policy. The generated WebApp catalog
contains only the public descriptor needed to select that exact profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    TaskEvaluationLaunchError,
    public_launch_profile_descriptor,
    validate_launch_profile,
    verify_profile_immutable_inputs,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TaskEvaluationLaunchError(f"profile_json_object_required:{path}")
    return dict(value)


def _write_exact(path: Path, payload: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
        return True
    except FileExistsError:
        if path.read_bytes() != payload:
            raise TaskEvaluationLaunchError(f"immutable_profile_conflict:{path.name}")
        return False


def publish_profiles(
    *,
    profile_paths: Sequence[str | Path],
    profile_dir: str | Path,
    webapp_catalog_out: str | Path,
) -> dict[str, Any]:
    published: list[dict[str, Any]] = []
    descriptors: list[dict[str, Any]] = []
    target_root = Path(profile_dir).expanduser().resolve()
    for source_value in profile_paths:
        source_input = Path(source_value).expanduser()
        if source_input.is_symlink():
            raise TaskEvaluationLaunchError(f"launch_profile_source_invalid:{source_input}")
        source = source_input.resolve()
        if not source.is_file():
            raise TaskEvaluationLaunchError(f"launch_profile_source_invalid:{source}")
        profile = _read(source)
        blockers = validate_launch_profile(profile)
        blockers.extend(verify_profile_immutable_inputs(profile))
        if blockers:
            raise TaskEvaluationLaunchError(",".join(sorted(set(blockers))))
        payload = (json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n").encode()
        target = target_root / f"{profile['profile_id']}.json"
        created = _write_exact(target, payload)
        published.append(
            {
                "profile_id": profile["profile_id"],
                "profile_digest": profile["profile_digest"],
                "path": str(target),
                "created": created,
            }
        )

    # The catalog is what the WebApp reads to resolve a profile_id, so it must
    # describe every published profile -- not just the ones named in this
    # invocation. Building it from the arguments meant a profile could be
    # published and still be invisible: a launch against it was rejected at
    # lookup, and the profile directory looked correct while the catalog listed
    # a single stale entry. Enumerating the directory makes the catalog a
    # function of published state instead of of a command line.
    for path in sorted(target_root.glob("*.json")):
        if path.is_symlink():
            raise TaskEvaluationLaunchError(f"launch_profile_source_invalid:{path}")
        catalog_profile = _read(path)
        blockers = validate_launch_profile(catalog_profile)
        blockers.extend(verify_profile_immutable_inputs(catalog_profile))
        if blockers:
            # A published directory holding an invalid profile is a real fault:
            # refuse rather than emit a catalog that silently omits it.
            raise TaskEvaluationLaunchError(
                f"published_profile_invalid:{path.name}:" + ",".join(sorted(set(blockers)))
            )
        descriptors.append(public_launch_profile_descriptor(catalog_profile))

    catalog_payload = (
        json.dumps(
            sorted(descriptors, key=lambda row: str(row["profile_id"])),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    catalog_path = Path(webapp_catalog_out).expanduser().resolve()
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_bytes(catalog_payload)
    return {
        "schema_version": "task_evaluation_launch_profile_publication.v1",
        "status": "published",
        "profiles": published,
        "webapp_catalog_path": str(catalog_path),
        "webapp_catalog_contains_allocator_arguments": False,
        "webapp_catalog_contains_secret_values": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", required=True)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--webapp-catalog-out", required=True)
    args = parser.parse_args(argv)
    try:
        result = publish_profiles(
            profile_paths=args.profile,
            profile_dir=args.profile_dir,
            webapp_catalog_out=args.webapp_catalog_out,
        )
    except (OSError, json.JSONDecodeError, TaskEvaluationLaunchError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "task_evaluation_launch_profile_publication.v1",
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                    "blockers": [str(exc)],
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
