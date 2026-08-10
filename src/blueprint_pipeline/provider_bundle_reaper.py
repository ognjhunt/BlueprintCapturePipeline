"""Reap a provider bundle once the run is over and it can be rebuilt.

Every paid launch stages a fresh bundle into its own job directory - assets,
runtime, natives, a zip - and nothing has ever removed one. On this lane that
is about 162 MB per launch, so a single working session left 19 of them, and
the tree as a whole accumulated 197 bundles totalling 25 GB. The laptop
eventually filled to the point where the harness could not write a file, which
killed a run that had nothing wrong with it.

The obvious fix - delete bundles - is wrong, and measuring said so. A bundle is
not a copy of anything: asset bindings rename files into it and some are
derived at build time, so two thirds of sampled bundle assets had no identical
copy anywhere outside a bundle. For a run whose inputs have since moved, the
bundle is the only surviving record.

So the rule is narrow: a bundle may be reaped only when every source it was
built from is still on disk, and only after the run has written its result. If
either is not true the bundle stays, and the receipt says which. Uncertainty
keeps bytes; it never spends them.

The run's own evidence - results, manifests, logs, avoidlist - is never
touched. Those are a few hundred kilobytes and they are the point of the run.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Sequence


PROVIDER_BUNDLE_REAPER_SCHEMA_VERSION = "provider_bundle_reaper.v1"
BUNDLE_DIRECTORY_NAME = "bundle"
# A run that has not written one of these is not finished with its bundle.
RESULT_FILENAMES = (
    "adp_arena_vast_result.json",
    "adp009d_native_microcheck.json",
    "vast_provider_adapter_result.json",
)


class ProviderBundleReaperError(ValueError):
    """Stable, sorted reaper failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _directory_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file() and not entry.is_symlink():
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    return total


def reap_provider_bundle(
    *,
    job_dir: str | Path,
    source_paths: Sequence[str | Path],
    bundle_directory_name: str = BUNDLE_DIRECTORY_NAME,
) -> dict[str, Any]:
    """Delete one job's bundle if and only if it could be rebuilt."""

    job = Path(job_dir).expanduser()
    if not job.is_dir():
        raise ProviderBundleReaperError(
            [f"provider_bundle_reaper_job_dir_missing:{job}"]
        )
    job = job.resolve()
    bundle = job / bundle_directory_name

    retained: list[str] = []
    if not bundle.is_dir():
        retained.append("provider_bundle_reaper_already_reaped_or_absent")

    # A run still in flight needs its bundle; the result file is the signal
    # that the provider is done with it.
    if not any((job / name).is_file() for name in RESULT_FILENAMES) and not any(
        job.rglob(RESULT_FILENAMES[0])
    ):
        retained.append("provider_bundle_reaper_no_result_yet")

    sources = [Path(str(value)).expanduser() for value in source_paths]
    if not sources:
        # "Nothing to check" is not "everything checks out". Without a source
        # list there is no evidence the bundle is rebuildable, and the whole
        # rule rests on that evidence.
        retained.append("provider_bundle_reaper_no_source_list")
    for source in sources:
        if not source.is_file():
            retained.append(f"provider_bundle_reaper_source_missing:{source.name}")

    if retained:
        return {
            "schema_version": PROVIDER_BUNDLE_REAPER_SCHEMA_VERSION,
            "job_dir": str(job),
            "reaped": False,
            "reclaimed_bytes": 0,
            "retained_because": sorted(set(retained)),
            "claim_boundary": {
                "retention_is_the_safe_outcome_not_a_failure": True,
                "run_evidence_is_never_reaped": True,
            },
        }

    reclaimed = _directory_bytes(bundle)
    shutil.rmtree(bundle)
    return {
        "schema_version": PROVIDER_BUNDLE_REAPER_SCHEMA_VERSION,
        "job_dir": str(job),
        "reaped": True,
        "reclaimed_bytes": reclaimed,
        "verified_sources": sorted(str(source) for source in sources),
        "retained_because": [],
        "claim_boundary": {
            "bundle_is_rebuildable_from_the_verified_sources": True,
            "run_evidence_is_never_reaped": True,
        },
    }


__all__ = [
    "BUNDLE_DIRECTORY_NAME",
    "PROVIDER_BUNDLE_REAPER_SCHEMA_VERSION",
    "RESULT_FILENAMES",
    "ProviderBundleReaperError",
    "reap_provider_bundle",
]
