"""Find an asset inside a provider bundle, and map the layout when you cannot.

A bundle does not preserve the names an asset was authored under. Asset
bindings rename the task object to ``approved_can.usda`` and the scene
collision to ``sage_collision.usd`` whatever they were called upstream, and the
bundle nests them under ``provider_runtime/assets`` rather than beside the spec
that names them. A composition spec written at authoring time therefore refers
to files that do not exist by that name on the provider.

The expensive version of this is one launch per renamed asset: the run boots
Isaac, provisions Arena, and dies reporting a single missing filename, which
tells you nothing about what *is* there. So a miss here lists every USD it
actually found. One launch then resolves the layout instead of one launch per
guess - the same reasoning as ``robot_asset_discovery``, and the same reason
that module reports its whole search on failure.

Aliases are explicit. Falling back to "the only USD in the directory" would
paper over a bundle that shipped the wrong asset, which is worse than a miss:
a miss costs a launch, silently loading the wrong scene costs a wrong result
that looks right.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


RUNTIME_ASSET_RESOLUTION_SCHEMA_VERSION = "runtime_asset_resolution.v1"

# Where bundles put assets, relative to the runtime directory. Ordered: the
# runtime directory itself first, so a payload that stages an asset beside its
# spec is not overridden by a same-named binding under assets/.
DEFAULT_RELATIVE_SEARCH_DIRS = (
    ".",
    "assets",
    # Resolution is rooted at the spec's directory, and the bundle stages the
    # spec into provider_runtime/native/ while the assets go to
    # provider_runtime/assets/ - siblings, not parent and child. Without the
    # climb, a payload searches only native/ and finds nothing.
    "../assets",
    "..",
    "provider_runtime/assets",
    "native",
)
USD_SUFFIXES = (".usd", ".usda", ".usdc", ".usdz")
MAX_REPORTED_PRESENT = 40


class RuntimeAssetResolutionError(ValueError):
    """Stable, sorted asset-resolution failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _search_dirs(root: Path, relative: Sequence[str]) -> list[Path]:
    seen: list[Path] = []
    for value in relative:
        candidate = (root / str(value)).resolve() if str(value) != "." else root
        if candidate.is_dir() and candidate not in seen:
            seen.append(candidate)
    return seen


def _present_usd_files(directories: Sequence[Path]) -> list[str]:
    present: list[str] = []
    for directory in directories:
        for entry in sorted(directory.iterdir()):
            if entry.is_file() and entry.suffix.lower() in USD_SUFFIXES:
                present.append(entry.name)
    # Names, not paths: the caller needs to know what to declare, and the same
    # asset appearing in two search directories is not two answers.
    return sorted(set(present))[:MAX_REPORTED_PRESENT]


def resolve_runtime_asset(
    *,
    runtime_dir: str | Path,
    declared_filename: str,
    aliases: Sequence[str] = (),
    role: str = "asset",
    relative_search_dirs: Sequence[str] = DEFAULT_RELATIVE_SEARCH_DIRS,
) -> dict[str, Any]:
    """Resolve one asset by declared name, then by alias, then fail loudly."""

    root = Path(runtime_dir).expanduser()
    if not root.is_dir():
        raise RuntimeAssetResolutionError(
            [f"runtime_asset_runtime_dir_missing:{role}:{root}"]
        )
    root = root.resolve()

    declared = str(declared_filename or "").strip()
    if not declared:
        raise RuntimeAssetResolutionError(
            [f"runtime_asset_declared_filename_missing:{role}"]
        )

    directories = _search_dirs(root, relative_search_dirs)
    checked: list[str] = []

    # Declared name first, across every search directory, before any alias: an
    # alias is a fallback for a rename, not a competing candidate.
    for matched_on, name in [("declared_filename", declared)] + [
        ("alias", str(value)) for value in aliases if str(value)
    ]:
        for directory in directories:
            candidate = directory / Path(name).name
            checked.append(str(candidate))
            if not candidate.is_file():
                continue
            resolved: dict[str, Any] = {
                "schema_version": RUNTIME_ASSET_RESOLUTION_SCHEMA_VERSION,
                "role": role,
                "declared_filename": declared,
                "resolved_path": str(candidate),
                "matched_on": matched_on,
                "paths_checked": checked,
                "claim_boundary": {
                    "existence_only_not_a_load_check": True,
                    "alias_match_is_a_rename_not_an_equivalence_proof": True,
                },
            }
            if matched_on == "alias":
                resolved["matched_alias"] = Path(name).name
            return resolved

    present = _present_usd_files(directories)
    raise RuntimeAssetResolutionError(
        [f"runtime_asset_not_found:{role}:{declared}"]
        + [f"runtime_asset_present:{role}:{name}" for name in present]
        + [f"runtime_asset_searched:{role}:{directory}" for directory in directories]
    )


__all__ = [
    "DEFAULT_RELATIVE_SEARCH_DIRS",
    "RUNTIME_ASSET_RESOLUTION_SCHEMA_VERSION",
    "RuntimeAssetResolutionError",
    "resolve_runtime_asset",
]
